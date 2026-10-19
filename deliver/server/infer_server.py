#!/usr/bin/env python3
"""infer_server.py —— 笔电侧 VLA 推理服务（D2 路线的服务端）。

与 `ros2_ws/src/vla_nav/vla_nav/vla_bridge_node.py` 的 HTTP 契约**逐字段对齐**：

    POST JSON: {
        "image_bgr": <hex>,       # BGR, H×W×3, uint8
        "h": H, "w": W,
        "scan": [...],            # 激光（车头为 0 逆时针，120 维或原生维）
        "state": [...32],         # [x, y, θ, v, ω] 归一化并 pad 到 32
        "instruction": "..."
    }
    → {"action": [v, ω], "latency_ms": ...}

前向路径与训练/仿真评测**完全相同**（复用工作区代码，不做复制品）：
    build_state / frame_batch / prepare_images / prepare_state / sample_actions

## 运行前提（笔电侧）

    笔电上需有 VLA 工作区 + .venv_smolvla（本包不含 Python 环境）：
        .venv_smolvla/bin/python infer_server.py \
            --project /path/to/VLA --ckpt ../models/ft_detf --port 8000

## 为什么归一化统计从 norm_stats.json 读

训练统计量（st_mean/st_std/act_mean/act_std）存在训练 h5 里（1.6 GB），
包里只带抽出来的 `models/ft_detf/norm_stats.json`（与 SimDemoData 同口径）。
**绝不能现场重算**——真机数据分布与仿真不同，重算等于换了坐标系。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

# 必须在任何 HF/transformers 导入之前：离线环境下联网重试会把启动卡死几分钟
# （实测：无 HF_HUB_OFFLINE=1 时 HEAD 请求重试 5 次阻塞启动）。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

STATE_DIM = 32
LOCK = threading.Lock()


def _load_model(project, ckpt_dir, dev):
    sys.path.insert(0, project)
    os.chdir(project)                      # base 模型路径按工作区相对路径解析
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from smolvla_finetune_sim import truncate_action_head

    policy = SmolVLAPolicy.from_pretrained("models/smolvla_base").to(dev)
    cfg = policy.config
    cfg.max_action_dim = 2
    cfg.n_action_steps = 50
    truncate_action_head(policy.model, 2)

    ck = torch.load(os.path.join(ckpt_dir, "policy.pt"), map_location="cpu",
                    weights_only=False)
    miss, unexp = policy.model.load_state_dict(ck["model"], strict=False)
    assert not unexp, f"多余键: {unexp[:5]}"
    policy.eval()
    meta = {k: ck.get(k) for k in ("n_frames", "with_lidar", "with_det",
                                   "det_features_mode", "max_state_dim",
                                   "tokenizer_max_length", "chunk")}
    print(f"[模型] 已载入 {ckpt_dir}（step={ck.get('step')}）"
          f" n_frames={meta['n_frames']} lidar={meta['with_lidar']} "
          f"det={meta['with_det']}({meta['det_features_mode']})")
    print(f"[模型] strict=False 未命中的键 {len(miss)} 个（expected）")
    return policy, meta


class DataShim:
    """SimDemoData 的最小替身：只保留推理需要的归一化统计与 tokenize。"""

    def __init__(self, stats_path, tok_len, dev):
        import torch
        s = json.load(open(stats_path))
        self.st_mean = torch.tensor(s["st_mean"], device=dev)
        self.st_std = torch.tensor(s["st_std"], device=dev)
        self.act_mean = np.array(s["act_mean"], dtype=np.float32)
        self.act_std = np.array(s["act_std"], dtype=np.float32)
        self.tok_len = tok_len
        self.device = dev
        self._tok = None
        print(f"[统计] st_mean[:3]={np.round(s['st_mean'][:3],3).tolist()} "
              f"act_mean={np.round(s['act_mean'],3).tolist()}")

    def tokenize(self, texts):
        import torch
        if self._tok is None:
            from transformers import AutoTokenizer
            # 与 SimDemoData 同一来源（HF 缓存的 fast tokenizer；
            # 本地 base 目录里的慢速 SP 转换需要 sentencepiece，反而不可用）
            from smolvla_sim_data import VLM_NAME
            self._tok = AutoTokenizer.from_pretrained(VLM_NAME)
            if self._tok.pad_token is None:
                self._tok.pad_token = self._tok.eos_token
        fixed = [t if t.endswith("\n") else t + "\n" for t in texts]
        enc = self._tok(fixed, padding="max_length", max_length=self.tok_len,
                        truncation=True, return_tensors="pt")
        return (enc["input_ids"].to(self.device),
                enc["attention_mask"].to(self.device).bool())


def make_handler(policy, data, meta, dev, project, detector="real", config=None):
    sys.path.insert(0, project)
    import torch
    from smolvla_finetune_sim import IMG_KEYS, STATE_KEY

    max_state_dim = int(meta["max_state_dim"] or STATE_DIM)
    n_frames = int(meta["n_frames"] or 1)
    use_det = bool(meta["with_det"])
    det_mode = str(meta["det_features_mode"] or "none")
    real_det = False
    if use_det and detector == "real":
        # 真机口径感知：robot.yaml 标定 HSV refs → backfill_real 检测器
        # （与训练真机帧 det_feat 的算法**完全相同**——仿真 detect() 的
        # KD-tree 参考色是仿真渲染模型，真机相机图像不匹配，绝不能混用）。
        from sim.perception import features_from_dets  # noqa: F401  (handler 内使用)
        sys.path.insert(0, os.path.join(project, "scripts", "real"))
        import backfill_real as RP
        _refs, _tol, _ovr, _hfov, _cam_h, use_refs = RP.load_cfg(config)
        print(f"[感知] 真机口径 detector=real：检测色={list(use_refs.keys())} "
              f"tol={_tol} 覆盖={_ovr} hfov={_hfov} cam_h={_cam_h} "
              f"（robot.yaml: {config}）")
        real_det = True
    elif use_det:
        from sim.perception import target_features  # 仿真口径（仅评测用）

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):        # 静默逐请求日志
            pass

        def do_POST(self):
            t0 = time.time()
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            try:
                img = np.frombuffer(bytes.fromhex(req["image_bgr"]), np.uint8)
                h, w = int(req["h"]), int(req["w"])
                img_bgr = img.reshape(h, w, 3)
                img = img_bgr[:, :, ::-1]                       # BGR→RGB（训练口径）
                scan = np.asarray(req["scan"], np.float32)
                st_in = np.asarray(req["state"], np.float32)[:5]  # [x,y,θ,v,ω]
                # 训练口径 state 是 5 维原始量（pad 到 max_state_dim 发生在归一化后），
                # 客户端若按旧契约发 32 维也兼容（只取前 5 维）。
                instr = str(req["instruction"])

                # 感知特征：真机帧必须走 robot.yaml HSV 检测器（与训练 det_feat 同口径）
                if use_det and real_det:
                    beam_step = 360.0 / max(1, len(scan))

                    def lidar_at(bdeg, _s=scan, _st=beam_step):
                        i = int(round((-bdeg) % 360.0 / _st)) % len(_s)
                        lo, hi = max(0, i - 1), min(len(_s), i + 2)
                        seg = _s[lo:hi]
                        seg = seg[np.isfinite(seg) & (seg < 12.0)]
                        return float(seg.min()) if seg.size else float("nan")

                    blobs = RP.detect_blobs_bboxes(img_bgr, use_refs, _tol, _ovr,
                                                   _hfov, _cam_h)
                    dets = RP.to_detections(blobs, lidar_beam_deg=lidar_at)
                    dfeat = np.asarray(
                        features_from_dets(dets, instr, mode=det_mode), np.float32)
                elif use_det:
                    dfeat = target_features(img, instr, lidar=scan,
                                            pose=st_in[:3], mode=det_mode)
                    dfeat = np.asarray(dfeat, np.float32)
                else:
                    dfeat = None

                st = np.concatenate([st_in, scan]) if meta["with_lidar"] else st_in
                if dfeat is not None:
                    st = np.concatenate([st, dfeat])
                t = torch.from_numpy(st.astype(np.float32)).to(dev)[None]
                t = (t - data.st_mean) / data.st_std
                t = torch.nn.functional.pad(t, (0, max_state_dim - t.shape[1]))

                imgd = {IMG_KEYS[k]: torch.from_numpy(img.astype(np.float32) / 255.0)
                        .permute(2, 0, 1)[None].to(dev) for k in range(n_frames)}
                images, img_masks = policy.prepare_images(imgd)
                state_p = policy.prepare_state({STATE_KEY: t})
                lang_t, lang_m = data.tokenize([instr])
                with LOCK, torch.no_grad():   # 推理必须 no_grad（否则 numpy() 报错）
                    pred = policy.model.sample_actions(images, img_masks,
                                                       lang_t, lang_m, state_p)
                a = (pred[0, 0].float().cpu().numpy() * data.act_std
                     + data.act_mean)                     # 反归一化 → 物理量
                out = {"action": [float(a[0]), float(a[1])],
                       "latency_ms": round((time.time() - t0) * 1000, 1)}
                self.send_response(200)
            except Exception as e:                        # noqa: BLE001
                out = {"error": f"{type(e).__name__}: {e}"}
                self.send_response(500)
            body = json.dumps(out).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="VLA 工作区路径（含 sim/ 与 smolvla_*.py）")
    ap.add_argument("--ckpt", default="../models/ft_detf")
    ap.add_argument("--stats", default="../models/ft_detf/norm_stats.json")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--detector", default="real", choices=["real", "sim"],
                    help="感知口径：real=robot.yaml HSV 检测（真机部署必须）；"
                         "sim=仿真调色板检测（仅离线评测）")
    ap.add_argument("--config", default=None,
                    help="robot.yaml 路径（detector=real 时必读；默认 <project>/deliver_v1/config/robot.yaml）")
    a = ap.parse_args()
    if a.config is None:
        a.config = os.path.join(a.project, "deliver_v1", "config", "robot.yaml")

    here = os.path.dirname(os.path.abspath(__file__))
    a.ckpt = a.ckpt if os.path.isabs(a.ckpt) else os.path.normpath(os.path.join(here, a.ckpt))
    a.stats = a.stats if os.path.isabs(a.stats) else os.path.normpath(os.path.join(here, a.stats))

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy, meta = _load_model(a.project, a.ckpt, dev)
    data = DataShim(a.stats, int(meta["tokenizer_max_length"] or 200), dev)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(
        policy, data, meta, dev, a.project, detector=a.detector, config=a.config))
    print(f"[服务] http://{a.host}:{a.port}/infer  设备={dev}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
