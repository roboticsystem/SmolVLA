#!/usr/bin/env python3
"""backfill_real.py —— 真机数据回流：ep_success 回填 + det_feat 重算。

与训练管线口径对齐的三件事：
1. ep_success 按人工判定回填（--success "回合:值,..."）
2. ep_instr 重写为训练同款句式（训练词表：去{英文色}的{箱子/椅子...}旁边）
3. det_feat 用「真机 HSV 检测（robot.yaml 标定 refs）→ features_from_dets」重算
   —— 注意：不能用仿真 detect()（KD-tree 参考色是仿真渲染模型，真机图像不匹配），
   也绝不能用 L2/L3 之外的方式偏离训练分布。

用法：
    python3 backfill_real.py --h5 deliver_v1/tools/data_recorded/session_red_02.h5 \
        --config deliver_v1/config/robot.yaml \
        --success "0:1,1:0,2:1,3:0,4:1" \
        --rewrite-instr "去red的箱子旁边"
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # 项目根
from sim.perception import Detection, features_from_dets  # noqa: E402


def load_cfg(path):
    import yaml
    cfg = yaml.safe_load(open(path)) or {}
    det = cfg.get("detection", {})
    refs = {k: tuple(v) for k, v in det.get("refs", {}).items()}
    tol = tuple(det.get("hsv_tol", [12, 90, 90]))
    overrides = {k: tuple(v) for k, v in det.get("hsv_tol_overrides", {}).items()}
    cam = cfg.get("camera", {})
    hfov = float(cam.get("hfov_deg", 69.4))
    cam_h = float(cam.get("cam_h", 0.08))
    measured = [k for k, v in refs.items() if "实测" in str(v)]
    # 占位色（未实测）不参与检测，防止 false positive
    use_refs = {k: v for k, v in refs.items() if k in measured or "占位" not in str(refs.get(k))}
    return refs, tol, overrides, hfov, cam_h, use_refs


def detect_blobs_bboxes(img_bgr, refs, tol, overrides, hfov_deg, cam_h,
                        min_px=150, max_range=12.0):
    """真机 HSV 检测（perception_smoke.detect_blobs 的扩展版：带 bbox + 每色容差）。

    Returns list[dict]: color, bbox(x,y,w,h), bearing_deg, d_mono_m
    """
    h, w = img_bgr.shape[:2]
    f = (w / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    horizon = h / 2.0
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    out = []
    for name, ref in refs.items():
        t = overrides.get(name, tol)
        lo = np.array([max(0, ref[0] - t[0]), max(0, ref[1] - t[1]), max(0, ref[2] - t[2])], np.uint8)
        hi = np.array([min(179, ref[0] + t[0]), min(255, ref[1] + t[1]), min(255, ref[2] + t[2])], np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            npx = int(stats[i, cv2.CC_STAT_AREA])
            if npx < min_px:
                continue
            x, y, bw, bh = stats[i, :4]
            r1 = int(y + bh)
            d = f * cam_h / max(1.0, (r1 - horizon)) if r1 > horizon else float("nan")
            d = float(np.clip(d, 0.15, max_range)) if np.isfinite(d) else float("nan")
            cx = cent[i][0]
            ang = np.degrees(np.arctan2(cx - (w - 1) / 2.0, f))
            out.append({"color": name, "n_px": npx, "bbox": (x, y, bw, bh),
                        "bearing_deg": ang, "d_mono_m": d})
    return sorted(out, key=lambda z: -z["n_px"])


def to_detections(blobs, lidar_beam_deg, max_range=12.0):
    """blob 字典 → sim.perception.Detection（与训练特征提取同一数据类）。

    dist：优先单目几何（与仿真 sweep_det 策略的「距离来源」同口径）；
    备选：该方位 ±3° 内的激光最小值；都不可用则 max_range（当作没测到深度）。
    cls 固定 "box"（与训练词表 _CLS_CN2KEY["箱子"] 对齐，指令须写「箱子」）。
    """
    dets = []
    for b in blobs:
        d = b["d_mono_m"]
        if not np.isfinite(d) and lidar_beam_deg is not None:
            d = lidar_beam_deg(b["bearing_deg"])
        if not np.isfinite(d):
            d = max_range
        x, y, bw, bh = b["bbox"]
        dets.append(Detection(bearing=np.deg2rad(b["bearing_deg"]), dist=float(d),
                              cls="box", color=b["color"],
                              height=float(bh), half=float(bw) / 2.0,
                              n_px=b["n_px"]))
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--config", default="deliver_v1/config/robot.yaml")
    ap.add_argument("--success", default="",
                    help='回合成功标签 "0:1,1:0,..."（回合号从 0 起）')
    ap.add_argument("--rewrite-instr", default="",
                    help='把 ep_instr 重写为训练同款句式，如 "去red的箱子旁边"')
    ap.add_argument("--det-mode", default="none", choices=["none", "ref", "resolve"],
                    help="与训练 meta 的 det_features_mode 一致（ft_detf=none，4 维）")
    a = ap.parse_args()

    refs_all, tol, overrides, hfov, cam_h, use_refs = load_cfg(a.config)
    print(f"[config] 检测色={list(use_refs.keys())} tol={tol} 覆盖={overrides} "
          f"hfov={hfov} cam_h={cam_h}")

    success_map = {}
    if a.success:
        for kv in a.success.split(","):
            k, v = kv.split(":")
            success_map[int(k)] = float(v)

    lid_beam_deg = None          # 从 h5 里取一帧激光建「方位→距离」查找
    with h5py.File(a.h5, "r+") as f:
        imgs = f["images"]
        n = imgs.shape[0]
        offs = f["ep_offset"][...]
        e = len(offs) - 1
        instrs = [x if isinstance(x, str) else x.decode() for x in f["ep_instr"][...]]

        # 激光方位查找表（beam 间隔 3°，车头为 0 逆时针）
        scan0 = f["lidar"][n // 2]
        beam_step = 360.0 / len(scan0)

        def lidar_at(bearing_deg):
            i = int(round((-bearing_deg) % 360.0 / beam_step)) % len(scan0)
            lo, hi = max(0, i - 1), min(len(scan0), i + 2)
            seg = scan0[lo:hi]
            seg = seg[np.isfinite(seg) & (seg < 12.0)]
            return float(seg.min()) if seg.size else float("nan")

        det_buf = np.zeros((n, 4), np.float32)
        found = 0
        for i in range(n):
            frame = imgs[i]                      # RGB（采集时已转）
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            blobs = detect_blobs_bboxes(bgr, use_refs, tol, overrides, hfov, cam_h)
            dets = to_detections(blobs, lidar_beam_deg=lidar_at)
            ep = int(np.searchsorted(offs, i, side="right") - 1)
            det_buf[i] = features_from_dets(dets, instrs[ep], mode=a.det_mode)
            found += int(det_buf[i][0] > 0.5)
            if i % 500 == 0:
                print(f"  {i}/{n}", flush=True)

        hit = 100.0 * found / n
        print(f"[det_feat] 目标检出率 {hit:.1f}%（{found}/{n} 帧）| 维度 {det_buf.shape[1]}"
              f" | mode={a.det_mode}")
        if hit < 20:
            print("[WARN] 检出率过低：确认目标物颜色已标定（robot.yaml refs），"
                  "且不是白色/低饱和物体")

        if "det_feat" in f:
            del f["det_feat"]
        f.create_dataset("det_feat", data=det_buf)
        print(f"[OK] det_feat 已写入 ({det_buf.shape[0]}, {det_buf.shape[1]})")

        if success_map:
            su = f["ep_success"][...]
            for k, v in success_map.items():
                if 0 <= k < e:
                    su[k] = v
            f["ep_success"][...] = su
            print(f"[OK] ep_success 回填: {list(su)}")

        if a.rewrite_instr:
            f["ep_instr"][...] = np.asarray([a.rewrite_instr] * e,
                                            dtype=h5py.special_dtype(vlen=str))
            print(f"[OK] ep_instr 重写: {a.rewrite_instr!r} × {e}")

    print("复验：python3 deliver_v1/tools/verify_dataset.py --h5", a.h5)


if __name__ == "__main__":
    main()
