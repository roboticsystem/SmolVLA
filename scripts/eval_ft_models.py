#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**微调模型的通用 200 回合闭环评测**（主创新 A 的模型级对照）。

为什么需要这个脚本：
`scripts/real/eval_ctl.py` 的模型组合是**硬编码**的（4 个 tag），
无法指定新模型目录；而各模型 `meta.json` 里那个 8 回合的小样本
`closed_loop_sr`（0.125 / 0.250）**统计功效完全不足**，不能用来排序模型。
本脚本按 §4.19 的权威口径（标定测试床、seed 9000+37·ep、200 回合、
`max_steps=200`、`calib=True`）重评任意模型，并输出逐回合明细以供**配对 McNemar**。

**特征档位自动跟随模型**：`det_features_mode` 从各模型的 `meta.json` 读取
（`none` 4 维 / `mem` 8 维），避免「用错档位评记忆模型」这类口径错误。

用法：
    python scripts/eval_ft_models.py \
        --models ft_detf:export/ft_detf,ft_detf_mem96k:export/ft_detf_mem96k \
        --episodes 200 --batch 8 --out export/ft_models.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "real"))

import torch  # noqa: E402
from smolvla_finetune_sim import (DEV, SmolVLAPolicy, SimDemoData,  # noqa: E402
                                  closed_loop_sr_batch, reconfigure_for_diff_drive)


def detect_mode(h5_path):
    """由演示数据的 `det_feat` 宽度推断特征档位：4 → `none`，8 → `mem`。"""
    import h5py
    with h5py.File(h5_path, "r") as f:
        for k in f.keys():
            if "det" in k.lower() or "feat" in k.lower():
                w = int(f[k].shape[1])
                return "mem" if w >= 8 else "none"
    return "none"


def load_policy(policy_dir):
    ck = torch.load(os.path.join(policy_dir, "policy.pt"), map_location=DEV)
    policy = SmolVLAPolicy.from_pretrained(
        os.path.join(ROOT, "models", "smolvla_base")).to(DEV)
    policy.eval()
    cfg = policy.config
    chunk = int(ck.get("chunk", 50))
    cfg.chunk_size = chunk
    cfg.n_action_steps = chunk
    reconfigure_for_diff_drive(policy, action_dim=2,
                               n_cam=int(ck.get("n_frames", 1)), state_dim=None)
    policy.model.load_state_dict(ck["model"], strict=False)
    policy.eval()
    return policy, cfg, ck


def main():
    ap = argparse.ArgumentParser(description="微调模型 200 回合闭环评测")
    ap.add_argument("--models", required=True,
                    help="逗号分隔的 tag:model_dir[:data_h5]")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="export/ft_models.json")
    ap.add_argument("--md", default="export/ft_models.md")
    a = ap.parse_args()

    res = {}
    for spec in [x for x in a.models.split(",") if x]:
        parts = spec.split(":")
        tag = parts[0]
        mdir = parts[1]
        meta = json.load(open(os.path.join(mdir, "meta.json"), encoding="utf-8"))
        data_h5 = parts[2] if len(parts) > 2 else meta.get("data")
        # **特征档位从 h5 的 `det_feat` 宽度自动推断**（4 维 → none，8 维 → mem）。
        # 不能只信 `meta["det_features_mode"]`：早期训练的模型根本没记这个字段
        #（读到 None），照用会把「记忆档模型」按「单帧档」评，
        # 结果是归一化维度不匹配（9 vs 13）或指标被系统性低估。
        dmode = detect_mode(data_h5)
        print(f"\n=== {tag} ({mdir}) | data={data_h5} | 档位={dmode} ===", flush=True)

        policy, cfg, ck = load_policy(mdir)
        data = SimDemoData(data_h5, device=DEV,
                           max_state_dim=ck.get("max_state_dim"),
                           tokenizer_max_length=ck.get("tokenizer_max_length"),
                           chunk_size=ck.get("chunk", 50), max_action_dim=2,
                           n_frames=1, with_lidar=False, with_det=True)
        det = []
        t0 = time.time()
        sr, spl, ne = closed_loop_sr_batch(
            policy, data, episodes=a.episodes, seed=a.seed, max_steps=a.max_steps,
            n_frames=1, visible_start=False, detail=det, batch=a.batch,
            det_mode=dmode, mem_ttl=int(meta.get("mem_ttl", 25) or 25), calib=True)
        res[tag] = {"model_dir": mdir, "data": data_h5, "det_features_mode": dmode,
                    "SR": float(sr), "SPL": float(spl), "NE": float(ne),
                    "episodes": a.episodes, "detail": det}
        print(f"  >>> {tag}: SR {100*sr:.2f}%  SPL {spl:.3f}  NE {ne:.2f} m  "
              f"({time.time()-t0:.0f}s)", flush=True)
        del policy, data
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    lines = [f"# 微调模型 200 回合闭环评测（seed {a.seed}，标定测试床，"
             f"max_steps={a.max_steps}）", "",
             "| 模型 | 特征档位 | 训练数据 | SR % | SPL | NE (m) |",
             "|---|---|---|---|---|---|"]
    for tag, r in res.items():
        lines.append(f"| {tag} | {r['det_features_mode']} | "
                     f"`{os.path.basename(str(r['data']))}` | "
                     f"{100*r['SR']:.2f} | {r['SPL']:.3f} | {r['NE']:.2f} |")
    with open(a.md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n已写出 {a.out} 与 {a.md}")


if __name__ == "__main__":
    main()
