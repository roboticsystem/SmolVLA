#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**模型级泛化 / 鲁棒矩阵评测**（布局矩阵 + 退化×遮挡矩阵）。

## 为什么需要

`export/bench_layouts.json` 只覆盖**经典策略**（oracle / geodesic / sweep /
random，n=10）；E12 的鲁棒性实验此前只有跟随任务（FollowEnv / TCR）的
`eval_e4_ablation.py`，**导航 VLA 模型**的「布局 × 退化」矩阵不存在。
本脚本把两者统一成「模型 × 单元格」的闭环评测矩阵，逐格输出 200 回合口径的
SR / SPL / NE 与逐回合 `detail`（可配对 McNemar）。

## 两种模式

**layout** —— 9 布局泛化：`lab`（标定单房间）+ `LAYOUT_STEPS` 的 8 个固定户型。
环境构造与 `scripts/bench_layouts.py` 完全一致（标定基准 + `layout=` 覆盖），
`max_steps` 取该布局的步数预算（`sim/env.py` 单一来源），保证与经典策略
的数字**同一口径、可直接同表**。

**degrade** —— 退化×遮挡：在标定基准上叠加单项退化旋钮。光照/材质的数值
**逐位取自** `sim/follow.py` 的 `SCENARIO_LIGHT`（dim=exposure 0.45、
backlight=light_dir π + glare 0.55、sunset、gloss），保证与主创新 B 的
场景族同一套几何；遮挡/传感类的旋钮用导航环境原生开关
（行人数量=动态遮挡、frame_drop=短时全黑、lidar_noise/odom_drift=传感器退化、
camera_augment=光度域移）。

## 用法

    python scripts/eval_matrix.py --mode layout \
        --models none:e5_none_s42,v2:e5_idas_v2_s42 \
        --episodes 50 --out export/e5_eval/matrix_layout.json

    python scripts/eval_matrix.py --mode degrade \
        --models none:e5_none_s42,v2:e5_idas_v2_s42 \
        --episodes 50 --out export/e5_eval/matrix_degrade.json
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
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from sim.env import LAYOUT_STEPS, CALIBRATED  # noqa: E402
from smolvla_finetune_sim import closed_loop_sr_batch  # noqa: E402
from eval_ft_models import detect_mode, load_policy  # noqa: E402

import torch  # noqa: E402

# 9 布局：标定单房间（lab）+ 8 个固定户型；步数预算单一来源 = LAYOUT_STEPS。
LAYOUTS = [("lab", int(CALIBRATED.get("max_steps", 200)))] + \
          [(k, int(v)) for k, v in LAYOUT_STEPS.items()]

# 退化×遮挡单元格：数值来源见模块 docstring；每格只动**一个**旋钮，
# 保证「哪个退化杀了 SR」可归因（大纲 §7.2 的同一条纪律）。
DEGRADE = {
    "normal":      {},
    "dim":         {"exposure": 0.45},
    "backlight":   {"light_dir": np.pi, "glare": 0.55},
    "sunset":      {"light_dir": np.pi, "glare": 0.80, "gloss": 0.15, "exposure": 0.75},
    "gloss":       {"light_dir": np.pi / 2, "gloss": 0.75},
    "crowd2":      {"n_peds": 2},                     # 标定床默认 0 行人 → 2 人遮挡
    "crowd4":      {"n_peds": 4},
    "frame_drop":  {"frame_drop": 0.30},              # 30% 帧丢弃 ≈ 短时全黑
    "lidar_noise": {"lidar_noise": 0.05},
    "odom_drift":  {"odom_drift": 0.02},
}


def main():
    ap = argparse.ArgumentParser(description="模型 × 布局 / 退化 矩阵评测")
    ap.add_argument("--mode", choices=["layout", "degrade"], required=True)
    ap.add_argument("--models", required=True, help="逗号分隔 tag:model_dir")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=200,
                    help="degrade 模式的步数预算（layout 模式自动取 LAYOUT_STEPS）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()

    cells = LAYOUTS if a.mode == "layout" else \
        [(k, a.max_steps) for k in DEGRADE]
    res = {}
    for spec in [x for x in a.models.split(",") if x]:
        tag, mdir = spec.split(":")[:2]
        meta = json.load(open(os.path.join(mdir, "meta.json"), encoding="utf-8"))
        data_h5 = meta.get("data")
        dmode = detect_mode(data_h5)
        policy, cfg, ck = load_policy(mdir)
        from smolvla_sim_data import SimDemoData
        from smolvla_finetune_sim import DEV
        data = SimDemoData(data_h5, device=DEV,
                           max_state_dim=ck.get("max_state_dim"),
                           tokenizer_max_length=ck.get("tokenizer_max_length"),
                           chunk_size=ck.get("chunk", 50), max_action_dim=2,
                           n_frames=1, with_lidar=False, with_det=True)
        res[tag] = {}
        for cell, msteps in cells:
            env_over = ({"layout": cell} if a.mode == "layout"
                        else dict(DEGRADE[cell]))
            det = []
            t0 = time.time()
            sr, spl, ne = closed_loop_sr_batch(
                policy, data, episodes=a.episodes, seed=a.seed, max_steps=msteps,
                n_frames=1, visible_start=False, detail=det, batch=a.batch,
                det_mode=dmode, mem_ttl=int(meta.get("mem_ttl", 25) or 25),
                calib=True, env_over=env_over)
            res[tag][cell] = {"SR": float(sr), "SPL": float(spl), "NE": float(ne),
                              "episodes": a.episodes, "max_steps": msteps,
                              "detail": det}
            print(f"  >>> {tag} | {cell}: SR {100*sr:.1f}%  SPL {spl:.3f}  "
                  f"NE {ne:.2f}  ({time.time()-t0:.0f}s)", flush=True)
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            with open(a.out, "w", encoding="utf-8") as f:   # 逐格落盘，断点友好
                json.dump(res, f, ensure_ascii=False)
        del policy, data
        torch.cuda.empty_cache()

    md = a.md or a.out.rsplit(".", 1)[0] + ".md"
    head = "布局" if a.mode == "layout" else "退化场景"
    lines = [f"# 模型 × {head} 矩阵（每格 {a.episodes} 回合，seed {a.seed}，"
             f"标定基准 + 单项旋钮）", "",
             f"| {head} | 步数预算 | " +
             " | ".join(f"{t} SR %" for t in res) + " |",
             "|---|---|" + "---|" * len(res)]
    for cell, msteps in cells:
        row = [cell, str(msteps)]
        for tag in res:
            c = res[tag].get(cell)
            row.append(f"{100*c['SR']:.1f}" if c else "—")
        lines.append("| " + " | ".join(row) + " |")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n已写出 {a.out} 与 {md}")


if __name__ == "__main__":
    main()
