#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**E4b · 场景化增强超参敏感性**（表 G1d-2）。

背景：E13 欠账清单里 E4b 一直挂着「⬜ 待做（需训练 GPU 时间）」，
但复核后发现 **E4 的三个模块（自适应增强 α / 空间约束因子 γ / 卡尔曼外推 λ_id）
全部在「策略层」**，不涉及模型训练 ⇒ **纯仿真即可完成，不需要 GPU、更不需要真机**。

扫三个超参（每个只改一个，其余保持模块默认开关状态）：

| 超参 | 含义 | 取值 |
|---|---|---|
| **α** `enhance_strength` | 自适应图像增强强度（gamma + 百分位拉伸的力度） | 0.5 / 1.0 / 1.5 / 2.0 |
| **γ** `gamma_clear_lo` | 空间约束因子的净空减速起点（米） | 0.3 / 0.5 / 0.7 / 0.9 |
| **λ_id** `kf_q` | 卡尔曼外推的过程噪声（越大越信自己的外推） | 0.15 / 0.35 / 0.60 / 0.90 |

口径：与 E4 消融完全一致（`FollowEnv`、`camera_augment=0.25`、`max_steps=300`、
种子 `3000 + 17·ep`），故本表与 `export/e4_ablation.json` 可直接并列。

用法：
    python scripts/eval_e4b_sensitivity.py --episodes 20 \
        --scenarios normal,light_shift,dim,crowd \
        --out export/e4b_sensitivity.json --md export/e4b_sensitivity.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.follow import FollowEnv, FollowPolicyE4  # noqa: E402

# 每个超参：(名称, 所属模块开关, 参数名, 取值列表)
AXES = [
    ("alpha", {"use_enhance": True}, "enhance_strength", [0.5, 1.0, 1.5, 2.0]),
    ("gamma", {"use_gamma": True}, "gamma_clear_lo", [0.3, 0.5, 0.7, 0.9]),
    ("lambda_id", {"use_kalman": True}, "kf_q", [0.15, 0.35, 0.60, 0.90]),
]


def run_one(kw, scenario, episodes, seed_base=3000, stride=17, max_steps=300):
    env = FollowEnv(seed=0, scenario=scenario, camera_augment=0.25, max_steps=max_steps)
    pol = FollowPolicyE4(**kw)
    rows = []
    for ep in range(episodes):
        obs = env.reset(seed=seed_base + ep * stride)
        pol.reset(env)
        while True:
            obs, r, done, info = env.step(pol.act(obs, env))
            if done:
                break
        rows.append(env.metrics())
    return {"TCR": float(np.mean([m["TCR"] for m in rows])),
            "err_mean": float(np.mean([m["err_mean"] for m in rows])),
            "vis_rate": float(np.mean([m["vis_rate"] for m in rows])),
            "rec_ms_median": float(np.mean([m["rec_ms_median"] for m in rows])),
            "n_collide": float(np.mean([m["n_collide"] for m in rows]))}


def main():
    ap = argparse.ArgumentParser(description="E4b 场景化增强超参敏感性")
    ap.add_argument("--scenarios", default="normal,light_shift,dim,crowd")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--out", default="export/e4b_sensitivity.json")
    ap.add_argument("--md", default="export/e4b_sensitivity.md")
    ap.add_argument("--resume", action="store_true",
                    help="读取 --out 中已有格子并跳过，只补跑缺失项（配合增量落盘）")
    a = ap.parse_args()

    scens = [x for x in a.scenarios.split(",") if x]
    out = {}
    _extra = {}
    if a.resume and os.path.exists(a.out):
        _old = json.load(open(a.out, encoding="utf-8"))
        out = _old.get("rows", {})
        _extra = {k: v for k, v in _old.items() if k not in ("rows", "episodes")}
        print(f"[resume] 已有 {len(out)} 格，跳过已完成项")
    t0 = time.time()
    for axis, mod, pname, vals in AXES:
        for sc in scens:
            for v in vals:
                key = f"{axis}|{pname}={v}|{sc}"
                if key in out:
                    continue
                kw = dict(mod)
                kw[pname] = v
                rec = run_one(kw, sc, a.episodes, max_steps=a.max_steps)
                out[key] = rec
                print(f"{axis:<9s} {pname}={v:<5} {sc:<12s} "
                      f"TCR {100*rec['TCR']:5.1f}%  可见 {100*rec['vis_rate']:5.1f}%  "
                      f"恢复 {rec['rec_ms_median']:6.0f} ms  碰撞 {rec['n_collide']:.1f}",
                      flush=True)
                # **增量落盘**：本轮扫描曾因进程被终止丢掉全部已算结果（46 格白跑），
                # 每格算完立刻写文件，kill 后最多只丢一格。
                os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
                with open(a.out, "w", encoding="utf-8") as f:
                    json.dump({"episodes": a.episodes, "rows": out, **_extra},
                              f, ensure_ascii=False, indent=2)
        print(f"-- {axis} 完成（{time.time()-t0:.0f}s）", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"episodes": a.episodes, "rows": out, **_extra},
                  f, ensure_ascii=False, indent=2)

    lines = [f"# E4b 场景化增强超参敏感性（{a.episodes} 回合/格，与 E4 消融同口径）", ""]
    for axis, mod, pname, vals in AXES:
        lines += [f"## {axis}（`{pname}`，模块开关 {mod}）", "",
                  "| 取值 | " + " | ".join(f"{sc} TCR %" for sc in scens) + " |",
                  "|---|" + "---|" * len(scens)]
        for v in vals:
            cells = [f"{100*out[f'{axis}|{pname}={v}|{sc}']['TCR']:.1f}"
                     for sc in scens]
            lines.append(f"| {v} | " + " | ".join(cells) + " |")
        lines.append("")
    with open(a.md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n已写出 {a.out} 与 {a.md}")


if __name__ == "__main__":
    main()
