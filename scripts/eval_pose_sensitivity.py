#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**E1b · 定位误差 → 导航性能 敏感度扫描（仿真侧）**。

背景（见 `docs/03_结果与诊断/E13_参考论文对照与欠账清单.md` §2.1）：
仿真里位姿是**真值完美的**，真机不是。参考论文2 第 3/4 章把定位作为独立方法章，
给出「平均 / 最大 / RMSE 定位误差」与「离群帧抑制率」；我方此前**没有任何位姿误差数字**，
E1 的已知限制里明写「基线假设位姿已知」。

本脚本在仿真里注入**三类定位误差**（与真机误差来源一一对应），测量导航性能衰减：

| 误差来源 | 仿真参数 | 真机对应 |
|---|---|---|
| 扫描匹配白噪声 | `pose_noise_gauss`（σ，米） | 激光匹配残差 |
| 里程计尺度/打滑造成的**持久偏差** | `pose_noise_drift`（米/秒，随里程累积） | 编码器里程计漂移（**最致命**） |
| 匹配失败导致的**离群帧** | `pose_noise_jump`（概率/步） | 论文2 §3.4 的离群帧问题 |

用法：
    python scripts/eval_pose_sensitivity.py --layouts lab,apartment,corridor \
        --policies geodesic,oracle --episodes 10 --sigmas 0,0.01,0.02,0.05,0.1,0.2
    python scripts/eval_pose_sensitivity.py --mode drift --drifts 0,0.002,0.005,0.01
    python scripts/eval_pose_sensitivity.py --mode jump --jumps 0,0.02,0.05,0.1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.baselines import make_policy                       # noqa: E402
from sim.env import LAYOUT_STEPS, calibrated_env, navigate_metrics   # noqa: E402


def run_one(layout, policy_name, episodes, level, seed, env_kwargs,
            max_steps=None, seed0=9000, stride=37):
    """跑一批回合，返回聚合指标。

    **口径与 `scripts/bench_layouts.py` 完全一致**（同一标定测试床、
    同一布局步数预算、同一 `9000 + 37*ep` 种子方案），
    这样 E1b/E1c 的数字与已记录基线**可比**。
    """
    env = calibrated_env(layout=layout,
                         max_steps=max_steps or LAYOUT_STEPS.get(layout, 350),
                         **env_kwargs)
    rows = []
    for ep in range(episodes):
        s = seed0 + ep * stride
        obs = env.reset(seed=s, level=level)
        policy = make_policy(policy_name)
        policy.reset(env)
        for i in range(env.cfg["max_steps"]):
            a = policy.act(obs, env)
            obs, r, done, info = env.step(a)
            if done:
                break
        m = navigate_metrics(info, env.cfg["success_radius"])
        rows.append({**m, "steps": i + 1, "n_collide": info["n_collide"],
                     "level": info["level"], "success": bool(info["success"])})
    agg = {
        "SR": float(np.mean([r["SR"] for r in rows])),
        "SPL": float(np.nanmean([r["SPL"] for r in rows])),
        "NE": float(np.mean([r["NE"] for r in rows])),
        "TL": float(np.mean([r["TL"] for r in rows])),
        "steps": float(np.mean([r["steps"] for r in rows])),
        "n_collide": float(np.mean([r["n_collide"] for r in rows])),
        "n": len(rows),
    }
    return agg


def main():
    ap = argparse.ArgumentParser(description="E1b 定位误差敏感度扫描")
    ap.add_argument("--layouts", default="lab,apartment,corridor,s_corridor")
    ap.add_argument("--policies", default="geodesic,oracle")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--level", default="mixed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed0", type=int, default=9000)
    ap.add_argument("--stride", type=int, default=37)
    ap.add_argument("--mode", choices=["gauss", "drift", "jump"], default="gauss")
    ap.add_argument("--sigmas", default="0,0.01,0.02,0.05,0.1,0.2")
    ap.add_argument("--drifts", default="0,0.002,0.005,0.01")
    ap.add_argument("--jumps", default="0,0.02,0.05,0.1")
    ap.add_argument("--jump-mag", type=float, default=0.30)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="0 = 按布局取 LAYOUT_STEPS（与基准脚本同口径）")
    ap.add_argument("--out", default="export/pose_sensitivity.json")
    ap.add_argument("--md", default="export/pose_sensitivity.md")
    a = ap.parse_args()

    layouts = [x for x in a.layouts.split(",") if x]
    policies = [x for x in a.policies.split(",") if x]
    if a.mode == "gauss":
        levels = [("sigma", [float(x) for x in a.sigmas.split(",")])]
    elif a.mode == "drift":
        levels = [("drift", [float(x) for x in a.drifts.split(",")])]
    else:
        levels = [("jump", [float(x) for x in a.jumps.split(",")])]

    results = []
    t0 = time.time()
    for lay in layouts:
        for pol in policies:
            for lname, vals in levels:
                for v in vals:
                    env_kwargs = {"use_pose_est": True,
                                  "pose_noise_gauss": v if a.mode == "gauss" else 0.0,
                                  "pose_noise_drift": v if a.mode == "drift" else 0.0,
                                  "pose_noise_jump": v if a.mode == "jump" else 0.0,
                                  "pose_noise_jump_mag": a.jump_mag}
                    agg = run_one(lay, pol, a.episodes, a.level, a.seed, env_kwargs,
                                     max_steps=a.max_steps,
                                     seed0=a.seed0, stride=a.stride)
                    rec = {"layout": lay, "policy": pol, "mode": a.mode,
                           "level_name": lname, "level_value": v, **agg}
                    results.append(rec)
                    print(f"{lay:<13s}{pol:<10s}{lname}={v:<8g} "
                          f"SR {100*agg['SR']:5.1f}%  SPL {agg['SPL']:.3f}  "
                          f"NE {agg['NE']:5.2f}  步 {agg['steps']:5.1f}  "
                          f"碰撞 {agg['n_collide']:5.1f}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"mode": a.mode, "episodes": a.episodes, "level": a.level,
                   "seed": a.seed, "results": results}, f,
                  ensure_ascii=False, indent=2)

    # ---- markdown 表 ----
    lname = levels[0][0]
    base = {(r["layout"], r["policy"]): r["SR"] for r in results
            if r["level_value"] == levels[0][1][0]}
    lines = [f"# E1b 定位误差敏感度（mode={a.mode}，{a.episodes} 回合/格，level={a.level}）", "",
             "| 布局 | 策略 | " + lname + " | SR % | SPL | NE (m) | 步数 | 碰撞 | SR 相对无噪 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        b = base.get((r["layout"], r["policy"]), float("nan"))
        lines.append(f"| {r['layout']} | {r['policy']} | {r['level_value']:g} | "
                     f"{100*r['SR']:.1f} | {r['SPL']:.3f} | {r['NE']:.2f} | "
                     f"{r['steps']:.1f} | {r['n_collide']:.1f} | "
                     f"{100*(r['SR']-b):+.1f} pp |")
    with open(a.md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n耗时 {time.time()-t0:.1f}s，已写出 {a.out} 与 {a.md}")


if __name__ == "__main__":
    main()
