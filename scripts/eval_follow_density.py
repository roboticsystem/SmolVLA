#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**E11b · 人群拥挤度分层 × 遮挡 的二维矩阵（仿真侧）**。

背景（`E13_参考论文对照与欠账清单.md` §3.2）：
参考论文4 §2.1 用「**半径 r 内局部占用率 × 意图权重**」量化拥堵度，
并把拥堵度作为规划代价的一部分。我方的动态行人目前只是「障碍物」，
E11 的「5 类干扰」是**定性分类**，缺一个可量化的自变量。

本脚本把 E11 的干扰维度从「5 类」升级为
**人群密度（稀疏/中/密集，按半径内行人数分档）× 遮挡（无/短/长）** 的二维矩阵，
并用论文4 的口径计算**拥挤度指标**：

    ρ(t) = r 内行人数量 / r 内可站立面积占比   （本实现用「r 内人数」与「占据率」两种）

同时统计跟随质量：TCR、跟随误差均值/标准差、可见率、丢失次数、恢复时间、碰撞。

用法：
    python scripts/eval_follow_density.py --peds 0,1,2,4 \
        --scenarios normal,short_occl,long_occl --episodes 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.follow import (FollowEnv, FollowPolicyE4,            # noqa: E402
                        crowding, place_peds)


def run(scenario, n_peds, episodes, policy_kw, radius, occlusion="none",
        seed0=3000, stride=17):
    env = FollowEnv(seed=0, scenario=scenario, n_peds=max(1, n_peds),
                    camera_augment=0.25, max_steps=300)
    pol = FollowPolicyE4(**policy_kw)
    rows, rho = [], []
    for ep in range(episodes):
        obs = env.reset(seed=seed0 + ep * stride)
        pol.reset(env)
        place_peds(env, n_peds, occlusion)
        while True:
            a = pol.act(obs, env)
            c = crowding(env, radius)
            rho.append(c)
            obs, r, done, info = env.step(a)
            if done:
                break
        rows.append(env.metrics())
    agg = {k: float(np.mean([m[k] for m in rows])) for k in rows[0]}
    agg["n_in_r_mean"] = float(np.mean([c["n_in_r"] for c in rho]))
    agg["area_frac_mean"] = float(np.mean([c["area_frac"] for c in rho]))
    agg["n_loss_mean"] = float(np.mean([m["n_loss"] for m in rows]))
    return agg


def main():
    ap = argparse.ArgumentParser(description="E11b 拥挤度分层 × 遮挡 矩阵")
    ap.add_argument("--peds", default="0,1,2,4")
    ap.add_argument("--scenarios", default="normal")
    ap.add_argument("--occlusions", default="side,partial,full",
                    help="受控干扰：side/partial/full（侧向 45° / 半遮挡 25° / 正对视线）")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--radius", type=float, default=2.0,
                    help="拥挤度统计半径（米）；论文4 用 2.5 m")
    ap.add_argument("--out", default="export/follow_density.json")
    ap.add_argument("--md", default="export/follow_density.md")
    a = ap.parse_args()

    peds = [int(x) for x in a.peds.split(",") if x]
    scens = [x for x in a.scenarios.split(",") if x]
    occs = [x for x in a.occlusions.split(",") if x]
    rows = []
    for sc in scens:
      for occ in occs:
        for np_ in peds:
            agg = run(sc, np_, a.episodes, {}, a.radius, occlusion=occ)
            rec = {"scenario": sc, "occlusion": occ, "n_peds": np_, **agg}
            rows.append(rec)
            print(f"{sc:<10s} {occ:<8s} 行人 {np_}  半径内 {agg['n_in_r_mean']:.2f}  "
                  f"占据率 {agg['area_frac_mean']:.2f}  TCR {100*agg['TCR']:5.1f}%  "
                  f"误差 {agg['err_mean']:.2f}±{agg['err_std']:.2f} m  "
                  f"可见 {100*agg['vis_rate']:5.1f}%  丢失 {agg['n_loss_mean']:.2f}  "
                  f"恢复 {agg['rec_ms_median']:.0f} ms  碰撞 {agg['n_collide']:.0f}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"episodes": a.episodes, "radius_m": a.radius, "rows": rows}, f,
                  ensure_ascii=False, indent=2)

    lines = [f"# E11b 人群拥挤度分层 × 遮挡（{a.episodes} 回合/格，统计半径 {a.radius} m）", "",
             "| 场景 | 遮挡 | 行人 | 半径内人数 | 占据率 | TCR % | 误差均值 (m) | "
             "误差标准差 (m) | 可见率 % | 丢失次数 | 恢复 (ms) | 碰撞 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['scenario']} | {r['occlusion']} | {r['n_peds']} | {r['n_in_r_mean']:.2f} | "
                     f"{r['area_frac_mean']:.2f} | {100*r['TCR']:.1f} | {r['err_mean']:.2f} | "
                     f"{r['err_std']:.2f} | {100*r['vis_rate']:.1f} | {r['n_loss_mean']:.2f} | "
                     f"{r['rec_ms_median']:.0f} | {r['n_collide']:.0f} |")
    with open(a.md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n已写出 {a.out} 与 {a.md}")


if __name__ == "__main__":
    main()
