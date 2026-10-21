#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_layouts.py —— 新增固定布局（窄走廊 / S 形走廊 / 仓库 / 公寓三档）
的可解性与基线基准。

用途：
- 校验新场景**连通且可解**（oracle 成功率应显著高于 random）；
- 产出「场景规模 → 成功率 / 步数」表，对齐论文4 的多规模实验口径；
- 回归监控：新增场景或改布局后重跑，发现成功率塌陷立刻能看出来。

用法:
    .venv_smolvla/bin/python scripts/bench_layouts.py --episodes 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.baselines import make_policy          # noqa: E402
from sim.env import calibrated_env, LAYOUT_STEPS   # 步数预算的单一来源在 sim/env.py            # noqa: E402




def run(layout, policy_name, episodes, seed0=9000, stride=37):
    env = calibrated_env(layout=layout, max_steps=LAYOUT_STEPS.get(layout, 350))
    pol = make_policy(policy_name)
    rows = []
    for ep in range(episodes):
        obs = env.reset(seed=seed0 + ep * stride)
        pol.reset(env)
        done, i = False, 0
        for i in range(env.cfg["max_steps"]):
            a = pol.act(obs, env)
            obs, _r, done, info = env.step(a)
            if done:
                break
        m = {"success": float(info["success"]), "steps": i + 1,
             "collide": int(info["n_collide"]),
             "level": str(info["level"]), "instr": info["instruction"]}
        rows.append(m)
    return {
        "layout": layout, "policy": policy_name, "n": len(rows),
        "SR": round(100 * float(np.mean([r["success"] for r in rows])), 1),
        "steps": round(float(np.mean([r["steps"] for r in rows])), 1),
        "collide": round(float(np.mean([r["collide"] for r in rows])), 1),
        "levels": sorted({r["level"] for r in rows}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", default="lab,apartment,apartment_s,apartment_l,"
                                         "corridor,s_corridor,warehouse,"
                                         "classroom,office")
    ap.add_argument("--policies", default="geodesic,geodesic_random,oracle,sweep,random")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = []
    print(f"{'布局':<12s} {'策略':<8s} {'n':>3s} {'SR%':>6s} {'步数':>7s} "
          f"{'碰撞':>6s}  层级")
    for lay in a.layouts.split(","):
        for pol in a.policies.split(","):
            r = run(lay, pol, a.episodes)
            out.append(r)
            print(f"{r['layout']:<12s} {r['policy']:<8s} {r['n']:>3d} "
                  f"{r['SR']:>6.1f} {r['steps']:>7.1f} {r['collide']:>6.1f}  "
                  f"{','.join(r['levels'])}", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("已写入", a.out)


if __name__ == "__main__":
    main()
