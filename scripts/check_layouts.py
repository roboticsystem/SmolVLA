#!/usr/bin/env python3
"""check_layouts.py —— 固定户型的**可解性体检**（连通性 / 目标可达 / 指令层级覆盖）。

为什么需要它：新加的场景（教室/办公室）最容易踩的坑不是「画得不像」，
而是「**目标根本到不了**」或「**指令只有 L1、L2/L3 采样不出来**」——
这类问题在跑基准之前就能查出来，比跑完 100 回合才发现塌陷便宜得多。

检查项：
  1. **连通性**：膨胀后的规划栅格里，出生点所在连通分量覆盖多少自由格
     （若 <95% 说明场景被家具切断成几块）。
  2. **目标可达**：每个语义目标是否落在出生点的测地距离场内（不可达 → 任务无效）。
  3. **指令层级覆盖**：L1/L2/L3 三级各自能采样出多少条**无歧义**表述
     （这决定该场景能不能做分级指令实验）。
  4. **可解性**：用测地场上界策略跑 N 回合的成功率（应显著高于随机）。

用法：
    .venv_smolvla/bin/python scripts/check_layouts.py --layouts classroom,office
    .venv_smolvla/bin/python scripts/check_layouts.py --layouts all --episodes 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.baselines import make_policy                   # noqa: E402
from sim.env import calibrated_env                     # noqa: E402
from sim.world import LAYOUTS                          # noqa: E402

LAYOUT_STEPS = {"apartment": 350, "apartment_s": 300, "apartment_l": 450,
                 "corridor": 700, "s_corridor": 700, "warehouse": 650,
                 "classroom": 600, "office": 600}


def check_connectivity(env):
    """膨胀后的规划栅格里，出生点所在连通分量的自由格占比。"""
    from collections import deque
    grid = env.plan_grid                      # True = 障碍
    H, W = grid.shape
    sc = env.scene
    r, c = sc.w2c(env.pose[:2])
    r0 = int(np.clip(r, 0, H - 1))
    c0 = int(np.clip(c, 0, W - 1))
    free_total = int((~grid).sum())
    if grid[r0, c0]:
        return {"free_ratio": 0.0, "free_total": free_total, "comp": 0}
    seen = np.zeros_like(grid)
    q = deque([(r0, c0)])
    seen[r0, c0] = True
    comp = 0
    while q:
        r, c = q.popleft()
        comp += 1
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not grid[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return {"free_ratio": comp / max(free_total, 1),
            "free_total": free_total, "comp": comp}


def check_reachability(env):
    """每个语义目标是否与出生点连通（目标自身是障碍，要看它周围的环带）。

    注意：测地距离场是在**膨胀后的规划栅格**上算的，物体自身的格必然是 −1，
    直接查 `field[obj_cell]` 会把所有目标都误判为「不可达」。
    正确做法：看物体足迹外围 0.25 m 的环带里有没有可达的格。
    """
    field = env._shortest_path_len(env.pose[:2])
    sc = env.scene
    bad = []
    ring = 5                                   # 5 格 = 0.25 m
    for o in sc.objects:
        r0, c0, r1, c1 = sc.cell_bbox(o.xy, o.half)
        r0 = max(0, r0 - ring); c0 = max(0, c0 - ring)
        r1 = min(field.shape[0], r1 + ring); c1 = min(field.shape[1], c1 + ring)
        if not (field[r0:r1, c0:c1] >= 0).any():
            bad.append([o.cls, o.color, round(float(o.xy[0]), 2),
                        round(float(o.xy[1]), 2)])
    return bad


def check_levels(env, seeds=40):
    """L1/L2/L3 三级各自能采样出多少条无歧义表述。"""
    from collections import Counter
    cnt = Counter()
    samples = {}
    for s in range(seeds):
        env.reset(seed=1000 + s)
        for lv, text, _ in (env.task.get("phrasings") or []):
            cnt[lv] += 1
            samples.setdefault(lv, text)
    return dict(cnt), samples


def run(layout, policy_name, episodes, seed0=9000, stride=37):
    env = calibrated_env(layout=layout, max_steps=LAYOUT_STEPS.get(layout, 400))
    pol = make_policy(policy_name)
    ok = 0
    for ep in range(episodes):
        obs = env.reset(seed=seed0 + ep * stride)
        pol.reset(env)
        done, i = False, 0
        for i in range(env.cfg["max_steps"]):
            a = pol.act(obs, env)
            obs, _r, done, info = env.step(a)
            if done:
                break
        ok += int(bool(info["success"]))
    return 100.0 * ok / max(episodes, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", default="all")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--policy", default="geodesic")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    layouts = list(LAYOUTS.keys()) if a.layouts == "all" else a.layouts.split(",")
    out = []
    print(f"{'布局':<12s} {'连通比':>7s} {'不可达目标':>10s} "
          f"{'L1':>4s} {'L2':>4s} {'L3':>4s}  {'%s SR%%' % a.policy:>12s}")
    for lay in layouts:
        env = calibrated_env(layout=lay, max_steps=LAYOUT_STEPS.get(lay, 400))
        env.reset(seed=1000)
        conn = check_connectivity(env)
        bad = check_reachability(env)
        lv, samples = check_levels(env)
        sr = run(lay, a.policy, a.episodes)
        out.append({"layout": lay, "connectivity": conn, "unreachable": bad,
                    "levels": lv, "samples": samples, "SR": sr})
        print(f"{lay:<12s} {conn['free_ratio'] * 100:6.1f}% {len(bad):>10d} "
              f"{lv.get('L1', 0):>4d} {lv.get('L2', 0):>4d} {lv.get('L3', 0):>4d} "
              f"{sr:>11.1f}%", flush=True)
        if bad:
            print(f"    不可达目标：{bad[:6]}")
        print(f"    指令样例：" + " | ".join(f"{k}:{v}" for k, v in samples.items()))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("已写入", a.out)


if __name__ == "__main__":
    main()
