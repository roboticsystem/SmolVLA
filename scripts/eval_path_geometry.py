#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**E1c · 路径几何质量评价**（ADO 安全裕度 / κmax / 总转角 / 最小通行间隙）。

背景（`docs/03_结果与诊断/E13_参考论文对照与欠账清单.md` §2.4）：
我方此前只有**任务量**（SR/SPL/NE/TL），没有**路径几何量**——
「画面抖动率」「速度指令平滑度」与「路径本身好不好」之间缺解释链条。
参考论文1 §4.4 定义了两个几何指标并做三方对比：

    ADO（平均避障安全裕度：路径离散点到最近障碍边界的平均距离）
    kappa_max（最大曲率，m^-1）
    其结果：A* 15.4 / 5.86，MAT 79.2 / 4.82，本文 81.8 / 1.21

本脚本在我方测试床上做**同构对比**，对象为三条策略：
`geodesic`（窗口测地场）、`astar`（经典 A* + 净空代价 + 视线捷径）、
`reactive`（检测 + PID 直连，无全局规划）。

同时输出**规划路径**（A* 的 A* 输出路径）与**执行轨迹**（机器人实际走过的路）
两套几何量：前者回答「规划器给出的路径好不好」，后者回答「控制后实际走的路好不好」。

用法：
    python scripts/eval_path_geometry.py --layouts apartment,classroom,office,corridor \
        --policies geodesic,astar,reactive --episodes 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.baselines import make_policy                       # noqa: E402
from sim.env import LAYOUT_STEPS, calibrated_env, navigate_metrics   # noqa: E402
from sim.world import CELL                                  # noqa: E402


# ------------------------------------------------------------------ 几何量
def obstacle_distance_map(scene):
    """**到最近障碍的距离图**（米）：多源 BFS，缓存到 scene 上。

    口径与论文1 的「到最近障碍边界的距离」一致（论文1 在渲染图上量的是像素，
    本实现量的是米）。
    """
    cache = getattr(scene, "_obs_dist_map", None)
    if cache is not None:
        return cache
    occ = scene.occ
    H, W = occ.shape
    dist = np.full((H, W), -1, dtype=np.int32)
    q = deque()
    ys, xs = np.nonzero(occ)
    for y, x in zip(ys, xs):
        dist[y, x] = 0
        q.append((int(y), int(x)))
    while q:
        y, x = q.popleft()
        d = dist[y, x] + 1
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and dist[ny, nx] < 0:
                dist[ny, nx] = d
                q.append((ny, nx))
    out = dist.astype(np.float64) * CELL
    out[dist < 0] = np.inf
    scene._obs_dist_map = out
    return out


def path_clearances(scene, pts):
    """路径各点到最近障碍的距离（米）。"""
    dmap = obstacle_distance_map(scene)
    H, W = dmap.shape
    out = []
    for p in pts:
        r, c = scene.w2c(np.asarray(p, dtype=float))
        r, c = int(np.floor(r)), int(np.floor(c))
        if not (0 <= r < H and 0 <= c < W):
            out.append(0.0)
            continue
        out.append(float(dmap[r, c]))
    return np.asarray(out, dtype=float)


def menger_curvature(pts):
    """离散曲率（m^-1）：三点 Menger 曲率 4*Area/(a*b*c)。"""
    P = np.asarray(pts, dtype=float)
    if len(P) < 3:
        return np.zeros(0)
    a = P[1:-1] - P[:-2]
    b = P[2:] - P[1:-1]
    c = P[2:] - P[:-2]
    la = np.linalg.norm(a, axis=1)
    lb = np.linalg.norm(b, axis=1)
    lc = np.linalg.norm(c, axis=1)
    area = 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    denom = la * lb * lc
    kappa = np.where(denom > 1e-9, 4.0 * area / np.maximum(denom, 1e-9), 0.0)
    return kappa


def path_metrics(scene, pts):
    """一组路径的几何质量指标。"""
    P = np.asarray(pts, dtype=float)
    if len(P) < 2:
        return {"ADO": float("nan"), "min_clear": float("nan"),
                "kappa_max": float("nan"), "kappa_mean": float("nan"),
                "turn_total": float("nan"), "length": float("nan"),
                "n_pts": int(len(P))}
    clr = path_clearances(scene, P)
    kappa = menger_curvature(P)
    d = np.diff(P, axis=0)
    seg = np.linalg.norm(d, axis=1)
    ang = np.arctan2(d[:, 1], d[:, 0])
    dang = np.diff(np.unwrap(ang))
    return {
        "ADO": float(np.mean(clr)),
        "min_clear": float(np.min(clr)),
        "kappa_max": float(np.max(kappa)) if len(kappa) else float("nan"),
        "kappa_mean": float(np.mean(kappa)) if len(kappa) else float("nan"),
        "turn_total": float(np.sum(np.abs(dang))),
        "length": float(np.sum(seg)),
        "n_pts": int(len(P)),
    }


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser(description="E1c 路径几何质量评价")
    ap.add_argument("--layouts", default="apartment,classroom,office,corridor")
    ap.add_argument("--policies", default="geodesic,astar,reactive")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--level", default="mixed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed0", type=int, default=9000)
    ap.add_argument("--stride", type=int, default=37)
    ap.add_argument("--out", default="export/path_geometry.json")
    ap.add_argument("--md", default="export/path_geometry.md")
    a = ap.parse_args()

    layouts = [x for x in a.layouts.split(",") if x]
    policies = [x for x in a.policies.split(",") if x]
    rows = []
    t0 = time.time()
    for lay in layouts:
        for pol in policies:
            env = calibrated_env(layout=lay,
                                max_steps=LAYOUT_STEPS.get(lay, 350))
            ex, pl, sr, coll, steps = [], [], [], [], []
            for ep in range(a.episodes):
                s = a.seed0 + ep * a.stride
                obs = env.reset(seed=s, level=a.level)
                policy = make_policy(pol)
                policy.reset(env)
                info = {}
                for i in range(env.cfg["max_steps"]):
                    act = policy.act(obs, env)
                    obs, r, done, info = env.step(act)
                    if done:
                        break
                m = navigate_metrics(info, env.cfg["success_radius"])
                traj = np.asarray(env.traj, dtype=float)
                ex.append(path_metrics(env.scene, traj))
                sr.append(m["SR"])
                coll.append(info["n_collide"])
                steps.append(i + 1)
                pth = getattr(policy, "_path", None)
                if pth is not None and len(pth) >= 2:
                    pl.append(path_metrics(env.scene, np.asarray(pth, dtype=float)))
            agg_exec = {k: float(np.nanmean([e[k] for e in ex])) for k in ex[0]}
            agg_exec["kappa_max"] = float(np.nanmax([e["kappa_max"] for e in ex]))
            agg_exec["min_clear"] = float(np.nanmin([e["min_clear"] for e in ex]))
            rec = {"layout": lay, "policy": pol, "exec": agg_exec,
                   "plan": ({k: float(np.nanmean([e[k] for e in pl]))
                             for k in pl[0]} if pl else None),
                   "SR": float(np.mean(sr)), "n_collide": float(np.mean(coll)),
                   "steps": float(np.mean(steps)), "episodes": a.episodes}
            rows.append(rec)
            print(f"{lay:<12s}{pol:<10s} 执行 ADO {agg_exec['ADO']:.3f} m  "
                  f"min_clear {agg_exec['min_clear']:.3f} m  "
                  f"kappa {agg_exec['kappa_max']:.3f}  "
                  f"转角 {agg_exec['turn_total']:.2f} rad  "
                  f"SR {100*rec['SR']:.0f}%  碰撞 {rec['n_collide']:.0f}", flush=True)
            if rec["plan"]:
                print(f"{'':22s}规划 ADO {rec['plan']['ADO']:.3f} m  "
                      f"min_clear {rec['plan']['min_clear']:.3f} m  "
                      f"kappa {rec['plan']['kappa_max']:.3f}  "
                      f"转角 {rec['plan']['turn_total']:.2f} rad", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"episodes": a.episodes, "level": a.level, "rows": rows}, f,
                  ensure_ascii=False, indent=2)

    lines = [f"# E1c 路径几何质量（{a.episodes} 回合/格，level={a.level}）", "",
             "## 执行轨迹（机器人实际走过的路）", "",
             "| 布局 | 策略 | ADO (m) | min_clear (m) | kappa_max | 总转角 (rad) | SR % | 碰撞 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        e = r["exec"]
        lines.append(f"| {r['layout']} | {r['policy']} | {e['ADO']:.3f} | "
                     f"{e['min_clear']:.3f} | {e['kappa_max']:.3f} | "
                     f"{e['turn_total']:.2f} | {100*r['SR']:.0f} | "
                     f"{r['n_collide']:.0f} |")
    lines += ["", "## 规划路径（A* 的 A* 输出路径）", "",
              "| 布局 | 策略 | ADO (m) | min_clear (m) | kappa_max | 总转角 (rad) |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        if not r["plan"]:
            continue
        p = r["plan"]
        lines.append(f"| {r['layout']} | {r['policy']} | {p['ADO']:.3f} | "
                     f"{p['min_clear']:.3f} | {p['kappa_max']:.3f} | "
                     f"{p['turn_total']:.2f} |")
    with open(a.md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n耗时 {time.time()-t0:.1f}s，已写出 {a.out} 与 {a.md}")


if __name__ == "__main__":
    main()
