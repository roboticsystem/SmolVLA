#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**E11c · 跟随阶段机 + 三级脱困策略的代价统计**（仿真侧）。

背景（`E13_参考论文对照与欠账清单.md` §3.3）：
E11 把「长时遮挡 >3 s 且目标离场返回」当成**一个干扰类**，但它其实是**另一个任务阶段**——
两种情形下「跟随误差」的口径根本不同（目标在视野内 vs 目标已丢失）。
论文4 表 3-5 用**阶段触发规则**驱动状态机自动切换，各阶段给各自判据；
论文3 §2.4 用**三级脱困**（速度调节 → 原地等待 → 局部重规划）处理「动不了」。

本脚本实现 `sim/follow.py: FollowPolicyStage`，并统计：
  1. **四阶段**（follow / static / lost / search）的时长占比与切换次数；
  2. **三级脱困**各自的触发次数、累计时长、占比；
  3. **L3 局部重规划的耗时**（ms，均值/中位/p90）与失败次数；
  4. 阶段机 + 脱困**是否真的提升跟随质量**（与 `reactive` 基线逐项对照）。

对照口径（**E4 模块全部关闭**，保证只比较「阶段机/脱困」这一个变量）：
  reactive      —— 基线（反应式跟随，无阶段机、无脱困）
  stage_s      —— **仅**阶段机（`use_replan=False`）
  stage_full   —— 阶段机 + 三级脱困（`use_replan=True`）

用法：
    python scripts/eval_follow_stage.py --scenarios normal,short_occl,long_occl \
        --episodes 20 --out export/follow_stage.json --md export/follow_stage.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.follow import STAGE_ORDER, FollowEnv, ReactiveFollow, FollowPolicyStage  # noqa: E402


def build_policy(name):
    """消融组合（**逐级开启**，保证只变一个变量）：

    reactive    —— 基线：无阶段机、无脱困
    stage_s     —— 仅阶段机（四阶段）
    stage_l1    —— 阶段机 + L1 速度调节
    stage_l2    —— 阶段机 + L1 + L2 原地等待
    stage_full  —— 阶段机 + L1 + L2 + L3 局部重规划
    """
    if name == "reactive":
        return ReactiveFollow()
    if name == "stage_s":
        return FollowPolicyStage(use_l1=False, use_l2=False, use_l3=False,
                                 use_replan=False, use_stage=True)
    if name == "stage_l1":
        return FollowPolicyStage(use_l1=True, use_l2=False, use_l3=False,
                                 use_replan=False, use_stage=True)
    if name == "stage_l2":
        return FollowPolicyStage(use_l1=True, use_l2=True, use_l3=False,
                                 use_replan=False, use_stage=True)
    if name == "stage_full":
        return FollowPolicyStage(use_l1=True, use_l2=True, use_l3=True,
                                 use_replan=True, use_stage=True)
    raise ValueError(f"未知策略：{name}")


def run(scenario, policy_name, episodes, seed0=3000, stride=17, max_steps=300):
    env = FollowEnv(seed=0, scenario=scenario, camera_augment=0.25, max_steps=max_steps)
    pol = build_policy(policy_name)
    rows, stages = [], []
    t0 = time.time()
    for ep in range(episodes):
        obs = env.reset(seed=seed0 + ep * stride)
        pol.reset(env)
        while True:
            obs, r, done, info = env.step(pol.act(obs, env))
            if done:
                break
        rows.append(env.metrics())
        if hasattr(pol, "stats"):
            stages.append(pol.stats)
    agg = {k: float(np.mean([m[k] for m in rows])) for k in rows[0]}
    agg["n_loss_mean"] = float(np.mean([m["n_loss"] for m in rows]))
    agg["wall_s"] = time.time() - t0

    out = {"scenario": scenario, "policy": policy_name, **agg}
    if stages:
        tot = sum(sum(s["stage_time"].values()) for s in stages)
        out["stage_pct"] = {
            st: (100.0 * sum(s["stage_time"][st] for s in stages) / max(tot, 1e-9))
            for st in STAGE_ORDER}
        out["stage_trans_mean"] = {
            st: float(np.mean([s["stage_trans"][st] for s in stages]))
            for st in STAGE_ORDER}
        out["l1_n_mean"] = float(np.mean([s["l1"]["n"] for s in stages]))
        out["l2_n_mean"] = float(np.mean([s["l2"]["n"] for s in stages]))
        out["l3_n_mean"] = float(np.mean([s["l3"]["n"] for s in stages]))
        out["l1_t_mean"] = float(np.mean([s["l1"]["t"] for s in stages]))
        out["l2_t_mean"] = float(np.mean([s["l2"]["t"] for s in stages]))
        out["l3_t_mean"] = float(np.mean([s["l3"]["t"] for s in stages]))
        rms = [x for s in stages for x in s["replan_ms"]]
        out["replan_ms_mean"] = float(np.mean(rms)) if rms else 0.0
        out["replan_ms_median"] = float(np.median(rms)) if rms else 0.0
        out["replan_ms_p90"] = float(np.percentile(rms, 90)) if rms else 0.0
        out["replan_fail_mean"] = float(np.mean([s["replan_fail"] for s in stages]))
    return out


def bench_replan(scenario="normal", n=200, seed=1234, max_steps=300):
    """**局部重规划（L3）延迟微基准**。

    为什么需要单独测：L3 是「连续卡住 ≥ `replan_after` 秒」才触发的兜底，
    在当前仿真测试床上**几乎不触发**（差速底盘总能原地旋转脱困，
    实测 L3 触发 0 次）⇒ 无法靠自然触发拿到样本。
    但论文必须给出**重规划耗时**这个代价数字（论文3 §2.4 的「重规划耗时」列），
    因此单独做微基准：在随机位姿上对**可见目标**做一次局部 A\\* 重规划，
    统计单次耗时分布。

    返回：耗时均值/中位/p90/p99 与失败率（规划不出路径的比例）。
    """
    from sim.planner import AStarPolicy

    env = FollowEnv(seed=0, scenario=scenario, camera_augment=0.25, max_steps=max_steps)
    planner = AStarPolicy(plan_clearance=0.15, clear_cost=2.0)
    rng = np.random.default_rng(seed)
    ms, fail = [], 0
    for i in range(n):
        env.reset(seed=7000 + i * 13)
        th = float(rng.uniform(-np.pi, np.pi))
        env.pose = np.array([env.pose[0], env.pose[1], th], dtype=float)
        d = float(rng.uniform(0.8, 2.5))
        goal = np.asarray(env.pose[:2], dtype=float) + \
            d * np.array([np.cos(th), np.sin(th)])
        t0 = time.perf_counter()
        try:
            path = planner._astar(env, goal)
        except Exception:
            path = None
        ms.append((time.perf_counter() - t0) * 1000.0)
        if not path:
            fail += 1
    ms = np.asarray(ms, dtype=float)
    return {"n": int(n), "scenario": scenario, "fail": int(fail),
            "ms_mean": float(ms.mean()), "ms_median": float(np.median(ms)),
            "ms_p90": float(np.percentile(ms, 90)),
            "ms_p99": float(np.percentile(ms, 99)), "ms_max": float(ms.max())}


def main():
    ap = argparse.ArgumentParser(description="E11c 跟随阶段机 + 三级脱困")
    ap.add_argument("--scenarios", default="normal,short_occl,long_occl")
    ap.add_argument("--policies", default="reactive,stage_s,stage_l1,stage_l2,stage_full")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--out", default="export/follow_stage.json")
    ap.add_argument("--md", default="export/follow_stage.md")
    ap.add_argument("--bench-n", type=int, default=200,
                    help="局部重规划延迟微基准样本数（0 = 不跑）")
    ap.add_argument("--bench-scenario", default="normal")
    ap.add_argument("--tag", default="",
                    help="分片标签：写入文件名（配合 scripts/merge_follow_stage.py）")
    a = ap.parse_args()
    if a.tag:
        a.out = f"export/_fs/part_{a.tag}.json"
        a.md = f"export/_fs/part_{a.tag}.md"

    scens = [x for x in a.scenarios.split(",") if x]
    pols = [x for x in a.policies.split(",") if x]
    bench = bench_replan(a.bench_scenario, a.bench_n) if a.bench_n > 0 else None
    if bench:
        print(f"[微基准] 局部重规划延迟 (n={bench['n']}, {bench['scenario']}): "
              f"均值 {bench['ms_mean']:.2f} ms | 中位 {bench['ms_median']:.2f} ms | "
              f"p90 {bench['ms_p90']:.2f} ms | p99 {bench['ms_p99']:.2f} ms | "
              f"规划失败 {bench['fail']}/{bench['n']}", flush=True)
    rows = []
    for sc in scens:
        for p in pols:
            rec = run(sc, p, a.episodes, max_steps=a.max_steps)
            rows.append(rec)
            extra = ""
            if "stage_pct" in rec:
                sp = rec["stage_pct"]
                extra = (f" | follow {sp['follow']:.0f}% static {sp['static']:.0f}% "
                         f"lost {sp['lost']:.0f}% search {sp['search']:.0f}%"
                         f" | L1 {rec['l1_n_mean']:.1f} L2 {rec['l2_n_mean']:.1f} "
                         f"L3 {rec['l3_n_mean']:.1f}"
                         f" | 重规划 {rec['replan_ms_median']:.1f} ms"
                         f"（失败 {rec['replan_fail_mean']:.1f}）")
            print(f"{sc:<12s} {p:<11s} TCR {100*rec['TCR']:5.1f}%  "
                  f"err {rec['err_mean']:.2f}±{rec['err_std']:.2f} m  "
                  f"可见 {100*rec['vis_rate']:5.1f}%  "
                  f"丢失 {rec['n_loss_mean']:.2f}  "
                  f"恢复中位 {rec['rec_ms_median']:6.1f} ms  "
                  f"碰撞 {rec['n_collide']:.0f}{extra}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"episodes": a.episodes, "max_steps": a.max_steps,
                   "rows": rows, "replan_bench": bench}, f, ensure_ascii=False, indent=2)

    head = ("| 场景 | 策略 | TCR % | 误差均值 (m) | 误差标准差 (m) | 可见率 % | "
            "丢失次数 | 恢复中位 (ms) | 碰撞 | follow % | static % | lost % | search % | "
            "L1 次数 | L2 次数 | L3 次数 | 重规划中位 (ms) | 重规划 p90 (ms) | 重规划失败 |")
    lines = [f"# E11c 跟随阶段机 + 三级脱困（{a.episodes} 回合/格）", "", head,
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sp = r.get("stage_pct", {})
        lines.append(
            f"| {r['scenario']} | {r['policy']} | {100*r['TCR']:.1f} | "
            f"{r['err_mean']:.2f} | {r['err_std']:.2f} | {100*r['vis_rate']:.1f} | "
            f"{r['n_loss_mean']:.2f} | {r['rec_ms_median']:.0f} | {r['n_collide']:.0f} | "
            f"{sp.get('follow', float('nan')):.1f} | {sp.get('static', float('nan')):.1f} | "
            f"{sp.get('lost', float('nan')):.1f} | {sp.get('search', float('nan')):.1f} | "
            f"{r.get('l1_n_mean', float('nan')):.1f} | {r.get('l2_n_mean', float('nan')):.1f} | "
            f"{r.get('l3_n_mean', float('nan')):.1f} | "
            f"{r.get('replan_ms_median', float('nan')):.1f} | "
            f"{r.get('replan_ms_p90', float('nan')):.1f} | "
            f"{r.get('replan_fail_mean', float('nan')):.1f} |")
    with open(a.md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n已写出 {a.out} 与 {a.md}")


if __name__ == "__main__":
    main()
