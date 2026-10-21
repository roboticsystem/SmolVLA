#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**E4 消融重跑**（当前代码口径）——分片执行 + 合并。

为什么必须重跑（2026-10-21 发现）：
`export/e4_ablation.json` 是 **2026-09-24** 的产物，而 commit `eaddbf4`
（2026-10-20「P1 场景族与退化几何 + P2 动态障碍/动力学退化/深度观测」）
**改了场景的光照/材质实现**：`dim` 从「单一光度旋钮」变成
`exposure=0.45` 的真实曝光几何，`light_shift/backlight/sunset/gloss/shadow`
新增眩光/镜面/侧光几何 ⇒ **旧消融数字与当前代码不再对应**
（实测旧表 `gamma|dim`=79.2%，当前代码下同配置约 11.7%）。
因此旧表**不能**与本轮 E4b 敏感性表并列，必须在当前代码上重跑。

口径与旧表一致（便于新旧对照）：
  · `sim.follow.run_e4_ablation` 的同一套模块开关（`E4_ABLATION`）
  · `FollowEnv(seed=0, want_dist=1.5, max_steps=300, camera_augment=0.25)`
  · 回合种子 `3000 + 17·ep`
  · 默认 30 回合/格

用法（分片并行）：
    for SC in normal short_occl long_occl light_shift narrow dim backlight; do
      python scripts/eval_e4_ablation.py --scenarios $SC --episodes 30 \
        --out export/_e4/$SC.json &
    done; wait
    python scripts/eval_e4_ablation.py --merge export/_e4 \
      --out export/e4_ablation.json --md export/e4_ablation.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.follow import E4_ABLATION, FollowEnv, FollowPolicyE4  # noqa: E402

OLD_SCENARIOS = ["normal", "short_occl", "long_occl", "light_shift",
                 "narrow", "dim", "backlight"]
NEW_SCENARIOS = ["sunset", "gloss", "shadow", "plain_wall", "crowd"]


def run_cell(kw, sc, episodes, max_steps=300):
    env = FollowEnv(seed=0, want_dist=1.5, max_steps=max_steps,
                    scenario=sc, camera_augment=0.25)
    pol = FollowPolicyE4(**kw)
    rows = []
    for ep in range(episodes):
        obs = env.reset(seed=3000 + ep * 17)
        pol.reset(env)
        while True:
            obs, r, done, info = env.step(pol.act(obs, env))
            if done:
                break
        rows.append(env.metrics())
    return {"TCR": float(np.mean([m["TCR"] for m in rows])),
            "err_mean": float(np.mean([m["err_mean"] for m in rows]))
            if "err_mean" in rows[0] else None,
            "vis_rate": float(np.mean([m["vis_rate"] for m in rows])),
            "rec_ms_median": float(np.mean([m["rec_ms_median"] for m in rows])),
            "n_collide": float(np.mean([m["n_collide"] for m in rows]))}


def main():
    ap = argparse.ArgumentParser(description="E4 消融（当前代码口径，可分片）")
    ap.add_argument("--scenarios", default=",".join(OLD_SCENARIOS))
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--out", default="export/e4_ablation.json")
    ap.add_argument("--md", default="export/e4_ablation.md")
    ap.add_argument("--merge", default=None,
                    help="分片目录：合并 export/_e4/*.json 为一张表")
    a = ap.parse_args()

    if a.merge:
        rows = {}
        for f in sorted(glob.glob(os.path.join(a.merge, "*.json"))):
            rows.update(json.load(open(f, encoding="utf-8")).get("rows", {}))
        scens = sorted({k.split("|")[1] for k in rows})
        pols = [p for p in E4_ABLATION if any(k.startswith(p + "|") for k in rows)]
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        lines = [f"# E4 场景化增强消融（当前代码重跑，{a.episodes} 回合/格）", "",
                 "| 模块 | " + " | ".join(f"{s} TCR %" for s in scens) + " |",
                 "|---|" + "---|" * len(scens)]
        for p in pols:
            cells = []
            for s in scens:
                k = f"{p}|{s}"
                cells.append(f"{100*rows[k]['TCR']:.1f}" if k in rows else "—")
            lines.append(f"| {p} | " + " | ".join(cells) + " |")
        with open(a.md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"已合并 {len(rows)} 格 → {a.out} / {a.md}")
        return

    out = {}
    t0 = time.time()
    for sc in [x for x in a.scenarios.split(",") if x]:
        for name, kw in E4_ABLATION.items():
            rec = run_cell(kw, sc, a.episodes, a.max_steps)
            out[f"{name}|{sc}"] = rec
            print(f"{sc:12s} {name:9s} TCR {100*rec['TCR']:5.1f}%  "
                  f"可见 {100*rec['vis_rate']:5.1f}%  恢复 {rec['rec_ms_median']:7.0f} ms  "
                  f"碰撞 {rec['n_collide']:5.1f}", flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"episodes": a.episodes, "rows": out}, f, ensure_ascii=False, indent=2)
    print(f"-- 完成（{time.time()-t0:.0f}s）→ {a.out}")


if __name__ == "__main__":
    main()
