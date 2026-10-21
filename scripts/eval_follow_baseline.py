#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""刷新「表 6 · E11 视觉跟随」的 `reactive` / `oracle` 基线（**当前代码**口径）。

起因（2026-10-21）：2026-10-20 的场景光照/深度改动使 2026-09 的跟随表失效
（`light_shift` 67.8% → 65.5%，`dim` 场景语义完全改变），
与 E4 消融表同源问题（见 `E14_本轮实验结果汇总.md` §7.2）。

口径与旧表一致：30 回合/格、`FollowEnv(seed=0, want_dist=1.5, max_steps=300,
camera_augment=0.25)`、种子 `3000 + 17·ep`。

用法：
    python scripts/eval_follow_baseline.py --scenarios normal,short_occl \
        --out export/follow_baseline.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.follow import run  # noqa: E402

SCENARIOS = ["normal", "short_occl", "long_occl", "light_shift", "narrow",
             "dim", "backlight"]


def main():
    ap = argparse.ArgumentParser(description="E11 跟随基线（reactive/oracle，当前代码）")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--policies", default="reactive,oracle")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--out", default="export/follow_baseline.json")
    a = ap.parse_args()

    out = {}
    for sc in [x for x in a.scenarios.split(",") if x]:
        for pol in [x for x in a.policies.split(",") if x]:
            res = run(policy_name=pol, scenario=sc, episodes=a.episodes,
                      seed_base=3000, stride=17, want_dist=1.5,
                      max_steps=a.max_steps, verbose=False)
            agg = res["agg"]
            out[f"{pol}|{sc}"] = {
                "TCR": agg["TCR"], "err_mean": agg["err_mean"],
                "err_std": agg.get("err_std"),
                "bearing_mean_deg": agg.get("bearing_mean_deg"),
                "vis_rate": agg["vis_rate"],
                "n_loss_mean": agg.get("n_loss"),
                "loss_mean_ms": agg.get("loss_mean_ms"),
                "rec_ms_median": agg["rec_ms_median"],
                "rec_ms_p90": agg.get("rec_ms_p90"),
                "n_collide": agg.get("n_collide"),
            }
            print(f"{sc:12s} {pol:9s} TCR {100*agg['TCR']:5.1f}%  "
                  f"err {agg['err_mean']:.2f}±{agg.get('err_std', 0):.2f} m  "
                  f"可见 {100*agg['vis_rate']:5.1f}%  丢失 {agg.get('n_loss', 0):.2f}  "
                  f"恢复 {agg['rec_ms_median']:6.0f} ms  碰撞 {agg.get('n_collide', 0):5.1f}",
                  flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"episodes": a.episodes, "rows": out}, f, ensure_ascii=False, indent=2)
    print(f"已写出 {a.out}")


if __name__ == "__main__":
    main()
