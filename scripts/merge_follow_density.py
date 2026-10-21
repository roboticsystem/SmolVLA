#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并 E11b 并行分片结果为单张拥挤度 × 遮挡矩阵。

`eval_follow_density.py` 可按 (场景, 遮挡) 分片并行跑（每片一个进程），
本脚本读取 `export/_fd/part_*.json`，按固定顺序合并输出：
    export/follow_density.json / export/follow_density.md

用法：
    python scripts/merge_follow_density.py --dir export/_fd \
        --out export/follow_density.json --md export/follow_density.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

SCEN_ORDER = ["normal", "short_occl", "long_occl"]
OCC_ORDER = ["side", "partial", "full"]


def sort_key(r):
    sc = SCEN_ORDER.index(r["scenario"]) if r["scenario"] in SCEN_ORDER else len(SCEN_ORDER)
    oc = OCC_ORDER.index(r["occlusion"]) if r["occlusion"] in OCC_ORDER else len(OCC_ORDER)
    return (sc, oc, r["n_peds"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="export/_fd")
    ap.add_argument("--out", default="export/follow_density.json")
    ap.add_argument("--md", default="export/follow_density.md")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "part_*.json")))
    if not files:
        print(f"没有找到分片文件：{a.dir}/part_*.json", file=sys.stderr)
        return 1
    rows = []
    episodes, radius = 10, 2.0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            blob = json.load(fh)
        rows.extend(blob["rows"])
        episodes = blob.get("episodes", episodes)
        radius = blob.get("radius_m", radius)
    rows.sort(key=sort_key)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"episodes": episodes, "radius_m": radius,
                   "rows": rows}, f, ensure_ascii=False, indent=2)

    lines = [f"# E11b 人群拥挤度分层 × 遮挡（{episodes} 回合/格，统计半径 "
             f"{radius} m）", "",
             "| 场景 | 遮挡 | 行人 | 半径内人数 | 占据率 | TCR % | 误差均值 (m) | "
             "误差标准差 (m) | 可见率 % | 丢失次数 | 恢复 (ms) | 碰撞 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['scenario']} | {r['occlusion']} | {r['n_peds']} | "
                     f"{r['n_in_r_mean']:.2f} | {r['area_frac_mean']:.2f} | "
                     f"{100*r['TCR']:.1f} | {r['err_mean']:.2f} | {r['err_std']:.2f} | "
                     f"{100*r['vis_rate']:.1f} | {r['n_loss_mean']:.2f} | "
                     f"{r['rec_ms_median']:.0f} | {r['n_collide']:.0f} |")
    with open(a.md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"合并 {len(files)} 个分片 / {len(rows)} 行 -> {a.out} / {a.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
