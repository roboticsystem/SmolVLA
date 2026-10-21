#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并 E11c 并行分片结果为「策略 × 场景」消融矩阵。

`eval_follow_stage.py` 可按场景分片并行跑，本脚本读取 `export/_fs/part_*.json`，
按固定顺序合并输出：
    export/follow_stage.json / export/follow_stage.md

用法：
    python scripts/merge_follow_stage.py --dir export/_fs \
        --out export/follow_stage.json --md export/follow_stage.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

SCEN_ORDER = ["normal", "short_occl", "long_occl", "light_shift", "narrow",
               "dim", "backlight", "sunset", "gloss", "shadow", "plain_wall", "crowd"]
POL_ORDER = ["reactive", "stage_s", "stage_l1", "stage_l2", "stage_full"]


def sort_key(r):
    sc = SCEN_ORDER.index(r["scenario"]) if r["scenario"] in SCEN_ORDER else len(SCEN_ORDER)
    po = POL_ORDER.index(r["policy"]) if r["policy"] in POL_ORDER else len(POL_ORDER)
    return (sc, po)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="export/_fs")
    ap.add_argument("--out", default="export/follow_stage.json")
    ap.add_argument("--md", default="export/follow_stage.md")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "part_*.json")))
    if not files:
        print("没有找到分片文件：%s/part_*.json" % a.dir, file=sys.stderr)
        return 1
    rows, bench, episodes, max_steps = [], None, 20, 300
    for f in files:
        with open(f, encoding="utf-8") as fh:
            blob = json.load(fh)
        # `part_bench.json` 只是微基准跑批（1 回合），**不能混进消融矩阵**
        if "bench" not in os.path.basename(f):
            rows.extend(blob.get("rows", []))
        if blob.get("replan_bench"):
            bench = blob["replan_bench"]
        episodes = blob.get("episodes", episodes)
        max_steps = blob.get("max_steps", max_steps)
    rows.sort(key=sort_key)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"episodes": episodes, "max_steps": max_steps,
                   "rows": rows, "replan_bench": bench}, f, ensure_ascii=False, indent=2)

    lines = ["# E11c 跟随阶段机 + 三级脱困（%d 回合/格）" % episodes, "",
             "| 场景 | 策略 | TCR % | 误差均值 (m) | 误差标准差 (m) | 可见率 % | "
             "丢失次数 | 恢复中位 (ms) | 碰撞 | follow % | static % | lost % | search % | "
             "L1 次数 | L2 次数 | L3 次数 | 重规划中位 (ms) | 重规划 p90 (ms) | 重规划失败 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    nan = float("nan")
    for r in rows:
        sp = r.get("stage_pct", {})
        lines.append(
            "| %s | %s | %.1f | %.2f | %.2f | %.1f | %.2f | %.0f | %.0f | "
            "%.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f |" % (
                r["scenario"], r["policy"], 100 * r["TCR"], r["err_mean"], r["err_std"],
                100 * r["vis_rate"], r["n_loss_mean"], r["rec_ms_median"], r["n_collide"],
                sp.get("follow", nan), sp.get("static", nan),
                sp.get("lost", nan), sp.get("search", nan),
                r.get("l1_n_mean", nan), r.get("l2_n_mean", nan), r.get("l3_n_mean", nan),
                r.get("replan_ms_median", nan), r.get("replan_ms_p90", nan),
                r.get("replan_fail_mean", nan)))
    if bench:
        lines += ["", "## 局部重规划（L3）延迟微基准", "",
                  "- 样本 n = %d，场景 %s" % (bench["n"], bench["scenario"]),
                  "- 单次局部 A* 重规划耗时：均值 **%.2f ms**、中位 **%.2f ms**、"
                  "p90 **%.2f ms**、p99 **%.2f ms**、最大 %.2f ms" % (
                      bench["ms_mean"], bench["ms_median"], bench["ms_p90"],
                      bench["ms_p99"], bench["ms_max"]),
                  "- 规划失败（规划不出路径）：**%d/%d**" % (bench["fail"], bench["n"])]
    with open(a.md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("合并 %d 个分片 / %d 行 -> %s / %s" % (len(files), len(rows), a.out, a.md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
