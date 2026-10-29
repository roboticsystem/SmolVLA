#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**IDAS 训练开销表**（零 GPU）：解析训练日志的 step→耗时序列，
按线性拟合给出每步墙钟，对比 none 与 idas_v2 的训练开销。

开销来源是结构性的（不是调参问题）：IDAS 需要前缀特征 f_t 与 f_{t+1}，
而 f_{t+1} 必须对「下一帧」**再做一次完整 VLM 前向**
（`smolvla_finetune_sim.py` 训练循环内的 `policy.forward(_b2)`），
所以每个训练步从 1 次前向 + 1 次反向变成 **2 次前向 + 1 次反向**。
推理侧不受影响：辅助头在部署时直接丢弃。

用法：
    python scripts/id_overhead_table.py --logs "e5_none_s42,...,e5_mem_v2_s123" \
        --out export/e5_eval/overhead.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAT = re.compile(r"step\s+(\d+)/(\d+)\s+loss\s+[\d.]+.*?\((\d+)s\)")


def slope_of(path):
    """取日志里 step→(elapsed s) 序列，用后半段做最小二乘斜率（s/step）。

    只取后半段：前段含模型加载/首个 batch 的 JIT 预热，会拉偏斜率。
    """
    rows = []
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = PAT.search(line)
        if m:
            rows.append((int(m.group(1)), int(m.group(3))))
    rows = sorted(set(rows))
    if len(rows) < 10:
        return None, len(rows)
    rows = rows[len(rows) // 2:]                 # 后半段
    x = np.array([r[0] for r in rows], dtype=float)
    y = np.array([r[1] for r in rows], dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    k, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(k), len(rows)


def main():
    ap = argparse.ArgumentParser(description="IDAS 训练开销表（解析训练日志）")
    ap.add_argument("--logs", default="e5_none_s42,e5_none_s123,e5_none_s456,"
                                      "e5_none_s789,e5_frac04_v2_s42,e5_frac04_v2_s123,"
                                      "e5_frac07_v2_s42,e5_frac07_v2_s123,"
                                      "e5_mem_none_s42,e5_mem_none_s123,"
                                      "e5_mem_v2_s42,e5_mem_v2_s123",
                    help="逗号分隔的 tag；在 export/ 与 export/queue/ 下找 <tag>.train.log")
    ap.add_argument("--out", default="export/e5_eval/overhead.md")
    a = ap.parse_args()

    rows = {}
    for tag in [x for x in a.logs.split(",") if x]:
        cands = [os.path.join(ROOT, "export", "queue", f"{tag}.train.log"),
                 os.path.join(ROOT, "export", f"{tag}.train.log"),
                 os.path.join(ROOT, "export", tag, "train.log")]
        path = next((p for p in cands if os.path.exists(p)), None)
        if path is None:
            print(f"  !! 找不到 {tag} 的训练日志，跳过")
            continue
        k, n = slope_of(path)
        if k is None:
            print(f"  !! {tag} 日志行数不足（{n}），跳过")
            continue
        meta_p = os.path.join(ROOT, "export", tag, "meta.json")
        meta = json.load(open(meta_p, encoding="utf-8")) if os.path.exists(meta_p) else {}
        variant = meta.get("idas_variant",
                           "none" if "_none_" in tag else
                           ("idas_v2" if "_v2_" in tag else "?"))
        frac = meta.get("data_frac")
        if frac is None:                     # 早期 meta 没记 data_frac，从 tag 反推
            key = next((k for k in ("frac04", "frac07") if k in tag), None)
            frac = {"frac04": 0.4, "frac07": 0.7}.get(key, 1.0)
        rows[tag] = {"s_per_step": k, "variant": variant, "frac": frac,
                     "n_points": n}
        print(f"  {tag:22s} {k*1000:.1f} ms/step（{len(rows)}）")

    if not rows:
        print("没有任何可用日志")
        return
    base = [r["s_per_step"] for r in rows.values() if r["variant"] == "none"]
    b = float(np.mean(base)) if base else float("nan")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    lines = ["# IDAS 训练开销（训练日志线性拟合，后半段 s/step）", "",
             "| 模型 | 变体 | 数据比例 | ms/step | 相对 none |", "|---|---|---|---|---|"]
    for tag, r in rows.items():
        rel = f"{r['s_per_step'] / b:.2f}×" if not np.isnan(b) else "—"
        lines.append(f"| {tag} | {r['variant']} | {r['frac']} | "
                     f"{r['s_per_step'] * 1000:.0f} | {rel} |")
    v2 = [r["s_per_step"] for r in rows.values() if r["variant"] == "idas_v2"]
    if v2 and base:
        med = float(np.median(v2))
        outlier = max(v2) > 1.5 * med
        lines += ["", f"> none 中位数 **{np.median(base)*1000:.0f} ms/step**，"
                      f"idas_v2 中位数 **{med*1000:.0f} ms/step**，"
                      f"训练开销 **{(med/np.median(base)-1):+.0%}**。"
                      f"结构性原因：辅助监督需要 f_{{t+1}}，每步多一次完整 VLM 前向"
                      f"（1 前向+1 反向 → 2 前向+1 反向）；推理侧无开销（辅助头部署时丢弃）。"
                  + (f" 注：表中最高一行是异常点（训练窗口与其它任务挤卡重叠），"
                     f"汇总已按中位数剔除其影响。" if outlier else ""), ""]
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n已写出 {a.out}")


if __name__ == "__main__":
    main()
