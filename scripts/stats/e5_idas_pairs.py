#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""e5_idas_pairs.py —— E5（IDAS 辅助监督）的 200 回合配对汇总。

## 为什么需要

`export/e5_*/meta.json` 里的 `closed_loop_sr` 只有 **8 回合**，方差大到
0.125↔0.375 能来回跳（16 个模型全是这个量级），**不能回答「IDAS 有没有用」**。
本脚本按主创新 A 的权威口径（`scripts/eval_ft_models.py`，200 回合、seed 9000+37·ep、
标定测试床、max_steps=200）重算，并做两件事：

1. **种子分层**：4 个训练种子的 SR 先各自算，再给均值 ± 种子间标准差
   （避免把「种子噪声」当成方法效应）；
2. **配对 McNemar**：同一批评测种子下逐回合配对，比较
   `idas_v2` vs `none` / `idas_v1` / `random_ctrl`。
   `random_ctrl` 是必需对照 —— 任何额外损失都有正则效应，没有它就无法
   把「学到了逆动力学」与「多一项损失」分开（见 smolvla_idas.py 文档）。

## 用法

    .venv_smolvla/bin/python scripts/stats/e5_idas_pairs.py \
        --inputs export/e5_eval/g1.json,export/e5_eval/g2.json \
        --md export/e5_eval/e5_summary.md

⚠ 需要 `none` 臂在同 4 个种子上训练后才有「方法 vs 基线」的配对结论；
缺哪个臂，表里对应行会显示 `—`，脚本不会用不同种子的数字硬凑。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
from scipy.stats import binomtest

MAIN = ("none", "idas_v1", "idas_v2", "random_ctrl")
# 数据效率臂（主方法 idas_v2 减数据）与 mem×IDAS 交互臂：单独成表，不混进表 1/3。
FRAC = ("frac04_v2", "frac07_v2")
MEM = ("mem_none", "mem_v2")
GROUPS = set(MAIN) | set(FRAC) | set(MEM)


def parse_tag(tag: str):
    """`e5_idas_v2_s42` → ('idas_v2', 42)；`e5_frac04_v2_s42` → ('frac04_v2', 42)；
    `e5_mem_v2_s42` → ('mem_v2', 42)；`e5_none_p02` → ('none_p02', None)。"""
    m = re.match(r"e5_frac0([47])_v2_s(\d+)$", tag)
    if m:
        return f"frac0{m.group(1)}_v2", int(m.group(2))
    m = re.match(r"e5_mem_(none|v2)_s(\d+)$", tag)
    if m:
        return f"mem_{m.group(1)}", int(m.group(2))
    m = re.match(r"e5_random_ctrl_s(\d+)$", tag)
    if m:
        return "random_ctrl", int(m.group(1))
    m = re.match(r"e5_none_s(\d+)$", tag)
    if m:
        return "none", int(m.group(1))
    m = re.match(r"e5_(idas_)?(v1|v2|p0[25]|l00?5|l02|none_p0[25])_s?(\d+)?$", tag)
    if m:
        name = ("idas_" if m.group(1) else "") + m.group(2)
        seed = int(m.group(3)) if m.group(3) else None
        return name, seed
    if tag.startswith("ft_detf"):
        return "ft_detf", None
    return tag, None


def load(rows: dict, tag: str):
    """取每个模型的 (per-episode success dict by seed, SR, SPL, NE)。"""
    r = rows[tag]
    succ = {int(d[0]): float(d[1]) for d in r["detail"]}
    return succ, float(r["SR"]), float(r["SPL"]), float(r["NE"])


def mcnemar(a: dict, b: dict):
    """配对 McNemar 精确检验；只用两边都跑过的回合（种子对齐）。"""
    keys = sorted(set(a) & set(b))
    x = np.array([a[k] for k in keys]) > 0.5
    y = np.array([b[k] for k in keys]) > 0.5
    n01 = int(np.sum(x & ~y))            # a 成、b 败
    n10 = int(np.sum(~x & y))            # a 败、b 成
    n = n01 + n10
    p = 1.0 if n == 0 else binomtest(n01, n, 0.5).pvalue
    return n01, n10, p, len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--md", default="export/e5_eval/e5_summary.md")
    a = ap.parse_args()

    rows = {}
    for p in a.inputs.split(","):
        with open(p, encoding="utf-8") as f:
            rows.update(json.load(f))

    data = {}
    for tag in rows:
        name, seed = parse_tag(tag)
        data.setdefault(name, {})[seed] = load(rows, tag)

    lines = ["# E5 · IDAS 200 回合闭环（标定测试床，seed 9000+37·ep）", ""]
    lines.append("## 表 1 主臂（4 变体 × 4 训练种子）")
    lines.append("")
    lines.append("| 变体 | 每种子 SR % | 均值±std | SPL | NE (m) |")
    lines.append("|---|---|---|---|---|")
    for name in MAIN:
        if name not in data:
            lines.append(f"| {name} | — | — | — | — |")
            continue
        srs = [100 * v[1] for s, v in sorted(data[name].items(), key=lambda z: (z[0] is None, z[0]))]
        per = " / ".join(f"{s:.1f}" for s in srs)
        mean = np.mean(srs)
        sd = np.std(srs, ddof=1) if len(srs) > 1 else float("nan")
        spl = np.mean([v[2] for v in data[name].values()])
        ne = np.mean([v[3] for v in data[name].values()])
        lines.append(f"| {name} | {per} | {mean:.1f} ± {sd:.1f} | {spl:.3f} | {ne:.2f} |")

    lines += ["", "## 表 2 配对 McNemar（同一评测种子，逐回合配对）", "",
              "| 对比 | 胜/负回合 | n | p |", "|---|---|---|---|"]
    pairs = [("idas_v2", "none"), ("idas_v2", "random_ctrl"),
             ("idas_v2", "idas_v1"), ("idas_v1", "none"),
             ("random_ctrl", "none")]
    for x_name, y_name in pairs:
        if x_name not in data or y_name not in data:
            lines.append(f"| {x_name} vs {y_name} | — | — | — |")
            continue
        seeds = sorted({s for s in data[x_name] if s in data[y_name]},
                       key=lambda z: (z is None, z))
        for s in seeds:
            n01, n10, p, n = mcnemar(data[x_name][s][0], data[y_name][s][0])
            lines.append(f"| {x_name} vs {y_name}（种子 {s}） | {n01}/{n10} | {n} | "
                         f"{'<0.001' if p < 1e-3 else f'{p:.4f}'} |")
        # 跨种子合并（回合独立，可直接拼接）
        ax, by = {}, {}
        for s in seeds:
            for k, v in data[x_name][s][0].items():
                ax[(s, k)] = v
            for k, v in data[y_name][s][0].items():
                by[(s, k)] = v
        n01, n10, p, n = mcnemar(ax, by)
        lines.append(f"| **{x_name} vs {y_name}（合并）** | {n01}/{n10} | {n} | "
                     f"{'<0.001' if p < 1e-3 else f'{p:.4f}'} |")

    lines += ["", "## 表 3 λ 与标签比例（单种子，配对仍有效）", "",
              "| 模型 | SR % | SPL | NE (m) |", "|---|---|---|---|"]
    for name in sorted(data):
        if name in GROUPS:
            continue
        for s, v in sorted(data[name].items(), key=lambda z: (z[0] is None, z[0])):
            lines.append(f"| {name}{'' if s is None else ' (s%d)' % s} | "
                         f"{100 * v[1]:.1f} | {v[2]:.3f} | {v[3]:.2f} |")

    # ---- 表 4 数据效率：同一个 idas_v2 配方，把回合内可用起点减到 70% / 40%
    if any(n in data for n in FRAC):
        lines += ["", "## 表 4 数据效率（idas_v2，100% / 70% / 40% 起始数据）", "",
                  "| 数据比例 | 种子 | SR % | SPL | NE (m) | 相对 100% 的配对 McNemar |",
                  "|---|---|---|---|---|---|"]
        for label, name in (("100%", "idas_v2"), ("70%", "frac07_v2"), ("40%", "frac04_v2")):
            if name not in data:
                lines.append(f"| {label} | — | — | — | — | — |")
                continue
            for s, v in sorted(data[name].items(), key=lambda z: (z[0] is None, z[0])):
                note = "—"
                if name != "idas_v2" and s in data.get("idas_v2", {}):
                    n01, n10, p, n = mcnemar(v[0], data["idas_v2"][s][0])
                    note = f"胜 {n01} / 负 {n10}，n={n}，p={p:.4f}"
                lines.append(f"| {label} | {s} | {100 * v[1]:.1f} | {v[2]:.3f} | "
                             f"{v[3]:.2f} | {note} |")

    # ---- 表 5 mem × IDAS 2×2 交互
    if any(n in data for n in MEM):
        lines += ["", "## 表 5 mem × IDAS 2×2（种子 42/123）", "",
                  "| 感知档位 | IDAS | 种子 | SR % | SPL | NE (m) |", "|---|---|---|---|---|---|"]
        for det, idas_name, det_label, idas_label in (
                ("none", "none", "单帧", "λ=0"), ("none", "idas_v2", "单帧", "idas_v2"),
                ("mem", "mem_none", "记忆", "λ=0"), ("mem", "mem_v2", "记忆", "idas_v2")):
            if idas_name not in data:
                lines.append(f"| {det_label} | {idas_label} | — | — | — | — |")
                continue
            for s, v in sorted(data[idas_name].items(), key=lambda z: (z[0] is None, z[0])):
                if det == "mem" and s not in (42, 123):
                    continue
                lines.append(f"| {det_label} | {idas_label} | {s} | {100 * v[1]:.1f} | "
                             f"{v[2]:.3f} | {v[3]:.2f} |")
        lines += ["", "> 表注：mem 两格用 `nav_detf_mem96k.h5`（95204 样本 / 1176 回合），"
                  "非 mem 两格沿用 `nav_detf.h5`（96147 / 1080），样本量差 1%，同一采集配置；"
                  "两档内部各自成对比较，跨档只作趋势参考。", ""]

    os.makedirs(os.path.dirname(a.md) or ".", exist_ok=True)
    with open(a.md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
