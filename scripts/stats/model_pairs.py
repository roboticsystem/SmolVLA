#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**模型级配对复核**（从权威明细重算 §4.23 的全部效应量）。

为什么需要这个脚本（2026-10-21）：
§4.23 的「记忆效应 +7.75 pp」此前只写在报告里，**没有可复算的脚本**；
一旦只看 `export/eval_detail/*_std.json` 的**单文件 200 回合**，
会得到**方向相反**的结论（记忆档 19.0% < 论文口径模型 24.5%），
原因是 ① 两臂**数据量不同**（96k vs 31k）本就不可配对比较；
② 忽略了追加批次 `*_s16400.json`（论文口径是**合并 n = 400**）。

本脚本按论文口径重算：
  · 回合集 = 主文件（seed 9000+37·ep）+ 追加批次（`_*_s16400.json`）**合并**；
  · 统计量 = **配对精确 McNemar**（双侧，二项精确）+ Wilson 区间；
  · 输出 `export/model_pairs.json` / `.md`，每个数字都能指回明细文件。

用法：
    python scripts/stats/model_pairs.py --out export/model_pairs.json --md export/model_pairs.md
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DETAIL = os.path.join(ROOT, "export", "eval_detail")

# (标签, 明细名 A, 明细名 B)：A 是被检验的「增强臂」
PAIRS = [
    ("记忆 8 维 vs 单帧 4 维（31k，数据完全相同）",
     "ft_detf_mem_std", "ft_detf_sf31k_std"),
    ("记忆 8 维 vs 单帧 4 维（96k）",
     "ft_detf_mem96k_std", "ft_detf_sf96k_std"),
    ("记忆 8 维 vs 无感知（31k）",
     "ft_detf_mem_std", "ft_detf_none31k_std"),
    ("单帧 4 维 vs 无感知（31k）",
     "ft_detf_sf31k_std", "ft_detf_none31k_std"),
    ("示范（观测条件化）vs 模型（96k 单帧）",
     "demo_sweep_det_std", "ft_detf_std"),
    ("示范 记忆开 vs 记忆关（n=200，单文件）",
     "demo_sweep_det_mm_mem25_std", "demo_sweep_det_mm_nomem_std"),
]


def load(name):
    """合并主文件与追加批次 → {seed: success}；并记录各批次来源。"""
    files = [f"{DETAIL}/{name}.json"] + sorted(glob.glob(f"{DETAIL}/{name}_s*.json"))
    succ, src = {}, []
    for f in files:
        if not os.path.exists(f):
            continue
        d = json.load(open(f, encoding="utf-8"))
        for row in d.get("detail", []):
            succ[row[0]] = float(row[1])
        src.append({"file": os.path.basename(f), "episodes": d.get("episodes"),
                    "settings": d.get("settings")})
    return succ, src


def mcnemar_exact(a, b):
    """配对精确 McNemar：返回 (只 a 成功, 只 b 成功, p 双侧)。"""
    n01 = sum(1 for x, y in zip(a, b) if x > 0.5 and y <= 0.5)
    n10 = sum(1 for x, y in zip(a, b) if x <= 0.5 and y > 0.5)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)
    return n01, n10, p


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main():
    ap = argparse.ArgumentParser(description="模型级配对复核（论文口径 n=400）")
    ap.add_argument("--out", default=os.path.join(ROOT, "export", "model_pairs.json"))
    ap.add_argument("--md", default=os.path.join(ROOT, "export", "model_pairs.md"))
    a = ap.parse_args()

    rows = []
    for label, na, nb in PAIRS:
        A, sa = load(na)
        B, sb = load(nb)
        if not A or not B:
            print(f"[skip] {label}: 明细缺失 ({na} / {nb})")
            continue
        common = sorted(set(A) & set(B))
        va = [A[k] for k in common]
        vb = [B[k] for k in common]
        n01, n10, p = mcnemar_exact(va, vb)
        ka = sum(1 for v in va if v > 0.5)
        kb = sum(1 for v in vb if v > 0.5)
        lo_a, hi_a = wilson(ka, len(common))
        lo_b, hi_b = wilson(kb, len(common))
        rows.append({
            "label": label, "a": na, "b": nb, "n": len(common),
            "sr_a": ka / len(common), "sr_b": kb / len(common),
            "ci_a": [lo_a, hi_a], "ci_b": [lo_b, hi_b],
            "delta_pp": 100.0 * (ka - kb) / len(common),
            "only_a": n01, "only_b": n10, "p": p, "sig": p < 0.05,
            "sources_a": sa, "sources_b": sb,
        })
        print(f"{label:42s} n={len(common):4d}  "
              f"{100*ka/len(common):6.2f}% vs {100*kb/len(common):6.2f}%  "
              f"Δ={100*(ka-kb)/len(common):+6.2f}pp  "
              f"只A={n01:3d} 只B={n10:3d}  p={p:.3e}  "
              f"{'显著' if p < 0.05 else '不显著'}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"pairs": rows}, f, ensure_ascii=False, indent=2)

    lines = ["# 模型级配对复核（论文口径：主文件 + 追加批次合并，配对精确 McNemar）", "",
             "| 对比 | n | SR A % | SR B % | Δ (pp) | 只 A 成 | 只 B 成 | p | 判定 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['n']} | {100*r['sr_a']:.2f} [{100*r['ci_a'][0]:.1f}, "
            f"{100*r['ci_a'][1]:.1f}] | {100*r['sr_b']:.2f} [{100*r['ci_b'][0]:.1f}, "
            f"{100*r['ci_b'][1]:.1f}] | **{r['delta_pp']:+.2f}** | {r['only_a']} | "
            f"{r['only_b']} | {r['p']:.2e} | {'✅ 显著' if r['sig'] else '❌ 不显著'} |")
    lines += ["", "明细来源（每个对比的文件）："]
    for r in rows:
        lines.append(f"- {r['label']}：`{r['a']}`（+ 追加批次） vs `{r['b']}`（+ 追加批次）")
    with open(a.md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n已写出 {a.out} 与 {a.md}")


if __name__ == "__main__":
    main()
