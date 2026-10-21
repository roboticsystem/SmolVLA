#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**末段停靠示范 vs 原示范**的对照评测（主创新 A 新增子模块的验证）。

背景（`主创新A_诊断报告.md` §4.26.5）：
原示范策略 `sweep_det` 是「全速冲刺、靠恰好擦过判据」的：
  · 单目地面几何在 0.752 m 饱和，而成功判据是「到包围盒表面 ≤ 0.30 m」
    ⇒ 示范在末段**不知道自己已经接近**；
  · 数据里没有任何精细末段控制样本 ⇒ 模型学不到末段；
  · 模型轨迹略有偏差就「擦不到」⇒ 保留率卡在 67–74%，VLA 闭环 SR 低于示范。

`sweep_det_dock` 用**激光表面距 + 激光命中点方位**做末段「先对准、再冲刺、
进判据区即停」，目标是：
  ① 提升**示范自身**的成功率（抬高模仿学习上限）；
  ② 在数据里造出**可学的末段停靠样本**（|v|→0 且 w=0，与扫描的 w≠0 可区分）。

口径（与 `主创新A_诊断报告.md` §4.19 完全一致，便于直接对照）：
  · 标定测试床 `calibrated_env`（success_standoff = 0.30 m）
  · 回合种子 `9000 + 37·ep`（与 VLA 评测共用同一套回合集）
  · `max_steps`：**默认 200**（与 `smolvla_finetune_sim.py --eval-max-steps` 同口径）；
    权威示范明细（`export/eval_detail/demo_sweep_det_std.json`）用的是
    `sim/run_eval.py` 的默认 **250 步**，⇒ 与示范并列时必须 `--max-steps 250`，
    否则两者不可比（本轮已实测：200 步下示范 SR 29.5% vs 权威 33.0%）。
  · `mem_ttl = 25`（`sweep_det` 的默认记忆配置）

本轮新增（2026-10-21）：
  · `--detail-out`：另存逐回合明细，供配对 McNemar 使用；
  · `--ref-detail`：与权威明细**逐回合**比对，判定本轮步数口径是否与权威一致。

用法：
    python scripts/eval_dock_demo.py --episodes 200 --max-steps 250 \
        --policies sweep_det,sweep_det_dock \
        --ref-detail export/eval_detail/demo_sweep_det_std.json \
        --detail-out export/eval_detail/dock_demo_ms250.json \
        --out export/dock_demo_ms250_paired.json --md export/dock_demo_ms250_paired.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.run_eval import evaluate  # noqa: E402

# `min_dsurf` 分档（与 §4.26.2 的分档一致，用于对比失败模式分布）
BANDS = [(0.30, 0.45, "只差一点 0.30-0.45 m"),
         (0.45, 0.70, "差一些 0.45-0.70 m"),
         (0.70, 1.50, "走到附近 0.70-1.50 m"),
         (1.50, 1e9, "从未接近 > 1.50 m")]


def band_of(d):
    for lo, hi, name in BANDS:
        if lo <= d < hi:
            return name
    return BANDS[-1][2]


def mcnemar_exact(a, b):
    """配对精确 McNemar（两列 0/1 成功标记，索引对齐）。

    a、b 为等长的 0/1 列表；返回 (只 a 成功, 只 b 成功, p 值)。
    p 值用二项精确检验（双侧，p = 2·P(X ≤ min)，截断到 1）。
    """
    n01 = sum(1 for x, y in zip(a, b) if x > 0.5 and y <= 0.5)
    n10 = sum(1 for x, y in zip(a, b) if x <= 0.5 and y > 0.5)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)
    return n01, n10, p


def load_ref_detail(path):
    """读权威明细（export/eval_detail/*.json），返回 {seed: success} 与 {seed: min_dist}。"""
    if not path or not os.path.exists(path):
        return {}, {}
    d = json.load(open(path, encoding="utf-8"))
    succ, dist = {}, {}
    for row in d.get("detail", []):
        succ[row[0]] = float(row[1])
        if len(row) > 4:
            dist[row[0]] = float(row[4])
    return succ, dist


def main():
    ap = argparse.ArgumentParser(description="末段停靠示范对照评测")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=9000)
    ap.add_argument("--ep-stride", type=int, default=37)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--policies", default="sweep_det,sweep_det_dock")
    ap.add_argument("--out", default="export/dock_demo.json")
    ap.add_argument("--md", default="export/dock_demo.md")
    ap.add_argument("--detail-out", default=None,
                    help="逐回合明细（seed, success, level, …）另存路径；"
                         "用于配对 McNemar 与口径核对")
    ap.add_argument("--ref-detail", default=None,
                    help="权威明细（export/eval_detail/*.json），用于核对本轮步数口径"
                         "是否与权威结果一致（逐回合比对成功标记）")
    a = ap.parse_args()

    out_rows, details = [], {}
    for pname in [x for x in a.policies.split(",") if x]:
        det = []
        agg, rows = evaluate(pname, episodes=a.episodes, level=None, seed=a.seed_base,
                             calib=True, env_kwargs={"max_steps": a.max_steps},
                             ep_base=a.seed_base, ep_stride=a.ep_stride,
                             policy_kw={"mem_ttl": 25}, verbose=True, detail_out=det)
        out_rows.append(agg)
        details[pname] = det
        print(f"  >>> {pname}: SR {100*agg['SR']:.1f}%  "
              f"SPL {agg['SPL']:.3f}  步数 {agg['steps']:.1f}  "
              f"碰撞 {agg['n_collide']:.1f}", flush=True)

    # 失败模式分档（min_dsurf）
    band_tbl = {}
    for pname, det in details.items():
        cnt = {}
        for d in det:
            b = band_of(float(d[4]))
            cnt[b] = cnt.get(b, 0) + 1
        tot = max(len(det), 1)
        band_tbl[pname] = {b: 100.0 * cnt.get(b, 0) / tot for _, _, b in BANDS}

    # 配对精确 McNemar（本轮两个策略之间）
    paired = {}
    pols = list(details.keys())
    if len(pols) == 2:
        s0 = {d[0]: d[1] for d in details[pols[0]]}
        s1 = {d[0]: d[1] for d in details[pols[1]]}
        common = sorted(set(s0) & set(s1))
        n01, n10, p = mcnemar_exact([s0[k] for k in common],
                                    [s1[k] for k in common])
        paired = {"a": pols[0], "b": pols[1], "n": len(common),
                  "only_a": n01, "only_b": n10, "p": p,
                  "sr_a": sum(s0[k] > 0.5 for k in common) / max(len(common), 1),
                  "sr_b": sum(s1[k] > 0.5 for k in common) / max(len(common), 1)}
        print(f"  >>> 配对 McNemar({pols[0]} vs {pols[1]}): "
              f"只{pols[0]}成 {n01} / 只{pols[1]}成 {n10}  p = {p:.4f}", flush=True)

    # 口径核对：本轮逐回合结果与权威明细是否逐回合一致
    ref_check = {}
    if a.ref_detail:
        ref, _ = load_ref_detail(a.ref_detail)
        for pname, det in details.items():
            cur = {d[0]: d[1] for d in det}
            common = sorted(set(ref) & set(cur))
            same = sum(1 for k in common
                       if (ref[k] > 0.5) == (cur[k] > 0.5))
            ref_check[pname] = {"ref_file": a.ref_detail, "n_common": len(common),
                                "n_same": same,
                                "n_diff": len(common) - same,
                                "sr_ref": sum(ref[k] > 0.5 for k in common) / max(len(common), 1),
                                "sr_cur": sum(cur[k] > 0.5 for k in common) / max(len(common), 1)}
            print(f"  >>> 口径核对 {pname}: 与 {a.ref_detail} 逐回合一致 "
                  f"{same}/{len(common)}（ref SR {100*ref_check[pname]['sr_ref']:.1f}% vs "
                  f"本轮 {100*ref_check[pname]['sr_cur']:.1f}%）", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"episodes": a.episodes, "seed_base": a.seed_base,
                   "ep_stride": a.ep_stride, "max_steps": a.max_steps,
                   "rows": out_rows, "bands": band_tbl,
                   "paired_mcnemar": paired, "ref_check": ref_check,
                   "detail": {k: v for k, v in details.items()}},
                  f, ensure_ascii=False, indent=2)
    if a.detail_out:
        os.makedirs(os.path.dirname(a.detail_out) or ".", exist_ok=True)
        with open(a.detail_out, "w", encoding="utf-8") as f:
            json.dump({"policy": pols[0] if len(pols) == 1 else pols,
                       "protocol": "std", "episodes": a.episodes,
                       "max_steps": a.max_steps,
                       "detail": details[pols[0]] if len(pols) == 1 else
                                 [d for p in pols for d in details[p]]},
                      f, ensure_ascii=False, indent=2)

    lines = [f"# 末段停靠示范对照评测（{a.episodes} 回合，标定测试床，"
             f"seed {a.seed_base}+{a.ep_stride}·ep，max_steps={a.max_steps}）", "",
             "| 策略 | SR % | SPL | NE (m) | 步数 | 碰撞 |",
             "|---|---|---|---|---|---|"]
    for r in out_rows:
        lines.append(f"| {r['policy']} | {100*r['SR']:.1f} | {r['SPL']:.3f} | "
                     f"{r['NE']:.2f} | {r['steps']:.1f} | {r['n_collide']:.1f} |")
    if paired:
        lines += ["", "## 配对精确 McNemar（同一回合集）", "",
                  f"- 对比：`{paired['a']}` vs `{paired['b']}`，n = {paired['n']}",
                  f"- SR：{100*paired['sr_a']:.1f}% vs {100*paired['sr_b']:.1f}%",
                  f"- 只前者成功 {paired['only_a']} 回合 / 只后者成功 {paired['only_b']} 回合",
                  f"- p = {paired['p']:.4f}"]
    if ref_check:
        lines += ["", "## 口径核对（与权威明细逐回合比对）", "",
                  "| 策略 | 权威文件 | 一致 / 总数 | 权威 SR % | 本轮 SR % |",
                  "|---|---|---|---|---|"]
        for pname, rc in ref_check.items():
            lines.append(f"| {pname} | `{os.path.basename(rc['ref_file'])}` | "
                         f"{rc['n_same']} / {rc['n_common']} | "
                         f"{100*rc['sr_ref']:.1f} | {100*rc['sr_cur']:.1f} |")
    lines += ["", "## 失败模式分布（按全程最近接近距离 `min_dsurf` 分档，%）", "",
              "| 档位 | " + " | ".join(details.keys()) + " |",
              "|---|" + "---|" * len(details)]
    for _, _, b in BANDS:
        lines.append(f"| {b} | " + " | ".join(
            f"{band_tbl[p].get(b, 0.0):.1f}" for p in details) + " |")
    with open(a.md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n已写出 {a.out} 与 {a.md}")


if __name__ == "__main__":
    main()
