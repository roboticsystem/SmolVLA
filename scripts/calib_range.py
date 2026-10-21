#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**E1b · 单目测距标定（仿真侧）**：距离-误差曲线。

背景（`E13_参考论文对照与欠账清单.md` §2.2）：
大纲 E11 写「目标距离用已知人体高度 + 针孔模型反推，**须在 E1 阶段先行标定并报告测距误差**」，
但 E1 阶段**一直没做**——这是明确的欠账。

参考论文2 §3.3 给出了「距离-误差」曲线（近线性、相对误差恒 <3%）；
参考论文1 §3.1 则给出「**绝对尺度不可逆**」的理论论证（针孔模型除以深度 Z_c
后绝对尺度丢失 ⇒ 单目测距必须靠已知尺寸或外部尺度补）。

本脚本在仿真里标定**跟随任务实际实现的那套测距**（底边接地法
`d = f·h_cam/(r_bottom - horizon)`），并同时给出**「已知身高反推法」**
`d = f·H_person/h_px` 在头顶**未被裁切**的距离区间内的表现，
以便论文里如实说明「为什么实现改用了底边接地法」以及两种口径的差异。

用法：
    python scripts/calib_range.py --distances 0.5,0.8,1.0,1.2,1.5,1.8,2.0,2.5,3.0 \
        --scenarios normal,backlight,gloss --repeats 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.follow import FollowEnv, PERSON_H, detect_person      # noqa: E402


def measure(env, d_true):
    """把目标摆到机器人正前方 d_true 处，返回一次测距结果。"""
    target = env.target
    # 机器人保持出生点与朝向，朝向改为「看向目标摆放位置」
    rx, ry = float(env.scene.spawn[0]), float(env.scene.spawn[1])
    ux, uy = np.asarray(target.xy, dtype=float) - np.array([rx, ry])
    n = float(np.hypot(ux, uy))
    if n < 1e-6:
        ux, uy, n = 1.0, 0.0, 1.0
    ux, uy = ux / n, uy / n
    # 目标摆在出生点前方 d_true 米
    target.xy = np.array([rx + ux * d_true, ry + uy * d_true], dtype=float)
    theta = float(np.arctan2(uy, ux))
    env.pose = np.array([rx, ry, theta], dtype=np.float64)
    obs = env._obs()
    r = detect_person(obs["image"], want_px=True)
    if r is None:
        return None
    bearing, d_est, npx, h_px, f, r0, r1 = r
    d_h = float(f * PERSON_H / h_px) if (h_px > 0 and r0 > 0) else float("nan")
    return {"d_true": float(d_true), "d_est": float(d_est), "d_by_height": d_h,
            "h_px": int(h_px), "top_clipped": bool(r0 == 0),
            "bearing": float(bearing), "npx": int(npx)}


def main():
    ap = argparse.ArgumentParser(description="E1b 单目测距标定（仿真侧）")
    ap.add_argument("--distances", default="0.5,0.8,1.0,1.2,1.5,1.8,2.0,2.5,3.0")
    ap.add_argument("--scenarios", default="normal,backlight,gloss,shadow")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="export/range_calib.json")
    ap.add_argument("--md", default="export/range_calib.md")
    a = ap.parse_args()

    dists = [float(x) for x in a.distances.split(",") if x]
    scen = [x for x in a.scenarios.split(",") if x]
    rows = []
    for sc in scen:
        for d in dists:
            errs, errs_h, det, clipped = [], [], 0, 0
            for k in range(a.repeats):
                env = FollowEnv(scenario=sc, camera_augment=0.0)
                env.reset(seed=1000 + k)
                m = measure(env, d)
                if m is None:
                    continue
                det += 1
                errs.append(m["d_est"] - m["d_true"])
                if m["d_by_height"] == m["d_by_height"]:
                    errs_h.append(m["d_by_height"] - m["d_true"])
                clipped += int(m["top_clipped"])
            n = max(1, det)
            rec = {"scenario": sc, "d_true": d, "det_rate": det / max(1, a.repeats),
                   "err_mean": float(np.mean(errs)) if errs else float("nan"),
                   "err_absmean": float(np.mean(np.abs(errs))) if errs else float("nan"),
                   "err_std": float(np.std(errs)) if errs else float("nan"),
                   "rel_err_pct": float(100 * np.mean(np.abs(errs)) / d) if errs else float("nan"),
                   "height_err_absmean": float(np.mean(np.abs(errs_h))) if errs_h else float("nan"),
                   "height_rel_pct": float(100 * np.mean(np.abs(errs_h)) / d) if errs_h else float("nan"),
                   "top_clipped_rate": clipped / n}
            rows.append(rec)
            print(f"{sc:<10s} d={d:4.1f} m  检出 {100*rec['det_rate']:5.1f}%  "
                  f"底边接地法 误差 {rec['err_mean']:+.3f} m "
                  f"(|e|={rec['err_absmean']:.3f}, {rec['rel_err_pct']:.1f}%)  "
                  f"身高反推法 |e|={rec['height_err_absmean']:.3f} "
                  f"({rec['height_rel_pct']:.1f}%)  头顶被裁 {100*rec['top_clipped_rate']:.0f}%",
                  flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"repeats": a.repeats, "rows": rows}, f, ensure_ascii=False, indent=2)

    lines = [f"# E1b 单目测距标定（仿真侧，{a.repeats} 次/格）", "",
             "| 场景 | 真实距离 (m) | 检出率 | 底边接地法 误差 (m) | 相对误差 | "
             "身高反推法 |e| (m) | 相对误差 | 头顶被裁 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['scenario']} | {r['d_true']:.1f} | {100*r['det_rate']:.0f}% | "
                     f"{r['err_mean']:+.3f} | {r['rel_err_pct']:.1f}% | "
                     f"{r['height_err_absmean']:.3f} | {r['height_rel_pct']:.1f}% | "
                     f"{100*r['top_clipped_rate']:.0f}% |")
    with open(a.md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n已写出 {a.out} 与 {a.md}")


if __name__ == "__main__":
    main()
