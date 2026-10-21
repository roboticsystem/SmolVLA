#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**E6b · 部署能耗与续航预算**（本机可做的部分：理论推算 + 云侧对照实测）。

三部分：
1. **仿真轨迹驱动的四段能耗**（行驶 / 转向 / BPU 推理 / 传感器待机）
   —— 用**真实跑出来的轨迹**驱动 `sim/energy.py`，不是拍脑袋的参数。
2. **推理频率 → 能耗 → 续航曲线** —— 主创新 B（轻量化部署）在**能量维度**上的落点。
3. **云侧对照实测** —— `nvidia-smi` 采样本机 4090 在持续推理负载下的功耗，
   作为 E9 云边分层的能量侧参照；**板载 BPU 功耗必须实测替换**（见文末清单）。

**纪律**（`E13_参考论文对照与欠账清单.md` §7）：参考论文3 的能耗模型是纯理论、
无实测标定、验证节没有误差统计。我方借鉴时**必须补实测**，
因此本脚本对「实测项」与「待测项」严格分开标注。

用法：
    python scripts/energy_report.py --layouts apartment,classroom --episodes 3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.baselines import make_policy                       # noqa: E402
from sim.energy import (DEFAULT, battery_budget, economic_speed,  # noqa: E402
                        full_breakdown)
from sim.env import LAYOUT_STEPS, calibrated_env           # noqa: E402


def gpu_power_w():
    """本机 GPU 当前功耗（W）；无 GPU 或 nvidia-smi 不可用时返回 None。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=10).decode()
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def measure_server_gpu(duration_s=6.0):
    """云侧对照：持续大矩阵负载下的 4090 平均功耗。

    用 450M 参数规模的矩阵乘负载**近似** SmolVLA 4×1 的推理强度，
    只为得到「云侧每帧能耗量级」这一**参照**，**不是**板载功耗的替代。
    """
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    dev = torch.device("cuda")
    w = [torch.randn(1536, 1536, device=dev, dtype=torch.float16) for _ in range(48)]
    x = torch.randn(1, 1536, device=dev, dtype=torch.float16)
    for _ in range(5):
        y = x
        for m in w:
            y = torch.nn.functional.gelu(y @ m)
        torch.cuda.synchronize()
    p0 = gpu_power_w()
    t0 = time.time()
    n = 0
    while time.time() - t0 < duration_s:
        y = x
        for m in w:
            y = torch.nn.functional.gelu(y @ m)
        n += 1
    torch.cuda.synchronize()
    dur = time.time() - t0
    p1 = gpu_power_w()
    if p0 is None or p1 is None:
        return None
    return {"gpu": "RTX 4090 (本机，云侧参照)", "power_W": (p0 + p1) / 2.0,
            "iter_per_s": n / dur,
            "note": "450M 参数规模持续负载；仅作云侧参照，非板载功耗"}


def run_episode(layout, policy_name, episodes, seed0=9000, stride=37):
    """跑若干回合，收集真实轨迹（位置、速度指令、角速度指令）。"""
    env = calibrated_env(layout=layout, max_steps=LAYOUT_STEPS.get(layout, 350))
    trajs = []
    for ep in range(episodes):
        obs = env.reset(seed=seed0 + ep * stride)
        pol = make_policy(policy_name)
        pol.reset(env)
        vs, ws, poses = [], [], []
        info = {}
        for i in range(env.cfg["max_steps"]):
            a = pol.act(obs, env)
            obs, _r, done, info = env.step(a)
            vs.append(float(a[0]))
            ws.append(float(a[1]))
            poses.append(env.pose[:2].copy())
            if done:
                break
        trajs.append({"pos": np.asarray(poses), "v": np.asarray(vs), "w": np.asarray(ws),
                      "success": bool(info.get("success", False)),
                      "steps": int(i + 1), "level": str(info.get("level", ""))})
    return trajs


def main():
    ap = argparse.ArgumentParser(description="E6b 部署能耗与续航预算")
    ap.add_argument("--layouts", default="apartment,classroom")
    ap.add_argument("--policy", default="geodesic")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--infer-hz", default="1,2,5,10,20")
    ap.add_argument("--infer-power-w", type=float, default=5.0,
                    help="板载 BPU 推理功耗的**占位估计**（W）；实测后必须替换")
    ap.add_argument("--out", default="export/energy_report.json")
    ap.add_argument("--md", default="export/energy_report.md")
    a = ap.parse_args()

    layouts = [x for x in a.layouts.split(",") if x]
    hz_list = [float(x) for x in a.infer_hz.split(",") if x]

    # ---------- 1. 轨迹驱动的四段能耗 ----------
    results = []
    for lay in layouts:
        trajs = run_episode(lay, a.policy, a.episodes)
        rows = []
        for tr in trajs:
            dtheta = np.abs(tr["w"]) * 0.1          # 每步转角 |Δθ| = |ω|·dt
            b = full_breakdown(tr["pos"], tr["v"], dtheta, len(tr["v"]),
                               a.infer_power_w, p=DEFAULT)
            rows.append({"success": tr["success"], "steps": tr["steps"],
                         "level": tr["level"], **b,
                         "turn_total_rad": float(np.sum(dtheta))})
        agg = {"layout": lay, "episodes": len(rows),
               "avg_steps": float(np.mean([r["steps"] for r in rows])),
               "avg_success_rate": float(np.mean([r["success"] for r in rows])),
               "avg_total_J": float(np.mean([r["total_J"] for r in rows])),
               "avg_shares": {k: float(np.mean([r["shares"][k] for r in rows]))
                              for k in rows[0]["shares"]},
               "avg_battery_mAh": float(np.mean([r["battery"]["mAh"] for r in rows])),
               "avg_runtime_min": float(np.mean([r["battery"]["runtime_min"] for r in rows])),
               "avg_turn_rad": float(np.mean([r["turn_total_rad"] for r in rows])),
               "avg_distance_m": float(np.mean([r["drive"]["distance_m"] for r in rows]))}
        results.append(agg)
        sh = agg["avg_shares"]
        print(f"[{lay}] 步数 {agg['avg_steps']:.0f}  里程 {agg['avg_distance_m']:.1f} m  "
              f"总转角 {agg['avg_turn_rad']:.1f} rad  总能耗 {agg['avg_total_J']:.1f} J  "
              + " ".join(f"{k}:{100*v:.0f}%" for k, v in sh.items())
              + f"  电池 {agg['avg_battery_mAh']:.1f} mAh", flush=True)

    # ---------- 2. 推理频率 → 能耗 → 续航 ----------
    freq_rows = []
    base = results[0]
    avg_steps = base["avg_steps"]
    for hz in hz_list:
        frames = avg_steps * (hz / 10.0)          # 以 10 Hz 为满载基准
        ej = frames * a.infer_power_w * 0.1
        total = base["avg_total_J"] * (1 - base["avg_shares"]["inference"]) + ej
        bat = battery_budget(total, p=DEFAULT)
        freq_rows.append({"hz": hz, "frames": frames, "inference_J": ej,
                           "total_J": total, "mAh": bat["mAh"],
                           "runtime_min": bat["runtime_min"]})
        print(f"推理 {hz:4.1f} Hz → 帧数 {frames:6.1f}  推理能耗 {ej:7.2f} J  "
              f"合计 {total:7.2f} J  电池 {bat['mAh']:6.2f} mAh  "
              f"可持续 {bat['runtime_min']:6.1f} min", flush=True)

    # ---------- 3. 云侧对照 ----------
    srv = measure_server_gpu()
    if srv:
        print(f"[云侧参照] {srv['gpu']} 持续负载 {srv['power_W']:.0f} W  "
              f"{srv['iter_per_s']:.1f} 次/s", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"layouts": results, "freq": freq_rows, "server_ref": srv,
                   "params": DEFAULT, "infer_power_w_assumed": a.infer_power_w},
                  f, ensure_ascii=False, indent=2)

    lines = ["# E6b 部署能耗与续航预算（本机可测部分）", "",
             f"## 1. 仿真轨迹驱动的四段能耗（{a.policy}，每布局 {a.episodes} 回合）", "",
             "| 布局 | 步数 | 里程 (m) | 总转角 (rad) | 合计 J | 行驶% | 转向% | 推理% | "
             "待机% | mAh | 可持续 (min) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        s = r["avg_shares"]
        lines.append(f"| {r['layout']} | {r['avg_steps']:.0f} | {r['avg_distance_m']:.1f} | "
                     f"{r['avg_turn_rad']:.1f} | {r['avg_total_J']:.1f} | "
                     f"{100*s['drive']:.0f} | {100*s['turn']:.0f} | {100*s['inference']:.0f} | "
                     f"{100*s['sensor']:.0f} | {r['avg_battery_mAh']:.1f} | "
                     f"{r['avg_runtime_min']:.1f} |")
    lines += ["", "## 2. 推理频率 → 能耗 → 续航", "",
              "| 推理频率 (Hz) | 每回合帧数 | 推理能耗 (J) | 合计 (J) | 电池 (mAh) | 可持续 (min) |",
              "|---|---|---|---|---|---|"]
    for r in freq_rows:
        lines.append(f"| {r['hz']:.0f} | {r['frames']:.0f} | {r['inference_J']:.2f} | "
                     f"{r['total_J']:.1f} | {r['mAh']:.2f} | {r['runtime_min']:.1f} |")
    lines += ["", "## 3. 经济时速", "",
              f"- **经济时速 = {economic_speed(DEFAULT):.3f} m/s**"
              "（按等距离比较能耗；对应论文3 的 U 型经济时速结论）", ""]
    if srv:
        lines += [f"- 云侧参照：{srv['gpu']}，持续负载 {srv['power_W']:.0f} W，"
                  f"{srv['iter_per_s']:.1f} 次/s（{srv['note']}）", ""]
    lines += ["## 4. 板载待测项（本机无法完成，需在 RDK-X3 上实测）", "",
              "| # | 待测量 | 测法 | 对应产出 |", "|---|---|---|---|",
              "| 1 | BPU 单帧推理功耗 P (W) | 板载电流采样（USB 电流表串在供电回路），"
              "空载基线扣除 | 表 G9b-2 |",
              "| 2 | 电机驱动功耗（行驶/转向分别） | 电机端电压电流采样 + 已知轮径/减速比 | 表 G9b-1 |",
              "| 3 | 传感器常开功耗 | 逐模块断电测静态功耗 | 表 G9b-1 |",
              "| 4 | 电源转换效率 | 空载/负载两端电压电流 | 参数表 |",
              "| 5 | 实测续航（连续跟随回合） | 记录 SoC 下降曲线 | 图 G9b-1 |", ""]
    with open(a.md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n已写出 {a.out} 与 {a.md}")


if __name__ == "__main__":
    main()
