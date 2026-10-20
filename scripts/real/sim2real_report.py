#!/usr/bin/env python3
"""sim2real_report.py —— **仿真↔实机一致性报告**生成器（论文 P5）。

输入：真机录制（`deliver/tools/record_dataset.py` 产出的 HDF5，schema：
      actions (N,2) / states (N,5) / lidar (N,120) / images (N,H,W,3)）
输出：一份 Markdown 报告（+ 可选 JSON），包含五节：
      1. 轨迹重放误差（真机指令 → 仿真执行，刚体对齐后）
      2. 图像域差（真机 vs 仿真，同位姿渲染）
      3. 激光域差
      4. 动力学参数标定（速度增益 / 一阶滞后 / 里程计漂移率）
      5. 误差预算分解（逐项消融）+ 闭环指标影响
      6. 等效密度换算（论文4 范式）

用法：
    # 基本用法（回放 + 域差 + 标定 + 预算）
    python3 scripts/real/sim2real_report.py \
        --h5 deliver/data_recorded/session.h5 \
        --layout apartment \
        --out export/sim2real/session.md

    # 附带闭环对照与等效密度换算
    python3 scripts/real/sim2real_report.py --h5 ... --layout apartment \
        --closed-loop --policy oracle --episodes 20 \
        --real-n 1 --real-area 6.0 \
        --out export/sim2real/session.md --json export/sim2real/session.json

    # 没有真机数据时，可用样例录制验证整条管线
    python3 scripts/real/make_mock_recording.py --out /tmp/mock.h5
    python3 scripts/real/sim2real_report.py --h5 /tmp/mock.h5 --layout apartment \
        --out export/sim2real/mock.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from sim.env import SemanticNavEnv                      # noqa: E402
from sim.replay import domain_gap, load_recording, replay_in_sim, write_report  # noqa: E402
from sim.sim2real import (closed_loop_impact, density_error_rate,       # noqa: E402
                          equivalent_density, error_budget, fit_kinematics,
                          scene_density)


def build_env(layout=None, seed=0, max_steps=400, obs_size=(160, 120),
              n_peds=0, lidar_noise=0.01, camera_augment=0.0):
    """按报告口径建环境（默认固定户型 `apartment`）。

    固定户型是**必须**的：真机录制是在真实房间里做的，场景几何必须与仿真场景
    对应，坐标锚定才有意义；随机 MultiRoom 与真实房间对不上，域差统计会失真。
    """
    kw = dict(seed=seed, layout=layout, max_steps=max_steps, obs_size=obs_size,
              n_peds=n_peds, lidar_noise=lidar_noise, camera_augment=camera_augment)
    return SemanticNavEnv(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="真机录制文件（HDF5/NPZ/JSON）")
    ap.add_argument("--layout", default="apartment",
                    help="仿真场景布局（apartment/corridor/s_corridor/warehouse）")
    ap.add_argument("--seed", type=int, default=0, help="场景种子（固定户型下仅影响行人/任务）")
    ap.add_argument("--frames", type=int, default=60, help="域差统计用帧数")
    ap.add_argument("--out", default="export/sim2real/report.md", help="Markdown 报告输出路径")
    ap.add_argument("--json", default=None, help="同时输出 JSON（机器可读）")
    ap.add_argument("--title", default="仿真—实机一致性报告")
    ap.add_argument("--no-budget", action="store_true", help="跳过误差预算分解（较慢）")
    ap.add_argument("--no-domain-gap", action="store_true", help="跳过观测域差统计")
    ap.add_argument("--closed-loop", action="store_true", help="追加闭环口径的退化项影响")
    ap.add_argument("--policy", default="geodesic",
                    help="闭环对照用策略（固定户型建议 geodesic：窗口测地场上界）")
    ap.add_argument("--episodes", type=int, default=20, help="闭环对照回合数")
    ap.add_argument("--real-n", type=float, default=None,
                    help="实机部署规模（等效密度换算用；如 1 台机器人）")
    ap.add_argument("--real-area", type=float, default=None,
                    help="实机场地面积 m²（等效密度换算用）")
    ap.add_argument("--real-sr", type=float, default=None,
                    help="实机实测的任务指标（如 SR=0.75），用于算误差率")
    ap.add_argument("--sim-sr", type=float, default=None,
                    help="仿真侧的任务指标（默认取闭环对照的基线 SR）")
    a = ap.parse_args()

    rec = load_recording(a.h5)
    env = build_env(layout=a.layout, seed=a.seed)
    notes = []

    print(f"[P5] 载入录制 {a.h5}：{len(rec['actions'])} 帧")
    rep = replay_in_sim(rec, env, seed=a.seed)
    print("  轨迹重放：位置 RMSE %.3f m（未对齐 %.3f m），航向 RMSE %.3f rad"
          % (rep.get("pos_rmse", float("nan")), rep.get("pos_rmse_raw", float("nan")),
             rep.get("yaw_rmse", float("nan"))))

    gap = None
    if not a.no_domain_gap:
        gap = domain_gap(rec, env, seed=a.seed, n_frames=a.frames)
        if "image" in gap:
            g = gap["image"]["mean"]
            print("  图像域差：亮度 真机 %.1f / 仿真 %.1f（差 %+.1f）"
                  % (g["real"], g["sim"], g["gap"]))

    kin = fit_kinematics(rec, dt=float(env.cfg["dt"]))
    print("  动力学标定：gain_v=%s  τ_v=%s  漂移=%s m/m"
          % (kin.get("gain_v"), kin.get("tau_v"), kin.get("drift_m_per_m")))

    bud = None
    if not a.no_budget:
        bud = error_budget(rec, env, seed=a.seed)
        print("  误差预算：基线 %.3f m → 全开 %.3f m，残差 %.3f m"
              % (bud["base_rmse"], bud["all_rmse"], bud["residual"]))

    clo = None
    if a.closed_loop:
        # 闭环对照必须用**同一个场景布局**，否则与轨迹/域差两节的口径不一致
        clo = closed_loop_impact(env, policy=a.policy, episodes=a.episodes, seed=a.seed,
                                 env_kwargs={"layout": a.layout, "n_peds": 0})
        print("  闭环对照：%s 策略 %d 回合完成" % (a.policy, a.episodes))

    dens = None
    sd = scene_density(env.scene)
    if a.real_n is not None and a.real_area is not None:
        eq = equivalent_density(a.real_n, a.real_area, sd["area"], metric="objects")
        dens = dict(eq)
        dens["sim_density"] = sd
        # 误差率：仿真指标 vs 实机指标（论文4 表 29 口径）
        sim_sr = a.sim_sr
        if sim_sr is None and clo is not None and clo.get("rows"):
            sim_sr = float(clo["rows"][0]["SR"])       # 闭环对照的基线 SR
        if sim_sr is not None and a.real_sr is not None:
            dens["metric_name"] = "SR"
            dens["error_rate"] = density_error_rate(sim_sr, a.real_sr)
            print("  误差率：仿真 SR %.3f vs 实机 SR %.3f → 相对误差 %s"
                  % (sim_sr, a.real_sr,
                     ("%.1f%%" % (100 * dens["error_rate"]["rel"]))
                     if dens["error_rate"]["rel"] is not None else "—"))
        else:
            notes.append("未提供 `--sim-sr/--real-sr`（或未跑闭环对照），"
                         "等效密度一节只给换算规模、不给误差率。")
        print("  等效密度：实机 %s/%s m² → 仿真等效规模 %.1f"
              % (a.real_n, a.real_area, eq["sim_n_equiv"]))
    else:
        notes.append("未提供 `--real-n/--real-area`，跳过等效密度换算（论文4 范式）。")

    write_report(a.out, replay=rep, gap=gap, title=a.title, kinematics=kin,
                 budget=bud, density=dens, closed_loop=clo, notes=notes)
    print(f"\n报告已写入：{a.out}")

    if a.json:
        def _clean(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            if isinstance(o, dict):
                return {k: _clean(v) for k, v in o.items() if k != "per_step"}
            if isinstance(o, (list, tuple)):
                return [_clean(x) for x in o]
            return o
        payload = {"source": os.path.abspath(a.h5), "layout": a.layout,
                   "replay": _clean(rep), "domain_gap": _clean(gap),
                   "kinematics": _clean(kin), "budget": _clean(bud),
                   "closed_loop": _clean(clo), "density": _clean(dens),
                   "sim_density": _clean(sd)}
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"JSON 已写入：{a.json}")


if __name__ == "__main__":
    main()
