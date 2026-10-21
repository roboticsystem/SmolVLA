# -*- coding: utf-8 -*-
"""**能耗模型**（四段拆解）—— E6b「部署能耗与续航预算」的理论部分。

借鉴参考论文3 第 1 章的**三段式能耗拆解**（驱动行驶 / 差速转向 / 待机与辅助），
并按本课题的 VLA 部署场景把它扩成**四段**：

    ① 底盘驱动（行驶）  ② 差速转向  ③ **BPU 推理**（VLA 特有）  ④ 传感器待机

**为什么不直接抄论文3 的数值**：论文3 的能耗模型是**纯理论、无实测标定**，
其「能耗模型验证」一节只有数值仿真与特性分析、**没有误差统计**
（只给了「辅助能耗占 8–12%」这类区间）。我方借鉴时必须**补实测标定**——
这恰好回应 IDAS 审稿意见对我方「只有理论估算无实测」的批评。

三条从论文3 借来的结论（本模块实现了这三条，可直接用于论文）：

    (a) 行驶能耗：`F = (m₀+m_load)a + (m₀+m_load)gμ_r + ½ρC_dAv²`，
        `E = ∫F·v dt / η(v)`；效率 η(v) 分段非线性 ⇒ 存在 **U 型经济时速**；
    (b) 转向能耗随转角**超线性**：`E ∝ k_θ·(1+β·m_total)·|Δθ|^n`（n>1），
        ⇒ 「两次 45° 的总能耗 < 一次 90°」；
    (c) 待机/辅助能耗占匀速总能耗的 8%–12%（论文3 的实测区间，我方用于校准），
        但**我方必须实测替换**。
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ 默认参数
# 机器人本体（与 BOM 一致：7.4 V 3000 mAh 电池、TT 电机、差速底盘）
DEFAULT = {
    # --- ① 行驶 ---
    "m_base": 3.2,          # kg，机器人本体（含电池与电子设备）
    "m_load": 0.0,          # kg，额外负载（跟随任务默认空载）
    "mu_r": 0.06,           # 滚动阻力系数
    "rho": 1.225,           # kg/m³，空气密度
    "Cd_A": 0.06,           # Cd·A 组合（迎风面积）
    "eta_lo": 0.55,         # 低速效率（铜损主导）
    "eta_mid": 0.82,        # 中速效率（高效区）
    "eta_hi": 0.68,         # 高速效率（弱磁/铁损）
    "v_lo": 0.20,           # m/s，效率曲线分段点 1
    "v_hi": 0.45,           # m/s，效率曲线分段点 2
    # --- ② 转向 ---
    "k_theta": 0.55,        # J/rad，基础转向能耗系数
    "beta_load": 0.12,      # 负载放大系数（(1+β·m_total)）
    "n_theta": 2.0,         # 超线性指数（|Δθ|^n）
    # --- ④ 待机 ---
    "p_sensor": 2.6,        # W，传感器 + 通信 + 控制常开功耗
    # --- 电池 ---
    "v_batt": 7.4,          # V
    "capacity_mah": 3000.0, # mAh
    "soc_reserve": 0.2,     # 保留 20% 电量（论文3 的熔断口径）
    "conv_eff": 0.85,       # 电源转换效率
}


def eta(v, p=DEFAULT):
    """电机效率 η(v)：低速铜损主导 → 中速高效区 → 高速弱磁铁损（分段线性）。

    ⇒ **单位距离能耗**对速度呈 **U 形**：存在一个**经济时速**，不是越快越省。
    （可向量化：传入数组即返回数组。）
    """
    v = np.abs(np.asarray(v, dtype=float))
    if v.ndim == 0:
        v = float(v)
        if v <= p["v_lo"]:
            return p["eta_lo"]
        if v <= p["v_hi"]:
            w = (v - p["v_lo"]) / max(1e-9, p["v_hi"] - p["v_lo"])
            return p["eta_lo"] + w * (p["eta_mid"] - p["eta_lo"])
        return p["eta_hi"]
    out = np.full(v.shape, p["eta_hi"], dtype=float)
    out[v <= p["v_lo"]] = p["eta_lo"]
    mid = (v > p["v_lo"]) & (v <= p["v_hi"])
    w = (v[mid] - p["v_lo"]) / max(1e-9, p["v_hi"] - p["v_lo"])
    out[mid] = p["eta_lo"] + w * (p["eta_mid"] - p["eta_lo"])
    return out


def economic_speed(p=DEFAULT, dist_m=5.0):
    """**经济时速**（m/s）：在 [0.02, 1.0] 上搜索「**走完固定距离 D 的能耗最小**」的速度。

    注意必须**按等距离**比较（而不是等时间）——等时间比较会得到「越慢越省」的平凡结论。
    结论形态与论文3 一致：**存在一个最优经济速度，不是越快越省**。
    """
    best_v, best_e = 0.0, None
    for v in np.linspace(0.02, 1.0, 99):
        dur = dist_m / v
        n = max(2, int(round(dur / 0.1)))
        pos = np.stack([np.arange(n) * v * 0.1, np.zeros(n)], axis=1)
        e = drive_energy(pos, np.full(n, v), dt=0.1, p=p)["J"]
        if best_e is None or e < best_e:
            best_v, best_e = float(v), float(e)
    return best_v


def drive_energy(pos, v_cmds, dt=0.1, p=DEFAULT):
    """① **底盘驱动（行驶）能耗**。

    `pos`: (N,2) 位置序列（用于算加速度）；`v_cmds`: (N,) 速度指令（m/s）。
    返回 {'J': 总焦耳, 'avg_power_W': 平均功率, 'm_per_J': 每米能耗}。
    """
    pos = np.asarray(pos, dtype=float).reshape(-1, 2)
    v = np.abs(np.asarray(v_cmds, dtype=float).reshape(-1))
    n = min(len(pos), len(v))
    pos, v = pos[:n], v[:n]
    if n < 2:
        return {"J": 0.0, "avg_power_W": 0.0, "m_per_J": float("nan")}
    m = p["m_base"] + p["m_load"]
    # 加速度（中心差分，裁剪到物理合理范围）
    a = np.diff(v) / dt
    a = np.clip(a, -2.0, 2.0)
    a = np.concatenate([[0.0], a])
    # 滚阻 + 风阻
    f_roll = m * 9.81 * p["mu_r"]
    f_aero = 0.5 * p["rho"] * p["Cd_A"] * v ** 2
    f = m * a + f_roll + f_aero
    f = np.maximum(f, 0.0)                     # 制动/倒车不回收能量（保守）
    e_step = f * v * dt / np.maximum(eta(v, p), 1e-3)
    dist = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    J = float(np.sum(e_step))
    dur = n * dt
    return {"J": J, "avg_power_W": J / max(dur, 1e-9), "m_per_J": dist / max(J, 1e-9),
            "distance_m": dist, "duration_s": dur}


def turn_energy(dtheta, p=DEFAULT):
    """② **差速转向能耗**：E ∝ k_θ·(1+β·m_total)·|Δθ|^n（n>1 超线性）。"""
    d = np.abs(np.asarray(dtheta, dtype=float).reshape(-1))
    if d.size == 0:
        return {"J": 0.0, "total_rad": 0.0}
    m = p["m_base"] + p["m_load"]
    k = p["k_theta"] * (1.0 + p["beta_load"] * m)
    J = float(np.sum(k * d ** p["n_theta"]))
    return {"J": J, "total_rad": float(np.sum(d))}


def inference_energy(n_frames, power_w, p=DEFAULT):
    """③ **BPU 推理能耗**：P_infer × 帧数 × dt（板载实测 P 后直接代入）。"""
    n = int(max(0, n_frames))
    return {"J": float(n * power_w * 0.1), "frames": n, "power_w": float(power_w)}


def sensor_energy(duration_s, p=DEFAULT):
    """④ **传感器与常开电路**能耗：P_sensor × t。"""
    return {"J": float(p["p_sensor"] * max(0.0, duration_s)), "power_w": p["p_sensor"]}


def battery_budget(total_J, p=DEFAULT):
    """总能耗 → 电池消耗 → 可持续时长。

    返回 {'mAh': 消耗容量, 'runtime_min': 可持续分钟数, 'soc_usable': 可用电量 mAh}。
    """
    usable_mah = p["capacity_mah"] * (1.0 - p["soc_reserve"])
    # 电池侧能量 = 负载侧能量 / 转换效率
    batt_J = total_J / max(1e-6, p["conv_eff"])
    mah = batt_J / (p["v_batt"] * 3.6)        # 1 mAh = 3.6 J（7.4 V 下）
    runtime_min = (usable_mah / max(mah, 1e-9)) if mah > 0 else float("inf")
    return {"mAh": float(mah), "runtime_min": float(runtime_min),
            "usable_mAh": float(usable_mah), "total_J": float(total_J)}


def full_breakdown(pos, v_cmds, dtheta, n_frames, p_infer_w, p=None):
    """四段合计 + 电池预算（一次调用给出论文表格需要的全部数字）。"""
    p = p or DEFAULT
    d = drive_energy(pos, v_cmds, dt=0.1, p=p)
    t = turn_energy(dtheta, p=p)
    i = inference_energy(n_frames, p_infer_w, p=p)
    s = sensor_energy(d["duration_s"], p=p)
    total = d["J"] + t["J"] + i["J"] + s["J"]
    return {
        "drive": d, "turn": t, "inference": i, "sensor": s,
        "total_J": total,
        "shares": {"drive": d["J"] / max(total, 1e-9),
                   "turn": t["J"] / max(total, 1e-9),
                   "inference": i["J"] / max(total, 1e-9),
                   "sensor": s["J"] / max(total, 1e-9)},
        "battery": battery_budget(total, p=p),
    }
