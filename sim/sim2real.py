"""sim/sim2real.py —— **仿真↔实机一致性闭环**（论文 P5）。

背景：四篇参考论文都有「仿真结论能否外推到实机」这一环，而本项目此前缺三件东西：

1. 论文4《面向群体机器人的能耗管理与智能规划研究》给出「**仿真—实机等效密度换算
   验证框架**」：实机受场地/成本限制只能部署 N 辆小车，于是把实机的**空间密度**
   换算成仿真场地里的**等效规模**，再比较两边的死锁/拥堵趋势，报告**误差率**。
   → 本模块的 `equivalent_density()` / `density_error_rate()` 复刻该范式，
   只是把「AGV 数量密度」换成**导航场景的拥挤度**（障碍占地比 + 目标密度）。
2. 论文2《多传感器室内自动搬运机器人》要求给出实机定位误差（cm 级）→
   本模块的 `fit_kinematics()` / `error_budget()` 把「真机 vs 仿真」的总误差
   **分解成可归因的预算项**（加减速滞后 / 轮滑 / 执行器延迟 / 里程计漂移 / 残差）。
3. 论文1《嵌入式视觉路径识别》把「图像质量退化」作为核心指标 →
   `sim/replay.py` 的 `domain_gap()` 给出图像/激光的**观测域差**统计。

本模块只做**标定与换算**，不改动任何已有策略/环境的行为（论文主结果不受影响）。

典型用法：
    from sim.sim2real import fit_kinematics, error_budget, equivalent_density

    kin = fit_kinematics(rec)                 # 从真机录制拟合动力学参数
    bud = error_budget(rec, env)              # 误差预算分解
    eq  = equivalent_density(1, 6.0, 98.0)    # 1 台/6 m² → 等效 16.3 台/98 m²
"""
from __future__ import annotations

import numpy as np

from .world import CELL

# 误差预算的默认分解项。每一项都是「仿真侧可开关的退化开关」，
# 与 `sim/env.py` 的同名 cfg 参数一一对应（默认全关，论文主结果不受影响）。
DEFAULT_BUDGET_ITEMS = [
    ("accel", {"accel": 0.8, "acc_ang": 3.0}, "加减速斜坡（电机+惯量）"),
    ("slip", {"slip": 0.35}, "轮滑（湿滑/地毯，实际位移打折）"),
    ("act_delay", {"act_delay": 2}, "执行器延迟（串口/网络排队）"),
    ("odom_drift", {"odom_drift": 0.02}, "里程计漂移（只污染上报位姿）"),
]


# ================================================================ 动力学拟合
def _wrap(a):
    """角度归一化到 (-π, π]。"""
    return (np.asarray(a, dtype=np.float64) + np.pi) % (2 * np.pi) - np.pi


def fit_kinematics(rec, dt: float = 0.10, min_cmd: float = 0.05):
    """从真机录制的 (指令, 里程计) 拟合动力学退化参数。

    录制 schema（与 `deliver/tools/record_dataset.py` 对齐）：
        actions (N,2) = (v_cmd, w_cmd)      遥操/策略下发的指令
        states  (N,5) = (x, y, θ, v, ω)      里程计上报的位姿与**实测速度**
        gt_pose (N,3)                        *可选*：外部真值位姿（视觉定位/动捕）

    拟合四项：

    | 拟合量 | 物理含义 | 仿真侧开关 |
    |---|---|---|
    | `motor_gain_v/w` | **电机标定系数**：实测速度 = gain × 指令速度 | （仿真默认 1.0） |
    | `tau_v/w`         | 一阶滞后时间常数（电机+惯量，加减速斜坡） | `accel` / `acc_ang` |
    | `odom_drift_*`    | 里程计自洽性：上报位姿 vs 由上报速度积分的位姿 | `odom_drift` |
    | `odo_bias_*`      | **里程计偏差**（有真值时）：上报位移 vs 真值位移 | `slip` |

    **可辨识性说明（重要）**：只含里程计的录制里，**轮滑与里程计漂移不可分辨** ——
    里程计既不测「轮子转了多少」也不测「车走了多远」，两者都以「位姿误差」的形式
    混在一起。要分离它们必须引入**独立真值**（`gt_pose`：AprilTag/视觉定位/动捕）。
    本函数在有 `gt_pose` 时给出 `odo_bias`（里程计高估/低估的百分比）与
    `slip_p_equiv`（等价滑移概率）；没有真值时明确标注**不可辨识**并给出区间。

    返回 dict（无法拟合的项为 None，并在 `notes` 里说明原因）。
    """
    acts = np.asarray(rec.get("actions"), dtype=np.float64)
    states = rec.get("states")
    out = {"n": int(len(acts)), "dt": float(dt), "notes": []}
    if states is None:
        out["notes"].append("录制里没有 states（里程计），无法拟合动力学参数")
        return out
    states = np.asarray(states, dtype=np.float64)
    n = min(len(acts), len(states))
    a = acts[:n]
    s = states[:n]
    v_cmd, w_cmd = a[:, 0], a[:, 1]
    v_meas, w_meas = s[:, 3], s[:, 4]

    # ---- 1) 电机标定系数（指令 → 实测速度）----
    def _gain(cmd, meas):
        m = np.abs(cmd) > min_cmd
        if m.sum() < 5:
            return None, 0
        den = float(np.sum(cmd[m] ** 2))
        if den <= 1e-9:
            return None, int(m.sum())
        g = float(np.sum(cmd[m] * meas[m]) / den)
        return g, int(m.sum())

    gv, nv = _gain(v_cmd, v_meas)
    gw, nw = _gain(w_cmd, w_meas)
    out["motor_gain_v"], out["motor_gain_w"] = gv, gw
    out["gain_v"], out["gain_w"] = gv, gw        # 兼容别名
    out["n_gain_v"], out["n_gain_w"] = nv, nw
    if gv is None:
        out["notes"].append("直行指令样本不足（|v_cmd|>%.2f 的帧 <5），无法拟合前进增益" % min_cmd)
    if gw is None:
        out["notes"].append("转向指令样本不足（|w_cmd|>%.2f 的帧 <5），无法拟合转向增益" % min_cmd)

    # ---- 2) 一阶滞后时间常数 ----
    def _tau(cmd, meas):
        if len(meas) < 5:
            return None, 0
        d = meas[1:] - meas[:-1]
        e = cmd[:-1] - meas[:-1]
        m = np.abs(e) > min_cmd
        if m.sum() < 5:
            return None, int(m.sum())
        den = float(np.sum(e[m] ** 2))
        if den <= 1e-9:
            return None, int(m.sum())
        alpha = float(np.sum(d[m] * e[m]) / den)      # alpha = dt / tau
        if alpha <= 1e-6:
            return None, int(m.sum())
        return float(dt / alpha), int(m.sum())

    tv, ntv = _tau(v_cmd, v_meas)
    tw, ntw = _tau(w_cmd, w_meas)
    out["tau_v"], out["tau_w"] = tv, tw
    out["n_tau_v"], out["n_tau_w"] = ntv, ntw
    if tv is None:
        out["notes"].append("速度阶跃样本不足，无法拟合加减速时间常数")

    # ---- 3) 里程计自洽性（漂移率）----
    # 由上报速度积分出「死推位姿」，与上报位姿比较
    ix, iy, ith = float(s[0, 0]), float(s[0, 1]), float(s[0, 2])
    path = 0.0
    for i in range(1, n):
        ix += v_meas[i - 1] * np.cos(ith) * dt
        iy += v_meas[i - 1] * np.sin(ith) * dt
        ith = float(_wrap(ith + w_meas[i - 1] * dt))
        path += abs(float(v_meas[i - 1])) * dt
    dp = float(np.hypot(s[-1, 0] - ix, s[-1, 1] - iy))
    dyaw = float(abs(_wrap(s[-1, 2] - ith)))
    out["path_len_m"] = path
    out["drift_total_m"] = dp
    out["drift_total_rad"] = dyaw
    out["drift_m_per_m"] = float(dp / path) if path > 0.2 else None
    out["drift_rad_per_m"] = float(dyaw / path) if path > 0.2 else None
    if path <= 0.2:
        out["notes"].append("路径过短（<0.2 m），漂移率不可靠")

    # ---- 4) 里程计偏差（需要独立真值）----
    gt = rec.get("gt_pose")
    if gt is not None:
        gt = np.asarray(gt, dtype=np.float64)
        m = min(len(gt), len(s))
        if m >= 5:
            d_odo = float(np.sum(np.hypot(np.diff(s[:m, 0]), np.diff(s[:m, 1]))))
            d_gt = float(np.sum(np.hypot(np.diff(gt[:m, 0]), np.diff(gt[:m, 1]))))
            if d_gt > 0.2:
                bias = d_odo / d_gt
                out["odo_bias"] = float(bias)
                out["odo_bias_pct"] = float(100.0 * (bias - 1.0))
                out["gt_path_len_m"] = d_gt
                out["odo_path_len_m"] = d_odo
                # 仿真侧：slip 概率 p、每次位移折扣 U(0.45,0.85) → 期望折扣 1 − 0.35p
                # 真值位移 = 里程计位移 / bias ⇒ 令 1 − 0.35p = 1 / bias
                out["slip_p_equiv"] = float(np.clip((1.0 - 1.0 / max(bias, 1e-6)) / 0.35,
                                                   0.0, 1.0))
            else:
                out["notes"].append("真值路径过短，无法估计里程计偏差")
        else:
            out["notes"].append("真值帧数不足，无法估计里程计偏差")
    else:
        out["notes"].append(
            "录制里**没有独立真值位姿（gt_pose）**：轮滑与里程计漂移在只含里程计的"
            "数据里**不可分辨**，二者都表现为「位姿误差」。要分离需补一路外部真值"
            "（AprilTag / 视觉定位 / 动捕）。")

    # ---- 5) 与仿真退化开关的换算关系（供标定用）----
    #   加减速：accel 是「每步允许的速度变化量」(m/s²·dt)，
    #          一阶滞后 v' = v + (cmd - v)·dt/τ ⇒ 每步变化 = (cmd - v)·dt/τ
    #          ⇒ accel ≈ v_max/τ（取最大指令速度做尺度）
    if tv is not None and tv > 0:
        out["accel_equiv"] = float(0.35 / tv)
    if tw is not None and tw > 0:
        out["acc_ang_equiv"] = float(1.60 / tw)
    return out


# ================================================================ 误差预算
def error_budget(rec, env, seed=None, items=None, align=True, ref="auto"):
    """把「真机轨迹 vs 仿真轨迹」的总误差**分解成可归因的预算项**。

    **参考口径 `ref`**（决定「跟谁比」）：
        - `"gt"`   ：仿真真值轨迹 vs 真机**地面真值**轨迹 —— 最干净，差异全部来自动力学；
        - `"odom"` ：仿真真值轨迹 vs 真机**里程计**轨迹 —— 只有里程计时唯一可选，
                     但差异里混着里程计自身的偏差（轮滑/漂移）；
        - `"auto"`（默认）：录制里有 `gt_pose` 就用 `"gt"`，否则用 `"odom"`。

    **第一层：分离「真机侧固有偏差」与「仿真侧动力学差异」**
        E_total ：指令回放（`mode="command"`）的轨迹误差
        E_odom  ：实测速度回放（`mode="measured"`）的轨迹误差
                  —— 这一步绕过仿真动力学、用真机自己的速度积分位姿，
                  得到的差异**只可能来自里程计本身**（滤波输出 vs 速度积分）。
        E_dyn   ：E_total − E_odom ≈ **可归因于动力学的部分**。

    **第二层：在动力学内部逐项消融（one-at-a-time）**
        基线   ：无任何退化的仿真重放 → RMSE_0
        逐项   ：只开第 i 项退化 → RMSE_i
        解释量 ：explained_i = RMSE_0 − RMSE_i
                 （>0 表示该退化**确实存在于真机**、能解释掉一部分差距；
                   <0 表示真机上没有这个现象，仿真里加它反而更差）
        全开   ：所有退化同时开 → RMSE_all
        残差   ：RMSE_all − RMSE_0 + Σ explained_i（交互项 + 未建模因素）

    **为什么 `odom_drift` 不出现在轨迹误差里**：仿真侧的 `odom_drift` 只污染
    **上报给策略的 state**（真值 `env.pose` 不变），所以开环重放的**真实轨迹**
    完全不受影响 —— 它的影响只体现在**闭环**性能上（策略被误导）。
    因此本函数把它单列为「闭环项」，并注明其量化方式（见 `closed_loop_impact`）。

    返回 dict：
        base_rmse / all_rmse / residual / odom_rmse / dyn_part / ref
        items  : [{name, label, rmse, explained, pct_of_base}, ...]
        kinematics : 拟合出的动力学参数（供交叉核对）
    """
    from .replay import replay_in_sim

    if ref == "auto":
        ref = "gt" if rec.get("gt_pose") is not None else "odom"
    items = list(items if items is not None else DEFAULT_BUDGET_ITEMS)
    base = replay_in_sim(rec, env, seed=seed, degrade={}, align=align,
                         mode="command", ref=ref)
    rmse0 = float(base.get("pos_rmse", float("nan")))

    # ---- 第一层：里程计口径（真机侧固有偏差）----
    odom = replay_in_sim(rec, env, seed=seed, degrade={}, align=align,
                         mode="measured", ref=ref)
    rmse_odom = float(odom.get("pos_rmse", float("nan")))

    rows = []
    for name, cfg, label in items:
        if name == "odom_drift":
            continue                      # 开环轨迹不受影响，见 docstring
        r = replay_in_sim(rec, env, seed=seed, degrade=cfg, align=align,
                          mode="command", ref=ref)
        rm = float(r.get("pos_rmse", float("nan")))
        explained = rmse0 - rm
        rows.append({"name": name, "label": label, "rmse": rm,
                     "explained": float(explained),
                     "pct_of_base": float(100.0 * explained / rmse0)
                     if rmse0 > 0 else float("nan")})

    allc = {}
    for name, cfg, _ in items:
        if name == "odom_drift":
            continue
        allc.update(cfg)
    allr = replay_in_sim(rec, env, seed=seed, degrade=allc, align=align,
                         mode="command", ref=ref)
    rmse_all = float(allr.get("pos_rmse", float("nan")))

    expl_sum = sum(r["explained"] for r in rows)
    residual = rmse_all - rmse0 + expl_sum

    kin = fit_kinematics(rec, dt=float(env.cfg.get("dt", 0.1)))
    return {"base_rmse": rmse0, "all_rmse": rmse_all,
            "odom_rmse": rmse_odom, "ref": ref,
            "dyn_part": float(rmse0 - rmse_odom),
            "sum_explained": float(expl_sum), "residual": float(residual),
            "items": rows, "n_steps": base.get("n", 0),
            "kinematics": kin}


def fit_dynamics_grid(rec, env, seed=None, align=True,
                      accels=(0.0, 0.4, 0.8, 1.5),
                      slips=(0.0, 0.2, 0.4, 0.6),
                      delays=(0, 1, 2, 3),
                      acc_ang_scale=3.0):
    """网格搜索：找一组退化参数，使**仿真重放的轨迹最贴近真机轨迹**。

    这是**轮滑/加减速/延迟**这三个量的真正标定手段 ——
    它们在「只含里程计」的录制里与里程计偏差不可分辨（见 `fit_kinematics`），
    但一旦给定真值轨迹（或用「实测速度回放」扣掉里程计偏差），
    就可以用**回放误差最小**这个判据把它们分开。

    返回 dict：best（参数组合与其 RMSE）、grid（全部候选，便于画图/查表）。
    """
    from .replay import replay_in_sim

    grid = []
    for ac in accels:
        for sl in slips:
            for dl in delays:
                cfg = {"accel": float(ac), "acc_ang": float(ac) * acc_ang_scale,
                       "slip": float(sl), "act_delay": int(dl)}
                r = replay_in_sim(rec, env, seed=seed, degrade=cfg, align=align)
                grid.append({"cfg": cfg, "rmse": float(r.get("pos_rmse", float("nan")))})
    grid.sort(key=lambda g: (g["rmse"] if g["rmse"] == g["rmse"] else 1e9))
    best = grid[0] if grid else None
    return {"best": best, "grid": grid, "n": len(grid)}


def closed_loop_impact(env, policy="oracle", episodes=20, seed=0,
                       items=None, env_kwargs=None):
    """闭环口径下各退化项对**任务指标**的影响（里程计漂移只在这里体现）。

    做法：同一策略、同一批回合种子，逐项打开退化，比较 SR / SPL 的变化。
    与论文的「同一回合集上的配对比较」口径一致（回合种子相同 → 可配对）。
    """
    from .run_eval import evaluate

    items = list(items if items is not None else DEFAULT_BUDGET_ITEMS)
    kw = dict(env_kwargs or {})
    out = []
    base_agg, _ = evaluate(policy, episodes, seed=seed, env_kwargs=kw, verbose=False)
    out.append({"name": "baseline", "label": "无退化", "SR": base_agg["SR"],
                "SPL": base_agg["SPL"], "dSR": 0.0, "dSPL": 0.0})
    for name, cfg, label in items:
        kw2 = dict(kw)
        kw2.update(cfg)
        agg, _ = evaluate(policy, episodes, seed=seed, env_kwargs=kw2, verbose=False)
        out.append({"name": name, "label": label, "SR": agg["SR"], "SPL": agg["SPL"],
                    "dSR": agg["SR"] - base_agg["SR"], "dSPL": agg["SPL"] - base_agg["SPL"]})
    return {"policy": policy, "episodes": episodes, "rows": out}


# ================================================================ 等效密度换算
def scene_density(scene, area_m2=None):
    """场景拥挤度统计（仿真场景 → 可与实机场景对比的量）。

    返回：
        area        : 场地面积（m²）。缺省用 `scene.bounds` 的外接矩形
        occ_area    : 障碍占地面积（m²）= 占据栅格面积
        occ_ratio   : **障碍占地比** ρ_occ（主密度指标，无量纲）
        free_ratio  : 自由面积占比
        n_objects   : 语义目标数量
        obj_density : **目标密度**（个/m²）
        n_rooms     : 房间数
    """
    b = scene.bounds
    if area_m2 is None:
        area_m2 = float((b[1] - b[0]) * (b[3] - b[2]))
    occ_cells = int(np.count_nonzero(scene.occ))
    occ_area = occ_cells * CELL * CELL
    rooms = getattr(scene, "rooms", None) or []
    n_obj = len(getattr(scene, "objects", []) or [])
    return {"area": float(area_m2),
            "occ_area": float(occ_area),
            "occ_ratio": float(occ_area / area_m2) if area_m2 > 0 else float("nan"),
            "free_ratio": float(1.0 - occ_area / area_m2) if area_m2 > 0 else float("nan"),
            "n_objects": int(n_obj),
            "obj_density": float(n_obj / area_m2) if area_m2 > 0 else float("nan"),
            "n_rooms": int(len(rooms))}


def equivalent_density(real_n, real_area, sim_area, metric="objects"):
    """**等效密度换算**（论文4 范式）。

    论文4 的做法：实机因场地/成本限制只能部署 6 辆 AGV（3 m × 2 m），
    于是把实机的**空间密度**换算到仿真中型场地（75 m × 45 m），
    得到「等效密度」下的仿真规模（80~100 辆），再比较两边的死锁趋势。

    本函数是同一范式在**导航场景**下的实例：
        ρ_real = real_n / real_area                （实机部署密度）
        n_sim_equiv = ρ_real × sim_area            （仿真场地的等效规模）

    `metric` 只用于给结果起名（"objects"=语义目标数，"agents"=机器人数量）。

    返回 dict：ρ_real、sim_n_equiv、sim_area、real_area。
    """
    real_area = float(real_area)
    sim_area = float(sim_area)
    rho = float(real_n) / real_area if real_area > 0 else float("nan")
    return {"metric": metric,
            "real_n": float(real_n), "real_area": real_area,
            "sim_area": sim_area,
            "rho_real": rho,
            "sim_n_equiv": float(rho * sim_area)}


def density_error_rate(sim_value, real_value):
    """**误差率**（论文4 表 29 的口径）：|仿真 − 实机| / 实机。

    实机为 0 时无法用相对误差，改为返回绝对差（并置 `rel=None`）。
    """
    s, r = float(sim_value), float(real_value)
    diff = abs(s - r)
    rel = float(diff / abs(r)) if abs(r) > 1e-12 else None
    return {"sim": s, "real": r, "abs_diff": diff, "rel": rel}


def density_matched_scene(real_density, seed=0, layouts=("apartment", "corridor",
                                                         "s_corridor", "warehouse"),
                           n_per_room=(1, 5), nx=(1, 4), ny=(1, 3)):
    """在仿真里找**与实机场景拥挤度等效**的场景配置。

    做法：遍历 (房间数 × 每房间物体数) 组合，算出各配置的 ρ_occ，
    取与 `real_density` 最接近的一组作为「等效密度场景」。

    论文4 的对应关系：实机 6 辆/3 m×2 m ↔ 仿真 80~100 辆/75 m×45 m。
    """
    from .world import make_scene
    best = None
    for lay in layouts:
        for nxx in range(nx[0], nx[1] + 1):
            for nyy in range(ny[0], ny[1] + 1):
                for lo in range(n_per_room[0], n_per_room[1] + 1):
                    for hi in range(lo, n_per_room[1] + 1):
                        sc = make_scene(seed=seed, nx=nxx, ny=nyy,
                                        n_objects_per_room=(lo, hi))
                        d = scene_density(sc)
                        err = abs(d["occ_ratio"] - float(real_density))
                        if best is None or err < best["err"]:
                            best = {"err": float(err), "layout": lay, "nx": nxx, "ny": nyy,
                                    "n_objects_per_room": (lo, hi), "density": d}
    return best
