"""sim/replay.py —— **仿真↔实机一致性**的回放与量化（论文 P5）。

背景（四篇参考论文都有、而本项目此前缺的一环）：
- 论文4《仓储多 AGV 分拣》有「**仿真—实机等效密度换算验证框架**」：把仿真
  场景的空间密度换算成实机可部署密度，论证仿真结论能外推；
- 论文2《多传感器搬运机器人》有「4 s 完全遮挡实验」与实机定位误差（cm 级）；
- 论文1《嵌入式视觉路径识别》把「端到端延迟 / 图像质量退化」作为核心指标。

本模块提供两件可复用的东西：

1. `replay_in_sim(rec, env)`：**把真机录到的动作序列在仿真里重放**，比较
   「真机实际走出来的轨迹」与「仿真按同样指令走出来的轨迹」，给出位置/航向
   误差统计 —— 这是量化 sim-to-real 差距最直接的证据（也用来标定仿真的
   动力学退化参数）。
   **真机与仿真的坐标系不可比**，所以误差统一在**刚体对齐（旋转+平移）后**统计
   （`align_2d`），否则坐标系差异会淹没真正的动力学差异。
2. `domain_gap(rec, env)`：**观测域差**统计：把真机图像/激光与仿真对应观测
   放在一起比（亮度/对比度/边缘密度/激光距离分布），量化「仿真画面离真机
   有多远」，为后续的域随机化提供依据。
   真机轨迹会先**锚定到仿真出生点**（`anchor="spawn"`）再渲染 —— 否则真机
   里程计坐标多半落在仿真场景外，统计毫无意义。

录制格式（与 `deliver/tools/record_dataset.py` 的 HDF5 对齐）：
    actions (N,2) f32   states (N,5) f32  [x,y,θ,v,ω]（真机里程计）
    lidar  (N,120) f32  images (N,H,W,3) u8（可选）

用法:
    from sim.replay import load_recording, replay_in_sim, domain_gap
    rec = load_recording("data/session.h5")
    st = replay_in_sim(rec, env)
    dg = domain_gap(rec, env)
"""
from __future__ import annotations

import json
import os

import numpy as np


# ---------------------------------------------------------------- 载入
def load_recording(path):
    """载入录制数据 → dict(actions, states, lidar, images, meta)。

    支持 HDF5（`deliver/tools/record_dataset.py` 的 schema）与 JSON/NPZ。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    out = {"actions": None, "states": None, "lidar": None, "images": None,
           "gt_pose": None, "meta": {}}
    if ext in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            for k in ("actions", "states", "lidar", "images", "gt_pose"):
                if k in f:
                    out[k] = np.asarray(f[k])
            for k in f.keys():
                if k not in out and hasattr(f[k], "shape"):
                    out["meta"][k] = np.asarray(f[k])
            for k, v in f.attrs.items():
                out["meta"].setdefault("_attrs", {})[k] = v
    elif ext == ".npz":
        d = np.load(path)
        for k in list(out):
            if k in d:
                out[k] = d[k]
    else:
        d = json.load(open(path, encoding="utf-8"))
        for k in list(out):
            if k in d:
                out[k] = np.asarray(d[k])
        out["meta"] = d.get("meta", {})
    if out["actions"] is None:
        raise ValueError("录制文件里没有 actions 字段：%s" % path)
    out["actions"] = np.asarray(out["actions"], dtype=np.float64)
    if out["states"] is not None:
        out["states"] = np.asarray(out["states"], dtype=np.float64)
    if out["lidar"] is not None:
        out["lidar"] = np.asarray(out["lidar"], dtype=np.float64)
    return out


# ---------------------------------------------------------------- 轨迹误差
def _pose_err(p_a, p_b, yaw_a=None, yaw_b=None):
    """位姿差（米 / 弧度）。

    位置取欧氏距离；航向差按**角度归一化**到 (-π, π]
    （否则 ±π 附近的微小差异会被算成 2π 的假误差）。
    """
    d = np.asarray(p_a, dtype=float) - np.asarray(p_b, dtype=float)
    dist = float(np.hypot(d[0], d[1]))
    if yaw_a is None or yaw_b is None:
        return dist, 0.0
    return dist, float(_wrap(float(yaw_a) - float(yaw_b)))


def align_2d(src_xy, dst_xy):
    """2D 刚体变换（旋转 + 平移），把 `src_xy` 最小二乘地对齐到 `dst_xy`。

    **为什么必须先对齐**：真机里程计的原点与朝向跟仿真坐标系没有任何可比性
    （真机的原点是它自己开机的地方）。不先对齐，「轨迹误差」里会混进坐标系差异，
    把真正的动力学差异淹没掉。

    用 Umeyama 的 2D 特例（允许反射时会被 `det` 符号修正掉），
    返回 `(R, t)`，使得 `R @ src + t ≈ dst`。
    """
    src = np.asarray(src_xy, dtype=np.float64)
    dst = np.asarray(dst_xy, dtype=np.float64)
    n = min(len(src), len(dst))
    if n < 2:
        t = (dst[:n].mean(axis=0) - src[:n].mean(axis=0)) if n else np.zeros(2)
        return np.eye(2), t
    src = src[:n]
    dst = dst[:n]
    sc = src - src.mean(axis=0)
    dc = dst - dst.mean(axis=0)
    U, S, Vt = np.linalg.svd(sc.T @ dc)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    t = dst.mean(axis=0) - R @ src.mean(axis=0)
    return R, t


def _wrap(a):
    return (np.asarray(a, dtype=np.float64) + np.pi) % (2 * np.pi) - np.pi


def replay_in_sim(rec, env, seed=None, degrade=None, dt=None, verbose=False,
                  align=True, mode="command", ref="odom"):
    """把录制动作在仿真里重放，量化「真机轨迹 vs 仿真轨迹」的偏差。

    `mode` 决定**重放什么**（这是误差归因的关键分界）：

    - `"command"`（默认）：重放**指令序列** `actions`，走仿真的完整动力学
      （含加减速斜坡 / 轮滑 / 执行器延迟等退化开关）。
      → 与真机里程计位姿的差异 = **动力学差异 + 里程计偏差**。
    - `"measured"`：**绕过仿真动力学**，直接用录制的**实测速度** (v, ω) 积分位姿。
      里程计本身就是「由实测速度积分而来」的，所以这一步测的是
      **里程计自身的自洽性/漂移**（滤波输出 vs 速度积分）。
      → 与真机里程计位姿的差异 = **纯里程计偏差**（真机侧固有，仿真无法消除）。

    两者相减 ≈ **可归因于动力学的部分**（见 `sim.sim2real.error_budget`）。

    `ref` 决定**跟谁比**：
    - `"odom"`（默认）：跟真机**里程计**位姿比 —— 只有里程计时唯一可选，
      但它本身有偏（轮滑/漂移都表现为位姿误差）。
    - `"gt"`：跟真机**地面真值**位姿比（录制里带 `gt_pose` 时才有）——
      这是**最干净**的口径：仿真的真值轨迹 vs 真机的真值轨迹，
      差异全部来自动力学/几何，不含里程计偏差。

    返回 dict：
        n            : 重放步数
        pos_rmse     : 位置误差 RMSE（米）。默认**刚体对齐后**的值
                       （真机与仿真坐标系不可比，不对齐会把坐标系差异算成误差）
        pos_rmse_raw : 未对齐的原始 RMSE（仅作参考）
        pos_final    : 终点位置误差（米，对齐后）
        yaw_rmse     : 航向误差 RMSE（弧度，对齐后）
        pos_err_p50/p90 : 位置误差分位数
        per_step     : 逐步误差（供画图）
        collided     : 仿真重放中的碰撞次数
        success      : 仿真重放是否到达目标
        align_rot_deg / align_trans : 对齐所用的旋转（度）和平移（米）
        init_offset  : 对齐前第 0 步的位姿差（米）——「坐标系/起点差异」的大小
    """
    env.cfg.update(degrade or {})
    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()
    acts = rec["actions"]
    states = rec["states"]
    gt = rec.get("gt_pose")
    if ref == "gt":
        if gt is None:
            raise ValueError("ref='gt' 需要录制里带 gt_pose（地面真值位姿）")
        ref_pose = np.asarray(gt, dtype=np.float64)
    else:
        ref_pose = np.asarray(states, dtype=np.float64) if states is not None else None
    dt = float(dt or env.cfg["dt"])
    per_step = []
    collided = 0
    info = None
    if mode == "measured":
        # 用实测速度积分位姿（绕过仿真动力学），只保留碰撞判定
        if states is None:
            raise ValueError("mode='measured' 需要 states 里的实测速度 (v, ω)")
        n_use = min(len(acts), len(states), len(ref_pose))
        for i in range(n_use):
            v = float(states[i][3])
            w = float(states[i][4])
            th = float(env.pose[2])
            nx = float(env.pose[0]) + v * np.cos(th) * dt
            ny = float(env.pose[1]) + v * np.sin(th) * dt
            nth = _wrap(th + w * dt)
            if env.scene.is_free((nx, ny), margin=0.10):
                env.pose = np.array([nx, ny, nth], dtype=np.float64)
            else:
                env.pose = np.array([env.pose[0], env.pose[1], nth], dtype=np.float64)
                collided += 1
            pe, ye = _pose_err(ref_pose[i][:2], env.pose[:2],
                               ref_pose[i][2], env.pose[2])
            per_step.append((float(ref_pose[i][0]), float(ref_pose[i][1]),
                             float(ref_pose[i][2]), float(env.pose[0]),
                             float(env.pose[1]), float(env.pose[2]),
                             pe, ye))
    else:
        for i in range(len(acts)):
            a = acts[i]
            obs, _r, done, info = env.step(a)
            collided = int(info["n_collide"])
            if ref_pose is not None and i < len(ref_pose):
                pe, ye = _pose_err(ref_pose[i][:2], env.pose[:2],
                                   ref_pose[i][2], env.pose[2])
                per_step.append((float(ref_pose[i][0]), float(ref_pose[i][1]),
                                 float(ref_pose[i][2]), float(env.pose[0]),
                                 float(env.pose[1]), float(env.pose[2]),
                                 pe, ye))
            if done:
                break
    out = {"n": len(per_step), "collided": collided, "mode": mode, "ref": ref,
           "success": bool(info and info.get("success"))}
    if per_step:
        arr = np.asarray(per_step)
        pe_raw = arr[:, 6]
        ye_raw = np.abs(arr[:, 7])
        # ---- 刚体对齐（真机坐标系 vs 仿真坐标系）----
        pe, ye = pe_raw, ye_raw
        rot_deg, trans = 0.0, float("nan")
        if align and len(arr) >= 2:
            R, t = align_2d(arr[:, 3:5], arr[:, 0:2])
            sim_al = (R @ arr[:, 3:5].T).T + t
            d_shift = float(np.arctan2(R[1, 0], R[0, 0]))
            yaw_sim_al = _wrap(arr[:, 5] + d_shift)
            pe = np.hypot(sim_al[:, 0] - arr[:, 0], sim_al[:, 1] - arr[:, 1])
            ye = np.abs(_wrap(yaw_sim_al - arr[:, 2]))
            rot_deg = float(np.rad2deg(d_shift))
            trans = float(np.linalg.norm(t))
        out.update({
            "pos_rmse": float(np.sqrt(np.mean(pe ** 2))),
            "pos_rmse_raw": float(np.sqrt(np.mean(pe_raw ** 2))),
            "pos_final": float(pe[-1]),
            "pos_err_p50": float(np.percentile(pe, 50)),
            "pos_err_p90": float(np.percentile(pe, 90)),
            "yaw_rmse": float(np.sqrt(np.mean(ye ** 2))),
            "yaw_rmse_raw": float(np.sqrt(np.mean(ye_raw ** 2))),
            "align_rot_deg": rot_deg,
            "align_trans": trans,
            "init_offset": float(pe_raw[0]),
            "per_step": per_step,
        })
    if verbose:
        print("[replay] 步数 %d 位置RMSE %.3f m（未对齐 %.3f）终点误差 %.3f m "
              "航向RMSE %.3f rad"
              % (out.get("n", 0), out.get("pos_rmse", float("nan")),
                 out.get("pos_rmse_raw", float("nan")),
                 out.get("pos_final", float("nan")),
                 out.get("yaw_rmse", float("nan"))))
    return out


# ---------------------------------------------------------------- 观测域差
def _img_stats(img):
    """图像统计量：亮度/对比度/饱和度/边缘密度/暗像素/过曝占比。"""
    g = np.asarray(img, dtype=np.float64)
    if g.ndim == 3:
        lum = g.mean(axis=2)
        sat = g.max(axis=2) - g.min(axis=2)
    else:
        lum = g
        sat = np.zeros_like(g)
    gy = np.abs(np.diff(lum, axis=0)).mean() if lum.shape[0] > 1 else 0.0
    gx = np.abs(np.diff(lum, axis=1)).mean() if lum.shape[1] > 1 else 0.0
    return {"mean": float(lum.mean()), "std": float(lum.std()),
            "sat": float(sat.mean()), "edge": float((gx + gy) / 2.0),
            "dark_frac": float((lum < 60).mean()),
            "over_frac": float((lum > 245).mean())}


def _lidar_stats(scan, max_range=12.0):
    s = np.asarray(scan, dtype=np.float64)
    v = s[s < max_range - 1e-6]
    if v.size == 0:
        return {"mean": float("nan"), "p10": float("nan"), "p50": float("nan"),
                "p90": float("nan"), "hit_frac": 0.0, "near_frac": 0.0}
    return {"mean": float(v.mean()), "p10": float(np.percentile(v, 10)),
            "p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)),
            "hit_frac": float(v.size / max(len(s), 1)),
            "near_frac": float((v < 1.0).mean())}


def _resize_img(img, w, h):
    """把图像缩放到 (w, h)（双线性重采样，无外部依赖）。

    域差统计必须在**同一分辨率**下比较：边缘密度（相邻像素差均值）
    随分辨率变化，不缩放就没有可比性（真机 640×480 vs 仿真 160×120）。
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 2:
        a = a[..., None]
    H, W = a.shape[:2]
    if (H, W) == (h, w):
        return a
    # 双线性：先沿 x 再沿 y，用两次线性插值实现
    xi = np.clip((np.arange(w) + 0.5) * W / w - 0.5, 0, W - 1)
    yi = np.clip((np.arange(h) + 0.5) * H / h - 0.5, 0, H - 1)
    x0 = np.floor(xi).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1)
    wx = (xi - x0)[None, :, None]
    y0 = np.floor(yi).astype(int)
    y1 = np.clip(y0 + 1, 0, H - 1)
    wy = (yi - y0)[:, None, None]
    top = a[:, x0, :] * (1 - wx) + a[:, x1, :] * wx
    bot = a[:, x0, :] * (1 - wx) + a[:, x1, :] * wx
    return top[y0, :, :] * (1 - wy) + bot[y1, :, :] * wy


def _anchor_transform(states, anchor_xy, anchor_yaw):
    """把真机位姿序列**刚体映射**到仿真场景的出生点。

    真机的里程计原点是「开机那一刻的位置」，与仿真坐标系没有任何关系；
    若直接把真机坐标塞进仿真去渲染，多数情况下会落在场景外（激光全是 12 m、
    图像全是墙），域差统计就失去意义。

    做法：以**仿真出生点**为锚点 —— 真机起始位姿对齐到仿真出生位姿，
    之后真机的相对运动（位移与转角）被原样搬到仿真里。
    返回 (R, t, d_yaw)，位姿映射为 `p_sim = R @ p_real + t`，`yaw_sim = yaw_real + d_yaw`。
    """
    r0 = np.asarray(states[0], dtype=np.float64)
    d_yaw = float(anchor_yaw - r0[2])
    c, s = np.cos(d_yaw), np.sin(d_yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    t = np.asarray(anchor_xy, dtype=np.float64) - R @ r0[:2]
    return R, t, d_yaw


def domain_gap(rec, env, seed=None, n_frames=60, degrade=None, anchor="spawn"):
    """真机观测 vs 仿真观测的域差统计（图像 + 激光）。

    做法：取录制中的前 `n_frames` 帧，把**真机的里程计位姿**（经刚体变换映射到
    仿真场景的出生点）放进仿真里渲染同位姿的观测，然后逐项比较统计量
    （亮度/对比度/边缘密度；激光距离分布、近距回波占比）。
    差得越多，说明仿真越需要加强域随机化。

    `anchor`：
        "spawn"（默认）：把真机轨迹对齐到仿真出生点 —— 真机坐标系与仿真坐标系
            不可比，不对齐时多数帧会落在场景外，统计失去意义；
        "raw"：直接用真机坐标（仅当真机录制就是在同一仿真场景里做的才合理）。
    """
    states = rec.get("states")
    if states is None:
        raise ValueError("录制里没有 states（真机位姿），无法对齐比较")
    states = np.asarray(states, dtype=np.float64)
    images = rec.get("images")
    lidar = rec.get("lidar")
    if degrade:
        env.cfg.update(degrade)
    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()
    if anchor == "spawn":
        R, t, d_yaw = _anchor_transform(states, env.scene.spawn, env.scene.spawn_theta)
    else:
        R, t, d_yaw = np.eye(2), np.zeros(2), 0.0
    n = min(n_frames, len(states))
    sim_img, sim_lid = [], []
    for i in range(n):
        p = R @ states[i][:2] + t
        env.pose = np.array([p[0], p[1], states[i][2] + d_yaw], dtype=np.float64)
        obs = env._obs()
        sim_img.append(obs["image"])
        sim_lid.append(obs["lidar"])
    out = {"n": n, "anchor": anchor,
           "anchor_trans": float(np.linalg.norm(t)), "anchor_rot_deg": float(np.rad2deg(d_yaw))}
    sim_h, sim_w = env.cfg["obs_size"][1], env.cfg["obs_size"][0]
    if images is not None and len(images) >= out["n"]:
        real_stats = [_img_stats(_resize_img(images[i], sim_w, sim_h))
                      for i in range(out["n"])]
        sim_stats = [_img_stats(im) for im in sim_img]
        keys = real_stats[0].keys()
        out["image"] = {
            k: {"real": float(np.mean([s[k] for s in real_stats])),
                "sim": float(np.mean([s[k] for s in sim_stats])),
                "gap": float(np.mean([s[k] for s in sim_stats])
                             - np.mean([s[k] for s in real_stats]))}
            for k in keys}
    if lidar is not None and len(lidar) >= out["n"]:
        real_l = [_lidar_stats(lidar[i]) for i in range(out["n"])]
        sim_l = [_lidar_stats(l) for l in sim_lid]
        keys = real_l[0].keys()
        out["lidar"] = {
            k: {"real": float(np.nanmean([s[k] for s in real_l])),
                "sim": float(np.nanmean([s[k] for s in sim_l])),
                "gap": float(np.nanmean([s[k] for s in sim_l])
                             - np.nanmean([s[k] for s in real_l]))}
            for k in keys}
    return out


# ---------------------------------------------------------------- 报告
def _fmt_table(rows, header):
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def write_report(path, replay=None, gap=None, title="仿真—实机一致性报告",
                 kinematics=None, budget=None, density=None, closed_loop=None,
                 notes=None):
    """把回放误差与域差写成 Markdown 报告（论文可直接引用/改写）。

    各节可独立缺省：
        replay      : 轨迹重放误差（`replay_in_sim` 的返回）
        gap         : 观测域差（`domain_gap` 的返回）
        kinematics  : 动力学参数拟合（`sim.sim2real.fit_kinematics`）
        budget      : 误差预算分解（`sim.sim2real.error_budget`）
        density     : 等效密度换算结果
        closed_loop : 闭环口径下各退化项对 SR/SPL 的影响
    """
    lines = ["# %s\n" % title, ""]
    if replay:
        lines += ["## 1. 轨迹重放误差（真机指令 → 仿真执行）\n",
                  "> 真机与仿真坐标系不可比，误差均在**刚体对齐（旋转+平移）后**统计；\n"
                  "> 未对齐的原始值仅作参考。\n",
                  _fmt_table([
                      ["重放步数", replay.get("n", 0)],
                      ["位置误差 RMSE (m)", "%.3f" % replay.get("pos_rmse", float("nan"))],
                      ["位置误差 RMSE（未对齐, m）", "%.3f" % replay.get("pos_rmse_raw", float("nan"))],
                      ["位置误差中位数 (m)", "%.3f" % replay.get("pos_err_p50", float("nan"))],
                      ["位置误差 P90 (m)", "%.3f" % replay.get("pos_err_p90", float("nan"))],
                      ["终点误差 (m)", "%.3f" % replay.get("pos_final", float("nan"))],
                      ["航向误差 RMSE (rad)", "%.3f" % replay.get("yaw_rmse", float("nan"))],
                      ["对齐旋转 (deg)", "%.2f" % replay.get("align_rot_deg", 0.0)],
                      ["初始位姿差 (m)", "%.3f" % replay.get("init_offset", float("nan"))],
                      ["重放中碰撞次数", replay.get("collided", 0)],
                  ], ["指标", "数值"]), ""]
    if gap and "image" in gap:
        rows = [[k, "%.2f" % v["real"], "%.2f" % v["sim"], "%+.2f" % v["gap"]]
                for k, v in gap["image"].items()]
        lines += ["## 2. 图像域差（真机 vs 仿真，同位姿渲染）\n",
                  _fmt_table(rows, ["统计量", "真机", "仿真", "差(仿真-真机)"]), ""]
    if gap and "lidar" in gap:
        rows = [[k, "%.2f" % v["real"], "%.2f" % v["sim"], "%+.2f" % v["gap"]]
                for k, v in gap["lidar"].items()]
        lines += ["## 3. 激光域差\n",
                  _fmt_table(rows, ["统计量", "真机", "仿真", "差(仿真-真机)"]), ""]
    if kinematics:
        k = kinematics
        lines += ["## 4. 动力学参数标定（从真机录制拟合）\n",
                  _fmt_table([
                      ["电机标定系数 gain_v（实测/指令）", _fmt_num(k.get("motor_gain_v"))],
                      ["电机标定系数 gain_w", _fmt_num(k.get("motor_gain_w"))],
                      ["前进一阶滞后 τ_v (s)", _fmt_num(k.get("tau_v"))],
                      ["转向一阶滞后 τ_w (s)", _fmt_num(k.get("tau_w"))],
                      ["里程计自洽性漂移 (m/m)", _fmt_num(k.get("drift_m_per_m"))],
                      ["里程计航向漂移 (rad/m)", _fmt_num(k.get("drift_rad_per_m"))],
                      ["里程计偏差（相对真值）", _fmt_num(k.get("odo_bias"))],
                      ["里程计高估比例", ("%+.1f%%" % k["odo_bias_pct"])
                       if k.get("odo_bias_pct") is not None else "—（无独立真值）"],
                      ["路径长度 (m)", "%.2f" % k.get("path_len_m", float("nan"))],
                      ["等效滑移概率", _fmt_num(k.get("slip_p_equiv"))],
                      ["等效加减速 (m/s²)", _fmt_num(k.get("accel_equiv"))],
                  ], ["参数", "数值"]), ""]
        if k.get("notes"):
            lines += ["注：" + "；".join(str(x) for x in k["notes"]) + "\n"]
    if budget:
        rows = []
        if budget.get("odom_rmse") is not None:
            rows.append(["真机侧：里程计口径（实测速度积分 vs 参考位姿）",
                         "%.3f" % budget["odom_rmse"], "—", "—"])
        rows.append(["基线（无退化的指令回放）",
                     "%.3f" % budget.get("base_rmse", float("nan")), "—", "—"])
        for it in budget.get("items", []):
            rows.append([it["label"], "%.3f" % it["rmse"],
                         "%+.3f" % it.get("explained", 0.0),
                         "%.1f%%" % it.get("pct_of_base", float("nan"))])
        rows.append(["**全开（组合）**", "%.3f" % budget.get("all_rmse", float("nan")),
                     "%+.3f" % (budget.get("base_rmse", 0) - budget.get("all_rmse", 0)),
                     "—"])
        rows.append(["残差（交互+未建模）", "%.3f" % budget.get("residual", float("nan")),
                     "—", "—"])
        ref_name = "地面真值 gt_pose" if budget.get("ref") == "gt" else "真机里程计"
        lines += ["## 5. 误差预算分解（逐项消融，单位：米）\n",
                  "> 参考口径：`%s`（%s）。\n"
                  "> **第一层**：先用「实测速度回放」把**真机侧固有偏差**（里程计自身）扣出来，\n"
                  "> 余下的才是可归因于动力学的部分。\n"
                  "> **第二层**：只开一项退化 → 误差**减少量**即该项「解释量」\n"
                  "> （>0 说明该退化确实存在于真机；<0 说明真机上没有这个现象）。\n"
                  "> 残差 = 全开误差 − 基线误差 + Σ解释量（交互项 + 未建模因素）。\n"
                  "> `里程计漂移`只污染上报位姿、不改变真值轨迹，故**不出现在开环轨迹误差里**（见第 6 节）。\n"
                  % (budget.get("ref", "-"), ref_name),
                  _fmt_table(rows, ["预算项", "位置误差 RMSE", "误差解释量", "占基线比例"]), ""]
        if budget.get("dyn_part") is not None:
            lines += ["可归因于动力学的部分 ≈ %.3f m（= 基线 %.3f − 里程计口径 %.3f）\n"
                      % (budget["dyn_part"], budget.get("base_rmse", float("nan")),
                         budget.get("odom_rmse", float("nan")))]
    if closed_loop:
        rows = [[r["label"], "%.1f%%" % (100 * r["SR"]), "%.3f" % r["SPL"],
                 "%+.1f pp" % (100 * r["dSR"]), "%+.3f" % r["dSPL"]]
                for r in closed_loop.get("rows", [])]
        lines += ["## 6. 闭环口径：各退化项对任务指标的影响\n",
                  "> 策略 `%s`，%d 回合（同一批回合种子，可配对比较）。\n"
                  "> 注意：使用**真值位姿**的特权基线（`geodesic`/`oracle`）不受「里程计漂移」影响，\n"
                  "> 漂移只惩罚**依赖上报位姿的策略**（VLA 属于此类）—— 这正是二者的差别所在。\n"
                  % (closed_loop.get("policy", "-"), closed_loop.get("episodes", 0)),
                  _fmt_table(rows, ["退化项", "SR", "SPL", "ΔSR", "ΔSPL"]), ""]
    if density:
        d = density
        lines += ["## 7. 等效密度换算（论文4 范式）\n",
                  _fmt_table([
                      ["实机部署规模", _fmt_num(d.get("real_n"))],
                      ["实机场地面积 (m²)", "%.2f" % d.get("real_area", float("nan"))],
                      ["实机空间密度 ρ", _fmt_num(d.get("rho_real"))],
                      ["仿真场地面积 (m²)", "%.2f" % d.get("sim_area", float("nan"))],
                      ["**等效规模**（同密度）", _fmt_num(d.get("sim_n_equiv"))],
                  ], ["量", "数值"]), ""]
        if d.get("sim_density"):
            sd = d["sim_density"]
            lines += ["仿真侧场景拥挤度：\n",
                      _fmt_table([
                          ["障碍占地比 ρ_occ", "%.4f" % sd.get("occ_ratio", float("nan"))],
                          ["自由面积占比", "%.4f" % sd.get("free_ratio", float("nan"))],
                          ["语义目标数", sd.get("n_objects", 0)],
                          ["目标密度 (个/m²)", "%.4f" % sd.get("obj_density", float("nan"))],
                      ], ["指标", "数值"]), ""]
        if d.get("error_rate"):
            er = d["error_rate"]
            rel = ("%.1f%%" % (100 * er["rel"])) if er.get("rel") is not None else "—"
            lines += ["与实机指标的**误差率**（论文4 表 29 口径）：\n",
                      _fmt_table([
                          [d.get("metric_name", "SR"),
                           "%.3f" % er["sim"], "%.3f" % er["real"],
                           "%.3f" % er["abs_diff"], rel],
                      ], ["指标", "仿真", "实机", "绝对差", "误差率"]), ""]
    if notes:
        lines += ["## 附注\n"]
        for nline in notes:
            lines.append("- " + str(nline))
        lines.append("")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _fmt_num(x, nd=3):
    """数值格式化（None → 不可用）。"""
    if x is None:
        return "—"
    try:
        return ("%." + str(nd) + "f") % float(x)
    except (TypeError, ValueError):
        return str(x)
