#!/usr/bin/env python3
"""make_mock_recording.py —— 生成**符合真机 schema 的样例录制**（无实机时验证 P5 管线）。

为什么需要它：P5 的整条链路是「真机录制 → 仿真回放 → 域差量化 → 参数标定 →
误差预算 → 等效密度换算」。没有实机数据时，这条链路**无法被验证** ——
只能等真机接上才第一次跑，风险很大（口径错、字段名错、单位错都会在最后一步
才暴露）。本脚本用仿真环境造一段「像真机」的录制，schema 与
`deliver/tools/record_dataset.py` **逐字段对齐**：

    actions (N,2) f32   states (N,5) f32  [x, y, θ, v, ω]
    lidar (N,120) f32   images (N,H,W,3) u8
    ep_offset (E+1,) i64  ep_instr (E,) str  ep_success (E,) f32
    attrs: schema="nav_v1" 等

并且**故意注入真机特有的不完美**（这才是验证的意义）：
  - 速度一阶滞后（电机+惯量）        → `fit_kinematics` 应拟合出非零 τ
  - 位移打折（轮滑）                  → 应拟合出 gain_v < 1
  - 里程计随机游走漂移                → 应拟合出非零漂移率
  - 图像亮度/对比度/噪声偏移          → 域差统计应显示非零 gap
  - 激光噪声 + 少量丢束                → 激光域差应显示非零 gap

用法：
    python3 scripts/real/make_mock_recording.py --out /tmp/mock.h5
    python3 scripts/real/sim2real_report.py --h5 /tmp/mock.h5 --layout apartment \
        --out export/sim2real/mock.md
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from sim.env import SemanticNavEnv                       # noqa: E402
from sim.sensors import photometric_augment               # noqa: E402
from sim.world import segment_free                        # noqa: E402


def _downscale(img, w, h):
    """面积平均降采样（无 cv2 依赖）。"""
    a = np.asarray(img, dtype=np.float64)
    H, W = a.shape[:2]
    ys = np.clip((np.arange(h) + 0.5) * H / h, 0, H - 1).astype(int)
    xs = np.clip((np.arange(w) + 0.5) * W / w, 0, W - 1).astype(int)
    return a[ys][:, xs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/mock_session.h5", help="输出 HDF5 路径")
    ap.add_argument("--layout", default="apartment", help="场景布局")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=120, help="录制步数")
    ap.add_argument("--img-w", type=int, default=320)
    ap.add_argument("--img-h", type=int, default=240)
    ap.add_argument("--tau", type=float, default=0.45, help="注入的一阶滞后时间常数 (s)")
    ap.add_argument("--gain", type=float, default=0.82, help="注入的位移增益（轮滑）")
    ap.add_argument("--drift", type=float, default=0.012, help="注入的里程计漂移率 (m/m)")
    ap.add_argument("--real-area", type=float, default=6.0,
                    help="模拟的「实机场地面积」(m²)，写进 attrs 供报告引用")
    a = ap.parse_args()

    try:
        import h5py
    except ImportError:
        raise SystemExit("需要 h5py：pip3 install h5py")

    rng = np.random.default_rng(a.seed)
    env = SemanticNavEnv(seed=a.seed, layout=a.layout, max_steps=10 ** 6,
                          obs_size=(160, 120), n_peds=0,
                          lidar_noise=0.0, camera_augment=0.0)
    env.reset(seed=a.seed)

    # ---- 真实感动力学：一阶滞后 + 轮滑 ----
    dt = float(env.cfg["dt"])
    v_act, w_act = 0.0, 0.0
    alpha = float(np.clip(dt / max(a.tau, 1e-3), 0.0, 1.0))

    actions, states, lidars, images, gts = [], [], [], [], []
    pose = np.array(env.pose, dtype=np.float64)
    odom = pose.copy()

    def _pick_wp(cur):
        """随机取一个**直线可达**的自由航点（距当前 ≥1.2 m）。

        必须要求「直线可达」：否则机器人会一头撞进墙里（真值不动、
        里程计却还在积分），测出来的「里程计偏差」就被碰撞污染了。
        """
        sc = env.scene
        for _ in range(400):
            p = np.array([rng.uniform(sc.bounds[0] + 0.6, sc.bounds[1] - 0.6),
                          rng.uniform(sc.bounds[2] + 0.6, sc.bounds[3] - 0.6)])
            if not sc.is_free(p, margin=0.30):
                continue
            if float(np.linalg.norm(p - cur[:2])) < 1.2:
                continue
            if not segment_free(sc, cur[:2], p, 0.22):
                continue
            return p
        return np.array(sc.spawn, dtype=np.float64)

    wp = _pick_wp(pose)

    for i in range(a.steps):
        # ---- 航点跟踪控制器：指令近乎恒速（便于拟合一阶滞后），转向用比例控制 ----
        if i % 22 == 0 or float(np.linalg.norm(wp - pose[:2])) < 0.45:
            wp = _pick_wp(pose)
        bearing = float(np.arctan2(wp[1] - pose[1], wp[0] - pose[0]) - pose[2])
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        w_cmd = float(np.clip(1.4 * bearing, -1.2, 1.2))
        v_cmd = float(0.30 * max(0.25, np.cos(min(abs(bearing), np.pi / 2))))
        v_act += alpha * (v_cmd - v_act)
        w_act += alpha * (w_cmd - w_act)

        # ---- 真值运动：轮滑打折（车实际走的距离 < 轮子转出来的距离）----
        th = pose[2]
        nx = pose[0] + v_act * a.gain * np.cos(th) * dt
        ny = pose[1] + v_act * a.gain * np.sin(th) * dt
        nth = (th + w_act * dt + np.pi) % (2 * np.pi) - np.pi
        if env.scene.is_free((nx, ny), margin=0.18):
            pose = np.array([nx, ny, nth])
        else:
            pose = np.array([pose[0], pose[1], nth])
        env.pose = pose.copy()

        # ---- 里程计（编码器口径）：**不打折**，所以会高估位移；再叠加随机游走漂移 ----
        odom[0] += v_act * np.cos(odom[2]) * dt
        odom[1] += v_act * np.sin(odom[2]) * dt
        odom[2] = (odom[2] + w_act * dt + np.pi) % (2 * np.pi) - np.pi
        odom[0] += rng.normal(0.0, a.drift * dt)
        odom[1] += rng.normal(0.0, a.drift * dt)
        odom[2] += rng.normal(0.0, a.drift * 0.4 * dt)

        obs = env._obs()
        # ---- 图像域差：亮度/对比度偏移 + 噪声（真机镜头的味道）----
        img = photometric_augment(obs["image"], rng, strength=0.18)
        img = np.clip(img.astype(np.float32) * 0.92 + 14.0, 0, 255).astype(np.uint8)
        img = _downscale(img, a.img_w, a.img_h).astype(np.uint8)

        # ---- 激光域差：测距噪声 + 少量丢束 ----
        lid = np.asarray(obs["lidar"], dtype=np.float64)
        lid = np.clip(lid + rng.normal(0.0, 0.02, lid.shape), 0.05, 12.0)
        drop = rng.random(lid.shape) < 0.03
        lid[drop] = 12.0

        actions.append([v_cmd, w_cmd])
        states.append([odom[0], odom[1], odom[2], v_act, w_act])
        gts.append([pose[0], pose[1], pose[2]])
        lidars.append(lid.astype(np.float32))
        images.append(img)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with h5py.File(a.out, "w") as f:
        d_img = f.create_dataset("images", shape=(len(images), a.img_h, a.img_w, 3),
                                 maxshape=(None, a.img_h, a.img_w, 3),
                                 dtype=np.uint8, chunks=(1, a.img_h, a.img_w, 3))
        d_act = f.create_dataset("actions", shape=(len(actions), 2),
                                 maxshape=(None, 2), dtype=np.float32)
        d_lds = f.create_dataset("lidar", shape=(len(lidars), 120),
                                 maxshape=(None, 120), dtype=np.float32)
        d_sts = f.create_dataset("states", shape=(len(states), 5),
                                 maxshape=(None, 5), dtype=np.float32)
        d_gt = f.create_dataset("gt_pose", shape=(len(gts), 3),
                                maxshape=(None, 3), dtype=np.float32)
        d_off = f.create_dataset("ep_offset", shape=(2,), maxshape=(None,), dtype=np.int64)
        for i in range(len(actions)):
            d_img[i] = images[i]
            d_act[i] = actions[i]
            d_lds[i] = lidars[i]
            d_sts[i] = states[i]
            d_gt[i] = gts[i]
        d_off[0] = 0
        d_off[1] = len(actions)
        dt2 = h5py.special_dtype(vlen=str)
        f["ep_instr"] = np.asarray(["<mock>"], dtype=dt2)
        f["ep_success"] = np.full(1, -1.0, np.float32)
        f.attrs.update({
            "schema": "nav_v1", "ros_mode": 0,
            "img_w": a.img_w, "img_h": a.img_h,
            "camera_hfov_deg": 64.5, "camera_h_m": 0.35,
            "instruction": "<mock>",
            "note": "**仿真生成的样例录制**（非真机），用于验证 sim2real 管线",
            "mock_injected": "tau=%.3f gain=%.3f drift=%.4f" % (a.tau, a.gain, a.drift),
            "mock_real_area_m2": a.real_area,
            "mock_layout": a.layout,
            "mock_gt_note": "gt_pose = 地面真值（区分轮滑与里程计漂移用）",
        })
    sz = os.path.getsize(a.out) / 2 ** 20
    print(f"已生成样例录制 {a.out}：{len(actions)} 帧 / {sz:.1f} MB")
    print("注入的不完美：tau=%.3f s, gain=%.3f, drift=%.4f m/m"
          % (a.tau, a.gain, a.drift))


if __name__ == "__main__":
    main()
