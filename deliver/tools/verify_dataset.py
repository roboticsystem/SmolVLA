#!/usr/bin/env python3
"""verify_dataset.py —— 采集数据的质量门（可训练性检查）。

用法：
    python3 tools/verify_dataset.py --h5 data_recorded/session_xxx.h5

检查项与判据：
  ① 结构完整性：字段齐全、ep_offset 单调、回合帧数 ≥ 5
  ② 动作：范围 |v| ≤ 0.35 / |ω| ≤ 1.6、非常数（全 0 或全定速 = 遥操没生效）
  ③ 激光：有限值比例（全 12.0 = 无 ROS 模式的占位，不可用于训练）
  ④ 图像：亮度分布（过暗/过曝）、帧间差异（全同帧 = 相机缓存没刷新）
  ⑤ 状态：航位模式会打警告（误差大，仅流程验证）

退出码：0 = 可训练；1 = 有硬伤（不可训练）。
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("pip3 install h5py")

OK, BAD, WARN = "✓", "✗", "!"
hard_fail = []


def sec(t):
    print(f"\n--- {t} ---")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    a = ap.parse_args()
    f = h5py.File(a.h5, "r")

    sec("① 结构")
    need = ["actions", "images", "lidar", "states", "ep_offset"]
    missing = [k for k in need if k not in f]
    if missing:
        print(f"  {BAD} 缺字段 {missing}"); sys.exit(1)
    offs = f["ep_offset"][()]
    lens = np.diff(offs)
    N, E = f["actions"].shape[0], len(lens)
    print(f"  {OK} {E} 回合 / {N} 帧 / {f['images'].shape[1]}x{f['images'].shape[2]} 图像")
    short = int((lens < 5).sum())
    print(f"  {OK if short == 0 else WARN} 回合帧数 最短 {lens.min()}，<5 帧的回合 {short} 个")
    monotonic = bool((np.diff(offs) > 0).all() and offs[0] == 0)
    if not monotonic:
        hard_fail.append("ep_offset 非法")

    sec("② 动作")
    A = f["actions"][()]
    v, w = A[:, 0], A[:, 1]
    oob = int(((np.abs(v) > 0.35 + 1e-6) | (np.abs(w) > 1.6 + 1e-6)).sum())
    print(f"  {OK if oob == 0 else BAD} 超限帧 {oob}（|v|≤0.35, |ω|≤1.6）")
    if oob:
        hard_fail.append("动作超限——遥操指令没经过限幅")
    moving = float((np.abs(v) > 0.05).mean())
    print(f"  {OK if 0.3 < moving < 0.95 else WARN} 非零速度帧占比 {moving*100:.0f}%"
          f"（全 0 = 没采到有效操作；全 1 = 一直在动，回合边界可能漏按 n）")
    print(f"  v: 中位 {np.median(v):+.3f}  ω: 中位 {np.median(w):+.3f} rad/s")

    sec("③ 激光")
    L = f["lidar"][()]
    finite = float(np.isfinite(L).mean())
    all_max = bool((np.abs(L - 12.0) < 1e-3).mean() > 0.99)
    if all_max:
        print(f"  {BAD} 激光全为 12.0（无 ROS 占位值）→ **不可用于训练**，需 --ros 模式重采")
        hard_fail.append("激光占位")
    else:
        print(f"  {OK} 有限值 {finite*100:.1f}%，中位 {np.median(L):.2f} m，"
              f"最小 {np.nanmin(L):.2f} m")
        print(f"  {WARN if np.nanmin(L) < 0.2 else OK} "
              f"最小距离 < 0.2 m 的帧占比 {float((L < 0.2).mean())*100:.2f}%"
              f"（偏高说明太贴墙/贴障碍，模仿这些帧会学到危险行为）")

    sec("④ 图像")
    I = f["images"]
    idx = np.linspace(0, N - 1, min(N, 300)).astype(int)
    bright = I[idx].mean(axis=(1, 2, 3))
    print(f"  {OK} 平均亮度 {bright.mean():.0f}"
          + (f"  {WARN} 偏暗" if bright.mean() < 40 else "")
          + (f"  {WARN} 过曝风险" if bright.mean() > 200 else ""))
    if N > 30:
        dup = float(np.mean([np.array_equal(I[i], I[i + 1])
                             for i in idx[:-1] if i + 1 < N]))
        print(f"  {BAD if dup > 0.5 else OK} 相邻重复帧比例 {dup*100:.0f}%"
              + ("  → 相机画面没刷新（缓冲问题），数据不可用" if dup > 0.5 else ""))
        if dup > 0.5:
            hard_fail.append("重复帧")

    sec("⑤ 状态来源")
    ros_mode = bool(f.attrs.get("ros_mode", 0))
    if not ros_mode:
        print(f"  {WARN} 无 ROS 模式：states 来自航位死推，误差随时间累积，"
              f"仅用于流程验证，不可训练")
        hard_fail.append("航位死推")
    else:
        print(f"  {OK} ROS 里程计")
    print(f"  内参回填检查：camera_hfov_deg={f.attrs.get('camera_hfov_deg')} "
          f"camera_h_m={f.attrs.get('camera_h_m')}"
          f"  {WARN}（-1 = 未回填，det_feat 重算前必须补）")

    print("\n=== 结论 ===")
    if hard_fail:
        print(f"  {BAD} 不可训练：{'；'.join(hard_fail)}")
        print("  修复后重采。把本输出与 selfcheck 报告一起回报。")
        sys.exit(1)
    print(f"  {OK} 通过质量门。回工作站后：")
    print("    1. 回填 ep_success（哪几个回合算成功）")
    print("    2. 用标定后的 HSV refs 重算 det_feat（with_det 的模型需要）")
    print("    3. 才能进微调管线；详见 README「采数回流」一节")


if __name__ == "__main__":
    main()
