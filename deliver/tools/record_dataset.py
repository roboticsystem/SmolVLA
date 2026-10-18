#!/usr/bin/env python3
"""record_dataset.py —— 遥操采数：录制 (图像, 激光, 里程计, 遥操动作) → HDF5。

落盘 schema 与训练管线 `export/nav_*.h5` **逐字段对齐**：

    actions (N,2) f32   images (N,H,W,3) u8   lidar (N,120) f32
    states  (N,5) f32   # [x, y, θ, v, ω]
    ep_offset (E+1,) i64    ep_instr (E,) str    ep_success (E,) f32（默认 -1，回填）
    meta: 采集参数、相机内参、归一化统计的来源（不重算！）

**为什么 schema 要对齐**：回工作站后可以直接进 `smolvla_finetune_sim.py` 的
数据类（只差 det_feat 一列——需要用标定后的 HSV refs 重算，见 verify 工具提示）。

两种模式：
    --ros     订阅 /camera/image_raw /scan /odom，发布 /cmd_out（推荐，需要 rclpy）
    （无 ROS） 仅相机 + 指令死推航位，激光全 12.0——只用于走通流程，不可训练

操作（cv2 窗口按键）：
    w/s 前进/后退   a/d 左转/右转   空格 急停
    n  结束当前回合（ep 边界）      q  保存并退出
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("需要 opencv：pip3 install opencv-python")

try:
    import h5py
except ImportError:
    raise SystemExit("需要 h5py：pip3 install h5py")

V_MAX, W_MAX = 0.35, 1.60        # 与训练动作分布一致，勿放大


def resample_lidar(r, n=120, max_range=12.0):
    """任意束数的 LaserScan → 120 维「车头为 0 逆时针」口径（与 vla_bridge 相同）。"""
    r = np.asarray(r, np.float32)
    if r.size == n:
        return np.where(np.isfinite(r), r, max_range)
    idx = np.linspace(0, r.size - 1, n)
    out = np.interp(idx, np.arange(r.size), np.where(np.isfinite(r), r, max_range))
    # 有限束插值会把「超出量程」拉低，这里恢复 out-of-range 常量
    out = np.where(out > max_range, max_range, out)
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ros", action="store_true", help="ROS2 模式（默认关）")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default="data_recorded/session_$(date +%m%d_%H%M).h5")
    ap.add_argument("--img-size", default="640x480", help="落盘分辨率 WxH（原图直存）")
    ap.add_argument("--max-episodes", type=int, default=50)
    ap.add_argument("--instruction", default="",
                    help="本批次所有回合的语言指令（写入 ep_instr；不填则 <manual>，微调前需回填）")
    ap.add_argument("--config", default="../config/robot.yaml",
                    help="robot.yaml 路径：自动读取已标定的相机内参写入 h5 attrs")
    ap.add_argument("--hfov", type=float, default=None, help="覆盖 config 中的 hfov_deg")
    ap.add_argument("--cam-h", type=float, default=None, help="覆盖 config 中的 cam_h")
    a = ap.parse_args()
    a.out = a.out.replace("$(date +%m%d_%H%M)", time.strftime("%m%d_%H%M"))

    # 相机内参：优先命令行 > robot.yaml > -1（未标定）
    hfov, cam_h = -1.0, -1.0
    if os.path.exists(a.config):
        try:
            import yaml
            cfg = yaml.safe_load(open(a.config)) or {}
            cam = cfg.get("camera", {})
            hfov = float(cam.get("hfov_deg", -1.0))
            cam_h = float(cam.get("cam_h", -1.0))
            print(f"[config] 从 {a.config} 读到内参：hfov={hfov} cam_h={cam_h}")
        except ImportError:
            print(f"[WARN] 未装 pyyaml（pip3 install pyyaml），内参不自动回填")
        except Exception as e:                                # noqa: BLE001
            print(f"[WARN] 读 config 失败（{e}），内参不自动回填")
    else:
        print(f"[WARN] 找不到 {a.config}，内参不自动回填")
    if a.hfov is not None:  hfov = a.hfov
    if a.cam_h is not None: cam_h = a.cam_h

    use_ros = False
    node = None
    if a.ros:
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import Image, LaserScan
            from nav_msgs.msg import Odometry

            class Sink(Node):
                def __init__(self):
                    super().__init__("vla_recorder")
                    self.img = self.scan = self.odom = None
                    self.create_subscription(Image, "/camera/image_raw", self._i, 10)
                    self.create_subscription(LaserScan, "/scan", self._s, 10)
                    self.create_subscription(Odometry, "/odom", self._o, 10)

                def _i(self, m):
                    self.img = np.frombuffer(m.data, np.uint8).reshape(
                        m.height, m.width, -1)[:, :, :3].copy()

                def _s(self, m):
                    ang = np.array([m.angle_min + i * m.angle_increment
                                    for i in range(len(m.ranges))])
                    order = np.argsort(((ang + np.pi) % (2 * np.pi)) - np.pi)
                    self.scan = np.asarray(m.ranges, np.float32)[order]

                def _o(self, m):
                    p, q = m.pose.pose.position, m.pose.pose.orientation
                    yaw = float(np.arctan2(2 * (q.w * q.z + q.x * q.y),
                                           1 - 2 * (q.y**2 + q.z**2)))
                    self.odom = np.array([p.x, p.y, yaw,
                                          m.twist.twist.linear.x,
                                          m.twist.twist.angular.z], np.float32)
            rclpy.init()
            node = Sink()
            use_ros = True
            print("[ROS] 订阅 /camera/image_raw /scan /odom")
        except Exception as e:                                # noqa: BLE001
            print(f"[WARN] rclpy 不可用（{e}）→ 退化为相机-only 模式（激光=12，航位死推）")

    W, H = map(int, a.img_size.split("x"))
    cap = cv2.VideoCapture(a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    imgs, acts, lds, sts, offs, instrs = [], [], [], [], [0], []
    ep, v_cmd, w_cmd = 0, 0.0, 0.0
    pose = np.zeros(3, np.float32)                # 死推航位（x,y,θ）
    dt = 1.0 / 10.0
    print("w/s 前后  a/d 转向  空格停  n 结束本回合  q 保存退出")
    t_last = time.time()
    while ep < a.max_episodes:
        ok, frame = cap.read()
        if not ok:
            print("抓帧失败，退出"); break
        if use_ros and node.odom is not None:
            pose = node.odom[:3]
            st = node.odom.copy()
        else:
            # 航位死推：仅无 ROS 模式（误差大，只用于流程验证）
            pose[2] += w_cmd * dt
            pose[0] += v_cmd * np.cos(pose[2]) * dt
            pose[1] += v_cmd * np.sin(pose[2]) * dt
            st = np.array([*pose, v_cmd, w_cmd], np.float32)
        scan = node.scan if (use_ros and node.scan is not None) else np.full(120, 12.0)
        key = cv2.waitKey(1) & 0xFF
        v_new, w_new = v_cmd, w_cmd
        if key == ord("w"):  v_new = V_MAX
        elif key == ord("s"): v_new = -V_MAX * 0.3
        elif key == ord("a"): w_new = W_MAX
        elif key == ord("d"): w_new = -W_MAX
        elif key == ord(" "): v_new = w_new = 0.0
        elif key == ord("n"):
            if len(imgs) > offs[-1] + 5:
                offs.append(len(imgs)); ep += 1
                print(f"  回合 {ep} 已记录（{offs[-1]-offs[-2]} 帧）")
            v_new = w_new = 0.0
        elif key == ord("q"):
            if len(imgs) > offs[-1] + 5:
                offs.append(len(imgs))
            break
        v_cmd, w_cmd = v_new, w_new
        imgs.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        acts.append([v_cmd, w_cmd])
        lds.append(resample_lidar(scan))
        sts.append(st)
        t_last = time.time()
        view = frame.copy()
        cv2.putText(view, f"ep={ep} frames={len(imgs)} v={v_cmd:+.2f} w={w_cmd:+.2f}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("record", view)
        if use_ros:
            import rclpy
            rclpy.spin_once(node, timeout_sec=0)
            # 遥操指令发到底盘（有 ROS 时）
            try:
                from geometry_msgs.msg import Twist
                if not hasattr(node, "pub"):
                    node.pub = node.create_publisher(Twist, "/cmd_out", 10)
                m = Twist(); m.linear.x = float(v_cmd); m.angular.z = float(w_cmd)
                node.pub.publish(m)
            except Exception:
                pass

    cap.release(); cv2.destroyAllWindows()
    n = len(imgs); e = len(offs) - 1
    if n < 10:
        print("样本太少，未保存"); return
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with h5py.File(a.out, "w") as f:
        f["actions"] = np.asarray(acts, np.float32)
        f["images"] = np.stack(imgs).astype(np.uint8)
        f["lidar"] = np.asarray(lds, np.float32)
        f["states"] = np.asarray(sts, np.float32)
        f["ep_offset"] = np.asarray(offs, np.int64)
        dt2 = h5py.special_dtype(vlen=str)
        instr_final = a.instruction if a.instruction else "<manual>"
        f["ep_instr"] = np.asarray(instrs if instrs else [instr_final] * e, dtype=dt2)
        f["ep_success"] = np.full(e, -1.0, np.float32)   # 遥操后回填 1/0
        f.attrs.update({
            "schema": "nav_v1", "ros_mode": int(use_ros),
            "img_w": W, "img_h": H,
            "camera_hfov_deg": hfov,                     # 自动来自 robot.yaml / 命令行
            "camera_h_m": cam_h,
            "instruction": instr_final,
            "note": "动作=遥操指令（无 ROS 模式含航位死推误差）；det_feat 需回工作站重算",
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    sz = os.path.getsize(a.out) / 2**20
    print(f"\n已保存 {a.out}: {e} 回合 / {n} 帧 / {sz:.1f} MB")
    print(f"指令: {instr_final} | 内参: hfov={hfov} cam_h={cam_h}")
    print(f"下一步：python3 tools/verify_dataset.py --h5 {a.out}")


if __name__ == "__main__":
    main()
