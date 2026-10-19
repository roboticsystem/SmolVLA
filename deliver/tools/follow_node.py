#!/usr/bin/env python3
"""follow_node.py —— E11 视觉跟随（跟拍）真机节点（RDK 上运行）。

任务定义（与仿真 sim/follow.py 同口径）：
    被跟随者穿/持场景中唯一的红色目标（红马甲/红盒），机器人保持
    ① 目标在视野内 ② 距离在带 [band_lo, band_hi] 内（默认 0.9–2.6 m，want 1.5 m）。
控制：规则式 P——方位误差→ω、纵向距离误差→v；近距护栏与部署契约一致：
    有效距离 = min(车头±18° 激光, 单目 det_dist)；<0.30 m 冻结、<0.40 m 限速 0.08。
跟拍：--record 时把带标注（方位/距离/v/ω）的画面录成 mp4（论文素材）。
指标：每帧落 CSV（found/bearing_deg/dist_m/v/w）——直接出纵向/横向误差表。

用法（车端；与 vla_bridge 二选一运行，两者都发 /cmd_out）：
    python3 follow_node.py --record --out-dir /tmp/follow_demo
"""
import argparse
import csv
import math
import os
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist


def load_cfg(path):
    import yaml
    cfg = yaml.safe_load(open(path)) or {}
    det = cfg.get("detection", {})
    refs = {k: tuple(v) for k, v in det.get("refs", {}).items()}
    tol = tuple(det.get("hsv_tol", [12, 90, 90]))
    overrides = {k: tuple(v) for k, v in det.get("hsv_tol_overrides", {}).items()}
    cam = cfg.get("camera", {})
    hfov = float(cam.get("hfov_deg", 69.4))
    cam_h = float(cam.get("cam_h", 0.08))
    measured = [k for k, v in refs.items() if "实测" in str(v)]
    use_refs = {k: v for k, v in refs.items() if k in measured or "占位" not in str(refs.get(k))}
    return refs, tol, overrides, hfov, cam_h, use_refs


def detect_target(img_bgr, color, refs, tol, overrides, hfov_deg, cam_h, min_px=150):
    """与 backfill_real 逐行同口径；返回 (bearing_deg, d_mono_m, r1, n_px) 或 None。"""
    h, w = img_bgr.shape[:2]
    f = (w / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    horizon = h / 2.0
    if color not in refs:
        return None
    ref = refs[color]
    t = overrides.get(color, tol)
    lo = np.array([max(0, ref[0] - t[0]), max(0, ref[1] - t[1]), max(0, ref[2] - t[2])], np.uint8)
    hi = np.array([min(179, ref[0] + t[0]), min(255, ref[1] + t[1]), min(255, ref[2] + t[2])], np.uint8)
    mask = cv2.inRange(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask)
    best = None
    for i in range(1, n):
        npx = int(stats[i, cv2.CC_STAT_AREA])
        if npx < min_px:
            continue
        x, y, bw, bh = stats[i, :4]
        r1 = int(y + bh)
        d = f * cam_h / max(1.0, r1 - horizon) if r1 > horizon else float("nan")
        d = float(np.clip(d, 0.15, 12.0)) if np.isfinite(d) else float("nan")
        ang = math.degrees(math.atan2(cent[i][0] - (w - 1) / 2.0, f))
        if best is None or npx > best[3]:
            best = (ang, d, r1, npx)
    return best


class Follow(Node):
    def __init__(self, a):
        super().__init__("follow_node")
        self.a = a
        self.refs, self.tol, self.ovr, self.hfov, self.cam_h, self.use_refs = \
            load_cfg(a.config)
        if a.color not in self.use_refs:
            raise SystemExit(f"颜色 {a.color!r} 不在 robot.yaml 实测参考色里")
        self.last_img = None
        self.last_scan = None
        self.lost = 0
        self.last_bearing = 0.0
        self.create_subscription(Image, "/camera/image_raw", self._img, 10)
        self.create_subscription(LaserScan, "/scan", self._scan, 10)
        self.pub = self.create_publisher(Twist, a.topic, 10)
        self.create_timer(0.1, self.tick)          # 10 Hz 控制回路
        os.makedirs(a.out_dir, exist_ok=True)
        self.csv_file = open(os.path.join(a.out_dir, "follow_metrics.csv"), "w", newline="")
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(["t", "found", "bearing_deg", "dist_m", "v", "w"])
        self.writer = None
        if a.record:
            self.writer = cv2.VideoWriter(
                os.path.join(a.out_dir, "follow_demo.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), 10, (a.width, a.height))
        self.t0 = time.time()
        self.get_logger().info(
            f"跟随节点就绪：目标色={a.color} want={a.want_dist}m "
            f"带[{a.band_lo},{a.band_hi}]m 场地话题={a.topic} 录像={a.record}")

    def _img(self, msg):
        self.last_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1)[:, :, :3]

    def _scan(self, msg):
        n = len(msg.ranges)
        ang = np.array([msg.angle_min + i * msg.angle_increment for i in range(n)])
        order = np.argsort(((ang + np.pi) % (2 * np.pi)) - np.pi)
        r = np.asarray(msg.ranges, dtype=np.float32)[order]
        self.last_scan = np.where(np.isfinite(r), r, 12.0)

    def _lidar_front_min(self, scan, bearing_deg):
        """车头 ±18° 内最小激光（beam 按逆时针 0 起约定，同 bridge/回填映射）。"""
        step = 360.0 / max(1, len(scan))
        i0 = int(round((-bearing_deg) % 360.0 / step)) % len(scan)
        seg = scan[max(0, i0 - 6):min(len(scan), i0 + 7)]
        seg = seg[np.isfinite(seg) & (seg < 12.0)]
        return float(seg.min()) if seg.size else float("nan")

    def tick(self):
        if self.last_img is None or self.last_scan is None:
            return
        a = self.a
        img = self.last_img
        h, w = img.shape[:2]
        res = detect_target(img, a.color, self.refs, self.tol, self.ovr,
                            self.hfov, self.cam_h)
        cmd = Twist()
        found = res is not None
        v = w = 0.0
        dist = bearing = float("nan")
        if found:
            bearing, d_mono, r1, _ = res
            self.lost = 0
            self.last_bearing = bearing
            # [SAT] 目标底沿贴住画面下缘：单目低估 → 激光兜底（与 infer_server det_dist 同规则）
            if r1 >= h - 2:
                d = self._lidar_front_min(self.last_scan, bearing)
            else:
                d = d_mono
            lidar_min = self._lidar_front_min(self.last_scan, bearing)
            dist = min(lidar_min, d) if np.isfinite(lidar_min) else d
            # 近距护栏（与部署契约一致）
            if dist < 0.30:
                v = 0.0                                   # 冻结
            else:
                v = a.k_v * (dist - a.want_dist)          # 纵向 P
                if dist < 0.40:
                    v = min(v, 0.08)
                if a.want_dist - a.band_dead <= dist <= a.want_dist + a.band_dead:
                    v = 0.0                               # 距离带死区
                v = float(np.clip(v, -0.10, a.v_max))
            w = float(np.clip(-a.k_w * math.radians(bearing),
                              -a.w_max, a.w_max))         # 横向 P（右偏→右转）
        else:
            self.lost += 1
            bearing = self.last_bearing                   # 朝最后可见方向慢扫找回
            if self.lost < 30:                            # 丢目标 3s 内：原地慢扫
                w = 0.5 if self.last_bearing >= 0 else -0.5
            # 超过 3s：停车待命（v=w=0）
        cmd.linear.x, cmd.angular.z = v, w
        self.pub.publish(cmd)
        t = time.time() - self.t0
        self.csv.writerow([round(t, 2), int(found),
                           round(bearing, 2) if found else "",
                           round(dist, 3) if found else "", round(v, 3), round(w, 3)])
        if self.writer is not None:
            vis = img.copy()
            if found:
                cv2.putText(vis, f"d={dist:.2f}m bear={bearing:.1f}deg v={v:.2f} w={w:.2f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(vis, "TARGET LOST", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            self.writer.write(vis)


def bearing_last_dir(self):
    """丢失后保持最后的转向趋势（简化：固定向右扫）。"""
    return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../config/robot.yaml")
    ap.add_argument("--color", default="red")
    ap.add_argument("--topic", default="/cmd_out",
                    help="控制话题（与 vla_bridge 二选一运行）")
    ap.add_argument("--want-dist", type=float, default=1.5)
    ap.add_argument("--band-lo", type=float, default=0.9)
    ap.add_argument("--band-hi", type=float, default=2.6)
    ap.add_argument("--band-dead", type=float, default=0.15, help="距离带死区半宽")
    ap.add_argument("--k-w", type=float, default=1.2)
    ap.add_argument("--k-v", type=float, default=0.6)
    ap.add_argument("--v-max", type=float, default=0.20)
    ap.add_argument("--w-max", type=float, default=1.2)
    ap.add_argument("--invert-w", action="store_true", help="转向反向（现场标定用）")
    ap.add_argument("--record", action="store_true", help="跟拍录像 + 指标 CSV")
    ap.add_argument("--out-dir", default="follow_demo")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    a = ap.parse_args()
    if a.invert_w:
        a.k_w = -a.k_w

    rclpy.init()
    node = Follow(a)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.csv_file.flush()
        node.csv_file.close()
        if node.writer is not None:
            node.writer.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
