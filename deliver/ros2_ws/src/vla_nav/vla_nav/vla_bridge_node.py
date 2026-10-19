#!/usr/bin/env python3
"""vla_bridge_node —— 端侧采集 + 局域网调用服务器侧 VLA（云边分层 D2）。

## 职责

订阅 `/camera/image_raw`、`/scan`、`/odom`，组装成与**仿真训练完全一致**的
观测格式，HTTP POST 到服务器侧的推理服务，拿回 2 维动作 (v, ω) 发布到 `/cmd_out`。

## 与训练观测的一致性（最容易错的地方）

训练时观测是：
    state = [x, y, θ, v_prev, ω_prev, (感知特征 4/8 维)]   → pad 到 32
    image = resize_with_pad 到 512×512，归一化 [-1, 1]
    language = 指令文本

本节点必须**逐项对齐**：
1. 状态顺序与维度（pose 3 + last_action 2 = 5，pad 到 32）
2. 图像 resize 用 `resize_with_pad`（不是普通 resize —— 长宽比不同会错位）
3. 归一化统计用训练数据集的 `st_mean/st_std`（**不能现场重算**）
4. 指令文本与训练时的措辞分布一致（L1/L2/L3 三级）

## 断网降级（E9 指标，真机必须实现）

连续 N 次请求失败 → 进入 `degraded` 状态：
- 停止前进（v=0），原地慢速旋转搜索目标（与仿真里 reactive 丢失时的行为一致）
- 网络恢复后自动回到正常模式，并记录降级时长（进 E9 的「断网可用性」指标）

> **本文件是骨架**：HTTP 客户端与降级逻辑已写，但**未经真机联调**。
"""
from __future__ import annotations

import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

STATE_DIM = 32          # 与训练一致（max_state_dim）
SERVER = "http://10.252.18.70:8000/infer"   # 服务器侧推理地址（热点网段入口；若 NAT 只转发 22，见 README_Spark 隧道方案）
TIMEOUT_S = 1.0         # 实测服务端 ~0.5 s/请求（4090），0.35 会全部超时进 degraded


class VLABridge(Node):
    def __init__(self):
        super().__init__("vla_bridge")
        self.declare_parameter("server", SERVER)
        self.declare_parameter("instruction", "去red的箱子旁边")
        self.declare_parameter("st_mean", [0.0] * 5)
        self.declare_parameter("st_std", [1.0] * 5)
        # 归一化开关：infer_server 端已按 norm_stats.json 归一化（单一口径来源），
        # 桥接端**默认关闭**，避免双重归一化（st 会被拉到 [-2,2] 之外，动作全歪）。
        # 仅当服务端是旧版（不做归一化）时才置 true 并配 st_mean/st_std。
        self.declare_parameter("normalize_state", False)
        self.server = self.get_parameter("server").value
        self.instr = self.get_parameter("instruction").value
        self.st_mean = np.array(self.get_parameter("st_mean").value, dtype=np.float32)
        self.st_std = np.array(self.get_parameter("st_std").value, dtype=np.float32)
        self.normalize_state = bool(self.get_parameter("normalize_state").value)
        # 语音换目标：voice_node 发布 /vla/instruction 即时更新指令
        # （std_msgs/String；只有 ok=True 的归一化指令才会被发布）
        from std_msgs.msg import String as _Str
        self.create_subscription(_Str, "/vla/instruction", self._voice_instr, 10)
        # chunk 批量执行（服务端每次推理返回前 K 步动作）：队列非空时 10Hz 逐步执行，
        # 取空才发起新请求——网络/推理压力降 K 倍，闭环期间请求频率 ~1Hz。
        self.declare_parameter("chunk_execute", True)
        self.chunk_execute = bool(self.get_parameter("chunk_execute").value)
        self.action_queue = []

    def _voice_instr(self, msg):
        if msg.data and msg.data != self.instr:
            self.instr = msg.data
            # 换目标后清空旧 chunk，避免用旧目标的动作继续走
            self.action_queue = []
            self.get_logger().info(f"[voice] 指令切换 → {self.instr!r}")

        self.last_img = None
        self.last_scan = None
        self.last_odom = None
        self.last_action = np.zeros(2, dtype=np.float32)
        self.fail_streak = 0
        self.degraded_since = None

        self.create_subscription(Image, "/camera/image_raw", self._img, 10)
        self.create_subscription(LaserScan, "/scan", self._scan, 10)
        self.create_subscription(Odometry, "/odom", self._odom, 10)
        self.pub = self.create_publisher(Twist, "/cmd_out", 10)
        self.create_timer(0.1, self.tick)          # 10 Hz 控制回路
        self.get_logger().info(f"vla_bridge 启动，服务器 {self.server}")

    # ---- 传感器回调：只缓存最新一帧 ----
    def _img(self, msg: Image):
        # 保留原始分辨率：检测口径（min_px=150 等）按 480x640 标定，
        # 缩小会让远处目标掉出阈值（实测 120x160 检出率 66.5%→31%）。
        # 传输体积由 JPEG 压缩解决（见 tick 的 payload 组装）。
        self.last_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1)[:, :, :3]

    def _scan(self, msg: LaserScan):
        n = len(msg.ranges)
        # **重排成「以车头为 0、逆时针」的 360° 约定**（与仿真 lidar_scan 一致）
        a = np.array([msg.angle_min + i * msg.angle_increment for i in range(n)])
        order = np.argsort(((a + np.pi) % (2 * np.pi)) - np.pi)
        r = np.asarray(msg.ranges, dtype=np.float32)[order]
        self.last_scan = np.where(np.isfinite(r), r, 12.0)

    def _odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = float(np.arctan2(2 * (q.w * q.z + q.x * q.y),
                               1 - 2 * (q.y ** 2 + q.z ** 2)))
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        self.last_odom = np.array([p.x, p.y, yaw, v, w], dtype=np.float32)

    # ---- 组装观测（与训练逐项对齐）----
    def build_state(self):
        o = self.last_odom
        if o is None:
            return None
        st = np.concatenate([o[:3], self.last_action]).astype(np.float32)
        if self.normalize_state:
            st = (st - self.st_mean) / self.st_std
        if st.size < STATE_DIM:
            st = np.pad(st, (0, STATE_DIM - st.size))
        return st[:STATE_DIM]

    def tick(self):
        if self.last_img is None or self.last_scan is None or self.last_odom is None:
            return
        cmd = Twist()
        # chunk 批量执行：队列里还有上一次推理的动作就直接执行，不发请求
        if self.chunk_execute and self.action_queue:
            a = self.action_queue.pop(0)
            cmd.linear.x, cmd.angular.z = float(a[0]), float(a[1])
            self.pub.publish(cmd)
            return
        try:
            import urllib.request
            # 图像传输：优先 JPEG 全尺寸（q=80，实测 ~37KB 且检出率无损）；
            # 无 cv2 时退回原始 hex（1.8MB，慢但能跑）
            import base64
            try:
                import cv2
                ok, buf = cv2.imencode(".jpg", self.last_img,
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                img_field = {"jpg_b64": base64.b64encode(buf.tobytes()).decode()}
            except ImportError:
                img_field = {"image_bgr": self.last_img.tobytes().hex(),
                             "h": int(self.last_img.shape[0]),
                             "w": int(self.last_img.shape[1])}
            payload = json.dumps({
                **img_field,
                "scan": self.last_scan.tolist(),
                "state": self.build_state().tolist(),
                "instruction": self.instr,
            }).encode()
            req = urllib.request.Request(self.server, data=payload,
                                         headers={"Content-Type": "application/json"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                out = json.loads(r.read())
            self.lat_ms = (time.time() - t0) * 1000
            if self.chunk_execute and out.get("actions"):
                acts = out["actions"]
                a = np.clip(np.asarray(acts[0], dtype=np.float32),
                            [-0.35, -1.6], [0.35, 1.6])
                self.action_queue = [
                    np.clip(np.asarray(x, dtype=np.float32), [-0.35, -1.6],
                            [0.35, 1.6]) for x in acts[1:]]
            else:
                self.action_queue = []
                a = np.clip(np.asarray(out["action"], dtype=np.float32),
                            [-0.35, -1.6], [0.35, 1.6])
            self.last_action = a
            cmd.linear.x, cmd.angular.z = float(a[0]), float(a[1])
            if self.fail_streak:
                self.get_logger().info(f"网络恢复（曾连续失败 {self.fail_streak} 次）")
            self.fail_streak = 0
            self.degraded_since = None
        except Exception as e:
            self.fail_streak += 1
            if self.fail_streak == 1:
                self.degraded_since = time.time()
                self.get_logger().warning(f"进入降级模式: {type(e).__name__}")
            # **降级**：停止前进、原地慢速扫描（与仿真 reactive 丢失行为一致）
            cmd.linear.x, cmd.angular.z = 0.0, 0.9
        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    n = VLABridge()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
