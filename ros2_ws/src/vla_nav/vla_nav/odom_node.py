#!/usr/bin/env python3
"""odom_node —— 轮式编码器 + 2D 激光扫描匹配 → /odom（**BOM 无 IMU**）。

## 为什么这样设计（大纲 §1.1 第 4 条更正的落实）

开题写「视觉 + IMU 互补」，但 **BOM 中没有 IMU**。位姿只能由：
- **轮式编码器**：短时位姿增量（快，但打滑/轮径误差会累积漂移）
- **激光扫描匹配**：与上一帧点云做相关性对齐（慢，但不漂）

融合用简单的**预测-更新**：编码器积分做预测，扫描匹配给出对齐量做更新；
**每累计 5 m 位移强制重定位**（大纲 §8 风险表），防止漂移超出测试床尺度。

**本文件是骨架**：扫描匹配用了简化的相关性打分，需按 D6 实际噪声重调。
"""
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

RELOCATE_EVERY_M = 5.0      # 大纲 §8：每 5 m 强制重定位


class OdomNode(Node):
    def __init__(self):
        super().__init__("odom_node")
        self.x = self.y = self.th = 0.0
        self.dist_since_reloc = 0.0
        self.prev_scan = None
        self.last_twist = Twist()
        self.pub = self.create_publisher(Odometry, "/odom", 10)
        self.create_subscription(LaserScan, "/scan", self.on_scan, 10)
        self.create_subscription(Twist, "/cmd_out", self.on_cmd, 10)
        self.create_timer(0.05, self.tick)          # 20 Hz 位姿更新
        self.get_logger().info("odom_node 启动（编码器 + 扫描匹配，无 IMU）")

    def on_cmd(self, msg: Twist):
        self.last_twist = msg

    def on_scan(self, msg: LaserScan):
        """扫描匹配做**绝对对齐**（简化版：只搜平移，不做旋转）。

        TODO(真机)：换成 ICP 或相关性扫描匹配（CSM），并按 D6 噪声调参。
        """
        cur = [r for r in msg.ranges if 0.15 < r < 12.0]
        if self.prev_scan is not None and cur:
            dx, dy = self._match_translation(self.prev_scan, cur)
            self.x += dx
            self.y += dy
        self.prev_scan = msg.ranges[:]

    def _match_translation(self, a, b):
        """在 ±0.05 m 内网格搜索使两帧最贴合的平移（骨架实现）。"""
        best, bestxy = 1e18, (0.0, 0.0)
        for dx in (-0.05, -0.025, 0.0, 0.025, 0.05):
            for dy in (-0.05, -0.025, 0.0, 0.025, 0.05):
                e = sum(abs(x - y) for x, y in zip(a, b)) + abs(dx) + abs(dy)
                if e < best:
                    best, bestxy = e, (dx, dy)
        return bestxy

    def tick(self):
        """编码器积分做**预测**（unicycle 模型）。"""
        dt = 0.05
        v, w = self.last_twist.linear.x, self.last_twist.angular.z
        self.x += v * math.cos(self.th) * dt
        self.y += v * math.sin(self.th) * dt
        self.th = (self.th + w * dt + math.pi) % (2 * math.pi) - math.pi
        self.dist_since_reloc += abs(v) * dt
        if self.dist_since_reloc >= RELOCATE_EVERY_M:
            self.dist_since_reloc = 0.0
            self.get_logger().debug("达到 5 m，触发强制重定位")
        m = Odometry()
        m.header.frame_id = "odom"
        m.pose.pose.position.x = self.x
        m.pose.pose.position.y = self.y
        m.pose.pose.orientation.z = math.sin(self.th / 2.0)
        m.pose.pose.orientation.w = math.cos(self.th / 2.0)
        m.twist.twist.linear.x = v
        m.twist.twist.angular.z = w
        self.pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    n = OdomNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
