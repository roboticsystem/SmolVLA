#!/usr/bin/env python3
"""control_node —— /cmd_out → 底盘（差速 2 维）。

**安全限速在这里实现**（大纲 §8 风险表）：无论上游发来什么，
输出都被夹到 `[v_min, v_max] × [-w_max, w_max]` 内。
真机还需接物理急停（硬件接线，不在软件范围）。

与仿真 `env.step` 的运动学逐项一致：
    x += v·cos(θ)·dt ; y += v·sin(θ)·dt ; θ += ω·dt
（位姿积分在 odom_node，本节点只做限幅与下发。）
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class ControlNode(Node):
    def __init__(self):
        super().__init__("control_node")
        self.declare_parameter("v_max", 0.35)      # 与仿真 V_MAX 一致
        self.declare_parameter("w_max", 1.60)      # 与仿真 W_MAX 一致
        self.declare_parameter("v_min_rev", -0.10)  # 倒车限速（安全）
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(Twist, "/cmd_out", self.on_cmd, 10)
        self.get_logger().info("control_node 启动（限幅后下发 /cmd_vel）")

    def on_cmd(self, msg: Twist):
        v_max = float(self.get_parameter("v_max").value)
        w_max = float(self.get_parameter("w_max").value)
        v_min = float(self.get_parameter("v_min_rev").value)
        out = Twist()
        out.linear.x = max(v_min, min(v_max, msg.linear.x))
        out.angular.z = max(-w_max, min(w_max, msg.angular.z))
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    n = ControlNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
