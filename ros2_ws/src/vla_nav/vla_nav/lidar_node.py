#!/usr/bin/env python3
"""lidar_node —— 国科光芯 D6 → /scan。

参数对齐仿真 `sensors.lidar_scan`：120 束 / 12 m / 360°。
D6 实际输出若是极坐标点云，这里重排成 ROS 标准 `LaserScan`
（angle_min = -π，逆时针），`vla_bridge` 再重排回「车头为 0」的约定 ——
两处约定必须成对，否则动作的方位参考系就错了。
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class LidarNode(Node):
    def __init__(self):
        super().__init__("lidar_node")
        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("beams", 120)
        self.declare_parameter("range_max", 12.0)
        self.declare_parameter("hz", 10.0)
        self.pub = self.create_publisher(LaserScan, "/scan", 10)
        self.create_timer(1.0 / self.get_parameter("hz").value, self.tick)
        self.get_logger().warning(
            "D6 驱动未接入（骨架模式）：发布全 12 m 占位扫描，真机联调时替换 _read_device")

    def _read_device(self):
        """TODO(真机)：从 D6 读取一帧，返回长度 `beams` 的距离数组（米）。"""
        return None

    def tick(self):
        n = int(self.get_parameter("beams").value)
        r = self._read_device()
        if r is None:
            r = [self.get_parameter("range_max").value] * n
        msg = LaserScan()
        msg.header.frame_id = "laser"
        msg.angle_min = -3.14159265
        msg.angle_max = 3.14159265
        msg.angle_increment = 2 * 3.14159265 / n
        msg.range_min = 0.15
        msg.range_max = float(self.get_parameter("range_max").value)
        msg.ranges = [float(x) for x in r]
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    n = LidarNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
