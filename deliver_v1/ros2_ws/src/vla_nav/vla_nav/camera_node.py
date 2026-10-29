#!/usr/bin/env python3
"""camera_node —— C70/C100 RGB 相机 → /camera/image_raw。

与训练的一致性要点：
- 内参 HFOV 64.5° 来自仿真（`sensors.CAMERA_HFOV_DEG`），真机需**棋盘格标定**后覆盖
- 统一发 RGB（通道序与 `vla_bridge` 的解释一致）
- 帧率 30 fps；`vla_bridge` 以 10 Hz 取最新帧（不排队，避免延迟堆积）
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        self.declare_parameter("device", "/dev/video0")
        self.declare_parameter("width", 1280)
        self.declare_parameter("height", 720)
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("hfov_deg", 64.5)      # 与仿真一致；真机标定后覆盖
        self.pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.cap = None
        self._open()
        self.create_timer(1.0 / self.get_parameter("fps").value, self.tick)
        self.get_logger().info(
            "camera {}x{} @{}Hz".format(self.get_parameter("width").value,
                                        self.get_parameter("height").value,
                                        self.get_parameter("fps").value))

    def _open(self):
        try:
            import cv2
            dev = self.get_parameter("device").value
            self.cap = cv2.VideoCapture(int(dev) if dev.isdigit() else dev)
            self.cap.set(3, self.get_parameter("width").value)
            self.cap.set(4, self.get_parameter("height").value)
            if not self.cap.isOpened():
                raise RuntimeError("打不开相机 " + str(dev))
        except Exception as e:
            self.get_logger().error("相机打开失败（骨架模式，等待真机）: " + str(e))
            self.cap = None

    def tick(self):
        if self.cap is None:
            return
        import cv2
        ok, frame = self.cap.read()
        if not ok:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        msg = Image()
        msg.height, msg.width = frame.shape[:2]
        msg.encoding = "rgb8"
        msg.step = frame.shape[1] * 3
        msg.data = frame.tobytes()
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    n = CameraNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
