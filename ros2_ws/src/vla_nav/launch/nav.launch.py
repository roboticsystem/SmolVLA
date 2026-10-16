from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    common = {"package": "vla_nav", "output": "screen"}
    return LaunchDescription([
        Node(**common, executable="camera_node"),
        Node(**common, executable="lidar_node"),
        Node(**common, executable="odom_node"),
        Node(**common, executable="control_node"),
        Node(**common, executable="vla_bridge_node",
             parameters=[{"server": "http://192.168.1.100:8000/infer",
                          "instruction": "go to the red chair"}]),
    ])
