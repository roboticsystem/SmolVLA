#!/usr/bin/env bash
# 本机原生 ROS2 Humble 环境加载脚本（沙箱内使用，容器方案见 docker/README.md）
# 用法：source docker/ros2_env.sh
#
# 关键点：本机 PATH 中 conda python3.14 优先于 /usr/bin/python3，
# 而 ROS Humble 的 python 组件只兼容 3.10，必须强制走系统 python。
export PATH="/usr/bin:$PATH"

source /opt/ros/humble/setup.bash

# 本沙箱内组播被限制，DDS 必须限定本机回环
export ROS_LOCALHOST_ONLY=1
export TURTLEBOT3_MODEL=burger

# 无显示器环境下为 Gazebo GUI 提供虚拟屏（gzserver 无头模式不需要）
if [ -z "${DISPLAY:-}" ]; then
  export USE_XVFB=1
fi

echo "[ros2_env] python -> $(which python3), ROS_DISTRO=$ROS_DISTRO, ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY"
