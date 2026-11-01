#!/usr/bin/env bash
# ROS2 仿真容器入口：source 环境 + 工作区叠加层
set -e
source /opt/ros/humble/setup.bash

# 若工作区已编译则叠加
if [ -f /ros2_ws/install/setup.bash ]; then
  source /ros2_ws/install/setup.bash
fi

# 无 DISPLAY 时可用 xvfb 提供虚拟屏（headless GUI 兜底）
if [ -z "${DISPLAY:-}" ] && [ "${USE_XVFB:-0}" = "1" ]; then
  Xvfb :99 -screen 0 1280x1024x24 &
  export DISPLAY=:99
fi

exec "$@"
