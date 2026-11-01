#!/usr/bin/env bash
# 冒烟测试：cmd_vel -> 底盘运动 -> odom 位移（对齐真机 vla_bridge 接口）
source /3240704007/sci/VLA/docker/ros2_env.sh

echo "== before =="
ros2 topic echo /odom --field pose.pose.position --once 2>/dev/null | grep -E "^x|^y" | head -2

ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}, angular: {z: 0.0}}" >/dev/null 2>&1 &
PUB_PID=$!
sleep 6
kill $PUB_PID 2>/dev/null

echo "== after 6s v=0.2 =="
ros2 topic echo /odom --field pose.pose.position --once 2>/dev/null | grep -E "^x|^y" | head -2
