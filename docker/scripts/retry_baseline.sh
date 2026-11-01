#!/usr/bin/env bash
# 回归基线测试：官方 launch（burger，今早验证过）+ 长 spawn 重试
source /3240704007/sci/VLA/docker/ros2_env.sh
S=/tmp/baseline_status.txt
echo "RUNNING $(date +%T)" > $S

ps -eo pid,cmd | grep -E "gzserver|gzclient|robot_state|spawn_entity|web_video" | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null
rm -f /dev/shm/fastrtps_* /dev/shm/fast_datasharing* 2>/dev/null
sleep 2

nohup ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py headless:=True > /tmp/gz6.log 2>&1 &

# 最多等 5 分钟（覆盖慢发现 + 自动 spawn）
ok=0
for i in $(seq 1 60); do
  ros2 topic list 2>/dev/null | grep -q /scan && ok=1 && break
  sleep 5
done
echo "scan_ok=$ok" >> $S
grep -E "died|Spawn.*success|spawn" /tmp/gz6.log | tail -4 >> $S
ros2 topic list 2>/dev/null | grep -E "scan|odom|clock" >> $S
echo "END $(date +%T)" >> $S
