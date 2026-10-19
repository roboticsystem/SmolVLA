#!/usr/bin/env bash
# 耐心版：DISPLAY=:99 下 Ogre 软渲染初始化极慢，全部等待放宽
source /3240704007/sci/VLA/docker/ros2_env.sh
S=/tmp/webview_status.txt
echo "RUNNING patient $(date +%T)" > $S

ps -eo pid,cmd | grep -E "gzserver|gzclient|robot_state|spawn_entity|web_video" | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null
rm -f /dev/shm/fastrtps_* /dev/shm/fast_datasharing* 2>/dev/null
sleep 2

export DISPLAY=:99 TURTLEBOT3_MODEL=waffle_pi
nohup ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py headless:=True > /tmp/gzA.log 2>&1 &

# 1) 轮询 spawn 服务（最多 10 分钟），就绪立即 spawn
spawned=0
for i in $(seq 1 60); do
  if timeout 20 ros2 service call /spawn_entity gazebo_msgs/srv/SpawnEntity \
     "{name: waffle_pi, initial_pose: {position: {x: -2.0, y: -0.5, z: 0.01}}}" 2>&1 | grep -q success; then
    spawned=1; echo "spawned at loop $i" >> $S; break
  fi
  sleep 2
done
echo "spawned=$spawned $(date +%T)" >> $S

# 2) 等相机图像（最多 10 分钟）
ok=0
for i in $(seq 1 60); do
  ros2 topic list 2>/dev/null | grep -q image && ok=1 && break
  sleep 10
done
echo "image=$ok $(date +%T)" >> $S
ros2 topic list 2>/dev/null | grep -E "image|scan" >> $S

# 3) 有图像就起视频服务，没有就起 SVG 态势页（至少有激光）
if [ $ok = 1 ]; then
  nohup ros2 run web_video_server web_video_server --ros-args -p port:=8080 > /tmp/wvs.log 2>&1 &
  sleep 10
  echo "DONE $(date +%T)" >> $S
  echo "URL1 first-person: http://localhost:8080/stream_viewer?topic=/camera/image_raw&type=mjpeg" >> $S
else
  echo "DONE_SVG $(date +%T)" >> $S
fi
