#!/usr/bin/env bash
# 网页观看：waffle_pi（自带相机）+ web_video_server
source /3240704007/sci/VLA/docker/ros2_env.sh
S=/tmp/webview_status.txt
echo "RUNNING $(date +%T)" > $S

ps -eo pid,cmd | grep -E "gzserver|gzclient|robot_state|spawn_entity|web_video" | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null
rm -f /dev/shm/fastrtps_* /dev/shm/fast_datasharing* 2>/dev/null
sleep 2

export DISPLAY=:99 TURTLEBOT3_MODEL=waffle_pi
nohup ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py headless:=True > /tmp/gz7.log 2>&1 &

# 等相机话题（渲染初始化慢，最多 5 分钟）
ok=0
for i in $(seq 1 60); do
  ros2 topic list 2>/dev/null | grep -q image && ok=1 && break
  sleep 5
done
echo "image_topics:" >> $S
ros2 topic list 2>/dev/null | grep -E "image|scan" >> $S

if [ $ok = 1 ]; then
  nohup ros2 run web_video_server web_video_server --ros-args -p port:=8080 > /tmp/wvs.log 2>&1 &
  sleep 10
  echo "DONE $(date +%T)" >> $S
  echo "URL0 list: http://localhost:8080/" >> $S
  echo "URL1 first-person: http://localhost:8080/stream_viewer?topic=/camera/image_raw&type=mjpeg" >> $S
else
  echo "FAIL: no image topic (gz7.log tail below)" >> $S
  grep -iE "error|died" /tmp/gz7.log | head -3 >> $S
fi
