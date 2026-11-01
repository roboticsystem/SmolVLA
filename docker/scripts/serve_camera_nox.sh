#!/usr/bin/env bash
# waffle_pi 不带 DISPLAY 测试：相机传感器在无 X 下是否可用
source /3240704007/sci/VLA/docker/ros2_env.sh
S=/tmp/webview_status.txt
echo "RUNNING_NOX $(date +%T)" > $S

ps -eo pid,cmd | grep -E "gzserver|gzclient|robot_state|spawn_entity|web_video" | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null
rm -f /dev/shm/fastrtps_* /dev/shm/fast_datasharing* 2>/dev/null
sleep 2

unset DISPLAY
export TURTLEBOT3_MODEL=waffle_pi
nohup ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py headless:=True > /tmp/gz8.log 2>&1 &

ok_scan=0; ok_img=0
for i in $(seq 1 40); do
  T=$(ros2 topic list 2>/dev/null)
  echo "$T" | grep -q /scan && ok_scan=1
  echo "$T" | grep -q image && ok_img=1
  [ $ok_scan = 1 ] && [ $ok_img = 1 ] && break
  sleep 5
done
echo "scan=$ok_scan image=$ok_img" >> $S
ros2 topic list 2>/dev/null | grep -E "image|scan" >> $S
if [ $ok_img = 1 ]; then
  nohup ros2 run web_video_server web_video_server --ros-args -p port:=8080 > /tmp/wvs.log 2>&1 &
  sleep 10
  echo "DONE $(date +%T)" >> $S
  echo "URL1 first-person: http://localhost:8080/stream_viewer?topic=/camera/image_raw&type=mjpeg" >> $S
else
  echo "FAIL_NOX $(date +%T)" >> $S
fi
