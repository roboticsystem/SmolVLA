#!/usr/bin/env bash
# 一键搭建网页观看：等基线仿真 -> 换 waffle_pi（带相机）-> 起 web_video_server
# 结果写入 /tmp/webview_status.txt（DONE/FAIL + URL）
source /3240704007/sci/VLA/docker/ros2_env.sh
S=/tmp/webview_status.txt
echo "RUNNING $(date +%T)" > $S

# 1) 基线没起来就自己拉起（burger + 标准世界）
if ! ros2 topic list 2>/dev/null | grep -q /scan; then
  export DISPLAY=:99
  nohup ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py headless:=True > /tmp/gz4.log 2>&1 &
fi
ok=0
for i in $(seq 1 30); do
  ros2 topic list 2>/dev/null | grep -q /scan && ok=1 && break
  sleep 3
done
if [ $ok = 0 ]; then
  echo "FAIL: baseline sim not up (no /scan)" >> $S; exit 1
fi
echo "baseline OK" >> $S

# 2) 关掉基线，起 waffle_pi（自带相机）
pkill -9 -f "ros2 launch"; pkill -9 -f gzserver; sleep 3
rm -f /dev/shm/fastrtps_* /dev/shm/fast_datasharing*
export DISPLAY=:99 TURTLEBOT3_MODEL=waffle_pi
nohup ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py headless:=True > /tmp/gz5.log 2>&1 &
ok=0
for i in $(seq 1 40); do
  ros2 topic list 2>/dev/null | grep -qE "camera/image|/scan" && ok=1 && break
  sleep 3
done
ros2 topic list 2>/dev/null | grep -E "camera|image|scan" >> $S
if [ $ok = 0 ]; then
  echo "FAIL: waffle_pi not up" >> $S; exit 1
fi
echo "waffle_pi OK" >> $S

# 3) 起 web_video_server（8080 端口，MJPEG）
export TURTLEBOT3_MODEL=waffle_pi
nohup ros2 run web_video_server web_video_server --ros-args -p port:=8080 > /tmp/wvs.log 2>&1 &
sleep 8
echo "DONE $(date +%T)" >> $S
echo "URL0 list: http://localhost:8080/" >> $S
echo "URL1 view: http://localhost:8080/stream_viewer?topic=/camera/image_raw" >> $S
