#!/usr/bin/env bash
# 网页观看最终版：gzserver(世界内置机器人+俯视相机) -> web_video_server -> 浏览器
# 结果写 /tmp/webview_status.txt
source /3240704007/sci/VLA/docker/ros2_env.sh
S=/tmp/webview_status.txt
echo "RUNNING $(date +%T)" > $S

# 清场（只清仿真相关，保留 daemon 无所谓）
ps -eo pid,cmd | grep -E "gzserver|gzclient|robot_state|spawn_entity|web_video" | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null
rm -f /dev/shm/fastrtps_* /dev/shm/fast_datasharing* 2>/dev/null
sleep 2

export DISPLAY=:99
export GAZEBO_MODEL_PATH=/usr/share/gazebo-11/models:/opt/ros/humble/share/turtlebot3_gazebo/models:$GAZEBO_MODEL_PATH

nohup gzserver /3240704007/sci/VLA/docker/webcam.world \
  -slibgazebo_ros_init.so -slibgazebo_ros_factory.so -slibgazebo_ros_force_system.so \
  > /tmp/gz_web.log 2>&1 &

# 等传感器话题（相机渲染慢，放宽到 4 分钟）
ok=0
for i in $(seq 1 80); do
  T=$(ros2 topic list 2>/dev/null)
  echo "$T" | grep -q image && ok=1 && break
  grep -q "died" /tmp/gz_web.log && break
  sleep 3
done
echo "topics found:" >> $S
ros2 topic list 2>/dev/null | grep -E "image|scan|clock" >> $S 2>&1
grep -iE "died|error" /tmp/gz_web.log | head -3 >> $S

if [ $ok = 0 ]; then
  echo "FAIL: no image topics" >> $S; exit 1
fi

nohup ros2 run web_video_server web_video_server --ros-args -p port:=8080 > /tmp/wvs.log 2>&1 &
sleep 10
echo "DONE $(date +%T)" >> $S
echo "URL0 list: http://localhost:8080/" >> $S
echo "URL1 first-person: http://localhost:8080/stream_viewer?topic=/camera/image_raw&type=mjpeg" >> $S
echo "URL2 top-down: http://localhost:8080/stream_viewer?topic=/scene/cam/image_raw&type=mjpeg" >> $S
