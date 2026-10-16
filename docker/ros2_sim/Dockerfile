# 路线B：ROS2 Humble + Gazebo Classic 11 + TurtleBot3 + Nav2 仿真容器
# 通用版：x86_64 与 aarch64（DGX Spark 等 ARM 机器）均可构建，
#         ros:humble 官方镜像双架构，build 时自动匹配本机架构。
#
# 镜像源切换：国内网络默认走阿里云；海外/其他网络构建时加：
#   docker build --build-arg USE_CN_MIRROR=0 -t vla-ros2-sim:humble .
FROM ros:humble-ros-base

ARG USE_CN_MIRROR=1

ENV DEBIAN_FRONTEND=noninteractive

# GPU 渲染（配合 nvidia-container-toolkit 的 graphics capability；
# DGX Spark / 带 NVIDIA 驱动的宿主机上 Gazebo 与相机传感器可用 GPU GL）
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    QT_X11_NO_MITSHM=1

RUN if [ "$USE_CN_MIRROR" = "1" ]; then \
      sed -i 's|http://archive.ubuntu.com|https://mirrors.aliyun.com|g; s|http://security.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list; \
    fi \
 && apt-get update -qq \
 && apt-get install -y --no-install-recommends \
      ros-humble-gazebo-ros-pkgs \
      ros-humble-turtlebot3-gazebo \
      ros-humble-turtlebot3-navigation2 \
      ros-humble-nav2-bringup \
      ros-humble-teleop-twist-keyboard \
      ros-humble-web-video-server \
      gazebo11 \
      xvfb \
      x11-apps \
      mesa-utils \
      ros-dev-tools \
 && rm -rf /var/lib/apt/lists/*

ENV TURTLEBOT3_MODEL=burger \
    GAZEBO_MODEL_PATH=/opt/ros/humble/share/turtlebot3_gazebo/models

# 工作区挂载点：把 VLA/ros2_ws 挂进来做 colcon 联调
VOLUME ["/ros2_ws"]
WORKDIR /ros2_ws

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
