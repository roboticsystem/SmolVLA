# DGX Spark 适配说明（ros2_sim）

日期：2026-10-16 ｜ 对应通用版：`Dockerfile` + `docker-compose.spark.yml`

## 0. Spark 上的前置检查（一次）

DGX OS 一般已带 docker + nvidia-container-toolkit，确认即可：

```bash
docker --version && nvidia-container-cli --version
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi   # 应列出 GB10
```

## 1. 构建与启动

```bash
cd docker/ros2_sim
docker compose -f docker-compose.spark.yml up -d --build
docker compose -f docker-compose.spark.yml exec ros2_sim bash
```

- 国内网络：默认阿里云镜像源；海外构建加 `--build-arg USE_CN_MIRROR=0`
- 镜像自动匹配 arm64（Spark 是 aarch64，ros:humble 官方双架构）

## 2. 验证清单（按序）

```bash
# 1) DDS 本机通信
ros2 topic pub -r 1 /t1 std_msgs/msg/Int32 "{data: 1}" &   # 另一终端 ros2 topic echo /t1 --once

# 2) Gazebo 世界 + GUI（Spark 有真实 GL，gzclient 能开）
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
#   无显示器的场合：headless:=True

# 3) 传感器话题（重点：本沙箱做不了的部分在这里应全部可用）
ros2 topic list                     # 应见 /scan /odom /imu /clock
ros2 topic hz /scan                 # 应 ≈5Hz 有真实数据（非 inf）

# 4) 相机（waffle_pi 才有相机）
export TURTLEBOT3_MODEL=waffle_pi   # 重启世界后
ros2 topic hz /camera/image_raw     # 应 ≈30Hz
# 网页看图：ros2 run web_video_server web_video_server   → 浏览器 http://localhost:8080

# 5) 运动链路
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}}"
ros2 topic echo /odom --field pose.pose.position --once   # x 应增大
```

> 沙箱里 /scan 全 inf、相机无话题的问题，在 Spark 上应消失；
> 若仍异常，优先怀疑 GPU 渲染（`glxinfo | grep renderer` 应显示 NVIDIA 而非 llvmpipe）。

## 3. 五节点联调（ros2_ws）

```bash
cd /ros2_ws && colcon build --symlink-install
source install/setup.bash
ros2 launch <你的launch>            # 5 节点首次联调
```

`ros2_ws` 与 `deliver_v1` 已挂载进容器，代码改动即时生效。

## 4. 与 RDK-X3 小车联动（Spark 作联调中枢）

拓扑（同一交换机/热点即可，无需隧道）：

```
RDK-X3 小车 ──ROS2 DDS(同网段)── Spark ──HTTP── 推理服务
```

- 小车与 Spark 同网段：`ROS_DOMAIN_ID` 两边一致（默认 0），`ROS_LOCALHOST_ONLY` **不要设**
- 推理服务二选一：
  - **留在这台 4090 服务器**（10.252.18.70:8002，走你现有隧道）——小车/Spark 都指它
  - **搬进 Spark 容器**：`/deliver_v1` 已挂载，装个 venv 跑 `infer_server.py --port 8000`；
    SmolVLA-500M 在 GB10 上推理无压力（aarch64 需装 aarch64 torch wheel）
- 小车 `vla_bridge` 配 Spark IP : 8000（本地闭环延迟最低）

## 5. Jazzy 原生路线（可选，不用容器）

DGX OS 24.04 可直接原生装（跳过容器层，性能最好）：

```bash
sudo apt install ros-jazzy-ros-base ros-jazzy-ros-gz ros-jazzy-turtlebot3-gazebo
```

注意：Humble→Jazzy 有 API 变化（参数声明、launch 语法小改），
`ros2_ws` 五节点需要适配；想零改动就留在 Humble 容器里。

## 6. 与沙箱版的行为差异速查

| 项 | 沙箱版 | Spark 版 |
|---|---|---|
| GUI (gzclient/rviz2) | ❌ 崩溃 | ✅ 直接可用 |
| /scan 数据 | inf（传感器管线死） | ✅ 正常 |
| 相机话题 | ❌ 无 | ✅ 正常（waffle_pi） |
| docker bridge 网络 | ❌ 不可用，强制 host | ✅ 可用（默认仍 host，省事） |
| 容器内 GPU GL | ❌ | ✅（graphics capability） |
| 端口转发/隧道 | 需要 | ❌ 不需要 |
