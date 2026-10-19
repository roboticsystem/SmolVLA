# 容器化与仿真环境说明

日期：2026-10-08 ｜ 适用工作区：`/3240704007/sci/VLA`

## 0. 本机（当前沙箱）结论：嵌套 Docker 不可行

当前工作区本身运行在一个受限沙箱容器内（Ubuntu 22.04，root）：

| 探测项 | 结果 | 对 Docker 的影响 |
|---|---|---|
| `dockerd` 启动 | ✅（需 `--iptables=false --bridge=none`，无 NET_ADMIN） | 镜像元数据/层下载正常 |
| `mount --bind` | ❌ `Operation not permitted` | 层解压失败 |
| `unshare -m`（mount ns） | ❌ | 层注册失败：`failed to register layer: unshare: operation not permitted` |
| `unshare -p`（pid ns） | ❌ | **无法创建任何容器**（容器至少需要 mount+pid ns） |
| Docker Hub 直连 | ❌ 超时（已配国内加速源，下载可用） | — |

即：daemon 能跑、镜像能下载，但**解包镜像和创建容器两步被外层 seccomp 封死**，
与存储驱动（overlay/vfs）无关，属硬阻塞。本目录资产请在**非嵌套环境**
（实验室服务器、个人机、或将来开放权限的宿主机）使用。

本机已装：`docker.io 29.1.3` + `docker-compose-v2`（daemon.json 已配国内加速 + vfs），
apt 源已切阿里云（原 `sources.list` 有备份）。

## 1. 路线A：sim/ 语义导航测试床容器化（轻量，~200MB）

`sim/` 依赖极轻：仅 `numpy` + `h5py`（画图另需 `matplotlib`），无 ROS 依赖。

```bash
cd docker/sim_testbed
./build_and_run.sh                              # 进入交互 shell（VLA 根目录挂载到 /work）
./build_and_run.sh python -m sim.run_eval --policy oracle --episodes 3
```

- 代码**不打进镜像**，运行时挂载，改代码不用重新 build
- 网络参数：脚本已带 `--network host`（兼容嵌套/受限环境；正常机器可去掉）
- 价值：论文实验可复现性——审稿人/答辩前在任意机器上原样复跑 E2/E4/E5

## 2. 路线B：ROS2 Humble + Gazebo + TurtleBot3 + Nav2（重量级，~4GB）

> **DGX Spark 用户**：直接看 `ros2_sim/README_Spark.md`（Spark 适配版：GPU 渲染启用、
> GUI/相机/激光全部可用、无需隧道；本节其余内容描述的是沙箱版环境）。

`docker/ros2_sim/` 提供 Dockerfile + compose + entrypoint：

```bash
cd docker/ros2_sim
docker compose up -d
docker compose exec ros2_sim bash
```

容器内三个入口（对应真机联调的三步）：

1. **世界仿真**：`ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py`
   （服务器无 GUI 加 `headless:=True`）
2. **遥操验证 /cmd_vel 链路**：`ros2 run teleop_twist_keyboard teleop_twist_keyboard`
   —— 这正是 `vla_bridge` 对接真机的同一接口
3. **Nav2 导航栈**：`ros2 launch nav2_bringup navigation_launch.py use_sim_time:=True`

设计对齐点（与《E1_仿真测试床说明.md》《部署可行性评估.md》衔接）：

- 差速底盘、`v_max=0.35 m/s`、10Hz 控制：TurtleBot3 burger 量级一致，
  可作为 sim/（运动学仿真）之上的**物理仿真中间层**
- `network_mode: host`：容器内 ROS2 与宿主机 `ros2_ws/` 五节点、vla_bridge HTTP 直连
- `ros2_ws` 直接挂载到 `/ros2_ws`：容器内 `colcon build` 即可联调现有 5 节点
- GPU 渲染：compose 中已留 nvidia 注释段，装 `nvidia-container-toolkit` 后启用

## 3. 本机的替代方案（已完成）：原生 ROS2

既然本机不能跑容器，已直接**原生安装**（apt，阿里云 ROS2 源）：

```
ros-humble-ros-base + gazebo(11.10) + ros-humble-gazebo-ros-pkgs
+ ros-humble-turtlebot3-gazebo + ros-humble-turtlebot3-navigation2 + ros-humble-nav2-bringup
+ web_video_server + ros-humble-rmw-cyclonedds-cpp
```

使用：`source docker/ros2_env.sh`，与上述容器内命令相同。

### 3.1 网页观看仿真态势（本沙箱可用）

```bash
bash docker/retry_baseline.sh        # 起 TurtleBot3 世界（burger，等 spawn 完成）
python3 docker/viz_server.py &       # 态势页服务（:8000）
# 浏览器打开 http://localhost:8000/  （IDE 内置浏览器可直接预览）
bash docker/smoke_drive.sh           # 同时开页面看机器人动起来
```

页面内容：机器人实时位姿（三角）+ 历史轨迹 + 激光点阵（/scan）+ 1m 网格。

### 3.2 本沙箱的硬限制（重要，别再踩）

1. **Gazebo 传感器管线不可用**：`/scan` 全 inf、相机话题不出——SensorManager 依赖
   渲染引擎；本沙箱无进程内 GLX（Xvfb 的 GLX 会让 gzserver 主循环死锁，
   无 DISPLAY 时渲染初始化失败→传感器永不更新）。物理引擎/里程计/服务/话题全正常。
2. **gzclient（Gazebo GUI）core dump**，web_video_server 无图可转。
   → 相机图像流（`serve_camera.sh`）在正常机器上可用，本机不行。
3. DDS 注意：进程必须**优雅退出**，`-9` 强杀会留下 /dev/shm/fastrtps_* 残留 +
   僵尸进程占满 localhost 参与者端口（症状：所有新节点 "service unavailable"）。
   清场命令见 `serve_camera.sh` 开头的 kill + rm 段。
4. conda python3.14 会抢占 ROS 的 python3.10 —— 必须 `source docker/ros2_env.sh`
   （内部强制 PATH 以 /usr/bin 开头）。
5. 组播在本沙箱不可靠：DDS 发现可能极慢（几十秒），脚本里的长轮询不是 bug。

## 4. 与 E7（BPU 量化）的区分

装好 docker **不解除** E7 阻塞：《部署可行性评估.md》阻塞 3
（`developer.horizon.ai` 不可达）依旧存在——地平线 OpenExplorer 工具链镜像
拿不到，与容器运行时无关，需要换网络或离线安装包。

## 5. 已知坑（本机特有，正常机器不适用）

1. apt：`apt-key` 执行异常（"Unknown error executing apt-key"），
   已绕过：`sources.list` 全部改用 `signed-by=/usr/share/keyrings/...` 写法
2. docker：`iptables` 无权限 → `bridge=none`，只能 `--network host`
3. docker：镜像层解包失败 → vfs 也救不了（根因是 unshare 被禁），本机别再尝试建容器
