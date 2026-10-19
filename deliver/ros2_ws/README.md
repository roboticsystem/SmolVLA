# ROS2 Humble 工作空间（E1 第 3 项）

> **状态：代码骨架完成，未经真机联调** —— 这是「不依赖硬件就能写」的部分。
> 联调需要 RDK-X3 与传感器（当前阻塞，见《部署可行性评估.md》）。

## 结构

```
ros2_ws/src/vla_nav/
├── package.xml
├── setup.py
├── launch/nav.launch.py        # 一键启动全部节点
├── vla_nav/
│   ├── camera_node.py          # C70/C100 RGB → /camera/image_raw
│   ├── lidar_node.py           # 国科光芯 D6 → /scan
│   ├── odom_node.py            # 编码器 + 扫描匹配 → /odom（**无 IMU**）
│   ├── control_node.py         # /cmd_out → 底盘（差速 2 维）
│   └── vla_bridge_node.py      # 局域网 RPC：图像+激光+里程 → /cmd_out
└── config/params.yaml
```

## 话题图

```
/camera/image_raw ─┐
/scan              ─┼─→ vla_bridge_node ──→ /cmd_out ──→ control_node → 电机
/odom              ─┘     （HTTP：服务器侧 VLA 推理）
```

## 三个关键设计决定（都来自本仓库的实测结论）

### ① 无 IMU 的里程计

BOM 无 IMU（大纲 §1.1 第 4 条更正）。位姿 = **轮式编码器 + 2D 激光扫描匹配**：
编码器做预测（快、会漂），扫描匹配做更新（慢、不漂），**每 5 m 强制重定位**
（大纲 §8 风险表的对策）。

### ② 动作空间是 2 维连续 (v, ω)

`vla_bridge_node` 输出的 `geometry_msgs/Twist` 只用 `linear.x` 与 `angular.z`，
与仿真训练的动作空间**逐维一致** —— 自建测试床的核心收益：
**仿真→真机没有动作维度再映射**（大纲 §5 E1 第 1 条）。

### ③ VLA 推理在服务器侧（云边分层 D2）

`vla_bridge_node` 走局域网 HTTP：端侧只采集与控制，VLA 在服务器推理。
这是 E6 闸门排除 Route A（450M 直接端侧）后的**兜底路线**；
E7 的 INT8 与板载实测打通后可换成本地推理（D1 纯端侧）。

## 与仿真管线的对应关系

| 仿真（`sim/`） | ROS2 节点 | 说明 |
|---|---|---|
| `sensors.render` | `camera_node` | 真机 720P → `vla_bridge` 里 resize_with_pad 到 512（与训练一致） |
| `sensors.lidar_scan` | `lidar_node` | 120 束 / 12 m / 360°，与 D6 对齐 |
| `env.step` 运动学 | `control_node` | 同一差速模型；**限速阈值在这里实现**（安全） |
| `perception.detect` | `vla_bridge`（服务器侧） | 感知+决策分离：感知在端、决策在云亦可 |

## 已知未实现 / 待真机联调

- [ ] `odom_node` 扫描匹配为简化相关性打分，需按 D6 实际噪声重调
- [ ] `camera_node` 内参（HFOV 64.5°）来自仿真，真机需棋盘格标定
- [ ] `vla_bridge` 断网降级未实现（E9 的指标之一，真机必须做）
- [ ] 物理急停是硬件接线，不在代码内（大纲 §8 风险表）
- [ ] 全部节点未经 `colcon build` / `ros2 launch` 验证（本机无 ROS2）
