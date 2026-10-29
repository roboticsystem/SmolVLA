# VLA 语义导航 · 真机交付包 v1

日期：2026-09-25 ｜ 对应工作区：`/3240704007/sci/VLA`（主创新A 全部实验完成态）

---

## 这是什么

把这个包放到真机（RDK-X3 + 差速底盘 + 单目相机 + 激光）上，完成
**bring-up → 标定 → 采数 → 闭环推理** 四件事。

**请先建立一个正确预期**（详见 `docs/已知问题与告警.md`）：

> 模型**只见过仿真图像**。本包的首要目标不是「直接跑出仿真的成功率」，
> 而是①打通真机链路、②标定传感器口径、③采集真机数据回传。
> 闭环 SR 的 sim→real 迁移本身就是下一个实验，不要把它当默认假设。

## 目录结构

```
deliver_v1/
├── README.md                     ← 本文件
├── config/robot.yaml             ← 全部可调参数（⚠ 占位值清单在里面）
├── models/
│   ├── ft_detf/                  ← 主模型（PyTorch，99.8M 可训参数）
│   │   ├── policy.pt             ←    权重（865 MB，笔电侧推理用）
│   │   ├── meta.json             ←    训练配置（chunk=50, action_dim=2 …）
│   │   └── norm_stats.json       ←    ★ 归一化统计（从训练集抽取，勿重算）
│   └── deploy_4x1/               ← 端侧蒸馏模型（ONNX，D1 路线原料）
│       ├── prefix.onnx (338M) / denoise.onnx (55M) / expert.pt (48M) / meta.json
├── server/infer_server.py        ← 笔电侧推理服务（D2 路线，对齐 vla_bridge 契约）
├── tools/                        ← 真机五件套（见下）
├── ros2_ws/                      ← ROS2 骨架（5 节点 + launch，未联调）
└── docs/
    ├── 部署可行性评估.md          ← D1/D2 路线与 RDK-X3 阻塞分析
    ├── E6_端侧可行性闸门实施手册.md
    └── 已知问题与告警.md          ← ★ 先读这个
```

## Bring-up 顺序（严格按序，每步通过再走下一步）

```
第 0 步  通读 docs/已知问题与告警.md                    （10 分钟，必读）
第 1 步  python3 tools/selfcheck.py                     （环境/传感器自检）
第 2 步  python3 tools/perception_smoke.py --camera 0 --measure
         → 对着真实目标物标定 HSV，写回 config/robot.yaml
第 3 步  python3 tools/calibrate_camera.py --focal --obj-width-cm 20 --dist-cm 150 --cam-h 0.32
         → 写回 robot.yaml（hfov / cam_h / 单目饱和下限）
第 4 步  python3 tools/record_dataset.py --ros          （遥操采 10-20 回合）
         python3 tools/verify_dataset.py --h5 data_recorded/*.h5
第 5 步  笔电侧：infer_server.py；真机侧：ros2_ws 联调（第 5 步前 1-4 必须全过）
```

### 各工具一览

| 工具 | 用途 | 需要什么 |
|---|---|---|
| `tools/selfcheck.py` | 系统/包/相机/串口/ROS2/网络 自检 | 纯标准库（cv2 可选） |
| `tools/perception_smoke.py` | 检测冒烟 + HSV 标定（`--measure`） | opencv |
| `tools/calibrate_camera.py` | 标定 f / hfov / cam_h / 单目饱和下限 | opencv + 卷尺 |
| `tools/record_dataset.py` | 遥操采数 → 训练对齐的 HDF5 | opencv + h5py（+rclpy） |
| `tools/verify_dataset.py` | 采数质量门（能不能拿去训练） | h5py |
| `tools/onnx_smoke.py` | ONNX 在目标机的形态验证 | onnxruntime |
| `server/infer_server.py` | D2 推理服务（笔电 GPU） | 工作区 venv + base 模型 |

## 推理链路（D2 路线）

```
RDK-X3 (ros2_ws)                      笔电 (GPU)
┌─────────────────────┐   HTTP    ┌──────────────────────────┐
│ camera_node → /camera│          │ server/infer_server.py    │
│ lidar_node  → /scan  │ ───────→ │  载入 models/ft_detf      │
│ odom_node   → /odom  │  JSON    │  norm_stats.json 归一化   │
│ vla_bridge  ─────────┘          │  det 特征（HSV refs 需标定）│
│   ↑ /cmd_out          ←─────── │  → action [v, ω]          │
│ control_node → 底盘               └──────────────────────────┘
```

笔电侧启动（需要 VLA 工作区 + `.venv_smolvla`）：

```bash
.venv_smolvla/bin/python server/infer_server.py \
    --project /path/to/VLA --ckpt <包路径>/models/ft_detf --port 8000
```

契约：POST `{image_bgr(hex), h, w, scan[], state[5], instruction}` → `{action:[v,ω]}`。
前向路径复用工作区代码（`build_state / prepare_images / sample_actions`），
与训练评测**同一条路**，没有复制品。

## 采数回流（第 4 步之后的路径）

1. `verify_dataset.py` 通过 → 数据才可训练；
2. 回工作站：回填 `ep_success`、用标定 HSV 重算 `det_feat`（tools 会提示具体项）；
3. 进 `smolvla_finetune_sim.py` 微调（schema 对齐，改动应当极小）；
4. 这就是 **sim→real 微调** 的数据基础——论文 E10 的第一步。

## 上报清单（回来问下一步时，请带上）

- [ ] `selfcheck.py --json report.json` 的输出
- [ ] `verify_dataset.py` 的完整输出（含退出码）
- [ ] 标定结果：hfov / cam_h / 单目饱和下限 / 各目标物 HSV
- [ ] ROS2 联调卡在哪一步（launch 报错原文）
- [ ] 底盘控制接口的实际类型（/cmd_vel Twist？串口协议？）

## 已知边界（务必先读 docs/已知问题与告警.md）

| # | 事项 | 一句话 |
|---|---|---|
| ① | 域差 | 模型只见过 160×120 仿真渲染；真机 SR 预期大幅下降，属正常 |
| ② | 末段 15 cm | 模型会停在 0.30–0.45 m（已定位为示范策略遗留），真机初期把判据放宽到 0.5 m |
| ③ | 单目测距饱和 | 近距（< f·cam_h/(H−H/2)）测距失效，必须用激光表面距 |
| ④ | 速度上限 | 训练分布 |v| ≤ 0.35 m/s，**不要**放大底盘限幅 |
| ⑤ | ROS2 骨架 | 未联调；launch 参数与话题名都可能要按实车改 |
