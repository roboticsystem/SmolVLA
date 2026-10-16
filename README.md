# SmolVLA —— 面向端侧语义导航的 VLA 增强框架

> 室内跟随/语义导航机器人（RDK-X3 + 差速底盘）· 毕业设计代码库
>
> 核心主张：**观测完备性**（Observation Completeness）——
> VLA 导航性能的瓶颈在于"给模型什么信息"，而非"给多少数据"。

## 核心结果（仿真，400 回合完全配对，×5 种子）

| 增强 | SR 提升 | 统计 |
|---|---|---|
| 感知特征通道（基线 11.5%） | **+13 pp → 24.5%** | p = 1.1×10⁻⁴ |
| 时序记忆（历史帧记忆） | +7.75 pp | 显著 |
| 架构效应 vs 数据效应分离 | 架构显著 / 数据不显著 | p = 0.0067 / 0.215 |
| LoRA vs 全参微调 | 1.2% 参数 ≥ 451M 全参 | 边缘（p=0.069） |
| 端侧蒸馏 + chunk 缓存 | 时延 −93%，≈140× 算力缓解 | 见 E6/E8 |

阴性结果与失败根因分析见 `docs/03_结果与诊断/`（答辩方法论素材）。

## 仓库结构

```
SmolVLA/
├── smolvla_finetune_sim.py    # 训练/评测主干（推理服务依赖此文件在根目录）
├── smolvla_sim_data.py        # 数据管线与归一化口径
├── sim/                       # 语义导航仿真测试床（E1，纯 Python：numpy+h5py）
├── exp_plan.yaml              # 实验事前登记（E1–E12 状态/阻塞/unblock 条件）
├── scripts/
│   ├── train/                 # LoRA / IDAS / 蒸馏 / chunk / DAgger
│   ├── deploy/                # ONNX 导出 / INT8 量化 / 基准 / 算子普查
│   ├── experiments/           # E2/E5 扫描与多种子驱动
│   └── stats/                 # Holm 多重比较校正
├── ros2_ws/                   # ROS2 五节点（camera/lidar/odom/vla_bridge/control）
├── deliver/                   # 真机交付包（server/tools/config/docs；模型不入库）
├── docker/
│   ├── ros2_sim/              # ROS2 Humble + Gazebo + TurtleBot3 + Nav2 容器
│   │   └── README_Spark.md    # DGX Spark (ARM64) 适配说明
│   ├── sim_testbed/           # sim/ 测试床容器
│   ├── scripts/               # 推理服务/冒烟/态势页等运维脚本
│   └── gazebo/                # 自定义世界（俯视相机）与 launch
└── docs/                      # 大纲 / 平台 / 结果 / 部署 / 工作总结
```

## 快速开始

### 环境

```bash
python3.12 -m venv .venv_smolvla && source .venv_smolvla/bin/activate
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e ./lerobot -c requirements-smolvla.txt   # lerobot 源码需另行获取（见下）
```

> `lerobot/`（第三方框架源码）与 `models/smolvla_base/`（基座权重）体积大且非原创，
> 不入库。获取方式见 `docs/04_部署/部署可行性评估.md` 或联系作者。

### 仿真评测（sim 测试床，零 ROS 依赖）

```bash
python -m sim.run_eval --policy oracle --episodes 15
```

### 真机推理服务（D2 路线：板端采数据 → 服务端 GPU 推理）

```bash
python deliver/server/infer_server.py --project <本仓库根目录> \
    --ckpt <ft_detf 权重目录> --port 8000
# 契约：POST {image_bgr(hex), h, w, scan[], state[5], instruction} → {action:[v,ω]}
```

### ROS2 仿真容器（含 DGX Spark ARM64 适配）

见 `docker/ros2_sim/README_Spark.md`。

## 实验规范

- 事前登记：`exp_plan.yaml`（10 条目，含 blocker 与 unblock 条件）
- 统计：≥5 随机种子、配对检验（McNemar / bootstrap 4000 次）、Holm 族校正
- 阴性结果全部归档（8 条），防止过度声称

## 文档索引

| 文档 | 内容 |
|---|---|
| `docs/01_大纲与设计/毕业论文实验流程大纲.md` | E1–E12 实验设计总纲 |
| `docs/02_平台说明/E1_仿真测试床说明.md` | 自建测试床设计与精度验证 |
| `docs/03_结果与诊断/主创新A_诊断报告.md` | §4.1–4.26 全部实验与根因分析 |
| `docs/03_结果与诊断/结果表_主创新A.md` | 表 1–8b，全部数字可复现 |
| `docs/04_部署/` | 端侧部署路线与可行性评估 |
