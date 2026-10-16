"""语义 MultiRoom 仿真测试床（E1）。

模块：
    world      场景生成（MultiRoom + 语义物体 + 行人）
    sensors    2D 激光雷达（DDA 光线投射）与第一人称语义 RGB 渲染
    env        差速底盘环境、L1/L2/L3 语言指令、导航指标
    baselines  Random / Frontier / Oracle 基线策略
    run_eval   评测驱动（CLI）

用法：
    cd /3240704007/sci/VLA
    .venv_smolvla/bin/python -m sim.run_eval --episodes 100 --out export/sim_baseline.json
"""

from .world import CELL, SEMANTICS, Obj, Pedestrian, Scene, make_pedestrians, make_scene

__all__ = ["CELL", "SEMANTICS", "Obj", "Pedestrian", "Scene", "make_pedestrians", "make_scene"]
