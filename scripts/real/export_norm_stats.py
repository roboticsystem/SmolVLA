"""从合并训练数据导出 norm_stats.json，与 ft_real_v1/policy.pt 配对部署。

统计口径必须与训练完全一致：SimDemoData 在加载时重算 st_mean/st_std/act_mean/act_std，
这里原样落盘。真机 infer_server.py / vla_bridge 消费 st_mean/st_std/act_mean/act_std 四个键。
"""

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # 项目根（scripts/<类>/ 上一级），保证 import sim/ smolvla_* 可用

import json

import numpy as np
import torch

from smolvla_finetune_sim import SmolVLAPolicy, SimDemoData

OUT = "export/ft_real_v1/norm_stats.json"
DATA = "export/nav_real_mix.h5"

# 先拿 policy 配置（tokenizer_max_length 等必须与训练一致）
ck = torch.load("export/ft_real_v1/policy.pt", map_location="cpu")
cfg = SmolVLAPolicy.from_pretrained("models/smolvla_base").config

data = SimDemoData(DATA, device="cpu", max_state_dim=cfg.max_state_dim,
                   tokenizer_max_length=cfg.tokenizer_max_length,
                   chunk_size=int(ck.get("chunk", 50)), max_action_dim=2,
                   n_frames=int(ck.get("n_frames", 1)),
                   with_lidar=False, with_det=True)

stats = {
    "st_mean": np.asarray(data.st_mean, dtype=np.float64).tolist(),
    "st_std": np.asarray(data.st_std, dtype=np.float64).tolist(),
    "act_mean": np.asarray(data.act_mean, dtype=np.float64).tolist(),
    "act_std": np.asarray(data.act_std, dtype=np.float64).tolist(),
    "source": DATA,
    "policy": "export/ft_real_v1/policy.pt",
    "n_frames": int(ck.get("n_frames", 1)),
    "chunk": int(ck.get("chunk", 50)),
    "det_features_mode": str(ck.get("det_features_mode", "none")),
    "note": "与训练数据 nav_real_mix.h5 同口径（sim 96147 + real 3812 帧共训统计量），"
            "部署时不得现场重算或替换为纯仿真统计量",
}
with open(OUT, "w") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print("已保存", OUT)
print("  st_mean[:5] =", np.round(data.st_mean[:5], 4).tolist())
print("  st_std[:5]  =", np.round(data.st_std[:5], 4).tolist())
print("  act_mean    =", np.round(data.act_mean, 4).tolist())
print("  act_std     =", np.round(data.act_std, 4).tolist())
