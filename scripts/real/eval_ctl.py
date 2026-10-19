"""受控对照评测：区分「策略真退化」vs「合并数据统计量带偏评测口径」。

四种组合，同一评测口径（calibrated_env、seed=9000、max_steps=200、det_mode=none）：
  A. ft_real_v1 + nav_real_mix.h5   （训练时的统计量）—— 预期复现 0%
  B. ft_real_v1 + nav_detf.h5       （换纯仿真统计量）—— 若 SR 恢复 → 统计量是元凶
  C. ft_detf     + nav_detf.h5      （基线自评）      —— 当场复现基线 SR
  D. ft_detf     + nav_real_mix.h5  （反向对照）      —— 若也崩 → 统计量主导，权重可能没坏

用法: python eval_ctl.py --episodes 20
"""

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # 项目根（scripts/<类>/ 上一级），保证 import sim/ smolvla_* 可用

import argparse
import os
import sys

import numpy as np
import torch

import smolvla_finetune_sim as S
from smolvla_finetune_sim import (DEV, SmolVLAPolicy, SimDemoData,
                                  closed_loop_sr, reconfigure_for_diff_drive)


def load_policy(policy_dir: str):
    ck = torch.load(os.path.join(policy_dir, "policy.pt"), map_location=DEV)
    policy = SmolVLAPolicy.from_pretrained(S.CKPT).to(DEV)
    policy.eval()
    cfg = policy.config
    chunk = int(ck.get("chunk", 50))
    cfg.chunk_size = chunk
    cfg.n_action_steps = chunk
    # 本项目 det-features-mode=none（4 维感知），_need = 5+4 = 9 ≤ 32 → state_dim=None
    reconfigure_for_diff_drive(policy, action_dim=2, n_cam=int(ck.get("n_frames", 1)),
                               state_dim=None)
    miss, unexp = policy.model.load_state_dict(ck["model"], strict=False)
    print(f"  [{policy_dir}] 加载 state_dict：缺失 {len(miss)} 项 / 多余 {len(unexp)} 项")
    policy.eval()
    return policy, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--only", default=None,
                    help="只跑指定组合（ft_real_v1 / ft_real_v1+simstats / "
                         "ft_detf_base / ft_detf+realstats），默认全部")
    a = ap.parse_args()

    combos = [
        ("ft_real_v1", "export/ft_real_v1", "export/nav_real_mix.h5"),
        ("ft_real_v1+simstats", "export/ft_real_v1", "export/nav_detf.h5"),
        ("ft_detf_base", "export/ft_detf", "export/nav_detf.h5"),
        ("ft_detf+realstats", "export/ft_detf", "export/nav_real_mix.h5"),
    ]
    results = {}
    for tag, pdir, dpath in combos:
        if a.only and tag != a.only:
            continue
        print(f"\n=== {tag}: {pdir} + {dpath} ===", flush=True)
        policy, cfg = load_policy(pdir)
        data = SimDemoData(dpath, device=DEV, max_state_dim=cfg.max_state_dim,
                           tokenizer_max_length=cfg.tokenizer_max_length,
                           chunk_size=cfg.chunk_size, max_action_dim=2, n_frames=1,
                           with_lidar=False, with_det=True)
        det = []
        sr, spl, ne = closed_loop_sr(policy, data, episodes=a.episodes, seed=a.seed,
                                     max_steps=a.max_steps, n_frames=1,
                                     visible_start=False, detail=det,
                                     det_mode="none", mem_ttl=25, calib=True)
        n_ok = sum(1 for d in det if d[1] > 0)
        print(f"  >>> {tag}: SR {100*sr:.1f}% ({n_ok}/{len(det)})  "
              f"SPL {spl:.3f}  NE {ne:.2f} m", flush=True)
        print(f"      detail: {det}", flush=True)
        results[tag] = (sr, spl, ne)
        del policy, data
        torch.cuda.empty_cache()

    print("\n===== 汇总（同口径，200 步 / seed 9000）=====")
    for tag, (sr, spl, ne) in results.items():
        print(f"  {tag:22s}  SR {100*sr:5.1f}%  SPL {spl:.3f}  NE {ne:.2f} m")


if __name__ == "__main__":
    sys.exit(main())
