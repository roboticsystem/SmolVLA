#!/usr/bin/env python3
"""E8-partial：chunk 缓存（一次推理执行 k 步）→ 推理次数 vs 反应性。

当前闭环评测每步都推理（只取预测 chunk 第 0 动作）。chunk 缓存 = 一次推理
产出 k 个动作、连发 k 步再推理：推理次数降为 1/k，但第 2..k 步开环，反应性下降。
纯部署侧权衡，不需要重训。独立副本——不动主评测路径（早先一版已回退）。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smolvla_finetune_sim import (DEV, IMG_KEYS, STATE_KEY, _slot_noise,
                                  build_state, truncate_action_head)
from smolvla_sim_data import SimDemoData
from sim.env import calibrated_env
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


@torch.no_grad()
def eval_chunked(policy, data, episodes=100, seed_base=9000, stride=37,
                 max_steps=200, exec_steps=5, batch=8):
    """返回 (sr, n_infer, total_steps)。"""
    policy.eval()
    K = max(1, int(exec_steps))
    n_infer = 0
    total_steps = 0
    per_ep = []
    done = 0
    ep_i = 0
    while done < episodes:
        B = min(batch, episodes - done)
        slots = []
        for i in range(B):
            s = seed_base + (ep_i + i) * stride
            env = calibrated_env(seed=s)
            obs = env.reset(seed=s)
            slots.append({"env": env, "obs": obs, "steps": 0, "seed": s,
                          "done": False, "sr": 0.0})
        ep_i += B
        while any(not s["done"] for s in slots):
            live = [s for s in slots if not s["done"]]
            if not live:
                break
            imgs = torch.stack([
                torch.from_numpy(np.asarray(s["obs"]["image"])).to(DEV).float()
                .permute(2, 0, 1) / 255.0 for s in live], 0)
            imgd = {IMG_KEYS[0]: imgs}
            st_t = torch.cat([build_state(s["obs"], data, DEV, None)
                              for s in live], 0)
            lang_t, lang_m = data.tokenize([s["obs"]["instruction"] for s in live])
            images, img_masks = policy.prepare_images(imgd)
            state_p = policy.prepare_state({STATE_KEY: st_t})
            noise = torch.cat([_slot_noise(policy.config, s["seed"], s["steps"], DEV)
                               for s in live], 0)
            pred = policy.model.sample_actions(images, img_masks, lang_t, lang_m,
                                               state_p, noise=noise)
            n_infer += 1
            for j in range(K):
                still = [s for s in live if not s["done"]]
                if not still:
                    break
                for i, s in enumerate(still):
                    a = data.denormalize_action(pred[i, min(j, pred.shape[1] - 1)])
                    a = np.clip(a.cpu().numpy(),
                                [-s["env"].cfg["v_max"], -s["env"].cfg["w_max"]],
                                [s["env"].cfg["v_max"], s["env"].cfg["w_max"]])
                    s["obs"], r, d, info = s["env"].step(a)
                    s["steps"] += 1
                    total_steps += 1
                    if info.get("success"):
                        s["sr"] = 1.0
                    if d or s["steps"] >= max_steps:
                        s["done"] = True
        for s in slots:
            per_ep.append(s["sr"])
        done += B
    return (float(np.mean(per_ep)) if per_ep else 0.0), n_infer, total_steps


def main():
    ap = argparse.ArgumentParser(description="E8-partial: chunk cache tradeoff")
    ap.add_argument("--ckpt", default="export/ft_detf")
    ap.add_argument("--data", default="export/nav_detf.h5")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--exec-steps", type=int, nargs="+", default=[1, 5, 10, 20])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="export/e8_chunk_cache.json")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    pol = SmolVLAPolicy.from_pretrained("models/smolvla_base").to(DEV)
    pol.config.max_action_dim = 2
    pol.config.n_action_steps = 50
    truncate_action_head(pol.model, 2)
    ck = torch.load(os.path.join(a.ckpt, "policy.pt"), map_location="cpu",
                    weights_only=False)
    pol.model.load_state_dict(ck["model"], strict=False)
    pol.model.chunk_size = 50
    pol.eval()
    data = SimDemoData(a.data, device=DEV, with_det=bool(ck.get("with_det")))
    res = []
    base = None
    for k in a.exec_steps:
        t0 = time.time()
        sr, n_inf, n_steps = eval_chunked(pol, data, episodes=a.episodes,
                                          max_steps=a.max_steps, exec_steps=k)
        if base is None:
            base = n_inf
        saved = 100.0 * (1 - n_inf / max(1, base))
        res.append({"k": k, "sr": sr, "n_infer": n_inf,
                    "total_steps": n_steps, "infer_saved_pct": saved,
                    "minutes": round((time.time() - t0) / 60, 1)})
        print("  k=%2d: SR %5.1f%%  推理 %5d 次（省 %.0f%%）  (%.1f min)"
              % (k, 100 * sr, n_inf, saved, (time.time() - t0) / 60), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("已写入 " + a.out)


if __name__ == "__main__":
    main()
