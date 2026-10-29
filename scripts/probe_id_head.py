#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**ID 头探针**：冻结训练好的策略，只训一个新鲜逆动力学头，
检验「VLM 前缀特征里到底有没有可学的逆动力学信息」。

## 为什么需要（E5 阴性结果的归因）

E5 的 200 回合评测显示 IDAS 无显著效应。这有两种互斥的解释：

1. **辅助任务没学起来**——前缀特征对「两帧之间执行了什么动作」不可预测，
   辅助梯度是噪声 ⇒ 方法本身在这个架构上不成立；
2. **学了但没用**——逆动力学信息可被头预测，但主任务（flow matching）
   本来就用得上这些信息，额外的显式监督不再提供新梯度 ⇒ 迁移不增益。

两种解释对论文的写法完全不同（§4.4 的局限性讨论），探针把它们分开：

* 对每个模型，取**冻结**特征 [f_t, f_{t+1}]，从头训一个小 ID 头；
* 对照 1：**随机标签**（动作打乱）——头能到多低纯属记忆训练集；
* 对照 2：**none 模型的特征**——如果 idas_v2 特征的探针误差显著低于
  none，说明辅助监督确实让特征更「动力学可读」（学起来了、但没帮到主任务）。

开销：每模型只需几百次前向抽特征（无梯度）+ 秒级 MLP 训练。

用法：
    python scripts/probe_id_head.py \
        --models idas_v2:export/e5_idas_v2_s42,none:export/e5_none_s42 \
        --data export/nav_detf.h5 --pairs 384 --out export/e5_eval/probe.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from eval_ft_models import load_policy  # noqa: E402
from smolvla_idas import (InverseDynamicsHead,  # noqa: E402
                          PrefixFeatureCapture, frame_feature,
                          next_frame_indices)
from smolvla_finetune_sim import DEV  # noqa: E402
from smolvla_sim_data import SimDemoData  # noqa: E402


def collect_features(policy, data, rng, n_pairs, batch=16):
    """前向抽 (f_t, f_t1, a)；无效对（跨回合/切不满 chunk）跳过。"""
    cap = PrefixFeatureCapture(policy.model)
    F_t, F_1, A = [], [], []
    got = 0
    while got < n_pairs:
        idx = data.sample_index(batch, rng)
        nxt, ok = next_frame_indices(idx, data.ep_off, chunk=int(data.chunk))
        if not ok.any():
            continue
        i_t = np.asarray(idx)[ok]
        i_1 = np.asarray(nxt)[ok]
        with torch.no_grad():
            policy.forward(data.batch(i_t), reduction="none")
            f_t = frame_feature(cap.prefix)
            policy.forward(data.batch(i_1), reduction="none")
            f_1 = frame_feature(cap.prefix)
        F_t.append(f_t.cpu())
        F_1.append(f_1.cpu())
        A.append(torch.as_tensor(data.actions[i_1], dtype=torch.float32))
        got += int(ok.sum())
    cap.remove()
    return torch.cat(F_t)[:n_pairs], torch.cat(F_1)[:n_pairs], torch.cat(A)[:n_pairs]


def probe_once(f_t, f_1, a, feat_dim, action_dim=2, iters=400, seed=0,
               shuffle_labels=False):
    """在冻结特征上从头训一个 ID 头，返回 (训练 MSE, 验证 MSE)。"""
    n = f_t.shape[0]
    ntr = int(n * 0.75)
    g = np.random.default_rng(seed)
    perm = g.permutation(n)
    tr, va = perm[:ntr], perm[ntr:]
    if shuffle_labels:                       # 随机标签对照：打乱训练与验证的配对
        a = a[g.permutation(n)]
    head = InverseDynamicsHead(feat_dim, action_dim=action_dim).to(f_t.device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    xtr = torch.cat([f_t[tr], f_1[tr]], dim=-1)
    ytr = a[tr]
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(head(f_t=xtr[:, :feat_dim],
                                                 f_t1=xtr[:, feat_dim:]), ytr)
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        xv = torch.cat([f_t[va], f_1[va]], dim=-1)
        v = head(xv[:, :feat_dim], xv[:, feat_dim:])
        v_mse = float(torch.nn.functional.mse_loss(v, a[va]).item())
    return float(loss.item()), v_mse


def main():
    ap = argparse.ArgumentParser(description="ID 头探针（冻结特征上的逆动力学可学习性）")
    ap.add_argument("--models", required=True, help="逗号分隔 tag:model_dir")
    ap.add_argument("--data", default=None, help="默认取各模型 meta.json 的 data")
    ap.add_argument("--pairs", type=int, default=384)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seeds", default="0,1,2", help="探针重复种子（取均值±std）")
    ap.add_argument("--out", default="export/e5_eval/probe.md")
    a = ap.parse_args()

    rows = {}
    for spec in [x for x in a.models.split(",") if x]:
        tag, mdir = spec.split(":")[:2]
        meta = json.load(open(os.path.join(mdir, "meta.json"), encoding="utf-8"))
        h5 = a.data or meta.get("data")
        policy, cfg, ck = load_policy(mdir)
        data = SimDemoData(h5, device=DEV,
                           max_state_dim=ck.get("max_state_dim"),
                           tokenizer_max_length=ck.get("tokenizer_max_length"),
                           chunk_size=ck.get("chunk", 50), max_action_dim=2,
                           n_frames=1, with_lidar=False, with_det=True)
        rng = np.random.default_rng(12345)
        f_t, f_1, act = collect_features(policy, data, rng, a.pairs, a.batch)
        feat_dim = int(f_t.shape[-1])
        var = float(act.var(dim=0).mean().item())          # 动作方差 = 哑基线 MSE
        seeds = [int(s) for s in a.seeds.split(",") if s]
        fit, val, rnd = [], [], []
        for s in seeds:
            tr, va = probe_once(f_t, f_1, act, feat_dim, seed=s)
            _, vr = probe_once(f_t, f_1, act, feat_dim, seed=s, shuffle_labels=True)
            fit.append(tr), val.append(va), rnd.append(vr)
        rows[tag] = {"feat_dim": feat_dim, "pairs": int(f_t.shape[0]),
                     "act_var": var,
                     "fit_mse": float(np.mean(fit)), "fit_std": float(np.std(fit)),
                     "val_mse": float(np.mean(val)), "val_std": float(np.std(val)),
                     "rand_mse": float(np.mean(rnd)), "rand_std": float(np.std(rnd))}
        print(f"{tag}: d={feat_dim} 拟合 {rows[tag]['fit_mse']:.5f} "
              f"验证 {rows[tag]['val_mse']:.5f}±{rows[tag]['val_std']:.5f} "
              f"随机标签 {rows[tag]['rand_mse']:.5f} 动作方差 {var:.5f}", flush=True)
        del policy, data
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    lines = ["# ID 头探针（冻结特征 + 从头训头，3 探针种子均值±std）", "",
             "| 模型 | 特征维 | 验证 MSE | 随机标签 MSE | 动作方差（哑基线） |",
             "|---|---|---|---|---|"]
    for tag, r in rows.items():
        lines.append(f"| {tag} | {r['feat_dim']} | {r['val_mse']:.5f}±{r['val_std']:.5f} "
                     f"| {r['rand_mse']:.5f}±{r['rand_std']:.5f} | {r['act_var']:.5f} |")
    lines += ["", "> 读法：验证 MSE ≪ 随机标签 MSE ≈ 动作方差 ⇒ 头真的学到了逆动力学；",
              "> idas_v2 的验证 MSE 低于 none ⇒ 辅助监督让特征更可读（学了但没帮主任务）。",
              ""]
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n已写出 {a.out}")


if __name__ == "__main__":
    main()
