"""E5：IDAS（逆动力学辅助监督）在 SmolVLA 上的移植。

## 来源与定位

IDAS 是已投稿小论文的方法（原为 JEPA 潜空间世界模型设计）。按大纲 §0 的定位，
它作为**主创新 A 的 4.4 节**纳入学位论文，不单独成章。

## 迁移的关键改动

原 IDAS 在 JEPA 潜空间里做「预测下一条潜状态」。SmolVLA 是
「VLM 主干 + flow matching 动作专家」，没有显式的潜空间状态序列，所以改成
**在 VLM 前缀特征上接一个逆动力学头**：

    head( [stop-grad(f_t), f_{t+1}] ) → â ∈ R²      （两帧之间执行的动作）
    L_idas = MSE(â, a_{t→t+1})
    L_total = L_flow + λ_id · L_idas

## 三个变体（大纲 §5 E5 的自变量 ①）

| 变体 | 说明 |
|---|---|
| `none` | 基线，λ_id = 0 |
| `idas_v1` | 辅助头直接吃两帧特征，**无停止梯度**（梯度会流回主干） |
| `idas_v2` | **停止梯度 + 预测器监督**（本文方法）：`f_t` 截断梯度，只训头 |
| `random_ctrl` | **随机对照**：同结构同 λ，但标签换成随机向量 —— 排除「多一个损失项本身就是正则」 |

> `random_ctrl` 很重要：任何额外损失都会改变优化动态（相当于正则化），
> 不设这个对照就无法把「IDAS 学到了动力学」与「多了一项正则」分开。

## 为什么用「底边接地法」不用「按身高」……不对，为什么用相邻帧

逆动力学的定义就是「给定前后两帧求动作」。数据集里 `images` 存了整段回合的
所有帧，`ep_offset` 给出回合边界，因此 `(t, t+1)` 对在数据里是现成的，
**不需要任何新的数据采集**。

## 与主创新的区分（避免自我重复）

主创新 A 的主线是「观测完备性」（把感知输出显式给模型，§4.12.4）；
IDAS 是**辅助监督**，动的是损失函数，不动输入。两者正交，
论文里 4.4 节应只谈辅助监督、不重复 4.1–4.3 的内容。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# ------------------------------------------------------------------ 变体
VARIANTS = ("none", "idas_v1", "idas_v2", "random_ctrl")


class InverseDynamicsHead(nn.Module):
    """逆动力学头：`[f_t, f_{t+1}] → â`。

    输入维度用 `2 * d`（两帧拼接），`d` 是**单帧特征维**。
    结构刻意保持很小（一层隐层）：它是辅助监督，不该与主任务争容量 ——
    若头太大，主任务会通过它把容量花在「重建动力学」上，反而干扰 flow matching。
    """

    def __init__(self, feat_dim: int, action_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * feat_dim, hidden), nn.GELU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, f_t, f_t1):
        return self.net(torch.cat([f_t, f_t1], dim=-1))


class PrefixFeatureCapture:
    """从 `VLAFlowMatching.forward` 里**截获 VLM 前缀特征**。

    ## 为什么不能用 `register_forward_hook`

    `modeling_smolvla.py` 的 `forward` 里写的是
    `self.vlm_with_expert.forward(...)` —— **直接调用方法**而不是 `module(...)`,
    所以 PyTorch 的 forward hook **根本不会触发**（实测 `cap` 为空）。
    因此改用**实例属性包装**：把 `forward` 换成一个先调原函数、再截获返回值的版本。

    ## 为什么取全部前缀 token 的均值

    前缀 = 图像 token + 语言 token + 状态 token。指令在回合内不变，
    所以帧间差异几乎全部来自视觉与状态 —— 正是逆动力学需要的信息。
    """

    def __init__(self, model):
        self.prefix = None
        self._mdl = model.vlm_with_expert
        self._orig = self._mdl.forward

        def _wrapped(*a, **kw):
            out = self._orig(*a, **kw)
            try:
                self.prefix = out[0][0]
            except Exception:
                self.prefix = None
            return out

        self._mdl.forward = _wrapped

    def remove(self):
        self._mdl.forward = self._orig


def frame_feature(prefix_out):
    """前缀特征 → 单帧向量：对所有 token 做掩码均值池化。

    返回 `(B, d)`。不做任何投影 —— 特征维就是 `prefix_out.shape[-1]`。
    """
    if prefix_out is None:
        raise RuntimeError("钩子没有截获到前缀特征（检查 vlm_with_expert 的返回结构）")
    return prefix_out.float().mean(dim=1)


# ------------------------------------------------------------------ 采样
def next_frame_indices(idx, ep_off, chunk=50):
    """每个样本的「下一帧」索引；无效样本置 -1 并在 `ok` 里标 False。

    两个必须满足的条件（缺一不可，都踩过）：
    ① `t+1` 必须仍在**同一回合**内 —— 跨回合的相邻帧是两个不同任务，
       其间的「动作」没有物理意义；
    ② `t+1` 之后还要能**切满一个动作 chunk** —— `data.batch()` 用
       `actions[i:i+chunk]` 取块，靠近回合末尾时切片会变短，
       `np.stack` 直接抛 `all input arrays must have the same shape`
       （实测训练在 step 800 崩掉，且 checkpoint 因此没保存）。
    """
    ep_off = np.asarray(ep_off)
    idx = np.asarray(idx)
    ep_i = np.searchsorted(ep_off, idx, side="right") - 1
    nxt = idx + 1
    end = ep_off[np.clip(ep_i + 1, 0, len(ep_off) - 1)]
    ok = (nxt < end) & (nxt + int(chunk) <= end)
    return np.where(ok, nxt, -1), ok


def action_between(data, i_t, i_t1):
    """`t → t+1` 之间执行的动作（物理量，未归一化）。"""
    return data.actions[np.asarray(i_t1)]


# ------------------------------------------------------------------ 损失
def idas_loss(head, f_t, f_t1, a_gt, variant, lam):
    """按变体计算辅助损失。

    - `idas_v1`：两帧特征都带梯度（会更新主干）
    - `idas_v2`：`f_t` **截断梯度**（本文方法）—— 主干不为辅助头「改变自己」，
      只有头在学。这样辅助任务不会与主任务争抢主干的表达能力，
      与「预测器 + 目标编码器停止梯度」的标准做法一致。
    - `random_ctrl`：标签换成与 `a_gt` 同分布的随机向量，结构与 λ 完全相同
    """
    if variant == "idas_v2":
        f_t = f_t.detach()
    pred = head(f_t, f_t1)
    if variant == "random_ctrl":
        a_gt = torch.randn_like(a_gt) * a_gt.std() + a_gt.mean()
    return lam * torch.nn.functional.mse_loss(pred, a_gt)


def label_mask(p, size, rng):
    """动作标签比例 p：**主损失与辅助损失同时缺失**（大纲 §5 E5 的下界实验）。

    返回 bool 掩码（True = 有标签）。注意是**掩蔽标签**而不是丢数据 ——
    丢数据会同时改变 batch 的组成，两个效应就分不开了（大纲 §7.2 同一约定）。
    """
    if p >= 1.0:
        return np.ones(size, dtype=bool)
    return rng.random(size) < p
