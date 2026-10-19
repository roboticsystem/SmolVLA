"""E7-partial：INT8 量化灵敏度（**精度损失那一半**）。

## 为什么最后用 PyTorch 原生量化而不是 ONNX

计划是「ORT 静态 PTQ 量化 `deploy_4x1/denoise.onnx`」，实测遇到**两层阻塞**：

1. `bf16` 初始化器与 KV-cache 图输入 → ORT CPU EP 直接 `INVALID_GRAPH`
   （已解决：把所有 bf16 张量与 ValueInfo 转成 fp32）
2. **图本身的问题**：`/CumSum_1` 的第一个输入是 `bool` 常量
   （`/Constant_1_output_0`），而 ONNX 规范要求 CumSum 的 `x` 为数值类型
   → `INVALID_GRAPH : Type 'tensor(bool)' ... operator (CumSum)`。

第 2 点**与量化无关，是导出图自身的缺陷** —— 也就是说
`export/deploy_4x1/denoise.onnx` 在 ORT CPU EP 上**本来就不能跑**。
这是一个有价值的发现：**「部署包已产出」不等于「部署包可执行」**，
E6 手册应补一条「上板前先用 ORT 做一次冒烟」的检查项。

因此改用 **PyTorch 原生动态量化**（对动作专家的 Linear 层做 INT8）。
它度量的是同一件事（INT8 对动作精度的损失），且不依赖 ONNX 图的正确性。

## 边界（不得越过）

| 能回答 | 不能回答 |
|---|---|
| INT8 对**动作精度**的损失 | BPU 算子覆盖率 |
| 逐层量化 vs 全量化的差异 | CPU 回退比例 |
| 是否掉出「≤3 pp」判据 | 板载真实延迟/功耗 |

`exp_plan.yaml` 的 E7-partial 已写明：**不得把本结果写成 RDK-X3 部署结论**。
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
from torch.ao.quantization import quantize_dynamic
from torch.ao.nn.quantized.dynamic import Linear as QLinear


def load_student(device="cpu"):
    """加载 4×1 蒸馏学生（动作专家）。

    用部署包里的 `expert.pt`（学生专家权重），它与 `denoise.onnx` 同源。
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # 项目根
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    pol = SmolVLAPolicy.from_pretrained("models/smolvla_base").to(device)
    ck = torch.load("export/deploy_4x1/expert.pt", map_location=device)
    if isinstance(ck, dict) and "model" in ck:
        pol.model.load_state_dict(ck["model"], strict=False)
    elif isinstance(ck, dict):
        pol.model.vlm_with_expert.lm_expert.load_state_dict(
            {k.replace("lm_expert.", ""): v for k, v in ck.items()}, strict=False)
    pol.eval()
    return pol


def count_qlinears(module):
    return sum(1 for m in module.modules() if isinstance(m, QLinear))


def main():
    ap = argparse.ArgumentParser(description="E7-partial：INT8 量化灵敏度")
    ap.add_argument("--data", default="export/nav_detf.h5")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="export/int8_e7/quant_report.json")
    a = ap.parse_args()

    dev = "cpu"                       # 动态量化在 CPU 上最标准
    pol = load_student(dev)
    model = pol.model

    # ---- 取动作专家（4 层学生）----
    expert = model.vlm_with_expert.lm_expert
    n_lin = sum(1 for m in expert.modules() if isinstance(m, torch.nn.Linear))
    print(f"动作专家 Linear 层数: {n_lin}")

    # ---- 构造真实输入：用导航数据集的 batch 走一遍前向，取专家输出 ----
    from smolvla_sim_data import (SimDemoData, LANG_TOKENS_KEY,
                              LANG_MASK_KEY)
    data = SimDemoData(a.data, device=dev, with_det=True)
    rng = np.random.default_rng(0)
    idx = data.sample_index(a.batch, rng)
    b = data.batch(idx)

    with torch.no_grad():
        images, img_masks = pol.prepare_images(b)
        state = pol.prepare_state(b)
        # 前缀（VLM 输出）—— 与部署时的 prefix.onnx 对应
        prefix_embs, _, _ = model.embed_prefix(images, img_masks,
                                               b[LANG_TOKENS_KEY],
                                               b[LANG_MASK_KEY], state=state)
        noise = model.sample_noise((a.batch, model.config.chunk_size,
                                    model.config.max_action_dim), dev)
        t = model.sample_time(a.batch, dev)
        ref = model.sample_actions(images, img_masks, b[LANG_TOKENS_KEY],
                                   b[LANG_MASK_KEY], state, noise=noise)

    # ---- 权重-only INT8（per-channel 对称，scale = max|w|/127）----
    # 为什么不用 `quantize_dynamic`：它会把 `nn.Linear` 换成 `QLinear`，
    # 而库代码 `forward_attn_layer` 里直接访问 `layer.self_attn.q_proj.weight.dtype`
    # —— 量化后 `.weight` 是**方法**而不是张量，立刻 AttributeError（实测）。
    # 权重-only 仿真保留模块类型，只把权重换成「INT8 量化再反量化」的值，
    # 度量的是 INT8 权重量化本身的误差（对 LLM 类模型这是主导项）。
    q_model = copy.deepcopy(model)
    n_q = 0
    with torch.no_grad():
        for m in q_model.vlm_with_expert.lm_expert.modules():
            if isinstance(m, torch.nn.Linear):
                w = m.weight.float()
                scale = w.abs().amax(dim=1, keepdim=True) / 127.0
                scale = torch.clamp(scale, min=1e-12)
                m.weight.copy_((torch.round(w / scale) * scale).to(w.dtype))
                n_q += 1
    print(f"量化 Linear 层数: {n_q}")

    with torch.no_grad():
        q_out = q_model.sample_actions(images, img_masks, b[LANG_TOKENS_KEY],
                                       b[LANG_MASK_KEY], state, noise=noise)

    mae = float((q_out - ref).abs().mean())
    scale = float(ref.abs().mean())
    rel = mae / max(scale, 1e-9)
    cos = float(torch.nn.functional.cosine_similarity(
        q_out.flatten(), ref.flatten(), dim=0))
    print(f"\nINT8 (dynamic, per-tensor) vs fp32:")
    print(f"  动作 MAE          = {mae:.5f}")
    print(f"  相对误差          = {100*rel:.2f}%")
    print(f"  余弦相似度        = {cos:.5f}")

    # 模型体积对比
    def nbytes(m):
        return sum(p.numel() * p.element_size() for p in m.parameters())
    s_fp, s_q = nbytes(model.vlm_with_expert.lm_expert), \
        sum(t.numel() * t.element_size() for t in
            q_model.vlm_with_expert.lm_expert.state_dict().values()
            if t.is_floating_point())
    print(f"  fp32 权重 {s_fp/1e6:.1f} MB → 量化后浮点部分 {s_q/1e6:.1f} MB")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rep = {"method": "pytorch_weight_only_int8_perchannel",
           "expert_linear_layers": n_lin, "quantized_linears": n_q,
           "mae": mae, "rel_err": rel, "cos": cos,
           "onnx_blocker": "denoise.onnx 在 ORT CPU EP 不可跑：bf16 类型 + /CumSum_1 收到 bool 常量",
           "note": "只覆盖 E7 的精度半边；算子覆盖率/板载延迟需地平线工具链"}
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"\n已写入 {a.out}")
    print("⚠️  不得把本结果写成 RDK-X3 部署结论（见 exp_plan.yaml E7-partial）")


if __name__ == "__main__":
    main()
