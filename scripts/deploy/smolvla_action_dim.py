"""动作空间重构：max_action_dim 32 → 2 的等价性分析（主创新 A1 的落点）。

背景：
    官方基座输出 6 维（SO-100/101 双机械臂），但内部统一 padding 到 max_action_dim=32。
    本课题是**无云台的差速底盘**，目标动作是 2 维 (v, ω)。

两条路线：
    (a) 直接截断：把 action_in_proj 的输入维度与 action_out_proj 的输出维度从 32 裁到 2
        —— 工程量极小，但 **不是无损变换**：action_in_proj 是 Linear(32→720)，
        会**跨动作维度混合**，截断输入必然改变输出。
    (b) 保留 32 维、只在后处理取前 2 维 —— 无损，但保留了 padding 的无效计算。

本脚本量化 (a) 的扰动幅度，判断它能否作为微调的初始点。

注意：前一轮已实测确认 **max_action_dim 裁剪几乎不省延迟**（只影响两个极小 Linear，
后缀序列长度与维度都与它无关），因此本项的定位是**正确性与可用性**，不是加速。

用法：
    cd /3240704007/sci/VLA
    CUDA_VISIBLE_DEVICES=5 .venv_smolvla/bin/python smolvla_action_dim.py
"""

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
import torch.nn as nn

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

CKPT = "models/smolvla_base"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NEW_DIM = 2


def truncate_action_dim(policy, new_dim: int) -> None:
    fm = policy.model
    w_in, b_in = fm.action_in_proj.weight, fm.action_in_proj.bias
    w_out, b_out = fm.action_out_proj.weight, fm.action_out_proj.bias

    new_in = nn.Linear(new_dim, w_in.shape[0], bias=b_in is not None).to(w_in.device, w_in.dtype)
    new_out = nn.Linear(w_out.shape[1], new_dim, bias=b_out is not None).to(w_out.device, w_out.dtype)
    with torch.no_grad():
        new_in.weight.copy_(w_in[:, :new_dim])
        if b_in is not None:
            new_in.bias.copy_(b_in)
        new_out.weight.copy_(w_out[:new_dim, :])
        if b_out is not None:
            new_out.bias.copy_(b_out[:new_dim])

    fm.action_in_proj, fm.action_out_proj = new_in, new_out
    policy.config.max_action_dim = new_dim


def main() -> None:
    print(f"设备: {DEV}")
    policy = SmolVLAPolicy.from_pretrained(CKPT).to(DEV).eval()
    cfg, fm = policy.config, policy.model

    pad = cfg.resize_imgs_with_padding
    h = w = pad[1] if pad else 512
    image = torch.randn(1, 3, h, w, device=DEV)
    img_mask = torch.ones(1, dtype=torch.bool, device=DEV)
    lang = torch.randint(0, 1000, (1, cfg.tokenizer_max_length), device=DEV)
    lang_m = torch.ones(1, cfg.tokenizer_max_length, dtype=torch.bool, device=DEV)
    state = torch.randn(1, cfg.max_state_dim, device=DEV)
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, device=DEV)

    print("\n=== 权重结构 ===")
    w_in = fm.action_in_proj.weight.detach().float()   # (720, 32)
    w_out = fm.action_out_proj.weight.detach().float()  # (32, 720)
    share_in = w_in[:, :NEW_DIM].norm() / w_in.norm()
    share_out = w_out[:NEW_DIM, :].norm() / w_out.norm()
    print(f"  action_in_proj  {tuple(w_in.shape)}  前 {NEW_DIM} 维权重占比 {share_in * 100:.1f}%")
    print(f"  action_out_proj {tuple(w_out.shape)} 前 {NEW_DIM} 维权重占比 {share_out * 100:.1f}%")

    print("\n=== 截断前后的输出差异 ===")
    # 先用原模型（32 维）算参考输出
    out32 = {}
    for ns in (10, 1):
        cfg.num_steps = ns
        with torch.no_grad():
            out32[ns] = fm.sample_actions(
                [image], [img_mask], lang, lang_m, state, noise=noise.clone()
            )

    # 原地截断到 2 维；此后模型只看到噪声的前 2 维（即部署时的真实情形）
    truncate_action_dim(policy, NEW_DIM)
    print(f"  截断完成，max_action_dim = {cfg.max_action_dim}")

    print(f"  {'num_steps':>10s} {'mean|Δ|':>12s} {'max|Δ|':>12s} {'相对误差':>10s}")
    for ns in (10, 1):
        cfg.num_steps = ns
        with torch.no_grad():
            out2 = fm.sample_actions(
                [image], [img_mask], lang, lang_m, state, noise=noise[..., :NEW_DIM].clone()
            )
        ref = out32[ns][..., :NEW_DIM]
        d = (out2 - ref).abs()
        scale = ref.abs().mean().item()
        print(f"  {ns:>10d} {d.mean().item():>12.4f} {d.max().item():>12.4f} "
              f"{100 * d.mean().item() / max(scale, 1e-9):>9.2f}%")

    print("\n=== 结论 ===")
    print("  截断会改变输出（action_in_proj 跨动作维度混合），因此 **不能当作等价变换**。")
    print("  正确定位：截断后的模型是「动作空间已对齐到 2 维差速底盘」的**微调初始点**，")
    print("  必须经目标域微调后才能使用；不应在论文中宣称其为无损改造。")
    print(f"  若追求严格无损：保留 max_action_dim=32，仅在后处理取前 {NEW_DIM} 维")
    print("  （代价是保留 padding 的无效参数，但实测这部分几乎不影响延迟）。")


if __name__ == "__main__":
    main()
