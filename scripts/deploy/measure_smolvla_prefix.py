"""实测 SmolVLA 的 prefix 长度与 KV cache 体积。

这是 E6 可行性闸门的第一步：得到真实的 prefix 长度，
后续所有延迟 / 内存 / KV 张量化预算都以它为准，不能靠推算。

用法：
    cd /3240704007/sci/VLA
    HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
        .venv_smolvla/bin/python measure_smolvla_prefix.py
"""

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch

from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

CKPT = "models/smolvla_base"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG_KEYS = [
    "chunk_size",
    "n_action_steps",
    "num_steps",
    "max_state_dim",
    "max_action_dim",
    "resize_imgs_with_padding",
    "tokenizer_max_length",
    "use_cache",
    "freeze_vision_encoder",
    "train_expert_only",
    "attention_mode",
    "num_vlm_layers",
]


def main() -> None:
    print(f"设备: {DEV}")
    policy = SmolVLAPolicy.from_pretrained(CKPT).to(DEV).eval()
    cfg = policy.config
    fm = policy.model

    print("\n=== 配置 ===")
    for k in CONFIG_KEYS:
        print(f"  {k:26s} = {getattr(cfg, k, '<缺失>')}")

    # 本体改造后只有 1 路相机（C70/C100）
    pad = cfg.resize_imgs_with_padding
    h = w = pad[1] if pad else 512
    images = [torch.randn(1, 3, h, w, device=DEV)]
    img_masks = [torch.ones(1, dtype=torch.bool, device=DEV)]
    lang_tokens = torch.randint(0, 1000, (1, cfg.tokenizer_max_length), device=DEV)
    lang_masks = torch.ones(1, cfg.tokenizer_max_length, dtype=torch.bool, device=DEV)
    state = torch.randn(1, cfg.max_state_dim, device=DEV)

    with torch.no_grad():
        prefix_embs, prefix_pad_masks, prefix_att_masks = fm.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

    prefix_len = int(prefix_pad_masks.sum())
    hidden = prefix_embs.shape[-1]
    print("\n=== prefix ===")
    print(f"  主干输入图像尺寸   : {h}x{w}")
    print(f"  prefix 长度（token）: {prefix_len}")
    print(f"  隐藏维度           : {hidden}")
    n_lang = int(lang_masks.sum())
    n_state = 1  # embed_prefix 中 state_proj 产生 1 个 token
    print(f"  语言 token 数      : {n_lang}")
    print(f"  状态 token 数      : {n_state}")
    print(f"  → 视觉 token 数    : {prefix_len - n_lang - n_state}")

    # 实测 KV cache：这是图 B 的输入张量，端侧内存的主要来源
    att = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
    with torch.no_grad():
        _, pkv = fm.vlm_with_expert.forward(
            attention_mask=att,
            position_ids=pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
    # 注意：transformers 5.5.4 已移除 DynamicCache.to_legacy_cache() / from_legacy_cache()，
    # 改为通过 DynamicCache.layers[i].keys / .values 访问。
    layers = pkv.layers
    n_layers = len(layers)
    k0, v0 = layers[0].keys, layers[0].values
    nbytes = sum(t.numel() * t.element_size() for lyr in layers for t in (lyr.keys, lyr.values))

    print("\n=== KV cache（图 B 的输入）===")
    print(f"  层数               : {n_layers}")
    print(f"  每层 K 形状        : {tuple(k0.shape)}")
    print(f"  每层 V 形状        : {tuple(v0.shape)}")
    print(f"  dtype              : {k0.dtype}")
    print(f"  单样本体积         : {nbytes / 1024 / 1024:.1f} MiB")

    print("\n=== 端侧含义 ===")
    print(f"  去噪步数 num_steps = {cfg.num_steps}")
    print(f"  → 图 B 需调用       : {cfg.num_steps} 次")
    print(f"  → KV 需常驻内存     : {nbytes / 1024 / 1024:.1f} MiB（1 路相机，fp32 基准）")


if __name__ == "__main__":
    main()
