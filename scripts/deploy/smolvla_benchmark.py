"""SmolVLA 组件级耗时基准。

E6 闸门的第一组数据：搞清时间花在哪（视觉编码器 / prefix prefill / 去噪循环），
据此判定优化优先级（去噪步数蒸馏 vs 动作维度裁剪 vs 降分辨率）。

【重要】本脚本运行在 RTX 4090 上，产出的是「相对占比」与「优化取舍依据」，
**不是 RDK-X3 的端侧实测值**，不可写进论文当作部署结论。

用法：
    cd /3240704007/sci/VLA
    CUDA_VISIBLE_DEVICES=5 .venv_smolvla/bin/python smolvla_benchmark.py
"""

import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

CKPT = "models/smolvla_base"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP = 20

RESULTS: list[tuple[str, float, float, float]] = []


def timeit(name: str, fn, runs: int = 100) -> float:
    """返回 p50 毫秒，并把 (name, mean, p50, p95) 记录到 RESULTS。"""
    with torch.no_grad():
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ts)
    mean, p50, p95 = a.mean(), float(np.percentile(a, 50)), float(np.percentile(a, 95))
    RESULTS.append((name, mean, p50, p95))
    print(f"  {name:38s} mean {mean:8.2f}  p50 {p50:8.2f}  p95 {p95:8.2f} ms")
    return p50


def main() -> None:
    print(f"设备: {DEV} | {torch.cuda.get_device_name(0)}")
    policy = SmolVLAPolicy.from_pretrained(CKPT).to(DEV).eval()
    cfg, fm = policy.config, policy.model

    h = w = cfg.resize_imgs_with_padding[1]
    images = [torch.randn(1, 3, h, w, device=DEV)]
    img_masks = [torch.ones(1, dtype=torch.bool, device=DEV)]
    lang_tokens = torch.randint(0, 1000, (1, cfg.tokenizer_max_length), device=DEV)
    lang_masks = torch.ones(1, cfg.tokenizer_max_length, dtype=torch.bool, device=DEV)
    state = torch.randn(1, cfg.max_state_dim, device=DEV)

    torch.cuda.reset_peak_memory_stats()

    print("\n=== 1. 组件耗时 ===")

    # a) 视觉编码器单独（SigLIP，512x512 → 约 1024 patch）
    timeit("SigLIP 视觉编码器 (512x512)", lambda: fm.vlm_with_expert.embed_image(images[0]))

    # b) 完整 prefix 嵌入（视觉 + 语言 + 状态）
    timeit("embed_prefix 全流程", lambda: fm.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state))

    with torch.no_grad():
        prefix_embs, prefix_pad_masks, prefix_att_masks = fm.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        att = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        pos = torch.cumsum(prefix_pad_masks, dim=1) - 1

    # c) prefix prefill（16 层，填 KV cache）
    def run_prefill():
        return fm.vlm_with_expert.forward(
            attention_mask=att,
            position_ids=pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

    timeit("Prefix prefill (16 层 + 填 KV)", run_prefill, runs=100)

    # d) 单次 denoise_step（走真实路径，cache 复用 + crop）
    _, pkv = run_prefill()
    pairs = [(layer.keys, layer.values) for layer in pkv.layers]
    cache = DynamicCache(ddp_cache_data=pairs)
    x_t = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, device=DEV)
    ts = torch.tensor([0.5], device=DEV)

    def run_denoise():
        return fm.denoise_step(prefix_pad_masks=prefix_pad_masks, past_key_values=cache, x_t=x_t, timestep=ts)

    timeit(f"单次 denoise_step (chunk={cfg.chunk_size})", run_denoise, runs=100)

    print("\n=== 2. 端到端 sample_actions ===")

    def run_e2e():
        return fm.sample_actions(images, img_masks, lang_tokens, lang_masks, state)

    p50_full = timeit(f"端到端 ({cfg.num_steps} 步去噪)", run_e2e, runs=30)

    orig_steps = cfg.num_steps
    cfg.num_steps = 1
    p50_1 = timeit("端到端 (1 步去噪)", run_e2e, runs=30)
    cfg.num_steps = orig_steps

    print("\n=== 3. chunk_size 对去噪步的影响 ===")
    orig_chunk = cfg.chunk_size
    for c in (50, 20, 10, 5, 1):
        cfg.chunk_size = c
        xc = torch.randn(1, c, cfg.max_action_dim, device=DEV)
        timeit(
            f"denoise_step (chunk={c})",
            lambda xc=xc: fm.denoise_step(
                prefix_pad_masks=prefix_pad_masks, past_key_values=cache, x_t=xc, timestep=ts
            ),
            runs=60,
        )
    cfg.chunk_size = orig_chunk

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n=== 4. 显存 ===\n  峰值分配: {peak:.2f} GiB")

    print("\n=== 汇总（p50, ms）===")
    for name, _m, p50, _p in RESULTS:
        print(f"  {name:38s} {p50:8.2f}")

    print("\n=== 闸门相关的关键结论 ===")
    print(f"  10 步 → 1 步去噪可省: {p50_full - p50_1:.1f} ms ({100 * (p50_full - p50_1) / p50_full:.0f}%)")
    print("  注：以上为 4090 数据，仅用于优化取舍，非端侧结论。")
    print("  将 chunk_size 由 50 缩短会减少专家序列长度，但降低反应性，需在论文中权衡。")


if __name__ == "__main__":
    main()
