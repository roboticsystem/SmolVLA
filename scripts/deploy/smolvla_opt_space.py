"""主创新 B 的优化空间量化（E6 后续 / 5.1 节）。

逐个自变量实测其对延迟与参数量的真实收益，给出「优化杠杆」的排序：
    ① 动作维度 max_action_dim:  32 → 2
    ② 动作专家层数 num_vlm_layers(专家侧): 16 → 4
    ③ 前缀 VLM 层数: 16 → 4（影响 prefix prefill）
    ④ 去噪步数 num_steps: 10 → 1（另见 smolvla_distill_steps.py）

【重要】本脚本同时用于**证伪一个先前判断**：曾推测「max_action_dim 32→2 可削减
专家计算量」，但按结构分析，max_action_dim 只影响 action_in_proj / action_out_proj
两个极小的 Linear，后缀序列长度与维度都与它无关，因此预期收益≈0。此处以实测确认。

用法：
    cd /3240704007/sci/VLA
    CUDA_VISIBLE_DEVICES=5 .venv_smolvla/bin/python smolvla_opt_space.py
"""

import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

CKPT = "models/smolvla_base"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP = 10

ROWS: list[tuple] = []


def bench(name, fn, runs=60):
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
    return float(np.percentile(ts, 50))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def shrink_action_dim(policy, new_dim: int):
    """把动作头从 max_action_dim 维裁到 new_dim 维（权重切片，非无损，见脚本说明）。"""
    fm = policy.model
    w_in, b_in = fm.action_in_proj.weight, fm.action_in_proj.bias
    w_out, b_out = fm.action_out_proj.weight, fm.action_out_proj.bias
    print(f"    裁前: action_in_proj {tuple(w_in.shape)} / action_out_proj {tuple(w_out.shape)}")

    new_in = nn.Linear(new_dim, w_in.shape[0], bias=b_in is not None).to(w_in.device, w_in.dtype)
    new_out = nn.Linear(w_out.shape[1], new_dim, bias=b_out is not None).to(w_out.device, w_out.dtype)
    with torch.no_grad():
        new_in.weight.copy_(w_in[:, :new_dim])
        if b_in is not None:
            new_in.bias.copy_(b_in)
        new_out.weight.copy_(w_out[:new_dim, :])
        if b_out is not None:
            new_out.bias.copy_(b_out[:new_dim])

    fm.action_in_proj = new_in
    fm.action_out_proj = new_out
    policy.config.max_action_dim = new_dim
    print(f"    裁后: action_in_proj {tuple(new_in.weight.shape)} / action_out_proj {tuple(new_out.weight.shape)}")


def main() -> None:
    print(f"设备: {DEV}")
    policy = SmolVLAPolicy.from_pretrained(CKPT).to(DEV).eval()
    cfg, fm = policy.config, policy.model

    h = w = cfg.resize_imgs_with_padding[1]
    image = torch.randn(1, 3, h, w, device=DEV)
    img_mask = torch.ones(1, dtype=torch.bool, device=DEV)
    lang = torch.randint(0, 1000, (1, cfg.tokenizer_max_length), device=DEV)
    lang_m = torch.ones(1, cfg.tokenizer_max_length, dtype=torch.bool, device=DEV)
    state = torch.randn(1, cfg.max_state_dim, device=DEV)
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, device=DEV)

    base_params = n_params(policy)
    print(f"\n基线参数量: {base_params / 1e6:.2f} M")

    with torch.no_grad():
        embs, pad, att = fm.embed_prefix([image], [img_mask], lang, lang_m, state=state)
        a2d = make_att_2d_masks(pad, att)
        pos = torch.cumsum(pad, dim=1) - 1
        _, pkv = fm.vlm_with_expert.forward(
            attention_mask=a2d, position_ids=pos, past_key_values=None,
            inputs_embeds=[embs, None], use_cache=True,
        )
    x_t = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, device=DEV)
    ts = torch.tensor([0.5], device=DEV)
    n_layer_full = len(pkv.layers)

    def make_cache(n):
        pairs = [(pkv.layers[i].keys, pkv.layers[i].values) for i in range(n)]
        return DynamicCache(ddp_cache_data=pairs)

    def step_cache(n):
        c = make_cache(n)
        return lambda: fm.denoise_step(prefix_pad_masks=pad, past_key_values=c, x_t=x_t, timestep=ts)

    def e2e():
        return fm.sample_actions([image], [img_mask], lang, lang_m, state)

    print("\n=== 基线 ===")
    t_den = bench("denoise_step(16层)", step_cache(n_layer_full))
    t_e2e = bench("e2e(10步)", e2e, runs=20)
    print(f"  denoise_step {t_den:.2f} ms | e2e(10步) {t_e2e:.2f} ms")
    ROWS.append(("基线 (dim=32, layers=16, steps=10)", 1.0, t_den, t_e2e))

    # ---------- ① 动作维度 ----------
    print("\n=== ① 动作维度 max_action_dim: 32 → 2 ===")
    p_before = n_params(policy)
    shrink_action_dim(policy, 2)
    p_after = n_params(policy)
    cfg2 = policy.config
    x_t2 = torch.randn(1, cfg2.chunk_size, 2, device=DEV)
    c2 = make_cache(n_layer_full)
    t_den_d2 = bench("denoise_step(dim=2)", lambda: fm.denoise_step(
        prefix_pad_masks=pad, past_key_values=c2, x_t=x_t2, timestep=ts))
    t_e2e_d2 = bench("e2e(10步, dim=2)", e2e, runs=20)
    dparam = p_before - p_after
    print(f"  参数量 {p_before/1e6:.2f} M → {p_after/1e6:.2f} M（省 {dparam/1e3:.1f} K，{100*dparam/base_params:.3f}%）")
    print(f"  denoise_step {t_den:.2f} → {t_den_d2:.2f} ms（{(t_den_d2-t_den)/t_den*100:+.1f}%）")
    print(f"  e2e(10步)    {t_e2e:.2f} → {t_e2e_d2:.2f} ms（{(t_e2e_d2-t_e2e)/t_e2e*100:+.1f}%）")
    ROWS.append(("① 动作维度 32→2", p_after / base_params, t_den_d2, t_e2e_d2))

    # ---------- ② 专家层数 ----------
    print("\n=== ② 层数裁剪（同时影响 prefill 与 denoise 循环）===")
    orig_layers = fm.vlm_with_expert.num_vlm_layers
    for n in (16, 12, 8, 4):
        fm.vlm_with_expert.num_vlm_layers = n
        with torch.no_grad():
            _, pkv_n = fm.vlm_with_expert.forward(
                attention_mask=a2d, position_ids=pos, past_key_values=None,
                inputs_embeds=[embs, None], use_cache=True,
            )
        t_pre = bench(f"prefill({n}层)", lambda pkv_n=pkv_n: fm.vlm_with_expert.forward(
            attention_mask=a2d, position_ids=pos, past_key_values=None,
            inputs_embeds=[embs, None], use_cache=True))
        cN = DynamicCache(ddp_cache_data=[(pkv_n.layers[i].keys, pkv_n.layers[i].values) for i in range(n)])
        t_dn = bench(f"denoise_step({n}层)", lambda cN=cN: fm.denoise_step(
            prefix_pad_masks=pad, past_key_values=cN, x_t=x_t2, timestep=ts))
        approx_e2e = t_pre + 10 * t_dn
        print(f"  {n:2d} 层: prefill {t_pre:6.2f} ms | denoise_step {t_dn:6.2f} ms | 估算 e2e {approx_e2e:7.2f} ms")
        ROWS.append((f"② 层数 {n}", None, t_dn, approx_e2e))
    fm.vlm_with_expert.num_vlm_layers = orig_layers

    # ---------- ③ 去噪步数 ----------
    print("\n=== ③ 去噪步数 num_steps ===")
    keep = cfg.num_steps
    for ns in (10, 5, 2, 1):
        cfg.num_steps = ns
        t = bench(f"e2e({ns}步)", e2e, runs=20)
        print(f"  {ns:2d} 步: e2e {t:7.2f} ms")
        ROWS.append((f"③ 去噪 {ns} 步", None, None, t))
    cfg.num_steps = keep

    print("\n=== 优化杠杆排序（相对基线的 e2e 降幅）===")
    ref = t_e2e
    for name, pr, _d, e in ROWS:
        if e is None:
            continue
        drop = (1 - e / ref) * 100
        pstr = f"params x{pr:.4f}" if pr else ""
        print(f"  {name:34s} e2e {e:7.2f} ms  降幅 {drop:+6.1f}%  {pstr}")


if __name__ == "__main__":
    main()
