"""SmolVLA 推理图拆分与 ONNX 导出（E6 闸门 Step 2）。

把 SmolVLA 的推理拆成两张图，使视觉语言主干只前向一次：
    图 A（前缀图，跑 1 次）：图像 + 语言 + 状态 → KV 张量
    图 B（去噪图，跑 num_steps 次）：KV + x_t + timestep → v_t

用法：
    cd /3240704007/sci/VLA
    CUDA_VISIBLE_DEVICES=5 .venv_smolvla/bin/python smolvla_export_onnx.py
"""

import json
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
OUT = "export"
# 实测结论：两张图对 opset 的下限不同，且图 A 必须关闭常量折叠。
#   图 A（含视觉塔 + VLM）：
#       · 用了 aten::scaled_dot_product_attention，该算子自 opset 14 才被支持
#         → 最低 opset = 14（实测 opset 11 报 UnsupportedOperatorError）
#       · do_constant_folding=True 会触发 "ScalarType ComplexDouble" 错误
#         （来自 LayerNorm 分解中的 Pow/Sqrt 在折叠时产生复数常量）
#         → 必须 do_constant_folding=False
#   图 B（仅有动作专家）：不含 SDPA，可低至 opset 11，常量折叠正常。
#
# 这一点直接关系到地平线工具链的兼容性（OE 传统上只支持 opset 10/11），
# 必须写进论文的部署讨论。
OPSET_PREFIX = 14
OPSET_DENOISE = 11
CF_PREFIX = False  # 图 A 必须关闭常量折叠
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- 图定义
class PrefixGraph(nn.Module):
    """图 A：图像 + 语言 + 状态 → 展平的 KV 张量。整图只跑 1 次。"""

    def __init__(self, fm):
        super().__init__()
        self.fm = fm

    def forward(self, image, img_mask, lang_tokens, lang_masks, state):
        embs, pad, att = self.fm.embed_prefix(
            [image], [img_mask], lang_tokens, lang_masks, state=state
        )
        a = make_att_2d_masks(pad, att)
        pos = torch.cumsum(pad, dim=1) - 1
        _, pkv = self.fm.vlm_with_expert.forward(
            attention_mask=a,
            position_ids=pos,
            past_key_values=None,
            inputs_embeds=[embs, None],
            use_cache=True,
        )
        flat = []
        for layer in pkv.layers:
            flat += [layer.keys, layer.values]
        return tuple(flat)


class DenoiseGraph(nn.Module):
    """图 B：KV + x_t + timestep → v_t。按 num_steps 重复调用。

    等价于 VLAFlowMatching.denoise_step，但省掉了末尾的 past_key_values.crop()
    （每步都用同一份 KV 重建 cache，crop 的副作用无需保留）。
    """

    def __init__(self, fm, num_layers):
        super().__init__()
        self.fm = fm
        self.num_layers = num_layers

    def forward(self, x_t, timestep, prefix_pad_masks, *kv):
        pairs = [(kv[2 * i], kv[2 * i + 1]) for i in range(self.num_layers)]
        return self.denoise(x_t, timestep, prefix_pad_masks, DynamicCache(ddp_cache_data=pairs))

    def denoise(self, x_t, timestep, prefix_pad_masks, cache):
        fm = self.fm
        suffix_embs, suffix_pad_masks, suffix_att_masks = fm.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outs, _ = fm.vlm_with_expert.forward(
            attention_mask=full_att,
            position_ids=position_ids,
            past_key_values=cache,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
        )
        suffix_out = outs[1][:, -fm.config.chunk_size :]
        return fm.action_out_proj(suffix_out.to(dtype=torch.float32))


# ---------------------------------------------------------------- 校验
def flatten_cache(cache):
    flat = []
    for layer in cache.layers:
        flat += [layer.keys, layer.values]
    return flat


def compare(name, a, b, tol=1e-4):
    a = a.detach().float().cpu().numpy()
    b = b.detach().float().cpu().numpy()
    max_abs = float(np.abs(a - b).max())
    denom = np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel())
    cos = float((a.ravel() @ b.ravel()) / denom) if denom > 0 else float("nan")
    ok = max_abs <= tol and cos >= 0.9999
    print(f"  {'PASS' if ok else 'FAIL'} {name:44s} max|Δ|={max_abs:.3e}  cos={cos:.6f}")
    return ok


# ---------------------------------------------------------------- 视觉塔 mask 规避
class VisionMaskPatch:
    """把视觉塔内 create_bidirectional_mask 的调用替换为 None。

    实测（spy 统计）：该函数在整个推理中**只被调用 1 次**，来自
    `config.model_type == "smolvlm_vision"`，mask 为全 True 的 (1, 1024)。
    全有效 mask 等价于无 mask，返回 None 在数学上完全等价；
    同时绕开 transformers 5.5.4 中 `q_length` 被当作 tensor 传入的 BC 分支
    （旧版 TorchScript 导出器无法追踪该分支，报 `IndexError`）。

    等价性由随后的 onnxruntime 数值校验确认。
    """

    def __enter__(self):
        import transformers.masking_utils as mu
        import transformers.models.smolvlm.modeling_smolvlm as msv

        orig = mu.create_bidirectional_mask
        self.count = 0

        def patched(config=None, inputs_embeds=None, attention_mask=None, **kw):
            if getattr(config, "model_type", "") == "smolvlm_vision":
                self.count += 1
                return None
            return orig(config=config, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kw)

        self.saved = []
        for mod in (mu, msv):
            if hasattr(mod, "create_bidirectional_mask"):
                self.saved.append((mod, mod.create_bidirectional_mask))
                mod.create_bidirectional_mask = patched
        return self

    def __exit__(self, *exc):
        for mod, fn in self.saved:
            mod.create_bidirectional_mask = fn
        return False


def verify_with_ort(onnx_path, feed, ref_outputs):
    """用 onnxruntime 跑导出的图，与 PyTorch 参考输出比对，返回最大绝对误差。"""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    got = sess.run(None, feed)
    worst = 0.0
    for g, r in zip(got, ref_outputs):
        g = np.asarray(g, dtype=np.float32)
        r = np.asarray(r, dtype=np.float32)
        worst = max(worst, float(np.abs(g - r).max()))
    return worst


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    print(f"设备: {DEV}")
    policy = SmolVLAPolicy.from_pretrained(CKPT).to(DEV).eval()
    cfg, fm = policy.config, policy.model

    h = w = cfg.resize_imgs_with_padding[1]
    image = torch.randn(1, 3, h, w, device=DEV)
    img_mask = torch.ones(1, dtype=torch.bool, device=DEV)
    lang_tokens = torch.randint(0, 1000, (1, cfg.tokenizer_max_length), device=DEV)
    lang_masks = torch.ones(1, cfg.tokenizer_max_length, dtype=torch.bool, device=DEV)
    state = torch.randn(1, cfg.max_state_dim, device=DEV)
    x_t = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, device=DEV)
    ts = torch.tensor([0.5], device=DEV)
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, device=DEV)

    # 原始参考量
    with torch.no_grad():
        embs, pad, att = fm.embed_prefix([image], [img_mask], lang_tokens, lang_masks, state=state)
        a2d = make_att_2d_masks(pad, att)
        pos = torch.cumsum(pad, dim=1) - 1
        _, pkv_orig = fm.vlm_with_expert.forward(
            attention_mask=a2d, position_ids=pos, past_key_values=None,
            inputs_embeds=[embs, None], use_cache=True,
        )
        kv_orig = flatten_cache(pkv_orig)
        v_orig = fm.denoise_step(
            prefix_pad_masks=pad,
            past_key_values=DynamicCache(ddp_cache_data=[(l.keys, l.values) for l in pkv_orig.layers]),
            x_t=x_t, timestep=ts,
        )

    n_layers = len(pkv_orig.layers)
    pg = PrefixGraph(fm).eval()
    dg = DenoiseGraph(fm, n_layers).eval()

    print("\n=== 第一关：图 A 的 KV == 原模型 KV ===")
    with torch.no_grad():
        kv_new = pg(image, img_mask, lang_tokens, lang_masks, state)
    ok1 = all(compare(f"KV[{i}]", kv_new[i], kv_orig[i]) for i in range(len(kv_orig)))

    print("\n=== 第二关：图 B 的 v_t == 原 denoise_step ===")
    with torch.no_grad():
        v_new = dg(x_t, ts, pad, *kv_orig)
    ok2 = compare("v_t", v_new, v_orig)

    print("\n=== 第三关：端到端 sample_actions ===")
    orig_denoise = fm.denoise_step

    def patched(prefix_pad_masks, past_key_values, x_t, timestep):
        return dg(x_t, timestep, prefix_pad_masks, *flatten_cache(past_key_values))

    with torch.no_grad():
        # 注意：sample_actions 的 images / img_masks 是 list（embed_prefix 内部 zip 它们）
        out_ref = fm.sample_actions([image], [img_mask], lang_tokens, lang_masks, state, noise=noise)
        fm.denoise_step = patched
        out_split = fm.sample_actions([image], [img_mask], lang_tokens, lang_masks, state, noise=noise)
        fm.denoise_step = orig_denoise
    ok3 = compare("最终动作", out_split, out_ref, tol=1e-3)

    # ------------------------------------------------------------ 导出
    print(f"\n=== ONNX 导出（legacy exporter；图 A opset {OPSET_PREFIX}，图 B opset {OPSET_DENOISE}）===")
    p_path = f"{OUT}/smolvla_prefix.onnx"
    d_path = f"{OUT}/smolvla_denoise.onnx"
    exported, errors = {}, {}

    # 图 A：需要规避视觉塔的 mask 分支；且因 SDPA 最低只能到 opset 14
    try:
        with VisionMaskPatch() as vmp:
            torch.onnx.export(
                pg, (image, img_mask, lang_tokens, lang_masks, state), p_path,
                input_names=["image", "img_mask", "lang_tokens", "lang_masks", "state"],
                output_names=[f"kv_{i}" for i in range(len(kv_orig))],
                opset_version=OPSET_PREFIX, do_constant_folding=CF_PREFIX,
                dynamic_axes=None, dynamo=False,
            )
        exported["prefix_mib"] = round(os.path.getsize(p_path) / 1024**2, 1)
        print(f"  图 A → {p_path}  ({exported['prefix_mib']:.1f} MiB)；规避 mask 调用 {vmp.count} 次")
        feed_a = {
            "image": image.cpu().numpy(),
            "img_mask": img_mask.cpu().numpy(),
            "lang_tokens": lang_tokens.cpu().numpy(),
            "lang_masks": lang_masks.cpu().numpy(),
            "state": state.cpu().numpy(),
        }
        try:
            err = verify_with_ort(p_path, feed_a, [t.float().cpu().numpy() for t in kv_orig])
            exported["prefix_ort_max_abs_delta"] = f"{err:.3e}"
            print(f"    onnxruntime vs PyTorch 最大绝对误差: {err:.3e}")
        except Exception as e:
            errors["prefix_ort"] = f"{type(e).__name__}: {str(e)[:150]}"
            print(f"    ORT 校验失败(不影响导出): {errors['prefix_ort']}")
    except Exception as e:
        errors["prefix_export"] = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"  图 A 导出失败: {errors['prefix_export']}")

    # 图 B
    try:
        torch.onnx.export(
            dg, (x_t, ts, pad, *kv_orig), d_path,
            input_names=["x_t", "timestep", "prefix_pad_masks"] + [f"kv_{i}" for i in range(len(kv_orig))],
            output_names=["v_t"],
            opset_version=OPSET_DENOISE, do_constant_folding=True,
            dynamic_axes=None, dynamo=False,
        )
        exported["denoise_mib"] = round(os.path.getsize(d_path) / 1024**2, 1)
        print(f"  图 B → {d_path}  ({exported['denoise_mib']:.1f} MiB)")
        feed_b = {
            "x_t": x_t.float().cpu().numpy(),
            "timestep": ts.float().cpu().numpy(),
            "prefix_pad_masks": pad.cpu().numpy(),
        }
        for i, t in enumerate(kv_orig):
            feed_b[f"kv_{i}"] = t.float().cpu().numpy()
        try:
            err = verify_with_ort(d_path, feed_b, [v_orig.float().cpu().numpy()])
            exported["denoise_ort_max_abs_delta"] = f"{err:.3e}"
            print(f"    onnxruntime vs PyTorch 最大绝对误差: {err:.3e}")
        except Exception as e:
            errors["denoise_ort"] = f"{type(e).__name__}: {str(e)[:150]}"
            print(f"    ORT 校验失败(不影响导出): {errors['denoise_ort']}")
    except Exception as e:
        errors["denoise_export"] = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"  图 B 导出失败: {errors['denoise_export']}")

    summary = {
        "checks": {"gate1_kv": bool(ok1), "gate2_denoise": bool(ok2), "gate3_end2end": bool(ok3)},
        "num_layers": n_layers,
        "prefix_len": int(pad.sum()),
        "opset": {"prefix": OPSET_PREFIX, "denoise": OPSET_DENOISE},
        "artifacts": exported,
        "errors": errors,
    }
    with open(f"{OUT}/export_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== 结论 ===")
    print(f"  三关校验: {'全部通过' if (ok1 and ok2 and ok3) else '存在失败项，见上'}")
    print(f"  导出产物: {exported or '无'}")
    if errors:
        print(f"  遗留问题: {list(errors)}")


if __name__ == "__main__":
    main()
