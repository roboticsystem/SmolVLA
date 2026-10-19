"""生成「裁剪 + 蒸馏」后的端侧部署包（主创新 B 的收口产物）。

链路：教师(16层×10步) → 层裁剪(N层) → 步数蒸馏(K步) → 导出 ONNX → 算子普查

产出（`--out` 目录下）：
    prefix.onnx      图 A：图像+语言+状态 → KV   （跑 1 次）
    denoise.onnx     图 B：KV+x_t+t → v_t        （跑 K 次）
    expert.pt        蒸馏后的专家权重（可继续微调）
    op_census.json   两张图的算子普查
    meta.json        体积 / 节点数 / 覆盖率 / 延迟 / 与 16 层基线的对比

【重要边界】本脚本产出的是 ONNX 模型，**不等于地平线 BPU 可用模型**。
真正的 BPU 量化需要地平线 OpenExplorer 工具链（Docker 分发），当前环境不可用。
本脚本的价值是：把「裁剪+蒸馏」的收益落到**模型体积与算子构成**上，为后续上板量化做准备。

用法：
    cd /3240704007/sci/VLA
    CUDA_VISIBLE_DEVICES=5 .venv_smolvla/bin/python smolvla_export_deploy.py \
        --layers 4 --student-steps 1 --distill-steps 600 --out export/deploy_4x1
"""

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # 项目根（scripts/<类>/ 上一级），保证 import sim/ smolvla_* 可用


import argparse
import json
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch

from smolvla_distill_joint import (
    CKPT,
    DEV,
    TEACHER_STEPS,
    denoise_fn,
    euler,
    load_pixels,
    load_state,
    prefix_kv,
    prune_layers,
    to_inputs,
    trainable_params,
)
from smolvla_export_onnx import DenoiseGraph, PrefixGraph, VisionMaskPatch

OPSET_PREFIX = 14   # 含 SDPA，最低 14
OPSET_DENOISE = 11
CF_PREFIX = False   # 图 A 必须关闭常量折叠


def census(path):
    """内联最小版算子普查，避免依赖 onnx_op_census 的命令行入口。"""
    import collections

    import onnx

    m = onnx.load(path, load_external_data=False)
    ops = collections.Counter(n.op_type for n in m.graph.node)
    hazard_ids = ["Where", "ScatterND", "NonZero", "IsNaN", "Einsum", "TopK", "ArgMax",
                  "LayerNormalization", "Pow", "Sqrt", "Reciprocal", "Erf", "Gelu",
                  "Resize", "Upsample", "Gather"]
    hz = {k: ops[k] for k in hazard_ids if ops[k]}
    return {
        "size_mib": round(os.path.getsize(path) / 1024**2, 1),
        "opset": {o.domain or "ai.onnx": o.version for o in m.opset_import},
        "nodes": sum(ops.values()),
        "op_top": dict(ops.most_common(12)),
        "hazards": hz,
        "hazard_nodes": sum(hz.values()),
        "coverage_pct": round(100 * (1 - sum(hz.values()) / max(sum(ops.values()), 1)), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--student-steps", type=int, default=1)
    ap.add_argument("--distill-steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train-frames", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-distill", action="store_true", help="只导出结构，不训练")
    ap.add_argument("--out", default="export/deploy")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    print(f"设备: {DEV} | 目标: {a.layers} 层 × {a.student_steps} 步")

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    teacher = SmolVLAPolicy.from_pretrained(CKPT).to(DEV).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    import copy

    import torch.nn as nn

    student = copy.deepcopy(teacher)
    if a.layers != teacher.model.vlm_with_expert.num_vlm_layers:
        prune_layers(student, a.layers)
    n = student.model.vlm_with_expert.num_vlm_layers
    n_t = teacher.model.vlm_with_expert.num_vlm_layers
    tps = trainable_params(student.model)
    print(f"学生层数={n}  可训参数={sum(p.numel() for p in tps) / 1e6:.2f} M")

    # ---------------- 蒸馏 ----------------
    if not a.skip_distill:
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        print(f"\n读取训练数据 {a.train_frames} 帧…")
        pix = load_pixels(100_000, a.train_frames)
        st = load_state(100_000, a.train_frames)
        for p in student.model.parameters():
            p.requires_grad_(False)
        for p in tps:
            p.requires_grad_(True)

        opt = torch.optim.AdamW(tps, lr=a.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.distill_steps)
        noise_shape = (a.batch, student.config.chunk_size, student.config.max_action_dim)
        t0 = time.time()
        for step in range(a.distill_steps):
            idx = np.random.randint(0, a.train_frames - a.batch, a.batch)
            x, im, lg, lm, stt = to_inputs(pix[idx], st[idx], student.config, DEV)
            with torch.no_grad():
                pad, flat = prefix_kv(teacher.model, x, im, lg, lm, stt)
                x0 = torch.randn(noise_shape, device=DEV)
                target = euler(denoise_fn(teacher.model, pad, flat, n_t), x0, TEACHER_STEPS)
            pred = euler(denoise_fn(student.model, pad, flat, n), x0, a.student_steps)
            loss = torch.nn.functional.mse_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tps, 1.0)
            opt.step()
            sched.step()
            if step % 200 == 0 or step == a.distill_steps - 1:
                print(f"  step {step:4d}/{a.distill_steps} loss {loss.item():.6f} ({time.time() - t0:.0f}s)")
        del pix, st
        torch.save({"expert": student.model.vlm_with_expert.lm_expert.state_dict(),
                    "action_out_proj": student.model.action_out_proj.state_dict(),
                    "layers": n, "steps": a.student_steps},
                   f"{a.out}/expert.pt")
        print(f"  已保存 {a.out}/expert.pt")

    # ---------------- 延迟 ----------------
    with torch.no_grad():
        va_pix = load_pixels(900_000, 8)
        va_st = load_state(900_000, 8)
        x, im, lg, lm, stt = to_inputs(va_pix[:1], va_st[:1], student.config, DEV)
        x0 = torch.randn(1, student.config.chunk_size, student.config.max_action_dim, device=DEV)

        def t_full(policy, k, nl):
            def once():
                p_s, f_s = prefix_kv(policy.model, x, im, lg, lm, stt)
                return euler(denoise_fn(policy.model, p_s, f_s, nl), x0, k)

            for _ in range(3):
                once()
            torch.cuda.synchronize()
            ts = []
            for _ in range(20):
                torch.cuda.synchronize()
                s = time.perf_counter()
                once()
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - s) * 1e3)
            return float(np.percentile(ts, 50))

        lat_t = t_full(teacher, TEACHER_STEPS, n_t)
        lat_s = t_full(student, a.student_steps, n)

    # ---------------- ONNX 导出 ----------------
    print(f"\n导出 ONNX（图 A opset {OPSET_PREFIX} / 图 B opset {OPSET_DENOISE}）…")
    cfg = student.config
    h = w = cfg.resize_imgs_with_padding[1]
    image = torch.randn(1, 3, h, w, device=DEV)
    img_mask = torch.ones(1, dtype=torch.bool, device=DEV)
    lang = torch.zeros(1, cfg.tokenizer_max_length, dtype=torch.long, device=DEV)
    lang_m = torch.ones(1, cfg.tokenizer_max_length, dtype=torch.bool, device=DEV)
    state = torch.randn(1, cfg.max_state_dim, device=DEV)

    with torch.no_grad():
        pad, flat = prefix_kv(student.model, image, img_mask, lang, lang_m, state)
    x_t = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, device=DEV)
    ts = torch.tensor([0.5], device=DEV)

    pg = PrefixGraph(student.model).eval()
    dg = DenoiseGraph(student.model, n).eval()
    p_path, d_path = f"{a.out}/prefix.onnx", f"{a.out}/denoise.onnx"

    errs = {}
    try:
        with VisionMaskPatch() as vmp:
            torch.onnx.export(
                pg, (image, img_mask, lang, lang_m, state), p_path,
                input_names=["image", "img_mask", "lang_tokens", "lang_masks", "state"],
                output_names=[f"kv_{i}" for i in range(len(flat))],
                opset_version=OPSET_PREFIX, do_constant_folding=CF_PREFIX,
                dynamic_axes=None, dynamo=False,
            )
        print(f"  图 A → {p_path}（规避 mask {vmp.count} 次）")
    except Exception as e:
        errs["prefix"] = f"{type(e).__name__}: {str(e)[:160]}"
        print(f"  图 A 导出失败: {errs['prefix']}")

    try:
        torch.onnx.export(
            dg, (x_t, ts, pad, *flat), d_path,
            input_names=["x_t", "timestep", "prefix_pad_masks"] + [f"kv_{i}" for i in range(len(flat))],
            output_names=["v_t"],
            opset_version=OPSET_DENOISE, do_constant_folding=True,
            dynamic_axes=None, dynamo=False,
        )
        print(f"  图 B → {d_path}")
    except Exception as e:
        errs["denoise"] = f"{type(e).__name__}: {str(e)[:160]}"
        print(f"  图 B 导出失败: {errs['denoise']}")

    # ---------------- 普查与汇总 ----------------
    cen = {}
    for name, path in (("prefix", p_path), ("denoise", d_path)):
        if os.path.exists(path):
            cen[name] = census(path)
            print(f"\n[{name}] {cen[name]['size_mib']} MiB, {cen[name]['nodes']} 节点, "
                  f"覆盖率 {cen[name]['coverage_pct']}%, 高危节点 {cen[name]['hazard_nodes']}")
            print(f"   高危算子: {cen[name]['hazards']}")

    meta = {
        "config": {"layers": n, "student_steps": a.student_steps,
                   "teacher_layers": n_t, "teacher_steps": TEACHER_STEPS},
        "trainable_M": round(sum(p.numel() for p in tps) / 1e6, 2),
        "latency_ms": {"teacher": round(lat_t, 1), "deploy": round(lat_s, 1),
                       "drop_pct": round(100 * (lat_t - lat_s) / lat_t, 1)},
        "onnx": cen,
        "errors": errs,
        "note": "ONNX 产物不等价于地平线 BPU 可用模型；BPU 量化需 OpenExplorer 工具链。",
    }
    with open(f"{a.out}/meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"部署包: {a.out}")
    print(f"  配置     {n} 层 × {a.student_steps} 步（教师 {n_t} 层 × {TEACHER_STEPS} 步）")
    print(f"  延迟     {lat_t:.1f} ms → {lat_s:.1f} ms（−{meta['latency_ms']['drop_pct']}%）")
    print(f"  产物     {sorted(os.listdir(a.out))}")


if __name__ == "__main__":
    main()
