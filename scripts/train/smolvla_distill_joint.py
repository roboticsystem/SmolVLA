"""步数 × 层数联合蒸馏（主创新 B 的主实验）。

两个杠杆（实测）：
    去噪步数 10→1  : e2e −79.5%
    层数     16→8  : e2e −51.0%
二者近似乘性叠加，本脚本在**同一框架内联合优化**，给出「精度 × 延迟」的帕累托设计空间。

层数裁剪要点（读源码确认）：
    `lm_expert` 的层数与 VLM 一一对应（`lm_expert_config.num_hidden_layers = num_vlm_layers`），
    `get_model_layers` 按 `expert_layer_index = i // multiple_of` 配对。
    因此裁剪必须**同时截断 text_model.layers 与 lm_expert.layers**，并同步三个计数属性，
    否则索引错位、KV 层数不匹配。

KV 复用：prefill 是逐层顺序的，前 N 层的 KV 与总层数无关，
         故可只算一次 16 层 prefill，按 `flat[:2N]` 切片供 N 层学生使用。
         脚本内含该假设的一致性校验。

用法：
    cd /3240704007/sci/VLA
    CUDA_VISIBLE_DEVICES=5 .venv_smolvla/bin/python smolvla_distill_joint.py \
        --configs "16:1,12:2,8:2,8:1,4:1" --steps 600
"""

import argparse
import copy
import json
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import h5py
import numpy as np
import torch
import torch.nn as nn

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

from transformers.cache_utils import DynamicCache

from lerobot.policies.common.vla_utils import make_att_2d_masks, pad_vector, resize_with_pad
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

CKPT = "models/smolvla_base"
H5 = "/3240704007/sci/pusht_expert_train.h5"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMG = 512
TEACHER_STEPS = 10


# ---------------------------------------------------------------- 数据
def load_pixels(start, n):
    with h5py.File(H5, "r") as f:
        return f["pixels"][start : start + n]


def load_state(start, n):
    with h5py.File(H5, "r") as f:
        return f["state"][start : start + n].astype(np.float32)


def to_inputs(pix, st, cfg, dev):
    x = torch.from_numpy(pix).to(dev).float().div_(255.0).permute(0, 3, 1, 2)
    x = resize_with_pad(x, IMG, IMG, pad_value=0) * 2.0 - 1.0
    img_mask = torch.ones(x.shape[0], dtype=torch.bool, device=dev)
    state = pad_vector(torch.from_numpy(st).to(dev), cfg.max_state_dim)
    lang = torch.zeros(x.shape[0], cfg.tokenizer_max_length, dtype=torch.long, device=dev)
    lang_m = torch.ones(x.shape[0], cfg.tokenizer_max_length, dtype=torch.bool, device=dev)
    return x, img_mask, lang, lang_m, state


# ---------------------------------------------------------------- 数据源
class PushtProvider:
    """原数据源：PushT（2D 推物仿真）。领域与室内机器人不同，仅作回归对照。"""

    name = "pusht"

    def __init__(self, a):
        print(f"读取 PushT 数据（训练 {a.train_frames} / 验证 {a.val_frames} 帧）…")
        self.tr = (load_pixels(100_000, a.train_frames), load_state(100_000, a.train_frames))
        self.va = (load_pixels(900_000, a.val_frames), load_state(900_000, a.val_frames))

    def train(self, batch):
        idx = np.random.randint(0, len(self.tr[0]) - batch, batch)
        return self.tr[0][idx], self.tr[1][idx], None

    def val(self, batch):
        return self.va[0][:batch], self.va[1][:batch], None

    def inputs(self, pix, st, instr, cfg, dev):
        return to_inputs(pix, st, cfg, dev)


class SimProvider:
    """新数据源：自建语义 MultiRoom 仿真（真实导航任务域）。"""

    name = "sim"

    def __init__(self, path, a, max_state_dim, tok_len, chunk):
        from smolvla_sim_data import SimDemoData
        self.d = SimDemoData(path, device="cpu", max_state_dim=max_state_dim,
                             tokenizer_max_length=tok_len, chunk_size=chunk,
                             max_action_dim=2)
        print("  " + self.d.summary().replace("\n", "\n  "))
        v = self.d.valid_idx
        cut = int(v.size * 0.9)
        self.tr_idx, self.va_idx = v[:cut], v[cut:][:max(1, a.val_frames)]

    def train(self, batch):
        idx = self.tr_idx[np.random.randint(0, self.tr_idx.size, batch)]
        return self.d.images[idx], self.d.states[idx], [self.d.instr[int(i)] for i in idx]

    def val(self, batch):
        # 验证集可能小于 batch（数据集小时尤甚）。必须**补齐到恰好 batch 个**，
        # 否则 prefix_pad_masks 的批维与预生成的 val_noise 不一致，
        # 报错形式是 "Sizes of tensors must match ... Expected size 5 but got size 8"。
        idx = self.va_idx if self.va_idx.size else self.d.valid_idx[:1]
        idx = np.resize(idx, batch) if idx.size < batch else idx[:batch]
        return self.d.images[idx], self.d.states[idx], [self.d.instr[int(i)] for i in idx]

    def inputs(self, pix, st, instr, cfg, dev):
        x = torch.from_numpy(pix).to(dev).float().div_(255.0).permute(0, 3, 1, 2)
        x = resize_with_pad(x, IMG, IMG, pad_value=0) * 2.0 - 1.0
        img_mask = torch.ones(x.shape[0], dtype=torch.bool, device=dev)
        state = pad_vector(torch.from_numpy(st).to(dev), cfg.max_state_dim)
        lang, lang_m = self.d.tokenize(instr)
        # SimDemoData 建在 CPU 上（不占显存），语言张量需搬到计算设备
        lang, lang_m = lang.to(dev), lang_m.to(dev)
        return x, img_mask, lang, lang_m, state


# ---------------------------------------------------------------- 前缀 / 积分
@torch.no_grad()
def prefix_kv(fm, image, img_mask, lang, lang_m, state):
    embs, pad, att = fm.embed_prefix([image], [img_mask], lang, lang_m, state=state)
    a2d = make_att_2d_masks(pad, att)
    pos = torch.cumsum(pad, dim=1) - 1
    _, pkv = fm.vlm_with_expert.forward(
        attention_mask=a2d, position_ids=pos, past_key_values=None,
        inputs_embeds=[embs, None], use_cache=True,
    )
    flat = []
    for layer in pkv.layers:
        flat += [layer.keys, layer.values]
    return pad, flat


def denoise_fn(fm, pad, flat, n_layers):
    def fn(x_t, time_tensor):
        cache = DynamicCache(
            ddp_cache_data=[(flat[2 * i], flat[2 * i + 1]) for i in range(n_layers)]
        )
        return fm.denoise_step(
            prefix_pad_masks=pad, past_key_values=cache, x_t=x_t, timestep=time_tensor
        )

    return fn


def euler(fn, noise, num_steps):
    dt = -1.0 / num_steps
    x = noise
    b = noise.shape[0]
    for step in range(num_steps):
        t = torch.full((b,), 1.0 + step * dt, dtype=torch.float32, device=noise.device)
        x = x + dt * fn(x, t)
    return x


# ---------------------------------------------------------------- 层裁剪
def prune_layers(policy, n):
    """同步截断 VLM 层与专家层，并更新三个计数属性。"""
    vwe = policy.model.vlm_with_expert
    vwe.get_vlm_model().text_model.layers = nn.ModuleList(
        list(vwe.get_vlm_model().text_model.layers)[:n]
    )
    vwe.lm_expert.layers = nn.ModuleList(list(vwe.lm_expert.layers)[:n])
    vwe.lm_expert.config.num_hidden_layers = n
    vwe.num_vlm_layers = n
    vwe.num_expert_layers = n


def trainable_params(fm):
    mods = [fm.vlm_with_expert.lm_expert, fm.action_in_proj, fm.action_out_proj,
            fm.action_time_mlp_in, fm.action_time_mlp_out]
    return [p for m in mods for p in m.parameters()]


# ---------------------------------------------------------------- 单配置
def run_one(teacher, cfg, args, prov, val_noise, order):
    n_layers, k_steps = cfg
    print(f"\n{'=' * 70}\n配置: {n_layers} 层 × {k_steps} 步\n{'=' * 70}")

    student = copy.deepcopy(teacher)
    if n_layers != teacher.model.vlm_with_expert.num_vlm_layers:
        prune_layers(student, n_layers)
    for p in student.model.parameters():
        p.requires_grad_(False)
    tps = trainable_params(student.model)
    for p in tps:
        p.requires_grad_(True)
    print(f"  学生层数={student.model.vlm_with_expert.num_vlm_layers}  "
          f"可训参数={sum(p.numel() for p in tps) / 1e6:.2f} M")

    n_layers_t = teacher.model.vlm_with_expert.num_vlm_layers
    noise_shape = (args.batch, student.config.chunk_size, student.config.max_action_dim)

    # --- 一致性校验：前 N 层的 KV 是否与总层数无关 ---
    with torch.no_grad():
        vp, vs, vi = prov.val(args.batch)
        x, im, lg, lm, stt = prov.inputs(vp, vs, vi, student.config, DEV)
        pad, flat_full = prefix_kv(teacher.model, x, im, lg, lm, stt)
        _, flat_full2 = prefix_kv(student.model, x, im, lg, lm, stt)
    d = max((flat_full[i] - flat_full2[i]).abs().max().item() for i in range(len(flat_full2)))
    print(f"  KV 切片一致性: max|Δ| = {d:.3e}  {'OK' if d < 1e-3 else '!! 不一致'}")

    def evaluate(policy, k):
        torch.manual_seed(1234)
        errs, scales = [], []
        with torch.no_grad():
            for bi in range(args.eval_batches):
                vp, vs, vi = prov.val(args.batch)
                xi, imi, lgi, lmi, sti = prov.inputs(vp, vs, vi, student.config, DEV)
                padi, flati = prefix_kv(teacher.model, xi, imi, lgi, lmi, sti)
                x0 = val_noise[bi]
                ref = euler(denoise_fn(teacher.model, padi, flati, n_layers_t), x0, TEACHER_STEPS)
                got = euler(denoise_fn(policy.model, padi, flati, n_layers), x0, k)
                errs.append((got - ref).norm(dim=(1, 2)).mean().item())
                scales.append(torch.linalg.norm(ref).item())
        e, sc = float(np.mean(errs)), float(np.mean(scales))
        return e, 100 * e / sc

    _, r_before = evaluate(student, k_steps)
    print(f"  [蒸馏前] {k_steps} 步 vs {TEACHER_STEPS} 步: 相对误差 {r_before:.2f}%")

    opt = torch.optim.AdamW(tps, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    t0 = time.time()
    hist = []
    for step in range(args.steps):
        tp, ts_, ti = prov.train(args.batch)
        xi, imi, lgi, lmi, sti = prov.inputs(tp, ts_, ti, student.config, DEV)
        with torch.no_grad():
            padi, flati = prefix_kv(teacher.model, xi, imi, lgi, lmi, sti)
            x0 = torch.randn(noise_shape, device=DEV)
            target = euler(denoise_fn(teacher.model, padi, flati, n_layers_t), x0, TEACHER_STEPS)
        pred = euler(denoise_fn(student.model, padi, flati, n_layers), x0, k_steps)
        loss = torch.nn.functional.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tps, 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == args.steps - 1:
            hist.append((step, loss.item()))
            print(f"    step {step:4d}/{args.steps} loss {loss.item():.6f} ({time.time() - t0:.0f}s)")

    _, r_after = evaluate(student, k_steps)

    # --- 延迟实测（完整流水线：embed_prefix + prefill(N) + K 次 denoise(N)）---
    with torch.no_grad():
        vp, vs, vi = prov.val(1)
        x, im, lg, lm, stt = prov.inputs(vp, vs, vi, student.config, DEV)
        x0 = torch.randn(1, student.config.chunk_size, student.config.max_action_dim, device=DEV)

        def t_full(policy, k, nl):
            def once():
                p_s, f_s = prefix_kv(policy.model, x, im, lg, lm, stt)
                return euler(denoise_fn(policy.model, p_s, f_s, nl), x0, k)

            for _ in range(3):
                once()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ts = []
            for _ in range(15):
                torch.cuda.synchronize()
                s = time.perf_counter()
                once()
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - s) * 1e3)
            return float(np.percentile(ts, 50))

        lat = t_full(student, k_steps, n_layers)
        lat_teacher = t_full(teacher, TEACHER_STEPS, n_layers_t)

    print(f"  [蒸馏后] 相对误差 {r_before:.2f}% → {r_after:.2f}%"
          f"（降低 {100 * (r_before - r_after) / r_before:.1f}%）")
    print(f"  延迟: 教师 {lat_teacher:.1f} ms → 本配置 {lat:.1f} ms"
          f"（−{100 * (lat_teacher - lat) / lat_teacher:.1f}%）")

    rec = {"layers": n_layers, "student_steps": k_steps, "teacher_steps": TEACHER_STEPS,
           "trainable_M": round(sum(p.numel() for p in tps) / 1e6, 2),
           "kv_slice_max_delta": d,
           "rel_err_before": round(r_before, 2), "rel_err_after": round(r_after, 2),
           "loss_history": hist, "latency_ms": round(lat, 1),
           "latency_teacher_ms": round(lat_teacher, 1),
           "latency_drop_pct": round(100 * (lat_teacher - lat) / lat_teacher, 1)}
    order.append(rec)
    del student
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="16:1,12:2,8:2,8:1,4:1")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train-frames", type=int, default=4096)
    ap.add_argument("--val-frames", type=int, default=128)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--seeds", default="42", help="逗号分隔的随机种子，用于统计严谨性")
    ap.add_argument("--data", default="pusht", choices=["pusht", "sim"],
                    help="观测数据源：pusht=原 2D 推物域；sim=自建语义导航域")
    ap.add_argument("--sim-data", default="export/nav_demos.h5")
    ap.add_argument("--out", default="export/joint_distill.json")
    a = ap.parse_args()

    print(f"设备: {DEV}")
    configs = []
    for c in a.configs.split(","):
        nl, ks = c.strip().split(":")
        configs.append((int(nl), int(ks)))
    seeds = [int(s) for s in a.seeds.split(",")]
    print(f"配置 {len(configs)} 个 × 种子 {len(seeds)} 组 = {len(configs) * len(seeds)} 次训练")

    teacher = SmolVLAPolicy.from_pretrained(CKPT).to(DEV).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    n_layers_t = teacher.model.vlm_with_expert.num_vlm_layers
    tcfg = teacher.config
    print(f"教师: {n_layers_t} 层 × {TEACHER_STEPS} 步")

    print(f"\n=== 数据源: {a.data} ===")
    if a.data == "sim":
        prov = SimProvider(a.sim_data, a, tcfg.max_state_dim, tcfg.tokenizer_max_length,
                           tcfg.chunk_size)
    else:
        prov = PushtProvider(a)

    # 验证用噪声固定（与训练种子无关），保证各配置/各种子在同一批样本上比较
    g = torch.Generator(device="cpu").manual_seed(2026)
    shape = (a.batch, teacher.config.chunk_size, teacher.config.max_action_dim)
    val_noise = [torch.randn(shape, generator=g).to(DEV) for _ in range(a.eval_batches)]

    order = []
    for si, seed in enumerate(seeds):
        print(f"\n{'#' * 70}\n# 种子 {seed}（{si + 1}/{len(seeds)}）\n{'#' * 70}")
        for cfg in configs:
            torch.manual_seed(seed)
            np.random.seed(seed)
            rec = run_one(teacher, cfg, a, prov, val_noise, order)
            rec["seed"] = seed

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(order, f, indent=2, ensure_ascii=False)

    # ---------------- 聚合：均值 ± 标准差 ----------------
    agg = {}
    for r in order:
        agg.setdefault((r["layers"], r["student_steps"]), []).append(r)

    print(f"\n{'=' * 78}")
    print(f"联合蒸馏汇总（{a.steps} 步训练，{len(seeds)} 组种子，误差为 {len(seeds)} 次的均值 ± 标准差）")
    print(f"{'=' * 78}")
    print(f"{'配置':>8s} {'可训(M)':>9s} {'误差(均值±SD)':>18s} {'延迟(均值±SD) ms':>20s} "
          f"{'速度增益':>9s} {'n':>3s}")
    summary = []
    for (nl, ks), rs in sorted(agg.items()):
        e = np.array([x["rel_err_after"] for x in rs], dtype=float)
        lat = np.array([x["latency_ms"] for x in rs], dtype=float)
        dd = np.array([x["latency_drop_pct"] for x in rs], dtype=float)
        sd = lambda v: v.std(ddof=1) if len(v) > 1 else 0.0
        print(f"{nl}x{ks:<5d} {rs[0]['trainable_M']:>9.2f} "
              f"{e.mean():>10.2f}% ± {sd(e):>5.2f} "
              f"{lat.mean():>11.1f} ± {sd(lat):>4.1f} "
              f"{dd.mean():>8.1f}% {len(rs):>3d}")
        summary.append({"layers": nl, "student_steps": ks, "n": len(rs),
                        "trainable_M": rs[0]["trainable_M"],
                        "err_mean": round(float(e.mean()), 3), "err_std": round(float(sd(e)), 3),
                        "lat_mean": round(float(lat.mean()), 2), "lat_std": round(float(sd(lat)), 2),
                        "drop_mean_pct": round(float(dd.mean()), 2)})

    # ---------------- 配对比较（对同一组种子做配对 bootstrap） ----------------
    def paired(cfg_a, cfg_b, label_a, label_b):
        A = {r["seed"]: r["rel_err_after"] for r in order
             if (r["layers"], r["student_steps"]) == cfg_a}
        B = {r["seed"]: r["rel_err_after"] for r in order
             if (r["layers"], r["student_steps"]) == cfg_b}
        keys = sorted(set(A) & set(B))
        if len(keys) < 2:
            return None
        d = np.array([B[k] - A[k] for k in keys], dtype=float)
        rng = np.random.default_rng(0)
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        sig = "显著（95% CI 不含 0）" if (lo > 0 or hi < 0) else "不显著（95% CI 含 0）"
        print(f"\n配对比较 {label_b} − {label_a}（n={len(keys)} 组配对种子）："
              f"均值差 {d.mean():+.3f} pp，95% CI [{lo:+.3f}, {hi:+.3f}] → {sig}")
        return {"comparison": f"{label_b} - {label_a}", "n_pairs": len(keys),
                "mean_diff_pp": round(float(d.mean()), 3),
                "ci95_pp": [round(lo, 3), round(hi, 3)], "significant": bool(lo > 0 or hi < 0)}

    comps = []
    for cfg_a, cfg_b, la, lb in [((16, 1), (8, 1), "16x1", "8x1"),
                                 ((16, 1), (4, 1), "16x1", "4x1"),
                                 ((8, 1), (4, 1), "8x1", "4x1")]:
        c = paired(cfg_a, cfg_b, la, lb)
        if c:
            comps.append(c)

    with open(a.out.replace(".json", "_summary.json"), "w") as f:
        json.dump({"summary": summary, "paired": comps, "steps": a.steps,
                   "seeds": seeds}, f, indent=2, ensure_ascii=False)
    print(f"\n已写入 {a.out} 与 {a.out.replace('.json', '_summary.json')}")


if __name__ == "__main__":
    main()
