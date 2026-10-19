"""去噪步数蒸馏（step distillation）——主创新 B 的核心优化。

问题：SmolVLA 用 10 步前向 Euler 积分生成动作 chunk，占端到端延迟的 79%。
目标：训练动作专家，使 **1 步** 积分即可逼近 10 步的结果。

方法：flow-map / 一致性蒸馏。
    teacher（冻结）：x1_T = Euler(v_θ_teacher, x0, num_steps=10)
    student（可训）：x1_S = Euler(v_θ_student, x0, num_steps=1)
    loss = MSE(x1_S, x1_T)
前缀 KV 由冻结的 VLM 计算一次，teacher/student 共用（前缀路径完全相同）。

数据：复用已有的 `pusht_expert_train.h5`（234 万帧真实 RGB），
     **领域不同**（2D 推物仿真 vs 室内机器人），仅用于验证蒸馏机制与测误差下降；
     真实场景的精度保持率需待 E1/E2 数据集。

用法：
    cd /3240704007/sci/VLA
    CUDA_VISIBLE_DEVICES=5 .venv_smolvla/bin/python smolvla_distill_steps.py --steps 400
"""

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # 项目根（scripts/<类>/ 上一级），保证 import sim/ smolvla_* 可用


import argparse
import copy
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import h5py
import numpy as np
import torch

try:  # 注册 Blosc 等 HDF5 滤波器（本数据集 pixels 用的 filter id = 32001）
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

from transformers.cache_utils import DynamicCache

from lerobot.policies.common.vla_utils import make_att_2d_masks, pad_vector, resize_with_pad
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

CKPT = "models/smolvla_base"
H5 = "/3240704007/sci/pusht_expert_train.h5"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMG = 512  # prepare_images 实际放大到 512


# ---------------------------------------------------------------- 数据
def load_pixels(start: int, n: int) -> np.ndarray:
    """顺序读取 n 帧（h5py 顺序读比花式索引快得多）。"""
    with h5py.File(H5, "r") as f:
        return f["pixels"][start : start + n]


def load_state(start: int, n: int) -> np.ndarray:
    with h5py.File(H5, "r") as f:
        return f["state"][start : start + n].astype(np.float32)


def to_inputs(pix: np.ndarray, st: np.ndarray, cfg, dev):
    """uint8 HWC 像素 → 模型输入（复刻 prepare_images 的 resize_with_pad + [-1,1] 归一化）。"""
    x = torch.from_numpy(pix).to(dev).float().div_(255.0).permute(0, 3, 1, 2)  # BCHW
    x = resize_with_pad(x, IMG, IMG, pad_value=0) * 2.0 - 1.0
    img_mask = torch.ones(x.shape[0], dtype=torch.bool, device=dev)
    state = pad_vector(torch.from_numpy(st).to(dev), cfg.max_state_dim)
    # 无真实语言指令：用固定的占位 token（仅验证机制，不代表真实指令接地能力）
    lang = torch.zeros(x.shape[0], cfg.tokenizer_max_length, dtype=torch.long, device=dev)
    lang_m = torch.ones(x.shape[0], cfg.tokenizer_max_length, dtype=torch.bool, device=dev)
    return x, img_mask, lang, lang_m, state


# ---------------------------------------------------------------- 前缀
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
    """与 lerobot.policies.common.flow_matching.euler_integrate 完全一致的积分方案。"""
    dt = -1.0 / num_steps
    x = noise
    b = noise.shape[0]
    for step in range(num_steps):
        t = torch.full((b,), 1.0 + step * dt, dtype=torch.float32, device=noise.device)
        x = x + dt * fn(x, t)
    return x


# ---------------------------------------------------------------- 训练
def trainable_params(fm):
    """只训后缀路径（动作专家 + 动作侧投影），前缀 VLM/视觉塔冻结。"""
    mods = [fm.vlm_with_expert.lm_expert, fm.action_in_proj, fm.action_out_proj,
            fm.action_time_mlp_in, fm.action_time_mlp_out]
    return [p for m in mods for p in m.parameters()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train-frames", type=int, default=4096)
    ap.add_argument("--val-frames", type=int, default=256)
    ap.add_argument("--teacher-steps", type=int, default=10)
    ap.add_argument("--student-steps", type=int, default=1)
    ap.add_argument("--out", default="export/distilled_expert.pt")
    a = ap.parse_args()

    print(f"设备: {DEV}")
    torch.manual_seed(42)

    teacher = SmolVLAPolicy.from_pretrained(CKPT).to(DEV).eval()
    student = copy.deepcopy(teacher)
    for p in teacher.parameters():
        p.requires_grad_(False)
    tps = trainable_params(student.model)
    for p in student.model.parameters():
        p.requires_grad_(False)
    for p in tps:
        p.requires_grad_(True)
    print(f"可训参数量: {sum(p.numel() for p in tps) / 1e6:.2f} M")
    n_layers = student.model.vlm_with_expert.num_vlm_layers
    cfg = student.config

    print(f"\n读取 PushT 数据（训练 {a.train_frames} 帧 / 验证 {a.val_frames} 帧）…")
    tr_pix = load_pixels(100_000, a.train_frames)
    tr_st = load_state(100_000, a.train_frames)
    va_pix = load_pixels(900_000, a.val_frames)
    va_st = load_state(900_000, a.val_frames)

    noise_shape = (a.batch, cfg.chunk_size, cfg.max_action_dim)

    @torch.no_grad()
    def evaluate(policy, pix, st, n_eval=4):
        errs_multi, errs_k = [], []
        for i in range(n_eval):
            s = i * a.batch
            x, im, lg, lm, stt = to_inputs(pix[s : s + a.batch], st[s : s + a.batch], cfg, DEV)
            pad, flat = prefix_kv(teacher.model, x, im, lg, lm, stt)
            x0 = torch.randn(noise_shape, device=DEV)
            ref = euler(denoise_fn(teacher.model, pad, flat, n_layers), x0, a.teacher_steps)
            got = euler(denoise_fn(policy.model, pad, flat, n_layers), x0, a.student_steps)
            errs_multi.append(torch.linalg.norm(ref).item())
            errs_k.append((got - ref).norm(dim=(1, 2)).mean().item())
        scale = np.mean(errs_multi)
        e = float(np.mean(errs_k))
        return e, 100 * e / scale

    e0, r0 = evaluate(student, va_pix, va_st)
    print(f"\n[蒸馏前] {a.student_steps} 步 vs {a.teacher_steps} 步："
          f"L2 = {e0:.4f}  相对误差 = {r0:.2f}%")

    opt = torch.optim.AdamW(tps, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)

    print(f"\n开始蒸馏（{a.steps} 步, batch={a.batch}, lr={a.lr}）…")
    t0 = time.time()
    hist = []
    n_train = a.train_frames - a.batch
    for step in range(a.steps):
        idx = np.random.randint(0, n_train, a.batch)
        x, im, lg, lm, stt = to_inputs(tr_pix[idx], tr_st[idx], cfg, DEV)

        with torch.no_grad():
            pad, flat = prefix_kv(teacher.model, x, im, lg, lm, stt)
            x0 = torch.randn(noise_shape, device=DEV)
            target = euler(denoise_fn(teacher.model, pad, flat, n_layers), x0, a.teacher_steps)

        pred = euler(denoise_fn(student.model, pad, flat, n_layers), x0, a.student_steps)
        loss = torch.nn.functional.mse_loss(pred, target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tps, 1.0)
        opt.step()
        sched.step()

        if step % 50 == 0 or step == a.steps - 1:
            hist.append((step, loss.item()))
            print(f"  step {step:4d}/{a.steps}  loss {loss.item():.6f}  "
                  f"({time.time() - t0:.0f}s)")

    e1, r1 = evaluate(student, va_pix, va_st)
    print(f"\n[蒸馏后] {a.student_steps} 步 vs {a.teacher_steps} 步："
          f"L2 = {e1:.4f}  相对误差 = {r1:.2f}%")
    print(f"  相对误差下降: {r0:.2f}% → {r1:.2f}%（降低 {100 * (r0 - r1) / r0:.1f}%）")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"expert": student.model.vlm_with_expert.lm_expert.state_dict(),
                "action_out_proj": student.model.action_out_proj.state_dict(),
                "meta": {"student_steps": a.student_steps, "teacher_steps": a.teacher_steps,
                         "loss_history": hist, "rel_err_before": r0, "rel_err_after": r1}},
               a.out)
    print(f"\n已保存蒸馏后的专家权重: {a.out}")


if __name__ == "__main__":
    main()
