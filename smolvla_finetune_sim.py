"""主创新 A：动作空间对齐 + 自建仿真域微调。

三件事：
    ① **动作空间重构**：官方基座是 3 路相机 / 6 维机械臂动作 / `max_action_dim=32`；
       本课题是 **1 路相机 / 2 维差速底盘动作 (v, ω)**。需要改 `input_features`、
       `output_features`、`max_action_dim`，并对动作头做权重外科。
    ② **域迁移微调**：官方默认 `freeze_vision_encoder=True` + `train_expert_only=True`
       （官方预设就是只训 98M 动作专家）。本脚本把它作为**成本基线**，
       并支持解冻视觉塔末层作为本文方法的对照。
    ③ **评价**：开环动作 MSE + **闭环导航成功率**（在自建仿真里真跑）。

用法：
    cd /3240704007/sci/VLA
    CUDA_VISIBLE_DEVICES=5 .venv_smolvla/bin/python smolvla_finetune_sim.py \
        --data export/nav_demos.h5 --steps 3000 --mode expert --out export/ft_expert
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
import torch.nn as nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from sim.perception import DET_DIM, TargetMemory, target_features
from smolvla_sim_data import (ACTION_KEY, IMG_KEY, IMG_KEYS, LANG_MASK_KEY,
                              LANG_TOKENS_KEY, STATE_KEY, SimDemoData)

CKPT = "models/smolvla_base"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- ① 动作空间对齐
def truncate_action_head(fm, new_dim: int):
    """把动作头从 `max_action_dim` 裁到 `new_dim`（权重切片）。

    注意：`action_in_proj` 是 Linear(32 → 720)，会**跨动作维度混合**，
    因此这不是无损变换，而是**目标域微调的初始点**（实测截断偏差 29.55%）。
    """
    w_in, b_in = fm.action_in_proj.weight, fm.action_in_proj.bias
    w_out, b_out = fm.action_out_proj.weight, fm.action_out_proj.bias
    nin = nn.Linear(new_dim, w_in.shape[0], bias=b_in is not None).to(w_in.device, w_in.dtype)
    nout = nn.Linear(w_out.shape[1], new_dim, bias=b_out is not None).to(w_out.device, w_out.dtype)
    with torch.no_grad():
        nin.weight.copy_(w_in[:, :new_dim])
        if b_in is not None:
            nin.bias.copy_(b_in)
        nout.weight.copy_(w_out[:new_dim, :])
        if b_out is not None:
            nout.bias.copy_(b_out[:new_dim])
    fm.action_in_proj, fm.action_out_proj = nin, nout


def reconfigure_for_diff_drive(policy, action_dim=2, cam_shape=(3, 120, 160), n_cam=1,
                               state_dim=None):
    """把 3 路相机 / 6 维动作的机械臂配置改成差速底盘配置。

    `n_cam > 1` 用于**帧堆叠**：把 t、t-1、t-2 … 各帧注册成 camera1、camera2、camera3 …
    复用 SmolVLA 原生的多相机接口，无需改模型结构（每帧各自过 SigLIP，
    token 按 camera1..K 顺序拼进 prefix）。这是缓解部分可观测 / 协变量偏移的结构性手段。
    """
    cfg = policy.config

    # 扩展状态维度（用于把激光雷达并进状态）。`state_proj` 是 Linear(max_state_dim → hidden)，
    # 因此改维度需要重建它：前 k 维一一对应地拷贝旧权重，**新增维度从零学起**。
    if state_dim and int(state_dim) != cfg.max_state_dim:
        old = policy.model.state_proj
        new = nn.Linear(int(state_dim), old.out_features,
                        bias=old.bias is not None).to(old.weight.device, old.weight.dtype)
        with torch.no_grad():
            new.weight.zero_()
            k = min(old.in_features, int(state_dim))
            new.weight[:, :k].copy_(old.weight[:, :k])     # Linear.weight 形状是 (out, in)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        policy.model.state_proj = new
        print(f"  state_proj 输入维 {old.in_features} → {state_dim}"
              f"（保留前 {k} 维权重，新增维从零学起）")
        cfg.max_state_dim = int(state_dim)

    feats = {STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(cfg.max_state_dim,))}
    for i in range(int(n_cam)):
        feats[IMG_KEYS[i]] = PolicyFeature(type=FeatureType.VISUAL, shape=cam_shape)
    cfg.input_features = feats
    cfg.output_features = {ACTION_KEY: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
    cfg.max_action_dim = action_dim
    truncate_action_head(policy.model, action_dim)
    return cfg


def set_trainable(policy, mode: str, lora_r=8, lora_alpha=16.0,
                  lora_targets=("vlm_attn",)):
    """按模式设置可训参数（E2 的「微调范式」自变量）。

    expert —— 官方默认（冻结视觉塔与 VLM，只训动作专家），作为**成本基线**
    lora   —— 与 expert 相同，另在 VLM 上**挂 LoRA 适配器**（本模块 §E2 ②）
    vision —— 额外解冻视觉塔末层（本文方法的域自适应对照）
    full   —— 全参微调（算力上界参考）

    **三档的公平口径**：动作专家在三档里都训（它是 A1 动作空间重构的落点，
    属正确性改造），差别只在「除专家外还改什么主干参数」。
    """
    fm = policy.model
    for p in policy.parameters():
        p.requires_grad_(False)

    mods = [fm.vlm_with_expert.lm_expert, fm.action_in_proj, fm.action_out_proj,
            fm.action_time_mlp_in, fm.action_time_mlp_out]
    if mode == "lora":
        import smolvla_lora as LoRA
        n, npar = LoRA.inject_lora(policy.model, r=int(lora_r), alpha=float(lora_alpha),
                                   targets=tuple(lora_targets))
        LoRA.mark_only_lora_trainable(policy.model, extra_modules=mods)
        print(f"  [E2] LoRA 注入 {n} 层（targets={lora_targets}），"
              f"可训 LoRA 参数 {npar/1e6:.2f} M，r={lora_r}, α={lora_alpha}")
    elif mode == "full":
        for p in policy.parameters():
            p.requires_grad_(True)
    elif mode == "vision":
        for m in mods:
            for p in m.parameters():
                p.requires_grad_(True)
        # 解冻视觉塔最后 2 层
        try:
            vlayers = fm.vlm_with_expert.get_vlm_model().vision_model.encoder.layers
            for lyr in list(vlayers)[-2:]:
                for p in lyr.parameters():
                    p.requires_grad_(True)
        except Exception as e:
            print(f"  [警告] 解冻视觉层失败（退回 expert 模式）: {e}")
    else:
        for m in mods:
            for p in m.parameters():
                p.requires_grad_(True)

    tps = [p for p in policy.parameters() if p.requires_grad]
    n = sum(p.numel() for p in tps)
    print(f"  训练模式 {mode}: 可训参数 {n / 1e6:.2f} M / 总 {sum(p.numel() for p in policy.parameters()) / 1e6:.2f} M")
    return tps


# ---------------------------------------------------------------- 前向
def policy_loss(policy, data, idx):
    """一次训练前向。

    直接调用 `SmolVLAPolicy.forward(batch)` —— 它与官方训练路径完全一致
    （内部会走 prepare_images / prepare_state / prepare_action，并按
    `action_feature.shape[0]` 裁掉 padding 维度）。
    注意 `prepare_images` / `prepare_state` 定义在 **SmolVLAPolicy** 上，
    不在 `VLAFlowMatching` 上。
    """
    b = data.batch(idx)
    out = policy.forward(b, reduction="mean")
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def open_loop_mse(policy, data, rng, batches=6, batch=8, chunk=10, noise_draws=2):
    """开环动作预测误差（物理量，单位 m/s 与 rad/s），只在 chunk 前几步上评。

    **必须与部署条件一致**：flow matching 的推理从**随机噪声** `x_0 ~ N(0, I)` 出发，
    并在 `num_steps` 步内积分到动作。早先的实现传了 `noise=zeros`，等于把起点固定在
    0 上，样本难度远低于真实部署 —— 实测会让指标明显偏乐观（0.21 vs 0.35）。
    因此这里改为用默认随机噪声，并在多次噪声抽样上取平均。
    """
    policy.eval()
    errs, per_dim = [], None
    for _ in range(batches):
        idx = data.sample_index(batch, rng)
        b = data.batch(idx)
        images, img_masks = policy.prepare_images(b)
        state = policy.prepare_state(b)
        for _ in range(noise_draws):
            pred = policy.model.sample_actions(          # 不传 noise → 随机起点
                images, img_masks, b[LANG_TOKENS_KEY], b[LANG_MASK_KEY], state)
            p = data.denormalize_action(pred[:, :chunk])
            g = data.denormalize_action(b[ACTION_KEY][:, :chunk])
            e = (p - g).abs()
            errs.append(e.mean().item())
            d = e.mean(dim=(0, 1)).cpu().numpy()
            per_dim = d if per_dim is None else per_dim + d
    policy.train()
    mae = float(np.mean(errs))
    pd = per_dim / max(1, batches * noise_draws)
    return mae, pd


def build_state(obs, data, dev, det_feat=None):
    """按数据集的配置构造状态张量（训练与闭环评测必须走同一条路径）。

    当 `data.with_lidar=True` 时把 120 维激光拼到状态尾部 ——
    见 `SimDemoData` 中的说明（探针显示激光值 22 pp 的闭环成功率）。
    """
    st = np.asarray(obs["state"], dtype=np.float32)
    if getattr(data, "with_lidar", False):
        st = np.concatenate([st, np.asarray(obs["lidar"], dtype=np.float32)])
    if getattr(data, "with_det", False):
        st = np.concatenate([st, det_feat if det_feat is not None
                             else np.zeros(4, dtype=np.float32)])
    t = torch.from_numpy(st).to(dev)[None]
    t = (t - torch.as_tensor(data.st_mean, device=dev)) / \
        torch.as_tensor(data.st_std, device=dev)
    return torch.nn.functional.pad(t, (0, data.max_state_dim - t.shape[1]))


def frame_batch(hist, n_frames, dev):
    """从帧历史构造 camera1..K 的输入字典（不足 n_frames 时重复最早一帧）。"""
    out = {}
    n = len(hist)
    for k in range(int(n_frames)):
        j = max(0, n - 1 - k)
        out[IMG_KEYS[k]] = (torch.from_numpy(hist[j]).to(dev).float().div_(255.0)
                            .permute(2, 0, 1)[None])
    return out




def _slot_noise(cfg, ep_seed, step, dev, n=1):
    """按 (回合种子, 步数) 生成**显式**的 flow-matching 起始噪声。

    为什么必须显式传 `noise=` 而不是 `torch.manual_seed`：
    批量评测里一次 `sample_actions` 会为整批抽一次噪声，若靠全局种子，
    每个槽位分到的噪声就取决于**槽位调度顺序**，而调度又取决于各回合何时结束
    —— 那依赖模型本身。于是两个模型拿到的噪声不同，**配对又被破坏**。
    显式按 (seed, step) 生成则可以保证**逐槽、逐模型完全一致**，
    同时批量版与单环境版逐回合一致。
    """
    g = torch.Generator(device=dev).manual_seed(
        (int(ep_seed) * 1000003 + int(step) * 7919) % (2 ** 31 - 1))
    return torch.randn(n, cfg.chunk_size, cfg.max_action_dim, generator=g, device=dev)


def _sample_seed(ep_seed, step):
    """为 flow matching 的**随机起点**确定一个种子。

    `sample_actions` 不加 `noise` 时会从 `x_0 ~ N(0, I)` 采样，因此**同一模型跑
    同一回合会得到不同动作** —— 评测本身带随机性（实测同一配置两次跑出
    0.0% 与 12.5%）。这会给所有对比额外灌入方差，也让「复现」无从谈起。
    这里按 (回合种子, 步数) 定种，使评测**逐回合可复现**。
    注意模型仍在评估模式、不做 dropout，因此定种后完全确定。
    """
    return (int(ep_seed) * 1000003 + int(step) * 7919) % (2 ** 31 - 1)


@torch.no_grad()
def closed_loop_sr_batch(policy, data, episodes=200, seed=9000, max_steps=200,
                         level=None, n_frames=1, visible_start=False,
                         detail=None, batch=8, verbose=True,
                         det_mode="none", mem_ttl=25, calib=True, trace_n=0,
                         env_over=None):
    """**批量**闭环评测：同时跑 `batch` 个环境，把模型前向合并成一批。

    ## 为什么需要

    单环境版每步只做 batch=1 的前向（SigLIP + 10 步 flow matching ≈ 0.13 s/步），
    200 回合要 ~75 分钟。把 B 个环境并行后，**每步的墙钟成本几乎不变**，
    所以墙钟时间降到约 1/B —— 这是把统计功效做到 200 回合的前提。
    `batch=1` 时与 `closed_loop_sr` **逐回合一致**（同一套 seed、同一套 env 逻辑、
    同样贪心取 chunk 的第 0 步），已对拍验证。

    各环境的回合长度不同，因此采用「槽位 + 补齐」：某个槽位结束后立刻从待跑
    队列里取一个新回合补上，直到所有回合跑完。
    """
    from sim.env import SemanticNavEnv, calibrated_env, navigate_metrics
    from sim.run_eval import goal_visible

    def _mk_env():
        # `calib=True` → 标定场地（单房间 8×6 m、0 行人）；
        # `calib=False` → 默认多房间 + 2 行人（**域泛化评测**用）。
        # 早先这里硬编码 `calibrated_env`，导致「多房间评测」实际仍在标定场地上跑
        #（数字对不上才发现）。
        # `env_over`：布局 / 退化矩阵的额外环境开关（layout、exposure、glare、
        # n_peds、frame_drop …），透传给构造函数；与 `calib` 的基准配置叠加。
        ov = env_over or {}
        if calib:
            return calibrated_env(seed=seed, max_steps=max_steps, task_level=level, **ov)
        return SemanticNavEnv(seed=seed, n_peds=2, max_steps=max_steps,
                              task_level=level, **ov)

    policy.eval()
    if detail is None:
        detail = []
    results = {}
    _traces = []

    def make_episode(ep):
        env = _mk_env()
        ep_seed = seed + ep * 37
        obs = env.reset(seed=ep_seed, level=level)
        if visible_start:
            tr = 0
            while not goal_visible(env) and tr < 40:
                tr += 1
                ep_seed = seed + ep * 37 + 100000 * tr
                obs = env.reset(seed=ep_seed, level=level)
            if not goal_visible(env):
                return None
        if env.task is None:
            return None
        # **每个回合一份独立的记忆**：记忆是「指令所指目标」的跨帧累积，
        # 跨回合复用会污染（上一回合的物体残留到下一回合）。见 `TargetMemory`。
        # **失败模式诊断用的逐回合记账**（见文件末尾 `detail` 落盘处）：
        # `ever_found` = 感知槽是否**曾经**检出目标（区分「看不见」与「看见了但走不到」）
        # `min_dsurf` = 全程离目标包围盒**最近**的表面距离（成功判据就是这个量 ≤ 0.30）
        return {"env": env, "obs": obs, "hist": [], "steps": 0,
                "seed": ep_seed, "ep": ep,
                "trace": [],          # 末段逐步轨迹（失败模式诊断用）
                "ever_found": False, "min_dsurf": float("inf"),
                "mem": (TargetMemory(ttl=mem_ttl) if det_mode == "mem" else None)}

    slots, next_ep = [], 0

    def fill():
        nonlocal next_ep
        while len(slots) < batch and next_ep < episodes:
            s = make_episode(next_ep)
            next_ep += 1
            if s is not None:
                slots.append(s)

    fill()
    keys = [IMG_KEYS[k] for k in range(int(n_frames))]
    t0 = time.time()
    while slots:
        imgd = {k: [] for k in keys}
        sts, langs = [], []
        for s in slots:
            s["hist"].append(s["obs"]["image"])
            if len(s["hist"]) > n_frames:
                s["hist"].pop(0)
            dfeat = target_features(
                s["obs"]["image"], s["obs"]["instruction"],
                pose=np.concatenate([s["env"].pose[:2], [s["env"].pose[2]]]),
                mode=det_mode, memory=s["mem"], t=s["steps"]) \
                if getattr(data, "with_det", False) else None
            # 该帧的感知槽是否命中目标（第 0 维 = 「是否检出」）
            if dfeat is not None and float(dfeat[0]) > 0.5:
                s["ever_found"] = True
            sts.append(build_state(s["obs"], data, DEV, dfeat))
            langs.append(s["obs"]["instruction"])
            fb = frame_batch(s["hist"], n_frames, DEV)
            for k in keys:
                imgd[k].append(fb[k])
        imgd = {k: torch.cat(v, 0) for k, v in imgd.items()}
        st_t = torch.cat(sts, 0)
        lang_t, lang_m = data.tokenize(langs)
        images, img_masks = policy.prepare_images(imgd)
        state_p = policy.prepare_state({STATE_KEY: st_t})
        noise = torch.cat([_slot_noise(policy.config, sl["seed"], sl["steps"], DEV)
                           for sl in slots], 0)
        pred = policy.model.sample_actions(images, img_masks, lang_t, lang_m, state_p,
                                           noise=noise)

        for i, s in enumerate(slots):
            a = data.denormalize_action(pred[i, 0]).cpu().numpy()
            a = np.clip(a, [-s["env"].cfg["v_max"], -s["env"].cfg["w_max"]],
                        [s["env"].cfg["v_max"], s["env"].cfg["w_max"]])
            s["obs"], r, done, info = s["env"].step(a)
            s["steps"] += 1
            _ds = info.get("dist_surf")
            if _ds == _ds:                       # 非 NaN 才计入
                s["min_dsurf"] = min(s["min_dsurf"], float(_ds))
            # 末段轨迹：记录 (步, v_cmd, ω_cmd, 真值表面距, 感知槽距离, 是否检出)
            # —— 用于回答「模型在最后 1 米到底在做什么」：停下？绕圈？过冲？
            if trace_n:
                s["trace"].append([int(s["steps"]), float(a[0]), float(a[1]),
                                   (None if _ds != _ds else float(_ds)),
                                   (None if dfeat is None else float(dfeat[3]) * 12.0),
                                   (None if dfeat is None else int(float(dfeat[0]) > 0.5))])
                if len(s["trace"]) > 30:
                    s["trace"].pop(0)
            s["done"] = bool(done) or s["steps"] >= max_steps
            if s["done"]:
                m = navigate_metrics(info)
                # 元组尾部携带**任务层级 + 失败模式诊断量**（见下方落盘处的说明）
                results[s["seed"]] = (s["ep"], float(m["SR"]), float(m["SPL"]),
                                      float(m["NE"]), str(info.get("level", "?")),
                                      int(s["ever_found"]),
                                      float(s["min_dsurf"]),
                                      int(bool(info.get("timeout", False))),
                                      int(info.get("n_collide", 0)))
                if trace_n:
                    _traces.append({"seed": int(s["seed"]), "sr": float(m["SR"]),
                                    "min_dsurf": float(s["min_dsurf"]),
                                    "ncol": int(info.get("n_collide", 0)),
                                    "trace": s["trace"]})
        slots = [s for s in slots if not s["done"]]
        fill()
        if verbose and len(results) and len(results) % 25 == 0 and \
                len(results) != getattr(closed_loop_sr_batch, "_last", -1):
            closed_loop_sr_batch._last = len(results)
            print(f"    批量评测 {len(results)}/{episodes} 回合 "
                  f"({time.time() - t0:.0f}s)")

    if trace_n:
        _tf = f"export/eval_detail/trace_{os.path.basename(a.eval_only)}.json"
        _failed = [s for s in _traces if s["sr"] == 0][:trace_n]
        with open(_tf, "w") as fh:
            json.dump({"model": a.eval_only, "n_fail_traced": len(_failed),
                       "fields": ["step", "v_cmd", "w_cmd", "true_dsurf",
                                  "feat_dist", "found"],
                       "episodes": _failed}, fh)
        print(f"  末段轨迹已写入 {_tf}（{len(_failed)} 个失败回合 × 最多 30 步）")

    _keys = sorted(results, key=lambda z: results[z][0])
    rows = [results[k] for k in _keys]

    def _rec(k, v):
        # 每回合记录：`[seed, SR, level, ever_found, min_dsurf, timeout, n_collide]`
        # - `level`：混合层级的平均值会掩盖结构（实测 L3 反而最好、L2 最差）
        # - `ever_found` / `min_dsurf`：**失败模式分解**用 —— 区分
        #   「从未检出目标」（感知/搜索失败）与「检出了但走不到」（控制/决策失败）。
        #   这是「模型只达到示范 67%」这一差距的定位手段。
        # 注意 `inf` 不能进 JSON（非标准），用 99.0 表示「从未记录到」。
        md = float(v[6])
        return (int(k), float(v[1]), str(v[4]), int(v[5]),
                (99.0 if md != md or md == float("inf") else md),
                int(v[7]), int(v[8]))

    # **批量路径与单环境路径是两处独立的记录点**，改一处不够（踩过）。
    detail.extend(_rec(k, v) for k, v in zip(_keys, rows))
    sr = float(np.mean([r[1] for r in rows])) if rows else 0.0
    spl = float(np.nanmean([r[2] for r in rows])) if rows else 0.0
    ne = float(np.mean([r[3] for r in rows])) if rows else 0.0
    return sr, spl, ne


@torch.no_grad()
def closed_loop_sr(policy, data, episodes=12, seed=9000, max_steps=400, level=None,
                   n_frames=1, visible_start=False, detail=None,
                   det_mode="none", mem_ttl=25, calib=True):
    """`detail` 传入一个 list 时，会写入每回合的 (seed, success) —— 供**配对检验**用。
    两个模型用同一套回合种子，因此可以直接做 McNemar 而不是独立的 Fisher。"""
    """闭环导航成功率：在自建仿真里真跑策略。"""
    from sim.env import SemanticNavEnv, calibrated_env, navigate_metrics
    # 用标定后的基准配置（见 主创新A_诊断报告.md §4.5）：
    # 开阔场地 + 「到物体**表面**距离 ≤ 0.30 m」的成功判据。
    # 旧配置用「到中心点距离」，而中心在实心物体内部、几何上不可达，
    # 导致测试床完全没有语义区分度（oracle 30% vs random_target 17.5%）。
    env = (calibrated_env(seed=seed, max_steps=max_steps, task_level=level) if calib
           else SemanticNavEnv(seed=seed, n_peds=2, max_steps=max_steps,
                               task_level=level))
    policy.eval()
    srs, spls, nes = [], [], []
    n_steps = n_stuck = 0
    if detail is None:
        detail = []
    sum_v = sum_w = 0.0
    from sim.run_eval import goal_visible
    n_skip = 0
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep * 37, level=level)
        ep_seed = seed + ep * 37
        if visible_start:
            tr = 0
            while not goal_visible(env) and tr < 40:
                tr += 1
                ep_seed = seed + ep * 37 + 100000 * tr
                obs = env.reset(seed=ep_seed, level=level)
            if not goal_visible(env):
                n_skip += 1
                continue
        if env.task is None:
            continue
        hist = []                                   # 帧历史（帧堆叠用）
        _mem = TargetMemory(ttl=mem_ttl) if det_mode == "mem" else None
        # 循环变量必须是 `step`：下面 `_slot_noise(..., ep_seed, step, ...)` 要用它
        # 按步定种。早先写成 `for _ in range(...)`，导致**最终评测一跑就 NameError**，
        # 而 checkpoint 保存在评测**之后** → 整轮训练白跑（已实测损失两轮）。
        for step in range(max_steps):
            hist.append(obs["image"])
            if len(hist) > n_frames:
                hist.pop(0)
            dfeat = target_features(
                obs["image"], obs["instruction"],
                pose=np.concatenate([env.pose[:2], [env.pose[2]]]),
                mode=det_mode, memory=_mem, t=step) \
                if getattr(data, "with_det", False) else None
            st = build_state(obs, data, DEV, dfeat)
            lang_t, lang_m = data.tokenize([obs["instruction"]])
            images, img_masks = policy.prepare_images(frame_batch(hist, n_frames, DEV))
            state_p = policy.prepare_state({STATE_KEY: st})
            noise = _slot_noise(policy.config, ep_seed, step, DEV)
            pred = policy.model.sample_actions(images, img_masks, lang_t, lang_m, state_p,
                                               noise=noise)
            a = data.denormalize_action(pred[0, 0]).cpu().numpy()
            a = np.clip(a, [-env.cfg["v_max"], -env.cfg["w_max"]],
                        [env.cfg["v_max"], env.cfg["w_max"]])
            obs, r, done, info = env.step(a)
            # 诊断：模型到底在输出什么动作？「几乎不动」与「乱动」是两种完全不同的失败
            n_steps += 1
            sum_v += abs(float(a[0])); sum_w += abs(float(a[1]))
            if abs(float(a[0])) < 0.05:
                n_stuck += 1
            if done:
                break
        m = navigate_metrics(info)
        srs.append(m["SR"]); spls.append(m["SPL"]); nes.append(m["NE"])
        # 连同**任务层级**一起落盘（见 `sim/run_eval.py` 同处的说明）：
        # 混合层级的平均值会掩盖「差距集中在 L3」这类结构，按层拆分才看得出。
        detail.append((int(ep_seed), float(m["SR"]), str(info.get("level", "?"))))
    policy.train()
    if n_steps:
        print(f"    [闭环诊断] 总步数 {n_steps} | |v|<0.05 占比 {100 * n_stuck / n_steps:.1f}%"
              f" | 平均 |v| {sum_v / n_steps:.4f} m/s | 平均 |ω| {sum_w / n_steps:.4f} rad/s"
              f"（示范动作参考：|v|≈0.27, |ω|≈0.42）")
    return float(np.mean(srs)), float(np.nanmean(spls)), float(np.mean(nes))


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="export/nav_demos.h5")
    ap.add_argument("--stats-from", default=None,
                    help="冻结归一化统计量：从该 h5（如纯仿真 nav_detf.h5）复算 "
                         "st/act 均值方差并覆盖 --data 的重算值。用于共训场景："
                         "训练/评测/部署全程用 sim 口径统计量，真机数据当作"
                         "「略带偏移的增广」，避免统计量被 real 帧拉偏（E10）")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--mode", default="expert",
                    choices=["expert", "vision", "full", "lora"],
                    help="微调范式（E2 自变量）：expert=仅动作专家 / lora=专家+VLM 上挂 LoRA"
                         " / vision=专家+解冻视觉塔末层 / full=全参")
    ap.add_argument("--lora-r", type=int, default=8, help="LoRA 秩 r")
    ap.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA 缩放 α")
    ap.add_argument("--data-frac", type=float, default=1.0,
                    help="E2 数据量子：按**回合内取前 frac 比例**的起点（保持场景分布不变）")
    ap.add_argument("--lora-targets", default="vlm_attn",
                    help="LoRA 挂点：vlm_attn / vlm_mlp / expert / all")
    ap.add_argument("--chunk", type=int, default=50, help="训练用的动作 chunk 长度")
    ap.add_argument("--n-frames", type=int, default=1,
                    help="帧堆叠帧数：>1 时把 t、t-1、… 注册成 camera1/2/3…（缓解部分可观测）")
    ap.add_argument("--exec-steps", type=int, default=1,
                    help="E8：一次推理执行多少步动作（chunk 缓存）。k>1 时推理次数降为 1/k，"
                         "但反应性下降 —— 这是「推理次数 vs 反应性」的权衡实验")
    ap.add_argument("--trace-fail", type=int, default=0,
                    help="记录指定数量**失败回合**的逐步轨迹（末段行为诊断）。"
                         "写出 export/eval_detail/trace_<model>.json，不动主结果文件")
    ap.add_argument("--eval-multiroom", action="store_true",
                    help="在**默认多房间 + 2 行人**场地上评测（域泛化），"
                         "而不是标定场地（单房间 8×6 m、0 行人）。"
                         "注意两条评测路径都要走同一场地，否则数字不可比")
    ap.add_argument("--eval-seed", type=int, default=9000,
                    help="闭环评测的**回合集基种子**（回合 seed = base + 37·ep）。"
                         "默认 9000 即论文口径。取别的值可**追加不相交的回合**，"
                         "与已有结果合并以提高配对检验的功效"
                         "（例：--eval-seed 16400 --eval-episodes 200 正好接在 9000..16363 之后）")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--visible-start", action="store_true",
                    help="评测起点即目标可见（把搜索从评测里分离出去，只测导航）")
    ap.add_argument("--eval-batch", type=int, default=1,
                    help="闭环评测并行环境数（>1 走批量评测，墙钟时间约降到 1/N，"
                         "结果与单环境逐回合一致）")
    ap.add_argument("--eval-max-steps", type=int, default=400,
                    help="闭环评测的步数上限（每步都要过一遍 SigLIP，是评测的主要开销）")
    ap.add_argument("--num-steps", type=int, default=None,
                    help="flow matching 的去噪步数（官方默认 10）。"
                         "探针显示一个直接回归器（一次前向）就能到 39%%，"
                         "而 VLA 只有 18.3%%，因此**采样过程**是首要嫌疑。"
                         "降到 1–2 可验证这一点，且无需重训（--eval-only 即可）")
    ap.add_argument("--lidar-in-state", action="store_true",
                    help="**把 120 维激光雷达拼进状态向量**（状态维 32 → 128）。"
                         "探针显示「有无激光」值 22 pp 的闭环成功率（§4.7），"
                         "而 VLA 原本只吃 RGB、拿不到这个几何模态。"
                         "实现上只需重建 state_proj 的输入维，不动视觉塔")
    ap.add_argument("--det-features-mode", default=None, choices=["none", "mem"],
                    help="感知特征档位（须与采集时一致）：`none`=4 维单帧；"
                         "`mem`=8 维（并列给出跨帧记忆的目标位置）。"
                         "`mem` 用于补齐模型相对示范缺失的时序记忆 —— "
                         "实测关掉示范的记忆会使它 33.0% → 16.0%")
    ap.add_argument("--mem-ttl", type=int, default=25,
                    help="记忆存活帧数，须与示范策略一致（默认 25）")
    ap.add_argument("--det-features", action="store_true",
                    help="把**感知模块输出**（4 维）拼进状态向量（感知+决策分离架构）。"
                         "对应采集时的 --det-features")
    ap.add_argument("--eval-only", default=None,
                    help="只加载该目录下的 policy.pt 做评测，不训练")
    ap.add_argument("--out", default="export/ft_expert")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--idas-variant", default="none", choices=list(__import__("smolvla_idas").VARIANTS),
                    help="E5：IDAS 变体（none / idas_v1 / idas_v2 / random_ctrl）")
    ap.add_argument("--idas-lam", type=float, default=0.1, help="E5：辅助损失权重 λ_id")
    ap.add_argument("--idas-label-p", type=float, default=1.0,
                    help="E5：动作标签比例 p。**主损失与辅助损失同时缺失**（下界实验）")
    a = ap.parse_args()

    print(f"设备: {DEV}")
    # 仅评测时**先把 checkpoint 的配置读出来**：帧堆叠数与训练数据必须与训练时一致，
    # 否则输入键对不上或归一化错位（两者都会让结果静默失真）。
    if a.eval_only:
        _ck = torch.load(os.path.join(a.eval_only, "policy.pt"), map_location="cpu")
        nf = int(_ck.get("n_frames", 1) or 1)
        if nf != a.n_frames:
            print(f"  [提示] checkpoint 的 n_frames={nf}，已采用（帧堆叠必须与训练一致）")
            a.n_frames = nf
        if _ck.get("data") and os.path.abspath(_ck["data"]) != os.path.abspath(a.data):
            print(f"  [提示] checkpoint 的训练数据为 {_ck['data']}，评测将使用其统计量")
        wd = bool(_ck.get("with_det", False))
        if wd != a.det_features:
            print(f"  [提示] checkpoint 的 with_det={wd}，已采用（必须与训练一致）")
            a.det_features = wd
        _dm = _ck.get("det_features_mode")
        if _dm and str(_dm) != str(a.det_features_mode or "none"):
            print(f"  [提示] checkpoint 的感知档位={_dm}，已采用（必须与训练一致）")
            a.det_features_mode = str(_dm)
            if a.det_features_mode == "mem":
                a.det_features = True
        if _ck.get("mem_ttl") and int(_ck["mem_ttl"]) != a.mem_ttl:
            print(f"  [提示] checkpoint 的 mem_ttl={_ck['mem_ttl']}，已采用")
            a.mem_ttl = int(_ck["mem_ttl"])
        wl = bool(_ck.get("with_lidar", False))
        if wl != a.lidar_in_state:
            print(f"  [提示] checkpoint 的 with_lidar={wl}，已采用（必须与训练一致）")
            a.lidar_in_state = wl
        wd = bool(_ck.get("with_det", False))
        if wd != a.det_features:
            print(f"  [提示] checkpoint 的 with_det={wd}，已采用（必须与训练一致）")
            a.det_features = wd
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    policy = SmolVLAPolicy.from_pretrained(CKPT).to(DEV)
    policy.train()
    cfg = policy.config
    if a.chunk and a.chunk != cfg.chunk_size:
        print(f"  chunk_size {cfg.chunk_size} → {a.chunk}")
        cfg.chunk_size = a.chunk
        cfg.n_action_steps = a.chunk
    if a.num_steps and a.num_steps != cfg.num_steps:
        print(f"  flow matching 去噪步数 num_steps {cfg.num_steps} → {a.num_steps}")
        cfg.num_steps = a.num_steps

    # 状态维：默认 32（够放 pose 3 + last_action 2 = 5）。
    # 启用附加模态后要重算：激光 120 维、感知 4 维。
    # 感知特征维度由档位决定：`none`=4（单帧）、`mem`=8（含跨帧记忆）
    # **`--det-features-mode` 任一取值都隐含 `--det-features`**。
    # 早先只让 `mem` 隐含，结果 `--det-features-mode none` 单独用时**根本没启用感知特征**
    # （状态仍是 5 维），把「记忆 vs 单帧」的对照悄悄变成了「记忆+感知 vs 什么都不给」——
    # 一个静默的开关不一致，与 gen_dataset 里 payload 漏传那处同类（见该文件注释）。
    if a.det_features_mode:
        a.det_features = True
    elif a.det_features:
        a.det_features_mode = "none"
    _ndet = 0 if not a.det_features else DET_DIM[a.det_features_mode]
    _need = 5 + (120 if a.lidar_in_state else 0) + _ndet
    state_dim = None
    if _need > 32:
        state_dim = 1
        while state_dim < _need:
            state_dim *= 2

    print("\n=== ① 动作空间对齐 ===")
    before = (cfg.max_action_dim, len(cfg.image_features), cfg.action_feature.shape)
    reconfigure_for_diff_drive(policy, action_dim=2, n_cam=a.n_frames, state_dim=state_dim)
    if a.lidar_in_state or a.det_features:
        extra = []
        if a.lidar_in_state:
            extra.append("120 维激光")
        if a.det_features:
            # 档位必须回显：`none`=4 维单帧、`mem`=8 维（多一个跨帧记忆槽）。
            # 采数据与训练两端的档位不一致会静默退化（维度对不上才会报错），
            # 所以在这里显式打印，便于核对。
            extra.append(f"{_ndet} 维感知输出（档位 {a.det_features_mode}："
                         f"目标槽{' + 跨帧记忆槽' if a.det_features_mode == 'mem' else ''}）")
        print(f"  状态向量       5 → {_need}（pose + 上一步动作 + {' + '.join(extra)}，"
              f"pad 到 {state_dim}）")
    after = (cfg.max_action_dim, len(cfg.image_features), cfg.action_feature.shape)
    print(f"  max_action_dim {before[0]} → {after[0]}")
    print(f"  相机路数       {before[1]} → {after[1]}" +
          (f"（帧堆叠 {a.n_frames} 帧：camera1=t, camera2=t-1, …）" if a.n_frames > 1 else ""))
    print(f"  动作维度       {before[2]} → {after[2]}")

    print("\n=== ② 数据 ===")
    data = SimDemoData(a.data, device=DEV, max_state_dim=cfg.max_state_dim,
                       tokenizer_max_length=cfg.tokenizer_max_length,
                       chunk_size=a.chunk, max_action_dim=2, n_frames=a.n_frames,
                       with_lidar=a.lidar_in_state, with_det=a.det_features)
    print("  " + data.summary().replace("\n", "\n  "))
    if a.stats_from:
        # 冻结统计量：从参考 h5 直接复算（不建第二个 SimDemoData——images 会整块
        # 进内存，白吃 ~5.5GB）。拼装口径必须与 SimDemoData.__init__ 完全一致：
        # states → [+lidar] → [+det_feat]，mean/std 同 lines 115-118。
        import h5py as _h5
        with _h5.File(a.stats_from, "r") as _f:
            _st = _f["states"][()].astype(np.float32)
            _act = _f["actions"][()].astype(np.float32)
            if a.lidar_in_state:
                if "lidar" not in _f:
                    raise ValueError(f"{a.stats_from} 缺 lidar，无法对齐状态维度")
                _st = np.concatenate([_st, _f["lidar"][()].astype(np.float32)], axis=1)
            if a.det_features:
                if "det_feat" not in _f:
                    raise ValueError(f"{a.stats_from} 缺 det_feat，无法对齐状态维度")
                _st = np.concatenate([_st, _f["det_feat"][()].astype(np.float32)], axis=1)
        data.act_mean = _act.mean(0)
        data.act_std = _act.std(0) + 1e-6
        data.st_mean = _st.mean(0)
        data.st_std = _st.std(0) + 1e-6
        print(f"  [统计量冻结] 来自 {a.stats_from}（sim 口径）：")
        print(f"    act_mean={np.round(data.act_mean, 4).tolist()} "
              f"act_std={np.round(data.act_std, 4).tolist()}")
        print(f"    st_mean[:5]={np.round(data.st_mean[:5], 4).tolist()}")
    rng = np.random.default_rng(a.seed)

    # 仅评测：加载已训练的 checkpoint，用**统一口径**重评（便于修正指标后回溯既有结果）
    if a.eval_only:
        ck = torch.load(os.path.join(a.eval_only, "policy.pt"), map_location=DEV)
        if ck.get("zeroshot"):
            # E5 外部锚点：**不加载任何权重**，直接用预训练 SmolVLA（models/smolvla_base）
            # 跑同一闭环协议 —— 证明「域迁移微调 + 感知分离架构」的必要性下界。
            print("  [zero-shot] 使用预训练 SmolVLA，不加载任何微调权重")
        elif "model" in ck:
            miss, unexp = policy.model.load_state_dict(ck["model"], strict=False)
            if miss or unexp:
                print(f"  [提示] 加载 state_dict：缺失 {len(miss)} 项 / 多余 {len(unexp)} 项")
        else:                      # 兼容旧格式（缺 action_time_mlp_*）
            print("  [警告] 旧格式 checkpoint（缺 action_time_mlp_*），结果不完全等价于训练态")
            policy.model.vlm_with_expert.lm_expert.load_state_dict(ck["lm_expert"])
            policy.model.action_in_proj.load_state_dict(ck["action_in_proj"])
            policy.model.action_out_proj.load_state_dict(ck["action_out_proj"])
        policy.eval()

        # **归一化统计量必须与训练时一致**，否则动作反归一化与状态归一化都会错位，
        # 评测结果无效（实测踩过：ω 标准差 0.556 vs 0.870 差 1.57 倍，
        # 同一模型评出 8.3% 与 25.0% 两个数）。
        ck_data = ck.get("data")
        if ck.get("zeroshot"):
            # zero-shot 锚点：dummy ckpt 不携带数据路径，统计量强制取本次评测数据（同一口径）
            ck_data = a.data
        # 若训练时冻结了统计量（stats_from），传入该 h5 即视为口径一致，
        # 不触发下面的「自动改用训练数据」——那会把 sim 冻结统计量换回混合统计量。
        _sf = ck.get("stats_from")
        if _sf and os.path.abspath(a.data) == os.path.abspath(_sf):
            ck_data = a.data
        if ck_data and os.path.abspath(ck_data) != os.path.abspath(a.data):
            print(f"\n  [!! 警告 !!] --data 与训练数据不一致：")
            print(f"      本次传入 : {a.data}")
            print(f"      训练使用 : {ck_data}")
            print("      已自动改用**训练数据**的统计量，避免评测口径错位。")
            data = SimDemoData(ck_data, device=DEV, max_state_dim=cfg.max_state_dim,
                               tokenizer_max_length=cfg.tokenizer_max_length,
                               chunk_size=a.chunk, max_action_dim=2, n_frames=a.n_frames,
                               with_lidar=a.lidar_in_state, with_det=a.det_features)
            print("  " + data.summary().replace("\n", "\n  "))
        print(f"\n=== 仅评测（{a.eval_only}，数据 {a.data}，"
              f"num_steps={cfg.num_steps}，起点可见={a.visible_start}）===")
        m, pd = open_loop_mse(policy, data, rng)
        det = []
        _fn = closed_loop_sr_batch if a.eval_batch > 1 else closed_loop_sr
        _kw = dict(batch=a.eval_batch) if a.eval_batch > 1 else {}
        # 感知档位/记忆参数必须与训练一致，且两条评测路径都要传（曾漏传导致档位静默退回）
        _kw.update(det_mode=(a.det_features_mode or "none"), mem_ttl=a.mem_ttl,
                   calib=not a.eval_multiroom, trace_n=int(a.trace_fail))
        sr, spl, ne = _fn(policy, data, episodes=a.eval_episodes,
                          max_steps=a.eval_max_steps, n_frames=a.n_frames,
                          visible_start=a.visible_start, detail=det,
                          seed=a.eval_seed, **_kw)
        # 文件名必须区分场地与步数预算：不同设置的评测**不得覆盖**论文口径的结果。
        # 踩过的坑：一次「多房间」实验误用了标定场地，却仍写进 `{m}_std.json`，
        # 把 200 步的论文口径结果**直接覆盖**掉了（只能重跑恢复）。
        tag = ("vis" if a.visible_start else "std")
        if a.eval_multiroom:
            tag = "mr_" + tag
        _settings = {"max_steps": int(a.eval_max_steps),
                     "multiroom": bool(a.eval_multiroom),
                     "visible_start": bool(a.visible_start)}
        os.makedirs("export/eval_detail", exist_ok=True)
        _nm = os.path.basename(a.eval_only)
        # 论文口径（base=9000）写主文件；追加批次写 `_s<base>` 后缀文件，
        # 由 plot_results.load() 自动合并 —— 追加回合不得覆盖主结果。
        fp = (f"export/eval_detail/{_nm}_{tag}.json" if a.eval_seed == 9000
              else f"export/eval_detail/{_nm}_{tag}_s{a.eval_seed}.json")
        # **写保护**：若目标文件已存在且其记录设置与本轮不同，则改写到带后缀的新文件。
        # 没有这道保护时，一次「标定 250 步」的实验会把「标定 200 步」的论文口径
        # 结果**静默覆盖**（已实测毁掉 4 个模型的主结果，只能重跑 108 分钟恢复）。
        if os.path.exists(fp):
            try:
                with open(fp) as fh:
                    _old = json.load(fh).get("settings")
            except Exception:
                _old = None
            if _old is not None and _old != _settings:
                alt = fp[:-5] + "_ms%d%s.json" % (
                    _settings["max_steps"], "_mr" if _settings["multiroom"] else "")
                print(f"  [!! 写保护 !!] {fp} 已存在且设置为 {_old}，与本轮 {_settings} 不同；"
                      f"\n                本轮改写到 {alt}（不覆盖论文口径结果）")
                fp = alt
        with open(fp, "w") as fh:
            json.dump({"model": a.eval_only, "protocol": tag,
                       "episodes": len(det), "detail": det,
                       "settings": _settings}, fh)
        print(f"  每回合结果已写入 {fp}（{len(det)} 回合，成功率 "
              f"{100*np.mean([x[1] for x in det]) if det else 0:.1f}%）")
        print(f"  开环 MAE {m:.4f}  (v {pd[0]:.4f} / ω {pd[1]:.4f})")
        print(f"  闭环 SR  {100 * sr:.1f}%  SPL {spl:.3f}  NE {ne:.2f} m")
        return

    # **E2 数据量子必须在这里真正生效**。
    # ⚠️ 早先这一段是用 `str.replace` 插入的，锚点没匹配上于是**静默失败**——
    # `--data-frac 0.4` 实际上从未被应用，整批 f06/f04 训练跑的都是 100% 数据，
    # 且 `expert@40%` 与 `expert@100%` 的 checkpoint **权重完全相同**
    # （200 回合逐回合 b=0 c=0），是「配置没生效」而不是「数据不影响结果」。
    # 教训与 §4.15 的 NN 检测器同源：**用 replace 改代码必须校验落点**。
    if getattr(a, "data_frac", 1.0) < 1.0:
        _nkept = data.set_data_fraction(a.data_frac)
        print(f"  [E2] 数据量子 frac={a.data_frac}: 可采样起点 "
              f"{data.valid_idx_full.size} → {_nkept}（按回合等比，场景分布不变）")

    print("\n=== ③ 可训参数 ===")
    tps = set_trainable(policy, a.mode, lora_r=a.lora_r,
                       lora_alpha=a.lora_alpha,
                       lora_targets=tuple(a.lora_targets.split(",")))

    print(f"\n=== ④ 微调前基线 ===")
    mse0, pd0 = open_loop_mse(policy, data, rng)
    print(f"  开环动作 MAE: {mse0:.4f}  (v {pd0[0]:.4f} / ω {pd0[1]:.4f})")

    opt = torch.optim.AdamW(tps, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)

    # ---------- E5：IDAS 辅助监督 ----------
    import smolvla_idas as idas
    _hook = None
    idas_head = None
    idas_head = None
    idas_hook = None
    if a.idas_variant != "none":
        idas_hook = idas.PrefixFeatureCapture(policy.model)
        # **头的输入维惰性确定**：前缀特征来自 **VLM 主干**（实测 1920 维），
        # 不是动作专家的 `expert_hidden_size`（720）—— 早先按后者建头，
        # 第一次前向就报 `mat1 and mat2 shapes cannot be multiplied (4x1920 ...)`。
        print(f"  [E5] IDAS 变体={a.idas_variant}  λ_id={a.idas_lam}  "
              f"标签比例 p={a.idas_label_p}（头输入维待首次前向确定）")

    def _ensure_idas_head(fdim):
        nonlocal idas_head
        if idas_head is not None:
            return
        idas_head = idas.InverseDynamicsHead(fdim, action_dim=int(data.max_action_dim)).to(DEV)
        # **头也进优化器**：不训头的话 λ 再大也不起作用。
        # ⚠️ 只能 **追加** 参数组，绝不能 `opt.param_groups = []` 再加 ——
        # 那会把主模型参数从优化器里**整个清掉**，于是主干从未更新：
        # 三个变体的 checkpoint 字节级相同、闭环 SR 全部 10.5%（未训练水平）。
        # 实测踩过，整轮 E5 结果作废一次。
        opt.add_param_group({"params": list(idas_head.parameters())})
        print(f"  [E5] 头输入维 = {2*fdim}（VLM 前缀特征），已追加进优化器"
              f"（共 {sum(len(g['params']) for g in opt.param_groups)} 个参数张量组）")

    hist = []
    _idas_last = float("nan")
    t0 = time.time()
    print(f"\n=== ⑤ 微调（{a.steps} 步, batch={a.batch}, lr={a.lr}, mode={a.mode}）===")
    for step in range(a.steps):
        idx = data.sample_index(a.batch, rng)
        if a.idas_variant == "none" and a.idas_label_p >= 1.0:
            loss = policy_loss(policy, data, idx)
        else:
            # **标签比例 p < 1 时走 per-sample 路径**：主损失与辅助损失用同一张掩码，
            # 这样「下界实验」才成立（若只掩一边，两个效应就分不开了）。
            b_ = data.batch(idx)
            per_s, _ = policy.forward(b_, reduction="none")
            _rng = np.random.default_rng(a.seed * 7919 + step)
            _m = torch.as_tensor(idas.label_mask(a.idas_label_p, len(idx), _rng),
                                 device=per_s.device, dtype=per_s.dtype)
            loss = (per_s * _m).sum() / _m.sum().clamp_min(1.0)
            if idas_hook is not None:
                # **顺序很重要**：主前向刚跑完，捕获器里此刻是 f_t —— 先取出来，
                # 再跑「下一帧」的前向拿 f_{t+1}（否则会被覆盖）。
                _nxt, _ok = idas.next_frame_indices(idx, data.ep_off, chunk=int(data.chunk))
                if _ok.any():
                    f_t = idas.frame_feature(idas_hook.prefix)[np.asarray(_ok)]
                    _ensure_idas_head(int(f_t.shape[-1]))
                    _b2 = data.batch(np.asarray(_nxt)[_ok])
                    policy.forward(_b2, reduction="none")           # 仅为截获 f_{t+1}
                    f_t1 = idas.frame_feature(idas_hook.prefix)
                    _a = idas.action_between(data, np.asarray(idx)[_ok],
                                             np.asarray(_nxt)[_ok])
                    _a = torch.as_tensor(_a, device=f_t.device, dtype=torch.float32)
                    # 标签比例 p 对辅助损失**同样生效**
                    _am = _m[_ok]
                    if _am.sum() > 0:
                        _l_id = idas.idas_loss(idas_head, f_t, f_t1, _a,
                                               a.idas_variant, 1.0)
                        _idas_last = float(_l_id.item())   # 不带 λ，便于与 λ 脱钩看
                        loss = loss + a.idas_lam * _l_id
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tps, 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == a.steps - 1:
            hist.append((step, float(loss.item())))
            _x = f"  L_idas={_idas_last:.5f}" if idas_hook is not None else ""
            print(f"  step {step:5d}/{a.steps}  loss {loss.item():.5f}"
                  f"{_x}  ({time.time() - t0:.0f}s)")
        if a.eval_every and step > 0 and step % a.eval_every == 0:
            mse, pdd = open_loop_mse(policy, data, rng)
            sr, spl, ne = closed_loop_sr(policy, data, episodes=a.eval_episodes,
                                         max_steps=a.eval_max_steps, n_frames=a.n_frames,
                                         visible_start=a.visible_start,
                                         det_mode=(a.det_features_mode or "none"),
                                         mem_ttl=a.mem_ttl,
                                         calib=not a.eval_multiroom)
            print(f"    [评测 step {step}] 开环 MAE {mse:.4f} "
                  f"(v {pdd[0]:.4f} / ω {pdd[1]:.4f}) | "
                  f"闭环 SR {100 * sr:.1f}% SPL {spl:.3f} NE {ne:.2f} m")

    # **先落盘，再评测**。顺序很关键：最终评测会真的跑 200 回合仿真，任何异常
    # 都会让 checkpoint 丢失。本项目实测因 `closed_loop_sr` 里一个 `NameError`
    # **白跑了两轮训练**（各 15 / 30 分钟）。评测可重跑，训练不行，故必须后置。
    os.makedirs(a.out, exist_ok=True)
    # **LoRA 先合并再保存**（E2 的 lora 范式）：
    # 把 (α/r)·B·A 折进 `base.weight`，落盘的就是一个普通满秩模型。
    # 必须这么做的原因：`--eval-only` 侧的模型**没有** LoRA 模块，加载时用的是
    # `load_state_dict(..., strict=False)` —— 若直接存 LoRA 权重，它们会落在
    # `unexpected_keys` 里被**静默丢弃**，评测得到的其实是「没适配的主干」。
    # 合并后 eval 与部署**完全不需要改动**。代价：不能从 checkpoint 续训（本课题不需要）。
    if a.mode == "lora":
        import smolvla_lora as LoRA
        n_merged = LoRA.merge_lora(policy.model)
        print(f"  [E2] LoRA 已合并进 {n_merged} 层权重后再落盘"
              f"（落盘为普通满秩模型，eval/部署无需改动）")
    # 保存**完整** model state_dict。早先只存了 lm_expert / action_{in,out}_proj，
    # 漏掉同样可训的 action_time_mlp_{in,out}，导致 --eval-only 复现的模型与训练时不一致。
    torch.save({"model": policy.model.state_dict(),
                "step": a.steps, "mode": a.mode, "chunk": a.chunk,
                "lora_r": int(a.lora_r), "lora_alpha": float(a.lora_alpha),
                "lora_targets": str(a.lora_targets),
                "action_dim": 2, "max_state_dim": cfg.max_state_dim,
                "tokenizer_max_length": cfg.tokenizer_max_length,
                "n_frames": int(a.n_frames),           # 帧堆叠数：评测必须与训练一致
                "with_lidar": bool(a.lidar_in_state),  # 状态是否含激光：评测必须与训练一致
                "with_det": bool(a.det_features),      # 状态是否含感知输出：同上
                "det_features_mode": str(a.det_features_mode or "none"),
                "mem_ttl": int(a.mem_ttl),             # 感知档位/记忆参数：评测必须一致
                "num_steps": int(cfg.num_steps),
                "data": os.path.abspath(a.data),       # 训练数据：评测必须对齐其归一化统计量
                "stats_from": (os.path.abspath(a.stats_from) if a.stats_from else None)},
               f"{a.out}/policy.pt")
    print(f"\n  已保存 {a.out}/policy.pt（在最终评测**之前**）")

    print(f"\n=== ⑦ 最终评测 ===")
    mse1, pd1 = open_loop_mse(policy, data, rng)
    sr0, spl0, ne0 = closed_loop_sr(policy, data, episodes=a.eval_episodes,
                                    max_steps=a.eval_max_steps, n_frames=a.n_frames,
                                    visible_start=a.visible_start,
                                    det_mode=(a.det_features_mode or "none"),
                                    mem_ttl=a.mem_ttl, calib=not a.eval_multiroom)
    print(f"  开环动作 MAE : {mse0:.4f} → {mse1:.4f}  (v {pd1[0]:.4f} / ω {pd1[1]:.4f})")
    print(f"  闭环导航 SR  : {100 * sr0:.1f}%  SPL {spl0:.3f}  NE {ne0:.2f} m")

    with open(f"{a.out}/meta.json", "w") as f:
        json.dump({"mode": a.mode, "steps": a.steps, "batch": a.batch, "lr": a.lr,
                   "chunk": a.chunk, "action_dim": 2,
                   "trainable_M": round(sum(p.numel() for p in tps) / 1e6, 2),
                   "mae_before": mse0, "mae_after": mse1,
                   "mae_before_per_dim": [float(x) for x in pd0],
                   "mae_after_per_dim": [float(x) for x in pd1],
                   "closed_loop_sr": sr0, "closed_loop_spl": spl0, "closed_loop_ne": ne0,
                   "eval_episodes": a.eval_episodes, "eval_max_steps": a.eval_max_steps,
                   "eval_metric": "随机噪声起点 + 多次抽样（与部署一致）",
                   "loss_history": hist, "data": a.data}, f, indent=2, ensure_ascii=False)
    print(f"\n已保存 {a.out}/policy.pt 与 meta.json")


if __name__ == "__main__":
    main()
