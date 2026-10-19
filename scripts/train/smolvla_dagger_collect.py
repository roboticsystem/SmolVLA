"""P0 · DAgger 式闭环纠正数据采集。

为什么需要（见 `主创新A_诊断报告.md` §2.4）：
    上界探针证明「开环拟合误差比 VLA 小 60 倍的学习器，闭环成功率仅 5.0%」——
    闭环失败源于**行为克隆的误差累积（covariate shift）**：示范数据只覆盖专家轨迹
    附近的分布，策略一旦偏出去就回不来，且**数据里没有任何「偏了该怎么纠回来」的样本**。

DAgger 的做法：
    1. 用当前策略 π_i 在仿真里 rollout（**不筛选成败**，覆盖学生自己的状态分布）
    2. 在**学生访问到的每个状态**上查询专家 π*，得到纠正动作标签
    3. 把 (观测, 纠正动作) 并入数据集 → 重训 → 下一轮

关于 β-mixing（HG-DAgger）：
    每步以概率 β 改用专家动作推进环境，其余用学生动作，避免学生过早走进死胡同
    采不到有效数据。**但无论谁的动作推进环境，记录的标签始终是专家在该状态的动作**
    —— 这正是 DAgger 的关键。

效率说明：
    学生单次前向约 130 ms（SigLIP 跑 512×512 + 10 步去噪），是本脚本的唯一瓶颈。
    因此采用**动作 chunk 执行**（`--exec-actions`）：一次前向出一个 chunk，
    连续执行若干步再重新规划；exec=8 时推理开销降到 1/8。

用法：
    cd /3240704007/sci/VLA
    CUDA_VISIBLE_DEVICES=5 .venv_smolvla/bin/python smolvla_dagger_collect.py \
        --ckpt export/ft_expert_react --data export/nav_demos_react.h5 \
        --episodes 200 --exec-actions 8 --beta 0.2 --out export/nav_dagger_r1.h5

    # 合并原数据 + 纠正数据（统一指令词表并重映射 id）
    .venv_smolvla/bin/python -m sim.gen_dataset \
        --merge export/nav_demos_react.h5,export/nav_dagger_r1.h5 \
        --out export/nav_demos_react_d1.h5
"""

from __future__ import annotations
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

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from sim.baselines import make_policy
from sim.env import SemanticNavEnv, calibrated_env, navigate_metrics
from sim.gen_dataset import write_h5
from smolvla_finetune_sim import CKPT, reconfigure_for_diff_drive
from smolvla_sim_data import ACTION_KEY, IMG_KEY, STATE_KEY, SimDemoData

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- 学生策略
def load_student(ckpt, data_path, chunk, action_dim=2):
    """加载微调后的学生策略（复用微调脚本的动作空间对齐逻辑）。"""
    policy = SmolVLAPolicy.from_pretrained(CKPT).to(DEV)
    cfg = policy.config
    if chunk and chunk != cfg.chunk_size:
        cfg.chunk_size = chunk
        cfg.n_action_steps = chunk
    reconfigure_for_diff_drive(policy, action_dim=action_dim)

    ck = torch.load(os.path.join(ckpt, "policy.pt"), map_location=DEV)
    if "model" in ck:
        miss, unexp = policy.model.load_state_dict(ck["model"], strict=False)
        if miss or unexp:
            print(f"  [提示] state_dict 缺失 {len(miss)} / 多余 {len(unexp)} 项")
    else:                                   # 兼容旧格式
        print("  [警告] 旧格式 checkpoint（缺 action_time_mlp_*）")
        policy.model.vlm_with_expert.lm_expert.load_state_dict(ck["lm_expert"])
        policy.model.action_in_proj.load_state_dict(ck["action_in_proj"])
        policy.model.action_out_proj.load_state_dict(ck["action_out_proj"])
    policy.eval()
    print(f"  已加载学生策略：{ckpt}（chunk={cfg.chunk_size}，动作维={action_dim}）")
    return policy


@torch.no_grad()
def student_chunk(policy, data, obs):
    """出一次前向，返回长度 chunk 的动作序列（物理量）。"""
    img = torch.from_numpy(obs["image"]).to(DEV).float().div_(255.0).permute(2, 0, 1)[None]
    st = torch.from_numpy(obs["state"]).to(DEV)[None]
    st = (st - torch.as_tensor(data.st_mean, device=DEV)) / \
         torch.as_tensor(data.st_std, device=DEV)
    st = torch.nn.functional.pad(st, (0, data.max_state_dim - st.shape[1]))
    # data 建在 CPU 上（不占显存），语言张量需显式搬到计算设备
    lang_t, lang_m = data.tokenize([obs["instruction"]])
    lang_t, lang_m = lang_t.to(DEV), lang_m.to(DEV)
    images, img_masks = policy.prepare_images({IMG_KEY: img})
    state_p = policy.prepare_state({STATE_KEY: st})
    pred = policy.model.sample_actions(images, img_masks, lang_t, lang_m, state_p)
    return data.denormalize_action(pred[0]).cpu().numpy()      # (chunk, 2)


# ---------------------------------------------------------------- DAgger 主循环
def dagger(ckpt, data_path, out, episodes=200, seed=0, max_steps=200, peds=0,
           exec_actions=8, beta=0.2, chunk=50, instr_variants=2, level=None,
           expert_name="sweep", verbose=True, calib=True):
    data = SimDemoData(data_path, device="cpu", chunk_size=chunk, max_action_dim=2)
    policy = load_student(ckpt, data_path, chunk)
    expert = make_policy(expert_name)
    # 标定配置（见 主创新A_诊断报告.md §4.5）：开阔场地 + 到物体表面距离的成功判据
    env = (calibrated_env(seed=seed, max_steps=max_steps, task_level=level) if calib
           else SemanticNavEnv(seed=seed, n_peds=peds, max_steps=max_steps, task_level=level))
    rng = np.random.default_rng(seed + 999)

    img_buf, act_buf, st_buf, lid_buf, ins_buf = [], [], [], [], []
    off_buf = [0]
    ep_meta = []
    instr2id: dict[str, int] = {}

    def instr_id(s):
        if s not in instr2id:
            instr2id[s] = len(instr2id)
        return instr2id[s]

    vmax, wmax = env.cfg["v_max"], env.cfg["w_max"]
    t0 = time.time()
    n_stu = n_exp = 0

    for ep in range(episodes):
        obs = env.reset(seed=seed + 100003 * (ep + 1), level=level)
        if env.task is None:
            continue
        expert.reset(env)

        rows = []                 # [(obs, expert_action)]
        buf, ptr = None, 0        # 学生动作 chunk 缓存
        done = False
        for _ in range(max_steps):
            a_star = np.asarray(expert.act(obs, env), dtype=np.float32)   # 专家标签
            rows.append((obs, a_star))

            use_expert = rng.random() < beta
            if use_expert:
                a_run = a_star
                n_exp += 1
            else:
                if buf is None or ptr >= min(exec_actions, len(buf)):
                    buf = student_chunk(policy, data, obs)
                    ptr = 0
                a_run = buf[ptr]
                ptr += 1
                n_stu += 1

            a_run = np.array([np.clip(a_run[0], 0.0, vmax),
                              np.clip(a_run[1], -wmax, wmax)], dtype=np.float32)
            obs, r, done, info = env.step(a_run)     # **用（混合后的）策略动作推进**
            if done:
                break

        if not rows:
            continue
        m = navigate_metrics(info)
        # DAgger **不筛选成败**：失败回合正是最有价值的纠正样本
        phr = [p[1] for p in (env.task.get("phrasings") or [])] or [env.task["instruction"]]
        k = int(max(1, min(instr_variants, len(phr))))
        chosen = list(phr) if k >= len(phr) else \
            [phr[int(j)] for j in rng.integers(0, len(phr), k)]
        for ins in chosen:
            for ob, av in rows:
                img_buf.append(ob["image"])
                act_buf.append(av)
                st_buf.append(ob["state"])
                lid_buf.append(ob["lidar"])
                ins_buf.append(instr_id(ins))
            off_buf.append(len(img_buf))
            ep_meta.append({"level": info["level"], "success": float(m["SR"]),
                            "SPL": m["SPL"], "NE": m["NE"], "instruction": ins,
                            "n_rows": len(rows), "n_variants": k})
        if verbose and (ep + 1) % 20 == 0:
            sr = np.mean([x["success"] for x in ep_meta])
            print(f"  {ep + 1:4d}/{episodes} 回合 | 样本 {len(img_buf):6d} | "
                  f"学生成功率 {100 * sr:.1f}% | 专家介入 {100 * n_exp / max(1, n_exp + n_stu):.1f}% "
                  f"| {time.time() - t0:.0f}s")

    vocab = [k for k, _ in sorted(instr2id.items(), key=lambda x: x[1])]
    d = {
        "images": np.stack(img_buf) if img_buf else np.zeros((0, 120, 160, 3), np.uint8),
        "actions": np.stack(act_buf) if act_buf else np.zeros((0, 2), np.float32),
        "states": np.stack(st_buf) if st_buf else np.zeros((0, 5), np.float32),
        "lidar": np.stack(lid_buf) if lid_buf else np.zeros((0, 120), np.float32),
        "instr_ids": np.asarray(ins_buf, dtype=np.int32),
        "instr_vocab": vocab,
        "ep_offset": np.asarray(off_buf, dtype=np.int64),
        "ep_level": np.array([m["level"] for m in ep_meta], dtype=object),
        "ep_success": np.array([m["success"] for m in ep_meta], dtype=np.float32),
        "ep_spl": np.array([m["SPL"] for m in ep_meta], dtype=np.float32),
        "ep_ne": np.array([m["NE"] for m in ep_meta], dtype=np.float32),
        "n_dropped": 0,
    }
    write_h5(out, d, attrs={
        "source": "dagger", "student_ckpt": ckpt, "expert": expert_name,
        "beta": beta, "exec_actions": exec_actions, "episodes": episodes,
        "student_rollout_sr": float(np.mean([m["success"] for m in ep_meta])) if ep_meta else 0.0,
        "instruction": "DAgger 纠正数据：标签为专家在**学生访问到的状态**上的动作",
    })
    stats = {"samples": len(d["images"]), "segments": len(ep_meta),
             "student_sr": float(np.mean([m["success"] for m in ep_meta])) if ep_meta else 0.0,
             "expert_intervention": n_exp / max(1, n_exp + n_stu),
             "seconds": time.time() - t0}
    print(f"\n完成 → {out}")
    print(f"  样本 {stats['samples']} / 段 {stats['segments']}")
    print(f"  学生策略自身成功率 {100 * stats['student_sr']:.1f}%"
          f"（这就是 DAgger 要纠正的分布）")
    print(f"  专家介入比例 {100 * stats['expert_intervention']:.1f}% | 耗时 {stats['seconds']:.0f}s")
    with open(out + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="学生策略目录（含 policy.pt）")
    ap.add_argument("--data", required=True, help="学生训练所用数据（取归一化统计与 tokenizer）")
    ap.add_argument("--out", default="export/nav_dagger_calib_r1.h5")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--no-calib", dest="calib", action="store_false",
                    help="使用旧配置（默认用标定配置）")
    ap.add_argument("--exec-actions", type=int, default=8,
                    help="一次前向执行多少步（推理是唯一瓶颈，8 可省 7/8 开销）")
    ap.add_argument("--beta", type=float, default=0.2, help="专家介入概率（HG-DAgger）")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--instr-variants", type=int, default=2)
    ap.add_argument("--level", default=None)
    ap.add_argument("--peds", type=int, default=2)
    ap.add_argument("--expert", default="sweep")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print(f"设备 {DEV}")
    print(f"=== DAgger 采集（学生 {a.ckpt} / 专家 {a.expert} / β={a.beta} "
          f"/ calib={a.calib}）===")
    dagger(a.ckpt, a.data, a.out, episodes=a.episodes, seed=a.seed,
           max_steps=a.max_steps, peds=a.peds, exec_actions=a.exec_actions,
           beta=a.beta, chunk=a.chunk, instr_variants=a.instr_variants,
           level=a.level, expert_name=a.expert, calib=a.calib)


if __name__ == "__main__":
    main()
