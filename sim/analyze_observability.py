"""可观测性分析：示范动作到底能不能从「单帧观测」中恢复？

背景（本轮发现的反常现象）：
    在自建测试床上微调 SmolVLA，开环动作 MAE 单调下降（0.56 → 0.22），
    输出动作的**统计量几乎与示范完全一致**（平均 |v| 0.2697 vs 示范 0.27；
    平均 |ω| 0.3938 vs 示范 0.42），但**闭环导航成功率始终为 0–12.5%**。

猜测：这不像欠训练，而像**信息不在观测里** ——
    示范策略 `oracle` 的动作是 `env._shortest_path_len(goal)`（**完整真值栅格地图上的
    测地距离场**）的函数，即依赖「地图 + 真值目标位置」这类**特权信息**；
    而策略只看「单帧 RGB + 自身位姿」。若目标在视野外，单帧观测往往**无法判断该左转
    还是右转**，此条件下示范动作的符号近似抛硬币 —— 那么任何单帧策略都不可能学会。

本脚本用**纯行为分析**验证这一点，不训练任何模型：
    1. 目标在相机视野内 / 视野外 两种情况下，示范角速度 ω 的符号分布
    2. 目标可见时，ω 的符号与「目标方位角」的一致性
    3. 与「忽略观测、直接预测动作均值」这个平凡基线的误差比较

判定标准：
    - 目标**可见**时 sign(ω) 与方位角高度一致 → 这些样本是可学的
    - 目标**不可见**时 sign(ω) 接近 50/50 → 这些样本**不可从单帧恢复**
    - 若后者占比很高，则「开环指标好、闭环必失败」就得到解释

用法：
    cd /3240704007/sci/VLA
    .venv_smolvla/bin/python -m sim.analyze_observability --episodes 60 --out export/observability.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .baselines import make_policy
from .env import SemanticNavEnv
from .sensors import CAMERA_HFOV_DEG


def run(episodes=60, seed=0, max_steps=600, peds=2, level=None, policy_name="oracle"):
    env = SemanticNavEnv(seed=seed, n_peds=peds, max_steps=max_steps, task_level=level)
    policy = make_policy(policy_name)
    hfov = np.deg2rad(CAMERA_HFOV_DEG) / 2.0

    rows = []          # (bearing_rad, w, v, goal_visible)
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep * 101, level=level)
        if env.task is None:
            continue
        policy.reset(env)
        for _ in range(max_steps):
            pose = env.pose.copy()
            goal = np.asarray(env.task["goal"], dtype=float)
            # 目标相对机器人朝向的方位角，归一化到 (-π, π]
            b = float(np.arctan2(goal[1] - pose[1], goal[0] - pose[0]) - pose[2])
            b = (b + np.pi) % (2 * np.pi) - np.pi
            a = policy.act(obs, env)
            rows.append((b, float(a[1]), float(a[0]), abs(b) <= hfov))
            obs, r, done, info = env.step(a)
            if done:
                break

    return np.asarray(rows)


def analyze(rows):
    b, w, v, vis = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3].astype(bool)
    out = {"n_steps": int(len(rows)), "fov_half_deg": float(np.degrees(np.pi) and 45.0),
           "goal_visible_frac": float(vis.mean())}

    # --- 1) 两种可见性下 ω 的符号分布 ---
    for tag, m in (("visible", vis), ("hidden", ~vis)):
        if m.sum() == 0:
            continue
        ww = w[m]
        nz = ww[np.abs(ww) > 0.05]                 # 排除几乎不转的样本
        pos = float((nz > 0).mean()) if nz.size else float("nan")
        out[f"{tag}_n"] = int(m.sum())
        out[f"{tag}_w_absmean"] = float(np.abs(ww).mean())
        out[f"{tag}_frac_w_near0"] = float((np.abs(ww) <= 0.05).mean())
        out[f"{tag}_frac_w_pos"] = pos             # 0.5 = 抛硬币
        out[f"{tag}_sign_entropy"] = _entropy(pos)

    # --- 2) 目标可见时：ω 符号与方位角的一致性 ---
    if vis.sum():
        bv, wv = b[vis], w[vis]
        m = np.abs(wv) > 0.05
        if m.sum():
            # 方位角 >0（目标在左侧）应伴随 ω>0（左转）
            agree = float(((bv[m] > 0) == (wv[m] > 0)).mean())
            out["visible_sign_agree_with_bearing"] = agree
            out["visible_signed_corr"] = float(np.corrcoef(bv[m], wv[m])[0, 1])

    # --- 3) 平凡基线：忽略观测，直接预测动作均值 ---
    out["const_baseline_mae"] = {"v": float(np.abs(v - v.mean()).mean()),
                                 "w": float(np.abs(w - w.mean()).mean()),
                                 "total": float(np.abs(np.stack([v, w], 1) -
                                                       np.stack([v, w], 1).mean(0)).mean())}
    out["action_std"] = {"v": float(v.std()), "w": float(w.std())}
    out["action_absmean"] = {"v": float(np.abs(v).mean()), "w": float(np.abs(w).mean())}
    return out


def _entropy(p):
    """二元符号分布的归一化熵：0 = 完全确定，1 = 完全随机（抛硬币）。"""
    if p is None or not np.isfinite(p) or p <= 0 or p >= 1:
        return 0.0 if (p == 0 or p == 1) else float("nan")
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--peds", type=int, default=2)
    ap.add_argument("--level", default=None)
    ap.add_argument("--policy", default="oracle", help="被分析的示范策略")
    ap.add_argument("--out", default="export/observability.json")
    a = ap.parse_args()

    rows = run(a.episodes, a.seed, a.max_steps, a.peds, a.level, a.policy)
    if len(rows) == 0:
        raise SystemExit("未采集到样本")
    r = analyze(rows)
    r["policy"] = a.policy

    print(f"\n=== 可观测性分析：示范策略 = {a.policy}（{r['n_steps']} 步）===")
    print(f"目标在视野内的比例: {100 * r['goal_visible_frac']:.1f}%\n")
    print(f"{'情形':<10} {'步数':>7} {'平均|ω|':>9} {'不动占比':>9} {'ω>0 占比':>10} {'符号熵':>8}")
    for tag, name in (("visible", "目标可见"), ("hidden", "目标不可见")):
        if f"{tag}_n" not in r:
            continue
        print(f"{name:<10} {r[f'{tag}_n']:>7} {r[f'{tag}_w_absmean']:>9.3f} "
              f"{100 * r[f'{tag}_frac_w_near0']:>8.1f}% "
              f"{100 * r[f'{tag}_frac_w_pos']:>9.1f}% {r[f'{tag}_sign_entropy']:>8.3f}")

    print(f"\n目标可见时 ω 符号与方位角一致率: "
          f"{100 * r.get('visible_sign_agree_with_bearing', float('nan')):.1f}%"
          f"（方位角-ω 相关系数 {r.get('visible_signed_corr', float('nan')):+.3f}）")
    cb = r["const_baseline_mae"]
    print(f"\n平凡基线（忽略观测、恒输出动作均值）MAE: "
          f"v {cb['v']:.4f} / ω {cb['w']:.4f} / 合计 {cb['total']:.4f}")
    print(f"  动作标准差: v {r['action_std']['v']:.4f} / ω {r['action_std']['w']:.4f}")

    print("\n--- 判定 ---")
    eh = r.get("hidden_sign_entropy", float("nan"))
    hp = r.get("hidden_frac_w_pos", float("nan"))
    if np.isfinite(eh) and eh > 0.9:
        print(f"✗ 目标不可见时 ω 符号熵 {eh:.3f}（≈1 = 抛硬币），占全部步数的 "
              f"{100 * (1 - r['goal_visible_frac']):.0f}%")
        print("  → 这些样本的转向方向**不由单帧观测决定**，单帧策略原理上无法学会；")
        print("    这解释了「开环指标持续下降、动作统计量吻合、闭环却 0%」的反常现象。")
        print("  修法：① 示范策略改为**反应式**（动作是观测的函数）；")
        print("        ② 或给策略加**记忆**（帧堆叠 / 循环），把它变成真正的 VLN 设定。")
    else:
        print(f"✓ 目标不可见时 ω 符号熵 {eh:.3f}，转向方向仍具可预测性 —— "
              "闭环失败更可能来自训练/容量，而非可观测性。")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    print(f"\n已写入 {a.out}")


if __name__ == "__main__":
    main()
