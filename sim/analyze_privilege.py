"""决定性实验：示范动作究竟能不能从「单帧观测」恢复？

为什么需要这个实验（`analyze_observability.py` 的判定不够严谨）：
    该脚本用「ω 符号的边际熵」当判据，但**边际熵 ≈ 1 只说明正负各半，并不等于不可学**。
    真正要问的是：ω 在给定**观测**下的条件不确定性。
    只有在「观测几乎不携带关于 ω 的信息」时，示范才是单帧不可学的。

做法（对照实验，直接判定信息归属）：
    用同一批 `oracle` 示范数据、同一回归器、同一划分，只换特征集：

        A（可观测）  激光(120) + 位姿(3) + 目标相对位置(2，仅当目标落在相机视场内)
        B（可观测+特权）A 再拼上**测地距离场在机器人所在格的梯度方向(2)**
                       —— 这正是 `oracle` 真正使用的特权信息（完整真值栅格图）

    若 **MAE(B) ≪ MAE(A)**，说明「决定动作的信息主要来自特权特征」，
    则用 A（＝策略实际能拿到的观测量）做的模仿学习在原理上就达不到示范水平；
    若 **MAE(A) ≈ MAE(B)**，说明观测已经够用，闭环失败更可能是训练/容量问题。

    同时报告 MAE(A) 与已训练的 SmolVLA 的开环 MAE，判断 VLA 是否已逼近 A 的可达下限。

用法：
    cd /3240704007/sci/VLA
    .venv_smolvla/bin/python -m sim.analyze_privilege --episodes 50 --out export/privilege.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .baselines import make_policy
from .env import SemanticNavEnv
from .sensors import CAMERA_HFOV_DEG
from .world import CELL


def collect(episodes=50, seed=0, max_steps=600, peds=2, level=None, policy_name="oracle"):
    """采集 (可观测特征, 特权特征, 动作)，按回合切分。"""
    env = SemanticNavEnv(seed=seed, n_peds=peds, max_steps=max_steps, task_level=level)
    policy = make_policy(policy_name)
    half = np.deg2rad(CAMERA_HFOV_DEG) / 2.0

    A, B, Y, EP = [], [], [], []
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep * 101, level=level)
        if env.task is None:
            continue
        policy.reset(env)
        goal = None
        for _ in range(max_steps):
            pose = env.pose.copy()
            goal = np.asarray(env.task["goal"], dtype=float)
            dv = goal - pose[:2]
            dist = float(np.linalg.norm(dv))
            bearing = float(np.arctan2(dv[1], dv[0]) - pose[2])
            bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
            visible = (abs(bearing) <= half) and (dist < 8.0)

            # --- A：只含策略真正能拿到的观测量 ---
            # **必须包含图像**，否则 A 不是「策略能看到的一切」，结论会站不住。
            # 用降采样灰度图（16×12=192 维）给树提供粗粒度视觉结构。
            if visible:      # 目标在视野外时，策略无从知道它相对自己的位置
                rel = np.array([dist * np.cos(bearing), dist * np.sin(bearing)])
            else:
                rel = np.zeros(2)
            # **必须包含上一步动作**：反应式示范带角速度速率限制，动作因此是有记忆的
            # （ω_t 依赖 ω_{t-1}）。漏掉它会让回归器退化到「不如恒输出均值」。
            # 注意这不构成特权信息 —— 上一步动作就在策略的 state 里（`obs["state"]` 含 last_action）。
            a_feat = np.concatenate([np.asarray(obs["lidar"], float), pose, rel,
                                     [1.0 if visible else 0.0],
                                     np.asarray(obs["state"][-2:], float),
                                     _img_feat(obs["image"])])

            # --- 特权：测地距离场在本格的梯度方向（oracle 真正的信息来源） ---
            grad = _field_grad(env, pose, goal)

            y = policy.act(obs, env)
            A.append(a_feat); B.append(np.concatenate([a_feat, grad]))
            Y.append([float(y[0]), float(y[1])]); EP.append(ep)

            obs, r, done, info = env.step(y)
            if done:
                break
    return (np.asarray(A, np.float32), np.asarray(B, np.float32),
            np.asarray(Y, np.float32), np.asarray(EP))


def _img_feat(img, h=12, w=16):
    """把 (H,W,3) 图像降采样成灰度特征向量（粗粒度视觉结构）。"""
    g = np.asarray(img, np.float32)
    if g.ndim == 3:
        g = g.mean(-1)
    H, W = g.shape
    gi = np.linspace(0, H, h + 1).astype(int)
    gj = np.linspace(0, W, w + 1).astype(int)
    out = np.empty((h, w), np.float32)
    for i in range(h):
        for j in range(w):
            out[i, j] = g[gi[i]:max(gi[i] + 1, gi[i + 1]),
                          gj[j]:max(gj[j] + 1, gj[j + 1])].mean()
    return (out.reshape(-1) / 255.0).astype(np.float32)


def _field_grad(env, pose, goal):
    """测地距离场在机器人所在栅格的**下降方向**（单位向量）；不可达则返回 0。"""
    try:
        f = env._shortest_path_len(goal)
    except Exception:
        return np.zeros(2, np.float32)
    H, W = f.shape
    r = int((pose[1] - env.scene.bounds[2]) / CELL)
    c = int((pose[0] - env.scene.bounds[0]) / CELL)
    r = int(np.clip(r, 1, H - 2)); c = int(np.clip(c, 1, W - 2))
    best, bk = np.inf, None
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            v = f[r + dr, c + dc]
            if v >= 0 and v < best:
                best, bk = v, (dr, dc)
    if bk is None or not np.isfinite(best) or f[r, c] < 0:
        return np.zeros(2, np.float32)
    g = np.array([bk[1], bk[0]], np.float32)       # (dx, dy) 栅格方向 = 世界方向
    n = np.linalg.norm(g)
    return g / n if n > 0 else np.zeros(2, np.float32)


def fit_eval(Xtr, Ytr, Xte, Yte, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    out = {}
    for j, nm in enumerate(("v", "w")):
        m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08,
                                          max_depth=None, random_state=seed)
        m.fit(Xtr, Ytr[:, j])
        out[nm] = float(np.abs(m.predict(Xte) - Yte[:, j]).mean())
    out["total"] = 0.5 * (out["v"] + out["w"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--policy", default="oracle")
    ap.add_argument("--val-ratio", type=float, default=0.3)
    ap.add_argument("--exclude-saturated", action="store_true", default=True,
                    help="剔除 ω 饱和（|ω|>=0.99*w_max）的步。实测不过滤时，新采回放里"
                         "有大量「原地打转卡死」的退化工况（|ω| 中位数就是饱和值），"
                         "会污染可学性判定")
    ap.add_argument("--keep-saturated", dest="exclude_saturated", action="store_false")
    ap.add_argument("--vla-mae", type=float, default=None,
                    help="已训练 VLA 的开环 MAE（可选，用于对比是否已逼近可达下限）")
    ap.add_argument("--out", default="export/privilege.json")
    a = ap.parse_args()

    print(f"采集示范（策略 {a.policy}，{a.episodes} 回合）…")
    A, B, Y, EP = collect(a.episodes, a.seed, a.max_steps, policy_name=a.policy)
    if len(A) < 100:
        raise SystemExit("样本过少")
    if a.exclude_saturated:
        from .env import W_MAX
        keep = np.abs(Y[:, 1]) < 0.99 * W_MAX
        print(f"  剔除 ω 饱和步：{int((~keep).sum())} / {len(Y)} "
              f"（{100 * (~keep).mean():.1f}%，属卡死/振荡的退化工况）")
        A, B, Y, EP = A[keep], B[keep], Y[keep], EP[keep]

    eps = np.unique(EP)
    n_val = max(1, int(len(eps) * a.val_ratio))
    val_eps = set(eps[-n_val:].tolist())           # **按回合划分**，避免时序泄漏
    m = np.array([e in val_eps for e in EP])
    print(f"  样本 {len(A)} | 训练 {int((~m).sum())} / 验证 {int(m.sum())}"
          f"（按回合划分，{n_val}/{len(eps)} 个回合做验证）")

    res = {}
    for tag, X in (("A_observable", A), ("B_observable+privileged", B)):
        r = fit_eval(X[~m], Y[~m], X[m], Y[m])
        res[tag] = r
        print(f"  [{tag:26s}] MAE  v {r['v']:.4f}  ω {r['w']:.4f}  合计 {r['total']:.4f}")

    # 平凡基线：忽略一切观测，恒输出训练集动作均值
    const = np.abs(Y[m] - Y[~m].mean(0)).mean(0)
    res["constant_baseline"] = {"v": float(const[0]), "w": float(const[1]),
                                "total": float(const.mean())}
    print(f"  [{'constant_baseline':26s}] MAE  v {const[0]:.4f}  ω {const[1]:.4f}  "
          f"合计 {const.mean():.4f}")

    ra, rb = res["A_observable"], res["B_observable+privileged"]
    gap = ra["total"] - rb["total"]
    rel = gap / max(rb["total"], 1e-9)
    impr = (const.mean() - ra["total"]) / max(const.mean(), 1e-9)   # 观测相对平凡基线的改善
    res["gap_total"] = float(gap)
    res["gap_rel"] = float(rel)
    res["observable_impr"] = float(impr)

    print("\n--- 判定 ---")
    print(f"① 可观测特征（含图像）相对平凡基线的改善: {100 * impr:+.1f}%")
    print(f"② 特权特征（测地场梯度）额外带来的改善:   {100 * rel:+.1f}%")
    if impr < 0.15 and rel < 0.15:
        print("✗ 观测（含图像）与特权特征**都无法**预测示范动作 —— 说明连 oracle")
        print("  自身的角速度都不是这几类量的函数（它由势场+避障的连续耦合决定），")
        print("  示范信号本身噪声极大。这解释了「开环指标降、闭环恒 0%」。")
    elif impr < 0.15 <= rel:
        print("✗ 决定性信息来自**特权特征**（完整真值栅格图），策略仅凭单帧观测")
        print("  原理上无法达到示范水平。这解释了「开环 MAE 单调下降 + 动作统计量")
        print("  吻合 + 闭环恒为 0%」。")
    elif impr >= 0.30:
        print("✓ 观测（含图像）已能较好预测示范动作 → 闭环失败更可能来自训练预算/")
        print("  容量/时序结构，而不是信息不足。")
    else:
        print("~ 观测能带来部分改善但不足以复现示范 → 单帧策略存在信息瓶颈，")
        print("  建议补记忆（帧堆叠/循环）或改用反应式示范。")
    if a.vla_mae is not None:
        print(f"\n对比：已训练 VLA 开环 MAE {a.vla_mae:.4f}"
              f" | 仅用可观测特征的可达水平 {ra['total']:.4f}"
              f"（VLA {'已逼近' if a.vla_mae <= ra['total'] * 1.15 else '仍高于'}该下限）")
        res["vla_mae"] = a.vla_mae

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n已写入 {a.out}")


if __name__ == "__main__":
    main()
