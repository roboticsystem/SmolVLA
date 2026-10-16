"""决定性实验：一个**开环几乎完美**的学习器，闭环能跑成什么样？

动机：
    主创新 A 的实证结果是「开环 MAE 单调下降（0.56 → 0.22）、动作统计量与示范几乎一致、
    闭环成功率恒为 0%」。要区分两种解释，需要一个能把「开环拟合质量」与
    「闭环可用性」解耦的实验：

        解释 ①（VLA 训练/容量不足）→ 若把开环误差压到远低于 VLA 的水平，闭环就该变好
        解释 ②（闭环误差累积不可逾越）→ 即使开环误差压到极低，闭环仍然失败

    做法：在同一批示范上训练一个**强非参数回归器**（梯度提升树），
    它的开环 MAE 只有 VLA 的约 1/6（实测 0.037 vs 0.2206），
    然后**把它当成策略在仿真里闭环保真跑**。

    注意这不是「更好的方法」，而是**上界探针**：它直接回答
    「单帧观测 + 开环拟合质量」这个组合的能力天花板在哪里。

特征（与策略可获得的观测量一致，不含特权信息）：
    激光(120) + 位姿(3) + 目标相对位置(2，仅当目标落在相机视场内) + 可见标志(1)
         + 上一步动作(2) + 降采样图像(192)

用法：
    cd /3240704007/sci/VLA
    .venv_smolvla/bin/python -m sim.eval_learned_baseline \
        --demo-policy reactive --episodes 80 --eval-episodes 20
"""

from __future__ import annotations

import argparse
import json
import os
import zlib

import numpy as np

from .baselines import make_policy
from .env import SemanticNavEnv, calibrated_env, navigate_metrics
from .sensors import CAMERA_HFOV_DEG

HALF_FOV = np.deg2rad(CAMERA_HFOV_DEG) / 2.0

# 是否把「目标在视野内时的相对位置 (dx, dy)」作为特征。
# **这是一个由真值目标位置算出来的量**，而 VLA 必须从图像自行判断目标方位。
# 因此开启它测得的成功率**不是 VLA 的公平上界**，只能说明「若已知目标方位则任务可解」。
USE_GOAL_REL = True

# 是否把激光雷达作为特征。**VLA 没有激光输入**（只吃 RGB + 位姿），
# 因此在开阔场地里激光是探针的一个巨大优势，必须单独关掉来做公平对标。
USE_LIDAR = True

# 是否把**指令**作为分类特征。开启后探针才与 VLA 信息对齐（图像 + 位姿 + 指令）；
# 关闭时探针只能学边缘行为，实测仅 10.0%，不具备判断可学性的资格。
USE_INSTR = True


def features(obs, env, pose=None, goal=None):
    """构造与策略可获观测量一致的特征（供采集与闭环保真共用）。"""
    from .analyze_privilege import _img_feat
    pose = env.pose if pose is None else pose
    goal = np.asarray(env.task["goal"], float) if goal is None else goal
    dv = goal - pose[:2]
    dist = float(np.linalg.norm(dv))
    bearing = float(np.arctan2(dv[1], dv[0]) - pose[2])
    bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
    visible = (abs(bearing) <= HALF_FOV) and (dist < 8.0)
    if USE_GOAL_REL and visible:
        rel = np.array([dist * np.cos(bearing), dist * np.sin(bearing)])
    else:
        rel = np.zeros(2)
    # **`visible`（目标是否在相机视野内）同样是特权信息**：它由真值目标位置算得，
    # 而「当前视野里是不是**那个指定的**目标」恰恰是语义判别本身要做的事。
    # 实测它单独就能让探针「读答案」—— 曾经只去掉 rel 而留下 visible，
    # 导致探针虚高。因此把二者一起由 `USE_GOAL_REL` 控制。
    parts = [pose.astype(np.float32),
             rel.astype(np.float32),
             [1.0 if (USE_GOAL_REL and visible) else 0.0],
             np.asarray(obs["state"][-2:], np.float32)]
    if USE_INSTR:
        # **指令必须以分类特征的形式给出**，否则探针根本不是「公平上界」：
        # 本任务是指令条件任务，「去椅子旁边」与「去桌子旁边」的目标不同。
        # 没有语言的模仿者只能学到**边缘行为**（对所有指令取平均），
        # 实测那样只有 10.0%，与纯随机同级 —— 完全无法用来判断任务是否可学。
        # 用稳定的 crc32 哈希把指令串映射成分类 id（不依赖词表顺序）。
        s = str(obs.get("instruction", ""))
        parts.append([float(zlib.crc32(s.encode()) % 100000)])
    parts.append(_img_feat(obs["image"]))
    if USE_LIDAR:                      # VLA **没有**激光输入，故可关掉以对标
        parts.insert(0, np.asarray(obs["lidar"], np.float32))
    return np.concatenate(parts).astype(np.float32)


def gather(episodes=80, seed=0, max_steps=200, policy_name="sweep", peds=0, calib=True):
    if calib:
        env = calibrated_env(seed=seed, max_steps=max_steps)
    else:
        env = SemanticNavEnv(seed=seed, n_peds=peds, max_steps=max_steps)
    pol = make_policy(policy_name)
    X, Y, EP = [], [], []
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep * 131)
        if env.task is None:
            continue
        pol.reset(env)
        for _ in range(max_steps):
            X.append(features(obs, env))
            y = pol.act(obs, env)
            Y.append([float(y[0]), float(y[1])])
            EP.append(ep)
            obs, r, done, info = env.step(y)
            if done:
                break
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), np.asarray(EP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-policy", default="sweep")
    ap.add_argument("--episodes", type=int, default=400, help="采集示范的回合数")
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--eval-max-steps", type=int, default=200)
    ap.add_argument("--no-calib", dest="calib", action="store_false",
                    help="使用旧配置（默认用标定配置，见 主创新A_诊断报告.md §4.5）")
    ap.add_argument("--no-instr", dest="instr", action="store_false",
                    help="**去掉指令特征**。仅用于展示「无语言模仿者」的天花板有多低"
                         "（实测仅 10.0%），不作为可学性判据")
    ap.add_argument("--no-lidar", dest="lidar", action="store_false",
                    help="**去掉激光雷达特征** —— VLA 只吃 RGB + 位姿，"
                         "开启激光会让探针获得 VLA 没有的信息")
    ap.add_argument("--no-goal-rel", dest="goal_rel", action="store_false",
                    help="**去掉「目标相对位置」特征**，只留策略真正能拿到的东西"
                         "（激光 + 位姿 + 上一步动作 + 降采样图像）。"
                         "这是 VLA 的公平上界；带该特征的结果只能说明"
                         "「已知目标方位时任务可解」")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--out", default="export/learned_baseline.json")
    a = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingRegressor

    global USE_GOAL_REL, USE_LIDAR, USE_INSTR
    USE_GOAL_REL = bool(a.goal_rel)
    USE_LIDAR = bool(a.lidar)
    USE_INSTR = bool(a.instr)
    print(f"① 采集示范（{a.demo_policy}，{a.episodes} 回合，calib={a.calib}，"
          f"目标相对位置={'开' if USE_GOAL_REL else '关'}，"
          f"激光={'开' if USE_LIDAR else '关'}，"
          f"指令={'开' if USE_INSTR else '关'}）…")
    X, Y, EP = gather(a.episodes, a.seed, policy_name=a.demo_policy, calib=a.calib)
    print(f"   样本 {len(X)}，动作 ω 标准差 {Y[:, 1].std():.4f}")

    print("② 训练回归器…")
    models = []
    for j, nm in enumerate(("v", "w")):
        m = HistGradientBoostingRegressor(max_iter=a.iters, learning_rate=0.08,
                                          random_state=a.seed)
        m.fit(X, Y[:, j])
        mse = float(np.abs(m.predict(X) - Y[:, j]).mean())
        models.append(m)
        print(f"   {nm}: 训练集 MAE {mse:.4f}（开环拟合质量）")

    # ③ 闭环保真
    print(f"\n③ 闭环保真（{a.eval_episodes} 回合，max_steps={a.eval_max_steps}）…")
    env = (calibrated_env(seed=9000, max_steps=a.eval_max_steps) if a.calib
           else SemanticNavEnv(seed=9000, n_peds=2, max_steps=a.eval_max_steps))
    srs, spls, nes, tls = [], [], [], []
    n_steps = n_sat = 0
    for ep in range(a.eval_episodes):
        obs = env.reset(seed=9000 + ep * 37)
        if env.task is None:
            continue
        for _ in range(a.eval_max_steps):
            f = features(obs, env)[None, :]
            act = np.array([models[0].predict(f)[0], models[1].predict(f)[0]], np.float32)
            act[0] = float(np.clip(act[0], 0.0, env.cfg["v_max"]))
            act[1] = float(np.clip(act[1], -env.cfg["w_max"], env.cfg["w_max"]))
            n_steps += 1
            if abs(act[1]) >= 0.99 * env.cfg["w_max"]:
                n_sat += 1
            obs, r, done, info = env.step(act)
            if done:
                break
        m = navigate_metrics(info)
        srs.append(m["SR"]); spls.append(m["SPL"]); nes.append(m["NE"]); tls.append(m["TL"])

    sr = float(np.mean(srs)); spl = float(np.nanmean(spls))
    ne = float(np.mean(nes)); tl = float(np.nanmean(tls))
    print(f"\n=== 学习器闭环结果 ===")
    print(f"  SR {100 * sr:.1f}% | SPL {spl:.3f} | NE {ne:.2f} m | TL {tl:.2f}")
    print(f"  总步数 {n_steps} | ω 饱和占比 {100 * n_sat / max(1, n_steps):.1f}%")

    print("\n参考基线（标定配置，40 回合）：oracle 67.5% / sweep 60.0% / "
          "random_target 17.5% / random 10.0%")
    print("参考 VLA（标定配置，60 回合）：n_frames=1 → 18.3%，n_frames=3 → 12.0%")

    print("\n--- 判定 ---")
    if sr >= 0.30:
        print("✓ 开环近乎完美的学习器闭环成功 → **瓶颈是 VLA 的训练/容量**，")
        print("  主创新 A 的空间在「把开环误差压到更低」，值得继续加训练预算/改架构。")
    elif sr <= 0.10:
        print("✗ 即使开环 MAE 已远低于 VLA，闭环依然失败 → **闭环误差累积是根本障碍**，")
        print("  单纯提升开环拟合精度无效。主创新 A 必须转向闭环修正：")
        print("  DAgger 式纠正数据 / 闭环微调 / 时序记忆结构。")
    else:
        print("~ 介于两者之间 → 开环精度对闭环有帮助但不充分，两条路都需要投入。")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"demo_policy": a.demo_policy, "episodes": a.episodes,
                   "eval_episodes": a.eval_episodes, "sr": sr, "spl": spl,
                   "ne": ne, "tl": tl, "saturation": n_sat / max(1, n_steps),
                   "train_mae": [float(np.abs(m.predict(X) - Y[:, j]).mean())
                                 for j, m in enumerate(models)]},
                  f, indent=2, ensure_ascii=False)
    print(f"\n已写入 {a.out}")


if __name__ == "__main__":
    main()
