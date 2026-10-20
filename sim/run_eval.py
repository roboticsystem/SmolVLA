"""E1 评测驱动：在自建语义 MultiRoom 上跑基线并输出指标。

用法：
    cd /3240704007/sci/VLA
    # 三条基线全跑，按指令层级分解，结果写入 JSON
    .venv_smolvla/bin/python -m sim.run_eval --episodes 40 --out export/sim_baseline.json

    # 只跑某条基线 / 某一级指令
    .venv_smolvla/bin/python -m sim.run_eval --policies oracle --level L3 --episodes 20

    # 顺带出一张轨迹图（俯视）
    .venv_smolvla/bin/python -m sim.run_eval --episodes 5 --figure export/sim_preview/traj.png
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .baselines import make_policy
from .env import SemanticNavEnv, calibrated_env, navigate_metrics
from .sensors import render_topdown


def goal_visible(env, max_range=9.0):
    """目标此刻是否**确实看得见**（在相机 FOV 内且未被墙遮挡）。

    只用于**构造评测起点**，不参与策略决策 —— 目的是把「搜索」从评测中分离出去，
    直接测「目标已在视野内时能否导航过去」。见 主创新A_诊断报告.md §4.17。
    """
    from .sensors import CAMERA_HFOV_DEG, raycast
    if env.task is None:
        return False
    g = np.asarray(env.task["goal"], dtype=float)
    p = env.pose
    v = g - p[:2]
    d = float(np.hypot(*v))
    if d < 0.6 or d > max_range:
        return False
    a = float(np.arctan2(v[1], v[0]) - p[2])
    a = (a + np.pi) % (2 * np.pi) - np.pi
    if abs(a) > np.deg2rad(CAMERA_HFOV_DEG) / 2:
        return False
    dist, h, _, _ = raycast(env.scene, p[:2], np.array([np.arctan2(v[1], v[0])]),
                            max_range=12.0, grid=env.scene.wall_grid)
    return (not h[0]) or float(dist[0]) >= d - 0.05


def evaluate(policy_name, episodes, level=None, seed=0, env_kwargs=None,
             figure=None, verbose=True, calib=False, policy_kw=None,
             visible_start=False, ep_base=None, ep_stride=1, detail_out=None,
             repeats=1):
    env_kwargs = dict(env_kwargs or {})
    policy_kw = dict(policy_kw or {})
    env = calibrated_env(seed=seed, **env_kwargs) if calib \
        else SemanticNavEnv(seed=seed, **env_kwargs)
    policy = make_policy(policy_name, **policy_kw)
    rows = []
    detail = []
    t0 = time.time()
    n_steps_total = 0

    n_skip = 0
    def _es(ep, tr=0, rep=0):
        # 回合种子方案：默认沿用老口径；给出 ep_base 时改用 `base + ep*stride`，
        # 以便与 VLA 评测（9000 + ep*37）**共用同一套回合集** ——
        # 保留率（模型/示范）只有在同一回合集上才是真正的配对量。
        #
        # `rep>0`（重复轮次）额外加 1e7*rep 的偏移：第 0 轮保持**原口径不变**
        # （配对检验仍用它），第 1 轮起用不同随机布局/任务/噪声重跑，
        # 用来估「重复实验之间的离散度」（参考论文统一的「重复 N 次取平均」口径）。
        # 偏移量取 1e7 是为了不与 visible_start 的重采样偏移 1e5*tr 撞上。
        off = 100000 * tr + 10000000 * rep
        return (ep_base + ep * ep_stride + off) if ep_base is not None \
            else (seed * 1000 + ep + off)

    rep_rows = []          # 每轮（repeat）的聚合结果
    all_rows = []
    all_detail = []
    for rep in range(repeats):
        rows = []
        detail = []
        for ep in range(episodes):
            s = _es(ep, 0, rep)
            obs = env.reset(seed=s, level=level)
            if visible_start:
                # 反复重采样起点，直到**目标在初始视野内且无遮挡**（最多 40 次）
                tries = 0
                while not goal_visible(env) and tries < 40:
                    tries += 1
                    s = _es(ep, tries, rep)
                    obs = env.reset(seed=s, level=level)
                if not goal_visible(env):
                    n_skip += 1
                    continue
            policy.reset(env)
            tot_r = 0.0
            _min_ds = float("inf")          # 全程离目标包围盒**最近**的表面距离
            for i in range(env.cfg["max_steps"]):
                a = policy.act(obs, env)
                obs, r, done, info = env.step(a)
                tot_r += r
                _ds = info.get("dist_surf")
                if _ds == _ds:
                    _min_ds = min(_min_ds, float(_ds))
                if done:
                    break
            n_steps_total += i + 1
            m = navigate_metrics(info, env.cfg["success_radius"])
            rows.append({**m, "steps": i + 1, "n_collide": info["n_collide"],
                         "level": info["level"], "instruction": info["instruction"],
                         "reward": tot_r, "success": info["success"]})
            # 每回合 `(seed, SR, level, ever_found, min_dsurf, timeout, n_collide)`：
            # 供配对 bootstrap / McNemar 使用（与 VLA 评测同口径）。
            # **level**：主结果是混合层级（L1/L2/L3）的平均，不记录就无法回答
            # 「差距集中在哪一层」。
            # **min_dsurf**：失败模式分解的关键量 —— 它把「失败」拆成
            # 「从未接近（搜索/感知）」与「接近了但没命中（控制/精度）」，
            # 是定位「模型只达到示范 67%」这一差距的手段。
            # 示范侧不额外跑检测，故 `ever_found` 记 -1（未知），不影响 min_dsurf 分析。
            detail.append((int(s), float(m["SR"]), str(info["level"]), -1,
                           (99.0 if _min_ds == float("inf") else float(_min_ds)),
                           int(bool(info.get("timeout", False))),
                           int(info["n_collide"])))

            if figure and ep == 0 and rep == 0:
                img = render_topdown(env.scene, env.pose, scale=8,
                                     pedestrians=env.peds, goal=env.task["goal"])
                _draw_path(img, env, scale=8)
                os.makedirs(os.path.dirname(figure), exist_ok=True)
                from PIL import Image
                Image.fromarray(img).save(figure)

        if rows:
            rep_rows.append({
                "rep": rep,
                "SR": float(np.mean([r["SR"] for r in rows])),
                "SPL": float(np.nanmean([r["SPL"] for r in rows])),
                "NE": float(np.mean([r["NE"] for r in rows])),
                "n": len(rows),
            })
        all_rows.extend(rows)
        all_detail.extend(detail)

    rows, detail = all_rows, all_detail
    dt = time.time() - t0
    from .plot_results import wilson
    _succ = [r["SR"] for r in rows]
    _k = int(sum(_succ))
    _lo, _hi = wilson(_k, len(rows)) if rows else (0.0, 0.0)
    agg = {
        "policy": policy_name,
        "level": level or "mixed",
        "episodes": episodes,
        "repeats": repeats,
        "SR": float(np.mean([r["SR"] for r in rows])),
        "SR_lo": float(_lo), "SR_hi": float(_hi),
        "SPL": float(np.nanmean([r["SPL"] for r in rows])),
        "NE": float(np.mean([r["NE"] for r in rows])),
        "TL": float(np.nanmean([r["TL"] for r in rows])),
        "steps": float(np.mean([r["steps"] for r in rows])),
        "n_collide": float(np.mean([r["n_collide"] for r in rows])),
        "sec_per_episode": round(dt / max(episodes * repeats, 1), 3),
        "ms_per_step": round(1000 * dt / max(n_steps_total, 1), 2),
    }
    if len(rep_rows) > 1:
        # 重复轮次之间的离散度（论文口径：重复 N 次取平均，报组间标准差）
        agg["rep_SR_mean"] = float(np.mean([r["SR"] for r in rep_rows]))
        agg["rep_SR_std"] = float(np.std([r["SR"] for r in rep_rows], ddof=1))
        agg["rep_SPL_mean"] = float(np.mean([r["SPL"] for r in rep_rows]))
        agg["rep_SPL_std"] = float(np.std([r["SPL"] for r in rep_rows], ddof=1))
    if verbose:
        extra = ""
        if len(rep_rows) > 1:
            extra = (f"  [重复 {len(rep_rows)} 轮: "
                     f"SR {100 * agg['rep_SR_mean']:.1f}% ± "
                     f"{100 * agg['rep_SR_std']:.1f}]")
        print(f"  {policy_name:9s} L={agg['level']:5s} n={episodes:3d} | "
              f"SR {100 * agg['SR']:5.1f}% [{100 * agg['SR_lo']:.1f},"
              f"{100 * agg['SR_hi']:.1f}]  SPL {agg['SPL']:.3f}  "
              f"NE {agg['NE']:5.2f} m  TL {agg['TL']:.2f}  "
              f"步数 {agg['steps']:5.1f}  碰撞 {agg['n_collide']:.0f}  "
              f"({agg['ms_per_step']:.1f} ms/步){extra}")
    if detail_out is not None:
        detail_out.extend(detail)
    return agg, rows


def _draw_path(img, env, scale=8):
    """在俯视图上叠加机器人轨迹。"""
    for p in env.traj:
        r, c = env.scene.w2c(p)
        rr, cc = int(r * scale), int(c * scale)
        img[max(0, rr - 1):rr + 2, max(0, cc - 1):cc + 2] = (255, 40, 40)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", default="random,frontier,oracle")
    ap.add_argument("--levels", default="mixed,L1,L2,L3")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--peds", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--lidar-noise", type=float, default=0.01)
    ap.add_argument("--camera-augment", type=float, default=0.25)
    ap.add_argument("--policy-kw", default=None,
                    help="策略构造参数的 JSON，如 '{\"mem_ttl\":25}'（用于调检测/记忆参数）")
    ap.add_argument("--detail-out", action="store_true",
                    help="把每回合 (seed, success) 写入 export/eval_detail/demo_<策略>_<协议>.json，"
                         "供配对 bootstrap / McNemar 使用")
    ap.add_argument("--ep-base", type=int, default=None,
                    help="回合种子的基值：给出后改用 base+ep*stride（与 VLA 评测对齐用）")
    ap.add_argument("--ep-stride", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1,
                    help="重复轮次：第 0 轮沿用原种子口径（配对检验用），"
                         "第 1 轮起用不同随机布局重跑，报「轮间均值±组间标准差」")
    ap.add_argument("--visible-start", action="store_true",
                    help="**起点即目标可见**（重采样直到目标在初始视野内且无遮挡）。"
                         "用于把「搜索」从评测里分离出去，只测导航能力")
    ap.add_argument("--calib", action="store_true",
                    help="用**标定场地**（单房间开阔场地，8×6 m，0 行人）。"
                         "注意：探针 `eval_learned_baseline` 与 VLA 评测必须用同一场地，"
                         "否则数字不可比（踩过：标定场地 sweep=60%%，多房间仅 15%%）")
    ap.add_argument("--figure", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.calib:
        env_kwargs = dict(max_steps=a.max_steps)     # calibrated_env 自带其余配置
    else:
        env_kwargs = dict(n_peds=a.peds, max_steps=a.max_steps,
                          lidar_noise=a.lidar_noise, camera_augment=a.camera_augment)

    print(f"\n=== E1 语义导航基线（{a.episodes} 回合/格，环境: "
          f"{'标定场地（单房间 8×6 m，0 行人）' if a.calib else f'{a.peds} 行人'}）===")
    results = []
    for pol in a.policies.split(","):
        for lv in a.levels.split(","):
            level = None if lv == "mixed" else lv
            det = [] if a.detail_out else None
            agg, _ = evaluate(pol, a.episodes, level=level, seed=a.seed,
                              env_kwargs=env_kwargs, calib=a.calib,
                              policy_kw=(json.loads(a.policy_kw) if a.policy_kw else None),
                              visible_start=a.visible_start,
                              ep_base=a.ep_base, ep_stride=a.ep_stride,
                              detail_out=det, repeats=a.repeats,
                              figure=a.figure if (pol == "oracle" and lv == "mixed") else None)
            if a.detail_out and det is not None:
                import json as _json
                os.makedirs("export/eval_detail", exist_ok=True)
                # 文件名必须同时区分**协议**与**场地**，否则不同场地的评测会互相覆盖。
                # 踩过的坑：一次「多房间」探测写进了标定场地的 `demo_sweep_det_std.json`，
                # 使该对象变成「主文件是多房间、追加批次是标定」的**混合协议**结果
                # （合并出 24.5%，而标定真值是 33.0%），且**不会报错**。
                tag = "vis" if a.visible_start else "std"
                if not a.calib:
                    tag = "mr_" + tag
                # 追加批次（`--ep-base` 取非默认值）写 `_s<base>` 后缀文件，
                # **不覆盖**主结果；`plot_results.load()` 会自动合并两者。
                _nm = (f"demo_{pol}_{tag}" if (a.ep_base is None or a.ep_base == 9000)
                       else f"demo_{pol}_{tag}_s{a.ep_base}")
                with open(f"export/eval_detail/{_nm}.json", "w") as fh:
                    _json.dump({"policy": pol, "protocol": tag,
                                "episodes": len(det), "detail": det}, fh)
            results.append(agg)

    # 汇总表（按策略 × 层级）
    print(f"\n{'策略':<10s} {'层级':<7s} {'SR [95%CI]':>16s} {'SPL':>7s} "
          f"{'NE(m)':>7s} {'TL':>6s} {'轮间σ':>7s}")
    for r in results:
        ci = f"{100 * r['SR']:.1f}% [{100 * r.get('SR_lo', 0):.0f}," \
             f"{100 * r.get('SR_hi', 0):.0f}]"
        rs = f"{100 * r['rep_SR_std']:.1f}" if "rep_SR_std" in r else "—"
        print(f"{r['policy']:<10s} {r['level']:<7s} {ci:>16s} "
              f"{r['SPL']:>7.3f} {r['NE']:>7.2f} {r['TL']:>6.2f} {rs:>7s}")

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n已写入 {a.out}")


if __name__ == "__main__":
    main()
