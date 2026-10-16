"""生成论文用的**森林图**（forest plot）—— 主创新 A 的最终结果。

数据来源：`export/eval_detail/*.json`（每回合的 `(seed, success)`）。
这些文件由

    sim/run_eval.py          --detail-out   （示范策略）
    smolvla_finetune_sim.py  --eval-only    （VLA）

产出，两边使用**同一套回合集**（`--ep-base 9000 --ep-stride 37`），
因此可以做**配对**统计。

统计方法：
- 单点比例的 95% 区间用 **Wilson score interval**（比例接近 0/1 时优于正态近似）；
- 保留率（模型 / 自身示范）的 95% 区间用**配对 bootstrap** —— 分子分母来自
  同一回合集，必须按回合重采样才能正确反映相关性；
- 两个模型的配对比较用 **McNemar 精确检验**（双侧）。

用法：
    /usr/bin/python3 -m sim.plot_results            # 默认输出 PNG + PDF
    /usr/bin/python3 -m sim.plot_results --out fig1 # 自定义文件名
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from math import comb, sqrt

import numpy as np

# ---------------------------------------------------------------- 统计工具
Z = 1.959964


def _detail_paths(name):
    """主文件 + 所有追加批次文件（`_s<base>.json`）。

    追加批次（`--eval-seed` 取非默认值）是为了**提高配对检验的功效**：
    回合数翻倍后 McNemar 才有机会看出 4 pp 级的数据效应。
    追加文件必须与主文件**合并**使用，且 seed 不得重叠。
    """
    base = os.path.join("export", "eval_detail", name)
    ps = [base + ".json"] + sorted(glob.glob(base + "_s*.json"))
    return [p for p in ps if os.path.exists(p)]


def load(name):
    """读每回合结果 → {seed: success}（**自动合并追加批次**）。

    兼容两种落盘格式：旧的 `[seed, SR]` 与新加的 `[seed, SR, level]`。
    """
    out = {}
    for p in _detail_paths(name):
        with open(p) as f:
            d = json.load(f)
        for e in d["detail"]:
            s = int(e[0])
            if s in out and out[s] != int(e[1]):
                raise ValueError(f"{name}: seed {s} 在多个文件中结果不一致")
            out[s] = int(e[1])
    return out


def load_levels(name):
    """读每回合层级 → {seed: level}；旧格式（无层级）**自动反推**。

    反推依据（已在 `env._new_scene` 处确认）：`reset(seed=s)` 会用 `s`
    **重新播种** `self.rng` 并由 `s` 重建场景，因此「回合内容」——包括
    任务层级——**完全由 seed 决定**，与父 env 实例、策略历史都无关。
    该结论已用带层级的 `demo_oracle_std.json` 对拍验证。
    """
    lv = {}
    seeds = []
    for p in _detail_paths(name):
        with open(p) as f:
            d = json.load(f)
        for e in d["detail"]:
            seeds.append(int(e[0]))
            if len(e) >= 3:
                lv[int(e[0])] = str(e[2])
    # **缺失的 seed 必须反推补上**：只有一部分批次带层级落盘时（旧格式 + 新格式混用），
    # 直接返回 lv 会让另一半回合**整段消失**，层级统计只算了一半样本
    # （踩过：`ft_detf` 的 L3 被算成 15.7%，而补全后是 22.1%）。
    miss = sorted({s for s in seeds if s not in lv})
    if miss:
        lv.update(levels_from_seeds(miss))
    return {s: lv[s] for s in seeds}


_LEVEL_CACHE = {}


def levels_from_seeds(seeds):
    """由 seed 反推任务层级（无需重跑评测）。"""
    todo = [s for s in seeds if s not in _LEVEL_CACHE]
    if todo:
        from .env import calibrated_env
        env = calibrated_env(seed=9000)
        for s in todo:
            env.reset(seed=s)
            _LEVEL_CACHE[s] = str(env.task["level"]) if env.task else "?"
    return {s: _LEVEL_CACHE[s] for s in seeds}


def wilson(k, n, z=Z):
    """Wilson score interval（比例接近 0/1 时优于正态近似）。"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b, c):
    """双侧 McNemar 精确检验。b/c 为两个不一致格的计数。"""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2.0 ** n)


def ratio_ci_paired(model, demo, idx, n_boot=10000, seed=0):
    """配对 bootstrap：按**回合**重采样，同时重算分子与分母。

    分子（模型 SR）与分母（示范 SR）来自同一批回合，二者相关；
    若按独立比例做区间会低估方差。这里对回合索引重采样即可正确反映。
    """
    common = sorted(set(model) & set(demo))
    def calc(ix):
        m = float(np.mean([model[common[i]] for i in ix]))
        d = float(np.mean([demo[common[i]] for i in ix]))
        return m / d if d > 0 else np.nan
    point = calc(range(len(common)))
    rg = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        v = calc(rg.integers(0, len(common), len(common)))
        if np.isfinite(v):
            bs.append(v)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return point, float(lo), float(hi), len(common)


def paired_p(a_name, b_name):
    A, B = load(a_name), load(b_name)
    cm = sorted(set(A) & set(B))
    a = np.array([A[s] for s in cm])
    b = np.array([B[s] for s in cm])
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    return mcnemar_exact(n10, n01), n10, n01, len(cm)


# ---------------------------------------------------------------- 绘图
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="export/fig_forest")
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--boot", type=int, default=10000,
                    help="配对 bootstrap 重采样次数（大纲 §3.2 要求 10 000）")
    ap.add_argument("--by-level", action="store_true",
                    help="按任务层级（L1/L2/L3）拆分成功率并打印对比表")
    ap.add_argument("--holm", action="store_true",
                    help="把全部配对比较做 Holm 多重比较校正（大纲 §3.2 要求）")
    ap.add_argument("--failure-modes", action="store_true",
                    help="失败模式分解：把失败按「最近接近距离」分成 搜索/接近/精度 三类")
    ap.add_argument("--data-variance", action="store_true",
                    help="数据侧方差：同配置、不同**示范集**训出的模型之间的离散度")
    ap.add_argument("--replication", action="store_true",
                    help="多种子重复实验分析：区分「训练侧方差」与效应量")
    ap.add_argument("--mem-figure", action="store_true",
                    help="生成「时序记忆消融」图（2 协议 × 2 数据规模 × 3 档位）")
    a = ap.parse_args()
    if a.by_level:
        return by_level(a)
    if a.holm:
        return holm_report(a)
    if a.failure_modes:
        return failure_modes(a)
    if a.data_variance:
        return data_variance(a)
    if a.replication:
        return replication(a)
    if a.mem_figure:
        return mem_figure(a)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8,
        "ytick.major.width": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    C_DET = "#1f6fb4"      # 无特权链路
    C_PRI = "#c0561f"      # 特权链路
    C_DEMO = "#7a7a7a"     # 示范策略（上界参照）
    C_MID = "#7fb3d9"      # 条件化示范 + 纯端到端（分离数据/架构效应所需的中间格）
    GREY = "#666666"
    XTXT = 70.0            # 右侧数值列 x
    XBRK = 108.0           # McNemar 括号 x

    # ---- 载入 ----
    data = {}
    for tag in ("std", "vis"):
        for key in ("demo_sweep", "demo_sweep_det", "ft_detf", "ft_calib_f1",
                    "ft_detf_e2e"):
            data[(key, tag)] = load(f"{key}_{tag}")
        # 参照对象：**同一回合集、同协议**测得的随机下界与 oracle 上界
        for key in ("demo_random", "demo_oracle"):
            try:
                _d = load(f"{key}_{tag}")
            except FileNotFoundError:
                continue
            if _d:                      # 空文件（不存在/正在写入）不入库
                data[(key, tag)] = _d

    # **回合数一致性自检**：各个对象必须评测在同样多的回合上。
    # 早先一次诊断脚本把 `demo_sweep_det_std.json` 改了名，导致该对象
    # 只剩第二批的 200 回合，而保留率的分母被静默换掉（33.0% → 另一个数）。
    # 这类「丢批次」不会报错，只会悄悄改变数字，所以在这里显式检查。
    _ns = {}
    for k, tag in data:
        if tag == "std":
            _ns[k] = len(data[(k, tag)])
    if _ns:
        _mode = max(set(_ns.values()), key=list(_ns.values()).count)
        _odd = {k: v for k, v in _ns.items()
                if v not in (_mode, 200) and not k.startswith("demo_")}
        if _odd:
            print(f"  [警告] 以下对象的回合数({_mode} 为主流)不一致：{_odd}")
            print("        常见原因：某个批次文件缺失/被改名（见 plot_results 注释）")

    def stat(key, tag):
        d = data[(key, tag)]
        n = len(d)
        if n == 0:            # 空文件（正在被写入）不应让整张图崩掉
            return float("nan"), float("nan"), float("nan"), 0, 0
        k = int(sum(d.values()))
        lo, hi = wilson(k, n)
        return k / n, lo, hi, n, k

    fig, axes = plt.subplots(
        1, 2, figsize=(11.2, 4.5), gridspec_kw={"width_ratios": [1.32, 1.0]})

    # ============ Panel A：闭环成功率 ============
    ax = axes[0]
    rows = []
    for tag, glabel in (("std", "Standard protocol"), ("vis", "Visible-start protocol")):
        rows.append(("group", glabel))
        rows.append(("bar", "sweep  —  demo policy (privileged)", "demo_sweep", tag, C_PRI))
        rows.append(("bar", "sweep_det  —  demo policy (image-only)",
                     "demo_sweep_det", tag, C_DET))
        rows.append(("bar", "VLA · privileged demos, end-to-end", "ft_calib_f1", tag, C_PRI))
        rows.append(("bar", "VLA · image-only demos, end-to-end",
                     "ft_detf_e2e", tag, C_MID))
        rows.append(("bar", "VLA · image-only demos + perception", "ft_detf", tag, C_DET))
        rows.append(("space", ""))

    ys, labels, colors = [], [], []
    y = 0.0
    for r in rows:
        if r[0] == "space":
            y -= 0.55
            continue
        ys.append(y)
        labels.append(r[1])
        colors.append(C_DEMO if r[0] == "group" else r[4])
        y -= 1.0

    for r, yy in zip([r for r in rows if r[0] != "space"], ys):
        if r[0] == "group":
            continue
        p, lo, hi, n, k = stat(r[2], r[3])
        if n == 0:
            continue
        is_demo = r[2].startswith("demo_")
        ax.plot([100 * lo, 100 * hi], [yy, yy], color=r[4], lw=1.6,
                solid_capstyle="round", zorder=2)
        # **实心 = 学到的模型；空心 = 示范策略（可达上界）**
        ax.plot([100 * p], [yy], "o" if not is_demo else "o", ms=6.8 if not is_demo else 6.2,
                markerfacecolor=("white" if is_demo else r[4]),
                markeredgecolor=r[4], markeredgewidth=1.3, zorder=3)
        # 数值放**右侧固定列**（标准森林图排法）：既对齐又不会与误差条/括号相撞
        ax.text(XTXT, yy,
                f"{100*p:5.1f}  [{100*lo:4.1f}, {100*hi:4.1f}]",
                va="center", ha="left", fontsize=7.3, color=r[4],
                family="DejaVu Sans Mono")

    # 分组标题
    gi = [i for i, r in enumerate([r for r in rows if r[0] != "space"])
          if r[0] == "group"]
    for idx in gi:
        ax.text(-1.0, ys[idx] + 0.42, labels[idx], fontsize=9.2, fontweight="bold",
                ha="left", va="bottom", color="#333333")

    ax.set_yticks(ys)
    ax.set_yticklabels(["" if r[0] == "group" else labels[i]
                        for i, r in enumerate([r for r in rows if r[0] != "space"])],
                       fontsize=8.2)
    ax.set_xlim(0, 152)
    ax.set_ylim(min(ys) - 1.0, max(ys) + 1.1)
    ax.set_xticks([0, 20, 40, 60])
    ax.axvline(XTXT - 6.0, color="#e8e8e8", lw=0.7, zorder=0)

    # ---- 配对检验标注：**标出两个可分离的效应** ----
    # 行序（Panel A，每个协议 6 行）：
    #   +0 组标题  +1 sweep  +2 sweep_det  +3 ft_calib_f1  +4 ft_detf_e2e  +5 ft_detf
    # std 组占 idx 0..5，vis 组占 6..11。
    # 相邻两行正好构成一个**干净的**主效应：
    #   ft_calib_f1 → ft_detf_e2e  换数据、架构不变  = 纯数据效应
    #   ft_detf_e2e → ft_detf      数据不变、换架构  = 纯架构效应
    # 早先只标 ft_detf vs ft_calib_f1，那一对**同时改了数据与架构**，不可归因。
    EFF = (("ft_detf_e2e", "ft_calib_f1", "data effect"),
           ("ft_detf", "ft_detf_e2e", "arch. effect"))
    for tag, base in (("std", 0), ("vis", 6)):
        for j, (an, bn, lab) in enumerate(EFF):
            i0, i1 = base + 3 + j, base + 4 + j
            p, n10, n01, ncm = paired_p(f"{an}_{tag}", f"{bn}_{tag}")
            xb = XBRK + j * 0.0
            y0, y1 = ys[i0], ys[i1]
            ax.plot([xb, xb], [y1 + 0.35, y0 - 0.35], color=GREY, lw=0.8, clip_on=False)
            for yy in (y1 + 0.35, y0 - 0.35):
                ax.plot([xb - 1.4, xb], [yy, yy], color=GREY, lw=0.8, clip_on=False)
            txt = (f"p = {p:.1e}" if p < 1e-3 else f"p = {p:.3f}")
            star = " *" if p <= 0.05 else " (n.s.)"
            ax.text(xb + 2.0, (y0 + y1) / 2.0, f"{lab} {txt}{star}",
                    fontsize=6.8, color="#333333" if p <= 0.05 else "#999999",
                    va="center", ha="left", clip_on=False)
            ax.plot([0, 66], [(y0 + y1) / 2.0] * 2, color="#ededed", lw=0.7, zorder=0)

    # ---- 同口径参照线：随机下界 / oracle 上界 ----
    # 二者都测在**与数据完全相同的回合集**上（seeds 9000 + 37k），可直接与各行比较。
    #
    # **参照线必须按协议分组画**：两种协议的随机下界差很多（常规 9.0% vs
    # 起点可见 24.0%）—— 后者通过重采样场景来保证「目标在视野内」，
    # 使随机策略的成功率被抬高 15 pp。用一条线贯穿两组会**误读**可见起点那一组。
    # 数值写进图例而不放在坐标区内（放区内会与刻度标签、数据相撞，踩过）。
    ref_handles = []
    for key, lab, col, ls in (("demo_random", "random policy", "#8a8a8a", (0, (4, 3))),
                              ("demo_oracle", "oracle planner (privileged)",
                               "#b4b4b4", (0, (1, 2)))):
        vals, seen = [], []
        for tag, i0, i1 in (("std", 0, 5), ("vis", 6, 11)):
            if (key, tag) not in data:
                continue
            pv, *_ = stat(key, tag)
            vals.append(pv)
            # 竖线只覆盖**该协议组**的行范围（y 轴向下增大）
            ax.plot([100 * pv, 100 * pv], [ys[i1] - 0.45, ys[i0] + 0.45],
                    color=col, lw=1.0, ls=ls, zorder=1)
            seen.append(f"{100*pv:.1f}%")
        if not vals:
            continue
        txt = seen[0] if len(seen) == 1 else f"{seen[0]} (std) / {seen[1]} (vis)"
        ref_handles.append(Line2D([], [], color=col, lw=1.0, ls=ls,
                                  label=f"{lab} — {txt}"))

    ax.set_xlabel("Closed-loop success rate  (%)", fontsize=9)
    _nA = len(data.get(("ft_detf", "std"), {}))
    ax.set_title(f"A · Navigation success (n = {_nA}, Wilson 95% CI)",
                 fontsize=9.8, loc="left", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", lw=0.4, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)
    handles = [
        Line2D([], [], marker="o", ls="", ms=6.2, markerfacecolor="white",
               markeredgecolor="#555555", markeredgewidth=1.3,
               label="demo policy (upper bound)"),
        Line2D([], [], marker="o", ls="", ms=6.8, markerfacecolor="#555555",
               markeredgecolor="#555555", label="VLA model (learned)"),
        Line2D([], [], color="#7fb3d9", lw=2.2,
               label="image-only demos, end-to-end"),
        Line2D([], [], color="#1f6fb4", lw=2.2, label="image-only + perception"),
        Line2D([], [], color="#c0561f", lw=2.2, label="privileged chain"),
    ] + ref_handles

    # ============ Panel B：保留率 ============
    bx = axes[1]
    ret_rows = []
    for tag, glabel in (("std", "Standard protocol"), ("vis", "Visible-start protocol")):
        ret_rows.append(("group", glabel))
        ret_rows.append(("bar", "privileged", "ft_calib_f1", "demo_sweep",
                         tag, C_PRI))
        ret_rows.append(("bar", "image-only, end-to-end", "ft_detf_e2e",
                         "demo_sweep_det", tag, C_MID))
        ret_rows.append(("bar", "image-only + perception", "ft_detf",
                         "demo_sweep_det", tag, C_DET))
        ret_rows.append(("space", ""))

    rys, rlabels = [], []
    y = 0.0
    for r in ret_rows:
        if r[0] == "space":
            y -= 0.8
            continue
        rys.append(y)
        rlabels.append(r[1])
        y -= 1.0

    for r, yy in zip([r for r in ret_rows if r[0] != "space"], rys):
        if r[0] == "group":
            continue
        m, d = data[(r[2], r[4])], data[(r[3], r[4])]
        if not m or not d:
            continue
        p, lo, hi, ncm = ratio_ci_paired(m, d, None, n_boot=a.boot)
        bx.plot([100 * lo, 100 * hi], [yy, yy], color=r[5], lw=1.6,
                solid_capstyle="round", zorder=2)
        bx.plot([100 * p], [yy], "o", color=r[5], ms=6.5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        bx.text(110.0, yy, f"{100*p:3.0f}%  [{100*lo:3.0f}, {100*hi:3.0f}]",
                va="center", ha="left", fontsize=7.3, color=r[5],
                family="DejaVu Sans Mono")

    for idx in [i for i, r in enumerate([r for r in ret_rows if r[0] != "space"])
                if r[0] == "group"]:
        bx.text(-1.0, rys[idx] + 0.42, rlabels[idx], fontsize=9.2,
                fontweight="bold", ha="left", va="bottom", color="#333333")

    bx.set_yticks(rys)
    bx.set_yticklabels(["" if r[0] == "group" else rlabels[i]
                        for i, r in enumerate([r for r in ret_rows if r[0] != "space"])],
                       fontsize=8.2)
    bx.set_xlim(0, 155)
    bx.set_ylim(min(rys) - 1.0, max(rys) + 1.1)
    bx.set_xticks([0, 20, 40, 60, 80, 100])
    bx.axvline(100, color=GREY, lw=0.8, ls=(0, (3, 3)), alpha=0.55)
    bx.axvline(105.0, color="#e8e8e8", lw=0.7, zorder=0)
    bx.set_xlabel("Retention rate  =  model SR / its own demo SR  (%)", fontsize=9)
    bx.set_title("B · Retention (paired bootstrap)", fontsize=9.8, loc="left", pad=8)
    bx.spines[["top", "right"]].set_visible(False)
    bx.grid(axis="x", lw=0.4, color="#dddddd", zorder=0)
    bx.set_axisbelow(True)

    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=7.4, bbox_to_anchor=(0.5, 0.030), handletextpad=0.6,
               columnspacing=1.8, labelspacing=0.5)
    fig.text(0.008, 0.002,
             "All comparisons use the identical episode set (seeds 9000+37k); "
             "protocols are not comparable to each other (visible-start resamples the initial pose).",
             fontsize=6.9, color=GREY)
    fig.tight_layout(rect=(0, 0.135, 1, 1))

    for ext in ("png", "pdf"):
        fp = f"{a.out}.{ext}"
        os.makedirs(os.path.dirname(fp) or ".", exist_ok=True)
        fig.savefig(fp, dpi=a.dpi, bbox_inches="tight")
        print(f"已保存 {fp}")

    # ---- 同时打印一份可复算的文本摘要 ----
    print("\n数值摘要（可直接核对）")
    for tag in ("std", "vis"):
        p, n10, n01, ncm = paired_p(f"ft_detf_{tag}", f"ft_calib_f1_{tag}")
        print(f"  [{tag}] McNemar  b={n10} c={n01} n={ncm}  p={p:.5f}")
        for m, d, lab in (("ft_detf", "demo_sweep_det", "image-only"),
                          ("ft_calib_f1", "demo_sweep", "privileged")):
            pt, lo, hi, nc = ratio_ci_paired(data[(m, tag)], data[(d, tag)],
                                             None, n_boot=a.boot)
            print(f"  [{tag}] 保留率 {lab:<11} {100*pt:.0f}% "
                  f"[{100*lo:.0f}, {100*hi:.0f}]  (配对 {nc} 回合)")




# ---------------------------------------------------------------- 按层级拆分
def by_level(a):
    """`--by-level`：把成功率按任务层级拆开。

    动因：混合层级的平均值会掩盖结构。已知 `perception.features_from_dets`
    **只编码「指令所指目标」的方位/距离，丢弃了参考物**（`_ref`/`_rel`），
    因此 L3（「桌子左边 的椅子」）所需的信息**根本不在输入里**。
    若该假设成立，应表现为 **L3 成功率显著低于 L1/L2**。
    """
    keys = [("demo_sweep", "std"), ("demo_sweep_det", "std"),
            ("ft_calib_f1", "std"), ("ft_detf", "std"),
            ("demo_sweep", "vis"), ("demo_sweep_det", "vis"),
            ("ft_calib_f1", "vis"), ("ft_detf", "vis")]
    lvls = ["L1", "L2", "L3"]
    print("%-16s %-5s %8s %8s %8s %8s" %
          ("对象", "协议", "整体", "L1", "L2", "L3"))
    print("-" * 60)
    tab = {}
    for name, tag in keys:
        try:
            res = load(name + "_" + tag)
            lv = load_levels(name + "_" + tag)
        except FileNotFoundError:
            continue
        if lv is None:
            print("%-16s %-5s   （无层级信息，需重跑评测）" % (name, tag))
            continue
        row = {"all": np.mean(list(res.values()))}
        cells = []
        for L in lvls:
            idx = [s for s in res if lv.get(s) == L]
            row[L] = float(np.mean([res[s] for s in idx])) if idx else np.nan
            cells.append("%7.1f%%" % (100 * row[L]) if idx else "       -")
        tab[(name, tag)] = row
        print("%-16s %-5s %7.1f%% %s" % (name, tag, 100 * row["all"], "".join(cells)))

    # 配对差：无特权链路 vs 特权链路，逐层看
    print("\n按层配对（同回合集，McNemar 精确检验）")
    print("%-6s %-28s %9s %9s %11s" % ("协议", "对比", "前者", "后者", "p"))
    for tag in ("std", "vis"):
        for a_name, b_name, lab in (
                ("demo_sweep_det", "demo_sweep", "无特权 vs 特权示范"),
                ("ft_detf", "ft_calib_f1", "VLA 无特权 vs VLA 特权")):
            try:
                A, B = load(a_name + "_" + tag), load(b_name + "_" + tag)
                lv = load_levels(a_name + "_" + tag)
            except FileNotFoundError:
                continue
            if lv is None:
                continue
            for L in lvls:
                idx = sorted(s for s in (set(A) & set(B)) if lv.get(s) == L)
                if not idx:
                    continue
                x = np.array([A[s] for s in idx]); y = np.array([B[s] for s in idx])
                b01 = int(((x == 1) & (y == 0)).sum())
                b10 = int(((x == 0) & (y == 1)).sum())
                pv = mcnemar_exact(b01, b10)
                print("%-6s %-28s %8.1f%% %8.1f%% %11.2g   (n=%d, b=%d, c=%d)" %
                      (tag, lab + " · " + L, 100 * x.mean(), 100 * y.mean(), pv,
                       len(idx), b01, b10))
    return tab




# ---------------------------------------------------------------- 记忆消融图
def mem_figure(a):
    """`--mem-figure`：时序记忆消融。

    这是一个**独立于图 1 的实验**（2 协议 × 2 数据规模 × 3 感知档位），
    数据全部**严格配对**（同命令、仅档位不同），因此不能塞进图 1 的行结构里。

    要点：示范策略依赖 `mem_ttl=25` 的跨帧记忆（关掉它使示范 33.0% → 16.0%），
    而模型原本只有单帧输入。补上记忆槽（8 维）后的效应见各格。
    """
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.linewidth": 0.8, "pdf.fonttype": 42})

    C = {"none": "#9aa0a6", "sf": "#1f6fb4", "mem": "#c0561f"}
    LAB = {"none": "no perception\n(state 5-D)", "sf": "single-frame\n(4-D)",
           "mem": "single-frame + memory\n(8-D)"}
    SIZES = [("31k", "ft_detf_none31k_%s", "ft_detf_sf31k_%s", "ft_detf_mem_%s",
              "31 125 samples"),
             ("96k", None, "ft_detf_sf96k_%s", "ft_detf_mem96k_%s",
              "95 204 samples")]
    TAGS = [("std", "Standard protocol"), ("vis", "Visible-start protocol")]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for ax, (tag, tname) in zip(axes, TAGS):
        xs, hs, los, his, cols = [], [], [], [], []
        xt, xl = [], []
        i = 0.0
        for _, n_none, n_sf, n_mem, slabel in SIZES:
            start = i
            for key, cls_ in (("none", "none"), ("sf", "sf"), ("mem", "mem")):
                nm = {"none": n_none, "sf": n_sf, "mem": n_mem}[key]
                if nm is None:
                    continue
                name = nm % tag
                try:
                    d = load(name)
                except Exception:
                    continue
                k = sum(d.values()); N = len(d)
                lo, hi = wilson(k, N)
                xs.append(i); hs.append(100 * k / N)
                los.append(100 * lo); his.append(100 * hi); cols.append(C[cls_])
                i += 1.0
            xt.append((start + i - 1) / 2.0)
            xl.append(slabel)
            i += 1.0
        ax.bar(xs, hs, color=cols, width=0.62, zorder=3)
        ax.errorbar(xs, hs, yerr=[np.array(hs) - np.array(los),
                                  np.array(his) - np.array(hs)],
                    fmt="none", ecolor="#333333", elinewidth=1.0, capsize=3, zorder=4)
        for xx, hh in zip(xs, hs):
            ax.text(xx, hh + 1.2, f"{hh:.1f}", ha="center", fontsize=7.6,
                    color="#333333")
        # 随机下界（同回合集）
        floor = load(f"demo_random_{tag}")
        fl = 100 * sum(floor.values()) / len(floor)
        ax.axhline(fl, color="#8a8a8a", lw=1.0, ls=(0, (4, 3)), zorder=2)
        # 参照标签放**坐标区内左侧**：放右侧会越过坐标区、还可能被裁掉
        ax.text(min(xs) - 0.42, fl, f"random\n{fl:.1f}%", fontsize=6.8,
                color="#8a8a8a", va="center", ha="left", linespacing=1.15)
        ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=8.2)
        ax.set_ylim(0, max(his) + 8)
        ax.set_ylabel("Closed-loop success rate (%)" if tag == "std" else "")
        ax.set_title(f"{'A' if tag == 'std' else 'B'} · {tname}", fontsize=9.4,
                     loc="left", pad=6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", lw=0.4, color="#dddddd", zorder=0)
        ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=C[k], label=LAB[k].replace("\n", " "))
                        for k in ("none", "sf", "mem")],
               loc="lower center", ncol=3, frameon=False, fontsize=7.8,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Temporal memory ablation — data strictly paired "
                 "(identical collection command, only the feature mode differs)",
                 fontsize=8.6, y=1.0, x=0.008, ha="left", color="#333333")
    fig.tight_layout(rect=(0, 0.075, 1, 0.955))
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=a.dpi, bbox_inches="tight")
        print(f"已保存 {a.out}.{ext}")
    return None




# ---------------------------------------------------------------- 多种子重复
# 配置 → (数据规模标签, 档位标签, 各种子对应的 detail 名前缀)
# 种子 0 是本轮之前的单次运行（`ft_detf_*`），种子 1/2 为 `rep_*`。
REPL_CONFIGS = [
    ("c1", "31k", "single-frame 4-D", "ft_detf_sf31k", "rep_c1_s1", "rep_c1_s2"),
    ("c2", "31k", "memory  8-D",       "ft_detf_mem",   "rep_c2_s1", "rep_c2_s2"),
    ("c3", "96k", "single-frame 4-D", "ft_detf_sf96k", "rep_c3_s1", "rep_c3_s2"),
    ("c4", "96k", "memory  8-D",       "ft_detf_mem96k", "rep_c4_s1", "rep_c4_s2"),
]


def replication(a, episodes=200):
    """`--replication`：每一配置跑 3 个训练种子，把「训练侧方差」显式量出来。

    单个训练种子之间的差异**不会被配对 McNemar 计入**（它只消除评测噪声），
    因此当效应量与种子间波动同量级时，单次运行得出的结论不可靠。
    这里对每个配置报告 3 个种子的 SR、均值 ± 标准差，并把
    **种子间标准差与效应量并列**，让「效应量是否超过噪声」一目了然。

    为公平起见，所有种子都只取**同一批 200 回合**（seed 9000 + 37k，ep 0..199）。
    """
    from itertools import combinations

    def first_n(name, n=episodes):
        d = load(name)
        ks = sorted(d)[:n]
        return {k: d[k] for k in ks}

    print(f"每配置 {len(REPL_CONFIGS[0]) - 3} 个训练种子，每种子同一批 {episodes} 回合\n")
    res = {}
    print("%-6s %-16s %-8s %8s %8s %8s | %8s %8s %7s" %
          ("配置", "档位", "数据", "种子0", "种子1", "种子2", "均值", "标准差", "n"))
    print("-" * 88)
    for cid, sz, lab, s0, s1, s2 in REPL_CONFIGS:
        vals, names = [], []
        for nm in (s0, s1, s2):
            try:
                d = first_n(nm)
            except Exception:
                continue
            if not d:
                continue
            vals.append(100 * sum(d.values()) / len(d))
            names.append(nm)
        if not vals:
            print("%-6s %-16s %-8s  （无数据）" % (cid, lab, sz))
            continue
        res[cid] = vals
        cells = [f"{v:7.1f}%" for v in vals] + ["      -"] * (3 - len(vals))
        m = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        print("%-6s %-16s %-8s %s | %7.2f%% %7.2f %7d" %
              (cid, lab, sz, " ".join(cells), m, sd, len(vals)))

    # **种子层面的 t 检验**：这才是与结论匹配的检验单位。
    # 配对 McNemar 把「同一模型的 200 个回合」当作独立样本，
    # 但真正的重复单位是**训练种子** —— 不同种子训练出的模型才算独立重复。
    print("\n种子层面的效应检验（重复单位 = 训练种子，这才是正确的检验单位）")
    print("%-22s %9s %9s %9s %9s %8s %9s" %
          ("对比", "均值A", "均值B", "效应", "种子SD", "t", "p"))
    for ca, cb, lab in (("c1", "c2", "31k: 记忆 − 单帧"),
                        ("c3", "c4", "96k: 记忆 − 单帧")):
        if ca not in res or cb not in res or len(res[ca]) < 2 or len(res[cb]) < 2:
            continue
        va, vb = np.array(res[ca]), np.array(res[cb])
        eff = float(vb.mean() - va.mean())
        se = float(np.sqrt(va.var(ddof=1) / len(va) + vb.var(ddof=1) / len(vb)))
        t = eff / se if se > 0 else float("inf")
        # Welch–Satterthwaite 自由度 → 双侧 p（用 t 分布的生存函数近似）
        df = ((va.var(ddof=1)/len(va) + vb.var(ddof=1)/len(vb))**2 /
              ((va.var(ddof=1)/len(va))**2/(len(va)-1) +
               (vb.var(ddof=1)/len(vb))**2/(len(vb)-1))) if se > 0 else 1.0
        from math import lgamma, log, exp
        def _tcdf(x, v):
            # 用不完全 Beta 的连分数近似双侧 t 检验 p 值（避免引入 scipy 依赖）
            import math
            def betacf(a, b, x, itmax=200, eps=3e-16):
                qab, qap, qam = a+b, a+1.0, a-1.0
                c, d = 1.0, 1.0 - qab*x/qap
                if abs(d) < 1e-300: d = 1e-300
                d = 1.0/d; h = d
                for m in range(1, itmax+1):
                    m2 = 2*m
                    aa = m*(b-m)*x/((qam+m2)*(a+m2))
                    d = 1.0 + aa*d
                    if abs(d) < 1e-300: d = 1e-300
                    c = 1.0 + aa/c
                    if abs(c) < 1e-300: c = 1e-300
                    d = 1.0/d; h *= d*c
                    aa = -(a+m)*(qab+m)*x/((a+m2)*(qap+m2))
                    d = 1.0 + aa*d
                    if abs(d) < 1e-300: d = 1e-300
                    c = 1.0 + aa/c
                    if abs(c) < 1e-300: c = 1e-300
                    d = 1.0/d; de = d*c; h *= de
                    if abs(de-1.0) < eps: break
                return h
            def betai(a, b, x):
                if x <= 0: return 0.0
                if x >= 1: return 1.0
                bt = math.exp(lgamma(a+b)-lgamma(a)-lgamma(b)+a*math.log(x)+b*math.log(1-x))
                if x < (a+1)/(a+b+2):
                    return bt*betacf(a, b, x)/a
                return 1.0 - bt*betacf(b, a, 1-x)/b
            p = 0.5 * betai(0.5*v, 0.5, v/(v+x*x))
            return 2*p
        pv = _tcdf(t, max(df, 1.0))
        pv = min(1.0, max(0.0, pv))
        sd_a = float(np.sqrt(va.var(ddof=1)/len(va) + vb.var(ddof=1)/len(vb)))
        print("%-22s %8.2f%% %8.2f%% %+8.2f pp %8.2f %8.2f %9.4f" %
              (lab, va.mean(), vb.mean(), eff, sd_a, t, pv))

    sds = [float(np.std(v, ddof=1)) for v in res.values() if len(v) > 1]
    if sds:
        print(f"\n种子间标准差：中位 {np.median(sds):.2f} pp，范围 "
              f"[{min(sds):.2f}, {max(sds):.2f}] pp")
    print("\n对照效应量（单次运行的配对估计）：")
    for lab, txt in (("31k 记忆 vs 单帧", "+7.75 pp (p = 3.7e-4)"),
                     ("96k 记忆 vs 单帧", "+2.75 pp (p = 0.215)"),
                     ("96k 单帧 vs 无感知", "+2.00 pp")):
        print(f"  {lab:<22} {txt}")

    print("\n逐种子配对（同评测回合集，故仍是配对比较）：")
    for cid, cid2 in (("c1", "c2"), ("c3", "c4")):
        if cid not in res or cid2 not in res:
            continue
        cfg_a = next(c for c in REPL_CONFIGS if c[0] == cid)
        cfg_b = next(c for c in REPL_CONFIGS if c[0] == cid2)
        print(f"  [{cfg_a[1]}] {cfg_b[2]} vs {cfg_a[2]}")
        for i in range(3):
            na = cfg_a[3 + i]; nb = cfg_b[3 + i]
            try:
                p, b01, b10, nm = paired_p(na + "_std", nb + "_std")
                da, db = first_n(na + "_std"), first_n(nb + "_std")
                print("    种子%d: %5.2f%% vs %5.2f%%  (差 %+5.2f pp，配对 p = %.3g)"
                      % (i, 100*sum(da.values())/len(da), 100*sum(db.values())/len(db),
                         100*(sum(db.values())/len(db) - sum(da.values())/len(da)), p))
            except Exception:
                pass
    return res




# ---------------------------------------------------------------- 数据侧方差
# 同配置（31k / 记忆 8 维）、不同**示范集**训出的模型：
#   seed 0  → `nav_detf_mem.h5`（本轮主线用的那一份）
#   101/202/303 → 各自独立采集的示范集
DSVAR_MODELS = [("示 范 集 0（seed=0）", "ft_detf_mem_std"),
                ("示范集 A（seed=101）", "rep_dv101_std"),
                ("示范集 B（seed=202）", "rep_dv202_std"),
                ("示范集 C（seed=303）", "rep_dv303_std")]


def data_variance(a, episodes=200):
    """`--data-variance`：把「采集（示范集）」当作重复单位。

    ## 为什么需要这个量

    §4.23.9 测出**训练种子**的 SD 只有 0.67 pp。但两个名义等价的 96k 单帧模型
    （`ft_detf` 24.5% vs `ft_detf_sf96k` 18.67%）差了 5.8 pp —— 远超种子噪声。
    这说明主要方差在**示范集**这一侧，而此前从未量化过。

    这里用**同一配置、只换示范集**的若干模型来估计「采集间 SD」，
    并与「同数据集内的种子 SD」并列，得到两个方差分量的相对大小。

    为公平，所有模型都只取同一批 200 回合。
    """
    def first(name, n=episodes):
        d = load(name)
        return {k: d[k] for k in sorted(d)[:n]}

    vals, labs = [], []
    print(f"每模型同一批 {episodes} 回合\n")
    print("%-24s %9s" % ("示范集", "SR"))
    print("-" * 36)
    for lab, nm in DSVAR_MODELS:
        try:
            d = first(nm)
        except Exception:
            print("%-24s         -" % lab)
            continue
        if not d:
            continue
        v = 100 * sum(d.values()) / len(d)
        vals.append(v); labs.append(lab)
        print("%-24s %8.2f%%" % (lab, v))
    if len(vals) < 2:
        print("\n（数据不足，需至少两份独立示范集）")
        return vals

    m = float(np.mean(vals))
    sd_data = float(np.std(vals, ddof=1))
    print("\n%-24s %8.2f%%" % ("均值", m))
    print("%-24s %8.2f pp" % ("**采集间 SD**", sd_data))
    print("%-24s %8.2f pp" % ("（对照）训练种子 SD", 0.67))
    print("%-24s %8.1f×" % ("采集 / 种子 方差比", sd_data / 0.67))

    # 两个方差分量：把总离散度写成 SD_total = sqrt(SD_data^2 + SD_seed^2)
    tot = float(np.sqrt(sd_data ** 2 + 0.67 ** 2))
    share = 100 * sd_data ** 2 / (sd_data ** 2 + 0.67 ** 2)
    print("\n方差分解（同数据集内 n 个种子 vs n 份数据集）：")
    print("  采集侧贡献 %.1f%% 的方差；种子侧 %.1f%%" % (share, 100 - share))
    print("  合成 SD_total ≈ %.2f pp" % tot)
    print("\n→ 若要检出 %.1f pp 的效应（当前量级），" % sd_data)
    print("  按 2 样本 t 检验、80%% 功效估算需 **约 %d 份独立示范集**"
          % max(2, int(np.ceil(2 * (2.8 * tot / max(sd_data, 0.1)) ** 2))))
    return vals




# ---------------------------------------------------------------- 失败模式分解
def _ep_records(name):
    """读逐回合记录 → `[{seed, sr, lvl, found, dmin, timeout, ncol}, ...]`。

    与 `load()` 的区别：这里要的是**除了成功与否之外的诊断量**。
    旧格式（3 字段）没有诊断量，`dmin` 记为 None 并在统计时排除。
    """
    out = []
    for p in _detail_paths(name):
        with open(p) as f:
            d = json.load(f)
        for e in d["detail"]:
            out.append({"seed": int(e[0]), "sr": int(e[1]),
                        "lvl": str(e[2]) if len(e) > 2 else "?",
                        "found": int(e[3]) if len(e) > 3 else -1,
                        "dmin": float(e[4]) if len(e) > 4 else None,
                        "timeout": int(e[5]) if len(e) > 5 else -1,
                        "ncol": int(e[6]) if len(e) > 6 else -1})
    return out


def failure_modes(a):
    """`--failure-modes`：把失败拆成「搜索失败」还是「接近失败」。

    ## 为什么需要这个量

    模型达到了自身示范的约 **67%**（保留率），剩下 33% 的差距一直只是
    「知道有差距」而**不知道差在哪里**。两种解释对应完全不同的修法：

    | 解释 | 表现 | 该修什么 |
    |---|---|---|
    | **搜索/感知失败** | 全程从未接近目标（`min_dsurf` 很大） | 感知召回、搜索策略、记忆 |
    | **接近/控制失败** | 接近了但没命中判据（`min_dsurf` 小但仍失败） | 控制精度、停止判据、误差累积 |

    判据用 `min_dsurf`（全程离目标**包围盒表面**的最小距离，即成功判据本身）：
    `≤ 0.30` 就是成功；越大说明越没靠近过。按四分档统计即可读出归属。
    """
    KEYS = [("demo_sweep_det_std", "sweep_det  (demo)"),
            ("ft_detf_std", "ft_detf   (96k, single-frame)"),
            ("ft_detf_sf31k_std", "ft_detf_sf31k (31k, single-frame)"),
            ("ft_detf_mem_std", "ft_detf_mem   (31k, +memory)")]

    def band(d):
        if d <= 0.30 + 1e-9:
            return "成功"
        if d <= 1.0:
            return "接近但未命中"
        if d <= 3.0:
            return "走近但不够近"
        return "从未接近"

    print("失败模式分解（按全程最近接近距离 `min_dsurf`；成功判据为 ≤ 0.30 m）\n")
    hdr = "%-28s %7s %9s %13s %13s %11s" % (
        "对象", "SR", "成功", "接近未命中", "走近不够近", "从未接近")
    print(hdr); print("-" * len(hdr))
    tabs = {}
    for nm, lab in KEYS:
        try:
            recs = _ep_records(nm)
        except Exception:
            continue
        # 只统计**带诊断量**的回合：追加批次是先前的旧格式（3 字段），
        # 混进来会被算成「无诊断量」而稀释比例。
        recs = [r for r in recs if r["dmin"] is not None]
        if not recs:
            print("%-28s   （无诊断量，需用新版重跑）" % lab)
            continue
        cnt = {}
        for r in recs:
            cnt[band(r["dmin"])] = cnt.get(band(r["dmin"]), 0) + 1
        n = len(recs)
        tabs[nm] = (recs, cnt)
        print("%-28s %6.1f%% %8d %12d %12d %10d" % (
            lab, 100 * sum(r["sr"] for r in recs) / n,
            cnt.get("成功", 0), cnt.get("接近但未命中", 0),
            cnt.get("走近但不够近", 0), cnt.get("从未接近", 0)))

    # 「看见了但走不到」 vs 「压根没看见」（仅 VLA 侧有 ever_found）
    print("\n感知槽是否曾命中目标（仅 VLA 侧记录；示范侧未额外跑检测故未记录）")
    print("%-28s %10s %12s %14s %12s" % ("对象", "曾检出", "曾检出且成功",
                                         "曾检出但失败", "从未检出"))
    for nm, lab in KEYS:
        if nm not in tabs:
            continue
        recs = tabs[nm][0]
        if all(r["found"] < 0 for r in recs):
            print("%-28s %10s" % (lab, "（未记录）"))
            continue
        ef = [r for r in recs if r["found"] == 1]
        ok = [r for r in ef if r["sr"] == 1]
        print("%-28s %9d%% %11d%% %13d%% %11d%%" % (
            lab, round(100 * len(ef) / len(recs)),
            round(100 * len(ok) / max(1, len(ef))),
            round(100 * (len(ef) - len(ok)) / max(1, len(ef))),
            round(100 * (len(recs) - len(ef)) / len(recs))))

    print("\n解读：把「从未接近」与「接近但未命中」的比例对比示范与模型，")
    print("      即可判断 33% 的保留率差距主要落在**搜索/感知**还是**控制/精度**上。")
    return tabs




# ---------------------------------------------------------------- 多重比较校正
def holm(pvals, alpha=0.05):
    """Holm–Bonferroni 逐步向下校正。

    大纲 §3.2 明确要求「跨方法 × 跨数据比例 × 跨场景统一 Holm 校正」。
    本项目的配对比较天然是**一批**（同一回合集上的多个模型两两对比），
    不做校正会系统性高估显著性 —— 尤其像 §4.26 里 p = 0.035 / 0.021
    这类刚过线的结果。

    Returns:
        `(校正后 p, 是否拒绝 H0)` 两个与输入等长的列表。
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [1.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        v = min(1.0, (m - rank) * pvals[i])
        running = max(running, v)          # 保证单调不减
        adj[i] = running
    return adj, [v < alpha for v in adj]


# 所有「同回合集上的配对比较」汇总一处，统一做 Holm
HOLM_COMPARISONS = [
    ("ft_detf", "ft_calib_f1", "std", "主结果：条件化+感知 vs 特权端到端"),
    ("ft_detf", "ft_detf_e2e", "std", "纯架构效应：+感知 vs 纯端到端（同数据）"),
    ("ft_detf_e2e", "ft_calib_f1", "std", "纯数据效应：条件化 vs 特权（同架构）"),
    ("ft_detf_mem", "ft_detf_sf31k", "std", "时序记忆效应（同数据规模 31k）"),
    ("ft_detf", "ft_detf_mem", "std", "记忆 vs 3 倍数据"),
    ("demo_sweep_det", "demo_sweep", "std", "无特权示范 vs 特权示范"),
    ("ft_detf", "ft_calib_f1", "vis", "主结果（起点可见协议）"),
    ("ft_detf", "ft_detf_e2e", "vis", "纯架构效应（起点可见协议）"),
]


def holm_report(a):
    """`--holm`：把全部配对比较列成一张表，附 Holm 校正后的 p 与判定。"""
    print("多重比较校正（Holm–Bonferroni，α = 0.05）\n")
    print("配对 McNemar 精确检验；bootstrap %d 次（大纲 §3.2）\n" % a.boot)
    got, raw, labels = [], [], []
    for x, y, tag, lab in HOLM_COMPARISONS:
        try:
            p, b01, b10, n = paired_p(f"{x}_{tag}", f"{y}_{tag}")
        except Exception:
            continue
        got.append((x, y, tag, lab, p, b01, b10, n))
        raw.append(p)
        labels.append(lab)
    if not raw:
        print("  （无可用数据）")
        return None
    adj, rej = holm(raw)
    print("%-44s %10s %10s %6s %s" % ("对比", "原始 p", "Holm p", "显著", "n"))
    print("-" * 84)
    for (x, y, tag, lab, p, b01, b10, n), ap_, rj in zip(got, adj, rej):
        print("%-44s %10.4g %10.4g %6s %5d   (b=%d, c=%d)" %
              (lab[:44], p, ap_, "✓" if rj else "✗", n, b01, b10))
    n_raw = sum(1 for p in raw if p < 0.05)
    n_adj = sum(rej)
    print("\n多重比较：%d 项对比；未校正时 %d 项显著，Holm 校正后 **%d 项**显著。"
          % (len(raw), n_raw, n_adj))
    if n_adj < n_raw:
        print("  → **有 %d 项在校正后失去显著性**，论文中不得只报未校正 p。"
              % (n_raw - n_adj))
    return {"raw": raw, "holm": adj, "reject": rej, "labels": labels}


if __name__ == "__main__":
    main()
