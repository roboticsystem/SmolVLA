"""E1 数据集生成器：把自建仿真 rollout 成 (image, instruction, action) 三元组。

用途（同时服务两条主线）：
    主创新 A（VLA 微调）  —— 目标域演示数据
    主创新 B（蒸馏）      —— 真实语义导航域上的观测，替换此前的 PushT 域

数据来源：用 `oracle`（完美目标知识 + 测地场规划）作为**示范策略**采集。
语义接地怎么学：
    oracle 的动作只取决于真实目标位置，而目标由指令唯一确定；同一段轨迹会配多种
    合法表述（L1/L2/L3 与不同参考物），因此模型必须**读指令**才能知道该去哪个物体
    —— 这正是 ObjectNav 的模仿学习设定。

并行采集：仿真纯 CPU，本机 256 核，用 `--workers N` 分片并行（每片独立写临时 h5，
最后合并）。实测单进程约 9 ms/步，并行后墙钟时间近似按 worker 数缩减。

输出（HDF5）：
    images       (N, H, W, 3) uint8      第一人称观测
    actions      (N, 2) float32          示范动作 (v, ω)
    states       (N, S) float32          本体状态（位姿 + 上一步动作）
    lidar        (N, B) float32          激光雷达
    instr_ids    (N,) int32              指令在 instr_vocab 中的索引
    instr_vocab  (M,) 变长 UTF-8 字符串表
    ep_offset    (E+1,) int64            每段在 N 维上的起止
    ep_level / ep_success / ep_spl / ep_ne   段级元信息

用法：
    cd /3240704007/sci/VLA
    # 单进程
    .venv_smolvla/bin/python -m sim.gen_dataset --episodes 150 --out export/nav_demos.h5
    # 分片并行（推荐）
    .venv_smolvla/bin/python -m sim.gen_dataset --episodes 2400 --workers 12 \
        --instr-variants 3 --out export/nav_demos_big.h5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np

from .baselines import make_policy
from .env import SemanticNavEnv, calibrated_env, navigate_metrics
from .perception import (TargetMemory, detect, features_from_dets,
                         target_features)

ARRAYS = ("images", "actions", "states", "lidar", "instr_ids", "ep_offset",
          "ep_success", "ep_spl", "ep_ne", "ep_level")


# ---------------------------------------------------------------- 采集
def collect(episodes, seed=0, obs_size=(160, 120), peds=2, max_steps=700, level=None,
            lidar_noise=0.01, camera_augment=0.25, policy_name="oracle",
            instr_variants=2, success_only=True, verbose=False, tag="", calib=False,
            min_rows=20, min_start_dist=0.5, policy_kw=None, det_features=False,
            det_mode="none", mem_ttl=25):
    """采集一个分片，返回内存中的数组字典（由 `write_h5` 落盘）。

    `calib=True` 时使用标定后的基准配置（开阔场地 + 表面距离成功判据）。
    """
    if calib:
        env = calibrated_env(seed=seed, obs_size=obs_size, task_level=level)
    else:
        env = SemanticNavEnv(seed=seed, obs_size=obs_size, n_peds=peds,
                             max_steps=max_steps, lidar_noise=lidar_noise,
                             camera_augment=camera_augment, task_level=level)
    policy = make_policy(policy_name, **(policy_kw or {}))

    img_buf, act_buf, st_buf, lid_buf, ins_buf = [], [], [], [], []
    det_buf = []      # 感知模块输出（见 --det-features 与 perception.target_features）
    off_buf = [0]
    ep_meta = []
    instr2id: dict[str, int] = {}

    def instr_id(s: str) -> int:
        if s not in instr2id:
            instr2id[s] = len(instr2id)
        return instr2id[s]

    t0 = time.time()
    n_dropped = 0      # 失败回合
    n_short = 0        # 退化回合（出生点太近 / 轨迹太短）
    rng = np.random.default_rng(seed + 12345)   # 语言表述抽样用，与场景种子解耦

    for ep in range(episodes):
        s = seed * 100000 + ep
        obs = env.reset(seed=s, level=level)
        if env.task is None:
            continue
        policy.reset(env)

        rows = []                      # [(obs, action)]
        done = False
        for _ in range(max_steps):
            a = policy.act(obs, env)
            rows.append((obs, np.asarray(a, dtype=np.float32)))
            obs, r, done, info = env.step(a)
            if done:
                break
        if not rows:
            continue
        m = navigate_metrics(info)
        # 模仿学习需要高质量示范：失败回合会稀释「抵达目标」的信号，
        # 默认丢弃（可用 --keep-failures 保留，用于对比实验）
        if success_only and not info["success"]:
            n_dropped += 1
            continue

        # **退化回合过滤**（实测在无特权示范策略上这类占 44%，噪声极大）：
        #   · 出生点本来就在目标附近 → 一两步就「成功」，学不到任何搜索行为
        #   · 轨迹过短 → 组不成有意义的动作 chunk
        # 两类都会把「几乎不动的动作」灌进训练集，严重偏斜动作分布。
        if getattr(env, "start_xy", None) is not None and env.task is not None:
            d_start = float(np.hypot(*(np.asarray(env.task["goal"]) - env.start_xy)))
            if d_start < min_start_dist:
                n_short += 1
                continue
        if len(rows) < min_rows:
            n_short += 1
            continue

        # 同一段视觉轨迹配 k 种不同合法表述 → 模型必须**读指令**才能区分同一场景下
        # 的不同指代，这是语言接地的核心监督信号。
        # （早先「回合中途换表述」的写法无效：回合一旦 done 就无法继续 step，
        #   实测 instr_variants 从未生效。）
        phr = [p[1] for p in (env.task.get("phrasings") or [])] or [env.task["instruction"]]
        k = int(max(1, min(instr_variants, len(phr))))
        if k >= len(phr):
            chosen = list(phr)
        else:
            chosen = [phr[int(j)] for j in rng.integers(0, len(phr), k)]
        # 检测与指令无关，**每帧只跑一遍**，再按各种表述做匹配（省 2/3 开销）
        det_cache = None
        if det_features:
            det_cache = [detect(ob["image"]) for ob, _ in rows]
        for ins in chosen:
            # **记忆必须逐表述独立重放**：记忆是「指令所指目标」的跨帧累积，
            # 而同一段轨迹要配多种表述，目标不同 → 记忆内容不同。
            mem = TargetMemory(ttl=mem_ttl) if (det_features and det_mode == "mem") else None
            for j, (ob, av) in enumerate(rows):
                img_buf.append(ob["image"])
                act_buf.append(av)
                st_buf.append(ob["state"])
                lid_buf.append(ob["lidar"])
                ins_buf.append(instr_id(ins))
                if det_features:
                    # `ob["state"][:3]` 就是 (x, y, θ)，来自里程计，不是特权信息
                    pose = np.asarray(ob["state"][:3], dtype=float)
                    det_buf.append(features_from_dets(det_cache[j], ins, pose=pose,
                                                      mode=det_mode, memory=mem, t=j))
            off_buf.append(len(img_buf))   # 每个表述副本各自成段，便于按段切 chunk
            # 段级元信息必须与 off_buf **同粒度**（每个表述副本一条）。
            # 若只在回合末尾记一次，len(ep_meta) != len(ep_offset)-1，
            # 合并时会出现 "Can't broadcast (2,) -> (4,)"。
            ep_meta.append({"level": info["level"], "success": float(m["SR"]),
                            "SPL": m["SPL"], "NE": m["NE"],
                            "instruction": ins, "n_rows": len(rows), "n_variants": k})
        if verbose and (ep + 1) % 25 == 0:
            print(f"{tag}  {ep + 1:4d}/{episodes} 回合 | 采纳 {len(ep_meta):4d} / "
                  f"丢弃 {n_dropped:4d} | 样本 {len(img_buf):6d} | {time.time() - t0:.0f}s")

    vocab = [k for k, _ in sorted(instr2id.items(), key=lambda x: x[1])]
    return {
        "images": np.stack(img_buf) if img_buf else np.zeros((0, obs_size[1], obs_size[0], 3), np.uint8),
        "actions": np.stack(act_buf) if act_buf else np.zeros((0, 2), np.float32),
        "states": np.stack(st_buf) if st_buf else np.zeros((0, 5), np.float32),
        "lidar": np.stack(lid_buf) if lid_buf else np.zeros((0, 0), np.float32),
        "instr_ids": np.asarray(ins_buf, dtype=np.int32),
        "instr_vocab": vocab,
        "ep_offset": np.asarray(off_buf, dtype=np.int64),
        "ep_level": np.array([m["level"] for m in ep_meta], dtype=object),
        "ep_success": np.array([m["success"] for m in ep_meta], dtype=np.float32),
        "ep_spl": np.array([m["SPL"] for m in ep_meta], dtype=np.float32),
        "ep_ne": np.array([m["NE"] for m in ep_meta], dtype=np.float32),
        "det_feat": np.stack(det_buf) if det_buf else np.zeros((0, 4), np.float32),
        "n_dropped": n_dropped, "n_short": n_short,
    }


def write_h5(path, d, attrs=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("images", data=d["images"], compression="gzip", compression_opts=4)
        for k in ("actions", "states", "lidar", "instr_ids", "ep_offset",
                  "ep_success", "ep_spl", "ep_ne"):
            f.create_dataset(k, data=d[k])
        if "det_feat" in d and getattr(d["det_feat"], "size", 0):
            f.create_dataset("det_feat", data=d["det_feat"])
        f.create_dataset("instr_vocab",
                         data=np.array(d["instr_vocab"], dtype=h5py.string_dtype("utf-8")))
        f.create_dataset("ep_level",
                         data=np.array(list(d["ep_level"]), dtype=h5py.string_dtype("utf-8")))
        f.attrs.update({"episodes": len(d["ep_level"]), "samples": len(d["images"]),
                        "n_dropped": int(d.get("n_dropped", 0))})
        if attrs:
            f.attrs.update(attrs)
    return path


# ---------------------------------------------------------------- 合并
def merge_h5(paths, out, attrs=None, progress=True):
    """把若干分片合并成一个数据集（统一指令词表并重映射 id）。"""
    # 第一遍：统一词表 + 统计总量
    vocab, index = [], {}
    tot_samples = tot_segments = 0
    for p in paths:
        with h5py.File(p, "r") as f:
            for s in f["instr_vocab"][()]:
                s = s.decode() if isinstance(s, bytes) else str(s)
                if s not in index:
                    index[s] = len(vocab)
                    vocab.append(s)
            tot_samples += int(f["images"].shape[0])
            tot_segments += int(f["ep_offset"].shape[0]) - 1

    if tot_samples == 0:
        raise RuntimeError("所有分片均为空，请检查采集参数（示范成功率过低？）")

    # 用第一个**非空**分片确定形状与 dtype —— 空分片的占位形状不可靠
    # （例如 lidar 占位为 (0, B)，会把 1 维当成 B=0，合并时报 broadcast 错）
    shape_rest = dtype = None
    for p in paths:
        with h5py.File(p, "r") as f:
            if f["images"].shape[0] > 0:
                keys = ["images", "actions", "states", "lidar"]
                if "det_feat" in f:
                    keys.append("det_feat")
                shape_rest = {k: f[k].shape[1:] for k in keys}
                dtype = {k: f[k].dtype for k in keys}
                break
    if shape_rest is None:
        raise RuntimeError("所有分片均无样本，未生成数据")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with h5py.File(out, "w") as fo:
        fo.create_dataset("images", shape=(tot_samples,) + shape_rest["images"],
                          dtype=dtype["images"], compression="gzip", compression_opts=4)
        for k in ("actions", "states", "lidar"):
            fo.create_dataset(k, shape=(tot_samples,) + shape_rest[k], dtype=dtype[k])
        if "det_feat" in shape_rest:
            fo.create_dataset("det_feat", shape=(tot_samples,) + shape_rest["det_feat"],
                              dtype=dtype["det_feat"])
        ids = fo.create_dataset("instr_ids", shape=(tot_samples,), dtype=np.int32)
        off = fo.create_dataset("ep_offset", shape=(tot_segments + 1,), dtype=np.int64)
        lvl = fo.create_dataset("ep_level", shape=(tot_segments,),
                                dtype=h5py.string_dtype("utf-8"))
        succ = fo.create_dataset("ep_success", shape=(tot_segments,), dtype=np.float32)
        spl = fo.create_dataset("ep_spl", shape=(tot_segments,), dtype=np.float32)
        ne = fo.create_dataset("ep_ne", shape=(tot_segments,), dtype=np.float32)

        s0 = e0 = 0
        for i, p in enumerate(paths):
            with h5py.File(p, "r") as fi:
                n = int(fi["images"].shape[0])
                if n == 0:
                    continue
                # instr_ids 重映射到全局词表
                local = [s.decode() if isinstance(s, bytes) else str(s)
                         for s in fi["instr_vocab"][()]]
                remap = np.array([index[s] for s in local], dtype=np.int32)
                ids[s0:s0 + n] = remap[fi["instr_ids"][()]]
                # **det_feat 必须一起搬**：早先漏了它，预创建的全零数据集从未被填充，
                # 于是「感知输入」实验实际喂进去的是零向量（静默失效，白跑一轮训练）。
                fill_keys = ["images", "actions", "states", "lidar"]
                if "det_feat" in shape_rest:
                    fill_keys.append("det_feat")
                for k in fill_keys:
                    fo[k][s0:s0 + n] = fi[k][()]
                e = int(fi["ep_offset"].shape[0]) - 1
                if e > 0:
                    o = fi["ep_offset"][()].astype(np.int64) + s0
                    off[e0:e0 + e + 1] = o
                    lvl[e0:e0 + e] = fi["ep_level"][()]
                    succ[e0:e0 + e] = fi["ep_success"][()]
                    spl[e0:e0 + e] = fi["ep_spl"][()]
                    ne[e0:e0 + e] = fi["ep_ne"][()]
                    e0 += e
                s0 += n
            if progress:
                print(f"  合并 {i + 1}/{len(paths)}：{s0}/{tot_samples} 样本")

        fo.create_dataset("instr_vocab",
                          data=np.array(vocab, dtype=h5py.string_dtype("utf-8")))
        fo.attrs.update({"episodes": tot_segments, "samples": tot_samples,
                         "n_shards": len(paths), "instr_vocab_size": len(vocab)})
        if attrs:
            f_attrs = {k: v for k, v in attrs.items() if v is not None}
            fo.attrs.update(f_attrs)
    return out


# ---------------------------------------------------------------- 并行
def _worker(payload):
    i, n_ep, seed, kw = payload
    kw = dict(kw)
    tmp = kw.pop("_tmp")                          # _tmp 只用于定位输出，不属于 collect 参数
    try:
        d = collect(episodes=n_ep, seed=seed, verbose=False, **kw)
    except Exception as ex:                       # 单个分片失败不应拖垮整体
        return {"i": i, "path": None, "error": f"{type(ex).__name__}: {ex}",
                "episodes": 0, "samples": 0}
    p = os.path.join(tmp, f"shard_{i:03d}.h5")
    write_h5(p, d)
    return {"i": i, "path": p, "error": None, "episodes": len(d["ep_level"]),
            "samples": len(d["images"]), "dropped": int(d.get("n_dropped", 0)),
            "short": int(d.get("n_short", 0))}


def gen(episodes=300, seed=0, out="export/nav_demos.h5", obs_size=(160, 120),
        peds=2, max_steps=700, level=None, lidar_noise=0.01, camera_augment=0.25,
        policy_name="oracle", instr_variants=2, verbose=True, success_only=True,
        workers=1, calib=False, min_rows=20, min_start_dist=0.5, policy_kw=None,
        det_features=False, det_mode="none", mem_ttl=25):
    """采集主入口。`workers > 1` 时按回合分片并行，最后合并。"""
    t0 = time.time()
    attrs = dict(policy=policy_name, seed=seed, obs_h=obs_size[1], obs_w=obs_size[0],
                 camera_augment=camera_augment, lidar_noise=lidar_noise,
                 max_steps=max_steps, instr_variants=instr_variants,
                 success_only=bool(success_only), level=level or "mixed",
                 calib=bool(calib), det_features=bool(det_features),
                 det_mode=str(det_mode), mem_ttl=int(mem_ttl))

    if workers <= 1:
        d = collect(episodes, seed, obs_size, peds, max_steps, level, lidar_noise,
                    camera_augment, policy_name, instr_variants, success_only, verbose,
                    calib=calib, min_rows=min_rows, min_start_dist=min_start_dist,
                    policy_kw=policy_kw, det_features=det_features,
                    det_mode=det_mode, mem_ttl=mem_ttl)
        write_h5(out, d, attrs)
        return _report(out, d, t0, success_only)

    # 分片：每个 worker 独立 seed，避免回合重叠
    tmp = tempfile.mkdtemp(prefix="navgen_")
    base = episodes // workers
    extra = episodes % workers
    payloads = []
    for i in range(workers):
        n_ep = base + (1 if i < extra else 0)
        if n_ep == 0:
            continue
        payloads.append((i, n_ep, seed + 7919 * (i + 1),
                         dict(obs_size=obs_size, peds=peds, max_steps=max_steps,
                              level=level, lidar_noise=lidar_noise,
                              camera_augment=camera_augment, policy_name=policy_name,
                              instr_variants=instr_variants, success_only=success_only,
                              calib=calib, min_rows=min_rows,
                              min_start_dist=min_start_dist, policy_kw=policy_kw,
                              det_features=det_features, det_mode=det_mode,
                              mem_ttl=mem_ttl, _tmp=tmp)))
    # **打印生效参数**：早先 payload 漏传 policy_kw/min_rows/det_features，
    # 分片静默退回默认值，导致「有记忆 vs 无记忆」两轮采集其实配置相同，
    # 白跑了两轮训练。这里显式回显，避免同类错误复发。
    print(f"并行采集：{len(payloads)} 个分片 × 约 {base} 回合（共 {episodes} 回合）")
    print(f"  生效参数：policy={policy_name} policy_kw={policy_kw} "
          f"min_rows={min_rows} min_start_dist={min_start_dist} "
          f"det_features={det_features} det_mode={det_mode} mem_ttl={mem_ttl}")
    print(f"预估耗时 ≈ {episodes * max_steps * 0.009 / workers / 60:.0f} 分钟"
          f"（按 9 ms/步、全部回合跑满估算，实际更快）")

    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        done = 0
        for res in ex.map(_worker, payloads):
            done += 1
            results.append(res)
            if res["error"]:
                print(f"  [{done}/{len(payloads)}] 分片 {res['i']} 失败: {res['error']}")
            else:
                print(f"  [{done}/{len(payloads)}] 分片 {res['i']}：采纳 {res['episodes']} / "
                      f"丢弃 {res['dropped']} / 样本 {res['samples']}")

    shards = [r["path"] for r in results if r["path"] and os.path.exists(r["path"])]
    if not shards:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError("所有分片都失败，未生成数据")

    print(f"\n合并 {len(shards)} 个分片 → {out}")
    merge_h5(shards, out, attrs)
    shutil.rmtree(tmp, ignore_errors=True)
    return _report(out, None, t0, success_only)


def _det_worker(c):
    """重算感知特征的分块工人（模块级，才能被 ProcessPoolExecutor pickle）。"""
    path, lo, hi, td, idx = c
    with h5py.File(path, "r") as f:
        imgs = f["images"][lo:hi]
        ids = f["instr_ids"][lo:hi]
        vocab = [s.decode() if isinstance(s, bytes) else str(s)
                 for s in f["instr_vocab"][()]]
    out = np.zeros((hi - lo, 4), dtype=np.float32)
    for i in range(hi - lo):
        out[i] = target_features(imgs[i], vocab[int(ids[i])])
    np.save(os.path.join(td, f"d{idx:05d}.npy"), out)
    return idx


def repair_det_features(path, workers=16):
    """**重算已有数据集的感知特征**（`det_feat`）。

    图像与指令都在 h5 里，所以**不需要重跑仿真** —— 直接对存好的帧重跑检测即可，
    比重新采集快一到两个数量级。用于修复 `merge_h5` 曾漏搬 `det_feat` 的历史数据。
    """
    from concurrent.futures import ProcessPoolExecutor

    with h5py.File(path, "r") as f:
        N = int(f["images"].shape[0])
    if N == 0:
        raise RuntimeError("数据集为空")
    tmp = tempfile.mkdtemp(prefix="detfix_")
    step = max(1, N // (workers * 4))
    chunks = [(path, lo, min(N, lo + step), tmp, i)
              for i, lo in enumerate(range(0, N, step))]
    print(f"重算感知特征：{N} 帧 / {len(chunks)} 分块 / {workers} 进程")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_det_worker, chunks))
    arr = np.concatenate([np.load(os.path.join(tmp, f"d{i:05d}.npy"))
                          for i in range(len(chunks))], axis=0)
    with h5py.File(path, "a") as f:
        if "det_feat" in f:
            del f["det_feat"]
        f.create_dataset("det_feat", data=arr)
    print(f"  检出率 {100 * arr[:, 0].mean():.1f}%（已写回）")
    return arr


def _report(out, d, t0, success_only):
    with h5py.File(out, "r") as f:
        n = int(f["images"].shape[0])
        ne = int(f.attrs["episodes"])
        nd = int(f.attrs.get("n_dropped", 0))
        vocab = [s.decode() if isinstance(s, bytes) else str(s) for s in f["instr_vocab"][()]]
    print(f"\n完成 → {out}")
    print(f"  采纳 {ne} 段 / 样本 {n}")
    if success_only:
        print("  （success_only：所有采纳段均来自成功回放，示范成功率恒为 100%）")
    print(f"  指令词表: {len(vocab)} 条，例: {vocab[:3]}")
    print(f"  体积: {os.path.getsize(out) / 1024**2:.1f} MiB，耗时 {time.time() - t0:.0f}s")
    return out


def load(path):
    """读取数据集，返回 numpy dict（instr 已解码）。"""
    with h5py.File(path, "r") as f:
        d = {k: f[k][()] for k in ("images", "actions", "states", "lidar",
                                   "instr_ids", "ep_offset", "ep_success", "ep_level")}
        d["instr_vocab"] = [s.decode() if isinstance(s, bytes) else str(s)
                            for s in f["instr_vocab"][()]]
        d["attrs"] = dict(f.attrs)
    d["instructions"] = [d["instr_vocab"][i] for i in d["instr_ids"]]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", default=None, help="L1/L2/L3；缺省为混合")
    ap.add_argument("--peds", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--obs-h", type=int, default=120)
    ap.add_argument("--obs-w", type=int, default=160)
    ap.add_argument("--lidar-noise", type=float, default=0.01)
    ap.add_argument("--camera-augment", type=float, default=0.25)
    ap.add_argument("--policy", default="oracle")
    ap.add_argument("--instr-variants", type=int, default=2,
                    help="同一目标采几种指令表述（语言接地增强）")
    ap.add_argument("--min-rows", type=int, default=20,
                    help="丢弃轨迹短于该步数的**退化回合**（出生点太近/一两步就成功）")
    ap.add_argument("--min-start-dist", type=float, default=0.5,
                    help="丢弃出生点距目标小于该距离的回合")
    ap.add_argument("--policy-kw", default=None,
                    help="策略构造参数 JSON，如 '{\"mem_ttl\":25}'")
    ap.add_argument("--repair-det", action="store_true",
                    help="重算已有数据集的 det_feat（图像已在库里，无需重跑仿真）")
    ap.add_argument("--det-features-mode", default=None,
                    choices=["none", "mem"],
                    help="感知特征档位：`none`（4 维，单帧）/ `mem`（8 维，含跨帧记忆）。"
                         "`mem` 用于补齐「示范依赖记忆、模型却只看单帧」这一不对称"
                         "（关记忆使示范 33.0% → 16.0%，见报告 §4.23）")
    ap.add_argument("--mem-ttl", type=int, default=25,
                    help="记忆存活帧数，须与示范策略一致（其默认 mem_ttl=25）")
    ap.add_argument("--det-features", action="store_true",
                    help="把**感知模块输出**（是否检出目标/方位/距离，4 维）存进数据集，"
                         "并拼接到状态向量 —— 感知+决策分离架构，见报告 §4.12.5")
    ap.add_argument("--keep-failures", action="store_true",
                    help="保留失败回合（默认只留成功示范）")
    ap.add_argument("--workers", type=int, default=1,
                    help="并行分片数（仿真纯 CPU，建议 ≤ 物理核数的一半）")
    ap.add_argument("--calib", action="store_true",
                    help="使用标定后的基准配置（开阔场地 + 到物体表面距离的成功判据）。"
                         "**新实验一律加此开关**，见 主创新A_诊断报告.md §4.5")
    ap.add_argument("--merge", default=None,
                    help="逗号分隔的 h5 路径：只做合并（统一词表并重映射 id），不采集。"
                         "配合 --out 使用，DAgger 聚合数据时用得上")
    ap.add_argument("--out", default="export/nav_demos.h5")
    a = ap.parse_args()

    if a.repair_det:
        repair_det_features(a.out, workers=max(1, a.workers))
        return

    if a.merge:
        paths = [p.strip() for p in a.merge.split(",") if p.strip()]
        print(f"合并 {len(paths)} 个数据集 → {a.out}")
        merge_h5(paths, a.out)
        _report(a.out, None, time.time(), success_only=False)
        return

    gen(episodes=a.episodes, seed=a.seed, out=a.out, level=a.level, peds=a.peds,
        max_steps=a.max_steps, obs_size=(a.obs_w, a.obs_h),
        lidar_noise=a.lidar_noise, camera_augment=a.camera_augment,
        policy_name=a.policy, instr_variants=a.instr_variants,
        success_only=not a.keep_failures, workers=a.workers, calib=a.calib,
        min_rows=a.min_rows, min_start_dist=a.min_start_dist,
        policy_kw=(json.loads(a.policy_kw) if a.policy_kw else None),
        det_features=a.det_features,
        det_mode=(a.det_features_mode or ("mem" if a.det_features else "none")),
        mem_ttl=a.mem_ttl)


if __name__ == "__main__":
    main()
