#!/usr/bin/env python3
"""merge_real_mix.py —— 把真机采集回合与仿真基线数据合并成共训数据集。

对齐处理：
- 真机图像 480x640 → resize 120x160（与仿真训练一致，宽高比同为 4:3）
- instr_vocab 取并集，真机帧的 instr_ids 指向新增词条
- ep_offset 平移、ep_success / ep_instr 拼接
- 归一化统计由 SimDemoData 在合并集上重算（仿真主导，真机小幅修正）

用法：
    python3 merge_real_mix.py \
        --sim export/nav_detf.h5 \
        --real deliver_v1/tools/data_recorded/session_red_02.h5 \
        --out export/nav_real_mix.h5
    （--real 可传多个，逗号分隔）
"""
from __future__ import annotations

import argparse

import cv2
import h5py
import numpy as np

TW, TH = 160, 120          # 训练口径 (W, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="export/nav_detf.h5")
    ap.add_argument("--real", required=True, help="逗号分隔的多个真机 h5")
    ap.add_argument("--out", default="export/nav_real_mix.h5")
    a = ap.parse_args()
    real_paths = [p.strip() for p in a.real.split(",") if p.strip()]

    sim = h5py.File(a.sim, "r")
    Ns = sim["images"].shape[0]
    vocab = [x.decode() if isinstance(x, bytes) else str(x)
             for x in sim["instr_vocab"][...]]
    print(f"[sim] {a.sim}: {Ns} 帧, vocab {len(vocab)}")

    reals = [h5py.File(p, "r") for p in real_paths]
    Nr = sum(r["images"].shape[0] for r in reals)
    print(f"[real] {Nr} 帧（{len(reals)} 个文件）")

    with h5py.File(a.out, "w") as o:
        N = Ns + Nr
        # ---- images（真机 resize 到训练口径）----
        o.create_dataset("images", shape=(N, TH, TW, 3), dtype=np.uint8,
                         chunks=(64, TH, TW, 3))
        o["images"][:Ns] = sim["images"][...]
        w = Ns
        for r in reals:
            m = r["images"].shape[0]
            for i in range(m):
                o["images"][w + i] = cv2.resize(r["images"][i], (TW, TH))
            w += m

        # ---- 一维可直拼的数据集 ----
        for k in ("actions", "states", "lidar", "det_feat"):
            if k not in sim or k not in reals[0]:
                print(f"[WARN] 跳过 {k}（一侧缺失）"); continue
            a_sim, a_real = sim[k][...], np.concatenate([r[k][...] for r in reals])
            if a_sim.shape[1:] != a_real.shape[1:]:
                print(f"[WARN] 跳过 {k}（形状不一致 {a_sim.shape[1:]} vs {a_real.shape[1:]}）")
                continue
            o.create_dataset(k, data=np.concatenate([a_sim, a_real], axis=0))
            print(f"[OK] {k}: {o[k].shape}")

        # ---- instr 词表与 ids（并集）----
        new_instr = sorted({(x if isinstance(x, str) else x.decode())
                            for r in reals for x in r["ep_instr"][...]})
        vid = {t: i for i, t in enumerate(vocab)}
        for t in new_instr:
            if t not in vid:
                vid[t] = len(vocab); vocab.append(t)
        o.create_dataset("instr_vocab",
                         data=np.asarray(vocab, dtype=h5py.special_dtype(vlen=str)))
        ids_sim = sim["instr_ids"][...]
        ids_real = []
        for r in reals:
            off = r["ep_offset"][...]
            instrs = [x if isinstance(x, str) else x.decode() for x in r["ep_instr"][...]]
            for a_, b_, t in zip(off[:-1], off[1:], instrs):
                ids_real += [vid[t]] * int(b_ - a_)
        o.create_dataset("instr_ids", data=np.asarray(ids_sim.tolist() + ids_real, np.int32))
        print(f"[OK] instr_vocab {len(vocab)} 词条, instr_ids {N} 帧")

        # ---- 回合边界 / 标签 / 指令 ----
        offs_sim = sim["ep_offset"][...]
        all_off = list(offs_sim)
        all_succ = list(sim["ep_success"][...]) if "ep_success" in sim else []
        all_instr = list(vocab[int(i)] for i in ids_sim)  # 帧级指令（按 vocab 反查）
        ep_instr_sim = []
        for a_, b_ in zip(offs_sim[:-1], offs_sim[1:]):
            ep_instr_sim.append(vocab[int(ids_sim[int(a_)])])
        w_off, w_instr = Ns, ep_instr_sim
        for r in reals:
            off = r["ep_offset"][...]
            e = len(off) - 1
            instrs = [x if isinstance(x, str) else x.decode() for x in r["ep_instr"][...]]
            instrs = (instrs + [instrs[-1]] * e)[:e] if len(instrs) < e else instrs[:e]
            all_off += [int(v) + w_off for v in off[1:]]
            all_succ += list(r["ep_success"][...]) if "ep_success" in r else [-1.0] * e
            w_instr += instrs[:e]
            w_off += int(off[-1])
        o.create_dataset("ep_offset", data=np.asarray(all_off, np.int64))
        o.create_dataset("ep_success", data=np.asarray(all_succ, np.float32))
        o.create_dataset("ep_instr",
                         data=np.asarray(w_instr, dtype=h5py.special_dtype(vlen=str)))
        print(f"[OK] ep_offset {len(all_off)-1} 回合, ep_success {len(all_succ)}, ep_instr {len(w_instr)}")

        o.attrs.update({"sim_source": a.sim, "real_sources": str(real_paths),
                        "real_frames": Nr, "total_frames": N})

    for r in reals:
        r.close()
    sim.close()
    print(f"\n[完成] {a.out} —— 共训数据集就绪")


if __name__ == "__main__":
    main()
