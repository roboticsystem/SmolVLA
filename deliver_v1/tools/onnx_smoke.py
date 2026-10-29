#!/usr/bin/env python3
"""onnx_smoke.py —— 在部署目标机上验证 ONNX 模型能否加载并跑通一次前向。

用法（在 RDK-X3 上，装 onnxruntime 后）：
    python3 tools/onnx_smoke.py --dir ../models/deploy_4x1

它做的事：
  ① 用 onnxruntime 打开 prefix.onnx / denoise.onnx，打印输入输出签名；
  ② 用随机张量各跑一次前向，报告耗时（CPU 上跑只是验形态，不是性能结论）；
  ③ 打印 meta.json 里的 BPU 风险点计数（hazard_nodes），提醒量化转换的难点。

注意：ONNX ≠ BPU 可用模型。上板需要地平线 OpenExplorer 工具链做量化
（本包 docs/部署可行性评估.md 的 E7-partial 小节有失败记录与替代路径）。
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    import sys
    sys.exit("需要 onnxruntime：pip3 install onnxruntime  （RDK 上可用 hrt 源或 CPU 版）")


def run_one(path, n_trials=3):
    print(f"\n—— {os.path.basename(path)} ({os.path.getsize(path)/2**20:.1f} MiB) ——")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    for i in sess.get_inputs():
        print(f"  输入  {i.name:<28} {i.shape} {i.type}")
    for o in sess.get_outputs():
        print(f"  输出  {o.name:<28} {o.shape} {o.type}")
    feeds = {}
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in i.shape]
        feeds[i.name] = np.random.randn(*shape).astype(
            np.float32 if "float" in i.type else np.int64)
    sess.run(None, feeds)                                    # 预热
    ts = []
    for _ in range(n_trials):
        t0 = time.time()
        sess.run(None, feeds)
        ts.append((time.time() - t0) * 1000)
    print(f"  随机前向 {n_trials} 次: 中位 {np.median(ts):.0f} ms"
          f"  （CPU 形态验证，非性能结论）")
    return sess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../models/deploy_4x1")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.dir, "meta.json")))
    print("meta.json 摘要:")
    print(f"  蒸馏配置 {meta['config']}  可训参数 {meta['trainable_M']} M")
    print(f"  时延 teacher {meta['latency_ms']['teacher']} ms → deploy "
          f"{meta['latency_ms']['deploy']} ms (−{meta['latency_ms']['drop_pct']}%)")
    for k, v in meta["onnx"].items():
        print(f"  {k}: {v['size_mib']} MiB, opset {v['opset']['ai.onnx']}, "
              f"BPU 风险节点 {v['hazard_nodes']}（覆盖率 {v['coverage_pct']}%）")
    print(f"  提示: {meta.get('note', '')}")
    run_one(os.path.join(a.dir, "prefix.onnx"))
    run_one(os.path.join(a.dir, "denoise.onnx"))
    print("\n结论判定：两个会话都能建立并跑通 → ONNX 形态在目标机可用；")
    print("后续问题是「量化到 BPU」与「算子回退治理」（E7 大纲，见 docs）。")


if __name__ == "__main__":
    main()
