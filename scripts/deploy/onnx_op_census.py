"""ONNX 算子普查（E6 闸门 Step 3）。

不依赖开发板即可给出「算子覆盖率」的初步判断：
统计全图算子直方图，并标记 RDK-X3（地平线 Bernoulli 架构）上已知的高危算子。

用法：
    .venv_smolvla/bin/python onnx_op_census.py export/smolvla_prefix.onnx
    .venv_smolvla/bin/python onnx_op_census.py export/*.onnx --json export/op_census.json
"""

import argparse
import collections
import json
import os

import onnx

# RDK-X3（地平线 Bernoulli/BPU）实测与社区反馈的高危算子。
# 分级：3 = 几乎必然回退 CPU；2 = 大概率回退或需改造；1 = 支持但代价高
HAZARD = {
    "ScatterND": (3, "KV cache 的 index_put_ 语义，BPU 不支持"),
    "ScatterElements": (3, "同上"),
    "IsNaN": (3, "softmax 数值稳定分支，BPU 不支持"),
    "Where": (3, "条件选择，BPU 不支持"),
    "Einsum": (3, "不支持"),
    "TopK": (3, "不支持"),
    "ArgMax": (3, "不支持"),
    "LayerNormalization": (2, "常回退，需分解为均值/方差 + 乘加"),
    "Pow": (2, "LayerNorm 分解产物，指数非常量时受限"),
    "Sqrt": (2, "同上"),
    "Reciprocal": (2, "同上"),
    "Erf": (2, "精确 GELU 的产物，需 tanh 近似"),
    "Gelu": (2, "需换近似"),
    "Resize": (2, "动态 scale 会失败，需固定"),
    "Upsample": (2, "同 Resize"),
    "Gather": (2, "动态索引会回退"),
    "Transpose": (1, "支持但引入大量内存搬运"),
    "Softmax": (1, "支持但慢"),
    "Split": (1, "支持"),
    "Cast": (1, "支持，但可优化消除"),
    "MatMul": (1, "支持，但任意形状受限"),
    "Range": (1, "需关注"),
    "NonZero": (3, "不支持"),
    "Unique": (3, "不支持"),
}

LEVEL_NAME = {3: "高危(几乎必然回退)", 2: "需改造", 1: "代价高"}


def census(path: str) -> dict:
    model = onnx.load(path, load_external_data=False)
    g = model.graph

    ops = collections.Counter(n.op_type for n in g.node)
    ext = sum(1 for i in g.initializer if i.data_location == onnx.TensorProto.EXTERNAL)

    # 参数量（仅统计 inline 的；外部数据按文件名不重复计算）
    params = 0
    for init in g.initializer:
        n = 1
        for d in init.dims:
            n *= int(d)
        params += n

    hazards = {op: (ops[op], *HAZARD[op]) for op in ops if op in HAZARD}
    n_hazard_nodes = sum(c for c, _lv, _r in hazards.values())
    total_nodes = sum(ops.values())

    return {
        "file": path,
        "size_mib": round(os.path.getsize(path) / 1024**2, 1),
        "ir_version": model.ir_version,
        "opset": {o.domain or "ai.onnx": o.version for o in model.opset_import},
        "nodes": total_nodes,
        "initializers": len(g.initializer),
        "external_data_initializers": ext,
        "params_million": round(params / 1e6, 2),
        "inputs": [
            (i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in g.input
        ],
        "outputs": len(g.output),
        "op_histogram": dict(ops.most_common()),
        "hazards": {k: {"count": v[0], "level": v[1], "reason": v[2]} for k, v in hazards.items()},
        "predicted_bpu_coverage_pct": round(100 * (1 - n_hazard_nodes / max(total_nodes, 1)), 1),
        "hazard_node_count": n_hazard_nodes,
    }


def render(r: dict) -> None:
    print(f"\n{'=' * 68}")
    print(f"文件: {r['file']}   ({r['size_mib']} MiB)")
    print(f"opset: {r['opset']}   IR: {r['ir_version']}")
    print(f"节点数: {r['nodes']}   张量: {r['initializers']}（外部数据 {r['external_data_initializers']}）")
    print(f"参数量: {r['params_million']} M")
    print(f"输入: {r['inputs']}")
    print(f"输出数: {r['outputs']}")

    print(f"\n-- 算子直方图（top 18）--")
    for op, c in list(r["op_histogram"].items())[:18]:
        mark = " <<<" if op in HAZARD and HAZARD[op][0] >= 2 else ""
        print(f"   {op:24s} {c:5d}{mark}")

    if r["hazards"]:
        print(f"\n-- 高危 / 需改造算子 --")
        for op, h in sorted(r["hazards"].items(), key=lambda x: (-x[1]["level"], -x[1]["count"])):
            print(f"   [{LEVEL_NAME[h['level']]}] {op:22s} ×{h['count']:<4d} {h['reason']}")

    print(f"\n-- 预测 BPU 算子覆盖率 --")
    print(f"   高危节点 {r['hazard_node_count']} / 总节点 {r['nodes']}"
          f"  →  估算覆盖率 {r['predicted_bpu_coverage_pct']}%")
    print(f"   判据：覆盖率 ≥ 85% 才考虑直接部署（手册 §9.1）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    out = []
    for p in a.paths:
        r = census(p)
        render(r)
        out.append(r)

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n已写入 {a.json}")


if __name__ == "__main__":
    main()
