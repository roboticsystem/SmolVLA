#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 E4b 运行日志（/tmp/e4b.log）重建 `export/e4b_sensitivity.json` 的已完成格子。

背景：本轮 E4b 扫描在 46/48 格时进程被终止（日志完整，但脚本末尾的 JSON 落盘未执行）。
日志里每行包含 TCR / 可见率 / 恢复时间 / 碰撞率 四项（脚本另有的 `err_mean` 未打印，
故此处置 None 并在文件中标注来源），用于给 `eval_e4b_sensitivity.py --resume` 做种子，
只补跑缺失的格子，不重复已完成的计算。
"""
from __future__ import annotations

import json
import os
import re
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/e4b.log"
OUT = sys.argv[2] if len(sys.argv) > 2 else "export/e4b_sensitivity.json"

PAT = re.compile(
    r"^(?P<axis>\S+)\s+(?P<pname>\S+?)=(?P<val>\S+)\s+(?P<sc>\S+)\s+"
    r"TCR\s+(?P<tcr>[\d.]+)%\s+可见\s+(?P<vis>[\d.]+)%\s+"
    r"恢复\s+(?P<rec>[\d.]+)\s+ms\s+碰撞\s+(?P<col>[\d.]+)"
)

rows = {}
with open(LOG, encoding="utf-8") as fh:
    for line in fh:
        m = PAT.match(line.strip())
        if not m:
            continue
        g = m.groupdict()
        key = f"{g['axis']}|{g['pname']}={g['val']}|{g['sc']}"
        rows[key] = {"TCR": float(g["tcr"]) / 100.0,
                     "err_mean": None,
                     "vis_rate": float(g["vis"]) / 100.0,
                     "rec_ms_median": float(g["rec"]),
                     "n_collide": float(g["col"])}

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump({"episodes": 20, "rows": rows,
               "_source": "reconstructed_from_run_log:/tmp/e4b.log "
                          "(err_mean 未打印，置 None；其余为原始记录)"},
              fh, ensure_ascii=False, indent=2)
print(f"已重建 {len(rows)} 格 → {OUT}")
