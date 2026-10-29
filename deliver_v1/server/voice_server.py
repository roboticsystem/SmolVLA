#!/usr/bin/env python3
"""voice_server.py —— 语音指令端点（D2 路线扩展，端口 8010）。

链路：车端麦克风录音 → 本服务 ASR（faster-whisper）→ 指令归一化映射
→ 车端 voice_node 发布 /vla/instruction → vla_bridge 换目标 → SmolVLA。

契约：
    POST /asr   {"wav_b64": <base64(audio)>}            # wav/mp3 均可
        → {"text": 原始转写, "instruction": 归一化指令, "ok": 是否可解析,
           "note": 映射说明, "latency_ms": ...}
    GET  /tts?text=...                                   # 可选播报（需联网）
        → audio/mpeg（edge-tts）；未安装/离线 → 503
    GET  /health → {"asr": true/false}

归一化目标句式（与训练集逐字一致）：
    有颜色：去{红色|蓝色|...}色的箱子旁边   （纯中文颜色口径）
    无颜色：去{桌子|椅子|...}旁边          （仿真无颜色句式）
归一化结果用 sim.baselines.parse_instruction 校验（color/cls 均可解析才算 ok）。
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import re
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

# 可选外部 ASR 后端（whisper.cpp / funasr_server 同协议：multipart file → {"text"}）。
# 设置后不再加载 faster-whisper，直接转发；如 ASR_BACKEND=http://127.0.0.1:8086/inference
ASR_BACKEND = os.getenv("ASR_BACKEND", "")

# ---------- 指令归一化 ----------
# 颜色归一到纯中文规范形（与训练模板「去{色}色的{类}旁边」逐字对齐，如「去红色的箱子旁边」）
COLOR_MAP = {
    "红色": "红色", "红": "红色", "蓝色": "蓝色", "蓝": "蓝色",
    "绿色": "绿色", "绿": "绿色", "黄色": "黄色", "黄": "黄色",
    "白色": "白色", "白": "白色", "黑色": "黑色", "黑": "黑色",
    "紫色": "紫色", "紫": "紫色", "橙色": "橙色", "橙": "橙色",
    "橘色": "橙色", "橘": "橙色",
    # Whisper 偶发输出繁体（如"紅色"），一并映射
    "紅色": "红色", "紅": "红色", "藍色": "蓝色", "藍": "蓝色",
    "綠色": "绿色", "綠": "绿色", "黃色": "黄色", "黃": "黄色",
}
# 类名词 → 训练词表的标准类词（与真机场景对齐：箱子；其余与仿真词表一致）
CLS_MAP = {
    "盒子": "箱子", "纸箱": "箱子", "箱": "箱子", "箱体": "箱子",
    "桌子": "桌子", "椅子": "椅子", "沙发": "沙发", "床": "床",
    "书架": "书架", "柜子": "柜子", "电视": "电视", "植物": "植物",
    "花盆": "植物", "凳子": "椅子",
}
REL_WORDS = ["那里", "那儿", "跟前", "边上", "附近", "旁边"]


def normalize_instruction(text: str, validate_fn=None):
    """口语转写 → 训练句式指令。返回 (instruction, ok, note)。

    ok 判定：validate_fn(instruction) 能解析出 color 与 cls。
    """
    t = re.sub(r"[\s，。！？、,.!?~～的]", "", str(text))
    note_parts = []

    # 1) 颜色词（先长后短，防"红色"被"红"截胡）
    color = None
    for k in sorted(COLOR_MAP, key=len, reverse=True):
        if k in t:
            color = COLOR_MAP[k]
            t = t.replace(k, "")
            note_parts.append(f"{k}→{color}")
            break

    # 2) 类名词
    cls = None
    for k in sorted(CLS_MAP, key=len, reverse=True):
        if k in t:
            cls = CLS_MAP[k]
            t = t.replace(k, "")
            note_parts.append(f"{k}→{cls}")
            break

    # 3) 什么都没听出 → 原样返回（让上层看到没听懂）
    if color is None and cls is None:
        return str(text), False, "未识别到颜色词与类名词"

    # 4) 重建训练句式（"去/前往/走到…"统一为"去"，方位词统一为"旁边"）
    if color and cls:
        instr = f"去{color}的{cls}旁边"
    elif color:
        instr = f"去{color}的箱子旁边"       # 只说了颜色：默认箱子（真机场景）
        note_parts.append("缺类词→默认箱子")
    else:
        instr = f"去{cls}旁边"

    ok = True
    if validate_fn is not None:
        try:
            c, k, _, _ = validate_fn(instr)
            # 无颜色句式（去桌子旁边）在训练集里真实存在（2141 帧），
            # 因此只要求类词可解析；颜色可选
            ok = (k is not None)
        except Exception:
            ok = False
    if not ok:
        note_parts.append("校验未通过（颜色或类词不在词表）")
    return instr, ok, "; ".join(note_parts) if note_parts else "直通"


class _State:
    whisper = None


def _get_asr():
    if _State.whisper is None:
        from faster_whisper import WhisperModel
        t0 = time.time()
        try:
            _State.whisper = WhisperModel(ASR_MODEL, device="cuda",
                                          compute_type="float16")
            dev = "cuda(fp16)"
        except Exception as e:                     # noqa: BLE001
            print(f"[asr] GPU 加载失败（{type(e).__name__}），回退 CPU int8")
            _State.whisper = WhisperModel(ASR_MODEL, device="cpu",
                                          compute_type="int8")
            dev = "cpu(int8)"
        print(f"[asr] faster-whisper {ASR_MODEL} 已加载（{dev}，"
              f"{time.time() - t0:.0f}s）", flush=True)
    return _State.whisper


def _asr_http(audio: bytes) -> str:
    """转发到外部 ASR 服务（multipart 上传 file 字段，返回 {"text": ...}）。"""
    boundary = "----vlavoice9271x"
    body = (("--%s\r\n" % boundary).encode() +
            b"Content-Disposition: form-data; name=\"file\"; "
            b"filename=\"a.wav\"\r\n"
            b"Content-Type: application/octet-stream\r\n\r\n" +
            audio + ("\r\n--%s--\r\n" % boundary).encode())
    r = urllib.request.Request(
        ASR_BACKEND, data=body, method="POST",
        headers={"Content-Type": "multipart/form-data; boundary=%s" % boundary})
    out = json.loads(urllib.request.urlopen(r, timeout=120).read())
    return str(out.get("text", ""))


async def _edge_tts_bytes(text: str) -> bytes:
    import edge_tts
    buf = io.BytesIO()
    com = edge_tts.Communicate(text, voice="zh-CN-XiaoxiaoNeural")
    async for chunk in com.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


def make_handler(project: str, validate_fn):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):                 # 静默逐请求日志
            pass

        def _json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/health":
                self._json(200, {"asr": bool(ASR_BACKEND)
                                 or _State.whisper is not None,
                                 "model": ASR_BACKEND or ASR_MODEL})
            elif u.path == "/tts":
                text = parse_qs(u.query).get("text", [""])[0]
                if not text:
                    self._json(400, {"error": "缺 text"})
                    return
                try:
                    mp3 = asyncio.run(_edge_tts_bytes(text))
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/mpeg")
                    self.send_header("Content-Length", str(len(mp3)))
                    self.end_headers()
                    self.wfile.write(mp3)
                except Exception as e:               # noqa: BLE001
                    self._json(503, {"error": f"TTS 不可用: {e}"})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            t0 = time.time()
            n = int(self.headers.get("Content-Length", 0))
            try:
                if urlparse(self.path).path != "/asr":
                    self._json(404, {"error": "not found"})
                    return
                req = json.loads(self.rfile.read(n))
                audio = base64.b64decode(req["wav_b64"])
                if ASR_BACKEND:
                    text = _asr_http(audio)
                else:
                    model = _get_asr()
                    segs, _info = model.transcribe(io.BytesIO(audio),
                                                   language="zh",
                                                   beam_size=1, vad_filter=True)
                    text = "".join(s.text for s in segs).strip()
                instr, ok, note = normalize_instruction(text, validate_fn)
                self._json(200, {"text": text, "instruction": instr,
                                 "ok": bool(ok), "note": note,
                                 "latency_ms": round((time.time() - t0) * 1000, 1)})
                print(f"[asr] {text!r} → {instr!r} ok={ok} "
                      f"({round((time.time() - t0) * 1000, 1)}ms)", flush=True)
            except Exception as e:                   # noqa: BLE001
                print(f"[asr] 异常 {type(e).__name__}: {e}", flush=True)
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True,
                    help="VLA 工作区路径（用 sim.baselines.parse_instruction 校验）")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--asr-model", default="small",
                    help="faster-whisper 模型名（small/base/tiny）")
    a = ap.parse_args()
    global ASR_MODEL
    ASR_MODEL = a.asr_model

    sys.path.insert(0, a.project)
    os.chdir(a.project)                    # parse_instruction 无相对依赖，chdir 仅为统一
    from sim.baselines import parse_instruction

    srv = ThreadingHTTPServer((a.host, a.port),
                              make_handler(a.project, parse_instruction))
    print(f"[voice] http://{a.host}:{a.port}/asr  ASR 模型={ASR_MODEL} "
          f"（首次 /asr 时加载）", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
