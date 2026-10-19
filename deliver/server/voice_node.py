#!/usr/bin/env python3
"""voice_node.py —— 车端语音指令节点（RDK）。
按回车录音(arecord 5s) → POST /asr → 发布 /vla/instruction → 可选 TTS 播报。
用法: python3 voice_node.py --server http://192.168.137.1:8010 [--no-ros] [--tts]
"""
import argparse
import base64
import json
import shutil
import subprocess
import time
import urllib.parse
import urllib.request

VOICES = "/vla/instruction"


def record_wav(path, seconds, device):
    cmd = ["arecord", "-D", device, "-f", "S16_LE", "-r", "16000",
           "-c", "1", "-d", str(int(seconds)), path]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("arecord 失败: " + r.stderr.decode(errors="replace")[:200])


def post_asr(server, wav_path):
    audio = base64.b64encode(open(wav_path, "rb").read()).decode()
    req = urllib.request.Request(server + "/asr",
                                 json.dumps({"wav_b64": audio}).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read())
    out["rtt_ms"] = round((time.time() - t0) * 1000, 1)
    return out


def play_tts(server, text):
    player = next((p for p in ("mpg123", "ffplay") if shutil.which(p)), None)
    if player is None:
        print("  [tts] 无 mpg123/ffplay，跳过播报")
        return
    try:
        with urllib.request.urlopen(server + "/tts?text="
                                    + urllib.parse.quote(text), timeout=30) as r:
            open("/tmp/vla_tts.mp3", "wb").write(r.read())
        if player == "mpg123":
            subprocess.run(["mpg123", "-q", "/tmp/vla_tts.mp3"])
        else:
            subprocess.run(["ffplay", "-nodisp", "-autoexit", "/tmp/vla_tts.mp3"])
    except Exception as e:                          # noqa: BLE001
        print("  [tts] 播报失败（不影响控制链路）: " + str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://192.168.137.1:8010")
    ap.add_argument("--no-ros", action="store_true")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--alsa-device", default="default")
    ap.add_argument("--tts", action="store_true")
    a = ap.parse_args()

    node = pub = None
    if not a.no_ros:
        try:
            import rclpy
            from std_msgs.msg import String
            rclpy.init()
            node = rclpy.create_node("voice_node")
            pub = node.create_publisher(String, VOICES, 10)
            print("[voice] ROS 就绪，指令发布到 " + VOICES)
        except Exception as e:                      # noqa: BLE001
            print("[voice] ROS 不可用(%s)，退化为纯测试模式" % e)

    print("[voice] 服务器 %s | 按回车说 %ds，Ctrl-C 退出" % (a.server, a.seconds))
    try:
        while True:
            input()
            wav = "/tmp/vla_voice.wav"
            try:
                record_wav(wav, a.seconds, a.alsa_device)
            except RuntimeError as e:
                print("  [rec] " + str(e))
                continue
            try:
                out = post_asr(a.server, wav)
            except Exception as e:                  # noqa: BLE001
                print("  [asr] 请求失败: " + str(e))
                continue
            print("  转写: %r -> 指令: %r (ok=%s, rtt=%sms)"
                  % (out["text"], out["instruction"], out["ok"], out["rtt_ms"]))
            if pub is not None and out["ok"]:
                msg = String()
                msg.data = out["instruction"]
                pub.publish(msg)
            if a.tts:
                play_tts(a.server, "收到，" + ("前往目标"
                                             if out["ok"] else "没听清，请再说一遍"))
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()


if __name__ == "__main__":
    main()
