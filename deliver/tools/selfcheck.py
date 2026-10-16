#!/usr/bin/env python3
"""selfcheck.py —— 真机第一步：环境与传感器自检（只读，不改任何配置）。

用法（在真机上）：
    python3 tools/selfcheck.py            # 全部检查
    python3 tools/selfcheck.py --json out.json   # 同时落盘报告

检查项：
  ① 系统与算力：架构 / 内存 / 磁盘 / 温度 / BPU 设备（RDK-X3 特征）
  ② Python 与关键包：numpy / opencv / onnxruntime / h5py / rclpy（可选缺失）
  ③ 相机：/dev/video* 枚举 + 能否抓一帧
  ④ 激光/串口：/dev/ttyUSB* /dev/ttyACM* 枚举
  ⑤ ROS2：rclpy 可用性、ROS_DISTRO、DDS 环境变量
  ⑥ 网络：能否 ping 通推理服务器（robot.yaml 里的 server_url）

每一项只报告事实，不做修复。输出末尾给出「下一步建议」。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import urllib.parse

OK, BAD, WARN, SKIP = "✓", "✗", "!", "-"


def sh(cmd, timeout=5):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except Exception as e:                                    # noqa: BLE001
        return 1, f"{type(e).__name__}: {e}"


def section(title):
    print(f"\n{'='*62}\n{title}\n{'='*62}")


def check_system(rep):
    section("① 系统与算力")
    print(f"  {OK} 主机名      {platform.node()}")
    print(f"  {OK} 架构        {platform.machine()}  {platform.release()}")
    mem = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")[:2]
        if k in ("MemTotal", "MemAvailable"):
            mem[k] = int(v.strip().split()[0]) // 1024
    print(f"  {OK} 内存        共 {mem.get('MemTotal','?')} MB / 可用 {mem.get('MemAvailable','?')} MB")
    st = os.statvfs("/")
    free_gb = st.f_bavail * st.f_frsize / 2**30
    print(f"  {OK} 磁盘        / 可用 {free_gb:.1f} GB" +
          (f"  {WARN} 不足 5 GB，采数前先清理" if free_gb < 5 else ""))
    rc, out = sh("cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | head -1")
    if rc == 0 and out.isdigit():
        print(f"  {OK} 温度        {int(out)/1000:.1f} °C" +
              (f"  {WARN} 偏高" if int(out) > 75000 else ""))
    # BPU / RDK 特征
    rdk = (os.path.exists("/usr/hobot") or os.path.exists("/opt/tros")
           or "sunrise" in open("/proc/cpuinfo").read().lower())
    print(f"  {OK if rdk else WARN} RDK/Hobot   "
          + ("检测到地平线运行时" if rdk else "未检测到（若为笔电侧可忽略）"))
    rep["system"] = {"machine": platform.machine(), "mem_total_mb": mem.get("MemTotal"),
                     "disk_free_gb": round(free_gb, 1), "rdk": bool(rdk)}


def check_python(rep):
    section("② Python 与关键包")
    print(f"  {OK} Python      {sys.version.split()[0]}（建议 ≥3.8）")
    pkgs = {"numpy": "数组", "cv2": "相机/图像", "onnxruntime": "ONNX 推理",
            "h5py": "采数落盘", "yaml": "读 robot.yaml", "rclpy": "ROS2"}
    rep["python"] = {}
    for m, use in pkgs.items():
        try:
            mod = __import__(m)
            ver = getattr(mod, "__version__", "?")
            print(f"  {OK} {m:<13} {ver:<12} {use}")
            rep["python"][m] = str(ver)
        except ImportError:
            tag = WARN if m in ("rclpy", "onnxruntime") else BAD
            note = ("真机联调前装：sudo apt install python3-rclpy 或 pip install "
                    "onnxruntime") if tag == WARN else f"必需（用于{use}）"
            print(f"  {tag} {m:<13} 缺失        {note}")
            rep["python"][m] = None


def check_camera(rep):
    section("③ 相机")
    devs = sorted(glob.glob("/dev/video*"))
    if not devs:
        print(f"  {BAD} 未发现 /dev/video*（检查 USB / 驱动 / 用户组 video）")
        rep["camera"] = {"devices": []}
        return
    print(f"  {OK} 设备        {' '.join(devs)}")
    try:
        import cv2
    except ImportError:
        print(f"  {SKIP} 抓帧测试跳过（无 cv2）")
        rep["camera"] = {"devices": devs}
        return
    rep["camera"] = {"devices": devs, "frame": None}
    for d in devs[:3]:
        cap = cv2.VideoCapture(int(d[-1]) if d[-1].isdigit() else d)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ok, frame = cap.read()
        if ok:
            import numpy as np
            mean = float(frame.mean())
            print(f"  {OK} {d:<14} 抓帧成功 {frame.shape} 平均亮度 {mean:.0f}" +
                  (f"  {WARN} 过暗，检查曝光" if mean < 25 else ""))
            rep["camera"]["frame"] = {"dev": d, "shape": list(frame.shape),
                                      "brightness": round(mean, 1)}
        else:
            print(f"  {WARN} {d:<14} 打开但抓不到帧（可能被占用或非采集口）")
        cap.release()


def check_serial(rep):
    section("④ 激光 / 串口")
    ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    if ports:
        print(f"  {OK} 串口设备    {' '.join(ports)}")
        for p in ports:
            print(f"      {p}  权限 {'可读写' if os.access(p, os.R_OK | os.W_OK) else '✗ 需 sudo 或 dialout 组'}")
    else:
        print(f"  {WARN} 无串口设备。若激光走网络（如 ROS2 驱动发布 /scan）则正常")
    rep["serial"] = ports


def check_ros(rep):
    section("⑤ ROS2")
    distro = os.environ.get("ROS_DISTRO", "")
    if distro:
        print(f"  {OK} ROS_DISTRO  {distro}")
        try:
            import rclpy                                              # noqa: F401
            print(f"  {OK} rclpy       可导入")
            rc, out = sh("ros2 node list", timeout=8)
            print(f"  {OK} ros2 CLI    {'正常' if rc == 0 else '异常: ' + out[:60]}")
        except ImportError:
            print(f"  {WARN} rclpy 不可导入（shell 未 source 或缺 python3-rclpy）")
    else:
        print(f"  {WARN} 未 source ROS2（请在 /opt/tros/setup.bash 或 "
              f"/opt/ros/*/setup.bash 后重试）")
    rep["ros"] = {"distro": distro}


def check_server(rep, url):
    section("⑥ 推理服务器连通性")
    if not url:
        print(f"  {SKIP} 未配置 server_url")
        return
    p = urllib.parse.urlparse(url)
    try:
        s = socket.create_connection((p.hostname, p.port or 80), timeout=3)
        s.close()
        print(f"  {OK} TCP 可达    {p.hostname}:{p.port or 80}"
              f"  {WARN}（端口开着≠服务在跑，完整测试用 tools/onnx_smoke.py 之外的 infer_server）")
    except Exception as e:                                    # noqa: BLE001
        print(f"  {BAD} 不可达      {p.hostname}:{p.port or 80}  ({e.__class__.__name__})")
        print(f"      → 若在笔电侧：先启动 server/infer_server.py")
    rep["server_url"] = url


def advice(rep):
    section("下一步建议")
    c = rep.get("camera", {}).get("frame")
    seq = []
    if not c:
        seq.append("相机没就绪 → 先解决相机（USB/驱动/权限），这是所有后续步骤的前提")
    else:
        seq.append("相机 OK → 跑 tools/perception_smoke.py --measure 标定目标 HSV")
    if rep["python"].get("cv2") is None:
        seq.append("pip3 install opencv-python（感知冒烟需要）")
    if rep["python"].get("h5py") is None:
        seq.append("pip3 install h5py pyyaml（采数需要）")
    if rep.get("ros", {}).get("distro") == "":
        seq.append("source ROS2 环境后重跑本脚本，确认 rclpy 可用")
    for i, s in enumerate(seq, 1):
        print(f"  {i}. {s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="报告落盘路径")
    ap.add_argument("--server-url", default=os.environ.get("VLA_SERVER_URL", ""))
    a = ap.parse_args()
    rep = {}
    check_system(rep)
    check_python(rep)
    check_camera(rep)
    check_serial(rep)
    check_ros(rep)
    check_server(rep, a.server_url)
    advice(rep)
    if a.json:
        json.dump(rep, open(a.json, "w"), ensure_ascii=False, indent=1)
        print(f"\n报告已写入 {a.json}（回报时请附上它）")


if __name__ == "__main__":
    main()
