#!/usr/bin/env python3
"""calibrate_camera.py —— 标定相机水平焦距 f 与安装高度 cam_h（写入 robot.yaml）。

原理（单目地面测距的两把尺子都由 f 与 cam_h 决定）：
    d = f·cam_h / (r1 - horizon)，horizon = H/2
两个量任一不准，检测距离就系统偏差——仿真里正是靠数据集最小距离 0.74 m
与理论饱和值 0.752 m 的吻合才定位到相机参数的（主创新A §4.26）。

标定步骤（约 5 分钟）：
  ① f：打印一张已知宽度 W_obj 的方块贴在墙上，机器人原地不动，
     用尺子量「镜头到墙」的直线距离 D，然后运行本脚本 --focal。
     f = (像素宽 × D) / W_obj
  ② cam_h：卷尺量「镜头光轴离地高度」即可（毫米级误差影响不大）。
  ③ 顺手算出地面测距的饱和下限 d_floor = f·cam_h/(H - H/2)，
     写进 robot.yaml 的 mono_range_floor_m——它就是「单目测距不可信区」。

用法：
    python3 tools/calibrate_camera.py --focal --obj-width-cm 20 --dist-cm 150
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("需要 opencv：pip3 install opencv-python")


def measure_focal_px(args):
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    print("把目标方块对准画面中央，按 s 抓拍（可多次，取中位数），q 结束")
    widths = []
    while True:
        ok, img = cap.read()
        if not ok:
            sys.exit("抓帧失败")
        view = img.copy()
        h, w = img.shape[:2]
        cv2.putText(view, f"frames={len(widths)}  s=抓拍 q=结束", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("calib", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("s"):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 60, 160)
            cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best, bw = None, 0
            for c in cnts:
                x, y, cw, ch = cv2.boundingRect(c)
                if 0.6 < cw / max(1, ch) < 1.7 and cw > bw and cw > w * 0.05:
                    best, bw = (x, y, cw, ch), cw
            if best:
                widths.append(bw)
                print(f"  捕捉目标宽 {bw} px（n={len(widths)}）")
            else:
                print("  未找到方形目标——把方块正对相机、光照均匀些")
        elif k == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    if not widths:
        sys.exit("没有有效抓拍")
    bw = float(np.median(widths))
    f = bw * (args.dist_cm / 100.0) / (args.obj_width_cm / 100.0)
    print(f"\n  中位像素宽 {bw:.1f} px, 物宽 {args.obj_width_cm} cm, 距离 {args.dist_cm} cm")
    print(f"  → 水平焦距 f = {f:.1f} px")
    print(f"  → hfov = {2*np.degrees(np.arctan((args.width/2)/f)):.1f}°")
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--focal", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--obj-width-cm", type=float, required=True,
                    help="标定物实际宽度（cm）")
    ap.add_argument("--dist-cm", type=float, required=True,
                    help="镜头到标定物的直线距离（cm）")
    ap.add_argument("--cam-h", type=float, default=None, help="镜头离地高度（米）")
    ap.add_argument("--robot-yaml", default="config/robot.yaml")
    a = ap.parse_args()
    if not a.focal:
        print(__doc__)
        return
    f = measure_focal_px(a)
    hfov = 2 * np.degrees(np.arctan((a.width / 2) / f))
    d_floor = None
    if a.cam_h:
        d_floor = f * a.cam_h / (a.height - a.height / 2)
        print(f"  → cam_h={a.cam_h} m 时，单目地面测距饱和下限 = {d_floor:.3f} m")
        print("    （真值小于该距离时必须改用激光表面距——见告警 ③）")
    print(f"""
—— 请把以下值写入 {a.robot_yaml} ——
camera:
  hfov_deg: {hfov:.1f}
  cam_h: {a.cam_h if a.cam_h else '⚠ 卷尺实测后填写'}
  mono_range_floor_m: {f'{d_floor:.3f}' if d_floor else '⚠ 填 cam_h 后重跑可算出'}""")


if __name__ == "__main__":
    main()
