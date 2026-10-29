#!/usr/bin/env python3
"""perception_smoke.py —— 相机 + 颜色块检测冒烟，兼做 HSV 参考色标定。

用法：
    # ① 看检测效果（对准场景里的目标物）
    python3 tools/perception_smoke.py --camera 0

    # ② 标定模式：把目标物放画面中央按 m 采样 HSV，q 退出后打印建议 refs
    python3 tools/perception_smoke.py --camera 0 --measure

    # ③ 离线图片
    python3 tools/perception_smoke.py --image path.jpg

检测算法与训练管线**同一套几何**（从 sim/perception.py 移植并简化）：
  HSV 阈值 → 连通域 → 底边行号 r1 → 地面测距 d = f·cam_h/(r1 - horizon)

⚠ 继承仿真 §4.26 的教训：当色块底边触到图像下沿（r1 ≥ H-2）时，
  地面公式**饱和**（仿真里是 0.752 m），此时必须改用激光读数，
  否则测距会卡死在饱和值——打印里会用 [SAT] 标出。
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("需要 opencv：pip3 install opencv-python")

# 仿真调色板的占位值（config/robot.yaml 有一份；--measure 后以实测为准）
DEFAULT_REFS = {
    "red": (0, 200, 120), "green": (60, 200, 120), "blue": (110, 200, 120),
    "yellow": (25, 200, 120), "purple": (140, 200, 120), "orange": (12, 220, 160),
}
DEFAULT_TOL = (12, 90, 90)
CAMERA_HFOV_DEG = 64.5          # ⚠ 占位：用 calibrate_camera.py 实测后修改
CAMERA_HEIGHT = 0.35            # ⚠ 占位：镜头离地高度（米）


def detect_blobs(img_bgr, refs, tol, hfov_deg, cam_h):
    """返回 [{color, n_px, cx, cy, r1, d_mono, sat}, ...]（按像素数降序）。"""
    h, w = img_bgr.shape[:2]
    f = (w / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    horizon = h / 2.0
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    out = []
    for name, ref in refs.items():
        lo = np.array([max(0, ref[0] - tol[0]), max(0, ref[1] - tol[1]),
                       max(0, ref[2] - tol[2])], np.uint8)
        hi = np.array([min(179, ref[0] + tol[0]), min(255, ref[1] + tol[1]),
                       min(255, ref[2] + tol[2])], np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            npx = int(stats[i, cv2.CC_STAT_AREA])
            if npx < 150:                                     # 噪声过滤
                continue
            x, y, bw, bh = stats[i, :4]
            r1 = int(y + bh)                                   # 色块底边行号
            sat = r1 >= h - 2
            d = f * cam_h / max(1.0, (r1 - horizon)) if r1 > horizon else float("nan")
            cx, cy = cent[i]
            ang = np.degrees(np.arctan2(cx - (w - 1) / 2.0, f))
            out.append({"color": name, "n_px": npx, "bearing_deg": round(ang, 1),
                        "r1": r1, "d_mono_m": round(float(d), 3), "sat": sat})
    return sorted(out, key=lambda z: -z["n_px"])


def annotate(img, dets):
    for dt in dets:
        tag = f"{dt['color']} {dt['bearing_deg']}deg {dt['d_mono_m']}m" + \
              (" [SAT]" if dt["sat"] else "")
        cv2.putText(img, tag, (10, 18 + 22 * dets.index(dt)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return img


def run_camera(args, refs, tol):
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    samples = []
    print("按 q 退出；--measure 模式下按 m 采样画面中心 HSV")
    while True:
        ok, img = cap.read()
        if not ok:
            print("抓帧失败")
            break
        dets = detect_blobs(img, refs, tol, args.hfov, args.cam_h)
        view = annotate(img.copy(), dets)
        if args.measure:
            h, w = img.shape[:2]
            patch = img[h // 4: 3 * h // 4, w // 3: 2 * w // 3]
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
            med = np.median(hsv, 0).astype(int)
            cv2.rectangle(view, (w // 3, h // 4), (2 * w // 3, 3 * h // 4), (0, 255, 255), 2)
            cv2.putText(view, f"center HSV={tuple(med)}", (10, view.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            samples.append(med)
        cv2.imshow("perception_smoke", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    if args.measure and samples:
        S = np.stack(samples)
        med = np.median(S, 0).astype(int)
        print("\n—— 建议写入 config/robot.yaml 的 refs ——")
        print(f"  target_object: [{med[0]}, {med[1]}, {med[2]}]   # n={len(S)} 次采样中位数")


def run_image(args, refs, tol):
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"读不到 {args.image}")
    dets = detect_blobs(img, refs, tol, args.hfov, args.cam_h)
    print(f"{'颜色':<8}{'像素':>7}{'方位°':>8}{'底边行':>8}{'单目测距':>10}  标记")
    for dt in dets[:10]:
        print(f"{dt['color']:<8}{dt['n_px']:>7}{dt['bearing_deg']:>8}"
              f"{dt['r1']:>8}{dt['d_mono_m']:>10.3f}  {'[SAT]' if dt['sat'] else ''}")
    if not dets:
        print("（无检测——先跑 --measure 重新标定 HSV）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--image", default=None)
    ap.add_argument("--measure", action="store_true", help="标定 HSV 参考色")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--hfov", type=float, default=CAMERA_HFOV_DEG, help="水平 FOV（度）")
    ap.add_argument("--cam-h", dest="cam_h", type=float, default=CAMERA_HEIGHT)
    ap.add_argument("--refs-yaml", default=None, help="从 robot.yaml 读 refs")
    a = ap.parse_args()

    refs, tol = dict(DEFAULT_REFS), DEFAULT_TOL
    if a.refs_yaml:
        import yaml
        cfg = yaml.safe_load(open(a.refs_yaml))
        d = cfg.get("detection", {})
        refs = {k: tuple(v) for k, v in d.get("refs", {}).items()} or refs
        tol = tuple(d.get("hsv_tol", DEFAULT_TOL))
        if a.hfov == CAMERA_HFOV_DEG:
            a.hfov = float(cfg["camera"].get("hfov_deg", a.hfov))
        if a.cam_h == CAMERA_HEIGHT:
            a.cam_h = float(cfg["camera"].get("cam_h", a.cam_h))
    print(f"refs={refs}\ntol={tol}  hfov={a.hfov}  cam_h={a.cam_h}")
    if a.image:
        run_image(a, refs, tol)
    else:
        run_camera(a, refs, tol)


if __name__ == "__main__":
    main()
