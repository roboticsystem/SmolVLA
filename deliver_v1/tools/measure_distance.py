#!/usr/bin/env python3
"""measure_distance.py —— E10 论文 6.x.3 单目测距精度采集（RDK 上运行）。

流程：把目标物（如红盒）依次放到若干已知距离（卷尺/米尺量），每个距离
采集 N 帧并跑与部署完全同款的 HSV 检测（robot.yaml refs + 单目几何），
输出「真值 vs 估计」误差表（CSV + 终端表格），直接作为论文表格素材。

用法（在 RDK 的 deliver_v1/tools 下）：
    python3 measure_distance.py --color red \
        --distances "0.5,1.0,1.5,2.0,2.5,3.0" --samples 8

依赖：opencv-python、pyyaml（与 perception_smoke 相同环境）。
检测算法与 backfill_real.detect_blobs_bboxes 逐行一致（保持口径）。
"""
import argparse
import csv
import time

import cv2
import numpy as np


def load_cfg(path):
    import yaml
    cfg = yaml.safe_load(open(path)) or {}
    det = cfg.get("detection", {})
    refs = {k: tuple(v) for k, v in det.get("refs", {}).items()}
    tol = tuple(det.get("hsv_tol", [12, 90, 90]))
    overrides = {k: tuple(v) for k, v in det.get("hsv_tol_overrides", {}).items()}
    cam = cfg.get("camera", {})
    hfov = float(cam.get("hfov_deg", 69.4))
    cam_h = float(cam.get("cam_h", 0.08))
    # 与 backfill_real 相同的占位色过滤：未实测的参考色不参与检测
    measured = [k for k, v in refs.items() if "实测" in str(v)]
    use_refs = {k: v for k, v in refs.items() if k in measured or "占位" not in str(refs.get(k))}
    return refs, tol, overrides, hfov, cam_h, use_refs


def detect_color_dmono(img_bgr, color, refs, tol, overrides, hfov_deg, cam_h,
                       min_px=150, max_range=12.0):
    """返回该颜色的单目测距（米）；未检出返回 None。与回填口径逐行一致。"""
    h, w = img_bgr.shape[:2]
    f = (w / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    horizon = h / 2.0
    if color not in refs:
        return None
    ref = refs[color]
    t = overrides.get(color, tol)
    lo = np.array([max(0, ref[0] - t[0]), max(0, ref[1] - t[1]), max(0, ref[2] - t[2])], np.uint8)
    hi = np.array([min(179, ref[0] + t[0]), min(255, ref[1] + t[1]), min(255, ref[2] + t[2])], np.uint8)
    mask = cv2.inRange(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask)
    best = None
    for i in range(1, n):
        npx = int(stats[i, cv2.CC_STAT_AREA])
        if npx < min_px:
            continue
        y, bh = stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_HEIGHT]
        r1 = int(y + bh)
        if r1 <= horizon:
            continue
        d = f * cam_h / (r1 - horizon)
        if not np.isfinite(d):
            continue
        d = float(np.clip(d, 0.15, max_range))
        if best is None or npx > best[1]:
            best = (d, npx, (cent[i][0], cent[i][1]))
    return best[0] if best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--config", default="../config/robot.yaml")
    ap.add_argument("--color", default="red", help="目标颜色键（须是 robot.yaml 实测色）")
    ap.add_argument("--distances", default="0.5,1.0,1.5,2.0,2.5,3.0",
                    help="逗号分隔的真值距离（米，卷尺测量）")
    ap.add_argument("--samples", type=int, default=8, help="每个距离采集帧数")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--out", default="dist_measures.csv")
    a = ap.parse_args()

    refs, tol, ovr, hfov, cam_h, use_refs = load_cfg(a.config)
    print(f"[config] 检测色={list(use_refs.keys())} hfov={hfov} cam_h={cam_h} "
          f"目标色={a.color}（参考 {refs.get(a.color)}）")
    if a.color not in use_refs:
        raise SystemExit(f"颜色 {a.color!r} 不在 robot.yaml 实测参考色里，先标定")

    cap = cv2.VideoCapture(a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened():
        raise SystemExit("相机打开失败")
    time.sleep(1.0)                      # 等自动曝光稳定

    rows = []
    for d_true in [float(x) for x in a.distances.split(",")]:
        input(f"\n>>> 把目标放到卷尺量好的 {d_true:.2f} m 处（正对相机），按回车采集… ")
        ests = []
        for k in range(a.samples + 2):   # 前 2 帧丢弃（曝光/对焦过渡）
            ok, frame = cap.read()
            if not ok:
                continue
            if k < 2:
                continue
            d = detect_color_dmono(frame, a.color, refs, tol, ovr, hfov, cam_h)
            ests.append(d)
            cv2.imwrite(f"measure_{a.color}_{d_true:.2f}m_{k}.png", frame)
        valid = [e for e in ests if e is not None]
        n_ok, n_all = len(valid), len(ests)
        if valid:
            m, s = float(np.mean(valid)), float(np.std(valid))
            err, pct = m - d_true, (m - d_true) / d_true * 100
            print(f"  检出 {n_ok}/{n_all} | 估计 {m:.3f}±{s:.3f} m | "
                  f"误差 {err:+.3f} m ({pct:+.1f}%)")
        else:
            m = s = err = pct = float("nan")
            print(f"  检出 0/{n_all} —— 该距离下不可检出（有效检测边界）")
        rows.append([d_true, n_ok, n_all, m, s, err, pct])

    cap.release()
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["true_m", "n_detect", "n_frames", "est_mean_m",
                    "est_std_m", "err_m", "err_pct"])
        w.writerows(rows)
    print(f"\n[OK] 已保存 {a.out}（{len(rows)} 行），过程帧已存当前目录，"
          f"论文表格直接取 est_mean_m / err_m / err_pct 三列")


if __name__ == "__main__":
    main()
