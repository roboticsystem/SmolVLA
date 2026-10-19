#!/usr/bin/env python3
"""web_sim.py —— E1/E10 网页版仿真演示系统（复用实验同款环境 sim.env）。

浏览器打开 http://<服务器>:8020 ：俯视地图 + 轨迹 + 激光 + 车载相机 + 状态，
可选策略（oracle/sweep/random_target/random/SmolVLA），批量评测出 SR/SPL/NE。
仅 Python 标准库 + sim 包；SmolVLA 模式需 infer_server(:8000) 已启动。
"""
from __future__ import annotations

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from sim.env import calibrated_env, navigate_metrics
from sim.baselines import (RandomPolicy, OraclePolicy, RandomTargetPolicy,
                           SweepDetectorPolicy)

LOCK = threading.Lock()
HTML_FILE = "web_sim.html"


class Sim:
    def __init__(self):
        self.env = calibrated_env()
        self.mode = "idle"
        self.policy_name = "oracle"
        self.seed = 9000
        self.level = None
        self.step = 0
        self.done = False
        self.obs = None
        self.result = None
        self.pose = [0.0, 0.0, 0.0]
        self.traj = []
        self.goal = None
        self.instruction = ""
        self.level_str = "-"
        self.map_png = None
        self.map_meta = None
        self.img_b64 = None
        self.lidar_pts = []
        self.batch = {"running": False, "done": 0, "n": 0, "sr": None,
                      "spl": None, "ne": None, "rows": []}
        self.worker = None
        self._stop = False

    def refresh_map(self):
        import cv2
        rgb = self.env.scene.rgb
        h, w = rgb.shape[:2]
        scale = 460.0 / max(h, w)
        small = cv2.resize(rgb, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode(".png", cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        self.map_png = base64.b64encode(buf.tobytes()).decode()
        b = self.env.scene.bounds
        self.map_meta = {"b": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                         "w": int(small.shape[1]), "h": int(small.shape[0])}

    def snapshot(self):
        import cv2
        lidar = []
        if self.obs is not None:
            th = self.pose[2]
            scan = self.obs["lidar"]
            n = len(scan)
            for i in range(0, n, 4):
                a = th + i * 2 * np.pi / n
                r = float(scan[i])
                if r < 11.5:
                    lidar.append([float(self.pose[0] + r * np.cos(a)),
                                  float(self.pose[1] + r * np.sin(a))])
            if self.step % 2 == 0 or self.done:
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(
                    self.obs["image"], cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 70])
                self.img_b64 = base64.b64encode(buf.tobytes()).decode()
        traj = self.traj[::max(1, len(self.traj) // 300)]
        return {"mode": self.mode, "policy": self.policy_name, "seed": self.seed,
                "step": self.step, "instruction": self.instruction,
                "level": self.level_str, "pose": self.pose, "traj": traj,
                "goal": self.goal, "lidar": lidar, "image": self.img_b64,
                "map": self.map_png, "map_meta": self.map_meta,
                "done": self.done, "result": self.result, "batch": self.batch}


SIM = Sim()


def cv2_jpeg(rgb):
    import cv2
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 80])
    return ok, buf.tobytes()


class VLAPolicy:
    """经 infer_server(:8000) 调真模型；chunk 执行（每次推理走 10 步）。"""

    def __init__(self, server="http://127.0.0.1:8000"):
        self.server = server
        self.queue = []

    def reset(self, env=None):
        self.queue = []

    def act(self, obs, env):
        if self.queue:
            return self.queue.pop(0)
        import urllib.request
        ok, buf = cv2_jpeg(obs["image"])
        req = {"jpg_b64": base64.b64encode(buf).decode(),
               "scan": obs["lidar"].tolist(),
               "state": [float(x) for x in obs["state"][:5]],
               "instruction": obs["instruction"]}
        r = urllib.request.urlopen(urllib.request.Request(
            self.server + "/infer", json.dumps(req).encode(),
            {"Content-Type": "application/json"}), timeout=30)
        out = json.loads(r.read())
        if "error" in out:
            raise RuntimeError(out["error"])
        acts = out.get("actions") or [out["action"]]
        self.queue = [np.asarray(a, np.float64) for a in acts[1:]]
        return np.asarray(acts[0], np.float64)


def make_policy(name):
    return {"oracle": OraclePolicy, "sweep": SweepDetectorPolicy,
            "random_target": RandomTargetPolicy, "random": RandomPolicy,
            "vla": VLAPolicy}[name]()


def run_episode(sim, policy_name, seed, level, collect_only=False):
    obs = sim.env.reset(seed=seed, level=level)
    pol = make_policy(policy_name)
    pol.reset(env=sim.env)
    steps = 0
    if not collect_only:
        sim.refresh_map()
    info = None
    while True:
        a = pol.act(obs, sim.env)
        obs, _r, done, info = sim.env.step(a)
        steps += 1
        if collect_only:
            if done or steps >= sim.env.cfg["max_steps"]:
                break
        else:
            with LOCK:
                sim.obs = obs
                sim.pose = [float(sim.env.pose[0]), float(sim.env.pose[1]),
                            float(sim.env.pose[2])]
                sim.traj = [list(map(float, p)) for p in sim.env.traj]
                sim.goal = [float(x) for x in sim.env.task["goal"]]
                sim.instruction = sim.env.task["instruction"]
                sim.level_str = str(sim.env.task.get("level", "-"))
                sim.step = steps
                sim.done = bool(done)
            time.sleep(0.05)
            if done or steps >= sim.env.cfg["max_steps"]:
                break
    m = navigate_metrics(info)
    return {"success": float(m["SR"]), "spl": float(m["SPL"]), "ne": float(m["NE"]),
            "steps": steps, "min_dist": float(sim.env.min_dist),
            "instruction": sim.env.task["instruction"],
            "level": str(sim.env.task.get("level", "-"))}


def worker_single(sim, policy_name, seed, level):
    try:
        res = run_episode(sim, policy_name, seed, level)
        with LOCK:
            sim.result = res
            sim.done = True
            sim.mode = "done"
    except Exception as e:                              # noqa: BLE001
        with LOCK:
            sim.mode = "error"
            sim.result = {"error": "%s: %s" % (type(e).__name__, e)}


def worker_batch(sim, policy_name, n, level):
    rows = []
    try:
        for ep in range(n):
            if sim._stop:
                break
            res = run_episode(sim, policy_name, 9000 + ep * 37, level,
                              collect_only=True)
            rows.append(res)
            with LOCK:
                sim.batch["done"] = ep + 1
                sim.batch["rows"] = rows[-8:]
                sim.batch["sr"] = round(100 * sum(r["success"] for r in rows)
                                        / len(rows), 1)
                sim.batch["spl"] = round(sum(r["spl"] for r in rows) / len(rows), 3)
                sim.batch["ne"] = round(sum(r["ne"] for r in rows) / len(rows), 2)
    except Exception as e:                              # noqa: BLE001
        rows.append({"error": "%s: %s" % (type(e).__name__, e)})
    with LOCK:
        sim.batch["running"] = False
        if rows:
            sim.batch["sr"] = round(100 * sum(r.get("success", 0) for r in rows)
                                    / len(rows), 1)
            sim.batch["spl"] = round(sum(r.get("spl", 0) for r in rows) / len(rows), 3)
            sim.batch["ne"] = round(sum(r.get("ne", 0) for r in rows) / len(rows), 2)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            body = open(HTML_FILE, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            with LOCK:
                self._json(SIM.snapshot())
        else:
            self._json({"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n)) if n else {}
        if self.path == "/api/start":
            with LOCK:
                if SIM.worker and SIM.worker.is_alive():
                    SIM._stop = True
                    SIM.worker.join(timeout=10)
                    SIM._stop = False
                SIM.mode = "running"
                SIM.done = False
                SIM.result = None
                SIM.step = 0
                SIM.traj = []
                SIM.obs = None
                SIM.policy_name = req.get("policy", "oracle")
                SIM.seed = int(req.get("seed", 9000))
                SIM.level = req.get("level")
                SIM.batch["running"] = False
                SIM.worker = threading.Thread(
                    target=worker_single,
                    args=(SIM, SIM.policy_name, SIM.seed, SIM.level), daemon=True)
                SIM.worker.start()
            self._json({"ok": True})
        elif self.path == "/api/batch":
            with LOCK:
                if SIM.worker and SIM.worker.is_alive():
                    SIM._stop = True
                    SIM.worker.join(timeout=10)
                    SIM._stop = False
                SIM.mode = "batch"
                SIM.batch = {"running": True, "done": 0,
                             "n": int(req.get("n", 20)), "sr": None,
                             "spl": None, "ne": None, "rows": []}
                SIM.worker = threading.Thread(
                    target=worker_batch,
                    args=(SIM, req.get("policy", "oracle"),
                          int(req.get("n", 20)), req.get("level")), daemon=True)
                SIM.worker.start()
            self._json({"ok": True})
        elif self.path == "/api/stop":
            SIM._stop = True
            self._json({"ok": True})
        else:
            self._json({"error": "not found"})


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8020)
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print("[web_sim] http://%s:%d  （浏览器打开即用；SmolVLA 模式需 infer_server:8000）"
          % (a.host, a.port), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
