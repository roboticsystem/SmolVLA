#!/usr/bin/env python3
"""web_sim.py —— E1/E10 网页版仿真演示系统（复用实验同款环境 sim.env）。

浏览器打开 http://<服务器>:8020 ：俯视地图 + 轨迹 + 激光 + 车载相机 + 状态，
可选策略（oracle/sweep/random_target/random/SmolVLA），批量评测出 SR/SPL/NE。
仅 Python 标准库 + sim 包；SmolVLA 模式需 infer_server(:8000) 已启动。
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from sim.env import calibrated_env, navigate_metrics, grid_value
from sim.baselines import (RandomPolicy, OraclePolicy, RandomTargetPolicy,
                           SweepDetectorPolicy)
from sim.follow import (FollowEnv, OracleFollow, ReactiveFollow, FollowPolicyE4)
from sim.world import segment_free

LOCK = threading.Lock()
HTML_FILE = "web_sim.html"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC = {"three.min.js": "application/javascript",
          "OrbitControls.js": "application/javascript"}


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
        self.target = None        # 跟随模式：被跟随者（红马甲）位置
        self.live = None          # 跟随模式：实时 距离/可见/带内
        self.fenv = None          # 跟随环境（懒创建，按 scenario 复用）
        self.aenv = None          # 公寓场景导航环境（懒创建）
        self.scene_name = "apartment"
        self.batch = {"running": False, "done": 0, "n": 0, "sr": None,
                      "spl": None, "ne": None, "rows": []}
        self.worker = None
        self._stop = False

    def refresh_map(self, env=None):
        import cv2
        env = env or self.env
        sc = env.scene
        rgb = sc.rgb
        h, w = rgb.shape[:2]
        scale = 640.0 / max(h, w)
        small = cv2.resize(rgb, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_NEAREST)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        self.map_png = base64.b64encode(buf.tobytes()).decode()
        b = sc.bounds
        # 3D 视图用：墙体栅格（2×2 最大池化到 0.1 m 格）+ 家具真实高度
        sem = sc.sem
        h2, w2 = sem.shape[0] // 2, sem.shape[1] // 2
        p = sem[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2)
        walls = (p == 1).any(axis=(1, 3))
        self.map_meta = {"b": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                         "w": int(small.shape[1]), "h": int(small.shape[0]),
                         "name": sc.name,
                         "walls": "".join("1" if v else "0"
                                          for row in walls for v in row),
                         "wh": [int(w2), int(h2)], "cell": 0.1,
                         "rooms": [{"name": n, "rect": [float(v) for v in rect]}
                                   for n, rect in getattr(sc, "room_names", [])],
                         "objs": [{"cls": o.cls, "color": o.color,
                                   "xy": [float(o.xy[0]), float(o.xy[1])],
                                   "half": float(o.half),
                                   "h": float(o.height)} for o in sc.objects]}

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
                "target": self.target, "live": self.live,
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


class FieldFollower:
    """**公寓场景专用**的测地场跟随器（demo 用，不影响论文基线）。

    为什么不用 `OraclePolicy`：它是在 8×6 m **单房间**标定场景（无墙无门）上
    调出来的 —— 避障减速 + 前瞻窗口的组合在门洞两侧墙角会陷入
    「贴墙 → 前方光束 <0.2 m → v=0 → 脱困自旋 → 又贴墙」的死循环
    （公寓 15 个跨房间目标实测 12 个失败）。而论文的 oracle 数据必须
    保持原实现，所以这里另写一个纯场梯度跟随器：

    - 每步在机器人周边 ±0.4 m 窗口找**测地场最小**的自由格作为临时瞄准点；
    - 角度差直接比例转向、无避障接管（场路径本身在膨胀栅格上，天然无碰撞）；
    - 卡住时窗口里总有更小的场值（场只会经门洞传播），因而无局部极小。
    """

    name = "field"

    def __init__(self, k_ang=3.0, win=14):
        self.k_ang = float(k_ang)
        self.win = int(win)
        self._f = None
        self._g = None

    def reset(self, env=None):
        self._f = None
        self._g = None

    def act(self, obs, env):
        goal = np.asarray(env.task["goal"], dtype=float)
        if self._f is None or self._g is None or not np.allclose(goal, self._g):
            self._f = env._shortest_path_len(goal)
            self._g = goal.copy()
        f = self._f
        H, W = f.shape
        r, c = env.scene.w2c(env.pose[:2])
        r, c = int(r), int(c)
        # 候选格按「场值 + 距离惩罚」排序，取第一个**直线可达**的
        # （纯取窗口最小会切家具角：目标格自由但连线穿角 → 顶死）
        cand = []
        for dr in range(-self.win, self.win + 1):
            for dc in range(-self.win, self.win + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and f[rr, cc] >= 0:
                    v = float(f[rr, cc]) + 0.15 * np.hypot(dr, dc) * CELL_C
                    cand.append((v, rr, cc))
        if not cand:
            return np.array([0.0, 0.5])
        # 末段直取：贴近目标且视线可达时直接瞄准目标中心
        # （成功由 env 的「包围盒表面距离 ≤0.30 m」判据触发，无需精确停靠）
        if env.d_surf == env.d_surf and env.d_surf < 0.9 and \
                segment_free(env.scene, env.pose[:2], goal, 0.11):
            ang = float(np.arctan2(goal[1] - env.pose[1],
                                   goal[0] - env.pose[0]) - env.pose[2])
            ang = (ang + np.pi) % (2 * np.pi) - np.pi
            v = env.cfg["v_max"] * float(np.clip(1.0 - abs(ang) / np.pi * 1.4,
                                                 0.2, 1.0))
            w = float(np.clip(self.k_ang * ang,
                              -env.cfg["w_max"], env.cfg["w_max"]))
            return np.array([v, w])
        cand.sort()
        tgt = None
        for _, rr, cc in cand:          # 全量扫描：前 N 个可能全被墙角挡住
            p = env.scene.c2w((rr, cc))
            if segment_free(env.scene, env.pose[:2], p, 0.11):
                tgt = p
                break
        if tgt is None:
            return np.array([0.0, 0.6])
        ang = float(np.arctan2(tgt[1] - env.pose[1], tgt[0] - env.pose[0])
                    - env.pose[2])
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        # 大角度先原地转向（边走边转会在墙角来回顶）
        if abs(ang) > 1.0:
            return np.array([0.0, float(np.clip(self.k_ang * ang,
                                                -env.cfg["w_max"], env.cfg["w_max"]))])
        v = env.cfg["v_max"] * float(np.clip(1.0 - abs(ang) / np.pi * 1.4, 0.2, 1.0))
        w = float(np.clip(self.k_ang * ang, -env.cfg["w_max"], env.cfg["w_max"]))
        d = float(np.linalg.norm(goal - env.pose[:2]))
        if d < 0.9:
            v *= max(0.3, d / 0.9)
        return np.array([v, w])


CELL_C = 0.05          # 与 sim.world.CELL 一致（独立常量，避免循环导入）


def make_policy(name, env=None):
    if name == "follow_reactive":
        return ReactiveFollow()
    if name == "follow_oracle":
        return OracleFollow()
    if name == "follow_e4":
        return FollowPolicyE4(use_kalman=True, use_depth=True,
                              use_gamma=True, use_enhance=True)
    if name == "oracle" and env is not None and \
            getattr(env.scene, "name", "") == "apartment":
        return FieldFollower()      # 公寓场景换专用跟随器（见类注释）
    return {"oracle": OraclePolicy, "sweep": SweepDetectorPolicy,
            "random_target": RandomTargetPolicy, "random": RandomPolicy,
            "vla": VLAPolicy}[name]()


def apply_instruction(env, instr):
    """把（语音归一化后的）指令落到当前场景：匹配场景中唯一指代表述并替换任务。

    与 voice_server 的归一化句式（去{color}的{cls}旁边 / 去{cls}旁边）逐字对齐
    `env._all_phrasings` 的产出；找不到则报错，让用户换说法或换场景。
    """
    instr = str(instr).strip()
    for oi, obj in enumerate(env.scene.objects):
        for lvl, text, ref in env._all_phrasings(obj):
            if text == instr:
                env.task = {"level": lvl, "goal": obj.xy.copy(),
                            "instruction": text, "target_cls": obj.cls,
                            "ref_cls": ref, "obj_index": int(oi),
                            "phrasings": env._all_phrasings(obj)}
                d = float(np.linalg.norm(env.task["goal"] - env.pose[:2]))
                env.prev_dist = env.d0 = env.min_dist = d
                env.lstar = grid_value(env._shortest_path_len(env.task["goal"]),
                                       env.scene, env.start_xy)
                return True
    return False


def _pick_env(sim, policy_name, scenario, scene_name=None):
    """按策略与场景选择环境（导航/跟随 × 标定实验场景/公寓，均懒创建复用）。"""
    scene_name = scene_name or "lab"
    if policy_name.startswith("follow"):
        sc = scenario or "normal"
        key = (sc, scene_name)
        if sim.fenv is None or getattr(sim.fenv, "_key", None) != key:
            sim.fenv = FollowEnv(seed=0, max_steps=300, scenario=sc,
                                 camera_augment=0.25,
                                 layout=scene_name if scene_name != "lab" else None)
            sim.fenv._key = key
        return sim.fenv
    if scene_name == "apartment":
        if sim.aenv is None:
            # 公寓 11.4×8.6 m 远大于 8×6 标定场景，路径更长 → 步数上限放宽
            sim.aenv = calibrated_env(layout="apartment", max_steps=350)
        return sim.aenv
    return sim.env


def run_episode(sim, policy_name, seed, level, collect_only=False,
                instruction=None, scenario=None, scene_name=None):
    env = _pick_env(sim, policy_name, scenario, scene_name)
    follow = policy_name.startswith("follow")
    obs = env.reset(seed=seed)
    pol = make_policy(policy_name, env=env)
    pol.reset(env=env)
    if instruction and not follow:
        if not apply_instruction(env, instruction):
            raise RuntimeError("指令「%s」在当前场景中无对应目标"
                               "（试试：去白色的箱子旁边 / 去沙发旁边）" % instruction)
    steps = 0
    if not collect_only:
        sim.refresh_map(env)
    info = None
    while True:
        a = pol.act(obs, env)
        obs, _r, done, info = env.step(a)
        steps += 1
        if collect_only:
            if done or steps >= env.cfg["max_steps"]:
                break
        else:
            with LOCK:
                sim.obs = obs
                sim.pose = [float(env.pose[0]), float(env.pose[1]),
                            float(env.pose[2])]
                sim.traj = [list(map(float, p)) for p in env.traj]
                if follow:
                    sim.goal = None
                    sim.target = [float(x) for x in env.target.xy]
                    sim.live = {"dist": float(info["dist"]),
                                "visible": bool(info["visible"]),
                                "in_band": bool(info["in_band"])}
                    sim.instruction = "跟随红马甲行人"
                    sim.level_str = "follow"
                else:
                    sim.goal = [float(x) for x in env.task["goal"]]
                    sim.target = None
                    sim.live = None
                    sim.instruction = env.task["instruction"]
                    sim.level_str = str(env.task.get("level", "-"))
                sim.step = steps
                sim.done = bool(done)
            time.sleep(0.05)
            if done or steps >= env.cfg["max_steps"]:
                break
    if follow:
        m = env.metrics()
        return {"follow": True,
                "success": float(m["TCR"] >= 0.7 and m["in_band_rate"] >= 0.7),
                "TCR": round(100 * m["TCR"], 1),
                "err_mean": round(m["err_mean"], 3),
                "err_std": round(m["err_std"], 3),
                "vis_rate": round(100 * m["vis_rate"], 1),
                "in_band_rate": round(100 * m["in_band_rate"], 1),
                "n_loss": m["n_loss"],
                "rec_ms_median": round(m["rec_ms_median"], 1),
                "steps": steps, "n_collide": m["n_collide"],
                "instruction": "跟随红马甲行人", "level": "follow"}
    m = navigate_metrics(info)
    return {"success": float(m["SR"]), "spl": float(m["SPL"]), "ne": float(m["NE"]),
            "steps": steps, "min_dist": float(env.min_dist),
            "instruction": env.task["instruction"],
            "level": str(env.task.get("level", "-"))}


def worker_single(sim, policy_name, seed, level, instruction=None, scenario=None,
                  scene_name=None):
    try:
        res = run_episode(sim, policy_name, seed, level,
                          instruction=instruction, scenario=scenario,
                          scene_name=scene_name)
        with LOCK:
            sim.result = res
            sim.done = True
            sim.mode = "done"
    except Exception as e:                              # noqa: BLE001
        with LOCK:
            sim.mode = "error"
            sim.result = {"error": "%s: %s" % (type(e).__name__, e)}


def worker_batch(sim, policy_name, n, level, scene_name=None):
    rows = []
    try:
        for ep in range(n):
            if sim._stop:
                break
            res = run_episode(sim, policy_name, 9000 + ep * 37, level,
                              collect_only=True, scene_name=scene_name)
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
        if self.path.startswith("/static/") and \
                self.path[len("/static/"):] in STATIC:
            fn = os.path.join(BASE_DIR, self.path.lstrip("/"))
            body = open(fn, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", STATIC[self.path[len("/static/"):]])
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/":
            body = open(HTML_FILE, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            with LOCK:
                if SIM.map_png is None:
                    # 首次访问即预渲染默认场景（公寓）的底图，页面打开就有户型图
                    try:
                        if SIM.aenv is None:
                            SIM.aenv = calibrated_env(layout="apartment",
                                                      max_steps=350)
                        SIM.refresh_map(SIM.aenv)
                    except Exception:                       # noqa: BLE001
                        pass
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
                SIM.target = None
                SIM.live = None
                SIM.batch["running"] = False
                SIM.worker = threading.Thread(
                    target=worker_single,
                    args=(SIM, SIM.policy_name, SIM.seed, SIM.level,
                          req.get("instruction"), req.get("scenario"),
                          req.get("scene")),
                    daemon=True)
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
                          int(req.get("n", 20)), req.get("level"),
                          req.get("scene")), daemon=True)
                SIM.worker.start()
            self._json({"ok": True})
        elif self.path == "/api/voice":
            # 代理到 voice_server(:8010) 的 /asr：录音 → 转写 → 归一化指令
            try:
                import urllib.request
                r = urllib.request.Request(
                    "http://127.0.0.1:8010/asr",
                    json.dumps({"wav_b64": req.get("wav_b64", "")}).encode(),
                    {"Content-Type": "application/json"})
                out = json.loads(urllib.request.urlopen(r, timeout=60).read())
            except Exception as e:                          # noqa: BLE001
                out = {"error": "语音服务(8010)不可达: %s" % e}
            self._json(out)
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
