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
                           SweepDetectorPolicy, ExploreMemoryPolicy)
from sim.follow import (FollowEnv, OracleFollow, ReactiveFollow, FollowPolicyE4,
                        FollowPolicyStage, place_peds, crowding)
from sim.planner import GeodesicWindowPolicy
from sim.world import segment_free

LOCK = threading.Lock()
HTML_FILE = "web_sim.html"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC = {"three.min.js": "application/javascript",
          "OrbitControls.js": "application/javascript"}


def _door_rects(mask, bounds, cell=0.1):
    """0.1 m 门洞栅格 → 米制矩形列表（4 邻域连通域外接框）。

    只给前端画门弧用：开口位置/朝向由矩形长短边决定，不参与任何物理计算。
    """
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    rects = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            rmin = rmax = r0
            cmin = cmax = c0
            while stack:
                r, c = stack.pop()
                rmin, rmax = min(rmin, r), max(rmax, r)
                cmin, cmax = min(cmin, c), max(cmax, c)
                for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if (0 <= rr < h and 0 <= cc < w and mask[rr, cc]
                            and not seen[rr, cc]):
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            rects.append([float(bounds[0] + cmin * cell),
                          float(bounds[2] + rmin * cell),
                          float(bounds[0] + (cmax + 1) * cell),
                          float(bounds[2] + (rmax + 1) * cell)])
    return rects


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
        self.map_meta = None
        self.map_v = 0             # 户型元数据版本号：前端带 ?mv=N 时只在变化才下发
        self.lidar_pts = []
        self.target = None        # 跟随模式：被跟随者（红马甲）位置
        self.live = None          # 跟随模式：实时 距离/可见/带内；explore 模式：阶段/覆盖/地标
        self.mem = None           # explore 模式：机器人记忆栅格（阶段一攒的占据图）
        self.fenv = None          # 跟随环境（懒创建，按 scenario 复用）
        self.aenv = None          # 公寓场景导航环境（懒创建）
        self.avenvs = {}          # 各固定布局的导航环境缓存（公寓/走廊/仓库…）
        self.scene_name = "apartment"
        self.active_env = None    # 当前回合使用的环境（清晰相机渲染用）
        self.batch = {"running": False, "done": 0, "n": 0, "sr": None,
                      "spl": None, "ne": None, "rows": []}
        self.worker = None
        self._stop = False

    def refresh_map(self, env=None):
        """生成 2D 户型图 / 3D 场景共用的矢量元数据（不再传栅格 PNG）。

        前端用 `walls` 画墙、`doors` 画门洞弧线、`rooms` 铺地板色并标名称，
        比传一张 640 px 位图省带宽，也更像真实户型图。
        """
        env = env or self.env
        sc = env.scene
        b = sc.bounds
        # 2×2 最大池化到 0.1 m 格；门洞单独取连通域矩形
        sem = sc.sem
        h2, w2 = sem.shape[0] // 2, sem.shape[1] // 2
        p = sem[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2)
        walls = (p == 1).any(axis=(1, 3))
        doors = _door_rects((p == 2).any(axis=(1, 3)), b)
        names = getattr(sc, "room_names", None)
        if not names:                                   # 随机 MultiRoom 无房间名
            names = [("", rect) for rect in sc.rooms]
        self.map_meta = {"b": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                         "name": sc.name,
                         "walls": "".join("1" if v else "0"
                                          for row in walls for v in row),
                         "wh": [int(w2), int(h2)], "cell": 0.1,
                         "doors": doors,
                         "rooms": [{"name": n, "rect": [float(v) for v in rect]}
                                   for n, rect in names],
                         "objs": [{"cls": o.cls, "color": o.color,
                                   "xy": [float(o.xy[0]), float(o.xy[1])],
                                   "half": float(o.half),
                                   "h": float(o.height)} for o in sc.objects]}
        self.map_v += 1

    def snapshot(self):
        lidar = []
        if self.obs is not None:
            th = self.pose[2]
            scan = self.obs["lidar"]
            n = len(scan)
            for i in range(0, n, 4):
                a = th + i * 2 * np.pi / n
                r = float(scan[i])
                if r < 11.5:
                    lidar.append([round(float(self.pose[0] + r * np.cos(a)), 2),
                                  round(float(self.pose[1] + r * np.sin(a)), 2)])
        # 车载相机画面不再由服务端渲染：前端用同一个 three.js 场景、按机器人位姿取
        # 第一视角，形状/颜色与 3D 视图天然一致（后端软件渲染只能画方盒，观感对不上），
        # 顺带把每帧 ~7 KB 的 JPEG 与编码开销整条去掉。
        # 轨迹/位姿只用于画图，降到 2~3 位小数即可，公网链路上能省掉一大半字节
        traj = [[round(float(v), 2) for v in p]
                for p in self.traj[::max(1, len(self.traj) // 300)]]
        goal = (None if self.goal is None
                else [round(float(v), 3) for v in self.goal])
        return {"mode": self.mode, "policy": self.policy_name, "seed": self.seed,
                "step": self.step, "instruction": self.instruction,
                "level": self.level_str,
                "pose": [round(float(v), 3) for v in self.pose], "traj": traj,
                "goal": goal, "lidar": lidar,
                "map_meta": self.map_meta, "map_v": self.map_v,
                "target": self.target, "live": self.live, "mem": self.mem,
                "done": self.done, "result": self.result, "batch": self.batch}

    def ensure_map(self):
        """首次访问即预渲染默认场景（公寓）的户型数据，页面打开就有图。"""
        if self.map_meta is not None:
            return
        try:
            if self.aenv is None:
                self.aenv = calibrated_env(layout="apartment", max_steps=350)
            self.refresh_map(self.aenv)
        except Exception:                                   # noqa: BLE001
            pass

    def frame(self, mv=-1):
        """取一份快照；与调用方已持有的户型版本号一致时不回传 map_meta。

        公网上每一帧的字节数直接决定刷新率，map_meta 约 12 KB，只在切换场景时
        （map_v 变化）下发，其余帧只有几 KB。
        """
        snap = self.snapshot()
        if snap["map_v"] == mv:
            snap["map_meta"] = None
        return snap


SIM = Sim()


def cv2_jpeg(rgb):
    import cv2
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 80])
    return ok, buf.tobytes()


def mem_payload(pol, step):
    """把 ExploreMemoryPolicy 阶段一攒下的记忆栅格编码给前端 2D 叠加显示。

    grid 值 -1/0/1（未知/自由/占据）→ +1 后按 uint8 base64（约 14k 格 ≈ 19 KB），
    `v` 是步数版本号：前端只在版本变化时重画暗区，避免每帧重画上万格。
    """
    fr = pol.front
    if fr.grid is None:
        return None
    g = (np.asarray(fr.grid, dtype=np.int16) + 1).astype(np.uint8)
    return {"v": int(step),
            "g": base64.b64encode(g.tobytes()).decode(),
            "shape": [int(g.shape[0]), int(g.shape[1])],
            "b": [float(x) for x in fr.bounds],
            "res": float(fr.res),
            "phase": pol.phase,
            "lm": (None if pol.landmark is None
                   else [float(pol.landmark[0]), float(pol.landmark[1])])}


class VLAPolicy:
    """经 infer_server(:8000) 调真模型；chunk 执行（每次推理走 10 步）。

    **工程侧指标**（对齐参考论文1 的端到端延迟/FPS、论文2 的推理时延）：
    每次 HTTP 推理的耗时、请求数、失败数、chunk 复用率都会记在 `self.stats`，
    由 `run_episode` 汇总进回合结果。
    """

    def __init__(self, server="http://127.0.0.1:8000"):
        self.server = server
        self.queue = []
        self.stats = {"requests": 0, "fails": 0, "total_ms": 0.0, "lat": [],
                      "cached_steps": 0, "chunk_len": 0}

    def reset(self, env=None):
        self.queue = []
        self.stats = {"requests": 0, "fails": 0, "total_ms": 0.0, "lat": [],
                      "cached_steps": 0, "chunk_len": 0}

    def act(self, obs, env):
        if self.queue:
            self.stats["cached_steps"] += 1
            return self.queue.pop(0)
        import urllib.request
        import urllib.error
        ok, buf = cv2_jpeg(obs["image"])
        req = {"jpg_b64": base64.b64encode(buf).decode(),
               "scan": obs["lidar"].tolist(),
               "state": [float(x) for x in obs["state"][:5]],
               "instruction": obs["instruction"]}
        t0 = time.perf_counter()
        try:
            r = urllib.request.urlopen(urllib.request.Request(
                self.server + "/infer", json.dumps(req).encode(),
                {"Content-Type": "application/json"}), timeout=30)
            out = json.loads(r.read())
        except urllib.error.HTTPError as e:              # 服务器 5xx：带出错误详情
            self.stats["fails"] += 1
            try:
                detail = json.loads(e.read().decode()).get("error", "")
            except Exception:                            # noqa: BLE001
                detail = ""
            raise RuntimeError(f"infer HTTP {e.code}: {detail or e.reason}")
        except Exception:                                    # noqa: BLE001
            self.stats["fails"] += 1
            raise
        finally:
            dt = (time.perf_counter() - t0) * 1000.0
            self.stats["requests"] += 1
            self.stats["total_ms"] += dt
            self.stats["lat"].append(dt)
        if "error" in out:
            raise RuntimeError(out["error"])
        acts = out.get("actions") or [out["action"]]
        self.queue = [np.asarray(a, np.float64) for a in acts[1:]]
        self.stats["chunk_len"] = max(self.stats["chunk_len"], len(acts))
        return np.asarray(acts[0], np.float64)


class FieldFollower(GeodesicWindowPolicy):
    """**公寓/固定户型专用**的测地场跟随器（demo 用，不影响论文基线）。

    实现已上移到 `sim/planner.GeodesicWindowPolicy`（CLI 评测与网页共用一份），
    这里只保留一个带名字的子类，方便网页策略下拉显示「field」。

    为什么不用 `OraclePolicy`：它是在 8×6 m **单房间**标定场景（无墙无门）上
    调出来的 —— 避障减速 + 前瞻窗口的组合在门洞两侧墙角会陷入
    「贴墙 → 前方光束 <0.2 m → v=0 → 脱困自旋 → 又贴墙」的死循环
    （公寓 15 个跨房间目标实测 12 个失败）。而论文的 oracle 数据必须
    保持原实现，所以另用**窗口式测地场规划器**：窄走廊/S 形走廊/仓库
    实测 4/4 成功（`oracle`/`planner_oracle` 在这些场景只有 0~1/4）。
    """

    name = "field"

    def __init__(self, **kw):
        kw.setdefault("use_gt", True)
        super().__init__(**kw)


CELL_C = 0.05          # 与 sim.world.CELL 一致（独立常量，避免循环导入）


class TourFollow(ReactiveFollow):
    """**全屋巡回专用**跟随（演示用，不改 E11 论文口径）。

    借鉴市售室内跟随产品（激光 SLAM 跟随底盘 / 视觉跟随行李箱等）的
    「跟踪不丢、跟得要紧」三板斧：

    ① **目标速度估计 + 前馈**（对应产品的 Kalman/EMA 速度跟踪）：
       对人的位置做指数滑动平均差分得到速度 `self._pv`，控制量里直接
       叠加 `v_ff = |pv|` 的前馈 —— 名义距离处机器人即以人速同行，
       不再等「落后 → 距离误差 → 加速」的滞后链条，跟得更紧；
    ② **预测瞄准**（对应产品的 lead-point 跟踪）：瞄准点取人位置沿
       速度方向外推 0.8 s，转弯提前量让过弯不再「切内丢目标」；
    ③ **测地寻回**（对应产品的 BLE/UWB 先验 + 建图寻回）：丢失或
       门洞遮挡时在行人膨胀栅格上对人当前位置做 BFS，沿场梯度尾随。
    """
    name = "tour_follow"
    LEAD_T = 0.8          # 预测瞄准外推时间（s）
    K_V_FF = 1.2          # 追赶增益（×v_max / m 距离误差）

    def reset(self, env):
        super().reset(env)
        self.lost = 0
        self._sf = None
        self._sg = None
        self._p_prev = None
        self._pv = np.zeros(2)

    def _seek(self, obs, env):
        """测地尾随：沿朝人（预测位置）的 BFS 场梯度走（演示主控制器）。"""
        goal = np.asarray(env.target.xy, dtype=float)
        pred = goal + self._pv * self.LEAD_T
        if self._sf is None or self._sg is None or \
                not np.allclose(goal, self._sg, atol=0.3):
            self._sf = env._shortest_path_len(goal)
            self._sg = goal.copy()
        f = self._sf
        H, W = f.shape
        r, c = map(int, env.scene.w2c(env.pose[:2]))
        cand = []
        for dr in range(-4, 5):
            for dc in range(-4, 5):
                r1, c1 = r + dr, c + dc
                if 0 <= r1 < H and 0 <= c1 < W and f[r1, c1] >= 0:
                    # 候选价值 = 场值 + 距预测点距离惩罚：同样接近人的
                    # 候选里优先选「顺路朝人前进方向」的，过弯提前切弯
                    v = float(f[r1, c1]) + 0.1 * np.hypot(dr, dc) * 0.05 \
                        + 0.03 * float(np.linalg.norm(
                            env.scene.c2w((r1, c1)) - pred))
                    cand.append((v, r1, c1))
        if not cand:
            return np.array([0.0, self.search_w])
        cand.sort()
        tgt = None
        tgt_fb = None                       # 兜底：场值最小者（不再原地自旋）
        from sim.world import segment_free
        for _, r1, c1 in cand:
            w_pt = env.scene.c2w((r1, c1))
            if tgt_fb is None:
                tgt_fb = w_pt
            if segment_free(env.scene, env.pose[:2], w_pt, 0.11):
                tgt = w_pt
                break
        if tgt is None:
            tgt = tgt_fb
        ang = float(np.arctan2(tgt[1] - env.pose[1], tgt[0] - env.pose[0])
                    - env.pose[2])
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        w = float(np.clip(self.k_ang * ang, -self.w_max, self.w_max))
        dist = float(np.linalg.norm(goal - env.pose[:2]))
        # 速度前馈：比例项（允许负修正，防过冲顶人）+ 人速前馈 ——
        # 名义距离处恰好 ≈ 人速同行，落后则加速、超前则减速
        pv = float(np.linalg.norm(self._pv))
        v_ff = min(pv, self.v_max) if pv > 0.02 else 0.0
        # 近距硬刹车：dist < 0.85 m（带内下限 0.9）时减速退避，防顶人/钻到人身后
        brake = float(np.clip((dist - 0.85) / 0.25, 0.0, 1.0))
        v = (v_ff + self.v_max * float(np.clip(self.K_V_FF * (dist - env.want_dist),
                                               -0.6, 1.0))) * brake
        # 大角度仍要转向，但比原来 1.0 放宽：边走边转跟随产品默认行为，
        # 场路径本身无碰撞，无需停转（停转反而落后、丢失）
        if abs(ang) > 1.2:
            return np.array([0.0, w])
        v *= float(np.clip(1.0 - abs(ang) / 1.6, 0.5, 1.0))
        return np.array([min(v, self.v_max), w])

    def act(self, obs, env):
        # 演示模式：测地尾随为主控制器；视觉检测仅驱动丢失计数/状态显示
        p = env._perceived(obs)
        self.lost = 0 if p is not None else self.lost + 1
        self._sf = None if p is not None else self._sf   # 重新看见即刷新场
        # 目标速度 EMA 估计（对应产品 Kalman 滤波的速度通道）
        cur = np.asarray(env.target.xy, dtype=float)
        if self._p_prev is not None:
            self._pv = 0.6 * self._pv + 0.4 * (cur - self._p_prev) / 0.1
        self._p_prev = cur.copy()
        return self._seek(obs, env)


def make_policy(name, env=None):
    if name == "follow_reactive":
        if env is not None and getattr(env, "scenario", "") == "tour":
            return TourFollow()      # 全屋巡回：门洞密集，降低激光安全间隙
        return ReactiveFollow()
    if name == "follow_stage":
        # E11c：显式跟随阶段机（follow/static/lost/search）+ 三级脱困（L1/L2/L3）
        return FollowPolicyStage()
    if name == "follow_stage_s":
        # E11c 逐级消融的「仅阶段机」臂：关掉 L1/L2/L3 与局部重规划
        return FollowPolicyStage(use_l1=False, use_l2=False, use_l3=False,
                                 use_replan=False)
    if name == "follow_oracle":
        return OracleFollow()
    if name == "follow_e4":
        return FollowPolicyE4(use_kalman=True, use_depth=True,
                              use_gamma=True, use_enhance=True)
    if name == "geodesic":
        return FieldFollower()      # 窗口测地场上界（新固定布局）
    if name == "explore":
        # 巡屋建图 → 瞥见记地标 → 记忆寻物（两阶段「视野外寻物」演示）
        return ExploreMemoryPolicy()
    if name == "oracle" and env is not None and \
            getattr(env.scene, "name", "") in LAYOUT_STEPS:
        # **固定户型**（公寓/窄走廊/S 形走廊/仓库）统一换测地场上界：
        # 原版 OraclePolicy 在门洞/货架端头会陷入贴墙死循环（实测 0~1/4），
        # 论文主结果仍用 CLI 的原始 oracle，不受影响。
        return FieldFollower()
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


# 各固定布局的步数上限（场景越大路径越长，步数按测地尺度放宽）
LAYOUT_STEPS = {"apartment": 350, "apartment_s": 300, "apartment_l": 450,
                 "corridor": 700, "s_corridor": 700, "warehouse": 650,
                 "classroom": 600, "office": 600}

# **硬件退化族**（论文1/2 报告的 sim-to-real 差距来源；默认全关，保全论文数字）
DEGRADE_KW = {
    "none": {},
    "slip": dict(slip=0.25),                  # 轮滑：实际位移小于指令
    "odom": dict(odom_drift=0.03),            # 里程计漂移：上报位姿带偏差
    "drop": dict(frame_drop=0.25),            # 掉帧：画面停在上一帧
    "delay": dict(act_delay=3),               # 执行器/网络延迟：动作延后生效
    "accel": dict(accel=0.5, acc_ang=1.0),    # 加减速斜坡：速度不能瞬变
    "peds": dict(dyn_peds=True, n_peds=3),    # 行人作为动态障碍（挡激光/可撞）
    "combo": dict(slip=0.2, odom_drift=0.02, frame_drop=0.15, act_delay=2,
                  accel=0.5, acc_ang=1.0, dyn_peds=True, n_peds=2),
}


def _pick_env(sim, policy_name, scenario, scene_name=None, dens=0):
    """按策略与场景选择环境（导航/跟随 × 标定实验场景/公寓/新场景，均懒创建复用）。

    `dens`（E11b 人群拥挤度分层）只影响跟随环境：非 0 时让 FollowEnv 生成
    对应数量的额外行人（随后由 `place_peds` 受控摆位），并参与缓存 key，
    改密度即重建环境。
    """
    scene_name = scene_name or "lab"
    if policy_name.startswith("follow"):
        sc = scenario or "normal"
        if sc == "tour":
            scene_name = "apartment"     # tour 航点按公寓 11.4×8.6 m 生成，强制绑定
        dens = int(dens or 0)
        key = (sc, scene_name, dens)
        if sim.fenv is None or getattr(sim.fenv, "_key", None) != key:
            # tour 全屋巡回路程约 45 m → 步数大幅放宽；跟随更紧、人速更慢
            steps = 2200 if sc == "tour" else LAYOUT_STEPS.get(scene_name, 300)
            kw = dict(want_dist=1.1, target_speed=0.15) if sc == "tour" else {}
            if dens > 0:
                kw["n_peds"] = dens
            sim.fenv = FollowEnv(seed=0, max_steps=steps, scenario=sc,
                                 camera_augment=0.25,
                                 layout=scene_name if scene_name != "lab" else None,
                                 **kw)
            sim.fenv._key = key
        return sim.fenv
    if scene_name in LAYOUT_STEPS:
        # 固定户型（公寓/窄走廊/S 形走廊/仓库…）：按布局缓存，步数按尺度放宽
        env = sim.avenvs.get(scene_name)
        if env is None:
            env = calibrated_env(layout=scene_name,
                                 max_steps=LAYOUT_STEPS[scene_name])
            sim.avenvs[scene_name] = env
            if scene_name == "apartment":
                sim.aenv = env
        return env
    return sim.env


def run_episode(sim, policy_name, seed, level, collect_only=False,
                instruction=None, scenario=None, scene_name=None, degrade=None,
                pose=0.0, dens=0, max_steps=None):
    env = _pick_env(sim, policy_name, scenario, scene_name, dens=dens)
    follow = policy_name.startswith("follow")
    sim.active_env = env
    # 硬件退化族：在 reset **之前**写入 cfg（n_peds 决定场景里生成几个行人，
    # 必须在建场景前生效）
    for k, v in DEGRADE_KW.get(degrade or "none", {}).items():
        env.cfg[k] = v
    # E1b 定位噪声：策略看到的位姿改用含高斯噪声的估计值（σ 单位：米）。
    # 环境是跨回合缓存的，所以这里**每回合显式写回**，避免上一次的设置残留。
    pose = float(pose or 0.0)
    env.cfg["use_pose_est"] = bool(pose > 0)
    env.cfg["pose_noise_gauss"] = pose
    obs = env.reset(seed=seed)
    if follow and int(dens or 0) > 0:
        # E11b：把额外行人受控摆到机器人初始位姿的侧向槽位（否则随机撒的
        # 行人远在 3 m 外，密度档位形同虚设）
        place_peds(env, int(dens), "side")
    pol = make_policy(policy_name, env=env)
    pol.reset(env=env)
    if policy_name == "explore":
        # 两阶段（巡屋建图 + 记忆寻物）比单程导航长得多 → 预算 ×5。
        # （实测晚瞥见种子：瞥见可发生在 ~1350 步，随后还需 ~140 步记忆寻物）
        # 环境跨回合缓存：原值记在 env 属性上并显式写回，避免乘法复利。
        if not hasattr(env, "_max_steps0"):
            env._max_steps0 = int(env.cfg["max_steps"])
        env.cfg["max_steps"] = env._max_steps0 * 5
    elif hasattr(env, "_max_steps0"):
        env.cfg["max_steps"] = env._max_steps0
    if max_steps is not None:
        # W1 口径实验：显式覆盖步数预算（每回合写入，无跨批次残留）
        env.cfg["max_steps"] = int(max_steps)
    if instruction and not follow:
        if not apply_instruction(env, instruction):
            raise RuntimeError("指令「%s」在当前场景中无对应目标"
                               "（试试：去白色的箱子旁边 / 去沙发旁边）" % instruction)
    steps = 0
    ctrl_ms = []
    if not collect_only:
        sim.refresh_map(env)
        # explore：预建记忆栅格，第一帧就有「暗区底图」；其他策略清掉残留
        sim.mem = None
        if policy_name == "explore":
            pol.front._ensure(env)
            sim.mem = mem_payload(pol, 0)
    info = None
    while True:
        if sim._stop:                # 「⏹ 停止」按钮：任何运行中的回合立即终止
            break
        _t0 = time.perf_counter()
        a = pol.act(obs, env)
        ctrl_ms.append((time.perf_counter() - _t0) * 1000.0)
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
                                "in_band": bool(info["in_band"]),
                                "rho": int(crowding(env, 2.0)["n_in_r"]),
                                "stage": getattr(pol, "stage", None)}
                    sim.instruction = "跟随红马甲行人"
                    sim.level_str = "follow"
                else:
                    if policy_name == "explore":
                        # 阶段一（巡屋）藏起目标点 —— 机器人自己还没瞥见，
                        # 观众看到的与机器人同信息量；瞥见换段后亮出地标。
                        if pol.phase == "navigate" and pol.landmark is not None:
                            sim.goal = [float(pol.landmark[0]),
                                        float(pol.landmark[1])]
                        else:
                            sim.goal = None
                        sim.live = dict(pol.live)
                        if steps % 10 == 0:
                            sim.mem = mem_payload(pol, steps)
                    else:
                        sim.goal = [float(x) for x in env.task["goal"]]
                        sim.live = None
                    sim.target = None
                    sim.instruction = env.task["instruction"]
                    sim.level_str = str(env.task.get("level", "-"))
                sim.step = steps
                sim.done = bool(done)
            time.sleep(0.05)
            if done or steps >= env.cfg["max_steps"]:
                break
    # ---- 工程侧指标：策略计算耗时（控制延迟）+ VLA 推理延迟/吞吐 ----
    eng = {"ctrl_ms_mean": round(float(np.mean(ctrl_ms)), 2) if ctrl_ms else 0.0,
           "ctrl_ms_p95": round(float(np.percentile(ctrl_ms, 95)), 2) if ctrl_ms else 0.0}
    if isinstance(pol, VLAPolicy):
        st = pol.stats
        lat = st["lat"]
        eng.update({
            "infer_ms_mean": round(float(np.mean(lat)), 1) if lat else 0.0,
            "infer_ms_p95": round(float(np.percentile(lat, 95)), 1) if lat else 0.0,
            "infer_reqs": st["requests"], "infer_fails": st["fails"],
            "chunk_reuse": (st["cached_steps"] /
                            max(st["cached_steps"] + st["requests"], 1)),
            "hz": round(1000.0 / max(float(np.mean(ctrl_ms)), 1e-6), 1),
        })
    if follow:
        m = env.metrics()
        out = {"follow": True,
               "success": float(m["TCR"] >= 0.7 and m["in_band_rate"] >= 0.7),
               "TCR": round(100 * m["TCR"], 1),
               "err_mean": round(m["err_mean"], 3),
               "err_std": round(m["err_std"], 3),
               "vis_rate": round(100 * m["vis_rate"], 1),
               "in_band_rate": round(100 * m["in_band_rate"], 1),
               "n_loss": m["n_loss"],
               "rec_ms_median": round(m["rec_ms_median"], 1),
               "steps": steps, "n_collide": m["n_collide"],
               "rho": int(crowding(env, 2.0)["n_in_r"]),
               "instruction": "跟随红马甲行人", "level": "follow", **eng}
        # E11c：阶段机停留时长 + 三级脱困触发次数（非阶段机策略无此字段）
        stg = getattr(pol, "stats", None)
        if isinstance(stg, dict) and "stage_time" in stg:
            out["stage_time"] = {k: round(v, 1)
                                 for k, v in stg["stage_time"].items()}
            out["stage_trans"] = dict(stg["stage_trans"])
            out["unstick"] = {"L1": int(stg["l1"]["n"]),
                              "L2": int(stg["l2"]["n"]),
                              "L3": int(stg["l3"]["n"])}
            lat = stg.get("replan_ms") or []
            out["replan_ms"] = round(float(np.median(lat)), 1) if lat else 0.0
        return out
    m = navigate_metrics(info)
    out = {"success": float(m["SR"]), "spl": float(m["SPL"]), "ne": float(m["NE"]),
           "steps": steps, "min_dist": float(env.min_dist),
           "instruction": env.task["instruction"],
           "level": str(env.task.get("level", "-")), **eng}
    # W1诊断：失败原因四分类（优先级 near_miss > wander > lost > progress_timeout）
    tl = m.get("TL")
    out["tl"] = round(float(tl), 2) if tl == tl else None
    if not m["SR"]:
        md = float(env.min_dist)
        tl_eff = tl if tl == tl else float("inf")
        if md <= 1.6:
            fm = "near_miss"         # 到过目标附近但未判定成功（半径/朝向边缘）
        elif tl_eff > 4.0:
            fm = "wander"            # 路程≥4倍最短路：绕圈/来回震荡（复合误差特征）
        elif md > 5.0:
            fm = "lost"              # 全程未接近目标（语义/感知失败）
        else:
            fm = "progress_timeout"  # 在推进但步数耗尽（效率问题）
        out["fail_mode"] = fm
    if policy_name == "explore" and not collect_only:
        # 两阶段战报：巡屋覆盖了多少、第几步瞥见地标、记忆寻物是否用了乐观补全
        sim.mem = mem_payload(pol, steps)      # 终版记忆图定格给前端
        out.update({"phase": pol.phase,
                    "cover": round(float(pol.cover), 3),
                    "seen": pol.landmark is not None,
                    "seen_step": pol.seen_step,
                    "explore_steps": int(pol.explore_t),
                    "optim": bool(pol.optim)})
    return out


def worker_single(sim, policy_name, seed, level, instruction=None, scenario=None,
                  scene_name=None, degrade=None, pose=0.0, dens=0):
    try:
        res = run_episode(sim, policy_name, seed, level,
                          instruction=instruction, scenario=scenario,
                          scene_name=scene_name, degrade=degrade,
                          pose=pose, dens=dens)
        with LOCK:
            if sim._stop:
                res = dict(res, stopped=True)    # 「⏹ 停止」终止：标注为手动停止
                sim.mode = "stopped"
            sim.result = res
            sim.done = True
            if sim.mode != "stopped":
                sim.mode = "done"
    except Exception as e:                              # noqa: BLE001
        with LOCK:
            sim.mode = "error"
            sim.result = {"error": "%s: %s" % (type(e).__name__, e)}


def _batch_scenes(scene_name):
    """批量评测的场景集合：'multi' → 跨场景联合统计（论文4 的多规模实验口径）。

    覆盖三种体量：标定单房间（论文主场景）/ 公寓 11.4×8.6 m / 仓库 14×10 m
    —— 规模跨度足够看出「场景大小对成功率/步数的影响」。
    """
    if scene_name in ("multi", "all"):
        return ["lab", "apartment", "warehouse"]
    return [scene_name]


def _pack_rows(rs):
    """把一批回合记录压成统计量（含 Wilson 95% 区间与标准差）。"""
    from sim.plot_results import wilson
    n = len(rs)
    if n == 0:
        return {}
    succ = [float(r.get("success", 0.0)) for r in rs]
    k = int(sum(succ))
    lo, hi = wilson(k, n)
    out = {"n": n,
           "SR": round(100 * sum(succ) / n, 1),
           "SR_lo": round(100 * lo, 1), "SR_hi": round(100 * hi, 1),
           "SR_std": round(100 * float(np.std(succ, ddof=1)), 1) if n > 1 else 0.0}
    for key, lab in (("spl", "SPL"), ("ne", "NE"), ("TCR", "TCR"),
                     ("err_mean", "err"), ("vis_rate", "vis"),
                     ("in_band_rate", "band"), ("rec_ms_median", "rec_ms"),
                     ("n_collide", "collide"),
                     ("ctrl_ms_mean", "ctrl_ms"), ("infer_ms_mean", "infer_ms"),
                     ("infer_ms_p95", "infer_p95"), ("hz", "hz")):
        vals = [float(r[key]) for r in rs if r.get(key) is not None]
        if vals:
            out[lab] = round(float(np.mean(vals)), 3)
            if len(vals) > 1:
                out[lab + "_std"] = round(float(np.std(vals, ddof=1)), 3)
    return out


def _batch_stats(rows, ref_rows=None):
    """分场景 / 合并统计 + 重复次数间的离散度 + 配对 McNemar 检验。"""
    from sim.plot_results import mcnemar_exact
    stats = {"pooled": _pack_rows(rows), "per_scene": {}}
    scenes = sorted({r.get("scene", "?") for r in rows})
    for sc in scenes:
        rs = [r for r in rows if r.get("scene") == sc]
        pack = _pack_rows(rs)
        # 「重复 N 次取平均 + 报方差」口径：按 repeat 分组求均值，再看组间离散度
        reps = sorted({int(r.get("rep", 0)) for r in rs})
        if len(reps) > 1:
            means = []
            for rp in reps:
                rr = [r for r in rs if int(r.get("rep", 0)) == rp]
                s = [float(x.get("success", 0.0)) for x in rr]
                means.append(100 * sum(s) / max(len(s), 1))
            pack["rep_SR_mean"] = round(float(np.mean(means)), 1)
            pack["rep_SR_std"] = round(float(np.std(means, ddof=1)), 2)
            pack["repeats"] = len(reps)
        stats["per_scene"][sc] = pack
    if ref_rows:
        stats["ref"] = _pack_rows(ref_rows)
        key = lambda r: (r.get("scene"), int(r.get("rep", 0)), r.get("seed"))
        A = {key(r): float(r.get("success", 0.0)) for r in rows}
        B = {key(r): float(r.get("success", 0.0)) for r in ref_rows}
        common = sorted(set(A) & set(B))
        n10 = sum(1 for k in common if A[k] == 1 and B[k] == 0)
        n01 = sum(1 for k in common if A[k] == 0 and B[k] == 1)
        stats["mcnemar"] = {"p": round(mcnemar_exact(n10, n01), 4),
                             "n10": n10, "n01": n01, "n": len(common)}
    return stats


def worker_batch(sim, policy_name, n, level, scene_name=None, scenario=None,
                 repeats=1, ref_policy=None, degrade=None, pose=0.0, dens=0,
                 max_steps=None):
    """批量评测：**可跨场景 + 可重复多次 + 可配对参考基线**。

    - 跨场景：`scene_name='multi'` 时在「标定场地 + 公寓」两种场地上跑同一批种子，
      分场景报指标并给出合并统计（论文4 的多规模实验口径）。
    - 重复：第 r 轮用种子 `9000 + ep*37 + 100000*r`，与第 0 轮**同场景不同布局**，
      报「各轮均值的均值 ± 组间标准差」（论文1/2/3/4 都采用「重复 N 次取平均」）。
    - 配对检验：给了 `ref_policy` 就在**同一批种子**上再跑一遍参考策略，
      用 McNemar 精确检验给出显著性 p 值。
    """
    rows = []
    ref_rows = []
    scenes = _batch_scenes(scene_name)
    try:
        total = n * len(scenes) * repeats
        done = 0
        for rep in range(repeats):
            if sim._stop:
                break
            for sc in scenes:
                if sim._stop:
                    break
                for ep in range(n):
                    if sim._stop:
                        break
                    seed = 9000 + ep * 37 + 100000 * rep
                    res = run_episode(sim, policy_name, seed, level,
                                      collect_only=True, scene_name=sc,
                                      scenario=scenario, degrade=degrade,
                                      pose=pose, dens=dens, max_steps=max_steps)
                    if sim._stop:
                        break        # 被停止打断的半个回合：丢弃，不入统计
                    res.update(scene=sc, rep=rep, seed=seed,
                               max_steps=int(max_steps) if max_steps else None)
                    rows.append(res)
                    if ref_policy:
                        r2 = run_episode(sim, ref_policy, seed, level,
                                         collect_only=True, scene_name=sc,
                                         scenario=scenario, degrade=degrade,
                                         pose=pose, dens=dens, max_steps=max_steps)
                        r2.update(scene=sc, rep=rep, seed=seed)
                        ref_rows.append(r2)
                    done += 1
                    with LOCK:
                        sim.batch["done"] = done
                        sim.batch["total"] = total
                        sim.batch["rows"] = rows[-8:]
                        sim.batch["sr"] = round(100 * sum(r.get("success", 0)
                                                          for r in rows)
                                                / len(rows), 1)
    except Exception as e:                              # noqa: BLE001
        rows.append({"error": "%s: %s" % (type(e).__name__, e)})
    with LOCK:
        sim.batch["running"] = False
        if sim._stop:
            sim.mode = "stopped"        # 停止要在前端可见（此前 mode 一直卡在 batch）
        if rows and "error" not in rows[-1]:
            sim.batch["stats"] = _batch_stats(rows, ref_rows or None)
        if rows:
            sim.batch["sr"] = round(100 * sum(r.get("success", 0) for r in rows)
                                    / len(rows), 1)
            sim.batch["spl"] = round(sum(r.get("spl", 0) for r in rows) / len(rows), 3)
            sim.batch["ne"] = round(sum(r.get("ne", 0) for r in rows) / len(rows), 2)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"      # 保活 + 让 SSE 能用分块传输成帧

    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _ver(qs, key):
        """从查询串里取一个整数版本号（形如 ?mv=3&iv=17）。"""
        for kv in qs.split("&"):
            if kv.startswith(key + "="):
                try:
                    return int(kv[len(key) + 1:])
                except ValueError:
                    return -1
        return -1

    def _stream(self):
        """SSE 推流：一条长连接持续下发快照，消除「每帧一次跨境往返」。

        公网隧道上单次请求 TTFB ≈0.6 s（冷启首包 1.5~2.5 s），前端轮询把相机
        帧率锁死在 ~1.2 fps；改成服务端主动推流后，帧率只受带宽（实测约
        100 KB/s）与渲染耗时限制，不再每帧付一次 RTT。分块传输（chunked）成帧，
        配合 `X-Accel-Buffering: no` 让中间 nginx 不做缓冲。
        """
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
        except OSError:
            return
        mv = -1
        paddle = time.time() + 3600.0               # 单连接上限 1 h，防幽灵线程
        try:
            with LOCK:
                SIM.ensure_map()
            while time.time() < paddle:
                with LOCK:
                    snap = SIM.frame(mv)
                    mv = snap["map_v"]
                body = ("data: " + json.dumps(snap, ensure_ascii=False)
                        + "\n\n").encode()
                self.wfile.write(b"%x\r\n" % len(body) + body + b"\r\n")
                self.wfile.flush()
                time.sleep(0.08)                    # 状态帧 ~10 Hz，相机帧由 Sim 内部节流
        except (BrokenPipeError, ConnectionResetError, OSError, ValueError):
            pass
        finally:
            self.close_connection = True

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
        elif self.path.split("?")[0] == "/api/stream":
            self._stream()
        elif self.path.split("?")[0] == "/api/state":
            # ?mv=N：前端已持有的户型版本号；一致时不再重复下发 map_meta（约 12 KB）
            qs = self.path.split("?", 1)[1] if "?" in self.path else ""
            mv = self._ver(qs, "mv")
            with LOCK:
                SIM.ensure_map()
                snap = SIM.frame(mv)
            self._json(snap)
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
                SIM._stop = False        # 无条件复位：上次「停止」不能毒死新回合
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
                          req.get("scene"), req.get("degrade"),
                          float(req.get("pose") or 0.0),
                          int(req.get("dens") or 0)),
                    daemon=True)
                SIM.worker.start()
            self._json({"ok": True})
        elif self.path == "/api/batch":
            with LOCK:
                if SIM.worker and SIM.worker.is_alive():
                    SIM._stop = True
                    SIM.worker.join(timeout=10)
                SIM._stop = False        # 无条件复位（与 /api/start 同理）
                SIM.mode = "batch"
                _ms = int(req.get("max_steps") or 0) or None
                SIM.batch = {"running": True, "done": 0,
                             "total": int(req.get("n", 20)) *
                             int(req.get("repeats", 1)) *
                             len(_batch_scenes(req.get("scene"))),
                             "n": int(req.get("n", 20)), "sr": None,
                             "spl": None, "ne": None, "rows": [],
                             "stats": None, "repeats": int(req.get("repeats", 1)),
                             "ref": req.get("ref_policy") or None,
                             "max_steps": _ms}
                SIM.worker = threading.Thread(
                    target=worker_batch,
                    args=(SIM, req.get("policy", "oracle"),
                          int(req.get("n", 20)), req.get("level"),
                          req.get("scene"), req.get("scenario"),
                          int(req.get("repeats", 1)),
                          req.get("ref_policy") or None,
                          req.get("degrade"),
                          float(req.get("pose") or 0.0),
                          int(req.get("dens") or 0),
                          _ms), daemon=True)
                SIM.worker.start()
            self._json({"ok": True})
        elif self.path == "/api/preview":
            # 场景/遮挡场景/人群密度下拉框切换 → 立即重建底图并归位姿态。
            # 2D 平面图与 3D 视图都消费同一份 snapshot 的 map/map_meta，
            # 切下拉框即换场景，无需先点 ▶（此前要等回合启动才刷新，
            # 表现为"下拉框切了、两块视图没跟上"）。
            with LOCK:
                busy = (SIM.mode in ("running", "batch")
                        or (SIM.worker and SIM.worker.is_alive())
                        or SIM.batch.get("running"))
                if busy:
                    self._json({"ok": False, "busy": True})
                    return
                try:
                    pol = req.get("policy") or "geodesic"
                    dens = int(req.get("dens") or 0)
                    env = _pick_env(SIM, pol, req.get("scenario"),
                                    req.get("scene"), dens=dens)
                    SIM.active_env = env
                    env.reset(seed=int(req.get("seed", 9000)))
                    if pol.startswith("follow") and dens > 0:
                        place_peds(env, dens, "side")
                    SIM.refresh_map(env)
                    # 清掉上个场景的残留显示（位姿/轨迹/结果），避免新旧串画
                    SIM.mode = "idle"
                    SIM.done = False
                    SIM.result = None
                    SIM.step = 0
                    SIM.traj = []
                    SIM.obs = None
                    SIM.mem = None
                    SIM.policy_name = pol
                    SIM.seed = int(req.get("seed", 9000))
                    SIM.pose = [float(env.pose[0]), float(env.pose[1]),
                                float(env.pose[2])]
                    if pol.startswith("follow"):
                        SIM.target = [float(x) for x in env.target.xy]
                        SIM.goal = None
                        SIM.live = None
                        SIM.instruction = "跟随红马甲行人"
                        SIM.level_str = "follow"
                    else:
                        SIM.target = None
                        SIM.live = None
                        # 同步任务显示：reset 已重采 env.task，否则切场景后
                        # 状态栏/目标点还挂着上一个场景的旧指令与旧目标
                        tk = env.task or {}
                        if tk.get("instruction"):
                            SIM.instruction = tk["instruction"]
                        SIM.level_str = str(tk.get("level", "-"))
                        SIM.goal = ([float(x) for x in tk["goal"]]
                                    if tk.get("goal") is not None else None)
                    self._json({"ok": True,
                                "name": (SIM.map_meta or {}).get("name")})
                except Exception as e:                      # noqa: BLE001
                    self._json({"ok": False, "error": str(e)})
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
