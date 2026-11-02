#!/usr/bin/env python3
"""hier_policy.py —— P18 真机分层混合策略（E20-hybrid 的服务端移植）。

「可见即规划，不可见即模型」的真机版仲裁（与仿真 web_sim.HierVLAPolicy
同构，行为对齐 sim.baselines.SweepDetectorPolicy(use_plan=True) + SweepDetDock）：

    检出（含 mem_ttl 短期记忆命中）→ 执行层：LocalPlanner 局部绕障逼近
                                    + 激光末段停靠（surf ≤ dock_target 停住）
    未检出                        → 搜索层：VLA chunk（infer_server 侧队列）

**非特权口径**（与 E20 info_boundary 逐条对齐）：
- 目标世界坐标 = 里程计位姿 + 相机方位 + 激光/单目距离，无真值；
- 执行层地图 = 激光 DDA 在线累积（LocalPlanner，numpy+scipy 自包含）；
- 记忆 = 世界坐标检测的指数平滑，物体静止假设（与仿真一致）。

## 与仿真的两处**有意差异**（真机口径决定，均已注释）

1. `det_every=1`：仿真里 HSV 检测 ~10 ms/帧隔帧做；真机检测在 infer_server
   本来就要算（det 特征），复用同一份 blobs，逐帧仲裁更反应。
2. 逐请求返回单步动作（不回 actions[]）：仲裁需要每步都能打断 chunk，
   chunk 复用在**服务端队列**完成（infer_server.hier），车侧 10Hz 轮询。

## bearing 约定（真机最容易错的地方，务必现场核验）

backfill_real 的 `bearing_deg = atan2(cx - (w-1)/2, f)`：目标在**画面右半**
则为正。若相机未镜像（画面右 = 机器人右），物理方位应为逆时针负角，
`ang_ccw = -bearing_deg`；若采集链路水平镜像（MV.py 同款 flip），则为正，
`ang_ccw = +bearing_deg`。由 robot.yaml `camera.mirror` 控制，
**默认 true（保持既有 lidar_at 行为）**，上板第一步必须按手册 §3 核验。
"""
from __future__ import annotations

import math

import numpy as np


def lidar_index(ang_ccw_deg: float, n: int) -> int:
    """相对方位角（CCW 为正，度）→ 激光束索引。

    scan 约定（vla_bridge._scan 重排后）：index 0 ↔ 相对角 -180°，
    逆时针递增，车头 = n/2 —— 与仿真 beam_index 同一口径。
    """
    return int(round(((float(ang_ccw_deg) + 180.0) % 360.0) / 360.0 * n)) % n


def surf_dist(scan, ang_ccw_deg: float, half_win_deg: float = 9.0,
              max_range: float = 12.0) -> float:
    """目标方位 ±half_win 内的激光最小值（表面距；末段停靠唯一有效距离）。"""
    n = len(scan)
    if n == 0:
        return float("nan")
    i0 = lidar_index(ang_ccw_deg, n)
    k = max(1, int(round(half_win_deg / 360.0 * n)))
    idx = (i0 + np.arange(-k, k + 1)) % n
    seg = np.asarray(scan)[idx]
    seg = seg[np.isfinite(seg) & (seg < max_range)]
    return float(seg.min()) if seg.size else float("nan")


class HierRealPolicy:
    """真机 hier 仲裁。由 infer_server 在 hier 模式下逐请求调用 `act`。"""

    def __init__(self, cfg: dict | None = None, mirror: bool = True):
        c = dict(cfg or {})
        # ---- 逼近/扫描参数（E20 最优配置 = SweepDetDock(use_plan=True) 默认值）----
        self.v_cruise = float(c.get("v_cruise", 0.16))
        self.w_scan = float(c.get("w_scan", 1.15))
        self.k_ang = float(c.get("k_ang", 2.2))
        self.stop_eps = float(c.get("stop_eps", 0.14))
        self.dock_dist = float(c.get("dock_dist", 0.8))
        self.max_dw = float(c.get("max_dw", 0.55))
        self.emergency = float(c.get("emergency", 0.26))
        self.v_max = float(c.get("v_max", 0.35))
        self.w_max = float(c.get("w_max", 1.6))
        self.scan_dir = float(c.get("scan_dir", 1.0))
        self.scan_arc = float(c.get("scan_arc", 1.5 * math.pi))
        self.relocate_steps = int(c.get("relocate_steps", 25))
        self.mem_ttl = int(c.get("mem_ttl", 25))
        self.mem_merge = float(c.get("mem_merge", 0.7))
        # ---- 末段停靠（SweepDetDock）----
        self.dock_win = float(c.get("dock_win", 0.9))
        self.dock_target = float(c.get("dock_target", 0.25))
        self.dock_k = float(c.get("dock_k", 0.8))
        self.align_tol_deg = float(c.get("align_tol_deg", 6.0))
        self.align_v = float(c.get("align_v", 0.08))
        self.align_max_steps = int(c.get("align_max_steps", 12))
        self.front_half_deg = float(c.get("front_half_deg", 28.0))
        self.mirror = bool(mirror)

        from sim.baselines import LocalPlanner, parse_instruction
        self._parse = parse_instruction
        self._planner = LocalPlanner()
        self.reset()

    # ---------------------------------------------------------------- 状态
    def reset(self):
        """回合/换指令时重置（infer_server 检测到 instruction 变化时调用）。"""
        self._t = 0
        self._pw = 0.0
        self._scan_acc = 0.0
        self._relocate = 0
        self._align_n = 0
        self._mem = []            # [world_xy(2), color, cls, born_t]
        self._planner.reset()
        self.mode = "search"      # search | plan（诊断信息，随响应返回）
        self._last_instr = None

    # ---------------------------------------------------------------- 感知
    def _ang_ccw(self, bearing_deg: float) -> float:
        """backfill bearing_deg（画面右半为正）→ 机器人系逆时针角（度）。"""
        return bearing_deg if self.mirror else -bearing_deg

    def _wpos(self, pose, ang_ccw_deg: float, dist: float) -> np.ndarray:
        a = float(pose[2]) + math.radians(ang_ccw_deg)
        return np.asarray(pose[:2]) + dist * np.array([math.cos(a), math.sin(a)])

    def _pick(self, blobs, instr, pose):
        """指令 → 命中 (ang_ccw_deg, dist, wpos) 或 None（含 L3/记忆回退）。"""
        want_color, want_cls, ref_cls, rel = self._parse(instr)
        hits = [b for b in blobs if b["color"] == want_color] if want_color else []
        # L3「X 的左边/右边」：世界坐标解参考物 + 方位角（与仿真 _pick 同式）
        if rel is not None and ref_cls is not None and hits:
            refs = [b for b in blobs if b["color"] != want_color]  # 真机 cls 恒 box，按颜色区分
            tgt_ang = {"右边": 0.0, "前边": 90.0, "后边": -90.0, "左边": 180.0}[rel]
            best = None
            for t in hits:
                ta = self._ang_ccw(t["bearing_deg"])
                td = t["d_mono_m"] if np.isfinite(t["d_mono_m"]) else 12.0
                wt = self._wpos(pose, ta, td)
                for r in refs:
                    ra = self._ang_ccw(r["bearing_deg"])
                    rd = r["d_mono_m"] if np.isfinite(r["d_mono_m"]) else 12.0
                    v = wt - self._wpos(pose, ra, rd)
                    if float(np.hypot(*v)) < 0.4:
                        continue
                    a = math.degrees(math.atan2(v[1], v[0]))
                    if abs((a - tgt_ang + 180.0) % 360.0 - 180.0) < 50.0:
                        if best is None or td < best[1]:
                            best = (ta, td, wt)
                        break
            return best
        if not hits:
            return self._from_memory(want_color, want_cls, pose)
        b0 = max(hits, key=lambda z: z["n_px"])
        ang = self._ang_ccw(b0["bearing_deg"])
        d = float(b0["d_mono_m"]) if np.isfinite(b0["d_mono_m"]) else 12.0
        return (ang, d, self._wpos(pose, ang, d))

    def _from_memory(self, want_color, want_cls, pose):
        """mem_ttl 短期记忆命中（与仿真 mem_ttl=25 组件同口径）。"""
        if not want_color:
            return None
        for m in self._mem:
            if m[1] == want_color and (want_cls is None or m[2] == want_cls):
                v = m[0] - np.asarray(pose[:2])
                return (math.degrees(math.atan2(v[1], v[0]) - float(pose[2])),
                        float(np.hypot(*v)), m[0].copy())
        return None

    def _mem_update(self, blobs, pose, now):
        if self.mem_ttl <= 0:
            return
        keep = [m for m in self._mem if now - m[3] <= self.mem_ttl]
        for b in blobs:
            ang = self._ang_ccw(b["bearing_deg"])
            d = b["d_mono_m"] if np.isfinite(b["d_mono_m"]) else 12.0
            p = self._wpos(pose, ang, d)
            for m in keep:
                if m[1] == b["color"] and float(np.hypot(*(p - m[0]))) < self.mem_merge:
                    m[0] = 0.5 * (m[0] + p)          # 位置指数平滑（同仿真）
                    m[3] = now
                    break
            else:
                keep.append([p, b["color"], "box", now])
        self._mem = keep

    # ---------------------------------------------------------------- 仲裁
    def wants_search(self, blobs, instr, pose):
        """纯查询：当前帧是否未检出（含记忆）→ 搜索层。

        infer_server 用它**先判模式再决定是否跑 VLA 前向**；随后 act 内部
        会再做一次同样的 pick（纯函数，结果一致），保证 mem 只更新一次。
        """
        return self._pick(blobs, instr, np.asarray(pose, dtype=float)) is None

    def act(self, blobs, scan, pose, instr, search_action=None):
        """单步仲裁。blobs = backfill 检出列表；pose = [x,y,θ,v,ω]。

        search_action：搜索层（VLA）给出的 (v, w)。未检出且提供时采用之
        （与仿真 HierVLAPolicy「未检出 → vla.act」一致）；未提供则退化为
        内建扫描（离线冒烟用）。公共尾部（急停/限幅）对**最终动作**生效。
        返回 (v, w, mode)；模式随 HTTP 响应回传（诊断 + 车侧显示）。
        """
        scan = np.asarray(scan, dtype=float)
        n = len(scan)
        ang = np.degrees(np.linspace(-math.pi, math.pi, n, endpoint=False)) \
            if n else np.zeros(0)
        pose = np.asarray(pose, dtype=float)

        if instr != self._last_instr:                 # 换指令 → 全状态重置
            self.reset()
            self._last_instr = instr                  # 必须在 reset() 之后（reset 不清此字段）
        self._t += 1

        hit = self._pick(blobs, instr, pose)
        self._mem_update(blobs, pose, self._t)

        if hit is not None:
            self.mode = "plan"
            ang_ccw, dist, wpos = hit
            surf = surf_dist(scan, ang_ccw)
            if np.isfinite(surf):                     # 激光表面距优先（同 use_lidar_dist）
                dist = min(dist, surf) if dist < 11.0 else surf
            # 停靠对准方位：surf 即在该方位测得，激光命中点方位 ≈ 色块方位
            v, w = self._act_hit(pose, scan, ang_ccw, dist, wpos, surf)
        elif search_action is not None:
            self.mode = "search"
            v, w = float(search_action[0]), float(search_action[1])
        else:
            self.mode = "search"
            v, w = self._act_scan()

        # ---- 公共尾部（与仿真 act 尾部逐行对应）----
        if n:
            front = np.abs(ang) <= self.front_half_deg
            d_front = float(scan[front].min()) if front.any() else 12.0
            if d_front < self.emergency:
                v = 0.0
                w = self.scan_dir * self.w_max * 0.9
        w = float(np.clip(w, self._pw - self.max_dw, self._pw + self.max_dw))
        w = float(np.clip(w, -self.w_max, self.w_max))
        self._pw = w
        return float(np.clip(v, 0.0, self.v_max)), w, self.mode

    def _act_hit(self, pose, scan, ang_ccw, dist, wpos, surf):
        """检出分支：逼近执行层（对齐 SweepDetDock 两段式 + LocalPlanner）。"""
        if dist <= self.stop_eps:
            self._pw = 0.0
            return 0.0, 0.0
        # 末段（激光表面距进入 dock_win）：先对准再停靠（同 SweepDetDock.act）
        if np.isfinite(surf) and surf <= self.dock_win:
            if abs(ang_ccw) > self.align_tol_deg and self._align_n < self.align_max_steps:
                self._align_n += 1
                return self.align_v, float(np.clip(self.k_ang * math.radians(ang_ccw),
                                                   -self.w_max, self.w_max))
            self._align_n = 0
            v = float(np.clip((surf - self.dock_target) * self.dock_k,
                              0.0, self.v_max))
            return v, float(np.clip(self.k_ang * math.radians(ang_ccw),
                                    -self.w_max, self.w_max))
        # 中段：局部绕障规划给航向（弧度 CCW）；规划失败回退直冲（同 _act_hit）
        hd = self._planner.step(pose, scan, np.asarray(wpos, dtype=float))
        bearing = hd if hd is not None else math.radians(ang_ccw)
        w = float(np.clip(self.k_ang * bearing, -self.w_max, self.w_max))
        v = self.v_max * min(1.0, max(0.0, (dist - self.stop_eps) / self.dock_dist))
        return v, w

    def _act_scan(self):
        """未检出分支：扫描 / 换观测点（同仿真 _act_scan）。"""
        if self._relocate <= 0 and self._scan_acc >= self.scan_arc:
            self._relocate = self.relocate_steps
            self._scan_acc = 0.0
        if self._relocate > 0:
            self._relocate -= 1
            w = 0.35 * self.scan_dir
            v = self.v_cruise * 1.6
        else:
            w = self.scan_dir * self.w_scan
            v = self.v_cruise
            self._scan_acc += abs(w) * 0.1
        return v, w
