"""语义导航环境：差速底盘 + L1/L2/L3 语言指令 + 导航指标。

动作空间：**2 维连续 (v, ω)**，与真机小车一致（无云台）。
观测：第一人称 RGB + 2D 激光雷达 + 本体状态 + 语言指令。

指令分级（对应 E2 的实验设计）：
    L1 类别     「去椅子上」
    L2 属性     「去红色的椅子上」
    L3 空间关系 「去桌子左边的沙发上」

指标：SR / SPL / NE / TL（定义见 metrics）。
"""

from __future__ import annotations

import numpy as np

from .sensors import (CAMERA_HFOV_DEG, LIDAR_MAX_RANGE, lidar_scan,
                      photometric_augment, render)
from .world import CELL, CLASS_CN, COLOR_NAMES, TARGET_CLASSES, make_pedestrians, make_scene

# 与真机对齐的运动学限制（TT 减速电机小型差速底盘）
V_MAX = 0.35      # m/s
W_MAX = 1.60      # rad/s
ROBOT_RADIUS = 0.18

# ---------------------------------------------------------------------------
# **标定后的基准配置**（见 `主创新A_诊断报告.md` §4.5）
#
# 关键点是 `success_standoff`：成功判据用「到目标**包围盒表面**的距离」而不是
# 「到中心点的距离」。目标中心在实心物体内部，后者在几何上不可达
# （实测示范策略最接近距离最小 0.43 m），是此前测试床没有区分度的根因。
#
# 实测语义区分度（40 回合）：oracle 67.5% / sweep 60.0% / random_target 17.5%
#                           / random 10.0%  →  差距 42.5–50 pp
# ---------------------------------------------------------------------------
CALIBRATED = dict(nx=1, ny=1, room_w=8.0, room_h=6.0, n_objects_per_room=(6, 9),
                  n_peds=0, max_steps=200, success_standoff=0.30,
                  lidar_noise=0.01, camera_augment=0.25)


def calibrated_env(seed=0, **over):
    """按标定配置建环境；`over` 可覆盖任意项。"""
    kw = dict(CALIBRATED)
    kw.update(over)
    return SemanticNavEnv(seed=seed, **kw)


class SemanticNavEnv:
    def __init__(self, seed=0, nx=3, ny=2, n_peds=2, max_steps=300, dt=0.10,
                 obs_size=(160, 120), lidar_beams=120,
                 v_max=V_MAX, w_max=W_MAX, success_radius=0.80,
                 lidar_noise=0.0, camera_augment=0.0, frame_stack=1,
                 reward_shaping=1.0, collide_penalty=0.05,
                 task_level=None, goal_image=False, n_instr_per_episode=1,
                 room_w=3.6, room_h=3.4, n_objects_per_room=(1, 3),
                 success_standoff=0.30, layout=None):
        self.seed = seed
        self.cfg = dict(nx=nx, ny=ny, n_peds=n_peds, max_steps=max_steps, dt=dt,
                        obs_size=obs_size, lidar_beams=lidar_beams,
                        v_max=v_max, w_max=w_max, success_radius=success_radius,
                        lidar_noise=lidar_noise, camera_augment=camera_augment,
                        frame_stack=frame_stack, reward_shaping=reward_shaping,
                        collide_penalty=collide_penalty,
                        task_level=task_level, goal_image=goal_image,
                        n_instr_per_episode=n_instr_per_episode,
                        room_w=room_w, room_h=room_h,
                        n_objects_per_room=n_objects_per_room,
                        success_standoff=success_standoff, layout=layout)
        self.rng = np.random.default_rng(seed)
        self._ep = 0
        self.scene = None
        self.peds = []
        self.reset(seed)

    # ------------------------------------------------------------ 场景
    def _new_scene(self, seed):
        # **必须按 seed 重新播种**：`self.rng` 同时被任务采样（`_sample_task`）与
        # 观测噪声（相机光度增强、激光噪声，都在 `_obs` 里）使用。若只在
        # `__init__` 播一次种，回合内容就会**依赖策略历史** —— 因为不同策略
        # 每步消耗的随机数个数不同（例如停住的策略不渲染噪声）。
        # 后果极严重：两个策略即使传相同 seed，实际跑的是**不同任务**，
        # 配对比较根本不成立（实测两个模型 100 回合只有 34 个种子重合）。
        self.rng = np.random.default_rng(int(seed))
        self.scene = make_scene(seed=int(seed), nx=self.cfg["nx"], ny=self.cfg["ny"],
                                room_w=self.cfg["room_w"], room_h=self.cfg["room_h"],
                                n_objects_per_room=tuple(self.cfg["n_objects_per_room"]),
                                layout=self.cfg.get("layout"))
        self.peds = make_pedestrians(self.scene, self.cfg["n_peds"], seed=int(seed))
        # 规划用栅格：把障碍按机器人半径膨胀，使测地路径天然保持安全间距。
        # 若直接用原栅格，最优路径会贴着墙/家具，任何控制器都会频繁碰撞
        # （实测 Oracle 单回合碰撞达 200+ 次）。
        k = int(np.ceil(ROBOT_RADIUS * 0.8 / CELL))
        self.plan_grid = _dilate(self.scene.occ, k)
        self._dist_cache = {}

    def _shortest_path_len(self, goal_xy):
        """可通行栅格上的测地距离场（米），带缓存。

        注意：目标点通常位于**物体包围盒内部**（不可通行），若直接从目标格起做 BFS
        会困在物体内部、整个距离场不可达。因此改从「离目标最近的自由格」起算，
        并把该自由格到目标中心的欧氏距离计入路径长度。
        """
        key = tuple(np.round(np.asarray(goal_xy) / CELL).astype(int))
        if key in self._dist_cache:
            return self._dist_cache[key]
        from collections import deque
        sc = self.scene
        H, W = sc.occ.shape
        free = ~self.plan_grid          # 用膨胀后的规划栅格
        gr, gc = sc.w2c(goal_xy)
        gr = int(np.clip(gr, 0, H - 1)); gc = int(np.clip(gc, 0, W - 1))

        if free[gr, gc]:
            seed_r, seed_c, off = gr, gc, 0.0
        else:
            rr, cc = np.nonzero(free)
            if rr.size == 0:
                self._dist_cache[key] = np.full((H, W), -1.0, dtype=np.float32)
                return self._dist_cache[key]
            k = int(np.argmin((rr - gr) ** 2 + (cc - gc) ** 2))
            seed_r, seed_c = int(rr[k]), int(cc[k])
            off = float(np.hypot(seed_r - gr, seed_c - gc)) * CELL

        dist = np.full((H, W), -1.0, dtype=np.float32)
        dist[seed_r, seed_c] = off
        q = deque([(seed_r, seed_c)])
        while q:
            r, c = q.popleft()
            for dr, dc, step in ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                                 (1, 1, 1.41421), (1, -1, 1.41421),
                                 (-1, 1, 1.41421), (-1, -1, 1.41421)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and free[nr, nc] and dist[nr, nc] < 0:
                    dist[nr, nc] = dist[r, c] + step * CELL
                    q.append((nr, nc))
        self._dist_cache[key] = dist
        return dist

    # ------------------------------------------------------------ 任务
    @staticmethod
    def _rel_word(ang):
        """方位词按**世界坐标系**定义：+x = 右，+y = 前。

        简化说明：本测试床家具为轴对齐包围盒、无自身朝向，故用全局坐标系而非
        「参考物自身朝向」定义方位（见 E1 文档 §9 已知限制 3）。
        """
        if -np.pi / 4 <= ang < np.pi / 4:
            return "右边"
        if np.pi / 4 <= ang < 3 * np.pi / 4:
            return "前边"
        if -3 * np.pi / 4 <= ang < -np.pi / 4:
            return "后边"
        return "左边"

    def _all_phrasings(self, tgt):
        """列出所有**合法且无歧义**地指代该目标的指令表述。

        返回 `[(level, text, ref_cls), ...]`。用途有二：
        ① 采样其中一条作为当前任务指令；
        ② 同一目标生成多种表述 → 语言接地的数据增强（见 `gen_dataset.py`）。

        无歧义约束：
        - L1 仅给类别 → 要求该类别在场景中**唯一**
        - L2 给颜色+类别 → 要求该「颜色+类别」组合**唯一**
        - L3 给参考物+方位 → 要求参考物类别**唯一**（否则「桌子左边的…」本身有歧义）
        """
        objs = self.scene.objects
        same = [o for o in objs if o.cls == tgt.cls]
        cname = CLASS_CN[tgt.cls]
        out = []

        if len(same) == 1:
            out.append(("L1", f"去{cname}旁边", None))
        if sum(1 for o in same if o.color == tgt.color) == 1:
            out.append(("L2", f"去{tgt.color}的{cname}旁边", None))

        for ref in objs:
            if ref is tgt or ref.cls == tgt.cls:
                continue
            if sum(1 for o in objs if o.cls == ref.cls) > 1:
                continue
            d = tgt.xy - ref.xy
            if np.hypot(*d) < 0.6:
                continue
            rel = self._rel_word(np.arctan2(d[1], d[0]))
            out.append(("L3", f"去{CLASS_CN[ref.cls]}{rel}的{cname}旁边", ref.cls))
        return out

    def _sample_task(self, level=None):
        """采样目标物体与对应分级指令；同时附带该目标的全部合法表述。"""
        if not self.scene.objects:
            return None
        lv = level if level is not None else self.cfg.get("task_level")
        if lv is None:
            lv = str(self.rng.choice(["L1", "L2", "L3"]))

        for oi in self.rng.permutation(len(self.scene.objects)):
            tgt = self.scene.objects[int(oi)]
            phrasings = self._all_phrasings(tgt)
            cand = [p for p in phrasings if p[0] == lv]
            if not cand:
                continue
            lvl, text, ref_cls = cand[0]
            return {"level": lvl, "goal": tgt.xy.copy(), "instruction": text,
                    "target_cls": tgt.cls, "ref_cls": ref_cls,
                    "phrasings": phrasings, "obj_index": int(oi)}
        return None

    def cycle_instruction(self):
        """在**不改变目标**的前提下换一种指令表述（语言接地数据增强）。

        返回新指令；若无其他合法表述则返回原指令。
        """
        phr = self.task.get("phrasings") or []
        if len(phr) <= 1:
            return self.task["instruction"]
        cur = self.task["instruction"]
        others = [p for p in phr if p[1] != cur]
        if not others:
            return cur
        lvl, text, ref_cls = others[int(self.rng.integers(len(others)))]
        self.task.update({"level": lvl, "instruction": text, "ref_cls": ref_cls})
        return text

    def goal_image(self):
        """E3 图像目标导航用：从目标正前方渲染一张「目标图像」。

        在目标周围按不同距离/方位寻找可见的自由视点，取第一个可用的。
        """
        g = np.asarray(self.task["goal"], dtype=np.float64)
        w, h = self.cfg["obs_size"]
        for r in (1.2, 1.6, 0.9, 2.0):
            for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                p = g - np.array([np.cos(a), np.sin(a)]) * r
                if not self.scene.is_free(p, margin=0.22):
                    continue
                th = float(np.arctan2(g[1] - p[1], g[0] - p[0]))
                return render(self.scene, np.array([p[0], p[1], th]),
                              self.peds, width=w, height=h)
        return None

    # ------------------------------------------------------------ 观测
    def _obs(self):
        cfg = self.cfg
        pose = self.pose
        img = render(self.scene, pose, self.peds, width=cfg["obs_size"][0],
                     height=cfg["obs_size"][1])
        if cfg["camera_augment"] > 0:
            img = photometric_augment(img, self.rng, strength=cfg["camera_augment"])
        lid = lidar_scan(self.scene, pose, n_beams=cfg["lidar_beams"],
                         noise_std=cfg["lidar_noise"], rng=self.rng)
        state = np.concatenate([pose, self.last_action]).astype(np.float32)
        out = {"image": img, "lidar": lid.astype(np.float32), "state": state,
               "instruction": self.task["instruction"], "task_level": self.task["level"]}
        if cfg["goal_image"]:                      # E3：目标图像（回合内不变，只渲染一次）
            if getattr(self, "_gimg", None) is None:
                self._gimg = self.goal_image()
            out["goal_image"] = self._gimg
        return out

    # ------------------------------------------------------------ API
    def reset(self, seed=None, level=None):
        cfg = self.cfg
        if seed is None:
            seed = self.seed + self._ep
        self._ep += 1
        self._new_scene(seed)
        self.pose = np.array([self.scene.spawn[0], self.scene.spawn[1],
                              self.scene.spawn_theta], dtype=np.float64)
        # 采样任务。严格化的无歧义规则（L1 要求类别唯一、L2 要求颜色+类别唯一、
        # L3 要求参考物类别唯一）会使**部分场景没有任何合法表述**，此时必须换场景重采，
        # 否则 self.task 为 None 会在后面直接崩溃（实测导致整轮采集白跑）。
        self.task = self._sample_task(level)
        tries = 0
        while self.task is None and tries < 10:
            tries += 1
            self._new_scene(seed + 10000 * tries)
            self.pose = np.array([self.scene.spawn[0], self.scene.spawn[1],
                                  self.scene.spawn_theta], dtype=np.float64)
            self.task = self._sample_task(level)
            if self.task is None:
                self.task = self._sample_task(None)      # 放宽到任意级别
        if self.task is None:
            raise RuntimeError(f"场景 seed={seed} 重采 {tries} 次仍无合法指令（场景过小？）")
        self.last_action = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.path_len = 0.0
        self.prev_dist = float(np.linalg.norm(self.task["goal"] - self.pose[:2]))
        self.d0 = self.prev_dist
        self.min_dist = self.prev_dist
        self.d_surf = float("nan")
        self.n_collide = 0
        self.traj = [self.pose[:2].copy()]
        self.done = False
        self.start_xy = self.pose[:2].copy()
        self._gimg = None
        # SPL 的最优路径长度：起点 → 目标的测地距离（episode 内固定）
        self.lstar = grid_value(self._shortest_path_len(self.task["goal"]),
                                self.scene, self.start_xy)
        return self._obs()

    def step(self, action):
        assert not self.done, "episode 已结束，请先 reset()"
        cfg = self.cfg
        dt = cfg["dt"]
        a = np.clip(np.asarray(action, dtype=np.float64),
                    [-cfg["v_max"], -cfg["w_max"]], [cfg["v_max"], cfg["w_max"]])
        self.last_action = a.astype(np.float32)

        v, w = float(a[0]), float(a[1])
        th = self.pose[2]
        nx = self.pose[0] + v * np.cos(th) * dt
        ny = self.pose[1] + v * np.sin(th) * dt
        nth = th + w * dt
        nth = (nth + np.pi) % (2 * np.pi) - np.pi

        # 碰撞：不可通行则原地停留
        moved = np.hypot(nx - self.pose[0], ny - self.pose[1])
        if self.scene.is_free((nx, ny), margin=ROBOT_RADIUS * 0.6):
            self.pose = np.array([nx, ny, nth])
            self.path_len += moved
        else:
            self.pose = np.array([self.pose[0], self.pose[1], nth])
            self.n_collide += 1

        for p in self.peds:
            p.step(dt)

        self.steps += 1
        self.traj.append(self.pose[:2].copy())
        d = float(np.linalg.norm(self.task["goal"] - self.pose[:2]))
        self.min_dist = min(self.min_dist, d)

        # **成功判据用「到目标物体包围盒表面的距离」而不是「到中心点的距离」**。
        # 目标中心在物体内部（物体是实心方盒），任何策略都不可能靠到中心 0.40 m 以内
        # ——实测示范策略的最接近距离最小 0.43 m、中位 0.88 m，也就是说旧判据
        # `success_radius` 在几何上根本不可达，标定因此一直失败。
        # 而「到物体旁边」的物理含义本来就是贴着它，所以用表面距离才正确。
        o = self.scene.objects[self.task["obj_index"]]
        rel = np.abs(self.pose[:2] - o.xy) - o.half
        self.d_surf = float(np.linalg.norm(np.maximum(rel, 0.0)))

        success = self.d_surf <= cfg["success_standoff"]
        timeout = self.steps >= cfg["max_steps"]
        self.done = bool(success or timeout)

        r = -0.01
        if cfg["reward_shaping"]:
            r += cfg["reward_shaping"] * (self.prev_dist - d)   # 前进奖励
        r -= cfg["collide_penalty"] * (1 if moved == 0 else 0)
        if success:
            r += 10.0
        self.prev_dist = d

        info = {
            "success": bool(success),
            "timeout": bool(timeout),
            "dist": d,                       # 到目标**中心**距离（规划用）
            "dist_surf": self.d_surf,        # 到目标**包围盒表面**距离（成功判据）
            "min_dist": self.min_dist,
            "path_len": self.path_len,
            "shortest_len": float(self.lstar),
            "n_collide": self.n_collide,
            "level": self.task["level"],
            "instruction": self.task["instruction"],
        }
        return self._obs(), float(r), self.done, info


def _dilate(occ: np.ndarray, k: int) -> np.ndarray:
    """把障碍膨胀 k 格，得到考虑机器人半径的规划栅格。

    采用**圆盘**而非切比雪夫（方盘）膨胀：方盘在对角方向会多膨胀约 1.41 倍，
    实测会把 0.9 m 宽的门洞压缩到接近不可通行、并切断自由空间连通性。
    """
    if k <= 0:
        return occ.copy()
    out = occ.copy()
    H, W = occ.shape
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            if dr * dr + dc * dc > k * k + k:      # 圆盘近似
                continue
            r0, r1 = max(0, dr), H + min(0, dr)
            c0, c1 = max(0, dc), W + min(0, dc)
            out[r0:r1, c0:c1] |= occ[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return out


def grid_value(dist_grid, scene, xy):
    """从测地距离图上取某点的路径长度（米）；不可达返回 nan。"""
    r, c = scene.w2c(xy)
    r = int(np.clip(r, 0, scene.H - 1))
    c = int(np.clip(c, 0, scene.W - 1))
    v = float(dist_grid[r, c])
    return v if v >= 0 else float("nan")


def navigate_metrics(info, success_radius=0.80):
    """由 info 计算单回合 SR / SPL / NE / TL 分量。

    SPL = S * L* / max(L, L*)；TL = L / L*
    """
    s = 1.0 if info["success"] else 0.0
    L, Ls = float(info["path_len"]), float(info["shortest_len"])
    spl = s * Ls / max(L, Ls) if (Ls == Ls and max(L, Ls) > 0) else 0.0
    tl = L / Ls if (Ls == Ls and Ls > 0) else float("nan")
    return {"SR": s, "SPL": spl, "NE": float(info["dist"]), "TL": tl}
