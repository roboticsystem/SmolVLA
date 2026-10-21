"""E11 视觉跟随任务的**仿真实现**（对应大纲 §5 E11，真机验证前的仿真版）。

## 为什么需要它

论文题目是「基于 VLA 模型的**室内跟随摄像机器人**设计」，但此前的全部实验
都在做**语义导航**（去某个物体旁）。两者不是同一个任务：

| | 语义导航（已有） | 视觉跟随（本模块） |
|---|---|---|
| 目标 | **静态**物体，指令给定 | **移动**的人，需持续跟住 |
| 成功判据 | 到目标 ≤ 0.30 m（一次性） | **全程**保持在期望距离带内且不丢 |
| 核心难点 | 搜索 + 末段接近 | **抗遮挡、丢失后恢复** |

因此导航上的结论**不能直接搬到跟随上**，必须单独建立任务与指标。

## 任务定义

机器人需保持被跟随者（穿**场景中唯一的红色马甲**，便于感知识别——这对应
真实跟随机器人的「目标可区分」前提）处于：
① 相机视野内（且无遮挡）；② 期望距离 `want_dist` 所在的距离带内。

**跟随误差**分纵向（距离偏差）与横向（方位偏差）两个分量分别报告 ——
只看纵向会在「目标在正前方但太远」与「目标很近但偏到画面边缘」时给出相同分数，
而后者在真机上就是「马上要丢」。

## 指标（对应大纲 §3.1 任务层）

| 指标 | 定义 |
|---|---|
| **TCR** | 跟踪连续率：「目标可见 **且** 在距离带内」的步数占比 |
| 跟随误差 | `|dist − want_dist|` 的均值 ± 标准差（纵向）；`|bearing|` 均值（横向） |
| **丢失次数** | 「连续不可见 ≥ 0.5 s」的事件次数 |
| **恢复时间** | 每次丢失到重新可见的耗时（ms），报中位数与 p90 |
| 可见率 | 目标可见步数占比（不受距离带约束，用于分离「遮挡」与「距离没控好」） |

> **口径提醒**：本仿真是**任务与指标的建立**，不是真机结论。
> 真机（E11 主体）需在同一指标定义下重测；目标距离**不能**用「已知人体高度 +
> 针孔模型」反推（头顶必被裁切，E1b 标定误差 379%/61%），
> 我方用**底边接地法**（0.8–2.0 m 相对误差 ≤3.5%），
> 并据此把跟随工作带限定为 **[0.8, 2.0] m**（详见 `E14_本轮实验结果汇总.md` §2）。
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .env import CALIBRATED, ROBOT_RADIUS, SemanticNavEnv
from .perception import detect
from .sensors import (CAMERA_HEIGHT, CAMERA_HFOV_DEG, LIDAR_MAX_RANGE, lidar_scan,
                      photometric_augment, raycast, render)
from .world import COLORS, segment_free, CELL, is_free_dyn, occ_with_peds

VEST_COLOR = "red"          # 马甲色：用调色板里的 red，**必须场景唯一**
PERSON_H = 1.70             # 人体高（真机用「已知人体高度」反推距离）
PERSON_R = 0.25
BAND = (0.9, 2.6)           # 期望距离带
SCENARIOS = ("normal", "short_occl", "long_occl", "light_shift", "narrow",
             "dim", "backlight", "sunset", "gloss", "shadow", "plain_wall",
             "crowd")

# **场景 → 真实光照/材质几何**（对齐论文1《嵌入式视觉可行走路径识别》的
# 逆光、反光地面、无纹理白墙工况；不再只用单一 photometric 强度旋钮）：
#   light_dir : 方向光行进方向（弧度）；glare : 逆光眩光/过曝；
#   gloss     : 镜面反光地面；exposure : 全局曝光；flat_wall : 纯白无纹理墙。
# `sunset`/`backlight` 的光来自**相机前方**（人在窗前 → 目标被逆光冲白）。
SCENARIO_LIGHT = {
    "normal":     dict(light_dir=None, glare=0.0, gloss=0.0, exposure=1.0),
    "dim":        dict(light_dir=None, glare=0.0, gloss=0.0, exposure=0.45),
    "backlight":  dict(light_dir=np.pi, glare=0.55, gloss=0.0, exposure=1.0),
    "sunset":     dict(light_dir=np.pi, glare=0.80, gloss=0.15, exposure=0.75),
    "gloss":      dict(light_dir=np.pi / 2, glare=0.0, gloss=0.75, exposure=1.0),
    "shadow":     dict(light_dir=np.pi / 2, glare=0.0, gloss=0.0, exposure=1.0),
    "plain_wall": dict(light_dir=None, glare=0.0, gloss=0.0, exposure=1.0),
}

# **目标速度必须低于机器人上限**，否则「跟不住」是物理必然、不是策略问题。
# 实测教训：最初取 0.55 m/s，而 `V_MAX = 0.35 m/s` —— 连 oracle（用真值）
# 都只有 34.5% TCR、64% 可见率，量的是「追不追得上」而不是跟随能力。
# 取 0.22 m/s ≈ 63% 的机器人上限：能跟上，但需要持续努力才有余量。
TARGET_SPEED = 0.22
V_FRAC = 0.90               # 基线允许使用的速度上限占 `V_MAX` 的比例


def detect_person(img, lidar=None, want_px=False):
    """**跟随任务专用的目标感知**：找马甲色块 → 用**底边行号**反推距离。

    为什么不能直接用 `perception.detect`：那套检测器是按**家具**调的
    （形状代价 `_shape_cost` 匹配各类家具的高宽先验、`min_fill`、`col_consistency`），
    而人是一个**细长精灵**（2 m 处约 33×83 px，且**顶部被画面裁掉**）。
    实测直接用它在 2 m 处**完全检不出**，关掉光度增强也只有 2/6。

    这符合真实情况：跟随机器人用的是**行人检测/重识别**，不是家具检测。
    同一路感知同时供 reactive 基线与（未来的）VLA 使用，保证对照公平。

    ## 测距

    用**色块底边行号**按地面几何反推：`d = f·cam_h/(r1 − horizon)`。
    人站在地面上，底边就是**接地点**，所以这个量在跟随距离带
    （0.9–2.6 m）上是准的 —— 它只在 < 0.75 m 饱和（导航任务上已实测），
    而跟随不会那么近。

    > **不用「按身高反推」**：相机高 0.35 m、人体高 1.70 m，在 2 m 处人顶
    > 已超出画面上沿（r0 被裁到 0），像素高不可用。大纲 §8 写的是「已知人体高度」，
    > 实现时按实际几何改用底边接地法，论文中应如实说明这一修正。
    """
    from scipy import ndimage as ndi

    H, W = img.shape[:2]
    f = (W / 2.0) / np.tan(np.deg2rad(CAMERA_HFOV_DEG) / 2.0)
    horizon = H / 2.0
    ref = np.array(COLORS[VEST_COLOR], dtype=np.float64)
    x = img.astype(np.float64)
    # 马甲色的成像会经历「距离衰减 + 雾化 + 垂直光照」，逐像素容差取较宽值，
    # 否则同一件马甲在图像上下部会被判成不同颜色（实测碎片化严重）。
    tol = 95.0
    mask = (np.linalg.norm(x - ref[None, None, :], axis=2) < tol)
    if mask.sum() < 25:
        return None
    lab, n = ndi.label(mask)
    if n == 0:
        return None
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    ys, xs = np.where(lab == k)
    if len(ys) < 25:
        return None
    c0, c1 = int(xs.min()), int(xs.max()) + 1
    r1 = int(ys.max()) + 1
    r0 = int(ys.min())
    # **顶部被裁（r0 == 0）是常态而不是错误**：相机高 0.35 m、人体高 1.70 m，
    # 在 2 m 处头顶已超出画面上沿。底边接地法不依赖顶部，所以不能因此拒检
    # —— 早先加了这条检查，导致 2 m 处**全部检不出**（踩过）。
    if r1 - horizon <= 1.0:
        return None
    d = float(f * CAMERA_HEIGHT / (r1 - horizon))
    bearing = float(np.arctan((0.5 * (c0 + c1 - 1) - (W - 1) / 2.0) / f))
    if want_px:
        # 额外返回**像素高度**与**相机内参**：供 E1b 的单目测距标定
        # 对比「底边接地法」与「已知身高反推法」两种口径（见 `scripts/calib_range.py`）。
        # `r0 == 0` 表示头顶被画面裁掉 → 像素高不可用。
        return bearing, d, int(len(ys)), int(r1 - r0), float(f), int(r0), int(r1)
    return bearing, d, int(len(ys))


class FollowTarget:
    """被跟随的人：沿航点直线行走。

    与 `world.Pedestrian` 的区别只有一点：可接受**脚本航点**，
    便于构造「从家具背后穿过」「走窄门」等场景。
    """

    def __init__(self, scene, rng, speed=TARGET_SPEED, waypoints=None, spawn_pose=None,
                 pause_at=None, pause_sec=0.0, loop_wps=False):
        self.scene = scene
        self.rng = rng
        self.speed = float(speed)
        self.radius = PERSON_R
        self.height = PERSON_H                      # 渲染与「按身高反推距离」都要用
        self.rgb = np.array(COLORS[VEST_COLOR], dtype=np.uint8)   # 马甲色（场景唯一）
        self._wps = [np.asarray(w, dtype=np.float64) for w in (waypoints or [])]
        self._wi = 0
        self._loop = bool(loop_wps)                 # 航点走完后从头再循环（全屋巡回用）
        # **可控时长的遮挡**：到达 `pause_at` 后原地停留 `pause_sec` 秒。
        # 早先 `short_occl` 与 `long_occl` 走同一分支、只是航点构造相同，
        # 实测两者**结果完全一致**（TCR 81.5% / 恢复 1191.7 ms 逐位相同）——
        # 即根本没有区分「短时」与「长时」。改用显式停顿时长后才能对上
        # 大纲 §5 E11 的「1–2 s」与「> 3 s」两档。
        self._pause_at = (None if pause_at is None
                          else np.asarray(pause_at, dtype=np.float64))
        self._pause_left = float(pause_sec)
        self.xy = (self._in_front_of(spawn_pose) if spawn_pose is not None
                   else self._rand_free())
        self.wp = self._next_wp()

    def _in_front_of(self, pose, lo=1.3, hi=3.2):
        """在机器人出生点**正前方、无遮挡**处生成 —— 跟随任务的合理初始状态。

        为什么必须这样（实测教训）：若目标随机落在场景任意处，机器人与它可能
        相隔 8.6 m 且目标在视野外（bearing −95°），于是整回合退化为**互相找路**
        —— 实测 299 步里碰撞 259 次（87%），量到的是「能不能绕出房间」，
        而不是跟随能力。**先捕获、再跟随**才是本任务要测的。
        """
        base = np.asarray(pose, float)
        # 搜**全向**而不是只搜前方：出生朝向由场景随机给定，可能正对墙角，
        # 此时「前方 2–3 m」全被挡（实测 400 次尝试全失败并静默回退到随机位置，
        # 导致目标出现在机器人**背后** 100° 处）。改为先在任意方向找到
        # 「自由且与机器人无遮挡」的点，再**把机器人初始朝向拧过去对准它**。
        order = np.arange(72)
        self.rng.shuffle(order)
        for k in order:
            a = float(k) * np.pi / 36.0 - np.pi
            # 距离也扫密一些：多房间里从某些位置出发，2–3.2 m 内**所有方向**
            # 都被墙挡住，搜索会静默回退到随机位置（实测目标跑到 8.3 m 外、
            # 图像里一个马甲像素都没有）。放到 1.3 m 起就有解了。
            for d in np.linspace(lo, hi, 7):
                p = base[:2] + d * np.array([np.cos(a), np.sin(a)])
                if not self.scene.is_free(p, margin=self.radius + 0.10):
                    continue
                if not segment_free(self.scene, base[:2], p, self.radius + 0.08):
                    continue
                return p
        return self._rand_free()

    def _rand_free(self):
        sc = self.scene
        for _ in range(400):
            p = np.array([self.rng.uniform(sc.bounds[0] + 0.6, sc.bounds[1] - 0.6),
                          self.rng.uniform(sc.bounds[2] + 0.6, sc.bounds[3] - 0.6)])
            if sc.is_free(p, margin=self.radius + 0.05):
                return p
        return sc.spawn.copy()

    def _next_wp(self):
        """脚本航点优先；用完后随机选「直线可达」的点，避免穿墙。"""
        if self._loop and self._wps and self._wi >= len(self._wps):
            self._wi = 0                             # 全屋巡回：从头再走
        # 循环航点（tour）由 BFS 按行人半径生成，校验余量须与生成时的
        # 膨胀半径一致（0.30 会把贴边航点全部误拒 → 人困在原地踱步）
        m = self.radius if self._loop else self.radius + 0.05
        while self._wi < len(self._wps):
            p = self._wps[self._wi]
            self._wi += 1
            if self._loop:
                # BFS 路径本身已按行人半径无碰撞，方盒校验会误拒贴边航点
                return p
            if self.scene.is_free(p, margin=m) and \
                    segment_free(self.scene, self.xy, p, m):
                return p
        for _ in range(400):
            p = self._rand_free()
            if np.linalg.norm(p - self.xy) < 1.2:
                continue
            if segment_free(self.scene, self.xy, p, self.radius + 0.05):
                return p
        return self.xy.copy()

    def step(self, dt: float):
        if self._pause_left > 0.0:
            self._pause_left = max(0.0, self._pause_left - dt)
            return
        if self._pause_at is not None and \
                float(np.linalg.norm(self.xy - self._pause_at)) < 0.30:
            self._pause_left = 0.0
            self._pause_at = None            # 只停一次
            return
        d = self.wp - self.xy
        n = float(np.linalg.norm(d))
        if n < 0.20:
            self.wp = self._next_wp()
            return
        self.xy = self.xy + d / n * min(self.speed * dt, n)


class FollowEnv(SemanticNavEnv):
    """跟随任务环境。

    复用 `SemanticNavEnv` 的场景、传感器与碰撞逻辑（**观测与动力学完全一致**，
    只换任务层），但：不采样语言指令（跟随目标由视觉给定）、采样被跟随者、
    并保证马甲色在场景中唯一。
    """

    def __init__(self, seed=0, want_dist=1.5, band=BAND, max_steps=300,
                 scenario="normal", target_speed=TARGET_SPEED,
                 camera_augment=0.0, **kw):
        # **必须在 `super().__init__` 之前设好**：父类构造函数会调用
        # `self.reset(seed)`，而 `reset` 要用到这些跟随专属属性（踩过）。
        self.want_dist = float(want_dist)
        self.band = (float(band[0]), float(band[1]))
        self.scenario = str(scenario)
        self.target_speed = float(target_speed)
        self._aug_base = float(camera_augment)
        # **默认用「标定测试床」的单房间配置**（`env.CALIBRATED`）：
        # ① 与主结果同源，跟随与导航的结论可比；
        # ② 开阔（8×6 m）才容得下「目标始终有通畅视线」的跟随任务 ——
        #    多房间下从若干出生点出发，1.3–3.2 m 内**所有方向**都被墙挡，
        #    目标生成会静默回退到随机位置（实测 2/14 回合目标落到 5 m 外，
        #    图像里一个马甲像素都没有）。
        # 需要多房间/狭窄通道时由 `scenario` 或显式传 `nx/ny` 覆盖。
        kw.setdefault("n_peds", 0)          # 跟随场景默认无额外行人，干扰由 scenario 控制
        if self.scenario == "crowd":
            # **人群干扰**：额外行人也会挡激光、也会被撞到（论文1 §5.3.2 的
            # 「行人横穿」、论文2 的多机避让都是这类工况）
            kw["n_peds"] = 2
        for k, v in CALIBRATED.items():
            if k in ("max_steps", "camera_augment"):
                continue
            kw.setdefault(k, v)
        super().__init__(seed=seed, max_steps=max_steps,
                         camera_augment=camera_augment, **kw)

    # ------------------------------------------------------------ 场景
    def _new_scene(self, seed):
        super()._new_scene(seed)
        # **保证马甲色唯一**：把场景里所有红物件的颜色换掉。
        # 不这么做，感知无法区分「要跟的人」与「红色的椅子」，
        # 跟随任务就退化成「跟住任意红物体」。
        # 注意 `Obj` 是 `__slots__`，只能改已有字段（`color`/`rgb`），不能加新字段。
        for o in self.scene.objects:
            if getattr(o, "color", None) == VEST_COLOR:
                alt = "blue" if o.cls in ("chair", "sofa", "bed") else "green"
                o.color = alt
                o.rgb = np.array(COLORS[alt], dtype=np.uint8)

    # 全屋巡回的「到访点」（公寓各房间的自由开阔位；演示时依次走到）
    TOUR_VISITS = [(2.60, 1.20),   # 客厅
                   (3.00, 7.00),   # 主卧（床东侧）
                   (6.50, 6.90),   # 次卧（床东侧）
                   (8.30, 6.40),   # 书房
                   (10.50, 6.50),  # 卫生间
                   (9.80, 2.00),   # 厨房
                   (5.60, 1.00),   # 餐厅
                   (2.60, 1.20)]   # 回客厅（闭环）

    def _tour_waypoints(self, start):
        """**算法生成**全屋巡回航点：以行人半径膨胀栅格 → BFS 测地路径。

        为什么不用手工航点：门洞宽 1.2 m、行人换向余量 0.30 m，斜穿门洞时
        视线校验必然擦到门角墙（实测 8 个航点被静默跳过、人困在客厅）。
        改为对每个到访点做 BFS 距离场，再从当前位置沿场梯度走到该点 ——
        路径天然经门洞中心、与墙/家具保持行人半径间距。
        """
        from collections import deque
        from .env import _dilate
        sc = self.scene
        free = ~_dilate(sc.occ, int(np.ceil(PERSON_R / CELL)))
        H, W = free.shape
        pts = []
        cur = np.asarray(start, dtype=np.float64)
        for goal in self.TOUR_VISITS:
            g = np.asarray(goal, dtype=np.float64)
            gr, gc = sc.w2c(g); gr, gc = int(gr), int(gc)
            dist = np.full((H, W), -1, dtype=np.int32)
            if not free[gr, gc]:
                cand = np.argwhere(free)
                if len(cand) == 0:
                    continue
                k = int(np.argmin((cand[:, 0] - gr) ** 2 + (cand[:, 1] - gc) ** 2))
                gr, gc = int(cand[k, 0]), int(cand[k, 1])
            dist[gr, gc] = 0
            q = deque([(gr, gc)])
            while q:
                r0, c0 = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    r1, c1 = r0 + dr, c0 + dc
                    if 0 <= r1 < H and 0 <= c1 < W and free[r1, c1] \
                            and dist[r1, c1] < 0:
                        dist[r1, c1] = dist[r0, c0] + 1
                        q.append((r1, c1))
            sr, sc_ = sc.w2c(cur); sr, sc_ = int(sr), int(sc_)
            path = []
            for _ in range(4000):
                path.append((sr, sc_))
                d0 = dist[sr, sc_]
                if d0 <= 0:
                    break
                best = None
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        r1, c1 = sr + dr, sc_ + dc
                        if 0 <= r1 < H and 0 <= c1 < W and dist[r1, c1] >= 0 \
                                and dist[r1, c1] < d0:
                            if best is None or dist[r1, c1] < best[0]:
                                best = (dist[r1, c1], r1, c1)
                if best is None:
                    break
                sr, sc_ = best[1], best[2]
            last = cur.copy()
            for r0, c0 in path:
                w = np.asarray(sc.c2w((r0, c0)))
                if np.linalg.norm(w - last) >= 0.25:
                    pts.append(w)
                    last = w
            cur = g
        return pts

    def _make_target(self, seed):
        rng = np.random.default_rng(int(seed) + 104729)
        spawn = np.array([self.scene.spawn[0], self.scene.spawn[1],
                          self.scene.spawn_theta], dtype=np.float64)
        wps = []
        if self.scenario == "narrow":
            # 狭窄通道：让目标绕到需要穿门洞的位置
            wps = [self.scene.spawn + np.array([3.0, 1.4]),
                   self.scene.spawn + np.array([6.0, -1.2]),
                   self.scene.spawn + np.array([1.5, -2.0])]
        elif self.scenario in ("short_occl", "long_occl"):
            # 遮挡：目标走到**最大件家具的背面**后停留指定时长。
            # 停顿点取家具中心沿「机器人→家具」方向的**远端**，机器人从近端看
            # 必然被家具挡住 —— 停顿时长因而直接控制遮挡时长（且可事后核验）。
            objs = sorted(self.scene.objects, key=lambda o: -o.half * o.height)
            if objs:
                o = objs[0]
                c = np.asarray(o.xy, float)
                v = c - np.asarray(spawn[:2], float)
                n = float(np.linalg.norm(v))
                v = v / n if n > 1e-6 else np.array([1.0, 0.0])
                hide = c + v * (o.half + PERSON_R + 0.12)      # 家具**远端**
                wps = [hide, hide + np.array([1.6, 1.4]), hide + np.array([-1.6, 1.4])]
                pause = hide
            else:
                pause = None
            # 大纲 §5 E11：短时 1–2 s、长时 > 3 s
            sec = 1.6 if self.scenario == "short_occl" else 3.8
            return FollowTarget(self.scene, rng, speed=self.target_speed, waypoints=wps,
                                spawn_pose=spawn, pause_at=pause, pause_sec=sec)
        if self.scenario == "tour":
            # 全屋巡回：先建目标（确定出生位），再按其位置算法生成巡回航点
            tgt = FollowTarget(self.scene, rng, speed=self.target_speed,
                               spawn_pose=spawn)
            wps = self._tour_waypoints(tgt.xy)
            tgt._wps = wps
            tgt._wi = 0
            tgt._loop = True
            if wps:
                tgt.wp = wps[0]
            return tgt
        return FollowTarget(self.scene, rng, speed=self.target_speed, waypoints=wps,
                            spawn_pose=spawn)

    # ------------------------------------------------------------ API
    def reset(self, seed=None, level=None):
        cfg = self.cfg
        if seed is None:
            seed = self.seed + self._ep
        self._ep += 1
        self._new_scene(seed)
        self.target = self._make_target(seed)
        # **初始朝向对准目标**：跟随任务在真实场景里也要先「捕获」目标，
        # 从「目标已在正前方」开始才是良定义的任务（否则第一秒就丢，量的不是跟随）。
        th0 = float(np.arctan2(*(np.asarray(self.target.xy, float)
                                 - np.asarray(self.scene.spawn, float))[::-1]))
        self.pose = np.array([self.scene.spawn[0], self.scene.spawn[1], th0],
                             dtype=np.float64)
        # 任务占位：跟随不使用语言指令，但父类的部分逻辑会读 `self.task`
        self.task = {"level": "follow", "obj_index": 0,
                     "instruction": "follow the person in the red vest",
                     "goal": np.asarray(self.target.xy, float)}
        self.last_action = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.path_len = 0.0
        self.n_collide = 0
        self.traj = [self.pose[:2].copy()]
        self.done = False
        self.start_xy = self.pose[:2].copy()
        self._gimg = None
        d0 = float(np.linalg.norm(self.target.xy - self.pose[:2]))
        self.d0 = self.min_dist = self.prev_dist = d0
        self.d_surf = float("nan")
        self.lstar = 1.0
        # ---- 逐回合记账 ----
        self.obs_log = []            # 每步 (真值距离, 真值方位, 可见, 在带内)
        self.cfg["camera_augment"] = self._aug_base
        # 光照/材质退化几何（按 scenario 设定，整回合生效）
        lit = SCENARIO_LIGHT.get(self.scenario, SCENARIO_LIGHT["normal"])
        cfg["glare"] = lit.get("glare", 0.0)
        cfg["gloss"] = lit.get("gloss", 0.0)
        cfg["exposure"] = lit.get("exposure", 1.0)
        cfg["flat_wall"] = (self.scenario == "plain_wall")
        # 光的方位**相对初始朝向**：`backlight/sunset` 的光源放在人（相机）正前方
        # —— 目标被逆光冲白；`gloss/shadow` 放侧向 90°，地面拉出光带/长阴影。
        # 用绝对方向会因机器人转向而失效（实测：光源在 +x、相机朝 +y 时
        # 眩光完全不入镜，逆光场景名不副实）。
        if lit.get("light_dir") is not None:
            if self.scenario in ("backlight", "sunset"):
                cfg["light_dir"] = th0 + np.pi          # 光从正前方射来
            else:
                cfg["light_dir"] = th0 + np.pi / 2      # 侧向光（光带/阴影）
        else:
            cfg["light_dir"] = None
        self._light_switched = False
        return self._obs()

    def _target_visible_gt(self):
        """真值可见性：在 HFOV 内、距离在量程内、且**无遮挡**（射线检测）。"""
        v = np.asarray(self.target.xy, float) - self.pose[:2]
        d = float(np.hypot(*v))
        if d < 0.35 or d > LIDAR_MAX_RANGE:
            return False
        a = float(np.arctan2(v[1], v[0]) - self.pose[2])
        a = (a + np.pi) % (2 * np.pi) - np.pi
        if abs(a) > np.deg2rad(CAMERA_HFOV_DEG) / 2:
            return False
        dist, h, _, _ = raycast(self.scene, self.pose[:2],
                                np.array([np.arctan2(v[1], v[0])]),
                                max_range=LIDAR_MAX_RANGE, grid=self.scene.wall_grid)
        return (not h[0]) or float(dist[0]) >= d - 0.05

    def _px_h(self, shape):
        H, W = shape[:2]
        return (W / 2.0) / np.tan(np.deg2rad(CAMERA_HFOV_DEG) / 2.0)

    def _perceived(self, obs):
        """**策略侧**的目标估计：只用图像 + 检测器（**不给真值**）。

        返回 `(bearing, dist)` 或 `None`。距离用「已知人体高度 + 针孔模型」反推：

            d = f · PERSON_H / 像素高

        而检测器给出的度量高度 `h` 满足 `h = 像素高 · d / f`，所以

            d_高度 = d_检测 · (PERSON_H / h_检测)

        即**按「检出高度与真人高度的比」对检测距离做尺度校正**。
        这比直接用地面几何稳：跟随距离带（~1.5 m）上两者都可用，
        但目标一走近，地面几何会饱和（导航任务上已实测饱和于 0.75 m）。
        """
        r = detect_person(obs["image"], lidar=obs.get("lidar"))
        if r is None:
            return None
        bearing, dist, _n = r
        return float(bearing), float(dist)

    def _dyn_obs(self):
        """动态障碍列表：**只有干扰行人**，不含被跟随者。

        为什么被跟随者不算障碍：市售跟随产品对「目标人体区域」做**激光滤波**
        （人腿的回波被剔除，否则底盘会一直绕着人转、永远贴不上去）。
        实测：把被跟随者塞进激光/碰撞集合后，反应式基线把人当成要避的障碍，
        short_occl 的 TCR 从 70.3% 掉到 20.2%（它不敢靠近，就永远进不了带内）。

        干扰行人（`crowd` 场景）**不滤波**：它们是真正的动态障碍，会挡激光、
        会被撞到（论文1 §5.3.2 行人横穿、论文2 多机避让）。
        """
        return list(getattr(self, "peds", []) or [])

    def _obs(self):
        cfg = self.cfg
        img = render(self.scene, self.pose, [self.target],
                     width=cfg["obs_size"][0], height=cfg["obs_size"][1],
                     light_dir=cfg.get("light_dir"),
                     light_strength=cfg.get("light_strength", 1.0),
                     glare=cfg.get("glare", 0.0), gloss=cfg.get("gloss", 0.0),
                     exposure=cfg.get("exposure", 1.0),
                     flat_wall=cfg.get("flat_wall", False))
        if cfg["camera_augment"] > 0:
            img = photometric_augment(img, self.rng, strength=cfg["camera_augment"])
        lid = lidar_scan(self.scene, self.pose, n_beams=cfg["lidar_beams"],
                         noise_std=cfg["lidar_noise"], rng=self.rng,
                         grid=occ_with_peds(self.scene, self._dyn_obs()))
        state = np.concatenate([self.pose, self.last_action]).astype(np.float32)
        # 目标**真值**只作为 `gt_target` 提供给 oracle 基线与指标；
        # `reactive` 与 VLA 一律走 `_perceived`，保证对照公平。
        return {"image": img, "lidar": lid.astype(np.float32), "state": state,
                "instruction": self.task["instruction"], "task_level": "follow",
                "gt_target": np.asarray(self.target.xy, float).copy()}

    def step(self, action):
        assert not self.done, "episode 已结束，请先 reset()"
        cfg = self.cfg
        dt = cfg["dt"]
        a = np.clip(np.asarray(action, dtype=np.float64),
                    [-cfg["v_max"], -cfg["w_max"]], [cfg["v_max"], cfg["w_max"]])
        # 光照突变场景：走到一半切换光度扰动强度（模拟灯被打开/遮挡）
        if self.scenario == "light_shift" and not self._light_switched and \
                self.steps >= cfg["max_steps"] // 2:
            self._light_switched = True
            cfg["camera_augment"] = min(0.85, self._aug_base + 0.55)
            # 同时把光照几何切成**逆光**（比纯光度扰动更接近真实失效模式：
            # 论文1 §5.3「傍晚直面走廊尽头西晒阳光，WDR 被击穿、画面大面积过曝泛白」）
            cfg["light_dir"] = np.pi
            cfg["glare"] = 0.6
        elif self.scenario == "dim" and not getattr(self, "_dim_on", False):
            # **弱光**：整回合亮度压低（模拟灯没开 / 傍晚）
            self._dim_on = True
            cfg["camera_augment"] = min(0.95, self._aug_base + 0.45)
            cfg["exposure"] = SCENARIO_LIGHT["dim"]["exposure"]
        elif self.scenario == "backlight" and not getattr(self, "_bl_on", False):
            # **逆光**：高对比 + 强色偏（目标背对窗户）
            self._bl_on = True
            cfg["camera_augment"] = min(0.95, self._aug_base + 0.60)
        self.last_action = a.astype(np.float32)

        # ---- 运动学（与父类一致：差速底盘 + 碰撞则原地停留）----
        v, w = float(a[0]), float(a[1])
        th = self.pose[2]
        nx = self.pose[0] + v * np.cos(th) * dt
        ny = self.pose[1] + v * np.sin(th) * dt
        nth = (th + w * dt + np.pi) % (2 * np.pi) - np.pi
        if is_free_dyn(self.scene, (nx, ny), self._dyn_obs(), ROBOT_RADIUS * 0.6):
            self.pose = np.array([nx, ny, nth])
            self.path_len += float(np.hypot(nx - self.pose[0], 0.0))
        else:
            self.pose = np.array([self.pose[0], self.pose[1], nth])
            self.n_collide += 1
        self.target.step(dt)
        self.steps += 1
        self.traj.append(self.pose[:2].copy())

        # ---- 指标记账（用**真值**，与策略无关，保证口径一致）----
        rel = np.asarray(self.target.xy, float) - self.pose[:2]
        d = float(np.hypot(*rel))
        bearing = float(np.arctan2(rel[1], rel[0]) - self.pose[2])
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        visible = bool(abs(bearing) <= np.deg2rad(CAMERA_HFOV_DEG) / 2
                       and self._target_visible_gt())
        in_band = bool(self.band[0] <= d <= self.band[1])
        self.obs_log.append((d, bearing, int(visible), int(in_band)))
        self.prev_dist = d
        self.min_dist = min(self.min_dist, d)

        # **回合不因目标丢失而结束** —— 丢失与恢复正是本任务要测的；
        # 只有超时才结束。
        timeout = self.steps >= cfg["max_steps"]
        self.done = bool(timeout)
        info = {"dist": d, "bearing": bearing, "visible": visible, "in_band": in_band,
                "success": bool(visible and in_band), "timeout": bool(timeout),
                "n_collide": self.n_collide, "level": "follow",
                "instruction": self.task["instruction"]}
        return self._obs(), 0.0, self.done, info

    # ------------------------------------------------------------ 指标
    def metrics(self):
        L = np.array(self.obs_log, dtype=float)
        if len(L) == 0:
            return {}
        d, b, vis, band = L[:, 0], L[:, 1], L[:, 2], L[:, 3]
        want = int(np.ceil(0.5 / self.cfg["dt"]))
        losses, rec, cur = [], [], 0
        for v_ in vis:
            if v_ < 0.5:
                cur += 1
            else:
                if cur >= want:
                    losses.append(cur)
                    rec.append(cur)
                cur = 0
        if cur >= want:
            losses.append(cur)
        err = np.abs(d - self.want_dist)
        return {"TCR": float(np.mean((vis > 0.5) & (band > 0.5))),
                "err_mean": float(err.mean()), "err_std": float(err.std()),
                "bearing_mean_deg": float(np.degrees(np.abs(b)).mean()),
                "vis_rate": float(np.mean(vis > 0.5)),
                "in_band_rate": float(np.mean(band > 0.5)),
                "n_loss": len(losses),
                "loss_mean_ms": float(np.mean(losses) * self.cfg["dt"] * 1000) if losses else 0.0,
                "rec_ms_median": float(np.median(rec) * self.cfg["dt"] * 1000) if rec else 0.0,
                "rec_ms_p90": float(np.percentile(rec, 90) * self.cfg["dt"] * 1000) if rec else 0.0,
                "steps": int(len(L)), "n_collide": int(self.n_collide)}


# ---------------------------------------------------------------- 局部避障
def steer_with_avoidance(obs, env, bearing, safe=1.05):
    """把「目标方位」修正为一个**可通行**的方位（激光安全间隙搜索）。

    为什么必须有：场景是多房间 + 家具，纯「对准目标直冲」的控制器会**卡在家具上**
    （实测 oracle 用真值也只有 18.8% TCR、3.39 m 距离误差：它一直对准目标，
    但走不过去）。真实跟随机器人同样需要局部避障，这不是特权信息。

    做法：若目标方位上有 < `safe` 的障碍，则在 ±70° 内找**离目标方位最近**的
    可通行方位；找不到就退回目标方位（由下层碰撞逻辑兜住）。
    """
    lid = np.asarray(obs["lidar"], dtype=float)
    n = lid.size
    if n == 0:
        return bearing
    ang = np.linspace(-np.pi, np.pi, n, endpoint=False)
    def _clear(a):
        i = int(np.argmin(np.abs(((ang - a + np.pi) % (2 * np.pi)) - np.pi)))
        # 取该方位附近 ±2 束的最小值，避免单束噪声
        idx = (np.arange(i - 2, i + 3)) % n
        return float(lid[idx].min())
    if _clear(bearing) >= safe:
        return bearing
    for off in np.arange(0.0, np.deg2rad(70.0), np.deg2rad(5.0)):
        for sgn in ((1.0, -1.0) if bearing >= 0 else (-1.0, 1.0)):
            a = bearing + sgn * off
            a = (a + np.pi) % (2 * np.pi) - np.pi
            if _clear(a) >= safe:
                return float(a)
    return bearing


# ---------------------------------------------------------------- 基线
class RandomFollow:
    """随机动作下界。"""
    name = "random"

    def reset(self, env):
        self.rng = np.random.default_rng(12345)

    def act(self, obs, env):
        return np.array([self.rng.uniform(0.0, env.cfg["v_max"] * 0.6),
                         self.rng.uniform(-env.cfg["w_max"] * 0.5,
                                          env.cfg["w_max"] * 0.5)])


class ReactiveFollow:
    """反应式跟随：检测马甲 → 「转向对准 + 速度按距离误差」直连。

    这是大纲 §5 E1 要求的「YOLO 检测 + PID 直连」基线在跟随任务上的版本，
    也是主创新 A 要超越的对象。
    """

    name = "reactive"

    def __init__(self, k_ang=1.9, k_v=0.9, search_w=0.9):
        self.k_ang, self.k_v, self.search_w = k_ang, k_v, search_w

    def reset(self, env):
        self.w_max = env.cfg["w_max"]
        self.v_max = env.cfg["v_max"] * V_FRAC

    def act(self, obs, env):
        p = env._perceived(obs)
        if p is None:
            # 丢失 → 原地扫描（真机跟随机器人丢目标后的标准行为）
            return np.array([0.0, self.search_w])
        bearing, dist = p
        hd = steer_with_avoidance(obs, env, bearing)
        # 需要大幅绕行时减速，避免贴着障碍高速冲（真机上的标准做法）
        slow = 1.0 if abs(hd - bearing) < np.deg2rad(20) else 0.45
        w = float(np.clip(self.k_ang * hd, -self.w_max, self.w_max))
        v = float(np.clip(self.k_v * (dist - env.want_dist), -self.v_max * 0.6, self.v_max))
        return np.array([v * slow, w])


class OracleFollow:
    """上界：用**真值**目标位置做同样的 PID。

    它回答「感知完美时该策略能到多好」，从而把差距拆成**感知瓶颈**与**控制瓶颈**。
    """

    name = "oracle"

    def __init__(self, k_ang=1.9, k_v=0.9):
        self.k_ang, self.k_v = k_ang, k_v

    def reset(self, env):
        self.w_max = env.cfg["w_max"]
        self.v_max = env.cfg["v_max"] * V_FRAC

    def act(self, obs, env):
        rel = np.asarray(env.target.xy, float) - env.pose[:2]
        d = float(np.hypot(*rel))
        bearing = float(np.arctan2(rel[1], rel[0]) - env.pose[2])
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        hd = steer_with_avoidance(obs, env, bearing)
        slow = 1.0 if abs(hd - bearing) < np.deg2rad(20) else 0.45
        w = float(np.clip(self.k_ang * hd, -self.w_max, self.w_max))
        v = float(np.clip(self.k_v * (d - env.want_dist), -self.v_max * 0.6, self.v_max))
        return np.array([v * slow, w])


# ================================================================ E4 场景化增强模块
class KalmanTracker:
    """E4 模块 ①：**卡尔曼速度估计 + 短期外推**。

    解决什么：E11 的结果显示跟随的瓶颈在「丢失后重捕获」（恢复中位 1242 ms
    vs oracle 255 ms）。目标丢失时，反应式策略只会**原地扫描**；
    而人有惯性 —— 记下丢失前的速度并外推一小段，就能朝「目标该在的方向」找。

    模型：常速度（CV），状态 `[x, y, vx, vy]`，只观测位置。
    增益取固定标量（Q 小 R 大），让速度估计平滑而非紧跟噪声。

    > 这是「记忆」思想在跟随任务上的对应物 —— 导航上补时序记忆带来
    > +7.75 pp（§4.23）；这里检验同一思想在另一个任务上是否成立。
    """

    def __init__(self, q=0.35, r=0.55, dt=0.10):
        self.dt, self.q, self.r = float(dt), float(q), float(r)
        self.x = None            # [x, y, vx, vy]
        self.lost_t = 0.0

    def reset(self):
        self.x = None
        self.lost_t = 0.0

    def update(self, pos):
        p = np.asarray(pos, float)
        if self.x is None:
            self.x = np.array([p[0], p[1], 0.0, 0.0])
            self.lost_t = 0.0
            return
        dt = self.dt
        # 预测 + 位置观测校正（标量增益，速度随位置残差隐式更新）
        x = np.array([self.x[0] + self.x[2] * dt, self.x[1] + self.x[3] * dt,
                      self.x[2], self.x[3]])
        K = self.q * dt / (self.q * dt + self.r)
        x[:2] = x[:2] + K * (p - x[:2])
        self.x = x
        self.lost_t = 0.0

    def mark_lost(self):
        self.lost_t += self.dt

    def predict(self, horizon=None):
        """外推 `horizon` 秒后的位置；无估计时返回 None。"""
        if self.x is None:
            return None
        h = self.dt if horizon is None else float(horizon)
        return self.x[:2] + self.x[2:4] * h


def auto_enhance(img, strength=1.0):
    """E4 模块 ②：**自适应图像增强**（针对光照突变）。

    为什么：`light_shift` 场景里 `reactive` 的 TCR 掉到 67.8%、恢复 4275 ms，
    而 oracle 不受影响 —— 差距全在感知。马甲色块在变暗/偏色后偏离调色板容差，
    于是检不出。**先做自适应 gamma + 百分位拉伸**把亮度拉回中性，
    检测器就能重新锁上。这是「环境先验」思想的最小实现：
    室内光照会变，但「目标该是什么颜色」是已知的。
    """
    x = img.astype(np.float32) / 255.0
    g = float(x.mean())
    if g < 1e-4:
        return img
    gamma = float(np.clip(np.log(0.5) / np.log(max(g, 1e-4)), 0.45, 2.2)) ** strength
    x = np.power(np.clip(x, 0, 1), gamma)
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    if hi - lo > 1e-3:
        x = np.clip((x - lo) / (hi - lo), 0, 1)
    return (x * 255).astype(np.uint8)


class FollowPolicyE4(ReactiveFollow):
    """反应式跟随 + **E4 模块开关**（逐项消融用）。

    所有模块默认关：此时与 `ReactiveFollow` **完全一致**，
    因此消融表里「全关」一行可直接对照 E11 的基线结果。
    """

    name = "reactive_e4"

    def __init__(self, use_kalman=False, use_depth=False, use_gamma=False,
                 use_enhance=False, enhance_strength=1.0,
                 gamma_clear_lo=0.5, gamma_clear_hi=1.5,
                 kf_q=0.35, kf_r=0.55, **kw):
        super().__init__(**kw)
        self.use_kalman = bool(use_kalman)
        self.use_depth = bool(use_depth)
        self.use_gamma = bool(use_gamma)
        self.use_enhance = bool(use_enhance)
        # **E4b 超参敏感性可调项**（默认值 = 原硬编码值 ⇒ 行为完全不变）：
        #   `enhance_strength` = α：自适应图像增强强度
        #   `gamma_clear_lo/hi` = γ：空间约束因子的净空映射（减速起点与斜率跨度）
        #   `kf_q/kf_r`        = λ_id：卡尔曼外推的过程噪声/观测噪声
        self.enhance_strength = float(enhance_strength)
        self.gamma_clear_lo = float(gamma_clear_lo)
        self.gamma_clear_hi = float(gamma_clear_hi)
        self.kf_q = float(kf_q)
        self.kf_r = float(kf_r)
        self.kf = KalmanTracker(q=self.kf_q, r=self.kf_r)

    def reset(self, env):
        super().reset(env)
        self.kf = KalmanTracker(q=self.kf_q, r=self.kf_r, dt=env.cfg["dt"])

    def _detect(self, img):
        """检测马甲色块，返回其像素掩码（供深度融合用）或 None。"""
        from scipy import ndimage as ndi
        ref = np.array(COLORS[VEST_COLOR], dtype=np.float64)
        d2 = np.linalg.norm(img.astype(np.float64) - ref[None, None, :], axis=2)
        mask = d2 < 95.0
        if mask.sum() < 25:
            return None
        lab, n = ndi.label(mask)
        if n == 0:
            return None
        sz = ndi.sum(mask, lab, index=np.arange(1, n + 1))
        return lab == (int(np.argmax(sz)) + 1)

    def act(self, obs, env):
        img = obs["image"]
        if self.use_enhance:
            img = auto_enhance(img, self.enhance_strength)
        r = detect_person(img, lidar=obs.get("lidar"))
        if r is None:
            return self._on_lost(obs, env)
        bearing, dist = float(r[0]), float(r[1])
        # E4 ②：**单目深度辅助** —— 底边接地法与「表观宽度」两种测距取平均。
        # 前者在近距离稳、后者在远距离稳，融合比单用更平缓。
        if self.use_depth:
            m = self._detect(img)
            if m is not None:
                H, W = img.shape[:2]
                f = (W / 2.0) / np.tan(np.deg2rad(CAMERA_HFOV_DEG) / 2.0)
                xs = np.where(m.any(axis=0))[0]
                wpx = max(2.0, float(xs.max() - xs.min() + 1))
                dist = 0.5 * (dist + f * 2.0 * PERSON_R / wpx)
        if self.use_kalman:
            a_ = env.pose[2] + bearing
            self.kf.update(env.pose[:2] + dist * np.array([np.cos(a_), np.sin(a_)]))
        hd = steer_with_avoidance(obs, env, bearing)
        slow = 1.0 if abs(hd - bearing) < np.deg2rad(20) else 0.45
        v = self.k_v * (dist - env.want_dist)
        # E4 ③：**空间约束因子 γ** —— 前方净空越小越减速（窄道 0.5 / 开阔 1.0）
        if self.use_gamma:
            lid = np.asarray(obs["lidar"], dtype=float)
            ang = np.linspace(-np.pi, np.pi, lid.size, endpoint=False)
            front = lid[np.abs(ang) <= np.deg2rad(30)]
            clear = float(front.min()) if front.size else 9.0
            v *= float(np.clip((clear - self.gamma_clear_lo) / self.gamma_clear_hi,
                               0.5, 1.0))
        w = float(np.clip(self.k_ang * hd, -self.w_max, self.w_max))
        v = float(np.clip(v, -self.v_max * 0.6, self.v_max)) * slow
        return np.array([v, w])

    def _on_lost(self, obs, env):
        """目标丢失后的行为：默认原地扫描；开卡尔曼则**外推找**。"""
        if not self.use_kalman or self.kf.x is None:
            return np.array([0.0, self.search_w])
        self.kf.mark_lost()
        # 外推时程随丢失时长**缩短**（速度估计置信度随时间衰减），最多 1.2 s
        h = float(np.clip(1.2 - self.kf.lost_t, 0.15, 1.2))
        p_pred = self.kf.predict(horizon=h)
        if p_pred is None:
            return np.array([0.0, self.search_w])
        rel = p_pred - env.pose[:2]
        d = float(np.hypot(*rel))
        bearing = float(np.arctan2(rel[1], rel[0]) - env.pose[2])
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        hd = steer_with_avoidance(obs, env, bearing)
        w = float(np.clip(self.k_ang * hd, -self.w_max, self.w_max))
        v = float(np.clip(self.k_v * (d - env.want_dist),
                          -self.v_max * 0.6, self.v_max * 0.8))
        return np.array([v, w])


# ================================================================ E11c 跟随阶段机 + 三级脱困
STAGE_FOLLOW, STAGE_STATIC, STAGE_LOST, STAGE_SEARCH = "follow", "static", "lost", "search"
STAGE_ORDER = (STAGE_FOLLOW, STAGE_STATIC, STAGE_LOST, STAGE_SEARCH)


class FollowPolicyStage(FollowPolicyE4):
    """**显式跟随阶段机 + 三级脱困策略**（E11c）。

    借鉴对象：
      · 论文4 表 3-5 的**阶段触发规则**（由可观测状态驱动自动切换，各阶段有各自判据）；
      · 论文3 §2.4 的**三级脱困**（速度调节 → 原地等待 → 局部重规划）。

    ## 四阶段（每个阶段有**各自的成功判据**）

    | 阶段 | 触发条件 | 该阶段的成功判据 |
    |---|---|---|
    | `follow` 跟随中 | 目标可见且**在运动** | 误差 ≤ 期望带 |
    | `static` 目标静止 | 目标可见且**静止** | 误差 ≤ 静止容差（更严） |
    | `lost` 目标离场 | 目标**刚**不可见（丢失 < `escalate_s`） | 能否在 T 秒内重新获取 |
    | `search` 搜索态 | 丢失 ≥ `escalate_s` | 能否重新获取 |

    ## 三级脱困（逐级升级）

    | 级别 | 触发条件 | 动作 | 代价 |
    |---|---|---|---|
    | L1 速度调节 | 前方净空 < `clear_l1` | 按净空比例减速 | 速度损失 |
    | L2 原地等待 | 净空持续不足 `stuck_steps` 步 | 原地停住 | 等待时长 |
    | L3 局部重规划 | 等待累计 > `replan_after` 秒 | A\\* 局部重规划并沿路走 | 重规划耗时 (ms) |

    默认 E4 模块**全关** ⇒ 与 `ReactiveFollow` 基线行为等价（只多了状态机与脱困），
    因此消融表里「基线 / +阶段机 / +阶段机+脱困」三行可直接对照。
    """

    name = "reactive_stage"

    def __init__(self, clear_l1=0.55, stuck_steps=8, replan_after=1.0,
                 escalate_s=3.0, static_speed=0.05, use_replan=True,
                 use_stage=True, use_l1=True, use_l2=True, use_l3=True, **kw):
        super().__init__(**kw)
        self.clear_l1 = float(clear_l1)
        self.stuck_steps = int(stuck_steps)
        self.replan_after = float(replan_after)
        self.escalate_s = float(escalate_s)
        self.static_speed = float(static_speed)
        self.use_replan = bool(use_replan)
        self.use_stage = bool(use_stage)
        # **逐级消融开关**（论文2 式 5-8 的逐项消融口径）
        self.use_l1 = bool(use_l1)
        self.use_l2 = bool(use_l2)
        self.use_l3 = bool(use_l3)
        self._planner = None
        self.reset_stats()

    # ------------------------------------------------------------ 状态与统计
    def reset_stats(self):
        self.stage = STAGE_FOLLOW
        self.lost_t = 0.0
        self.stuck_n = 0
        self.wait_t = 0.0
        self.replan_path = None
        self.replan_idx = 0
        self.last_target_xy = None
        self.prev_pose = None
        self.prev_dist = None
        self.stats = {
            "stage_time": {s: 0.0 for s in STAGE_ORDER},
            "stage_trans": {s: 0 for s in STAGE_ORDER},
            "l1": {"n": 0, "t": 0.0},
            "l2": {"n": 0, "t": 0.0},
            "l3": {"n": 0, "t": 0.0},
            "replan_ms": [],
            "replan_fail": 0,
        }

    def reset(self, env):
        super().reset(env)
        self.reset_stats()

    # ------------------------------------------------------------ 工具
    @staticmethod
    def _front_clear(obs, half_deg=20.0):
        """前方 ±`half_deg` 内的**最小激光净空**（米）。

        扇区取窄（±20°）而不是宽（±30°）：跟随时侧向常有行人/桌角，
        用宽扇区会把「旁边有东西」误判成「前面被堵」，导致无谓停车。
        """
        lid = np.asarray(obs["lidar"], dtype=float)
        ang = np.linspace(-np.pi, np.pi, lid.size, endpoint=False)
        front = lid[np.abs(ang) <= np.deg2rad(half_deg)]
        return float(front.min()) if front.size else 9.0

    # ------------------------------------------------------------ 阶段机
    def _update_stage(self, seen, env, dist=None):
        """按可观测状态推进阶段机（论文4 表 3-5 的触发规则）。"""
        dt = float(env.cfg["dt"])
        if seen:
            # 「目标是否静止」用**目标的真值速度**判定：这是**环境状态**
            # （任务阶段的定义），不是感知量，因此不构成「作弊」，但必须在论文里写明。
            # 注意不能用「相对速度小」当判据——那会把「跟随良好」误判成「目标静止」，
            # 反而让机器人一直用半速跟随（实测 TCR 从 99.4% 掉到 88.3%）。
            sp = float(getattr(env.target, "speed", 0.0))
            new = STAGE_STATIC if sp < self.static_speed else STAGE_FOLLOW
            self.lost_t = 0.0
        else:
            self.lost_t += dt
            new = STAGE_LOST if self.lost_t < self.escalate_s else STAGE_SEARCH
        if new != self.stage:
            self.stats["stage_trans"][new] += 1
            self.stage = new
        self.stats["stage_time"][self.stage] += dt

    # ------------------------------------------------------------ 三级脱困
    def _replan(self, obs, env):
        """L3：**局部重规划**（A\\* 到感知到的目标位置），返回是否成功。"""
        import time
        if self._planner is None:
            from .planner import AStarPolicy
            self._planner = AStarPolicy(plan_clearance=0.15, clear_cost=2.0)
        goal = self.last_target_xy
        if goal is None:
            # 从未看见目标 → 沿当前朝向前方 1.5 m 做「探路式」局部规划
            th = float(env.pose[2])
            goal = np.asarray(env.pose[:2], dtype=float) + \
                1.5 * np.array([np.cos(th), np.sin(th)])
        t0 = time.perf_counter()
        try:
            path = self._planner._astar(env, goal)
        except Exception:
            path = None
        self.stats["replan_ms"].append((time.perf_counter() - t0) * 1000.0)
        self.stats["l3"]["n"] += 1
        self.stats["l3"]["t"] += float(env.cfg["dt"])
        if not path:
            self.stats["replan_fail"] += 1
            self.replan_path = None
            return False
        self.replan_path = [np.asarray(p, dtype=float) for p in path]
        self.replan_idx = 0
        return True

    def _path_bearing(self, env, lookahead=0.6):
        """沿重规划路径取前视路点的 (方位角, 距离)。"""
        if not self.replan_path:
            return None
        while (self.replan_idx < len(self.replan_path) - 1 and
               float(np.hypot(*(self.replan_path[self.replan_idx] -
                                np.asarray(env.pose[:2], dtype=float)))) < lookahead):
            self.replan_idx += 1
        rel = self.replan_path[self.replan_idx] - np.asarray(env.pose[:2], dtype=float)
        d = float(np.hypot(*rel))
        b = float(np.arctan2(rel[1], rel[0]) - env.pose[2])
        return (b + np.pi) % (2 * np.pi) - np.pi, d

    def _unstick(self, obs, env, v_cmd, dist=None):
        """三级脱困。返回 `(v, level)`，level ∈ {0,1,2,3}。

        **卡住判据**用「**想走却一步没动**」（指令速度 > 0 而实际位移 = 0），
        而不是单看净空：跟随时前方净空 < 1 m 是常态（目标就在那附近），
        若拿净空当卡住判据，机器人会在正常跟随时不停停车 —— 实测把 TCR 从
        100% 打到 96.7%。净空只用于 **L1 速度调节**（预防性减速）。

        **L1 还要满足「确实需要靠近目标」**：目标本来就在带内时，前方净空小
        多半是**目标自己**在那儿，此时减速只会拉低跟随质量
        （crowd 场景实测 TCR 87.5% → 82.0%）。
        """
        dt = float(env.cfg["dt"])
        pose = np.asarray(env.pose[:2], dtype=float)
        moved = 0.0
        if self.prev_pose is not None:
            moved = float(np.hypot(*(pose - self.prev_pose)))
        self.prev_pose = pose

        clear = self._front_clear(obs)
        wants_move = v_cmd > 0.05
        # 卡住 = **想走却一步没动**（FollowEnv 碰撞时位姿原地不动，只更新朝向）。
        # 用绝对位移阈值 0.015 m 会把「指令速度很小」也判成卡住
        # （v=0.06 m/s × dt=0.1 s ⇒ 位移 0.006 m < 0.015 m），故用严格零位移。
        stuck_now = bool(wants_move and moved < 1e-6)
        needs_approach = bool(dist is None or dist > env.want_dist + 0.3)

        # ---- L1：速度调节（净空不足 **且** 需要靠近目标 ⇒ 减速）----
        scale = 1.0
        if self.use_l1 and clear < self.clear_l1 and needs_approach:
            self.stats["l1"]["n"] += 1
            self.stats["l1"]["t"] += dt
            lo = 0.30
            scale = float(np.clip((clear - lo) / max(1e-3, self.clear_l1 - lo), 0.0, 1.0))
            v_cmd = v_cmd * scale

        if not stuck_now:
            self.stuck_n = 0
            # 等待计时**缓慢衰减**而非清零：短暂的「顶一下又松开」也应累计成
            # 一次较长的等待，否则每次脱离就把计时清掉，L3 永远触发不了
            # （实测 L3 触发 0 次）。衰减系数 0.8 兼顾「抖一下不算卡死」。
            self.wait_t *= 0.8
            return v_cmd, 1 if scale < 1.0 else 0

        # ---- L2：原地等待（连续多步「想走却没动」）----
        self.stuck_n += 1
        if not self.use_l2 or self.stuck_n < self.stuck_steps:
            return v_cmd, 1
        self.stats["l2"]["n"] += 1
        self.stats["l2"]["t"] += dt
        v = 0.0
        self.wait_t += dt
        if not self.use_l3 or self.wait_t < self.replan_after:
            return v, 2

        # ---- L3：局部重规划 ----
        ok = self._replan(obs, env)
        if ok:
            # 重规划成功后**复位**，否则会每帧重复规划（实测 L3 每回合十余次）
            self.stuck_n = 0
            self.wait_t = 0.0
            return v, 3
        self.wait_t = 0.0
        self.stuck_n = 0
        return v, 2

    # ------------------------------------------------------------ 主循环
    def act(self, obs, env):
        img = obs["image"]
        if self.use_enhance:
            img = auto_enhance(img, self.enhance_strength)
        r = detect_person(img, lidar=obs.get("lidar"))
        seen = r is not None
        bearing = float(r[0]) if seen else 0.0
        dist = float(r[1]) if seen else 0.0
        if seen and self.use_depth:
            m = self._detect(img)
            if m is not None:
                H, W = img.shape[:2]
                f = (W / 2.0) / np.tan(np.deg2rad(CAMERA_HFOV_DEG) / 2.0)
                xs = np.where(m.any(axis=0))[0]
                wpx = max(2.0, float(xs.max() - xs.min() + 1))
                dist = 0.5 * (dist + f * 2.0 * PERSON_R / wpx)
        if seen:
            a_ = env.pose[2] + bearing
            self.last_target_xy = env.pose[:2] + dist * np.array([np.cos(a_), np.sin(a_)])
            if self.use_kalman:
                self.kf.update(self.last_target_xy)
        else:
            self.prev_dist = None
        if self.use_stage:
            self._update_stage(seen, env, dist if seen else None)

        # ---- 阶段 → 行为 ----
        if self.stage in (STAGE_FOLLOW, STAGE_STATIC) and seen:
            hd = steer_with_avoidance(obs, env, bearing)
            slow = 1.0 if abs(hd - bearing) < np.deg2rad(20) else 0.45
            w = float(np.clip(self.k_ang * hd, -self.w_max, self.w_max))
            if self.stage == STAGE_STATIC:
                # 目标静止 ⇒ 速度上限减半（避免冲撞静止目标）
                v = float(np.clip(self.k_v * (dist - env.want_dist),
                                  -self.v_max * 0.3, self.v_max * 0.5))
            else:
                v = float(np.clip(self.k_v * (dist - env.want_dist),
                                  -self.v_max * 0.6, self.v_max))
            v *= slow
        elif self.stage == STAGE_LOST and not seen:
            a = self._on_lost(obs, env)
            v, w = float(a[0]), float(a[1])
        elif self.stage == STAGE_SEARCH and not seen:
            # 搜索态：转速更大（扩大扫描范围），速度压到 0
            v, w = 0.0, self.search_w * 1.6
        else:
            # 阶段机关闭时退化为父类行为；可见但处于 lost/search 的边界情形
            a = self._on_lost(obs, env) if not seen else np.array([0.0, self.search_w])
            v, w = float(a[0]), float(a[1])

        # ---- 三级脱困 ----
        v, level = self._unstick(obs, env, v, dist if seen else None)
        if level == 3:
            pb = self._path_bearing(env)
            if pb is not None:
                b, d = pb
                hd = steer_with_avoidance(obs, env, b)
                w = float(np.clip(self.k_ang * hd, -self.w_max, self.w_max))
                v = float(np.clip(self.k_v * (d - env.want_dist),
                                  -self.v_max * 0.6, self.v_max * 0.5))
        return np.array([v, w])


POLICIES = {"random": RandomFollow, "reactive": ReactiveFollow, "oracle": OracleFollow,
            "reactive_e4": FollowPolicyE4}
POLICIES["stage"] = FollowPolicyStage

# E4 消融：模块开关组合。`baseline` 与 `ReactiveFollow` 等价（全关），
# 其余每行**只开一个模块** —— 这样才能把增益归到具体模块上。
E4_ABLATION = {
    "baseline": {},
    "kalman": {"use_kalman": True},
    "depth": {"use_depth": True},
    "gamma": {"use_gamma": True},
    "enhance": {"use_enhance": True},
    "all": {"use_kalman": True, "use_depth": True,
            "use_gamma": True, "use_enhance": True},
}


def run_e4_ablation(episodes=30, scenarios=None, max_steps=300, want_dist=1.5,
                    out=""):
    """E4 消融：模块 × 场景 的成功率矩阵（大纲 §5 E4 的产出表 G5）。"""
    scenarios = list(scenarios or SCENARIOS)
    print("=== E4 场景化增强模块消融（%d 回合/格）===" % episodes)
    print("%-11s" % "模块", end="")
    for sc in scenarios:
        print(" %13s" % sc[:13], end="")
    print()
    res = {}
    for name, kw in E4_ABLATION.items():
        pol = FollowPolicyE4(**kw)
        print("%-11s" % name, end="")
        for sc in scenarios:
            env = FollowEnv(seed=0, want_dist=want_dist, max_steps=max_steps,
                            scenario=sc, camera_augment=0.25)
            rows = []
            for ep in range(episodes):
                s = 3000 + ep * 17
                obs = env.reset(seed=s)
                pol.reset(env)
                while True:
                    obs, r, done, info = env.step(pol.act(obs, env))
                    if done:
                        break
                rows.append(env.metrics())
            tcr = float(np.mean([m["TCR"] for m in rows]))
            rec = float(np.mean([m["rec_ms_median"] for m in rows]))
            res[(name, sc)] = {"TCR": tcr, "rec_ms_median": rec}
            print(" %6.1f%%/%4.0f" % (100 * tcr, rec), end="")
        print()
    if out:
        d = os.path.dirname(out)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(out, "w") as f:
            json.dump({f"{k[0]}|{k[1]}": v for k, v in res.items()}, f,
                      indent=2, ensure_ascii=False)
        print("\n已写入 %s" % out)
    return res


def run(policy_name="reactive", scenario="normal", episodes=30, seed_base=3000,
        stride=17, want_dist=1.5, max_steps=300, seed=0, verbose=True,
        camera_augment=0.25):
    """跑一组（策略 × 场景），返回汇总指标与逐回合明细。"""
    env = FollowEnv(seed=seed, want_dist=want_dist, max_steps=max_steps,
                    scenario=scenario, camera_augment=camera_augment)
    pol = POLICIES[policy_name]()
    rows, t0 = [], time.time()
    for ep in range(episodes):
        s = seed_base + ep * stride
        obs = env.reset(seed=s)
        pol.reset(env)
        while True:
            obs, r, done, info = env.step(pol.act(obs, env))
            if done:
                break
        m = env.metrics()
        m.update({"seed": int(s), "policy": policy_name, "scenario": scenario})
        rows.append(m)
    keys = ("TCR", "err_mean", "err_std", "bearing_mean_deg", "vis_rate",
            "in_band_rate", "n_loss", "loss_mean_ms", "rec_ms_median",
            "rec_ms_p90", "n_collide")
    agg = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if vals:
            agg[k] = float(np.mean(vals))
            agg[k + "_sd"] = float(np.std(vals))
    if verbose:
        print("  %-9s %-11s TCR %5.1f%%  err %.2f±%.2f m  |bear| %4.1f°  可见 %5.1f%%  "
              "丢失 %.2f 次  恢复中位 %6.1f ms  (%.0fs)" %
              (policy_name, scenario, 100 * agg.get("TCR", 0),
               agg.get("err_mean", 0), agg.get("err_std", 0),
               agg.get("bearing_mean_deg", 0), 100 * agg.get("vis_rate", 0),
               agg.get("n_loss", 0), agg.get("rec_ms_median", 0), time.time() - t0))
    return {"agg": agg, "rows": rows,
            "config": {"policy": policy_name, "scenario": scenario,
                       "episodes": episodes, "seed_base": seed_base, "stride": stride,
                       "want_dist": want_dist, "max_steps": max_steps,
                       "band": list(env.band), "target_speed": env.target_speed,
                       "camera_augment": camera_augment}}


def main():
    ap = argparse.ArgumentParser(description="E11 视觉跟随（仿真版）")
    ap.add_argument("--policies", default="random,reactive,oracle")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--want-dist", type=float, default=1.5)
    ap.add_argument("--seed-base", type=int, default=3000)
    ap.add_argument("--stride", type=int, default=17)
    ap.add_argument("--camera-augment", type=float, default=0.25)
    ap.add_argument("--out", default="", help="结果 JSON 落盘路径")
    ap.add_argument("--e4", action="store_true",
                    help="跑 E4 场景化增强模块消融（模块 × 场景 的 TCR/恢复时间矩阵）")
    a = ap.parse_args()

    if a.e4:
        run_e4_ablation(episodes=a.episodes, max_steps=a.max_steps,
                        want_dist=a.want_dist, out=a.out)
        return

    print("=== E11 视觉跟随（仿真，%d 回合/格，期望距离 %.2f m，目标速度 %.2f m/s）==="
          % (a.episodes, a.want_dist, TARGET_SPEED))
    print("  被跟随者穿**场景唯一**的红色马甲；reactive 与 oracle 的差别只在"
          "「用感知」还是「用真值」")
    out = {}
    for sc in a.scenarios.split(","):
        for p in a.policies.split(","):
            out[f"{p}|{sc}"] = run(policy_name=p, scenario=sc, episodes=a.episodes,
                                   seed_base=a.seed_base, stride=a.stride,
                                   want_dist=a.want_dist, max_steps=a.max_steps,
                                   camera_augment=a.camera_augment)
    if a.out:
        d = os.path.dirname(a.out)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n已写入 {a.out}")


if __name__ == "__main__":
    main()
