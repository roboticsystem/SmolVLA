"""非 VLA 基线策略（E1）。

对应用户大纲中的基线设计：
    Random        随机游走 —— 性能下界
    Frontier      几何前沿探索（激光雷达建图 + 前沿目标）—— 「无语义」基线
    Oracle        完美目标知识 + 反应式避障 —— 控制上界（把感知与决策分离）

说明：
- 三个基线都假设**位姿已知**（仿真真值）。本测试床评估的是导航决策与感知，
  定位由 E1 的 odom_node / 扫描匹配承担，不在基线对比范围内。
- 真机版本的反应式基线（YOLO 检测 + PID 直连）需接入相机检测器后替换 Oracle 的
  「完美目标知识」，接口保持一致。
"""

from __future__ import annotations

import numpy as np

from .env import V_MAX, W_MAX
from .planner import (AStarPolicy, AStarRandomTargetPolicy,
                      GeodesicRandomTargetPolicy, GeodesicWindowPolicy,
                      PlannerOraclePolicy, PlannerPolicy, PlannerRandomTargetPolicy)
from .sensors import CAMERA_HFOV_DEG, LIDAR_MAX_RANGE
from .world import CELL, COLORS


def beam_index(ang, n):
    """相对角度（弧度）→ 激光束索引。

    束角为 ``linspace(-pi, pi, n, endpoint=False)``，即束 k 对应 ``-pi + k*2pi/n``，
    因此换算须先加 pi 偏置（漏掉会导致索引整体偏移半圈）。
    """
    return int(round(((float(ang) + np.pi) / (2 * np.pi)) * n)) % n


class RandomPolicy:
    name = "random"

    def __init__(self, seed=0, v_range=(0.0, 0.30), w_range=(-1.2, 1.2)):
        self.rng = np.random.default_rng(seed)
        self.v_range = v_range
        self.w_range = w_range

    def reset(self, env=None):
        pass

    def act(self, obs, env):
        return np.array([self.rng.uniform(*self.v_range), self.rng.uniform(*self.w_range)])


class OraclePolicy:
    """完美地图规划上界：沿**测地距离场**梯度行进 + 反应式避障。

    为什么不用「直冲目标」：纯贪心在室内会陷入局部极小（撞到墙角后原地打转，
    实测路径长度仅 L* 的 13%）。改为在环境提供的测地距离场上做带前瞻的梯度下降，
    得到一个真正可用的控制上界，从而把「感知/语义理解」与「控制能力」分离。

    仍然假设目标位置已知（完美感知），因此它衡量的是控制与规划的上界。
    """

    name = "oracle"

    def __init__(self, k_ang=2.4, slow_radius=0.9, stop_radius=0.45,
                 avoid_dist=0.38, lookahead=0.45, v_max=V_MAX, w_max=W_MAX):
        self.k_ang = k_ang
        self.slow_radius = slow_radius
        self.stop_radius = stop_radius
        self.avoid_dist = avoid_dist
        self.lookahead = lookahead
        self.v_max = v_max
        self.w_max = w_max
        self._field = None
        self._field_goal = None       # 不能叫 self._goal，否则会覆盖 _goal() 方法

    def reset(self, env=None):
        self._field = None
        self._field_goal = None       # 注意：不能叫 self._goal，否则会覆盖 _goal() 方法
        self._stuck = 0
        self._esc = 0
        self._esc_dir = 1.0
        self._last_xy = None

    def _goal(self, env):
        """规划目标。子类可覆写（如 RandomTargetPolicy 换成随机目标）。"""
        return env.task["goal"]

    def _field_for(self, env, goal):
        if (self._field is None or self._field_goal is None
                or not np.allclose(goal, self._field_goal)):
            self._field = env._shortest_path_len(goal)
            self._field_goal = np.asarray(goal, dtype=float).copy()
        return self._field

    def act(self, obs, env):
        lidar = obs["lidar"]
        n = len(lidar)
        pose = env.robot_pose()
        goal = self._goal(env)
        f = self._field_for(env, goal)
        H, W = f.shape
        sc = env.scene

        # ---- 卡死脱困：连续多步几乎没有位移则原地旋转若干步 ----
        # （纯反应式转向会与路径跟踪的转向相互抵消，必须靠显式脱困跳出振荡）
        if self._esc > 0:
            self._esc -= 1
            return np.array([0.0, self._esc_dir * self.w_max])
        if self._last_xy is not None:
            moved_now = float(np.linalg.norm(pose[:2] - self._last_xy))
            self._stuck = self._stuck + 1 if moved_now < 2e-3 else 0
        self._last_xy = pose[:2].copy()
        if self._stuck > 10:
            self._stuck = 0
            self._esc = 12
            ang_goal = float(np.arctan2(goal[1] - pose[1], goal[0] - pose[0]) - pose[2])
            ang_goal = (ang_goal + np.pi) % (2 * np.pi) - np.pi
            fwd0 = beam_index(ang_goal, n)
            left = float(np.min(lidar[[(fwd0 + j) % n for j in range(2, 26)]]))
            right = float(np.min(lidar[[(fwd0 - j) % n for j in range(2, 26)]]))
            self._esc_dir = 1.0 if left >= right else -1.0
            return np.array([0.0, self._esc_dir * self.w_max])

        d = float(np.linalg.norm(goal - pose[:2]))
        r0, c0 = sc.w2c(pose[:2])
        r0 = int(np.clip(r0, 0, H - 1)); c0 = int(np.clip(c0, 0, W - 1))

        # 在半径 lookahead 的窗口内选距离场最小的格，作为前瞻目标点
        k = max(2, int(round(self.lookahead / CELL)))
        r_lo, r_hi = max(0, r0 - k), min(H, r0 + k + 1)
        c_lo, c_hi = max(0, c0 - k), min(W, c0 + k + 1)
        sub = f[r_lo:r_hi, c_lo:c_hi]
        valid = sub >= 0
        if valid.any():
            sub2 = np.where(valid, sub, np.inf)
            dr, dc = np.unravel_index(int(np.argmin(sub2)), sub2.shape)
            tx = sc.bounds[0] + (c_lo + dc + 0.5) * CELL
            ty = sc.bounds[2] + (r_lo + dr + 0.5) * CELL
        else:
            rr, cc = np.nonzero(f >= 0)
            if rr.size == 0:
                return np.array([0.0, 0.0])
            j = int(np.argmin((rr - r0) ** 2 + (cc - c0) ** 2))
            tx = sc.bounds[0] + (cc[j] + 0.5) * CELL
            ty = sc.bounds[2] + (rr[j] + 0.5) * CELL

        # 接近目标时直接朝向目标，避免前瞻点在目标附近抖动
        aim = np.array([tx, ty]) if d > 1.0 else np.asarray(goal, dtype=float)
        ang = float(np.arctan2(aim[1] - pose[1], aim[0] - pose[0]) - pose[2])
        ang = (ang + np.pi) % (2 * np.pi) - np.pi

        fwd = beam_index(ang, n)
        front_d = float(np.min(lidar[[(fwd + j) % n for j in range(-4, 5)]]))

        v = self.v_max * float(np.clip(1.0 - abs(ang) / np.pi * 1.6, 0.0, 1.0))
        if d < self.slow_radius:
            v *= max(0.2, d / max(self.slow_radius, 1e-6))
        if d < self.stop_radius:
            v = 0.0
        w = float(np.clip(self.k_ang * ang, -self.w_max, self.w_max))

        # 避障：**只按接近程度连续减速，不接管转向**。
        # 关键教训（三轮实测）：任何「避障改写 w」的做法都会与路径跟踪的转向相互抵消
        # ——出生点距障碍可能仅 0.21 m，机器人长期处于避障态并原地振荡
        # （实测 93% 的步数没有位移、SR 跌到 6.7%）。
        # 路径本身已由膨胀栅格上的测地场保证无碰撞，转向交给路径跟踪；
        # 万一真的卡死，交给上面的显式脱困逻辑。
        if front_d < self.avoid_dist:
            floor_d = 0.20
            scale = float(np.clip((front_d - floor_d) / max(self.avoid_dist - floor_d, 1e-6), 0.0, 1.0))
            v *= scale
        return np.array([v, w])


class FrontierPolicy:
    """几何前沿探索：激光雷达累积栅格 → 前沿 → 最近前沿。

    **不使用任何语义信息**，因此预期在 L1/L2/L3 上都接近下界；它的作用是把
    「语义先验的价值」从「控制能力」中分离出来（E2 的核心对照）。
    """

    name = "frontier"

    def __init__(self, res=0.10, max_range=12.0, v_max=V_MAX, w_max=W_MAX,
                 replan_every=8, obstacle_inflate=0.25, goal_stop=0.35):
        self.res = res
        self.max_range = max_range
        self.v_max = v_max
        self.w_max = w_max
        self.replan_every = replan_every
        self.inflate = obstacle_inflate
        self.goal_stop = goal_stop
        self.reset()

    def reset(self, env=None):
        self.grid = None
        self.bounds = None
        self._t = 0
        self._target = None
        self._stuck = 0
        self._esc = 0
        self._esc_dir = 1.0
        self._last_xy = None
        self._blacklist = set()      # 长时间无法抵达的前沿格
        self._best_d = None
        self._tgt_age = 0

    # ---------- 建图 ----------
    def _ensure(self, env):
        if self.grid is not None:
            return
        sc = env.scene
        pad = 1.0
        self.bounds = (sc.bounds[0] - pad, sc.bounds[1] + pad,
                       sc.bounds[2] - pad, sc.bounds[3] + pad)
        W = int(np.ceil((self.bounds[1] - self.bounds[0]) / self.res))
        H = int(np.ceil((self.bounds[3] - self.bounds[2]) / self.res))
        self.grid = np.full((H, W), -1, dtype=np.int8)   # -1 未知, 0 自由, 1 占据
        self._inflate_n = max(1, int(round(self.inflate / self.res)))

    def _w2c(self, pts):
        c = np.floor((pts[:, 0] - self.bounds[0]) / self.res).astype(np.int64)
        r = np.floor((pts[:, 1] - self.bounds[2]) / self.res).astype(np.int64)
        return r, c

    def _integrate(self, pose, lidar):
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
        n = len(lidar)
        ang = np.linspace(-np.pi, np.pi, n, endpoint=False) + th
        ds = np.arange(0.0, self.max_range, self.res)
        px = x + np.cos(ang)[:, None] * ds[None, :]
        py = y + np.sin(ang)[:, None] * ds[None, :]
        r, c = self._w2c(np.stack([px.ravel(), py.ravel()], 1))
        H, W = self.grid.shape
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        r, c = r[ok], c[ok]
        d_flat = np.broadcast_to(ds[None, :], (n, ds.size)).ravel()[ok]
        lid_flat = np.broadcast_to(lidar[:, None], (n, ds.size)).ravel()[ok]
        hit = lid_flat < self.max_range - 1e-6

        free = d_flat < (lid_flat - self.res)
        self.grid[r[free], c[free]] = 0
        # 只在命中点**附近**标障碍。若写成 `d >= lidar - res`，则射线命中点之后的
        # 全部采样点都会被标成障碍（120 束 × 120 采样 → 上万个假障碍），
        # 地图几乎全被占满、BFS 直接失败（实测 target 恒为 None、机器人原地打转）。
        occ = hit & (d_flat >= (lid_flat - self.res)) & (d_flat <= (lid_flat + self.res))
        self.grid[r[occ], c[occ]] = 1

        # 起点附近标记为自由（清理自我遮挡）
        rr, cc = self._w2c(np.array([[x, y]]))
        if 0 <= rr[0] < H and 0 <= cc[0] < W:
            k = self._inflate_n
            self.grid[max(0, rr[0] - k):rr[0] + k + 1, max(0, cc[0] - k):cc[0] + k + 1] = 0

    # ---------- 前沿 ----------
    def _frontiers(self):
        g = self.grid
        unknown = g < 0
        free = g == 0
        nb_unknown = np.zeros_like(free)
        nb_unknown[1:, :] |= unknown[:-1, :]
        nb_unknown[:-1, :] |= unknown[1:, :]
        nb_unknown[:, 1:] |= unknown[:, :-1]
        nb_unknown[:, :-1] |= unknown[:, 1:]
        return free & nb_unknown

    def _bfs_from(self, start_rc):
        from collections import deque
        g = self.grid
        H, W = g.shape
        blocked = (g == 1)
        # 膨胀障碍
        k = self._inflate_n
        if k > 0:
            bl = blocked.copy()
            for dr in range(-k, k + 1):
                for dc in range(-k, k + 1):
                    rr0, rr1 = max(0, dr), H + min(0, dr)
                    cc0, cc1 = max(0, dc), W + min(0, dc)
                    bl[rr0:rr1, cc0:cc1] |= blocked[rr0 - dr:rr1 - dr, cc0 - dc:cc1 - dc]
            blocked = bl
        dist = np.full((H, W), -1, dtype=np.int32)
        sr, sc = start_rc
        if not (0 <= sr < H and 0 <= sc < W):
            return dist
        # 起点本身可能在障碍膨胀范围内（机器人贴墙/贴家具），必须放行，
        # 否则 BFS 直接失败、策略退化为原地打转（实测路径长度仅 L* 的 4%）
        blocked = blocked.copy()
        blocked[sr, sc] = False
        dist[sr, sc] = 0
        q = deque([(sr, sc)])
        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not blocked[nr, nc] and dist[nr, nc] < 0:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
        return dist

    def act(self, obs, env):
        self._ensure(env)
        self._integrate(env.robot_pose(), obs["lidar"])
        self._t += 1

        # 卡死脱困（与 OraclePolicy 同一套机制：反应式转向会与目标转向相互抵消）
        if self._esc > 0:
            self._esc -= 1
            return np.array([0.0, self._esc_dir * self.w_max])
        if self._last_xy is not None:
            moved_now = float(np.linalg.norm(env.robot_pose()[:2] - self._last_xy))
            self._stuck = self._stuck + 1 if moved_now < 2e-3 else 0
        self._last_xy = env.robot_pose()[:2].copy()
        if self._stuck > 10:
            self._stuck = 0
            self._esc = 12
            lid = obs["lidar"]
            nn = len(lid)
            self._esc_dir = 1.0 if float(np.min(lid[: nn // 2])) >= float(np.min(lid[nn // 2:])) else -1.0
            return np.array([0.0, self._esc_dir * self.w_max])

        if self._t % self.replan_every == 1 or self._target is None:
            fr = self._frontiers()
            rr, cc = self._w2c(np.array([[env.robot_pose()[0], env.robot_pose()[1]]]))
            dist = self._bfs_from((int(rr[0]), int(cc[0])))
            self._target = None
            if fr.any() and (dist >= 0).any():
                frr, fcc = np.nonzero(fr)
                dv = dist[frr, fcc]
                good = dv >= 0
                if self._blacklist:
                    bl = np.array([(int(a), int(b)) not in self._blacklist
                                   for a, b in zip(frr, fcc)])
                    good &= bl
                if good.any():
                    frr, fcc, dv = frr[good], fcc[good], dv[good]
                    # 取最近的前沿（格数最少）
                    k = int(np.argmin(dv))
                    self._target = (int(frr[k]), int(fcc[k]))
                    self._best_d = None
                    self._tgt_age = 0

        # 进度跟踪：长期无法靠近当前前沿则拉黑，避免在两个前沿间来回震荡
        if self._target is not None:
            tr, tc = self._target
            tx = self.bounds[0] + (tc + 0.5) * self.res
            ty = self.bounds[2] + (tr + 0.5) * self.res
            dnow = float(np.hypot(tx - env.robot_pose()[0], ty - env.robot_pose()[1]))
            if self._best_d is None or dnow < self._best_d - 0.1:
                self._best_d = dnow
                self._tgt_age = 0
            else:
                self._tgt_age += 1
                if self._tgt_age > 40:
                    self._blacklist.add((tr, tc))
                    self._target = None
                    self._best_d = None
                    self._tgt_age = 0

        if self._target is None:
            return np.array([0.0, 0.6 * self.w_max])

        tr, tc = self._target
        tx = self.bounds[0] + (tc + 0.5) * self.res
        ty = self.bounds[2] + (tr + 0.5) * self.res
        ang = float(np.arctan2(ty - env.robot_pose()[1], tx - env.robot_pose()[0]) - env.robot_pose()[2])
        ang = (ang + np.pi) % (2 * np.pi) - np.pi

        lidar = obs["lidar"]
        n = len(lidar)
        fwd = beam_index(ang, n)
        front_d = float(np.min(lidar[[(fwd + k) % n for k in range(-5, 6)]]))
        v = self.v_max * float(np.clip(1.0 - abs(ang) / np.pi * 1.8, 0.0, 1.0))
        w = float(np.clip(2.0 * ang, -self.w_max, self.w_max))
        # 与 OraclePolicy 同样的教训：只减速、不改写转向（改写会与目标转向抵消）。
        if front_d < 0.45:
            v *= float(np.clip((front_d - 0.18) / 0.27, 0.0, 1.0))
        return np.array([v, w])


class RandomTargetPolicy(OraclePolicy):
    """无语义对照（推荐的主力对照）：**导航机制与 oracle 完全相同**，唯一区别是
    规划目标从空闲栅格中随机选取，而非由指令指代的物体。

    这样做的实验价值：把「语义/语言理解」的增益从「导航与规划能力」中干净地
    分离出来 —— 若 VLA 显著优于本策略，增益只能归因于语义；若持平，说明任务
    退化为纯探索。

    这一条比 `frontier` 更本质：frontier 同时改变了「目标选择」与「导航机制」
    两个变量，且其实测探索效率很差（TL ≈ 0.10），不适合作为主要对照。
    """

    name = "random_target"

    def __init__(self, replan_steps=60, min_dist=1.2, reach_eps=0.6, seed=0, **kw):
        super().__init__(**kw)
        self.replan_steps = replan_steps
        self.min_dist = min_dist
        self.reach_eps = reach_eps
        self._rng = np.random.default_rng(seed)
        self._cur_goal = None
        self._age = 0

    def reset(self, env=None):
        super().reset(env)
        self._cur_goal = None
        self._age = 0

    def _pick(self, env):
        sc = env.scene
        H, W = sc.occ.shape
        m = int(np.ceil(0.35 / CELL))
        for _ in range(400):
            r = int(self._rng.integers(m, max(m + 1, H - m)))
            c = int(self._rng.integers(m, max(m + 1, W - m)))
            if env.plan_grid[r, c]:
                continue
            p = np.array([sc.bounds[0] + (c + 0.5) * CELL,
                          sc.bounds[2] + (r + 0.5) * CELL])
            if float(np.linalg.norm(p - env.robot_pose()[:2])) < self.min_dist:
                continue
            return p
        return np.asarray(env.task["goal"], dtype=float).copy()

    def _goal(self, env):
        self._age += 1
        need = (self._cur_goal is None or self._age > self.replan_steps
                or float(np.linalg.norm(self._cur_goal - env.robot_pose()[:2])) < self.reach_eps)
        if need:
            self._cur_goal = self._pick(env)
            self._age = 0
            self._field = None          # 目标变了 → 强制重算距离场
        return self._cur_goal


class ReactiveGoalPolicy:
    """反应式示范策略：动作**只依赖当前观测**，因此采出的示范是「单帧可学」的。

    为什么要它（见 `sim/analyze_observability.py` 的实测）：
        `OraclePolicy` 走的是**真值栅格图上的测地距离场 + 避障势场** —— 角速度由
        全局地图决定。实测其 ω 的符号与「目标方位角」**完全不相关**
        （目标可见时一致率仅 52.5%，相关系数 −0.001；符号熵 0.998 ≈ 抛硬币），
        且目标只有 **29.5%** 的时间在视野内。
        用这种示范做模仿学习，策略只能收敛到条件均值：表现为
        「开环 MAE 单调下降、动作统计量几乎与示范一致、闭环成功率却为 0」。

    本策略把动作限制为**可观测量的函数**：
        ① 目标在相机视场内 → 朝目标方位转向，按角度与距离衰减前进
        ② 目标不可见       → 朝激光最开阔方向转向 + 低速前进（墙跟随式探索）
        ③ 两种情况均叠加避障项（前扇区最近距离）
    代价：不做全局规划，成功率低于 oracle；收益：示范是可学的。

    注意：判断「目标是否在视野内」用真值方位角 **与相机视场比较** —— 这等价于
    「相机能不能看到目标」，是策略自己也能获得的信息，不构成特权信息。
    """

    name = "reactive"

    def __init__(self, v_max=V_MAX, w_max=W_MAX, k_ang=1.8,
                 hfov_deg=CAMERA_HFOV_DEG, avoid_dist=0.38, stop_radius=0.28,
                 slow_radius=0.9, search_v=0.45, max_sight=8.0,
                 avoid_w_frac=0.7, max_dw=0.22):
        self.v_max = v_max
        self.w_max = w_max
        self.k_ang = k_ang
        self.hfov_deg = hfov_deg
        self.avoid_dist = avoid_dist
        self.stop_radius = stop_radius
        self.slow_radius = slow_radius
        self.search_v = search_v
        self.max_sight = max_sight
        self.avoid_w_frac = avoid_w_frac   # 避障转向用比例而非满速，降低饱和
        self.max_dw = max_dw               # 角速度速率限制（rad/s 每步），使示范平滑
        self._prev_w = 0.0

    def reset(self, env=None):
        self._prev_w = 0.0

    def act(self, obs, env):
        lid = np.asarray(obs["lidar"], dtype=float)
        n = len(lid)
        ang = np.linspace(-np.pi, np.pi, n, endpoint=False)   # 与 lidar_scan 一致
        pose = env.robot_pose()
        g = np.asarray(env.task["goal"], dtype=float)
        dv = g - pose[:2]
        dist = float(np.linalg.norm(dv))
        bearing = float(np.arctan2(dv[1], dv[0]) - pose[2])
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        half = np.deg2rad(self.hfov_deg) / 2.0
        visible = (abs(bearing) <= half) and (dist < self.max_sight)

        front = np.abs(ang) <= np.deg2rad(35.0)
        d_front = float(lid[front].min()) if front.any() else LIDAR_MAX_RANGE

        if visible:
            w = float(np.clip(self.k_ang * bearing, -self.w_max, self.w_max))
            v = self.v_max * max(0.30, 1.0 - abs(bearing) / half)
            if dist <= self.stop_radius:
                v = 0.0
                w = float(np.clip(self.k_ang * bearing, -self.w_max, self.w_max))
            else:
                v = min(v, self.v_max * max(0.0, (dist - self.stop_radius) / self.slow_radius))
        else:
            # 朝激光最开阔方向转（环形平滑，避免单束噪声）
            k = 11
            sm = np.convolve(np.concatenate([lid[-k:], lid, lid[:k]]),
                             np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]
            w = float(np.clip(self.k_ang * float(ang[int(np.argmax(sm))]),
                              -self.w_max, self.w_max))
            v = self.search_v * self.v_max

        # 避障优先：前扇区受阻则停止前进，转向更开阔的一侧。
        # **但目标本身就是障碍物** —— 接近目标时若仍触发避障，会被自己的避障项推开，
        # 永远进不了成功半径（实测 SR=0% 而 TL=1.12 就是这个原因）。
        near_goal = visible and dist < 1.0
        if (not near_goal) and d_front < self.avoid_dist:
            lm = (ang > 0.10) & (ang <= 0.75 * np.pi)
            rm = (ang < -0.10) & (ang >= -0.75 * np.pi)
            cl = float(lid[lm].mean()) if lm.any() else 0.0
            cr = float(lid[rm].mean()) if rm.any() else 0.0
            w = self.avoid_w_frac * self.w_max * (1.0 if cl >= cr else -1.0)
            v = 0.0

        # 角速度速率限制：抑制抖动，使示范更平滑、更易学。
        # 上一步动作已包含在观测量中（state 含 last_action），故这不引入特权信息。
        w = float(np.clip(w, self._prev_w - self.max_dw, self._prev_w + self.max_dw))
        w = float(np.clip(w, -self.w_max, self.w_max))
        self._prev_w = w

        return np.array([float(np.clip(v, 0.0, self.v_max)), w], dtype=np.float32)


class SmoothOraclePolicy(OraclePolicy):
    """`oracle` + 角速度速率限制与幅度上限。

    动机：`oracle` 的导航能力**明显强于** `reactive`（SR 40.0% vs 13.3%），
    是更有价值的示范来源；它的唯一问题是 **79.6% 的步处于角速度饱和**（原地打转），
    导致示范信号噪声过大、策略学不动角速度（实测学习后 |ω| 仍偏 2.9 倍）。

    本类保留 oracle 的测地场规划与避障逻辑，只对输出做两件事：
      ① 幅度上限 `w_cap_frac · w_max` —— 主动避开饱和区
      ② 速率限制 `max_dw` —— 抑制势场/避障项切换造成的抖动

    上一步动作已在观测量中（`obs["state"]` 含 last_action），故不引入特权信息。
    目标是得到「导航能力接近 oracle、但动作可学」的示范策略。
    """

    name = "oracle_smooth"

    def __init__(self, max_dw=0.22, w_cap_frac=0.85, **kw):
        super().__init__(**kw)
        self.max_dw = max_dw
        self.w_cap = w_cap_frac * self.w_max
        self._pw = 0.0

    def reset(self, env=None):
        super().reset(env)
        self._pw = 0.0

    def act(self, obs, env):
        a = super().act(obs, env)
        w = float(np.clip(float(a[1]), -self.w_cap, self.w_cap))
        w = float(np.clip(w, self._pw - self.max_dw, self._pw + self.max_dw))
        self._pw = w
        return np.array([float(a[0]), w], dtype=np.float32)


class SweepPolicy:
    """扫描式示范策略：目标可见则前往，否则**匀速旋转扫描 + 低速前进**。

    为什么加它（标定结论，见 `主创新A_诊断报告.md` §4.4）：
      · `oracle` / `planner` 在障碍密集场景会卡死（v=0 占 65–81%），
        作为「可达上界」太低，没有测量余量；
      · `reactive` 在开阔场地里用「朝激光最开阔方向转」，结果**贴着墙转圈**，
        永远看不到场地中央的目标 —— NE 只有 1.70 m 但成功率上不去；
      · `random_target` 反而因为一直在动、覆盖面大，拿到了与有语义策略相当的成功率。

    本策略用一个确定性的搜索动作取代启发式探索：**匀速自转 + 低速直行**。
    因为相机 HFOV 只有 64.5°，匀速扫描保证目标在约 2π/ω_scan 秒内必定进入视野；
    一旦看到就切换成「朝目标纯追踪 + 停靠减速」。这使示范策略在开阔场地里
    既**覆盖地面**又**不漏看目标**，从而把成功率抬到可作为上界的水平。
    """

    name = "sweep"

    def __init__(self, v_cruise=0.16, w_scan=1.15, k_ang=2.2, hfov_deg=CAMERA_HFOV_DEG,
                 stop_eps=0.14, dock_dist=0.8, max_dw=0.55, emergency=0.26,
                 max_sight=9.0, v_max=V_MAX, w_max=W_MAX, scan_dir=1.0):
        self.v_cruise = v_cruise
        self.w_scan = w_scan
        self.k_ang = k_ang
        self.hfov_deg = hfov_deg
        self.stop_eps = stop_eps
        self.dock_dist = dock_dist
        self.max_dw = max_dw
        self.emergency = emergency
        self.max_sight = max_sight
        self.v_max = v_max
        self.w_max = w_max
        self.scan_dir = scan_dir
        self._pw = 0.0
        self._last_xy = None
        self._stuck = 0

    def reset(self, env=None):
        self._pw = 0.0
        self._last_xy = None
        self._stuck = 0

    def act(self, obs, env):
        lid = np.asarray(obs["lidar"], dtype=float)
        n = len(lid)
        ang = np.linspace(-np.pi, np.pi, n, endpoint=False) if n else np.zeros(0)
        pose = env.robot_pose()
        g = np.asarray(env.task["goal"], dtype=float)
        dv = g - pose[:2]
        dist = float(np.linalg.norm(dv))
        bearing = float(np.arctan2(dv[1], dv[0]) - pose[2])
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        half = np.deg2rad(self.hfov_deg) / 2.0
        visible = (abs(bearing) <= half) and (dist < self.max_sight)

        if dist <= self.stop_eps:
            self._pw = 0.0
            return np.array([0.0, 0.0], dtype=np.float32)

        if visible:
            w = float(np.clip(self.k_ang * bearing, -self.w_max, self.w_max))
            v = self.v_max * min(1.0, max(0.0, (dist - self.stop_eps) / self.dock_dist))
        else:
            w = self.scan_dir * self.w_scan        # 匀速扫描，保证不漏看
            v = self.v_cruise

        front = np.abs(ang) <= np.deg2rad(28.0)
        d_front = float(lid[front].min()) if front.any() else LIDAR_MAX_RANGE
        if d_front < self.emergency:
            v = 0.0
            w = self.scan_dir * self.w_max * 0.9

        xy = pose[:2]
        moved = (self._last_xy is None
                 or float(np.linalg.norm(xy - self._last_xy)) >= 0.006)
        if v > 0.02:
            self._stuck = 0 if moved else self._stuck + 1
        else:
            self._stuck = 0
        self._last_xy = xy.copy()
        if self._stuck > 15:
            v = 0.0
            w = self.scan_dir * self.w_max * 0.9

        w = float(np.clip(w, self._pw - self.max_dw, self._pw + self.max_dw))
        w = float(np.clip(w, -self.w_max, self.w_max))
        self._pw = w
        return np.array([float(np.clip(v, 0.0, self.v_max)), w], dtype=np.float32)


class LocalPlanner:
    """**只用激光 + 里程计**的局部绕障逼近（不读任何真值地图）。

    ## 动机（见 `主创新A_诊断报告.md` §4.15.4）

    把检测器从 64.8% 召回提到 100%，示范策略 SR 反而从 37.5% 掉到 **25.0%**、
    碰撞翻倍 —— 因为 `sweep_det` 只会**直冲目标**。整张实验表都指向同一个结论：
    **瓶颈在「逼近 + 避障」，不在感知。**

    ## 做法

    1. 把激光按里程计累积成占据栅格（与 `FrontierPolicy` 同一套 DDA 建图），
       并对障碍做膨胀，留出安全间隙；
    2. 每隔 `replan_every` 步，在机器人附近的**局部窗口**内做一次 BFS；
    3. 沿 BFS 距离场的梯度取一个前视航点（默认 0.7 m），转向它。

    **不依赖真值地图的细节**：栅格边界以**出生点为中心固定外扩**，
    不读 `scene.bounds`（那相当于告知建筑尺寸）。障碍全部来自激光观测。
    """

    def __init__(self, res=0.10, max_range=12.0, inflate=0.28, half_size=15.0,
                 window=6.0, lookahead=0.7, replan_every=6, v_max=V_MAX, w_max=W_MAX):
        self.res = res
        self.max_range = max_range
        self.inflate = inflate
        self.half_size = half_size
        self.window = window
        self.lookahead = lookahead
        self.replan_every = replan_every
        self.v_max = v_max
        self.w_max = w_max
        self.grid = None
        self.bounds = None
        self._n_inf = 1
        self._t = 0
        self._wp = None

    def reset(self, env=None):
        self.grid = None
        self.bounds = None
        self._t = 0
        self._wp = None

    # ---------- 建图 ----------
    def _ensure(self, pose):
        if self.grid is not None:
            return
        x, y = float(pose[0]), float(pose[1])
        self.bounds = (x - self.half_size, x + self.half_size,
                       y - self.half_size, y + self.half_size)
        W = int(np.ceil(2 * self.half_size / self.res))
        self.grid = np.full((W, W), -1, dtype=np.int8)     # -1 未知 0 自由 1 占据
        self._n_inf = max(1, int(round(self.inflate / self.res)))

    def _w2c(self, pts):
        c = np.floor((pts[:, 0] - self.bounds[0]) / self.res).astype(np.int64)
        r = np.floor((pts[:, 1] - self.bounds[2]) / self.res).astype(np.int64)
        return r, c

    def _integrate(self, pose, lidar):
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
        n = len(lidar)
        ang = np.linspace(-np.pi, np.pi, n, endpoint=False) + th
        ds = np.arange(0.0, self.max_range, self.res)
        px = x + np.cos(ang)[:, None] * ds[None, :]
        py = y + np.sin(ang)[:, None] * ds[None, :]
        r, c = self._w2c(np.stack([px.ravel(), py.ravel()], 1))
        H, W = self.grid.shape
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        r, c = r[ok], c[ok]
        d_flat = np.broadcast_to(ds[None, :], (n, ds.size)).ravel()[ok]
        lid_flat = np.broadcast_to(np.asarray(lidar)[:, None], (n, ds.size)).ravel()[ok]
        hit = lid_flat < self.max_range - 1e-6
        free = d_flat < (lid_flat - self.res)
        self.grid[r[free], c[free]] = 0
        occ = hit & (d_flat >= (lid_flat - self.res)) & (d_flat <= (lid_flat + self.res))
        self.grid[r[occ], c[occ]] = 1
        rr, cc = self._w2c(np.array([[x, y]]))
        if 0 <= rr[0] < H and 0 <= cc[0] < W:
            k = self._n_inf
            self.grid[max(0, rr[0] - k):rr[0] + k + 1,
                      max(0, cc[0] - k):cc[0] + k + 1] = 0

    def _inflated_blocked(self):
        """障碍膨胀后的「不可通行」掩码（未知格也**不算**可通过）。"""
        from scipy import ndimage
        occ = (self.grid == 1)
        k = self._n_inf
        return ndimage.binary_dilation(occ, np.ones((2 * k + 1, 2 * k + 1), bool))

    # ---------- 规划 ----------
    def _bfs(self, pose, goal_xy):
        """在机器人附近的局部窗口内做 BFS，返回前视航点（世界坐标）或 None。"""
        from collections import deque

        H, W = self.grid.shape
        rr, cc = self._w2c(np.array([[pose[0], pose[1]]]))
        r0, c0 = int(rr[0]), int(cc[0])
        if not (0 <= r0 < H and 0 <= c0 < W):
            return None
        wr = int(self.window / self.res)
        ra, rb = max(0, r0 - wr), min(H, r0 + wr + 1)
        ca, cb = max(0, c0 - wr), min(W, c0 + wr + 1)

        gr, gc = self._w2c(np.array([[goal_xy[0], goal_xy[1]]]))
        gr, gc = int(gr[0]) - ra, int(gc[0]) - ca
        h, w = rb - ra, cb - ca
        # **未知格也算可通过**：否则一开始地图近乎全空、BFS 直接失败
        # （这与「未知格不可通行」的两难里，选择「乐观」一侧，靠障碍膨胀保安全）。
        blocked_i = (self.grid[ra:rb, ca:cb] == 1)
        from scipy import ndimage
        k = self._n_inf
        blocked_i = ndimage.binary_dilation(blocked_i, np.ones((2 * k + 1, 2 * k + 1), bool))

        if not (0 + 1 <= r0 - ra < h - 1 and 0 + 1 <= c0 - ca < w - 1):
            return None
        dist = np.full((h, w), -1, dtype=np.int32)
        srr, scc = r0 - ra, c0 - ca
        if blocked_i[srr, scc]:
            return None
        # 紧贴的障碍（膨胀后）也放行起点 3×3，避免被自身膨胀格困住
        blocked_i[max(0, srr - 1):srr + 2, max(0, scc - 1):scc + 2] = False
        q = deque([(srr, scc)])
        dist[srr, scc] = 0
        while q:
            r, c = q.popleft()
            d = dist[r, c] + 1
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] < 0 and not blocked_i[nr, nc]:
                    dist[nr, nc] = d
                    q.append((nr, nc))

        # 目标格可能落在障碍/未知上 → 取窗口内距目标最近的**可达**格
        if 0 <= gr < h and 0 <= gc < w and dist[gr, gc] >= 0:
            tr, tc = gr, gc
        else:
            ys, xs = np.nonzero(dist >= 0)
            if ys.size == 0:
                return None
            k2 = np.argmin((ys - gr) ** 2 + (xs - gc) ** 2)
            tr, tc = int(ys[k2]), int(xs[k2])

        # 沿梯度回溯，取距起点约 lookahead 的格作为航点
        need = int(self.lookahead / self.res)
        r, c = tr, tc
        for _ in range(need):
            if dist[r, c] <= 0:
                break
            d = dist[r, c]
            moved = False
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and 0 <= dist[nr, nc] < d:
                    r, c = nr, nc
                    moved = True
                    break
            if not moved:
                break
        return np.array([self.bounds[0] + (ca + c + 0.5) * self.res,
                         self.bounds[2] + (ra + r + 0.5) * self.res])

    def step(self, pose, lidar, goal_xy):
        """返回 (v, w)；规划失败返回 None（交给调用方回退到直冲）。"""
        self._ensure(pose)
        self._integrate(pose, lidar)
        self._t += 1
        if self._wp is None or self._t % self.replan_every == 1:
            self._wp = self._bfs(pose, goal_xy)
        if self._wp is None:
            return None
        v = self._wp - pose[:2]
        dist = float(np.linalg.norm(v))
        if dist < 1e-3:
            return None
        ang = float(np.arctan2(v[1], v[0]) - pose[2])
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        return ang


# ------------------------------------------------------------------ 指令解析
_CLS_CN2KEY = {"椅子": "chair", "桌子": "table", "沙发": "sofa", "床": "bed",
               "书架": "shelf", "植物": "plant", "箱子": "box", "电视": "tv"}
# 方位词 → 世界坐标下的目标角度（+x = 右，+y = 前，与 env._rel_word 的定义一致）
_REL_ANG = {"右边": 0.0, "前边": np.pi / 2, "后边": -np.pi / 2, "左边": np.pi}


def parse_instruction(text):
    """从**指令文本**解析目标描述。

    刻意只读文本、不读 `env.task` 的字段 —— 真实系统也只能拿到指令字符串，
    读 task 等于又绕回特权信息。
    """
    want_color = next((c for c in COLORS if c in text), None)
    found = []
    for cn, key in _CLS_CN2KEY.items():
        pos = text.find(cn)
        if pos >= 0:
            found.append((pos, key))
    found.sort()
    want_cls = found[-1][1] if found else None
    ref_cls = found[0][1] if len(found) >= 2 else None
    rel = next((w for w in _REL_ANG if w in text), None)
    return want_color, want_cls, ref_cls, rel


class SweepDetectorPolicy:
    """扫描式示范策略 —— **完全基于图像**，不使用任何真值目标信息。

    与 `sweep` 的唯一区别（也是本策略存在的全部理由）：

    | | `sweep`（旧） | `sweep_det`（本类） |
    |---|---|---|
    | 方位来源 | **真值** `env.task["goal"]` | `perception.detect(obs["image"])` |
    | 可见性判据 | `\\|bearing\\| ≤ HFOV/2`（真值） | 图像里有没有检测到匹配物体 |
    | 距离来源 | **真值** ‖goal − pose‖ | 单目几何 `f·cam_h/(r1−horizon)` |

    为什么必须改（见 `主创新A_诊断报告.md` §4.10）：旧 `sweep` 的动作是
    `f(图像, 目标位置)` 的函数，而**目标位置不在图像里**，这对模仿者是
    **一对多**映射（同一张图、目标不同则动作不同），只能学到条件平均。
    实测 VLA 闭环 13.3% 与「图像+位姿+指令」的公平上界 13.0% **完全相等**，
    正是这个原因的后果。

    改成本策略后，「观测 → 动作」成为良定义的映射，模仿才有意义。
    检测器不完美（召回约 62%、有方位误差）会**降低示范策略自身的成功率**，
    但这不影响可学性 —— 关键是动作只依赖图像。
    """

    name = "sweep_det"

    def __init__(self, v_cruise=0.16, w_scan=1.15, k_ang=2.2,
                 stop_eps=0.14, dock_dist=0.8, max_dw=0.55, emergency=0.26,
                 v_max=V_MAX, w_max=W_MAX, scan_dir=1.0, det_every=2,
                 rel_tol_deg=50.0, use_det_dist=True, det_kw=None,
                 use_lidar_dist=False,
                 # `mem_ttl` 默认 25：实测 200 回合下 16.7% → 33%+，
                 # 是当前唯一稳定有效的组件（物体静止，短期记忆能补上「暂时看不见」）。
                 mem_ttl=25, mem_merge=0.7, scan_arc=1.5 * 3.141592653589793,
                 relocate_steps=25, nn_det=None, use_plan=False, plan_kw=None,
                 use_map=False):
        self.v_cruise = v_cruise
        self.w_scan = w_scan
        self.k_ang = k_ang
        self.stop_eps = stop_eps
        self.dock_dist = dock_dist
        self.max_dw = max_dw
        self.emergency = emergency
        self.v_max = v_max
        self.w_max = w_max
        self.scan_dir = scan_dir
        # 检测较贵（每帧 ~10 ms），隔帧做即可 —— 扫描阶段本来就在匀速转
        self.det_every = int(max(1, det_every))
        self.rel_tol_deg = rel_tol_deg
        self.use_det_dist = use_det_dist
        # `use_lidar_dist`：用**激光测得的表面距**替代单目中心距来做停靠判据。
        # 动机：单目地面公式在色块底边触到图像下沿时饱和于 0.752 m，
        # 而成功判据是 0.30 m —— 该公式在判据附近失效（数据集里检测距离的最小值
        # 恰为 0.74 m 可证），所以 `stop_eps`/`dock_dist` 的减速逻辑**永不触发**。
        self.use_lidar_dist = use_lidar_dist
        self._last_surf = float('nan')
        self.det_kw = dict(det_kw or {})
        # `nn_det` 给的是 checkpoint 路径时，改用**监督学习的检测器**（见 sim/nn_detector.py）。
        # 它比颜色-连通域强得多（同口径：召回 64.8%→100%、判类 74.3%→85.2%、颜色 88.6%→100%）。
        self._nn = None
        if nn_det:
            from .nn_detector import NNDetector
            self._nn = NNDetector(nn_det)
        # **逼近阶段改用局部绕障规划**（只用激光 + 里程计）。
        # 见 主创新A_诊断报告.md §4.15.4：检测器变好反而让策略变差，根因就是只会直冲。
        self._planner = LocalPlanner(**(plan_kw or {})) if use_plan else None
        # **在线语义地图**：把「看到过的物体」永久记在世界坐标里。
        # 这是补上「看不见也知道方向」那条通道的显式版本（报告 §4.16.3）。
        self._map = None
        if use_map:
            from .semantic_map import SemanticMap
            self._map = SemanticMap()
        self.mem_ttl = int(mem_ttl)
        self.mem_merge = float(mem_merge)
        # **高效搜索**：原地匀速自转会让「扫描」动作长期占据多数类，
        # 使模仿者塌缩成「永远扫描」（实测占 72% 的步数，闭环 0.0%）。
        # 改成「转够 scan_arc 仍未检出就换观测点」，压低扫描占比。
        self.scan_arc = float(scan_arc)
        self.relocate_steps = int(relocate_steps)
        self._scan_acc = 0.0
        self._relocate = 0
        self._pw = 0.0
        self._t = 0
        self._last = None                  # 缓存上一次检测 (bearing, dist)
        self._mem = []                     # 世界坐标下的检测记忆

    def reset(self, env=None):
        self._pw = 0.0
        self._t = 0
        self._last = None
        self._mem = []
        self._scan_acc = 0.0
        self._relocate = 0
        if self._planner is not None:
            self._planner.reset()
        if self._map is not None:
            self._map.reset()

    def _mem_update(self, dets, env):
        """把本帧检测投影到**世界坐标**并累积成短期记忆。

        单帧检测会漏检（同色物体合并、被别的物体遮挡）；但机器人转动后，
        同一物体在别的角度往往能被检出。按世界坐标累积历史检测，
        可显著降低「目标明明在附近却检测不到」的比例 —— 真实感知系统正是
        这么做多帧关联的。

        **这不是特权信息**：世界位置完全由「自车位姿（里程计）+ 相机测得的
        方位与距离」算出，不涉及任何真值。但它确实让示范策略拥有**超出单帧**
        的记忆能力，属于「能力差」而非「信息差」。故默认关闭（`mem_ttl=0`），
        便于做「有/无记忆」的对照实验。
        """
        if self.mem_ttl <= 0:
            return
        now = self._t
        keep = [m for m in self._mem if now - m[5] <= self.mem_ttl]
        for dt in dets:
            a = env.robot_pose()[2] + dt.bearing
            p = env.robot_pose()[:2] + dt.dist * np.array([np.cos(a), np.sin(a)])
            for m in keep:
                if m[1] == dt.color and float(np.hypot(*(p - m[0]))) < self.mem_merge:
                    m[0] = 0.5 * (m[0] + p)          # 位置指数平滑，抑制测距抖动
                    m[3], m[4], m[5] = dt.half, dt.height, now
                    break
            else:
                keep.append([p, dt.color, dt.cls, dt.half, dt.height, now])
        self._mem = keep

    def _pick(self, obs, env):
        """在图像里选出指令指代的物体；返回 (bearing, dist) 或 None。

        同时把命中的**激光表面距**记到 `self._last_surf`（供 `use_lidar_dist` 用）。
        """
        from .perception import detect, match

        want_color, want_cls, ref_cls, rel = parse_instruction(env.task["instruction"])
        if self._nn is not None:
            dets = self._nn.detect(obs["image"])
        else:
            dets = detect(obs["image"], lidar=obs.get("lidar"), **self.det_kw)
        self._last_surf = float('nan')
        self._mem_update(dets, env)
        if self._map is not None:
            self._map.update(env.robot_pose(), obs.get("lidar"), dets, self._t)
        if not dets:
            return self._from_map(env, want_color, want_cls) or \
                self._from_memory(env, want_color, want_cls, ref_cls, rel)

        if rel is not None and ref_cls is not None:
            # L3：先用世界坐标把「参考物 + 方位」解出来，再挑目标
            # 世界位置 = 自车位姿（里程计可观测） + 相机测得的方位与距离
            def wpos(d):
                a = env.robot_pose()[2] + d.bearing
                return env.robot_pose()[:2] + d.dist * np.array([np.cos(a), np.sin(a)])

            refs = [d for d in dets if match([d], cls=ref_cls) is not None]
            tgts = [d for d in dets if match([d], color=want_color, cls=want_cls) is not None]
            tgt_ang = _REL_ANG[rel]
            tol = np.deg2rad(self.rel_tol_deg)
            best = None
            for t in tgts:
                for r in refs:
                    v = wpos(t) - wpos(r)
                    if np.hypot(*v) < 0.4:
                        continue
                    a = float(np.arctan2(v[1], v[0]))
                    if abs((a - tgt_ang + np.pi) % (2 * np.pi) - np.pi) < tol:
                        if best is None or t.dist < best.dist:
                            best = t
                        break
            if best is not None:
                a = env.robot_pose()[2] + best.bearing
                return (best.bearing, best.dist,
                        env.robot_pose()[:2] + best.dist * np.array([np.cos(a), np.sin(a)]))

        d = match(dets, color=want_color, cls=want_cls)
        if d is None:
            return self._from_memory(env, want_color, want_cls, ref_cls, rel)
        # 记下命中的**激光表面距**（`use_lidar_dist` 的停靠判据要用）
        self._last_surf = float(getattr(d, "surf", float("nan")))
        a = env.robot_pose()[2] + d.bearing
        return (d.bearing, d.dist,
                env.robot_pose()[:2] + d.dist * np.array([np.cos(a), np.sin(a)]))

    def _from_map(self, env, want_color, want_cls):
        """**当前帧没看到目标时，从语义地图里取位置** —— 这是「看不见也知道方向」。

        与 `_from_memory` 的区别：地图里的实例是**永久保留**的，且按
        （颜色, 类别）双重关联（`_from_memory` 只按颜色），因此更不容易记错。
        """
        if self._map is None:
            return None
        cand = self._map.query(color=want_color, cls=want_cls, pose=env.robot_pose())
        if not cand:
            return None
        v = cand[0][0] - np.asarray(env.robot_pose()[:2])
        dist = float(np.hypot(*v))
        a = float(np.arctan2(v[1], v[0]) - env.robot_pose()[2])
        a = (a + np.pi) % (2 * np.pi) - np.pi
        return a, dist, np.asarray(cand[0][0], dtype=np.float64).copy()

    def _from_memory(self, env, want_color, want_cls, ref_cls, rel):
        """单帧没检出时回退到**世界坐标下的检测记忆**。

        记忆条目的世界位置仍然有效（物体静止），但机器人已移动/转动，
        所以方位与距离必须按**当前位姿**重算 —— 这正是「用记忆」而非「用真值」。
        """
        from .perception import _CLASSES, _LOG_SHAPE

        if not self._mem:
            return None
        cand = []
        for m in self._mem:
            if want_color is not None and m[1] != want_color:
                continue
            if want_cls is not None and m[2] != want_cls:
                q = np.log(np.array([max(m[4], 1e-3), max(m[3], 1e-3)]))
                tgt = _LOG_SHAPE[_CLASSES.index(want_cls)]
                if float(np.linalg.norm(tgt - q)) > 0.5:      # 同 match 的软匹配
                    continue
            cand.append(m)
        if not cand:
            return None

        if rel is not None and ref_cls is not None:
            refs = [m for m in self._mem if m[2] == ref_cls]
            tgt_ang = _REL_ANG[rel]
            tol = np.deg2rad(self.rel_tol_deg)
            kept = []
            for m in cand:
                for r in refs:
                    v = m[0] - r[0]
                    if np.hypot(*v) < 0.4:
                        continue
                    a = float(np.arctan2(v[1], v[0]))
                    if abs((a - tgt_ang + np.pi) % (2 * np.pi) - np.pi) < tol:
                        kept.append(m)
                        break
            if kept:
                cand = kept

        best = min(cand, key=lambda m: float(np.hypot(*(m[0] - env.robot_pose()[:2]))))
        v = best[0] - env.robot_pose()[:2]
        a = float(np.arctan2(v[1], v[0]) - env.robot_pose()[2])
        a = (a + np.pi) % (2 * np.pi) - np.pi
        return a, float(np.hypot(*v)), np.asarray(best[0], dtype=np.float64).copy()

    def act(self, obs, env):
        lid = np.asarray(obs["lidar"], dtype=float)
        n = len(lid)
        ang = np.linspace(-np.pi, np.pi, n, endpoint=False) if n else np.zeros(0)
        pose = env.robot_pose()

        self._t += 1
        if self._last is None or self._t % self.det_every == 1:
            self._last = self._pick(obs, env)

        hit = self._last
        if hit is not None:
            bearing, dist = float(hit[0]), float(hit[1])
            # **停靠判据用的距离**：默认单目中心距；开启后改用激光表面距
            # （近距离唯一有效，且正是成功判据所用的量）。
            if self.use_lidar_dist and np.isfinite(self._last_surf):
                dist = float(self._last_surf)
            if dist <= self.stop_eps:
                self._pw = 0.0
                return np.array([0.0, 0.0], dtype=np.float32)
            # 先用**局部绕障规划**给出航向；规划失败（地图未建好/目标不可达）
            # 再回退到直冲，保证行为不会因规划器失效而完全停摆。
            if self._planner is not None and len(hit) > 2:
                # **变量名不能叫 `ang`**：上面 `ang` 是激光束的角度数组，
                # 覆盖它会让后面的 `np.abs(ang)` 拿到 None（规划失败时）而崩溃。
                hd = self._planner.step(pose, lid, hit[2])
                if hd is not None:
                    bearing = hd
            w = float(np.clip(self.k_ang * bearing, -self.w_max, self.w_max))
            v = self.v_max * min(1.0, max(0.0, (dist - self.stop_eps) / self.dock_dist))
        else:
            # 扫描：转够 scan_arc 仍未检出 → 前进换观测点（避免原地空转占满多数类）
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

        front = np.abs(ang) <= np.deg2rad(28.0)
        d_front = float(lid[front].min()) if front.any() else LIDAR_MAX_RANGE
        if d_front < self.emergency:
            v = 0.0
            w = self.scan_dir * self.w_max * 0.9

        w = float(np.clip(w, self._pw - self.max_dw, self._pw + self.max_dw))
        w = float(np.clip(w, -self.w_max, self.w_max))
        self._pw = w
        return np.array([float(np.clip(v, 0.0, self.v_max)), w], dtype=np.float32)


POLICIES = {"random": RandomPolicy, "frontier": FrontierPolicy,
            "oracle": OraclePolicy, "random_target": RandomTargetPolicy,
            "reactive": ReactiveGoalPolicy, "oracle_smooth": SmoothOraclePolicy,
            "planner": PlannerPolicy, "planner_oracle": PlannerOraclePolicy,
            "planner_random": PlannerRandomTargetPolicy, "sweep": SweepPolicy,
            "sweep_det": SweepDetectorPolicy,
            # 窗口测地场上界（新增固定布局用；见 planner.GeodesicWindowPolicy）
            "geodesic": GeodesicWindowPolicy,
            "geodesic_random": GeodesicRandomTargetPolicy,
            # 经典 A* 传统规划对照（见 planner.AStarPolicy；E1c 路径几何对比 /
            # 第 3 章算法选型论证用），astar_random 为同口径无语义对照
            "astar": AStarPolicy,
            "astar_random": AStarRandomTargetPolicy}


def make_policy(name, **kw):
    if name not in POLICIES:
        raise ValueError(f"未知基线: {name}（可选 {list(POLICIES)}）")
    return POLICIES[name](**kw)
