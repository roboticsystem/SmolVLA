"""基于测地距离场的**纯追踪规划器**，用于替换 `OraclePolicy`。

为什么要换（见 `主创新A_诊断报告.md` §3.1、§4.4）：
    `OraclePolicy` 用「势场梯度 + 独立避障项」**直接**生成 (v, ω)。两项连续耦合，
    导致：
      · **79.6% 的步处于角速度饱和**（原地满速打转），示范信号严重退化
      · 停靠精度差（`stop_radius=0.45`），够不到 0.30 m 的成功半径
      · 绝对成功率只有 30–40%，作为「可达上界」太低，没有测量余量

本策略把「路径」与「控制」解耦：
    ① 在**带间隙的膨胀栅格**上算测地距离场（已有 `env._shortest_path_len`）
    ② 沿梯度下降抽取**前视路点**（`lookahead` 米之外的一个点）
    ③ 用**纯追踪（pure pursuit）**朝该路点转向
    ④ 按「剩余距离 + 航向误差」调度速度，并在目标处**精确停靠**

因为路径本身已带间隙，**不需要独立的避障项**，角速度因此平滑（饱和率低），
又能精确停靠到 0.2–0.3 m —— 这正是拉开「有语义 vs 无语义」区分度的关键。
"""

from __future__ import annotations

import math

import numpy as np

from .env import SAFE_CLEARANCE, V_MAX, W_MAX, _dilate
from .sensors import LIDAR_MAX_RANGE
from .world import CELL


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class PlannerPolicy:
    """测地场 + 纯追踪 + 精确停靠。作为测试床的「可达上界」示范策略。"""

    name = "planner"

    def __init__(self, v_max=V_MAX, w_max=W_MAX, k_ang=2.0, lookahead=0.95,
                 dock_dist=0.9, stop_eps=0.18, slow_down=1.1, max_dw=0.60,
                 emergency=0.28, corner_slow=0.55, replan_every=1,
                 turn_only_deg=55.0, min_v_frac=0.40, min_v_deg=80.0):
        self.v_max = v_max
        self.w_max = w_max
        self.k_ang = k_ang
        self.lookahead = lookahead
        self.dock_dist = dock_dist          # 进入该距离后按比例减速
        self.stop_eps = stop_eps            # 小于该距离认为已到位
        self.slow_down = slow_down          # 航向误差导致的速度衰减系数
        self.max_dw = max_dw                # 角速度速率限制
        self.emergency = emergency          # 前向急停距离
        self.corner_slow = corner_slow
        self.replan_every = replan_every
        self.turn_only_deg = turn_only_deg    # 航向误差超过此角度则只转不走
        self.min_v_frac = min_v_frac          # 航向误差小于 min_v_deg 时的前进速度下限
        self.min_v_deg = min_v_deg
        self._field = None
        self._goal = None
        self._pw = 0.0
        self._t = 0
        self._last_xy = None
        self._stuck = 0

    # ---------------------------------------------------------- 内部工具
    def reset(self, env=None):
        self._field = None
        self._goal = None
        self._pw = 0.0
        self._t = 0
        self._last_xy = None
        self._stuck = 0

    def _field_for(self, env, goal):
        if (self._field is None or self._goal is None
                or not np.allclose(goal, self._goal)):
            self._field = env._shortest_path_len(goal)
            self._goal = np.asarray(goal, dtype=float).copy()
        return self._field

    @staticmethod
    def _cell_of(env, p):
        sc = env.scene
        r = (p[1] - sc.bounds[2]) / CELL
        c = (p[0] - sc.bounds[0]) / CELL
        return int(r), int(c)

    def _lookahead_point(self, env, field, pose):
        """沿距离场梯度下降，返回约 `lookahead` 米之外的世界坐标点。

        这是本策略与 `OraclePolicy` 的核心差别：oracle 只用了**当前位置**的梯度
        （等于每步都盯着脚边的方向），因此在小障碍附近频繁来回摆；
        用前视点可以把转向提前、平滑掉这些抖动。
        """
        H, W = field.shape
        r, c = self._cell_of(env, pose)
        r = int(np.clip(r, 0, H - 1)); c = int(np.clip(c, 0, W - 1))

        # **关键**：`plan_grid` 是按机器人半径膨胀过的，机器人自身所在格经常被判为
        # 「不可通行」（`field < 0`）。若此时直接用当前位置做梯度下降会立刻失败，
        # 表现为「原地不动、TL 极低」。因此在窗口内找**最近的合法格**作为起点
        # ——`OraclePolicy` 也是这样做的，此前漏掉这一步导致规划器 SR 反而低于 oracle。
        if field[r, c] < 0:
            R = 8
            r0, r1 = max(0, r - R), min(H, r + R + 1)
            c0, c1 = max(0, c - R), min(W, c + R + 1)
            sub = field[r0:r1, c0:c1]
            ok = sub >= 0
            if not ok.any():
                return np.asarray(env.task["goal"], dtype=float)
            rr, cc = np.nonzero(ok)
            k = int(np.argmin((rr + r0 - r) ** 2 + (cc + c0 - c) ** 2))
            r, c = int(rr[k] + r0), int(cc[k] + c0)

        traveled = 0.0
        for _ in range(400):
            best, bk = field[r, c], None
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        v = field[rr, cc]
                        if 0 <= v < best:
                            best, bk = v, (rr, cc)
            if bk is None:
                break
            dr, dc = bk[0] - r, bk[1] - c
            r, c = bk
            traveled += CELL * float(np.hypot(dr, dc))
            if traveled >= self.lookahead:
                break
        sc = env.scene
        return np.array([sc.bounds[0] + (c + 0.5) * CELL,
                         sc.bounds[2] + (r + 0.5) * CELL])

    def _escape(self, lid, ang):
        """朝左右更开阔的一侧转，并在连续卡死时翻转方向，避免在死角里来回顶。"""
        lm = (ang > 0.15) & (ang <= 0.7 * np.pi)
        rm = (ang < -0.15) & (ang >= -0.7 * np.pi)
        cl = float(lid[lm].mean()) if lm.any() else 0.0
        cr = float(lid[rm].mean()) if rm.any() else 0.0
        d = 1.0 if cl >= cr else -1.0
        if self._stuck > 34:                 # 长时间脱困失败 → 换一侧
            d = -d
            self._stuck = 14
        return d * self.w_max * 0.9

    # ---------------------------------------------------------- 控制
    def act(self, obs, env):
        self._t += 1
        lid = np.asarray(obs["lidar"], dtype=float)
        n = len(lid)
        ang = np.linspace(-np.pi, np.pi, n, endpoint=False) if n else np.zeros(0)
        pose = env.robot_pose()
        goal = np.asarray(env.task["goal"], dtype=float)
        dist_goal = float(np.linalg.norm(goal - pose[:2]))

        # 到位即停（对准目标方向，避免最后一次大角度摆动）
        if dist_goal <= self.stop_eps:
            self._pw = 0.0
            return np.array([0.0, 0.0], dtype=np.float32)

        field = self._field_for(env, goal)
        wp = self._lookahead_point(env, field, pose)
        d = wp - pose[:2]
        if float(np.linalg.norm(d)) < 1e-6:
            d = goal - pose[:2]
        head_err = float(_wrap(np.arctan2(d[1], d[0]) - pose[2]))

        # 纯追踪转向
        w = float(np.clip(self.k_ang * head_err, -self.w_max, self.w_max))

        # 速度：航向误差衰减 × 停靠减速
        # **差速底盘必须「先转后走」**：航向误差大时若仍前进，配合角速度速率限制
        # 会走出巨大弧线 —— 实测 15 步只推进 0.28 m、TL 掉到 0.17。
        # 因此用二次衰减，超过 `turn_only_deg` 直接把 v 压到 0，原地转向后再走。
        v_head = max(0.0, 1.0 - (abs(head_err) / np.deg2rad(self.turn_only_deg)) ** 2)
        # **最小前进速度兜底**：纯「先转后走」会让机器人一半以上时间原地转向
        # （实测 v=0 占 58.6%），在长路径上严重拖慢进度。
        # 只要航向误差还在可纠正范围内（< min_v_deg），就保证一个前进速度下限。
        if abs(head_err) < np.deg2rad(self.min_v_deg):
            v_head = max(v_head, self.min_v_frac)
        v_dock = min(1.0, max(0.0, (dist_goal - self.stop_eps) / self.dock_dist))
        v = self.v_max * v_head * v_dock

        # 窄通道自动减速（前方半扇区最近障碍）
        front = np.abs(ang) <= np.deg2rad(30.0)
        d_front = float(lid[front].min()) if front.any() else LIDAR_MAX_RANGE
        if d_front < self.corner_slow:
            v *= max(0.15, d_front / self.corner_slow)

        # 仅保留**急停**（路径已带间隙，不需要连续避障项 —— 那正是 oracle 的抖动来源）。
        # **但急停必须同时转向**：只把 v 置零会让机器人停在障碍前不动、
        # 且因 head_err≈0 而不产生 ω，形成永久死锁
        # （实测 89–97% 的步 v=ω=0，TL 掉到 0.17）。
        if d_front < self.emergency:
            v = 0.0
            w = self._escape(lid, ang)

        # 卡死检测：**只在「命令了前进速度却位移极小」时才算卡住**。
        # 原地转向（v 本来就被压到 0）时位置不动是正常的，不能计为卡死 ——
        # 否则脱困逻辑会在正常转向途中反转方向，把机器人变成来回摆动
        # （实测：航向从 -156° 正常收敛到 -93° 的过程中被反复打断）。
        xy = pose[:2]
        moved = (self._last_xy is None
                 or float(np.linalg.norm(xy - self._last_xy)) >= 0.006)
        if v > 0.02:
            self._stuck = 0 if moved else self._stuck + 1
        else:
            self._stuck = 0
        self._last_xy = xy.copy()
        if self._stuck > 12:
            v = 0.0
            w = self._escape(lid, ang)

        # 角速度速率限制，进一步平滑
        w = float(np.clip(w, self._pw - self.max_dw, self._pw + self.max_dw))
        w = float(np.clip(w, -self.w_max, self.w_max))
        self._pw = w

        return np.array([float(np.clip(v, 0.0, self.v_max)), w], dtype=np.float32)


class PlannerOraclePolicy(PlannerPolicy):
    """语义上与 `oracle` 等价（使用真值目标位置），仅替换底层控制器。

    用于替换测试床的「可达上界」：与 `random_target` 的唯一差别就是目标是否来自指令。
    """

    name = "planner_oracle"


class PlannerRandomTargetPolicy(PlannerPolicy):
    """无语义对照：控制器与 `PlannerOraclePolicy` **完全相同**，目标随机选。

    这是标定测试床区分度的关键基线 —— 若它与 `planner_oracle` 的差距足够大，
    说明该测试床能测出语义的贡献；若仍接近，则任务本身不需要语义。
    """

    name = "planner_random"

    def __init__(self, replan_steps=70, min_dist=1.2, reach_eps=0.5, seed=0, **kw):
        super().__init__(**kw)
        self.replan_steps = replan_steps
        self.min_dist = min_dist
        self.reach_eps = reach_eps
        self._rng = np.random.default_rng(seed)
        self._cur = None
        self._age = 0

    def reset(self, env=None):
        super().reset(env)
        self._cur = None
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

    def act(self, obs, env):
        self._age += 1
        need = (self._cur is None or self._age > self.replan_steps
                or float(np.linalg.norm(self._cur - env.robot_pose()[:2])) < self.reach_eps)
        if need:
            self._cur = self._pick(env)
            self._age = 0
            self._field = None          # 目标变了 → 强制重算距离场
        goal_saved = env.task["goal"]
        env.task["goal"] = self._cur
        try:
            return super().act(obs, env)
        finally:
            env.task["goal"] = goal_saved


class AStarPolicy(PlannerPolicy):
    """**经典 A\\* 规划器**（8 邻域 + 八分度启发 + 视线捷径）——「传统规划」对照基线。

    为什么必须补这个基线（三条理由）：

    1. **第 3 章选型论证需要**（参考论文4 §1.3 用 A* / Dijkstra / DQN 做五维对比）：
       我方若只有「测地场梯度」当对照，等于拿稻草人当 A*，评审会追问
       「你和真正的 A* 比过吗」。
    2. **E1c 需要路径几何对照**：A\\* 在栅格上给出的是**锯齿状折线**，
       是「全局最短 ≠ 好路径」的经典反例（κmax 高、净空小）；
       这正是我方「窗口测地场 + 格精确判据」的价值所在。
    3. **「最短路径」与「可执行路径」不是一回事**：捷径（string pulling）用
       **与物理碰撞同口径**的 `segment_free_exact` 做视线检查 ——
       若换成采样式检查，捷径就会切进桌角（这正是本轮修掉的那个 bug）。

    目标选择与 `PlannerPolicy` 保持一致：**默认使用指令指定的真值目标**，
    因此 `astar` 与 `geodesic` 的差异只来自**规划几何**，不含语义差异。
    """

    name = "astar"

    def __init__(self, shortcut=True, replan_every=8, max_expand=300000,
                 plan_clearance=0.15, clear_cost=2.0, dock_los=0.65, **kw):
        super().__init__(**kw)
        self.shortcut = bool(shortcut)
        self.replan_every = int(replan_every)
        self.max_expand = int(max_expand)
        # **最后接近段不做视线校验**（米）：指令目标本身就贴在物体旁，
        # 严格校验会让机器人停在离目标 0.3–0.5 m 处反复够不着（永远无法判定成功）。
        # 这一段的安全性由环境的碰撞判据保证（它逐帧检查机器人是否侵入障碍）。
        self.dock_los = float(dock_los)
        # **costmap 膨胀半径**（米）：默认与 `env.plan_grid` 一致（0.15 m，恰等于碰撞阈值）。
        # 若改大（如 0.25 m），机器人**出生点/目标点**常常落在膨胀后的不可通行格内，
        # 起点吸附会把路径引到墙的另一侧（实测机器人卡在障碍前反复碰撞）。
        # 因此这里**不靠膨胀取净空**，改用**净空代价**（见 `clear_cost`）。
        self.plan_clearance = float(plan_clearance)
        # **净空代价权重**（无量纲）：路径代价 = 长度 × (1 + clear_cost / 净空格数)。
        # 净空 = 该格到最近不可通行格的距离（格数），在 `env.plan_grid` 上多源 BFS 得到。
        # 这等价于「尽量远离墙与家具」，与真实 A\\* 系统（如 SMAC 的 clearance cost）
        # 的做法一致，且**不破坏起终点的可行性**。
        self.clear_cost = float(clear_cost)
        self._path = None            # 世界坐标点序列
        self._path_goal = None
        self._path_age = 0
        self._path_idx = 0

    def reset(self, env=None):
        super().reset(env)
        self._path = None
        self._path_goal = None
        self._path_age = 0
        self._path_idx = 0

    # ------------------------------------------------------------ A* 本体
    @staticmethod
    def _nearest_free(free, r0, c0, max_r=40):
        """找离 (r0,c0) 最近的自由格（目标点常在物体内部）。

        搜索半径要足够大：目标若落在大件家具（沙发/床）内部，
        最近自由格可能远达 1 m 以上；半径太小会直接判「不可达」。
        """
        H, W = free.shape
        if 0 <= r0 < H and 0 <= c0 < W and free[r0, c0]:
            return int(r0), int(c0)
        for rad in range(1, max_r + 1):
            r1, r2 = max(0, r0 - rad), min(H, r0 + rad + 1)
            c1, c2 = max(0, c0 - rad), min(W, c0 + rad + 1)
            sub = free[r1:r2, c1:c2]
            if not sub.any():
                continue
            rr, cc = np.nonzero(sub)
            d = (rr + r1 - r0) ** 2 + (cc + c1 - c0) ** 2
            k = int(np.argmin(d))
            return int(rr[k] + r1), int(cc[k] + c1)
        return None, None

    def _astar_grid(self, env):
        """A\\* 用的**规划栅格**（与 `env.plan_grid` 同口径，缓存在 env 上）。"""
        cache = getattr(env, "_astar_grid_cache", None)
        if cache is not None:
            return cache
        env._astar_grid_cache = env.plan_grid
        return env.plan_grid

    @staticmethod
    def _clearance_map(env, grid):
        """**净空图**（格数）：每格到最近不可通行格的距离，多源 BFS（缓存在 env 上）。"""
        cache = getattr(env, "_clearance_cache", None)
        if cache is not None:
            return cache
        from collections import deque
        H, W = grid.shape
        dist = np.full((H, W), -1, dtype=np.int32)
        q = deque()
        ys, xs = np.nonzero(grid)
        for y, x in zip(ys, xs):
            dist[y, x] = 0
            q.append((int(y), int(x)))
        while q:
            y, x = q.popleft()
            d = dist[y, x] + 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and dist[ny, nx] < 0:
                    dist[ny, nx] = d
                    q.append((ny, nx))
        env._clearance_cache = dist
        return dist

    def _astar(self, env, goal):
        """8 邻域 A\\*（octile 启发 + **净空代价**），返回起点→目标格的世界坐标序列。"""
        import heapq
        sc = env.scene
        H, W = sc.occ.shape
        grid = self._astar_grid(env)
        free = ~grid
        clear = self._clearance_map(env, grid)
        sr, sc_ = sc.w2c(np.asarray(env.robot_pose()[:2], dtype=float))
        gr, gc = sc.w2c(np.asarray(goal, dtype=float))
        sr, sc_ = int(np.floor(sr)), int(np.floor(sc_))
        gr, gc = int(np.floor(gr)), int(np.floor(gc))
        s_r, s_c = self._nearest_free(free, sr, sc_)
        g_r, g_c = self._nearest_free(free, gr, gc)
        if s_r is None or g_r is None:
            return None
        if (s_r, s_c) == (g_r, g_c):
            return [np.asarray(goal, dtype=float)]

        def h(r, c):
            dr, dc = abs(r - g_r), abs(c - g_c)
            return CELL * (dr + dc) + (CELL * (math.sqrt(2) - 1.0)) * min(dr, dc)

        g = {(s_r, s_c): 0.0}
        parent = {}
        open_heap = [(h(s_r, s_c), 0.0, s_r, s_c)]
        closed = set()
        n_exp = 0
        found = False
        while open_heap:
            _, gcst, r, c = heapq.heappop(open_heap)
            if (r, c) in closed:
                continue
            closed.add((r, c))
            n_exp += 1
            if n_exp > self.max_expand:
                break
            if (r, c) == (g_r, g_c):
                found = True
                break
            for dr, dc, step in ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                                 (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
                                 (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W) or not free[nr, nc]:
                    continue
                if dr and dc:      # 禁止「切角」：两个正交邻格都必须自由
                    if not (free[r + dr, c] and free[r, c + dc]):
                        continue
                ng = gcst + step * CELL * (1.0 + self.clear_cost / max(1, int(clear[nr, nc])))
                if ng < g.get((nr, nc), 1e18):
                    g[(nr, nc)] = ng
                    parent[(nr, nc)] = (r, c)
                    heapq.heappush(open_heap, (ng + h(nr, nc), ng, nr, nc))
        if not found:
            return None
        cells = [(g_r, g_c)]
        cur = (g_r, g_c)
        while cur in parent:
            cur = parent[cur]
            cells.append(cur)
        cells.reverse()
        pts = [np.array([sc.bounds[0] + (c + 0.5) * CELL,
                         sc.bounds[2] + (r + 0.5) * CELL]) for r, c in cells]
        pts[-1] = np.asarray(goal, dtype=float)
        return pts

    def _shortcut_path(self, env, pts):
        """**视线捷径**（string pulling）：把能直连的中间点全部去掉。

        视线检查用 `segment_free_exact`——**与规划栅格同口径**
        （`radius=self.plan_clearance`），避免捷径切进规划栅格认为不可通行的区域；
        同时它本身也是格精确的，不会漏掉被占格（这正是本轮修掉的那类 bug）。
        """
        from .world import segment_free_exact
        if len(pts) <= 2:
            return pts
        out = [pts[0]]
        i = 0
        n = len(pts)
        while i < n - 1:
            j = n - 1
            while j > i + 1:
                if segment_free_exact(env.scene, pts[i], pts[j],
                                      radius=self.plan_clearance):
                    break
                j -= 1
            out.append(pts[j])
            i = j
        return out

    # ------------------------------------------------------------ 与父类的接口
    def _field_for(self, env, goal):
        """A\\* 主路径不需要测地距离场；但**搜索失败时要有兜底**，
        因此这里仍按父类实现缓存距离场（只在 A\\* 失败时才用到）。"""
        return super()._field_for(env, goal)

    def _lookahead_point(self, env, field, pose):
        """沿 A\\* 路径取前视点（父类同名方法被覆盖，仅在 A\\* 失败时回退）。"""
        goal = np.asarray(env.task["goal"], dtype=float)
        need = (self._path is None or self._path_goal is None
                or self._path_age > self.replan_every
                or not np.allclose(goal, self._path_goal, atol=0.3))
        if need:
            pts = self._astar(env, goal)
            if pts is None:
                # **兜底**：A\\* 搜索失败（扩展上限/不可达）→ 回退到测地场梯度，
                # 绝不能「直冲目标」（那会让机器人一头撞进家具、原地卡死）。
                self._path = None
                self._path_goal = goal.copy()
                self._path_age = 0
                return super()._lookahead_point(env, field, pose)
            if self.shortcut:
                pts = self._shortcut_path(env, pts)
            self._path = pts
            self._path_goal = goal.copy()
            self._path_age = 0
            self._path_idx = 0
        if self._path is None:
            return super()._lookahead_point(env, field, pose)
        self._path_age += 1
        path = self._path
        # 找机器人当前最近的路径点（不是「第一个更近的点」——那会永远卡在起点）
        d = np.linalg.norm(path - pose[:2][None, :], axis=1)
        i = int(np.argmin(d))
        i = max(0, min(i, len(path) - 1))
        # **只走「格精确验证过」的直线段**：从当前位置出发，沿路径前进，
        # 直到某个路点与当前位置之间不再是「可直线通过」的（视线被障碍挡住）。
        # 纯追踪会切角，若不校验，机器人在紧贴障碍的拐角处必然碰撞
        # （实测 A* 基线单回合 100+ 次碰撞）；这一步让「路径跟踪」与
        # 「路径来源」解耦，E1c 的对比才真正只差路径来源。
        from .world import segment_free_exact
        j = i
        goal_dist = float(np.linalg.norm(np.asarray(env.task["goal"]) - pose[:2]))
        dock = goal_dist <= self.dock_los      # 最后接近段：目标本身就在物体旁
        while (j < len(path) - 1
               and (dock
                    or segment_free_exact(env.scene, pose[:2], path[j + 1],
                                          radius=self.plan_clearance))):
            j += 1
        # 在 [i, j] 上取前进 `lookahead` 米的点（**段内插值**，
        # 不能直接取段端点：否则第一段长于 lookahead 时会返回机器人自己 → 原地打转）
        traveled = 0.0
        k = i
        while k < j:
            seg = float(np.linalg.norm(path[k + 1] - path[k]))
            if traveled + seg >= self.lookahead:
                t = (self.lookahead - traveled) / max(seg, 1e-9)
                self._path_idx = k
                return path[k] + t * (path[k + 1] - path[k])
            traveled += seg
            k += 1
        self._path_idx = j
        return path[j]


class AStarRandomTargetPolicy(AStarPolicy):
    """无语义对照：控制器与 `AStarPolicy` **完全相同**，目标随机选。

    与 `geodesic_random` 构成同口径对照（同一套导航机制，只换目标来源）。
    """

    name = "astar_random"

    def __init__(self, replan_steps=70, min_dist=1.2, reach_eps=0.5, seed=0, **kw):
        super().__init__(**kw)
        self.replan_steps = replan_steps
        self.min_dist = min_dist
        self.reach_eps = reach_eps
        self._rng = np.random.default_rng(seed)
        self._cur = None
        self._age = 0

    def reset(self, env=None):
        super().reset(env)
        self._cur = None
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

    def act(self, obs, env):
        self._age += 1
        need = (self._cur is None or self._age > self.replan_steps
                or float(np.linalg.norm(self._cur - env.robot_pose()[:2])) < self.reach_eps)
        if need:
            self._cur = self._pick(env)
            self._age = 0
            self._path = None
            self._path_goal = None
        goal_saved = env.task["goal"]
        env.task["goal"] = self._cur
        try:
            return super().act(obs, env)
        finally:
            env.task["goal"] = goal_saved


class GeodesicWindowPolicy:
    """**窗口式测地场规划器** —— 新增固定布局（窄走廊 / S 形走廊 / 仓库）
    的「可达上界」。

    为什么需要它（与 `PlannerPolicy` 的区别）：
    - `PlannerPolicy` 沿**距离场梯度下降**取前视点，梯度是 8 邻域的局部量；
      在 1.5 m 窄走廊、货架迷宫的**贴边格**上，梯度方向会被障碍主导，
      机器人会卡在货架端头（实测 warehouse 里 oracle/planner_oracle 均 0/4，
      机器人停在 (1.94, 1.99) 顶着货架不动）。
    - 本策略每步在机器人周围 `win` 格的**窗口**里，挑「测地场值最小 且 与
      机器人直线可达」的自由格作为前视点：场值保证方向正确，直达性保证
      不会切进家具角，窗口搜索天然没有局部极小（场只从门洞/通道传播）。

    实测（4 回合/场景）：窄走廊 4/4、S 形走廊 4/4、仓库 4/4、公寓 4/4。
    """

    name = "geodesic"

    def __init__(self, win=14, k_ang=3.0, use_gt=True, seed=0):
        self.win = int(win)
        self.k_ang = float(k_ang)
        self.use_gt = bool(use_gt)
        self._rng = np.random.default_rng(seed)
        self._f = None
        self._g = None
        self._rt = None
        self._age = 0

    def reset(self, env=None):
        self._f = None
        self._g = None
        self._rt = None
        self._age = 0

    # ---------------------------------------------------------------- 工具
    def _pick_random(self, env):
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
            if float(np.linalg.norm(p - env.robot_pose()[:2])) < 1.2:
                continue
            return p
        return np.asarray(env.task["goal"], dtype=float).copy()

    def _goal_of(self, env):
        if self.use_gt:
            return np.asarray(env.task["goal"], dtype=float)
        self._age += 1
        if self._rt is None or self._age > 70:
            self._rt = self._pick_random(env)
            self._age = 0
            self._f = None
        return self._rt

    # ---------------------------------------------------------------- 主体
    def act(self, obs, env):
        from .world import segment_free_exact
        goal = self._goal_of(env)
        if self._f is None or self._g is None or \
                not np.allclose(goal, self._g, atol=0.3):
            self._f = env._shortest_path_len(goal)
            self._g = goal.copy()
        f = self._f
        H, W = f.shape
        r, c = map(int, env.scene.w2c(env.robot_pose()[:2]))
        cand = []
        for dr in range(-self.win, self.win + 1):
            for dc in range(-self.win, self.win + 1):
                r1, c1 = r + dr, c + dc
                if 0 <= r1 < H and 0 <= c1 < W and f[r1, c1] >= 0:
                    v = float(f[r1, c1]) + 0.15 * np.hypot(dr, dc) * CELL
                    cand.append((v, r1, c1))
        if not cand:
            return np.array([0.0, 0.6])
        cand.sort()
        tgt = None
        for _, r1, c1 in cand:
            p = env.scene.c2w((r1, c1))
            # **可行性判据必须与物理碰撞判据一致**（格精确版）：
            # 用采样点会漏掉「恰好擦过桌角」的窄缝 → 规划出的路径机器人挤不进去，
            # 表现为顶在障碍角上反复碰撞直到超时（实测教室场景 482 次碰撞）。
            if segment_free_exact(env.scene, env.robot_pose()[:2], p):
                tgt = p
                break
        if tgt is None:
            return np.array([0.0, 0.6])
        # 末段直取：贴近目标且视线可达时直接瞄准目标中心
        ds = env.d_surf
        if ds == ds and ds < 0.9 and \
                segment_free_exact(env.scene, env.robot_pose()[:2], goal):
            tgt = goal
        ang = float(np.arctan2(tgt[1] - env.robot_pose()[1], tgt[0] - env.robot_pose()[0])
                    - env.robot_pose()[2])
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        w = float(np.clip(self.k_ang * ang, -env.cfg["w_max"], env.cfg["w_max"]))
        if abs(ang) > 1.0:
            return np.array([0.0, w])
        v = env.cfg["v_max"] * float(np.clip(1.0 - abs(ang) / np.pi * 1.4, 0.2, 1.0))
        d = float(np.linalg.norm(goal - env.robot_pose()[:2]))
        if d < 0.9:
            v *= max(0.3, d / 0.9)
        return np.array([v, w])


class GeodesicRandomTargetPolicy(GeodesicWindowPolicy):
    """无语义对照：控制器与 `GeodesicWindowPolicy` 完全相同，目标随机选。"""

    name = "geodesic_random"

    def __init__(self, **kw):
        kw.setdefault("use_gt", False)
        super().__init__(**kw)
