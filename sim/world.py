"""语义 MultiRoom 仿真测试床 —— 世界模型（E1）。

设计目标（与真机本体/传感器对齐，便于 sim-to-real）：
- **差速底盘**：2 维连续动作 (v, ω)，与 RDK-X3 小车一致（无云台）
- **传感器**：第一人称 RGB（C70 级 ~64.5° HFOV）+ 2D 激光雷达（D6 级，360°/12 m）
- **语义实例**：类别 + 颜色属性 → 支撑 L1（类别）/ L2（属性）/ L3（空间关系）三级指令
- **动态行人**：支撑遮挡与动态干扰的鲁棒性实验
- **零外部授权依赖**，完全确定性（按种子复现）

坐标系：世界坐标 (x, y) 单位米；朝向 θ 弧度，0 = +x。
栅格分辨率 0.05 m。
"""

from __future__ import annotations

import numpy as np

CELL = 0.05

# id -> (名称, 默认 RGB, 高度 m, 是否可作为导航目标)
SEMANTICS = {
    0: ("free", (238, 238, 238), 0.0, False),
    1: ("wall", (92, 98, 110), 1.80, False),
    2: ("door", (168, 128, 82), 1.60, False),
    3: ("rug", (205, 195, 175), 0.01, False),
    4: ("chair", (205, 95, 85), 0.90, True),
    5: ("table", (150, 105, 70), 0.75, True),
    6: ("sofa", (100, 130, 175), 0.85, True),
    7: ("bed", (140, 155, 200), 0.60, True),
    8: ("shelf", (120, 120, 92), 1.70, True),
    9: ("plant", (85, 150, 95), 1.00, True),
    10: ("box", (190, 160, 100), 0.50, True),
    11: ("tv", (58, 58, 62), 0.60, True),
}
FREE, WALL, DOOR = 0, 1, 2
TARGET_CLASSES = [k for k, v in SEMANTICS.items() if v[3]]

# L2 属性指令用的颜色（渲染进栅格 + 生成指令文本）
COLORS = {
    "red": (198, 62, 55),
    "blue": (62, 92, 198),
    "green": (70, 158, 82),
    "yellow": (220, 186, 58),
    "white": (238, 238, 238),
    "black": (52, 52, 58),
}
COLOR_NAMES = list(COLORS)

CLASS_CN = {
    "chair": "椅子", "table": "桌子", "sofa": "沙发", "bed": "床",
    "shelf": "书架", "plant": "植物", "box": "箱子", "tv": "电视",
}


class Obj:
    """语义物体实例。方形足迹，用于碰撞、渲染与任务生成。"""

    __slots__ = ("cls", "color", "xy", "half", "height", "rgb")

    def __init__(self, cls: str, color: str, xy, half: float):
        self.cls = cls
        self.color = color
        self.xy = np.asarray(xy, dtype=np.float64)
        self.half = float(half)
        self.height = SEMANTICS[[k for k, v in SEMANTICS.items() if v[0] == cls][0]][2]
        self.rgb = np.array(COLORS.get(color, (150, 150, 150)), dtype=np.uint8)

    @property
    def cls_id(self) -> int:
        return _CLS2ID[self.cls]


_CLS2ID = {v[0]: k for k, v in SEMANTICS.items()}


class Scene:
    """一个可复现的室内场景。"""

    def __init__(self, sem: np.ndarray, rgb: np.ndarray, objects, spawn, spawn_theta,
                 rooms, bounds, name="scene"):
        self.sem = sem                      # (H,W) uint8 语义 id
        self.rgb = rgb                      # (H,W,3) uint8 渲染颜色
        self.objects: list[Obj] = objects
        self.spawn = np.asarray(spawn, dtype=np.float64)
        self.spawn_theta = float(spawn_theta)
        self.rooms = rooms                  # [(x0,y0,x1,y1)] 米
        self.bounds = bounds                # (xmin,xmax,ymin,ymax)
        self.name = name
        self.H, self.W = sem.shape
        self.occ = (sem == WALL)  # 门洞(DOOR)是**可通行**的，仅用于渲染区分
        # 相机背景只对墙体做光线投射（物体作为 sprite 单独绘制并做深度测试），
        # 而碰撞与激光雷达用 occ（墙 + 物体），二者物理一致。
        self.wall_grid = (sem == WALL)
        # 物体足迹也视为不可通行（贴边可绕行）
        for o in objects:
            self._stamp_occ(o)

    def _stamp_occ(self, o: Obj):
        i0, j0, i1, j1 = self.cell_bbox(o.xy, o.half)
        self.occ[i0:i1, j0:j1] = True

    # ---------------- 坐标换算 ----------------
    def w2c(self, xy) -> np.ndarray:
        """世界坐标(米) → 栅格下标 (row, col) 浮点。"""
        x, y = xy
        col = (x - self.bounds[0]) / CELL
        row = (y - self.bounds[2]) / CELL
        return np.array([row, col], dtype=np.float64)

    def c2w(self, rc) -> np.ndarray:
        row, col = rc
        return np.array([self.bounds[0] + (col + 0.5) * CELL,
                         self.bounds[2] + (row + 0.5) * CELL])

    def cell_bbox(self, xy, half):
        r, c = self.w2c(xy)
        h = half / CELL
        return (max(0, int(r - h)), max(0, int(c - h)),
                min(self.H, int(np.ceil(r + h))), min(self.W, int(np.ceil(c + h))))

    def is_free(self, xy, margin=0.0) -> bool:
        r, c = self.w2c(xy)
        m = int(np.ceil(margin / CELL))
        r, c = int(r), int(c)
        if not (m <= r < self.H - m and m <= c < self.W - m):
            return False
        return not self.occ[r - m:r + m + 1, c - m:c + m + 1].any()


# ---------------------------------------------------------------- 场景生成
def _layout(nx: int, ny: int, room_w: float, room_h: float, wall: float, seed: int):
    """生成 nx×ny 个房间的矩形列表（米）。

    注意：房间尺寸只能**向内收缩**，不能超出网格节距，否则相邻房间会重叠、
    墙体间隙消失（曾在调试中出现连通性仅 14% 的问题）。
    """
    rng = np.random.default_rng(seed)
    rooms = []
    for iy in range(ny):
        for ix in range(nx):
            x0 = wall + ix * (room_w + wall)
            y0 = wall + iy * (room_h + wall)
            w = room_w * rng.uniform(0.85, 1.0)   # ≤ room_w，保证间隙 ≥ wall
            h = room_h * rng.uniform(0.85, 1.0)
            rooms.append((x0, y0, x0 + w, y0 + h))
    total_w = wall + nx * (room_w + wall)
    total_h = wall + ny * (room_h + wall)
    return rooms, (0.0, total_w, 0.0, total_h)


def make_scene(seed: int = 0, nx: int = 3, ny: int = 2, room_w: float = 3.6,
               room_h: float = 3.4, wall: float = 0.18, n_objects_per_room=(1, 3),
               pad_objects: bool = True, layout: str | None = None) -> Scene:
    """程序化生成一个语义 MultiRoom 场景（确定性）。

    `layout="apartment"` 时返回固定的「三室两厅」户型（演示用），
    其余参数被忽略；默认 None 保持原有随机 MultiRoom（论文实验用）。
    """
    if layout == "apartment":
        return make_apartment_scene(seed)
    rng = np.random.default_rng(seed)
    rooms, (xmin, xmax, ymin, ymax) = _layout(nx, ny, room_w, room_h, wall, seed)

    Wg = int(np.ceil((xmax - xmin) / CELL))
    Hg = int(np.ceil((ymax - ymin) / CELL))
    sem = np.full((Hg, Wg), WALL, dtype=np.uint8)
    rgb = np.zeros((Hg, Wg, 3), dtype=np.uint8)
    rgb[:] = np.array(SEMANTICS[WALL][1], dtype=np.uint8)

    def carve_r(x0, y0, x1, y1, val=FREE):
        c0 = int(np.floor((x0 - xmin) / CELL)); c1 = int(np.ceil((x1 - xmin) / CELL))
        r0 = int(np.floor((y0 - ymin) / CELL)); r1 = int(np.ceil((y1 - ymin) / CELL))
        r0, r1 = max(0, r0), min(Hg, r1); c0, c1 = max(0, c0), min(Wg, c1)
        sem[r0:r1, c0:c1] = val
        rgb[r0:r1, c0:c1] = np.array(SEMANTICS[val][1], dtype=np.uint8)
        return r0, r1, c0, c1

    # 1) 挖出房间内部
    for (x0, y0, x1, y1) in rooms:
        carve_r(x0, y0, x1, y1, FREE)

    # 2) 开连通门：直接在两房间之间的整段间隙上开洞（不假设间隙恰好等于 wall）
    door_cells = []
    for iy in range(ny):
        for ix in range(nx):
            i = iy * nx + ix
            # 向右邻居
            if ix + 1 < nx:
                x0a, y0a, x1a, y1a = rooms[i]
                x0b, y0b, x1b, y1b = rooms[i + 1]
                lo, hi = max(y0a, y0b) + 0.6, min(y1a, y1b) - 0.6
                if hi > lo:
                    yy = rng.uniform(lo, hi)
                    carve_r(x1a - 0.02, yy - 0.45, x0b + 0.02, yy + 0.45, DOOR)
                    door_cells.append((0.5 * (x1a + x0b), yy))
            # 向上邻居
            if iy + 1 < ny:
                x0a, y0a, x1a, y1a = rooms[i]
                x0b, y0b, x1b, y1b = rooms[i + nx]
                lo, hi = max(x0a, x0b) + 0.6, min(x1a, x1b) - 0.6
                if hi > lo:
                    xx = rng.uniform(lo, hi)
                    carve_r(xx - 0.45, y1a - 0.02, xx + 0.45, y0b + 0.02, DOOR)
                    door_cells.append((xx, 0.5 * (y1a + y0b)))

    # 3) 放语义物体
    objects: list[Obj] = []
    room_of_obj = {}
    for i, (x0, y0, x1, y1) in enumerate(rooms):
        n = rng.integers(n_objects_per_room[0], n_objects_per_room[1] + 1)
        for _ in range(int(n)):
            for _try in range(60):
                cls = rng.choice(list(CLASS_CN.keys()))
                half = {"chair": 0.22, "table": 0.45, "sofa": 0.55, "bed": 0.75,
                        "shelf": 0.25, "plant": 0.20, "box": 0.25, "tv": 0.18}[cls]
                pad = half + (0.35 if pad_objects else 0.1)
                cx = rng.uniform(x0 + pad, x1 - pad)
                cy = rng.uniform(y0 + pad, y1 - pad)
                if any(np.hypot(cx - o.xy[0], cy - o.xy[1]) < o.half + half + 0.35
                       for o in objects):
                    continue
                color = rng.choice(COLOR_NAMES)
                o = Obj(cls, str(color), (cx, cy), half)
                objects.append(o)
                room_of_obj[id(o)] = i
                break

    scene = Scene(sem, rgb, objects, (0, 0), 0.0, rooms,
                  (xmin, xmax, ymin, ymax), name=f"multiroom_s{seed}")

    # 4) 把物体画进栅格（语义 + 颜色）
    for o in objects:
        r0, c0, r1, c1 = scene.cell_bbox(o.xy, o.half)
        oid = o.cls_id
        scene.sem[r0:r1, c0:c1] = oid
        scene.rgb[r0:r1, c0:c1] = o.rgb
    scene.wall_grid = (scene.sem == WALL)
    scene.occ = (scene.sem == WALL)
    for o in objects:
        scene._stamp_occ(o)

    # 5) 选出生点：第一个房间的空闲格中心
    x0, y0, x1, y1 = rooms[0]
    for _ in range(200):
        p = (rng.uniform(x0 + 0.5, x1 - 0.5), rng.uniform(y0 + 0.5, y1 - 0.5))
        if scene.is_free(p, margin=0.25):
            scene.spawn = np.array(p)
            scene.spawn_theta = float(rng.uniform(-np.pi, np.pi))
            break
    scene._door_cells = door_cells
    scene._room_of_obj = room_of_obj
    return scene


# ---------------------------------------------------------------- 三室两厅公寓（演示场景）
# 固定户型：客厅 + 餐厅（两厅）、主卧 + 次卧 + 书房（三室）、厨房 + 卫生间 + 走廊。
# 与 E1 实验场景**同物理/传感器/指令体系**（差速底盘、激光、L1-L3 指令、SPL 口径），
# 仅布局固定且家具摆放贴近真实住宅 —— 供网页演示与跟随拍摄任务使用；
# 论文实验一律用默认随机 MultiRoom（`layout=None`），两套场景互不影响。
# 家具颜色刻意**避开红色**：红色保留给跟随任务的「红马甲」（场景唯一前提）。
_APARTMENT_ROOMS = [
    ("客厅", (0.18, 0.18, 5.00, 3.82)),
    ("餐厅", (5.18, 0.18, 8.00, 3.82)),
    ("厨房", (8.18, 0.18, 11.22, 3.82)),
    ("走廊", (0.18, 4.00, 11.22, 4.90)),
    ("主卧", (0.18, 5.08, 3.60, 8.42)),
    ("次卧", (3.78, 5.08, 7.00, 8.42)),
    ("书房", (7.18, 5.08, 9.60, 8.42)),
    ("卫生间", (9.78, 5.08, 11.22, 8.42)),
]
_APARTMENT_FLOORS = {
    "客厅": (232, 219, 190), "餐厅": (232, 219, 190),          # 木地板
    "厨房": (213, 226, 233), "卫生间": (213, 226, 233),        # 瓷砖
    "走廊": (226, 226, 220),
    "主卧": (235, 222, 224), "次卧": (222, 230, 240), "书房": (224, 233, 222),
}
# 门洞 (x0, y0, x1, y1)，内门统一 1.2 m（0.9 m 时 oracle 前瞻+避障会在门口墙角卡死）
_APARTMENT_DOORS = [
    (5.00, 1.30, 5.18, 2.70),      # 客厅 ↔ 餐厅（宽开口 1.4 m）
    (8.00, 1.30, 8.18, 2.50),      # 餐厅 ↔ 厨房
    (2.20, 3.82, 3.40, 4.00),      # 客厅 ↔ 走廊（宽开口 1.2 m）
    (6.10, 3.82, 7.30, 4.00),      # 餐厅 ↔ 走廊
    (9.20, 3.82, 10.40, 4.00),     # 厨房 ↔ 走廊
    (1.20, 4.90, 2.40, 5.08),      # 主卧 ↔ 走廊
    (4.90, 4.90, 6.10, 5.08),      # 次卧 ↔ 走廊
    (7.70, 4.90, 8.90, 5.08),      # 书房 ↔ 走廊
    (9.90, 4.90, 11.10, 5.08),     # 卫生间 ↔ 走廊
]
# (cls, color, x, y, half)。指令可达性：L1 唯一类 = 沙发/电视/箱子；
# L2 唯一「颜色+类」组合覆盖床/桌/椅/架/植物等（见 _all_phrasings 的唯一性校验）。
_APARTMENT_FURNITURE = [
    ("sofa",  "blue",  1.10, 2.90, 0.55),   # 客厅沙发
    ("tv",    "black", 4.60, 3.30, 0.18),   # 电视柜（北墙侧，西侧 approached 开阔）
    ("table", "white", 3.60, 2.30, 0.40),   # 茶几（避开出生点→走廊门洞的直线路径）
    ("plant", "green", 0.70, 0.80, 0.22),   # 客厅绿植
    ("table", "yellow", 6.40, 1.90, 0.45),  # 餐桌
    ("chair", "green", 7.55, 0.75, 0.22),   # 餐椅（四周留 0.5m+ 站位，勿夹缝）
    ("chair", "green", 5.60, 3.30, 0.22),   # 餐椅
    ("shelf", "blue", 10.70, 1.00, 0.25),   # 橱柜
    ("box",   "white", 9.00, 1.00, 0.25),   # 纸箱（L1「去箱子旁边」→ 与真机指令对齐）
    ("bed",   "white", 1.60, 7.10, 0.75),   # 主卧床
    ("shelf", "yellow", 0.95, 5.55, 0.25),  # 衣柜（南墙西段，绕床通道保持 ≥0.4m）
    ("plant", "yellow", 10.50, 6.50, 0.22),  # 卫生间植物（主卧衣柜挪出后原位改放）
    ("bed",   "blue",  5.30, 7.10, 0.75),   # 次卧床
    ("shelf", "black", 8.90, 7.60, 0.25),   # 书房书架
    ("chair", "blue",  7.60, 5.70, 0.22),   # 书房椅（避开门口动线）
]
_APARTMENT_SPAWN = (2.60, 1.00)             # 客厅（玄关侧），面向沙发/电视


def make_apartment_scene(seed: int = 0) -> Scene:
    """固定「三室两厅」户型场景（确定性，`seed` 仅保持接口兼容）。"""
    del seed
    xmin, ymin, xmax, ymax = 0.0, 0.0, 11.4, 8.6
    Wg = int(np.ceil((xmax - xmin) / CELL))
    Hg = int(np.ceil((ymax - ymin) / CELL))
    sem = np.full((Hg, Wg), WALL, dtype=np.uint8)
    rgb = np.zeros((Hg, Wg, 3), dtype=np.uint8)
    rgb[:] = np.array(SEMANTICS[WALL][1], dtype=np.uint8)

    def carve(x0, y0, x1, y1, val=FREE, rgbv=None):
        c0 = int(np.floor((x0 - xmin) / CELL)); c1 = int(np.ceil((x1 - xmin) / CELL))
        r0 = int(np.floor((y0 - ymin) / CELL)); r1 = int(np.ceil((y1 - ymin) / CELL))
        r0, r1 = max(0, r0), min(Hg, r1); c0, c1 = max(0, c0), min(Wg, c1)
        sem[r0:r1, c0:c1] = val
        rgb[r0:r1, c0:c1] = np.array(
            rgbv if rgbv is not None else SEMANTICS[val][1], dtype=np.uint8)

    for name, (x0, y0, x1, y1) in _APARTMENT_ROOMS:
        carve(x0, y0, x1, y1, FREE, _APARTMENT_FLOORS[name])
    for d in _APARTMENT_DOORS:
        carve(*d, DOOR, (201, 160, 99))

    objects = [Obj(cls, color, (x, y), half)
               for cls, color, x, y, half in _APARTMENT_FURNITURE]
    rooms = [rect for _, rect in _APARTMENT_ROOMS]
    scene = Scene(sem, rgb, objects, np.array(_APARTMENT_SPAWN), np.pi / 2.0,
                  rooms, (xmin, xmax, ymin, ymax), name="apartment")

    # 家具画进栅格（语义 + 颜色 + 深色描边增强俯视观感）
    dark = (60, 56, 52)
    for o in objects:
        r0, c0, r1, c1 = scene.cell_bbox(o.xy, o.half)
        scene.sem[r0:r1, c0:c1] = o.cls_id
        scene.rgb[r0:r1, c0:c1] = o.rgb
        scene.rgb[r0, c0:c1] = dark; scene.rgb[r1 - 1, c0:c1] = dark
        scene.rgb[r0:r1, c0] = dark; scene.rgb[r0:r1, c1 - 1] = dark
    scene.wall_grid = (scene.sem == WALL)
    scene.occ = (scene.sem == WALL)
    for o in objects:
        scene._stamp_occ(o)
    scene._door_cells = []
    scene._room_of_obj = {}
    scene.room_names = list(_APARTMENT_ROOMS)     # 前端房间名标注用
    return scene


# ---------------------------------------------------------------- 行人
def segment_free(scene: "Scene", a, b, radius: float, n: int = 24) -> bool:
    """沿线段采样，检查是否始终位于自由空间（含半径余量）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    for t in np.linspace(0.0, 1.0, n):
        if not scene.is_free(a + (b - a) * t, margin=radius):
            return False
    return True


class Pedestrian:
    """在自由空间内沿航点直线行走的动态行人。

    航点选择同时要求「终点自由」与「直线路径无碰撞」，避免穿墙
    （调试中曾出现行人在障碍物内部的问题）。
    """

    def __init__(self, scene: Scene, rng: np.random.Generator, speed=0.5, radius=0.25,
                 min_dist_from=None):
        self.scene = scene
        self.radius = radius
        self.speed = speed
        self._rng = rng
        self.xy = np.asarray(min_dist_from, dtype=np.float64) if min_dist_from is not None else None
        self._sample_start(min_dist_from)
        self.wp = self.xy.copy()
        self._pick_waypoint()

    def _rand_free(self, margin, avoid=None, min_sep=0.0):
        sc = self.scene
        for _ in range(300):
            p = np.array([self._rng.uniform(sc.bounds[0] + 0.5, sc.bounds[1] - 0.5),
                          self._rng.uniform(sc.bounds[2] + 0.5, sc.bounds[3] - 0.5)])
            if not sc.is_free(p, margin=margin):
                continue
            if avoid is not None and np.linalg.norm(p - avoid) < min_sep:
                continue
            return p
        return sc.spawn.copy()

    def _sample_start(self, avoid):
        self.xy = self._rand_free(self.radius + 0.05, avoid=avoid, min_sep=1.0)

    def _pick_waypoint(self):
        for _ in range(300):
            p = self._rand_free(self.radius + 0.05)
            if segment_free(self.scene, self.xy, p, self.radius + 0.05):
                self.wp = p
                return

    def step(self, dt: float):
        d = self.wp - self.xy
        n = float(np.linalg.norm(d))
        if n < 0.15:
            self._pick_waypoint()
            return
        self.xy = self.xy + d / n * min(self.speed * dt, n)


def make_pedestrians(scene: Scene, n: int, seed: int = 0):
    """生成 n 个行人；起点避开机器人出生点（至少 1 m）。"""
    rng = np.random.default_rng(seed + 7919)
    return [Pedestrian(scene, rng, min_dist_from=scene.spawn) for _ in range(n)]
