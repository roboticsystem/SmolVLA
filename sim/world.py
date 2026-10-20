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
    if layout in LAYOUTS:
        return LAYOUTS[layout](seed)
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


def _build_fixed_scene(name, rooms, floors, doors, furniture, spawn, bounds,
                       door_rgb=(201, 160, 99)):
    """由「房间矩形 + 门洞 + 家具清单」构建固定户型场景。

    抽出来是为了让公寓 / 窄走廊 / S 形走廊 / 仓库四类场景共用同一套
    栅格化与占据图逻辑（论文1 的窄走廊、论文2/4 的仓库货架都靠它扩展）。
    """
    xmin, xmax, ymin, ymax = bounds
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

    for nm, (x0, y0, x1, y1) in rooms:
        carve(x0, y0, x1, y1, FREE, floors.get(nm))
    for d in doors:
        carve(*d, DOOR, door_rgb)

    objects = [Obj(cls, color, (x, y), half)
               for cls, color, x, y, half in furniture]
    scene = Scene(sem, rgb, objects, np.array(spawn, dtype=np.float64),
                  np.pi / 2.0, [rect for _, rect in rooms], bounds, name=name)

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
    scene.room_names = list(rooms)      # 前端房间名标注用：(名称, 矩形) 对
    return scene


def _scale_door(d, s):
    """门洞随场景缩放：**开口宽度保持不变**（真实世界的门宽不随房间变大），
    只缩放位置与墙厚。

    `s == 1` 时**原样返回**：缩放公式里的 `cy ± dy/2` 会把 1.30 变成
    1.3000000000000003，除以 CELL 后 floor/ceil 的结果整整差一格（5 cm 墙沿），
    会让重构后的公寓栅格与历史结果不可比。
    """
    if abs(s - 1.0) < 1e-9:
        return tuple(d)
    x0, y0, x1, y1 = d
    dx, dy = x1 - x0, y1 - y0
    if dx <= dy:                      # 墙薄在 x 方向 → 开口沿 y
        cy = 0.5 * (y0 + y1) * s
        return (round(x0 * s, 6), round(cy - dy / 2.0, 6),
                round(x1 * s, 6), round(cy + dy / 2.0, 6))
    cx = 0.5 * (x0 + x1) * s
    return (round(cx - dx / 2.0, 6), round(y0 * s, 6),
            round(cx + dx / 2.0, 6), round(y1 * s, 6))


def make_apartment_scene(seed: int = 0, scale: float = 1.0) -> Scene:
    """固定「三室两厅」户型场景（确定性，`seed` 仅保持接口兼容）。

    `scale` 用于做**场景规模阶梯**（论文4 的 45×35 / 75×45 / 100×60 三档做法）：
    0.82 / 1.0 / 1.22 三档共用同一套家具相对布局，门宽保持 1.2~1.4 m。
    """
    del seed
    s = float(scale)
    rooms = [(nm, tuple(v * s for v in rect)) for nm, rect in _APARTMENT_ROOMS]
    doors = [_scale_door(d, s) for d in _APARTMENT_DOORS]
    furn = [(cls, color, x * s, y * s, half * s)
            for cls, color, x, y, half in _APARTMENT_FURNITURE]
    spawn = (_APARTMENT_SPAWN[0] * s, _APARTMENT_SPAWN[1] * s)
    bounds = (0.0, 11.4 * s, 0.0, 8.6 * s)
    scene = _build_fixed_scene("apartment", rooms, _APARTMENT_FLOORS, doors,
                               furn, spawn, bounds)
    scene.spawn_theta = np.pi / 2.0
    return scene


# ---------------------------------------------------------------- 窄走廊 / S 形走廊
# 参照论文1《基于嵌入式视觉的室内无人车可行走路径的识别系统研究》§5.3：
#   「宽度仅为 1.5 米的走廊」，前方单侧与双侧交错放置多个纸箱，
#   地面为大理石瓷砖（存在反光纹理倒影）。
# 我们的版本保留「1.5 m 净宽 + 交错障碍 + 尽端目标」三要素，
# 差别是障碍用语义类别的家具（椅子/沙发）而非纯纸箱，以复用同一套
# L1/L2/L3 指令与目标判据。
_CORRIDOR_ROOMS = [
    ("走廊前段", (0.18, 0.18, 8.50, 1.68)),      # 净宽 1.50 m
    ("走廊后段", (7.00, 1.68, 8.80, 9.60)),      # 净宽 1.80 m，L 形拐角
]
_CORRIDOR_FLOORS = {"走廊前段": (214, 208, 196), "走廊后段": (214, 208, 196)}
# 障碍布置约束：**同侧可交错，对侧不得在 x 上重叠**（否则两侧夹出一个 0.26 m
# 的死缝，膨胀栅格下直接断路 —— 实测踩过：s 形走廊曾整条不可达）。
# 另一条约束：家具靠墙放，保证剩余净宽 ≥0.9 m（扣除膨胀后仍可通行）。
_CORRIDOR_FURNITURE = [
    ("chair", "black", 2.20, 0.45, 0.22),        # 下侧
    ("tv", "white", 4.20, 0.45, 0.18),
    ("chair", "black", 6.20, 0.45, 0.22),
    ("plant", "yellow", 3.20, 1.42, 0.22),       # 上侧（与下侧 x 不重叠）
    ("tv", "black", 5.20, 1.42, 0.18),
    ("table", "yellow", 8.30, 3.10, 0.40),       # 靠右墙
    ("plant", "green", 8.45, 6.20, 0.22),       # 靠右墙
    ("box", "white", 8.30, 9.05, 0.25),          # 尽端唯一「箱子」→ L1 可达
]
_CORRIDOR_SPAWN = (0.90, 0.93)


def make_corridor_scene(seed: int = 0) -> Scene:
    """1.5 m 窄走廊（L 形拐角）——对齐论文1 §5.3.1 的窄走廊工况。"""
    del seed
    scene = _build_fixed_scene(
        "corridor", _CORRIDOR_ROOMS, _CORRIDOR_FLOORS, [], _CORRIDOR_FURNITURE,
        _CORRIDOR_SPAWN, (0.0, 8.7, 0.0, 9.8))
    scene.spawn_theta = 0.0                    # 面向 +x（沿走廊前进）
    return scene


_S_CORRIDOR_ROOMS = [
    ("S前段", (0.18, 0.18, 6.20, 1.88)),        # 净宽 1.70 m
    ("S竖段", (4.50, 1.88, 6.20, 5.20)),
    ("S中段", (0.18, 5.20, 6.20, 6.70)),
    ("S后段", (0.18, 6.70, 1.88, 10.20)),       # 净宽 1.70 m
]
_S_CORRIDOR_FLOORS = {k: (214, 208, 196) for k, _ in _S_CORRIDOR_ROOMS}
_S_CORRIDOR_FURNITURE = [
    ("chair", "black", 1.60, 0.45, 0.22),        # S前段 下侧
    ("tv", "white", 2.80, 0.45, 0.18),
    ("plant", "yellow", 3.90, 1.62, 0.22),       # S前段 上侧
    ("chair", "black", 1.90, 1.62, 0.22),        # 与上/下侧 x 均不重叠
    ("chair", "white", 5.30, 3.10, 0.22),        # S竖段
    ("tv", "black", 5.30, 4.40, 0.18),
    ("chair", "white", 3.60, 5.95, 0.22),        # S中段 下侧
    ("tv", "black", 2.20, 5.95, 0.18),
    ("plant", "green", 0.90, 6.45, 0.22),       # S中段 上侧
    ("table", "yellow", 5.40, 0.95, 0.40),       # 靠右墙
    ("box", "white", 1.00, 9.60, 0.25),          # 尽端唯一「箱子」
]
_S_CORRIDOR_SPAWN = (0.90, 0.93)


def make_s_corridor_scene(seed: int = 0) -> Scene:
    """S 形窄走廊 —— 对齐论文1 §5.4.2 的 S 型路径跟踪工况（此处改为
    「长距离绕行到尽端目标」的导航工况）。"""
    del seed
    scene = _build_fixed_scene(
        "s_corridor", _S_CORRIDOR_ROOMS, _S_CORRIDOR_FLOORS, [],
        _S_CORRIDOR_FURNITURE, _S_CORRIDOR_SPAWN, (0.0, 6.4, 0.0, 10.4))
    scene.spawn_theta = 0.0
    return scene


# ---------------------------------------------------------------- 仓库货架场景
# 参照论文2《基于多传感器的室内自动搬运机器人的研究与实现》§6.2 的
# 「货到人」仓库（井字形货架 + 拣选工位 + 活动货柜），与论文4 的
# 「载货点—卸货点—停车点」分区；此处用同一套语义类别表达：
#   货架 = shelf（书架语义）、货箱 = box、工位台 = table。
_WAREHOUSE_ROOMS = [("仓库", (0.18, 0.18, 13.82, 9.82))]
_WAREHOUSE_FLOORS = {"仓库": (198, 198, 194)}
_WAREHOUSE_FURNITURE = [
    # 3 排货架，每排 3 段（段间留 0.2 m），排间通道 2.2 m
    ("shelf", "black", 2.60, 1.60, 0.50), ("shelf", "black", 2.60, 3.20, 0.50),
    ("shelf", "black", 2.60, 4.80, 0.50),
    ("shelf", "yellow", 6.00, 1.60, 0.50), ("shelf", "yellow", 6.00, 3.20, 0.50),
    ("shelf", "yellow", 6.00, 4.80, 0.50),
    ("shelf", "blue", 9.40, 1.60, 0.50), ("shelf", "blue", 9.40, 3.20, 0.50),
    ("shelf", "blue", 9.40, 4.80, 0.50),
    # 拣选工位（右侧）
    ("table", "white", 12.60, 6.60, 0.45), ("table", "white", 11.20, 8.20, 0.45),
    ("table", "green", 12.90, 8.30, 0.45),
    # 地面货箱（避开出生点 (0.90, 0.90) 的 0.2 m 净空）
    ("box", "white", 1.90, 0.80, 0.25), ("box", "black", 7.60, 7.20, 0.25),
    ("box", "green", 3.40, 8.60, 0.25),
    ("plant", "green", 13.20, 1.00, 0.22),
    ("chair", "blue", 10.40, 8.40, 0.22),
]
_WAREHOUSE_SPAWN = (0.90, 0.90)


def make_warehouse_scene(seed: int = 0) -> Scene:
    """仓库货架作业场景（货架 + 工位 + 地面货箱）——对齐论文2/4 的作业语境。"""
    del seed
    scene = _build_fixed_scene(
        "warehouse", _WAREHOUSE_ROOMS, _WAREHOUSE_FLOORS, [],
        _WAREHOUSE_FURNITURE, _WAREHOUSE_SPAWN, (0.0, 14.0, 0.0, 10.0))
    scene.spawn_theta = 0.0
    return scene


# 布局注册表：新增场景只需在此登记（env / web 侧用 layout 名选择）
LAYOUTS = {
    "apartment": lambda s: make_apartment_scene(s, 1.0),
    "apartment_s": lambda s: make_apartment_scene(s, 0.82),   # 紧凑档
    "apartment_l": lambda s: make_apartment_scene(s, 1.22),   # 放大档
    "corridor": make_corridor_scene,                          # 1.5 m 窄走廊
    "s_corridor": make_s_corridor_scene,                      # S 形窄走廊
    "warehouse": make_warehouse_scene,                        # 仓库货架
}


# ---------------------------------------------------------------- 行人
def segment_free(scene: "Scene", a, b, radius: float, n: int = 24) -> bool:
    """沿线段采样，检查是否始终位于自由空间（含半径余量）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    for t in np.linspace(0.0, 1.0, n):
        if not scene.is_free(a + (b - a) * t, margin=radius):
            return False
    return True


def occ_with_peds(scene: "Scene", peds=()):
    """静态占据图 + **行人动态足迹**。

    为什么需要：此前行人**不进占据栅格** —— 既不会被激光雷达看到，也不会与
    机器人碰撞，等于「全程没有动态障碍」。真实室内（论文1 §5.3.2 的行人横穿、
    论文2 的多机避让）里行人必须挡激光、必须能被撞到。

    默认**不启用**（`cfg['dyn_peds']=False`）：论文里已记录的基线数字是在
    「静态障碍」口径下测的，打开会改变 SR/SPL，必须作为**对照实验**单独报。
    """
    if not peds:
        return scene.occ
    occ = scene.occ.copy()
    for p in peds:
        r = float(getattr(p, "radius", 0.25))
        r0, c0, r1, c1 = scene.cell_bbox(p.xy, r)
        occ[r0:r1, c0:c1] = True
    return occ


def is_free_dyn(scene: "Scene", xy, peds=(), margin: float = 0.0) -> bool:
    """自由空间判定 + 行人动态避让（用于碰撞检测）。"""
    if not scene.is_free(xy, margin=margin):
        return False
    for p in peds:
        r = margin + float(getattr(p, "radius", 0.25))
        if float(np.hypot(xy[0] - p.xy[0], xy[1] - p.xy[1])) < r:
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
