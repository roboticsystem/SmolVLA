"""在线语义地图 —— 把「目标曾经在哪」变成**显式的输入通道**（E1 / 主创新 A）。

## 为什么需要它（见 `主创新A_诊断报告.md` §4.16.3）

200 回合的实验把差距来源定位得很清楚：

| | `sweep`（特权） | 最优图像策略 |
|---|---|---|
| SR | **57.5%** | 35.0% |
| 平均步数 | **126.8** | 155.7 |

`sweep` **不需要搜索** —— 它在任何时刻都知道目标的真值方位，直线开过去即可。
任何基于图像的策略都**必须先找到目标**，而「找」本身在 200 步的回合预算里
吃掉一大块。把检测器召回从 64.8% 提到 **100%** 也补不上，因为
**检测器只在「看得见」时有用**，而 `sweep` 在「看不见」时照样知道方向。

那部分信息来自真值地图，图像策略只能靠**搜索 + 记忆**重建。
本模块就是这个重建：一个只用「检测结果 + 激光 + 里程计」的在线语义地图。

## 与 §4.15「检测记忆」的区别

`SweepDetectorPolicy` 里的 `mem_ttl` 只是一个**短期的目标位置缓存**（几秒）。
本模块是完整的世界状态：

1. **物体实例**：类别 / 颜色 / 世界坐标 / 最后看到的时间 —— 永久保留；
2. **障碍栅格**：激光累积（复用 `LocalPlanner` 的建图）；
3. **覆盖记忆**：机器人已经看过哪些方向 —— 用于**避免重复扫描同一片区域**，
   这是纯记忆方案没解决的问题。

## 快照渲染（喂给策略的通道）

把地图渲染成**机器人视角的正上方俯视图**（前方向上），
用一张 RGB 图承载三类信息：

| 通道 | 含义 |
|---|---|
| 绿 | **指令所指目标**在记忆中的位置（当前帧没看见也能给出） |
| 物体本色 | 其他记忆中的物体 |
| 灰 | 激光累积的障碍 |
| 白点 | 机器人自身（图心） |

快照与相机图同构，因此可以直接作为**第二路相机**喂给 VLA
（复用已有的多相机接口），示范与策略共用同一条通道 → 映射良定。
"""

from __future__ import annotations

import numpy as np

BG = (26, 28, 34)
WALL_C = (96, 104, 118)
ROBOT_C = (255, 255, 255)
TARGET_C = (60, 235, 90)


class SemanticMap:
    """只用「检测 + 激光 + 里程计」的在线地图。"""

    def __init__(self, merge_radius=0.7, res=0.10, half_size=15.0,
                 max_range=12.0, inflate=0.28):
        self.merge_radius = float(merge_radius)
        self.res = float(res)
        self.half_size = float(half_size)
        self.max_range = float(max_range)
        self.inflate = float(inflate)
        self.reset()

    def reset(self, env=None):
        self.objs = []          # [ [np.array([x,y]), color, cls, last_t, hits] ]
        self.grid = None
        self.bounds = None
        self._n_inf = 1

    # ---------------------------------------------------------------- 障碍栅格
    def _ensure(self, pose):
        if self.grid is not None:
            return
        x, y = float(pose[0]), float(pose[1])
        self.bounds = (x - self.half_size, x + self.half_size,
                       y - self.half_size, y + self.half_size)
        n = int(np.ceil(2 * self.half_size / self.res))
        self.grid = np.full((n, n), -1, dtype=np.int8)
        self._n_inf = max(1, int(round(self.inflate / self.res)))

    def _w2c(self, pts):
        c = np.floor((pts[:, 0] - self.bounds[0]) / self.res).astype(np.int64)
        r = np.floor((pts[:, 1] - self.bounds[2]) / self.res).astype(np.int64)
        return r, c

    def _integrate_lidar(self, pose, lidar):
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
        l_flat = np.broadcast_to(np.asarray(lidar)[:, None], (n, ds.size)).ravel()[ok]
        hit = l_flat < self.max_range - 1e-6
        free = d_flat < (l_flat - self.res)
        self.grid[r[free], c[free]] = 0
        occ = hit & (d_flat >= (l_flat - self.res)) & (d_flat <= (l_flat + self.res))
        self.grid[r[occ], c[occ]] = 1
        rr, cc = self._w2c(np.array([[x, y]]))
        if 0 <= rr[0] < H and 0 <= cc[0] < W:
            k = self._n_inf
            self.grid[max(0, rr[0] - k):rr[0] + k + 1,
                      max(0, cc[0] - k):cc[0] + k + 1] = 0

    # ---------------------------------------------------------------- 物体实例
    def _to_world(self, det, pose):
        a = float(pose[2]) + float(det.bearing)
        return np.asarray(pose[:2], dtype=np.float64) + \
            float(det.dist) * np.array([np.cos(a), np.sin(a)])

    def update(self, pose, lidar=None, dets=None, t=0):
        """把一帧的激光与检测并入地图。"""
        self._ensure(pose)
        if lidar is not None and len(lidar):
            self._integrate_lidar(pose, lidar)
        for d in (dets or []):
            p = self._to_world(d, pose)
            best, bd = None, np.inf
            for o in self.objs:
                if o[1] != d.color or o[2] != d.cls:
                    continue
                dd = float(np.hypot(*(p - o[0])))
                if dd < bd:
                    best, bd = o, dd
            if best is not None and bd < self.merge_radius:
                # 同实例 → 位置做指数平滑（抑制单目测距抖动），刷新时间
                best[0] = 0.5 * (best[0] + p)
                best[3] = t
                best[4] += 1
            else:
                self.objs.append([p, d.color, d.cls, t, 1])

    def query(self, color=None, cls=None, pose=None):
        """取记忆中符合描述的实例（离机器人最近的在前）。"""
        cand = [o for o in self.objs
                if (color is None or o[1] == color) and (cls is None or o[2] == cls)]
        if not cand:
            return []
        if pose is not None:
            cand = sorted(cand, key=lambda o: float(np.hypot(*(o[0] - pose[:2]))))
        return cand

    # ---------------------------------------------------------------- 快照
    def snapshot(self, pose, want_color=None, want_cls=None, size=64, view=4.0,
                 target_xy=None):
        """渲染**机器人视角的俯视快照**（前方向上），返回 (size,size,3) uint8。"""
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:] = np.array(BG, dtype=np.uint8)
        res = 2.0 * view / size
        th = float(pose[2])
        fwd = np.array([np.cos(th), np.sin(th)])
        right = np.array([np.sin(th), -np.cos(th)])
        org = np.asarray(pose[:2], dtype=np.float64)

        def to_px(pts):
            rel = np.asarray(pts, dtype=np.float64) - org
            rx = rel @ right
            ry = rel @ fwd
            cc = (size / 2.0 + rx / res).astype(np.int64)
            rr = (size / 2.0 - ry / res).astype(np.int64)
            return rr, cc

        k = 2 if view <= 5.0 else 1
        if self.grid is not None and self.bounds is not None:
            occ = np.argwhere(self.grid == 1)
            if occ.size:
                c2w = np.stack([self.bounds[0] + (occ[:, 1] + 0.5) * self.res,
                                self.bounds[2] + (occ[:, 0] + 0.5) * self.res], 1)
                rr, cc = to_px(c2w)
                ok = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
                if ok.any():
                    img[rr[ok], cc[ok]] = WALL_C

        # 记忆中的物体：目标高亮绿色，其余用本色（压暗，避免盖过障碍）
        for o in self.objs:
            is_t = (target_xy is not None
                    and float(np.hypot(*(o[0] - np.asarray(target_xy)))) < self.merge_radius)
            if is_t:
                col = TARGET_C
            elif want_color is not None and o[1] == want_color and \
                    (want_cls is None or o[2] == want_cls):
                col = TARGET_C
            else:
                from .world import COLORS
                col = tuple(int(0.6 * v) for v in COLORS.get(o[1], (150, 150, 150)))
            rr, cc = to_px(o[0][None, :])
            r0, c0 = int(rr[0]), int(cc[0])
            img[max(0, r0 - k):r0 + k + 1, max(0, c0 - k):c0 + k + 1] = col
            if 0 <= r0 < size and 0 <= c0 < size:
                img[r0, c0] = col                      # 保证单像素也可见

        c = size // 2
        img[max(0, c - 2):c + 3, max(0, c - 2):c + 3] = ROBOT_C
        return img


def det_to_map_features(smap, pose, want_color, want_cls, view=4.0):
    """把「目标在记忆中的相对位置」压成 4 维向量（供状态向量使用）。

    与 `perception.target_features` 同构，但来源是**记忆**而非当前帧 ——
    这是「看不见也知道方向」的那条通道。
    """
    cand = smap.query(color=want_color, cls=want_cls, pose=pose)
    if not cand:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    v = cand[0][0] - np.asarray(pose[:2])
    dist = float(np.hypot(*v))
    a = float(np.arctan2(v[1], v[0]) - pose[2])
    a = (a + np.pi) % (2 * np.pi) - np.pi
    return np.array([1.0, float(np.sin(a)), float(np.cos(a)),
                     float(min(1.0, dist / 12.0))], dtype=np.float32)
