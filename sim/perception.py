"""从渲染图像做目标检测 —— 让示范策略摆脱**特权信息**（E1 / 主创新 A）。

## 为什么需要这个模块

见 `主创新A_诊断报告.md` §4.10。此前的示范策略（`sweep` / `oracle`）都用
**真值目标方位**（`|bearing| ≤ HFOV/2`）来决定「扫描 ↔ 逼近」的切换，而真实
机器人必须**从相机里认出任一指令指代的物体**才能做到同样的事。

后果是致命的：示范动作变成了 `f(图像, 目标位置)`，而目标位置**不在图像里**。
对模仿者而言这是**一对多**映射（同一张图，目标不同则动作不同），
它只能学到条件平均。这恰好解释了实测结果 —— VLA 闭环 13.3%，
而「图像 + 位姿 + 指令」的公平上界是 13.0%，**两者相等**。

本模块让示范策略**只看图像**：检测候选物体 → 按指令描述匹配 → 输出方位角。
检测器**不追求完美**（有漏检、误检、方位误差都无所谓），关键在于
**动作只依赖图像**，从而使「观测 → 动作」成为良定义、可学习的映射。

## 检测原理（单目几何，不需要任何真值）

渲染器把物体画成矩形 sprite，其投影为

    cx       = (W-1)/2 + f·tan(ang)          ← 水平位置 → 方位角
    half_px  = f·half/d                       ← 像素半宽
    r0       = horizon - f·(h - cam_h)/d      ← 顶边
    r1       = horizon + f·cam_h/d            ← **底边（落在地面上）**

因此：
1. 由**底边行号**反推距离 `d = f·cam_h/(r1 - horizon)`（单目测距的标准做法，
   不依赖颜色，比用雾化亮度稳健得多）；
2. 由像素高/宽反推物体真实尺寸 `h = (r1-r0)·d/f`、`half = half_px·d/f`；
3. `(h, half)` 在类别间分得很开（椅子 0.90×0.22、桌子 0.75×0.45、书架 1.70×0.25…），
   最近邻即可判类；
4. 颜色用**逐像素到「调色板色 → 雾色」线段的垂距**来分（渲染的明暗/雾化
   恰好就是这条线段上的位置，所以这个模型是匹配的）。

## 已知限制（诚实交代）

- **同方位的相邻物体会并成一个 blob**（按列聚类无法分开）
- **墙的灰度可能被判成 "black"**：靠 `(h, half)` 的合理性区间 + 逐列颜色一致性
  （墙有纹理抖动）来抑制，但挡不住时会漏进误检
- **白色物体与天花板/地面接近**，可能漏检
- 被完全遮挡的物体**检测不到**（这正是我们想要的：可见性必须来自图像）
"""

from __future__ import annotations

import numpy as np

from .sensors import CAMERA_HEIGHT, CAMERA_HFOV_DEG
from .world import COLORS

# 渲染参数（必须与 sensors.render 的默认值一致，否则几何反推会偏）
FOG_COLOR = np.array((150.0, 160.0, 175.0))
MAX_RANGE = 12.0

# 类别 → (高度 m, 半宽 m)。与 world.py 的 SEMANTICS 高度 / 摆放 half 一致。
CLASS_SHAPE = {
    "chair": (0.90, 0.22), "table": (0.75, 0.45), "sofa": (0.85, 0.55),
    "bed":   (0.60, 0.75), "shelf": (1.70, 0.25), "plant": (1.00, 0.20),
    "box":   (0.50, 0.25), "tv":    (0.60, 0.18),
}
_CLASSES = list(CLASS_SHAPE)
_SHAPE_ARR = np.array([CLASS_SHAPE[c] for c in _CLASSES], dtype=np.float64)
# 类别判定在 log 空间做（尺寸是乘性关系，log 空间下距离更合理）
_LOG_SHAPE = np.log(_SHAPE_ARR)


class Detection:
    """一次检测结果。`bearing` 为**相对车头**的方位角（弧度）。"""

    __slots__ = ("bearing", "dist", "cls", "color", "height", "half", "n_px", "surf",
                 "hit_bearing")

    def __init__(self, bearing, dist, cls, color, height, half, n_px, surf=float("nan"),
                 hit_bearing=float("nan")):
        self.bearing = float(bearing)
        # `dist` = 由色块底边按地面几何反推的**中心距**（近距离会饱和，见 `detect`）
        self.dist = float(dist)
        self.cls = cls
        self.color = color
        self.height = float(height)
        self.half = float(half)
        self.n_px = int(n_px)
        # `surf` = **激光在目标方位上的读数 = 到物体表面的距离**（NaN 表示未测）。
        # 这是**唯一在近距离仍然有效**的距离量：地面几何在 r1 触底时饱和于
        # f·cam_h/(H−horizon) ≈ 0.75 m（判据就是 0.30 m！），而表观尺寸在
        # 0.30 m 处物体已纵向溢出画面（254 px vs 图高 120 px）。
        self.surf = float(surf)
        # `hit_bearing` = 激光命中点（色块角跨度内**读数最小**的那一束）的方位角，
        # 相对车头（与 `bearing` 同系）。**仅当 `detect(want_hit=True)` 时有效**。
        # 用途：末段停靠要「朝真实表面点走」，而单目 `bearing` 指的是色块**中心**
        # （有检测方位误差），激光命中点才是真正会被撞到的那个表面点。
        self.hit_bearing = float(hit_bearing)

    def __repr__(self):
        return (f"<{self.color} {self.cls} bearing={np.rad2deg(self.bearing):+.1f}° "
                f"d={self.dist:.1f}m {self.height:.2f}×{2*self.half:.2f} px={self.n_px}>")


def _build_refs():
    """构造「参考色字典」：把渲染的成像模型离散成有限个观测量。

    成像模型（`sensors.render`）为

        obs = (base · df · (1-fw) + fog · fw) · s

    其中 `df = 1-0.75·d/R`（距离衰减）、`fw = 0.45·d/R`（雾化）、
    `s ∈ [0.78, 1.0]`（sprite 的垂直光照因子）。

    **早先只建模「调色板色 → 雾色」这条线段是不够的**：`s` 会把整个颜色
    （含雾分量）乘性缩小，使观测点明显偏离线段 —— 实测蓝色 B 通道
    198×0.78=154，偏离 26，超出容差 24，导致蓝色物体全部漏检（召回仅 10%）。
    把三个因素都离散进去即可。
    """
    cols, idx = [], []
    for i, name in enumerate(COLORS):
        c = np.array(COLORS[name], dtype=np.float64)
        for df in (0.30, 0.50, 0.70, 0.85, 0.975):
            for fw in (0.0, 0.15, 0.30, 0.45):
                for s in (0.78, 0.90, 1.0):
                    cols.append((c * df * (1.0 - fw) + FOG_COLOR * fw) * s)
                    idx.append(i)
    return np.array(cols), np.array(idx, dtype=np.int64)


REFS, REF_IDX = _build_refs()
_REFTREE = None


def _pixel_labels(img, tol=16.0):
    """把每个像素归到最近的参考色。

    Returns:
        lab (H,W) int  调色板下标，-1 表示不属于任何颜色
        names list     下标 → 颜色名
    """
    from scipy.spatial import cKDTree

    global _REFTREE
    if _REFTREE is None:
        _REFTREE = cKDTree(REFS)

    H, W, _ = img.shape
    px = img.reshape(-1, 3).astype(np.float64)
    dist, j = _REFTREE.query(px, k=1)
    lab = np.where(dist < tol, REF_IDX[j], -1).reshape(H, W)
    return lab, list(COLORS)


def detect(img, hfov_deg=CAMERA_HFOV_DEG, cam_h=CAMERA_HEIGHT,
           max_range=MAX_RANGE, tol=16.0, min_px=40, col_consistency=True,
           max_col_std=6.0, min_fill=0.65, max_shape_err=0.50, lidar=None,
           want_hit=False):
    """从一帧 RGB 图里检测候选物体。

    Args:
        col_consistency: 要求 blob 的**逐列均色标准差**足够小。
            墙有逐格纹理抖动（`render` 里 ±10%，实测列色标准差约 10），
            物体列色近乎恒定（≈0）；这一项用来抑制「墙被误判成 black 物体」。
        min_fill: 最小填充度（blob 像素数 / 外接框面积）。物体是实心矩形，
            填充度接近 1；散乱的误检区域填充度低。
        lidar: (n_beams,) 一帧激光（360°）。**强烈建议传入**：它用来按列把
            色块切成「属于同一物体」的若干簇，解决「物体与墙并在一起」
            （占漏检 95.7%）。不传时退化为纯几何。
        want_hit: 是否额外返回**激光命中点方位**（`Detection.hit_bearing`）。
            默认 **False**（行为与既有完全一致）；末段停靠控制器才需要它。

    Returns:
        list[Detection]
    """
    from scipy import ndimage

    H, W, _ = img.shape
    lab, names = _pixel_labels(img, tol=tol)
    imgf = img.astype(np.float64)
    f = (W / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    horizon = H / 2.0
    # 激光读数按方位索引（用于给每个检测配一个**表面距离**）。
    # 为什么必须有它：地面几何 `f·cam_h/(r1−horizon)` 在色块底边触到图像下沿时
    # **饱和于 f·cam_h/(H−horizon) = 0.752 m**，而成功判据就是 0.30 m ——
    # 即该公式在判据附近完全失效（数据集里检测距离的最小值恰为 0.74 m 可证）。
    # 表观尺寸也不行：0.30 m 处物体像素高 254 px > 图高 120 px，已溢出画面。
    # 激光是唯一在近距离仍有效的量。
    _lv = (np.asarray(lidar, dtype=np.float64).reshape(-1)
           if (lidar is not None and np.size(lidar) > 0) else None)

    def _lidar_surf(c0, c1):
        """色块**角跨度内**激光读数的最小值 = 到该物体包围盒**最近表面**的距离。

        不用单束：单束可能打到更近的遮挡物或擦不到物体而打到墙上（实测都有）。
        取跨度内的最小值在几何上正好对应「到包围盒表面的最短距离」——**就是成功判据**。

        `want_hit=True` 时额外返回**命中点的方位角**（那一束最小读数所在的角），
        它是「真正会被撞到的表面点」的方向；单目 `bearing` 指的是色块**中心**，
        两者在检测有方位误差时并不一致 —— 末段停靠要靠它对准。
        """
        if _lv is None or _lv.size == 0:
            return (float("nan"), float("nan")) if want_hit else float("nan")
        nl = _lv.size
        cols = np.arange(int(c0), max(int(c0) + 1, int(c1)), dtype=np.float64)
        angs = np.arctan((cols - (W - 1) / 2.0) / f)
        bi = np.round(((angs + np.pi) / (2 * np.pi)) * nl).astype(np.int64) % nl
        v = _lv[bi]
        ok = np.isfinite(v)
        if not ok.any():
            return (float("nan"), float("nan")) if want_hit else float("nan")
        v = v[ok]
        ang_ok = angs[ok]
        k = int(np.argmin(v))
        dmin = float(v[k])
        if want_hit:
            return dmin, float(ang_ok[k])
        return dmin

    dets = []
    # **逐颜色做二维连通域**。早先用「逐列合并」会把整幅画面并成一个 blob
    # （墙被判成 black，占据 117/160 列），全部被尺寸过滤掉 → 召回仅 5%。
    st = np.ones((3, 3), dtype=bool)
    for ci, name in enumerate(names):
        m = lab == ci
        if int(m.sum()) < min_px:
            continue
        cc, n = ndimage.label(m, structure=st)
        if n == 0:
            continue
        objs = ndimage.find_objects(cc)
        for k, sl in enumerate(objs, start=1):
            ys, xs = np.nonzero(cc[sl] == k)
            npx = ys.size
            if npx < min_px:
                continue
            r0 = sl[0].start + int(ys.min())
            r1 = sl[0].start + int(ys.max()) + 1
            c0 = sl[1].start + int(xs.min())
            c1 = sl[1].start + int(xs.max()) + 1
            if r1 - horizon <= 1.0:            # 底边须在地平线下方才能反推距离
                continue
            d0 = f * cam_h / (r1 - horizon)
            if not (0.4 <= d0 <= max_range):
                continue
            sub_m = (cc[sl] == k)              # 该连通域在裁剪窗内的掩码

            # ---------------------------------------------------------------
            # **用激光按列把色块切成「属于同一物体」的若干簇**。
            #
            # 同一条激光束打到的必然是同一个最近障碍，所以若某列的实际距离与
            # 该簇不符，它就不属于这个物体。这一招同时解决两类合并：
            #   ① 物体与**墙**并在一起（墙列的距离是墙距，与前景物体不同）
            #   ② 相邻同色**物体**并在一起（各自距离不同）
            # 实测漏检里 95.7% 是这两类合并，而仅有「等宽二分」对二者都无能为力。
            # ---------------------------------------------------------------
            dcols = np.arange(c0, c1, dtype=np.float64)
            if lidar is not None and np.size(lidar) > 0:
                nl = int(np.size(lidar))
                lv = np.asarray(lidar, dtype=np.float64).reshape(-1)
                angs = np.arctan((dcols - (W - 1) / 2.0) / f)
                bi = np.round(((angs + np.pi) / (2 * np.pi)) * nl).astype(np.int64) % nl
                dl = lv[bi]
                # **前景提取（只剔除背景列），不做距离聚类。**
                # 色块底边给出 d0 = 物体中心距；激光打到物体表面，读数 ≤ d0；
                # 而背景（墙）列的读数会明显大于 d0。把背景列剔掉，剩下的就是前景。
                # **不能按激光距离聚类**：同一个大件（如床 half=0.75）自身激光读数
                # 跨度可达 2·half ≈ 1.5 m，聚类会把单体切碎 —— 实测那样召回
                # 反而从 67% 掉到 57%。
                keep = dl <= d0 + np.maximum(0.5, 0.30 * d0)
                segs = []
                s = 0
                for j in range(1, len(keep) + 1):
                    if j == len(keep) or keep[j] != keep[s]:
                        if keep[s] and (j - s) >= 2:
                            segs.append((int(dcols[s]), int(dcols[j - 1]) + 1, d0))
                        s = j
            else:
                segs = [(c0, c1, d0)]

            for (ca, cb, dseg) in segs:
                la, lb = ca - sl[1].start, cb - sl[1].start
                sm = sub_m[:, la:lb]
                if not sm.any():
                    continue
                yy, xx = np.nonzero(sm)
                if yy.size < min_px * 0.5:
                    continue
                # **行号必须加上裁剪窗偏移**：`yy` 是 `cc[sl]` 窗口内的局部行号，
                # 而 `horizon` 是整幅图像的全局行号。漏掉 `sl[0].start` 会让
                # `rr1 - horizon` 完全错位（召回从 67% 崩到 13%）。
                rr0 = sl[0].start + int(yy.min())
                rr1 = sl[0].start + int(yy.max()) + 1
                cc0 = ca + int(xx.min())
                cc1 = ca + int(xx.max()) + 1
                if rr1 - horizon <= 1.0:
                    continue
                d = f * cam_h / (rr1 - horizon)
                if abs(d - d0) > max(0.8, 0.5 * d0):
                    continue                   # 与整块几何不符，判为噪声
                clip_l = (cc0 == 0)
                clip_r = (cc1 == W)
                clip_top = (rr0 == 0)
                clip_bot = (rr1 == H)
                h = (rr1 - rr0) * d / f
                if not (0.20 <= h <= 2.60):
                    continue
                npx_s = yy.size
                if npx_s / float((rr1 - rr0) * (cc1 - cc0)) < \
                        min_fill * (0.80 if (clip_top or clip_bot) else 1.0):
                    continue
                if col_consistency:
                    smm = np.zeros_like(sub_m)
                    smm[yy, xx] = True
                    cmm = smm.sum(axis=0)
                    cmean = (np.where(smm[:, :, None], imgf[sl[0], sl[1]], 0.0).sum(axis=0)
                             / np.maximum(cmm, 1)[:, None])
                    # **只统计有标注像素的列**：没有标注像素的列 `cmean` 会是 0，
                    # 把整体标准差虚高。被**部分遮挡**的目标（如跟随任务里
                    # 被家具挡掉一半的人）外接框内必然出现整列无标注像素，
                    # 于是被误拒 —— 实测跟随任务中目标在 2 m 处**完全检不出**，
                    # 关闭本项后立刻正常检出（npx 460、距离 1.93 m 对真值 2.00 m）。
                    _nz = cmm > 0
                    if int(_nz.sum()) > 1 and \
                            float(np.linalg.norm(cmean[_nz].std(axis=0))) > max_col_std:
                        continue

                # 同一簇内仍可能并着同色邻居，再用等宽拆分兜底（k=1 能过就不拆）
                seg_w = cc1 - cc0
                got = None
                for kk in (1, 2, 3, 4):
                    sw = seg_w / kk
                    cls_k, cost = _shape_cost(h, sw * d / (2.0 * f), clip_l, clip_r,
                                              clip_top, clip_bot)
                    if cost <= max_shape_err:
                        got = (kk, sw, cls_k)
                        break
                if got is None:
                    continue
                kk, sw, cls = got
                half_e = sw * d / (2.0 * f)
                if not (0.06 <= half_e <= 1.30):
                    continue
                for j in range(kk):
                    ccx = cc0 + sw * (j + 0.5)
                    ang = float(np.arctan((ccx - (W - 1) / 2.0) / f))
                    # **表面距离按「单目为主、失效时切激光」融合**：
                    # - `clip_bot=False`：地面几何有效，表面距 = 中心距 − 半尺寸（单目误差小）
                    # - `clip_bot=True` ：色块底边触到图像下沿 → 地面公式**饱和**（0.752 m），
                    #   此时唯一可用的量是激光在角跨度内的最小读数
                    _sw = sw
                    _hit = float("nan")
                    if clip_bot:
                        if want_hit:
                            _surf, _hit = _lidar_surf(ccx - _sw / 2.0, ccx + _sw / 2.0)
                        else:
                            _surf = _lidar_surf(ccx - _sw / 2.0, ccx + _sw / 2.0)
                    else:
                        _surf = d - half_e
                    dets.append(Detection(ang, d, cls, name, h, half_e, npx_s // kk,
                                          surf=_surf, hit_bearing=_hit))
    return dets


def _shape_cost(h, half, clip_l=False, clip_r=False, clip_top=False, clip_bot=False):
    """**几何感知**的形状匹配代价（越小越像某个类别）。

    关键在于**被截断的维度只能当下界用**：

    | 情形 | 观测量的性质 | 正确的判据 |
    |---|---|---|
    | 物体贴画面左右边 | 观测宽度是真实宽度的**一部分** | 只要求 `类别半宽 ≥ 观测半宽` |
    | 顶边被画面裁掉 | 观测高度是真实高度的**一部分** | 只要求 `类别高度 ≥ 观测高度` |
    | 底边被画面裁掉 | 距离估计失真，两者都不可信 | 两个维度都只放下界 |

    早先直接对截断后的尺寸做**双侧**匹配，导致**贴边（35%）与顶部截断（11%）
    全部被误杀** —— 它们合计占漏检的 46%。

    Returns: (cls, cost)
    """
    qh = float(np.log(max(h, 1e-3)))
    qw = float(np.log(max(half, 1e-3)))
    lo_h = clip_top or clip_bot          # 高度只能当下界
    lo_w = clip_l or clip_r or clip_bot  # 半宽只能当下界
    ch, cw = _LOG_SHAPE[:, 0], _LOG_SHAPE[:, 1]
    ph = np.maximum(0.0, ch - qh) if lo_h else np.abs(ch - qh)
    pw = np.maximum(0.0, cw - qw) if lo_w else np.abs(cw - qw)
    cost = np.hypot(ph, pw)
    k = int(np.argmin(cost))
    return _CLASSES[k], float(cost[k])


def target_features(img, instruction, det_kw=None, max_range=MAX_RANGE, lidar=None,
                    pose=None, mode="none", memory=None, t=0):
    """把「**指令所指目标**在当前图像里的位置」编码成定长向量。

    这是「**感知 + 决策分离**」架构的接口：感知模块输出结构化结果，
    策略只消费它。**示范与策略共用同一路检测**，所以「观测 → 动作」
    是良定义映射。

    动机（见 `主创新A_诊断报告.md` §4.12.4）：端到端只喂 RGB 时，
    「从扫描切到追击」这个切换信号在图像上是个**稀疏、难感知**的事件，
    模仿者学不到，只能塌缩到多数类（永远扫描，闭环 0.0%）。
    把这些量显式给出来，切换就变成一次简单的阈值判断。

    Args:
        pose: `(x, y, theta)`。**仅 `mode="resolve"` 需要** —— 解析 L3 的
            「参考物 + 方位」要把两个检测投影到同一坐标系再比角度，
            而自车位姿本身就是模型的输入之一，不构成额外特权信息。
        mode: `"none"` / `"ref"` / `"resolve"`，见 `features_from_dets`。

    Returns:
        4 维（`mode="none"` / `"resolve"`）或 8 维（`mode="ref"`）float32。
    """
    return features_from_dets(detect(img, lidar=lidar, **(det_kw or {})),
                              instruction, max_range, pose=pose, mode=mode,
                              memory=memory, t=t)


def _obj_slot(d, max_range):
    """一个「物体位置」→ 4 维槽：`[是否检出, sin(方位), cos(方位), 距离/max_range]`。

    `d` 是 `(bearing, dist)` 二元组或 `None`（统一用元组，使**检测结果**与
    **记忆查询结果**能走同一个编码函数，不必为后者造一个假 `Detection`）。

    `d is None`（未检出/记忆为空）时返回 `[0, 0, 0, 1]` —— 用第 0 维的 0
    表示「无此物体」，因此不会与「方位 0、距离 0」混淆。
    """
    if d is None:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    b, dd = float(d[0]), float(d[1])
    return np.array([1.0, float(np.sin(b)), float(np.cos(b)),
                     float(min(1.0, dd / max_range))], dtype=np.float32)


def _slot_of(det, max_range):
    """`_obj_slot` 的检测版本：把 `Detection` 转成 `(bearing, dist)` 再编码。"""
    return _obj_slot(None if det is None else (det.bearing, det.dist), max_range)


def resolve_relation(dets, ref_cls, rel, pose, rel_tol_deg=50.0):
    """用**几何**解出「参考物 `rel` 方位上的那个目标」，返回该检测或 None。

    ## 为什么必须有这一步

    这是 L3 的**信息缺口**所在。`env._all_phrasings` 对 L3
    **只要求参考物类别唯一，不要求目标类别唯一**：

    ```python
    if sum(1 for o in objs if o.cls == ref.cls) > 1: continue   # 只约束参考物
    ```

    所以场景里可以同时存在**两个椅子**，而指令说「去桌子**左边**的椅子旁边」。
    旧特征只调 `match(dets, color, cls)`，它按「最近」取一个 —— 于是
    两个椅子时**约一半回合会指向错误的那一个**，而模型**无从判别**：
    桌子在哪根本没有进入输入。实测表现为 L3 成功率显著低于 L1/L2。

    ## 几何

    两个物体的世界坐标都可以由**已观测量**算出：

        world = pose_xy + dist · [cos(pose_θ + bearing), sin(pose_θ + bearing)]

    其中位姿来自里程计、方位与距离来自相机（单目几何），**不含任何真值**。
    再比较「参考物 → 目标」连线方向与指令要求的角度
    （`baselines._REL_ANG`：右 0、前 π/2、左 π、后 −π/2）。
    """
    from .baselines import _REL_ANG                 # 延迟导入，避免环状依赖

    if not dets or rel is None or ref_cls is None or pose is None:
        return None
    want_ang = _REL_ANG.get(rel)
    if want_ang is None:
        return None

    def wpos(d):
        a = float(pose[2]) + d.bearing
        return np.asarray(pose[:2], float) + d.dist * np.array([np.cos(a), np.sin(a)])

    refs = [d for d in dets if match([d], cls=ref_cls) is not None]
    if not refs:
        return None
    tol = np.deg2rad(rel_tol_deg)
    best = None
    for t in dets:
        for r in refs:
            v = wpos(t) - wpos(r)
            if float(np.hypot(*v)) < 0.4:            # 太近，方位不可靠
                continue
            a = float(np.arctan2(v[1], v[0]))
            if abs((a - want_ang + np.pi) % (2 * np.pi) - np.pi) < tol:
                if best is None or t.dist < best.dist:
                    best = t
                break
    return best


class TargetMemory:
    """跨帧累积「检测到的物体」的**世界坐标**估计 —— 与示范策略同机制。

    ## 为什么模型需要它

    `SweepDetectorPolicy` 默认 `mem_ttl=25`，**实测该记忆值 17 pp**
    （同一 200 回合集：开记忆 33.0% → 关记忆 16.0%），是示范策略最大的单一机制。
    但 `target_features` 给模型的只有**单帧**检测。

    后果与 §4.10 的特权信息问题**同构**：示范的动作是 `f(图像, **记忆**)` 的函数，
    而记忆不在单帧图像里 → 对模仿者是**一对多**映射（同一帧、记忆不同则动作不同），
    只能学到条件平均。**这不是特权信息**：世界坐标完全由「自车位姿（里程计）+
    相机测得的方位/距离」算出，示范与策略用的是同一路信息。

    ## 机制（刻意与 `SweepDetectorPolicy._mem_update` 保持一致）

    - 每帧把检测投影到世界坐标（物体静止，世界坐标可跨帧比较）
    - 与已有条目按「同色 + 距离 < `merge`」关联，位置做指数平滑（抑制测距抖动）
    - 超过 `ttl` 帧未再观测到的条目丢弃
    - 查询时按**当前位姿**重算方位/距离（物体不动、机器人动了）
    """

    def __init__(self, ttl=25, merge=0.7):
        self.ttl = int(ttl)
        self.merge = float(merge)
        self.items = []          # [world_xy(2,), color, cls, half, height, t]

    def reset(self):
        self.items = []

    def update(self, dets, pose, t):
        if self.ttl <= 0 or pose is None:
            return
        self.items = [m for m in self.items if t - m[5] <= self.ttl]
        for d in dets:
            a = float(pose[2]) + d.bearing
            p = np.asarray(pose[:2], float) + d.dist * np.array([np.cos(a), np.sin(a)])
            for m in self.items:
                if m[1] == d.color and float(np.hypot(*(p - m[0]))) < self.merge:
                    m[0] = 0.5 * (m[0] + p)              # 位置指数平滑
                    m[3], m[4], m[5] = d.half, d.height, t
                    break
            else:
                self.items.append([p, d.color, d.cls, d.half, d.height, t])

    def query(self, want_color, want_cls, pose):
        """→ `(bearing, dist)` 或 `None`。按**当前位姿**重算，取最近的一个。"""
        if not self.items or pose is None:
            return None
        cand = [m for m in self.items
                if (want_color is None or m[1] == want_color)
                and (want_cls is None or m[2] == want_cls)]
        if not cand:
            return None
        p0 = np.asarray(pose[:2], float)
        best = min(cand, key=lambda m: float(np.hypot(*(m[0] - p0))))
        v = best[0] - p0
        a = float(np.arctan2(v[1], v[0]) - pose[2])
        a = (a + np.pi) % (2 * np.pi) - np.pi
        return a, float(np.hypot(*v))


def features_from_dets(dets, instruction, max_range=MAX_RANGE, pose=None, mode="none",
                       memory=None, t=0):
    """`target_features` 的**复用版**：`dets` 由调用方预先算好。

    采集时同一帧要配多种指令表述，`detect` 与指令无关、只需跑一遍，
    用这个接口可以省掉 2/3 的检测开销。

    ## `mode` 三档（L3 空间关系的三种处理方式）

    | mode | 输出 | 参考物 | 说明 |
    |---|---|---|---|
    | `"none"` | 4 维 | **丢弃** | 旧行为。L3 时靠「最近」取目标，两个椅子时约一半指错 |
    | `"ref"` | **8 维** | 作为 4 维槽给出 | 信息**完备**了，但空间关系要模型自己从指令文本 + 两个坐标推 |
    | `"resolve"` | 4 维 | 由 `resolve_relation` 解掉 | 感知模块做几何解算，目标槽直接指向**正确**那个 |

    `"ref"` 与 `"resolve"` 是同一问题的两种分工：前者把推理交给模型
    （更强的「跨模态」主张，但要求模型学会坐标变换 + 角度比较），
    后者把推理交给感知模块（「感知 + 决策分离」，容易奏效）。
    两者的对照本身就是一条结论：**关系推理该放在哪一侧**。

    Args:
        pose: 仅 `mode="resolve"` 需要（见 `resolve_relation`）。
    """
    from .baselines import parse_instruction          # 延迟导入，避免环状依赖

    want_color, want_cls, ref_cls, rel = parse_instruction(str(instruction))
    if mode == "resolve" and rel is not None and ref_cls is not None:
        d = resolve_relation(dets, ref_cls, rel, pose)
        if d is None:                                 # 关系解不出来 → 退回最近者
            d = match(dets, color=want_color, cls=want_cls)
        return _slot_of(d, max_range)

    d = match(dets, color=want_color, cls=want_cls)
    tgt = _slot_of(d, max_range)

    if mode == "mem":
        # **记忆档**：并列给出「上一帧之前累积到的目标位置」。
        # 演示数据的动作依赖记忆（关记忆则示范 33.0% → 16.0%），
        # 单帧特征会让「观测 → 动作」变成一对多映射。见 `TargetMemory`。
        if memory is None:
            mem = _obj_slot(None, max_range)
        else:
            memory.update(dets, pose, t)
            q = memory.query(want_color, want_cls, pose)
            mem = _obj_slot(q, max_range)
        return np.concatenate([tgt, mem])

    if mode != "ref":
        return tgt
    # `ref` 档：并列给出参考物，使 L3 所需信息完整
    r = match(dets, color=None, cls=ref_cls) if ref_cls is not None else None
    return np.concatenate([tgt, _slot_of(r, max_range)])


# 各 mode 的感知特征维度（写数据集 / 建模型时都要用）
DET_DIM = {"none": 4, "ref": 8, "resolve": 4, "mem": 8}


def match(dets, color=None, cls=None, prefer_near=True, tol_deg=12.0):
    """在检测结果里挑出符合指令描述的那个。

    `color` / `cls` 为 None 表示该维度不约束（L1 只给类别、L2 给颜色+类别）。
    类别判定本来就带误差，所以对类别用**相似尺寸**做软匹配而不是硬相等。
    """
    cand = []
    for dt in dets:
        if color is not None and dt.color != color:
            continue
        if cls is not None and dt.cls != cls:
            # 软匹配：尺寸接近（log 距离 < 0.5）也接受
            q = np.log(np.array([max(dt.height, 1e-3), max(dt.half, 1e-3)]))
            tgt = _LOG_SHAPE[_CLASSES.index(cls)]
            if float(np.linalg.norm(tgt - q)) > 0.5:
                continue
        cand.append(dt)
    if not cand:
        return None
    if prefer_near:
        cand.sort(key=lambda d: d.dist)
    return cand[0]


# ---------------------------------------------------------------- 自检
def _self_test(seeds=(0, 1, 2), n_pose=40, verbose=True,
               max_shape_err=0.50, min_fill=0.65, max_col_std=6.0, use_lidar=True,
               detect_fn=None):
    """与真值对比，量化检测器的**召回率 / 方位误差 / 判类准确率**。

    这是必需的验证步骤：在用它替换示范策略的真值方位之前，
    必须先知道它有多准。
    """
    from .sensors import lidar_scan, raycast, render
    from .world import make_scene

    tot = hit = 0
    berr, derr = [], []
    cls_ok = cls_n = 0
    color_ok = color_n = 0
    fp = 0

    for seed in seeds:
        sc = make_scene(seed=seed)
        rng = np.random.default_rng(seed)
        for _ in range(n_pose):
            # 取一个自由位姿（与 env 的出生点采样同一套判据）
            p = sc.spawn.copy()
            for _try in range(200):
                q = np.array([rng.uniform(sc.bounds[0] + 0.4, sc.bounds[1] - 0.4),
                              rng.uniform(sc.bounds[2] + 0.4, sc.bounds[3] - 0.4)])
                if sc.is_free(q, margin=0.25):
                    p = q
                    break
            th = float(rng.uniform(-np.pi, np.pi))
            img = render(sc, np.array([p[0], p[1], th]), width=160, height=120)
            # 激光与策略实际拿到的一致（`scene.occ` = 墙 + 物体）
            img_lidar = lidar_scan(sc, np.array([p[0], p[1], th]), n_beams=120,
                                   noise_std=0.0) if use_lidar else None
            if detect_fn is not None:
                dets = detect_fn(img, lidar=img_lidar)
            else:
                dets = detect(img, lidar=img_lidar, max_shape_err=max_shape_err,
                              min_fill=min_fill, max_col_std=max_col_std)

            # ---- 真值：可见（FOV 内且未被墙遮挡）的物体 ----
            truth = []
            for o in sc.objects:
                v = o.xy - p
                d = float(np.hypot(*v))
                if d > MAX_RANGE:
                    continue
                ang = float(np.arctan2(v[1], v[0]) - th)
                ang = (ang + np.pi) % (2 * np.pi) - np.pi
                if abs(ang) > np.deg2rad(CAMERA_HFOV_DEG) / 2:
                    continue
                # 到物体中心投射射线，检查中途是否被墙挡住
                # 遮挡必须按**墙 + 其他物体**一起判（`sc.occ`）。早先只用
                # `wall_grid`，于是「被别的物体挡住」的也算真值可见，而那种情况
                # 根本不可能被检出 —— 召回被系统性低估。
                # 射线打到该物体自身近表面（≈ d − o.half）属正常，不算遮挡。
                dist, h_, _, _ = raycast(sc, p, np.array([np.arctan2(v[1], v[0])]),
                                         max_range=MAX_RANGE, grid=sc.occ)
                if h_[0] and dist[0] < d - o.half - 0.06:
                    continue
                truth.append((o, ang, d))

            for o, ang, d in truth:
                tot += 1
                # 找最近的检测（方位差 < tol_deg）
                best, bd = None, np.inf
                for dt in dets:
                    dd = abs(dt.bearing - ang)
                    if dd < bd:
                        best, bd = dt, dd
                if best is not None and bd < np.deg2rad(20.0) and abs(best.dist - d) < max(1.0, 0.5 * d):
                    hit += 1
                    berr.append(np.rad2deg(bd))
                    derr.append(abs(best.dist - d))
                    color_n += 1
                    if best.color == o.color:
                        color_ok += 1
                    cls_n += 1
                    if best.cls == o.cls:
                        cls_ok += 1
            # 误检：没有任何真值物体在附近的检测
            for dt in dets:
                if not any(abs(dt.bearing - a) < np.deg2rad(20.0) for _, a, _ in truth):
                    fp += 1

    if verbose and tot:
        print(f"  检测器自检（{len(seeds)} 场景 × {n_pose} 位姿）")
        print(f"    真值可见物体 {tot} 个 | 召回 {hit} 个（{100*hit/max(1,tot):.1f}%）| 误检 {fp} 个")
        if berr:
            print(f"    方位误差 中位 {np.median(berr):.1f}° / 均值 {np.mean(berr):.1f}°")
            print(f"    测距误差 中位 {np.median(derr):.2f} m")
            print(f"    判类准确率 {100*cls_ok/max(1,cls_n):.1f}% | 颜色准确率 {100*color_ok/max(1,color_n):.1f}%")
    return {"recall": hit / max(1, tot), "fp": fp,
            "bearing_med_deg": float(np.median(berr)) if berr else None,
            "cls_acc": cls_ok / max(1, cls_n), "color_acc": color_ok / max(1, color_n)}


def _diagnose(seeds=(0, 1, 2), n_pose=30, **kw):
    """统计**漏检原因分布** —— 不做这一步，优化召回就是瞎猜。

    对每个「真值可见但没被检出」的物体，记录它命中了哪些几何/外观限制：
    太远、贴边（部分出视野）、顶/底截断、被墙遮挡，或检测到了但颜色/类别判错。
    """
    from .sensors import lidar_scan, raycast, render
    from .world import make_scene

    names = list(COLORS)
    cnt = {}
    cls_miss = {}
    n_tot = 0
    for seed in seeds:
        sc = make_scene(seed=seed)
        rng = np.random.default_rng(seed)
        for _ in range(n_pose):
            p = sc.spawn.copy()
            for _t in range(200):
                q = np.array([rng.uniform(sc.bounds[0] + 0.4, sc.bounds[1] - 0.4),
                              rng.uniform(sc.bounds[2] + 0.4, sc.bounds[3] - 0.4)])
                if sc.is_free(q, margin=0.25):
                    p = q
                    break
            th = float(rng.uniform(-np.pi, np.pi))
            img = render(sc, np.array([p[0], p[1], th]), width=160, height=120)
            H, W, _ = img.shape
            dets = detect(img, lidar=lidar_scan(sc, np.array([p[0], p[1], th]),
                                                n_beams=120), **kw)
            f = (W / 2.0) / np.tan(np.deg2rad(CAMERA_HFOV_DEG) / 2.0)
            horizon = H / 2.0

            for o in sc.objects:
                v = o.xy - p
                d = float(np.hypot(*v))
                if d > MAX_RANGE:
                    continue
                ang = float(np.arctan2(v[1], v[0]) - th)
                ang = (ang + np.pi) % (2 * np.pi) - np.pi
                if abs(ang) > np.deg2rad(CAMERA_HFOV_DEG) / 2:
                    continue
                # 同 `_self_test`：遮挡要连其他物体一起判（见那里的说明）
                dist, hh, _, _ = raycast(sc, p, np.array([np.arctan2(v[1], v[0])]),
                                         max_range=MAX_RANGE, grid=sc.occ)
                occluded = bool(hh[0] and dist[0] < d - o.half - 0.06)
                if occluded:
                    continue                       # 被挡住 → 真值不可见，不算漏检
                n_tot += 1

                # 是否被检出（方位差 < 8° 且距离接近）
                got = any(abs(dt.bearing - ang) < np.deg2rad(8.0)
                          and abs(dt.dist - d) < max(0.6, 0.4 * d) for dt in dets)
                if got:
                    continue

                # 未被检出 → 归因（可多条同时命中）
                cx = (W - 1) / 2.0 + f * np.tan(ang)
                half_px = f * o.half / d
                c0, c1 = cx - half_px, cx + half_px
                r0 = horizon - f * (o.height - CAMERA_HEIGHT) / d
                r1 = horizon + f * CAMERA_HEIGHT / d
                rs = []
                if (c1 - c0) * (min(r1, H) - max(r0, 0)) < 40:
                    rs.append("像素太少")
                if d > 7.0:
                    rs.append("太远")
                # **先做基于检测结果的细分**（这才是真实失败模式），
                # 几何标志只作为附加信息。早先把「贴边」放在最前面，
                # 会把「贴边且完全没检出」和「贴边但检出判类错」混为一类。
                if not rs:
                    # 细分「其他」：按方位找最近的检测，看它到底差在哪一环节
                    near = None
                    for dt in dets:
                        if abs(dt.bearing - ang) < np.deg2rad(8.0):
                            if near is None or abs(dt.dist - d) < abs(near.dist - d):
                                near = dt
                    if near is None:
                        # 再看投影区域内的像素**有没有被归到该物体的颜色**上：
                        # 有 → blob 被合并/形状被拒；没有 → 颜色未归类
                        cc0 = int(max(0, np.floor(c0)))
                        cc1 = int(min(W, np.ceil(c1)))
                        rr0 = int(max(0, np.floor(r0)))
                        rr1 = int(min(H, np.ceil(r1)))
                        lab, nms = _pixel_labels(img, tol=kw.get("tol", 16.0))
                        ci = nms.index(o.color)
                        if cc1 > cc0 and rr1 > rr0:
                            frac = float((lab[rr0:rr1, cc0:cc1] == ci).mean())
                        else:
                            frac = 0.0
                        rs.append("A1 颜色未归类" if frac < 0.25 else "A2 blob合并/形状被拒")
                    elif near.color != o.color and near.cls != o.cls:
                        rs.append("B 检测到但颜色+判类都错")
                    elif near.color != o.color:
                        rs.append("C 检测到但颜色错")
                    elif near.cls != o.cls:
                        rs.append("D 检测到但判类错")
                    else:
                        rs.append("E 检测到但测距偏差大")
                if c0 < 0 or c1 > W:
                    rs[-1] += "｜贴边"
                if r0 <= 0.5:
                    rs[-1] += "｜顶部截断"
                if r1 >= H - 0.5:
                    rs[-1] += "｜底部截断"
                for r in rs:
                    cnt[r] = cnt.get(r, 0) + 1
                cls_miss[o.cls] = cls_miss.get(o.cls, 0) + 1

    tot_miss = sum(cls_miss.values())
    print(f"\n漏检归因（{n_tot} 个真值可见物体，漏检 {tot_miss} 个 = "
          f"{100*tot_miss/max(1,n_tot):.1f}%）")
    for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
        print(f"  {k:<22} {v:4d}  ({100*v/max(1,tot_miss):.1f}% 的漏检)")
    print("  按类别:", ", ".join(f"{k}:{v}" for k, v in
                                 sorted(cls_miss.items(), key=lambda x: -x[1])))
    return cnt


def _diag_blob(seeds=(0, 1, 2), n_pose=30, **kw):
    """对漏检物体，直接量**它所在色块的几何量与真值之比**。

    用来区分 `A2` 的两种成因：
    - 比值 ≈ 2/3 → **合并**（相邻同色物体并成一个连通域）
    - 比值 < 1 → **碎裂/截断**（被别的物体挡住一部分，或贴边）
    """
    from scipy import ndimage
    from .sensors import lidar_scan, raycast, render
    from .world import make_scene

    rw, rh = [], []
    for seed in seeds:
        sc = make_scene(seed=seed)
        rng = np.random.default_rng(seed)
        for _ in range(n_pose):
            p = sc.spawn.copy()
            for _t in range(200):
                q = np.array([rng.uniform(sc.bounds[0] + 0.4, sc.bounds[1] - 0.4),
                              rng.uniform(sc.bounds[2] + 0.4, sc.bounds[3] - 0.4)])
                if sc.is_free(q, margin=0.25):
                    p = q
                    break
            th = float(rng.uniform(-np.pi, np.pi))
            img = render(sc, np.array([p[0], p[1], th]), width=160, height=120)
            H, W, _ = img.shape
            f = (W / 2.0) / np.tan(np.deg2rad(CAMERA_HFOV_DEG) / 2.0)
            horizon = H / 2.0
            dets = detect(img, lidar=lidar_scan(sc, np.array([p[0], p[1], th]),
                                                n_beams=120), **kw)
            lab, names = _pixel_labels(img, tol=kw.get("tol", 16.0))
            for o in sc.objects:
                v = o.xy - p
                d = float(np.hypot(*v))
                if d > MAX_RANGE:
                    continue
                ang = float(np.arctan2(v[1], v[0]) - th)
                ang = (ang + np.pi) % (2 * np.pi) - np.pi
                if abs(ang) > np.deg2rad(CAMERA_HFOV_DEG) / 2:
                    continue
                dist, hh, _, _ = raycast(sc, p, np.array([np.arctan2(v[1], v[0])]),
                                         max_range=MAX_RANGE, grid=sc.occ)
                if hh[0] and dist[0] < d - o.half - 0.06:
                    continue
                if any(abs(dt.bearing - ang) < np.deg2rad(8.0)
                       and abs(dt.dist - d) < max(0.6, 0.4 * d) for dt in dets):
                    continue                       # 已检出，不统计

                ci = names.index(o.color)
                m = lab == ci
                if not m.any():
                    continue
                cl, _ = ndimage.label(m, structure=np.ones((3, 3), bool))
                col = int(round((W - 1) / 2.0 + f * np.tan(ang)))
                if not (0 <= col < W):
                    continue
                ls = cl[:, col]
                ls = ls[ls > 0]
                if ls.size == 0:
                    continue
                k = int(np.bincount(ls).argmax())
                ys, xs = np.nonzero(cl == k)
                r0, r1 = int(ys.min()), int(ys.max()) + 1
                c0, c1 = int(xs.min()), int(xs.max()) + 1
                if r1 - horizon <= 1:
                    continue
                de = f * CAMERA_HEIGHT / (r1 - horizon)
                rw.append((c1 - c0) * de / (2.0 * f) / o.half)
                rh.append((r1 - r0) * de / f / o.height)

    if rw:
        rw = np.array(rw)
        rh = np.array(rh)
        print(f"\n漏检色块的几何比值（{len(rw)} 个漏检）")
        print(f"  半宽 实测/真值: 中位 {np.median(rw):.2f}  "
              f"分位 25/50/75 = {np.percentile(rw, 25):.2f}/"
              f"{np.median(rw):.2f}/{np.percentile(rw, 75):.2f}")
        print(f"  高度 实测/真值: 中位 {np.median(rh):.2f}  "
              f"分位 25/50/75 = {np.percentile(rh, 25):.2f}/"
              f"{np.median(rh):.2f}/{np.percentile(rh, 75):.2f}")
        print(f"  判读：半宽比 ≈2 → 合并；<1 → 碎裂/截断")
    return rw, rh


def _rel_selftest(seeds=(0, 1, 2), n_pose=25, rel_tol_deg=50.0, verbose=True):
    """量化 L3 的**目标选择准确率**：`match`（旧）vs `resolve_relation`（新）。

    这是检验 L3 修法是否有效的直接判据。注意不能只看「闭环成功率」——
    成功率里混着检测、控制、搜索的误差；这里单独把**语义选择**这一环隔离出来。

    判定方法：把选中的检测按「自车位姿 + 方位/距离」投影回世界坐标，
    与任务真值目标比较（< 0.6 m 记为选对）。

    Returns:
        {mode: (选对次数, 总次数)}
    """
    from .env import calibrated_env
    from .sensors import render
    from .baselines import parse_instruction

    out = {}
    for mode in ("none", "resolve"):
        ok = tot = miss_det = 0
        for seed in seeds:
            env = calibrated_env(seed=seed)
            for ep in range(200):
                s = seed + ep * 37
                env.reset(seed=s)
                t = env.task
                if not t or t.get("level") != "L3":
                    continue
                # 只在「目标类不唯一」的回合上评估 —— 那才是真正有歧义的场景；
                # 目标唯一的 L3 回合两种方法都会选对，计入分母只会稀释差异。
                same = [o for o in env.scene.objects if o.cls == t["target_cls"]]
                if len(same) < 2:
                    continue
                goal = np.asarray(t["goal"], float)
                rng = np.random.default_rng(s)
                for _ in range(n_pose):
                    p = env.scene.spawn.copy()
                    for _try in range(200):
                        q = np.array([rng.uniform(env.scene.bounds[0] + 0.4,
                                                  env.scene.bounds[1] - 0.4),
                                      rng.uniform(env.scene.bounds[2] + 0.4,
                                                  env.scene.bounds[3] - 0.4)])
                        if env.scene.is_free(q, margin=0.25):
                            p = q
                            break
                    th = float(rng.uniform(-np.pi, np.pi))
                    pose = np.array([p[0], p[1], th])
                    img = render(env.scene, pose, width=160, height=120)
                    dets = detect(img)
                    want_c, want_k, ref_k, rel = parse_instruction(t["instruction"])
                    if mode == "resolve":
                        d = resolve_relation(dets, ref_k, rel, pose, rel_tol_deg)
                    else:
                        d = match(dets, color=want_c, cls=want_k)
                    if d is None:
                        miss_det += 1
                        continue
                    a = th + d.bearing
                    wp = pose[:2] + d.dist * np.array([np.cos(a), np.sin(a)])
                    tot += 1
                    ok += int(float(np.hypot(*(wp - goal))) < 0.6)
        out[mode] = (ok, tot, miss_det)

    if verbose:
        print(f"\nL3 目标选择准确率（仅「目标类不唯一」的回合，{n_pose} 个随机位姿/回合）")
        print("%-10s %10s %8s %12s" % ("选择器", "选对/总", "准确率", "无候选"))
        for m, (ok, tot, md) in out.items():
            lab = "match(旧)" if m == "none" else "resolve(新)"
            print("%-10s %10s %7.1f%% %12d" %
                  (lab, f"{ok}/{tot}", 100 * ok / max(1, tot), md))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="目标检测器自检")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--poses", type=int, default=40)
    ap.add_argument("--diag", action="store_true", help="漏检归因诊断")
    ap.add_argument("--rel-test", action="store_true",
                    help="L3 目标选择准确率：match(旧) vs resolve(新)")
    ap.add_argument("--diag-blob", action="store_true", help="漏检色块的几何比值诊断")
    ap.add_argument("--sweep", action="store_true",
                    help="扫描形状/填充度/纹理阈值，找召回与误检的平衡点")
    a = ap.parse_args()
    if a.rel_test:
        _rel_selftest(seeds=tuple(range(a.seeds)), n_pose=a.poses)
    elif a.diag_blob:
        _diag_blob(seeds=tuple(range(a.seeds)), n_pose=a.poses)
    elif a.diag:
        _diagnose(seeds=tuple(range(a.seeds)), n_pose=a.poses)
    elif a.sweep:
        import itertools
        combos = [(0.50, 0.65, 6.0), (0.50, 0.70, 5.5), (0.50, 0.75, 5.0),
                  (0.45, 0.70, 6.0), (0.45, 0.75, 5.5), (0.60, 0.70, 5.5),
                  (0.50, 0.70, 6.0), (0.55, 0.70, 5.5)]
        print("%-8s %-8s %-7s | %-8s %-8s %-8s %-8s" %
              ("shapeErr", "fill", "colStd", "召回", "误检", "方位°", "判类"))
        for se, mf, cs in combos:
            r = _self_test(seeds=tuple(range(a.seeds)), n_pose=a.poses,
                           verbose=False, max_shape_err=se, min_fill=mf, max_col_std=cs)
            print("%-8.2f %-8.2f %-7.1f | %7.1f%% %8d %8.2f %7.1f%%" %
                  (se, mf, cs, 100 * r["recall"], r["fp"],
                   r["bearing_med_deg"] or -1, 100 * r["cls_acc"]))
    else:
        _self_test(seeds=tuple(range(a.seeds)), n_pose=a.poses)
