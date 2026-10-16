"""传感器仿真：2D 激光雷达 + 第一人称语义 RGB 相机。

对齐真机（不新增传感器）：
- 激光雷达：D6 级 —— 360° 扫描、最大 12 m、可配束数与角分辨率
- 相机：C70 级 —— 水平 FOV 64.5°，针孔模型

实现要点：
- 雷达用 **DDA（Amanatides-Woo）光线投射**，对全部射线向量化同步推进，无 Python 逐射线循环
- 相机背景只对**墙体**投射，家具/行人作为 **sprite** 按真实高度绘制并做逐列深度测试，
  因此近处物体会正确遮挡远处墙面
"""

from __future__ import annotations

import numpy as np

from .world import CELL, SEMANTICS

# 真机参数
LIDAR_MAX_RANGE = 12.0
CAMERA_HFOV_DEG = 64.5
CAMERA_HEIGHT = 0.35  # 相机离地高度（小车的相机大致高度）


def raycast(scene, origin, angles, max_range=LIDAR_MAX_RANGE, grid=None, max_steps=600):
    """对栅格做 2D 光线投射（DDA）。

    Args:
        scene: Scene
        origin: (2,) 世界坐标
        angles: (N,) 弧度
        grid: 布尔栅格；默认用 scene.occ（墙 + 物体）

    Returns:
        dist (N,)  命中距离（未命中为 max_range）
        hit  (N,)  是否命中
        rows, cols (N,) 命中格索引（未命中为 -1）
    """
    g = scene.occ if grid is None else grid
    H, W = g.shape
    xmin, _, ymin, _ = scene.bounds
    ox, oy = float(origin[0]), float(origin[1])

    dx = np.cos(angles).astype(np.float64)
    dy = np.sin(angles).astype(np.float64)
    N = angles.shape[0]

    gx = (ox - xmin) / CELL
    gy = (oy - ymin) / CELL
    # 必须是长度为 N 的数组：若写成标量，只有进入 DDA 循环后才会被 np.where 广播成数组，
    # 而「起点即在障碍内」的分支会在广播前索引标量而报错。
    ix = np.full(N, int(np.floor(gx)), dtype=np.int64)
    iy = np.full(N, int(np.floor(gy)), dtype=np.int64)

    stepx = np.where(dx > 0, 1, -1).astype(np.int64)
    stepy = np.where(dy > 0, 1, -1).astype(np.int64)

    inf = np.inf
    adx = np.abs(dx)
    ady = np.abs(dy)
    tdx = np.where(adx > 1e-12, CELL / np.where(adx > 1e-12, adx, 1.0), inf)
    tdy = np.where(ady > 1e-12, CELL / np.where(ady > 1e-12, ady, 1.0), inf)

    # 到第一条格边界的距离
    bxx = (np.floor(gx) + (stepx > 0)) * CELL + xmin
    byy = (np.floor(gy) + (stepy > 0)) * CELL + ymin
    tmx = np.where(adx > 1e-12, (bxx - ox) / np.where(adx > 1e-12, dx, 1.0), inf)
    tmy = np.where(ady > 1e-12, (byy - oy) / np.where(ady > 1e-12, dy, 1.0), inf)

    dist = np.full(N, float(max_range))
    hit = np.zeros(N, dtype=bool)
    rows = np.full(N, -1, dtype=np.int64)
    cols = np.full(N, -1, dtype=np.int64)
    done = np.zeros(N, dtype=bool)

    # 起点本身就在障碍内 → 距离 0
    inb = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    start_hit = np.zeros(N, dtype=bool)
    start_hit[inb] = g[iy[inb], ix[inb]]
    if start_hit.any():
        dist[start_hit] = 0.0
        hit[start_hit] = True
        rows[start_hit] = iy[start_hit]
        cols[start_hit] = ix[start_hit]
        done |= start_hit

    for _ in range(max_steps):
        if done.all():
            break
        take_x = tmx <= tmy
        t_cur = np.minimum(tmx, tmy)

        ix = np.where(take_x, ix + stepx, ix)
        iy = np.where(take_x, iy, iy + stepy)
        tmx = np.where(take_x, tmx + tdx, tmx)
        tmy = np.where(take_x, tmy, tmy + tdy)

        oob = (ix < 0) | (ix >= W) | (iy < 0) | (iy >= H)
        beyond = t_cur > max_range
        newly_done = oob | beyond
        check = ~done & ~newly_done
        if check.any():
            ci = np.clip(iy, 0, H - 1)
            cj = np.clip(ix, 0, W - 1)
            blocked = check & g[ci, cj]
            if blocked.any():
                dist[blocked] = t_cur[blocked]
                hit[blocked] = True
                rows[blocked] = iy[blocked]
                cols[blocked] = ix[blocked]
                newly_done = newly_done | blocked
        done |= newly_done

    return dist, hit, rows, cols


def lidar_scan(scene, pose, n_beams=120, max_range=LIDAR_MAX_RANGE, fov_deg=360.0,
               noise_std=0.0, rng=None, grid=None):
    """一帧 2D 激光雷达。返回 (n_beams,) 距离（米，未命中为 max_range）。"""
    if fov_deg >= 360.0:
        ang = np.linspace(-np.pi, np.pi, n_beams, endpoint=False)
    else:
        half = np.deg2rad(fov_deg) / 2
        ang = np.linspace(-half, half, n_beams)
    ang = ang + pose[2]
    d, _, _, _ = raycast(scene, pose[:2], ang, max_range=max_range, grid=grid)
    if noise_std > 0:
        r = rng if rng is not None else np.random.default_rng(0)
        d = np.clip(d + r.normal(0, noise_std, d.shape), 0.05, max_range)
    return d


def _shade(rgb, d, max_range):
    """按距离做亮度衰减（近亮远暗），返回 (...,3) uint8。"""
    f = np.clip(1.0 - 0.75 * (d / max_range), 0.25, 1.0)
    return np.clip(np.asarray(rgb, dtype=np.float64) * f[..., None], 0, 255).astype(np.uint8)


def render(scene, pose, pedestrians=(), width=160, height=120,
           hfov_deg=CAMERA_HFOV_DEG, max_range=LIDAR_MAX_RANGE,
           cam_h=CAMERA_HEIGHT, ceiling=(238, 238, 244), floor=(206, 200, 188),
           texture=True, fog_strength=0.45, fog_color=(150, 160, 175)):
    """第一人称 RGB 渲染（针孔模型 + 逐列深度测试）。

    为缩小 sim-to-real 视觉差距，加入三项廉价但有效的真实感处理：
    - **纹理**：墙体按命中格的哈希做逐格色彩抖动
    - **光照**：墙面垂直渐变 + 地面按透视距离衰减
    - **雾化**：按距离向雾色混合

    Returns:
        img (height, width, 3) uint8
    """
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    f = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)   # 焦距（像素）
    horizon = height / 2.0
    fog_c = np.asarray(fog_color, dtype=np.float64)

    cols_px = np.arange(width, dtype=np.float64)
    ang_off = np.arctan((cols_px - (width - 1) / 2.0) / f)
    dist_wall, _, wrow, wcol = raycast(scene, (x, y), ang_off + th, max_range=max_range,
                                       grid=scene.wall_grid)

    rows_idx = np.arange(height)[:, None]
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[:] = np.array(ceiling, dtype=np.uint8)

    # ---- 地面：按透视行距计算深度并衰减 ----
    below = rows_idx - horizon
    floor_dist = np.where(below > 0.5, f * cam_h / np.maximum(below, 0.5), max_range)
    floor_dist = np.clip(floor_dist, 0.1, max_range)
    fsh = np.asarray(floor, dtype=np.float64)[None, None, :] * \
        np.clip(1.0 - 0.72 * floor_dist / max_range, 0.2, 1.0)[:, :, None]
    fog_f = np.clip(fog_strength * floor_dist / max_range, 0.0, 1.0)[:, :, None]
    fsh = fsh * (1 - fog_f) + fog_c[None, None, :] * fog_f
    mask_floor_all = below > 0.5
    img[:] = np.where(mask_floor_all[..., None], np.clip(fsh, 0, 255).astype(np.uint8), img)

    # ---- 墙体列 ----
    wall_rgb = np.array(SEMANTICS[1][1], dtype=np.float64)
    hw = SEMANTICS[1][2]
    d_safe = np.maximum(dist_wall, 1e-3)
    topf = horizon - f * (hw - cam_h) / d_safe
    botf = horizon + f * cam_h / d_safe
    top = np.clip(np.round(topf), 0, height).astype(np.int64)
    bot = np.clip(np.round(botf), 0, height).astype(np.int64)

    # 纹理：命中格哈希 → 逐列抖动系数
    if texture:
        ri = np.clip(wrow, 0, scene.H - 1)
        ci = np.clip(wcol, 0, scene.W - 1)
        hsh = ((ri * 73856093) ^ (ci * 19349663)) % 997
        tex = 0.90 + 0.20 * (hsh / 996.0)
    else:
        tex = np.ones(width)

    base = wall_rgb[None, :] * tex[:, None]
    dist_f = 1.0 - 0.75 * (dist_wall / max_range)
    fog_w = np.clip(fog_strength * dist_wall / max_range, 0.0, 1.0)
    wall_col = base * dist_f[:, None]
    wall_col = wall_col * (1 - fog_w[:, None]) + fog_c[None, :] * fog_w[:, None]

    mask_wall = (rows_idx >= top[None, :]) & (rows_idx < bot[None, :])
    span = np.maximum(bot - top, 1)[None, :]
    rel = np.clip((rows_idx - top[None, :]) / span, 0.0, 1.0)   # 0=顶 1=底
    light = (0.78 + 0.22 * rel)                                  # 顶部稍暗
    wall_px = wall_col[None, :, :] * light[..., None]
    img[:] = np.where(mask_wall[..., None], np.clip(wall_px, 0, 255).astype(np.uint8), img)

    zbuf = dist_wall.copy()

    # ---- sprite：家具 + 行人 ----
    sprites = [(o.xy, o.half, o.height, o.rgb) for o in scene.objects]
    for p in pedestrians:
        # **颜色/身高优先取行人自带的**：跟随任务的目标穿「可辨识的马甲」，
        # 硬编码肤色会让目标与普通行人无法区分（实测：目标被画成肤色后
        # 检测器在 2 m 处只找到 3 px 的红块 —— 因为根本没有红块）。
        sprites.append((np.asarray(p.xy),
                        float(getattr(p, "radius", 0.25)),
                        float(getattr(p, "height", 1.70)),
                        np.asarray(getattr(p, "rgb", (214, 168, 132)),
                                   dtype=np.uint8)))

    for xy, half, oh, rgb in sprites:
        relv = np.asarray(xy, dtype=np.float64) - np.array([x, y])
        d = float(np.hypot(relv[0], relv[1]))
        if d < 1e-3 or d > max_range:
            continue
        ang = np.arctan2(relv[1], relv[0]) - th
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        if abs(ang) > np.deg2rad(hfov_deg) / 2 * 1.4:
            continue

        cx = int(round((width - 1) / 2.0 + f * np.tan(ang)))
        half_px = max(1, int(round(f * half / d)))
        c0, c1 = cx - half_px, cx + half_px + 1
        if c1 <= 0 or c0 >= width:
            continue
        c0c, c1c = max(0, c0), min(width, c1)

        r0 = int(np.clip(np.floor(horizon - f * (oh - cam_h) / d), 0, height))
        r1 = int(np.clip(np.ceil(horizon + f * cam_h / d), 0, height))
        if r1 <= r0:
            continue

        df = 1.0 - 0.75 * (d / max_range)
        fw = fog_strength * d / max_range
        shade = np.asarray(rgb, dtype=np.float64) * df
        shade = shade * (1 - fw) + fog_c * fw
        # sprite 垂直光照：朝地面略亮
        rloc = np.linspace(0.78, 1.0, max(r1 - r0, 1))[:, None]
        col2 = np.clip(shade[None, :] * rloc, 0, 255)

        sub = zbuf[c0c:c1c]
        vis = d < sub
        if not vis.any():
            continue
        cols_sel = np.where(vis)[0]
        block = img[r0:r1, c0c:c1c, :]
        # col2 为逐行色 (nrows,3)，按选中的列广播
        block[:, cols_sel] = col2[:, None, :].astype(np.uint8)
        zbuf[c0c:c1c] = np.where(vis, d, sub)

    return img


def photometric_augment(img, rng, strength=0.25):
    """域随机化：亮度 / 对比度 / 伽马 / 色偏 / 噪声。

    用于缩小 sim-to-real 视觉差距（E1），也可作为微调时的数据增强。
    """
    x = img.astype(np.float32) / 255.0
    bright = 1.0 + rng.uniform(-strength, strength)
    contrast = 1.0 + rng.uniform(-strength, strength)
    gamma = np.exp(rng.uniform(-0.35, 0.35))
    x = np.clip(x * bright, 0, 1)
    x = np.clip((x - 0.5) * contrast + 0.5, 0, 1)
    x = np.power(x, gamma)
    cast = 1.0 + rng.uniform(-strength * 0.4, strength * 0.4, size=3).astype(np.float32)
    x = np.clip(x * cast[None, None, :], 0, 1)
    if rng.random() < 0.7:
        x = x + rng.normal(0, 0.02 + 0.03 * rng.random(), x.shape).astype(np.float32)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def render_topdown(scene, pose=None, scale=24, pedestrians=(), goal=None):
    """俯视图（用于论文插图与调试）。"""
    H, W = scene.sem.shape
    img = scene.rgb.copy()
    if scale != 1:
        img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    if pose is not None:
        r, c = scene.w2c(pose[:2])
        rr, cc = int(r * scale), int(c * scale)
        s = max(2, scale // 3)
        img[max(0, rr - s):rr + s, max(0, cc - s):cc + s] = (255, 40, 40)
    for p in pedestrians:
        r, c = scene.w2c(p.xy)
        rr, cc = int(r * scale), int(c * scale)
        s = max(2, scale // 4)
        img[max(0, rr - s):rr + s, max(0, cc - s):cc + s] = (240, 170, 60)
    if goal is not None:
        r, c = scene.w2c(goal)
        rr, cc = int(r * scale), int(c * scale)
        s = max(3, scale // 2)
        img[max(0, rr - s):rr + s, max(0, cc - s):cc + s] = (30, 200, 30)
    return img
