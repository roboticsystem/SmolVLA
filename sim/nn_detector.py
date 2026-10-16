"""仿真真值监督的小型检测器（替代颜色-连通域方案）。

## 为什么换掉颜色-连通域（见 `主创新A_诊断报告.md` §4.14）

上一代检测器是「按颜色分段 + 二维连通域 + 几何反推」，实测：

- 召回上界 **75.9%**（把所有过滤条件全关）；要到这个上界必须接受 181 个误检
- 而**误检比漏检更有害**：示范策略 SR 在「更严的检测器」下反而更高
  （42.5% vs 37.5%），因为误检会让策略去追幻影目标
- 剩余漏检里 **95.7%** 是「物体与墙同色合并」——颜色本身就是不可靠的判据

**本模块改用监督学习**：让网络直接从图像学「哪一列有物体、是什么、多远」。

## 为什么是**逐列**预测

`sim/sensors.py` 的渲染器把每个物体画成一段**按列占据的矩形 sprite**：

```
cx      = (W-1)/2 + f·tan(ang)      ← 列位置 → 方位角
r1      = horizon + f·cam_h/d       ← 底边行 → 距离
c0..c1  = cx ± f·half/d             ← 列范围
```

因此「每一列的可见物体是唯一的」（由 z-buffer 决定），
逐列多任务预测天然贴合渲染结构，不需要 anchor box 或 NMS：

| 每列输出 | 含义 | 损失 |
|---|---|---|
| `obj` (1) | 该列是否被物体覆盖 | BCE |
| `cls` (8) | 物体类别 | CE（仅物体列） |
| `color` (6) | 物体颜色 | CE（仅物体列） |
| `dist` (1) | 物体**中心距**（米，log 空间） | L1（仅物体列） |
| `half` (1) | 物体半宽（米，log 空间） | L1（仅物体列） |

推理时把 objectness 高的连续同类列聚成一个检测，方位取列中心、距离取中位。

## 真值从哪来

不重新实现渲染，而是**复刻 `sensors.render` 的 sprite 循环**并记录 z-buffer 的
归属：每个被绘制的列属于哪个物体、其距离是多少。这样真值与渲染器**严格一致**，
也自动继承了「被墙挡住 = 不可见」的语义。
"""

from __future__ import annotations

import os
import time

import numpy as np

from .sensors import (CAMERA_HEIGHT, CAMERA_HFOV_DEG, LIDAR_MAX_RANGE, render)
from .world import COLORS, SEMANTICS

# 类别顺序与 `perception.CLASS_SHAPE` 保持一致，便于两者对比
CLASSES = ["chair", "table", "sofa", "bed", "shelf", "plant", "box", "tv"]
_CLS2ID = {c: i for i, c in enumerate(CLASSES)}
COLOR_NAMES = list(COLORS)
_COLOR2ID = {c: i for i, c in enumerate(COLOR_NAMES)}

N_CLS = len(CLASSES)
N_COL = len(COLOR_NAMES)


# ---------------------------------------------------------------- 真值提取
def column_truth(scene, pose, pedestrians=(), width=160, height=120,
                 hfov_deg=CAMERA_HFOV_DEG, max_range=LIDAR_MAX_RANGE,
                 cam_h=CAMERA_HEIGHT):
    """复刻渲染器的 sprite 循环，返回**逐列真值**。

    Returns:
        owner (W,) int   该列可见物体的索引（-1 = 背景）
        dist  (W,) f32   该物体的**中心距**（米）；背景为 0
    """
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    W, H = int(width), int(height)
    f = (W / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    horizon = H / 2.0

    # 墙体深度（与 render 的第一步一致）
    from .sensors import raycast
    cols_px = np.arange(W, dtype=np.float64)
    ang_off = np.arctan((cols_px - (W - 1) / 2.0) / f)
    dist_wall, _, _, _ = raycast(scene, (x, y), ang_off + th,
                                 max_range=max_range, grid=scene.wall_grid)
    zbuf = dist_wall.copy()
    owner = np.full(W, -1, dtype=np.int64)
    dist = np.zeros(W, dtype=np.float32)

    objs = list(scene.objects)
    sprites = [(i, o.xy, o.half, o.height) for i, o in enumerate(objs)]
    for j, p in enumerate(pedestrians):
        sprites.append((len(objs) + j, np.asarray(p.xy), p.radius, 1.70))

    for oi, xy, half, oh in sprites:
        relv = np.asarray(xy, dtype=np.float64) - np.array([x, y])
        d = float(np.hypot(relv[0], relv[1]))
        if d < 1e-3 or d > max_range:
            continue
        ang = np.arctan2(relv[1], relv[0]) - th
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        if abs(ang) > np.deg2rad(hfov_deg) / 2 * 1.4:
            continue
        cx = int(round((W - 1) / 2.0 + f * np.tan(ang)))
        half_px = max(1, int(round(f * half / d)))
        c0, c1 = cx - half_px, cx + half_px + 1
        if c1 <= 0 or c0 >= W:
            continue
        c0c, c1c = max(0, c0), min(W, c1)
        r0 = int(np.clip(np.floor(horizon - f * (oh - cam_h) / d), 0, H))
        r1 = int(np.clip(np.ceil(horizon + f * cam_h / d), 0, H))
        if r1 <= r0:
            continue
        sub = zbuf[c0c:c1c]
        vis = d < sub
        if not vis.any():
            continue
        sel = np.nonzero(vis)[0] + c0c
        owner[sel] = oi
        dist[sel] = d
        zbuf[c0c:c1c] = np.where(vis, d, sub)

    return owner, dist


def truth_targets(scene, pose, pedestrians=(), width=160, height=120, **kw):
    """把逐列真值转成训练目标张量。"""
    owner, dist = column_truth(scene, pose, pedestrians, width, height, **kw)
    W = width
    obj = (owner >= 0).astype(np.float32)
    cls = np.zeros(W, dtype=np.int64)
    color = np.zeros(W, dtype=np.int64)
    dd = np.zeros(W, dtype=np.float32)
    hh = np.zeros(W, dtype=np.float32)
    objs = list(scene.objects)
    for c in range(W):
        oi = int(owner[c])
        if oi < 0 or oi >= len(objs):        # 行人列按背景处理（本课题目标不含行人）
            obj[c] = 0.0
            continue
        o = objs[oi]
        cls[c] = _CLS2ID[o.cls]
        color[c] = _COLOR2ID[o.color]
        dd[c] = dist[c]
        hh[c] = o.half
    return obj, cls, color, dd, hh


# ---------------------------------------------------------------- 网络
def build_net(width=160, height=120, ch=48):
    """极小的逐列预测网络。

    高度方向逐步池化到 1，宽度方向保持分辨率（每列一个输出）——
    这正是「逐列预测」的结构性先验，参数量只有几十万，
    在单卡上几分钟就能训好。
    """
    import torch.nn as nn

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                             nn.ReLU(inplace=True))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                blk(3, ch), nn.MaxPool2d((2, 1)),
                blk(ch, ch * 2), nn.MaxPool2d((2, 1)),
                blk(ch * 2, ch * 2), nn.MaxPool2d((2, 1)),
                blk(ch * 2, ch * 2), nn.MaxPool2d((2, 1)),
            )
            self.trunk = nn.Sequential(nn.Conv1d(ch * 2, ch * 2, 3, padding=1),
                                       nn.ReLU(inplace=True))
            # 注意：**不能叫 `self.half`** —— `nn.Module.half()` 是内置方法，
            # 覆盖它会在训练时报 "Module.half() takes 1 positional argument but 2 were given"。
            self.h_obj = nn.Conv1d(ch * 2, 1, 1)
            self.h_cls = nn.Conv1d(ch * 2, N_CLS, 1)
            self.h_col = nn.Conv1d(ch * 2, N_COL, 1)
            self.h_dist = nn.Conv1d(ch * 2, 1, 1)
            self.h_half = nn.Conv1d(ch * 2, 1, 1)

        def forward(self, x):
            z = self.body(x)                     # (B, C, 1, W)
            z = z.mean(dim=2)                    # 高度方向取均值 → (B, C, W)
            z = self.trunk(z)
            return {"obj": self.h_obj(z).squeeze(1),
                    "cls": self.h_cls(z),
                    "color": self.h_col(z),
                    "dist": self.h_dist(z).squeeze(1),
                    "half": self.h_half(z).squeeze(1)}

    return Net()


def loss_fn(out, tgt, w_dist=1.0):
    """逐列多任务损失。分类与回归只在**物体列**上计算（背景列无意义）。"""
    import torch
    import torch.nn.functional as F

    obj, cls, color, dd, hh = tgt
    m = obj > 0.5
    o = F.binary_cross_entropy_with_logits(out["obj"], obj)
    if m.any():
        ce_c = F.cross_entropy(out["cls"].transpose(1, 2)[m], cls[m])
        ce_k = F.cross_entropy(out["color"].transpose(1, 2)[m], color[m])
        r_d = F.smooth_l1_loss(out["dist"][m], torch.log(dd[m].clamp(min=0.2)))
        r_h = F.smooth_l1_loss(out["half"][m], torch.log(hh[m].clamp(min=0.05)))
    else:
        ce_c = ce_k = r_d = r_h = o * 0.0
    return o + ce_c + ce_k + w_dist * (r_d + r_h)


# ---------------------------------------------------------------- 推理
# ---------------------------------------------------------------- 数据生成
def _rand_pose(sc, rng):
    for _ in range(300):
        p = np.array([rng.uniform(sc.bounds[0] + 0.4, sc.bounds[1] - 0.4),
                      rng.uniform(sc.bounds[2] + 0.4, sc.bounds[3] - 0.4)])
        if sc.is_free(p, margin=0.25):
            return p, float(rng.uniform(-np.pi, np.pi))
    return sc.spawn.copy(), 0.0


def gen_frames(n_frames=30000, seed=0, width=160, height=120, peds=0,
               calib=True, verbose=True):
    """采集「图像 + 逐列真值」对。

    刻意**同时覆盖两种场地**（标定开阔场地 + 多房间），
    否则模型会把「没有墙」当成先验。
    """
    from .env import SemanticNavEnv
    from .world import make_scene

    rng = np.random.default_rng(seed)
    imgs = np.zeros((n_frames, height, width, 3), dtype=np.uint8)
    O = np.zeros((n_frames, width), dtype=np.float32)
    C = np.zeros((n_frames, width), dtype=np.int64)
    K = np.zeros((n_frames, width), dtype=np.int64)
    D = np.zeros((n_frames, width), dtype=np.float32)
    Hh = np.zeros((n_frames, width), dtype=np.float32)

    t0 = time.time()
    scene = None
    ped_objs = []
    per_scene = 24
    for i in range(n_frames):
        if i % per_scene == 0:
            s = seed * 1000 + i // per_scene
            if calib and (i // per_scene) % 2 == 0:
                env = SemanticNavEnv(seed=s, nx=1, ny=1, room_w=8.0, room_h=6.0,
                                     n_objects_per_room=(6, 9), n_peds=0)
            else:
                env = SemanticNavEnv(seed=s, n_peds=0)
            scene = env.scene
            ped_objs = []
            if peds:
                from .world import make_pedestrians
                ped_objs = make_pedestrians(scene, peds, seed=s)
        p, th = _rand_pose(scene, rng)
        pose = np.array([p[0], p[1], th])
        imgs[i] = render(scene, pose, ped_objs, width=width, height=height)
        o, c, k, d, hh = truth_targets(scene, pose, ped_objs, width, height)
        O[i], C[i], K[i], D[i], Hh[i] = o, c, k, d, hh
        if verbose and (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{n_frames} 帧  前景列占比 "
                  f"{100 * O.mean():.1f}%  {time.time() - t0:.0f}s")
    return imgs, O, C, K, D, Hh


def save_frames(path, data):
    np.savez_compressed(path, imgs=data[0], obj=data[1], cls=data[2],
                        color=data[3], dist=data[4], half=data[5])


def load_frames(path):
    z = np.load(path)
    return z["imgs"], z["obj"], z["cls"], z["color"], z["dist"], z["half"]


# ---------------------------------------------------------------- 训练
def train(data, out="models/nn_det.pt", epochs=12, batch=64, lr=3e-3,
          val_frac=0.08, device=None, verbose=True, augment=True):
    """训练逐列检测器。数据量在几万帧量级时，单卡几分钟即可收敛。"""
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    imgs, O, C, K, D, Hh = data
    n = len(imgs)
    idx = np.random.default_rng(0).permutation(n)
    nv = max(1, int(n * val_frac))
    va, tr = idx[:nv], idx[nv:]

    net = build_net(imgs.shape[2], imgs.shape[1]).to(dev)
    # 前景列很少（几只物体占几十列 / 共 160 列），用 pos_weight 抵消类别不平衡
    pos_w = torch.tensor([max(1.0, (1 - O.mean()) / max(O.mean(), 1e-3))],
                         device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(1, len(tr) // batch))

    def to_t(ids, augment=False):
        x = torch.from_numpy(imgs[ids]).to(dev).float().div_(255.0).permute(0, 3, 1, 2)
        if augment:
            # **在线光度增强**，与 `sensors.photometric_augment` 同一套扰动
            # （亮度 / 对比度 / 伽马 / 噪声）。**这一步是必需的**：训练帧是干净渲染，
            # 而评测环境开着 `camera_augment=0.25`；不加增强的话 NN 在评测时掉点，
            # 实测示范策略 SR 会从 37.5% 掉到 25.0%。规则式检测器不受影响，
            # 所以这个坑只在换成 NN 之后才暴露出来。
            n = len(ids)
            sh = (n, 1, 1, 1)
            u = lambda a, b: torch.empty(sh, device=dev).uniform_(a, b)
            x = x * u(0.75, 1.25)
            x = (x - 0.5) * u(0.75, 1.25) + 0.5
            x = x.clamp(0, 1).pow(u(0.70, 1.40))
            x = x + torch.randn_like(x) * (0.02 + 0.03 * u(0, 1))
            x = x.clamp(0, 1)
        t = (torch.from_numpy(O[ids]).to(dev),
             torch.from_numpy(C[ids]).to(dev),
             torch.from_numpy(K[ids]).to(dev),
             torch.from_numpy(D[ids]).to(dev),
             torch.from_numpy(Hh[ids]).to(dev))
        return x, t

    def batch_loss(out, t):
        """训练用损失：在 `loss_fn` 基础上给 objectness 加 `pos_weight`。

        前景列只占约 1/5，不加权的话网络会倾向全部预测为背景。
        """
        import torch.nn.functional as F

        obj = t[0]
        m = obj > 0.5
        l = F.binary_cross_entropy_with_logits(out["obj"], obj, pos_weight=pos_w)
        if m.any():
            l = l + F.cross_entropy(out["cls"].transpose(1, 2)[m], t[1][m])
            l = l + F.cross_entropy(out["color"].transpose(1, 2)[m], t[2][m])
            l = l + F.smooth_l1_loss(out["dist"][m],
                                     torch.log(t[3][m].clamp(min=0.2)))
            l = l + F.smooth_l1_loss(out["half"][m],
                                     torch.log(t[4][m].clamp(min=0.05)))
        return l

    hist = []
    for ep in range(epochs):
        net.train()
        perm = np.random.default_rng(ep).permutation(len(tr))
        tot, nb = 0.0, 0
        for b in range(0, len(tr) - batch + 1, batch):
            ids = tr[perm[b:b + batch]]
            x, t = to_t(ids, augment=True)
            l = batch_loss(net(x), t)
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            tot += float(l.item())
            nb += 1
        net.eval()
        with torch.no_grad():
            accs, f1s = [], []
            for b in range(0, len(va), batch):
                ids = va[b:b + batch]
                x, t = to_t(ids)
                pred = net(x)          # 不能叫 out：会遮蔽函数参数（保存路径）
                pr = (torch.sigmoid(pred["obj"]) > 0.5).float()
                gt = t[0]
                tp = float((pr * gt).sum())
                fp = float((pr * (1 - gt)).sum())
                fn = float(((1 - pr) * gt).sum())
                f1s.append(2 * tp / max(1e-6, 2 * tp + fp + fn))
                accs.append(float((pr == gt).float().mean()))
        hist.append((ep, tot / max(1, nb), float(np.mean(f1s)), float(np.mean(accs))))
        if verbose:
            print(f"  epoch {ep + 1:2d}/{epochs}  loss {hist[-1][1]:.4f}  "
                  f"列 F1 {hist[-1][2]:.3f}  逐列准确率 {hist[-1][3]:.3f}")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({"state": net.state_dict(), "width": imgs.shape[2],
                "height": imgs.shape[1], "hist": hist}, out)
    return net, hist


# ---------------------------------------------------------------- 推理封装
class NNDetector:
    """把训练好的逐列检测器包装成与 `perception.detect` 同签名的接口。

    `lidar` 参数**刻意忽略**：本模型完全从图像学习「哪一列有物体」，
    不再依赖激光做前景提取（上一代方案正是靠它，但代价是引入额外假设）。
    """

    def __init__(self, ckpt="models/nn_det.pt", threshold=0.5, device=None):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt, map_location=self.device)
        self.net = build_net(ck["width"], ck["height"]).to(self.device)
        self.net.load_state_dict(ck["state"])
        self.net.eval()
        self.threshold = float(threshold)

    def detect(self, img, lidar=None, **kw):
        import torch
        x = torch.from_numpy(np.asarray(img)).to(self.device).float() \
            .div_(255.0).permute(2, 0, 1)[None]
        with torch.no_grad():
            o = self.net(x)
        out = {k: v[0].cpu().numpy() for k, v in o.items()}
        return detections_from_net(out, threshold=self.threshold)


def detections_from_net(out, threshold=0.5, tol_deg=None, width=160,
                        hfov_deg=CAMERA_HFOV_DEG):
    """把逐列输出聚成检测结果。"""
    from .perception import Detection

    W = out["obj"].shape[-1]
    f = (W / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    p_obj = out["obj"]
    # 注意轴号：类别 logits 形状是 **(N_CLS, W)**，要拿「每一列的类别」必须
    # 在 **axis 0**（类别维）上取 argmax。写成 `argmax(1)` 得到的是 (N_CLS,)，
    # 后续按列索引就越界（踩过：报 "index 15 is out of bounds for axis 0 with size 8"）。
    cls = out["cls"].argmax(0)
    color = out["color"].argmax(0)
    dist = np.exp(np.clip(out["dist"], -2.0, 3.0))
    half = np.exp(np.clip(out["half"], -4.0, 1.0))

    dets = []
    c = 0
    while c < W:
        if p_obj[c] < threshold:
            c += 1
            continue
        c1 = c
        while (c1 + 1 < W and p_obj[c1 + 1] >= threshold
               and cls[c1 + 1] == cls[c] and color[c1 + 1] == color[c]):
            c1 += 1
        if c1 - c + 1 >= 2:
            k = int(color[c])
            dets.append(Detection(float(np.arctan(((c + c1) / 2.0 - (W - 1) / 2.0) / f)),
                                  float(np.median(dist[c:c1 + 1])),
                                  CLASSES[int(cls[c])], COLOR_NAMES[k],
                                  float(np.median(half[c:c1 + 1])) * 2.0,
                                  float(np.median(half[c:c1 + 1])),
                                  int((p_obj[c:c1 + 1] >= threshold).sum())))
        c = c1 + 1
    return dets


# ---------------------------------------------------------------- 命令行
def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="仿真真值监督的小型检测器")
    ap.add_argument("--gen", type=int, default=0, help="生成 N 帧训练数据后退出")
    ap.add_argument("--data", default="export/det_frames.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--no-augment", action="store_true",
                    help="关闭在线光度增强（默认开启，见 train 中的说明）")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="models/nn_det.pt")
    ap.add_argument("--eval", action="store_true", help="与真值对比（同 perception 的口径）")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--poses", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()

    if a.gen:
        print(f"生成 {a.gen} 帧「图像 + 逐列真值」…")
        d = gen_frames(n_frames=a.gen, seed=a.seed)
        save_frames(a.data, d)
        print(f"  前景列占比 {100 * d[1].mean():.1f}%  →  {a.data}")
        return
    if a.train:
        print(f"读取 {a.data} …")
        d = load_frames(a.data)
        print(f"  {len(d[0])} 帧，前景列占比 {100 * d[1].mean():.1f}%")
        net, hist = train(d, out=a.out, epochs=a.epochs, batch=a.batch,
                          augment=not a.no_augment)
        print(f"  已保存 → {a.out}")
        return
    if a.eval:
        from .perception import _self_test
        det = NNDetector(a.out, threshold=a.threshold)
        r = _self_test(seeds=tuple(range(a.seeds)), n_pose=a.poses,
                       detect_fn=det.detect)
        print(f"  （对照：颜色-连通域方案 同口径 召回 64.8% / 误检 30 / 判类 74.3%）")
        return
    ap.print_help()


if __name__ == "__main__":
    _cli()
