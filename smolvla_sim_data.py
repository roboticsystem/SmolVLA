"""把自建仿真（`sim/gen_dataset.py`）的演示数据桥接到 SmolVLA 的输入约定。

官方预处理链（`processor_smolvla.py`，已核对）：
    1. 重命名观测  2. 加 batch 维  3. **task 必须以 "\\n" 结尾**
    4. TokenizerProcessorStep(tokenizer_name=config.vlm_model_name,
                              padding="max_length", padding_side="right",
                              max_length=config.tokenizer_max_length=48)
    5. 移到设备  6. 归一化

本模块只负责「数据 → 模型输入」这一层；归一化单独提供，便于与官方口径对齐或
替换为自定义统计量。

动作 chunk：flow matching 训练目标是一个长度为 `chunk_size` 的动作序列，
因此从同一条回合内切片 `actions[i : i + chunk_size]`（跨回合处丢弃）。
"""

from __future__ import annotations

import h5py
import numpy as np
import torch

VLM_NAME = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
IMG_KEY = "observation.images.camera1"          # 当前帧（兼容单帧用法）
STATE_KEY = "observation.state"
ACTION_KEY = "action"
# 帧堆叠用的相机键：camera1 = 当前帧，camera2 = 1 步前，camera3 = 2 步前 …
# 复用 SmolVLA 原生的多相机接口，**无需改模型结构**——每帧各自过 SigLIP，
# token 按 camera1..K 的顺序拼进 prefix。
IMG_KEYS = [f"observation.images.camera{i}" for i in range(1, 5)]
# 官方常量（lerobot.utils.constants），SmolVLAPolicy.forward 按这两个键取语言输入
LANG_TOKENS_KEY = "observation.language.tokens"
LANG_MASK_KEY = "observation.language.attention_mask"


class SimDemoData:
    """加载仿真演示数据，提供切块采样与归一化统计。"""

    def __init__(self, path, device="cuda", max_state_dim=32, tokenizer_max_length=48,
                 chunk_size=50, max_action_dim=2, n_frames=1, with_lidar=False,
                 with_det=False):
        self.path = path
        self.device = device
        self.max_state_dim = max_state_dim
        self.tok_len = tokenizer_max_length
        self.chunk = chunk_size
        self.max_action_dim = max_action_dim

        with h5py.File(path, "r") as f:
            self.images = f["images"][()]          # (N,H,W,3) uint8
            self.actions = f["actions"][()].astype(np.float32)   # (N,2)
            self.states = f["states"][()].astype(np.float32)
            self.lidar = f["lidar"][()].astype(np.float32)
            vocab = f["instr_vocab"][()]
            ids = f["instr_ids"][()]
            self.attrs = dict(f.attrs)
            ep_off = f["ep_offset"][()]
            # **必须在 with 块内读**：块外 f 已关闭，`"det_feat" in f` 会静默为假、
            # 取值会直接抛异常（踩过：训练一开始就报「数据集没有 det_feat 字段」，
            # 而 h5py 检查时该字段明明存在）。
            det_feat = f["det_feat"][()].astype(np.float32) if "det_feat" in f else None
        self.instr_vocab = [s.decode() if isinstance(s, bytes) else str(s) for s in vocab]
        self.instr_ids = ids
        self.instr = [self.instr_vocab[int(i)] for i in ids]
        self.ep_off = ep_off
        self.N = len(self.images)
        self.n_frames = int(max(1, n_frames))

        # **把激光雷达并进状态向量**（可选）。
        # 动机（见 主创新A_诊断报告.md §4.7）：上界探针显示「有无激光」造成 **22 pp** 的
        # 闭环成功率差异（61.0% → 39.0%），而 VLA 只吃 RGB、拿不到这个几何模态。
        # 并入方式很轻：拼到状态向量尾部，再把 `state_proj` 的输入维改掉即可，
        # **不需要动视觉塔或任何跨模态结构**。
        self.with_lidar = bool(with_lidar)
        if self.with_lidar:
            self.states = np.concatenate([self.states, self.lidar], axis=1)

        # **感知模块输出**（是否检出目标 / 方位 sin,cos / 距离）并进状态向量。
        # 这是「感知 + 决策分离」架构：把「切换信号」从图像里的稀疏事件
        # 变成显式输入，见 主创新A_诊断报告.md §4.12.4-4.12.5。
        self.with_det = bool(with_det)
        if self.with_det:
            if det_feat is None:
                raise ValueError("数据集没有 det_feat 字段，采集时需加 --det-features")
            self.det_feat = det_feat
            self.states = np.concatenate([self.states, self.det_feat], axis=1)
        else:
            self.det_feat = None

        # 帧堆叠的回溯索引：_fback[k][i] = 第 k 帧（k 步前）所在的行号。
        # **不额外存储历史帧** —— 数据集本就按时间有序存放，直接回溯即可。
        # 段首不足 k 步时 clamp 到段首帧（重复第一帧），避免跨回合串帧。
        seg_start = np.zeros(self.N, dtype=bool)
        for a, b in zip(ep_off[:-1], ep_off[1:]):
            if b > a:
                seg_start[int(a)] = True
        idx_self = np.arange(self.N)
        prev1 = np.where(seg_start, idx_self, idx_self - 1)
        self._fback = [idx_self]
        for _ in range(1, self.n_frames):
            self._fback.append(prev1[self._fback[-1]])

        # 可切 chunk 的起点：同一回合内且留足 chunk 长度
        ok = np.zeros(self.N, dtype=bool)
        for a, b in zip(ep_off[:-1], ep_off[1:]):
            if b - a >= self.chunk:
                ok[a : b - self.chunk + 1] = True
        self.valid_idx = np.nonzero(ok)[0]
        self.valid_idx_full = self.valid_idx.copy()   # 数据量子实验的基准池
        if self.valid_idx.size == 0:                 # 回合太短则退化为单步
            self.valid_idx = np.arange(max(1, self.N - self.chunk))
            self.valid_idx_full = self.valid_idx.copy()

        # 归一化统计（仅用训练集索引统计时请自行传入 subset）
        self.act_mean = self.actions.mean(0)
        self.act_std = self.actions.std(0) + 1e-6
        self.st_mean = self.states.mean(0)
        self.st_std = self.states.std(0) + 1e-6
        self.tokenizer = None

    def set_data_fraction(self, frac: float):
        """把可采样池限制为**每个回合内前 `frac` 比例**的起点（E2 数据量子）。

        ## 为什么是「按回合取比例」而不是「全局随机抽 frac」

        若全局随机抽，短回合贡献少、长回合贡献多，且**某些场景可能被整体抽空** ——
        那样「数据量减少」就同时改变了**场景/任务的分布**，
        于是「LoRA 在低数据量下更好」与「分布变了所以结果变了」**分不开**。
        按回合等比抽取可保证：场景集合、每场景的指令分布、回合数**完全不变**，
        只有每回合贡献的**帧数**减少 —— 这才是干净的数据量子。

        Returns:
            实际保留的起点数。
        """
        frac = float(frac)
        if frac >= 1.0:
            self.valid_idx = self.valid_idx_full.copy()
            return int(self.valid_idx.size)
        keep = []
        for a, b in zip(self.ep_off[:-1], self.ep_off[1:]):
            lo, hi = int(a), int(b - self.chunk + 1)
            if hi <= lo:
                continue
            m = max(1, int(round((hi - lo) * frac)))
            keep.append(np.arange(lo, lo + m))
        self.valid_idx = (np.concatenate(keep) if keep
                          else self.valid_idx_full.copy())
        return int(self.valid_idx.size)

    # ---------------- 语言 ----------------
    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(VLM_NAME)
            if tok.pad_token is None:                # 与官方 padding 行为对齐
                tok.pad_token = tok.eos_token
            self.tokenizer = tok
        return self.tokenizer

    def tokenize(self, texts):
        """官方口径：文本先补 "\\n"，右填充到 tokenizer_max_length。"""
        tok = self._ensure_tokenizer()
        fixed = [t if t.endswith("\n") else t + "\n" for t in texts]
        enc = tok(fixed, padding="max_length", max_length=self.tok_len,
                  truncation=True, return_tensors="pt")
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device).bool()

    # ---------------- 采样 ----------------
    def sample_index(self, batch_size, rng):
        sel = rng.choice(self.valid_idx, size=batch_size, replace=self.valid_idx.size < batch_size)
        return np.asarray(sel)

    def batch(self, idx, normalize=True, with_action_chunk=True):
        """构造一个训练 batch。

        图像保持 [0,1] 的 (B,3,H,W) —— 尺寸变换交给模型的 `prepare_images`，
        以保证与推理路径完全一致（它也负责 resize_with_pad 到 512 与 [-1,1] 归一化）。
        """
        idx = np.asarray(idx)
        # 按回溯索引取 n_frames 帧，各自作为一个「相机」键（camera1 = 当前帧）
        img_items = {}
        for k in range(self.n_frames):
            im = torch.from_numpy(self.images[self._fback[k][idx]]).to(self.device)
            im = im.float().div_(255.0).permute(0, 3, 1, 2).contiguous()
            img_items[IMG_KEYS[k]] = im

        st = torch.from_numpy(self.states[idx]).to(self.device)
        if normalize:
            st = (st - torch.as_tensor(self.st_mean, device=self.device)) / \
                 torch.as_tensor(self.st_std, device=self.device)
        if st.shape[1] < self.max_state_dim:
            st = torch.nn.functional.pad(st, (0, self.max_state_dim - st.shape[1]))

        lang_t, lang_m = self.tokenize([self.instr[int(i)] for i in idx])

        out = dict(img_items)
        out.update({STATE_KEY: st, "task": [self.instr[int(i)] for i in idx],
                    LANG_TOKENS_KEY: lang_t, LANG_MASK_KEY: lang_m})

        if with_action_chunk:
            B = len(idx)
            acts = np.stack([self.actions[i : i + self.chunk] for i in idx])   # (B,chunk,2)
            a = torch.from_numpy(acts).to(self.device)
            if normalize:
                a = (a - torch.as_tensor(self.act_mean, device=self.device)) / \
                    torch.as_tensor(self.act_std, device=self.device)
            if a.shape[-1] < self.max_action_dim:      # pad 到模型动作维度
                a = torch.nn.functional.pad(a, (0, self.max_action_dim - a.shape[-1]))
            out[ACTION_KEY] = a
        return out

    def denormalize_action(self, a):
        """把模型输出的归一化动作还原到物理量 (v, ω)。"""
        a = torch.as_tensor(a, dtype=torch.float32)
        mean = torch.as_tensor(self.act_mean, dtype=torch.float32, device=a.device)
        std = torch.as_tensor(self.act_std, dtype=torch.float32, device=a.device)
        return a * std + mean

    def summary(self):
        # 状态维度可能很大（含激光时 125），只打印前 5 维避免刷屏
        head = np.round(self.st_mean[:5], 4).tolist()
        tail = ""
        if self.with_lidar:
            tail += " (+激光 120 维)"
        if self.with_det:
            tail += " (+感知 4 维)"
        return (f"{self.N} 样本 / {len(self.ep_off) - 1} 回合 / "
                f"可切块起点 {self.valid_idx.size} / 指令词表 {len(self.instr_vocab)}\n"
                f"  动作 均值 {np.round(self.act_mean, 4).tolist()} "
                f"标准差 {np.round(self.act_std, 4).tolist()}\n"
                f"  状态 {self.states.shape[1]} 维 均值(前5) {head}{tail}")


def load(path, **kw):
    return SimDemoData(path, **kw)
