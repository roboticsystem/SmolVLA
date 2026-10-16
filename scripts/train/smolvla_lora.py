"""E2：LoRA / PEFT 微调范式（大纲 §5 E2 的自变量「微调范式」②）。

## 为什么自己实现而不是用 peft 库

peft 的 `LoRALinear` 是 `nn.Linear` 的**替换**层（把原权重塞进 `base_layer`），
之后 `mod.weight` 访问不到张量。而本项目的上游库
（`smolvlm_with_expert.forward_attn_layer`）里直接写了

    hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)

—— 一旦 `.weight` 不是张量就立刻崩（量化那次已经踩过同类坑：见
`smolvla_quant_int8.py` 顶部对 `quantize_dynamic` 的说明）。

因此这里让 `LoRALinear` **继承 `nn.Linear`**：`self.weight` 仍是**冻结的**原权重
张量（保持 `.weight.dtype` 可用），另外挂 `lora_A` / `lora_B` 两个**可训**小矩阵。

## 三档范式的公平口径

大纲 §5 E2 的自变量是「微调范式」三档。本项目的动作专家**必须**训
（它是「动作空间重构」A1 的落点，属**正确性改造**而非可选微调），
所以三档的差别只在于「**除动作专家外还改什么**」：

| 范式 | 除动作专家外 |
|---|---|
| `expert`（③，已有） | 什么都不改 —— 成本基线 |
| **`lora`（②，本模块）** | VLM 注意力上加 LoRA 适配器 |
| `full`（①，已有） | 全部参数 |

这样三档才是**同一任务、同一动作专家、只差主干适配方式**的公平对照。

## 保存：先合并再落盘

训练完把 `(α/r)·B·A` 折进 `base.weight` 再保存。好处：
① `--eval-only` 与部署**不需要任何改动**，也不会出现「LoRA 权重被 `strict=False`
静默丢弃」这种哑火；② 得到的就是可直接部署的**满秩**普通模型。
代价是不能从 checkpoint 继续训练（本课题不需要）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Linear):
    """`nn.Linear` + LoRA 旁路。

    `self.weight` / `self.bias` 保持**冻结**（原权重），
    新增 `lora_A: (r, in)`、`lora_B: (out, r)` 为可训参数。
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__(base.in_features, base.out_features,
                         bias=base.bias is not None,
                         device=base.weight.device, dtype=base.weight.dtype)
        # 复制原权重（随后冻结）
        with torch.no_grad():
            self.weight.copy_(base.weight)
            if base.bias is not None:
                self.bias.copy_(base.bias)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        self.r = int(r)
        self.scaling = float(alpha) / float(r) if r > 0 else 0.0
        # **必须显式指定 device**：`torch.zeros` 默认建在 CPU，而模型可能已在
        # CUDA 上 —— 实测报错 `mat2 is on cpu, different from ... cuda`。
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features,
                                               dtype=base.weight.dtype,
                                               device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r,
                                               dtype=base.weight.dtype,
                                               device=base.weight.device))
        # A 用 Kaiming/均匀初始化、B 置 0 —— 保证训练起点**严格等价于原层**（W + B·A = W）
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.r > 0 and self.scaling != 0.0:
            # x @ A^T @ B^T * scaling
            lx = self.drop(x)
            out = out + self.scaling * F.linear(F.linear(lx, self.lora_A), self.lora_B)
        return out

    @torch.no_grad()
    def merge_(self):
        """把 LoRA 折进 `weight`，之后该层退化为普通 Linear。"""
        if self.r > 0 and self.scaling != 0.0:
            delta = (self.lora_B @ self.lora_A) * self.scaling
            self.weight.add_(delta.to(self.weight.dtype))
        return self

    def lora_parameters(self):
        return [self.lora_A, self.lora_B]


# ---------------------------------------------------------------- 注入
def _iter_named_linears(model):
    """遍历所有 `nn.Linear` 的 (父模块, 名字, 模块)。"""
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                yield parent, name, child


def target_filter(name, targets):
    """判断某个 Linear 是否该挂 LoRA。

    `targets`:
      - `vlm_attn`：VLM 注意力的四个投影（q/k/v/o）—— 默认，标准做法
      - `vlm_mlp` ：VLM 的 FFN（gate/up/down）
      - `expert`  ：动作专家的全部 Linear
      - `all`     ：以上全部
    """
    t = set(targets)
    low = name.lower()
    is_attn = any(k in low for k in ("q_proj", "k_proj", "v_proj", "o_proj"))
    is_mlp = any(k in low for k in ("gate_proj", "up_proj", "down_proj", "fc1", "fc2"))
    if "all" in t:
        return True
    if "vlm_attn" in t:
        return is_attn
    if "vlm_mlp" in t:
        return is_mlp
    if "expert" in t:
        return True
    return is_attn


def _resolve_root(model, targets):
    """按 targets 决定注入的作用域。

    `vlm_attn` / `vlm_mlp` 只在 **VLM 主干**内注入 ——
    不能按名字全局匹配：动作专家里的投影也叫 `q_proj/k_proj/v_proj/o_proj`，
    会被一起挂上（实测注入 164 层、其中相当一部分在专家里），
    而专家在 `expert` 范式里**本来就是全参可训**的，LoRA 挂在上面是重复且
    说不清的。限定到 VLM 主干后，「LoRA 范式 = expert + 主干低秩适配」
    这个口径才干净。
    """
    t = set(targets)
    if t & {"vlm_attn", "vlm_mlp"}:
        try:
            return model.vlm_with_expert.vlm
        except AttributeError:
            pass
    return model


def inject_lora(model, r=8, alpha=16.0, targets=("vlm_attn",),
                scope=None, dropout=0.0):
    """把作用域之内、名字匹配 `targets` 的 Linear 换成 `LoRALinear`。

    Returns:
        `(n_injected, n_lora_params)`
    """
    root = scope if scope is not None else _resolve_root(model, targets)
    n = 0
    seen = set()
    for parent, name, child in list(_iter_named_linears(root)):
        if id(child) in seen or isinstance(child, LoRALinear):
            continue
        if not target_filter(name, targets):
            continue
        seen.add(id(child))
        parent._modules[name] = LoRALinear(child, r, alpha, dropout)
        n += 1
    npar = sum(p.numel() for m in model.modules()
               if isinstance(m, LoRALinear) for p in m.lora_parameters())
    return n, npar


def mark_only_lora_trainable(model, extra_modules=()):
    """冻结一切，只放开 LoRA 参数与 `extra_modules`（动作专家等）。"""
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)
    for m in extra_modules:
        for p in m.parameters():
            p.requires_grad_(True)


def count_lora(model):
    ms = [m for m in model.modules() if isinstance(m, LoRALinear)]
    return len(ms), sum(p.numel() for m in ms for p in m.lora_parameters())


@torch.no_grad()
def merge_lora(model):
    """把所有 LoRA 折进权重（不可逆）。返回折入的层数。"""
    n = 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge_()
            n += 1
    return n
