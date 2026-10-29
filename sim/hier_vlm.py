"""VLM 版分层基线 —— 与 `HierRulePolicy` 共用执行层，仅替换「子目标层」。

分层 VLA 的结构假设：指令 → 子目标（去哪个物体）→ 测地执行。Track 2a 用
规则（`parse_instruction` + 颜色/类别匹配）实现子目标层；本模块把同一步换成
**多模态大模型 zero-shot 判定**（Qwen3-VL-4B，vLLM OpenAI 兼容接口）：

- **探索期绑定**：每隔 `ask_every` 帧（且画面里有检测）把相机帧 + 指令 +
  检测列表（编号/类别/颜色）发给 VLM，让它指出哪个检测是指令目标。
  VLM 直接看像素，负责颜色绑定与 L3 关系理解 —— 这正是 W1 定位的
  端到端 VLA 失效点（颜色 grounding 0/18）。
- **换段后**：目标实例已确定，末段修正回落到规则匹配（检测器对同一物体
  的颜色/类别输出稳定，无需再问 VLM，控制末段延迟）。

记忆条目照旧进 `SemanticMap`，但**只有 VLM 判定过的实例才可作为地标**
（`_verified` 列表），与规则版的「按（颜色, 类别）查询全量记忆」区分开 ——
两个臂的子目标层语义各自纯粹。
"""

from __future__ import annotations

import base64
import io
import json
import urllib.request

import numpy as np

from .baselines import HierRulePolicy


class HierVLMPolicy(HierRulePolicy):
    """子目标层 = VLM zero-shot 判定；执行层 = 父类记忆测地导航（不变）。"""

    name = "hier_vlm"

    def __init__(self, api="http://127.0.0.1:8002/v1", model="qwen3-vl-4b",
                 ask_every=10, timeout=25, max_tokens=8, scale=2,
                 verify_fresh=True, **kw):
        super().__init__(**kw)
        self.api = api.rstrip("/")
        self.model = model
        self.ask_every = int(max(1, ask_every))
        self.timeout = int(timeout)
        self.max_tokens = int(max_tokens)
        self.scale = int(scale)              # 160x120 原图偏小，默认放大 2 倍
        self.verify_fresh = bool(verify_fresh)
        self._verified = []                  # VLM 判定过的目标实例（世界坐标）
        self._asks = 0
        self._ask_fails = 0
        self._fail_streak = 0                # 连续失败计数（vLLM 首次视觉请求
        self._server_ok = None               # 冷启动可能 >25s，不因单次超时判死）

    def reset(self, env=None):
        super().reset(env)
        self._verified = []
        self._asks = 0
        self._ask_fails = 0
        self._fail_streak = 0

    # ------------------------------------------------------------ VLM 调用
    def _encode_jpeg(self, img):
        from PIL import Image

        im = Image.fromarray(np.asarray(img))
        if self.scale > 1:
            im = im.resize((im.width * self.scale, im.height * self.scale),
                           Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _chat(self, prompt, img_b64):
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/jpeg;base64," + img_b64}},
                {"type": "text", "text": prompt},
            ]}],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.api + "/chat/completions", data=body, method="POST",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read().decode("utf-8"))
        return out["choices"][0]["message"]["content"]

    def _vlm_pick(self, img, dets, instruction):
        """问 VLM：画面里哪个检测是指令目标。返回检测下标（None=都不是/失败）。"""
        if self._server_ok is False:
            return None
        lines = ["你是导航机器人的目标定位模块。", "指令：" + instruction,
                 "当前画面中检测到以下物体（编号 | 类别 | 颜色）："]
        for i, d in enumerate(dets, 1):
            lines.append("%d. %s | %s" % (i, d.cls, d.color))
        lines.append("指令所指的目标是画面中的哪个物体？只回答一个编号数字；"
                     "若画面中没有指令所指的目标，只回答 0。")
        self._asks += 1
        try:
            txt = self._chat("\n".join(lines), self._encode_jpeg(img))
        except Exception as e:                          # noqa: BLE001
            self._ask_fails += 1
            self._fail_streak += 1
            if self._fail_streak >= 3 and self._server_ok is None:
                self._server_ok = False                 # 开局即连续失败 → 快速失败
                raise RuntimeError("VLM 服务连续 3 次不可达（%s: %s）—— "
                                   "hier_vlm 臂无法运行" % (type(e).__name__, e))
            return None                                 # 局部瞬时失败 → 跳过本帧
        self._fail_streak = 0
        self._server_ok = True
        digits = "".join(ch for ch in str(txt) if ch.isdigit())
        if not digits:
            return None
        k = int(digits[:2])                             # 容错：把 "1." 读成 1
        if 1 <= k <= len(dets):
            return k - 1
        return None

    # ------------------------------------------------------------ 子目标层替换
    def _subgoal(self, obs, env):
        """探索期用 VLM 绑定目标；换段后回落父类规则末段修正。"""
        from .perception import detect

        if self.phase == "navigate":
            return super()._subgoal(obs, env)           # 末段规则修正（实例已定）

        self._frame += 1
        dets = []
        if self._frame % self.det_every == 1:
            dets = list(detect(obs["image"], lidar=obs.get("lidar")))
        self._map.update(env.robot_pose(), None, dets, self._frame)
        self._fresh = None
        hit = None
        if dets and self._frame % self.ask_every == 1:
            k = self._vlm_pick(obs["image"], dets, env.task["instruction"])
            if k is not None:
                d = dets[k]
                a = env.robot_pose()[2] + d.bearing
                hit = env.robot_pose()[:2] + d.dist * np.array(
                    [np.cos(a), np.sin(a)])
                self._fresh = np.asarray(hit, dtype=np.float64).copy()
                # 记入已验证实例（最近邻合并，抑制测距抖动）
                merged = False
                for v in self._verified:
                    if float(np.hypot(*(self._fresh - v))) < 0.7:
                        v[:] = 0.5 * (v + self._fresh)
                        merged = True
                        break
                if not merged:
                    self._verified.append(self._fresh.copy())
        if hit is None and self._verified:
            p = env.robot_pose()[:2]
            hit = min(self._verified, key=lambda v: float(np.hypot(*(v - p))))
        return hit
