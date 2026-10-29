#!/usr/bin/env bash
# E5b 补充实验队列：三卡三路 + 零成本项，覆盖用户 2026-10-27 指定的全部五件事。
#
#   GPU4  laneA  ID头探针(分钟级) → mem反常核查 g7 → 1000回合长时稳定性 g8
#   GPU2  laneB  模型×9布局泛化矩阵（50 回合/格，步数预算取 LAYOUT_STEPS）
#   GPU3  laneC  退化×遮挡鲁棒矩阵（10 场景 × 2 模型，50 回合/格）
#   CPU   零成本  IDAS 训练开销表（解析训练日志）
#
# 幂等：产物 json/md 已存在则跳过；日志在 export/queue/lane_{a,b,c}.log。
set -u

ROOT=/3240704007/sci/VLA
cd "$ROOT" || exit 1
PY=.venv_smolvla/bin/python
Q=export/queue
mkdir -p "$Q" export/e5_eval

say() { echo "[$(date '+%F %T')] $*" | tee -a "$Q/queue.log"; }
skip_if() { [ -f "$1" ] && { say "跳过 $2（已存在 $(basename "$1")）"; return 0; } || return 1; }

# ---------------------------------------------------------------- 零成本：开销表
skip_if export/e5_eval/overhead.md "开销表" || \
  "$PY" scripts/id_overhead_table.py --out export/e5_eval/overhead.md \
  >> "$Q/lane_a.log" 2>&1 && say "完成开销表"

# ---------------------------------------------------------------- laneA：GPU6
(
  CUDA_VISIBLE_DEVICES=6 "$PY" scripts/probe_id_head.py \
      --models idas_v2:export/e5_idas_v2_s42,none:export/e5_none_s42 \
      --data export/nav_detf.h5 --pairs 384 \
      --out export/e5_eval/probe.md >> "$Q/lane_a.log" 2>&1 \
      && say "完成 ID 头探针" || say "!! ID 头探针失败"

  skip_if export/e5_eval/g7.json "mem反常核查 g7" || {
    say "评测 g7（mem 反常核查：新 none/mem + 旧 96k 对 ft_detf / ft_detf_mem96k）"
    CUDA_VISIBLE_DEVICES=6 "$PY" scripts/eval_ft_models.py \
      --models none:export/e5_none_s42,mem:export/e5_mem_none_s42,old_sf:export/ft_detf,old_mem:export/ft_detf_mem96k \
      --episodes 200 --batch 16 \
      --out export/e5_eval/g7.json --md export/e5_eval/g7.md \
      >> "$Q/lane_a.log" 2>&1 && say "完成评测 g7" || say "!! 评测 g7 失败"
  }

  skip_if export/e5_eval/g8.json "长时稳定性 g8" || {
    say "评测 g8（1000 回合长时稳定性：idas_v2_s42 vs none_s42）"
    CUDA_VISIBLE_DEVICES=6 "$PY" scripts/eval_ft_models.py \
      --models idas_v2_s42:export/e5_idas_v2_s42,none_s42:export/e5_none_s42 \
      --episodes 1000 --batch 16 \
      --out export/e5_eval/g8.json --md export/e5_eval/g8.md \
      >> "$Q/lane_a.log" 2>&1 && say "完成评测 g8" || say "!! 评测 g8 失败"
  }
  say "laneA 结束"
) >> "$Q/lane_a.log" 2>&1 &

# ---------------------------------------------------------------- laneB：GPU1 布局矩阵
(
  skip_if export/e5_eval/matrix_layout.json "布局矩阵" || {
    say "评测 布局矩阵（2 模型 × 9 布局 × 50 回合）"
    CUDA_VISIBLE_DEVICES=1 "$PY" scripts/eval_matrix.py --mode layout \
      --models none:export/e5_none_s42,v2:export/e5_idas_v2_s42 \
      --episodes 50 --batch 8 \
      --out export/e5_eval/matrix_layout.json \
      >> "$Q/lane_b.log" 2>&1 && say "完成布局矩阵" || say "!! 布局矩阵失败"
  }
  say "laneB 结束"
) >> "$Q/lane_b.log" 2>&1 &

# ---------------------------------------------------------------- laneC：GPU3 退化矩阵
(
  skip_if export/e5_eval/matrix_degrade.json "退化矩阵" || {
    say "评测 退化矩阵（2 模型 × 10 场景 × 50 回合）"
    CUDA_VISIBLE_DEVICES=3 "$PY" scripts/eval_matrix.py --mode degrade \
      --models none:export/e5_none_s42,v2:export/e5_idas_v2_s42 \
      --episodes 50 --batch 8 \
      --out export/e5_eval/matrix_degrade.json \
      >> "$Q/lane_c.log" 2>&1 && say "完成退化矩阵" || say "!! 退化矩阵失败"
  }
  say "laneC 结束"
) >> "$Q/lane_c.log" 2>&1 &

wait
say "E5b 队列全部结束"
