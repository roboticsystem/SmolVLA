#!/usr/bin/env bash
# E5 本机补全队列：GPU3 训练 / GPU4 评测，两路交错，互不抢卡。
#
# 顺序（2026-10-26 盘点结论，按价值排序）：
#   ① none 基线 4 种子 —— 旧 e5_none_s42 的 checkpoint 是 with_det=False
#      （训练时漏了 --det-features-mode none），与其余三臂口径不一致，
#      已归档到 export/_archive/e5_none_s42_wodet，这里重训。
#   ② λ 臂 / 标签比例臂的 200 回合评测（none_p02 / none_p05 已在 g2 里评）。
#   ③ 数据效率 frac 0.4 / 0.7 × 2 种子（补 A2「数据 −60%、损失 ≤3 pp」）。
#   ④ mem × IDAS 2×2 交互（只补 mem 两个格子，none 档复用①和既有 idas_v2 种子）。
#
# 幂等：已有 policy.pt 的训练跳过；已有 json 的评测跳过；完成标记在 export/queue/done_*。
# 日志：export/queue/queue.log（主）、lane_eval.log（评测链）、<tag>.train.log。
set -u

ROOT=/3240704007/sci/VLA
cd "$ROOT" || exit 1
PY=.venv_smolvla/bin/python
Q=export/queue
mkdir -p "$Q" export/e5_eval

say() { echo "[$(date '+%F %T')] $*" | tee -a "$Q/queue.log"; }
wait_proc() { while pgrep -f "$1" >/dev/null 2>&1; do sleep 30; done; }
wait_marker() { while [ ! -f "$Q/done_$1" ]; do sleep 30; done; }

# ---------------------------------------------------------------- 训练
# train <tag> <seed> <data> <det_mode> <variant> <frac>
train() {
  local tag=$1 seed=$2 data=$3 dmode=$4 variant=$5 frac=$6
  if [ -f "export/$tag/policy.pt" ]; then
    say "跳过训练 $tag（已存在 policy.pt）"; touch "$Q/done_$tag"; return
  fi
  say "训练 $tag | seed=$seed data=$(basename "$data") det=$dmode variant=$variant frac=$frac"
  CUDA_VISIBLE_DEVICES=3 "$PY" smolvla_finetune_sim.py \
      --data "$data" --det-features-mode "$dmode" --data-frac "$frac" \
      --steps 10000 --batch 8 --lr 1e-4 --chunk 50 --mode expert \
      --seed "$seed" --idas-variant "$variant" --idas-lam 0.1 \
      --eval-every 0 --eval-episodes 8 --out "export/$tag" \
      > "$Q/$tag.train.log" 2>&1
  if [ -f "export/$tag/policy.pt" ] && [ -f "export/$tag/meta.json" ]; then
    say "完成训练 $tag"; touch "$Q/done_$tag"
  else
    say "!! 训练失败 $tag（见 $Q/$tag.train.log）"
  fi
}

# ---------------------------------------------------------------- 评测
# evalset <outname> <tag...>
evalset() {
  local out=$1; shift
  local specs="" t
  for t in "$@"; do
    [ -f "export/$t/policy.pt" ] && specs="${specs:+$specs,}$t:export/$t"
  done
  if [ -z "$specs" ]; then say "跳过评测 $out（无可用模型）"; return; fi
  if [ -f "export/e5_eval/$out.json" ]; then say "跳过评测 $out（已存在）"; return; fi
  say "评测 $out <- $specs"
  CUDA_VISIBLE_DEVICES=4 "$PY" scripts/eval_ft_models.py --models "$specs" \
      --episodes 200 --batch 16 \
      --out "export/e5_eval/$out.json" --md "export/e5_eval/$out.md" \
      >> "$Q/$out.log" 2>&1
  [ -f "export/e5_eval/$out.json" ] && say "完成评测 $out" || say "!! 评测失败 $out"
}

# ---------------------------------------------------------------- 评测链（GPU4）
(
  wait_proc "export/e5_eval/g2.json"
  evalset g3 e5_idas_l005 e5_idas_l02 e5_idas_p02 e5_idas_p05

  for t in e5_none_s42 e5_none_s123 e5_none_s456 e5_none_s789; do wait_marker "$t"; done
  evalset g4 e5_none_s42 e5_none_s123 e5_none_s456 e5_none_s789

  for t in e5_frac04_v2_s42 e5_frac04_v2_s123 e5_frac07_v2_s42 e5_frac07_v2_s123; do wait_marker "$t"; done
  evalset g5 e5_frac04_v2_s42 e5_frac04_v2_s123 e5_frac07_v2_s42 e5_frac07_v2_s123

  for t in e5_mem_none_s42 e5_mem_none_s123 e5_mem_v2_s42 e5_mem_v2_s123; do wait_marker "$t"; done
  evalset g6 e5_mem_none_s42 e5_mem_none_s123 e5_mem_v2_s42 e5_mem_v2_s123

  say "评测链完成（g3/g4/g5/g6）"
) >> "$Q/lane_eval.log" 2>&1 &

# ---------------------------------------------------------------- 训练链（GPU3）
wait_proc "export/e5_eval/g1.json"

# ① none 基线 4 种子（统一带 --det-features-mode none）
for s in 42 123 456 789; do train "e5_none_s$s" "$s" export/nav_detf.h5 none none 1.0; done

# ③ 数据效率：主方法臂（idas_v2）减到 40% / 70%
for f in 0.4 0.7; do
  case "$f" in 0.4) tag=frac04 ;; 0.7) tag=frac07 ;; esac
  for s in 42 123; do
    train "e5_${tag}_v2_s$s" "$s" export/nav_detf.h5 none idas_v2 "$f"
  done
done

# ④ mem × IDAS：只补 mem 两个格子（none 档复用①，v2 档复用 e5_idas_v2_s*）
#    注意 mem 格子必须用带 8 维 det_feat 的 mem96k（95204 样本 / 1176 回合），
#    与①的 nav_detf.h5（96147 / 1080）样本量差 1%，同一采集配置；两格对照靠同一
#    数据集内部比较，跨档比较时在表注里写明这 1% 差异。
for s in 42 123; do
  train "e5_mem_none_s$s" "$s" export/nav_detf_mem96k.h5 mem none 1.0
  train "e5_mem_v2_s$s" "$s" export/nav_detf_mem96k.h5 mem idas_v2 1.0
done

say "训练链完成，等待评测链收尾"
wait
say "队列全部结束"
