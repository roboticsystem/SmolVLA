#!/bin/bash
# 等待任一候选卡空闲 >= 12GB，然后跑 E8 chunk 缓存评测
cd /3240704007/sci/VLA
while true; do
  for g in 0 5 6 7; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
    if [ "$f" -ge 12000 ]; then
      echo "GPU$g free ${f} MiB -> start E8"
      .venv_smolvla/bin/python -u scripts/train/smolvla_e8_chunk.py --gpu $g --episodes 100
      exit 0
    fi
  done
  sleep 120
done
