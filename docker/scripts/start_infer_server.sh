#!/usr/bin/env bash
# 真机实验·服务器端推理服务一键启动（D2 路线）
# 用法：bash docker/start_infer_server.sh
# 端口：8002（8001 被本服务器上的 vLLM 服务占用；对外地址 10.252.18.70:8002）
set -e
cd /3240704007/sci/VLA

# venv 断链自愈（IDE 更新可能再次清掉 workbuddy python）
if ! .venv_smolvla/bin/python -c "import torch" 2>/dev/null; then
  QWEN_PY=/root/miniconda3/envs/Qwen/bin/python3.12
  mkdir -p /root/.workbuddy/binaries/python/versions/3.12.8/bin
  for n in python3.12 python3 python; do
    ln -sf $QWEN_PY /root/.workbuddy/binaries/python/versions/3.12.8/bin/$n
    ln -sf $QWEN_PY .venv_smolvla/bin/$n
  done
  echo "[自愈] venv python 已重链到 $QWEN_PY"
fi

# 幂等：先杀旧实例
ps -eo pid,cmd | grep infer_server.py | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null
sleep 1

nohup .venv_smolvla/bin/python deliver_v1/server/infer_server.py \
  --project /3240704007/sci/VLA \
  --ckpt  /3240704007/sci/VLA/deliver_v1/models/ft_detf \
  --port 8002 > /tmp/infer_server.log 2>&1 &

# 等模型加载完（约 40s，首帧 CUDA 预热另需一次请求）
for i in $(seq 1 30); do
  grep -q "设备=cuda" /tmp/infer_server.log && break
  sleep 2
done
tail -3 /tmp/infer_server.log
echo
echo "[OK] 对外地址：10.252.18.70:8002  （小车 vla_bridge 配这个）"
echo "[注意] 首次请求含 CUDA 预热（约 5s），之后稳态约 190ms"
