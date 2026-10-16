#!/usr/bin/env bash
# 构建 sim 测试床镜像并以其运行命令
# 用法：
#   ./build_and_run.sh                      # 进入交互 shell（代码目录已挂载）
#   ./build_and_run.sh python -m sim.run_eval --policy oracle --episodes 3
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VLA_ROOT="$(cd "$HERE/../.." && pwd)"

docker build -t vla-sim "$HERE"

# 注意：必须 --network host（沙箱/嵌套环境下 docker 无 bridge）；
# 正常机器上可去掉此参数使用默认 bridge 网络。
docker run --rm -it --network host \
  -v "$VLA_ROOT":/work \
  -w /work \
  vla-sim "$@"
