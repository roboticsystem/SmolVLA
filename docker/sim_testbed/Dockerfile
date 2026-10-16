# 路线A：语义导航仿真测试床（sim/）容器化
# 依赖极轻：numpy + h5py（plot_results 还可选 matplotlib）
# 用法见 ../README.md；在非嵌套沙箱的机器上执行：
#   docker build -t vla-sim .
#   ../run_sim.sh python -m sim.run_eval --policy oracle --episodes 3

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install --no-cache-dir numpy h5py matplotlib

WORKDIR /work
# 代码不打入镜像，运行时挂载，保证改代码不用重新 build
VOLUME ["/work"]

CMD ["bash"]
