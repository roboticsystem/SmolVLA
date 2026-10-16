#!/usr/bin/env python3
"""补跑 E5 的 `none × p=0.2` 控制格（带等待空闲 GPU）。

这一格在两处出问题过：
① 先因 `smolvla_sim_data.py` 的 `st_mean` 初始化 bug 失败（已修）；
② 重试时又因挑到一块被别的任务占满的卡而 OOM。
所以这里改成**轮询等待显存足够**再开跑，避免再撞同样的墙。
"""
import os
import subprocess
import time

P = ".venv_smolvla/bin/python"
NEED = 9000          # 需要的空闲显存 MiB
WAIT = 60
CANDIDATES = [int(x) for x in os.environ.get("E5RETRY_GPUS", "0,5,6,7").split(",")]


def free_mib(gpu):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits", "-i", str(gpu)],
                             capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def run(gpu, log):
    out = "export/e5_none_p02"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    cmds = [
        [P, "-u", "smolvla_finetune_sim.py", "--data", "export/nav_detf.h5",
         "--steps", "10000", "--batch", "8", "--chunk", "50", "--mode", "expert",
         "--det-features", "--idas-variant", "none", "--idas-label-p", "0.2",
         "--eval-every", "0", "--eval-episodes", "8", "--eval-max-steps", "200",
         "--eval-batch", "8", "--out", out],
        [P, "-u", "smolvla_finetune_sim.py", "--eval-only", out,
         "--data", "export/nav_detf.h5", "--det-features", "--eval-episodes", "200",
         "--eval-max-steps", "200", "--chunk", "50", "--eval-batch", "8"],
    ]
    with open(log, "a") as fh:
        fh.write(f"[选定 GPU{gpu}]\n")
        for c in cmds:
            fh.write("#" * 70 + "\n$ " + " ".join(c) + "\n")
            fh.flush()
            r = subprocess.run(c, env=env, stdout=fh, stderr=subprocess.STDOUT)
            fh.write(f"  [退出码 {r.returncode}]\n")
            fh.flush()
            if r.returncode != 0:
                return r.returncode
        fh.write("E5_NONE_P02_DONE\n")
        fh.flush()
    return 0


def main():
    log = "/tmp/e5_retry_p02.log"
    open(log, "w").close()
    while True:
        for g in CANDIDATES:
            f = free_mib(g)
            if f >= NEED:
                print(f"GPU{g} 空闲 {f} MiB，开始补跑", flush=True)
                rc = run(g, log)
                print(f"结束 rc={rc}", flush=True)
                return
        print("所有候选卡显存不足，等待...", flush=True)
        time.sleep(WAIT)


if __name__ == "__main__":
    main()
