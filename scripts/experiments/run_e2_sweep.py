#!/usr/bin/env python3
"""E2 数据量子扫描启动器：范式 × 数据量（大纲 §5 E2 的自变量「演示数据量」）。

大纲 §5 E2 的预期是「LoRA 在数据量 ≤ 60% 时性价比最优」——
这一维正是该预期 remaining 的检验。

配置：
    (expert, lora) × frac(0.6, 0.4)   各 4 格
    full × 0.4                         1 格（上界在低数据量下的表现，可选）
100% 那一行由 E2 范式对照（ft_detf / e2_lora / e2_full）提供。

**每个 GPU 串行跑自己的队列；开跑前会等到该卡显存足够** ——
这样可以现在就挂上，等 E5 扫描与 E2 范式对照腾出卡后自动接上，
不必人工盯着。
"""
import os
import subprocess
import sys
import time

P = ".venv_smolvla/bin/python"
NEED_MIB = int(os.environ.get("E2SWEEP_NEED_MIB", "12000"))   # 需要的空闲显存
WAIT_S = int(os.environ.get("E2SWEEP_WAIT_S", "60"))

CONFIGS = [
    # (name, extra_args)
    ("e2_expert_f06", ["--mode", "expert", "--data-frac", "0.6"]),
    ("e2_lora_f06",   ["--mode", "lora",   "--data-frac", "0.6",
                       "--lora-r", "8", "--lora-alpha", "16"]),
    ("e2_expert_f04", ["--mode", "expert", "--data-frac", "0.4"]),
    ("e2_lora_f04",   ["--mode", "lora",   "--data-frac", "0.4",
                       "--lora-r", "8", "--lora-alpha", "16"]),
    ("e2_full_f04",   ["--mode", "full",   "--data-frac", "0.4"]),
]


def free_mib(gpu):
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
             "-i", str(gpu)], capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def wait_gpu(gpu, log):
    while True:
        f = free_mib(gpu)
        if f >= NEED_MIB:
            with open(log, "a") as fh:
                fh.write(f"[GPU{gpu}] 空闲 {f} MiB ≥ {NEED_MIB}，开始\n")
            return
        with open(log, "a") as fh:
            fh.write(f"[GPU{gpu}] 空闲仅 {f} MiB，等待 {WAIT_S}s...\n")
            fh.flush()
        time.sleep(WAIT_S)


def run_one(name, extra, gpu, log):
    wait_gpu(gpu, log)
    out = f"export/{name}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    # full 模式显存吃紧，batch 降到 4
    batch = "4" if "full" in extra else "8"
    cmds = [
        [P, "-u", "smolvla_finetune_sim.py", "--data", "export/nav_detf.h5",
         "--steps", "10000", "--batch", batch, "--chunk", "50", "--det-features",
         "--eval-every", "0", "--eval-episodes", "8", "--eval-max-steps", "200",
         "--eval-batch", "8", "--out", out] + extra,
        [P, "-u", "smolvla_finetune_sim.py", "--eval-only", out,
         "--data", "export/nav_detf.h5", "--det-features", "--eval-episodes", "200",
         "--eval-max-steps", "200", "--chunk", "50", "--eval-batch", "8"],
    ]
    with open(log, "a") as fh:
        for c in cmds:
            fh.write("#" * 70 + "\n$ " + " ".join(c) + "\n")
            fh.flush()
            r = subprocess.run(c, env=env, stdout=fh, stderr=subprocess.STDOUT)
            fh.write(f"  [退出码 {r.returncode}]\n")
            fh.flush()
            if r.returncode != 0:
                return r.returncode
        fh.write(f"E2SWEEP_{name}_DONE\n")
        fh.flush()
    return 0


def main():
    gpus = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 5, 6, 7]
    procs = []
    for i, gpu in enumerate(gpus):
        mine = [(n, e) for j, (n, e) in enumerate(CONFIGS) if j % len(gpus) == i]
        if not mine:
            continue
        log = f"/tmp/e2sweep_g{gpu}.log"
        open(log, "w").close()
        pid = os.fork()
        if pid == 0:
            for n, e in mine:
                t0 = time.time()
                rc = run_one(n, e, gpu, log)
                with open(log, "a") as fh:
                    fh.write(f"  [{n} 结束 rc={rc} 用时 {(time.time()-t0)/60:.1f} min]\n")
                if rc != 0:
                    break
            os._exit(0)
        procs.append((gpu, pid, [n for n, _ in mine]))
        print(f"  GPU{gpu}: {[n for n, _ in mine]}", flush=True)
    for gpu, pid, names in procs:
        os.waitpid(pid, 0)
        print(f"  GPU{gpu} 队列完成: {names}", flush=True)


if __name__ == "__main__":
    main()
