#!/usr/bin/env python3
"""消融实验的种子扩展（大纲 §3.2「随机种子 ≥ 5 组」）。

主对比（expert）已有 5 组训练种子（`seed{42,123,456,789,1024}_ft_detf`），
但 **E2 范式对照与 E5 变体对照目前都只有 seed 0**。

范围（刻意收窄到「进入结论的对比」）：

| 对比 | 已有 | 需补 |
|---|---|---|
| E2 范式：`lora` | seed 0 | 42/123/456/789（4） |
| E2 范式：`full` | seed 0 | 42/123/456/789（4） |
| E5 变体：`idas_v1/v2/random_ctrl` | seed 0 | 各 4（共 12） |

共 20 格，每格 ≈ 1 GPU·h，4 卡并行约 5 h。
（E4 是策略级消融不训练，它的「种子」是回合集种子，单独处理。）
"""
import os
import subprocess
import sys
import time

P = ".venv_smolvla/bin/python"
SEEDS = [42, 123, 456, 789]
NEED_MIB = int(os.environ.get("SEED_NEED_MIB", "12000"))
WAIT_S = int(os.environ.get("SEED_WAIT_S", "60"))


def build_cmd(name):
    """按配置名**显式**构造训练命令（避免隐式拼装出错）。"""
    seed = name.rsplit("_s", 1)[1]
    out = f"export/{name}"
    base = ["--steps", "10000", "--chunk", "50", "--seed", seed,
            "--eval-every", "0", "--eval-episodes", "8",
            "--eval-max-steps", "200", "--eval-batch", "8", "--out", out]
    if name.startswith("e2_lora"):
        return ([P, "-u", "smolvla_finetune_sim.py", "--data", "export/nav_detf.h5",
                 "--batch", "8", "--mode", "lora", "--lora-r", "8",
                 "--lora-alpha", "16", "--det-features"] + base)
    if name.startswith("e2_full"):
        return ([P, "-u", "smolvla_finetune_sim.py", "--data", "export/nav_detf.h5",
                 "--batch", "4", "--mode", "full", "--det-features"] + base)
    if name.startswith("e5_idas_v1"):
        v = "idas_v1"
    elif name.startswith("e5_idas_v2"):
        v = "idas_v2"
    elif name.startswith("e5_random"):
        v = "random_ctrl"
    else:
        raise ValueError(name)
    return ([P, "-u", "smolvla_finetune_sim.py", "--data", "export/nav_detf.h5",
             "--batch", "8", "--mode", "expert", "--det-features",
             "--idas-variant", v, "--idas-lam", "0.1"] + base)


def free_mib(gpu):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits", "-i", str(gpu)],
                             capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def wait_gpu(gpu, log):
    while True:
        f = free_mib(gpu)
        if f >= NEED_MIB:
            with open(log, "a") as fh:
                fh.write(f"[GPU{gpu}] 空闲 {f} MiB，开始\n")
            return
        with open(log, "a") as fh:
            fh.write(f"[GPU{gpu}] 空闲 {f} MiB，等待...\n")
            fh.flush()
        time.sleep(WAIT_S)


def run_one(name, gpu, log):
    wait_gpu(gpu, log)
    cmd = build_cmd(name)
    out = f"export/{name}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    ev = [P, "-u", "smolvla_finetune_sim.py", "--eval-only", out,
          "--data", "export/nav_detf.h5", "--det-features",
          "--eval-episodes", "200", "--eval-max-steps", "200",
          "--chunk", "50", "--eval-batch", "8"]
    with open(log, "a") as fh:
        for c in (cmd, ev):
            fh.write("#" * 70 + "\n$ " + " ".join(c) + "\n")
            fh.flush()
            r = subprocess.run(c, env=env, stdout=fh, stderr=subprocess.STDOUT)
            fh.write(f"  [退出码 {r.returncode}]\n")
            fh.flush()
            if r.returncode != 0:
                return r.returncode
        fh.write(f"SEED_{name}_DONE\n")
        fh.flush()
    return 0


def main():
    gpus = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 5, 6, 7]
    names = []
    for s in SEEDS:
        names += [f"e2_lora_s{s}", f"e2_full_s{s}"]
    for s in SEEDS:
        names += [f"e5_idas_v1_s{s}", f"e5_idas_v2_s{s}", f"e5_random_ctrl_s{s}"]
    procs = []
    for i, gpu in enumerate(gpus):
        mine = [n for j, n in enumerate(names) if j % len(gpus) == i]
        if not mine:
            continue
        log = f"/tmp/seedexp_g{gpu}.log"
        open(log, "w").close()
        pid = os.fork()
        if pid == 0:
            for n in mine:
                t0 = time.time()
                rc = run_one(n, gpu, log)
                with open(log, "a") as fh:
                    fh.write(f"  [{n} 结束 rc={rc} 用时 {(time.time()-t0)/60:.1f} min]\n")
                if rc != 0:
                    break
            os._exit(0)
        procs.append((gpu, pid, mine))
        print(f"  GPU{gpu}: {len(mine)} 格", flush=True)
    for gpu, pid, mine in procs:
        os.waitpid(pid, 0)
        print(f"  GPU{gpu} 完成: {mine}", flush=True)


if __name__ == "__main__":
    main()
