#!/usr/bin/env python3
"""E5 扫描启动器：把剩余配置按 GPU 排队串行执行。

变体对照（λ=0.1, p=1.0）已完成（见表 1h，阴性结果）。
这里补大纲 §5 E5 要求的另外两维：
  ① **标签比例 p**（下界实验）：主损失与辅助损失同时按 p 掩蔽。
     大纲预期「低标签比例下 IDAS 的增益更大」—— 这是该假设 remaining 的关键检验。
  ② **λ_id 敏感性**：0.05 / 0.2（0.1 已在变体对照里）。

每格 = 训练 10000 步 + 闭环评测 200 回合。
对照必须**同 p**（否则掩蔽本身的损失与 IDAS 的增益混在一起）。
"""
import json
import os
import subprocess
import sys
import time

P = ".venv_smolvla/bin/python"
GPUS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 5, 6]

# (name, extra_args)
CONFIGS = [
    ("e5_none_p05",   ["--idas-variant", "none",       "--idas-label-p", "0.5"]),
    ("e5_idas_p05",   ["--idas-variant", "idas_v2",    "--idas-label-p", "0.5",
                       "--idas-lam", "0.1"]),
    ("e5_idas_p02",   ["--idas-variant", "idas_v2",    "--idas-label-p", "0.2",
                       "--idas-lam", "0.1"]),
    ("e5_none_p02",   ["--idas-variant", "none",       "--idas-label-p", "0.2"]),
    ("e5_idas_l005",  ["--idas-variant", "idas_v2",    "--idas-lam", "0.05"]),
    ("e5_idas_l02",   ["--idas-variant", "idas_v2",    "--idas-lam", "0.2"]),
]


def run_one(name, extra, gpu, log):
    out = f"export/{name}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               HF_HUB_OFFLINE="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    cmds = [
        [P, "-u", "smolvla_finetune_sim.py", "--data", "export/nav_detf.h5",
         "--steps", "10000", "--batch", "8", "--chunk", "50", "--mode", "expert",
         "--det-features", "--eval-every", "0", "--eval-episodes", "8",
         "--eval-max-steps", "200", "--eval-batch", "8", "--out", out] + extra,
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
        fh.write(f"E5SWEEP_{name}_DONE\n")
        fh.flush()
    return 0


def main():
    # 把配置轮流分给各 GPU，每个 GPU 串行处理自己的队列
    procs = []
    for i, gpu in enumerate(GPUS):
        mine = [(n, e) for j, (n, e) in enumerate(CONFIGS) if j % len(GPUS) == i]
        if not mine:
            continue
        log = f"/tmp/e5sweep_g{gpu}.log"
        open(log, "w").close()
        pid = os.fork()
        if pid == 0:
            code = 0
            for n, e in mine:
                t0 = time.time()
                rc = run_one(n, e, gpu, log)
                with open(log, "a") as fh:
                    fh.write(f"  [{n} 结束 rc={rc} 用时 {(time.time()-t0)/60:.1f} min]\n")
                if rc != 0:
                    code = rc
                    break
            os._exit(code)
        procs.append((gpu, pid, mine))
        print(f"  GPU{gpu}: {[n for n, _ in mine]}", flush=True)
    fails = 0
    for gpu, pid, mine in procs:
        _, st = os.waitpid(pid, 0)
        if st != 0:
            fails += 1
            print(f"  GPU{gpu} 队列有失败: {[n for n, _ in mine]}", flush=True)
    print(f"\n全部队列结束（失败 {fails} 个 GPU 队列）", flush=True)


if __name__ == "__main__":
    main()
