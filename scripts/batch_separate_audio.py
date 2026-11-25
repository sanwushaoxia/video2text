#!/usr/bin/env python3
"""Queue multi-stem jobs with bounded CPU workers and isolated atomic bundles."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def initialize(device, threads, keep_model):
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(threads)
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if keep_model:
        from video2text.bs_roformer_split import enable_cpu_model_reuse

        enable_cpu_model_reuse()


def job_record(source, root):
    suffix = sha256(str(source).encode()).hexdigest()[:10]
    name = f"{source.stem}-{suffix}"
    return {"input": str(source), "output": str(root / name),
            "performance": str(root / f"{name}.performance.json"), "log": str(root / f"{name}.log")}


def separate_job(job):
    source, root, cache_dir, options = job
    from cli.separate_audio import main

    record = job_record(source, root)
    argv = [str(source), "--multi-stems", "--split-mode", "diarize_bss", "--atomic-output",
            "--out-dir", record["output"], "--performance-report", record["performance"]]
    if cache_dir:
        argv.extend(("--cache-dir", str(cache_dir)))
    with Path(record["log"]).open("w", encoding="utf-8") as file, redirect_stdout(file), redirect_stderr(file):
        try:
            status = main(argv + options)
        except SystemExit as exc:
            status = exc.code or 0
        except Exception as exc:
            print(type(exc).__name__, str(exc), file=sys.stderr)
            status = 1
    return {**record, "exit_code": status}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--reuse-cpu-roformer", action="store_true")
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    forward = argv[split + 1:]
    reserved = {"--out-dir", "--performance-report", "--cache-dir",
                "--count-speakers", "--three-stems", "--atomic-output"}
    forwarded_options = [arg.split("=", 1)[0] for arg in forward if arg.startswith("--")]
    if any(option.startswith(arg) for option in reserved for arg in forwarded_options):
        parser.error("批处理负责输出、缓存、计时和发布选项")
    if args.workers < 1 or args.threads_per_worker < 1:
        parser.error("工作进程数和每进程线程数必须为正")
    if args.workers > 1 and args.device != "cpu":
        parser.error("并行批处理需要显式选择 --device cpu，自动设备使用单工作进程")
    if args.reuse_cpu_roformer and args.device != "cpu":
        parser.error("--reuse-cpu-roformer 需要 --device cpu")
    root = args.out_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    inputs = dict.fromkeys(path.expanduser().resolve() for path in args.inputs)
    jobs = [(source, root, args.cache_dir, forward) for source in inputs]

    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize, initargs=(args.device, args.threads_per_worker, args.reuse_cpu_roformer),
    ) as executor:
        results = []
        pending = [executor.submit(separate_job, job) for job in jobs]
        for job, future in zip(jobs, pending):
            try:
                results.append(future.result())
            except BrokenProcessPool as exc:
                results.append({**job_record(job[0], root), "exit_code": 1, "error": str(exc)})
    index = root / "batch_report.json"
    index.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(index)
    return 0 if all(row["exit_code"] == 0 for row in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
