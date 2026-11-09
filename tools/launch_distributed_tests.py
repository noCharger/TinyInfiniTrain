#!/usr/bin/env python3
"""Launch TinyInfiniTrain's multi-process NCCL integration tests.

Each mode gets a fresh filesystem rendezvous directory and run identifier, so
stale ncclUniqueId files from an earlier crash can never join a new run.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid


ALL_MODES = ("tp", "tp_gpt2", "zero1", "zero2", "zero2_bf16", "zero3", "pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("build/test_distributed_nccl"),
        help="Path to the USE_CUDA=ON, USE_NCCL=ON test executable",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=ALL_MODES,
        default=list(ALL_MODES),
        help="Integration cases to run",
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated physical GPU ids; defaults to CUDA_VISIBLE_DEVICES or 0..world-size-1",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout per mode in seconds")
    return parser.parse_args()


def launch_mode(binary: Path, mode: str, world_size: int, gpu_list: str, timeout: float) -> None:
    with tempfile.TemporaryDirectory(prefix=f"tinyinfinitrain-{mode}-") as rendezvous:
        run_id = uuid.uuid4().hex
        processes: list[subprocess.Popen[str]] = []
        log_paths: list[Path] = []
        log_files = []
        try:
            for rank in range(world_size):
                env = os.environ.copy()
                env.update(
                    {
                        "CUDA_VISIBLE_DEVICES": gpu_list,
                        "RANK": str(rank),
                        "WORLD_SIZE": str(world_size),
                        "LOCAL_RANK": str(rank),
                        "TINYINFINITRAIN_DISTRIBUTED_TEST": mode,
                        "TINYINFINITRAIN_RENDEZVOUS_DIR": rendezvous,
                        "TINYINFINITRAIN_RUN_ID": run_id,
                    }
                )
                log_path = Path(rendezvous) / f"rank-{rank}.log"
                log_file = log_path.open("w", encoding="utf-8")
                log_paths.append(log_path)
                log_files.append(log_file)
                processes.append(
                    subprocess.Popen(
                        [str(binary), "--gtest_color=no"],
                        env=env,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                )

            deadline = time.monotonic() + timeout
            while any(process.poll() is None for process in processes):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{mode} exceeded {timeout:.1f}s")
                time.sleep(0.05)

            failures = [rank for rank, process in enumerate(processes) if process.returncode != 0]
            if failures:
                raise RuntimeError(f"{mode} failed on rank(s) {failures}")
        except BaseException:
            for process in processes:
                if process.poll() is None:
                    process.kill()
            for process in processes:
                process.wait()
            raise
        finally:
            for log_file in log_files:
                log_file.close()
            for rank, log_path in enumerate(log_paths):
                print(f"\n===== {mode}: rank {rank} =====")
                print(log_path.read_text(encoding="utf-8"), end="")


def main() -> None:
    args = parse_args()
    binary = args.binary.expanduser().resolve()
    if not binary.is_file():
        raise SystemExit(f"test binary not found: {binary}")
    if args.world_size != 2:
        raise SystemExit("the current integration cases require --world-size=2")
    gpu_list = args.gpus or os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpu_list is None:
        gpu_list = ",".join(str(rank) for rank in range(args.world_size))
    if len([gpu for gpu in gpu_list.split(",") if gpu.strip()]) < args.world_size:
        raise SystemExit(f"need at least {args.world_size} visible GPUs, got {gpu_list!r}")

    for mode in args.modes:
        print(f"\n>>> running distributed mode: {mode}")
        launch_mode(binary, mode, args.world_size, gpu_list, args.timeout)
    print("\nAll requested distributed modes passed.")


if __name__ == "__main__":
    main()
