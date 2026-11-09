#!/usr/bin/env python3
"""Measure pipeline bubble and CUDA allocator high-water memory across PP/M.

The output CSV contains observations from real GPipe/1F1B executions. It is
deliberately separate from parallel_analysis.cc, which remains an analytical
capacity-planning model.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/pipeline_benchmark"))
    parser.add_argument("--pipeline-depths", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--microbatches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--schedules", nargs="+", choices=("gpipe", "1f1b"), default=["gpipe", "1f1b"])
    parser.add_argument(
        "--activation-recomputation",
        nargs="+",
        choices=("off", "on"),
        default=["off", "on"],
        help="Benchmark full-graph retention and block activation recomputation",
    )
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--layers-per-stage", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--gpus", default=None, help="Comma-separated physical GPU ids")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path, default=Path("pipeline_benchmark.csv"))
    return parser.parse_args()


def parse_metric(line: str) -> dict[str, str]:
    fields = line.strip().split()
    if not fields or fields[0] != "METRIC":
        raise ValueError(f"not a metric line: {line!r}")
    return dict(field.split("=", 1) for field in fields[1:])


def run_case(
    binary: Path,
    schedule: str,
    pp: int,
    microbatches: int,
    activation_recomputation: str,
    model_arguments: list[str],
    gpu_ids: list[str],
    timeout: float,
) -> dict[str, str]:
    with tempfile.TemporaryDirectory(
        prefix=f"tinyinfinitrain-bench-{pp}-{microbatches}-{activation_recomputation}-"
    ) as rendezvous:
        run_id = uuid.uuid4().hex
        processes: list[subprocess.Popen[str]] = []
        logs: list[Path] = []
        handles = []
        visible = ",".join(gpu_ids[:pp])
        try:
            for rank in range(pp):
                env = os.environ.copy()
                env.update(
                    {
                        "CUDA_VISIBLE_DEVICES": visible,
                        "RANK": str(rank),
                        "WORLD_SIZE": str(pp),
                        "LOCAL_RANK": str(rank),
                        "TINYINFINITRAIN_RENDEZVOUS_DIR": rendezvous,
                        "TINYINFINITRAIN_RUN_ID": run_id,
                    }
                )
                log = Path(rendezvous) / f"rank-{rank}.log"
                handle = log.open("w", encoding="utf-8")
                logs.append(log)
                handles.append(handle)
                processes.append(
                    subprocess.Popen(
                        [
                            str(binary),
                            "--schedule",
                            schedule,
                            "--microbatches",
                            str(microbatches),
                            "--activation-recomputation",
                            activation_recomputation,
                            *model_arguments,
                        ],
                        env=env,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                )

            deadline = time.monotonic() + timeout
            while any(process.poll() is None for process in processes):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"{schedule}, PP={pp}, M={microbatches}, "
                        f"recompute={activation_recomputation} exceeded {timeout:.1f}s"
                    )
                time.sleep(0.05)
            failed = [rank for rank, process in enumerate(processes) if process.returncode != 0]
            if failed:
                raise RuntimeError(f"benchmark failed on ranks {failed}")
        except BaseException:
            for process in processes:
                if process.poll() is None:
                    process.kill()
            for process in processes:
                process.wait()
            for handle in handles:
                handle.flush()
            for rank, log in enumerate(logs):
                print(f"\n===== failed case rank {rank} =====\n{log.read_text(encoding='utf-8')}", end="")
            raise
        finally:
            for handle in handles:
                handle.close()

        rank_zero = logs[0].read_text(encoding="utf-8")
        metric_lines = [line for line in rank_zero.splitlines() if line.startswith("METRIC ")]
        if len(metric_lines) != 1:
            for rank, log in enumerate(logs):
                print(f"\n===== failed case rank {rank} =====\n{log.read_text(encoding='utf-8')}", end="")
            raise RuntimeError(f"expected one METRIC line, found {len(metric_lines)}")
        return parse_metric(metric_lines[0])


def main() -> None:
    args = parse_args()
    binary = args.binary.expanduser().resolve()
    if not binary.is_file():
        raise SystemExit(f"benchmark binary not found: {binary}")
    configured_gpus = args.gpus or os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured_gpus is None:
        configured_gpus = ",".join(str(index) for index in range(max(args.pipeline_depths)))
    gpu_ids = [gpu.strip() for gpu in configured_gpus.split(",") if gpu.strip()]
    if len(gpu_ids) < max(args.pipeline_depths):
        raise SystemExit(f"need {max(args.pipeline_depths)} GPUs, got {configured_gpus!r}")

    rows = []
    model_arguments = [
        "--micro-batch-size",
        str(args.micro_batch_size),
        "--sequence-length",
        str(args.sequence_length),
        "--layers-per-stage",
        str(args.layers_per_stage),
        "--n-head",
        str(args.n_head),
        "--n-embd",
        str(args.n_embd),
        "--vocab-size",
        str(args.vocab_size),
    ]
    for schedule in args.schedules:
        for pp in args.pipeline_depths:
            for microbatches in args.microbatches:
                for activation_recomputation in args.activation_recomputation:
                    print(
                        f">>> schedule={schedule} PP={pp} M={microbatches} "
                        f"recompute={activation_recomputation}",
                        flush=True,
                    )
                    row = run_case(
                        binary,
                        schedule,
                        pp,
                        microbatches,
                        activation_recomputation,
                        model_arguments,
                        gpu_ids,
                        args.timeout,
                    )
                    rows.append(row)
                    print(
                        f"    bubble={float(row['measured_bubble']):.3f}, "
                        f"activation_delta={int(row['activation_delta_bytes']) / 2**20:.2f} MiB"
                    )

    fieldnames = [
        "schedule",
        "pp",
        "microbatches",
        "activation_recomputation",
        "micro_batch_size",
        "sequence_length",
        "layers_per_stage",
        "n_head",
        "n_embd",
        "vocab_size",
        "elapsed_ms",
        "compute_sum_ms",
        "measured_bubble",
        "baseline_bytes",
        "peak_bytes",
        "activation_delta_bytes",
        "peak_inflight",
    ]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} measured configurations to {output}")


if __name__ == "__main__":
    main()
