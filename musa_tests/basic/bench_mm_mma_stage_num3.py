"""
Benchmark all mm_mma_stage_num3 kernel configurations using the built-in profiler.
"""

import argparse
import itertools
from typing import Iterable, Tuple, List

import tilelang

from musa_tests.basic.mm_mma_stage_num3 import DEVICE, TARGET, matmul


MNK_CASES = [
    (1024, 1024, 1024),
    # (4096, 4096, 4096),
    (8192, 8192, 8192),
    (16384, 16384, 16384),
]

BLOCK_CASES = [
    (32, 32, 32),
    (128, 128, 64),
    # (128, 128, 32),
]

TYPE_CASES = [
    ("float16", "float32"),
    ("bfloat16", "float32"),
    ("float8_e4m3", "float32"),
]

SPECIAL_CASES = []


def iter_cases() -> Iterable[Tuple[int, int, int, int, int, int, str, str]]:
    for (M, N, K), (bm, bn, bk), (dtype, acc_type) in itertools.product(
        MNK_CASES, BLOCK_CASES, TYPE_CASES
    ):
        yield M, N, K, bm, bn, bk, dtype, acc_type
    for case in SPECIAL_CASES:
        yield case


def bench_one_case(
    M: int,
    N: int,
    K: int,
    bm: int,
    bn: int,
    bk: int,
    dtype: str,
    acc_type: str,
    *,
    n_warmup: int,
    n_repeat: int,
    backend: str,
    quantiles: List[float] | None,
    return_mode: str,
) -> float | List[float]:
    program = matmul(M, N, K, bm, bn, bk, dtype=dtype, accum_dtype=acc_type)
    kernel = tilelang.compile(
        program,
        out_idx=-1,
        target=TARGET,
        execution_backend="cython",
    )
    profiler = kernel.get_profiler()
    latency_ms = profiler.do_bench(
        n_warmup=n_warmup,
        n_repeat=n_repeat,
        backend=backend,
        quantiles=quantiles,
        return_mode=return_mode,
    )
    return latency_ms


def main():
    tilelang.env.TILELANG_PRINT_ON_COMPILATION = "0"
    parser = argparse.ArgumentParser(description="Benchmark mm_mma_stage_num3 kernels.")
    parser.add_argument("-f", "--format", choices=["csv", "human"], default="human")
    parser.add_argument("--backend", choices=["event", "cupti"], default="event")
    parser.add_argument(
        "--return-mode", choices=["min", "max", "mean", "median"], default="mean"
    )
    parser.add_argument("--quantiles", type=float, nargs="*", default=None)
    parser.add_argument(
        "--n-warmup", type=int, default=3, help="Warmup iterations override."
    )
    parser.add_argument(
        "--n-repeat", type=int, default=20, help="Benchmark iterations override."
    )
    args = parser.parse_args()

    cases = list(iter_cases())

    print(f"Running {len(cases)} cases on {DEVICE} (backend={args.backend})")
    if args.format == "csv" and not args.quantiles:
        print(f"ID,Latency(ms),Throughput(TFLOPs)")
    perf_items = []
    for idx, (M, N, K, bm, bn, bk, dtype, acc_type) in enumerate(cases, start=1):
        latency_ms = bench_one_case(
            M,
            N,
            K,
            bm,
            bn,
            bk,
            dtype,
            acc_type,
            n_warmup=args.n_warmup,
            n_repeat=args.n_repeat,
            backend=args.backend,
            quantiles=args.quantiles,
            return_mode=args.return_mode,
        )
        case_id = f"M{M}-N{N}-K{K}-bm{bm}-bn{bn}-bk{bk}-{dtype}-{acc_type}"
        if not args.quantiles:
            latency = latency_ms / 1e3
            tflops = (2.0 * M * N * K) / latency / 1e12
            perf_items.append((idx, case_id, latency_ms, tflops))

    if args.quantiles:
        return

    for perf_item in perf_items:
        (idx, case_id, latency_ms, tflops) = perf_item
        if args.format == "csv":
            print(f"{case_id},{latency_ms:.6f}ms,{tflops:.3f}")
        else:
            print(
                f"[{idx}/{len(cases)}] "
                f"{case_id}: Latency={latency_ms:.6f}ms  TFlops={tflops:.3f}"
            )


if __name__ == "__main__":
    main()
