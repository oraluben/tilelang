#!/usr/bin/env python3
"""Comprehensive Metal GEMM benchmark for performance optimization.

Tests multiple configurations to find the best TFLOPS for fp16 GEMM on Metal.
Compares: thread counts, block_K, num_stages, with/without pipeline.
"""

import tilelang
import tilelang.language as T
import torch
import time
import sys

tilelang.env.disable_cache()

# --------------------------------------------------------------------------- #
# Kernel builders
# --------------------------------------------------------------------------- #


@tilelang.jit
def matmul_smem(M, N, K, block_M, block_N, block_K, num_stages=0, threads=None, dtype=T.float16, accum_dtype=T.float32):
    """Shared memory path: global → threadgroup → register."""
    if threads is None:
        threads = max(32, (block_M // 16) * (block_N // 32) * 32)

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return kernel


@tilelang.jit
def matmul_global(M, N, K, block_M, block_N, block_K, threads=None, dtype=T.float16, accum_dtype=T.float32):
    """Global path: global → register directly (no shared memory)."""
    if threads is None:
        threads = max(32, (block_M // 16) * (block_N // 32) * 32)

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_local = T.alloc_fragment((block_M, block_K), dtype)
            B_local = T.alloc_fragment((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.serial(T.ceildiv(K, block_K)):
                T.copy(A[by * block_M, ko * block_K], A_local)
                T.copy(B[ko * block_K, bx * block_N], B_local)
                T.gemm(A_local, B_local, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return kernel


def bench_kernel(jit_kernel, M, N, K, warmup=5, iters=20):
    """Benchmark a compiled kernel. Returns TFLOPS."""
    torch_dtype = torch.float16
    torch_accum = torch.float32
    a = torch.randn(M, K, dtype=torch_dtype, device="mps")
    b = torch.randn(K, N, dtype=torch_dtype, device="mps")
    c = torch.zeros(M, N, dtype=torch_accum, device="mps")

    # Warmup
    for _ in range(warmup):
        jit_kernel(a, b, c)
    torch.mps.synchronize()

    # Timed
    start = time.perf_counter()
    for _ in range(iters):
        jit_kernel(a, b, c)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    flops = 2 * M * N * K * iters
    tflops = flops / elapsed / 1e12
    return tflops


def bench_torch(M, N, K, warmup=5, iters=20):
    """Benchmark torch.matmul on MPS."""
    a = torch.randn(M, K, dtype=torch.float16, device="mps")
    b = torch.randn(K, N, dtype=torch.float16, device="mps")

    for _ in range(warmup):
        _ = a @ b
    torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = a @ b
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    flops = 2 * M * N * K * iters
    tflops = flops / elapsed / 1e12
    return tflops


def bench_mlx(M, N, K, warmup=5, iters=20):
    """Benchmark MLX matmul if available."""
    try:
        import mlx.core as mx

        a = mx.random.normal((M, K)).astype(mx.float16)
        b = mx.random.normal((K, N)).astype(mx.float16)
        mx.eval(a, b)

        for _ in range(warmup):
            c = a @ b
            mx.eval(c)

        start = time.perf_counter()
        for _ in range(iters):
            c = a @ b
            mx.eval(c)
        elapsed = time.perf_counter() - start

        flops = 2 * M * N * K * iters
        return flops / elapsed / 1e12
    except ImportError:
        return None


# --------------------------------------------------------------------------- #
# Main benchmark
# --------------------------------------------------------------------------- #


def main():
    sizes = [4096, 8192]
    if len(sys.argv) > 1:
        sizes = [int(x) for x in sys.argv[1:]]

    # Configurations to test: (block_M, block_N, block_K, num_stages, path, label)
    configs = [
        (64, 64, 16, 0, "smem", "smem 64x64x16 s=0"),
        (64, 64, 16, 2, "smem", "smem 64x64x16 s=2"),
        (64, 64, 32, 0, "smem", "smem 64x64x32 s=0"),
        (64, 64, 32, 2, "smem", "smem 64x64x32 s=2"),
        (64, 128, 16, 0, "smem", "smem 64x128x16 s=0"),
        (64, 128, 16, 2, "smem", "smem 64x128x16 s=2"),
        (64, 128, 32, 0, "smem", "smem 64x128x32 s=0"),
        (64, 128, 32, 2, "smem", "smem 64x128x32 s=2"),
        (128, 128, 32, 0, "smem", "smem 128x128x32 s=0"),
        (128, 128, 32, 2, "smem", "smem 128x128x32 s=2"),
    ]

    thread_options = [128, 256]

    for size in sizes:
        M = N = K = size
        print(f"\n{'=' * 72}")
        print(f"  GEMM {M}x{N}x{K} fp16")
        print(f"{'=' * 72}")

        # Baselines
        torch_tflops = bench_torch(M, N, K)
        print(f"  torch.matmul:  {torch_tflops:.2f} TFLOPS")
        mlx_tflops = bench_mlx(M, N, K)
        if mlx_tflops is not None:
            print(f"  MLX matmul:    {mlx_tflops:.2f} TFLOPS")
        print()

        best_tflops = 0
        best_label = ""

        print(f"  {'Config':<40} {'Threads':>8} {'TFLOPS':>8} {'vs MLX':>8} {'vs Torch':>10}")
        print(f"  {'-' * 40} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10}")

        for bM, bN, bK, ns, path, label in configs:
            for threads in thread_options:
                full_label = f"{label} t={threads}"
                try:
                    if path == "smem":
                        jit_kernel = matmul_smem(M, N, K, bM, bN, bK, num_stages=ns, threads=threads)
                    else:
                        jit_kernel = matmul_global(M, N, K, bM, bN, bK, threads=threads)

                    tflops = bench_kernel(jit_kernel, M, N, K)

                    vs_mlx = f"{tflops / mlx_tflops * 100:.0f}%" if mlx_tflops else "N/A"
                    vs_torch = f"{tflops / torch_tflops * 100:.0f}%"

                    marker = " <-- BEST" if tflops > best_tflops else ""
                    print(f"  {full_label:<40} {threads:>8} {tflops:>8.2f} {vs_mlx:>8} {vs_torch:>10}{marker}")

                    if tflops > best_tflops:
                        best_tflops = tflops
                        best_label = full_label
                except Exception as e:
                    import traceback

                    print(f"  {full_label:<40} {threads:>8} {'FAIL':>8}   {str(e)[:60]}")
                    traceback.print_exc()

        print(f"\n  Best: {best_label} = {best_tflops:.2f} TFLOPS")
        if mlx_tflops:
            print(f"  vs MLX: {best_tflops / mlx_tflops * 100:.0f}%, vs Torch: {best_tflops / torch_tflops * 100:.0f}%")


if __name__ == "__main__":
    main()
