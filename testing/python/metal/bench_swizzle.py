#!/usr/bin/env python3
"""Test correctness and benchmark block swizzle on Metal."""

import tilelang
import tilelang.language as T
import torch
import time

tilelang.env.disable_cache()


@tilelang.jit
def matmul_swizzle(M, N, K, block_M, block_N, block_K, threads, panel_size=0):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            if panel_size > 0:
                T.use_swizzle(panel_size)
            A_shared = T.alloc_shared((block_M, block_K), T.float16, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N), T.float16, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return kernel


def bench(kernel_fn, M, N, K, warmup=5, iters=20):
    a = torch.randn(M, K, dtype=torch.float16, device="mps")
    b = torch.randn(K, N, dtype=torch.float16, device="mps")
    c = torch.zeros(M, N, dtype=torch.float32, device="mps")
    for _ in range(warmup):
        kernel_fn(a, b, c)
    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        kernel_fn(a, b, c)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    return 2 * M * N * K * iters / elapsed / 1e12


def check_correctness(M, N, K, bM, bN, bK, threads, panel_size):
    a = torch.randn(M, K, dtype=torch.float16, device="mps")
    b = torch.randn(K, N, dtype=torch.float16, device="mps")
    c = torch.zeros(M, N, dtype=torch.float32, device="mps")
    k = matmul_swizzle(M, N, K, bM, bN, bK, threads, panel_size=panel_size)
    k(a, b, c)
    ref = a.to(torch.float32) @ b.to(torch.float32)
    max_diff = (ref - c).abs().max().item()
    ok = max_diff < 1.0
    print(f"  Correctness {M}x{N}x{K} panel={panel_size}: max_diff={max_diff:.4f} {'OK' if ok else 'FAIL'}")
    return ok


print("=== Correctness ===")
check_correctness(128, 128, 128, 64, 128, 32, 128, panel_size=4)
check_correctness(1024, 1024, 1024, 64, 128, 32, 128, panel_size=4)
check_correctness(4096, 4096, 4096, 64, 128, 32, 128, panel_size=8)

print("\n=== Benchmark 4096 ===")
M = N = K = 4096
bM, bN, bK, threads = 64, 128, 32, 128

print(f"  {'Config':<30} {'TFLOPS':>8}")
for ps in [0, 2, 4, 8, 16]:
    try:
        k = matmul_swizzle(M, N, K, bM, bN, bK, threads, panel_size=ps)
        t = bench(k, M, N, K)
        label = f"panel_size={ps}" if ps > 0 else "no swizzle"
        print(f"  {label:<30} {t:>8.2f}")
    except Exception as e:
        print(f"  panel_size={ps:<25} FAIL  {str(e)[:50]}")

print("\n=== Benchmark 8192 ===")
M = N = K = 8192
for ps in [0, 4, 8]:
    try:
        k = matmul_swizzle(M, N, K, bM, bN, bK, threads, panel_size=ps)
        t = bench(k, M, N, K)
        label = f"panel_size={ps}" if ps > 0 else "no swizzle"
        print(f"  {label:<30} {t:>8.2f}")
    except Exception as e:
        print(f"  panel_size={ps:<25} FAIL  {str(e)[:50]}")
