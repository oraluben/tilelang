import torch
import tilelang
import tilelang.language as T
from typing import Tuple, Optional


FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"


pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: True,
}

tilelang.disable_cache()

@tilelang.jit(target="musa", pass_configs=pass_configs)
def fp8_gemm_kernel(N, K, out_dtype=BF16, accum_dtype="float32"):
    assert out_dtype in [BF16, "float32"]

    M = T.symbolic("M")
    group_size = 32
    block_M = 128
    block_N = 128
    block_K = 64

    @T.prim_func
    def fp8_gemm_kernel_(
        A: T.Tensor[(M, K), FP8],
        B: T.Tensor[(N, K), FP8],
        C: T.Tensor[(M, N), out_dtype],
        scales_a: T.Tensor[(M, T.ceildiv(K, group_size)), FP32],
        scales_b: T.Tensor[(T.ceildiv(N, group_size), T.ceildiv(K, group_size)), FP32],
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), FP8)
            B_shared = T.alloc_shared((block_N, block_K), FP8)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            Scale_C_shared = T.alloc_shared((block_M), FP32)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Improve L2 Cache
            # T.use_swizzle(panel_size=10)

            T.clear(C_local)
            T.clear(C_local_accum)
            K_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(K_iters, num_stages=1):
                # Load A into shared memory
                T.copy(A[by * block_M, k * block_K], A_shared)
                # Load B into shared memory
                T.copy(B[bx * block_N, k * block_K], B_shared)
                # Load scale into shared memory
                Scale_B = scales_b[bx * block_N // group_size, k]
                for i in T.Parallel(block_M):
                    Scale_C_shared[i] = scales_a[by * block_M + i, k] * Scale_B

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                # Promote to enable 2xAcc
                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * Scale_C_shared[i]
                T.clear(C_local)
            # TMA store
            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return fp8_gemm_kernel_


def fp8_gemm(
    a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor
) -> torch.Tensor:
    """
    Perform a matrix multiplication using FP8 precision.
    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.
    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert a_s.is_contiguous() and b_s.is_contiguous(), (
        "Scaling factor tensors must be contiguous"
    )
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.bfloat16)
    kernel = fp8_gemm_kernel(N, K)
    print(kernel.get_kernel_source())
    kernel(a.view(M, K), b, c.view(M, N), a_s.view(M, -1), b_s)
    return c

device = "musa"
M, N, K = 4096, 4096, 4096
group_size = 32
A_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
B_fp32 = torch.randn(N, K, device=device, dtype=torch.float32)
A_fp8 = A_fp32.to(torch.float8_e4m3fn)
B_fp8 = B_fp32.to(torch.float8_e4m3fn)
def ceildiv(a, b):
    return (a + b - 1) // b
num_k_groups = ceildiv(K, group_size)
num_n_groups = ceildiv(N, group_size)
a_s = torch.ones(M, num_k_groups, device=device, dtype=torch.float32)
b_s = torch.ones(num_n_groups, num_k_groups, device=device, dtype=torch.float32)
C = fp8_gemm(A_fp8, a_s, B_fp8, b_s)
ref_C = A_fp8 @ B_fp8.T
print(C)
print(ref_C)
torch.testing.assert_close(C.to(torch.float32), ref_C.to(torch.float32), rtol=1.25e-1, atol=1.25e-1)

import triton

def get_tflops(latency_ms, M, N, K):
    latency = latency_ms / 1e3
    tflops = (2.0 * M * N * K) / latency / 1e12
    return tflops

ms_torch = triton.musa_testing.do_bench(lambda: torch.mm(A_fp8, B_fp8.T))
ms_tilelang = triton.musa_testing.do_bench(lambda: fp8_gemm(A_fp8, a_s, B_fp8, b_s))
print(f"torch latency: {ms_torch:.4f} ms")
print(f"tilelang kernel latency: {ms_tilelang:.4f} ms")
print(f"torch tflops: {get_tflops(ms_torch, M, N, K):.4f}")
print(f"tilelang kernel tflops: {get_tflops(ms_tilelang, M, N, K):.4f}")
print(f"tilelang2torch speed_up: {round(ms_torch / ms_tilelang, 6)}")
