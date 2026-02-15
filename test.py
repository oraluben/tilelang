import tilelang
from tilelang import tvm as tvm
from time import sleep
# import tilelang.testing
import tilelang.language as T
import json
import torch
import os


from tilelang.engine.callback import register_metal_postproc_callback

@register_metal_postproc_callback
def _p(code, target):
    print(code)
    return code


@tilelang.jit
def matmul(M, N, K, block_M, block_N, block_K, dtype="float32", accum_dtype="float"):

    @T.prim_func
    def gemm(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
                    bx,
                    by,
                ):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)

                T.gemm_v2(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def benchmark(f, n, *args, **kwargs):
    # trigger jit
    f(*args, **kwargs)

    torch.mps.synchronize()
    with torch.mps.profiler.profile(mode="interval,event", wait_until_completed=True):
        start = torch.mps.Event(enable_timing=True)
        end = torch.mps.Event(enable_timing=True)
        start.record()

        for _ in range(n):
            f(*args, **kwargs)

        end.record()

        start.synchronize()
        end.synchronize()

        return start.elapsed_time(end) / 1000


if __name__ == "__main__":
    m = n = k = 1024
    torch_dtype = torch.float16
    dtype = 'float16'

    a = torch.randn(m, k, device="mps", dtype=torch_dtype)
    b = torch.randn(k, n, device="mps", dtype=torch_dtype)
    c = torch.zeros(m, n, device="mps", dtype=torch_dtype)

    # torch_add = lambda: torch.matmul(a, b, out=c)
    # torch_add()
    # print(benchmark(torch_add, n=100))

    jit_kernel = tilelang.compile(matmul(m, n, k, 16, 16, 16, dtype=dtype, accum_dtype="float"), target="mps")

    # jit_kernel(a, b, c)
