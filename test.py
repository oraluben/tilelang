import tilelang

print("Imported tilelang")
from tilelang import tvm as tvm

# import tilelang.testing
import tilelang.language as T
import torch

print("Imports done", flush=True)


# @register_metal_postproc_callback
# def _p(code, target):
#     print(code)
#     return code


@tilelang.jit
def matmul(M, N, K, block_M, block_N, block_K, dtype="float32", accum_dtype="float"):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared")
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype, scope="shared")

            T.clear(C_shared)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)

                T.gemm_v2(A_shared, B_shared, C_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return gemm


if __name__ == "__main__":
    m = n = k = 1024
    torch_dtype = torch.float16
    dtype = "float16"

    a = torch.randn(m, k, device="mps", dtype=torch_dtype)
    b = torch.randn(k, n, device="mps", dtype=torch_dtype)
    c = torch.zeros(m, n, device="mps", dtype=torch_dtype)

    # torch_add = lambda: torch.matmul(a, b, out=c)
    # torch_add()
    # print(benchmark(torch_add, n=100))

    print("Starting compilation...", flush=True)
    jit_kernel = matmul(m, n, k, 16, 16, 8, dtype=dtype, accum_dtype="float")
    print("Compilation finished.", flush=True)

    print(jit_kernel.get_kernel_source())
    jit_kernel(a, b, c)
    print(c)
    print(a @ b)
