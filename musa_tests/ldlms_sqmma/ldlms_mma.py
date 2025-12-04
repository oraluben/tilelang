import tilelang
import tilelang.language as T

tilelang.disable_cache()

def matmul(M, N, K,
           block_M, block_N, block_K,
           dtype="float16", accum_dtype="float"):
    @T.prim_func
    def gemm(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return gemm


if __name__ == "__main__":
    M = 256
    N = 256
    K = 256
    block_M = 32
    block_N = 32
    block_K = 32
    program = matmul(M, N, K, block_M, block_N, block_K)
    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: True,
    }
    kernel = tilelang.compile(
        program,
        out_idx=-1,
        target="musa",
        execution_backend="cython",
        verbose=True,
        pass_configs=pass_configs,
    )
    print(kernel.get_kernel_source())
    import torch
    a = torch.randn(M, K, device="musa", dtype=torch.float16)
    b = torch.randn(K, N, device="musa", dtype=torch.float16)
    print('start kernel')
    c = kernel(a, b)
    print('start ref')
    ref_c = a @ b
    print('compare')
    print(ref_c)
    print(c)
    torch.testing.assert_close(c.to(torch.float32), ref_c.to(torch.float32), rtol=1e-2, atol=1e-2)
    print("matmul matches torch reference.")

