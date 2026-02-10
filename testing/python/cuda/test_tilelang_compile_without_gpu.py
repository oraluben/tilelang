"""Validate that TileLang compiles CUDA kernels without a real GPU device."""

import pytest
import tilelang
import tilelang.language as T
from tilelang.utils.target import check_cuda_availability

requires_cuda_toolkit = pytest.mark.skipif(
    not check_cuda_availability(),
    reason="CUDA toolkit (nvcc) not available",
)


@requires_cuda_toolkit
def test_compile_without_gpu():
    """Compilation with auto target should succeed with only the CUDA toolkit."""

    @T.prim_func
    def matmul(
        A: T.Tensor((128, 128), "float16"),
        B: T.Tensor((128, 128), "float16"),
        C: T.Tensor((128, 128), "float16"),
    ):
        with T.Kernel(T.ceildiv(128, 64), T.ceildiv(128, 64), threads=128) as (bx, by):
            A_shared = T.alloc_shared((64, 64), "float16")
            B_shared = T.alloc_shared((64, 64), "float16")
            C_local = T.alloc_fragment((64, 64), "float16")
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(128, 64), num_stages=2):
                T.copy(A[bx * 64, k * 64], A_shared)
                T.copy(B[k * 64, by * 64], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[bx * 64, by * 64])

    kernel = tilelang.compile(matmul)
    source = kernel.get_kernel_source()
    assert source and "matmul_kernel" in source
