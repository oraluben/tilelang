import tilelang
import tilelang.language as T

import tvm_ffi
def func(c: str):
    print(f'1:', c)
tvm_ffi.register_global_func("tilelang_dbg", f=func)


@tilelang.jit
def add(M, block_M, dtype="float32"):
    @T.prim_func
    def add_kernel(
        A: T.Tensor((M, ), dtype),
        B: T.Tensor((M, ), dtype),
        C: T.Tensor((M, ), dtype),
    ):
        num_per_thread = 8
        with T.Kernel(T.ceildiv(M, block_M * num_per_thread), threads=128) as bx:
            for local_x, i in T.Parallel(block_M, num_per_thread):
                x = (bx * block_M + local_x) * num_per_thread
                C[x + i] = A[x + i] + B[x + i]

    return add_kernel


size = 1024 * 16 - 1
jit_kernel = add(size, 128)

print(jit_kernel.get_kernel_source())
