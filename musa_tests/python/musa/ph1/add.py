import tilelang
import tilelang.language as T
import torch
 
def elementwise_add(N, num_per_thread=8, threads=256, dtype="float32"):
 
    @T.prim_func
    def main(A: T.Tensor((N), dtype), B: T.Tensor((N), dtype), C: T.Tensor((N), dtype)):
        with T.Kernel(T.ceildiv(N, threads * num_per_thread), threads=threads) as (b_x):
            # vector add.
            for i, j in T.Parallel(threads, num_per_thread):
                offsets = (b_x * threads + i) * num_per_thread
                C[offsets + j] = A[offsets + j] + B[offsets + j]
 
    return main
 
def ref_program(x, y):
    return x + y
 
N = 4096
 
program = elementwise_add(N)
kernel = tilelang.compile(program, out_idx=-1, target="musa", execution_backend="cython", verbose=True)
print(kernel.get_kernel_source())
 
a = torch.randn(N, dtype=torch.float32, device="musa")
b = torch.randn(N, dtype=torch.float32, device="musa")
 
c = kernel(a, b)
 
print(c)
 
torch.testing.assert_close(c, ref_program(a, b), rtol=1e-2, atol=1e-2)
