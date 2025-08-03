import tilelang
import tilelang.language as T

import torch


@tilelang.jit
def add(M, block_M, dtype="float32"):

    @T.prim_func
    def add_kernel(
        A: T.Tensor((M, ), dtype),
        B: T.Tensor((M, ), dtype),
        C: T.Tensor((M, ), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=128) as bx:
            start_x = bx * block_M
            for local_x in T.Parallel(block_M):
                x = start_x + local_x
                C[x] = A[x] + B[x]

    return add_kernel


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 128
    jit_kernel = add(N, BLOCK_SIZE)

    jit_kernel(a, b, c)


import torch

tile_add = solve


def benchmark(f, n=None, nvtx=None, *args, **kwargs):
    # trigger jit
    f(*args, **kwargs)

    if n is None:
        assert False, 'TODO: estimate time'

    if nvtx:
        torch.cuda.nvtx.range_push(nvtx)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for _ in range(n):
        f(*args, **kwargs)

    end.record()
    end.synchronize()

    if nvtx:
        torch.cuda.nvtx.range_pop()

    return start.elapsed_time(end) / 1000


sizes = [128 * 2 ** p for p in range(0, 1)]
y1 = []
y2 = []
y3 = []

device = 'mps'
device = 'cpu'

for size in sizes:
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    c = torch.zeros(size, device=device)

    tile_add(a, b, c, size)
