import torch
import tilelang
import tilelang.language as T


FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"


pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: True,
}

tilelang.disable_cache()


@tilelang.jit(target="musa", pass_configs=pass_configs, out_idx=[4])
def fp8_index_kernel(h: int, d: int):
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")

    blk_n1 = 64
    blk_n2 = 32

    @T.prim_func
    def fp8_index_kernel_(
        q: T.Tensor[(b, m, h, d), FP8],
        q_s: T.Tensor[(b, m, h), FP32],
        k: T.Tensor[(b, n, d), FP8],
        k_s: T.Tensor[(b, n), FP32],
        o: T.Tensor[(b, m, n), FP32],
    ) -> None:
        with T.Kernel(b, m, T.ceildiv(n, blk_n1)) as (i_b, i_m, i1_n):
            q_smem = T.alloc_shared((h, d), FP8)
            T.copy(q[i_b, i_m, 0, 0], q_smem)

            q_s_frag = T.alloc_fragment(h, FP32)
            T.copy(q_s[i_b, i_m, 0], q_s_frag)

            for i2_n in T.Pipelined(blk_n1 // blk_n2, num_stages=1):
                k_smem = T.alloc_shared((blk_n2, d), FP8)
                T.copy(k[i_b, i1_n * blk_n1 + i2_n * blk_n2, 0], k_smem)

                k_s_frag = T.alloc_fragment(blk_n2, FP32)
                T.copy(k_s[i_b, i1_n * blk_n1 + i2_n * blk_n2], k_s_frag)

                logits = T.alloc_fragment((blk_n2, h), FP32)
                T.gemm(
                    k_smem,
                    q_smem,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=True,
                )

                for i_h, i3_n in T.Parallel(h, blk_n2):
                    logits[i3_n, i_h] = T.max(logits[i3_n, i_h], 0) * q_s_frag[i_h]

                logits_sum = T.alloc_fragment(blk_n2, FP32)
                T.reduce_sum(logits, logits_sum, dim=1)

                for i3_n in T.Parallel(blk_n2):
                    logits_sum[i3_n] *= k_s_frag[i3_n]

                T.copy(logits_sum, o[i_b, i_m, i1_n * blk_n1 + i2_n * blk_n2])

    return fp8_index_kernel_


def fp8_index(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    """
    Perform index score using FP8 precision.
    Args:
        q (torch.Tensor): The Q tensor, must be contiguous.
        q_s (torch.Tensor): The scaling factor for Q (float), must be contiguous.
        k (torch.Tensor): The K tensor, must be contiguous.
        k_s (torch.Tensor): The scaling factor for K (e8m0 here), must be contiguous.
        fp8 q @ fp8 k -> fp32 logits
        relu(fp32 logits) * q_s (weights) -> fp32 logits
        fp32 logits -> fp32 logits_sum
        fp32 logits_sum * k_s (e8m0) -> fp32 index_score
    """
    kernel = fp8_index_kernel(q.shape[2], q.shape[3])
    return kernel(q, q_s, k, k_s)


def fp8_index_torch(q, q_s, k, k_s):
    # q: (B, M, H, D) fp8_e4m3fn
    # q_s: (B, M, H) float
    # k: (B, N, D) fp8_e4m3fn
    # k_s: (B, N) float
    B, M, H, D = q.shape
    assert k.shape[0] == B and k.shape[2] == D
    N = k.shape[1]
    assert q_s.shape == (B, M, H)
    assert k_s.shape == (B, N)

    q_f = q.float()  # accum in fp32
    k_f = k.float()
    # logits: (B, M, N, H)
    logits = torch.einsum("bnd,bmhd->bmnh", k_f, q_f)
    logits = torch.relu(logits) * q_s.unsqueeze(2)  # broadcast over n
    out = logits.sum(dim=-1) * k_s.unsqueeze(1)  # (B, M, N)
    return out  # float32


device = "musa"

B = 256
M = 256
H = 32
D = 32
N = 256

q_fp32 = torch.randn(B, M, H, D, device=device, dtype=torch.float32)
k_fp32 = torch.randn(B, N, D, device=device, dtype=torch.float32)

q_fp8 = q_fp32.to(torch.float8_e4m3fn)
k_fp8 = k_fp32.to(torch.float8_e4m3fn)

q_s = torch.rand(B, M, H, device=device, dtype=torch.float32)
k_s = torch.rand(B, N, device=device, dtype=torch.float32)
o = fp8_index(q_fp8, q_s, k_fp8, k_s)

o_ref = fp8_index_torch(q_fp8, q_s, k_fp8, k_s)

print("q shape:", q_fp8.shape)
print("q_s shape:", q_s.shape)
print("k shape:", k_fp8.shape)
print("k_s shape:", k_s.shape)
print("o shape:", o.shape)
print(o_ref)
print(o)
torch.testing.assert_close(o.to(torch.float32), o_ref.to(torch.float32), rtol=1.25e-2, atol=1.25e-2)
