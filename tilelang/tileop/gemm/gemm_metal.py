from __future__ import annotations

from .gemm_base import GemmBase
from .inst import GemmInst
from tilelang.layout import Layout, make_linear_layout
from tilelang.utils.language import is_shared, is_global, is_full_region, is_metal_cooperative_tensor, is_fragment
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.ir import Range
from tvm import tir
from tilelang import language as T
from tilelang.transform.simplify import _Simplify


def _make_padded_layout(buffer):
    shape = buffer.shape
    stride = int(shape[-2])
    continuous = int(shape[-1])
    element_bits = int(tvm.DataType(buffer.dtype).bits)
    padded = continuous
    if (element_bits * continuous) % 256 == 0:
        padded += 128 // element_bits
    return Layout([stride, continuous], lambda i, j: i * padded + j)


class GemmMetal(GemmBase):
    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def is_gemm_gg(self) -> bool:
        return is_global(self.A) and is_global(self.B)

    def infer_layout(self, target: Target, thread_nums: int):
        if self.is_gemm_ss():
            return {
                self.A: make_linear_layout(self.A),
                self.B: make_linear_layout(self.B),
            }
        return {}

    @staticmethod
    def _get_padded_stride(buffer):
        continuous = int(buffer.shape[-1])
        element_bits = int(tvm.DataType(buffer.dtype).bits)
        padded = continuous
        if (element_bits * continuous) % 256 == 0:
            padded += 128 // element_bits
        return padded

    def lower(
        self, layout_map: dict, target: Target, thread_bounds: Range, thread_var: tir.Var, mbar_phase_expr: tir.PrimExpr | None = None
    ):
        thread_nums = thread_bounds.extent
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GemmInst.METAL)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)

        from tilelang.intrinsics.metal_macro_generator import MPSIntrinEmitter

        a_stride = None
        b_stride = None

        mps_emitter = MPSIntrinEmitter(
            a_dtype=self.in_dtype,
            b_dtype=self.in_dtype,
            accum_dtype=self.accum_dtype,
            a_transposed=self.trans_A,
            b_transposed=self.trans_B,
            block_row_warps=m_warp,
            block_col_warps=n_warp,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=self.chunk,
            thread_var=thread_var,
            a_stride_override=a_stride,
            b_stride_override=b_stride,
        )

        in_dtype = self.in_dtype
        accum_dtype = self.accum_dtype
        warp_rows = mps_emitter.warp_rows
        warp_cols = mps_emitter.warp_cols
        num_simd_c = warp_rows * warp_cols
        block_K = mps_emitter.chunk
        micro_size_k = mps_emitter.micro_size_k

        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion

        C_buf = C_region.buffer

        clear_accum = self.clear_accum
        c_in_cooperative_tensor = is_metal_cooperative_tensor(C_buf) or is_fragment(C_buf)

        assert block_K >= micro_size_k, f"block_K ({block_K}) must be >= micro_size_k ({micro_size_k})"
        assert is_full_region(C_region), "Fragment output C must be a full region"
        if self.is_gemm_ss() or self.is_gemm_gg():
            if c_in_cooperative_tensor:

                @T.prim_func
                def _gemm_cooperative_tensor() -> None:
                    A_local = T.alloc_local((warp_rows * 512), in_dtype, scope="metal.cooperative_tensor")
                    B_local = T.alloc_local((warp_cols * 512), in_dtype, scope="metal.cooperative_tensor")
                    if clear_accum:
                        for _i in T.serial(num_simd_c):
                            T.cooperative_tensor_fill(C_buf.data, _i, T.cast(0, accum_dtype), 16, 16)
                    for ki in T.serial(0, (block_K // micro_size_k)):
                        mps_emitter.ldmatrix_a(A_local, A_region, ki)
                        mps_emitter.ldmatrix_b(B_local, B_region, ki)
                        mps_emitter.mma(A_local, B_local, C_buf)

                return _Simplify(_gemm_cooperative_tensor, inline_let=True)
            else:

                @T.prim_func
                def _gemm_with_c_writeback() -> None:
                    A_local = T.alloc_local((warp_rows * 512), in_dtype, scope="metal.cooperative_tensor")
                    B_local = T.alloc_local((warp_cols * 512), in_dtype, scope="metal.cooperative_tensor")
                    C_ct = T.alloc_local((num_simd_c * 256), accum_dtype, scope="metal.cooperative_tensor")
                    if clear_accum:
                        for _i in T.serial(num_simd_c):
                            T.cooperative_tensor_fill(C_ct.data, _i, T.cast(0, accum_dtype), 16, 16)
                    else:
                        mps_emitter.simd_load(C_ct, C_buf)
                    for ki in T.serial(0, (block_K // micro_size_k)):
                        mps_emitter.ldmatrix_a(A_local, A_region, ki)
                        mps_emitter.ldmatrix_b(B_local, B_region, ki)
                        mps_emitter.mma(A_local, B_local, C_ct)

                    mps_emitter.simd_store(C_ct, C_buf)

                return _Simplify(_gemm_with_c_writeback, inline_let=True)
        else:
            raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")
