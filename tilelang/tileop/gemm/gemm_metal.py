from .gemm_base import GemmBase
from .inst import GemmInst
from tilelang.layout import make_swizzled_layout
from tilelang.intrinsics.metal_macro_generator import (
    MetalSimdgroupIntrinEmitter,
)
from tilelang.utils.language import is_full_region
from tvm.target import Target
from tvm.ir import Range
from tvm import tir
from tilelang import language as T
from tilelang.transform.simplify import _Simplify


class GemmMetal(GemmBase):
    def _create_emitter(self, thread_nums, target, thread_var=None):
        m_warp, n_warp = self.policy.compute_warp_partition(
            self.M, self.N, thread_nums, target, GemmInst.METAL
        )
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        return MetalSimdgroupIntrinEmitter(
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
        )

    def infer_layout(self, target: Target, thread_nums: int):
        emitter = self._create_emitter(thread_nums, target)
        if self.is_gemm_ss():
            return {
                self.A: make_swizzled_layout(self.A),
                self.B: make_swizzled_layout(self.B),
                self.C: emitter.make_metal_store_layout(self.C),
            }
        else:
            raise ValueError(
                f"Unsupported gemm combination for Metal, A: {self.A.scope()}, B: {self.B.scope()}"
            )

    def lower(self, layout_map: dict, target: Target, thread_bounds: Range, thread_var: tir.Var):
        thread_nums = thread_bounds.extent
        emitter = self._create_emitter(thread_nums, target, thread_var)

        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion

        C_buf = C_region.buffer

        clear_accum = self.clear_accum

        assert is_full_region(C_region), "Fragment output C must be a full region"

        if self.is_gemm_ss():

            @T.prim_func
            def _gemm_ssr() -> None:
                emitter.gemm_ss(A_region, B_region, C_buf, clear_accum)

            return _Simplify(_gemm_ssr, inline_let=True)
        else:
            raise ValueError(
                f"Unsupported gemm combination for Metal, A: {self.A.scope()}, B: {self.B.scope()}"
            )
