from .gemm_base import GemmBase
from .inst import GemmInst
from tilelang.utils.language import is_shared, is_fragment, is_full_region, is_metal_simdgroup
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.ir import Range
from tvm import tir
from tilelang import language as T
from tilelang.transform.simplify import _Simplify


def _is_register_like(buffer) -> bool:
    """Check if a buffer is register-like (fragment or metal.simdgroup)."""
    return is_fragment(buffer) or is_metal_simdgroup(buffer)


class GemmMetal(GemmBase):

    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def is_gemm_sr(self) -> bool:
        return is_shared(self.A) and _is_register_like(self.B)

    def is_gemm_rs(self) -> bool:
        return _is_register_like(self.A) and is_shared(self.B)

    def is_gemm_rr(self) -> bool:
        return _is_register_like(self.A) and _is_register_like(self.B)

    def infer_layout(self, target: Target, thread_nums: int):
        m_warp, n_warp = self.policy.compute_warp_partition(
            self.M, self.N, thread_nums, target, GemmInst.METAL)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)

        from tilelang.intrinsics.metal_macro_generator import MPSIntrinEmitter

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
        )

        if self.is_gemm_ss():
            return {
                self.C: mps_emitter.make_mma_store_layout(self.C),
            }
        elif self.is_gemm_sr():
            return {
                self.B: mps_emitter.make_mma_store_layout(self.B),
                self.C: mps_emitter.make_mma_store_layout(self.C),
            }
        elif self.is_gemm_rs():
            return {
                self.A: mps_emitter.make_mma_store_layout(self.A),
                self.C: mps_emitter.make_mma_store_layout(self.C),
            }
        elif self.is_gemm_rr():
            return {
                self.A: mps_emitter.make_mma_store_layout(self.A),
                self.B: mps_emitter.make_mma_store_layout(self.B),
                self.C: mps_emitter.make_mma_store_layout(self.C),
            }
        else:
            raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")

    def lower(self, layout_map: dict, target: Target, thread_bounds: Range, thread_var: tir.Var):
        thread_nums = thread_bounds.extent
        m_warp, n_warp = self.policy.compute_warp_partition(
            self.M, self.N, thread_nums, target, GemmInst.METAL)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)

        from tilelang.intrinsics.metal_macro_generator import MPSIntrinEmitter

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
        )

        in_dtype = self.in_dtype
        accum_dtype = self.accum_dtype
        warp_rows = mps_emitter.warp_rows
        warp_cols = mps_emitter.warp_cols
        num_simd_c = warp_rows * warp_cols
        block_K = mps_emitter.chunk
        micro_size_k = mps_emitter.micro_size_k

        M = int(self.M)
        N = int(self.N)

        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion

        A_buf = A_region.buffer
        B_buf = B_region.buffer
        C_buf = C_region.buffer

        clear_accum = self.clear_accum

        assert block_K >= micro_size_k, f"block_K ({block_K}) must be >= micro_size_k ({micro_size_k})"
        assert is_full_region(C_region), "Fragment output C must be a full region"

        if self.is_gemm_ss():

            @T.prim_func
            def _gemm_ssr() -> None:
                A_local = T.alloc_local((warp_rows * 64), in_dtype, scope="metal.simdgroup")
                B_local = T.alloc_local((warp_cols * 64), in_dtype, scope="metal.simdgroup")
                C_simd = T.alloc_local((num_simd_c * 64), accum_dtype, scope="metal.simdgroup")
                C_shared = T.alloc_shared((M, N), accum_dtype, scope="shared")
                if clear_accum:
                    for _i in T.serial(num_simd_c):
                        T.make_filled_simdgroup_matrix(
                            C_simd.data, _i, T.cast(0, accum_dtype))
                else:
                    mps_emitter.fragment_to_simd(C_buf, C_simd, C_shared)
                for ki in T.serial(0, (block_K // micro_size_k)):
                    mps_emitter.ldmatrix_a(A_local, A_region, ki)
                    mps_emitter.ldmatrix_b(B_local, B_region, ki)
                    mps_emitter.mma(A_local, B_local, C_simd, ki)
                mps_emitter.simd_to_fragment(C_simd, C_buf, C_shared)

            return _Simplify(_gemm_ssr, inline_let=True)
        elif self.is_gemm_sr():
            assert is_full_region(B_region), "Fragment input B must be a full region"

            @T.prim_func
            def _gemm_srr() -> None:
                A_local = T.alloc_local((warp_rows * 64), in_dtype, scope="metal.simdgroup")
                C_simd = T.alloc_local((num_simd_c * 64), accum_dtype, scope="metal.simdgroup")
                C_shared = T.alloc_shared((M, N), accum_dtype, scope="shared")
                if clear_accum:
                    for _i in T.serial(num_simd_c):
                        T.make_filled_simdgroup_matrix(
                            C_simd.data, _i, T.cast(0, accum_dtype))
                else:
                    mps_emitter.fragment_to_simd(C_buf, C_simd, C_shared)
                for ki in T.serial(0, (block_K // micro_size_k)):
                    mps_emitter.ldmatrix_a(A_local, A_region, ki)
                    mps_emitter.mma(A_local, B_buf, C_simd, ki)
                mps_emitter.simd_to_fragment(C_simd, C_buf, C_shared)

            return _Simplify(_gemm_srr, inline_let=True)
        elif self.is_gemm_rs():
            assert is_full_region(A_region), "Fragment input A must be a full region"

            @T.prim_func
            def _gemm_rsr() -> None:
                B_local = T.alloc_local((warp_cols * 64), in_dtype, scope="metal.simdgroup")
                C_simd = T.alloc_local((num_simd_c * 64), accum_dtype, scope="metal.simdgroup")
                C_shared = T.alloc_shared((M, N), accum_dtype, scope="shared")
                if clear_accum:
                    for _i in T.serial(num_simd_c):
                        T.make_filled_simdgroup_matrix(
                            C_simd.data, _i, T.cast(0, accum_dtype))
                else:
                    mps_emitter.fragment_to_simd(C_buf, C_simd, C_shared)
                for ki in T.serial(0, (block_K // micro_size_k)):
                    mps_emitter.ldmatrix_b(B_local, B_region, ki)
                    mps_emitter.mma(A_buf, B_local, C_simd, ki)
                mps_emitter.simd_to_fragment(C_simd, C_buf, C_shared)

            return _Simplify(_gemm_rsr, inline_let=True)
        elif self.is_gemm_rr():
            assert is_full_region(A_region), "Fragment input A must be a full region"
            assert is_full_region(B_region), "Fragment input B must be a full region"

            @T.prim_func
            def _gemm_rrr() -> None:
                C_simd = T.alloc_local((num_simd_c * 64), accum_dtype, scope="metal.simdgroup")
                C_shared = T.alloc_shared((M, N), accum_dtype, scope="shared")
                if clear_accum:
                    for _i in T.serial(num_simd_c):
                        T.make_filled_simdgroup_matrix(
                            C_simd.data, _i, T.cast(0, accum_dtype))
                else:
                    mps_emitter.fragment_to_simd(C_buf, C_simd, C_shared)
                for ki in T.serial(0, (block_K // micro_size_k)):
                    mps_emitter.mma(A_buf, B_buf, C_simd, ki)
                mps_emitter.simd_to_fragment(C_simd, C_buf, C_shared)

            return _Simplify(_gemm_rrr, inline_let=True)
        else:
            raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")
