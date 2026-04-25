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

        # inner_k_steps=2 batches two K-micro-steps per load cycle (MLX's TK=2).
        # This halves K-loop iterations but doubles A/B register footprint.
        # Only safe when C register pressure is low enough (c_per_thread ≤ 128B).
        # MMA tile: 16×32 output, 32 threads/simdgroup → 16 floats = 64 bytes per tile per thread
        c_bytes_per_thread = warp_row_tiles * warp_col_tiles * 64
        inner_k_steps = 2 if c_bytes_per_thread <= 128 else 1
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
            inner_k_steps=inner_k_steps,
        )

        in_dtype = self.in_dtype
        accum_dtype = self.accum_dtype
        warp_rows = mps_emitter.warp_rows
        warp_cols = mps_emitter.warp_cols
        num_simd_c = warp_rows * warp_cols
        block_K = mps_emitter.chunk
        micro_size_x = mps_emitter.micro_size_x
        micro_size_y = mps_emitter.micro_size_y
        micro_size_k = mps_emitter.micro_size_k
        inner_k_steps = mps_emitter.inner_k_steps
        a_tile_elems = micro_size_x * micro_size_k
        b_tile_elems = micro_size_k * micro_size_y
        c_tile_elems = micro_size_x * micro_size_y

        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion

        C_buf = C_region.buffer

        clear_accum = self.clear_accum
        c_in_cooperative_tensor = is_metal_cooperative_tensor(C_buf) or is_fragment(C_buf)

        assert block_K >= micro_size_k, f"block_K ({block_K}) must be >= micro_size_k ({micro_size_k})"
        assert is_full_region(C_region), "Fragment output C must be a full region"

        if not (self.is_gemm_ss() or self.is_gemm_gg()):
            raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")

        num_k_iters = block_K // micro_size_k
        # c_per_thread: bytes of C accumulator each thread must hold in registers.
        #   num_simd_c = warp_rows * warp_cols  (MMA output tiles per simdgroup)
        #   c_tile_elems = micro_size_x * micro_size_y = 16 * 32 = 512  (elements per tile)
        #   * 4 / 32: 4 bytes per float32 element, 32 threads per simdgroup
        # Examples (128×128 block):
        #   1024 threads (32 sg): 1 tile  →  64 B/thread
        #    512 threads (16 sg): 2 tiles → 128 B/thread
        #    256 threads  (8 sg): 4 tiles → 256 B/thread
        c_per_thread = num_simd_c * c_tile_elems * 4 // 32
        # Double-buffer requires:
        #   - ≥4 K iterations (otherwise prologue/epilogue overhead dominates)
        #   - inner_k_steps == 1 (batched K not yet compatible with DB)
        #   - C fits in registers alongside A0/A1/B0/B1 double-buffer pairs
        #     (DB adds ~192 B/thread for A/B copies; 128 B/thread C is the
        #      empirical limit before register spilling hurts more than DB helps)
        use_double_buffer = num_k_iters >= 4 and inner_k_steps == 1 and c_per_thread <= 128

        if c_in_cooperative_tensor:
            if use_double_buffer:

                @T.prim_func
                def _gemm_cooperative_tensor_db() -> None:
                    A0 = T.alloc_local((warp_rows * a_tile_elems), in_dtype, scope="metal.cooperative_tensor")
                    A1 = T.alloc_local((warp_rows * a_tile_elems), in_dtype, scope="metal.cooperative_tensor")
                    B0 = T.alloc_local((warp_cols * b_tile_elems), in_dtype, scope="metal.cooperative_tensor")
                    B1 = T.alloc_local((warp_cols * b_tile_elems), in_dtype, scope="metal.cooperative_tensor")
                    if clear_accum:
                        for _i in T.serial(num_simd_c):
                            T.cooperative_tensor_fill(C_buf.data, _i, T.cast(0, accum_dtype), micro_size_x, micro_size_y)

                    mps_emitter.ldmatrix_a(A0, A_region, 0)
                    mps_emitter.ldmatrix_b(B0, B_region, 0)

                    if num_k_iters % 2 == 0:
                        for ki in T.serial(1, num_k_iters - 1, 2):
                            mps_emitter.ldmatrix_a(A1, A_region, ki)
                            mps_emitter.ldmatrix_b(B1, B_region, ki)
                            mps_emitter.mma(A0, B0, C_buf)
                            mps_emitter.ldmatrix_a(A0, A_region, ki + 1)
                            mps_emitter.ldmatrix_b(B0, B_region, ki + 1)
                            mps_emitter.mma(A1, B1, C_buf)
                        mps_emitter.ldmatrix_a(A1, A_region, num_k_iters - 1)
                        mps_emitter.ldmatrix_b(B1, B_region, num_k_iters - 1)
                        mps_emitter.mma(A0, B0, C_buf)
                        mps_emitter.mma(A1, B1, C_buf)
                    else:
                        for ki in T.serial(1, num_k_iters, 2):
                            mps_emitter.ldmatrix_a(A1, A_region, ki)
                            mps_emitter.ldmatrix_b(B1, B_region, ki)
                            mps_emitter.mma(A0, B0, C_buf)
                            mps_emitter.ldmatrix_a(A0, A_region, ki + 1)
                            mps_emitter.ldmatrix_b(B0, B_region, ki + 1)
                            mps_emitter.mma(A1, B1, C_buf)
                        mps_emitter.mma(A0, B0, C_buf)

                return _Simplify(_gemm_cooperative_tensor_db, inline_let=True)
            else:

                @T.prim_func
                def _gemm_cooperative_tensor() -> None:
                    A_local = T.alloc_local((warp_rows * a_tile_elems * inner_k_steps), in_dtype, scope="metal.cooperative_tensor")
                    B_local = T.alloc_local((warp_cols * b_tile_elems * inner_k_steps), in_dtype, scope="metal.cooperative_tensor")
                    if clear_accum:
                        for _i in T.serial(num_simd_c):
                            T.cooperative_tensor_fill(C_buf.data, _i, T.cast(0, accum_dtype), micro_size_x, micro_size_y)
                    for k_outer in T.serial(0, (block_K // (micro_size_k * inner_k_steps))):
                        for k_inner in T.serial(0, inner_k_steps):
                            ki = k_outer * inner_k_steps + k_inner
                            mps_emitter.ldmatrix_a(A_local, A_region, ki, k_inner)
                            mps_emitter.ldmatrix_b(B_local, B_region, ki, k_inner)
                        for k_inner in T.serial(0, inner_k_steps):
                            mps_emitter.mma(A_local, B_local, C_buf, k_inner)

                return _Simplify(_gemm_cooperative_tensor, inline_let=True)
        else:
            if use_double_buffer:

                @T.prim_func
                def _gemm_with_c_writeback_db() -> None:
                    A0 = T.alloc_local((warp_rows * a_tile_elems), in_dtype, scope="metal.cooperative_tensor")
                    A1 = T.alloc_local((warp_rows * a_tile_elems), in_dtype, scope="metal.cooperative_tensor")
                    B0 = T.alloc_local((warp_cols * b_tile_elems), in_dtype, scope="metal.cooperative_tensor")
                    B1 = T.alloc_local((warp_cols * b_tile_elems), in_dtype, scope="metal.cooperative_tensor")
                    C_ct = T.alloc_local((num_simd_c * c_tile_elems), accum_dtype, scope="metal.cooperative_tensor")
                    if clear_accum:
                        for _i in T.serial(num_simd_c):
                            T.cooperative_tensor_fill(C_ct.data, _i, T.cast(0, accum_dtype), micro_size_x, micro_size_y)
                    else:
                        mps_emitter.simd_load(C_ct, C_buf)

                    mps_emitter.ldmatrix_a(A0, A_region, 0)
                    mps_emitter.ldmatrix_b(B0, B_region, 0)

                    if num_k_iters % 2 == 0:
                        for ki in T.serial(1, num_k_iters - 1, 2):
                            mps_emitter.ldmatrix_a(A1, A_region, ki)
                            mps_emitter.ldmatrix_b(B1, B_region, ki)
                            mps_emitter.mma(A0, B0, C_ct)
                            mps_emitter.ldmatrix_a(A0, A_region, ki + 1)
                            mps_emitter.ldmatrix_b(B0, B_region, ki + 1)
                            mps_emitter.mma(A1, B1, C_ct)
                        mps_emitter.ldmatrix_a(A1, A_region, num_k_iters - 1)
                        mps_emitter.ldmatrix_b(B1, B_region, num_k_iters - 1)
                        mps_emitter.mma(A0, B0, C_ct)
                        mps_emitter.mma(A1, B1, C_ct)
                    else:
                        for ki in T.serial(1, num_k_iters, 2):
                            mps_emitter.ldmatrix_a(A1, A_region, ki)
                            mps_emitter.ldmatrix_b(B1, B_region, ki)
                            mps_emitter.mma(A0, B0, C_ct)
                            mps_emitter.ldmatrix_a(A0, A_region, ki + 1)
                            mps_emitter.ldmatrix_b(B0, B_region, ki + 1)
                            mps_emitter.mma(A1, B1, C_ct)
                        mps_emitter.mma(A0, B0, C_ct)

                    mps_emitter.simd_store(C_ct, C_buf)

                return _Simplify(_gemm_with_c_writeback_db, inline_let=True)
            else:

                @T.prim_func
                def _gemm_with_c_writeback() -> None:
                    A_local = T.alloc_local((warp_rows * a_tile_elems * inner_k_steps), in_dtype, scope="metal.cooperative_tensor")
                    B_local = T.alloc_local((warp_cols * b_tile_elems * inner_k_steps), in_dtype, scope="metal.cooperative_tensor")
                    C_ct = T.alloc_local((num_simd_c * c_tile_elems), accum_dtype, scope="metal.cooperative_tensor")
                    if clear_accum:
                        for _i in T.serial(num_simd_c):
                            T.cooperative_tensor_fill(C_ct.data, _i, T.cast(0, accum_dtype), micro_size_x, micro_size_y)
                    else:
                        mps_emitter.simd_load(C_ct, C_buf)
                    for k_outer in T.serial(0, (block_K // (micro_size_k * inner_k_steps))):
                        for k_inner in T.serial(0, inner_k_steps):
                            ki = k_outer * inner_k_steps + k_inner
                            mps_emitter.ldmatrix_a(A_local, A_region, ki, k_inner)
                            mps_emitter.ldmatrix_b(B_local, B_region, ki, k_inner)
                        for k_inner in T.serial(0, inner_k_steps):
                            mps_emitter.mma(A_local, B_local, C_ct, k_inner)

                    mps_emitter.simd_store(C_ct, C_buf)

                return _Simplify(_gemm_with_c_writeback, inline_let=True)
