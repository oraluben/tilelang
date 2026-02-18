from __future__ import annotations
import tilelang.language as T
from tvm import tir
from tvm.ir import Range
from tvm.tir import PrimExpr, IndexMap, Buffer, Var, BufferRegion, BufferLoad
from tilelang.utils import is_fragment
from tilelang.language.utils import get_buffer_region_from_load
from .mma_metal_layout import shared_8x8_to_mma_a_32x2_layout


def metal_store_index_map(thread_id, local_id):
    """Reverse mapping: (thread_id, local_id) -> (row, col) for 8x8 Metal simdgroup layout.

    Uses modular arithmetic instead of bitwise ops so that TVM's iter-map
    analysis can parse the expression (required for IndexMap.inverse()).
    """
    r0 = (thread_id // 2) % 2
    r1 = (thread_id // 4) % 2
    r2 = (thread_id // 16) % 2
    c1 = thread_id % 2
    c2 = (thread_id // 8) % 2
    row = r0 + 2 * r1 + 4 * r2
    col = local_id + 2 * c1 + 4 * c2
    return row, col


class MetalSimdgroupIntrinEmitter:
    """
    Emitter for Metal simdgroup-based matrix multiply intrinsics.

    Metal uses simdgroup_matrix<T, 8, 8> with 32 threads per simdgroup.
    Each thread holds 2 elements per 8x8 tile, mapped via Z-order (Morton) curve.

    Since no hardware MMA intrinsic is available in TIR, this emitter generates
    software-emulated matrix multiply by having each thread compute its assigned
    output elements via direct shared-memory reads.
    """

    M_DIM = 8
    N_DIM = 8
    WARP_SIZE = 32

    # Represent the thread binding in the form of (tx, warp_n, warp_m)
    is_m_first: bool = False

    def __init__(
        self,
        a_dtype: str = T.float16,
        b_dtype: str = T.float16,
        accum_dtype: str = T.float32,
        a_transposed: bool = False,
        b_transposed: bool = False,
        block_row_warps: int = 2,
        block_col_warps: int = 2,
        warp_row_tiles: int = 8,
        warp_col_tiles: int = 8,
        chunk: int = 8,
        is_m_first: bool | None = False,
        thread_var: Var | None = None,
    ):
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.accum_dtype = accum_dtype
        self.a_transposed = a_transposed
        self.b_transposed = b_transposed
        self.block_row_warps = block_row_warps
        self.block_col_warps = block_col_warps
        self.warp_row_tiles = warp_row_tiles
        self.warp_col_tiles = warp_col_tiles
        self.chunk = chunk

        # Metal simdgroup uses 8x8 tiles, k_dim = 8
        self.k_dim = 8
        self.micro_size_x = self.M_DIM  # 8
        self.micro_size_y = self.N_DIM  # 8
        self.micro_size_k = self.k_dim  # 8

        # Each thread holds 2 elements per 8x8 tile
        self.local_size_out = (self.M_DIM * self.N_DIM) // self.WARP_SIZE  # 2

        # Number of micro tiles per warp
        self.warp_rows = warp_row_tiles // self.M_DIM
        self.warp_cols = warp_col_tiles // self.N_DIM

        self.threads = self.WARP_SIZE * (block_row_warps * block_col_warps)

        if is_m_first is not None:
            self.is_m_first = is_m_first

        self.thread_var = thread_var

    def get_thread_binding(self):
        if self.thread_var is None:
            current_frame = T.KernelLaunchFrame.Current()
            assert current_frame is not None, "Must be called in a T.Kernel Frame"
            return current_frame.get_thread_binding()
        else:
            return self.thread_var

    def extract_thread_binding(self, thread_id, is_m_first=None) -> tuple[PrimExpr, PrimExpr, PrimExpr]:
        WARP_SIZE = self.WARP_SIZE
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps

        if is_m_first is None:
            is_m_first = self.is_m_first

        if is_m_first:
            lane_id, warp_n, warp_m = (
                thread_id % WARP_SIZE,
                (thread_id // WARP_SIZE) % block_col_warps,
                (thread_id // (WARP_SIZE * block_col_warps)) % block_row_warps,
            )
            return lane_id, warp_n, warp_m
        else:
            lane_id, warp_m, warp_n = (
                thread_id % WARP_SIZE,
                (thread_id // WARP_SIZE) % block_row_warps,
                (thread_id // (WARP_SIZE * block_row_warps)) % block_col_warps,
            )
            return lane_id, warp_n, warp_m

    def get_store_index_map(self, inverse: bool = False) -> IndexMap:
        warp_size, local_size_c = self.WARP_SIZE, self.local_size_out
        index_map = IndexMap.from_func(metal_store_index_map, index_dtype=T.int32)
        if not inverse:
            return index_map
        inverse_index_map = index_map.inverse([warp_size, local_size_c])
        return inverse_index_map

    def gemm_ss(
        self,
        A_region: BufferRegion,
        B_region: BufferRegion,
        C_buf: Buffer,
        clear_accum: PrimExpr,
    ):
        """Generate software-emulated GEMM for shared-shared case.

        Each thread computes its assigned output elements by reading directly
        from shared memory A and B, accumulating into local fragment C.
        """
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        warp_row_tiles = self.warp_row_tiles
        warp_col_tiles = self.warp_col_tiles
        local_size_out = self.local_size_out
        micro_size_x = self.micro_size_x
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        block_K = self.chunk
        a_transposed = self.a_transposed
        b_transposed = self.b_transposed

        thread_binding = self.get_thread_binding()

        A_buf = A_region.buffer
        B_buf = B_region.buffer
        A_base0 = A_region.region[-2].min
        A_base1 = A_region.region[-1].min
        B_base0 = B_region.region[-2].min
        B_base1 = B_region.region[-1].min

        # Check if C_buf is 1D (flat local buffer) or 2D (fragment buffer)
        c_is_flat = len(C_buf.shape) == 1

        @T.macro
        def _gemm_ss_impl(C_buf, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)

            if clear_accum:
                T.clear(C_buf)

            for ki in T.serial(block_K // micro_size_k):
                for i, j in T.grid(warp_rows, warp_cols):
                    for local_id in T.serial(local_size_out):
                        c_row, c_col = T.meta_var(metal_store_index_map(tx, local_id))
                        for k in T.serial(micro_size_k):
                            a_row = warp_m * warp_row_tiles + i * micro_size_x + c_row
                            b_col = warp_n * warp_col_tiles + j * micro_size_y + c_col
                            k_idx = ki * micro_size_k + k

                            if a_transposed:
                                a_val = A_buf[A_base0 + k_idx, A_base1 + a_row]
                            else:
                                a_val = A_buf[A_base0 + a_row, A_base1 + k_idx]

                            if b_transposed:
                                b_val = B_buf[B_base0 + b_col, B_base1 + k_idx]
                            else:
                                b_val = B_buf[B_base0 + k_idx, B_base1 + b_col]

                            if c_is_flat:
                                C_buf[i * warp_cols * local_size_out + j * local_size_out + local_id] += (
                                    a_val * b_val
                                )
                            else:
                                # 2D fragment indexing: the fragment layout maps
                                # these coordinates to the per-thread local storage
                                C_buf[a_row, b_col] += a_val * b_val

        return _gemm_ss_impl(C_buf, thread_binding)

    def make_metal_store_layout(self, local_buf: Buffer) -> T.Fragment:
        """Create fragment layout for Metal simdgroup output (C matrix).

        Uses the forward map (row, col) -> (thread_id, local_id) from
        shared_8x8_to_mma_a_32x2_layout directly, avoiding the need to
        invert the reverse map (which uses bit-interleaving that TVM's
        iter-map analysis cannot handle).
        """
        shape = local_buf.shape
        assert is_fragment(local_buf), f"local_buf {local_buf} must be a fragment, but got {local_buf.scope()}"

        # Forward map: (row, col) -> (thread_id, local_id) for 8x8 simdgroup tile
        shared_to_fragment_layout = IndexMap.from_func(shared_8x8_to_mma_a_32x2_layout, index_dtype=T.int32)

        micro_size_x, micro_size_y = self.micro_size_x, self.micro_size_y
        local_size_out = self.local_size_out
        block_row_warps, block_col_warps = self.block_row_warps, self.block_col_warps
        warp_rows, warp_cols = self.warp_rows, self.warp_cols
        warp_size = self.WARP_SIZE
        is_m_first = self.is_m_first

        def forward_thread(i: int, j: int) -> int:
            block_i, block_j = (i // micro_size_x) // warp_rows, (j // micro_size_y) // warp_cols
            mma_i, mma_j = i % micro_size_x, j % micro_size_y
            lane_id, _ = shared_to_fragment_layout.map_indices([mma_i, mma_j])
            if is_m_first:
                thread_id = block_i * (block_col_warps * warp_size) + block_j * warp_size + lane_id
            else:
                thread_id = block_j * (block_row_warps * warp_size) + block_i * warp_size + lane_id
            return thread_id

        def forward_index(i: int, j: int) -> int:
            warp_i, warp_j = (i // micro_size_x) % warp_rows, (j // micro_size_y) % warp_cols
            mma_i, mma_j = i % micro_size_x, j % micro_size_y
            _, local_id = shared_to_fragment_layout.map_indices([mma_i, mma_j])
            return warp_i * (warp_cols * local_size_out) + warp_j * local_size_out + local_id

        return T.Fragment(
            shape,
            forward_thread_fn=forward_thread,
            forward_index_fn=forward_index,
        )

    @staticmethod
    def _legalize_to_buffer_region(obj: Buffer | BufferLoad | BufferRegion) -> BufferRegion:
        if isinstance(obj, BufferRegion):
            return obj
        if isinstance(obj, Buffer):
            mins = [tir.IntImm("int32", 0) for _ in obj.shape]
            ranges = [Range.from_min_extent(m, e) for m, e in zip(mins, obj.shape)]
            return BufferRegion(obj, ranges)
        if isinstance(obj, BufferLoad):
            region = get_buffer_region_from_load(obj)
            if region is not None:
                return region
            mins = [idx for idx in obj.indices]
            ones = [tir.IntImm("int32", 1) for _ in obj.indices]
            ranges = [Range.from_min_extent(m, e) for m, e in zip(mins, ones)]
            return BufferRegion(obj.buffer, ranges)
        raise ValueError(f"Unsupported argument type for BufferRegion: {type(obj)}")
