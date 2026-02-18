import tilelang.language as T
from tvm import tir
from tvm import DataType
from tvm.tir import Buffer, BufferRegion


class MPSIntrinEmitter:

    WARP_SIZE = 32

    def __init__(
        self,
        a_dtype: str = "float16",
        b_dtype: str = "float16",
        accum_dtype: str = "float32",
        a_transposed: bool = False,
        b_transposed: bool = False,
        block_row_warps: int = 1,
        block_col_warps: int = 1,
        warp_row_tiles: int = 8,
        warp_col_tiles: int = 8,
        chunk: int = 32,
        thread_var: tir.Var | None = None,
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
        self.thread_var = thread_var

        # Metal simdgroup matrix size (always 8x8)
        self.micro_size_x = 8
        self.micro_size_y = 8
        self.micro_size_k = 8

        # Number of 8x8 tiles per warp
        self.warp_rows = warp_row_tiles // self.micro_size_x
        self.warp_cols = warp_col_tiles // self.micro_size_y

        # Elements per thread in each local tile (64 elements per 8x8 matrix / 32 threads)
        self.local_size_a = 2  # 64 / 32
        self.local_size_b = 2  # 64 / 32

    def get_thread_binding(self):
        if self.thread_var is None:
            current_frame = T.KernelLaunchFrame.Current()
            assert current_frame is not None, "Must be called in a T.Kernel Frame"
            return current_frame.get_thread_binding()
        else:
            return self.thread_var

    def _get_warp_indices(self):
        thread_binding = self.get_thread_binding()
        WARP_SIZE = self.WARP_SIZE
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps

        warp_m = (thread_binding // WARP_SIZE) % block_row_warps
        warp_n = (thread_binding // (WARP_SIZE * block_row_warps)) % block_col_warps
        return warp_m, warp_n

    def ldmatrix_a(self, A_local_buf, A_shared_buf: Buffer | BufferRegion, ki):
        warp_rows = self.warp_rows
        micro_size_x = self.micro_size_x
        micro_size_k = self.micro_size_k
        a_transposed = self.a_transposed

        warp_m, _ = self._get_warp_indices()

        if hasattr(A_shared_buf, "buffer"):
            buffer = A_shared_buf.buffer
            offset_m = A_shared_buf.region[-2].min
            offset_k = A_shared_buf.region[-1].min
            stride = buffer.shape[-1]
        else:
            buffer = A_shared_buf
            offset_m = 0
            offset_k = 0
            stride = buffer.shape[-1]

        @T.macro
        def _warp_ldmatrix_a(A_local_buf, buffer, offset_m, offset_k, stride, warp_m, ki):
            for i in T.serial(warp_rows):
                if a_transposed:
                    row_idx = offset_k + ki * micro_size_k
                    col_idx = offset_m + warp_m * (self.warp_row_tiles) + i * micro_size_x
                else:
                    row_idx = offset_m + warp_m * (self.warp_row_tiles) + i * micro_size_x
                    col_idx = offset_k + ki * micro_size_k

                ptr = T.access_ptr(buffer[row_idx, col_idx], "r")

                T.simdgroup_load(
                    A_local_buf.data,
                    i,
                    ptr,
                    stride,
                    micro_size_x,
                    micro_size_k,
                    T.bool(a_transposed),
                )

        return _warp_ldmatrix_a(A_local_buf, buffer, offset_m, offset_k, stride, warp_m, ki)

    def ldmatrix_b(self, B_local_buf, B_shared_buf: Buffer | BufferRegion, ki):
        warp_cols = self.warp_cols
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        b_transposed = self.b_transposed

        _, warp_n = self._get_warp_indices()

        if hasattr(B_shared_buf, "buffer"):
            buffer = B_shared_buf.buffer
            offset_k = B_shared_buf.region[-2].min
            offset_n = B_shared_buf.region[-1].min
            stride = buffer.shape[-1]
        else:
            buffer = B_shared_buf
            offset_k = 0
            offset_n = 0
            stride = buffer.shape[-1]

        @T.macro
        def _warp_ldmatrix_b(B_local_buf, buffer, offset_k, offset_n, stride, warp_n, ki):
            for j in T.serial(warp_cols):
                if b_transposed:
                    row_idx = offset_n + warp_n * (self.warp_col_tiles) + j * micro_size_y
                    col_idx = offset_k + ki * micro_size_k
                else:
                    row_idx = offset_k + ki * micro_size_k
                    col_idx = offset_n + warp_n * (self.warp_col_tiles) + j * micro_size_y

                ptr = T.access_ptr(buffer[row_idx, col_idx], "r")

                T.simdgroup_load(
                    B_local_buf.data,
                    j,
                    ptr,
                    stride,
                    micro_size_k,
                    micro_size_y,
                    T.bool(b_transposed),
                )

        return _warp_ldmatrix_b(B_local_buf, buffer, offset_k, offset_n, stride, warp_n, ki)

    def mma(self, A_local_buf, B_local_buf, C_local_buf, ki=0):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols

        @T.macro
        def _warp_mma(A_local_buf, B_local_buf, C_local_buf):
            for i, j in T.grid(warp_rows, warp_cols):
                T.simdgroup_multiply_accumulate(
                    C_local_buf.data,
                    i * warp_cols + j,
                    A_local_buf.data,
                    i,
                    B_local_buf.data,
                    j,
                    C_local_buf.data,
                    i * warp_cols + j,
                )

        return _warp_mma(A_local_buf, B_local_buf, C_local_buf)

    def make_mma_store_layout(self, buffer):
        from tilelang.intrinsics.mma_metal_layout import metal_store_layout
        from tilelang.utils import is_fragment

        shape = buffer.shape

        micro_size_x = self.micro_size_x
        micro_size_y = self.micro_size_y
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        warp_size = self.WARP_SIZE
        local_size = (micro_size_x * micro_size_y) // self.WARP_SIZE

        def forward_thread(i: int, j: int) -> int:
            # Determine which warp (block_i, block_j) and which tile within the warp
            block_i = (i // micro_size_x) // warp_rows
            block_j = (j // micro_size_y) // warp_cols
            # Position within the 8x8 tile
            mma_i = i % micro_size_x
            mma_j = j % micro_size_y
            lane_id, _ = metal_store_layout.map_indices([mma_i, mma_j])
            # Thread binding: [warp_size, block_row_warps, block_col_warps]
            thread_id = block_j * (block_row_warps * warp_size) + block_i * warp_size + lane_id
            return thread_id

        def forward_index(i: int, j: int) -> int:
            # Which tile within the warp
            warp_i = (i // micro_size_x) % warp_rows
            warp_j = (j // micro_size_y) % warp_cols
            # Position within the 8x8 tile
            mma_i = i % micro_size_x
            mma_j = j % micro_size_y
            _, local_id = metal_store_layout.map_indices([mma_i, mma_j])
            return warp_i * (warp_cols * local_size) + warp_j * local_size + local_id

        return T.Fragment(
            shape,
            forward_thread_fn=forward_thread,
            forward_index_fn=forward_index,
        )

    def stmatrix_c(self, C_local_buf, C_shared_buf):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        micro_size_x = self.micro_size_x
        micro_size_y = self.micro_size_y
        
        warp_m, warp_n = self._get_warp_indices()
        
        # Handle BufferRegion or Buffer for C_shared_buf
        if hasattr(C_shared_buf, "buffer"):
            buffer = C_shared_buf.buffer
            offset_m = C_shared_buf.region[0].min
            offset_n = C_shared_buf.region[1].min
            stride = buffer.shape[-1]
        else:
            buffer = C_shared_buf
            offset_m = 0
            offset_n = 0
            stride = buffer.shape[-1]
            
        @T.macro
        def _warp_stmatrix_c(C_local_buf, buffer, offset_m, offset_n, stride, warp_m, warp_n):
            for i, j in T.grid(warp_rows, warp_cols):
                row_in_block = warp_m * (self.warp_row_tiles * self.micro_size_x) + i * self.micro_size_x
                col_in_block = warp_n * (self.warp_col_tiles * self.micro_size_y) + j * self.micro_size_y
                
                row_idx = row_in_block + offset_m
                col_idx = col_in_block + offset_n
                
                index_c = i * warp_cols + j
                
                ptr = T.access_ptr(buffer[row_idx, col_idx], "w")
                
                T.simdgroup_store(
                    C_local_buf.data,
                    index_c,
                    ptr,
                    stride,
                    micro_size_x,
                    micro_size_y,
                    T.bool(False)
                )
        
        return _warp_stmatrix_c(C_local_buf, buffer, offset_m, offset_n, stride, warp_m, warp_n)

    def ldmatrix_c(self, C_local_buf, C_shared_buf):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        micro_size_x = self.micro_size_x
        micro_size_y = self.micro_size_y
        
        warp_m, warp_n = self._get_warp_indices()
        
        # Handle BufferRegion or Buffer for C_shared_buf
        if hasattr(C_shared_buf, "buffer"):
            buffer = C_shared_buf.buffer
            offset_m = C_shared_buf.region[0].min
            offset_n = C_shared_buf.region[1].min
            stride = buffer.shape[-1]
        else:
            buffer = C_shared_buf
            offset_m = 0
            offset_n = 0
            stride = buffer.shape[-1]
            
        @T.macro
        def _warp_ldmatrix_c(C_local_buf, buffer, offset_m, offset_n, stride, warp_m, warp_n):
            for i, j in T.grid(warp_rows, warp_cols):
                row_in_block = warp_m * (self.warp_row_tiles * self.micro_size_x) + i * self.micro_size_x
                col_in_block = warp_n * (self.warp_col_tiles * self.micro_size_y) + j * self.micro_size_y
                
                row_idx = row_in_block + offset_m
                col_idx = col_in_block + offset_n
                
                index_c = i * warp_cols + j
                
                ptr = T.access_ptr(buffer[row_idx, col_idx], "r")
                
                T.simdgroup_load(
                    C_local_buf.data,
                    index_c,
                    ptr,
                    stride,
                    micro_size_x,
                    micro_size_y,
                    T.bool(False)
                )
        
        return _warp_ldmatrix_c(C_local_buf, buffer, offset_m, offset_n, stride, warp_m, warp_n)

    def simd_to_fragment(self, C_simd_buf, C_frag_buf, C_scratch_buf):
        """Store simdgroup matrices to a fragment buffer via shared memory scratch.

        The flow is:
        1. simdgroup_store writes each 8x8 simdgroup tile to shared memory
        2. Each thread reads its 2 elements per tile from shared memory
           into the fragment buffer using the metal_store_fragment layout

        Args:
            C_simd_buf: The simdgroup matrix buffer (metal.simdgroup scope)
            C_frag_buf: The fragment buffer (local.fragment scope)
            C_scratch_buf: A shared memory scratch buffer for intermediary storage
        """
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        micro_size_x = self.micro_size_x
        micro_size_y = self.micro_size_y
        WARP_SIZE = self.WARP_SIZE

        warp_m, warp_n = self._get_warp_indices()

        stride = C_scratch_buf.shape[-1]
        thread_binding = self.get_thread_binding()

        @T.macro
        def _simd_to_fragment(C_simd_buf, C_frag_buf, C_scratch_buf, stride,
                              warp_m, warp_n, thread_binding):
            # Step 1: Store all simdgroup tiles to shared memory
            for i, j in T.grid(warp_rows, warp_cols):
                row_in_block = warp_m * (self.warp_row_tiles) + i * micro_size_x
                col_in_block = warp_n * (self.warp_col_tiles) + j * micro_size_y

                index_c = i * warp_cols + j

                ptr = T.access_ptr(C_scratch_buf[row_in_block, col_in_block], "w")

                T.simdgroup_store(
                    C_simd_buf.data,
                    index_c,
                    ptr,
                    stride,
                    micro_size_x,
                    micro_size_y,
                    T.bool(False),
                )

            # Step 2: Each thread reads its elements from shared memory
            # Inverse of the Z-order (Morton) layout:
            # Given lane_id, compute which (row, col) in the 8x8 tile this thread owns
            lane_id = thread_binding % WARP_SIZE
            r0 = (lane_id >> 1) & 1
            r1 = (lane_id >> 2) & 1
            r2 = (lane_id >> 4) & 1
            c1 = lane_id & 1
            c2 = (lane_id >> 3) & 1
            tile_row = r0 + 2 * r1 + 4 * r2
            tile_col_base = 2 * c1 + 4 * c2

            for i, j in T.grid(warp_rows, warp_cols):
                row_base = warp_m * (self.warp_row_tiles) + i * micro_size_x
                col_base = warp_n * (self.warp_col_tiles) + j * micro_size_y
                for local_id in T.serial(2):
                    C_frag_buf[
                        row_base + tile_row,
                        col_base + tile_col_base + local_id
                    ] = C_scratch_buf[
                        row_base + tile_row,
                        col_base + tile_col_base + local_id]

        return _simd_to_fragment(C_simd_buf, C_frag_buf, C_scratch_buf, stride,
                                 warp_m, warp_n, thread_binding)

    def fragment_to_simd(self, C_frag_buf, C_simd_buf, C_scratch_buf):
        """Load fragment buffer data into simdgroup matrices via shared memory scratch.

        The flow is:
        1. Each thread writes its fragment elements to shared memory
        2. simdgroup_load reads from shared memory into simdgroup matrices

        Args:
            C_frag_buf: The fragment buffer (local.fragment scope)
            C_simd_buf: The simdgroup matrix buffer (metal.simdgroup scope)
            C_scratch_buf: A shared memory scratch buffer for intermediary storage
        """
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        micro_size_x = self.micro_size_x
        micro_size_y = self.micro_size_y
        WARP_SIZE = self.WARP_SIZE

        warp_m, warp_n = self._get_warp_indices()

        stride = C_scratch_buf.shape[-1]
        thread_binding = self.get_thread_binding()

        @T.macro
        def _fragment_to_simd(C_frag_buf, C_simd_buf, C_scratch_buf, stride,
                              warp_m, warp_n, thread_binding):
            # Step 1: Each thread writes its fragment elements to shared memory
            lane_id = thread_binding % WARP_SIZE
            r0 = (lane_id >> 1) & 1
            r1 = (lane_id >> 2) & 1
            r2 = (lane_id >> 4) & 1
            c1 = lane_id & 1
            c2 = (lane_id >> 3) & 1
            tile_row = r0 + 2 * r1 + 4 * r2
            tile_col_base = 2 * c1 + 4 * c2

            for i, j in T.grid(warp_rows, warp_cols):
                row_base = warp_m * (self.warp_row_tiles) + i * micro_size_x
                col_base = warp_n * (self.warp_col_tiles) + j * micro_size_y
                for local_id in T.serial(2):
                    C_scratch_buf[
                        row_base + tile_row,
                        col_base + tile_col_base + local_id
                    ] = C_frag_buf[
                        row_base + tile_row,
                        col_base + tile_col_base + local_id]

            # Step 2: simdgroup_load from shared memory
            for i, j in T.grid(warp_rows, warp_cols):
                row_in_block = warp_m * (self.warp_row_tiles) + i * micro_size_x
                col_in_block = warp_n * (self.warp_col_tiles) + j * micro_size_y

                index_c = i * warp_cols + j

                ptr = T.access_ptr(C_scratch_buf[row_in_block, col_in_block], "r")

                T.simdgroup_load(
                    C_simd_buf.data,
                    index_c,
                    ptr,
                    stride,
                    micro_size_x,
                    micro_size_y,
                    T.bool(False),
                )

        return _fragment_to_simd(C_frag_buf, C_simd_buf, C_scratch_buf, stride,
                                 warp_m, warp_n, thread_binding)
