import tilelang.language as T
from tvm import tir
from tvm import DataType

class MPSIntrinEmitter:
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

        # Metal simdgroup matrix size
        self.micro_size_x = 8
        self.micro_size_y = 8
        self.micro_size_k = 8

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
