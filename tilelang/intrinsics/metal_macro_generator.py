import tilelang.language as T
from typing import Union, Tuple, Optional, Literal, Callable
from tilelang.common import TransformKind
from tvm import DataType
from tvm.tir import PrimExpr, IndexMap, Buffer
from tvm.runtime import convert
from .utils import (
    mma_store_index_map,
    get_ldmatrix_offset,
)
from tilelang.utils import is_fragment

lift = convert


class MetalSIMDGroupIntrinEmitter(object):
    """
    Metal SIMD Group Intrinsics Emitter.
    To eliminate Python syntax within TIR Macro.
    Based on Metal's SIMD group matrix multiplication operations.
    """

    M_DIM = 16
    N_DIM = 16
    WARP_SIZE = 32  # Metal thread execution width
    dtype_abbrv = {
        "float16": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "int8": "int8",
        "int32": "int32",
        "float8_e4m3": "e4m3",
        "float8_e5m2": "e5m2",
    }

    # k_pack represents the number of elements in a vectorized instruction
    # For Metal SIMD groups, this depends on the data type and matrix size
    k_pack = 1
    # Represent the thread binding in the form of (tx, warp_n, warp_m)
    is_m_first = False

    def __init__(
        self,
        a_dtype: str = "float16",
        b_dtype: str = "float16",
        accum_dtype: str = "float16",
        a_transposed: bool = False,
        b_transposed: bool = False,
        block_row_warps: int = 2,
        block_col_warps: int = 2,
        warp_row_tiles: int = 8,
        warp_col_tiles: int = 8,
        chunk: int = 16,
        reduce_k: int = 1,
        num_elems_per_byte: int = 1,
        k_pack: Optional[int] = None,
        is_m_first: Optional[bool] = False,
    ):
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.accum_dtype = accum_dtype
        self.a_transposed = a_transposed
        self.b_transposed = b_transposed
        # Hint Information
        self.block_row_warps = block_row_warps
        self.block_col_warps = block_col_warps
        self.warp_row_tiles = warp_row_tiles
        self.warp_col_tiles = warp_col_tiles
        self.chunk = chunk
        self._initialize_k_dim(a_dtype)
        self._initialize_abbrev(a_dtype, b_dtype, accum_dtype)
        self._initialize_local_size(self.M_DIM, self.N_DIM, self.k_dim, self.WARP_SIZE)
        self._initialize_metal_simdgroup_prefix(self.k_dim)
        self._initialize_micro_size(self.M_DIM, self.N_DIM, self.k_dim)
        self._initialize_k_pack(k_pack)
        self._initialize_is_m_first(is_m_first)

        self.warp_rows = warp_row_tiles // self.micro_size_x
        self.warp_cols = warp_col_tiles // self.micro_size_y
        self.reduce_k = reduce_k
        self.threads = self.WARP_SIZE * (block_row_warps * block_col_warps) * reduce_k
        self.num_elems_per_byte = num_elems_per_byte

        if self.warp_rows == 0 or self.warp_cols == 0:
            raise ValueError(
                f"Invalid threads configuration for this tile shape, {self.warp_rows} x {self.warp_cols} with threads {self.threads}"
            )

    def _initialize_k_dim(self, a_dtype="float16"):
        if isinstance(a_dtype, str):
            if a_dtype in ["float8_e4m3", "float8_e5m2"]:
                self.k_dim = 32
                return
            a_dtype = DataType(a_dtype)

        if a_dtype.bits == 32:
            self.k_dim = 4
        elif a_dtype.bits in {16, 8}:
            self.k_dim = 16
        else:
            raise ValueError(f"Unsupported a_dtype = {a_dtype}")

    def _initialize_local_size(self, m_dim=16, n_dim=16, k_dim=16, warp_size=32):
        self.local_size_a = (m_dim * k_dim) // warp_size
        self.local_size_b = (n_dim * k_dim) // warp_size
        self.local_size_out = (m_dim * n_dim) // warp_size

    def _initialize_abbrev(self, a_dtype, b_dtype, accum_dtype):
        self.a_dtype_abbrv = self.dtype_abbrv[a_dtype]
        self.b_dtype_abbrv = self.dtype_abbrv[b_dtype]
        self.accum_dtype_abbrv = self.dtype_abbrv[accum_dtype]

    def _initialize_metal_simdgroup_prefix(self, k_dim=16):
        in_dtype, out_dtype = self.a_dtype, self.accum_dtype
        M_DIM, N_DIM = self.M_DIM, self.N_DIM
        out_dtype_abbrv = {
            "float16": "f16",
            "float32": "f32",
            "int8": "i8",
            "int32": "i32"
        }[out_dtype]

        in_dtype_abbrv = {
            "float16": "f16",
            "float32": "f32",
            "int8": "i8",
            "int32": "i32",
            "float8_e4m3": "fp8",
            "float8_e5m2": "fp8",
        }[in_dtype]

        # Metal SIMD group operation naming convention
        if in_dtype_abbrv == "fp8":
            self.simdgroup_suffix = f"{out_dtype_abbrv}_{M_DIM}x{N_DIM}x{k_dim}_fp8_fp8"
        else:
            self.simdgroup_suffix = f"{out_dtype_abbrv}_{M_DIM}x{N_DIM}x{k_dim}{in_dtype_abbrv}"

    def _initialize_micro_size(self, m_dim=16, n_dim=16, k_dim=16):
        self.micro_size_x = m_dim
        self.micro_size_y = n_dim
        self.micro_size_k = k_dim

    def _initialize_k_pack(self, k_pack: Optional[int] = None):
        if k_pack is not None:
            self.k_pack = k_pack

    def _initialize_is_m_first(self, is_m_first: Optional[bool] = False):
        if is_m_first is not None:
            self.is_m_first = is_m_first

    def get_ldsimdgroup_index_map(self, is_b=False):
        """
        Get index map for loading data into SIMD groups.
        """
        from .metal_layout import (
            shared_16x4_to_local_64x1_layout_A,
            shared_4x16_to_local_64x1_layout_B,
            shared_16x16_to_local_64x4_layout_A,
            shared_16x16_to_local_64x4_layout_B,
            shared_16x32_to_local_64x8_layout_A,
            shared_16x32_to_local_64x8_layout_B,
            shared_16x64_to_local_64x16_layout_A,
            shared_16x64_to_local_64x16_layout_B,
            thread_id_shared_access_64x1_to_16x4_layout_A,
            thread_id_shared_access_64x1_to_4x16_layout_B,
            thread_id_shared_access_64x4_to_16x16_layout_A,
            thread_id_shared_access_64x4_to_16x16_layout_B,
            thread_id_shared_access_64x8_to_16x32_layout_A,
            thread_id_shared_access_64x8_to_16x32_layout_B,
            thread_id_shared_access_64x16_to_16x64_layout_A,
            thread_id_shared_access_64x16_to_16x64_layout_B,
        )

        k_dim = self.k_dim * self.k_pack
        transposed = self.a_transposed if not is_b else self.b_transposed
        if k_dim == 4:
            index_map = shared_16x4_to_local_64x1_layout_A
            reverse_index_map = thread_id_shared_access_64x1_to_16x4_layout_A
            if is_b:
                index_map = shared_16x4_to_local_64x1_layout_A if transposed else shared_4x16_to_local_64x1_layout_B
                reverse_index_map = thread_id_shared_access_64x1_to_16x4_layout_A if transposed else thread_id_shared_access_64x1_to_4x16_layout_B
        elif k_dim == 16:
            index_map = shared_16x16_to_local_64x4_layout_B if transposed else shared_16x16_to_local_64x4_layout_A
            reverse_index_map = thread_id_shared_access_64x4_to_16x16_layout_B if transposed else thread_id_shared_access_64x4_to_16x16_layout_A

            if is_b:
                index_map = shared_16x16_to_local_64x4_layout_A if transposed else shared_16x16_to_local_64x4_layout_B
                reverse_index_map = thread_id_shared_access_64x4_to_16x16_layout_A if transposed else thread_id_shared_access_64x4_to_16x16_layout_B
        elif k_dim == 32:
            index_map = shared_16x32_to_local_64x8_layout_B if transposed else shared_16x32_to_local_64x8_layout_A
            reverse_index_map = thread_id_shared_access_64x8_to_16x32_layout_B if transposed else thread_id_shared_access_64x8_to_16x32_layout_A

            if is_b:
                index_map = shared_16x32_to_local_64x8_layout_A if transposed else shared_16x32_to_local_64x8_layout_B
                reverse_index_map = thread_id_shared_access_64x8_to_16x32_layout_A if transposed else thread_id_shared_access_64x8_to_16x32_layout_B
        elif k_dim == 64:
            index_map = shared_16x64_to_local_64x16_layout_B if transposed else shared_16x64_to_local_64x16_layout_A
            reverse_index_map = thread_id_shared_access_64x16_to_16x64_layout_B if transposed else thread_id_shared_access_64x16_to_16x64_layout_A

            if is_b:
                index_map = shared_16x64_to_local_64x16_layout_A if transposed else shared_16x64_to_local_64x16_layout_B
                reverse_index_map = thread_id_shared_access_64x16_to_16x64_layout_A if transposed else thread_id_shared_access_64x16_to_16x64_layout_B
        else:
            raise ValueError("k_dim must be 4 or 16 or 32 or 64 currently")

        return index_map, reverse_index_map

    def extract_thread_binding(self,
                               thread_id,
                               is_m_first=None) -> Tuple[PrimExpr, PrimExpr, PrimExpr]:
        '''
            is_m_first: True if the thread binding is in the form of (tx, warp_n, warp_m)
            which represents [warp_size, block_row_warps (split n), block_col_warps (split m)]
            Otherwise, it is in the form of [warp_size, block_col_warps (split m), block_row_warps (split n)]
        '''
        WARP_SIZE = self.WARP_SIZE
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps

        # if is_m_first is None, then use the default value
        if is_m_first is None:
            is_m_first = self.is_m_first

        if is_m_first:
            lane_id, warp_n, warp_m = thread_id % WARP_SIZE, (
                thread_id //
                WARP_SIZE) % block_col_warps, (thread_id //
                                               (WARP_SIZE * block_col_warps)) % block_row_warps,
            return lane_id, warp_n, warp_m
        else:
            lane_id, warp_m, warp_n = thread_id % WARP_SIZE, (
                thread_id //
                WARP_SIZE) % block_row_warps, (thread_id //
                                               (WARP_SIZE * block_row_warps)) % block_col_warps,
            return lane_id, warp_n, warp_m

    def ldsimdgroup_a(self, A_local_buf, A_shared_buf, ki, rk=0):
        """
        Load matrix A into SIMD group registers.
        This is the Metal equivalent of CUDA's ldmatrix operation.
        """
        warp_row_tiles = self.warp_row_tiles
        warp_rows = self.warp_rows
        chunk = self.chunk
        micro_size_x = self.micro_size_x
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        k_pack = self.k_pack
        is_transposed = self.a_transposed
        current_frame = T.KernelLaunchFrame.Current()
        thread_binding = current_frame.get_thread_binding()
        _, reverse_index_map = self.get_ldsimdgroup_index_map(is_b=False)

        @T.macro
        def _warp_ldsimdgroup_a(
            A_local_buf,
            A_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, _, warp_m = self.extract_thread_binding(thread_binding)
            if is_transposed:
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (rk * chunk + ki * (k_pack * micro_size_k),
                                warp_m * warp_row_tiles + i * micro_size_x)
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_shared_buf[l + row,
                                                                                         r + col]
            else:
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (warp_m * warp_row_tiles + i * micro_size_x,
                                rk * chunk + ki * (k_pack * micro_size_k))
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_shared_buf[l + row,
                                                                                         r + col]

        return _warp_ldsimdgroup_a(A_local_buf, A_shared_buf, ki, thread_binding, rk)

    def ldsimdgroup_b(self, B_local_buf, B_shared_buf, ki, rk=0):
        """
        Load matrix B into SIMD group registers.
        This is the Metal equivalent of CUDA's ldmatrix operation.
        """
        warp_col_tiles = self.warp_col_tiles
        warp_cols = self.warp_cols
        chunk = self.chunk
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        k_pack = self.k_pack
        is_transposed = self.b_transposed
        current_frame = T.KernelLaunchFrame.Current()
        thread_binding = current_frame.get_thread_binding()
        _, reverse_index_map = self.get_ldsimdgroup_index_map(is_b=True)

        @T.macro
        def _warp_ldsimdgroup_b(
            B_local_buf,
            B_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)

            if is_transposed:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            warp_n * warp_col_tiles + j * micro_size_y,
                            rk * chunk + ki * (k_pack * micro_size_k),
                        )
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_shared_buf[l + row,
                                                                                         r + col]
            else:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            rk * chunk + ki * (k_pack * micro_size_k),
                            warp_n * warp_col_tiles + j * micro_size_y,
                        )
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_shared_buf[l + row,
                                                                                         r + col]

        return _warp_ldsimdgroup_b(B_local_buf, B_shared_buf, ki, thread_binding, rk)

    def simdgroup_mma(self, A_local_buf, B_local_buf, C_local_buf):
        """
        Perform SIMD group matrix multiplication.
        This is the Metal equivalent of CUDA's mma operation.
        """
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        k_pack = self.k_pack
        simdgroup_suffix = self.simdgroup_suffix
        a_dtype, b_dtype, out_dtype = self.a_dtype, self.b_dtype, self.accum_dtype
        compute_a_dtype = a_dtype if local_size_a == 1 else f"{a_dtype}x{local_size_a}"
        compute_b_dtype = b_dtype if local_size_b == 1 else f"{b_dtype}x{local_size_b}"
        compute_out_dtype = out_dtype if local_size_out == 1 else f"{out_dtype}x{local_size_out}"

        @T.macro
        def _warp_mma(A_local_buf, B_local_buf, C_local_buf):
            for kp, i, j in T.grid(k_pack, warp_rows, warp_cols):
                # Metal SIMD group matrix multiplication intrinsic
                T.tvm_metal_simdgroup_mma(
                    simdgroup_suffix,
                    "row",
                    "row",
                    compute_a_dtype,
                    compute_b_dtype,
                    compute_out_dtype,
                    B_local_buf.data,
                    ((j * k_pack + kp) * local_size_b) // local_size_b,
                    A_local_buf.data,
                    ((i * k_pack + kp) * local_size_a) // local_size_a,
                    C_local_buf.data,
                    (i * warp_cols * local_size_out + j * local_size_out) // local_size_out,
                    dtype=compute_out_dtype,
                )

        return _warp_mma(A_local_buf, B_local_buf, C_local_buf)

    def stsimdgroup(self, C_local_buf, C_buf, pid_m=None, pid_n=None):
        """
        Store results from SIMD group to memory.
        This is the Metal equivalent of CUDA's stmatrix operation.
        """
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_out = self.local_size_out
        current_frame = T.KernelLaunchFrame.Current()
        thread_binding = current_frame.get_thread_binding()
        is_global = pid_m is not None and pid_n is not None
        BLOCK_M = block_row_warps * warp_rows
        BLOCK_N = block_col_warps * warp_cols
        M_DIM, N_DIM = self.M_DIM, self.N_DIM
        C_buf_dims = len(C_buf.shape)
        assert C_buf_dims in {2, 4}, "C_buf should be 2D or 4D"

        # STS
        # MMA Store must be in simulated instead of TVM Intrins
        # As TVM Intrins is like a hack that the threadIdx.x should be always
        # equal to the warp_size
        @T.macro
        def _warp_stsimdgroup_shared(C_local_buf, C_buf, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            for i, j in T.grid(warp_rows, warp_cols):
                for local_id in T.vectorized(local_size_out):
                    # Use the Metal-specific store index map
                    from .metal_layout import thread_id_shared_access_64x4_to_16x16_layout_C_n_m
                    row, col = T.meta_var(thread_id_shared_access_64x4_to_16x16_layout_C_n_m(tx, local_id))
                    if C_buf_dims == 2:
                        C_buf[(warp_m * warp_rows + i) * M_DIM + row,
                              (warp_n * warp_cols + j) * N_DIM +
                              col] = C_local_buf[i * (warp_cols * local_size_out) +
                                                 j * local_size_out + local_id]
                    else:
                        C_buf[warp_m * warp_rows + i, warp_n * warp_cols + j, row,
                              col] = C_local_buf[i * warp_cols * local_size_out +
                                                 j * local_size_out + local_id]

        @T.macro
        def _warp_stsimdgroup_global(C_local_buf, C_buf, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            for i, j in T.grid(warp_rows, warp_cols):
                for local_id in T.vectorized(local_size_out):
                    # Use the Metal-specific store index map
                    from .metal_layout import thread_id_shared_access_64x4_to_16x16_layout_C_n_m
                    row, col = T.meta_var(thread_id_shared_access_64x4_to_16x16_layout_C_n_m(tx, local_id))
                    C_buf[(pid_m * BLOCK_M + warp_m * warp_rows + i) * M_DIM + row,
                          (pid_n * BLOCK_N + warp_n * warp_cols + j) * N_DIM +
                          col] = C_local_buf[i * warp_cols * local_size_out + j * local_size_out +
                                             local_id]

        return _warp_stsimdgroup_global(C_local_buf, C_buf,
                                     thread_binding) if is_global else _warp_stsimdgroup_shared(
                                         C_local_buf, C_buf, thread_binding)

    def make_simdgroup_load_layout(self,
                                 local_buf: Buffer,
                                 matrix: Literal["A", "B"] = "A") -> T.Fragment:
        """
        Create a layout function for loading data into SIMD group registers.
        This layout is used in conjunction with `inverse_simdgroup_store_layout` to
        map fragment indices to threads and local indices.

        Parameters
        ----------
        local_buf : tir.Buffer
            The local buffer representing a fragment of a matrix.

        Returns
        -------
        T.Fragment
            A fragment object that describes how threads and indices
            in `local_buf` are laid out.

        Raises
        ------
        AssertionError
            If `local_buf` is not detected to be a fragment buffer.
        """
        from tilelang.utils import is_fragment
        from .metal_layout import (
            shared_16x16_to_local_64x4_layout_A,
            shared_16x16_to_local_64x4_layout_B,
            shared_16x32_to_local_64x8_layout_A,
            shared_16x32_to_local_64x8_layout_B,
            shared_16x64_to_local_64x16_layout_A,
            shared_16x64_to_local_64x16_layout_B,
            thread_id_shared_access_64x4_to_16x16_layout_A,
            thread_id_shared_access_64x4_to_16x16_layout_B,
            thread_id_shared_access_64x8_to_16x32_layout_A,
            thread_id_shared_access_64x8_to_16x32_layout_B,
            thread_id_shared_access_64x16_to_16x64_layout_A,
            thread_id_shared_access_64x16_to_16x64_layout_B,
        )
        assert matrix in ["A", "B"], "matrix should be either A or B"
        dtype = self.a_dtype if matrix == "A" else self.b_dtype
        dtype_bits = DataType(dtype).bits
        transposed = self.a_transposed
        assert transposed is False, "transposed is not supported yet"
        # s represents spatial axis
        # r represents reduction axis
        # sr represents the two dims are spatial + reduction
        # rs represents the two dims are reduction + spatial
        transform_func_sr: Callable = None
        transform_func_rs: Callable = None
        if dtype_bits == 16:
            transform_func_sr = shared_16x16_to_local_64x4_layout_A
            transform_func_rs = thread_id_shared_access_64x4_to_16x16_layout_A
        elif dtype_bits == 8:
            transform_func_sr = shared_16x32_to_local_64x8_layout_A
            transform_func_rs = thread_id_shared_access_64x8_to_16x32_layout_A
        elif dtype_bits == 32:
            transform_func_sr = shared_16x4_to_local_64x1_layout_A
            transform_func_rs = thread_id_shared_access_64x1_to_16x4_layout_A
        else:
            raise ValueError(f"Unsupported dtype {dtype}")
        is_sr_conditions = [False]
        is_sr_conditions.append(matrix == "A" and not transposed)
        is_sr_conditions.append(matrix == "B" and transposed)
        is_sr_axis_order = any(is_sr_conditions)

        transform_func: Callable = transform_func_sr if is_sr_axis_order else transform_func_rs

        assert is_fragment(local_buf), "local_buf must be a fragment, but got {}".format(
            local_buf.scope())

        if matrix == "A":
            micro_size_s, micro_size_r = self.micro_size_x, self.micro_size_k
        else:
            micro_size_r, micro_size_s = self.micro_size_k, self.micro_size_y

        block_row_warps, block_col_warps = (
            self.block_row_warps,
            self.block_col_warps,
        )
        warp_rows, warp_cols = self.warp_rows, self.warp_cols
        warp_s = warp_rows if matrix == "A" else warp_cols
        chunk = self.chunk
        transform_func = transform_func
        inverse_simdgroup_load_layout = IndexMap.from_func(transform_func, index_dtype="int32")

        def forward_thread(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            """
            lane_id, _ = inverse_simdgroup_load_layout.map_indices([i, j])
            return lane_id

        def forward_index(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            """
            _, local_id = inverse_simdgroup_load_layout.map_indices([i, j])
            return local_id

        base_fragment = T.Fragment(
            [micro_size_r, micro_size_s],
            forward_thread_fn=forward_thread,
            forward_index_fn=forward_index,
        )
        warp_fragment = base_fragment.repeat([block_row_warps, 1],
                                             repeat_on_thread=True).replicate(block_col_warps)
        block_fragment = warp_fragment.repeat([warp_s, chunk // micro_size_r],
                                              repeat_on_thread=False,
                                              lower_dim_first=False)
        return block_fragment

    def make_simdgroup_store_layout(self, local_buf: Buffer) -> T.Fragment:
        """
        Create a layout function for storing SIMD group results into a fragment buffer.
        This layout is used in conjunction with `inverse_simdgroup_store_layout` to
        map fragment indices to threads and local indices.

        Parameters
        ----------
        local_buf : tir.Buffer
            The local buffer representing a fragment of a matrix.

        Returns
        -------
        T.Fragment
            A fragment object that describes how threads and indices
            in `local_buf` are laid out.

        Raises
        ------
        AssertionError
            If `local_buf` is not detected to be a fragment buffer.
        """
        from tilelang.utils import is_fragment

        shape = local_buf.shape
        # Use Metal-specific store index map
        from .metal_layout import thread_id_shared_access_64x4_to_16x16_layout_C_n_m
        inverse_simdgroup_store_layout = IndexMap.from_func(thread_id_shared_access_64x4_to_16x16_layout_C_n_m, index_dtype="int32")
        assert is_fragment(local_buf), "local_buf must be a fragment"
        micro_size_x, micro_size_y = self.micro_size_x, self.micro_size_y
        local_size_out = self.local_size_out
        block_row_warps, block_col_warps = self.block_row_warps, self.block_col_warps
        warp_rows, warp_cols = self.warp_rows, self.warp_cols
        warp_size = self.WARP_SIZE
        is_m_first = self.is_m_first

        def forward_thread(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            map them to a thread index according to `inverse_simdgroup_store_layout`.
            """
            # the upper bounds of i and j are block_row_warps * warp_rows * micro_size_x and block_col_warps * warp_cols * micro_size_y
            # the upper bounds of block_row_warps and block_col_warps are warp_rows and warp_cols
            block_i, block_j = (i // micro_size_x) // warp_rows, (j // micro_size_y) // warp_cols
            # upper bounds of mma_i and mma_j are micro_size_x and micro_size_y
            mma_i, mma_j = i % micro_size_x, j % micro_size_y
            lane_id, _ = inverse_simdgroup_store_layout.map_indices([mma_i, mma_j])
            if is_m_first:
                thread_id = block_i * (block_col_warps * warp_cols) + block_j * warp_size + lane_id
            else:
                thread_id = block_j * (block_row_warps * warp_size) + block_i * warp_size + lane_id
            return thread_id

        def forward_index(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            map them to a local index in a single thread according
            to `inverse_simdgroup_store_layout`.
            """
            # the upper bounds of i and j are block_row_warps * warp_rows * micro_size_x and block_col_warps * warp_cols * micro_size_y
            # the upper bounds of warp_i and warp_j are warp_rows and warp_cols
            warp_i, warp_j = (i // micro_size_x) % warp_rows, (j // micro_size_y) % warp_cols
            # upper bounds of mma_i and mma_j are micro_size_x and micro_size_y
            mma_i, mma_j = i % micro_size_x, j % micro_size_y
            _, local_id = inverse_simdgroup_store_layout.map_indices([mma_i, mma_j])
            return warp_i * (warp_cols * local_size_out) + warp_j * local_size_out + local_id

        return T.Fragment(
            shape,
            forward_thread_fn=forward_thread,
            forward_index_fn=forward_index,
        )


class MetalSIMDGroupIntrinEmitterWithLadderTransform(MetalSIMDGroupIntrinEmitter):
    """
    Metal SIMD Group Intrinsics Emitter with Ladder Transform Plugin.
    To eliminate Python syntax within TIR Macro.
    With Ladder Transform Plugin.
    """

    def __init__(
        self,
        a_dtype: str = "float16",
        b_dtype: str = "float16",
        accum_dtype: str = "float16",
        a_transposed: bool = False,
        b_transposed: bool = False,
        block_row_warps: int = 2,
        block_col_warps: int = 2,
        warp_row_tiles: int = 8,
        warp_col_tiles: int = 8,
        chunk: int = 16,
        reduce_k: int = 1,
        num_elems_per_byte: int = 1,
        is_m_first: Optional[bool] = False,
        transform_kind_a: Union[int, TransformKind] = 0,
        transform_kind_b: Union[int, TransformKind] = 0,
    ):
        super().__init__(
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            accum_dtype=accum_dtype,
            a_transposed=a_transposed,
            b_transposed=b_transposed,
            block_row_warps=block_row_warps,
            block_col_warps=block_col_warps,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=chunk,
            reduce_k=reduce_k,
            num_elems_per_byte=num_elems_per_byte,
            is_m_first=is_m_first,
        )
        self._initialize_transform_kind(transform_kind_a, transform_kind_b)

    def _initialize_k_dim(self, a_dtype="float16"):
        self.k_dim = 256 // DataType(a_dtype).bits

    def _initialize_local_size(self, m_dim=16, n_dim=16, k_dim=16, warp_size=32):
        self.local_size_a = (m_dim * k_dim) // warp_size
        self.local_size_b = (n_dim * k_dim) // warp_size
        self.local_size_out = (m_dim * n_dim) // warp_size

    def _initialize_abbrev(self, a_dtype, b_dtype, accum_dtype):
        self.a_dtype_abbrv = self.dtype_abbrv[a_dtype]
        self.b_dtype_abbrv = self.dtype_abbrv[b_dtype]
        self.accum_dtype_abbrv = self.dtype_abbrv[accum_dtype]

    def _initialize_metal_simdgroup_prefix(self, k_dim=16):
        if k_dim == 16:
            self.simdgroup_suffix = "m16n8k16"
        elif k_dim == 32:
            self.simdgroup_suffix = "m16n8k32"
        else:
            raise ValueError("Unsupported k_dim")

    def _initialize_micro_size(self, m_dim=16, n_dim=16, k_dim=16):
        self.micro_size_x = m_dim
        self.micro_size_y = n_dim
        self.micro_size_k = k_dim

    def _initialize_transform_kind(self, transform_kind_a, transform_kind_b):
        if isinstance(transform_kind_a, int):
            self.transform_kind_a = TransformKind(transform_kind_a)
        elif isinstance(transform_kind_a, TransformKind):
            self.transform_kind_a = transform_kind_a
        else:
            raise ValueError("Unsupported transform_kind_a")

        if isinstance(transform_kind_b, int):
            self.transform_kind_b = TransformKind(transform_kind_b)
        elif isinstance(transform_kind_b, TransformKind):
            self.transform_kind_b = transform_kind_b
        else:
            raise ValueError("Unsupported transform_kind_b")

        assert transform_kind_a in [0, 1, 2, 3], "Input transform stage should be 0, 1, 2, or 3"
        assert transform_kind_b in [0, 1, 2, 3], "Weight transform stage should be 0, 1, 2, or 3"

    def ldsimdgroup_a(self, A_local_buf, A_shared_buf, ki, rk=0):
        """
        Load matrix A into SIMD group registers with ladder transform.
        """
        warp_row_tiles = self.warp_row_tiles
        warp_rows = self.warp_rows
        chunk = self.chunk
        micro_size_x = self.micro_size_x
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        a_dtype = self.a_dtype
        a_transposed = self.a_transposed
        transform_kind_a = self.transform_kind_a

        current_frame = T.KernelLaunchFrame.Current()
        thread_binding = current_frame.get_thread_binding()

        @T.macro
        def _warp_ldsimdgroup_a(
            A_local_buf,
            A_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            stride = A_shared_buf.shape[-1]
            tx, _, warp_m = self.extract_thread_binding(thread_binding)
            if transform_kind_a == TransformKind.NonTransform:
                for i in T.serial(warp_rows):
                    # Use Metal-specific SIMD group load
                    for local_id in T.vectorized(local_size_a):
                        from .metal_layout import thread_id_shared_access_64x4_to_16x16_layout_A
                        row, col = T.meta_var(thread_id_shared_access_64x4_to_16x16_layout_A(tx, local_id))
                        l, r = (
                            warp_m * warp_row_tiles + i * micro_size_x,
                            rk * chunk + ki * micro_size_k,
                        )
                        A_local_buf[i * local_size_a + local_id] = A_shared_buf[l + row, r + col]
            elif transform_kind_a == TransformKind.InterWarpTransform:
                for i in T.serial(warp_rows):
                    # Assign B_shared_elem
                    ri, rj = (
                        warp_m * warp_row_tiles + i * micro_size_x,
                        rk * chunk + ki * micro_size_k,
                    )
                    ni, nj, nii, njj = (
                        (ri) // micro_size_x,
                        (rj) // micro_size_k,
                        (ri) % micro_size_x,
                        (rj) % micro_size_k,
                    )
                    args = (ni, nj, nii, njj) if transform_kind_a > 0 else (ri, rj)
                    A_shared_elem = A_shared_buf[args]

                    # Use Metal-specific SIMD group load
                    for local_id in T.vectorized(local_size_a):
                        from .metal_layout import thread_id_shared_access_64x4_to_16x16_layout_A
                        row, col = T.meta_var(thread_id_shared_access_64x4_to_16x16_layout_A(tx, local_id))
                        A_local_buf[i * local_size_a + local_id] = A_shared_elem[row, col]
            elif transform_kind_a == TransformKind.IntraWarpTransform:
                for i in T.serial(warp_rows):
                    # Assign B_shared_elem
                    ri, rj = (
                        warp_m * warp_row_tiles + i * micro_size_x,
                        rk * chunk + ki * micro_size_k,
                    )
                    ni, nj, nii, njj = (
                        (ri) // micro_size_x,
                        (rj) // micro_size_k,
                        (ri) % micro_size_x,
                        (rj) % micro_size_k,
                    )
                    A_shared_elem = A_shared_buf[ni, nj, nii, njj]

                    # Use Metal-specific SIMD group load
                    for local_id in T.vectorized(local_size_a):
                        from .metal_layout import thread_id_shared_access_64x4_to_16x16_layout_A
                        row, col = T.meta_var(thread_id_shared_access_64x4_to_16x16_layout_A(tx, local_id))
                        A_local_buf[i * local_size_a + local_id] = A_shared_elem[row, col]
            elif transform_kind_a == TransformKind.LDMatrixTransform:
                for j in T.serial(warp_rows):
                    for local_id in T.vectorized(local_size_a):
                        # Assign A_shared_elem
                        ri, rj = (
                            warp_m * warp_rows + j,
                            rk * (chunk // micro_size_k) + ki,
                        )
                        rii, rjj = (tx * local_size_a +
                                    local_id) // micro_size_k, (tx * local_size_a + local_id) % (
                                        micro_size_k)
                        A_local_buf[j * local_size_a + local_id] = (A_shared_buf[ri, rj, rii, rjj])
            else:
                raise ValueError("Unsupported TransformKind for Input A")

        return _warp_ldsimdgroup_a(A_local_buf, A_shared_buf, ki, thread_binding, rk)

    def ldsimdgroup_b(self, B_local_buf, B_shared_buf, ki, rk=0):
        """
        Load matrix B into SIMD group registers with ladder transform.
        """
        warp_col_tiles = self.warp_col_tiles
        warp_cols = self.warp_cols
        chunk = self.chunk
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        b_dtype = self.b_dtype
        transform_kind_b = self.transform_kind_b
        b_transposed = self.b_transposed
        num_elems_per_byte = self.num_elems_per_byte

        current_frame = T.KernelLaunchFrame.Current()
        thread_binding = current_frame.get_thread_binding()

        @T.macro
        def _warp_ldsimdgroup_b(
            B_local_buf,
            B_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            stride = B_shared_buf.shape[-1]
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)

            if transform_kind_b == TransformKind.NonTransform:
                for j in T.serial(warp_cols):
                    # Assign B_shared_elem
                    ri, rj = (
                        warp_n * warp_col_tiles + j * micro_size_y,
                        rk * chunk + ki * micro_size_k,
                    )
                    B_shared_elem = B_shared_buf[ri, rj]

                    # Use Metal-specific SIMD group load
                    for local_id in T.vectorized(local_size_b):
                        from .metal_layout import thread_id_shared_access_64x4_to_16x16_layout_B
                        row, col = T.meta_var(thread_id_shared_access_64x4_to_16x16_layout_B(tx, local_id))
                        B_local_buf[j * local_size_b + local_id] = B_shared_elem[row, col]
            elif transform_kind_b == TransformKind.InterWarpTransform:
                for j in T.serial(warp_cols):
                    # Assign B_shared_elem
                    ri, rj = (
                        warp_n * warp_col_tiles + j * micro_size_y,
                        rk * chunk + ki * micro_size_k,
                    )
                    ni, nj, nii, njj = (
                        (ri) // micro_size_y,
                        (rj) // micro_size_k,
                        (ri) % micro_size_y,
                        (rj) % micro_size_k,
                    )
                    B_shared_elem = B_shared_buf[ni, nj, nii, njj]

                    # Use Metal-specific SIMD group load
                    for local_id in T.vectorized(local_size_b):
                        from .metal_layout import thread_id_shared_access_64x4_to_16x16_layout_B
                        row, col = T.meta_var(thread_id_shared_access_64x4_to_16x16_layout_B(tx, local_id))
                        B_local_buf[j * local_size_b + local_id] = B_shared_elem[row, col]
            elif transform_kind_b == TransformKind.IntraWarpTransform:
                for j in T.serial(warp_cols):
                    # Assign B_shared_elem
                    ri, rj = (
                        warp_n * warp_col_tiles + j * micro_size_y,
                        rk * chunk + ki * micro_size_k,
                    )
                    ni, nj, nii, njj = (
                        (ri) // micro_size_y,
                        (rj) // micro_size_k,
                        (ri) % micro_size_y,
                        (rj) % micro_size_k,
                    )
                    B_shared_elem = B_shared_buf[ni, nj, nii, njj]

                    # Use Metal-specific SIMD group load
                    for local_id in T.vectorized(local_size_b):
                        from .metal_layout import thread_id_shared_access_64x4_to_16x16_layout_B
                        row, col = T.meta_var(thread_id_shared_access_64x4_to_16x16_layout_B(tx, local_id))
                        B_local_buf[j * local_size_b + local_id] = B_shared_elem[row, col]
            elif transform_kind_b == TransformKind.LDMatrixTransform:
                local_size_dequantize = local_size_b // num_elems_per_byte
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(local_size_dequantize):
                        # Assign B_shared_elem
                        ri, rj = (
                            warp_n * warp_cols + j,
                            rk * (chunk // micro_size_k) + ki,
                        )
                        rii, rjj = (tx * local_size_dequantize +
                                    local_id) // (micro_size_k // num_elems_per_byte), (
                                        tx * local_size_dequantize + local_id) % (
                                            micro_size_k // num_elems_per_byte)
                        B_local_buf[j * local_size_dequantize + local_id] = (
                            B_shared_buf[ri, rj, rii, rjj])
            else:
                raise ValueError("Unsupported TransformKind for Input B")

        return _warp_ldsimdgroup_b(B_local_buf, B_shared_buf, ki, thread_binding, rk)

    def simdgroup_mma(self, A_local_buf, B_local_buf, C_local_buf):
        """
        Perform SIMD group matrix multiplication with ladder transform.
        """
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        k_pack = self.k_pack
        simdgroup_suffix = self.simdgroup_suffix
        a_dtype, b_dtype, out_dtype = self.a_dtype, self.b_dtype, self.accum_dtype
        compute_a_dtype = a_dtype if local_size_a == 1 else f"{a_dtype}x{local_size_a}"
        compute_b_dtype = b_dtype if local_size_b == 1 else f"{b_dtype}x{local_size_b}"
        compute_out_dtype = out_dtype if local_size_out == 1 else f"{out_dtype}x{local_size_out}"

        @T.macro
        def _warp_mma(A_local_buf, B_local_buf, C_local_buf):
            for kp, i, j in T.grid(k_pack, warp_rows, warp_cols):
                # Metal SIMD group matrix multiplication intrinsic
                T.tvm_metal_simdgroup_mma(
                    simdgroup_suffix,
                    "row",
                    "row",
                    compute_a_dtype,
                    compute_b_dtype,
                    compute_out_dtype,
                    B_local_buf.data,
                    ((j * k_pack + kp) * local_size_b) // local_size_b,
                    A_local_buf.data,
                    ((i * k_pack + kp) * local_size_a) // local_size_a,
                    C_local_buf.data,
                    (i * warp_cols * local_size_out + j * local_size_out) // local_size_out,
                    dtype=compute_out_dtype,
                )

        return _warp_mma(A_local_buf, B_local_buf, C_local_buf)


# Utility functions for Metal SIMD group operations
def metal_simdgroup_store_index_map(thread_id, local_id):
    """
    Metal-specific store index map function.
    """
    from .metal_layout import thread_id_shared_access_64x4_to_16x16_layout_C_n_m
    return thread_id_shared_access_64x4_to_16x16_layout_C_n_m(thread_id, local_id)