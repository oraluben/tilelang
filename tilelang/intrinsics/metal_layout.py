from typing import Union
from tvm import arith, DataType
import tilelang.language as T


def ldsimdgroup_32x8_to_shared_16x16_layout(thread_id, local_id):
    """
    Layout function for loading 32x8 data from shared memory to SIMD group (16x16 matrix).
    This is the Metal equivalent of CUDA's ldmatrix operation.
    """
    row = thread_id % 16
    col = 8 * (thread_id // 16) + local_id % 8
    return row, col


def ldsimdgroup_trans_32x8_to_shared_16x16_layout(thread_id, local_id):
    """
    Transposed layout function for loading 32x8 data from shared memory to SIMD group (16x16 matrix).
    """
    row = 8 * (thread_id // 16) + (thread_id % 8)
    col = 8 * ((thread_id % 16) // 8) + local_id % 8
    return row, col


def ldsimdgroup_16x32_to_shared_16x32_layout_a(thread_id, local_id):
    """
    Layout function for loading 16x32 data from shared memory to SIMD group for matrix A.
    """
    row = thread_id % 16
    col = 16 * (thread_id // 16) + local_id % 16
    return row, col


def ldsimdgroup_16x32_to_shared_16x32_layout_b(thread_id, local_id):
    """
    Layout function for loading 16x32 data from shared memory to SIMD group for matrix B.
    """
    row = 8 * (thread_id // 16) + (thread_id % 8)
    col = 16 * ((thread_id % 16) // 8) + local_id % 16
    return row, col


def ldsimdgroup_32x16_to_shared_16x32_layout_a(thread_id, local_id):
    """
    Layout function for loading 32x16 data from shared memory to SIMD group for matrix A.
    """
    row = thread_id % 16
    col = local_id + (thread_id // 16) * 16
    return row, col


def ldsimdgroup_32x16_to_shared_16x32_layout_b(thread_id, local_id):
    """
    Layout function for loading 32x16 data from shared memory to SIMD group for matrix B.
    """
    row = (thread_id // 16) * 8 + (thread_id % 8)
    col = local_id + 16 * ((thread_id % 16) // 8)
    return row, col


def simdgroup_store_32x8_to_shared_16x16_layout(thread_id, local_id):
    """
    Layout function for storing 32x8 data from SIMD group to shared memory (16x16 matrix).
    This is the Metal equivalent of CUDA's stmatrix operation.
    """
    row = 8 * (local_id % 4 // 2) + (thread_id // 4)
    col = 8 * (local_id // 4) + (thread_id % 4) * 2 + (local_id % 2)
    return row, col


# sr represents spatial + reduction layout
# the first axis is spatial while the second axis is reduction
def shared_16x16_to_simdgroup_32x8_layout_sr(i, j):
    """
    Shared memory to SIMD group layout mapping (spatial + reduction).
    """
    thread_id = 4 * (i % 8) + (j % 8) // 2
    return thread_id, 4 * (j // 8) + (i // 8) * 2 + (j % 2)


def shared_16x16_to_simdgroup_32x8_layout_rs(i, j):
    """
    Shared memory to SIMD group layout mapping (reduction + spatial).
    """
    thread_id = 4 * (j % 8) + (i % 8) // 2
    return thread_id, 4 * (i // 8) + (j // 8) * 2 + (i % 2)


shared_16x16_to_simdgroup_32x8_layout = shared_16x16_to_simdgroup_32x8_layout_sr
shared_16x16_to_simdgroup_32x8_layout_trans = shared_16x16_to_simdgroup_32x8_layout_rs


def shared_16x32_to_simdgroup_32x16_layout(i, j):
    """
    Shared memory to SIMD group layout mapping for 16x32 matrices.
    """
    thread_id = 4 * (i % 8) + (j % 16) // 4
    return thread_id, 8 * (j // 16) + (i // 8) * 4 + j % 4


def shared_32x16_to_simdgroup_32x16_layout(i, j):
    """
    Shared memory to SIMD group layout mapping for 32x16 matrices.
    """
    thread_id = (i % 16) // 4 + 4 * (j % 8)
    return thread_id, 8 * (j // 8) + (i // 16) * 4 + i % 4


def simdgroup_32x8_to_shared_16x16_layout(thread_id, local_id):
    """
    SIMD group to shared memory layout mapping (32x8 to 16x16).
    """
    row = 8 * (local_id % 4 // 2) + (thread_id // 4)
    col = 8 * (local_id // 4) + (thread_id % 4) * 2 + (local_id % 2)
    return row, col


def shared_16x16_to_simdgroup_32x8_smoothlayout(i, j):
    """
    Smooth layout mapping for shared memory to SIMD group.
    """
    return (i * 2 + j // 8, j % 8)


def shared_16x32_to_simdgroup_32x16_smoothlayout(i, j):
    """
    Smooth layout mapping for shared memory to SIMD group (16x32).
    """
    return (i * 2 + j // 16, j % 16)


def shared_32x16_to_simdgroup_32x16_smoothlayout(i, j):
    """
    Smooth layout mapping for shared memory to SIMD group (32x16).
    """
    return (i * 2 + j // 16, j % 16)


def get_swizzle_layout(row_idx, col_idx, row_size, dtype: Union[DataType, str], swizzle_bytes=None):
    """
    Get swizzled layout for memory access optimization.
    """
    ana = arith.Analyzer()
    if isinstance(dtype, str):
        dtype = DataType(dtype)
    row_bytes = dtype.bits * row_size // 8
    assert row_bytes % 32 == 0, "Row size must be multiple of 32B."
    if swizzle_bytes is None:
        swizzle_bytes = min(128, row_bytes)
    # 128B swizzle
    #   Use 8 * 8 permuted layout
    #   Every number below corresponds to 16B
    #   0  1  2  3  4  5  6  7    ==>    0  1  2  3  4  5  6  7
    #   0  1  2  3  4  5  6  7    ==>    1  0  3  2  5  4  7  6
    #   0  1  2  3  4  5  6  7    ==>    2  3  0  1  6  7  4  5
    #   0  1  2  3  4  5  6  7    ==>    3  2  1  0  7  6  5  4
    #   0  1  2  3  4  5  6  7    ==>    4  5  6  7  0  1  2  3
    #   0  1  2  3  4  5  6  7    ==>    5  4  7  6  1  0  3  2
    #   0  1  2  3  4  5  6  7    ==>    6  7  4  5  2  3  0  1
    #   0  1  2  3  4  5  6  7    ==>    7  6  5  4  3  2  1  0
    # 64B swizzle
    #  Use 8 * 4 permuted layout
    #  Every number below corresponds to 16B
    #  0  1  2  3  4  0  1  2  3    ==>    0  1  2  3  0  1  2  3
    #  0  1  2  3  4  0  1  2  3    ==>    1  0  3  2  1  0  3  2
    #  0  1  2  3  4  0  1  2  3    ==>    2  3  0  1  2  3  0  1
    #  0  1  2  3  4  0  1  2  3    ==>    3  2  1  0  3  2  1  0
    # 32B swizzle
    #  Use 8 * 2 permuted layout
    #  Every number below corresponds to 16B
    #  0  1  2  3  4  5  6  7    ==>    0  1  2  3  4  5  6  7
    #  0  1  2  3  4  5  6  7    ==>    1  0  3  2  5  4  7  6
    elem_per_16B = 128 // dtype.bits
    col_idx_16B = col_idx // elem_per_16B
    col_idx_in_16B = col_idx % elem_per_16B
    new_col_idx_16B = col_idx_16B ^ (row_idx % (swizzle_bytes // 16))
    return row_idx, ana.simplify(new_col_idx_16B * elem_per_16B + col_idx_in_16B)


def make_metal_swizzle_layout(shared_buf, is_smooth: bool = False):
    """
    Create a swizzled layout for Metal shared memory buffers.
    """
    dtype = shared_buf.dtype
    shape = shared_buf.shape

    can_swizzle = shape[-1] * DataType(dtype).bits % 512 == 0
    if is_smooth or (not can_swizzle):
        return T.Layout(shape, lambda *args: args)

    def transform_func(*args):
        i, j = args[-2:]
        new_warp_i, new_warp_j = get_swizzle_layout(i, j, shape[-1], dtype)
        return [*args[:-2], new_warp_i, new_warp_j]

    return T.Layout(shape, transform_func)


# Metal-specific SIMD group layouts
def shared_16x4_to_local_64x1_layout_A(i, j):
    """
    Shared memory to local layout for 16x4 matrix A.
    """
    thread_id = (j * 16 + i)
    return thread_id, 0


def thread_id_shared_access_64x1_to_16x4_layout_A(thread_id, local_id):
    """
    Thread ID to shared access layout for 16x4 matrix A.
    """
    i = thread_id % 16
    j = thread_id // 16
    return i, j


def shared_4x16_to_local_64x1_layout_B(i, j):
    """
    Shared memory to local layout for 4x16 matrix B.
    """
    thread_id = (i * 16 + j)
    return thread_id, 0


def thread_id_shared_access_64x1_to_4x16_layout_B(thread_id, local_id):
    """
    Thread ID to shared access layout for 4x16 matrix B.
    """
    i = thread_id // 16
    j = thread_id % 16
    return i, j


def shared_16x16_to_local_64x4_layout_C(i, j):
    """
    Shared memory to local layout for 16x16 matrix C.
    """
    thread_id = j + (i // 4) * 16
    local = (i % 4)
    return thread_id, local


def shared_16x16_to_ldsimdgroup_64x4_layout(ind):
    """
    Shared memory to SIMD group layout mapping.
    """
    i, j = ind[0], ind[1]
    thread_id, local_id = shared_16x16_to_local_64x4_layout_C(i, j)
    return [thread_id, local_id]


def thread_id_shared_access_64x4_to_16x16_layout_A(thread_id, local_id):
    """
    Thread ID to shared access layout for 16x16 matrix A.
    """
    i = thread_id % 16
    j = (thread_id // 16) * 4 + local_id
    return i, j


def shared_16x16_to_local_64x4_layout_A(i, j):
    """
    Shared memory to local layout for 16x16 matrix A.
    """
    thread_id = i + 16 * (j // 4)
    local = (j % 4)
    return thread_id, local


def thread_id_shared_access_64x4_to_16x16_layout_B(thread_id, local_id):
    """
    Thread ID to shared access layout for 16x16 matrix B.
    """
    i = local_id + (thread_id // 16) * 4
    j = thread_id % 16
    return i, j


def shared_16x16_to_local_64x4_layout_B(i, j):
    """
    Shared memory to local layout for 16x16 matrix B.
    """
    thread_id = j + (i // 4) * 16
    local = (i % 4)
    return thread_id, local


shared_16x16_to_local_64x4_layout_m_n = shared_16x16_to_local_64x4_layout_A
shared_16x16_to_local_64x4_layout_n_k = shared_16x16_to_local_64x4_layout_A
shared_16x16_to_local_64x4_layout_n_m = shared_16x16_to_local_64x4_layout_B
shared_16x16_to_local_64x4_layout_k_n = shared_16x16_to_local_64x4_layout_B


def thread_id_shared_access_64x4_to_16x16_layout_C_m_n(thread_id, local_id):
    """
    Thread ID to shared access layout for 16x16 matrix C (m_n).
    """
    i = local_id + (thread_id // 16) * 4
    j = thread_id % 16
    return i, j


def thread_id_shared_access_64x4_to_16x16_layout_C_n_m(thread_id, local_id):
    """
    Thread ID to shared access layout for 16x16 matrix C (n_m).
    """
    i = thread_id % 16
    j = local_id + (thread_id // 16) * 4
    return i, j


def thread_id_shared_access_64x8_to_16x32_layout_A(thread_id, local_id):
    """
    Thread ID to shared access layout for 16x32 matrix A.
    """
    i = thread_id % 16
    j = (thread_id // 16) * 8 + local_id
    return i, j


def shared_16x32_to_local_64x8_layout_A(i, j):
    """
    Shared memory to local layout for 16x32 matrix A.
    """
    thread_id = i + 16 * (j // 8)
    local = (j % 8)
    return thread_id, local


def thread_id_shared_access_64x8_to_16x32_layout_B(thread_id, local_id):
    """
    Thread ID to shared access layout for 16x32 matrix B.
    """
    i = local_id + (thread_id // 16) * 8
    j = thread_id % 16
    return i, j


def shared_16x32_to_local_64x8_layout_B(i, j):
    """
    Shared memory to local layout for 16x32 matrix B.
    """
    thread_id = j + (i // 8) * 16
    local = (i % 8)
    return thread_id, local


def thread_id_shared_access_64x16_to_16x64_layout_A(thread_id, local_id):
    """
    Thread ID to shared access layout for 16x64 matrix A.
    """
    i = thread_id % 16
    j = local_id + (thread_id // 16) * 16
    return i, j


def shared_16x64_to_local_64x16_layout_A(i, j):
    """
    Shared memory to local layout for 16x64 matrix A.
    """
    thread_id = i + 16 * (j // 16)
    local = (j % 16)
    return thread_id, local


def thread_id_shared_access_64x16_to_16x64_layout_B(thread_id, local_id):
    """
    Thread ID to shared access layout for 16x64 matrix B.
    """
    i = local_id + (thread_id // 16) * 16
    j = thread_id % 16
    return i, j


def shared_16x64_to_local_64x16_layout_B(i, j):
    """
    Shared memory to local layout for 16x64 matrix B.
    """
    thread_id = i + 16 * (j // 16)
    local = (j % 16)
    return thread_id, local


def make_metal_simdgroup_swizzle_layout(shared_buf, vecSize=8):
    """
    Create a swizzled layout for Metal SIMD group operations.
    """
    dtype = shared_buf.dtype
    shape = shared_buf.shape

    numBanks = 32
    bankBitWidth = 32
    SIMDWidth = 32  # Metal thread execution width

    innerDimLength = shape[-1]
    typeWidthInBit = DataType(dtype).bits

    elemsPerOneBanksRow = (numBanks * bankBitWidth) // typeWidthInBit
    perPhase = max(1, elemsPerOneBanksRow // innerDimLength)
    maxPhase = min(SIMDWidth // perPhase, innerDimLength // vecSize)

    def transform(row, col):
        phase = (row // perPhase) % maxPhase
        colOffSwizzled = ((col // vecSize) ^ phase) * vecSize
        colOffOrdered = col % vecSize
        colOff = colOffSwizzled + colOffOrdered
        return row, colOff

    return T.Layout(shape, transform)