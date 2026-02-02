def shared_8x8_to_mma_a_32x2_layout(row, col):
    """
    Z-Order (Morton) curve from philipturner/metal-flash-attention
    """

    # assert rep == 0
    # if rep != 0:
    #     import ipdb; ipdb.set_trace()

    local_id = col % 2

    r0 = row % 2
    r1 = (row // 2) % 2
    r2 = (row // 4) % 2

    c1 = (col // 2) % 2
    c2 = (col // 4) % 2

    tid = c1 + 2 * r0 + 4 * r1 + 8 * c2 + 16 * r2

    return tid, local_id


import tilelang.language as T
from tvm.tir import IndexMap


metal_store_layout = IndexMap.from_func(shared_8x8_to_mma_a_32x2_layout, index_dtype=T.int32)


def forward_thread(i: int, j: int) -> int:
    """
    Given the row index `i` and column index `j` in the fragment,
    """
    lane_id, _ = metal_store_layout.map_indices([i, j])
    return lane_id


def forward_index(i: int, j: int) -> int:
    """
    Given the row index `i` and column index `j` in the fragment,
    """
    _, local_id = metal_store_layout.map_indices([i, j])
    return local_id


metal_store_fragment = T.Fragment(
    [8, 8],
    forward_thread_fn=forward_thread,
    forward_index_fn=forward_index,
)


# # Print the layout structure (optional for debugging)
# print(base_fragment)

# from tilelang.tools import plot_layout

# # Plot and save the layout visualization
# plot_layout(base_fragment, name="base_layout")
