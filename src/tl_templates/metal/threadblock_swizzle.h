#pragma once

#include <metal_stdlib>
using namespace metal;

namespace tl {

template <int panel_width>
static inline uint3 rasterization2DRow(uint3 blockIdx, uint3 gridDim) {
  auto ceil_div = [](uint a, uint b) { return (a + b - 1) / b; };
  const uint block_idx = blockIdx.x + blockIdx.y * gridDim.x;
  const uint grid_size = gridDim.x * gridDim.y;
  const uint panel_size = panel_width * gridDim.x;
  const uint panel_offset = block_idx % panel_size;
  const uint panel_idx = block_idx / panel_size;
  const uint total_panel = ceil_div(grid_size, panel_size);
  const uint stride =
      panel_idx + 1 < total_panel
          ? panel_width
          : (grid_size - panel_idx * panel_size) / gridDim.x;
  const uint col_idx = (panel_idx & 1)
                           ? gridDim.x - 1 - panel_offset / stride
                           : panel_offset / stride;
  const uint row_idx = panel_offset % stride + panel_idx * panel_width;
  return uint3(col_idx, row_idx, blockIdx.z);
}

template <int panel_width>
static inline uint3 rasterization2DColumn(uint3 blockIdx, uint3 gridDim) {
  auto ceil_div = [](uint a, uint b) { return (a + b - 1) / b; };
  const uint block_idx = blockIdx.x + blockIdx.y * gridDim.x;
  const uint grid_size = gridDim.x * gridDim.y;
  const uint panel_size = panel_width * gridDim.y;
  const uint panel_offset = block_idx % panel_size;
  const uint panel_idx = block_idx / panel_size;
  const uint total_panel = ceil_div(grid_size, panel_size);
  const uint stride =
      panel_idx + 1 < total_panel
          ? panel_width
          : (grid_size - panel_idx * panel_size) / gridDim.y;
  const uint row_idx = (panel_idx & 1)
                           ? gridDim.y - 1 - panel_offset / stride
                           : panel_offset / stride;
  const uint col_idx = panel_offset % stride + panel_idx * panel_width;
  return uint3(col_idx, row_idx, blockIdx.z);
}

} // namespace tl
