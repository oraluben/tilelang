#pragma once

#ifndef __MUSACC_RTC__
#include <musa.h>
#endif

#include "common.h"

namespace tl {

template <SmemSwizzleGranularity sg = SmemSwizzleGranularity::NONE,
          SmemSwizzleStride ss = SmemSwizzleStride::B256,
          SmemSwizzleLine sl = SmemSwizzleLine::B256,
          CacheHint inner_hint = CacheHint::CACHE_NORMAL,
          CacheHint outer_hint = CacheHint::CACHE_NORMAL,
          PrefetchSize prefetch = PrefetchSize::NONE>
TL_DEVICE void tma_load(const MUtensorDescriptor &descriptor,
                        uint32_t const &bar_id, void *smem_ptr,
                        int32_t const &crd0, int32_t const &dim0) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  __musa_tme_ld_tile_1d(
      bar_id, make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr),
      gmem_int_desc, dim0, crd0, static_cast<int32_t>(sg),
      static_cast<int32_t>(ss), static_cast<int32_t>(sl),
      static_cast<int32_t>(prefetch), static_cast<int32_t>(inner_hint),
      static_cast<int32_t>(outer_hint), 0);
}

template <SmemSwizzleGranularity sg = SmemSwizzleGranularity::NONE,
          SmemSwizzleStride ss = SmemSwizzleStride::B256,
          SmemSwizzleLine sl = SmemSwizzleLine::B256,
          CacheHint inner_hint = CacheHint::CACHE_NORMAL,
          CacheHint outer_hint = CacheHint::CACHE_NORMAL,
          PrefetchSize prefetch = PrefetchSize::NONE>
TL_DEVICE void tma_load(const MUtensorDescriptor &descriptor,
                        uint32_t const &bar_id, void *smem_ptr,
                        int32_t const &crd0, int32_t const &crd1,
                        int32_t const &dim0, int32_t const &dim1) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  v2i32_t crd{crd0, crd1};
  v2i32_t dim{dim0, dim1};

  __musa_tme_ld_tile_2d(
      bar_id, make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr),
      gmem_int_desc, dim, crd, static_cast<int32_t>(sg),
      static_cast<int32_t>(ss), static_cast<int32_t>(sl),
      static_cast<int32_t>(prefetch), static_cast<int32_t>(inner_hint),
      static_cast<int32_t>(outer_hint), 0);
}

template <SmemSwizzleGranularity sg = SmemSwizzleGranularity::NONE,
          SmemSwizzleStride ss = SmemSwizzleStride::B256,
          SmemSwizzleLine sl = SmemSwizzleLine::B256>
TL_DEVICE void tma_store(const MUtensorDescriptor &descriptor,
                         void const *smem_ptr, int32_t const &crd0,
                         int32_t const &dim0) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  __musa_tme_st_1d(
      make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr),
      gmem_int_desc, dim0, crd0, static_cast<int32_t>(sg),
      static_cast<int32_t>(ss), static_cast<int32_t>(sl));
}

template <SmemSwizzleGranularity sg = SmemSwizzleGranularity::NONE,
          SmemSwizzleStride ss = SmemSwizzleStride::B256,
          SmemSwizzleLine sl = SmemSwizzleLine::B256>
TL_DEVICE void tma_store(const MUtensorDescriptor &descriptor,
                         void const *smem_ptr, int32_t const &crd0,
                         int32_t const &crd1, int32_t const &dim0,
                         int32_t const &dim1) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  v2i32_t crd{crd0, crd1};
  v2i32_t dim{dim0, dim1};
  __musa_tme_st_2d(
      make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr),
      gmem_int_desc, dim, crd, static_cast<int32_t>(sg),
      static_cast<int32_t>(ss), static_cast<int32_t>(sl));
}
} // namespace tl
