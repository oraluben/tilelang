#pragma once

#ifndef __MUSACC_RTC__
#include <musa.h>
#endif

#include "barrier.h"
#include "common.h"

namespace tl {
enum class CacheHintSm90 : uint64_t {
  EVICT_NORMAL = 0x1000000000000000,
  EVICT_FIRST = 0x12F0000000000000,
  EVICT_LAST = 0x14F0000000000000,
};

template <typename BarrierType = uint64_t>
TL_DEVICE void tma_load(void *smem_ptr, void *gmem_ptr, BarrierType &smem_mbar,
                        uint32_t size) {
  uint32_t smem_int_mbar =
      smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::"
               "bytes [%0], [%1], %2, [%3]; \n" ::"r"(smem_int_ptr),
               "l"(gmem_ptr), "r"(size), "r"(smem_int_mbar)
               :);
}

TL_DEVICE void tma_load_multicast(void *smem_ptr, void *gmem_ptr,
                                  uint64_t &smem_mbar, uint32_t size,
                                  uint16_t mask) {
  // uint32_t smem_int_mbar = smem_ptr_to_uint(&smem_mbar);
  // uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  // asm volatile(
  //     "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes."
  //     "multicast::cluster [%0], [%1], %2, [%3], %4; \n" ::"r"(smem_int_ptr),
  //     "l"(gmem_ptr), "r"(size), "r"(smem_int_mbar), "h"(mask)
  //     :);
}

TL_DEVICE void tma_load(void const *&descriptor, BarrierType &smem_mbar,
                        void const *const smem_ptr, int32_t const &crd0) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  __musa_tme_ld_tile_1d(
      bar_id, make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr),
      gmem_int_desc, dim0, crd0, static_cast<int32_t>(sg),
      static_cast<int32_t>(ss), static_cast<int32_t>(sl),
      static_cast<int32_t>(prefetch), static_cast<int32_t>(inner_hint),
      static_cast<int32_t>(outer_hint), 0);
}

TL_DEVICE void tma_load(void const *descriptor, BarrierType &smem_mbar,
                        void const *const smem_ptr, int32_t const &crd0,
                        int32_t const &crd1) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  mute::v2i32_t crd{crd0, crd1};
  mute::v2i32_t dim{dim0, dim1};
  __musa_tme_ld_tile_2d(
      bar_id, make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr),
      gmem_int_desc, dim, crd, static_cast<int32_t>(sg),
      static_cast<int32_t>(ss), static_cast<int32_t>(sl),
      static_cast<int32_t>(prefetch), static_cast<int32_t>(inner_hint),
      static_cast<int32_t>(outer_hint), 0);
}

TL_DEVICE void tma_load(void const *&descriptor, BarrierType &smem_mbar,
                        void const *const smem_ptr, int32_t const &crd0,
                        int32_t const &crd1, int32_t const &crd2) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  mute::v3i32_t crd{crd0, crd1, crd2};
  mute::v3i32_t dim{dim0, dim1, dim2};
  __musa_tme_ld_tile_3d(
      bar_id, make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr),
      gmem_int_desc, dim, crd, static_cast<int32_t>(sg),
      static_cast<int32_t>(ss), static_cast<int32_t>(sl),
      static_cast<int32_t>(prefetch), static_cast<int32_t>(inner_hint),
      static_cast<int32_t>(outer_hint), 0);
}

TL_DEVICE void tma_load(void const *&descriptor, BarrierType &smem_mbar,
                        void const *const smem_ptr, int32_t const &crd0,
                        int32_t const &crd1, int32_t const &crd2,
                        int32_t const &crd3) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  mute::v4i32_t crd{crd0, crd1, crd2, crd3};
  mute::v4i32_t dim{dim0, dim1, dim2, dim3};
  __musa_tme_ld_tile_4d(
      bar_id, make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr),
      gmem_int_desc, dim, crd, static_cast<int32_t>(sg),
      static_cast<int32_t>(ss), static_cast<int32_t>(sl),
      static_cast<int32_t>(prefetch), static_cast<int32_t>(inner_hint),
      static_cast<int32_t>(outer_hint), 0);
}

TL_DEVICE void tma_load(void const *&descriptor, BarrierType &smem_mbar,
                        void const *const smem_ptr, int32_t const &crd0,
                        int32_t const &crd1, int32_t const &crd2,
                        int32_t const &crd3, int32_t const &crd4) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  mute::v5i32_t crd{crd0, crd1, crd2, crd3, crd4};
  mute::v5i32_t dim{dim0, dim1, dim2, dim3, dim4};
  __musa_tme_ld_tile_5d(
      bar_id, make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr),
      gmem_int_desc, dim, crd, static_cast<int32_t>(sg),
      static_cast<int32_t>(ss), static_cast<int32_t>(sl),
      static_cast<int32_t>(prefetch), static_cast<int32_t>(inner_hint),
      static_cast<int32_t>(outer_hint), 0);
}

// TODO:
template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void tma_load_im2col(void const *&descriptor, BarrierType &smem_mbar,
                               void const *const smem_ptr,
                               int32_t const &coord_c, int32_t const &coord_w,
                               int32_t const &coord_h, int32_t const &coord_n,
                               uint16_t const &offset_w,
                               uint16_t const &offset_h) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar =
      smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier:"
               ":complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3, %4, %5, %6}], [%2], {%7, %8}, %9;"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(coord_c), "r"(coord_w), "r"(coord_h), "r"(coord_n),
                 "h"(offset_w), "h"(offset_h), "l"(cache_hint)
               : "memory");
}

TL_DEVICE void tma_store(void *gmem_ptr, void *smem_ptr, uint32_t size) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group"
               ".L2::cache_hint [%0], [%1], %2, %3;"
               :
               : "l"(gmem_ptr), "r"(smem_int_ptr), "r"(size), "l"(cache_hint)
               :);
}

TL_DEVICE void tma_store(void const *&descriptor, void const *const smem_ptr,
                         int32_t const &crd0) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  __musa_tme_st_1d(
      make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr),
      gmem_int_desc, dim0, crd0, static_cast<int32_t>(sg),
      static_cast<int32_t>(ss), static_cast<int32_t>(sl));
}

TL_DEVICE void tma_store(void const *&descriptor, void const *const smem_ptr,
                         int32_t const &crd0, int32_t const &crd1) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  mute::v2i32_t crd{crd0, crd1};
  mute::v2i32_t dim{dim0, dim1};
  __musa_tme_st_2d(
      make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr),
      gmem_int_desc, dim, crd, static_cast<int32_t>(sg),
      static_cast<int32_t>(ss), static_cast<int32_t>(sl));
}

TL_DEVICE void tma_store(void const *&descriptor, void const *const smem_ptr,
                         int32_t const &crd0, int32_t const &crd1,
                         int32_t const &crd2) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  mute::v3i32_t crd{crd0, crd1, crd2};
  mute::v3i32_t dim{dim0, dim1, dim2};
  __musa_tme_st_3d(
      make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr),
      gmem_int_desc, dim, crd, static_cast<int32_t>(sg),
      static_cast<int32_t>(ss), static_cast<int32_t>(sl));
}

TL_DEVICE void tma_store(void const *&descriptor, void const *const smem_ptr,
                         int32_t const &crd0, int32_t const &crd1,
                         int32_t const &crd2, int32_t const &crd3) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  mute::v4i32_t crd{crd0, crd1, crd2, crd3};
  mute::v4i32_t dim{dim0, dim1, dim2, dim3};
  __musa_tme_st_4d(
      make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr),
      gmem_int_desc, dim, crd, static_cast<int32_t>(sg),
      static_cast<int32_t>(ss), static_cast<int32_t>(sl));
}

TL_DEVICE void tma_store(void const *&descriptor, void const *const smem_ptr,
                         int32_t const &crd0, int32_t const &crd1,
                         int32_t const &crd2, int32_t const &crd3,
                         int32_t const &crd4) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  mute::v5i32_t crd{crd0, crd1, crd2, crd3, crd4};
  mute::v5i32_t dim{dim0, dim1, dim2, dim3, dim4};
  __musa_tme_st_5d(
      make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr),
      gmem_int_desc, dim, crd, static_cast<int32_t>(sg),
      static_cast<int32_t>(ss), static_cast<int32_t>(sl));
}

TL_DEVICE void tma_store_add(float *const smem_ptr, float *gmem_ptr,
                             int32_t const &store_bytes) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 "
               "[%0], [%1], %2;\n"
               :
               : "l"(gmem_ptr), "r"(smem_int_ptr), "r"(store_bytes)
               : "memory");
}

TL_DEVICE void prefetch_tma_descriptor(void const *&descriptor) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  asm volatile("prefetch.tensormap [%0];" : : "l"(gmem_int_desc) : "memory");
}

} // namespace tl
