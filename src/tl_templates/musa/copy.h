#pragma once

#include "common.h"

#define _AS1 __attribute__((address_space(1)))
#define _AS3 __attribute__((address_space(3)))

namespace tl {

TL_DEVICE void cp_async_commit() {
}

template <int N> TL_DEVICE void cp_async_wait() {
  // __musa_memcpy_g2s_wait();
}

template <int N>
TL_DEVICE void cp_async_gs(void const *const smem_addr, void *global_ptr) {
  __musa_memcpy_g2s((void _AS3 *)smem_addr, (void _AS1 *)global_ptr, N /* total_bytes */, 0 /* prefetch_size */);
}

template <int N>
TL_DEVICE void cp_async_gs_conditional(void const *const smem_addr,
                                       void *global_ptr, bool cond) {}

} // namespace tl
