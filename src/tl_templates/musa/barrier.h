#pragma once

#include "common.h"
// #include <mutlass/arch/barrier.hpp>

// Reuse cutlass advanced barrier abstraction
// using Barrier = cutlass::arch::ClusterTransactionBarrier;

namespace tl {
TL_DEVICE void
mbarrier_init(uint32_t barrier_id, // 32 bits user-managed barrier's id
              uint32_t warp_count =
                  1, // Warp count expected to arrive/wait on this barrier
              uint32_t init_phase = 0) { // Init phase on this barrier
// #if __MUSA_ARCH__ >= 310
  __musa_async_init_arrival(barrier_id, warp_count, init_phase);
// #endif
}

TL_DEVICE uint32_t mbarrier_try_wait(uint64_t &smem_barrier, int phase_bit) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  uint32_t waitComplete;

  asm volatile("{\n\t"
               ".reg .pred P1; \n\t"
               "mbarrier.try_wait.parity.shared.b64 P1, [%1], %2; \n\t"
               "selp.b32 %0, 1, 0, P1; \n\t"
               "}"
               : "=r"(waitComplete)
               : "r"(smem_int_ptr), "r"(phase_bit));

  return waitComplete;
}

TL_DEVICE void mbarrier_wait(uint32_t barrier_id, int phase_bit) {
// #if __MUSA_ARCH__ >= 310
  __musa_async_wait(barrier_id, phase_bit);
// #endif
}

TL_DEVICE void mbarrier_test_wait(uint64_t &smem_barrier, int phase_bit) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  asm volatile(
      "{\n"
      ".reg .pred                P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.test_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1                       bra.uni DONE;\n"
      "nanosleep.u32 5;\n" // wait a few nanoseconds on pre-Hopper architectures
                           // to save instruction issue slots
      "bra.uni                   LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(smem_int_ptr),
      "r"(phase_bit));
}

TL_DEVICE void mbarrier_arrive(uint32_t barrier_id) {
// #if __MUSA_ARCH__ >= 310
  __musa_async_arrive(barrier_id);
// #endif
}

TL_DEVICE uint32_t mbarrier_arrive(uint32_t barrier_id) {
// #if __MUSA_ARCH__ >= 310
  return __musa_async_arrive(barrier_id);
// #endif
}

TL_DEVICE void mbarrier_arrive(uint64_t &smem_barrier, int cta_id,
                               uint32_t pred) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  if (pred) {
    asm volatile("{\n\t"
                 ".reg .b32 remAddr32;\n\t"
                 "mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
                 "mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n\t"
                 "}"
                 :
                 : "r"(smem_int_ptr), "r"(cta_id));
  }
}

TL_DEVICE void mbarrier_expect_tx(uint64_t &smem_barrier,
                                  uint32_t transaction_bytes) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  asm volatile("mbarrier.expect_tx.shared.b64 [%1], %0;"
               :
               : "r"(transaction_bytes), "r"(smem_int_ptr));
}

TL_DEVICE void mbarrier_arrive_expect_tx(uint32_t barrier_id,
                                         uint32_t transaction_bytes) {
// #if __MUSA_ARCH__ >= 310
  __musa_async_add_trans(barrier_id, transaction_bytes);
// #endif
}

template <typename BarrierType = uint64_t>
TL_DEVICE void mbarrier_cp_async_arrive(BarrierType &smem_mbar) {
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  asm volatile("cp.async.mbarrier.arrive.shared.b64 [%0];"
               :
               : "r"(smem_int_mbar));
}

template <typename BarrierType = uint64_t>
TL_DEVICE void mbarrier_cp_async_arrive_noinc(BarrierType &smem_mbar) {
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  asm volatile("{\n\t"
               "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
               "}"
               :
               : "r"(smem_int_mbar));
  cutlass::arch::synclog_emit_cpasync_barrier_arrive(__LINE__, smem_int_mbar);
}

TL_DEVICE void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" : :);
}

TL_DEVICE void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" : :);
}

// Indicate arrival of warp issuing TMA_STORE
TL_DEVICE void tma_store_arrive() {
  asm volatile("cp.async.bulk.commit_group;");
}

template <int Count> TL_DEVICE void tma_store_wait() {
// #if __MUSA_ARCH__ >= 310
  __musa_tme_idf_l2();
// #endif
}

TL_DEVICE void syncthreads_partial(uint64_t &smem_barrier) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  uint64_t state = 0;
  asm volatile("{\n"
               ".reg .pred                P1;\n"
               "mbarrier.arrive.shared.b64 %1, [%0];\n"
               "LAB_WAIT:\n"
               "mbarrier.try_wait.shared.b64 P1, [%0], %1;\n"
               "@!P1                      bra.uni LAB_WAIT;\n"
               "}\n"
               :
               : "r"(smem_int_ptr), "l"(state));
}
} // namespace tl