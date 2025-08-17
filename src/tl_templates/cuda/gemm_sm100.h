#pragma once

#include <cute/arch/mma_sm100.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/underscore.hpp>

#include "common.h"

namespace tl {

namespace detail_sm100 {

using namespace cute;

template <typename A_type, typename B_type, typename C_type,
          int M, int N, bool trans_A, bool trans_B>
struct UmmaInstructionSelector;

// F16/BF16 -> F32 accumulate
template <int M, int N, bool trans_A, bool trans_B>
struct UmmaInstructionSelector<half_t, half_t, float, M, N, trans_A, trans_B> {
  static constexpr UMMA::Major MajorA = trans_A ? UMMA::Major::MN : UMMA::Major::K;
  static constexpr UMMA::Major MajorB = trans_B ? UMMA::Major::K  : UMMA::Major::MN;
  using MMA = SM100_MMA_F16BF16_SS<half_t, half_t, float, M, N, MajorA, MajorB>;
};

template <int M, int N, bool trans_A, bool trans_B>
struct UmmaInstructionSelector<bfloat16_t, bfloat16_t, float, M, N, trans_A, trans_B> {
  static constexpr UMMA::Major MajorA = trans_A ? UMMA::Major::MN : UMMA::Major::K;
  static constexpr UMMA::Major MajorB = trans_B ? UMMA::Major::K  : UMMA::Major::MN;
  using MMA = SM100_MMA_F16BF16_SS<bfloat16_t, bfloat16_t, float, M, N, MajorA, MajorB>;
};

// TF32 -> F32 accumulate
template <int M, int N, bool trans_A, bool trans_B>
struct UmmaInstructionSelector<tfloat32_t, tfloat32_t, float, M, N, trans_A, trans_B> {
  static constexpr UMMA::Major MajorA = trans_A ? UMMA::Major::MN : UMMA::Major::K;
  static constexpr UMMA::Major MajorB = trans_B ? UMMA::Major::K  : UMMA::Major::MN;
  using MMA = SM100_MMA_TF32_SS<tfloat32_t, tfloat32_t, float, M, N, MajorA, MajorB>;
};

// INT8 -> S32 accumulate
template <int M, int N, bool trans_A, bool trans_B>
struct UmmaInstructionSelector<int8_t, int8_t, int32_t, M, N, trans_A, trans_B> {
  static constexpr UMMA::Major MajorA = trans_A ? UMMA::Major::MN : UMMA::Major::K;
  static constexpr UMMA::Major MajorB = trans_B ? UMMA::Major::K  : UMMA::Major::MN;
  using MMA = SM100_MMA_S8_SS<int8_t, int8_t, int32_t, M, N, MajorA, MajorB>;
};

// Helper to build swizzled SMEM layout for UMMA from the MMA partition shapes
template <class Type, UMMA::Major MajorMode, class MmaShape>
static CUTE_DEVICE auto make_smem_layout(MmaShape const &mma_shape) {
  if constexpr (MajorMode == UMMA::Major::K) {
    return UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<Type>{}, mma_shape);
  } else {
    return UMMA::tile_to_mma_shape(UMMA::Layout_MN_SW128_Atom<Type>{}, mma_shape);
  }
}

template <int M, int N, int K, bool trans_A, bool trans_B, bool clear_accum,
          typename A_type, typename B_type, typename C_type>
struct GemmTensorOpSS {
  using Selector = UmmaInstructionSelector<A_type, B_type, C_type, M, N, trans_A, trans_B>;
  using MmaAtom  = typename Selector::MMA;

  static CUTE_DEVICE void body(A_type *pA_smem, B_type *pB_smem, uint32_t tmem_c_addr) {
    // Constraints per SM100 UMMA ISA (single-CTA instruction forms)
    static_assert(M == 64 || M == 128, "SM100 UMMA SS requires M in {64,128}");
    static_assert(N % 8 == 0 && N >= 8 && N <= 256, "SM100 UMMA SS requires 8<=N<=256 and N%8==0");

    const int tid = threadIdx.x;

    // Build the tiled MMA
    auto tiled_mma = make_tiled_mma(MmaAtom{});
    auto thr_mma   = tiled_mma.get_thread_slice(tid);

    // Determine post-partitioned shapes for A/B to create swizzled SMEM layouts
    auto mma_shape_A = partition_shape_A(tiled_mma, Shape<Int<M>, Int<(256 / sizeof_bits<A_type>::value)>>{});
    auto mma_shape_B = partition_shape_B(tiled_mma, Shape<Int<N>, Int<(256 / sizeof_bits<B_type>::value)>>{});

    constexpr UMMA::Major MajorA = (trans_A ? UMMA::Major::MN : UMMA::Major::K);
    constexpr UMMA::Major MajorB = (trans_B ? UMMA::Major::K  : UMMA::Major::MN);

    using SmemLayoutA = decltype(make_smem_layout<A_type, MajorA>(mma_shape_A));
    using SmemLayoutB = decltype(make_smem_layout<B_type, MajorB>(mma_shape_B));

    Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<A_type *>(pA_smem)), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<B_type *>(pB_smem)), SmemLayoutB{});

    // Fragments and accumulator (TMEM on SM100)
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);

    Tensor acc = make_tensor(
        make_tmem_ptr<C_type>(tmem_c_addr),
        partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));

    // Control accumulation behavior
    if constexpr (clear_accum) {
      tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), acc);
      tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
  }
};

} // namespace detail_sm100

// Public API: SS variant on SM100 (A,B in SMEM). Accumulator C is a TMEM base address.
// Note: On SM100 (tcgen05), accumulators live in TMEM. Pass TMEM address for C via tmem_c_addr.
template <int M, int N, int K,
          int num_warp_m, int num_warp_n,
          bool trans_A, bool trans_B,
          bool clear_accum = false,
          typename A_type, typename B_type, typename C_type>
TL_DEVICE void gemm_ss(A_type *pA_smem, B_type *pB_smem, uint32_t tmem_c_addr) {
  // num_warp_* are unused for SM100 UMMA single-CTA form, kept for API parity
  (void)num_warp_m; (void)num_warp_n;
  detail_sm100::GemmTensorOpSS<M, N, K, trans_A, trans_B, clear_accum,
                               A_type, B_type, C_type>::body(pA_smem, pB_smem, tmem_c_addr);
}

} // namespace tl


