#pragma once

#include "common.h"
#include "intrin.h"

#include <mutlass/arch/barrier.hpp>
#include <mutlass/mutlass.h>
#include <mutlass/gemm/collective/collective_builder.hpp>

namespace mute {

using mutlass::gemm::collective::detail;

namespace tl_wgmma {

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, typename A_type_raw,
          typename B_type_raw, typename C_type_raw>
class GemmTensorOp {
public:
  using A_type = A_type_raw;
  using B_type = B_type_raw;
  using C_type = C_type_raw;

  static constexpr TCE::Major GmmaMajorA =
      trans_A ? TCE::Major::MN : TCE::Major::K;
  static constexpr TCE::Major GmmaMajorB =
      trans_B ? TCE::Major::K : TCE::Major::MN;

  using TileShape_MNK = Shape<Int<4 * M / num_warp_m>, Int<N / num_warp_n>, Int<K>>

  using SqmmaOp = decltype(mute::MP31::SQMMA::ss_op_selector<A_type, B_type, C_type, TileShape_MNK, GmmaMajorA, GmmaMajorB>());

  using SmemLayoutAtomA =
      decltype(ss_smem_selector_A<GmmaMajorA, A_type, SqmmaOp, TileShape_MNK>());
  using SmemLayoutAtomB =
      decltype(ss_smem_selector_B<GmmaMajorB, B_type, SqmmaOp, TileShape_MNK>());

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{}, Shape<Int<M>, Int<K>>{},
      conditional_t<trans_A, Step<_2, _1>, Step<_1, _2>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{}, Shape<Int<N>, Int<K>>{},
      conditional_t<trans_B, Step<_1, _2>, Step<_2, _1>>{}));

  template <int wg_wait = 0>
  static CUTE_DEVICE void body(A_type_raw *pA, B_type_raw *pB, C_type_raw *pC) {
    const int tid = threadIdx.x;
    Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<A_type *>(pA)),
                            SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<B_type *>(pB)),
                            SmemLayoutB{});
    auto tiled_mma = make_tiled_mma(
        SqmmaOp{},
        Layout<Shape<Int<num_warp_m / 4>, Int<num_warp_n>, _1>>{});
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    // Allocate registers for pipelining
    Tensor tCsA = thr_mma.partition_A(sA); // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB); // (MMA,MMA_N,MMA_K,PIPE)

    Tensor tCrA = thr_mma.make_fragment_A(tCsA); // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB); // (MMA,MMA_M,MMA_N,PIPE)

    Tensor acc =
        make_tensor(make_rmem_ptr(reinterpret_cast<C_type *>(pC)),
                    partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));

    warpgroup_fence_operand(acc);
    warpgroup_arrive();
    if constexpr (clear_accum) {
      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      // warpgroup_arrive();
      // (V,M) x (V,N) => (V,M,N)
      gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), acc);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }

    warpgroup_commit_batch();
    if constexpr (wg_wait >= 0) {
      warpgroup_wait<wg_wait>();
    }
    warpgroup_fence_operand(acc);
  }
};

} // namespace tl_wgmma

} // namespace cute

namespace tl {

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum = false, int lda = 0, int ldb = 0,
          int offset_a = 0, int offset_b = 0, bool use_wgmma = true,
          int wg_wait = 0, typename A_type, typename B_type, typename C_type>
TL_DEVICE void gemm_ss(A_type *pA, B_type *pB, C_type *accum) {

static_assert(use_wgmma, "Only Support SQMMA");

static_assert((trans_A && lda == M) || (!trans_A && lda == K),
                "Hopper wgmma doesn't support custom stride for A");
static_assert((trans_B && ldb == K) || (!trans_B && ldb == N),
                "Hopper wgmma doesn't support custom stride for B");
static_assert(offset_a == 0 && offset_b == 0,
                "offset_a and offset_b must be zero for wgmma");
using MMA = cute::tl_wgmma::GemmTensorOp<M, N, K, num_warp_m, num_warp_n,
                                            trans_A, trans_B, clear_accum,
                                            A_type, B_type, C_type>;
MMA::body<wg_wait>(pA, pB, accum);
}


template <int num_mma>
TL_DEVICE /**
           * Wait for all WMMA/MMA warps in the current warp-group to
           * synchronize.
           *
           * Blocks until the warp-group-wide rendezvous for `num_mma` MMA lanes
           * completes, ensuring all participating warps have arrived before
           * proceeding.
           */
    void
    wait_wgmma() {
  mute::warpgroup_wait<num_mma>();
}

template <int NumMmaThreads> TL_DEVICE void warp_scheduler_barrier_sync() {
  mutlass::arch::NamedBarrier::sync(NumMmaThreads,
                                    mutlass::canonical_warp_group_idx() /*id*/);
}

template <int NumMmaThreads> TL_DEVICE void warp_scheduler_barrier_arrive() {
  static_assert(NumMmaThreads == 256 || NumMmaThreads == 384);
  if constexpr (NumMmaThreads == 256) {
    mutlass::arch::NamedBarrier::arrive(
        NumMmaThreads, (1 - mutlass::canonical_warp_group_idx()) /*id*/);
  } else {
    mutlass::arch::NamedBarrier::arrive(
        NumMmaThreads,
        (mutlass::canonical_warp_group_idx() <= 1
             ? mutlass::canonical_warp_group_idx() + 1
             : mutlass::canonical_warp_group_idx() + 1 - 3) /*id*/);
    mutlass::arch::NamedBarrier::arrive(
        NumMmaThreads,
        (mutlass::canonical_warp_group_idx() <= 0
             ? mutlass::canonical_warp_group_idx() + 2
             : mutlass::canonical_warp_group_idx() + 2 - 3) /*id*/);
  }
}

template <int NumMmaThreads> TL_DEVICE void mma_init() {
  static_assert(NumMmaThreads == 256 || NumMmaThreads == 384);
  if (mutlass::canonical_warp_group_idx() > 0) {
    mutlass::arch::NamedBarrier::arrive(NumMmaThreads, 0);
  }
  if constexpr (NumMmaThreads == 384) {
    if (mutlass::canonical_warp_group_idx() > 1) {
      mutlass::arch::NamedBarrier::arrive(NumMmaThreads, 1 /*id*/);
    }
  }
}
} // namespace tl
