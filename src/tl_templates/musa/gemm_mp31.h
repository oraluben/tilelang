#pragma once

#include "common.h"
#include "intrin.h"

#include <mute/tensor.hpp>
#include <mute/atom/mma_atom.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>


namespace mute {

namespace tl_wgmma {

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, typename A_type_raw,
          typename B_type_raw, typename C_type_raw>
class GemmTensorOp {
public:
  using A_type_mute = typename tl::to_mute_type<A_type_raw>::type;
  using B_type_mute = typename tl::to_mute_type<B_type_raw>::type;
  using A_type = conditional_t<std::is_same<A_type_mute, float>::value,
                               tfloat32_t, A_type_mute>;
  using B_type = conditional_t<std::is_same<B_type_mute, float>::value,
                               tfloat32_t, B_type_mute>;
  using C_type = C_type_raw;

  static constexpr TCE::Major SqmmaMajorA =
      trans_A ? TCE::Major::MN : TCE::Major::K;
  static constexpr TCE::Major SqmmaMajorB =
      trans_B ? TCE::Major::K : TCE::Major::MN;

  using TileShape_MNK = Shape<Int<M>, Int<N>, Int<K>>;

  using SqmmaOp = decltype(mute::MP31::SQMMA::ss_op_selector<A_type, B_type, C_type, TileShape_MNK, SqmmaMajorA, SqmmaMajorB>());
  using SqmmaTraits = MMA_Traits<SqmmaOp>;
  using InstructionShape_MNK = typename SqmmaTraits::Shape_MNK;


  static_assert(size<0>(TileShape_MNK{}) % size<0>(InstructionShape_MNK{}) == 0, "TileM must align to SQMMA M.");
  static_assert(size<1>(TileShape_MNK{}) % size<1>(InstructionShape_MNK{}) == 0, "TileN must align to SQMMA N.");
  static_assert(size<2>(TileShape_MNK{}) % size<2>(InstructionShape_MNK{}) == 0, "TileK must align to SQMMA K.");

  using AtomLayout = decltype(make_layout(
    Shape<Int<size<0>(TileShape_MNK{}) / size<0>(InstructionShape_MNK{})>,
          Int<size<1>(TileShape_MNK{}) / size<1>(InstructionShape_MNK{})>,
          Int<1>>{},
    LayoutRight{}));


  using TiledMma = decltype(make_tiled_mma(SqmmaOp{}, AtomLayout{}));

  using SmemLayoutAtomA =
      decltype(mutlass::gemm::collective::detail::ss_smem_selector_A<SqmmaMajorA, A_type, SqmmaOp, TileShape_MNK>());
  using SmemLayoutAtomB =
      decltype(mutlass::gemm::collective::detail::ss_smem_selector_B<SqmmaMajorB, B_type, SqmmaOp, TileShape_MNK>());

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<1>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<1>{})));

  static TL_DEVICE void body(A_type_raw *pA, B_type_raw *pB, C_type_raw *pC) {
    const int tid = threadIdx.x;
    Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<A_type *>(pA)),
                            SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<B_type *>(pB)),
                            SmemLayoutB{});
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    // Allocate registers for pipelining
    Tensor tCsA = thr_mma.partition_A(sA); // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB); // (MMA,MMA_N,MMA_K,PIPE)

    Tensor tCrA = thr_mma.make_fragment_A(tCsA); // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB); // (MMA,MMA_M,MMA_N,PIPE)

    Tensor acc = make_tensor(make_rmem_ptr(reinterpret_cast<C_type *>(pC)),
                    partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));

    if constexpr (clear_accum) {
      tiled_mma.accumulate_ = mute::MP31::SQMMA::ScaleOut::Zero;
    }

    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      // (V,M) x (V,N) => (V,M,N)
      gemm(tiled_mma, tCrA(_, _, k_block, 0), tCrB(_, _, k_block, 0), acc);
      tiled_mma.accumulate_ = mute::MP31::SQMMA::ScaleOut::One;
    }

  }
};

} // namespace tl_wgmma

} // namespace mute

namespace tl {

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum = false, int lda = 0, int ldb = 0,
          int offset_a = 0, int offset_b = 0, bool use_sqmma = true,
          int wg_wait = 0, typename A_type, typename B_type, typename C_type>
TL_DEVICE void gemm_ss(A_type *pA, B_type *pB, C_type *accum) {

static_assert(use_sqmma, "Only Support SQMMA");

static_assert((trans_A && lda == M) || (!trans_A && lda == K),
                "SQMMA doesn't support custom stride for A");
static_assert((trans_B && ldb == K) || (!trans_B && ldb == N),
                "SQMMA doesn't support custom stride for B");
static_assert(offset_a == 0 && offset_b == 0,
                "offset_a and offset_b must be zero for SQMMA");

using MMA = mute::tl_wgmma::GemmTensorOp<M, N, K, num_warp_m, num_warp_n,
                                            trans_A, trans_B, clear_accum,
                                            A_type, B_type, C_type>;
MMA::body(pA, pB, accum);
}

} // namespace tl
