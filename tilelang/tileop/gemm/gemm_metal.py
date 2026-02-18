from .gemm_base import GemmBase
from tilelang.utils.language import is_shared, is_fragment, is_full_region
from tilelang import tvm as tvm
from tvm.target import Target
from tvm import tir
from tilelang import language as T
from tilelang.transform.simplify import _Simplify

from tilelang.layout import make_swizzled_layout


class GemmMetal(GemmBase):
    def infer_layout(self, target: Target, thread_nums: int):

        from tilelang.intrinsics.mma_metal_layout import metal_store_fragment

        return {
            self.A: make_swizzled_layout(self.A),
            self.B: make_swizzled_layout(self.B),
            self.C: metal_store_fragment,
        }

    def lower(self, layout_map: dict, target: Target, thread_nums: int, thread_var: tir.Var):
        assert False
