from .gemm_base import GemmBase
from tilelang.utils.language import is_shared, is_fragment, is_full_region
from tilelang import tvm as tvm
from tvm.target import Target
from tvm import tir
from tilelang import language as T
from tilelang.transform.simplify import _Simplify


class GemmMetal(GemmBase):
    pass
