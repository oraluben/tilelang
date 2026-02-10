"""Tests that TileLang can generate and compile CUDA code without a real GPU device.

These tests validate that the compilation pipeline works correctly when
only the CUDA toolkit (nvcc) is available but no GPU device is present.
"""

import pytest
import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.utils.target import determine_target, check_cuda_availability, _default_cuda_arch_from_nvcc
from tilelang.contrib import nvcc
from tvm.target import Target


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _simple_matmul_func():
    """Return a simple matmul PrimFunc for testing compilation."""

    @T.prim_func
    def matmul(
        A: T.Tensor((128, 128), "float16"),
        B: T.Tensor((128, 128), "float16"),
        C: T.Tensor((128, 128), "float16"),
    ):
        with T.Kernel(T.ceildiv(128, 64), T.ceildiv(128, 64), threads=128) as (bx, by):
            A_shared = T.alloc_shared((64, 64), "float16")
            B_shared = T.alloc_shared((64, 64), "float16")
            C_local = T.alloc_fragment((64, 64), "float16")
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(128, 64), num_stages=2):
                T.copy(A[bx * 64, k * 64], A_shared)
                T.copy(B[k * 64, by * 64], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[bx * 64, by * 64])

    return matmul


requires_cuda_toolkit = pytest.mark.skipif(
    not check_cuda_availability(),
    reason="CUDA toolkit (nvcc) not available",
)


# ---------------------------------------------------------------------------
# target detection tests
# ---------------------------------------------------------------------------


@requires_cuda_toolkit
def test_determine_target_auto_has_arch():
    """determine_target('auto') should always include an arch."""
    target = determine_target("auto", return_object=True)
    assert target.arch is not None
    assert target.arch.startswith("sm_")


@requires_cuda_toolkit
def test_determine_target_cuda_has_arch():
    """determine_target('cuda') should infer an arch when none is specified."""
    target = determine_target("cuda", return_object=True)
    assert target.arch is not None
    assert target.arch.startswith("sm_")


def test_determine_target_explicit_arch_preserved():
    """An explicit arch like 'cuda -arch=sm_80' should be preserved."""
    target = determine_target("cuda -arch=sm_80", return_object=True)
    assert target.arch == "sm_80"


@requires_cuda_toolkit
def test_default_cuda_arch_from_nvcc():
    """_default_cuda_arch_from_nvcc() should return a valid arch string."""
    arch = _default_cuda_arch_from_nvcc()
    # Arch strings are numeric (e.g. "75") or numeric with "a" suffix for
    # architectures >= sm_90 (e.g. "90a").
    assert arch.rstrip("a").isdigit()
    arch_int = int(arch.rstrip("a"))
    assert arch_int >= 50


@requires_cuda_toolkit
def test_get_target_compute_version_with_target():
    """get_target_compute_version should work when target has arch."""
    target = determine_target("auto", return_object=True)
    version = nvcc.get_target_compute_version(target)
    assert "." in version
    major, minor = nvcc.parse_compute_version(version)
    assert major >= 5


# ---------------------------------------------------------------------------
# compilation tests (no GPU device required, only nvcc)
# ---------------------------------------------------------------------------


@requires_cuda_toolkit
def test_compile_auto_target():
    """Compilation with auto target should succeed without GPU."""
    func = _simple_matmul_func()
    kernel = tilelang.compile(func)
    source = kernel.get_kernel_source()
    assert len(source) > 0
    assert "matmul_kernel" in source


@requires_cuda_toolkit
def test_compile_cuda_target():
    """Compilation with explicit 'cuda' target should succeed without GPU."""
    func = _simple_matmul_func()
    kernel = tilelang.compile(func, target="cuda")
    source = kernel.get_kernel_source()
    assert len(source) > 0


@requires_cuda_toolkit
def test_compile_explicit_arch():
    """Compilation with explicit arch should succeed."""
    arch = _default_cuda_arch_from_nvcc()
    func = _simple_matmul_func()
    kernel = tilelang.compile(func, target=f"cuda -arch=sm_{arch}")
    source = kernel.get_kernel_source()
    assert len(source) > 0


@requires_cuda_toolkit
def test_lower_auto_target():
    """Lower (IR generation + codegen) with auto target should succeed."""
    func = _simple_matmul_func()
    target = determine_target("auto", return_object=True)
    with tvm.transform.PassContext(opt_level=3), target:
        artifact = tilelang.lower(func, target=target)
    assert artifact is not None
    assert artifact.kernel_source is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
