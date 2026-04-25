"""Rewrite local.fragment → metal.cooperative_tensor for GEMM accumulators on Metal."""

from __future__ import annotations

from tvm import tir, IRModule
from tvm.ir import Op, PointerType
from tvm.tir.transform import prim_func_pass

_GEMM_OPS = None


def _get_gemm_ops():
    global _GEMM_OPS
    if _GEMM_OPS is None:
        _GEMM_OPS = {
            Op.get("tl.tileop.gemm"),
            Op.get("tl.tileop.wgmma_gemm"),
            Op.get("tl.tileop.tcgen05_gemm"),
        }
    return _GEMM_OPS


def _extract_buffer_var_from_region(region_call):
    if not isinstance(region_call, tir.Call):
        return None
    if len(region_call.args) < 1:
        return None
    buf_load = region_call.args[0]
    if isinstance(buf_load, tir.BufferLoad):
        return buf_load.buffer.data
    return None


def _collect_fragment_gemm_accum_vars(body: tir.Stmt) -> set:
    accum_vars: set = set()
    gemm_ops = _get_gemm_ops()

    def _visitor(stmt):
        if isinstance(stmt, tir.Evaluate) and isinstance(stmt.value, tir.Call):
            call = stmt.value
            if call.op in gemm_ops and len(call.args) >= 3:
                var = _extract_buffer_var_from_region(call.args[2])
                if var is not None and hasattr(var, "type_annotation"):
                    ta = var.type_annotation
                    if ta is not None and hasattr(ta, "storage_scope") and ta.storage_scope == "local.fragment":
                        accum_vars.add(var)

    tir.stmt_functor.post_order_visit(body, _visitor)
    return accum_vars


def _remap_buffer(buf, var_map, num_warps=1):
    old_data = buf.data
    new_data = var_map.get(old_data, None)
    if new_data is None:
        return buf
    total = 1
    for s in buf.shape:
        total *= s.value if isinstance(s, tir.IntImm) else s
    new_total = total // num_warps if num_warps > 1 else total
    return tir.decl_buffer(
        [tir.IntImm("int32", new_total)],
        buf.dtype,
        buf.name,
        data=new_data,
        scope="metal.cooperative_tensor",
        data_alignment=buf.data_alignment,
        offset_factor=buf.offset_factor,
    )


def _is_fill_on_accum_var(stmt, accum_vars):
    """Check if stmt is a tl.fill call targeting one of the resized accum buffers."""
    if not isinstance(stmt, tir.Evaluate):
        return False
    call = stmt.value
    if not isinstance(call, tir.Call):
        return False
    op_name = str(call.op) if hasattr(call.op, "__str__") else ""
    if "fill" not in op_name:
        return False
    if len(call.args) < 1:
        return False
    arg0 = call.args[0]
    if isinstance(arg0, tir.BufferLoad):
        return arg0.buffer.data in accum_vars
    if isinstance(arg0, tir.Call):
        if len(arg0.args) >= 1:
            inner = arg0.args[0]
            if isinstance(inner, tir.BufferLoad):
                return inner.buffer.data in accum_vars
            if isinstance(inner, tir.Var):
                return inner in accum_vars
    return False


def _remove_fills_on_accum(body, accum_vars):
    """Remove fill/clear ops on resized accum buffers (gemm handles its own clear)."""

    def _pre_order(stmt):
        if _is_fill_on_accum_var(stmt, accum_vars):
            return tir.Evaluate(tir.const(0, "int32"))
        if isinstance(stmt, tir.SeqStmt):
            new_stmts = []
            changed = False
            for s in stmt.seq:
                if _is_fill_on_accum_var(s, accum_vars):
                    changed = True
                else:
                    new_stmts.append(s)
            if changed:
                if len(new_stmts) == 0:
                    return tir.Evaluate(tir.const(0, "int32"))
                if len(new_stmts) == 1:
                    return new_stmts[0]
                return tir.SeqStmt(new_stmts)
        return None

    return tir.stmt_functor.ir_transform(body, _pre_order, None, ["tir.Evaluate", "tir.SeqStmt"])


def _rewrite_scope(body, var_map, num_warps=1):
    buf_map = {}

    def _pre_order(stmt):
        if isinstance(stmt, tir.Block):
            new_alloc_bufs = []
            changed = False
            for buf in stmt.alloc_buffers:
                new_buf = _remap_buffer(buf, var_map, num_warps)
                new_alloc_bufs.append(new_buf)
                if not new_buf.same_as(buf):
                    buf_map[buf] = new_buf
                    changed = True
            if changed:
                new_body = tir.stmt_functor.substitute(stmt.body, var_map)
                new_body = _remove_fills_on_accum(new_body, set(var_map.values()))
                new_block = tir.Block(
                    stmt.iter_vars,
                    stmt.reads,
                    stmt.writes,
                    stmt.name_hint,
                    new_body,
                    stmt.init,
                    new_alloc_bufs,
                    stmt.match_buffers,
                    stmt.annotations,
                )
                return (
                    tir.BlockRealize(
                        stmt.iter_vars,
                        tir.const(True, "bool"),
                        new_block,
                    )
                    if False
                    else new_block
                )
        elif isinstance(stmt, tir.Allocate):
            new_var = var_map.get(stmt.buffer_var, None)
            if new_var is not None:
                new_body = tir.stmt_functor.substitute(stmt.body, var_map)
                new_body = _remove_fills_on_accum(new_body, set(var_map.values()))
                total = 1
                for ext in stmt.extents:
                    total *= ext.value if isinstance(ext, tir.IntImm) else ext
                new_total = total // num_warps if num_warps > 1 else total
                new_extents = [tir.IntImm("int32", new_total)]
                return tir.Allocate(new_var, stmt.dtype, new_extents, stmt.condition, new_body, stmt.annotations)
        return None

    return tir.stmt_functor.ir_transform(body, _pre_order, None, ["tir.Block", "tir.Allocate"])


def _get_num_warps(func):
    warp_size = 32
    num_threads = None

    def _visitor(stmt):
        nonlocal num_threads
        if isinstance(stmt, tir.AttrStmt):
            if stmt.attr_key == "thread_extent":
                if hasattr(stmt.node, "thread_tag") and "threadIdx.x" in str(stmt.node.thread_tag):
                    val = stmt.value
                    if isinstance(val, tir.IntImm):
                        num_threads = val.value

    tir.stmt_functor.post_order_visit(func.body, _visitor)
    if num_threads is not None:
        return num_threads // warp_size
    return 1


@prim_func_pass(opt_level=0, name="tl.MetalFragmentToCooperativeTensor")
class MetalFragmentToCooperativeTensor:
    def transform_function(self, func: tir.PrimFunc, mod: IRModule, ctx) -> tir.PrimFunc:
        target = func.attrs.get("target", None)
        if target is None or target.kind.name != "metal":
            return func

        accum_vars = _collect_fragment_gemm_accum_vars(func.body)
        if not accum_vars:
            return func

        num_warps = _get_num_warps(func)

        var_map: dict = {}
        for var in accum_vars:
            ptr_type = var.type_annotation
            new_ptr = PointerType(ptr_type.element_type, "metal.cooperative_tensor")
            new_var = tir.Var(var.name, new_ptr)
            var_map[var] = new_var

        new_body = _rewrite_scope(func.body, var_map, num_warps)
        all_accum_vars = set(var_map.keys()) | set(var_map.values())
        new_body = _remove_fills_on_accum(new_body, all_accum_vars)
        return func.with_body(new_body)


# Keep backward-compatible alias
MetalFragmentToSimdgroup = MetalFragmentToCooperativeTensor
