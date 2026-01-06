/*!
 * \file vectorize_boundry_check.cc
 * \brief Vectorize loop with boundry check
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include <utility>

#include "../op/builtin.h"
#include "../op/parallel.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "loop_partition.h"
#include "loop_vectorize.h"
#include "tvm/ffi/cast.h"
#include "tvm/ir/expr.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/stmt.h"

namespace tvm {
namespace tl {

using namespace tir;

class PromoteBoundryCheck : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {

    // Create an instance of the legalizer with the analyzer
    PromoteBoundryCheck substituter;
    // Get a mutable copy of the function node
    PrimFuncNode *fptr = f.CopyOnWrite();
    // Apply the legalizer to the function body
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

private:
  bool _all_add_with(const PrimExpr expr, const Var var) {
    if (const AddNode *add_node = expr.as<AddNode>()) {
        return _all_add_with(add_node->a, var) && _all_add_with(add_node->b, var);
    } else if (expr->IsInstance<SubNode>()) {
        return false;
    } else if (expr->IsInstance<ModNode>() || expr->IsInstance<MulNode>() || expr->IsInstance<DivNode>()) {
        return true;
    }
    return false;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    /*
    for vec in T.vectorized(extent):
        if <x> + vec < N:
            block
        else:
            block2
    =>
    if <x> + extent < N:
        for vec in T.vectorized(extent):
            block
    else:
        for vec in T.vectorized(extent):
            if <x> + vec < N:
                block
            else:
                block2
     */
    For forStmt = Downcast<For>(StmtExprMutator::VisitStmt_(op));

    std::ostringstream code;

    if (const IfThenElseNode *if_node = forStmt->body.as<IfThenElseNode>()) {
        if (const LTNode *lt_node = if_node->condition.as<LTNode>()) {
            if (is_const_number(lt_node->b) && _all_add_with(lt_node->a, forStmt->loop_var)) {
                
            }
        }
    }

    // LOG(WARNING) << forStmt << forStmt->annotations << '\n';

    if (const auto f =
          ffi::Function::GetGlobal("tilelang_dbg")) {
        code << "loop_var: " << forStmt->loop_var << '\n';
        code << "extent: " << forStmt->extent << '\n';
        code << "kind: " << forStmt->kind << '\n';
        code << "min: " << forStmt->min << '\n';
        code << "body (" << forStmt->body.GetTypeKey() << "): " << forStmt->body << '\n';
        code << "";
        (*f)(code.str());
    }

    return forStmt;
  }
};

// Create a pass that legalizes vectorized loops in the IRModule
tvm::transform::Pass VectorizeWithBoundry() {
  using namespace tir::transform;
  // Define the transformation function to be applied
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    return PromoteBoundryCheck::Substitute(std::move(f));
  };
  // Create and return a PrimFunc pass with the transformation function
  return CreatePrimFuncPass(pass_func, 0, "tl.VectorizeWithBoundry", {});
}

// Register the pass globally so it can be used in the compilation pipeline
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.VectorizeWithBoundry",
                        VectorizeWithBoundry);
}

} // namespace tl
} // namespace tvm
