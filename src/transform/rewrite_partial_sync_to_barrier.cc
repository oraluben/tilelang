/*!
 * \file rewrite_partial_sync_to_barrier.cc
 * \brief Rewrite partial shared sync into mbarrier arrive/wait for MUSA.
 */
#include "../op/builtin.h"
#include "tvm/tir/builtin.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/op.h"
#include "tvm/tir/stmt_functor.h"
#include "tvm/tir/transform.h"
#include <tvm/ffi/reflection/registry.h>

#include <algorithm>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

class PartialSyncCollector : public StmtExprVisitor {
public:
  void VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.same_as(builtin::tvm_storage_sync())) {
        HandleStorageSync(call);
      } else if (call->op.same_as(builtin::create_barriers())) {
        HandleCreateBarriers(call);
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }


  const std::unordered_map<int, PrimExpr> &partial_syncs() const {
    return partial_syncs_;
  }
  int barrier_count() const { return barrier_count_; }

private:
  void HandleStorageSync(const CallNode *call) {
    if (call->args.size() != 3)
      return;
    const auto *scope = call->args[0].as<StringImmNode>();
    if (!scope)
      return;
    if (scope->value != "shared" && scope->value != "shared.dyn")
      return;
    const auto *barrier_id = call->args[1].as<IntImmNode>();
    if (!barrier_id)
      return;
    partial_syncs_[static_cast<int>(barrier_id->value)] = call->args[2];
  }

  void HandleCreateBarriers(const CallNode *call) {
    if (call->args.size() != 1)
      return;
    if (const auto *n = call->args[0].as<IntImmNode>()) {
      barrier_count_ += static_cast<int>(n->value);
    }
  }

  std::unordered_map<int, PrimExpr> partial_syncs_;
  int barrier_count_{0};
};

class PartialSyncRewriter : public StmtExprMutator {
public:
  PartialSyncRewriter(std::unordered_map<int, int> id_remap,
                      std::vector<std::pair<int, PrimExpr>> barrier_inits,
                      int new_barrier_count)
      : id_remap_(std::move(id_remap)),
        barrier_inits_(std::move(barrier_inits)),
        new_barrier_count_(new_barrier_count) {}

  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.same_as(builtin::tvm_storage_sync())) {
        if (auto rewritten = RewriteStorageSync(call)) {
          return rewritten.value();
        }
      } else if (call->op.same_as(builtin::create_barriers())) {
        // Drop existing create_barriers; we'll emit a single unified one later.
        return Stmt();
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    if (!init_inserted_ && ShouldInsertInit(op->condition) &&
        !barrier_inits_.empty()) {
      auto then_case = StmtExprMutator::VisitStmt(op->then_case);
      Array<Stmt> stmts = MakeInitStmts();
      stmts.push_back(then_case);
      auto seq = stmts.size() == 1 ? stmts[0] : SeqStmt(stmts);
      init_inserted_ = true;
      Stmt else_case;
      if (op->else_case) {
        else_case = StmtExprMutator::VisitStmt(op->else_case.value());
      }
      return IfThenElse(op->condition, seq, else_case);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  bool init_inserted() const { return init_inserted_; }

  Array<Stmt> MakeInitStmts() {
    Array<Stmt> stmts;
    for (const auto &[id, thread_count] : barrier_inits_) {
      auto barrier = Call(DataType::Handle(), get_mbarrier(),
                          {IntImm(DataType::Int(32), id)});
      auto count = VisitExpr(thread_count);
      auto init =
          Call(DataType::Handle(), builtin::ptx_init_barrier_thread_count(),
               {barrier, count});
      stmts.push_back(Evaluate(init));
    }
    return stmts;
  }

private:
  std::optional<Stmt> RewriteStorageSync(const CallNode *call) {
    if (call->args.size() != 3)
      return std::nullopt;
    const auto *scope = call->args[0].as<StringImmNode>();
    if (!scope || (scope->value != "shared" && scope->value != "shared.dyn"))
      return std::nullopt;
    const auto *barrier_id = call->args[1].as<IntImmNode>();
    if (!barrier_id)
      return std::nullopt;
    auto it = id_remap_.find(static_cast<int>(barrier_id->value));
    if (it == id_remap_.end())
      return std::nullopt;
    int new_id = it->second;
    Array<PrimExpr> args = {
        Call(DataType::Handle(), get_mbarrier(),
             {IntImm(DataType::Int(32), new_id)}),
        VisitExpr(call->args[2])};
    auto new_call = Call(call->dtype, musa_sync(), args);
    return Evaluate(new_call);
  }

  // todo: 修改成不需要找哪里可以插入 barrier init，而是创建一个块，插入 barrier init
  bool ShouldInsertInit(const PrimExpr &cond) {
    if (const auto *call = cond.as<CallNode>()) {
      if (call->op.same_as(tl_shuffle_elect()) && !call->args.empty()) {
        if (const auto *imm = call->args[0].as<IntImmNode>()) {
          return imm->value == 0;
        }
      }
    } else if (const auto *eq = cond.as<EQNode>()) {
      if (const auto *rhs = eq->b.as<IntImmNode>()) {
        if (rhs->value == 0)
          return true;
      } else if (const auto *lhs = eq->a.as<IntImmNode>()) {
        if (lhs->value == 0)
          return true;
      }
    }
    return false;
  }

  std::unordered_map<int, int> id_remap_;
  std::vector<std::pair<int, PrimExpr>> barrier_inits_;
  int new_barrier_count_{0};
  bool init_inserted_{false};
};

PrimFunc RewritePartialSyncToBarrier(PrimFunc f) {
  PartialSyncCollector collector;
  collector(f->body);
  if (collector.partial_syncs().empty()) {
    return f;
  }

  std::vector<std::pair<int, PrimExpr>> syncs(collector.partial_syncs().begin(),
                                              collector.partial_syncs().end());
  std::sort(syncs.begin(), syncs.end(),
            [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

  int base_count = collector.barrier_count();

  std::unordered_map<int, int> id_remap;
  std::vector<std::pair<int, PrimExpr>> barrier_inits;
  int next_barrier_id = base_count;
  for (const auto &[old_id, thread_count] : syncs) {
    id_remap[old_id] = next_barrier_id;
    barrier_inits.push_back({next_barrier_id, thread_count});
    ++next_barrier_id;
  }
  int new_barrier_count = next_barrier_id;

  PartialSyncRewriter rewriter(std::move(id_remap), barrier_inits,
                               new_barrier_count);
  auto *n = f.CopyOnWrite();
  Stmt body = rewriter(std::move(n->body));

  Array<Stmt> prefix;
  // Always insert a fresh create_barriers with the new count.
  auto create =
      Evaluate(Call(DataType::Handle(), builtin::create_barriers(),
                    {IntImm(DataType::Int(32), new_barrier_count)}));
  prefix.push_back(create);

  if (!rewriter.init_inserted()) {
    auto cond = Call(DataType::Bool(), tl_shuffle_elect(),
                     {IntImm(DataType::Int(32), 0)});
    auto init_stmts = rewriter.MakeInitStmts();
    auto seq = init_stmts.size() == 1 ? init_stmts[0] : SeqStmt(init_stmts);
    prefix.push_back(IfThenElse(cond, seq));
  }

  if (!prefix.empty()) {
    prefix.push_back(body);
    body = SeqStmt(prefix);
  }

  n->body = std::move(body);
  return f;
}

namespace transform {

tvm::transform::Pass RewriteMUSAPartialSync() {
  auto pass_func = [](PrimFunc f, IRModule m,
                     const tvm::transform::PassContext &ctx) {
    return RewritePartialSyncToBarrier(std::move(f));
  };
  return tir::transform::CreatePrimFuncPass(pass_func, 0,
                                            "tl.RewriteMUSAPartialSync", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.RewriteMUSAPartialSync",
                        RewriteMUSAPartialSync);
}

} // namespace transform
} // namespace tl
} // namespace tvm
