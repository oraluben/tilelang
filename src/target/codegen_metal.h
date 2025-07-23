#include <tvm/target/codegen.h>

#include <string>

#include "target/source/codegen_c.h"


namespace tvm {
namespace codegen {

/*!
 * \brief WebGPU code generator.
 *
 * Note WGSL have a different syntax from normal C.
 * We only leverage the C for expression generation and
 * write most of the language generations.
 */
class CodeGenTileLangMetal final : public CodeGenC {
public:
  explicit CodeGenTileLangMetal(Target target);
  // overrides
  std::string Finish() final;
  using CodeGenC::AddFunction;
  runtime::FunctionInfo AddFunction(const PrimFunc &f,
                                    bool skip_readonly_decl); // NOLINT(*)
  void InitFuncState(const PrimFunc &f) final;
  void PrintStorageSync(const CallNode *op) final;    // NOLINT(*)
  void PrintType(DataType t, std::ostream &os) final; // NOLINT(*)
  void BindThreadIndex(const IterVar &iv) final;      // NOLINT(*)

  // assignment printing
  void PrintSSAAssign(const std::string &target, const std::string &src,
                      DataType type) final;

  // overload visitor
  void VisitExpr_(const BroadcastNode *op, std::ostream &os) final; // NOLINT(*)
  void VisitExpr_(const CallNode *op, std::ostream &os) final;      // NOLINT(*)
  void VisitExpr_(const BufferLoadNode *op,
                  std::ostream &os) final;                          // NOLINT(*)
  void VisitExpr_(const CastNode *op, std::ostream &os) final;      // NOLINT(*)
  void VisitExpr_(const SelectNode *op, std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const FloatImmNode *op, std::ostream &os) final;  // NOLINT(*)
  void VisitExpr_(const IntImmNode *op, std::ostream &os) final;    // NOLINT(*)

  // stmt printing
  void VisitStmt_(const LetStmtNode *op) final;
  void VisitStmt_(const BufferStoreNode *op) final;
  void VisitStmt_(const ForNode *op) final;
  void VisitStmt_(const AllocateNode *op) final;
  void VisitStmt_(const AssertStmtNode *op) final;
  void VisitStmt_(const AllocateConstNode *op) final;
  void VisitStmt_(const WhileNode *op) final;
};
} // namespace codegen
} // namespace tvm
