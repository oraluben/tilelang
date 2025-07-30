#include "codegen_metal.h"
#include "runtime/metal/metal_module.h"

namespace tvm::codegen {

CodeGenTileLangMetal::CodeGenTileLangMetal() = default;

auto CodeGenTileLangMetal::Finish() -> std::string {
  decl_stream << "#include <metal_stdlib>\n";
  decl_stream << "\n";

  return CodeGenC::Finish();
}

void CodeGenTileLangMetal::InitFuncState(const PrimFunc &f) {
  CodeGenC::InitFuncState(f);
}
void CodeGenTileLangMetal::PrintStorageSync(const CallNode *op) {
  CodeGenC::PrintStorageSync(op);
}
void CodeGenTileLangMetal::PrintType(DataType t, std::ostream &os) {
  CodeGenC::PrintType(t, os);
}
void CodeGenTileLangMetal::BindThreadIndex(const IterVar &iv) {
  CodeGenC::BindThreadIndex(iv);
}

// assignment printing
void CodeGenTileLangMetal::PrintSSAAssign(const std::string &target,
                                          const std::string &src,
                                          DataType type) {
  CodeGenC::PrintSSAAssign(target, src, type);
}

void CodeGenTileLangMetal::VisitExpr_(const BroadcastNode *op,
                                      std::ostream &os) {
  CodeGenC::VisitExpr_(op, os);
}
void CodeGenTileLangMetal::VisitExpr_(const CallNode *op, std::ostream &os) {
  CodeGenC::VisitExpr_(op, os);
}
void CodeGenTileLangMetal::VisitExpr_(const BufferLoadNode *op,
                                      std::ostream &os) {
  CodeGenC::VisitExpr_(op, os);
}
void CodeGenTileLangMetal::VisitExpr_(const CastNode *op, std::ostream &os) {
  CodeGenC::VisitExpr_(op, os);
}
void CodeGenTileLangMetal::VisitExpr_(const SelectNode *op, std::ostream &os) {
  CodeGenC::VisitExpr_(op, os);
}
void CodeGenTileLangMetal::VisitExpr_(const FloatImmNode *op,
                                      std::ostream &os) {
  CodeGenC::VisitExpr_(op, os);
}
void CodeGenTileLangMetal::VisitExpr_(const IntImmNode *op, std::ostream &os) {
  CodeGenC::VisitExpr_(op, os);
}

// stmt printing
void CodeGenTileLangMetal::VisitStmt_(const LetStmtNode *op) {}
void CodeGenTileLangMetal::VisitStmt_(const BufferStoreNode *op) {}
void CodeGenTileLangMetal::VisitStmt_(const ForNode *op) {}
void CodeGenTileLangMetal::VisitStmt_(const AllocateNode *op) {}
void CodeGenTileLangMetal::VisitStmt_(const AssertStmtNode *op) {}
void CodeGenTileLangMetal::VisitStmt_(const AllocateConstNode *op) {}
void CodeGenTileLangMetal::VisitStmt_(const WhileNode *op) {}

auto BuildTileLangMetal(const IRModule &mod, const Target &target)
    -> runtime::Module {
  std::unordered_map<std::string, std::string> smap;
  std::unordered_map<std::string, runtime::FunctionInfo> fmap;

  CodeGenTileLangMetal cg;

  std::string code = cg.Finish();

  return runtime::MetalModuleCreate(smap, fmap, "metal", code);
}

const TVM_REGISTER_GLOBAL("target.build.tilelang_metal")
    .set_body_typed(BuildTileLangMetal);

} // namespace tvm::codegen
