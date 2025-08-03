#include <tvm/target/codegen.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <string>
// #include <unordered_map>

#include "target/source/codegen_c.h"

namespace tvm::codegen {

/*!
 * \brief WebGPU code generator.
 *
 * Note WGSL have a different syntax from normal C.
 * We only leverage the C for expression generation and
 * write most of the language generations.
 */
class CodeGenTileLangMetal final : public CodeGenC {
public:
  CodeGenTileLangMetal();
  void AddFunction(const GlobalVar &gvar, const PrimFunc &f) override;

  // overrides
  auto Finish() -> std::string final;
  void InitFuncState(const PrimFunc &f) final;
  void PrintStorageSync(const CallNode *op) final;    // NOLINT(*)
  void PrintType(DataType t, std::ostream &os) final; // NOLINT(*)
  void BindThreadIndex(const IterVar &iv) final;      // NOLINT(*)

  // assignment printing
  void PrintSSAAssign(const std::string &target, const std::string &src,
                      DataType type) final;
};
} // namespace tvm::codegen
