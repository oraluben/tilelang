#include "codegen_metal.h"
#include "runtime/metal/metal_module.h"


namespace tvm {
namespace codegen {

runtime::Module BuildTileLangMetal(const IRModule& mod, const Target& target) {
    std::unordered_map<std::string, std::string> smap;
    std::unordered_map<std::string, runtime::FunctionInfo> fmap;


  return runtime::MetalModuleCreate(smap, fmap, "metal", "");
}

TVM_REGISTER_GLOBAL("target.build.tilelang_metal")
    .set_body_typed(BuildTileLangMetal);

} // namespace codegen
} // namespace tvm
