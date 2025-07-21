
namespace tvm {
namespace codegen {

TVM_REGISTER_GLOBAL("target.build.tilelang_metal")
    .set_body_typed([](IRModule mod, Target target) {
    //   return BuildTileLangWebGPU(mod, target);
    });

} // namespace codegen
} // namespace tvm
