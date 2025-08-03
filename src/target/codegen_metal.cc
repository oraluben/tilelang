#include "codegen_metal.h"
#include "runtime/metal/metal_module.h"
#include "target/source/codegen_c.h"
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>

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
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK_EQ(lanes, 1) << "do not yet support vector types";
    os << "void*";
    return;
  }

  if (t.is_void()) {
    os << "void";
    return;
  }
  if (t == DataType::Bool()) {
    os << "bool";
    return;
  }
  bool fail = false;
  if (t.is_float()) {
    // Need to care about sizes and alignment of half3/float3 because tir
    // representation might not be aware of Metal half3/float3 details and can
    // treat them as just three elements, while sizes and alignmnents of
    // half3/float3 are one element more (half3-8 bytes/ float13 - 16bytes).
    // Example of problematic pattern: filling of threadgroup packed array using
    // float3 elements by threads concurrently can lead to datarace and wrong
    // data in threadgroup shared array. packed_(half3/float3) are exactly
    // datatypes dealing with 3 elements and per-element alignment
    if (lanes == 3) {
      os << "packed_";
    }
    switch (t.bits()) {
    case 16:
      os << "half";
      break;
    case 32:
      os << "float";
      break;
    default:
      fail = true;
      break;
    }
    if (!fail && lanes == 1)
      return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << 'u';
    }
    switch (t.bits()) {
    case 8:
      os << "char";
      break;
    case 16:
      os << "short";
      break;
    case 32:
      os << "int";
      break;
    case 64:
      os << "long";
      break;
    case 1:
      os << "bool";
      break;
    default:
      fail = true;
      break;
    }
    if (!fail && lanes == 1)
      return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  } else if (t.is_bfloat16()) {
    os << "bfloat";
    return;
  }
  LOG(FATAL) << "Cannot convert type " << t << " to Metal type";
}
void CodeGenTileLangMetal::BindThreadIndex(const IterVar &iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  var_idmap_[iv->var.get()] =
      CastFromTo(iv->thread_tag, DataType::UInt(32), iv->var.dtype());
}

// assignment printing
void CodeGenTileLangMetal::PrintSSAAssign(const std::string &target,
                                          const std::string &src,
                                          DataType type) {
  printf("PrintSSAAssign\n");
  CodeGenC::PrintSSAAssign(target, src, type);
}

void CodeGenTileLangMetal::AddFunction(const GlobalVar &gvar,
                                       const PrimFunc &f) {
  // If the function has already been forward-declared, this is a
  // no-op.
  CodeGenC::DeclareFunction(gvar, f);
  // clear previous generated state.
  this->InitFuncState(f);
  // reserve keywords
  ReserveKeywordsAsUnique();

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";
  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);

  this->PrintFuncPrefix(decl_stream);
  CodeGenC::PrintType(f->ret_type, decl_stream);
  this->PrintExtraAttrs(f, decl_stream);

  decl_stream << " " << static_cast<std::string>(global_symbol.value()) << "(";

  for (size_t i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];
    std::string vid = AllocVarID(v.get());
    printf("AddFunction handle arg: %s\n", vid.c_str());
    if (i != 0) {
      decl_stream << ", ";
    }
    decl_stream << "device ";
    CodeGenC::PrintType(GetType(v), decl_stream);
    decl_stream << " " << vid;
  }

  decl_stream << ") {\n";
  this->PreFunctionBody(f);

  printf("AddFunction 6: %s\n", decl_stream.str().c_str());
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  printf("AddFunction 7\n");
  this->EndScope(func_scope);
  printf("AddFunction 8\n");
  this->PrintIndent();
  printf("AddFunction 9\n");
  this->decl_stream << "}\n\n";
}

auto BuildTileLangMetal(const IRModule &mod, const Target &target)
    -> runtime::Module {
  std::unordered_map<std::string, std::string> smap;
  std::unordered_map<std::string, runtime::FunctionInfo> fmap;

  CodeGenTileLangMetal cg;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangMetal: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();

  return runtime::MetalModuleCreate(smap, fmap, "metal", code);
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.tilelang_metal", BuildTileLangMetal);
});

} // namespace tvm::codegen
