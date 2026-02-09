/**
 * \file nvrtc.cc
 * \brief Implementation of NVRTC API stub library.
 *
 * This file resolves NVRTC symbols that are already loaded in the current
 * process (e.g., by PyTorch) using dlsym(RTLD_DEFAULT, ...).
 *
 * Unlike the CUDA driver stub (cuda.cc) which uses dlopen() to load
 * libcuda.so, NVRTC symbols are expected to already be in memory.
 * Using RTLD_DEFAULT ensures we reuse the exact same libnvrtc that the
 * framework (e.g., torch) has already loaded, avoiding version mismatches
 * that could occur if we dlopen()-ed a different copy.
 */

#include "nvrtc.h"

#include <dlfcn.h>
#include <stdexcept>
#include <string>

namespace tvm::tl::nvrtc {

namespace {

/**
 * \brief Get symbol already loaded in the process, returning nullptr on
 * failure.
 *
 * Uses RTLD_DEFAULT to search all shared objects in the process — this
 * finds the libnvrtc symbols that torch (or another framework) has already
 * loaded, without opening a separate copy of the library.
 */
template <typename T> T get_symbol(const char *name) {
  // Clear any existing error
  (void)dlerror();
  void *sym = dlsym(RTLD_DEFAULT, name);
  // Check for error (symbol could legitimately be nullptr in some cases)
  const char *error = dlerror();
  if (error != nullptr) {
    return nullptr;
  }
  return reinterpret_cast<T>(sym);
}

/**
 * \brief Create and initialize the NVRTCApi singleton.
 *
 * Resolves all NVRTC function symbols from the already-loaded process image.
 * Required symbols that are missing will cause an exception.
 *
 * \return The initialized NVRTCApi instance.
 * \throws std::runtime_error if a required symbol is missing.
 */
NVRTCApi create_nvrtc_api() {
  NVRTCApi api{};

// Lookup required symbols - throw if missing
#define LOOKUP_REQUIRED(name)                                                  \
  api.name##_ = get_symbol<decltype(&name)>(#name);                            \
  if (api.name##_ == nullptr) {                                                \
    const char *error = dlerror();                                             \
    throw std::runtime_error(                                                  \
        std::string("Failed to load required NVRTC symbol: ") + #name +        \
        ". Error: " + (error ? error : "unknown"));                            \
  }
  TILELANG_LIBNVRTC_API_REQUIRED(LOOKUP_REQUIRED)
#undef LOOKUP_REQUIRED

  return api;
}

} // namespace

bool NVRTCApi::is_available() {
  // Check if at least one NVRTC symbol is already loaded in the process
  return dlsym(RTLD_DEFAULT, "nvrtcCreateProgram") != nullptr;
}

NVRTCApi *NVRTCApi::get() {
  static NVRTCApi singleton = create_nvrtc_api();

  if (!is_available()) {
    throw std::runtime_error(
        "NVRTC symbols not found in the current process. "
        "Please ensure a CUDA runtime (e.g., PyTorch) has loaded libnvrtc.");
  }

  return &singleton;
}

} // namespace tvm::tl::nvrtc

// ============================================================================
// Global wrapper function implementations
// ============================================================================
// These are the implementations for the extern "C" functions declared in the
// header. They provide ABI-compatible replacements for libnvrtc.so functions.

using tvm::tl::nvrtc::NVRTCApi;

extern "C" {

const char *nvrtcGetErrorString(nvrtcResult result) {
  return NVRTCApi::get()->nvrtcGetErrorString_(result);
}

nvrtcResult nvrtcCreateProgram(nvrtcProgram *prog, const char *src,
                               const char *name, int numHeaders,
                               const char *const *headers,
                               const char *const *includeNames) {
  return NVRTCApi::get()->nvrtcCreateProgram_(prog, src, name, numHeaders,
                                              headers, includeNames);
}

nvrtcResult nvrtcDestroyProgram(nvrtcProgram *prog) {
  return NVRTCApi::get()->nvrtcDestroyProgram_(prog);
}

nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions,
                                const char *const *options) {
  return NVRTCApi::get()->nvrtcCompileProgram_(prog, numOptions, options);
}

nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t *logSizeRet) {
  return NVRTCApi::get()->nvrtcGetProgramLogSize_(prog, logSizeRet);
}

nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char *log) {
  return NVRTCApi::get()->nvrtcGetProgramLog_(prog, log);
}

nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t *ptxSizeRet) {
  return NVRTCApi::get()->nvrtcGetPTXSize_(prog, ptxSizeRet);
}

nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char *ptx) {
  return NVRTCApi::get()->nvrtcGetPTX_(prog, ptx);
}

} // extern "C"
