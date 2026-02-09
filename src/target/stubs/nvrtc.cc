/**
 * \file nvrtc.cc
 * \brief Implementation of NVRTC API stub library.
 *
 * This file implements lazy loading of libnvrtc.so and provides global wrapper
 * functions that serve as drop-in replacements for the NVRTC API.
 *
 * The library is loaded on first API call using dlopen(). This allows
 * tilelang to be built against one CUDA version while working with
 * another at runtime, since the versioned libnvrtc.so (e.g., libnvrtc.so.12
 * or libnvrtc.so.13) is resolved dynamically.
 */

#include "nvrtc.h"

#include <dlfcn.h>
#include <stdexcept>
#include <string>

namespace tvm::tl::nvrtc {

namespace {

// Library names to try loading (in order of preference)
constexpr const char *kLibNvrtcPaths[] = {
    "libnvrtc.so.12", // CUDA 12.x
    "libnvrtc.so.13", // CUDA 13.x
    "libnvrtc.so",    // Unversioned library
};

/**
 * \brief Try to load libnvrtc.so from various paths.
 * \return The dlopen handle, or nullptr if loading failed.
 */
void *try_load_libnvrtc() {
  void *handle = nullptr;
  for (const char *path : kLibNvrtcPaths) {
    handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    if (handle != nullptr) {
      break;
    }
  }
  return handle;
}

/**
 * \brief Get symbol from library handle, returning nullptr on failure.
 */
template <typename T> T get_symbol(void *handle, const char *name) {
  // Clear any existing error
  (void)dlerror();
  void *sym = dlsym(handle, name);
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
 * This function loads libnvrtc.so and resolves all function symbols.
 * Required symbols that are missing will cause an exception.
 *
 * \return The initialized NVRTCApi instance.
 * \throws std::runtime_error if a required symbol is missing.
 */
NVRTCApi create_nvrtc_api() {
  NVRTCApi api{};
  void *handle = NVRTCApi::get_handle();

  if (handle == nullptr) {
    return api;
  }

// Lookup required symbols - throw if missing
#define LOOKUP_REQUIRED(name)                                                  \
  api.name##_ = get_symbol<decltype(&name)>(handle, #name);                    \
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

void *NVRTCApi::get_handle() {
  // Static handle ensures library is loaded only once
  static void *handle = try_load_libnvrtc();
  return handle;
}

bool NVRTCApi::is_available() { return get_handle() != nullptr; }

NVRTCApi *NVRTCApi::get() {
  static NVRTCApi singleton = create_nvrtc_api();

  if (!is_available()) {
    throw std::runtime_error(
        "NVRTC library (libnvrtc.so) not found. "
        "Please ensure CUDA toolkit is installed.");
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
