/**
 * \file nvrtc.h
 * \brief Stub library for lazy loading libnvrtc.so at runtime.
 *
 * This library provides drop-in replacements for NVRTC API functions.
 * It allows tilelang to be built against a single CUDA version while
 * working with both CUDA 12 and 13 at runtime, since the actual
 * libnvrtc.so is loaded lazily on first API call via dlopen().
 *
 * Usage:
 *
 * 1. Link against libnvrtc_stub.so instead of libnvrtc.so
 *
 * 2. Call NVRTC API functions normally - they are provided as
 *    exported global functions with C linkage:
 *
 *    ```cpp
 *    #include "target/stubs/nvrtc.h"
 *    nvrtcResult result = nvrtcCreateProgram(&prog, src, name, 0, NULL, NULL);
 *    ```
 *
 * 3. For advanced use, access the singleton directly:
 *
 *    ```cpp
 *    auto* api = tvm::tl::nvrtc::NVRTCApi::get();
 *    bool available = tvm::tl::nvrtc::NVRTCApi::is_available();
 *    ```
 */

#pragma once

// Define guard before including vendor/nvrtc.h
// This ensures vendor/nvrtc.h can only be included through this stub header.
#define _TILELANG_NVRTC_STUB_INCLUDE_GUARD

#include "vendor/nvrtc.h" // include the NVRTC API types

#undef _TILELANG_NVRTC_STUB_INCLUDE_GUARD

// Symbol visibility macros for shared library export
#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef TILELANG_NVRTC_STUB_EXPORTS
#define TILELANG_NVRTC_STUB_API __declspec(dllexport)
#else
#define TILELANG_NVRTC_STUB_API __declspec(dllimport)
#endif
#else
#define TILELANG_NVRTC_STUB_API __attribute__((visibility("default")))
#endif

// X-macro for listing all required NVRTC API functions.
// Format: _(function_name)
// These are the core functions used by TVM's CUDA compilation path.
#define TILELANG_LIBNVRTC_API_REQUIRED(_)                                      \
  _(nvrtcGetErrorString)                                                       \
  _(nvrtcCreateProgram)                                                        \
  _(nvrtcDestroyProgram)                                                       \
  _(nvrtcCompileProgram)                                                       \
  _(nvrtcGetProgramLogSize)                                                    \
  _(nvrtcGetProgramLog)                                                        \
  _(nvrtcGetPTXSize)                                                           \
  _(nvrtcGetPTX)

namespace tvm::tl::nvrtc {

/**
 * \brief NVRTC API accessor struct with lazy loading support.
 *
 * This struct provides lazy loading of libnvrtc.so symbols at first use,
 * allowing tilelang to be built against one CUDA version while working
 * with another at runtime.
 *
 * Usage:
 *   nvrtcResult result = NVRTCApi::get()->nvrtcCreateProgram_(...);
 */
struct TILELANG_NVRTC_STUB_API NVRTCApi {
// Create function pointer members for each API function
#define CREATE_MEMBER(name) decltype(&name) name##_;
  TILELANG_LIBNVRTC_API_REQUIRED(CREATE_MEMBER)
#undef CREATE_MEMBER

  /**
   * \brief Get the singleton instance of NVRTCApi.
   *
   * On first call, this loads libnvrtc.so and resolves all symbols.
   * Subsequent calls return the cached instance.
   *
   * \return Pointer to the singleton NVRTCApi instance.
   * \throws std::runtime_error if libnvrtc.so cannot be loaded or
   *         required symbols are missing.
   */
  static NVRTCApi *get();

  /**
   * \brief Check if NVRTC library is available without throwing.
   *
   * \return true if libnvrtc.so can be loaded, false otherwise.
   */
  static bool is_available();

  /**
   * \brief Get the raw library handle for libnvrtc.so.
   *
   * \return The dlopen handle, or nullptr if not loaded.
   */
  static void *get_handle();
};

} // namespace tvm::tl::nvrtc

// ============================================================================
// Global wrapper functions for lazy-loaded NVRTC API
// ============================================================================
// These functions provide drop-in replacements for NVRTC API calls.
// They are exported from the stub library and can be linked against directly.
// The implementations are in nvrtc.cc.

extern "C" {

TILELANG_NVRTC_STUB_API const char *nvrtcGetErrorString(nvrtcResult result);
TILELANG_NVRTC_STUB_API nvrtcResult nvrtcCreateProgram(
    nvrtcProgram *prog, const char *src, const char *name, int numHeaders,
    const char *const *headers, const char *const *includeNames);
TILELANG_NVRTC_STUB_API nvrtcResult nvrtcDestroyProgram(nvrtcProgram *prog);
TILELANG_NVRTC_STUB_API nvrtcResult
nvrtcCompileProgram(nvrtcProgram prog, int numOptions,
                    const char *const *options);
TILELANG_NVRTC_STUB_API nvrtcResult
nvrtcGetProgramLogSize(nvrtcProgram prog, size_t *logSizeRet);
TILELANG_NVRTC_STUB_API nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog,
                                                       char *log);
TILELANG_NVRTC_STUB_API nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog,
                                                    size_t *ptxSizeRet);
TILELANG_NVRTC_STUB_API nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char *ptx);

} // extern "C"
