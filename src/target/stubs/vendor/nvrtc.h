/**
 * \file vendor/nvrtc.h
 * \brief Minimal NVRTC type definitions for the stub library.
 *
 * This header provides the subset of NVRTC types and constants that
 * the tilelang/TVM CUDA runtime actually references.  It is NOT the
 * full NVIDIA nvrtc.h — only the stable, publicly-documented ABI
 * surface used by build_cuda_on.cc is reproduced here so that the
 * stub library can be compiled without an NVRTC SDK installation.
 *
 * Guard: this file must only be included through target/stubs/nvrtc.h.
 */

#ifndef _TILELANG_NVRTC_STUB_INCLUDE_GUARD
#error "vendor/nvrtc.h should only be included by target/stubs/nvrtc.h. " \
       "Do not include this file directly and use target/stubs/nvrtc.h instead."
#endif

#ifndef TILELANG_VENDOR_NVRTC_H_
#define TILELANG_VENDOR_NVRTC_H_

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief NVRTC API call result codes.
 */
typedef enum {
  NVRTC_SUCCESS = 0,
  NVRTC_ERROR_OUT_OF_MEMORY = 1,
  NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
  NVRTC_ERROR_INVALID_INPUT = 3,
  NVRTC_ERROR_INVALID_PROGRAM = 4,
  NVRTC_ERROR_INVALID_OPTION = 5,
  NVRTC_ERROR_COMPILATION = 6,
  NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
  NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
  NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
  NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
  NVRTC_ERROR_INTERNAL_ERROR = 11,
} nvrtcResult;

/**
 * \brief Opaque handle to an NVRTC program.
 */
typedef struct _nvrtcProgram *nvrtcProgram;

#ifdef __cplusplus
}
#endif

#endif /* TILELANG_VENDOR_NVRTC_H_ */
