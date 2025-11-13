#pragma once

#include <musa_runtime.h>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <cstdio>

#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short

#define TL_DEVICE __forceinline__ __device__
#define TL_DEVICE_NOINLINE __noinline__ __device__
#define TL_PATCH

#define TILELANG_CHECK(stmt)                                                   \
  do {                                                                         \
    musaError_t __err = (stmt);                                                \
    if (__err != musaSuccess) {                                                \
      snprintf(error_buf, ERROR_BUF_SIZE, "%s:%d: %s - %s", __FILE__,          \
               __LINE__, musaGetErrorName(__err), musaGetErrorString(__err));  \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define TILELANG_CHECK_LAST_ERROR(kernel_name)                                 \
  do {                                                                         \
    musaError_t __err = musaGetLastError();                                    \
    if (__err != musaSuccess) {                                                \
      snprintf(error_buf, ERROR_BUF_SIZE, kernel_name ": %s - %s",             \
               musaGetErrorName(__err), musaGetErrorString(__err));            \
      return -1;                                                               \
    }                                                                          \
  } while (0)

using bfloat16_t = __mt_bfloat16;
using half_t = __half;
