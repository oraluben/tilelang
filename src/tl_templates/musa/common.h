#pragma once

#include <cstdio>       // snprintf
#include <sstream>      // std::stringstream
#include <musa_runtime.h>
#include <musa_bf16.h>
#include <musa_fp16.h>

#define TL_HOST_DEVICE __forceinline__ __host__ __device__
#define TL_DEVICE __forceinline__ __device__
#define TL_DEVICE_NOINLINE __noinline__ __device__

#define _AS1 __attribute__((address_space(1)))
#define _AS3 __attribute__((address_space(3)))

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


using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;


using bfloat16_t = __mt_bfloat16;
using half_t = __half;
using  v2i32_t = int32_t __attribute__((vector_size(8)));
using  v3i32_t = int32_t __attribute__((vector_size(12)));
using  v4i32_t = int32_t __attribute__((vector_size(16)));
using  v5i32_t = int32_t __attribute__((vector_size(20)));
using  v8i32_t = int32_t __attribute__((vector_size(32)));
using v16i32_t = int32_t __attribute__((vector_size(64)));
using v32i32_t = int32_t __attribute__((vector_size(128)));

static const int NumThreadsPerWarpBeforeMP31 = 128;
static const int NumThreadsPerWarp = 32;
static const int NumThreadsPerWarpSquad = 128;
static const int NumWarpsPerWarpSquad = NumThreadsPerWarpSquad / NumThreadsPerWarp;
static const int NumThreadsPerHalfWarp = NumThreadsPerWarp / 2;

enum class SmemSwizzleGranularity : uint8_t {
  NONE = 0,
  B16  = 1,
  B32  = 2,
  B64  = 3,
};

enum class SmemSwizzleStride : uint8_t {
  B32  = 0,
  B64  = 1,
  B128 = 2,
  B256 = 3,
};

enum class SmemSwizzleLine : uint8_t {
  B128 = 0,
  B256 = 1,
};

enum class CacheHint : uint8_t {
  CACHE_NONE    = 0,
  CACHE_ONCE    = 1,
  CACHE_NORMAL  = 2,
  CACHE_PERSIST = 3,
};

enum class PrefetchSize : uint8_t {
  NONE = 0,
  B64  = 64,
  B128 = 128,
};

enum class AddressSpace {
  Generic = 0,
  Global  = 1,
  Shared  = 3,
};

template <AddressSpace AS>
TL_HOST_DEVICE constexpr
void __attribute__((address_space(static_cast<int>(AS))))*
make_ptr_with_address_space(uint64_t ptr) {
  return reinterpret_cast<void __attribute__((address_space(static_cast<int>(AS))))*>(ptr);
}

/// MUTE helper to cast SMEM pointer to unsigned
TL_HOST_DEVICE
uint32_t
cast_smem_ptr_to_uint(void const* const ptr)
{
  /// MUTE helper to get SMEM pointer
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}
