#ifndef VIZION_CORE_H
#define VIZION_CORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  if defined(VIZION_BUILD)
#    define VIZION_EXPORT __declspec(dllexport)
#  else
#    define VIZION_EXPORT __declspec(dllimport)
#  endif
#  if defined(_MSC_VER) && defined(_M_IX86)
#    define VIZION_CALL __cdecl
#  else
#    define VIZION_CALL
#  endif
#else
#  define VIZION_EXPORT __attribute__((visibility("default")))
#  define VIZION_CALL
#endif

/* bit0 SSE2 可用，bit1 AVX2 可用，bit2 目前走 AVX2，bit3 走 SSE2，bit4 走純 C。 */
#define VIZION_CPU_SSE2 1
#define VIZION_CPU_AVX2 2
#define VIZION_CPU_SELECTED_AVX2 4
#define VIZION_CPU_SELECTED_SSE2 8
#define VIZION_CPU_SELECTED_SCALAR 16

VIZION_EXPORT int VIZION_CALL vizion_cpu_flags(void);

/* 灰階落在 min 與 max（含）的像素收成水平行程。
   rows 傳 NULL 時只回傳段數。成功回 0，參數錯誤 -1，緩衝不夠 -2。 */
VIZION_EXPORT int VIZION_CALL vizion_threshold_runs(
    const uint8_t* gray, int width, int height, int min_gray, int max_gray,
    int* rows, int* columns, int* lengths, int capacity, int* written);

VIZION_EXPORT int VIZION_CALL vizion_threshold_selftest(void);

#ifdef __cplusplus
}
#endif

#endif
