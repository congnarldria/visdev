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

/* 八連通。輸出行程依物件排好，物件順序是掃描時第一次遇到的那段。回傳物件數，失敗 -1。 */
VIZION_EXPORT int VIZION_CALL vizion_connection8(
    const int* rows, const int* columns, const int* lengths, int count, int height,
    int* out_rows, int* out_columns, int* out_lengths, int* out_offsets, int* out_counts);

VIZION_EXPORT int VIZION_CALL vizion_area_center(
    const int* rows, const int* columns, const int* lengths, int count,
    int* area, double* row, double* column);

VIZION_EXPORT int VIZION_CALL vizion_bounds(
    const int* rows, const int* columns, const int* lengths, int count,
    int* row1, int* column1, int* row2, int* column2);

VIZION_EXPORT int VIZION_CALL vizion_region_selftest(void);

#ifdef __cplusplus
}
#endif

#endif
