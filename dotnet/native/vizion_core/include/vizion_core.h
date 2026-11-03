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

/* 灰階矩形原點是 k/2，不把偶數補成奇數。dilate 非 0 表示膨脹，圖外膨脹當 0、侵蝕當 255。 */
VIZION_EXPORT int VIZION_CALL vizion_gray_rect(
    const uint8_t* src, uint8_t* dst, int width, int height, int mask_height, int mask_width, int dilate);

VIZION_EXPORT int VIZION_CALL vizion_gray_ellipse(
    const uint8_t* src, uint8_t* dst, int width, int height, int radius, int dilate);

/* 區域矩形偶數寬高會加 1。out 傳 NULL 時只回傳段數。 */
VIZION_EXPORT int VIZION_CALL vizion_region_rect(
    const int* rows, const int* columns, const int* lengths, int count,
    int width, int height, int kernel_width, int kernel_height, int dilate,
    int* out_rows, int* out_columns, int* out_lengths, int capacity, int* written);

VIZION_EXPORT int VIZION_CALL vizion_region_disk(
    const int* rows, const int* columns, const int* lengths, int count,
    int width, int height, double radius, int dilate,
    int* out_rows, int* out_columns, int* out_lengths, int capacity, int* written);

VIZION_EXPORT int VIZION_CALL vizion_morph_selftest(void);

#ifdef __cplusplus
}
#endif

#endif
