#include "vizion_core.h"

#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#  include <malloc.h>
#endif

#if defined(_MSC_VER)
#  include <intrin.h>
#else
#  include <emmintrin.h>
#endif

void vizion_line_minmax(const uint8_t* src, uint8_t* dst, int n, int k, int anchor, int dilate, uint8_t* scratch);
int vizion_gray_rect_scalar(const uint8_t* src, uint8_t* dst, int width, int height, int mask_height, int mask_width, int dilate);

static void* vizion_aligned(size_t bytes)
{
    void* pointer = 0;
#if defined(_WIN32)
    pointer = _aligned_malloc(bytes, 32);
#else
    if (posix_memalign(&pointer, 32, bytes))
        return 0;
#endif
    return pointer;
}

static void vizion_aligned_free(void* pointer)
{
#if defined(_WIN32)
    _aligned_free(pointer);
#else
    free(pointer);
#endif
}

static void vizion_line_minmax_sse(const __m128i* src, __m128i* dst, int n, int k, int anchor, int dilate, __m128i* scratch)
{
    int left = anchor;
    int padded = n + k - 1;
    int chunk;
    __m128i border = _mm_set1_epi8(dilate ? 0 : -1);
    __m128i* pad = scratch;
    __m128i* prefix = scratch + padded;
    __m128i* suffix = scratch + padded * 2;
    int i;
    for (i = 0; i < left; i++)
        pad[i] = border;
    memcpy(pad + left, src, (size_t)n * sizeof(__m128i));
    for (i = left + n; i < padded; i++)
        pad[i] = border;
    for (chunk = 0; chunk * k < padded; chunk++) {
        int begin = chunk * k;
        int end = begin + k;
        int j;
        if (end > padded)
            end = padded;
        prefix[begin] = pad[begin];
        for (j = begin + 1; j < end; j++)
            prefix[j] = dilate ? _mm_max_epu8(prefix[j - 1], pad[j]) : _mm_min_epu8(prefix[j - 1], pad[j]);
        suffix[end - 1] = pad[end - 1];
        for (j = end - 2; j >= begin; j--)
            suffix[j] = dilate ? _mm_max_epu8(suffix[j + 1], pad[j]) : _mm_min_epu8(suffix[j + 1], pad[j]);
    }
    for (i = 0; i < n; i++) {
        __m128i a = suffix[i];
        __m128i b = prefix[i + k - 1];
        dst[i] = dilate ? _mm_max_epu8(a, b) : _mm_min_epu8(a, b);
    }
}

static int vizion_horizontal_sse(const uint8_t* src, uint8_t* dst, int width, int height, int mask_width, int dilate)
{
    int anchor = mask_width / 2;
    int padded = width + mask_width;
    int row = 0;
    __m128i* cols = (__m128i*)vizion_aligned((size_t)width * sizeof(__m128i));
    __m128i* filtered = (__m128i*)vizion_aligned((size_t)width * sizeof(__m128i));
    __m128i* scratch = (__m128i*)vizion_aligned((size_t)padded * 3 * sizeof(__m128i));
    uint8_t* line_scratch = (uint8_t*)malloc((size_t)padded * 3);
    if (!cols || !filtered || !scratch || !line_scratch) {
    vizion_aligned_free(cols);
    vizion_aligned_free(filtered);
    vizion_aligned_free(scratch);
    free(line_scratch);
    return -1;
}
    for (; row + 16 <= height; row += 16) {
        int col;
        int lane;
        for (col = 0; col < width; col++) {
            uint8_t packed[16];
            for (lane = 0; lane < 16; lane++)
                packed[lane] = src[((size_t)row + (size_t)lane) * (size_t)width + col];
            cols[col] = _mm_loadu_si128((const __m128i*)packed);
        }
        vizion_line_minmax_sse(cols, filtered, width, mask_width, anchor, dilate, scratch);
        for (col = 0; col < width; col++) {
            uint8_t packed[16];
            _mm_storeu_si128((__m128i*)packed, filtered[col]);
            for (lane = 0; lane < 16; lane++)
                dst[((size_t)row + (size_t)lane) * (size_t)width + col] = packed[lane];
        }
    }
    for (; row < height; row++)
        vizion_line_minmax(src + (size_t)row * (size_t)width, dst + (size_t)row * (size_t)width, width, mask_width, anchor, dilate, line_scratch);
    vizion_aligned_free(cols);
    vizion_aligned_free(filtered);
    vizion_aligned_free(scratch);
    free(line_scratch);
    return 0;
}

static int vizion_vertical_sse(const uint8_t* src, uint8_t* dst, int width, int height, int mask_height, int dilate)
{
    int anchor = mask_height / 2;
    int padded = height + mask_height;
    int col = 0;
    __m128i* rows = (__m128i*)vizion_aligned((size_t)height * sizeof(__m128i));
    __m128i* filtered = (__m128i*)vizion_aligned((size_t)height * sizeof(__m128i));
    __m128i* scratch = (__m128i*)vizion_aligned((size_t)padded * 3 * sizeof(__m128i));
    uint8_t* column = (uint8_t*)malloc((size_t)height);
    uint8_t* column_out = (uint8_t*)malloc((size_t)height);
    uint8_t* line_scratch = (uint8_t*)malloc((size_t)padded * 3);
    if (!rows || !filtered || !scratch || !column || !column_out || !line_scratch) {
    vizion_aligned_free(rows);
    vizion_aligned_free(filtered);
    vizion_aligned_free(scratch);
    free(column);
    free(column_out);
    free(line_scratch);
    return -1;
}
    for (; col + 16 <= width; col += 16) {
        int row;
        for (row = 0; row < height; row++)
            rows[row] = _mm_loadu_si128((const __m128i*)(src + (size_t)row * (size_t)width + col));
        vizion_line_minmax_sse(rows, filtered, height, mask_height, anchor, dilate, scratch);
        for (row = 0; row < height; row++)
            _mm_storeu_si128((__m128i*)(dst + (size_t)row * (size_t)width + col), filtered[row]);
    }
    for (; col < width; col++) {
        int row;
        for (row = 0; row < height; row++)
            column[row] = src[(size_t)row * (size_t)width + col];
        vizion_line_minmax(column, column_out, height, mask_height, anchor, dilate, line_scratch);
        for (row = 0; row < height; row++)
            dst[(size_t)row * (size_t)width + col] = column_out[row];
    }
    vizion_aligned_free(rows);
    vizion_aligned_free(filtered);
    vizion_aligned_free(scratch);
    free(column);
    free(column_out);
    free(line_scratch);
    return 0;
}

int vizion_gray_rect_sse2(const uint8_t* src, uint8_t* dst, int width, int height, int mask_height, int mask_width, int dilate)
{
    uint8_t* temp;
    int code;
    if (!src || !dst || width < 1 || height < 1 || mask_height < 1 || mask_width < 1)
        return -1;
    temp = (uint8_t*)malloc((size_t)width * (size_t)height);
    if (!temp)
        return vizion_gray_rect_scalar(src, dst, width, height, mask_height, mask_width, dilate);
    code = vizion_horizontal_sse(src, temp, width, height, mask_width, dilate);
    if (code == 0)
        code = vizion_vertical_sse(temp, dst, width, height, mask_height, dilate);
    if (code != 0)
        code = vizion_gray_rect_scalar(src, dst, width, height, mask_height, mask_width, dilate);
    free(temp);
    return code;
}
