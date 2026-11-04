#include "vizion_core.h"

#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#  include <intrin.h>
#else
#  include <emmintrin.h>
#endif

int vizion_reflect101(int position, int length);
int vizion_mean_div(unsigned int sum, int area);
int vizion_mean_scalar(const uint8_t* src, uint8_t* dst, int width, int height, int mask_width, int mask_height);

static __m128i vizion_load4(const uint8_t* src, int width, int row, int col)
{
    return _mm_setr_epi32(
        src[(size_t)row * (size_t)width + col],
        src[((size_t)row + 1) * (size_t)width + col],
        src[((size_t)row + 2) * (size_t)width + col],
        src[((size_t)row + 3) * (size_t)width + col]);
}

static void vizion_store4(unsigned int* dst, int width, int row, int col, __m128i value)
{
    unsigned int packed[4];
    _mm_storeu_si128((__m128i*)packed, value);
    dst[(size_t)row * (size_t)width + col] = packed[0];
    dst[((size_t)row + 1) * (size_t)width + col] = packed[1];
    dst[((size_t)row + 2) * (size_t)width + col] = packed[2];
    dst[((size_t)row + 3) * (size_t)width + col] = packed[3];
}

static __m128i vizion_load4_sum(const unsigned int* src, int width, int row, int col)
{
    return _mm_setr_epi32(
        (int)src[(size_t)row * (size_t)width + col],
        (int)src[(size_t)row * (size_t)width + col + 1],
        (int)src[(size_t)row * (size_t)width + col + 2],
        (int)src[(size_t)row * (size_t)width + col + 3]);
}

int vizion_mean_sse2(const uint8_t* src, uint8_t* dst, int width, int height, int mask_width, int mask_height)
{
    unsigned int* horizontal;
    int anchor_x = mask_width / 2;
    int anchor_y = mask_height / 2;
    int area = mask_width * mask_height;
    int row = 0;
    int col;
    if (!src || !dst || width < 1 || height < 1 || mask_width < 1 || mask_height < 1)
        return -1;
    horizontal = (unsigned int*)malloc((size_t)width * (size_t)height * sizeof(unsigned int));
    if (!horizontal)
        return vizion_mean_scalar(src, dst, width, height, mask_width, mask_height);
    for (; row + 4 <= height; row += 4) {
        __m128i sum = _mm_setzero_si128();
        int tap;
        for (tap = 0; tap < mask_width; tap++)
            sum = _mm_add_epi32(sum, vizion_load4(src, width, row, vizion_reflect101(tap - anchor_x, width)));
        vizion_store4(horizontal, width, row, 0, sum);
        for (col = 1; col < width; col++) {
            sum = _mm_add_epi32(sum, vizion_load4(src, width, row, vizion_reflect101(col + mask_width - 1 - anchor_x, width)));
            sum = _mm_sub_epi32(sum, vizion_load4(src, width, row, vizion_reflect101(col - 1 - anchor_x, width)));
            vizion_store4(horizontal, width, row, col, sum);
        }
    }
    for (; row < height; row++) {
        const uint8_t* line = src + (size_t)row * (size_t)width;
        unsigned int sum = 0;
        int tap;
        for (tap = 0; tap < mask_width; tap++)
            sum += line[vizion_reflect101(tap - anchor_x, width)];
        horizontal[(size_t)row * (size_t)width] = sum;
        for (col = 1; col < width; col++) {
            sum += line[vizion_reflect101(col + mask_width - 1 - anchor_x, width)];
            sum -= line[vizion_reflect101(col - 1 - anchor_x, width)];
            horizontal[(size_t)row * (size_t)width + col] = sum;
        }
    }
    for (col = 0; col + 4 <= width; col += 4) {
        __m128i sum = _mm_setzero_si128();
        int tap;
        int y;
        for (tap = 0; tap < mask_height; tap++)
            sum = _mm_add_epi32(sum, vizion_load4_sum(horizontal, width, vizion_reflect101(tap - anchor_y, height), col));
        {
            unsigned int packed[4];
            _mm_storeu_si128((__m128i*)packed, sum);
            for (y = 0; y < 4; y++)
                dst[col + y] = (uint8_t)vizion_mean_div(packed[y], area);
        }
        for (y = 1; y < height; y++) {
            sum = _mm_add_epi32(sum, vizion_load4_sum(horizontal, width, vizion_reflect101(y + mask_height - 1 - anchor_y, height), col));
            sum = _mm_sub_epi32(sum, vizion_load4_sum(horizontal, width, vizion_reflect101(y - 1 - anchor_y, height), col));
            {
                unsigned int packed[4];
                _mm_storeu_si128((__m128i*)packed, sum);
                for (tap = 0; tap < 4; tap++)
                    dst[(size_t)y * (size_t)width + col + tap] = (uint8_t)vizion_mean_div(packed[tap], area);
            }
        }
    }
    for (; col < width; col++) {
        unsigned int sum = 0;
        int tap;
        for (tap = 0; tap < mask_height; tap++)
            sum += horizontal[(size_t)vizion_reflect101(tap - anchor_y, height) * (size_t)width + col];
        dst[col] = (uint8_t)vizion_mean_div(sum, area);
        for (row = 1; row < height; row++) {
            sum += horizontal[(size_t)vizion_reflect101(row + mask_height - 1 - anchor_y, height) * (size_t)width + col];
            sum -= horizontal[(size_t)vizion_reflect101(row - 1 - anchor_y, height) * (size_t)width + col];
            dst[(size_t)row * (size_t)width + col] = (uint8_t)vizion_mean_div(sum, area);
        }
    }
    free(horizontal);
    return 0;
}
