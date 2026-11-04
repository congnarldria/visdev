#include "vizion_core.h"

#include <stdint.h>

#if defined(_MSC_VER)
#  include <immintrin.h>
#else
#  include <immintrin.h>
#endif

uint16_t vizion_pyrdown_hpixel(const uint8_t* src, int width, int row, int column);

static __m256i vizion_load16(const uint8_t* pixel)
{
    return _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)pixel));
}

static void vizion_horizontal_row16(const uint8_t* src, uint16_t* dst, int col)
{
    __m256i left2 = vizion_load16(src + col - 2);
    __m256i left1 = vizion_load16(src + col - 1);
    __m256i center = vizion_load16(src + col);
    __m256i right1 = vizion_load16(src + col + 1);
    __m256i right2 = vizion_load16(src + col + 2);
    __m256i sum = _mm256_add_epi16(left2, right2);
    sum = _mm256_add_epi16(sum, _mm256_slli_epi16(left1, 2));
    sum = _mm256_add_epi16(sum, _mm256_slli_epi16(right1, 2));
    sum = _mm256_add_epi16(sum, _mm256_slli_epi16(center, 2));
    sum = _mm256_add_epi16(sum, _mm256_slli_epi16(center, 1));
    _mm256_storeu_si256((__m256i*)(dst + col), sum);
}

void vizion_pyrdown_horizontal_avx2(const uint8_t* src, uint16_t* horizontal, int width, int height)
{
    int row;
    int col;
    for (row = 0; row < height; row++) {
        const uint8_t* line = src + (size_t)row * (size_t)width;
        uint16_t* dest = horizontal + (size_t)row * (size_t)width;
        for (col = 0; col < width && col < 2; col++)
            dest[col] = vizion_pyrdown_hpixel(src, width, row, col);
        for (; col + 18 <= width; col += 16)
            vizion_horizontal_row16(line, dest, col);
        for (; col < width; col++)
            dest[col] = vizion_pyrdown_hpixel(src, width, row, col);
    }
}
