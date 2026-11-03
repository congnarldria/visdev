#include "vizion_core.h"

#include <stdint.h>

#if defined(_MSC_VER)
#  include <intrin.h>
#else
#  include <emmintrin.h>
#endif

uint16_t vizion_pyrdown_hpixel(const uint8_t* src, int width, int row, int column);
uint8_t vizion_pyrdown_vpixel(const uint16_t* horizontal, int width, int height, int row, int column);

static __m128i vizion_load8(const uint8_t* pixel)
{
    return _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)pixel), _mm_setzero_si128());
}

static void vizion_horizontal_row8(const uint8_t* src, uint16_t* dst, int col)
{
    __m128i left2 = vizion_load8(src + col - 2);
    __m128i left1 = vizion_load8(src + col - 1);
    __m128i center = vizion_load8(src + col);
    __m128i right1 = vizion_load8(src + col + 1);
    __m128i right2 = vizion_load8(src + col + 2);
    __m128i sum = _mm_add_epi16(left2, right2);
    sum = _mm_add_epi16(sum, _mm_slli_epi16(left1, 2));
    sum = _mm_add_epi16(sum, _mm_slli_epi16(right1, 2));
    sum = _mm_add_epi16(sum, _mm_slli_epi16(center, 2));
    sum = _mm_add_epi16(sum, _mm_slli_epi16(center, 1));
    _mm_storeu_si128((__m128i*)(dst + col), sum);
}

void vizion_pyrdown_horizontal_sse2(const uint8_t* src, uint16_t* horizontal, int width, int height)
{
    int row;
    int col;
    for (row = 0; row < height; row++) {
        const uint8_t* line = src + (size_t)row * (size_t)width;
        uint16_t* dest = horizontal + (size_t)row * (size_t)width;
        for (col = 0; col < width && col < 2; col++)
            dest[col] = vizion_pyrdown_hpixel(src, width, row, col);
        for (; col + 10 <= width; col += 8) {
            vizion_horizontal_row8(line, dest, col);
        }
        for (; col < width; col++)
            dest[col] = vizion_pyrdown_hpixel(src, width, row, col);
    }
}

static __m128i vizion_even8(const uint16_t* pixel)
{
    __m128i mask = _mm_set1_epi32(0x0000FFFF);
    __m128i left = _mm_and_si128(_mm_loadu_si128((const __m128i*)pixel), mask);
    __m128i right = _mm_and_si128(_mm_loadu_si128((const __m128i*)(pixel + 8)), mask);
    return _mm_packs_epi32(left, right);
}

static void vizion_vertical_8(const uint16_t* r0, const uint16_t* r1, const uint16_t* r2, const uint16_t* r3, const uint16_t* r4, int source_col, uint8_t* dst)
{
    __m128i s0 = vizion_even8(r0 + source_col);
    __m128i s1 = vizion_even8(r1 + source_col);
    __m128i s2 = vizion_even8(r2 + source_col);
    __m128i s3 = vizion_even8(r3 + source_col);
    __m128i s4 = vizion_even8(r4 + source_col);
    __m128i sum = _mm_add_epi16(s0, s4);
    sum = _mm_add_epi16(sum, _mm_slli_epi16(s1, 2));
    sum = _mm_add_epi16(sum, _mm_slli_epi16(s3, 2));
    sum = _mm_add_epi16(sum, _mm_slli_epi16(s2, 2));
    sum = _mm_add_epi16(sum, _mm_slli_epi16(s2, 1));
    sum = _mm_srli_epi16(_mm_add_epi16(sum, _mm_set1_epi16(128)), 8);
    _mm_storel_epi64((__m128i*)dst, _mm_packus_epi16(sum, _mm_setzero_si128()));
}

void vizion_pyrdown_vertical_sse2(const uint16_t* horizontal, int width, int height, uint8_t* dst)
{
    int dest_width = (width + 1) / 2;
    int dest_height = (height + 1) / 2;
    int row;
    for (row = 0; row < dest_height; row++) {
        int source_row = row * 2;
        int col = 0;
        uint8_t* row_dst = dst + (size_t)row * (size_t)dest_width;
        if (source_row >= 2 && source_row + 2 < height) {
            const uint16_t* r0 = horizontal + (size_t)(source_row - 2) * (size_t)width;
            const uint16_t* r1 = horizontal + (size_t)(source_row - 1) * (size_t)width;
            const uint16_t* r2 = horizontal + (size_t)source_row * (size_t)width;
            const uint16_t* r3 = horizontal + (size_t)(source_row + 1) * (size_t)width;
            const uint16_t* r4 = horizontal + (size_t)(source_row + 2) * (size_t)width;
            for (; col + 8 <= dest_width && col * 2 + 16 <= width; col += 8)
                vizion_vertical_8(r0, r1, r2, r3, r4, col * 2, row_dst + col);
        }
        for (; col < dest_width; col++)
            row_dst[col] = vizion_pyrdown_vpixel(horizontal, width, height, row, col);
    }
}
