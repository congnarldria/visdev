#include "vizion_core.h"

#if defined(_MSC_VER)
#  include <intrin.h>
#else
#  include <emmintrin.h>
#endif

void vizion_bgr_scalar(const uint8_t* bgr, uint8_t* gray, int begin, int count);

static __m128i vizion_mullo_epi32(__m128i left, __m128i right)
{
    __m128i even = _mm_mul_epu32(left, right);
    __m128i odd = _mm_mul_epu32(_mm_srli_si128(left, 4), _mm_srli_si128(right, 4));
    even = _mm_shuffle_epi32(even, _MM_SHUFFLE(0, 0, 2, 0));
    odd = _mm_shuffle_epi32(odd, _MM_SHUFFLE(0, 0, 2, 0));
    return _mm_unpacklo_epi32(even, odd);
}

static void vizion_bgr4(const uint8_t* pixel, uint8_t* gray)
{
    __m128i blue = _mm_setr_epi32(pixel[0], pixel[3], pixel[6], pixel[9]);
    __m128i green = _mm_setr_epi32(pixel[1], pixel[4], pixel[7], pixel[10]);
    __m128i red = _mm_setr_epi32(pixel[2], pixel[5], pixel[8], pixel[11]);
    __m128i sum = vizion_mullo_epi32(red, _mm_set1_epi32(9798));
    sum = _mm_add_epi32(sum, vizion_mullo_epi32(green, _mm_set1_epi32(19235)));
    sum = _mm_add_epi32(sum, vizion_mullo_epi32(blue, _mm_set1_epi32(3735)));
    sum = _mm_srli_epi32(_mm_add_epi32(sum, _mm_set1_epi32(16384)), 15);
    sum = _mm_packs_epi32(sum, _mm_setzero_si128());
    sum = _mm_packus_epi16(sum, _mm_setzero_si128());
    gray[0] = (uint8_t)_mm_cvtsi128_si32(sum);
    gray[1] = (uint8_t)_mm_cvtsi128_si32(_mm_srli_si128(sum, 1));
    gray[2] = (uint8_t)_mm_cvtsi128_si32(_mm_srli_si128(sum, 2));
    gray[3] = (uint8_t)_mm_cvtsi128_si32(_mm_srli_si128(sum, 3));
}

void vizion_bgr_sse2(const uint8_t* bgr, uint8_t* gray, int count)
{
    int index = 0;
    for (; index + 4 <= count; index += 4)
        vizion_bgr4(bgr + (size_t)index * 3, gray + index);
    vizion_bgr_scalar(bgr, gray, index, count);
}
