#include "vizion_core.h"

#if defined(_MSC_VER)
#  include <immintrin.h>
#else
#  include <immintrin.h>
#endif

void vizion_bgr_scalar(const uint8_t* bgr, uint8_t* gray, int begin, int count);

static __m128i vizion_lane4(__m128i bgr)
{
    __m128i zero = _mm_setzero_si128();
    __m128i blue = _mm_shuffle_epi8(bgr, _mm_setr_epi8(0, 3, 6, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    __m128i green = _mm_shuffle_epi8(bgr, _mm_setr_epi8(1, 4, 7, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    __m128i red = _mm_shuffle_epi8(bgr, _mm_setr_epi8(2, 5, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    blue = _mm_unpacklo_epi16(_mm_unpacklo_epi8(blue, zero), zero);
    green = _mm_unpacklo_epi16(_mm_unpacklo_epi8(green, zero), zero);
    red = _mm_unpacklo_epi16(_mm_unpacklo_epi8(red, zero), zero);
    __m128i sum = _mm_mullo_epi32(red, _mm_set1_epi32(9798));
    sum = _mm_add_epi32(sum, _mm_mullo_epi32(green, _mm_set1_epi32(19235)));
    sum = _mm_add_epi32(sum, _mm_mullo_epi32(blue, _mm_set1_epi32(3735)));
    return _mm_srli_epi32(_mm_add_epi32(sum, _mm_set1_epi32(16384)), 15);
}

void vizion_bgr_avx2(const uint8_t* bgr, uint8_t* gray, int count)
{
    int index = 0;
    for (; index + 8 <= count; index += 8) {
        const uint8_t* pixel = bgr + (size_t)index * 3;
        __m128i low = _mm_loadu_si128((const __m128i*)pixel);
        __m128i high = _mm_loadl_epi64((const __m128i*)(pixel + 16));
        __m128i rest = _mm_alignr_epi8(high, low, 12);
        __m128i packed = _mm_packs_epi32(vizion_lane4(low), vizion_lane4(rest));
        packed = _mm_packus_epi16(packed, _mm_setzero_si128());
        _mm_storel_epi64((__m128i*)(gray + index), packed);
    }
    vizion_bgr_scalar(bgr, gray, index, count);
}
