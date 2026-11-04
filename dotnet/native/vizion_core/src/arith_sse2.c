#include "vizion_core.h"

#if defined(_MSC_VER)
#  include <intrin.h>
#else
#  include <emmintrin.h>
#endif

void vizion_arith_scalar(
    const uint8_t* a, const uint8_t* b, uint8_t* dst, int begin, int count,
    int op, float mult, float add);

static void vizion_store4(uint8_t* dst, __m128 value)
{
    __m128i bits = _mm_cvttps_epi32(_mm_min_ps(_mm_max_ps(value, _mm_setzero_ps()), _mm_set1_ps(255.f)));
    bits = _mm_packs_epi32(bits, _mm_setzero_si128());
    bits = _mm_packus_epi16(bits, _mm_setzero_si128());
    dst[0] = (uint8_t)_mm_cvtsi128_si32(bits);
    dst[1] = (uint8_t)_mm_cvtsi128_si32(_mm_srli_si128(bits, 1));
    dst[2] = (uint8_t)_mm_cvtsi128_si32(_mm_srli_si128(bits, 2));
    dst[3] = (uint8_t)_mm_cvtsi128_si32(_mm_srli_si128(bits, 3));
}

static __m128 vizion_load4(const uint8_t* src)
{
    __m128i bytes = _mm_cvtsi32_si128((int)(uint32_t)(src[0] | (src[1] << 8) | (src[2] << 16) | (src[3] << 24)));
    __m128i wide = _mm_unpacklo_epi8(bytes, _mm_setzero_si128());
    wide = _mm_unpacklo_epi16(wide, _mm_setzero_si128());
    return _mm_cvtepi32_ps(wide);
}

static void vizion_float4(const uint8_t* a, const uint8_t* b, uint8_t* dst, int op, float mult, float add)
{
    __m128 left = vizion_load4(a);
    __m128 right = b ? vizion_load4(b) : _mm_setzero_ps();
    __m128 scale = _mm_set1_ps(mult);
    __m128 offset = _mm_set1_ps(add);
    __m128 value;
    switch (op) {
    case 1:
        value = _mm_add_ps(_mm_mul_ps(left, scale), offset);
        break;
    case 2:
        value = _mm_add_ps(_mm_mul_ps(_mm_add_ps(left, right), scale), offset);
        break;
    case 3:
        value = _mm_add_ps(_mm_mul_ps(_mm_sub_ps(left, right), scale), offset);
        break;
    case 4:
        value = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(left, right), scale), offset);
        break;
    case 5:
        value = _mm_add_ps(_mm_mul_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), _mm_sub_ps(left, right)), scale), offset);
        break;
    default:
        value = left;
        break;
    }
    vizion_store4(dst, value);
}

void vizion_arith_sse2(
    const uint8_t* a, const uint8_t* b, uint8_t* dst, int count,
    int op, float mult, float add)
{
    int index = 0;
    if (op == 0) {
        __m128i fill = _mm_set1_epi8((char)255);
        for (; index + 16 <= count; index += 16) {
            __m128i block = _mm_loadu_si128((const __m128i*)(a + index));
            _mm_storeu_si128((__m128i*)(dst + index), _mm_xor_si128(block, fill));
        }
    } else if (op == 6 || op == 7) {
        for (; index + 16 <= count; index += 16) {
            __m128i left = _mm_loadu_si128((const __m128i*)(a + index));
            __m128i right = _mm_loadu_si128((const __m128i*)(b + index));
            __m128i block = op == 6 ? _mm_min_epu8(left, right) : _mm_max_epu8(left, right);
            _mm_storeu_si128((__m128i*)(dst + index), block);
        }
    } else {
        for (; index + 4 <= count; index += 4)
            vizion_float4(a + index, b ? b + index : NULL, dst + index, op, mult, add);
    }
    if (index < count)
        vizion_arith_scalar(a, b, dst, index, count, op, mult, add);
}
