#include "vizion_core.h"

#if defined(_MSC_VER)
#  include <intrin.h>
#else
#  include <emmintrin.h>
#endif

void vizion_bits_scalar(const uint8_t* a, const uint8_t* b, uint8_t* dst, int begin, int count, int op);

void vizion_bits_sse2(const uint8_t* a, const uint8_t* b, uint8_t* dst, int count, int op)
{
    int index = 0;
    if (op == 3) {
        __m128i fill = _mm_set1_epi8((char)255);
        for (; index + 16 <= count; index += 16) {
            __m128i block = _mm_loadu_si128((const __m128i*)(a + index));
            _mm_storeu_si128((__m128i*)(dst + index), _mm_xor_si128(block, fill));
        }
    } else {
        for (; index + 16 <= count; index += 16) {
            __m128i left = _mm_loadu_si128((const __m128i*)(a + index));
            __m128i right = _mm_loadu_si128((const __m128i*)(b + index));
            __m128i block;
            if (op == 0)
                block = _mm_and_si128(left, right);
            else if (op == 1)
                block = _mm_or_si128(left, right);
            else
                block = _mm_xor_si128(left, right);
            _mm_storeu_si128((__m128i*)(dst + index), block);
        }
    }
    if (index < count)
        vizion_bits_scalar(a, b, dst, index, count, op);
}
