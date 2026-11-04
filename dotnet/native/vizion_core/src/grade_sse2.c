#include "vizion_core.h"

#if defined(_MSC_VER)
#  include <intrin.h>
#else
#  include <emmintrin.h>
#endif

/* 四個相鄰輸出的取樣都在影像內。係數依序乘加，不使用融合乘加。 */

void vizion_sep_quad_sse2(const float* src, float* dst, int col, const float* kernel, int length, int anchor)
{
    int tap;
    __m128 acc = _mm_setzero_ps();
    for (tap = 0; tap < length; tap++) {
        __m128 sample = _mm_loadu_ps(src + col + tap - anchor);
        __m128 weight = _mm_set1_ps(kernel[tap]);
        acc = _mm_add_ps(acc, _mm_mul_ps(sample, weight));
    }
    _mm_storeu_ps(dst + col, acc);
}
