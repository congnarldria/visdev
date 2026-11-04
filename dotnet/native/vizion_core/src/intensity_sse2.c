#include "vizion_core.h"

#include <stddef.h>
#include <stdint.h>

#if defined(_MSC_VER)
#  include <intrin.h>
#else
#  include <emmintrin.h>
#endif

static int64_t vizion_sum_run16(const uint8_t* pixel, int length)
{
    int64_t total = 0;
    int index = 0;
    for (; index + 16 <= length; index += 16) {
        __m128i block = _mm_loadu_si128((const __m128i*)(pixel + index));
        __m128i sad = _mm_sad_epu8(block, _mm_setzero_si128());
        /* sad 的兩段都放得進 32 位元。_mm_cvtsi128_si64 在 Win32 沒有這個內建。 */
        total += (unsigned int)_mm_cvtsi128_si32(sad);
        total += (unsigned int)_mm_cvtsi128_si32(_mm_srli_si128(sad, 8));
    }
    for (; index < length; index++)
        total += pixel[index];
    return total;
}

void vizion_intensity_sum_sse2(const uint8_t* gray, int width, const int* rows, const int* columns, const int* lengths, int runs, int64_t* sum, int64_t* count)
{
    int64_t total = 0;
    int64_t pixels = 0;
    int run;
    for (run = 0; run < runs; run++) {
        int length = lengths[run];
        const uint8_t* pixel;
        if (length <= 0)
            continue;
        pixel = gray + (size_t)rows[run] * (size_t)width + columns[run];
        total += vizion_sum_run16(pixel, length);
        pixels += length;
    }
    *sum = total;
    *count = pixels;
}

static void vizion_square_run2(const uint8_t* pixel, int length, double mean, double* square)
{
    int index = 0;
    __m128d center = _mm_set1_pd(mean);
    double total = *square;
    for (; index + 2 <= length; index += 2) {
        __m128i bytes = _mm_cvtsi32_si128(pixel[index] | (pixel[index + 1] << 16));
        __m128i words = _mm_unpacklo_epi16(bytes, _mm_setzero_si128());
        __m128d value = _mm_cvtepi32_pd(words);
        __m128d delta = _mm_sub_pd(value, center);
        __m128d squared = _mm_mul_pd(delta, delta);
        double parts[2];
        _mm_storeu_pd(parts, squared);
        total += parts[0];
        total += parts[1];
    }
    for (; index < length; index++) {
        double delta = (double)pixel[index] - mean;
        total += delta * delta;
    }
    *square = total;
}

void vizion_intensity_square_sse2(const uint8_t* gray, int width, const int* rows, const int* columns, const int* lengths, int runs, double mean, double* square)
{
    double total = 0;
    int run;
    for (run = 0; run < runs; run++) {
        int length = lengths[run];
        const uint8_t* pixel;
        if (length <= 0)
            continue;
        pixel = gray + (size_t)rows[run] * (size_t)width + columns[run];
        vizion_square_run2(pixel, length, mean, &total);
    }
    *square = total;
}
