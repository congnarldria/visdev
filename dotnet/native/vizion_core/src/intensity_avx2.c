#include "vizion_core.h"

#include <stddef.h>
#include <stdint.h>

#if defined(_MSC_VER)
#  include <immintrin.h>
#else
#  include <immintrin.h>
#endif

static int64_t vizion_sum_run32(const uint8_t* pixel, int length)
{
    int64_t total = 0;
    int index = 0;
    for (; index + 32 <= length; index += 32) {
        __m256i block = _mm256_loadu_si256((const __m256i*)(pixel + index));
        __m256i sad = _mm256_sad_epu8(block, _mm256_setzero_si256());
        __m128i low = _mm256_castsi256_si128(sad);
        __m128i high = _mm256_extracti128_si256(sad, 1);
        total += _mm_cvtsi128_si64(low);
        total += _mm_cvtsi128_si64(_mm_srli_si128(low, 8));
        total += _mm_cvtsi128_si64(high);
        total += _mm_cvtsi128_si64(_mm_srli_si128(high, 8));
    }
    for (; index + 16 <= length; index += 16) {
        __m128i block = _mm_loadu_si128((const __m128i*)(pixel + index));
        __m128i sad = _mm_sad_epu8(block, _mm_setzero_si128());
        total += _mm_cvtsi128_si64(sad);
        total += _mm_cvtsi128_si64(_mm_srli_si128(sad, 8));
    }
    for (; index < length; index++)
        total += pixel[index];
    return total;
}

void vizion_intensity_sum_avx2(const uint8_t* gray, int width, const int* rows, const int* columns, const int* lengths, int runs, int64_t* sum, int64_t* count)
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
        total += vizion_sum_run32(pixel, length);
        pixels += length;
    }
    *sum = total;
    *count = pixels;
}

static void vizion_square_run4(const uint8_t* pixel, int length, double mean, double* square)
{
    int index = 0;
    __m256d center = _mm256_set1_pd(mean);
    double total = *square;
    for (; index + 4 <= length; index += 4) {
        __m128i bytes = _mm_cvtsi32_si128((int)pixel[index] | ((int)pixel[index + 1] << 8) | ((int)pixel[index + 2] << 16) | ((int)pixel[index + 3] << 24));
        __m128i words = _mm_unpacklo_epi8(bytes, _mm_setzero_si128());
        __m128i values = _mm_unpacklo_epi16(words, _mm_setzero_si128());
        __m256d delta = _mm256_sub_pd(_mm256_cvtepi32_pd(values), center);
        __m256d squared = _mm256_mul_pd(delta, delta);
        double parts[4];
        _mm256_storeu_pd(parts, squared);
        total += parts[0];
        total += parts[1];
        total += parts[2];
        total += parts[3];
    }
    for (; index < length; index++) {
        double delta = (double)pixel[index] - mean;
        total += delta * delta;
    }
    *square = total;
}

void vizion_intensity_square_avx2(const uint8_t* gray, int width, const int* rows, const int* columns, const int* lengths, int runs, double mean, double* square)
{
    double total = 0;
    int run;
    for (run = 0; run < runs; run++) {
        int length = lengths[run];
        const uint8_t* pixel;
        if (length <= 0)
            continue;
        pixel = gray + (size_t)rows[run] * (size_t)width + columns[run];
        vizion_square_run4(pixel, length, mean, &total);
    }
    *square = total;
}
