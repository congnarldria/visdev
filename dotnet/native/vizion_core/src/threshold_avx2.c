#include "threshold_internal.h"

#if defined(_MSC_VER)
#  include <immintrin.h>
#else
#  include <immintrin.h>
#endif

#if !defined(__AVX2__) && !defined(_MSC_VER)
#  error "threshold_avx2.c 需要 -mavx2 或 /arch:AVX2"
#endif

int vizion_threshold_avx2(
    const uint8_t* gray, int width, int height, int min_gray, int max_gray,
    int* rows, int* columns, int* lengths, int capacity, int* written)
{
    int counting = rows == 0;
    int count = 0;
    int row;
    __m256i vmin = _mm256_set1_epi8((char)(unsigned char)min_gray);
    __m256i vmax = _mm256_set1_epi8((char)(unsigned char)max_gray);
    unsigned min_value = (unsigned)min_gray;
    unsigned max_value = (unsigned)max_gray;
    for (row = 0; row < height; row++) {
        const uint8_t* line = gray + (size_t)row * (size_t)width;
        int start = -1;
        int column = 0;
        for (; column + 32 <= width; column += 32) {
            __m256i value = _mm256_loadu_si256((const __m256i*)(line + column));
            __m256i above = _mm256_cmpeq_epi8(_mm256_max_epu8(value, vmin), value);
            __m256i below = _mm256_cmpeq_epi8(_mm256_min_epu8(value, vmax), value);
            int bits = _mm256_movemask_epi8(_mm256_and_si256(above, below));
            if (vizion_emit_mask((uint32_t)bits, 32, row, column, &start, rows, columns, lengths, capacity, &count, counting))
                return -2;
        }
        for (; column < width; column++) {
            unsigned value = line[column];
            int on = value >= min_value && value <= max_value;
            if (on) {
                if (start < 0)
                    start = column;
            } else if (start >= 0) {
                if (vizion_flush_run(row, start, column - 1, rows, columns, lengths, capacity, &count, counting))
                    return -2;
                start = -1;
            }
        }
        if (start >= 0 && vizion_flush_run(row, start, width - 1, rows, columns, lengths, capacity, &count, counting))
            return -2;
    }
    *written = count;
    return 0;
}
