#include "threshold_internal.h"

#if defined(_MSC_VER)
#  include <intrin.h>
#else
#  include <emmintrin.h>
#endif

#if !defined(__SSE2__) && !defined(_M_X64) && !defined(_M_IX86) && !defined(__i386__) && !defined(__x86_64__)
#  error "threshold_sse2.c 只給 x86/x64"
#endif

int vizion_threshold_sse2(
    const uint8_t* gray, int width, int height, int min_gray, int max_gray,
    int* rows, int* columns, int* lengths, int capacity, int* written)
{
    int counting = rows == 0;
    int count = 0;
    int row;
    __m128i vmin = _mm_set1_epi8((char)(unsigned char)min_gray);
    __m128i vmax = _mm_set1_epi8((char)(unsigned char)max_gray);
    unsigned min_value = (unsigned)min_gray;
    unsigned max_value = (unsigned)max_gray;
    for (row = 0; row < height; row++) {
        const uint8_t* line = gray + (size_t)row * (size_t)width;
        int start = -1;
        int column = 0;
        for (; column + 16 <= width; column += 16) {
            __m128i value = _mm_loadu_si128((const __m128i*)(line + column));
            __m128i above = _mm_cmpeq_epi8(_mm_max_epu8(value, vmin), value);
            __m128i below = _mm_cmpeq_epi8(_mm_min_epu8(value, vmax), value);
            int bits = _mm_movemask_epi8(_mm_and_si128(above, below));
            if (vizion_emit_mask((uint32_t)bits, 16, row, column, &start, rows, columns, lengths, capacity, &count, counting))
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
