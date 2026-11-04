#include "threshold_internal.h"

/* 純 C。不使用 SSE/AVX。x86 沒有 SSE2、或 SIMD 路徑不能用時走這裡。 */
int vizion_threshold_scalar(
    const uint8_t* gray, int width, int height, int min_gray, int max_gray,
    int* rows, int* columns, int* lengths, int capacity, int* written)
{
    int counting = rows == 0;
    int count = 0;
    int row;
    unsigned min_value = (unsigned)min_gray;
    unsigned max_value = (unsigned)max_gray;
    for (row = 0; row < height; row++) {
        const uint8_t* line = gray + (size_t)row * (size_t)width;
        int start = -1;
        int column;
        for (column = 0; column < width; column++) {
            unsigned value = line[column];
            if (value >= min_value && value <= max_value) {
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
