#include "vizion_core.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* 行程上的平均和標準差。總和用整數，平方和按像素順序加，和 C# 的兩趟相同。 */

void vizion_intensity_sum_sse2(const uint8_t* gray, int width, const int* rows, const int* columns, const int* lengths, int runs, int64_t* sum, int64_t* count);
void vizion_intensity_square_sse2(const uint8_t* gray, int width, const int* rows, const int* columns, const int* lengths, int runs, double mean, double* square);
void vizion_intensity_sum_avx2(const uint8_t* gray, int width, const int* rows, const int* columns, const int* lengths, int runs, int64_t* sum, int64_t* count);
void vizion_intensity_square_avx2(const uint8_t* gray, int width, const int* rows, const int* columns, const int* lengths, int runs, double mean, double* square);

static void vizion_intensity_sum_scalar(const uint8_t* gray, int width, const int* rows, const int* columns, const int* lengths, int runs, int64_t* sum, int64_t* count)
{
    int run;
    int64_t total = 0;
    int64_t pixels = 0;
    for (run = 0; run < runs; run++) {
        int length = lengths[run];
        int index;
        const uint8_t* pixel;
        if (length <= 0)
            continue;
        pixel = gray + (size_t)rows[run] * (size_t)width + columns[run];
        for (index = 0; index < length; index++)
            total += pixel[index];
        pixels += length;
    }
    *sum = total;
    *count = pixels;
}

static void vizion_intensity_square_scalar(const uint8_t* gray, int width, const int* rows, const int* columns, const int* lengths, int runs, double mean, double* square)
{
    int run;
    double total = 0;
    for (run = 0; run < runs; run++) {
        int length = lengths[run];
        int index;
        const uint8_t* pixel;
        if (length <= 0)
            continue;
        pixel = gray + (size_t)rows[run] * (size_t)width + columns[run];
        for (index = 0; index < length; index++) {
            double delta = (double)pixel[index] - mean;
            total += delta * delta;
        }
    }
    *square = total;
}

int vizion_intensity(
    const uint8_t* gray, int width,
    const int* rows, const int* columns, const int* lengths, int runs,
    double* mean, double* deviation)
{
    int64_t sum = 0;
    int64_t count = 0;
    double square = 0;
    int flags;
    if (!gray || !mean || !deviation || width < 1 || runs < 0)
        return -1;
    if (runs > 0 && (!rows || !columns || !lengths))
        return -1;
    flags = vizion_cpu_flags();
    if (flags & VIZION_CPU_SELECTED_AVX2) {
        vizion_intensity_sum_avx2(gray, width, rows, columns, lengths, runs, &sum, &count);
        if (count == 0) {
            *mean = 0;
            *deviation = 0;
            return 0;
        }
        *mean = (double)sum / (double)count;
        vizion_intensity_square_avx2(gray, width, rows, columns, lengths, runs, *mean, &square);
    } else if (flags & VIZION_CPU_SELECTED_SSE2) {
        vizion_intensity_sum_sse2(gray, width, rows, columns, lengths, runs, &sum, &count);
        if (count == 0) {
            *mean = 0;
            *deviation = 0;
            return 0;
        }
        *mean = (double)sum / (double)count;
        vizion_intensity_square_sse2(gray, width, rows, columns, lengths, runs, *mean, &square);
    } else {
        vizion_intensity_sum_scalar(gray, width, rows, columns, lengths, runs, &sum, &count);
        if (count == 0) {
            *mean = 0;
            *deviation = 0;
            return 0;
        }
        *mean = (double)sum / (double)count;
        vizion_intensity_square_scalar(gray, width, rows, columns, lengths, runs, *mean, &square);
    }
    *deviation = sqrt(square / (double)count);
    return 0;
}

static int vizion_close(double left, double right)
{
    double diff = left - right;
    if (diff < 0)
        diff = -diff;
    return diff <= 1e-12;
}

int vizion_intensity_selftest(void)
{
    uint8_t line[3] = {1, 2, 3};
    uint8_t image[48];
    int row = 0;
    int column = 0;
    int length = 3;
    int rows[3];
    int columns[3];
    int lengths[3];
    double mean = 1;
    double deviation = 1;
    double scalar_mean;
    double scalar_deviation;
    double fast_mean;
    double fast_deviation;
    int64_t sum = 0;
    int64_t count = 0;
    double square = 0;
    int i;
    int flags = vizion_cpu_flags();
    if (vizion_intensity(line, 3, &row, &column, &length, 1, &mean, &deviation) != 0)
        return 1;
    if (!vizion_close(mean, 2.0) || !vizion_close(deviation, sqrt(2.0 / 3.0)))
        return 2;
    if (vizion_intensity(line, 3, 0, 0, 0, 0, &mean, &deviation) != 0 || mean != 0 || deviation != 0)
        return 3;
    for (i = 0; i < 48; i++)
        image[i] = (uint8_t)((i * 19 + 3) & 255);
    rows[0] = 0;
    columns[0] = 1;
    lengths[0] = 17;
    rows[1] = 1;
    columns[1] = 0;
    lengths[1] = 32;
    rows[2] = 2;
    columns[2] = 5;
    lengths[2] = 0;
    vizion_intensity_sum_scalar(image, 16, rows, columns, lengths, 3, &sum, &count);
    scalar_mean = (double)sum / (double)count;
    vizion_intensity_square_scalar(image, 16, rows, columns, lengths, 3, scalar_mean, &square);
    scalar_deviation = sqrt(square / (double)count);
    if (flags & VIZION_CPU_SSE2) {
        vizion_intensity_sum_sse2(image, 16, rows, columns, lengths, 3, &sum, &count);
        fast_mean = (double)sum / (double)count;
        vizion_intensity_square_sse2(image, 16, rows, columns, lengths, 3, fast_mean, &square);
        fast_deviation = sqrt(square / (double)count);
        if (!vizion_close(scalar_mean, fast_mean) || !vizion_close(scalar_deviation, fast_deviation))
            return 4;
    }
    if (flags & VIZION_CPU_AVX2) {
        vizion_intensity_sum_avx2(image, 16, rows, columns, lengths, 3, &sum, &count);
        fast_mean = (double)sum / (double)count;
        vizion_intensity_square_avx2(image, 16, rows, columns, lengths, 3, fast_mean, &square);
        fast_deviation = sqrt(square / (double)count);
        if (!vizion_close(scalar_mean, fast_mean) || !vizion_close(scalar_deviation, fast_deviation))
            return 5;
    }
    return 0;
}
