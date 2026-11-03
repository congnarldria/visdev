#include "vizion_core.h"

#include <stdlib.h>

#if defined(_MSC_VER)
#  include <immintrin.h>
#else
#  include <immintrin.h>
#endif

static int vizion_reflect(int position, int length)
{
    if ((unsigned)position < (unsigned)length)
        return position;
    if (length <= 1)
        return 0;
    do {
        if (position < 0)
            position = -position - 1;
        else
            position = length - 1 - (position - length);
    } while ((unsigned)position >= (unsigned)length);
    return position;
}

int vizion_sep_scalar(const float* src, float* dst, int width, int height, const float* across, int across_len, const float* along, int along_len);

static void vizion_sep_row_scalar(const float* src, float* dst, int width, int row, const float* across, int across_len, int col0, int col1)
{
    int radius = across_len / 2;
    int col;
    for (col = col0; col < col1; col++) {
        float sum = 0;
        int tap;
        for (tap = 0; tap < across_len; tap++) {
            int xx = vizion_reflect(col + tap - radius, width);
            sum += src[(size_t)row * (size_t)width + xx] * across[tap];
        }
        dst[(size_t)row * (size_t)width + col] = sum;
    }
}

int vizion_sep_avx2(const float* src, float* dst, int width, int height, const float* across, int across_len, const float* along, int along_len)
{
    int radius_x = across_len / 2;
    int radius_y = along_len / 2;
    float* horizontal = (float*)malloc((size_t)width * (size_t)height * sizeof(float));
    int row;
    int col;
    int interior;
    if (!horizontal)
        return vizion_sep_scalar(src, dst, width, height, across, across_len, along, along_len);
    interior = width - radius_x;
    for (row = 0; row < height; row++) {
        vizion_sep_row_scalar(src, horizontal, width, row, across, across_len, 0, radius_x < width ? radius_x : width);
        for (col = radius_x; col + 8 <= interior; col += 8) {
            __m256 sum = _mm256_setzero_ps();
            int tap;
            for (tap = 0; tap < across_len; tap++)
                sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(src + (size_t)row * (size_t)width + col + tap - radius_x), _mm256_set1_ps(across[tap])));
            _mm256_storeu_ps(horizontal + (size_t)row * (size_t)width + col, sum);
        }
        vizion_sep_row_scalar(src, horizontal, width, row, across, across_len, col, width);
    }
    for (col = 0; col + 8 <= width; col += 8) {
        for (row = 0; row < height; row++) {
            __m256 sum = _mm256_setzero_ps();
            int tap;
            for (tap = 0; tap < along_len; tap++) {
                int yy = vizion_reflect(row + tap - radius_y, height);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(horizontal + (size_t)yy * (size_t)width + col), _mm256_set1_ps(along[tap])));
            }
            _mm256_storeu_ps(dst + (size_t)row * (size_t)width + col, sum);
        }
    }
    for (; col < width; col++) {
        for (row = 0; row < height; row++) {
            float sum = 0;
            int tap;
            for (tap = 0; tap < along_len; tap++) {
                int yy = vizion_reflect(row + tap - radius_y, height);
                sum += horizontal[(size_t)yy * (size_t)width + col] * along[tap];
            }
            dst[(size_t)row * (size_t)width + col] = sum;
        }
    }
    free(horizontal);
    return 0;
}
