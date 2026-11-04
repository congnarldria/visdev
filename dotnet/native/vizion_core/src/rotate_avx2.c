#include "vizion_core.h"

#include "rotate_internal.h"

#include <string.h>

#if defined(_MSC_VER)
#  include <immintrin.h>
#else
#  include <immintrin.h>
#endif

static void vizion_rotate_tail(const uint8_t* src, uint8_t* dst, int width, int height, int row, int column, double center_row, double center_column, double cosine, double sine)
{
    double dx = (double)column - center_column;
    double dy = (double)row - center_row;
    double source_column = center_column + dx * cosine - dy * sine;
    double source_row = center_row + dx * sine + dy * cosine;
    vizion_rotate_write(src, dst, width, height, row, column, source_row, source_column);
}

void vizion_rotate_avx2(const uint8_t* src, uint8_t* dst, int width, int height, double cosine, double sine)
{
    double center_column = (width - 1) / 2.0;
    double center_row = (height - 1) / 2.0;
    int row;
    int column;
    __m256d cos_v = _mm256_set1_pd(cosine);
    __m256d sin_v = _mm256_set1_pd(sine);
    __m256d center_c = _mm256_set1_pd(center_column);
    __m256d center_r = _mm256_set1_pd(center_row);
    memset(dst, 0, (size_t)width * (size_t)height);
    for (row = 0; row < height; row++) {
        double dy = (double)row - center_row;
        __m256d dy_sin = _mm256_set1_pd(dy * sine);
        __m256d dy_cos = _mm256_set1_pd(dy * cosine);
        for (column = 0; column + 4 <= width; column += 4) {
            double dx_store[4];
            double columns[4];
            double rows[4];
            int lane;
            __m256d dx;
            __m256d source_column;
            __m256d source_row;
            for (lane = 0; lane < 4; lane++)
                dx_store[lane] = (double)(column + lane) - center_column;
            dx = _mm256_loadu_pd(dx_store);
            source_column = _mm256_sub_pd(_mm256_add_pd(center_c, _mm256_mul_pd(dx, cos_v)), dy_sin);
            source_row = _mm256_add_pd(_mm256_add_pd(center_r, _mm256_mul_pd(dx, sin_v)), dy_cos);
            _mm256_storeu_pd(columns, source_column);
            _mm256_storeu_pd(rows, source_row);
            for (lane = 0; lane < 4; lane++)
                vizion_rotate_write(src, dst, width, height, row, column + lane, rows[lane], columns[lane]);
        }
        for (; column < width; column++)
            vizion_rotate_tail(src, dst, width, height, row, column, center_row, center_column, cosine, sine);
    }
}
