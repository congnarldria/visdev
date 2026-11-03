#include "vizion_core.h"

#include "rotate_internal.h"

#include <string.h>

#if defined(_MSC_VER)
#  include <intrin.h>
#else
#  include <emmintrin.h>
#endif

static void vizion_rotate_tail(const uint8_t* src, uint8_t* dst, int width, int height, int row, int column, double center_row, double center_column, double cosine, double sine)
{
    double dx = (double)column - center_column;
    double dy = (double)row - center_row;
    double source_column = center_column + dx * cosine - dy * sine;
    double source_row = center_row + dx * sine + dy * cosine;
    vizion_rotate_write(src, dst, width, height, row, column, source_row, source_column);
}

void vizion_rotate_sse2(const uint8_t* src, uint8_t* dst, int width, int height, double cosine, double sine)
{
    double center_column = (width - 1) / 2.0;
    double center_row = (height - 1) / 2.0;
    int row;
    int column;
    __m128d cos_v = _mm_set1_pd(cosine);
    __m128d sin_v = _mm_set1_pd(sine);
    __m128d center_c = _mm_set1_pd(center_column);
    __m128d center_r = _mm_set1_pd(center_row);
    memset(dst, 0, (size_t)width * (size_t)height);
    for (row = 0; row < height; row++) {
        double dy = (double)row - center_row;
        __m128d dy_sin = _mm_set1_pd(dy * sine);
        __m128d dy_cos = _mm_set1_pd(dy * cosine);
        for (column = 0; column + 2 <= width; column += 2) {
            double dx_store[2];
            double columns[2];
            double rows[2];
            __m128d dx;
            __m128d source_column;
            __m128d source_row;
            dx_store[0] = (double)column - center_column;
            dx_store[1] = (double)(column + 1) - center_column;
            dx = _mm_loadu_pd(dx_store);
            source_column = _mm_sub_pd(_mm_add_pd(center_c, _mm_mul_pd(dx, cos_v)), dy_sin);
            source_row = _mm_add_pd(_mm_add_pd(center_r, _mm_mul_pd(dx, sin_v)), dy_cos);
            _mm_storeu_pd(columns, source_column);
            _mm_storeu_pd(rows, source_row);
            vizion_rotate_write(src, dst, width, height, row, column, rows[0], columns[0]);
            vizion_rotate_write(src, dst, width, height, row, column + 1, rows[1], columns[1]);
        }
        for (; column < width; column++)
            vizion_rotate_tail(src, dst, width, height, row, column, center_row, center_column, cosine, sine);
    }
}
