#include "vizion_core.h"

#include <math.h>

#if defined(_MSC_VER)
#  include <intrin.h>
#else
#  include <emmintrin.h>
#endif

static int vizion_round_away(double value)
{
    int rounded = (int)(fabs(value) + 0.5);
    return value < 0 ? -rounded : rounded;
}

static uint8_t vizion_remap_one(const uint8_t* src, int width, int height, double column, double row)
{
    int x0, y0, x1, y1, rounded;
    double wx, wy, top, bottom, value;
    if (column < 0 || row < 0 || column > (double)width - 1 || row > (double)height - 1)
        return 0;
    x0 = (int)column;
    y0 = (int)row;
    x1 = x0 + 1 < width ? x0 + 1 : width - 1;
    y1 = y0 + 1 < height ? y0 + 1 : height - 1;
    wx = column - x0;
    wy = row - y0;
    top = src[(size_t)y0 * (size_t)width + x0] * (1.0 - wx) + src[(size_t)y0 * (size_t)width + x1] * wx;
    bottom = src[(size_t)y1 * (size_t)width + x0] * (1.0 - wx) + src[(size_t)y1 * (size_t)width + x1] * wx;
    value = top * (1.0 - wy) + bottom * wy;
    rounded = vizion_round_away(value);
    if (rounded < 0)
        return 0;
    if (rounded > 255)
        return 255;
    return (uint8_t)rounded;
}

void vizion_remap_sse2(const uint8_t* src, int width, int height, const double* map_x, const double* map_y, uint8_t* dst)
{
    int count = width * height;
    int index = 0;
    for (; index + 2 <= count; index += 2) {
        double mx[2], my[2], a[2], b[2], c[2], d[2], result[2];
        int valid[2];
        int lane;
        __m128d wx, wy, top, bottom, blended;
        for (lane = 0; lane < 2; lane++) {
            double column = map_x[index + lane];
            double row = map_y[index + lane];
            int x0, y0, x1, y1;
            valid[lane] = !(column < 0 || row < 0 || column > (double)width - 1 || row > (double)height - 1);
            if (!valid[lane]) {
                mx[lane] = my[lane] = a[lane] = b[lane] = c[lane] = d[lane] = 0;
                continue;
            }
            x0 = (int)column;
            y0 = (int)row;
            x1 = x0 + 1 < width ? x0 + 1 : width - 1;
            y1 = y0 + 1 < height ? y0 + 1 : height - 1;
            mx[lane] = column - x0;
            my[lane] = row - y0;
            a[lane] = src[(size_t)y0 * (size_t)width + x0];
            b[lane] = src[(size_t)y0 * (size_t)width + x1];
            c[lane] = src[(size_t)y1 * (size_t)width + x0];
            d[lane] = src[(size_t)y1 * (size_t)width + x1];
        }
        wx = _mm_loadu_pd(mx);
        wy = _mm_loadu_pd(my);
        top = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(a), _mm_sub_pd(_mm_set1_pd(1.0), wx)), _mm_mul_pd(_mm_loadu_pd(b), wx));
        bottom = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(c), _mm_sub_pd(_mm_set1_pd(1.0), wx)), _mm_mul_pd(_mm_loadu_pd(d), wx));
        blended = _mm_add_pd(_mm_mul_pd(top, _mm_sub_pd(_mm_set1_pd(1.0), wy)), _mm_mul_pd(bottom, wy));
        _mm_storeu_pd(result, blended);
        for (lane = 0; lane < 2; lane++) {
            int rounded;
            if (!valid[lane]) {
                dst[index + lane] = 0;
                continue;
            }
            rounded = vizion_round_away(result[lane]);
            if (rounded < 0)
                rounded = 0;
            if (rounded > 255)
                rounded = 255;
            dst[index + lane] = (uint8_t)rounded;
        }
    }
    for (; index < count; index++)
        dst[index] = vizion_remap_one(src, width, height, map_x[index], map_y[index]);
}
