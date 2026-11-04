#include "vizion_core.h"

#include <math.h>

#if defined(_MSC_VER)
#  include <intrin.h>
#else
#  include <emmintrin.h>
#endif

static double vizion_sample(const uint8_t* gray, int width, int height, double x, double y)
{
    int x0, y0, x1, y1;
    double dx, dy, top, bottom;
    if (x < 0 || y < 0 || x > (double)width - 1 || y > (double)height - 1)
        return nan("");
    x0 = (int)floor(x);
    y0 = (int)floor(y);
    x1 = x0 + 1 < width ? x0 + 1 : width - 1;
    y1 = y0 + 1 < height ? y0 + 1 : height - 1;
    dx = x - x0;
    dy = y - y0;
    top = gray[(size_t)y0 * (size_t)width + x0] * (1.0 - dx) + gray[(size_t)y0 * (size_t)width + x1] * dx;
    bottom = gray[(size_t)y1 * (size_t)width + x0] * (1.0 - dx) + gray[(size_t)y1 * (size_t)width + x1] * dx;
    return top * (1.0 - dy) + bottom * dy;
}

void vizion_sample_line_sse2(const uint8_t* gray, int width, int height, double x0, double y0, double dx, double dy, const double* ts, int count, double* values)
{
    int index = 0;
    for (; index + 2 <= count; index += 2) {
        double wxs[2], wys[2], a[2], b[2], c[2], d[2], result[2];
        int valid[2];
        int lane;
        __m128d wx, wy, top, bottom, blended;
        for (lane = 0; lane < 2; lane++) {
            double px = x0 + dx * ts[index + lane];
            double py = y0 + dy * ts[index + lane];
            int x0i, y0i, x1i, y1i;
            valid[lane] = !(px < 0 || py < 0 || px > (double)width - 1 || py > (double)height - 1);
            if (!valid[lane]) {
                wxs[lane] = wys[lane] = a[lane] = b[lane] = c[lane] = d[lane] = 0;
                continue;
            }
            x0i = (int)floor(px);
            y0i = (int)floor(py);
            x1i = x0i + 1 < width ? x0i + 1 : width - 1;
            y1i = y0i + 1 < height ? y0i + 1 : height - 1;
            wxs[lane] = px - x0i;
            wys[lane] = py - y0i;
            a[lane] = gray[(size_t)y0i * (size_t)width + x0i];
            b[lane] = gray[(size_t)y0i * (size_t)width + x1i];
            c[lane] = gray[(size_t)y1i * (size_t)width + x0i];
            d[lane] = gray[(size_t)y1i * (size_t)width + x1i];
        }
        wx = _mm_loadu_pd(wxs);
        wy = _mm_loadu_pd(wys);
        top = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(a), _mm_sub_pd(_mm_set1_pd(1.0), wx)), _mm_mul_pd(_mm_loadu_pd(b), wx));
        bottom = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(c), _mm_sub_pd(_mm_set1_pd(1.0), wx)), _mm_mul_pd(_mm_loadu_pd(d), wx));
        blended = _mm_add_pd(_mm_mul_pd(top, _mm_sub_pd(_mm_set1_pd(1.0), wy)), _mm_mul_pd(bottom, wy));
        _mm_storeu_pd(result, blended);
        for (lane = 0; lane < 2; lane++)
            values[index + lane] = valid[lane] ? result[lane] : nan("");
    }
    for (; index < count; index++)
        values[index] = vizion_sample(gray, width, height, x0 + dx * ts[index], y0 + dy * ts[index]);
}
