#include "vizion_core.h"

#include <math.h>

#if defined(_MSC_VER)
#  include <intrin.h>
#else
#  include <emmintrin.h>
#endif

double vizion_shape_score_scalar(
    const float* gx, const float* gy, const float* mag, int width, int height,
    double column, double row, const float* mx, const float* my, const float* mgx, const float* mgy,
    int count, double min_contrast);
double vizion_ncc_scalar(
    const float* gray, int width, int height, const float* x, const float* y, const float* value,
    int count, double column, double row, int absolute);

static int vizion_floor_int(double value)
{
    return (int)floor(value);
}

static __m128 vizion_blend4(__m128 left, __m128 right, __m128 lower, __m128 bottom, __m128 wx, __m128 wy)
{
    __m128 one = _mm_set1_ps(1.f);
    __m128 ix = _mm_sub_ps(one, wx);
    __m128 iy = _mm_sub_ps(one, wy);
    __m128 upper = _mm_add_ps(_mm_mul_ps(ix, left), _mm_mul_ps(wx, right));
    __m128 down = _mm_add_ps(_mm_mul_ps(ix, lower), _mm_mul_ps(wx, bottom));
    return _mm_add_ps(_mm_mul_ps(iy, upper), _mm_mul_ps(wy, down));
}

static void vizion_prepare4(const float* ox, const float* oy, int index, double column, double row, int width, int height, int shape,
    int* x0, int* y0, float* wx, float* wy, int* valid)
{
    int lane;
    for (lane = 0; lane < 4; lane++) {
        double x = column + (double)ox[index + lane];
        double y = row + (double)oy[index + lane];
        int inside = shape
            ? (x >= 1 && y >= 1 && x < (double)width - 2 && y < (double)height - 2)
            : 1;
        int fx = vizion_floor_int(x);
        int fy = vizion_floor_int(y);
        if (!shape)
            inside = fx >= 0 && fy >= 0 && fx + 1 < width && fy + 1 < height;
        x0[lane] = inside ? fx : 0;
        y0[lane] = inside ? fy : 0;
        wx[lane] = inside ? (float)(x - fx) : 0;
        wy[lane] = inside ? (float)(y - fy) : 0;
        valid[lane] = inside;
    }
}

double vizion_shape_score_sse2(
    const float* gx, const float* gy, const float* mag, int width, int height,
    double column, double row, const float* mx, const float* my, const float* mgx, const float* mgy,
    int count, double min_contrast)
{
    double sum = 0;
    int index = 0;
    if (!gx || !gy || !mag || !mx || !my || !mgx || !mgy || width < 3 || height < 3 || count <= 0)
        return 0;
    for (; index + 4 <= count; index += 4) {
        int x0[4], y0[4], valid[4], lane;
        float wx[4], wy[4], magv[4], gxv[4], gyv[4];
        __m128 mag_r;
        vizion_prepare4(mx, my, index, column, row, width, height, 1, x0, y0, wx, wy, valid);
        {
            float m00[4], m10[4], m01[4], m11[4];
            float x00[4], x10[4], x01[4], x11[4];
            float y00[4], y10[4], y01[4], y11[4];
            __m128 weight_x = _mm_loadu_ps(wx);
            __m128 weight_y = _mm_loadu_ps(wy);
            for (lane = 0; lane < 4; lane++) {
                int i00 = y0[lane] * width + x0[lane];
                m00[lane] = mag[i00];
                m10[lane] = mag[i00 + 1];
                m01[lane] = mag[i00 + width];
                m11[lane] = mag[i00 + width + 1];
                x00[lane] = gx[i00];
                x10[lane] = gx[i00 + 1];
                x01[lane] = gx[i00 + width];
                x11[lane] = gx[i00 + width + 1];
                y00[lane] = gy[i00];
                y10[lane] = gy[i00 + 1];
                y01[lane] = gy[i00 + width];
                y11[lane] = gy[i00 + width + 1];
            }
            mag_r = vizion_blend4(_mm_loadu_ps(m00), _mm_loadu_ps(m10), _mm_loadu_ps(m01), _mm_loadu_ps(m11), weight_x, weight_y);
            _mm_storeu_ps(magv, mag_r);
            _mm_storeu_ps(gxv, vizion_blend4(_mm_loadu_ps(x00), _mm_loadu_ps(x10), _mm_loadu_ps(x01), _mm_loadu_ps(x11), weight_x, weight_y));
            _mm_storeu_ps(gyv, vizion_blend4(_mm_loadu_ps(y00), _mm_loadu_ps(y10), _mm_loadu_ps(y01), _mm_loadu_ps(y11), weight_x, weight_y));
        }
        for (lane = 0; lane < 4; lane++) {
            float inverse;
            float dot;
            if (!valid[lane] || (double)magv[lane] < min_contrast)
                continue;
            inverse = 1.f / (magv[lane] > 1e-3f ? magv[lane] : 1e-3f);
            dot = mgx[index + lane] * gxv[lane] * inverse + mgy[index + lane] * gyv[lane] * inverse;
            if (dot > 0)
                sum += dot;
        }
    }
    if (index < count) {
        double tail = vizion_shape_score_scalar(gx, gy, mag, width, height, column, row, mx + index, my + index, mgx + index, mgy + index, count - index, min_contrast);
        sum += tail * (count - index);
    }
    return sum / count;
}

double vizion_ncc_sse2(
    const float* gray, int width, int height, const float* x, const float* y, const float* value,
    int count, double column, double row, int absolute)
{
    double sum_s = 0, sum_t = 0, sum_ss = 0, sum_tt = 0, sum_st = 0;
    int seen = 0;
    int min_count;
    int index = 0;
    double var_s, var_t, cov, score;
    if (!gray || !x || !y || !value || width < 2 || height < 2 || count <= 0)
        return 0;
    for (; index + 4 <= count; index += 4) {
        int x0[4], y0[4], valid[4], lane;
        float wx[4], wy[4], sample[4];
        vizion_prepare4(x, y, index, column, row, width, height, 0, x0, y0, wx, wy, valid);
        {
            float g00[4], g10[4], g01[4], g11[4];
            for (lane = 0; lane < 4; lane++) {
                int i00 = valid[lane] ? y0[lane] * width + x0[lane] : 0;
                g00[lane] = gray[i00];
                g10[lane] = gray[i00 + 1];
                g01[lane] = gray[i00 + width];
                g11[lane] = gray[i00 + width + 1];
            }
            _mm_storeu_ps(sample, vizion_blend4(_mm_loadu_ps(g00), _mm_loadu_ps(g10), _mm_loadu_ps(g01), _mm_loadu_ps(g11), _mm_loadu_ps(wx), _mm_loadu_ps(wy)));
        }
        for (lane = 0; lane < 4; lane++) {
            float template;
            float sampled;
            if (!valid[lane])
                continue;
            sampled = sample[lane];
            template = value[index + lane];
            sum_s += sampled;
            sum_t += template;
            sum_ss += (double)sampled * sampled;
            sum_tt += (double)template * template;
            sum_st += (double)sampled * template;
            seen++;
        }
    }
    for (; index < count; index++) {
        double px = column + (double)x[index];
        double py = row + (double)y[index];
        int x0 = vizion_floor_int(px);
        int y0 = vizion_floor_int(py);
        float wx, wy, sampled, template;
        int i00;
        if (x0 < 0 || y0 < 0 || x0 + 1 >= width || y0 + 1 >= height)
            continue;
        wx = (float)(px - x0);
        wy = (float)(py - y0);
        i00 = y0 * width + x0;
        sampled = (1.f - wy) * ((1.f - wx) * gray[i00] + wx * gray[i00 + 1]) + wy * ((1.f - wx) * gray[i00 + width] + wx * gray[i00 + width + 1]);
        template = value[index];
        sum_s += sampled;
        sum_t += template;
        sum_ss += (double)sampled * sampled;
        sum_tt += (double)template * template;
        sum_st += (double)sampled * template;
        seen++;
    }
    min_count = (int)(count * 0.65);
    if (min_count < 16)
        min_count = 16;
    if (seen < min_count)
        return 0;
    var_s = sum_ss - sum_s * sum_s / seen;
    var_t = sum_tt - sum_t * sum_t / seen;
    cov = sum_st - sum_s * sum_t / seen;
    if (var_s < 1e-6 || var_t < 1e-6)
        return 0;
    score = cov / sqrt(var_s * var_t);
    return absolute ? fabs(score) : score;
}
