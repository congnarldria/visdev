#include "vizion_core.h"

#include <math.h>
#include <stdlib.h>

/* 形狀分數和正規相關的取樣迴圈。雙線性公式和 C# 相同，累加順序也相同。 */

double vizion_shape_score_sse2(
    const float* gx, const float* gy, const float* mag, int width, int height,
    double column, double row, const float* mx, const float* my, const float* mgx, const float* mgy,
    int count, double min_contrast);
double vizion_shape_score_avx2(
    const float* gx, const float* gy, const float* mag, int width, int height,
    double column, double row, const float* mx, const float* my, const float* mgx, const float* mgy,
    int count, double min_contrast);
double vizion_ncc_sse2(
    const float* gray, int width, int height, const float* x, const float* y, const float* value,
    int count, double column, double row, int absolute);
double vizion_ncc_avx2(
    const float* gray, int width, int height, const float* x, const float* y, const float* value,
    int count, double column, double row, int absolute);

static int vizion_floor_int(double value)
{
    return (int)floor(value);
}

static float vizion_blend(float left, float right, float top, float bottom, float wx, float wy)
{
    float upper = (1.f - wx) * left + wx * right;
    float lower = (1.f - wx) * top + wx * bottom;
    return (1.f - wy) * upper + wy * lower;
}

double vizion_shape_score_scalar(
    const float* gx, const float* gy, const float* mag, int width, int height,
    double column, double row, const float* mx, const float* my, const float* mgx, const float* mgy,
    int count, double min_contrast)
{
    double sum = 0;
    int index;
    if (!gx || !gy || !mag || !mx || !my || !mgx || !mgy || width < 3 || height < 3 || count <= 0)
        return 0;
    for (index = 0; index < count; index++) {
        double x = column + (double)mx[index];
        double y = row + (double)my[index];
        int x0;
        int y0;
        float wx;
        float wy;
        int i00;
        float scene_mag;
        float scene_gx;
        float scene_gy;
        float inverse;
        float dot;
        if (x < 1 || y < 1 || x >= (double)width - 2 || y >= (double)height - 2)
            continue;
        x0 = vizion_floor_int(x);
        y0 = vizion_floor_int(y);
        wx = (float)(x - x0);
        wy = (float)(y - y0);
        i00 = y0 * width + x0;
        scene_mag = vizion_blend(mag[i00], mag[i00 + 1], mag[i00 + width], mag[i00 + width + 1], wx, wy);
        if ((double)scene_mag < min_contrast)
            continue;
        scene_gx = vizion_blend(gx[i00], gx[i00 + 1], gx[i00 + width], gx[i00 + width + 1], wx, wy);
        scene_gy = vizion_blend(gy[i00], gy[i00 + 1], gy[i00 + width], gy[i00 + width + 1], wx, wy);
        inverse = 1.f / (scene_mag > 1e-3f ? scene_mag : 1e-3f);
        dot = mgx[index] * scene_gx * inverse + mgy[index] * scene_gy * inverse;
        if (dot > 0)
            sum += dot;
    }
    return sum / count;
}

double vizion_ncc_scalar(
    const float* gray, int width, int height, const float* x, const float* y, const float* value,
    int count, double column, double row, int absolute)
{
    double sum_s = 0;
    double sum_t = 0;
    double sum_ss = 0;
    double sum_tt = 0;
    double sum_st = 0;
    int seen = 0;
    int min_count;
    int index;
    double var_s;
    double var_t;
    double cov;
    double score;
    if (!gray || !x || !y || !value || width < 2 || height < 2 || count <= 0)
        return 0;
    min_count = (int)(count * 0.65);
    if (min_count < 16)
        min_count = 16;
    for (index = 0; index < count; index++) {
        double px = column + (double)x[index];
        double py = row + (double)y[index];
        int x0 = vizion_floor_int(px);
        int y0 = vizion_floor_int(py);
        float wx;
        float wy;
        int i00;
        float sample;
        float template;
        if (x0 < 0 || y0 < 0 || x0 + 1 >= width || y0 + 1 >= height)
            continue;
        wx = (float)(px - x0);
        wy = (float)(py - y0);
        i00 = y0 * width + x0;
        sample = vizion_blend(gray[i00], gray[i00 + 1], gray[i00 + width], gray[i00 + width + 1], wx, wy);
        template = value[index];
        sum_s += sample;
        sum_t += template;
        sum_ss += (double)sample * sample;
        sum_tt += (double)template * template;
        sum_st += (double)sample * template;
        seen++;
    }
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

double vizion_shape_score(
    const float* gx, const float* gy, const float* mag, int width, int height,
    double column, double row, const float* mx, const float* my, const float* mgx, const float* mgy,
    int count, double min_contrast)
{
    int flags = vizion_cpu_flags();
    if (flags & VIZION_CPU_SELECTED_AVX2)
        return vizion_shape_score_avx2(gx, gy, mag, width, height, column, row, mx, my, mgx, mgy, count, min_contrast);
    if (flags & VIZION_CPU_SELECTED_SSE2)
        return vizion_shape_score_sse2(gx, gy, mag, width, height, column, row, mx, my, mgx, mgy, count, min_contrast);
    return vizion_shape_score_scalar(gx, gy, mag, width, height, column, row, mx, my, mgx, mgy, count, min_contrast);
}

double vizion_ncc_correlate(
    const float* gray, int width, int height, const float* x, const float* y, const float* value,
    int count, double column, double row, int absolute)
{
    int flags = vizion_cpu_flags();
    if (flags & VIZION_CPU_SELECTED_AVX2)
        return vizion_ncc_avx2(gray, width, height, x, y, value, count, column, row, absolute);
    if (flags & VIZION_CPU_SELECTED_SSE2)
        return vizion_ncc_sse2(gray, width, height, x, y, value, count, column, row, absolute);
    return vizion_ncc_scalar(gray, width, height, x, y, value, count, column, row, absolute);
}

static int vizion_near(double left, double right)
{
    double diff = left - right;
    if (diff < 0)
        diff = -diff;
    /* x86 純 C 走 x87，SSE 是 32 位元浮點，允許這一點捨入差。 */
    return diff <= 1e-5 || diff <= 1e-5 * (fabs(left) + fabs(right) + 1.0);
}

int vizion_score_selftest(void)
{
    enum { width = 24, height = 18, count = 40 };
    float gx[width * height];
    float gy[width * height];
    float mag[width * height];
    float gray[width * height];
    float mx[count];
    float my[count];
    float mgx[count];
    float mgy[count];
    float value[count];
    int i;
    int flags = vizion_cpu_flags();
    double scalar;
    double fast;
    for (i = 0; i < width * height; i++) {
        gray[i] = (float)((i * 17) % 251);
        gx[i] = gray[i] - 80.f;
        gy[i] = (float)(i % width) - 6.f;
        mag[i] = fabsf(gx[i]) + fabsf(gy[i]) + 2.f;
    }
    for (i = 0; i < count; i++) {
        mx[i] = (float)((i % 9) - 4);
        my[i] = (float)((i % 7) - 3);
        mgx[i] = (i & 1) ? 0.6f : -0.4f;
        mgy[i] = 0.2f;
        value[i] = gray[(i * 3) % (width * height)];
    }
    scalar = vizion_shape_score_scalar(gx, gy, mag, width, height, 11.25, 8.5, mx, my, mgx, mgy, count, 3.0);
    if (flags & VIZION_CPU_SSE2) {
        fast = vizion_shape_score_sse2(gx, gy, mag, width, height, 11.25, 8.5, mx, my, mgx, mgy, count, 3.0);
        if (!vizion_near(scalar, fast))
            return 1;
        fast = vizion_ncc_sse2(gray, width, height, mx, my, value, count, 11.25, 8.5, 1);
        scalar = vizion_ncc_scalar(gray, width, height, mx, my, value, count, 11.25, 8.5, 1);
        if (!vizion_near(scalar, fast))
            return 2;
    }
    scalar = vizion_shape_score_scalar(gx, gy, mag, width, height, 11.25, 8.5, mx, my, mgx, mgy, count, 3.0);
    if (flags & VIZION_CPU_AVX2) {
        fast = vizion_shape_score_avx2(gx, gy, mag, width, height, 11.25, 8.5, mx, my, mgx, mgy, count, 3.0);
        if (!vizion_near(scalar, fast))
            return 3;
        fast = vizion_ncc_avx2(gray, width, height, mx, my, value, count, 4.5, 6.25, 0);
        scalar = vizion_ncc_scalar(gray, width, height, mx, my, value, count, 4.5, 6.25, 0);
        if (!vizion_near(scalar, fast))
            return 4;
    }
    if (!vizion_near(vizion_shape_score(gx, gy, mag, width, height, 11.25, 8.5, mx, my, mgx, mgy, count, 3.0),
            vizion_shape_score_scalar(gx, gy, mag, width, height, 11.25, 8.5, mx, my, mgx, mgy, count, 3.0)))
        return 5;
    return 0;
}
