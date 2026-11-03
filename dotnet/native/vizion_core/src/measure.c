#include "vizion_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* 沿掃描線雙線性取樣，再用梯度拋物線算邊緣偏差。公式和 gauge._locate 相同。 */

void vizion_sample_line_sse2(const uint8_t* gray, int width, int height, double x0, double y0, double dx, double dy, const double* ts, int count, double* values);
void vizion_sample_line_avx2(const uint8_t* gray, int width, int height, double x0, double y0, double dx, double dy, const double* ts, int count, double* values);

static int vizion_round_even(double value)
{
    double base = floor(value);
    double diff = value - base;
    int whole = (int)base;
    if (diff > 0.5)
        return whole + 1;
    if (diff < 0.5)
        return whole;
    return (whole & 1) ? whole + 1 : whole;
}

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

void vizion_sample_line_scalar(const uint8_t* gray, int width, int height, double x0, double y0, double dx, double dy, const double* ts, int count, double* values)
{
    int index;
    for (index = 0; index < count; index++)
        values[index] = vizion_sample(gray, width, height, x0 + dx * ts[index], y0 + dy * ts[index]);
}

static void vizion_sample_line(const uint8_t* gray, int width, int height, double x0, double y0, double dx, double dy, const double* ts, int count, double* values)
{
    int flags = vizion_cpu_flags();
    if (flags & VIZION_CPU_SELECTED_AVX2) {
        vizion_sample_line_avx2(gray, width, height, x0, y0, dx, dy, ts, count, values);
        return;
    }
    if (flags & VIZION_CPU_SELECTED_SSE2) {
        vizion_sample_line_sse2(gray, width, height, x0, y0, dx, dy, ts, count, values);
        return;
    }
    vizion_sample_line_scalar(gray, width, height, x0, y0, dx, dy, ts, count, values);
}

static double vizion_shift(const uint8_t* row, int column, int width)
{
    double left, center, right, denom, shift;
    if (column < 1 || column + 2 >= width)
        return 0;
    left = fabs((double)row[column] - row[column - 1]);
    center = fabs((double)row[column + 1] - row[column]);
    right = fabs((double)row[column + 2] - row[column + 1]);
    denom = left - 2 * center + right;
    if (fabs(denom) < 1e-6 || center < left || center < right)
        return 0;
    shift = 0.5 * (left - right) / denom;
    if (shift < -0.5)
        return -0.5;
    if (shift > 0.5)
        return 0.5;
    return shift;
}

int vizion_measure_row(const uint8_t* row, int width, double* left, double* right)
{
    double best_rise = 0;
    double best_fall = 0;
    int rise_at = -1;
    int fall_at = -1;
    int column;
    if (!row || !left || !right || width < 3)
        return -1;
    for (column = 0; column < width - 1; column++) {
        double delta = (double)row[column + 1] - row[column];
        if (delta > best_rise) {
            best_rise = delta;
            rise_at = column;
        }
        if (-delta > best_fall) {
            best_fall = -delta;
            fall_at = column;
        }
    }
    if (rise_at < 0 || fall_at < 0 || fall_at <= rise_at || best_rise <= 0 || best_fall <= 0)
        return 1;
    *left = rise_at + 0.5 + vizion_shift(row, rise_at, width);
    *right = fall_at + 0.5 + vizion_shift(row, fall_at, width);
    return 0;
}

static void vizion_smooth(double* values, int count, double sigma_px, double step)
{
    double sigma;
    int radius;
    int kernel_len;
    double* kernel;
    double* number;
    double* weight;
    double sum;
    int i;
    int offset;
    if (sigma_px <= 0.05 || count < 5)
        return;
    sigma = fmax(sigma_px / step, 0.25);
    radius = vizion_round_even(sigma * 3.0);
    if (radius < 1)
        radius = 1;
    if (radius > (count - 1) / 2)
        radius = (count - 1) / 2;
    kernel_len = radius * 2 + 1;
    kernel = (double*)malloc((size_t)kernel_len * sizeof(double));
    number = (double*)calloc((size_t)count, sizeof(double));
    weight = (double*)calloc((size_t)count, sizeof(double));
    if (!kernel || !number || !weight) {
        free(kernel);
        free(number);
        free(weight);
        return;
    }
    sum = 0;
    for (offset = -radius; offset <= radius; offset++) {
        double weight_tap = exp(-0.5 * ((double)offset / sigma) * ((double)offset / sigma));
        kernel[offset + radius] = weight_tap;
        sum += weight_tap;
    }
    for (offset = 0; offset < kernel_len; offset++)
        kernel[offset] /= sum;
    for (i = 0; i < count; i++) {
        double filled = isfinite(values[i]) ? values[i] : 0;
        double mask = isfinite(values[i]) ? 1 : 0;
        for (offset = -radius; offset <= radius; offset++) {
            int at = i + offset;
            if (at < 0 || at >= count)
                continue;
            number[at] += filled * kernel[offset + radius];
            weight[at] += mask * kernel[offset + radius];
        }
    }
    for (i = 0; i < count; i++)
        values[i] = weight[i] > 0.5 ? number[i] / weight[i] : nan("");
    free(kernel);
    free(number);
    free(weight);
}

int vizion_measure_scan(
    const uint8_t* gray, int width, int height,
    double x0, double y0, double x1, double y1,
    double sigma, double threshold, int transition,
    double* out_rows, double* out_columns, double* out_amps, double* out_dists,
    int capacity, int* written)
{
    double length = hypot(x1 - x0, y1 - y0);
    double step = 0.5;
    int count;
    double* ts;
    double* xs;
    double* ys;
    double* values;
    double* grad;
    double dx, dy, spacing;
    int index;
    int hits = 0;
    if (!written || !gray || width < 2 || height < 2)
        return -1;
    if (length < 1.5) {
        *written = 0;
        return 0;
    }
    count = (int)(length / step) + 1;
    if (count < 5) {
        *written = 0;
        return 0;
    }
    ts = (double*)malloc((size_t)count * sizeof(double));
    xs = (double*)malloc((size_t)count * sizeof(double));
    ys = (double*)malloc((size_t)count * sizeof(double));
    values = (double*)malloc((size_t)count * sizeof(double));
    grad = (double*)malloc((size_t)count * sizeof(double));
    if (!ts || !xs || !ys || !values || !grad) {
        free(ts);
        free(xs);
        free(ys);
        free(values);
        free(grad);
        return -1;
    }
    dx = (x1 - x0) / length;
    dy = (y1 - y0) / length;
    for (index = 0; index < count; index++) {
        ts[index] = length * (double)index / (double)(count - 1);
        xs[index] = x0 + dx * ts[index];
        ys[index] = y0 + dy * ts[index];
        grad[index] = nan("");
    }
    spacing = ts[1] - ts[0];
    vizion_sample_line(gray, width, height, x0, y0, dx, dy, ts, count, values);
    vizion_smooth(values, count, sigma, step);
    for (index = 1; index < count - 1; index++) {
        if (isfinite(values[index - 1]) && isfinite(values[index + 1]))
            grad[index] = (values[index + 1] - values[index - 1]) / (2 * step);
    }
    for (index = 2; index < count - 2; index++) {
        double left = grad[index - 1];
        double mid = grad[index];
        double right = grad[index + 1];
        int positive;
        int negative;
        double denom;
        double delta;
        double at;
        int i0;
        int i1;
        double frac;
        if (!isfinite(left) || !isfinite(mid) || !isfinite(right))
            continue;
        positive = mid >= threshold && mid >= left && mid > right;
        negative = mid <= -threshold && mid <= left && mid < right;
        if (transition == 1 && !positive)
            continue;
        if (transition == 2 && !negative)
            continue;
        if (transition == 0 && !(positive || negative))
            continue;
        denom = left - 2 * mid + right;
        delta = fabs(denom) < 1e-9 ? 0 : 0.5 * (left - right) / denom;
        if (delta < -1)
            delta = -1;
        if (delta > 1)
            delta = 1;
        at = index + delta;
        i0 = (int)floor(at);
        i1 = i0 + 1 < count ? i0 + 1 : count - 1;
        frac = at - i0;
        if (out_rows && hits < capacity) {
            out_rows[hits] = ys[i0] * (1 - frac) + ys[i1] * frac;
            out_columns[hits] = xs[i0] * (1 - frac) + xs[i1] * frac;
            out_amps[hits] = mid;
            out_dists[hits] = at * spacing;
        }
        hits++;
    }
    *written = hits;
    free(ts);
    free(xs);
    free(ys);
    free(values);
    free(grad);
    if (out_rows && hits > capacity)
        return -2;
    return 0;
}

static int vizion_close(double left, double right)
{
    double diff = left - right;
    if (diff < 0)
        diff = -diff;
    return diff <= 1e-6;
}

int vizion_measure_selftest(void)
{
    uint8_t row[64];
    uint8_t image[32 * 64];
    double left, right;
    double rows[4], cols[4], amps[4], dists[4];
    int written = 0;
    int column;
    int flags;
    memset(row, 0, sizeof(row));
    for (column = 16; column < 48; column++)
        row[column] = 255;
    if (vizion_measure_row(row, 64, &left, &right) != 0 || !vizion_close(left, 15.5) || !vizion_close(right, 47.5))
        return 1;
    memset(image, 0, sizeof(image));
    for (column = 16; column < 48; column++) {
        int y;
        for (y = 0; y < 32; y++)
            image[y * 64 + column] = 255;
    }
    if (vizion_measure_scan(image, 64, 32, 0, 16, 63, 16, 0, 20, 0, rows, cols, amps, dists, 4, &written) != 0 || written != 2)
        return 2;
    if (!vizion_close(rows[0], 16) || !vizion_close(cols[0], 15.5) || !vizion_close(amps[0], 255) || !vizion_close(dists[0], 15.5))
        return 3;
    if (!vizion_close(cols[1], 47.5) || !vizion_close(amps[1], -255) || !vizion_close(dists[1], 47.5))
        return 4;
    if (vizion_measure_scan(image, 64, 32, 0, 16, 63, 16, 1, 10, 1, rows, cols, amps, dists, 4, &written) != 0 || written != 1)
        return 5;
    if (!vizion_close(cols[0], 15.5) || !vizion_close(amps[0], 95.851631) || !vizion_close(dists[0], 15.5))
        return 6;
    {
        uint8_t diagonal[40 * 40];
        int y;
        memset(diagonal, 0, sizeof(diagonal));
        for (y = 0; y < 40; y++) {
            for (column = 20; column < 40; column++)
                diagonal[y * 40 + column] = 180;
        }
        if (vizion_measure_scan(diagonal, 40, 40, 5, 5, 34, 34, 0.8, 15, 1, rows, cols, amps, dists, 4, &written) != 0 || written != 1)
            return 10;
        if (!vizion_close(rows[0], 19.5) || !vizion_close(cols[0], 19.5) || !vizion_close(amps[0], 74.712939) || !vizion_close(dists[0], 20.506097))
            return 11;
    }
    flags = vizion_cpu_flags();
    if (flags & VIZION_CPU_AVX2) {
        double straight[8];
        double fast[8];
        double ts[8];
        int i;
        for (i = 0; i < 8; i++)
            ts[i] = i * 0.5;
        vizion_sample_line_scalar(image, 64, 32, 14, 16, 1, 0, ts, 8, straight);
        vizion_sample_line_avx2(image, 64, 32, 14, 16, 1, 0, ts, 8, fast);
        for (i = 0; i < 8; i++) {
            if (!vizion_close(straight[i], fast[i]))
                return 7;
        }
    }
    return 0;
}
