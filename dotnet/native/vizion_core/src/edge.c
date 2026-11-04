#include "vizion_core.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 次像素邊緣的可分離梯度和非極大抑制。卷積順序和 C# 相同，-0.5 的邊界留在同一側。 */

int vizion_sep_sse2(const float* src, float* dst, int width, int height, const float* across, int across_len, const float* along, int along_len);
int vizion_sep_avx2(const float* src, float* dst, int width, int height, const float* across, int across_len, const float* along, int along_len);
void vizion_remap_avx2(const uint8_t* src, int width, int height, const double* map_x, const double* map_y, uint8_t* dst);
void vizion_remap_sse2(const uint8_t* src, int width, int height, const double* map_x, const double* map_y, uint8_t* dst);

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

int vizion_sep_scalar(const float* src, float* dst, int width, int height, const float* across, int across_len, const float* along, int along_len)
{
    int radius_x = across_len / 2;
    int radius_y = along_len / 2;
    float* horizontal = (float*)malloc((size_t)width * (size_t)height * sizeof(float));
    int row;
    int col;
    if (!horizontal)
        return -1;
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            float sum = 0;
            int tap;
            for (tap = 0; tap < across_len; tap++) {
                int xx = vizion_reflect(col + tap - radius_x, width);
                sum += src[(size_t)row * (size_t)width + xx] * across[tap];
            }
            horizontal[(size_t)row * (size_t)width + col] = sum;
        }
    }
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
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

static int vizion_sep(const float* src, float* dst, int width, int height, const float* across, int across_len, const float* along, int along_len)
{
    int flags = vizion_cpu_flags();
    if (flags & VIZION_CPU_SELECTED_AVX2)
        return vizion_sep_avx2(src, dst, width, height, across, across_len, along, along_len);
    if (flags & VIZION_CPU_SELECTED_SSE2)
        return vizion_sep_sse2(src, dst, width, height, across, across_len, along, along_len);
    return vizion_sep_scalar(src, dst, width, height, across, across_len, along, along_len);
}

static float vizion_sample(const float* field, int width, int height, float row, float column)
{
    int x0, y0, x1, y1;
    float wx, wy, top, bottom;
    if (column < 0)
        column = 0;
    if (row < 0)
        row = 0;
    if (column > (float)width - 1)
        column = (float)width - 1;
    if (row > (float)height - 1)
        row = (float)height - 1;
    x0 = (int)floorf(column);
    y0 = (int)floorf(row);
    x1 = x0 + 1 < width ? x0 + 1 : width - 1;
    y1 = y0 + 1 < height ? y0 + 1 : height - 1;
    wx = column - (float)x0;
    wy = row - (float)y0;
    top = (1.f - wx) * field[(size_t)y0 * (size_t)width + x0] + wx * field[(size_t)y0 * (size_t)width + x1];
    bottom = (1.f - wx) * field[(size_t)y1 * (size_t)width + x0] + wx * field[(size_t)y1 * (size_t)width + x1];
    return (1.f - wy) * top + wy * bottom;
}

int vizion_subpix(
    const float* gray, int width, int height,
    const float* smooth, int smooth_len, const float* deriv, int deriv_len,
    float* dx, float* dy, float* magnitude, float* normal_x, float* normal_y,
    float* shift, uint8_t* center)
{
    int count;
    int index;
    if (!gray || !smooth || !deriv || !dx || !dy || !magnitude || !normal_x || !normal_y || !shift || !center)
        return -1;
    if (width < 3 || height < 3 || smooth_len < 1 || deriv_len < 1 || (smooth_len & 1) == 0 || (deriv_len & 1) == 0)
        return -1;
    if (vizion_sep(gray, dx, width, height, deriv, deriv_len, smooth, smooth_len) != 0)
        return -1;
    if (vizion_sep(gray, dy, width, height, smooth, smooth_len, deriv, deriv_len) != 0)
        return -1;
    count = width * height;
    for (index = 0; index < count; index++) {
        float sq = dx[index] * dx[index] + dy[index] * dy[index];
        float mag = (float)sqrt((double)sq);
        float scale = mag > 1e-6f ? mag : 1e-6f;
        magnitude[index] = mag;
        normal_x[index] = dx[index] / scale;
        normal_y[index] = dy[index] / scale;
    }
    for (index = 0; index < count; index++) {
        int col = index % width;
        int row = index / width;
        float ahead = vizion_sample(magnitude, width, height, (float)row + normal_y[index], (float)col + normal_x[index]);
        float behind = vizion_sample(magnitude, width, height, (float)row - normal_y[index], (float)col - normal_x[index]);
        float denom = behind + ahead - 2.f * magnitude[index];
        float moved = fabsf(denom) > 1e-6f ? 0.5f * (behind - ahead) / denom : 9.f;
        int on = magnitude[index] >= ahead && magnitude[index] >= behind && magnitude[index] > 0.f
            && moved > -0.5f + 1e-6f && moved <= 0.5f;
        shift[index] = moved;
        center[index] = (uint8_t)(on ? 1 : 0);
    }
    return 0;
}

static void vizion_remap_scalar(const uint8_t* src, int width, int height, const double* map_x, const double* map_y, uint8_t* dst)
{
    int count = width * height;
    int index;
    double limit_x = (double)width - 1;
    double limit_y = (double)height - 1;
    for (index = 0; index < count; index++) {
        double column = map_x[index];
        double row = map_y[index];
        int x0, y0, x1, y1, rounded;
        double wx, wy, top, bottom, value;
        if (column < 0 || row < 0 || column > limit_x || row > limit_y) {
            dst[index] = 0;
            continue;
        }
        x0 = (int)column;
        y0 = (int)row;
        x1 = x0 + 1 < width ? x0 + 1 : width - 1;
        y1 = y0 + 1 < height ? y0 + 1 : height - 1;
        wx = column - x0;
        wy = row - y0;
        top = src[(size_t)y0 * (size_t)width + x0] * (1.0 - wx) + src[(size_t)y0 * (size_t)width + x1] * wx;
        bottom = src[(size_t)y1 * (size_t)width + x0] * (1.0 - wx) + src[(size_t)y1 * (size_t)width + x1] * wx;
        value = top * (1.0 - wy) + bottom * wy;
        rounded = (int)(value + 0.5);
        if (rounded < 0)
            rounded = 0;
        if (rounded > 255)
            rounded = 255;
        dst[index] = (uint8_t)rounded;
    }
}

int vizion_remap(
    const uint8_t* src, int width, int height,
    const double* map_x, const double* map_y, uint8_t* dst)
{
    if (!src || !map_x || !map_y || !dst || width < 1 || height < 1)
        return -1;
    /* 來源座標不連續，逐像素算比先裝進 SSE/AVX 暫存器快。自我檢查仍核對那兩條路徑。 */
    vizion_remap_scalar(src, width, height, map_x, map_y, dst);
    return 0;
}

static int vizion_near(float left, float right)
{
    float diff = left - right;
    if (diff < 0)
        diff = -diff;
    return diff <= 1e-5f;
}

static void vizion_test_kernels(double sigma, float* smooth, float* deriv, int* out_len)
{
    int radius = (int)ceil(sigma * 4.0);
    int length;
    int i;
    double acc = 0;
    double smooth64[32];
    double deriv64[32];
    double variance;
    if (radius < 1)
        radius = 1;
    length = radius * 2 + 1;
    variance = sigma * sigma;
    for (i = 0; i < length; i++) {
        double axis = i - radius;
        smooth64[i] = exp(-0.5 * axis * axis / variance);
        acc += smooth64[i];
    }
    for (i = 0; i < length; i++) {
        double axis = i - radius;
        smooth64[i] /= acc;
        deriv64[i] = (-axis / variance) * smooth64[i];
        smooth[i] = (float)smooth64[i];
        deriv[i] = (float)deriv64[i];
    }
    *out_len = length;
}

int vizion_edge_selftest(void)
{
    enum { kStep = 32 * 24 };
    uint8_t image[41 * 41];
    double map_x[41 * 41];
    double map_y[41 * 41];
    uint8_t dst[41 * 41];
    uint8_t other[41 * 41];
    float gray[16 * 12];
    float kernel_a[3] = {0.25f, 0.5f, 0.25f};
    float kernel_b[3] = {-0.5f, 0.f, 0.5f};
    float scalar[16 * 12];
    float fast[16 * 12];
    float step[kStep];
    float smooth[16];
    float deriv[16];
    float dx[kStep];
    float dy[kStep];
    float magnitude[kStep];
    float normal_x[kStep];
    float normal_y[kStep];
    float shift[kStep];
    uint8_t center[kStep];
    int kernel_len = 0;
    int centers = 0;
    int i;
    int flags = vizion_cpu_flags();
    memset(image, 0, sizeof(image));
    image[20 * 41 + 35] = 255;
    for (i = 0; i < 41 * 41; i++) {
        map_x[i] = 0;
        map_y[i] = 0;
    }
    map_x[20 * 41 + 36] = 34.72;
    map_y[20 * 41 + 36] = 20;
    vizion_remap_scalar(image, 41, 41, map_x, map_y, other);
    if (other[20 * 41 + 36] != 184)
        return 1;
    if (vizion_remap(image, 41, 41, map_x, map_y, dst) != 0 || memcmp(dst, other, sizeof(dst)) != 0)
        return 1;
    if (flags & VIZION_CPU_SSE2) {
        memset(dst, 0, sizeof(dst));
        vizion_remap_sse2(image, 41, 41, map_x, map_y, dst);
        if (memcmp(dst, other, sizeof(dst)) != 0)
            return 2;
    }
    if (flags & VIZION_CPU_AVX2) {
        memset(dst, 0, sizeof(dst));
        vizion_remap_avx2(image, 41, 41, map_x, map_y, dst);
        if (memcmp(dst, other, sizeof(dst)) != 0)
            return 3;
    }
    for (i = 0; i < 16 * 12; i++)
        gray[i] = (float)((i * 13) % 90);
    if (vizion_sep_scalar(gray, scalar, 16, 12, kernel_b, 3, kernel_a, 3) != 0)
        return 4;
    if (flags & VIZION_CPU_SSE2) {
        vizion_sep_sse2(gray, fast, 16, 12, kernel_b, 3, kernel_a, 3);
        for (i = 0; i < 16 * 12; i++) {
            if (!vizion_near(scalar[i], fast[i]))
                return 5;
        }
    }
    if (flags & VIZION_CPU_AVX2) {
        vizion_sep_avx2(gray, fast, 16, 12, kernel_b, 3, kernel_a, 3);
        for (i = 0; i < 16 * 12; i++) {
            if (!vizion_near(scalar[i], fast[i]))
                return 6;
        }
    }
    memset(step, 0, sizeof(step));
    for (i = 0; i < 24; i++) {
        int col;
        for (col = 16; col < 32; col++)
            step[i * 32 + col] = 255.f;
    }
    vizion_test_kernels(1.0, smooth, deriv, &kernel_len);
    if (vizion_subpix(step, 32, 24, smooth, kernel_len, deriv, kernel_len, dx, dy, magnitude, normal_x, normal_y, shift, center) != 0)
        return 7;
    for (i = 0; i < kStep; i++) {
        int col;
        int row;
        float column;
        if (!center[i] || magnitude[i] < 20.f)
            continue;
        col = i % 32;
        row = i / 32;
        column = (float)col + shift[i] * normal_x[i];
        if (fabsf(column - 15.5f) > 1e-3f || fabsf((float)row + shift[i] * normal_y[i] - (float)row) > 1e-3f) {
            fprintf(stderr, "邊緣 %d 行 %d 列 %.6f\n", i, col, column);
            return 8;
        }
        centers++;
    }
    if (centers != 24) {
        fprintf(stderr, "階躍中心 %d\n", centers);
        return 9;
    }
    return 0;
}
