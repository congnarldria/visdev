#include "vizion_core.h"

#include <math.h>
#include <string.h>

/* 3x3 Sobel。邊界和 C# 的 Reflect 相同：-1 映到 0。梯度是整數，float 算完再以 double 開根號。 */

int vizion_sobel_sse2(const uint8_t* gray, int width, int height, float* gx, float* gy, float* mag);
int vizion_sobel_avx2(const uint8_t* gray, int width, int height, float* gx, float* gy, float* mag);

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

static float vizion_at(const uint8_t* gray, int width, int height, int row, int column)
{
    return (float)gray[(size_t)vizion_reflect(row, height) * (size_t)width + vizion_reflect(column, width)];
}

void vizion_sobel_pixel(const uint8_t* gray, int width, int height, int row, int column, float* dx, float* dy)
{
    float up_left = vizion_at(gray, width, height, row - 1, column - 1);
    float up = vizion_at(gray, width, height, row - 1, column);
    float up_right = vizion_at(gray, width, height, row - 1, column + 1);
    float left = vizion_at(gray, width, height, row, column - 1);
    float right = vizion_at(gray, width, height, row, column + 1);
    float down_left = vizion_at(gray, width, height, row + 1, column - 1);
    float down = vizion_at(gray, width, height, row + 1, column);
    float down_right = vizion_at(gray, width, height, row + 1, column + 1);
    *dx = -up_left + up_right - 2.f * left + 2.f * right - down_left + down_right;
    *dy = -up_left - 2.f * up - up_right + down_left + 2.f * down + down_right;
}

void vizion_sobel_mag(const float* gx, const float* gy, float* mag, int count)
{
    int index;
    for (index = 0; index < count; index++) {
        float sq = gx[index] * gx[index] + gy[index] * gy[index];
        mag[index] = (float)sqrt((double)sq);
    }
}

int vizion_sobel_scalar(const uint8_t* gray, int width, int height, float* gx, float* gy, float* mag)
{
    int row;
    int col;
    int count = width * height;
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            float dx, dy;
            vizion_sobel_pixel(gray, width, height, row, col, &dx, &dy);
            gx[(size_t)row * (size_t)width + col] = dx;
            gy[(size_t)row * (size_t)width + col] = dy;
        }
    }
    vizion_sobel_mag(gx, gy, mag, count);
    return 0;
}

int vizion_sobel(const uint8_t* gray, int width, int height, float* gx, float* gy, float* mag)
{
    int flags;
    if (!gray || !gx || !gy || !mag || width < 1 || height < 1)
        return -1;
    flags = vizion_cpu_flags();
    if (flags & VIZION_CPU_SELECTED_AVX2)
        return vizion_sobel_avx2(gray, width, height, gx, gy, mag);
    if (flags & VIZION_CPU_SELECTED_SSE2)
        return vizion_sobel_sse2(gray, width, height, gx, gy, mag);
    return vizion_sobel_scalar(gray, width, height, gx, gy, mag);
}

static int vizion_same(const float* left, const float* right, int count)
{
    int index;
    for (index = 0; index < count; index++) {
        if (memcmp(left + index, right + index, sizeof(float)) != 0)
            return 0;
    }
    return 1;
}

int vizion_sobel_selftest(void)
{
    enum { kW = 18, kH = 11, kN = kW * kH };
    uint8_t step[5 * 3];
    uint8_t image[kN];
    float gx[kN], gy[kN], mag[kN];
    float fgx[kN], fgy[kN], fmag[kN];
    int row;
    int i;
    int flags = vizion_cpu_flags();
    memset(step, 0, sizeof(step));
    for (row = 0; row < 3; row++) {
        step[row * 5 + 3] = 255;
        step[row * 5 + 4] = 255;
    }
    if (vizion_sobel_scalar(step, 5, 3, gx, gy, mag) != 0)
        return 1;
    if (gx[1 * 5 + 2] != 1020.f || gy[1 * 5 + 2] != 0.f || mag[1 * 5 + 2] != 1020.f)
        return 2;
    for (i = 0; i < kN; i++)
        image[i] = (uint8_t)((i * 17 + 3) & 255);
    if (vizion_sobel_scalar(image, kW, kH, gx, gy, mag) != 0)
        return 3;
    if (flags & VIZION_CPU_SSE2) {
        vizion_sobel_sse2(image, kW, kH, fgx, fgy, fmag);
        if (!vizion_same(gx, fgx, kN) || !vizion_same(gy, fgy, kN) || !vizion_same(mag, fmag, kN))
            return 4;
    }
    if (flags & VIZION_CPU_AVX2) {
        vizion_sobel_avx2(image, kW, kH, fgx, fgy, fmag);
        if (!vizion_same(gx, fgx, kN) || !vizion_same(gy, fgy, kN) || !vizion_same(mag, fmag, kN))
            return 5;
    }
    if (vizion_sobel(image, kW, kH, fgx, fgy, fmag) != 0 || !vizion_same(gx, fgx, kN) || !vizion_same(mag, fmag, kN))
        return 6;
    return 0;
}
