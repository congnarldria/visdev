#include "vizion_core.h"

#include "rotate_internal.h"

#include <string.h>

/* 繞中心最近鄰旋轉。角度的正弦餘弦由 C# 算好。四捨五入遠離 0，圖外補 0。 */

void vizion_rotate_sse2(const uint8_t* src, uint8_t* dst, int width, int height, double cosine, double sine);
void vizion_rotate_avx2(const uint8_t* src, uint8_t* dst, int width, int height, double cosine, double sine);

void vizion_rotate_scalar(const uint8_t* src, uint8_t* dst, int width, int height, double cosine, double sine)
{
    double center_column = (width - 1) / 2.0;
    double center_row = (height - 1) / 2.0;
    int row;
    int column;
    memset(dst, 0, (size_t)width * (size_t)height);
    for (row = 0; row < height; row++) {
        double dy = (double)row - center_row;
        for (column = 0; column < width; column++) {
            double dx = (double)column - center_column;
            double source_column = center_column + dx * cosine - dy * sine;
            double source_row = center_row + dx * sine + dy * cosine;
            vizion_rotate_write(src, dst, width, height, row, column, source_row, source_column);
        }
    }
}

int vizion_rotate_nearest(const uint8_t* src, uint8_t* dst, int width, int height, double cosine, double sine)
{
    int flags;
    if (!src || !dst || width < 1 || height < 1)
        return -1;
    flags = vizion_cpu_flags();
    if (flags & VIZION_CPU_SELECTED_AVX2)
        vizion_rotate_avx2(src, dst, width, height, cosine, sine);
    else if (flags & VIZION_CPU_SELECTED_SSE2)
        vizion_rotate_sse2(src, dst, width, height, cosine, sine);
    else
        vizion_rotate_scalar(src, dst, width, height, cosine, sine);
    return 0;
}

int vizion_rotate_selftest(void)
{
    enum { kW = 40, kH = 40, kN = 40 * 40, kP = 18 * 13 };
    uint8_t band[kN];
    uint8_t turned[kN];
    uint8_t image[kP];
    uint8_t scalar[kP];
    uint8_t fast[kP];
    int row;
    int column;
    int i;
    int flags = vizion_cpu_flags();
    memset(band, 0, sizeof(band));
    for (row = 16; row <= 23; row++) {
        for (column = 0; column < kW; column++)
            band[row * kW + column] = 255;
    }
    vizion_rotate_scalar(band, turned, kW, kH, 0.0, 1.0);
    if (turned[20 * kW + 15] != 0 || turned[20 * kW + 16] != 255 || turned[20 * kW + 23] != 255 || turned[20 * kW + 24] != 0)
        return 1;
    for (i = 0; i < kP; i++)
        image[i] = (uint8_t)((i * 13 + 4) & 255);
    vizion_rotate_scalar(image, scalar, 18, 13, 0.7, 0.2);
    if ((flags & VIZION_CPU_SSE2) && (flags & VIZION_CPU_AVX2)) {
        vizion_rotate_sse2(image, scalar, 18, 13, 0.7, 0.2);
        vizion_rotate_avx2(image, fast, 18, 13, 0.7, 0.2);
        if (memcmp(scalar, fast, kP) != 0)
            return 2;
    }
    vizion_rotate_nearest(image, fast, 18, 13, 1.0, 0.0);
    if (memcmp(image, fast, kP) != 0)
        return 3;
    return 0;
}
