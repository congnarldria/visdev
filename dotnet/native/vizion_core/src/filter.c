#include "vizion_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* 平均是可分離的盒狀濾波，邊界用 OpenCV 的 REFLECT_101。面積是 2 的次方時，捨入比四捨五入多 1/面積。
   高斯用和 cv2.GaussianBlur(..., 0) 相同的整數核，兩次卷積後再四捨五入。 */

int vizion_mean_sse2(const uint8_t* src, uint8_t* dst, int width, int height, int mask_width, int mask_height);
int vizion_mean_avx2(const uint8_t* src, uint8_t* dst, int width, int height, int mask_width, int mask_height);

int vizion_reflect101(int position, int length)
{
    if (length <= 1)
        return 0;
    while (position < 0 || position >= length) {
        if (position < 0)
            position = -position;
        else
            position = length * 2 - position - 2;
    }
    return position;
}

int vizion_mean_div(unsigned int sum, int area)
{
    if (area <= 1)
        return (int)sum;
    if ((area & (area - 1)) == 0)
        return (int)((sum + (unsigned int)(area / 2 + 1)) / (unsigned int)area);
    return (int)((double)sum / (double)area + 0.5);
}

static unsigned int vizion_box_sum(const uint8_t* src, int length, int index, int kernel, int anchor)
{
    unsigned int sum = 0;
    int tap;
    for (tap = 0; tap < kernel; tap++)
        sum += src[vizion_reflect101(index + tap - anchor, length)];
    return sum;
}

int vizion_mean_scalar(const uint8_t* src, uint8_t* dst, int width, int height, int mask_width, int mask_height)
{
    unsigned int* horizontal;
    int row;
    int col;
    int anchor_x = mask_width / 2;
    int anchor_y = mask_height / 2;
    int area = mask_width * mask_height;
    if (!src || !dst || width < 1 || height < 1 || mask_width < 1 || mask_height < 1)
        return -1;
    horizontal = (unsigned int*)malloc((size_t)width * (size_t)height * sizeof(unsigned int));
    if (!horizontal)
        return -1;
    for (row = 0; row < height; row++) {
        const uint8_t* line = src + (size_t)row * (size_t)width;
        unsigned int* out = horizontal + (size_t)row * (size_t)width;
        unsigned int sum = vizion_box_sum(line, width, 0, mask_width, anchor_x);
        out[0] = sum;
        for (col = 1; col < width; col++) {
            sum += line[vizion_reflect101(col + mask_width - 1 - anchor_x, width)];
            sum -= line[vizion_reflect101(col - 1 - anchor_x, width)];
            out[col] = sum;
        }
    }
    for (col = 0; col < width; col++) {
        unsigned int sum = 0;
        int tap;
        for (tap = 0; tap < mask_height; tap++)
            sum += horizontal[(size_t)vizion_reflect101(tap - anchor_y, height) * (size_t)width + col];
        dst[col] = (uint8_t)vizion_mean_div(sum, area);
        for (row = 1; row < height; row++) {
            sum += horizontal[(size_t)vizion_reflect101(row + mask_height - 1 - anchor_y, height) * (size_t)width + col];
            sum -= horizontal[(size_t)vizion_reflect101(row - 1 - anchor_y, height) * (size_t)width + col];
            dst[(size_t)row * (size_t)width + col] = (uint8_t)vizion_mean_div(sum, area);
        }
    }
    free(horizontal);
    return 0;
}

static void vizion_copy_kernel(int* dst, const int* src, int count, int* scale)
{
    int i;
    int sum = 0;
    for (i = 0; i < count; i++) {
        dst[i] = src[i];
        sum += src[i];
    }
    *scale = sum;
}

static void vizion_kernel_formula(int n, int* dst, int* scale)
{
    double sigma = 0.3 * (((n - 1) * 0.5) - 1.0) + 0.8;
    double inv = -0.5 / (sigma * sigma);
    double raw[128];
    double sum = 0;
    int i;
    int total = 0;
    if (n > 128)
        n = 128;
    for (i = 0; i < n; i++) {
        double x = i - (n - 1) * 0.5;
        raw[i] = exp(inv * x * x);
        sum += raw[i];
    }
    for (i = 0; i < n; i++) {
        double scaled = raw[i] / sum * 256.0;
        dst[i] = (int)(scaled + 0.5);
        total += dst[i];
    }
    dst[n / 2] += 256 - total;
    *scale = 0;
    for (i = 0; i < n; i++)
        *scale += dst[i];
}

static int vizion_gauss_kernel(int n, int* dst, int* scale)
{
    static const int k3[] = {64, 128, 64};
    static const int k5[] = {16, 64, 96, 64, 16};
    static const int k7[] = {8, 28, 56, 72, 56, 28, 8};
    static const int k9[] = {4, 13, 30, 51, 60, 51, 30, 13, 4};
    static const int k11[] = {2, 7, 17, 31, 45, 52, 45, 31, 17, 7, 2};
    static const int k13[] = {1, 5, 10, 19, 30, 41, 44, 41, 30, 19, 10, 5, 1};
    static const int k15[] = {1, 3, 6, 12, 20, 30, 36, 40, 36, 30, 20, 12, 6, 3, 1};
    static const int k17[] = {1, 2, 4, 8, 13, 21, 28, 33, 36, 33, 28, 21, 13, 8, 4, 2, 1};
    static const int k19[] = {1, 1, 3, 5, 10, 15, 20, 27, 30, 32, 30, 27, 20, 15, 10, 5, 3, 1, 1};
    static const int k21[] = {0, 2, 2, 4, 6, 11, 15, 20, 25, 28, 30, 28, 25, 20, 15, 11, 6, 4, 2, 2, 0};
    static const int k23[] = {0, 1, 2, 3, 5, 8, 11, 15, 20, 24, 26, 26, 26, 24, 20, 15, 11, 8, 5, 3, 2, 1, 0};
    static const int k25[] = {0, 1, 1, 3, 3, 6, 9, 12, 15, 19, 22, 25, 24, 25, 22, 19, 15, 12, 9, 6, 3, 3, 1, 1, 0};
    static const int k27[] = {0, 1, 1, 2, 3, 4, 7, 9, 12, 15, 19, 21, 22, 24, 22, 21, 19, 15, 12, 9, 7, 4, 3, 2, 1, 1, 0};
    static const int k29[] = {0, 1, 1, 1, 2, 4, 5, 7, 10, 12, 15, 18, 20, 21, 22, 21, 20, 18, 15, 12, 10, 7, 5, 4, 2, 1, 1, 1, 0};
    static const int k31[] = {0, 1, 0, 1, 2, 3, 4, 6, 7, 10, 13, 15, 17, 19, 20, 20, 20, 19, 17, 15, 13, 10, 7, 6, 4, 3, 2, 1, 0, 1, 0};
    const int* table = 0;
    if (n == 3) table = k3;
    else if (n == 5) table = k5;
    else if (n == 7) table = k7;
    else if (n == 9) table = k9;
    else if (n == 11) table = k11;
    else if (n == 13) table = k13;
    else if (n == 15) table = k15;
    else if (n == 17) table = k17;
    else if (n == 19) table = k19;
    else if (n == 21) table = k21;
    else if (n == 23) table = k23;
    else if (n == 25) table = k25;
    else if (n == 27) table = k27;
    else if (n == 29) table = k29;
    else if (n == 31) table = k31;
    if (table) {
        vizion_copy_kernel(dst, table, n, scale);
        return 0;
    }
    vizion_kernel_formula(n, dst, scale);
    return 0;
}

int vizion_gauss_scalar(const uint8_t* src, uint8_t* dst, int width, int height, int size)
{
    int kernel[128];
    int scale = 0;
    int anchor;
    long long* horizontal;
    long long divisor;
    int row;
    int col;
    if (!src || !dst || width < 1 || height < 1 || size < 3 || (size & 1) == 0 || size > 127)
        return -1;
    vizion_gauss_kernel(size, kernel, &scale);
    if (scale <= 0)
        return -1;
    anchor = size / 2;
    divisor = (long long)scale * (long long)scale;
    horizontal = (long long*)malloc((size_t)width * (size_t)height * sizeof(long long));
    if (!horizontal)
        return -1;
    for (row = 0; row < height; row++) {
        const uint8_t* line = src + (size_t)row * (size_t)width;
        for (col = 0; col < width; col++) {
            long long sum = 0;
            int tap;
            for (tap = 0; tap < size; tap++)
                sum += (long long)line[vizion_reflect101(col + tap - anchor, width)] * kernel[tap];
            horizontal[(size_t)row * (size_t)width + col] = sum;
        }
    }
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            long long sum = 0;
            int tap;
            int value;
            for (tap = 0; tap < size; tap++)
                sum += horizontal[(size_t)vizion_reflect101(row + tap - anchor, height) * (size_t)width + col] * kernel[tap];
            value = (int)((sum + divisor / 2) / divisor);
            if (value < 0)
                value = 0;
            if (value > 255)
                value = 255;
            dst[(size_t)row * (size_t)width + col] = (uint8_t)value;
        }
    }
    free(horizontal);
    return 0;
}

int vizion_mean(const uint8_t* src, uint8_t* dst, int width, int height, int mask_width, int mask_height)
{
    int flags = vizion_cpu_flags();
    if ((unsigned long long)mask_width * (unsigned long long)mask_height * 255ull > 4294967295ull)
        return vizion_mean_scalar(src, dst, width, height, mask_width, mask_height);
    if (flags & VIZION_CPU_SELECTED_AVX2)
        return vizion_mean_avx2(src, dst, width, height, mask_width, mask_height);
    if (flags & VIZION_CPU_SELECTED_SSE2)
        return vizion_mean_sse2(src, dst, width, height, mask_width, mask_height);
    return vizion_mean_scalar(src, dst, width, height, mask_width, mask_height);
}

int vizion_gauss(const uint8_t* src, uint8_t* dst, int width, int height, int size)
{
    return vizion_gauss_scalar(src, dst, width, height, size);
}

static int vizion_same(const uint8_t* left, const uint8_t* right, int count)
{
    int i;
    for (i = 0; i < count; i++) {
        if (left[i] != right[i])
            return 0;
    }
    return 1;
}

static void vizion_naive_mean(const uint8_t* src, uint8_t* dst, int width, int height, int mask_width, int mask_height)
{
    int anchor_x = mask_width / 2;
    int anchor_y = mask_height / 2;
    int area = mask_width * mask_height;
    int row;
    int col;
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            unsigned int sum = 0;
            int dy;
            int dx;
            for (dy = 0; dy < mask_height; dy++) {
                int sample_row = vizion_reflect101(row + dy - anchor_y, height);
                for (dx = 0; dx < mask_width; dx++) {
                    int sample_col = vizion_reflect101(col + dx - anchor_x, width);
                    sum += src[(size_t)sample_row * (size_t)width + sample_col];
                }
            }
            dst[(size_t)row * (size_t)width + col] = (uint8_t)vizion_mean_div(sum, area);
        }
    }
}

int vizion_filter_selftest(void)
{
    static const uint8_t grid[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    static const uint8_t mean3[16] = {3, 4, 5, 5, 5, 5, 6, 6, 9, 9, 10, 10, 10, 10, 11, 12};
    static const uint8_t mean2[16] = {3, 3, 4, 5, 3, 3, 4, 5, 7, 7, 8, 9, 11, 11, 12, 13};
    static const uint8_t spike[49] = {
        0, 0, 0, 0, 0, 0, 0,
        0, 1, 4, 6, 4, 1, 0,
        0, 4, 16, 24, 16, 4, 0,
        0, 6, 24, 36, 24, 6, 0,
        0, 4, 16, 24, 16, 4, 0,
        0, 1, 4, 6, 4, 1, 0,
        0, 0, 0, 0, 0, 0, 0
    };
    uint8_t src[49];
    uint8_t dst[49];
    uint8_t naive[64];
    uint8_t fast[64];
    int flags = vizion_cpu_flags();
    int i;
    if (vizion_mean_scalar(grid, dst, 4, 4, 3, 3) != 0 || !vizion_same(dst, mean3, 16))
        return 1;
    if (vizion_mean_scalar(grid, dst, 4, 4, 2, 2) != 0 || !vizion_same(dst, mean2, 16))
        return 2;
    memset(src, 0, sizeof(src));
    src[3 * 7 + 3] = 255;
    if (vizion_gauss_scalar(src, dst, 7, 7, 5) != 0 || !vizion_same(dst, spike, 49))
        return 3;
    for (i = 0; i < 48; i++)
        naive[i] = (uint8_t)(i * 17);
    vizion_naive_mean(naive, dst, 8, 6, 4, 3);
    if (vizion_mean(naive, fast, 8, 6, 4, 3) != 0 || !vizion_same(dst, fast, 48))
        return 4;
    if ((flags & VIZION_CPU_SSE2) && (vizion_mean_sse2(naive, fast, 8, 6, 4, 3) != 0 || !vizion_same(dst, fast, 48)))
        return 5;
    if ((flags & VIZION_CPU_AVX2) && (vizion_mean_avx2(naive, fast, 8, 6, 4, 3) != 0 || !vizion_same(dst, fast, 48)))
        return 6;
    {
        uint8_t wide[16 * 12];
        uint8_t expect[16 * 12];
        uint8_t got[16 * 12];
        for (i = 0; i < 16 * 12; i++)
            wide[i] = (uint8_t)(i * 13);
        vizion_naive_mean(wide, expect, 16, 12, 5, 2);
        if (vizion_mean_scalar(wide, got, 16, 12, 5, 2) != 0 || !vizion_same(expect, got, 16 * 12))
            return 8;
        if ((flags & VIZION_CPU_AVX2) && (vizion_mean_avx2(wide, got, 16, 12, 5, 2) != 0 || !vizion_same(expect, got, 16 * 12)))
            return 9;
    }
    if (vizion_gauss(naive, fast, 8, 6, 5) != 0)
        return 7;
    return 0;
}
