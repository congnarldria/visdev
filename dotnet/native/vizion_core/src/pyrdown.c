#include "vizion_core.h"

#include <stdlib.h>
#include <string.h>

/* 5 點 [1,4,6,4,1]，再隔一點取樣。(sum + 128) / 256。邊界和 C# 的 Reflect 相同。 */

static const int vizion_pyr_kernel[5] = {1, 4, 6, 4, 1};

void vizion_pyrdown_horizontal_sse2(const uint8_t* src, uint16_t* horizontal, int width, int height);
void vizion_pyrdown_horizontal_avx2(const uint8_t* src, uint16_t* horizontal, int width, int height);
void vizion_pyrdown_vertical_sse2(const uint16_t* horizontal, int width, int height, uint8_t* dst);

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

uint16_t vizion_pyrdown_hpixel(const uint8_t* src, int width, int row, int column)
{
    int sum = 0;
    int tap;
    for (tap = -2; tap <= 2; tap++)
        sum += vizion_pyr_kernel[tap + 2] * src[(size_t)row * (size_t)width + vizion_reflect(column + tap, width)];
    return (uint16_t)sum;
}

void vizion_pyrdown_horizontal_scalar(const uint8_t* src, uint16_t* horizontal, int width, int height)
{
    int row;
    int col;
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++)
            horizontal[(size_t)row * (size_t)width + col] = vizion_pyrdown_hpixel(src, width, row, col);
    }
}

uint8_t vizion_pyrdown_vpixel(const uint16_t* horizontal, int width, int height, int row, int column)
{
    int source_row = row * 2;
    int source_col = column * 2;
    int sum = 0;
    int tap;
    for (tap = 0; tap < 5; tap++) {
        int rr = vizion_reflect(source_row + tap - 2, height);
        sum += vizion_pyr_kernel[tap] * horizontal[(size_t)rr * (size_t)width + source_col];
    }
    sum = (sum + 128) / 256;
    if (sum < 0)
        sum = 0;
    if (sum > 255)
        sum = 255;
    return (uint8_t)sum;
}

void vizion_pyrdown_vertical_scalar(const uint16_t* horizontal, int width, int height, uint8_t* dst)
{
    int dest_width = (width + 1) / 2;
    int dest_height = (height + 1) / 2;
    int row;
    for (row = 0; row < dest_height; row++) {
        int source_row = row * 2;
        int rows[5];
        int tap;
        int col;
        for (tap = 0; tap < 5; tap++)
            rows[tap] = vizion_reflect(source_row + tap - 2, height);
        for (col = 0; col < dest_width; col++) {
            int source_col = col * 2;
            int sum = 0;
            for (tap = 0; tap < 5; tap++)
                sum += vizion_pyr_kernel[tap] * horizontal[(size_t)rows[tap] * (size_t)width + source_col];
            sum = (sum + 128) / 256;
            if (sum < 0)
                sum = 0;
            if (sum > 255)
                sum = 255;
            dst[(size_t)row * (size_t)dest_width + col] = (uint8_t)sum;
        }
    }
}

int vizion_pyrdown(const uint8_t* src, int width, int height, uint8_t* dst)
{
    uint16_t* horizontal;
    int flags;
    if (!src || !dst || width < 1 || height < 1)
        return -1;
    horizontal = (uint16_t*)malloc((size_t)width * (size_t)height * sizeof(uint16_t));
    if (!horizontal)
        return -1;
    flags = vizion_cpu_flags();
    if (flags & VIZION_CPU_SELECTED_AVX2)
        vizion_pyrdown_horizontal_avx2(src, horizontal, width, height);
    else if (flags & VIZION_CPU_SELECTED_SSE2)
        vizion_pyrdown_horizontal_sse2(src, horizontal, width, height);
    else
        vizion_pyrdown_horizontal_scalar(src, horizontal, width, height);
    if (flags & (VIZION_CPU_SELECTED_AVX2 | VIZION_CPU_SELECTED_SSE2))
        vizion_pyrdown_vertical_sse2(horizontal, width, height, dst);
    else
        vizion_pyrdown_vertical_scalar(horizontal, width, height, dst);
    free(horizontal);
    return 0;
}

int vizion_pyrdown_selftest(void)
{
    enum { kW = 40, kH = 21, kN = 40 * 21, kD = 20 * 11 };
    uint8_t tiny[1] = {200};
    uint8_t down[1];
    uint8_t image[kN];
    uint8_t scalar[kD];
    uint8_t fast[kD];
    uint16_t* horizontal;
    int i;
    int flags = vizion_cpu_flags();
    if (vizion_pyrdown(tiny, 1, 1, down) != 0 || down[0] != 200)
        return 1;
    for (i = 0; i < kN; i++)
        image[i] = (uint8_t)((i * 13 + 5) & 255);
    horizontal = (uint16_t*)malloc(kN * sizeof(uint16_t));
    if (!horizontal)
        return 2;
    vizion_pyrdown_horizontal_scalar(image, horizontal, kW, kH);
    vizion_pyrdown_vertical_scalar(horizontal, kW, kH, scalar);
    if (flags & VIZION_CPU_SSE2) {
        vizion_pyrdown_horizontal_sse2(image, horizontal, kW, kH);
        vizion_pyrdown_vertical_sse2(horizontal, kW, kH, fast);
        if (memcmp(scalar, fast, kD) != 0) {
            free(horizontal);
            return 3;
        }
    }
    if (flags & VIZION_CPU_AVX2) {
        vizion_pyrdown_horizontal_avx2(image, horizontal, kW, kH);
        vizion_pyrdown_vertical_sse2(horizontal, kW, kH, fast);
        if (memcmp(scalar, fast, kD) != 0) {
            free(horizontal);
            return 4;
        }
    }
    free(horizontal);
    if (vizion_pyrdown(image, kW, kH, fast) != 0 || memcmp(scalar, fast, kD) != 0)
        return 5;
    return 0;
}
