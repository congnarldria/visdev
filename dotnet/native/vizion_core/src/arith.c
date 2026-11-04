#include "vizion_core.h"

#include <stdlib.h>
#include <string.h>

/* 小數直接丟掉，負數先截成 0，避免把負的 float 轉成 uint8。 */

void vizion_arith_sse2(
    const uint8_t* a, const uint8_t* b, uint8_t* dst, int count,
    int op, float mult, float add);

static uint8_t vizion_trunc(float value)
{
    if (value <= 0.f)
        return 0;
    if (value >= 255.f)
        return 255;
    return (uint8_t)value;
}

static uint8_t vizion_arith_one(uint8_t left, uint8_t right, int op, float mult, float add)
{
    float a = (float)left;
    float b = (float)right;
    float value;
    switch (op) {
    case 0:
        return (uint8_t)(255 - left);
    case 1:
        value = a * mult + add;
        break;
    case 2:
        value = (a + b) * mult + add;
        break;
    case 3:
        value = (a - b) * mult + add;
        break;
    case 4:
        value = (a * b) * mult + add;
        break;
    case 5:
        value = (a >= b ? a - b : b - a) * mult + add;
        break;
    case 6:
        return left < right ? left : right;
    case 7:
        return left > right ? left : right;
    case 8:
        value = (b == 0.f ? 0.f : a / b) * mult + add;
        break;
    case 9:
        value = a + (a - b) * mult;
        break;
    case 10:
        value = (a - b) + b * mult + add;
        break;
    default:
        return left;
    }
    return vizion_trunc(value);
}

void vizion_arith_scalar(
    const uint8_t* a, const uint8_t* b, uint8_t* dst, int begin, int count,
    int op, float mult, float add)
{
    int index;
    for (index = begin; index < count; index++) {
        uint8_t right = b ? b[index] : 0;
        dst[index] = vizion_arith_one(a[index], right, op, mult, add);
    }
}

int vizion_gray_arith(
    const uint8_t* a, const uint8_t* b, uint8_t* dst, int count,
    int op, double mult, double add)
{
    int flags;
    if (!a || !dst || count < 1 || op < 0 || op > 10)
        return -1;
    if (op != 0 && op != 1 && !b)
        return -1;
    flags = vizion_cpu_flags();
    if ((flags & VIZION_CPU_SELECTED_AVX2) || (flags & VIZION_CPU_SELECTED_SSE2)) {
        vizion_arith_sse2(a, b, dst, count, op, (float)mult, (float)add);
        return 0;
    }
    vizion_arith_scalar(a, b, dst, 0, count, op, (float)mult, (float)add);
    return 0;
}

static int vizion_clamp_index(int position, int length)
{
    if (position < 0)
        return 0;
    if (position >= length)
        return length - 1;
    return position;
}

static void vizion_insert(uint8_t* values, int count, uint8_t value)
{
    int index = count;
    while (index > 0 && values[index - 1] > value) {
        values[index] = values[index - 1];
        index--;
    }
    values[index] = value;
}

int vizion_median_image(const uint8_t* src, uint8_t* dst, int width, int height, int ksize)
{
    int radius;
    int window;
    int row;
    int column;
    uint8_t* values;
    if (!src || !dst || width < 1 || height < 1 || ksize < 1 || (ksize & 1) == 0)
        return -1;
    if (ksize == 1) {
        memcpy(dst, src, (size_t)width * (size_t)height);
        return 0;
    }
    radius = ksize / 2;
    window = ksize * ksize;
    values = (uint8_t*)malloc((size_t)window);
    if (!values)
        return -1;
    for (row = 0; row < height; row++) {
        for (column = 0; column < width; column++) {
            int dy;
            int filled = 0;
            for (dy = -radius; dy <= radius; dy++) {
                int dx;
                int yy = vizion_clamp_index(row + dy, height);
                const uint8_t* line = src + (size_t)yy * (size_t)width;
                for (dx = -radius; dx <= radius; dx++) {
                    int xx = vizion_clamp_index(column + dx, width);
                    vizion_insert(values, filled, line[xx]);
                    filled++;
                }
            }
            dst[(size_t)row * (size_t)width + (size_t)column] = values[window / 2];
        }
    }
    free(values);
    return 0;
}

int vizion_arith_selftest(void)
{
    uint8_t left[20];
    uint8_t right[20];
    uint8_t scalar[20];
    uint8_t fast[20];
    uint8_t image[9] = {1, 9, 3, 8, 2, 7, 4, 6, 5};
    uint8_t median[9];
    uint8_t expect_median[9] = {2, 3, 3, 4, 5, 5, 4, 5, 5};
    int index;
    int flags = vizion_cpu_flags();
    int op;
    for (index = 0; index < 20; index++) {
        left[index] = (uint8_t)(index * 13 + 7);
        right[index] = (uint8_t)(255 - index * 9);
    }
    if (vizion_gray_arith(left, NULL, scalar, 20, 0, 0.0, 0.0) != 0)
        return 1;
    if (scalar[0] != (uint8_t)(255 - 7) || scalar[3] != (uint8_t)(255 - (3 * 13 + 7)))
        return 2;
    if (vizion_gray_arith(left, NULL, scalar, 5, 1, 1.5, -10.0) != 0)
        return 3;
    if (scalar[0] != vizion_trunc(7.f * 1.5f - 10.f))
        return 4;
    if (vizion_gray_arith(left, right, scalar, 8, 3, 1.0, 128.0) != 0)
        return 5;
    if (scalar[1] != vizion_trunc(((float)left[1] - (float)right[1]) + 128.f))
        return 6;
    if (vizion_median_image(image, median, 3, 3, 3) != 0)
        return 7;
    if (memcmp(median, expect_median, 9) != 0)
        return 8;
    if ((flags & VIZION_CPU_SSE2) == 0)
        return 0;
    for (op = 0; op <= 10; op++) {
        const uint8_t* second = (op == 0 || op == 1) ? NULL : right;
        vizion_arith_scalar(left, second, scalar, 0, 20, op, 0.5f, 3.f);
        memset(fast, 0, sizeof(fast));
        vizion_arith_sse2(left, second, fast, 20, op, 0.5f, 3.f);
        if (memcmp(scalar, fast, 20) != 0)
            return 10 + op;
        vizion_arith_scalar(left, second, scalar, 0, 7, op, 1.25f, -4.f);
        memset(fast, 0, sizeof(fast));
        vizion_arith_sse2(left, second, fast, 7, op, 1.25f, -4.f);
        if (memcmp(scalar, fast, 7) != 0)
            return 20 + op;
    }
    return 0;
}
