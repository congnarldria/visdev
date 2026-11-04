#include "vizion_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* 高通、局部標準差、矩形灰階差、拉普拉斯。模糊和 OpenCV 的 float 盒狀濾波相同。 */

int vizion_box_float(const float* src, float* dst, int width, int height, int mask_width, int mask_height);
int vizion_reflect101(int position, int length);

static uint8_t local_trunc(float value)
{
    if (!(value > 0.f))
        return 0;
    if (value >= 255.f)
        return 255;
    return (uint8_t)value;
}

static uint8_t local_round(float value)
{
    int whole;
    float frac;
    int rounded;
    if (!(value > 0.f))
        return 0;
    if (value >= 255.f)
        return 255;
    whole = (int)value;
    frac = value - (float)whole;
    if (frac > 0.5f)
        rounded = whole + 1;
    else if (frac < 0.5f)
        rounded = whole;
    else
        rounded = (whole & 1) ? whole + 1 : whole;
    if (rounded > 255)
        return 255;
    return (uint8_t)rounded;
}

static int local_blur_bytes(const uint8_t* src, float* dst, int width, int height, int mask_width, int mask_height)
{
    float* plane;
    int count = width * height;
    int index;
    int code;
    plane = (float*)malloc((size_t)count * sizeof(float));
    if (!plane)
        return -1;
    for (index = 0; index < count; index++)
        plane[index] = (float)src[index];
    code = vizion_box_float(plane, dst, width, height, mask_width, mask_height);
    free(plane);
    return code;
}

int vizion_highpass(const uint8_t* src, uint8_t* dst, int width, int height, int mask_width, int mask_height)
{
    float* mean;
    int count;
    int index;
    int code;
    if (!src || !dst || width < 1 || height < 1 || mask_width < 1 || mask_height < 1)
        return -1;
    count = width * height;
    mean = (float*)malloc((size_t)count * sizeof(float));
    if (!mean)
        return -1;
    code = local_blur_bytes(src, mean, width, height, mask_width, mask_height);
    if (code != 0) {
        free(mean);
        return code;
    }
    for (index = 0; index < count; index++)
        dst[index] = local_round(128.f + (float)src[index] - mean[index]);
    free(mean);
    return 0;
}

int vizion_deviation(const uint8_t* src, uint8_t* dst, int width, int height, int mask_width, int mask_height)
{
    float* mean;
    float* square;
    float* plane;
    int count;
    int index;
    int code;
    if (!src || !dst || width < 1 || height < 1 || mask_width < 1 || mask_height < 1)
        return -1;
    count = width * height;
    mean = (float*)malloc((size_t)count * sizeof(float) * 3);
    if (!mean)
        return -1;
    square = mean + count;
    plane = square + count;
    for (index = 0; index < count; index++) {
        float value = (float)src[index];
        plane[index] = value * value;
    }
    code = local_blur_bytes(src, mean, width, height, mask_width, mask_height);
    if (code == 0)
        code = vizion_box_float(plane, square, width, height, mask_width, mask_height);
    if (code != 0) {
        free(mean);
        return code;
    }
    for (index = 0; index < count; index++) {
        float variance = square[index] - mean[index] * mean[index];
        if (variance < 0.f)
            variance = 0.f;
        dst[index] = local_round(sqrtf(variance));
    }
    free(mean);
    return 0;
}

int vizion_gray_span(const uint8_t* src, uint8_t* dst, int width, int height, int mask_height, int mask_width)
{
    uint8_t* bright;
    uint8_t* dark;
    int count;
    int index;
    if (!src || !dst || width < 1 || height < 1 || mask_height < 1 || mask_width < 1)
        return -1;
    count = width * height;
    bright = (uint8_t*)malloc((size_t)count * 2);
    if (!bright)
        return -1;
    dark = bright + count;
    if (vizion_gray_rect(src, bright, width, height, mask_height, mask_width, 1) != 0
        || vizion_gray_rect(src, dark, width, height, mask_height, mask_width, 0) != 0) {
        free(bright);
        return -1;
    }
    for (index = 0; index < count; index++)
        dst[index] = (uint8_t)(bright[index] - dark[index]);
    free(bright);
    return 0;
}

static void local_conv3(const float* src, float* dst, int width, int height, int eight)
{
    static const float cross[9] = { 0, 1, 0, 1, -4, 1, 0, 1, 0 };
    static const float box[9] = { 1, 1, 1, 1, -8, 1, 1, 1, 1 };
    const float* kernel = eight ? box : cross;
    int row;
    int col;
    int dy;
    int dx;
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            float sum = 0.f;
            for (dy = 0; dy < 3; dy++) {
                int yy = vizion_reflect101(row + dy - 1, height);
                for (dx = 0; dx < 3; dx++) {
                    int xx = vizion_reflect101(col + dx - 1, width);
                    sum += src[(size_t)yy * (size_t)width + xx] * kernel[dy * 3 + dx];
                }
            }
            dst[(size_t)row * (size_t)width + col] = sum;
        }
    }
}

static int local_sep(const float* src, float* dst, int width, int height, const float* across, int across_count, const float* along, int along_count)
{
    float* horizontal;
    int row;
    int col;
    int tap;
    int anchor_x = across_count / 2;
    int anchor_y = along_count / 2;
    horizontal = (float*)malloc((size_t)width * (size_t)height * sizeof(float));
    if (!horizontal)
        return -1;
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            float sum = 0.f;
            for (tap = 0; tap < across_count; tap++) {
                int xx = vizion_reflect101(col + tap - anchor_x, width);
                sum += src[(size_t)row * (size_t)width + xx] * across[tap];
            }
            horizontal[(size_t)row * (size_t)width + col] = sum;
        }
    }
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            float sum = 0.f;
            for (tap = 0; tap < along_count; tap++) {
                int yy = vizion_reflect101(row + tap - anchor_y, height);
                sum += horizontal[(size_t)yy * (size_t)width + col] * along[tap];
            }
            dst[(size_t)row * (size_t)width + col] = sum;
        }
    }
    free(horizontal);
    return 0;
}

int vizion_laplace(const uint8_t* src, uint8_t* dst, int width, int height, int ksize, int eight, int absolute)
{
    float* plane;
    float* response;
    float* extra;
    int count;
    int index;
    int code = 0;
    static const float second5[5] = { 1.f, 0.f, -2.f, 0.f, 1.f };
    static const float smooth5[5] = { 1.f, 4.f, 6.f, 4.f, 1.f };
    static const float second7[7] = { 1.f, 2.f, -1.f, -4.f, -1.f, 2.f, 1.f };
    static const float smooth7[7] = { 1.f, 6.f, 15.f, 20.f, 15.f, 6.f, 1.f };
    if (!src || !dst || width < 1 || height < 1 || (ksize != 3 && ksize != 5 && ksize != 7))
        return -1;
    count = width * height;
    plane = (float*)malloc((size_t)count * sizeof(float) * 3);
    if (!plane)
        return -1;
    response = plane + count;
    extra = response + count;
    for (index = 0; index < count; index++)
        plane[index] = (float)src[index];
    if (ksize == 3)
        local_conv3(plane, response, width, height, eight);
    else {
        const float* second = ksize == 5 ? second5 : second7;
        const float* smooth = ksize == 5 ? smooth5 : smooth7;
        code = local_sep(plane, response, width, height, second, ksize, smooth, ksize);
        if (code == 0)
            code = local_sep(plane, extra, width, height, smooth, ksize, second, ksize);
        if (code == 0) {
            for (index = 0; index < count; index++)
                response[index] += extra[index];
        }
    }
    if (code == 0) {
        for (index = 0; index < count; index++) {
            float value = absolute ? fabsf(response[index]) : response[index] + 128.f;
            dst[index] = local_trunc(value);
        }
    }
    free(plane);
    return code;
}

static int local_same(const uint8_t* left, const uint8_t* right, int count)
{
    int index;
    for (index = 0; index < count; index++) {
        if (left[index] != right[index])
            return 0;
    }
    return 1;
}

int vizion_local_selftest(void)
{
    static const uint8_t src[12] = { 1, 9, 3, 8, 2, 7, 4, 6, 5, 0, 12, 20 };
    static const uint8_t abs4[12] = { 18, 18, 13, 14, 12, 13, 12, 12, 16, 31, 20, 44 };
    static const uint8_t signed4[12] = { 146, 110, 141, 114, 140, 115, 140, 140, 112, 159, 108, 84 };
    static const uint8_t abs8[12] = { 42, 42, 27, 30, 22, 20, 33, 18, 8, 43, 42, 108 };
    static const uint8_t lap5[12] = { 24, 28, 116, 200, 104, 68, 52, 136, 184, 108, 220, 255 };
    static const uint8_t high[12] = { 123, 133, 125, 131, 126, 130, 124, 126, 129, 123, 133, 140 };
    static const uint8_t dev[12] = { 3, 3, 2, 2, 4, 4, 5, 5, 3, 3, 5, 5 };
    static const uint8_t span[12] = { 8, 8, 6, 5, 9, 12, 20, 17, 7, 12, 20, 16 };
    uint8_t out[12];
    if (vizion_laplace(src, out, 4, 3, 3, 0, 1) != 0 || !local_same(out, abs4, 12))
        return 1;
    if (vizion_laplace(src, out, 4, 3, 3, 0, 0) != 0 || !local_same(out, signed4, 12))
        return 2;
    if (vizion_laplace(src, out, 4, 3, 3, 1, 1) != 0 || !local_same(out, abs8, 12))
        return 3;
    if (vizion_laplace(src, out, 4, 3, 5, 0, 1) != 0 || !local_same(out, lap5, 12))
        return 4;
    if (vizion_highpass(src, out, 4, 3, 3, 3) != 0 || !local_same(out, high, 12))
        return 5;
    if (vizion_deviation(src, out, 4, 3, 3, 3) != 0 || !local_same(out, dev, 12))
        return 6;
    if (vizion_gray_span(src, out, 4, 3, 3, 3) != 0 || !local_same(out, span, 12))
        return 7;
    return 0;
}
