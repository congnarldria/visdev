#include "vizion_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* 斜坡、拼接、卷積、帶通、加邊、金字塔和局部熵。卷積是 OpenCV 的相關。 */

int vizion_box_float(const float* src, float* dst, int width, int height, int mask_width, int mask_height);
int vizion_reflect101(int position, int length);

static uint8_t stage_round(double value)
{
    int whole;
    double frac;
    int rounded;
    if (!(value > 0.0))
        return 0;
    if (value >= 255.0)
        return 255;
    whole = (int)value;
    frac = value - (double)whole;
    if (frac > 0.5)
        rounded = whole + 1;
    else if (frac < 0.5)
        rounded = whole;
    else
        rounded = (whole & 1) ? whole + 1 : whole;
    if (rounded > 255)
        return 255;
    return (uint8_t)rounded;
}

static uint8_t stage_round_f(float value)
{
    return stage_round((double)value);
}

int vizion_gray_ramp(uint8_t* dst, int width, int height, double alpha, double beta, double gamma, double row, double column)
{
    int y;
    int x;
    float a;
    float b;
    float c;
    float origin_row;
    float origin_column;
    if (!dst || width < 1 || height < 1)
        return -1;
    a = (float)alpha;
    b = (float)beta;
    c = (float)gamma;
    origin_row = (float)row;
    origin_column = (float)column;
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            float value = a * ((float)y - origin_row) + b * ((float)x - origin_column) + c;
            dst[(size_t)y * (size_t)width + x] = stage_round_f(value);
        }
    }
    return 0;
}

int vizion_tile(
    const uint8_t* left, int left_width, int left_height,
    const uint8_t* right, int right_width, int right_height,
    uint8_t* dst, int channels, int vertical)
{
    int width;
    int height;
    int row;
    int col;
    int channel;
    if (!left || !right || !dst || left_width < 1 || left_height < 1 || right_width < 1 || right_height < 1)
        return -1;
    if (channels != 1 && channels != 3)
        return -1;
    width = vertical ? (left_width > right_width ? left_width : right_width) : left_width + right_width;
    height = vertical ? left_height + right_height : (left_height > right_height ? left_height : right_height);
    memset(dst, 0, (size_t)width * (size_t)height * (size_t)channels);
    for (row = 0; row < left_height; row++) {
        for (col = 0; col < left_width; col++) {
            for (channel = 0; channel < channels; channel++)
                dst[((size_t)row * (size_t)width + col) * channels + channel] =
                    left[((size_t)row * (size_t)left_width + col) * channels + channel];
        }
    }
    for (row = 0; row < right_height; row++) {
        int out_row = vertical ? row + left_height : row;
        int out_col = vertical ? 0 : left_width;
        for (col = 0; col < right_width; col++) {
            for (channel = 0; channel < channels; channel++)
                dst[((size_t)out_row * (size_t)width + (out_col + col)) * channels + channel] =
                    right[((size_t)row * (size_t)right_width + col) * channels + channel];
        }
    }
    return 0;
}

static int stage_border_index(int position, int length, int border)
{
    int wrapped;
    if (border == 1) {
        wrapped = position % length;
        if (wrapped < 0)
            wrapped += length;
        return wrapped;
    }
    if (border == 2)
        return vizion_reflect101(position, length);
    if (position < 0)
        return 0;
    if (position >= length)
        return length - 1;
    return position;
}

int vizion_convol(
    const uint8_t* src, uint8_t* dst, int width, int height,
    const float* kernel, int kernel_height, int kernel_width, int border)
{
    float* plane;
    int count;
    int row;
    int col;
    int anchor_y;
    int anchor_x;
    double sum;
    int tap;
    if (!src || !dst || !kernel || width < 1 || height < 1)
        return -1;
    if (kernel_height < 1 || kernel_width < 1 || kernel_height > 15 || kernel_width > 15 || border < 0 || border > 2)
        return -1;
    count = width * height;
    plane = (float*)malloc((size_t)count * sizeof(float));
    if (!plane)
        return -1;
    for (tap = 0; tap < count; tap++)
        plane[tap] = (float)src[tap];
    sum = 0.0;
    for (tap = 0; tap < kernel_height * kernel_width; tap++)
        sum += (double)kernel[tap];
    anchor_y = kernel_height / 2;
    anchor_x = kernel_width / 2;
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            float response = 0.f;
            int dy;
            for (dy = 0; dy < kernel_height; dy++) {
                int yy = stage_border_index(row + dy - anchor_y, height, border);
                int dx;
                for (dx = 0; dx < kernel_width; dx++) {
                    int xx = stage_border_index(col + dx - anchor_x, width, border);
                    response += plane[(size_t)yy * (size_t)width + xx] * kernel[dy * kernel_width + dx];
                }
            }
            if (sum > -1e-6 && sum < 1e-6)
                response += 128.f;
            dst[(size_t)row * (size_t)width + col] = stage_round_f(response);
        }
    }
    free(plane);
    return 0;
}

int vizion_bandpass(const uint8_t* src, uint8_t* dst, int width, int height, int inner, int outer)
{
    float* plane;
    float* narrow;
    float* wide;
    int count;
    int index;
    int code;
    if (!src || !dst || width < 1 || height < 1 || inner < 1 || outer < 1 || inner >= outer)
        return -1;
    if ((inner & 1) == 0 || (outer & 1) == 0)
        return -1;
    count = width * height;
    plane = (float*)malloc((size_t)count * sizeof(float) * 3);
    if (!plane)
        return -1;
    narrow = plane + count;
    wide = narrow + count;
    for (index = 0; index < count; index++)
        plane[index] = (float)src[index];
    code = vizion_box_float(plane, narrow, width, height, inner, inner);
    if (code == 0)
        code = vizion_box_float(plane, wide, width, height, outer, outer);
    if (code == 0) {
        for (index = 0; index < count; index++)
            dst[index] = stage_round_f(128.f + narrow[index] - wide[index]);
    }
    free(plane);
    return code;
}

int vizion_make_border(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels, int border, int fill)
{
    int out_width;
    int out_height;
    int row;
    int col;
    int channel;
    if (!src || !dst || width < 1 || height < 1 || border < 0 || fill < 0 || fill > 255)
        return -1;
    if (channels != 1 && channels != 3)
        return -1;
    out_width = width + border * 2;
    out_height = height + border * 2;
    for (row = 0; row < out_height; row++) {
        for (col = 0; col < out_width; col++) {
            int inside = row >= border && col >= border && row < border + height && col < border + width;
            for (channel = 0; channel < channels; channel++) {
                uint8_t value = (uint8_t)fill;
                if (inside)
                    value = src[(((size_t)(row - border) * (size_t)width) + (size_t)(col - border)) * channels + channel];
                dst[(((size_t)row * (size_t)out_width) + (size_t)col) * channels + channel] = value;
            }
        }
    }
    return 0;
}

int vizion_pyramid(const uint8_t* src, uint8_t* dst, int width, int height, int channels, int mode)
{
    int out_width;
    int out_height;
    int channel;
    int row;
    int col;
    int plane;
    if (!src || !dst || width < 2 || height < 2 || mode < 0 || mode > 3)
        return -1;
    if (channels != 1 && channels != 3)
        return -1;
    out_width = mode >= 2 ? width / 2 : (width + 1) / 2;
    out_height = mode >= 2 ? height / 2 : (height + 1) / 2;
    if (mode == 0) {
        uint16_t* horizontal = (uint16_t*)malloc((size_t)width * (size_t)height * sizeof(uint16_t));
        uint8_t* source_plane = (uint8_t*)malloc((size_t)width * (size_t)height);
        uint8_t* dest_plane = (uint8_t*)malloc((size_t)out_width * (size_t)out_height);
        static const int kernel[5] = { 1, 4, 6, 4, 1 };
        if (!horizontal || !source_plane || !dest_plane) {
            free(horizontal);
            free(source_plane);
            free(dest_plane);
            return -1;
        }
        for (channel = 0; channel < channels; channel++) {
            int y;
            int x;
            for (plane = 0; plane < width * height; plane++)
                source_plane[plane] = src[(size_t)plane * channels + channel];
            for (y = 0; y < height; y++) {
                for (x = 0; x < width; x++) {
                    int sum = 0;
                    int tap;
                    for (tap = -2; tap <= 2; tap++)
                        sum += kernel[tap + 2] * source_plane[(size_t)y * (size_t)width + vizion_reflect101(x + tap, width)];
                    horizontal[(size_t)y * (size_t)width + x] = (uint16_t)sum;
                }
            }
            for (y = 0; y < out_height; y++) {
                for (x = 0; x < out_width; x++) {
                    int sum = 0;
                    int tap;
                    int source_col = x * 2;
                    for (tap = -2; tap <= 2; tap++) {
                        int yy = vizion_reflect101(y * 2 + tap, height);
                        sum += kernel[tap + 2] * horizontal[(size_t)yy * (size_t)width + source_col];
                    }
                    sum = (sum + 128) / 256;
                    if (sum < 0)
                        sum = 0;
                    if (sum > 255)
                        sum = 255;
                    dest_plane[(size_t)y * (size_t)out_width + x] = (uint8_t)sum;
                }
            }
            for (plane = 0; plane < out_width * out_height; plane++)
                dst[(size_t)plane * channels + channel] = dest_plane[plane];
        }
        free(horizontal);
        free(source_plane);
        free(dest_plane);
        return 0;
    }
    for (row = 0; row < out_height; row++) {
        for (col = 0; col < out_width; col++) {
            for (channel = 0; channel < channels; channel++) {
                uint8_t value;
                if (mode == 1)
                    value = src[((size_t)(row * 2) * (size_t)width + (size_t)(col * 2)) * channels + channel];
                else {
                    int r0 = row * 2;
                    int c0 = col * 2;
                    value = src[((size_t)r0 * (size_t)width + c0) * channels + channel];
                    {
                        uint8_t samples[4];
                        int index;
                        samples[0] = value;
                        samples[1] = src[((size_t)r0 * (size_t)width + (c0 + 1)) * channels + channel];
                        samples[2] = src[((size_t)(r0 + 1) * (size_t)width + c0) * channels + channel];
                        samples[3] = src[((size_t)(r0 + 1) * (size_t)width + (c0 + 1)) * channels + channel];
                        value = samples[0];
                        for (index = 1; index < 4; index++) {
                            if (mode == 2 && samples[index] < value)
                                value = samples[index];
                            if (mode == 3 && samples[index] > value)
                                value = samples[index];
                        }
                    }
                }
                dst[((size_t)row * (size_t)out_width + col) * channels + channel] = value;
            }
        }
    }
    return 0;
}

int vizion_entropy(const uint8_t* src, uint8_t* dst, int width, int height, int mask_width, int mask_height)
{
    int anchor_x;
    int anchor_y;
    int area;
    int row;
    int col;
    if (!src || !dst || width < 1 || height < 1 || mask_width < 1 || mask_height < 1)
        return -1;
    if ((mask_width & 1) == 0 || (mask_height & 1) == 0)
        return -1;
    anchor_x = mask_width / 2;
    anchor_y = mask_height / 2;
    area = mask_width * mask_height;
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            int hist[256];
            int dy;
            double entropy = 0.0;
            int value;
            memset(hist, 0, sizeof(hist));
            for (dy = 0; dy < mask_height; dy++) {
                int yy = vizion_reflect101(row + dy - anchor_y, height);
                int dx;
                for (dx = 0; dx < mask_width; dx++) {
                    int xx = vizion_reflect101(col + dx - anchor_x, width);
                    hist[src[(size_t)yy * (size_t)width + xx]]++;
                }
            }
            for (value = 0; value < 256; value++) {
                double probability;
                if (!hist[value])
                    continue;
                probability = (double)hist[value] / (double)area;
                entropy -= probability * (log(probability) / log(2.0));
            }
            dst[(size_t)row * (size_t)width + col] = stage_round(entropy / 8.0 * 255.0);
        }
    }
    return 0;
}

static int stage_same(const uint8_t* left, const uint8_t* right, int count)
{
    int index;
    for (index = 0; index < count; index++) {
        if (left[index] != right[index])
            return 0;
    }
    return 1;
}

int vizion_stage_selftest(void)
{
    static const uint8_t src[9] = { 1, 9, 3, 8, 2, 7, 4, 6, 5 };
    static const uint8_t ramp_expect[12] = { 12, 11, 10, 8, 13, 12, 10, 8, 14, 12, 10, 9 };
    static const float sobel[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    static const uint8_t sobel_expect[9] = { 146, 133, 115, 126, 129, 131, 128, 130, 130 };
    static const float smooth[9] = {
        1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f
    };
    static const uint8_t smooth_expect[9] = { 4, 5, 5, 5, 5, 5, 5, 5, 5 };
    static const uint8_t pepper[9] = { 255, 255, 255, 0, 0, 0, 0, 0, 0 };
    static const uint8_t entropy_expect[9] = { 29, 29, 29, 29, 29, 29, 0, 0, 0 };
    uint8_t out[12];
    uint8_t tiled[6];
    uint8_t framed[8];
    uint8_t half[4];
    if (vizion_gray_ramp(out, 4, 3, 0.5, -1.5, 10.0, 1.0, 2.0) != 0 || !stage_same(out, ramp_expect, 12))
        return 1;
    if (vizion_convol(src, out, 3, 3, sobel, 3, 3, 0) != 0 || !stage_same(out, sobel_expect, 9))
        return 2;
    if (vizion_convol(src, out, 3, 3, smooth, 3, 3, 0) != 0 || !stage_same(out, smooth_expect, 9))
        return 3;
    if (vizion_entropy(pepper, out, 3, 3, 3, 3) != 0 || !stage_same(out, entropy_expect, 9))
        return 4;
    if (vizion_tile(src, 3, 1, src + 3, 2, 1, tiled, 1, 0) != 0 || tiled[0] != 1 || tiled[3] != 8 || tiled[4] != 2)
        return 5;
    if (vizion_make_border(src, framed, 2, 1, 1, 1, 9) != 0 || framed[0] != 9 || framed[1 * 4 + 1] != 1 || framed[1 * 4 + 2] != 9)
        return 6;
    if (vizion_pyramid(src, half, 3, 3, 1, 1) != 0 || half[0] != 1 || half[1] != 3)
        return 7;
    if (vizion_pyramid(src, half, 3, 3, 1, 2) != 0 || half[0] != 1)
        return 8;
    return 0;
}
