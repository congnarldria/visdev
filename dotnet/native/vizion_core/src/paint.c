#include "vizion_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* 查表、填灰、塗灰、排序、Prewitt、Roberts、二項平滑。
   排序邊界 0 是重複邊緣、1 是 REFLECT_101、2 是循環。 */

int vizion_reflect101(int position, int length);

static uint8_t paint_round(double value)
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

static uint8_t paint_round_f(float value)
{
    return paint_round((double)value);
}

int vizion_fill(uint8_t* dst, int count, int value)
{
    int index;
    uint8_t gray;
    if (!dst || count < 1 || value < 0 || value > 255)
        return -1;
    gray = (uint8_t)value;
    for (index = 0; index < count; index++)
        dst[index] = gray;
    return 0;
}

int vizion_lut(const uint8_t* src, uint8_t* dst, int count, const uint8_t* table)
{
    int index;
    if (!src || !dst || !table || count < 1)
        return -1;
    for (index = 0; index < count; index++)
        dst[index] = table[src[index]];
    return 0;
}

int vizion_rgb3_gray(const uint8_t* red, const uint8_t* green, const uint8_t* blue, uint8_t* gray, int count)
{
    int index;
    if (!red || !green || !blue || !gray || count < 1)
        return -1;
    for (index = 0; index < count; index++) {
        double value = 0.299 * (double)red[index] + 0.587 * (double)green[index] + 0.114 * (double)blue[index];
        gray[index] = paint_round(value);
    }
    return 0;
}

int vizion_paint(
    const uint8_t* base, const uint8_t* source, uint8_t* dst,
    int width, int height, int base_channels, int source_channels, const uint8_t* mask)
{
    int count;
    int index;
    if (!base || !source || !dst || width < 1 || height < 1)
        return -1;
    if (!((base_channels == source_channels && (base_channels == 1 || base_channels == 3))
            || (source_channels == 1 && base_channels == 3)))
        return -1;
    count = width * height;
    if (dst != base)
        memcpy(dst, base, (size_t)count * (size_t)base_channels);
    for (index = 0; index < count; index++) {
        if (mask && !mask[index])
            continue;
        if (base_channels == source_channels) {
            int channel;
            for (channel = 0; channel < base_channels; channel++)
                dst[index * base_channels + channel] = source[index * source_channels + channel];
        } else {
            dst[index * 3] = source[index];
            dst[index * 3 + 1] = source[index];
            dst[index * 3 + 2] = source[index];
        }
    }
    return 0;
}

static int paint_index(int position, int length, int border)
{
    int wrapped;
    if (border == 2) {
        wrapped = position % length;
        if (wrapped < 0)
            wrapped += length;
        return wrapped;
    }
    if (border == 1)
        return vizion_reflect101(position, length);
    if (position < 0)
        return 0;
    if (position >= length)
        return length - 1;
    return position;
}

int vizion_rank(
    const uint8_t* src, uint8_t* dst, int width, int height,
    int mask_width, int mask_height, int rank, int border)
{
    int row;
    int col;
    int window;
    int anchor_x;
    int anchor_y;
    uint8_t samples[49];
    if (!src || !dst || width < 1 || height < 1 || border < 0 || border > 2)
        return -1;
    if (mask_width < 1 || mask_height < 1 || mask_width > 7 || mask_height > 7)
        return -1;
    if ((mask_width & 1) == 0 || (mask_height & 1) == 0)
        return -1;
    window = mask_width * mask_height;
    if (rank < 0 || rank >= window)
        return -1;
    anchor_x = mask_width / 2;
    anchor_y = mask_height / 2;
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            int dy;
            int filled = 0;
            for (dy = 0; dy < mask_height; dy++) {
                int yy = paint_index(row + dy - anchor_y, height, border);
                int dx;
                for (dx = 0; dx < mask_width; dx++) {
                    int xx = paint_index(col + dx - anchor_x, width, border);
                    uint8_t value = src[(size_t)yy * (size_t)width + xx];
                    int index = filled;
                    while (index > 0 && samples[index - 1] > value) {
                        samples[index] = samples[index - 1];
                        index--;
                    }
                    samples[index] = value;
                    filled++;
                }
            }
            dst[(size_t)row * (size_t)width + col] = samples[rank];
        }
    }
    return 0;
}

static float paint_conv3(const float* src, int width, int height, int row, int col, const float* kernel)
{
    float sum = 0.f;
    int dy;
    for (dy = 0; dy < 3; dy++) {
        int yy = row + dy - 1;
        int dx;
        if (yy < 0)
            yy = 0;
        else if (yy >= height)
            yy = height - 1;
        for (dx = 0; dx < 3; dx++) {
            int xx = col + dx - 1;
            if (xx < 0)
                xx = 0;
            else if (xx >= width)
                xx = width - 1;
            sum += src[(size_t)yy * (size_t)width + xx] * kernel[dy * 3 + dx];
        }
    }
    return sum;
}

int vizion_prewitt(const uint8_t* src, uint8_t* dst, int width, int height)
{
    static const float horizontal[9] = { -1.f, 0.f, 1.f, -1.f, 0.f, 1.f, -1.f, 0.f, 1.f };
    static const float vertical[9] = { -1.f, -1.f, -1.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };
    float* plane;
    int count;
    int row;
    int col;
    if (!src || !dst || width < 1 || height < 1)
        return -1;
    count = width * height;
    plane = (float*)malloc((size_t)count * sizeof(float));
    if (!plane)
        return -1;
    for (row = 0; row < count; row++)
        plane[row] = (float)src[row];
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            float gx = paint_conv3(plane, width, height, row, col, horizontal);
            float gy = paint_conv3(plane, width, height, row, col, vertical);
            dst[(size_t)row * (size_t)width + col] = paint_round_f(hypotf(gx, gy) / 3.f);
        }
    }
    free(plane);
    return 0;
}

int vizion_roberts(const uint8_t* src, uint8_t* dst, int width, int height, int sum_abs)
{
    int row;
    int col;
    if (!src || !dst || width < 1 || height < 1)
        return -1;
    memset(dst, 0, (size_t)width * (size_t)height);
    for (row = 0; row < height - 1; row++) {
        for (col = 0; col < width - 1; col++) {
            float first = (float)src[(size_t)row * (size_t)width + col] - (float)src[(size_t)(row + 1) * (size_t)width + (col + 1)];
            float second = (float)src[(size_t)row * (size_t)width + (col + 1)] - (float)src[(size_t)(row + 1) * (size_t)width + col];
            float strength = sum_abs ? fabsf(first) + fabsf(second) : hypotf(first, second);
            dst[(size_t)row * (size_t)width + col] = paint_round_f(strength);
        }
    }
    return 0;
}

static void paint_binomial(float* kernel, int size)
{
    int degree = size - 1;
    int index;
    double coeff = 1.0;
    double sum = 0.0;
    double raw[31];
    for (index = 0; index < size; index++) {
        raw[index] = coeff;
        sum += coeff;
        coeff = coeff * (double)(degree - index) / (double)(index + 1);
    }
    for (index = 0; index < size; index++)
        kernel[index] = (float)(raw[index] / sum);
}

static int paint_sep_replicate(
    const float* src, float* dst, int width, int height, const float* across, int across_count, const float* along, int along_count)
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
                int xx = col + tap - anchor_x;
                if (xx < 0)
                    xx = 0;
                else if (xx >= width)
                    xx = width - 1;
                sum += src[(size_t)row * (size_t)width + xx] * across[tap];
            }
            horizontal[(size_t)row * (size_t)width + col] = sum;
        }
    }
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            float sum = 0.f;
            for (tap = 0; tap < along_count; tap++) {
                int yy = row + tap - anchor_y;
                if (yy < 0)
                    yy = 0;
                else if (yy >= height)
                    yy = height - 1;
                sum += horizontal[(size_t)yy * (size_t)width + col] * along[tap];
            }
            dst[(size_t)row * (size_t)width + col] = sum;
        }
    }
    free(horizontal);
    return 0;
}

int vizion_binomial(const uint8_t* src, uint8_t* dst, int width, int height, int mask_width, int mask_height)
{
    float across[31];
    float along[31];
    float* plane;
    float* blurred;
    int count;
    int index;
    int code;
    if (!src || !dst || width < 1 || height < 1)
        return -1;
    if (mask_width < 1 || mask_height < 1 || mask_width > 31 || mask_height > 31)
        return -1;
    if ((mask_width & 1) == 0 || (mask_height & 1) == 0)
        return -1;
    count = width * height;
    plane = (float*)malloc((size_t)count * sizeof(float) * 2);
    if (!plane)
        return -1;
    blurred = plane + count;
    for (index = 0; index < count; index++)
        plane[index] = (float)src[index];
    paint_binomial(across, mask_width);
    paint_binomial(along, mask_height);
    code = paint_sep_replicate(plane, blurred, width, height, across, mask_width, along, mask_height);
    if (code == 0) {
        for (index = 0; index < count; index++)
            dst[index] = paint_round_f(blurred[index]);
    }
    free(plane);
    return code;
}

static int paint_same(const uint8_t* left, const uint8_t* right, int count)
{
    int index;
    for (index = 0; index < count; index++) {
        if (left[index] != right[index])
            return 0;
    }
    return 1;
}

int vizion_paint_selftest(void)
{
    static const uint8_t src[9] = { 1, 9, 3, 8, 2, 7, 4, 6, 5 };
    static const uint8_t rank0[9] = { 1, 1, 2, 1, 1, 2, 2, 2, 2 };
    static const uint8_t rank4[9] = { 2, 3, 3, 4, 5, 5, 4, 5, 5 };
    static const uint8_t mirror[9] = { 2, 7, 3, 6, 5, 6, 4, 6, 5 };
    static const uint8_t prewitt[9] = { 4, 2, 2, 2, 1, 1, 1, 1, 1 };
    static const uint8_t roberts[9] = { 1, 2, 0, 3, 3, 0, 0, 0, 0 };
    static const uint8_t roberts_abs[9] = { 2, 3, 0, 4, 4, 0, 0, 0, 0 };
    static const uint8_t binomial[9] = { 4, 5, 5, 5, 5, 5, 5, 5, 5 };
    uint8_t out[9];
    uint8_t table[256];
    uint8_t red = 10, green = 20, blue = 30, gray = 0;
    uint8_t base[3] = { 1, 2, 3 };
    uint8_t painted[3];
    uint8_t mask[1] = { 0 };
    int index;
    if (vizion_rank(src, out, 3, 3, 3, 3, 0, 0) != 0 || !paint_same(out, rank0, 9))
        return 1;
    if (vizion_rank(src, out, 3, 3, 3, 3, 4, 0) != 0 || !paint_same(out, rank4, 9))
        return 2;
    if (vizion_rank(src, out, 3, 3, 3, 3, 4, 1) != 0 || !paint_same(out, mirror, 9))
        return 3;
    if (vizion_prewitt(src, out, 3, 3) != 0 || !paint_same(out, prewitt, 9))
        return 4;
    if (vizion_roberts(src, out, 3, 3, 0) != 0 || !paint_same(out, roberts, 9))
        return 5;
    if (vizion_roberts(src, out, 3, 3, 1) != 0 || !paint_same(out, roberts_abs, 9))
        return 6;
    if (vizion_binomial(src, out, 3, 3, 3, 3) != 0 || !paint_same(out, binomial, 9))
        return 7;
    if (vizion_rgb3_gray(&red, &green, &blue, &gray, 1) != 0 || gray != 18)
        return 8;
    for (index = 0; index < 256; index++)
        table[index] = (uint8_t)(255 - index);
    if (vizion_lut(src, out, 9, table) != 0 || out[0] != 254 || out[1] != 246)
        return 9;
    if (vizion_fill(out, 9, 7) != 0 || out[4] != 7)
        return 10;
    if (vizion_paint(base, src, painted, 1, 1, 3, 1, mask) != 0 || painted[0] != 1 || painted[2] != 3)
        return 11;
    mask[0] = 255;
    if (vizion_paint(base, src, painted, 1, 1, 3, 1, mask) != 0 || painted[0] != 1 || painted[1] != 1 || painted[2] != 1)
        return 12;
    return 0;
}
