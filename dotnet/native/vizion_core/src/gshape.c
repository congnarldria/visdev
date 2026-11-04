#include "vizion_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* 二值核取最大或最小，圖外重複邊緣。開運算先侵蝕再膨脹，閉運算相反。 */

void vizion_plane_minmax_sse2(
    const uint8_t* pad, int pad_width, uint8_t* dst, int width, int height,
    const uint8_t* kernel, int kernel_height, int kernel_width, int dilate);

static int shape_hits(const uint8_t* kernel, int count)
{
    int hits = 0;
    int i;
    for (i = 0; i < count; i++) {
        if (kernel[i])
            hits++;
    }
    return hits;
}

static uint8_t* shape_pad(const uint8_t* src, int width, int height, int kernel_height, int kernel_width, int zero_border, int* pad_width)
{
    int anchor_y = kernel_height / 2;
    int anchor_x = kernel_width / 2;
    int pad_h;
    int pad_w;
    size_t bytes;
    uint8_t* pad;
    int row;
    int col;
    if (kernel_height > 200000000 || kernel_width > 200000000 || width > 200000000 || height > 200000000)
        return NULL;
    pad_h = height + kernel_height - 1;
    pad_w = width + kernel_width - 1;
    bytes = (size_t)pad_w * (size_t)pad_h;
    if (pad_h != 0 && bytes / (size_t)pad_h != (size_t)pad_w)
        return NULL;
    pad = (uint8_t*)malloc(bytes);
    if (!pad)
        return NULL;
    *pad_width = pad_w;
    if (zero_border) {
        memset(pad, 0, bytes);
        for (row = 0; row < height; row++)
            memcpy(pad + ((size_t)row + (size_t)anchor_y) * (size_t)pad_w + anchor_x, src + (size_t)row * (size_t)width, (size_t)width);
        return pad;
    }
    for (row = 0; row < pad_h; row++) {
        int src_row = row - anchor_y;
        if (src_row < 0)
            src_row = 0;
        if (src_row >= height)
            src_row = height - 1;
        for (col = 0; col < pad_w; col++) {
            int src_col = col - anchor_x;
            if (src_col < 0)
                src_col = 0;
            if (src_col >= width)
                src_col = width - 1;
            pad[(size_t)row * (size_t)pad_w + col] = src[(size_t)src_row * (size_t)width + src_col];
        }
    }
    return pad;
}

static void shape_minmax_scalar(
    const uint8_t* pad, int pad_width, uint8_t* dst, int width, int height,
    const uint8_t* kernel, int kernel_height, int kernel_width, int dilate)
{
    int ky;
    int kx;
    int row;
    int col;
    memset(dst, dilate ? 0 : 255, (size_t)width * (size_t)height);
    for (ky = 0; ky < kernel_height; ky++) {
        for (kx = 0; kx < kernel_width; kx++) {
            if (!kernel[ky * kernel_width + kx])
                continue;
            for (row = 0; row < height; row++) {
                const uint8_t* line = pad + ((size_t)row + (size_t)ky) * (size_t)pad_width + kx;
                uint8_t* out = dst + (size_t)row * (size_t)width;
                for (col = 0; col < width; col++) {
                    uint8_t sample = line[col];
                    if (dilate) {
                        if (sample > out[col])
                            out[col] = sample;
                    } else if (sample < out[col]) {
                        out[col] = sample;
                    }
                }
            }
        }
    }
}

static int shape_plane_minmax(
    const uint8_t* src, uint8_t* dst, int width, int height,
    const uint8_t* kernel, int kernel_height, int kernel_width, int dilate, int use_sse)
{
    int pad_width = 0;
    uint8_t* pad = shape_pad(src, width, height, kernel_height, kernel_width, 0, &pad_width);
    if (!pad)
        return -1;
    if (use_sse)
        vizion_plane_minmax_sse2(pad, pad_width, dst, width, height, kernel, kernel_height, kernel_width, dilate);
    else
        shape_minmax_scalar(pad, pad_width, dst, width, height, kernel, kernel_height, kernel_width, dilate);
    free(pad);
    return 0;
}

static uint8_t shape_round(float value)
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
    rounded = frac > 0.5f ? whole + 1 : frac < 0.5f ? whole : ((whole & 1) ? whole + 1 : whole);
    if (rounded > 255)
        return 255;
    return (uint8_t)rounded;
}

static int shape_prefix(int flat, int lanes)
{
    int index = 0;
    int wide = 4 * lanes;
    int mid = 2 * lanes;
    if (lanes < 1)
        return 0;
    while (index <= flat - wide)
        index += wide;
    if (index <= flat - mid)
        index += mid;
    if (index <= flat - lanes)
        index += lanes;
    return index;
}

/* 和 OpenCV 的 float filter2D 對齊：有 AVX2 時每一列前面用融合乘加，剩下的像素分開乘加。 */

static void shape_mean_plane(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels,
    const uint8_t* kernel, int kernel_height, int kernel_width, float weight)
{
    int anchor_y = kernel_height / 2;
    int anchor_x = kernel_width / 2;
    int lanes = (vizion_cpu_flags() & (VIZION_CPU_AVX2 | VIZION_CPU_SELECTED_AVX2)) ? 8 : 0;
    int prefix = shape_prefix(width * channels, lanes);
    int row;
    int col;
    int channel;
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            for (channel = 0; channel < channels; channel++) {
                int use_fma = col * channels + channel < prefix;
                float acc = 0.f;
                int ky;
                for (ky = 0; ky < kernel_height; ky++) {
                    int kx;
                    for (kx = 0; kx < kernel_width; kx++) {
                        int sample_row;
                        int sample_col;
                        float sample;
                        if (!kernel[ky * kernel_width + kx])
                            continue;
                        sample_row = row + ky - anchor_y;
                        sample_col = col + kx - anchor_x;
                        sample = 0.f;
                        if ((unsigned)sample_row < (unsigned)height && (unsigned)sample_col < (unsigned)width)
                            sample = (float)src[((size_t)sample_row * (size_t)width + sample_col) * (size_t)channels + channel];
                        if (use_fma)
                            acc = fmaf(sample, weight, acc);
                        else
                            acc += sample * weight;
                    }
                }
                dst[((size_t)row * (size_t)width + col) * (size_t)channels + channel] = shape_round(acc);
            }
        }
    }
}

static int shape_fast(void)
{
    int flags = vizion_cpu_flags();
    return (flags & VIZION_CPU_SELECTED_AVX2) || (flags & VIZION_CPU_SELECTED_SSE2);
}

static void shape_take(const uint8_t* src, uint8_t* plane, int count, int channels, int channel)
{
    int i;
    for (i = 0; i < count; i++)
        plane[i] = src[(size_t)i * (size_t)channels + channel];
}

static void shape_put(const uint8_t* plane, uint8_t* dst, int count, int channels, int channel)
{
    int i;
    for (i = 0; i < count; i++)
        dst[(size_t)i * (size_t)channels + channel] = plane[i];
}

int vizion_gray_kernel(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels,
    const uint8_t* kernel, int kernel_height, int kernel_width, int mode)
{
    int count;
    int fast;
    int channel;
    uint8_t* first;
    uint8_t* second;
    uint8_t* third;
    if (!src || !dst || !kernel || width < 1 || height < 1 || (channels != 1 && channels != 3))
        return -1;
    if (kernel_height < 1 || kernel_width < 1 || mode < 0 || mode > 3)
        return -1;
    if (shape_hits(kernel, kernel_height * kernel_width) == 0)
        return -1;
    count = width * height;
    first = (uint8_t*)malloc((size_t)count * 3);
    if (!first)
        return -1;
    second = first + count;
    third = second + count;
    fast = shape_fast();
    for (channel = 0; channel < channels; channel++) {
        int code = 0;
        shape_take(src, first, count, channels, channel);
        if (mode == 2) {
            code = shape_plane_minmax(first, second, width, height, kernel, kernel_height, kernel_width, 0, fast);
            if (code == 0)
                code = shape_plane_minmax(second, third, width, height, kernel, kernel_height, kernel_width, 1, fast);
        } else if (mode == 3) {
            code = shape_plane_minmax(first, second, width, height, kernel, kernel_height, kernel_width, 1, fast);
            if (code == 0)
                code = shape_plane_minmax(second, third, width, height, kernel, kernel_height, kernel_width, 0, fast);
        } else {
            code = shape_plane_minmax(first, third, width, height, kernel, kernel_height, kernel_width, mode == 0, fast);
        }
        if (code != 0) {
            free(first);
            return code;
        }
        shape_put(third, dst, count, channels, channel);
    }
    free(first);
    return 0;
}

/* 前景權重相等，圖外補 0。列首融合乘加，列尾分開乘加，再四捨五入到偶數。 */

int vizion_mean_kernel(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels,
    const uint8_t* kernel, int kernel_height, int kernel_width)
{
    int hits;
    if (!src || !dst || !kernel || width < 1 || height < 1 || (channels != 1 && channels != 3))
        return -1;
    if (kernel_height < 1 || kernel_width < 1)
        return -1;
    hits = shape_hits(kernel, kernel_height * kernel_width);
    if (hits == 0)
        return -1;
    shape_mean_plane(src, dst, width, height, channels, kernel, kernel_height, kernel_width, 1.f / (float)hits);
    return 0;
}

/* 索引 1 是紅、2 是綠、3 是藍。灰階只留索引 1，其餘變 0。 */

int vizion_select_gray(
    const uint8_t* src, const uint8_t* index, uint8_t* dst, int width, int height, int channels)
{
    int count;
    int i;
    if (!src || !index || !dst || width < 1 || height < 1 || (channels != 1 && channels != 3))
        return -1;
    count = width * height;
    if (channels == 1) {
        for (i = 0; i < count; i++)
            dst[i] = index[i] == 1 ? src[i] : 0;
        return 0;
    }
    for (i = 0; i < count; i++) {
        uint8_t choice = index[i];
        const uint8_t* pixel = src + (size_t)i * 3;
        if (choice == 1)
            dst[i] = pixel[2];
        else if (choice == 2)
            dst[i] = pixel[1];
        else if (choice == 3)
            dst[i] = pixel[0];
        else
            dst[i] = 0;
    }
    return 0;
}

static int shape_bankers(double value)
{
    double whole = floor(value);
    double frac = value - whole;
    int base = (int)whole;
    if (frac > 0.5)
        return base + 1;
    if (frac < 0.5)
        return base;
    return (base & 1) ? base + 1 : base;
}

/* 矩形、菱形，或和 OpenCV 橢圓結構元素相同的八角形。高寬要是正奇數。 */

int vizion_shape_kernel(uint8_t* kernel, int height, int width, int shape)
{
    int center_row;
    int center_col;
    int row;
    int radius;
    int half_width;
    double inverse;
    if (!kernel || height < 1 || width < 1 || (height & 1) == 0 || (width & 1) == 0 || shape < 0 || shape > 2)
        return -1;
    memset(kernel, 0, (size_t)height * (size_t)width);
    if (shape == 0) {
        memset(kernel, 1, (size_t)height * (size_t)width);
        return 0;
    }
    if (shape == 1) {
        center_row = height / 2;
        center_col = width / 2;
        for (row = 0; row < height; row++) {
            int dist = row - center_row;
            int span;
            int begin;
            int end;
            int col;
            double denom = center_row == 0 ? 1.0 : (double)center_row;
            if (dist < 0)
                dist = -dist;
            span = shape_bankers((1.0 - (double)dist / denom) * (double)center_col);
            begin = center_col - span;
            end = center_col + span + 1;
            if (begin < 0)
                begin = 0;
            if (end > width)
                end = width;
            for (col = begin; col < end; col++)
                kernel[row * width + col] = 1;
        }
        return 0;
    }
    radius = height / 2;
    half_width = width / 2;
    inverse = radius == 0 ? 0.0 : 1.0 / ((double)radius * (double)radius);
    for (row = 0; row < height; row++) {
        int dy = row - radius;
        int ady = dy < 0 ? -dy : dy;
        int begin = 0;
        int end = 0;
        int col;
        if (ady <= radius) {
            double inside = ((double)radius * (double)radius - (double)dy * (double)dy) * inverse;
            int reach;
            if (inside < 0.0)
                inside = 0.0;
            reach = (int)((double)half_width * sqrt(inside) + 0.5);
            begin = half_width - reach;
            end = half_width + reach + 1;
            if (begin < 0)
                begin = 0;
            if (end > width)
                end = width;
        }
        for (col = begin; col < end; col++)
            kernel[row * width + col] = 1;
    }
    return 0;
}

static int shape_same(const uint8_t* left, const uint8_t* right, int count)
{
    int i;
    for (i = 0; i < count; i++) {
        if (left[i] != right[i])
            return 0;
    }
    return 1;
}

static unsigned shape_lcg(unsigned* state)
{
    *state = *state * 1103515245u + 12345u;
    return (*state >> 16) & 255u;
}

int vizion_gshape_selftest(void)
{
    static const uint8_t rect3[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    static const uint8_t rhombus[25] = {
        0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0};
    static const uint8_t octagon[25] = {
        0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0};
    static const uint8_t dilate9[25] = {
        0, 0, 0, 0, 0, 0, 9, 9, 9, 0, 0, 9, 9, 9, 0, 0, 9, 9, 9, 0, 0, 0, 0, 0, 0};
    static const uint8_t erode0[25] = {
        9, 9, 9, 9, 9, 9, 0, 0, 0, 9, 9, 0, 0, 0, 9, 9, 0, 0, 0, 9, 9, 9, 9, 9, 9};
    static const uint8_t dilate_rh[25] = {
        0, 0, 9, 0, 0, 0, 9, 9, 9, 0, 9, 9, 9, 9, 9, 0, 9, 9, 9, 0, 0, 0, 9, 0, 0};
    static const uint8_t dilate_oc[25] = {
        0, 0, 9, 0, 0, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 9, 0, 0};
    static const uint8_t se[9] = {1, 0, 0, 1, 1, 0, 0, 0, 1};
    static const uint8_t se_src[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    static const uint8_t se_dilate[16] = {5, 6, 7, 7, 9, 10, 11, 11, 13, 14, 15, 15, 13, 14, 15, 15};
    static const uint8_t mean_want[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    uint8_t kernel[25];
    uint8_t spot[25];
    uint8_t field[25];
    uint8_t out[48];
    uint8_t slow[323];
    uint8_t fast[323];
    uint8_t picture[323];
    uint8_t taps[20];
    uint8_t mean_src[9];
    uint8_t color[27];
    uint8_t index[4];
    uint8_t picked[4];
    unsigned state = 7;
    int i;
    int open_blank = 1;
    int close_full = 1;
    memset(spot, 0, sizeof(spot));
    spot[12] = 9;
    memset(field, 9, sizeof(field));
    field[12] = 0;
    if (vizion_shape_kernel(kernel, 3, 3, 0) != 0 || !shape_same(kernel, rect3, 9))
        return 1;
    if (vizion_shape_kernel(kernel, 5, 5, 1) != 0 || !shape_same(kernel, rhombus, 25))
        return 2;
    if (vizion_shape_kernel(kernel, 5, 5, 2) != 0 || !shape_same(kernel, octagon, 25))
        return 3;
    if (vizion_shape_kernel(kernel, 4, 3, 0) == 0 || vizion_shape_kernel(kernel, 3, 3, 4) == 0)
        return 4;
    if (vizion_gray_kernel(spot, out, 5, 5, 1, rect3, 3, 3, 0) != 0 || !shape_same(out, dilate9, 25))
        return 5;
    if (vizion_gray_kernel(spot, out, 5, 5, 1, rect3, 3, 3, 2) != 0)
        return 6;
    for (i = 0; i < 25; i++) {
        if (out[i] != 0)
            open_blank = 0;
    }
    if (!open_blank)
        return 6;
    if (vizion_gray_kernel(field, out, 5, 5, 1, rect3, 3, 3, 1) != 0 || !shape_same(out, erode0, 25))
        return 7;
    if (vizion_gray_kernel(field, out, 5, 5, 1, rect3, 3, 3, 3) != 0)
        return 8;
    for (i = 0; i < 25; i++) {
        if (out[i] != 9)
            close_full = 0;
    }
    if (!close_full)
        return 8;
    if (vizion_gray_kernel(spot, out, 5, 5, 1, rhombus, 5, 5, 0) != 0 || !shape_same(out, dilate_rh, 25))
        return 9;
    if (vizion_gray_kernel(spot, out, 5, 5, 1, octagon, 5, 5, 0) != 0 || !shape_same(out, dilate_oc, 25))
        return 10;
    if (vizion_gray_kernel(se_src, out, 4, 4, 1, se, 3, 3, 0) != 0 || !shape_same(out, se_dilate, 16))
        return 11;
    memset(mean_src, 0, sizeof(mean_src));
    mean_src[4] = 9;
    if (vizion_mean_kernel(mean_src, out, 3, 3, 1, rect3, 3, 3) != 0 || !shape_same(out, mean_want, 9))
        return 12;
    memset(color, 0, sizeof(color));
    color[3 * 4] = 30;
    color[3 * 4 + 1] = 20;
    color[3 * 4 + 2] = 10;
    if (vizion_gray_kernel(color, out, 3, 3, 3, rect3, 3, 3, 0) != 0)
        return 13;
    for (i = 0; i < 9; i++) {
        if (out[i * 3] != 30 || out[i * 3 + 1] != 20 || out[i * 3 + 2] != 10)
            return 13;
    }
    index[0] = 1;
    index[1] = 2;
    index[2] = 3;
    index[3] = 0;
    for (i = 0; i < 4; i++) {
        color[i * 3] = 30;
        color[i * 3 + 1] = 20;
        color[i * 3 + 2] = 10;
    }
    if (vizion_select_gray(color, index, picked, 4, 1, 3) != 0)
        return 14;
    if (picked[0] != 10 || picked[1] != 20 || picked[2] != 30 || picked[3] != 0)
        return 14;
    for (i = 0; i < 323; i++)
        picture[i] = (uint8_t)shape_lcg(&state);
    for (i = 0; i < 20; i++)
        taps[i] = (shape_lcg(&state) & 3) == 0 ? 0 : 1;
    taps[0] = 1;
    if (shape_plane_minmax(picture, slow, 19, 17, taps, 5, 4, 1, 0) != 0
        || shape_plane_minmax(picture, fast, 19, 17, taps, 5, 4, 1, 1) != 0
        || !shape_same(slow, fast, 323))
        return 15;
    if (shape_plane_minmax(picture, slow, 19, 17, taps, 5, 4, 0, 0) != 0
        || shape_plane_minmax(picture, fast, 19, 17, taps, 5, 4, 0, 1) != 0
        || !shape_same(slow, fast, 323))
        return 16;
    if (vizion_mean_kernel(picture, slow, 19, 17, 1, taps, 5, 4) != 0)
        return 17;
    if (vizion_gray_kernel(picture, slow, 19, 17, 1, taps, 5, 4, 0) != 0
        || shape_plane_minmax(picture, fast, 19, 17, taps, 5, 4, 1, 1) != 0
        || !shape_same(slow, fast, 323))
        return 18;
    if (vizion_gray_kernel(NULL, out, 1, 1, 1, rect3, 3, 3, 0) == 0)
        return 19;
    return 0;
}
