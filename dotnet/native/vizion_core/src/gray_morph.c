#include "vizion_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* 矩形灰階用可分離的 van Herk/Gil-Werman。圓形是 OpenCV 的橢圓遮罩。圖外侵蝕當 255、膨脹當 0。原點是 k/2。 */

int vizion_gray_rect_sse2(const uint8_t* src, uint8_t* dst, int width, int height, int mask_height, int mask_width, int dilate);
int vizion_gray_rect_avx2(const uint8_t* src, uint8_t* dst, int width, int height, int mask_height, int mask_width, int dilate);
int vizion_region_morph_selftest(void);

static int cv_round(double value)
{
    return (int)(value + (value >= 0.0 ? 0.5 : -0.5));
}

void vizion_line_minmax(const uint8_t* src, uint8_t* dst, int n, int k, int anchor, int dilate, uint8_t* scratch)
{
    int left = anchor;
    int padded = n + k - 1;
    int chunk;
    uint8_t border = dilate ? 0 : 255;
    uint8_t* pad = scratch;
    uint8_t* prefix = scratch + padded;
    uint8_t* suffix = scratch + padded * 2;
    int i;
    if (n < 1 || k < 1)
        return;
    for (i = 0; i < left; i++)
        pad[i] = border;
    memcpy(pad + left, src, (size_t)n);
    for (i = left + n; i < padded; i++)
        pad[i] = border;
    for (chunk = 0; chunk * k < padded; chunk++) {
        int begin = chunk * k;
        int end = begin + k;
        int j;
        if (end > padded)
            end = padded;
        prefix[begin] = pad[begin];
        for (j = begin + 1; j < end; j++)
            prefix[j] = dilate ? (uint8_t)(prefix[j - 1] > pad[j] ? prefix[j - 1] : pad[j])
                               : (uint8_t)(prefix[j - 1] < pad[j] ? prefix[j - 1] : pad[j]);
        suffix[end - 1] = pad[end - 1];
        for (j = end - 2; j >= begin; j--)
            suffix[j] = dilate ? (uint8_t)(suffix[j + 1] > pad[j] ? suffix[j + 1] : pad[j])
                               : (uint8_t)(suffix[j + 1] < pad[j] ? suffix[j + 1] : pad[j]);
    }
    for (i = 0; i < n; i++) {
        uint8_t a = suffix[i];
        uint8_t b = prefix[i + k - 1];
        dst[i] = dilate ? (uint8_t)(a > b ? a : b) : (uint8_t)(a < b ? a : b);
    }
}

int vizion_gray_rect_scalar(const uint8_t* src, uint8_t* dst, int width, int height, int mask_height, int mask_width, int dilate)
{
    int longer = width > height ? width : height;
    int thicker = mask_width > mask_height ? mask_width : mask_height;
    int padded = longer + thicker;
    uint8_t* scratch;
    uint8_t* horizontal;
    uint8_t* column;
    uint8_t* filtered;
    int row;
    int col;
    if (!src || !dst || width < 1 || height < 1 || mask_height < 1 || mask_width < 1)
        return -1;
    scratch = (uint8_t*)malloc((size_t)padded * 3);
    horizontal = (uint8_t*)malloc((size_t)width * (size_t)height);
    column = (uint8_t*)malloc((size_t)height);
    filtered = (uint8_t*)malloc((size_t)height);
    if (!scratch || !horizontal || !column || !filtered) {
        free(scratch);
        free(horizontal);
        free(column);
        free(filtered);
        return -1;
    }
    for (row = 0; row < height; row++)
        vizion_line_minmax(src + (size_t)row * (size_t)width, horizontal + (size_t)row * (size_t)width, width, mask_width, mask_width / 2, dilate, scratch);
    for (col = 0; col < width; col++) {
        for (row = 0; row < height; row++)
            column[row] = horizontal[(size_t)row * (size_t)width + col];
        vizion_line_minmax(column, filtered, height, mask_height, mask_height / 2, dilate, scratch);
        for (row = 0; row < height; row++)
            dst[(size_t)row * (size_t)width + col] = filtered[row];
    }
    free(scratch);
    free(horizontal);
    free(column);
    free(filtered);
    return 0;
}

static int vizion_ellipse_offsets(int radius, int* rows, int* cols, int capacity)
{
    int size = radius * 2 + 1;
    int center = radius;
    double inverse = center == 0 ? 0.0 : 1.0 / (double)(center * center);
    int count = 0;
    int row;
    for (row = 0; row < size; row++) {
        int dy = row - center;
        int reach;
        int start;
        int end;
        int column;
        if (dy > center || -dy > center)
            continue;
        reach = center == 0 ? 0 : cv_round((double)center * sqrt(fmax(0.0, 1.0 - (double)dy * (double)dy * inverse)));
        start = center - reach;
        if (start < 0)
            start = 0;
        end = center + reach + 1;
        if (end > size)
            end = size;
        for (column = start; column < end; column++) {
            if (count < capacity) {
                rows[count] = row - center;
                cols[count] = column - center;
            }
            count++;
        }
    }
    return count;
}

int vizion_gray_ellipse(const uint8_t* src, uint8_t* dst, int width, int height, int radius, int dilate)
{
    int* off_row;
    int* off_col;
    int offsets;
    int capacity;
    int row;
    int col;
    uint8_t border = dilate ? 0 : 255;
    if (!src || !dst || width < 1 || height < 1 || radius < 0)
        return -1;
    capacity = (radius * 2 + 1) * (radius * 2 + 1);
    off_row = (int*)malloc((size_t)capacity * sizeof(int));
    off_col = (int*)malloc((size_t)capacity * sizeof(int));
    if (!off_row || !off_col) {
        free(off_row);
        free(off_col);
        return -1;
    }
    offsets = vizion_ellipse_offsets(radius, off_row, off_col, capacity);
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            uint8_t value = border;
            int index;
            for (index = 0; index < offsets; index++) {
                int sample_row = row + off_row[index];
                int sample_col = col + off_col[index];
                uint8_t sample = border;
                if ((unsigned)sample_row < (unsigned)height && (unsigned)sample_col < (unsigned)width)
                    sample = src[(size_t)sample_row * (size_t)width + sample_col];
                if (dilate) {
                    if (sample > value)
                        value = sample;
                } else if (sample < value) {
                    value = sample;
                }
            }
            dst[(size_t)row * (size_t)width + col] = value;
        }
    }
    free(off_row);
    free(off_col);
    return 0;
}

int vizion_gray_rect(const uint8_t* src, uint8_t* dst, int width, int height, int mask_height, int mask_width, int dilate)
{
    int flags = vizion_cpu_flags();
    if (flags & VIZION_CPU_SELECTED_AVX2)
        return vizion_gray_rect_avx2(src, dst, width, height, mask_height, mask_width, dilate);
    if (flags & VIZION_CPU_SELECTED_SSE2)
        return vizion_gray_rect_sse2(src, dst, width, height, mask_height, mask_width, dilate);
    return vizion_gray_rect_scalar(src, dst, width, height, mask_height, mask_width, dilate);
}

static void vizion_naive_rect(const uint8_t* src, uint8_t* dst, int width, int height, int mask_height, int mask_width, int dilate)
{
    int anchor_row = mask_height / 2;
    int anchor_col = mask_width / 2;
    uint8_t border = dilate ? 0 : 255;
    int row;
    int col;
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            uint8_t value = border;
            int mask_row;
            int mask_col;
            for (mask_row = 0; mask_row < mask_height; mask_row++) {
                int sample_row = row + mask_row - anchor_row;
                for (mask_col = 0; mask_col < mask_width; mask_col++) {
                    int sample_col = col + mask_col - anchor_col;
                    uint8_t sample = border;
                    if ((unsigned)sample_row < (unsigned)height && (unsigned)sample_col < (unsigned)width)
                        sample = src[(size_t)sample_row * (size_t)width + sample_col];
                    if (dilate) {
                        if (sample > value)
                            value = sample;
                    } else if (sample < value) {
                        value = sample;
                    }
                }
            }
            dst[(size_t)row * (size_t)width + col] = value;
        }
    }
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

static int vizion_gray_selftest(void)
{
    int flags = vizion_cpu_flags();
    int widths[] = {1, 2, 3, 4, 7, 17, 32, 33};
    int heights[] = {1, 3, 5, 18};
    int kernels[] = {1, 2, 3, 5};
    int wi, hi, khi, kwi, dilate, i;
    uint32_t state = 7;
    for (wi = 0; wi < 8; wi++) {
        for (hi = 0; hi < 4; hi++) {
            int width = widths[wi];
            int height = heights[hi];
            int pixels = width * height;
            uint8_t* src = (uint8_t*)malloc((size_t)pixels);
            uint8_t* naive = (uint8_t*)malloc((size_t)pixels);
            uint8_t* scalar = (uint8_t*)malloc((size_t)pixels);
            uint8_t* fast = (uint8_t*)malloc((size_t)pixels);
            if (!src || !naive || !scalar || !fast)
                return 1;
            for (i = 0; i < pixels; i++) {
                state = state * 1664525u + 1013904223u;
                src[i] = (uint8_t)(state >> 24);
            }
            for (khi = 0; khi < 4; khi++) {
                for (kwi = 0; kwi < 4; kwi++) {
                    for (dilate = 0; dilate < 2; dilate++) {
                        vizion_naive_rect(src, naive, width, height, kernels[khi], kernels[kwi], dilate);
                        if (vizion_gray_rect_scalar(src, scalar, width, height, kernels[khi], kernels[kwi], dilate) != 0 || !vizion_same(naive, scalar, pixels)) {
                            free(src); free(naive); free(scalar); free(fast);
                            return 2;
                        }
                        if ((flags & VIZION_CPU_SSE2) && (vizion_gray_rect_sse2(src, fast, width, height, kernels[khi], kernels[kwi], dilate) != 0 || !vizion_same(naive, fast, pixels))) {
                            free(src); free(naive); free(scalar); free(fast);
                            return 3;
                        }
                        if ((flags & VIZION_CPU_AVX2) && (vizion_gray_rect_avx2(src, fast, width, height, kernels[khi], kernels[kwi], dilate) != 0 || !vizion_same(naive, fast, pixels))) {
                            free(src); free(naive); free(scalar); free(fast);
                            return 4;
                        }
                    }
                }
            }
            free(src); free(naive); free(scalar); free(fast);
        }
    }
    {
        uint8_t src[25];
        uint8_t dst[25];
        int rows[32];
        int cols[32];
        int count;
        int index;
        memset(src, 0, sizeof(src));
        src[12] = 255;
        if (vizion_gray_ellipse(src, dst, 5, 5, 2, 1) != 0)
            return 5;
        count = vizion_ellipse_offsets(2, rows, cols, 32);
        for (index = 0; index < 25; index++) {
            int on = 0;
            int k;
            int row = index / 5;
            int col = index - row * 5;
            for (k = 0; k < count; k++) {
                if (row == 2 + rows[k] && col == 2 + cols[k])
                    on = 1;
            }
            if ((on ? 255 : 0) != dst[index])
                return 6;
        }
    }
    return 0;
}

int vizion_morph_selftest(void)
{
    int gray = vizion_gray_selftest();
    int region;
    if (gray != 0)
        return gray;
    region = vizion_region_morph_selftest();
    return region == 0 ? 0 : 10 + region;
}
