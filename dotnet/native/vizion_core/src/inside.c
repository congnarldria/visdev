#include "vizion_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* 圓形排序、到邊界的最低瓶頸、矩形等化、區域平均和一次灰階平面。 */

int vizion_equalize(const uint8_t* src, uint8_t* dst, int count);

static uint8_t inside_round(double value)
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

static int inside_replicate(int position, int length)
{
    if (position < 0)
        return 0;
    if (position >= length)
        return length - 1;
    return position;
}

int vizion_rank_disk(const uint8_t* src, uint8_t* dst, int width, int height, int radius, int rank)
{
    int row;
    int col;
    int count;
    int dy;
    int dx;
    uint8_t samples[128];
    if (!src || !dst || width < 1 || height < 1 || radius < 1 || radius > 5)
        return -1;
    count = 0;
    for (dy = -radius; dy <= radius; dy++) {
        for (dx = -radius; dx <= radius; dx++) {
            if (dy * dy + dx * dx <= radius * radius)
                count++;
        }
    }
    if (rank < 0 || rank >= count || count > 128)
        return -1;
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            int filled = 0;
            for (dy = -radius; dy <= radius; dy++) {
                int yy;
                if (dy * dy > radius * radius)
                    continue;
                yy = inside_replicate(row + dy, height);
                for (dx = -radius; dx <= radius; dx++) {
                    int xx;
                    uint8_t value;
                    int index;
                    if (dy * dy + dx * dx > radius * radius)
                        continue;
                    xx = inside_replicate(col + dx, width);
                    value = src[(size_t)yy * (size_t)width + xx];
                    index = filled;
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

typedef struct InsideNode {
    int value;
    int row;
    int col;
} InsideNode;

static int inside_push(InsideNode** heap, int* size, int* capacity, InsideNode node)
{
    int index;
    if (*size + 1 > *capacity) {
        int next = *capacity < 16 ? 16 : *capacity * 2;
        InsideNode* grown = (InsideNode*)realloc(*heap, (size_t)next * sizeof(InsideNode));
        if (!grown)
            return -1;
        *heap = grown;
        *capacity = next;
    }
    index = (*size)++;
    while (index > 0) {
        int parent = (index - 1) / 2;
        if ((*heap)[parent].value <= node.value)
            break;
        (*heap)[index] = (*heap)[parent];
        index = parent;
    }
    (*heap)[index] = node;
    return 0;
}

static InsideNode inside_pop(InsideNode* heap, int* size)
{
    InsideNode result = heap[0];
    InsideNode last = heap[--(*size)];
    int index = 0;
    if (*size == 0)
        return result;
    while (index * 2 + 1 < *size) {
        int child = index * 2 + 1;
        if (child + 1 < *size && heap[child + 1].value < heap[child].value)
            child++;
        if (last.value <= heap[child].value)
            break;
        heap[index] = heap[child];
        index = child;
    }
    heap[index] = last;
    return result;
}

int vizion_gray_inside(const uint8_t* src, uint8_t* dst, int width, int height)
{
    int* result;
    uint8_t* seen;
    InsideNode* heap;
    int heap_size = 0;
    int heap_capacity;
    int row;
    int col;
    static const int neighbors[8][2] = {
        { -1, -1 }, { -1, 0 }, { -1, 1 }, { 0, -1 }, { 0, 1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }
    };
    if (!src || !dst || width < 1 || height < 1)
        return -1;
    result = (int*)malloc((size_t)width * (size_t)height * sizeof(int));
    seen = (uint8_t*)calloc((size_t)width * (size_t)height, 1);
    heap_capacity = width * height + 8;
    heap = (InsideNode*)malloc((size_t)heap_capacity * sizeof(InsideNode));
    if (!result || !seen || !heap) {
        free(result);
        free(seen);
        free(heap);
        return -1;
    }
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++)
            result[(size_t)row * (size_t)width + col] = 1000000000;
    }
    for (col = 0; col < width; col++) {
        result[col] = src[col];
        result[(size_t)(height - 1) * (size_t)width + col] = src[(size_t)(height - 1) * (size_t)width + col];
    }
    for (row = 0; row < height; row++) {
        result[(size_t)row * (size_t)width] = src[(size_t)row * (size_t)width];
        result[(size_t)row * (size_t)width + (width - 1)] = src[(size_t)row * (size_t)width + (width - 1)];
    }
    for (row = 0; row < height; row++) {
        InsideNode node;
        node.value = result[(size_t)row * (size_t)width];
        node.row = row;
        node.col = 0;
        inside_push(&heap, &heap_size, &heap_capacity, node);
        node.value = result[(size_t)row * (size_t)width + (width - 1)];
        node.col = width - 1;
        inside_push(&heap, &heap_size, &heap_capacity, node);
    }
    for (col = 0; col < width; col++) {
        InsideNode node;
        node.value = result[col];
        node.row = 0;
        node.col = col;
        inside_push(&heap, &heap_size, &heap_capacity, node);
        node.value = result[(size_t)(height - 1) * (size_t)width + col];
        node.row = height - 1;
        inside_push(&heap, &heap_size, &heap_capacity, node);
    }
    while (heap_size > 0) {
        InsideNode node = inside_pop(heap, &heap_size);
        int index;
        if (seen[(size_t)node.row * (size_t)width + node.col])
            continue;
        seen[(size_t)node.row * (size_t)width + node.col] = 1;
        for (index = 0; index < 8; index++) {
            int nrow = node.row + neighbors[index][0];
            int ncol = node.col + neighbors[index][1];
            int candidate;
            if (nrow < 0 || ncol < 0 || nrow >= height || ncol >= width)
                continue;
            candidate = node.value > src[(size_t)nrow * (size_t)width + ncol]
                ? node.value
                : src[(size_t)nrow * (size_t)width + ncol];
            if (candidate < result[(size_t)nrow * (size_t)width + ncol]) {
                InsideNode next;
                result[(size_t)nrow * (size_t)width + ncol] = candidate;
                next.value = candidate;
                next.row = nrow;
                next.col = ncol;
                inside_push(&heap, &heap_size, &heap_capacity, next);
            }
        }
    }
    for (row = 0; row < height * width; row++) {
        int value = result[row];
        if (value < 0)
            value = 0;
        if (value > 255)
            value = 255;
        dst[row] = (uint8_t)value;
    }
    free(result);
    free(seen);
    free(heap);
    return 0;
}

static int inside_descale(int value, int shift)
{
    return (value + (1 << (shift - 1))) >> shift;
}

static int inside_sat(int value)
{
    if (value < 0)
        return 0;
    if (value > 255)
        return 255;
    return value;
}

static void inside_to_ycrcb(uint8_t b, uint8_t g, uint8_t r, int* y, int* cr, int* cb)
{
    int yy = inside_descale((int)b * 1868 + (int)g * 9617 + (int)r * 4899, 14);
    *y = yy;
    *cr = inside_sat(inside_descale(((int)r - yy) * 11682 + (128 << 14), 14));
    *cb = inside_sat(inside_descale(((int)b - yy) * 9241 + (128 << 14), 14));
}

static void inside_from_ycrcb(int y, int cr, int cb, uint8_t* b, uint8_t* g, uint8_t* r)
{
    *b = (uint8_t)inside_sat(y + inside_descale((cb - 128) * 29049, 14));
    *g = (uint8_t)inside_sat(y + inside_descale((cb - 128) * -5636 + (cr - 128) * -11698, 14));
    *r = (uint8_t)inside_sat(y + inside_descale((cr - 128) * 22987, 14));
}

int vizion_equalize_rect(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels,
    int top, int left, int bottom, int right)
{
    int rect_height;
    int rect_width;
    int count;
    uint8_t* crop;
    int row;
    if (!src || !dst || width < 1 || height < 1 || (channels != 1 && channels != 3))
        return -1;
    if (top < 0 || left < 0 || bottom < top || right < left || bottom >= height || right >= width)
        return -1;
    if (dst != src)
        memcpy(dst, src, (size_t)width * (size_t)height * (size_t)channels);
    rect_height = bottom - top + 1;
    rect_width = right - left + 1;
    count = rect_height * rect_width;
    crop = (uint8_t*)malloc((size_t)count);
    if (!crop)
        return -1;
    if (channels == 1) {
        for (row = 0; row < rect_height; row++)
            memcpy(crop + (size_t)row * (size_t)rect_width,
                dst + ((size_t)(top + row) * (size_t)width + left), (size_t)rect_width);
        if (vizion_equalize(crop, crop, count) != 0) {
            free(crop);
            return -1;
        }
        for (row = 0; row < rect_height; row++)
            memcpy(dst + ((size_t)(top + row) * (size_t)width + left),
                crop + (size_t)row * (size_t)rect_width, (size_t)rect_width);
        free(crop);
        return 0;
    }
    {
        uint8_t* y = (uint8_t*)malloc((size_t)width * (size_t)height * 3);
        uint8_t* cr;
        uint8_t* cb;
        int index;
        if (!y) {
            free(crop);
            return -1;
        }
        cr = y + width * height;
        cb = cr + width * height;
        for (index = 0; index < width * height; index++) {
            int yy, rr, bb;
            inside_to_ycrcb(dst[index * 3], dst[index * 3 + 1], dst[index * 3 + 2], &yy, &rr, &bb);
            y[index] = (uint8_t)yy;
            cr[index] = (uint8_t)rr;
            cb[index] = (uint8_t)bb;
        }
        for (row = 0; row < rect_height; row++)
            memcpy(crop + (size_t)row * (size_t)rect_width,
                y + ((size_t)(top + row) * (size_t)width + left), (size_t)rect_width);
        if (vizion_equalize(crop, crop, count) != 0) {
            free(crop);
            free(y);
            return -1;
        }
        for (row = 0; row < rect_height; row++)
            memcpy(y + ((size_t)(top + row) * (size_t)width + left),
                crop + (size_t)row * (size_t)rect_width, (size_t)rect_width);
        for (index = 0; index < width * height; index++)
            inside_from_ycrcb(y[index], cr[index], cb[index], &dst[index * 3], &dst[index * 3 + 1], &dst[index * 3 + 2]);
        free(crop);
        free(y);
    }
    return 0;
}

int vizion_region_mean(
    const uint8_t* gray, uint8_t* dst, int width,
    const int* rows, const int* columns, const int* lengths, int runs)
{
    double sum = 0.0;
    int count = 0;
    int index;
    int gray_value;
    if (!gray || !dst || width < 1 || runs < 0)
        return -1;
    if (runs == 0)
        return 0;
    if (!rows || !columns || !lengths)
        return -1;
    for (index = 0; index < runs; index++) {
        int step;
        if (lengths[index] <= 0)
            continue;
        for (step = 0; step < lengths[index]; step++) {
            sum += gray[(size_t)rows[index] * (size_t)width + columns[index] + step];
            count++;
        }
    }
    if (count == 0)
        return 0;
    gray_value = inside_round(sum / (double)count);
    for (index = 0; index < runs; index++) {
        int step;
        for (step = 0; step < lengths[index]; step++)
            dst[(size_t)rows[index] * (size_t)width + columns[index] + step] = (uint8_t)gray_value;
    }
    return 0;
}

int vizion_surface(uint8_t* dst, int width, int height, double alpha, double beta, double gamma, double row, double column)
{
    int y;
    int x;
    if (!dst || width < 1 || height < 1)
        return -1;
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            double value = alpha * ((double)y - row) + beta * ((double)x - column) + gamma;
            dst[(size_t)y * (size_t)width + x] = inside_round(value);
        }
    }
    return 0;
}

int vizion_surface_quad(
    uint8_t* dst, int width, int height,
    double alpha, double beta, double gamma, double delta, double epsilon, double zeta,
    double row, double column)
{
    int y;
    int x;
    if (!dst || width < 1 || height < 1)
        return -1;
    for (y = 0; y < height; y++) {
        double yy = (double)y - row;
        for (x = 0; x < width; x++) {
            double xx = (double)x - column;
            double value = alpha * yy * yy + beta * xx * xx + gamma * yy * xx + delta * yy + epsilon * xx + zeta;
            dst[(size_t)y * (size_t)width + x] = inside_round(value);
        }
    }
    return 0;
}

static int inside_same(const uint8_t* left, const uint8_t* right, int count)
{
    int index;
    for (index = 0; index < count; index++) {
        if (left[index] != right[index])
            return 0;
    }
    return 1;
}

int vizion_inside_selftest(void)
{
    static const uint8_t src[9] = { 1, 9, 3, 8, 2, 7, 4, 6, 5 };
    static const uint8_t rank0[9] = { 1, 1, 3, 1, 2, 2, 4, 2, 5 };
    static const uint8_t rank2[9] = { 1, 3, 3, 4, 7, 5, 4, 5, 5 };
    static const uint8_t surface_expect[8] = { 10, 13, 16, 19, 12, 15, 18, 21 };
    uint8_t out[9];
    uint8_t surface[8];
    int rows[1] = { 1 };
    int columns[1] = { 0 };
    int lengths[1] = { 3 };
    if (vizion_rank_disk(src, out, 3, 3, 1, 0) != 0 || !inside_same(out, rank0, 9))
        return 1;
    if (vizion_rank_disk(src, out, 3, 3, 1, 2) != 0 || !inside_same(out, rank2, 9))
        return 2;
    if (vizion_gray_inside(src, out, 3, 3) != 0 || !inside_same(out, src, 9))
        return 3;
    if (vizion_surface(surface, 4, 2, 2.0, 3.0, 10.0, 0.0, 0.0) != 0 || !inside_same(surface, surface_expect, 8))
        return 4;
    if (vizion_surface_quad(surface, 1, 3, 1.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0) != 0 || surface[2] != 8)
        return 6;
    memset(out, 0, sizeof(out));
    if (vizion_region_mean(src, out, 3, rows, columns, lengths, 1) != 0 || out[3] != inside_round((8 + 2 + 7) / 3.0) || out[0] != 0)
        return 5;
    return 0;
}
