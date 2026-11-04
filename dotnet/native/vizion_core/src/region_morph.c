#include "vizion_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* 區域膨脹、侵蝕只改水平行程，不把區域畫成遮罩。矩形偶數寬高加 1。圓沿用 C# 的偶數捨入。 */

typedef struct {
    int start;
    int end;
} VizSpan;

typedef struct {
    VizSpan* data;
    int count;
    int capacity;
} VizRow;

static int vizion_round_even(double value)
{
    double base = floor(value);
    double diff = value - base;
    int whole = (int)base;
    if (diff > 0.5)
        return whole + 1;
    if (diff < 0.5)
        return whole;
    return (whole & 1) == 0 ? whole : whole + 1;
}

static int vizion_add_span(VizRow* row, int start, int end)
{
    if (start > end)
        return 0;
    if (row->count == row->capacity) {
        int next = row->capacity < 8 ? 8 : row->capacity * 2;
        VizSpan* grown = (VizSpan*)realloc(row->data, (size_t)next * sizeof(VizSpan));
        if (!grown)
            return -1;
        row->data = grown;
        row->capacity = next;
    }
    row->data[row->count].start = start;
    row->data[row->count].end = end;
    row->count++;
    return 0;
}

static int vizion_span_cmp(const void* left, const void* right)
{
    const VizSpan* a = (const VizSpan*)left;
    const VizSpan* b = (const VizSpan*)right;
    if (a->start != b->start)
        return a->start < b->start ? -1 : 1;
    return 0;
}

static int vizion_merge_spans(VizRow* row)
{
    int i;
    int kept = 0;
    if (row->count < 2)
        return 0;
    qsort(row->data, (size_t)row->count, sizeof(VizSpan), vizion_span_cmp);
    for (i = 0; i < row->count; i++) {
        if (kept == 0 || row->data[i].start > row->data[kept - 1].end + 1)
            row->data[kept++] = row->data[i];
        else if (row->data[i].end > row->data[kept - 1].end)
            row->data[kept - 1].end = row->data[i].end;
    }
    row->count = kept;
    return 0;
}

static void vizion_free_rows(VizRow* rows, int height)
{
    int row;
    if (!rows)
        return;
    for (row = 0; row < height; row++)
        free(rows[row].data);
    free(rows);
}

static VizRow* vizion_rows_from_runs(const int* rows, const int* columns, const int* lengths, int count, int width, int height)
{
    VizRow* image = (VizRow*)calloc((size_t)height, sizeof(VizRow));
    int index;
    if (!image)
        return 0;
    for (index = 0; index < count; index++) {
        int row = rows[index];
        int start = columns[index];
        int end = columns[index] + lengths[index] - 1;
        if ((unsigned)row >= (unsigned)height)
            continue;
        if (start < 0)
            start = 0;
        if (end >= width)
            end = width - 1;
        if (vizion_add_span(&image[row], start, end) != 0) {
            vizion_free_rows(image, height);
            return 0;
        }
    }
    for (index = 0; index < height; index++) {
        if (vizion_merge_spans(&image[index]) != 0) {
            vizion_free_rows(image, height);
            return 0;
        }
    }
    return image;
}

static int vizion_intersect_rows(const VizRow* left, const VizRow* right, VizRow* dest)
{
    int index = 0;
    int item;
    for (item = 0; item < left->count; item++) {
        while (index < right->count && right->data[index].end < left->data[item].start)
            index++;
        {
            int probe = index;
            while (probe < right->count && right->data[probe].start <= left->data[item].end) {
                int start = left->data[item].start > right->data[probe].start ? left->data[item].start : right->data[probe].start;
                int end = left->data[item].end < right->data[probe].end ? left->data[item].end : right->data[probe].end;
                if (start <= end && vizion_add_span(dest, start, end) != 0)
                    return -1;
                probe++;
            }
        }
    }
    return 0;
}

static int vizion_emit(const VizRow* image, int height, int* out_rows, int* out_columns, int* out_lengths, int capacity, int* written)
{
    int row;
    int count = 0;
    for (row = 0; row < height; row++)
        count += image[row].count;
    *written = count;
    if (!out_rows)
        return 0;
    if (count > capacity)
        return -2;
    count = 0;
    for (row = 0; row < height; row++) {
        int index;
        for (index = 0; index < image[row].count; index++) {
            out_rows[count] = row;
            out_columns[count] = image[row].data[index].start;
            out_lengths[count] = image[row].data[index].end - image[row].data[index].start + 1;
            count++;
        }
    }
    return 0;
}

static int vizion_odd(int size)
{
    if (size < 1)
        return -1;
    return size % 2 == 0 ? size + 1 : size;
}

int vizion_region_rect(
    const int* rows, const int* columns, const int* lengths, int count,
    int width, int height, int kernel_width, int kernel_height, int dilate,
    int* out_rows, int* out_columns, int* out_lengths, int capacity, int* written)
{
    int wide = vizion_odd(kernel_width);
    int high = vizion_odd(kernel_height);
    int radius_col;
    int radius_row;
    VizRow* source;
    VizRow* result;
    int row;
    int code;
    if (!written || !rows || !columns || !lengths || width < 1 || height < 1 || count < 0)
        return -1;
    if (wide < 0 || high < 0)
        return -1;
    radius_col = wide / 2;
    radius_row = high / 2;
    source = vizion_rows_from_runs(rows, columns, lengths, count, width, height);
    result = (VizRow*)calloc((size_t)height, sizeof(VizRow));
    if (!source || !result) {
        vizion_free_rows(source, height);
        vizion_free_rows(result, height);
        return -1;
    }
    if (dilate) {
        for (row = 0; row < height; row++) {
            int index;
            for (index = 0; index < source[row].count; index++) {
                int start = source[row].data[index].start - radius_col;
                int end = source[row].data[index].end + radius_col;
                int dy;
                if (start < 0)
                    start = 0;
                if (end >= width)
                    end = width - 1;
                if (start > end)
                    continue;
                for (dy = -radius_row; dy <= radius_row; dy++) {
                    int dest = row + dy;
                    if ((unsigned)dest >= (unsigned)height)
                        continue;
                    if (vizion_add_span(&result[dest], start, end) != 0) {
                        code = -1;
                        goto done;
                    }
                }
            }
        }
        for (row = 0; row < height; row++)
            vizion_merge_spans(&result[row]);
    } else {
        VizRow* shrunk = (VizRow*)calloc((size_t)height, sizeof(VizRow));
        if (!shrunk) {
            code = -1;
            goto done;
        }
        for (row = 0; row < height; row++) {
            int index;
            for (index = 0; index < source[row].count; index++) {
                int start = source[row].data[index].start + radius_col;
                int end = source[row].data[index].end - radius_col;
                if (start <= end && vizion_add_span(&shrunk[row], start, end) != 0) {
                    vizion_free_rows(shrunk, height);
                    code = -1;
                    goto done;
                }
            }
            vizion_merge_spans(&shrunk[row]);
        }
        for (row = radius_row; row < height - radius_row; row++) {
            VizRow current = shrunk[row - radius_row];
            VizRow combined = {0, 0, 0};
            int other;
            int first = 1;
            for (other = row - radius_row; other <= row + radius_row; other++) {
                VizRow next = {0, 0, 0};
                if (first) {
                    int index;
                    for (index = 0; index < shrunk[other].count; index++) {
                        if (vizion_add_span(&combined, shrunk[other].data[index].start, shrunk[other].data[index].end) != 0) {
                            free(combined.data);
                            vizion_free_rows(shrunk, height);
                            code = -1;
                            goto done;
                        }
                    }
                    first = 0;
                    (void)current;
                    continue;
                }
                if (vizion_intersect_rows(&combined, &shrunk[other], &next) != 0) {
                    free(combined.data);
                    free(next.data);
                    vizion_free_rows(shrunk, height);
                    code = -1;
                    goto done;
                }
                free(combined.data);
                combined = next;
            }
            result[row] = combined;
        }
        vizion_free_rows(shrunk, height);
    }
    code = vizion_emit(result, height, out_rows, out_columns, out_lengths, capacity, written);
done:
    vizion_free_rows(source, height);
    vizion_free_rows(result, height);
    return code;
}

static int vizion_disk_count(double radius, int* dy, int* dx, int capacity)
{
    int size = vizion_round_even(radius * 2.0);
    int center;
    double inverse;
    int count = 0;
    int row;
    if (size < 1)
        size = 1;
    if (size % 2 == 0)
        size++;
    center = size / 2;
    inverse = center == 0 ? 0.0 : 1.0 / (double)(center * center);
    for (row = 0; row < size; row++) {
        double delta = (double)row - (double)center;
        int reach;
        int start;
        int end;
        int column;
        if (fabs(delta) > center)
            continue;
        reach = center == 0 ? 0 : vizion_round_even((double)center * sqrt(fmax(0.0, 1.0 - delta * delta * inverse)));
        start = center - reach;
        if (start < 0)
            start = 0;
        end = center + reach + 1;
        if (end > size)
            end = size;
        for (column = start; column < end; column++) {
            if (count < capacity && dy && dx) {
                dy[count] = row - center;
                dx[count] = column - center;
            }
            count++;
        }
    }
    return count;
}

int vizion_region_disk(
    const int* rows, const int* columns, const int* lengths, int count,
    int width, int height, double radius, int dilate,
    int* out_rows, int* out_columns, int* out_lengths, int capacity, int* written)
{
    int offsets;
    int* dy;
    int* dx;
    VizRow* source;
    VizRow* result;
    int index;
    int code = 0;
    if (!written || !rows || !columns || !lengths || width < 1 || height < 1 || count < 0 || radius < 0)
        return -1;
    offsets = vizion_disk_count(radius, 0, 0, 0);
    dy = (int*)malloc((size_t)(offsets > 0 ? offsets : 1) * sizeof(int));
    dx = (int*)malloc((size_t)(offsets > 0 ? offsets : 1) * sizeof(int));
    source = vizion_rows_from_runs(rows, columns, lengths, count, width, height);
    result = (VizRow*)calloc((size_t)height, sizeof(VizRow));
    if (!dy || !dx || !source || !result || vizion_disk_count(radius, dy, dx, offsets) != offsets) {
        free(dy);
        free(dx);
        vizion_free_rows(source, height);
        vizion_free_rows(result, height);
        return -1;
    }
    if (dilate) {
        for (index = 0; index < offsets; index++) {
            int row;
            for (row = 0; row < height; row++) {
                int dest = row + dy[index];
                int span;
                if ((unsigned)dest >= (unsigned)height)
                    continue;
                for (span = 0; span < source[row].count; span++) {
                    int start = source[row].data[span].start + dx[index];
                    int end = source[row].data[span].end + dx[index];
                    if (start < 0)
                        start = 0;
                    if (end >= width)
                        end = width - 1;
                    if (vizion_add_span(&result[dest], start, end) != 0) {
                        code = -1;
                        goto done;
                    }
                }
            }
        }
        for (index = 0; index < height; index++)
            vizion_merge_spans(&result[index]);
    } else if (offsets > 0) {
        for (index = 0; index < offsets; index++) {
            VizRow* shifted = (VizRow*)calloc((size_t)height, sizeof(VizRow));
            int row;
            if (!shifted) {
                code = -1;
                goto done;
            }
            for (row = 0; row < height; row++) {
                int dest = row + dy[index];
                int span;
                if ((unsigned)dest >= (unsigned)height)
                    continue;
                for (span = 0; span < source[dest].count; span++) {
                    int start = source[dest].data[span].start - dx[index];
                    int end = source[dest].data[span].end - dx[index];
                    if (start < 0)
                        start = 0;
                    if (end >= width)
                        end = width - 1;
                    if (vizion_add_span(&shifted[row], start, end) != 0) {
                        vizion_free_rows(shifted, height);
                        code = -1;
                        goto done;
                    }
                }
                vizion_merge_spans(&shifted[row]);
            }
            if (index == 0) {
                vizion_free_rows(result, height);
                result = shifted;
            } else {
                VizRow* next = (VizRow*)calloc((size_t)height, sizeof(VizRow));
                if (!next) {
                    vizion_free_rows(shifted, height);
                    code = -1;
                    goto done;
                }
                for (row = 0; row < height; row++) {
                    if (vizion_intersect_rows(&result[row], &shifted[row], &next[row]) != 0) {
                        vizion_free_rows(shifted, height);
                        vizion_free_rows(next, height);
                        code = -1;
                        goto done;
                    }
                }
                vizion_free_rows(shifted, height);
                vizion_free_rows(result, height);
                result = next;
            }
        }
    }
    if (code == 0)
        code = vizion_emit(result, height, out_rows, out_columns, out_lengths, capacity, written);
done:
    free(dy);
    free(dx);
    vizion_free_rows(source, height);
    vizion_free_rows(result, height);
    return code;
}

static int vizion_expect_runs(const int* rows, const int* columns, const int* lengths, int count, int width, int height,
    int kernel_width, int kernel_height, int dilate, const int* expect, int expect_count)
{
    int written = 0;
    int* out_rows;
    int* out_columns;
    int* out_lengths;
    int code;
    int index;
    code = vizion_region_rect(rows, columns, lengths, count, width, height, kernel_width, kernel_height, dilate, 0, 0, 0, 0, &written);
    if (code != 0 || written != expect_count)
        return -1;
    if (expect_count == 0)
        return 0;
    out_rows = (int*)malloc((size_t)written * sizeof(int) * 3);
    out_columns = out_rows + written;
    out_lengths = out_columns + written;
    code = vizion_region_rect(rows, columns, lengths, count, width, height, kernel_width, kernel_height, dilate, out_rows, out_columns, out_lengths, written, &written);
    if (code != 0)
        code = -1;
    for (index = 0; code == 0 && index < expect_count; index++) {
        if (out_rows[index] != expect[index * 3] || out_columns[index] != expect[index * 3 + 1] || out_lengths[index] != expect[index * 3 + 2])
            code = -1;
    }
    free(out_rows);
    return code;
}

int vizion_region_morph_selftest(void)
{
    int rows[3] = {3, 4, 5};
    int columns[3] = {3, 3, 3};
    int lengths[3] = {3, 3, 3};
    int erode_expect[3] = {4, 4, 1};
    int dilate_expect[15] = {
        2, 2, 5, 3, 2, 5, 4, 2, 5, 5, 2, 5, 6, 2, 5
    };
    int circle_written = 0;
    int code = vizion_expect_runs(rows, columns, lengths, 3, 9, 9, 3, 3, 0, erode_expect, 1);
    if (code != 0)
        return 1;
    code = vizion_expect_runs(rows, columns, lengths, 3, 9, 9, 2, 2, 1, dilate_expect, 5);
    if (code != 0)
        return 2;
    code = vizion_region_disk(rows, columns, lengths, 3, 9, 9, 1.0, 0, 0, 0, 0, 0, &circle_written);
    if (code != 0 || circle_written != 1)
        return 3;
    code = vizion_region_disk(rows, columns, lengths, 3, 9, 9, 1.0, 1, 0, 0, 0, 0, &circle_written);
    if (code != 0 || circle_written != 5)
        return 4;
    return 0;
}
