#include "vizion_core.h"
#include "threshold_internal.h"

#include <stdio.h>
#include <stdlib.h>

vizion_threshold_fn vizion_threshold_selected(void);

static int vizion_prepare_bounds(int* min_gray, int* max_gray)
{
    if (*min_gray > *max_gray || *min_gray > 255 || *max_gray < 0)
        return 1;
    if (*min_gray < 0)
        *min_gray = 0;
    if (*max_gray > 255)
        *max_gray = 255;
    return 0;
}

int vizion_threshold_runs(
    const uint8_t* gray, int width, int height, int min_gray, int max_gray,
    int* rows, int* columns, int* lengths, int capacity, int* written)
{
    vizion_threshold_fn impl;
    if (!written)
        return -1;
    if (!gray || width < 1 || height < 1)
        return -1;
    if (!rows && (columns || lengths))
        return -1;
    if (rows && (!columns || !lengths || capacity < 0))
        return -1;
    if (vizion_prepare_bounds(&min_gray, &max_gray)) {
        *written = 0;
        return 0;
    }
    impl = vizion_threshold_selected();
    return impl(gray, width, height, min_gray, max_gray, rows, columns, lengths, capacity, written);
}

static int vizion_same_runs(int count, const int* a_row, const int* a_col, const int* a_len,
    const int* b_row, const int* b_col, const int* b_len)
{
    int i;
    for (i = 0; i < count; i++) {
        if (a_row[i] != b_row[i] || a_col[i] != b_col[i] || a_len[i] != b_len[i])
            return 0;
    }
    return 1;
}

static int vizion_check_impl(vizion_threshold_fn impl, const uint8_t* gray, int width, int height,
    int min_gray, int max_gray, const int* expect_row, const int* expect_col, const int* expect_len, int expect)
{
    int count = 0;
    int written = 0;
    int* rows;
    int* columns;
    int* lengths;
    int code;
    if (impl(gray, width, height, min_gray, max_gray, 0, 0, 0, 0, &count) != 0)
        return -1;
    if (count != expect)
        return -1;
    if (count == 0)
        return 0;
    rows = (int*)malloc((size_t)count * sizeof(int));
    columns = (int*)malloc((size_t)count * sizeof(int));
    lengths = (int*)malloc((size_t)count * sizeof(int));
    if (!rows || !columns || !lengths) {
        free(rows);
        free(columns);
        free(lengths);
        return -1;
    }
    code = impl(gray, width, height, min_gray, max_gray, rows, columns, lengths, count, &written);
    if (code != 0 || written != count || !vizion_same_runs(count, rows, columns, lengths, expect_row, expect_col, expect_len))
        code = -1;
    else if (count > 0 && impl(gray, width, height, min_gray, max_gray, rows, columns, lengths, count - 1, &written) != -2)
        code = -1;
    else
        code = 0;
    free(rows);
    free(columns);
    free(lengths);
    return code;
}

int vizion_threshold_selftest(void)
{
    static const int widths[] = {1, 15, 16, 17, 31, 32, 33, 40, 64, 100};
    static const int heights[] = {1, 2, 5};
    static const int bounds[][2] = {{0, 255}, {0, 0}, {255, 255}, {128, 128}, {1, 254}, {200, 250}};
    int flags = vizion_cpu_flags();
    int pattern, width_index, height_index, bound_index;
    uint32_t state = 1;
    for (pattern = 0; pattern < 3; pattern++) {
        for (width_index = 0; width_index < (int)(sizeof(widths) / sizeof(widths[0])); width_index++) {
            for (height_index = 0; height_index < (int)(sizeof(heights) / sizeof(heights[0])); height_index++) {
                int width = widths[width_index];
                int height = heights[height_index];
                int pixels = width * height;
                uint8_t* gray = (uint8_t*)malloc((size_t)pixels);
                int* row = (int*)malloc((size_t)pixels * sizeof(int));
                int* column = (int*)malloc((size_t)pixels * sizeof(int));
                int* length = (int*)malloc((size_t)pixels * sizeof(int));
                int b;
                if (!gray || !row || !column || !length) {
                    free(gray);
                    free(row);
                    free(column);
                    free(length);
                    fprintf(stderr, "閾值自我檢查記憶體不足\n");
                    return 2;
                }
                for (b = 0; b < pixels; b++) {
                    if (pattern == 0) {
                        state = state * 1664525u + 1013904223u;
                        gray[b] = (uint8_t)(state >> 24);
                    } else if (pattern == 1) {
                        gray[b] = (uint8_t)((b % 2) ? 255 : 0);
                    } else {
                        gray[b] = (uint8_t)(b == 31 || b == 32 || b == 33 ? 180 : 10);
                    }
                }
                for (bound_index = 0; bound_index < (int)(sizeof(bounds) / sizeof(bounds[0])); bound_index++) {
                    int min_gray = bounds[bound_index][0];
                    int max_gray = bounds[bound_index][1];
                    int expect = 0;
                    int code = vizion_threshold_scalar(gray, width, height, min_gray, max_gray, row, column, length, pixels, &expect);
                    if (code != 0) {
                        free(gray);
                        free(row);
                        free(column);
                        free(length);
                        fprintf(stderr, "純 C 閾值失敗\n");
                        return 3;
                    }
                    if ((flags & VIZION_CPU_SSE2) && vizion_check_impl(vizion_threshold_sse2, gray, width, height, min_gray, max_gray, row, column, length, expect)) {
                        free(gray);
                        free(row);
                        free(column);
                        free(length);
                        fprintf(stderr, "SSE2 閾值和純 C 不同，寬 %d 高 %d\n", width, height);
                        return 4;
                    }
                    if ((flags & VIZION_CPU_AVX2) && vizion_check_impl(vizion_threshold_avx2, gray, width, height, min_gray, max_gray, row, column, length, expect)) {
                        free(gray);
                        free(row);
                        free(column);
                        free(length);
                        fprintf(stderr, "AVX2 閾值和純 C 不同，寬 %d 高 %d\n", width, height);
                        return 5;
                    }
                    {
                        int dispatched = 0;
                        int* drow = (int*)malloc((size_t)(expect > 0 ? expect : 1) * sizeof(int));
                        int* dcol = (int*)malloc((size_t)(expect > 0 ? expect : 1) * sizeof(int));
                        int* dlen = (int*)malloc((size_t)(expect > 0 ? expect : 1) * sizeof(int));
                        int ok;
                        code = vizion_threshold_runs(gray, width, height, min_gray, max_gray, expect ? drow : 0, expect ? dcol : 0, expect ? dlen : 0, expect, &dispatched);
                        ok = code == 0 && dispatched == expect && (expect == 0 || vizion_same_runs(expect, row, column, length, drow, dcol, dlen));
                        free(drow);
                        free(dcol);
                        free(dlen);
                        if (!ok) {
                            free(gray);
                            free(row);
                            free(column);
                            free(length);
                            fprintf(stderr, "分派後的閾值和純 C 不同\n");
                            return 6;
                        }
                    }
                }
                free(gray);
                free(row);
                free(column);
                free(length);
            }
        }
    }
    {
        uint8_t sample[10] = {0, 10, 10, 0, 10, 10, 10, 10, 10, 0};
        int rows[4], columns[4], lengths[4], written = 0;
        int code = vizion_threshold_runs(sample, 5, 2, 10, 10, rows, columns, lengths, 4, &written);
        if (code != 0 || written != 3 || rows[0] != 0 || columns[0] != 1 || lengths[0] != 2
            || rows[1] != 0 || columns[1] != 4 || lengths[1] != 1
            || rows[2] != 1 || columns[2] != 0 || lengths[2] != 4) {
            fprintf(stderr, "手算的 5x2 閾值不對\n");
            return 7;
        }
        code = vizion_threshold_runs(sample, 5, 2, -5, 1000, 0, 0, 0, 0, &written);
        if (code != 0 || written != 2) {
            fprintf(stderr, "超出 0 到 255 的門檻應留下整列\n");
            return 8;
        }
        code = vizion_threshold_runs(sample, 5, 2, 5, 4, 0, 0, 0, 0, &written);
        if (code != 0 || written != 0) {
            fprintf(stderr, "上下顛倒的門檻應是空的\n");
            return 9;
        }
        if (vizion_threshold_runs(0, 5, 2, 0, 255, 0, 0, 0, 0, &written) != -1)
            return 10;
    }
    return 0;
}
