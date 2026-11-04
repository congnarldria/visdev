#include "vizion_core.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 遲滯和串鏈。鄰居順序、起點排序和 C# 的 Link 相同。座標先用 float 再存成 double。 */

static const int vizion_drow[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
static const int vizion_dcol[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

static void vizion_mark(const int* neighbors, unsigned char* used, const unsigned char* degree, int left, int right)
{
    int slot;
    for (slot = 0; slot < degree[left]; slot++) {
        if (neighbors[left * 8 + slot] == right)
            used[left * 8 + slot] = 1;
    }
    for (slot = 0; slot < degree[right]; slot++) {
        if (neighbors[right * 8 + slot] == left)
            used[right * 8 + slot] = 1;
    }
}

static int vizion_edge_open(const int* neighbors, const unsigned char* used, const unsigned char* degree, int left, int right)
{
    int slot;
    for (slot = 0; slot < degree[left]; slot++) {
        if (neighbors[left * 8 + slot] == right)
            return !used[left * 8 + slot];
    }
    return 0;
}

static int vizion_hysteresis(const unsigned char* center, const float* magnitude, double low, double high, int width, int height, unsigned char* keep)
{
    int count = width * height;
    unsigned char* seen = (unsigned char*)calloc((size_t)count, 1);
    int* stack = (int*)malloc((size_t)count * sizeof(int));
    int index;
    if (!seen || !stack) {
        free(seen);
        free(stack);
        return -1;
    }
    for (index = 0; index < count; index++) {
        int top = 0;
        if (!center[index] || (double)magnitude[index] < high || seen[index])
            continue;
        stack[top++] = index;
        seen[index] = 1;
        while (top > 0) {
            int current = stack[--top];
            int row = current / width;
            int column = current - row * width;
            int direction;
            keep[current] = 1;
            for (direction = 0; direction < 8; direction++) {
                int next_row = row + vizion_drow[direction];
                int next_column = column + vizion_dcol[direction];
                int next;
                if (next_row < 0 || next_column < 0 || next_row >= height || next_column >= width)
                    continue;
                next = next_row * width + next_column;
                if (seen[next] || !center[next] || (double)magnitude[next] < low)
                    continue;
                seen[next] = 1;
                stack[top++] = next;
            }
        }
    }
    free(stack);
    free(seen);
    return 0;
}

static int vizion_emit(double* rows, double* columns, int* offsets, int* counts, int point_capacity, int chain_capacity, int* point_count, int* chain_count, int write, int pixel, int width, const float* shift, const float* normal_x, const float* normal_y, int chain_index, int* chain_length)
{
    int row;
    int column;
    float moved;
    if (*chain_length == 0) {
        if (write) {
            if (*chain_count >= chain_capacity)
                return -2;
            offsets[*chain_count] = *point_count;
        }
    }
    if (write) {
        if (*point_count >= point_capacity)
            return -2;
        row = pixel / width;
        column = pixel - row * width;
        moved = shift[pixel];
        rows[*point_count] = (double)((float)row + moved * normal_y[pixel]);
        columns[*point_count] = (double)((float)column + moved * normal_x[pixel]);
    }
    (*point_count)++;
    (*chain_length)++;
    (void)counts;
    (void)chain_index;
    return 0;
}

int vizion_edge_chains(
    const unsigned char* center, const float* magnitude, double low, double high,
    int width, int height,
    const float* shift, const float* normal_x, const float* normal_y,
    double* rows, double* columns, int* offsets, int* counts,
    int point_capacity, int chain_capacity, int* point_count, int* chain_count)
{
    int count;
    int index;
    int point_total = 0;
    int chain_total = 0;
    int write;
    unsigned char* keep;
    int* map_index;
    int* points;
    int* neighbors;
    unsigned char* degree;
    unsigned char* used;
    int* starts;
    int start_count = 0;
    int point;
    if (!center || !magnitude || !shift || !normal_x || !normal_y || !point_count || !chain_count || width < 1 || height < 1)
        return -1;
    count = width * height;
    write = rows != 0 && columns != 0 && offsets != 0 && counts != 0;
    keep = (unsigned char*)calloc((size_t)count, 1);
    map_index = (int*)malloc((size_t)count * sizeof(int));
    if (!keep || !map_index) {
        free(keep);
        free(map_index);
        return -1;
    }
    if (vizion_hysteresis(center, magnitude, low, high, width, height, keep) != 0) {
        free(keep);
        free(map_index);
        return -1;
    }
    for (index = 0; index < count; index++) {
        map_index[index] = -1;
        if (keep[index])
            point_total++;
    }
    if (point_total == 0) {
        *point_count = 0;
        *chain_count = 0;
        free(keep);
        free(map_index);
        return 0;
    }
    points = (int*)malloc((size_t)point_total * sizeof(int));
    neighbors = (int*)malloc((size_t)point_total * 8 * sizeof(int));
    degree = (unsigned char*)calloc((size_t)point_total, 1);
    used = (unsigned char*)calloc((size_t)point_total * 8, 1);
    starts = (int*)malloc((size_t)point_total * sizeof(int));
    if (!points || !neighbors || !degree || !used || !starts) {
        free(keep);
        free(map_index);
        free(points);
        free(neighbors);
        free(degree);
        free(used);
        free(starts);
        return -1;
    }
    point_total = 0;
    for (index = 0; index < count; index++) {
        if (!keep[index])
            continue;
        map_index[index] = point_total;
        points[point_total++] = index;
    }
    for (point = 0; point < point_total; point++) {
        int pixel = points[point];
        int row = pixel / width;
        int column = pixel - row * width;
        int direction;
        for (direction = 0; direction < 8; direction++) {
            int next_row = row + vizion_drow[direction];
            int next_column = column + vizion_dcol[direction];
            int next;
            int neighbor;
            if (next_row < 0 || next_column < 0 || next_row >= height || next_column >= width)
                continue;
            next = next_row * width + next_column;
            neighbor = map_index[next];
            if (neighbor < 0)
                continue;
            neighbors[point * 8 + degree[point]] = neighbor;
            degree[point]++;
        }
        if (degree[point] != 2)
            starts[start_count++] = point;
    }
    if (start_count == 0)
        starts[start_count++] = 0;
    *point_count = 0;
    *chain_count = 0;
    for (index = 0; index < start_count; index++) {
        int start = starts[index];
        int slot;
        for (slot = 0; slot < degree[start]; slot++) {
            int next = neighbors[start * 8 + slot];
            int current;
            int chain_length = 0;
            int chain_index;
            if (!vizion_edge_open(neighbors, used, degree, start, next))
                continue;
            chain_index = *chain_count;
            vizion_mark(neighbors, used, degree, start, next);
            if (vizion_emit(rows, columns, offsets, counts, point_capacity, chain_capacity, point_count, chain_count, write, points[start], width, shift, normal_x, normal_y, chain_index, &chain_length) != 0) {
                free(keep);
                free(map_index);
                free(points);
                free(neighbors);
                free(degree);
                free(used);
                free(starts);
                return -2;
            }
            current = next;
            for (;;) {
                int unused = 0;
                int choice = -1;
                int neighbor_slot;
                int previous;
                if (vizion_emit(rows, columns, offsets, counts, point_capacity, chain_capacity, point_count, chain_count, write, points[current], width, shift, normal_x, normal_y, chain_index, &chain_length) != 0) {
                    free(keep);
                    free(map_index);
                    free(points);
                    free(neighbors);
                    free(degree);
                    free(used);
                    free(starts);
                    return -2;
                }
                for (neighbor_slot = 0; neighbor_slot < degree[current]; neighbor_slot++) {
                    int candidate = neighbors[current * 8 + neighbor_slot];
                    if (!vizion_edge_open(neighbors, used, degree, current, candidate))
                        continue;
                    unused++;
                    if (choice < 0)
                        choice = candidate;
                }
                if (unused != 1)
                    break;
                previous = current;
                current = choice;
                vizion_mark(neighbors, used, degree, previous, current);
            }
            if (chain_length >= 2) {
                if (write)
                    counts[chain_index] = chain_length;
                (*chain_count)++;
            } else if (write) {
                *point_count -= chain_length;
            } else {
                *point_count -= chain_length;
            }
        }
    }
    chain_total = *chain_count;
    (void)chain_total;
    free(keep);
    free(map_index);
    free(points);
    free(neighbors);
    free(degree);
    free(used);
    free(starts);
    return 0;
}

static int vizion_near(double left, double right)
{
    double diff = left - right;
    if (diff < 0)
        diff = -diff;
    return diff <= 1e-6;
}

int vizion_edge_link_selftest(void)
{
    unsigned char center[4] = {1, 1, 1, 1};
    float magnitude[4] = {100.f, 100.f, 100.f, 100.f};
    float shift[4] = {0.f, 0.f, 0.f, 0.f};
    float normal_x[4] = {0.f, 0.f, 0.f, 0.f};
    float normal_y[4] = {0.f, 0.f, 0.f, 0.f};
    double rows[16];
    double columns[16];
    int offsets[8];
    int counts[8];
    int point_count = 0;
    int chain_count = 0;
    unsigned char line_center[4] = {1, 1, 1, 1};
    float line_magnitude[4] = {80.f, 80.f, 80.f, 80.f};
    float line_shift[4] = {0.f, 0.f, 0.f, 0.f};
    float line_nx[4] = {0.f, 0.f, 0.f, 0.f};
    float line_ny[4] = {0.f, 0.f, 0.f, 0.f};
    if (vizion_edge_chains(line_center, line_magnitude, 20.0, 40.0, 4, 1, line_shift, line_nx, line_ny, rows, columns, offsets, counts, 16, 8, &point_count, &chain_count) != 0)
        return 1;
    if (chain_count != 1 || counts[0] != 4 || !vizion_near(columns[0], 0) || !vizion_near(columns[3], 3))
        return 2;
    if (vizion_edge_chains(center, magnitude, 20.0, 40.0, 2, 2, shift, normal_x, normal_y, rows, columns, offsets, counts, 16, 8, &point_count, &chain_count) != 0)
        return 3;
    if (chain_count != 4)
        return 4;
    if (counts[0] != 2 || !vizion_near(rows[0], 0) || !vizion_near(columns[0], 0) || !vizion_near(columns[1], 1))
        return 5;
    if (counts[3] != 4 || !vizion_near(rows[offsets[3]], 0) || !vizion_near(columns[offsets[3]], 1))
        return 6;
    return 0;
}
