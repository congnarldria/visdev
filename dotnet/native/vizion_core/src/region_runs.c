#include "vizion_core.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* 行程上的八連通和面積、中心、外接框。這是整段的合併，不是逐像素，所以維持純 C。 */

typedef struct {
    int row;
    int start;
    int end;
} VizRun;

typedef struct {
    int root;
    int row;
    int start;
} VizObject;

static int vizion_cmp_run(const void* left, const void* right)
{
    const VizRun* a = (const VizRun*)left;
    const VizRun* b = (const VizRun*)right;
    if (a->row != b->row)
        return a->row < b->row ? -1 : 1;
    if (a->start != b->start)
        return a->start < b->start ? -1 : 1;
    return 0;
}

static int vizion_cmp_object(const void* left, const void* right)
{
    const VizObject* a = (const VizObject*)left;
    const VizObject* b = (const VizObject*)right;
    if (a->row != b->row)
        return a->row < b->row ? -1 : 1;
    if (a->start != b->start)
        return a->start < b->start ? -1 : 1;
    return 0;
}

static int vizion_find(int* parent, int index)
{
    while (parent[index] != index) {
        parent[index] = parent[parent[index]];
        index = parent[index];
    }
    return index;
}

int vizion_connection8(
    const int* rows, const int* columns, const int* lengths, int count, int height,
    int* out_rows, int* out_columns, int* out_lengths, int* out_offsets, int* out_counts)
{
    VizRun* runs = 0;
    int* parent = 0;
    int* row_count = 0;
    int* row_offset = 0;
    int* row_fill = 0;
    int* row_index = 0;
    int* root_of = 0;
    int* min_index = 0;
    int* members = 0;
    int* cursor = 0;
    int* place = 0;
    VizObject* objects = 0;
    int row;
    int index;
    int object_count = 0;
    int total = 0;
    if (count < 0 || height < 1 || !rows || !columns || !lengths)
        return -1;
    if (count == 0)
        return 0;
    if (!out_rows || !out_columns || !out_lengths || !out_offsets || !out_counts)
        return -1;
    runs = (VizRun*)malloc((size_t)count * sizeof(VizRun));
    parent = (int*)malloc((size_t)count * sizeof(int));
    row_count = (int*)calloc((size_t)height, sizeof(int));
    row_offset = (int*)malloc(((size_t)height + 1) * sizeof(int));
    row_fill = (int*)malloc((size_t)height * sizeof(int));
    row_index = (int*)malloc((size_t)count * sizeof(int));
    root_of = (int*)malloc((size_t)count * sizeof(int));
    min_index = (int*)malloc((size_t)count * sizeof(int));
    members = (int*)calloc((size_t)count, sizeof(int));
    cursor = (int*)malloc((size_t)count * sizeof(int));
    place = (int*)malloc((size_t)count * sizeof(int));
    objects = (VizObject*)malloc((size_t)count * sizeof(VizObject));
    if (!runs || !parent || !row_count || !row_offset || !row_fill || !row_index || !root_of || !min_index || !members || !cursor || !place || !objects) {
        object_count = -1;
        goto done;
    }
    for (index = 0; index < count; index++) {
        runs[index].row = rows[index];
        runs[index].start = columns[index];
        runs[index].end = columns[index] + lengths[index] - 1;
        parent[index] = index;
        min_index[index] = -1;
    }
    qsort(runs, (size_t)count, sizeof(VizRun), vizion_cmp_run);
    for (index = 0; index < count; index++) {
        if ((unsigned)runs[index].row < (unsigned)height)
            row_count[runs[index].row]++;
    }
    row_offset[0] = 0;
    for (row = 0; row < height; row++)
        row_offset[row + 1] = row_offset[row] + row_count[row];
    memcpy(row_fill, row_offset, (size_t)height * sizeof(int));
    for (index = 0; index < count; index++) {
        row = runs[index].row;
        if ((unsigned)row < (unsigned)height)
            row_index[row_fill[row]++] = index;
    }
    for (row = 1; row < height; row++) {
        int above = row_offset[row - 1];
        int above_count = row_count[row - 1];
        int here = row_offset[row];
        int here_count = row_count[row];
        int seen = 0;
        int item;
        for (item = 0; item < here_count; item++) {
            int current = row_index[here + item];
            int low = runs[current].start - 1;
            int high = runs[current].end + 1;
            int probe;
            while (seen < above_count && runs[row_index[above + seen]].end < low)
                seen++;
            probe = seen;
            while (probe < above_count && runs[row_index[above + probe]].start <= high) {
                int left = vizion_find(parent, current);
                int right = vizion_find(parent, row_index[above + probe]);
                if (left != right)
                    parent[right] = left;
                probe++;
            }
        }
    }
    for (index = 0; index < count; index++) {
        int root = vizion_find(parent, index);
        root_of[index] = root;
        if (min_index[root] < 0)
            min_index[root] = index;
        members[root]++;
    }
    for (index = 0; index < count; index++) {
        if (min_index[root_of[index]] != index)
            continue;
        objects[object_count].root = root_of[index];
        objects[object_count].row = runs[index].row;
        objects[object_count].start = runs[index].start;
        object_count++;
    }
    qsort(objects, (size_t)object_count, sizeof(VizObject), vizion_cmp_object);
    for (index = 0; index < object_count; index++) {
        int root = objects[index].root;
        cursor[root] = total;
        out_offsets[index] = total;
        out_counts[index] = members[root];
        total += members[root];
    }
    for (index = 0; index < count; index++)
        place[index] = cursor[root_of[index]]++;
    for (index = 0; index < count; index++) {
        int dest = place[index];
        out_rows[dest] = runs[index].row;
        out_columns[dest] = runs[index].start;
        out_lengths[dest] = runs[index].end - runs[index].start + 1;
    }
done:
    free(runs);
    free(parent);
    free(row_count);
    free(row_offset);
    free(row_fill);
    free(row_index);
    free(root_of);
    free(min_index);
    free(members);
    free(cursor);
    free(place);
    free(objects);
    return object_count;
}

int vizion_area_center(
    const int* rows, const int* columns, const int* lengths, int count,
    int* area, double* row, double* column)
{
    long long sum = 0;
    double row_moment = 0;
    double column_moment = 0;
    int index;
    if (count < 0 || !rows || !columns || !lengths || !area || !row || !column)
        return -1;
    for (index = 0; index < count; index++) {
        int length = lengths[index];
        int start;
        if (length <= 0)
            continue;
        start = columns[index];
        sum += length;
        row_moment += (double)rows[index] * (double)length;
        column_moment += (double)length * (2.0 * (double)start + (double)length - 1.0) / 2.0;
    }
    if (sum == 0) {
        *area = 0;
        *row = 0;
        *column = 0;
        return 0;
    }
    *area = (int)sum;
    *row = row_moment / (double)sum;
    *column = column_moment / (double)sum;
    return 0;
}

int vizion_bounds(
    const int* rows, const int* columns, const int* lengths, int count,
    int* row1, int* column1, int* row2, int* column2)
{
    int top = INT_MAX;
    int left = INT_MAX;
    int bottom = INT_MIN;
    int right = INT_MIN;
    int index;
    if (count < 1 || !rows || !columns || !lengths || !row1 || !column1 || !row2 || !column2)
        return -1;
    for (index = 0; index < count; index++) {
        int edge = columns[index] + lengths[index] - 1;
        if (rows[index] < top)
            top = rows[index];
        if (rows[index] > bottom)
            bottom = rows[index];
        if (columns[index] < left)
            left = columns[index];
        if (edge > right)
            right = edge;
    }
    *row1 = top;
    *column1 = left;
    *row2 = bottom;
    *column2 = right;
    return 0;
}

static int vizion_expect_one(const int* rows, const int* columns, const int* lengths, int count, int height, int expect_objects)
{
    int* out_rows = (int*)malloc((size_t)(count > 0 ? count : 1) * sizeof(int));
    int* out_columns = (int*)malloc((size_t)(count > 0 ? count : 1) * sizeof(int));
    int* out_lengths = (int*)malloc((size_t)(count > 0 ? count : 1) * sizeof(int));
    int* offsets = (int*)malloc((size_t)(count > 0 ? count : 1) * sizeof(int));
    int* counts = (int*)malloc((size_t)(count > 0 ? count : 1) * sizeof(int));
    int objects;
    int sum = 0;
    int index;
    int code = 0;
    if (!out_rows || !out_columns || !out_lengths || !offsets || !counts)
        code = -1;
    else {
        objects = vizion_connection8(rows, columns, lengths, count, height, out_rows, out_columns, out_lengths, offsets, counts);
        if (objects != expect_objects)
            code = -1;
        for (index = 0; index < objects && code == 0; index++)
            sum += counts[index];
        if (code == 0 && sum != count)
            code = -1;
    }
    free(out_rows);
    free(out_columns);
    free(out_lengths);
    free(offsets);
    free(counts);
    return code;
}

int vizion_region_selftest(void)
{
    int diagonal_rows[2] = {0, 1};
    int diagonal_columns[2] = {0, 1};
    int diagonal_lengths[2] = {1, 1};
    int gap_columns[2] = {0, 2};
    int block_rows[3] = {1, 2, 3};
    int block_columns[3] = {2, 2, 2};
    int block_lengths[3] = {4, 4, 4};
    int touch_rows[3] = {0, 0, 1};
    int touch_columns[3] = {0, 5, 1};
    int touch_lengths[3] = {2, 1, 5};
    int area = 0;
    double center_row = 0;
    double center_column = 0;
    int row1, column1, row2, column2;
    if (vizion_connection8(diagonal_rows, diagonal_columns, diagonal_lengths, 0, 3, 0, 0, 0, 0, 0) != 0)
        return 1;
    if (vizion_expect_one(diagonal_rows, diagonal_columns, diagonal_lengths, 2, 3, 1))
        return 2;
    if (vizion_expect_one(diagonal_rows, gap_columns, diagonal_lengths, 2, 3, 2))
        return 3;
    if (vizion_expect_one(block_rows, block_columns, block_lengths, 3, 6, 1))
        return 4;
    if (vizion_expect_one(touch_rows, touch_columns, touch_lengths, 3, 4, 1))
        return 5;
    if (vizion_area_center(block_rows, block_columns, block_lengths, 3, &area, &center_row, &center_column) != 0)
        return 6;
    if (area != 12 || fabs(center_row - 2.0) > 1e-12 || fabs(center_column - 3.5) > 1e-12)
        return 7;
    if (vizion_bounds(block_rows, block_columns, block_lengths, 3, &row1, &column1, &row2, &column2) != 0)
        return 8;
    if (row1 != 1 || column1 != 2 || row2 != 3 || column2 != 5)
        return 9;
    return 0;
}
