#include "vizion_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int vizion_reflect101(int position, int length);

static int thresh_inside(const uint8_t* domain, int index)
{
    return !domain || domain[index];
}

static int thresh_round(double value)
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

static int thresh_box(const float* src, float* dst, int width, int height, int radius)
{
    int window = radius * 2 + 1;
    int row;
    int col;
    int tap;
    double* horizontal;
    if (radius <= 0) {
        memcpy(dst, src, (size_t)width * (size_t)height * sizeof(float));
        return 0;
    }
    horizontal = (double*)malloc((size_t)width * (size_t)height * sizeof(double));
    if (!horizontal)
        return -1;
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            double acc = 0.0;
            for (tap = -radius; tap <= radius; tap++)
                acc += (double)src[(size_t)row * (size_t)width + vizion_reflect101(col + tap, width)];
            horizontal[(size_t)row * (size_t)width + col] = acc / (double)window;
        }
    }
    for (col = 0; col < width; col++) {
        for (row = 0; row < height; row++) {
            double acc = 0.0;
            for (tap = -radius; tap <= radius; tap++)
                acc += horizontal[(size_t)vizion_reflect101(row + tap, height) * (size_t)width + col];
            dst[(size_t)row * (size_t)width + col] = (float)(acc / (double)window);
        }
    }
    free(horizontal);
    return 0;
}

/* light：原圖 >= 平滑圖 + Offset。dark 是反過來。equal 是絕對差不超過 Offset。 */

int vizion_dyn_threshold(
    const uint8_t* image, const uint8_t* smooth, const uint8_t* domain, uint8_t* mask,
    int count, double offset, int mode)
{
    int index;
    if (!image || !smooth || !mask || count < 1 || mode < 0 || mode > 3)
        return -1;
    for (index = 0; index < count; index++) {
        int base = image[index];
        int other = smooth[index];
        int delta = base - other;
        int abs_delta = delta < 0 ? -delta : delta;
        int keep = 0;
        if (!thresh_inside(domain, index)) {
            mask[index] = 0;
            continue;
        }
        if (mode == 0)
            keep = (double)base >= (double)other + offset;
        else if (mode == 1)
            keep = (double)base <= (double)other - offset;
        else if (mode == 2)
            keep = (double)abs_delta <= offset;
        else
            keep = (double)abs_delta > offset;
        mask[index] = keep ? 255 : 0;
    }
    return 0;
}

/* dark 比局部平均低超過 Offset。light 比局部平均高超過 Offset。 */

int vizion_local_threshold(
    const uint8_t* gray, uint8_t* mask, int width, int height, int side, double offset, int light)
{
    int pixels;
    int index;
    int radius;
    float* values;
    float* mean;
    if (!gray || !mask || width < 1 || height < 1 || side < 1 || (side & 1) == 0)
        return -1;
    pixels = width * height;
    values = (float*)malloc((size_t)pixels * sizeof(float) * 2);
    if (!values)
        return -1;
    mean = values + pixels;
    for (index = 0; index < pixels; index++)
        values[index] = (float)gray[index];
    radius = side / 2;
    if (thresh_box(values, mean, width, height, radius) != 0) {
        free(values);
        return -1;
    }
    for (index = 0; index < pixels; index++) {
        int keep;
        if (light)
            keep = (double)values[index] > (double)mean[index] + offset + 1e-6;
        else
            keep = (double)values[index] + 1e-6 < (double)mean[index] - offset;
        mask[index] = keep ? 255 : 0;
    }
    free(values);
    return 0;
}

/* 鄰域裡每個像素都一樣才算平坦。圖外膨脹當 0、侵蝕當 255。 */

int vizion_plateau(const uint8_t* gray, const uint8_t* domain, uint8_t* mask, int width, int height, int side)
{
    int radius;
    int row;
    int column;
    if (!gray || !mask || width < 1 || height < 1 || side < 1 || (side & 1) == 0)
        return -1;
    radius = side / 2;
    for (row = 0; row < height; row++) {
        for (column = 0; column < width; column++) {
            int value = gray[row * width + column];
            int highest = 0;
            int lowest = 255;
            int dy;
            int dx;
            int keep;
            for (dy = -radius; dy <= radius; dy++) {
                for (dx = -radius; dx <= radius; dx++) {
                    int rr = row + dy;
                    int cc = column + dx;
                    int high_sample = (rr < 0 || cc < 0 || rr >= height || cc >= width) ? 0 : gray[rr * width + cc];
                    int low_sample = (rr < 0 || cc < 0 || rr >= height || cc >= width) ? 255 : gray[rr * width + cc];
                    if (high_sample > highest)
                        highest = high_sample;
                    if (low_sample < lowest)
                        lowest = low_sample;
                }
            }
            keep = value == highest && value == lowest && thresh_inside(domain, row * width + column);
            mask[row * width + column] = keep ? 255 : 0;
        }
    }
    return 0;
}

/* binary 留下大於門檻的像素。binary_inv 留下小於等於門檻的像素。 */

int vizion_opencv_threshold(
    const uint8_t* gray, const uint8_t* domain, uint8_t* mask, int count, int level, int inverse)
{
    int index;
    if (!gray || !mask || count < 1 || level < 0 || level > 255)
        return -1;
    for (index = 0; index < count; index++) {
        int keep = inverse ? gray[index] <= level : gray[index] > level;
        if (!thresh_inside(domain, index))
            keep = 0;
        mask[index] = keep ? 255 : 0;
    }
    return 0;
}

static int thresh_otsu(const uint8_t* gray, int count)
{
    double hist[256];
    double sum_all = 0.0;
    double sum_b = 0.0;
    double weight_b = 0.0;
    double best = -1.0;
    int index;
    int threshold = 0;
    memset(hist, 0, sizeof(hist));
    for (index = 0; index < count; index++)
        hist[gray[index]] += 1.0;
    for (index = 0; index < 256; index++)
        sum_all += (double)index * hist[index];
    for (index = 0; index < 256; index++) {
        double weight_f;
        double mean_b;
        double mean_f;
        double between;
        weight_b += hist[index];
        if (weight_b == 0.0)
            continue;
        weight_f = (double)count - weight_b;
        if (weight_f == 0.0)
            break;
        sum_b += (double)index * hist[index];
        mean_b = sum_b / weight_b;
        mean_f = (sum_all - sum_b) / weight_f;
        between = weight_b * weight_f * (mean_b - mean_f) * (mean_b - mean_f);
        if (between > best) {
            best = between;
            threshold = index;
        }
    }
    return threshold;
}

/* 全圖同一個灰階時是空的。否則留下比 Otsu 門檻亮的像素。 */

int vizion_bin_threshold(const uint8_t* gray, uint8_t* mask, int count)
{
    int index;
    int low;
    int high;
    int level;
    if (!gray || !mask || count < 1)
        return -1;
    low = gray[0];
    high = gray[0];
    for (index = 1; index < count; index++) {
        if (gray[index] < low)
            low = gray[index];
        if (gray[index] > high)
            high = gray[index];
    }
    if (low == high) {
        memset(mask, 0, (size_t)count);
        return 0;
    }
    level = thresh_otsu(gray, count);
    for (index = 0; index < count; index++)
        mask[index] = gray[index] > level ? 255 : 0;
    return 0;
}

static void thresh_dilate(const uint8_t* src, uint8_t* dst, int width, int height, const uint8_t* kernel, int side)
{
    int radius = side / 2;
    int row;
    int column;
    for (row = 0; row < height; row++) {
        for (column = 0; column < width; column++) {
            int dy;
            int hot = 0;
            for (dy = -radius; dy <= radius && !hot; dy++) {
                int dx;
                for (dx = -radius; dx <= radius; dx++) {
                    int rr = row + dy;
                    int cc = column + dx;
                    int kr = dy + radius;
                    int kc = dx + radius;
                    if (!kernel[kr * side + kc])
                        continue;
                    if (rr < 0 || cc < 0 || rr >= height || cc >= width)
                        continue;
                    if (src[rr * width + cc]) {
                        hot = 1;
                        break;
                    }
                }
            }
            dst[row * width + column] = hot ? 255 : 0;
        }
    }
}

static void thresh_flood(const uint8_t* allow, uint8_t* mask, int width, int height, int start)
{
    int capacity = width * height;
    int* queue = (int*)malloc((size_t)capacity * sizeof(int));
    int head = 0;
    int tail = 0;
    static const int drow[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    static const int dcol[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    if (!queue)
        return;
    queue[tail++] = start;
    mask[start] = 255;
    while (head < tail) {
        int index = queue[head++];
        int row = index / width;
        int column = index - row * width;
        int neighbor;
        for (neighbor = 0; neighbor < 8; neighbor++) {
            int nr = row + drow[neighbor];
            int nc = column + dcol[neighbor];
            int next;
            if (nr < 0 || nc < 0 || nr >= height || nc >= width)
                continue;
            next = nr * width + nc;
            if (!allow[next] || mask[next])
                continue;
            mask[next] = 255;
            queue[tail++] = next;
        }
    }
    free(queue);
}

/* 高門檻是種子。低門檻以上、和種子八連通的像素留下。MaxLength 先把種子做距離膨脹。 */

int vizion_hysteresis_threshold(
    const uint8_t* gray, const uint8_t* domain, uint8_t* mask, int width, int height,
    double low, double high, double length)
{
    int pixels;
    int index;
    int radius = 0;
    uint8_t* low_mask;
    uint8_t* seeds;
    uint8_t* grown;
    if (!gray || !mask || width < 1 || height < 1 || high < low || length < 0.0)
        return -1;
    pixels = width * height;
    low_mask = (uint8_t*)calloc((size_t)pixels, 3);
    if (!low_mask)
        return -1;
    seeds = low_mask + pixels;
    grown = seeds + pixels;
    for (index = 0; index < pixels; index++) {
        if (!thresh_inside(domain, index))
            continue;
        if ((double)gray[index] >= low)
            low_mask[index] = 255;
        if ((double)gray[index] >= high)
            seeds[index] = 255;
    }
    if (length > 0.0) {
        int side;
        uint8_t* kernel;
        radius = thresh_round(length);
        if (radius < 1)
            radius = 1;
        side = radius * 2 + 1;
        kernel = (uint8_t*)malloc((size_t)side * (size_t)side);
        if (!kernel || vizion_shape_kernel(kernel, side, side, 2) != 0) {
            free(kernel);
            free(low_mask);
            return -1;
        }
        thresh_dilate(seeds, grown, width, height, kernel, side);
        free(kernel);
        memcpy(seeds, grown, (size_t)pixels);
    }
    memset(mask, 0, (size_t)pixels);
    for (index = 0; index < pixels; index++) {
        if (seeds[index] && low_mask[index] && !mask[index])
            thresh_flood(low_mask, mask, width, height, index);
    }
    free(low_mask);
    return 0;
}

static int thresh_box_rect(const float* src, float* dst, int width, int height, int mask_width, int mask_height)
{
    int radius_x = mask_width / 2;
    int radius_y = mask_height / 2;
    int row;
    int col;
    int tap;
    double* horizontal = (double*)malloc((size_t)width * (size_t)height * sizeof(double));
    if (!horizontal)
        return -1;
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            double acc = 0.0;
            for (tap = -radius_x; tap <= radius_x; tap++)
                acc += (double)src[(size_t)row * (size_t)width + vizion_reflect101(col + tap, width)];
            horizontal[(size_t)row * (size_t)width + col] = acc / (double)mask_width;
        }
    }
    for (col = 0; col < width; col++) {
        for (row = 0; row < height; row++) {
            double acc = 0.0;
            for (tap = -radius_y; tap <= radius_y; tap++)
                acc += horizontal[(size_t)vizion_reflect101(row + tap, height) * (size_t)width + col];
            dst[(size_t)row * (size_t)width + col] = (float)(acc / (double)mask_height);
        }
    }
    free(horizontal);
    return 0;
}

static int thresh_stats(const float* values, float* mean, float* deviation, int width, int height, int mask_width, int mask_height)
{
    int pixels = width * height;
    int index;
    float* square = (float*)malloc((size_t)pixels * sizeof(float));
    float* mean_sq = (float*)malloc((size_t)pixels * sizeof(float));
    if (!square || !mean_sq) {
        free(square);
        free(mean_sq);
        return -1;
    }
    for (index = 0; index < pixels; index++)
        square[index] = values[index] * values[index];
    if (thresh_box_rect(values, mean, width, height, mask_width, mask_height) != 0
        || thresh_box_rect(square, mean_sq, width, height, mask_width, mask_height) != 0) {
        free(square);
        free(mean_sq);
        return -1;
    }
    for (index = 0; index < pixels; index++) {
        float var32 = mean_sq[index] - mean[index] * mean[index];
        double spread = (double)var32;
        if (spread < 0.0)
            spread = 0.0;
        deviation[index] = (float)sqrt(spread);
    }
    free(square);
    free(mean_sq);
    return 0;
}

/* 門檻是 max(標準差 × 倍率, AbsThreshold)。light 高於平均加門檻。 */

int vizion_var_threshold(
    const uint8_t* gray, const uint8_t* domain, uint8_t* mask, int width, int height,
    int mask_width, int mask_height, double scale, double absolute, int mode)
{
    int pixels;
    int index;
    float* values;
    float* mean;
    float* deviation;
    if (!gray || !mask || width < 1 || height < 1 || mask_width < 1 || mask_height < 1 || mode < 0 || mode > 3)
        return -1;
    if ((mask_width & 1) == 0)
        mask_width++;
    if ((mask_height & 1) == 0)
        mask_height++;
    pixels = width * height;
    values = (float*)malloc((size_t)pixels * sizeof(float));
    mean = (float*)malloc((size_t)pixels * sizeof(float));
    deviation = (float*)malloc((size_t)pixels * sizeof(float));
    if (!values || !mean || !deviation) {
        free(values);
        free(mean);
        free(deviation);
        return -1;
    }
    for (index = 0; index < pixels; index++)
        values[index] = (float)gray[index];
    if (thresh_stats(values, mean, deviation, width, height, mask_width, mask_height) != 0) {
        free(values);
        free(mean);
        free(deviation);
        return -1;
    }
    for (index = 0; index < pixels; index++) {
        double limit = (double)deviation[index] * scale;
        int keep = 0;
        if (absolute > limit)
            limit = absolute;
        if (!thresh_inside(domain, index)) {
            mask[index] = 0;
            continue;
        }
        if (mode == 0)
            keep = (double)values[index] >= (double)mean[index] + limit;
        else if (mode == 1)
            keep = (double)values[index] <= (double)mean[index] - limit;
        else if (mode == 2)
            keep = fabs((double)values[index] - (double)mean[index]) <= limit;
        else
            keep = fabs((double)values[index] - (double)mean[index]) > limit;
        mask[index] = keep ? 255 : 0;
    }
    free(values);
    free(mean);
    free(deviation);
    return 0;
}

/* 比局部平均乘上 1 + K×(標準差/128 - 1) 還暗的像素留下。 */

int vizion_sauvola(
    const uint8_t* gray, const uint8_t* domain, uint8_t* mask, int width, int height, int side, double factor)
{
    int pixels;
    int index;
    float* values;
    float* mean;
    float* deviation;
    if (!gray || !mask || width < 1 || height < 1 || side < 3 || (side & 1) == 0 || !(factor > 0.0) || factor > 1.0)
        return -1;
    pixels = width * height;
    values = (float*)malloc((size_t)pixels * sizeof(float));
    mean = (float*)malloc((size_t)pixels * sizeof(float));
    deviation = (float*)malloc((size_t)pixels * sizeof(float));
    if (!values || !mean || !deviation) {
        free(values);
        free(mean);
        free(deviation);
        return -1;
    }
    for (index = 0; index < pixels; index++)
        values[index] = (float)gray[index];
    if (thresh_stats(values, mean, deviation, width, height, side, side) != 0) {
        free(values);
        free(mean);
        free(deviation);
        return -1;
    }
    for (index = 0; index < pixels; index++) {
        double spread = (double)deviation[index];
        double level = (double)mean[index] * (1.0 + factor * (spread / 128.0 - 1.0));
        int keep = thresh_inside(domain, index) && (double)values[index] < level;
        mask[index] = keep ? 255 : 0;
    }
    free(values);
    free(mean);
    free(deviation);
    return 0;
}

static void thresh_paint(int* labels, int width, int height, int start, int label, const uint8_t* allow)
{
    int capacity = width * height;
    int* queue = (int*)malloc((size_t)capacity * sizeof(int));
    int head = 0;
    int tail = 0;
    static const int drow[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    static const int dcol[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    labels[start] = label;
    if (!queue)
        return;
    queue[tail++] = start;
    while (head < tail) {
        int index = queue[head++];
        int row = index / width;
        int column = index - row * width;
        int neighbor;
        for (neighbor = 0; neighbor < 8; neighbor++) {
            int nr = row + drow[neighbor];
            int nc = column + dcol[neighbor];
            int next;
            if (nr < 0 || nc < 0 || nr >= height || nc >= width)
                continue;
            next = nr * width + nc;
            if (!allow[next] || labels[next])
                continue;
            labels[next] = label;
            queue[tail++] = next;
        }
    }
    free(queue);
}

/* 比 Threshold 更亮或更暗的像素做八連通。面積夠、而且有更強的像素才留下。 */

int vizion_dual_labels(
    const uint8_t* gray, const uint8_t* domain, int* labels, int width, int height,
    double limit, double strong, int minimum, int* count)
{
    int pixels;
    int index;
    int label = 0;
    uint8_t* allow;
    uint8_t* peak;
    if (!gray || !labels || !count || width < 1 || height < 1 || limit < 0.0 || strong < 0.0 || minimum < 0)
        return -1;
    pixels = width * height;
    allow = (uint8_t*)calloc((size_t)pixels, 2);
    if (!allow)
        return -1;
    peak = allow + pixels;
    memset(labels, 0, (size_t)pixels * sizeof(int));
    for (index = 0; index < pixels; index++) {
        double value = gray[index];
        if (!thresh_inside(domain, index))
            continue;
        if (value > 128.0 + limit || value < 128.0 - limit)
            allow[index] = 255;
        if (value > 128.0 + strong || value < 128.0 - strong)
            peak[index] = 255;
    }
    for (index = 0; index < pixels; index++) {
        int area = 0;
        int hot = 0;
        int cursor;
        if (!allow[index] || labels[index])
            continue;
        thresh_paint(labels, width, height, index, -1, allow);
        for (cursor = 0; cursor < pixels; cursor++) {
            if (labels[cursor] != -1)
                continue;
            area++;
            if (peak[cursor])
                hot = 1;
        }
        if (area >= minimum && hot) {
            label++;
            for (cursor = 0; cursor < pixels; cursor++) {
                if (labels[cursor] == -1)
                    labels[cursor] = label;
            }
        } else {
            for (cursor = 0; cursor < pixels; cursor++) {
                if (labels[cursor] == -1)
                    labels[cursor] = -2;
            }
        }
    }
    *count = label;
    free(allow);
    return 0;
}

/* 從種子八鄰域成長。和相鄰已加入的像素灰階差不超過 Tolerance。 */

int vizion_region_grow(
    const uint8_t* gray, const uint8_t* domain, int* labels, int width, int height,
    const int* rows, const int* columns, int seeds, double tolerance, int minimum, int* count)
{
    int label = 0;
    int seed;
    int* visited;
    int* queue;
    int pixels;
    static const int drow[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    static const int dcol[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    if (!gray || !labels || !rows || !columns || !count || width < 1 || height < 1 || seeds < 1 || minimum < 0)
        return -1;
    pixels = width * height;
    visited = (int*)calloc((size_t)pixels, sizeof(int));
    queue = (int*)malloc((size_t)pixels * sizeof(int));
    if (!visited || !queue) {
        free(visited);
        free(queue);
        return -1;
    }
    memset(labels, 0, (size_t)pixels * sizeof(int));
    for (seed = 0; seed < seeds; seed++) {
        int row = rows[seed];
        int column = columns[seed];
        int start;
        int head = 0;
        int tail = 0;
        int area = 0;
        int* pixels_at;
        int cursor;
        if (row < 0 || column < 0 || row >= height || column >= width)
            continue;
        start = row * width + column;
        if (visited[start] || !thresh_inside(domain, start))
            continue;
        pixels_at = (int*)malloc((size_t)pixels * sizeof(int));
        if (!pixels_at) {
            free(visited);
            free(queue);
            return -1;
        }
        visited[start] = 1;
        queue[tail++] = start;
        while (head < tail) {
            int index = queue[head++];
            int r = index / width;
            int c = index - r * width;
            int neighbor;
            pixels_at[area++] = index;
            for (neighbor = 0; neighbor < 8; neighbor++) {
                int nr = r + drow[neighbor];
                int nc = c + dcol[neighbor];
                int next;
                int delta;
                if (nr < 0 || nc < 0 || nr >= height || nc >= width)
                    continue;
                next = nr * width + nc;
                if (visited[next] || !thresh_inside(domain, next))
                    continue;
                delta = (int)gray[next] - (int)gray[index];
                if (delta < 0)
                    delta = -delta;
                if ((double)delta <= tolerance) {
                    visited[next] = 1;
                    queue[tail++] = next;
                }
            }
        }
        if (area >= minimum) {
            label++;
            for (cursor = 0; cursor < area; cursor++)
                labels[pixels_at[cursor]] = label;
        }
        free(pixels_at);
    }
    *count = label;
    free(visited);
    free(queue);
    return 0;
}
