#include "vizion_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* 直方圖平滑、區域成長和自動門檻。高斯核與 OpenCV getGaussianKernel 相同，半值四捨五入到偶數。 */

int vizion_gray_ellipse(const uint8_t* src, uint8_t* dst, int width, int height, int radius, int dilate);

static int reflect101(int position, int length)
{
    if (length <= 1)
        return 0;
    while (position < 0 || position >= length) {
        if (position < 0)
            position = -position;
        if (position >= length)
            position = length + length - position - 2;
    }
    return position;
}

static int round_even(double value)
{
    double down = floor(value);
    double fraction = value - down;
    int base;
    if (fraction > 0.5)
        return (int)down + 1;
    if (fraction < 0.5)
        return (int)down;
    base = (int)down;
    if ((base & 1) != 0)
        return base + 1;
    return base;
}

static int clamp_index(int position, int length)
{
    if (position < 0)
        return 0;
    if (position >= length)
        return length - 1;
    return position;
}

static int inside_mask(const uint8_t* domain, int index)
{
    return domain == NULL || domain[index] != 0;
}

static void gauss_kernel(int size, double sigma, double* kernel)
{
    double sigma_x = sigma > 0.0 ? sigma : ((size - 1) * 0.5 - 1.0) * 0.3 + 0.8;
    double scale = -0.5 / (sigma_x * sigma_x);
    double sum = 0.0;
    int index;
    for (index = 0; index < size; index++) {
        double offset = index - (size - 1) * 0.5;
        kernel[index] = exp(scale * offset * offset);
        sum += kernel[index];
    }
    for (index = 0; index < size; index++)
        kernel[index] /= sum;
}

static void convolve_same(const double* signal, int length, const double* kernel, int size, double* out)
{
    int radius = size / 2;
    int index;
    for (index = 0; index < length; index++) {
        double sum = 0.0;
        int tap;
        for (tap = 0; tap < size; tap++) {
            int sample = index + tap - radius;
            if (sample >= 0 && sample < length)
                sum += signal[sample] * kernel[tap];
        }
        out[index] = sum;
    }
}

static void blur_reflect(const double* signal, int length, const double* kernel, int size, double* out)
{
    int radius = size / 2;
    int index;
    for (index = 0; index < length; index++) {
        double sum = 0.0;
        int tap;
        for (tap = 0; tap < size; tap++)
            sum += signal[reflect101(index + tap - radius, length)] * kernel[tap];
        out[index] = sum;
    }
}

static int odd_span(int length, double sigma)
{
    int raw = round_even(sigma * 6.0);
    int odd = raw | 1;
    int cap = (length & 1) ? length : length - 1;
    int size = odd < 3 ? 3 : odd;
    if (size > cap)
        size = cap;
    return size;
}

static int otsu_level(const uint8_t* gray, int count)
{
    double hist[256];
    double sum_all = 0.0;
    double sum_back = 0.0;
    double weight_back = 0.0;
    double best = -1.0;
    int index;
    int threshold = 0;
    memset(hist, 0, sizeof(hist));
    for (index = 0; index < count; index++)
        hist[gray[index]] += 1.0;
    for (index = 0; index < 256; index++)
        sum_all += (double)index * hist[index];
    for (index = 0; index < 256; index++) {
        double weight_fore;
        double mean_back;
        double mean_fore;
        double between;
        weight_back += hist[index];
        if (weight_back == 0.0)
            continue;
        weight_fore = (double)count - weight_back;
        if (weight_fore == 0.0)
            break;
        sum_back += (double)index * hist[index];
        mean_back = sum_back / weight_back;
        mean_fore = (sum_all - sum_back) / weight_fore;
        between = weight_back * weight_fore * (mean_back - mean_fore) * (mean_back - mean_fore);
        if (between > best) {
            best = between;
            threshold = index;
        }
    }
    return threshold;
}

static int triangle_level(const uint8_t* gray, int count)
{
    int hist[256];
    int index;
    int left = 0;
    int right = 0;
    int peak = 0;
    int peak_count = 0;
    int flipped = 0;
    double thresh;
    double height;
    double slope;
    double best = 0.0;
    memset(hist, 0, sizeof(hist));
    for (index = 0; index < count; index++)
        hist[gray[index]]++;
    for (index = 0; index < 256; index++) {
        if (hist[index] > 0) {
            left = index;
            break;
        }
    }
    if (left > 0)
        left--;
    for (index = 255; index > 0; index--) {
        if (hist[index] > 0) {
            right = index;
            break;
        }
    }
    if (right < 255)
        right++;
    for (index = 0; index < 256; index++) {
        if (hist[index] > peak_count) {
            peak_count = hist[index];
            peak = index;
        }
    }
    if (peak - left < right - peak) {
        int cursor = 0;
        int other = 255;
        flipped = 1;
        while (cursor < other) {
            int swap = hist[cursor];
            hist[cursor] = hist[other];
            hist[other] = swap;
            cursor++;
            other--;
        }
        left = 255 - right;
        peak = 255 - peak;
    }
    thresh = (double)left;
    height = (double)peak_count;
    slope = (double)(left - peak);
    for (index = left + 1; index <= peak; index++) {
        double distance = height * (double)index + slope * (double)hist[index];
        if (distance > best) {
            best = distance;
            thresh = (double)index;
        }
    }
    thresh -= 1.0;
    if (flipped)
        thresh = 255.0 - thresh;
    return (int)thresh;
}

static int smooth_histo_level(const uint8_t* gray, int count)
{
    double hist[256];
    double kernel[15];
    double smooth[256];
    int peaks[256];
    int peak_count = 0;
    int index;
    int left;
    int right;
    int valley;
    memset(hist, 0, sizeof(hist));
    for (index = 0; index < count; index++)
        hist[gray[index]] += 1.0;
    gauss_kernel(15, 2.0, kernel);
    convolve_same(hist, 256, kernel, 15, smooth);
    for (index = 1; index < 255; index++) {
        if (smooth[index] >= smooth[index - 1] && smooth[index] >= smooth[index + 1] && smooth[index] > 0.0)
            peaks[peak_count++] = index;
    }
    if (peak_count < 2)
        return otsu_level(gray, count);
    for (index = 1; index < peak_count; index++) {
        int key = peaks[index];
        int cursor = index - 1;
        while (cursor >= 0 && (smooth[peaks[cursor]] < smooth[key]
            || (smooth[peaks[cursor]] == smooth[key] && peaks[cursor] > key))) {
            peaks[cursor + 1] = peaks[cursor];
            cursor--;
        }
        peaks[cursor + 1] = key;
    }
    left = peaks[0] < peaks[1] ? peaks[0] : peaks[1];
    right = peaks[0] < peaks[1] ? peaks[1] : peaks[0];
    valley = left;
    for (index = left + 1; index <= right; index++) {
        if (smooth[index] < smooth[valley])
            valley = index;
    }
    return valley;
}

static int box_mean_u8(const uint8_t* source, uint8_t* mean, int width, int height, int side)
{
    int radius = side / 2;
    int padded_width = width + radius * 2;
    int padded_height = height + radius * 2;
    int stride = padded_width + 1;
    uint8_t* padded;
    int64_t* integral;
    int row;
    int column;
    double area = (double)side * (double)side;
    padded = (uint8_t*)malloc((size_t)padded_width * (size_t)padded_height);
    integral = (int64_t*)calloc((size_t)(padded_height + 1) * (size_t)stride, sizeof(int64_t));
    if (!padded || !integral) {
        free(padded);
        free(integral);
        return -1;
    }
    for (row = 0; row < padded_height; row++) {
        int source_row = clamp_index(row - radius, height);
        for (column = 0; column < padded_width; column++) {
            int source_column = clamp_index(column - radius, width);
            padded[(size_t)row * (size_t)padded_width + column] = source[source_row * width + source_column];
        }
    }
    for (row = 0; row < padded_height; row++) {
        for (column = 0; column < padded_width; column++) {
            int64_t sum = padded[(size_t)row * (size_t)padded_width + column];
            sum += integral[(size_t)row * (size_t)stride + column + 1];
            sum += integral[(size_t)(row + 1) * (size_t)stride + column];
            sum -= integral[(size_t)row * (size_t)stride + column];
            integral[(size_t)(row + 1) * (size_t)stride + column + 1] = sum;
        }
    }
    for (row = 0; row < height; row++) {
        for (column = 0; column < width; column++) {
            int64_t sum = integral[(size_t)(row + side) * (size_t)stride + column + side];
            sum -= integral[(size_t)row * (size_t)stride + column + side];
            sum -= integral[(size_t)(row + side) * (size_t)stride + column];
            sum += integral[(size_t)row * (size_t)stride + column];
            mean[row * width + column] = (uint8_t)round_even((double)sum / area);
        }
    }
    free(padded);
    free(integral);
    return 0;
}

static int gauss_mean_u8(const uint8_t* source, uint8_t* mean, int width, int height, int side)
{
    double exact[64];
    float kernel[64];
    float* horizontal;
    float* vertical;
    int radius = side / 2;
    int row;
    int column;
    int tap;
    if (side > 64)
        return -1;
    gauss_kernel(side, 0.0, exact);
    for (tap = 0; tap < side; tap++)
        kernel[tap] = (float)exact[tap];
    horizontal = (float*)malloc((size_t)width * (size_t)height * sizeof(float));
    vertical = (float*)malloc((size_t)width * (size_t)height * sizeof(float));
    if (!horizontal || !vertical) {
        free(horizontal);
        free(vertical);
        return -1;
    }
    for (row = 0; row < height; row++) {
        for (column = 0; column < width; column++) {
            float sum = 0.0f;
            for (tap = 0; tap < side; tap++) {
                int sample = clamp_index(column + tap - radius, width);
                sum = sum + (float)source[row * width + sample] * kernel[tap];
            }
            horizontal[row * width + column] = sum;
        }
    }
    for (column = 0; column < width; column++) {
        for (row = 0; row < height; row++) {
            float sum = 0.0f;
            for (tap = 0; tap < side; tap++) {
                int sample = clamp_index(row + tap - radius, height);
                sum = sum + horizontal[sample * width + column] * kernel[tap];
            }
            vertical[row * width + column] = sum;
        }
    }
    for (row = 0; row < width * height; row++) {
        int rounded = round_even((double)vertical[row]);
        if (rounded < 0)
            rounded = 0;
        if (rounded > 255)
            rounded = 255;
        mean[row] = (uint8_t)rounded;
    }
    free(horizontal);
    free(vertical);
    return 0;
}

static void apply_domain(uint8_t* mask, const uint8_t* domain, int count, int dark)
{
    int index;
    for (index = 0; index < count; index++) {
        if (dark)
            mask[index] = mask[index] ? 0 : 255;
        if (!inside_mask(domain, index))
            mask[index] = 0;
    }
}

static int find_valleys(const double* counts, int bins, double sigma, int* lows, int* highs)
{
    double* smooth;
    double* kernel = NULL;
    int* valleys;
    int valley_count = 0;
    int index = 1;
    int kept = 0;
    int size = 0;
    smooth = (double*)malloc((size_t)bins * sizeof(double));
    valleys = (int*)malloc((size_t)bins * sizeof(int));
    if (!smooth || !valleys) {
        free(smooth);
        free(valleys);
        return -1;
    }
    if (sigma > 0.0) {
        size = odd_span(bins, sigma);
        kernel = (double*)malloc((size_t)size * sizeof(double));
        if (!kernel) {
            free(smooth);
            free(valleys);
            return -1;
        }
        gauss_kernel(size, sigma, kernel);
        blur_reflect(counts, bins, kernel, size, smooth);
    } else {
        memcpy(smooth, counts, (size_t)bins * sizeof(double));
    }
    while (index < bins - 1) {
        if (smooth[index] < smooth[index - 1]) {
            int end = index;
            while (end + 1 < bins && smooth[end + 1] <= smooth[end])
                end++;
            if (end + 1 < bins && smooth[end + 1] > smooth[end])
                valleys[valley_count++] = (index + end) / 2;
            index = end + 1;
        } else {
            index++;
        }
    }
    for (index = 0; index <= valley_count; index++) {
        int start = index == 0 ? 0 : valleys[index - 1] + 1;
        int end = index == valley_count ? bins - 1 : valleys[index];
        double total = 0.0;
        int bin;
        if (end < start)
            continue;
        for (bin = start; bin <= end; bin++)
            total += counts[bin];
        if (total <= 0.0)
            continue;
        lows[kept] = start;
        highs[kept] = end;
        kept++;
    }
    if (kept == 0) {
        lows[0] = 0;
        highs[0] = bins - 1;
        kept = 1;
    }
    free(kernel);
    free(smooth);
    free(valleys);
    return kept;
}

/* 種子八鄰域成長。鄰居和彈出像素當下的平均差不超過 Tolerance，平均隨新像素更新。 */

int vizion_region_grow_mean(
    const uint8_t* gray, const uint8_t* domain, int* labels, int width, int height,
    const int* rows, const int* columns, int seeds, double tolerance, int minimum, int* count)
{
    static const int drow[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    static const int dcol[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    int pixels;
    int label = 0;
    int seed;
    uint8_t* visited;
    int* queue;
    if (!gray || !labels || !rows || !columns || !count || width < 1 || height < 1 || seeds < 1)
        return -1;
    pixels = width * height;
    visited = (uint8_t*)calloc((size_t)pixels, 1);
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
        int64_t total;
        if (row < 0 || column < 0 || row >= height || column >= width)
            continue;
        start = row * width + column;
        if (visited[start] || !inside_mask(domain, start))
            continue;
        visited[start] = 1;
        queue[tail++] = start;
        total = gray[start];
        area = 1;
        while (head < tail) {
            int index = queue[head++];
            int y = index / width;
            int x = index - y * width;
            double mean = (double)total / (double)area;
            int neighbor;
            for (neighbor = 0; neighbor < 8; neighbor++) {
                int next_row = y + drow[neighbor];
                int next_column = x + dcol[neighbor];
                int next;
                double delta;
                if (next_row < 0 || next_column < 0 || next_row >= height || next_column >= width)
                    continue;
                next = next_row * width + next_column;
                if (visited[next] || !inside_mask(domain, next))
                    continue;
                delta = (double)gray[next] - mean;
                if (delta < 0.0)
                    delta = -delta;
                if (delta <= tolerance) {
                    visited[next] = 1;
                    queue[tail++] = next;
                    total += gray[next];
                    area++;
                }
            }
        }
        if (area >= minimum) {
            int cursor;
            label++;
            for (cursor = 0; cursor < tail; cursor++)
                labels[queue[cursor]] = label;
        }
    }
    *count = label;
    free(visited);
    free(queue);
    return 0;
}

static double channel_gap(const uint8_t* pixels, int channels, int left, int right)
{
    if (channels == 1) {
        int delta = (int)pixels[left] - (int)pixels[right];
        if (delta < 0)
            delta = -delta;
        return (double)delta;
    }
    {
        double sum = 0.0;
        int channel;
        for (channel = 0; channel < channels; channel++) {
            double delta = (double)pixels[left * channels + channel] - (double)pixels[right * channels + channel];
            sum += delta * delta;
        }
        return sqrt(sum);
    }
}

/* 四鄰域依通道歐氏距離分割。距離落在 low 到 high 才合併。 */

int vizion_region_grow_n(
    const uint8_t* pixels, int* labels, int width, int height, int channels,
    double low, double high, int minimum, int* count)
{
    static const int drow[4] = {-1, 1, 0, 0};
    static const int dcol[4] = {0, 0, -1, 1};
    int area_pixels;
    int label = 0;
    int row;
    uint8_t* visited;
    int* queue;
    if (!pixels || !labels || !count || width < 1 || height < 1 || (channels != 1 && channels != 3) || low < 0.0 || high < low)
        return -1;
    area_pixels = width * height;
    visited = (uint8_t*)calloc((size_t)area_pixels, 1);
    queue = (int*)malloc((size_t)area_pixels * sizeof(int));
    if (!visited || !queue) {
        free(visited);
        free(queue);
        return -1;
    }
    memset(labels, 0, (size_t)area_pixels * sizeof(int));
    for (row = 0; row < height; row++) {
        int column;
        for (column = 0; column < width; column++) {
            int start = row * width + column;
            int head = 0;
            int tail = 0;
            if (visited[start])
                continue;
            visited[start] = 1;
            queue[tail++] = start;
            while (head < tail) {
                int index = queue[head++];
                int y = index / width;
                int x = index - y * width;
                int neighbor;
                for (neighbor = 0; neighbor < 4; neighbor++) {
                    int next_row = y + drow[neighbor];
                    int next_column = x + dcol[neighbor];
                    int next;
                    double gap;
                    if (next_row < 0 || next_column < 0 || next_row >= height || next_column >= width)
                        continue;
                    next = next_row * width + next_column;
                    if (visited[next])
                        continue;
                    gap = channel_gap(pixels, channels, next, index);
                    if (low <= gap && gap <= high) {
                        visited[next] = 1;
                        queue[tail++] = next;
                    }
                }
            }
            if (tail >= minimum) {
                int cursor;
                label++;
                for (cursor = 0; cursor < tail; cursor++)
                    labels[queue[cursor]] = label;
            }
        }
    }
    *count = label;
    free(visited);
    free(queue);
    return 0;
}

/* 直方圖高斯平滑後以波谷切段。沒有波谷或全空時是整段。 */

int vizion_histo_to_thresh(const double* counts, int bins, double sigma, int* lows, int* highs, int* segments)
{
    int kept;
    if (!counts || !lows || !highs || !segments || bins < 3 || sigma < 0.0)
        return -1;
    kept = find_valleys(counts, bins, sigma, lows, highs);
    if (kept < 0)
        return -1;
    *segments = kept;
    return 0;
}

/* 依直方圖波谷把灰階切成多塊。標號從 1 開始，暗的在前。 */

int vizion_auto_threshold(
    const uint8_t* gray, const uint8_t* domain, int* labels, int count, double sigma, int* segments)
{
    double hist[256];
    int lows[256];
    int highs[256];
    int kept;
    int index;
    int band;
    int samples = 0;
    if (!gray || !labels || !segments || count < 1 || sigma < 0.0)
        return -1;
    memset(hist, 0, sizeof(hist));
    memset(labels, 0, (size_t)count * sizeof(int));
    for (index = 0; index < count; index++) {
        if (!inside_mask(domain, index))
            continue;
        hist[gray[index]] += 1.0;
        samples++;
    }
    if (samples == 0) {
        *segments = 0;
        return 0;
    }
    kept = find_valleys(hist, 256, sigma, lows, highs);
    if (kept < 0)
        return -1;
    for (band = 0; band < kept; band++) {
        int label = band + 1;
        for (index = 0; index < count; index++) {
            if (!inside_mask(domain, index))
                continue;
            if (gray[index] >= lows[band] && gray[index] <= highs[band])
                labels[index] = label;
        }
    }
    *segments = kept;
    return 0;
}

/* 用區域直方圖的兩個波峰決定字元門檻。找不到第二峰時改用 Otsu，較小的那一側留下。 */

int vizion_char_threshold(
    const uint8_t* gray, const uint8_t* histo, const uint8_t* domain, uint8_t* mask,
    int count, double sigma, double percent, int* limit)
{
    double hist[256];
    double smooth[256];
    double* kernel;
    int radius;
    int size;
    int index;
    int peak = 0;
    int secondary = -1;
    int level;
    int dark_side;
    double best = 0.0;
    double floor_value;
    int samples = 0;
    if (!gray || !histo || !mask || !limit || count < 1 || !(sigma > 0.0) || percent < 0.0 || percent > 100.0)
        return -1;
    memset(hist, 0, sizeof(hist));
    for (index = 0; index < count; index++) {
        if (histo[index] == 0)
            continue;
        hist[gray[index]] += 1.0;
        samples++;
    }
    if (samples == 0)
        return -1;
    radius = round_even(sigma * 3.0);
    if (radius < 1)
        radius = 1;
    size = radius * 2 + 1;
    kernel = (double*)malloc((size_t)size * sizeof(double));
    if (!kernel)
        return -1;
    gauss_kernel(size, sigma, kernel);
    convolve_same(hist, 256, kernel, size, smooth);
    free(kernel);
    for (index = 1; index < 256; index++) {
        if (smooth[index] > smooth[peak])
            peak = index;
    }
    floor_value = smooth[peak] * 0.01;
    for (index = 1; index < 255; index++) {
        int distance = index - peak;
        if (distance < 0)
            distance = -distance;
        if (distance < 8)
            continue;
        if (smooth[index] > smooth[index - 1] && smooth[index] > smooth[index + 1]
            && smooth[index] > best && smooth[index] > floor_value) {
            best = smooth[index];
            secondary = index;
        }
    }
    if (secondary < 0) {
        int dark = 0;
        level = otsu_level(gray, count);
        for (index = 0; index < count; index++) {
            if (gray[index] <= level)
                dark++;
        }
        dark_side = dark <= count - dark;
    } else {
        int low = peak < secondary ? peak : secondary;
        int high = peak < secondary ? secondary : peak;
        int valley = low;
        for (index = low + 1; index <= high; index++) {
            if (smooth[index] < smooth[valley])
                valley = index;
        }
        level = round_even((double)valley + ((double)peak - (double)valley) * (100.0 - percent) / 100.0);
        if (level < 0)
            level = 0;
        if (level > 255)
            level = 255;
        dark_side = secondary < peak;
    }
    for (index = 0; index < count; index++) {
        int keep = dark_side ? gray[index] <= level : gray[index] >= level;
        if (!inside_mask(domain, index))
            keep = 0;
        mask[index] = keep ? 255 : 0;
    }
    *limit = level;
    return 0;
}

/* 自動二值化。method：0 Otsu、1 平滑直方圖、2 三角形、3 局部平均、4 局部高斯。dark 為 1 時反相。 */

int vizion_binary_threshold(
    const uint8_t* gray, const uint8_t* domain, uint8_t* mask, int width, int height,
    int method, int dark, int* used)
{
    int count;
    int index;
    int level;
    uint8_t* local = NULL;
    if (!gray || !mask || !used || width < 1 || height < 1 || method < 0 || method > 4 || (dark != 0 && dark != 1))
        return -1;
    count = width * height;
    if (method <= 2) {
        if (method == 0)
            level = otsu_level(gray, count);
        else if (method == 1)
            level = smooth_histo_level(gray, count);
        else
            level = triangle_level(gray, count);
        for (index = 0; index < count; index++)
            mask[index] = gray[index] > level ? 255 : 0;
        *used = level;
    } else {
        local = (uint8_t*)malloc((size_t)count);
        if (!local)
            return -1;
        if ((method == 3 ? box_mean_u8(gray, local, width, height, 31)
                         : gauss_mean_u8(gray, local, width, height, 31)) != 0) {
            free(local);
            return -1;
        }
        for (index = 0; index < count; index++)
            mask[index] = (int)gray[index] + 5 > (int)local[index] ? 255 : 0;
        free(local);
        *used = 5;
    }
    apply_domain(mask, domain, count, dark);
    return 0;
}

static uint8_t saturated_diff(uint8_t left, uint8_t right)
{
    return left > right ? (uint8_t)(left - right) : 0;
}

/* 橢圓頂帽或黑帽找小點。kind：0 亮、1 暗、2 兩者。diameter 是 3 到 31 的奇數。 */

int vizion_dots_image(
    const uint8_t* gray, const uint8_t* domain, uint8_t* mask, int width, int height,
    int diameter, int kind, double shift)
{
    int count;
    int index;
    int radius;
    uint8_t* eroded;
    uint8_t* opened;
    if (!gray || !mask || width < 1 || height < 1 || diameter < 3 || diameter > 31 || (diameter & 1) == 0
        || kind < 0 || kind > 2 || shift < 0.0)
        return -1;
    count = width * height;
    radius = diameter / 2;
    eroded = (uint8_t*)malloc((size_t)count);
    opened = (uint8_t*)malloc((size_t)count);
    if (!eroded || !opened) {
        free(eroded);
        free(opened);
        return -1;
    }
    memset(mask, 0, (size_t)count);
    if (kind == 0 || kind == 2) {
        if (vizion_gray_ellipse(gray, eroded, width, height, radius, 0) != 0
            || vizion_gray_ellipse(eroded, opened, width, height, radius, 1) != 0) {
            free(eroded);
            free(opened);
            return -1;
        }
        for (index = 0; index < count; index++) {
            if ((double)saturated_diff(gray[index], opened[index]) >= shift)
                mask[index] = 255;
        }
    }
    if (kind == 1 || kind == 2) {
        if (vizion_gray_ellipse(gray, eroded, width, height, radius, 1) != 0
            || vizion_gray_ellipse(eroded, opened, width, height, radius, 0) != 0) {
            free(eroded);
            free(opened);
            return -1;
        }
        for (index = 0; index < count; index++) {
            if ((double)saturated_diff(opened[index], gray[index]) >= shift)
                mask[index] = 255;
        }
    }
    for (index = 0; index < count; index++) {
        if (!inside_mask(domain, index))
            mask[index] = 0;
    }
    free(eroded);
    free(opened);
    return 0;
}

int vizion_segment_more_selftest(void)
{
    uint8_t line[6] = {0, 0, 0, 255, 255, 255};
    uint8_t grown[3] = {10, 12, 40};
    uint8_t mask[6];
    int labels[6];
    int rows[1] = {0};
    int columns[1] = {0};
    int count = 0;
    int used = 0;
    int lows[8];
    int highs[8];
    double histo[5] = {5.0, 0.0, 0.0, 0.0, 5.0};
    int segments = 0;
    if (vizion_binary_threshold(line, NULL, mask, 6, 1, 0, 0, &used) != 0 || used != 0)
        return 1;
    if (mask[0] != 0 || mask[3] != 255)
        return 2;
    if (vizion_region_grow_mean(grown, NULL, labels, 3, 1, rows, columns, 1, 5.0, 1, &count) != 0 || count != 1)
        return 3;
    if (labels[0] != 1 || labels[1] != 1 || labels[2] != 0)
        return 4;
    if (vizion_region_grow_n(grown, labels, 3, 1, 1, 0.0, 5.0, 1, &count) != 0 || count != 2)
        return 5;
    if (labels[0] != 1 || labels[1] != 1 || labels[2] != 2)
        return 6;
    if (vizion_histo_to_thresh(histo, 5, 0.0, lows, highs, &segments) != 0 || segments != 2)
        return 7;
    if (lows[0] != 0 || highs[0] != 2 || lows[1] != 3 || highs[1] != 4)
        return 8;
    {
        uint8_t ramp[3] = {10, 12, 40};
        if (vizion_binary_threshold(ramp, NULL, mask, 3, 1, 2, 0, &used) != 0 || used != 12)
            return 9;
    }
    return 0;
}
