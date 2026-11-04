#include "vizion_core.h"

static int segment_at(const uint8_t* gray, int width, int height, int row, int column, int outside)
{
    if (row < 0 || column < 0 || row >= height || column >= width)
        return outside;
    return gray[row * width + column];
}

/* 3×3 裡比周圍高，或比周圍暗。平坦的高台或暗區內部不算。 */

int vizion_local_peak(const uint8_t* gray, uint8_t* mask, int width, int height, double level, int minima)
{
    int row;
    int column;
    if (!gray || !mask || width < 1 || height < 1)
        return -1;
    for (row = 0; row < height; row++) {
        for (column = 0; column < width; column++) {
            int value = gray[row * width + column];
            int highest = 0;
            int lowest = 255;
            int dy;
            int dx;
            int keep;
            for (dy = -1; dy <= 1; dy++) {
                for (dx = -1; dx <= 1; dx++) {
                    int high_sample = segment_at(gray, width, height, row + dy, column + dx, 0);
                    int low_sample = segment_at(gray, width, height, row + dy, column + dx, 255);
                    if (high_sample > highest)
                        highest = high_sample;
                    if (low_sample < lowest)
                        lowest = low_sample;
                }
            }
            if (!minima)
                keep = value == highest && value > lowest && (!(level > 0.0) || (double)value >= level);
            else
                keep = value == lowest && value < highest && (!(level < 255.0) || (double)value <= level);
            mask[row * width + column] = keep ? 255 : 0;
        }
    }
    return 0;
}

/* 灰階差超過門檻。mode：0 絕對差、1 比參考亮、2 比參考暗。 */

int vizion_check_diff(
    const uint8_t* image, const uint8_t* pattern, uint8_t* mask, int count, double limit, int mode)
{
    int index;
    if (!image || !pattern || !mask || count < 1 || limit < 0.0 || mode < 0 || mode > 2)
        return -1;
    for (index = 0; index < count; index++) {
        int delta = (int)image[index] - (int)pattern[index];
        int keep;
        if (mode == 1)
            keep = (double)delta > limit;
        else if (mode == 2)
            keep = (double)(-delta) > limit;
        else
            keep = (double)(delta < 0 ? -delta : delta) > limit;
        mask[index] = keep ? 255 : 0;
    }
    return 0;
}

/* 灰階 128 當作 0。相鄰一正一負時，留下離 0 較近的那一側。 */

int vizion_zero_cross(const uint8_t* gray, uint8_t* mask, int width, int height)
{
    int row;
    int column;
    if (!gray || !mask || width < 1 || height < 1)
        return -1;
    for (row = 0; row < height; row++) {
        for (column = 0; column < width; column++)
            mask[row * width + column] = 0;
    }
    for (row = 0; row < height; row++) {
        for (column = 0; column < width - 1; column++) {
            int left = (int)gray[row * width + column] - 128;
            int right = (int)gray[row * width + column + 1] - 128;
            int change = (left * right < 0) || ((left == 0) ^ (right == 0));
            int left_abs = left < 0 ? -left : left;
            int right_abs = right < 0 ? -right : right;
            if (change && left_abs <= right_abs)
                mask[row * width + column] = 255;
            if (change && right_abs <= left_abs)
                mask[row * width + column + 1] = 255;
        }
    }
    for (row = 0; row < height - 1; row++) {
        for (column = 0; column < width; column++) {
            int up = (int)gray[row * width + column] - 128;
            int down = (int)gray[(row + 1) * width + column] - 128;
            int change = (up * down < 0) || ((up == 0) ^ (down == 0));
            int up_abs = up < 0 ? -up : up;
            int down_abs = down < 0 ? -down : down;
            if (change && up_abs <= down_abs)
                mask[row * width + column] = 255;
            if (change && down_abs <= up_abs)
                mask[(row + 1) * width + column] = 255;
        }
    }
    return 0;
}

/* 同一列上比 MinSize 短的連續亮段丟掉。 */

int vizion_fast_threshold(
    const uint8_t* gray, uint8_t* mask, int width, int height, int low, int high, int minimum)
{
    int row;
    int column;
    if (!gray || !mask || width < 1 || height < 1 || low > high || minimum < 1)
        return -1;
    for (row = 0; row < height; row++) {
        column = 0;
        while (column < width) {
            int value = gray[row * width + column];
            int start;
            int length;
            if (value < low || value > high) {
                mask[row * width + column] = 0;
                column++;
                continue;
            }
            start = column;
            while (column < width) {
                int sample = gray[row * width + column];
                if (sample < low || sample > high)
                    break;
                column++;
            }
            length = column - start;
            while (start < column) {
                mask[row * width + start] = length >= minimum ? 255 : 0;
                start++;
            }
        }
    }
    return 0;
}

int vizion_segment_selftest(void)
{
    uint8_t flat[9];
    uint8_t step[2] = {0, 255};
    uint8_t line[5] = {10, 200, 200, 10, 220};
    uint8_t mask[9];
    int index;
    for (index = 0; index < 9; index++)
        flat[index] = 40;
    if (vizion_local_peak(flat, mask, 3, 3, 0.0, 0) != 0)
        return 1;
    for (index = 0; index < 9; index++) {
        if (mask[index] != 0)
            return 2;
    }
    if (vizion_zero_cross(step, mask, 2, 1) != 0)
        return 3;
    if (mask[0] != 0 || mask[1] != 255)
        return 4;
    if (vizion_fast_threshold(line, mask, 5, 1, 200, 255, 2) != 0)
        return 5;
    if (mask[0] != 0 || mask[1] != 255 || mask[2] != 255 || mask[3] != 0 || mask[4] != 0)
        return 6;
    return 0;
}
