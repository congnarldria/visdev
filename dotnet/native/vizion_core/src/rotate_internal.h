#ifndef VIZION_ROTATE_INTERNAL_H
#define VIZION_ROTATE_INTERNAL_H

#include <stddef.h>

static int vizion_round_away(double value)
{
    if (value >= 0.0)
        return (int)(value + 0.5);
    return -(int)(0.5 - value);
}

static void vizion_rotate_write(const uint8_t* src, uint8_t* dst, int width, int height, int row, int column, double source_row, double source_column)
{
    int sample_row = vizion_round_away(source_row);
    int sample_column = vizion_round_away(source_column);
    if ((unsigned)sample_row >= (unsigned)height || (unsigned)sample_column >= (unsigned)width)
        return;
    dst[(size_t)row * (size_t)width + column] = src[(size_t)sample_row * (size_t)width + sample_column];
}

#endif
