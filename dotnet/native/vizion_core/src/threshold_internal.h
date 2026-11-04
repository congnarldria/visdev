#ifndef VIZION_THRESHOLD_INTERNAL_H
#define VIZION_THRESHOLD_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

typedef int (*vizion_threshold_fn)(
    const uint8_t* gray, int width, int height, int min_gray, int max_gray,
    int* rows, int* columns, int* lengths, int capacity, int* written);

int vizion_threshold_scalar(
    const uint8_t* gray, int width, int height, int min_gray, int max_gray,
    int* rows, int* columns, int* lengths, int capacity, int* written);

int vizion_threshold_sse2(
    const uint8_t* gray, int width, int height, int min_gray, int max_gray,
    int* rows, int* columns, int* lengths, int capacity, int* written);

int vizion_threshold_avx2(
    const uint8_t* gray, int width, int height, int min_gray, int max_gray,
    int* rows, int* columns, int* lengths, int capacity, int* written);

static inline int vizion_flush_run(
    int row, int start, int end,
    int* rows, int* columns, int* lengths, int capacity, int* written, int counting)
{
    if (end < start)
        return 0;
    if (!counting) {
        if (*written >= capacity || !rows || !columns || !lengths)
            return -2;
        rows[*written] = row;
        columns[*written] = start;
        lengths[*written] = end - start + 1;
    }
    (*written)++;
    return 0;
}

/* bits 的第 0 位是 col_base。start 小於 0 表示這一段還沒開始。 */
static inline int vizion_emit_mask(
    uint32_t bits, int nbits, int row, int col_base, int* start,
    int* rows, int* columns, int* lengths, int capacity, int* written, int counting)
{
    uint32_t full = nbits >= 32 ? 0xFFFFFFFFu : (1u << nbits) - 1u;
    int bit;
    bits &= full;
    if (bits == full) {
        if (*start < 0)
            *start = col_base;
        return 0;
    }
    if (bits == 0) {
        if (*start >= 0) {
            int closed = vizion_flush_run(row, *start, col_base - 1, rows, columns, lengths, capacity, written, counting);
            *start = -1;
            return closed;
        }
        return 0;
    }
    for (bit = 0; bit < nbits; bit++) {
        int column = col_base + bit;
        if ((bits >> bit) & 1u) {
            if (*start < 0)
                *start = column;
        } else if (*start >= 0) {
            int closed = vizion_flush_run(row, *start, column - 1, rows, columns, lengths, capacity, written, counting);
            *start = -1;
            if (closed)
                return closed;
        }
    }
    return 0;
}

#endif
