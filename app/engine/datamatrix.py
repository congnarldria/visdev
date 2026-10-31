"""ECC200 Data Matrix, written here.

Square symbols from 10×10 through 26×26. ASCII data, Reed-Solomon over GF(256)
with polynomial 0x12D and first root α^1, then the diagonal Utah placement used
by ECC200. The finder is the solid L on the left and bottom and the clock on
the top and right. Encode and decode share that placement, so a printed symbol
reads back as the same text.
"""

from __future__ import annotations

import numpy as np

# (symbol modules, data codewords, error-correction codewords). One data region.
_SIZES = (
    (10, 3, 5),
    (12, 5, 7),
    (14, 8, 10),
    (16, 12, 12),
    (18, 18, 14),
    (20, 22, 18),
    (22, 30, 20),
    (24, 36, 24),
    (26, 44, 28),
)

_EXP = [0] * 512
_LOG = [0] * 256


def _build_field() -> None:
    value = 1
    for index in range(255):
        _EXP[index] = value
        _LOG[value] = index
        value <<= 1
        if value & 0x100:
            value ^= 0x12D
    for index in range(255, 512):
        _EXP[index] = _EXP[index - 255]


_build_field()


def _mul(left: int, right: int) -> int:
    if left == 0 or right == 0:
        return 0
    return _EXP[_LOG[left] + _LOG[right]]


def _generator(length: int) -> list[int]:
    poly = [1]
    for index in range(length):
        root = _EXP[index + 1]
        grown = [0] * (len(poly) + 1)
        for coef_index, coef in enumerate(poly):
            grown[coef_index] ^= coef
            grown[coef_index + 1] ^= _mul(coef, root)
        poly = grown
    return poly


def _rs_encode(data: list[int], ec_len: int) -> list[int]:
    generator = _generator(ec_len)
    message = list(data) + [0] * ec_len
    for index in range(len(data)):
        coef = message[index]
        if coef == 0:
            continue
        for gen_index in range(1, len(generator)):
            message[index + gen_index] ^= _mul(generator[gen_index], coef)
    return list(data) + message[-ec_len:]


def _syndromes(code: list[int], ec_len: int) -> bool:
    for index in range(ec_len):
        total = 0
        root = _EXP[index + 1]
        for coef in code:
            total = _mul(total, root) ^ coef
        if total:
            return False
    return True


def _ascii_words(payload: str, data_len: int) -> list[int]:
    try:
        raw = payload.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("Data Matrix 這版只寫 ASCII") from exc
    words: list[int] = []
    index = 0
    while index < len(raw):
        if index + 1 < len(raw) and 48 <= raw[index] <= 57 and 48 <= raw[index + 1] <= 57:
            words.append(130 + (raw[index] - 48) * 10 + (raw[index + 1] - 48))
            index += 2
        else:
            words.append(raw[index] + 1)
            index += 1
    if len(words) > data_len:
        raise ValueError("這段文字放不進 26×26 的 Data Matrix")
    if len(words) < data_len:
        words.append(129)
    while len(words) < data_len:
        position = len(words) + 1
        pad = 129 + ((149 * position) % 253) + 1
        if pad > 254:
            pad -= 254
        words.append(pad)
    return words


def _ascii_text(words: list[int]) -> str:
    chars: list[str] = []
    for word in words:
        if word == 129:
            break
        if 1 <= word <= 128:
            chars.append(chr(word - 1))
        elif 130 <= word <= 229:
            digits = word - 130
            chars.append(f"{digits:02d}")
        else:
            break
    return "".join(chars)


def _choose(payload: str) -> tuple[int, list[int]]:
    for symbol, data_len, ec_len in _SIZES:
        try:
            words = _ascii_words(payload, data_len)
        except ValueError as exc:
            if "放不進" in str(exc):
                continue
            raise
        return symbol, _rs_encode(words, ec_len)
    raise ValueError("這段文字放不進 26×26 的 Data Matrix")


def _placement(rows: int, cols: int, codewords: list[int]) -> list[list[int | None]]:
    bits = []
    for word in codewords:
        for shift in range(7, -1, -1):
            bits.append((word >> shift) & 1)
    grid: list[list[int | None]] = [[None] * cols for _ in range(rows)]

    def module(row: int, col: int, pos: int, bit: int) -> None:
        if row < 0:
            row += rows
            col += 4 - ((rows + 4) % 8)
        if col < 0:
            col += cols
            row += 4 - ((cols + 4) % 8)
        grid[row][col] = bits[pos * 8 + bit]

    def utah(row: int, col: int, pos: int) -> None:
        module(row - 2, col - 2, pos, 0)
        module(row - 2, col - 1, pos, 1)
        module(row - 1, col - 2, pos, 2)
        module(row - 1, col - 1, pos, 3)
        module(row - 1, col, pos, 4)
        module(row, col - 2, pos, 5)
        module(row, col - 1, pos, 6)
        module(row, col, pos, 7)

    def corner1(pos: int) -> None:
        module(rows - 1, 0, pos, 0)
        module(rows - 1, 1, pos, 1)
        module(rows - 1, 2, pos, 2)
        module(0, cols - 2, pos, 3)
        module(0, cols - 1, pos, 4)
        module(1, cols - 1, pos, 5)
        module(2, cols - 1, pos, 6)
        module(3, cols - 1, pos, 7)

    def corner2(pos: int) -> None:
        module(rows - 3, 0, pos, 0)
        module(rows - 2, 0, pos, 1)
        module(rows - 1, 0, pos, 2)
        module(0, cols - 4, pos, 3)
        module(0, cols - 3, pos, 4)
        module(0, cols - 2, pos, 5)
        module(0, cols - 1, pos, 6)
        module(1, cols - 1, pos, 7)

    def corner3(pos: int) -> None:
        module(rows - 3, 0, pos, 0)
        module(rows - 2, 0, pos, 1)
        module(rows - 1, 0, pos, 2)
        module(0, cols - 2, pos, 3)
        module(0, cols - 1, pos, 4)
        module(1, cols - 1, pos, 5)
        module(2, cols - 1, pos, 6)
        module(3, cols - 1, pos, 7)

    def corner4(pos: int) -> None:
        module(rows - 1, 0, pos, 0)
        module(rows - 1, cols - 1, pos, 1)
        module(0, cols - 3, pos, 2)
        module(0, cols - 2, pos, 3)
        module(0, cols - 1, pos, 4)
        module(1, cols - 3, pos, 5)
        module(1, cols - 2, pos, 6)
        module(1, cols - 1, pos, 7)

    pos = 0
    row = 4
    col = 0
    while row < rows or col < cols:
        if row == rows and col == 0:
            corner1(pos)
            pos += 1
        if row == rows - 2 and col == 0 and cols % 4 != 0:
            corner2(pos)
            pos += 1
        if row == rows - 2 and col == 0 and cols % 8 == 4:
            corner3(pos)
            pos += 1
        if row == rows + 4 and col == 2 and cols % 8 == 0:
            corner4(pos)
            pos += 1
        while row >= 0 and col < cols:
            if row < rows and col >= 0 and grid[row][col] is None:
                utah(row, col, pos)
                pos += 1
            row -= 2
            col += 2
        row += 1
        col += 3
        while row < rows and col >= 0:
            if row >= 0 and col < cols and grid[row][col] is None:
                utah(row, col, pos)
                pos += 1
            row += 2
            col -= 2
        row += 3
        col += 1
    if grid[rows - 1][cols - 1] is None:
        grid[rows - 1][cols - 1] = 1
        grid[rows - 2][cols - 2] = 1
    return grid


def _read_codewords(region: list[list[int]], count: int) -> list[int]:
    rows = len(region)
    cols = len(region[0])
    slots: list[list[int | None]] = [[None] * 8 for _ in range(count)]
    grid: list[list[tuple[int, int] | None]] = [[None] * cols for _ in range(rows)]
    bits = [0] * (count * 8)

    def module(row: int, col: int, pos: int, bit: int) -> None:
        if row < 0:
            row += rows
            col += 4 - ((rows + 4) % 8)
        if col < 0:
            col += cols
            row += 4 - ((cols + 4) % 8)
        grid[row][col] = (pos, bit)

    def utah(row: int, col: int, pos: int) -> None:
        module(row - 2, col - 2, pos, 0)
        module(row - 2, col - 1, pos, 1)
        module(row - 1, col - 2, pos, 2)
        module(row - 1, col - 1, pos, 3)
        module(row - 1, col, pos, 4)
        module(row, col - 2, pos, 5)
        module(row, col - 1, pos, 6)
        module(row, col, pos, 7)

    def corner1(pos: int) -> None:
        module(rows - 1, 0, pos, 0)
        module(rows - 1, 1, pos, 1)
        module(rows - 1, 2, pos, 2)
        module(0, cols - 2, pos, 3)
        module(0, cols - 1, pos, 4)
        module(1, cols - 1, pos, 5)
        module(2, cols - 1, pos, 6)
        module(3, cols - 1, pos, 7)

    def corner2(pos: int) -> None:
        module(rows - 3, 0, pos, 0)
        module(rows - 2, 0, pos, 1)
        module(rows - 1, 0, pos, 2)
        module(0, cols - 4, pos, 3)
        module(0, cols - 3, pos, 4)
        module(0, cols - 2, pos, 5)
        module(0, cols - 1, pos, 6)
        module(1, cols - 1, pos, 7)

    def corner3(pos: int) -> None:
        module(rows - 3, 0, pos, 0)
        module(rows - 2, 0, pos, 1)
        module(rows - 1, 0, pos, 2)
        module(0, cols - 2, pos, 3)
        module(0, cols - 1, pos, 4)
        module(1, cols - 1, pos, 5)
        module(2, cols - 1, pos, 6)
        module(3, cols - 1, pos, 7)

    def corner4(pos: int) -> None:
        module(rows - 1, 0, pos, 0)
        module(rows - 1, cols - 1, pos, 1)
        module(0, cols - 3, pos, 2)
        module(0, cols - 2, pos, 3)
        module(0, cols - 1, pos, 4)
        module(1, cols - 3, pos, 5)
        module(1, cols - 2, pos, 6)
        module(1, cols - 1, pos, 7)

    pos = 0
    row = 4
    col = 0
    while (row < rows or col < cols) and pos < count:
        if row == rows and col == 0:
            corner1(pos)
            pos += 1
        if row == rows - 2 and col == 0 and cols % 4 != 0:
            corner2(pos)
            pos += 1
        if row == rows - 2 and col == 0 and cols % 8 == 4:
            corner3(pos)
            pos += 1
        if row == rows + 4 and col == 2 and cols % 8 == 0:
            corner4(pos)
            pos += 1
        while row >= 0 and col < cols:
            if row < rows and col >= 0 and grid[row][col] is None and pos < count:
                utah(row, col, pos)
                pos += 1
            row -= 2
            col += 2
        row += 1
        col += 3
        while row < rows and col >= 0:
            if row >= 0 and col < cols and grid[row][col] is None and pos < count:
                utah(row, col, pos)
                pos += 1
            row += 2
            col -= 2
        row += 3
        col += 1
    words = [0] * count
    for yy, line in enumerate(grid):
        for xx, owner in enumerate(line):
            if owner is None:
                continue
            word_index, bit = owner
            if region[yy][xx]:
                words[word_index] |= 1 << (7 - bit)
            slots[word_index][bit] = 1
    if any(bit is None for word in slots for bit in word):
        raise ValueError("Data Matrix 的模組沒排滿")
    return words


def encode_modules(payload: str) -> np.ndarray:
    if payload == "":
        raise ValueError("Text 不能是空的")
    symbol, codewords = _choose(payload)
    region_size = symbol - 2
    data_len = next(item[1] for item in _SIZES if item[0] == symbol)
    ec_len = len(codewords) - data_len
    if not _syndromes(codewords, ec_len):
        raise ValueError("Data Matrix 的校正碼不對")
    placed = _placement(region_size, region_size, codewords)
    canvas = np.full((symbol, symbol), 255, np.uint8)
    for row in range(symbol):
        canvas[row, 0] = 0
        canvas[row, symbol - 1] = 0 if row % 2 else 255
    for col in range(symbol):
        canvas[symbol - 1, col] = 0
        canvas[0, col] = 0 if col % 2 == 0 else 255
    for row, line in enumerate(placed):
        for col, bit in enumerate(line):
            if bit:
                canvas[row + 1, col + 1] = 0
    return canvas


def _symbol_boxes(gray: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    dark = gray < 128
    height, width = dark.shape
    visited = np.zeros_like(dark, dtype=np.uint8)
    boxes = []
    for row in range(height):
        for col in range(width):
            if not dark[row, col] or visited[row, col]:
                continue
            stack = [(row, col)]
            visited[row, col] = 1
            top = bottom = row
            left = right = col
            count = 0
            while stack:
                yy, xx = stack.pop()
                count += 1
                top = min(top, yy)
                bottom = max(bottom, yy)
                left = min(left, xx)
                right = max(right, xx)
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny = yy + dy
                    nx = xx + dx
                    if 0 <= ny < height and 0 <= nx < width and dark[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = 1
                        stack.append((ny, nx))
            if count >= 20:
                boxes.append((top, bottom, left, right, count))
    return boxes


def _sample_symbol(gray: np.ndarray, top: int, bottom: int, left: int, right: int) -> list[list[int]] | None:
    module = 0
    x = left
    while x <= right and gray[top, x] < 128:
        module += 1
        x += 1
    span = right - left + 1
    if module < 1 or span < module * 8:
        return None
    size = int(round(span / module))
    if size not in {item[0] for item in _SIZES}:
        return None
    if abs((bottom - top + 1) - span) > module:
        return None
    modules = []
    for row in range(size):
        line = []
        yy = min(bottom, top + int((row + 0.5) * (bottom - top + 1) / size))
        for col in range(size):
            xx = min(right, left + int((col + 0.5) * span / size))
            line.append(1 if gray[yy, xx] < 128 else 0)
        modules.append(line)
    return modules


def decode_image(gray: np.ndarray) -> list[tuple[str, list[list[float]]]]:
    picture = np.asarray(gray)
    if picture.ndim == 3:
        picture = picture[:, :, 0]
    hits = []
    for top, bottom, left, right, _count in _symbol_boxes(picture):
        modules = _sample_symbol(picture, top, bottom, left, right)
        if modules is None:
            continue
        symbol = len(modules)
        spec = next(item for item in _SIZES if item[0] == symbol)
        region = [line[1:-1] for line in modules[1:-1]]
        try:
            words = _read_codewords(region, spec[1] + spec[2])
        except ValueError:
            continue
        if not _syndromes(words, spec[2]):
            continue
        text = _ascii_text(words[: spec[1]])
        if not text:
            continue
        quad = [
            [float(left), float(top)],
            [float(right), float(top)],
            [float(right), float(bottom)],
            [float(left), float(bottom)],
        ]
        hits.append((text, quad))
    return hits
