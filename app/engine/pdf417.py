"""PDF417 symbols that this program can print and read back.

The bar patterns are every 4-bar, 4-space codeword that sums to 17 modules.
Each cluster is numbered in the same order here, so the encoder and the decoder
agree. Reed-Solomon is over the prime field modulo 929. Nothing is copied from
a vendor codeword table.
"""

from __future__ import annotations

import numpy as np

_MOD = 929
_PRIMITIVE = 3
_EC_LEVEL = 2
_EC_COUNT = {0: 2, 1: 4, 2: 8, 3: 16, 4: 32, 5: 64, 6: 128, 7: 256, 8: 512}
_ROW_MODULES = 3
_START = (8, 1, 1, 1, 1, 1, 1, 3)
_STOP = (7, 1, 1, 3, 1, 1, 1, 2, 1)
_ALPHA = {chr(ord("A") + i): i for i in range(26)}
_ALPHA[" "] = 26
_ALPHA_REV = {value: char for char, value in _ALPHA.items()}


def _patterns() -> dict[int, list[tuple[int, ...]]]:
    clusters: dict[int, list[tuple[int, ...]]] = {0: [], 3: [], 6: []}
    for bars in np.ndindex((6, 6, 6, 6, 6, 6, 6, 6)):
        widths = tuple(int(item) + 1 for item in bars)
        if sum(widths) != 17:
            continue
        cluster = (widths[0] - widths[2] + widths[4] - widths[6] + 9) % 9
        if cluster in clusters:
            clusters[cluster].append(widths)
    selected = {}
    for cluster, items in clusters.items():
        items.sort()
        if len(items) < 929:
            raise RuntimeError(f"PDF417 cluster {cluster} 只有 {len(items)} 個")
        selected[cluster] = items[:929]
    return selected


_CLUSTERS = _patterns()
_ENCODE = {cluster: patterns for cluster, patterns in _CLUSTERS.items()}
_DECODE = {
    cluster: {pattern: index for index, pattern in enumerate(patterns)}
    for cluster, patterns in _CLUSTERS.items()
}


def _generator(nsym: int) -> list[int]:
    gen = [1]
    for power in range(1, nsym + 1):
        root = pow(_PRIMITIVE, power, _MOD)
        grown = [0] * (len(gen) + 1)
        for index, coef in enumerate(gen):
            grown[index] = (grown[index] - coef * root) % _MOD
            grown[index + 1] = (grown[index + 1] + coef) % _MOD
        gen = grown
    return list(reversed(gen))


def _rs_encode(data: list[int], nsym: int) -> list[int]:
    gen = _generator(nsym)
    message = list(data) + [0] * nsym
    for index in range(len(data)):
        coef = message[index]
        if coef == 0:
            continue
        for offset in range(1, len(gen)):
            message[index + offset] = (message[index + offset] - coef * gen[offset]) % _MOD
    parity = [(-value) % _MOD for value in message[len(data) :]]
    return list(data) + parity


def _syndromes_clear(code: list[int], nsym: int) -> bool:
    for power in range(1, nsym + 1):
        root = pow(_PRIMITIVE, power, _MOD)
        total = 0
        for value in code:
            total = (total * root + value) % _MOD
        if total != 0:
            return False
    return True


def _text_codewords(payload: str) -> list[int]:
    if all(char in _ALPHA for char in payload):
        values = [_ALPHA[char] for char in payload]
        if len(values) % 2:
            values.append(29)
        pairs = [30 * values[index] + values[index + 1] for index in range(0, len(values), 2)]
        return [900, *pairs]
    try:
        raw = payload.encode("ascii")
    except UnicodeError as exc:
        raise ValueError("PDF417 這版只寫 ASCII") from exc
    if any(byte >= 900 for byte in raw):
        raise ValueError("PDF417 這版只寫 ASCII")
    return [901, *list(raw)]


def _layout(count: int) -> tuple[int, int]:
    needed = count + _EC_COUNT[_EC_LEVEL]
    best: tuple[int, int, int] | None = None
    for cols in range(1, 7):
        for rows in range(3, 31):
            slots = cols * rows
            if slots < needed or slots - _EC_COUNT[_EC_LEVEL] > 928:
                continue
            if best is None or slots < best[0]:
                best = (slots, rows, cols)
    if best is None:
        raise ValueError("這段文字放不進 PDF417")
    return best[1], best[2]


def _indicators(row: int, rows: int, cols: int) -> tuple[int, int]:
    base = 30 * (row // 3)
    if row % 3 == 0:
        return base + (rows - 1) // 3, base + cols - 1
    if row % 3 == 1:
        return base + _EC_LEVEL * 3 + (rows - 1) % 3, base + (rows - 1) // 3
    return base + cols - 1, base + _EC_LEVEL * 3 + (rows - 1) % 3


def _paint(runs: tuple[int, ...], black: bool, row: list[int]) -> bool:
    for width in runs:
        row.extend([0 if black else 255] * width)
        black = not black
    return black


def encode_modules(payload: str) -> np.ndarray:
    if payload == "":
        raise ValueError("Text 不能是空的")
    words = _text_codewords(payload)
    rows, cols = _layout(1 + len(words))
    data_slots = rows * cols - _EC_COUNT[_EC_LEVEL]
    data = [data_slots, *words]
    data.extend([900] * (data_slots - len(data)))
    stream = _rs_encode(data, _EC_COUNT[_EC_LEVEL])
    modules = []
    cursor = 0
    for row_index in range(rows):
        cluster = (row_index % 3) * 3
        left, right = _indicators(row_index, rows, cols)
        pieces = [left, *stream[cursor : cursor + cols], right]
        cursor += cols
        line: list[int] = []
        black = True
        black = _paint(_START, black, line)
        for value in pieces:
            black = _paint(_ENCODE[cluster][value], black, line)
        _paint(_STOP, black, line)
        modules.extend([line] * _ROW_MODULES)
    return np.asarray(modules, np.uint8)


def _runs(line: np.ndarray) -> list[tuple[int, int]]:
    if line.size == 0:
        return []
    runs = []
    current = int(line[0])
    length = 1
    for value in line[1:]:
        value = int(value)
        if value == current:
            length += 1
        else:
            runs.append((current, length))
            current = value
            length = 1
    runs.append((current, length))
    return runs


def _quantize(lengths: list[int], module: float) -> tuple[int, ...]:
    return tuple(max(1, int(round(length / module))) for length in lengths)


def _codewords(line: np.ndarray, module: float, cluster: int) -> list[int] | None:
    runs = _runs(line)
    if not runs or runs[0][0] != 1:
        return None
    if runs[-1][0] == 0:
        runs = runs[:-1]
    lengths = [length for _color, length in runs]
    modules = _quantize(lengths, module)
    if len(modules) < len(_START) + 8 + len(_STOP) or modules[: len(_START)] != _START:
        return None
    pos = len(_START)
    words = []
    while pos < len(modules):
        if modules[pos:] == _STOP:
            break
        if pos + 8 > len(modules):
            return None
        pattern = modules[pos : pos + 8]
        if sum(pattern) != 17:
            return None
        value = _DECODE[cluster].get(pattern)
        if value is None:
            return None
        words.append(value)
        pos += 8
    if modules[pos:] != _STOP or len(words) < 3:
        return None
    return words


def _text(body: list[int]) -> str:
    chars: list[str] = []
    index = 0
    while index < len(body):
        code = body[index]
        if code == 900:
            index += 1
            produced = False
            while index < len(body) and body[index] < 900:
                pair = body[index]
                high, low = divmod(pair, 30)
                if high not in _ALPHA_REV or (low not in _ALPHA_REV and low != 29):
                    return ""
                chars.append(_ALPHA_REV[high])
                if low != 29:
                    chars.append(_ALPHA_REV[low])
                produced = True
                index += 1
            if not produced:
                break
        elif code == 901:
            index += 1
            while index < len(body) and body[index] < 900:
                if body[index] > 255:
                    return ""
                chars.append(chr(body[index]))
                index += 1
        else:
            break
    return "".join(chars)


def _read_symbol(binary: np.ndarray, top: int, left: int) -> tuple[str, list[list[float]]] | None:
    height, width = binary.shape
    samples = []
    for y in range(0, height, max(1, height // 8)):
        runs = _runs(binary[y])
        if runs and runs[0][0] == 1 and runs[0][1] >= 8:
            samples.append(runs[0][1] / 8.0)
    if not samples:
        return None
    module = float(np.median(samples))
    if module < 0.5:
        return None
    row_pitch = _ROW_MODULES * module
    rows = int(round(height / row_pitch))
    if rows < 3 or rows > 30:
        return None
    grid = []
    for row_index in range(rows):
        y = min(height - 1, int((row_index + 0.5) * row_pitch))
        words = _codewords(binary[y], module, (row_index % 3) * 3)
        if words is None:
            return None
        grid.append(words)
    cols = len(grid[0]) - 2
    if cols < 1 or any(len(row) != cols + 2 for row in grid):
        return None
    level = None
    for row_index, words in enumerate(grid):
        if row_index % 3 == 1:
            base = 30 * (row_index // 3)
            raw = words[0] - base - (rows - 1) % 3
            if raw % 3 != 0:
                return None
            level = raw // 3
            break
    if level not in _EC_COUNT:
        return None
    for row_index, words in enumerate(grid):
        left_word, right_word = _indicators(row_index, rows, cols)
        if row_index % 3 == 1:
            left_word = 30 * (row_index // 3) + level * 3 + (rows - 1) % 3
        elif row_index % 3 == 2:
            right_word = 30 * (row_index // 3) + level * 3 + (rows - 1) % 3
        if words[0] != left_word or words[-1] != right_word:
            return None
    stream = [value for row in grid for value in row[1:-1]]
    nsym = _EC_COUNT[level]
    if len(stream) <= nsym:
        return None
    data_slots = len(stream) - nsym
    if stream[0] != data_slots or not _syndromes_clear(stream, nsym):
        return None
    text = _text(stream[1:data_slots])
    if not text:
        return None
    bottom = top + height - 1
    right = left + width - 1
    quad = [
        [float(left), float(top)],
        [float(right), float(top)],
        [float(right), float(bottom)],
        [float(left), float(bottom)],
    ]
    return text, quad


def decode_image(gray: np.ndarray) -> list[tuple[str, list[list[float]]]]:
    picture = np.asarray(gray)
    if picture.ndim == 3:
        picture = picture[:, :, 0]
    dark = picture < 128
    ys, xs = np.nonzero(dark)
    if len(ys) < 40:
        return []
    top, bottom = int(ys.min()), int(ys.max())
    left, right = int(xs.min()), int(xs.max())
    binary = dark[top : bottom + 1, left : right + 1].astype(np.uint8)
    hit = _read_symbol(binary, top, left)
    return [hit] if hit else []
