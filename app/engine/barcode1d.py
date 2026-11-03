"""More 1D symbologies that print and read with one table each.

EAN-8 and UPC-E use the same digit patterns as EAN-13. Codabar, the two 2/5
codes, Code 93, and MSI each keep one pattern list for both directions.
"""

from __future__ import annotations

import numpy as np

L = {
    "0": "0001101", "1": "0011001", "2": "0010011", "3": "0111101", "4": "0100011",
    "5": "0110001", "6": "0101111", "7": "0111011", "8": "0110111", "9": "0001011",
}
G = {
    "0": "0100111", "1": "0110011", "2": "0011011", "3": "0100001", "4": "0011101",
    "5": "0111001", "6": "0000101", "7": "0010001", "8": "0001001", "9": "0010111",
}
R = {digit: "".join("1" if bit == "0" else "0" for bit in bits) for digit, bits in L.items()}
PARITY = {
    "0": "LLLLLL", "1": "LLGLGG", "2": "LLGGLG", "3": "LLGGGL", "4": "LGLLGG",
    "5": "LGGLLG", "6": "LGGGLL", "7": "LGLGLG", "8": "LGLGGL", "9": "LGGLGL",
}
_L_REV = {bits: digit for digit, bits in L.items()}
_G_REV = {bits: digit for digit, bits in G.items()}
_R_REV = {bits: digit for digit, bits in R.items()}

# 1 is wide. Seven elements: bar, space, bar, space, bar, space, bar.
_CODABAR = {
    "0": "0000011", "1": "0000110", "2": "0001001", "3": "1100000", "4": "0010010",
    "5": "1000010", "6": "0100001", "7": "0100100", "8": "0110000", "9": "1001000",
    "-": "0001100", "$": "0011000", ":": "1000101", "/": "1010001", ".": "1010100",
    "+": "0010101", "A": "0011010", "B": "0101001", "C": "0001011", "D": "0001110",
}
_CODABAR_REV = {pattern: char for char, pattern in _CODABAR.items()}

# Two wide bars or spaces out of five. Shared by interleaved and industrial 2/5.
_WIDE = {
    "0": "00110", "1": "10001", "2": "01001", "3": "11000", "4": "00101",
    "5": "10100", "6": "01100", "7": "00011", "8": "10010", "9": "01010",
}
_WIDE_REV = {pattern: digit for digit, pattern in _WIDE.items()}

_CODE93_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%"
_CODE93 = (
    "131112", "111213", "111312", "111411", "121113", "121212", "121311", "111114", "131211", "141111",
    "211113", "211212", "211311", "221112", "221211", "231111", "112113", "112212", "112311", "122112",
    "132111", "111123", "111222", "111321", "121122", "131121", "212112", "212211", "211122", "211221",
    "221121", "222111", "112122", "112221", "122121", "123111", "121131", "311112", "311211", "321111",
    "112131", "113121", "211131", "121221", "312111", "311121", "122211", "111141",
)
_CODE93_REV = {pattern: index for index, pattern in enumerate(_CODE93)}

_UPCE_PARITY = {
    "0": "EEEOOO", "1": "EEOEOO", "2": "EEOOEO", "3": "EEOOOE", "4": "EOEEOO",
    "5": "EOOEEO", "6": "EOOOEE", "7": "EOEOEO", "8": "EOEOOE", "9": "EOOEOE",
}
_UPCE_PARITY_REV = {pattern: digit for digit, pattern in _UPCE_PARITY.items()}

# 1 is black. Each digit is 12 modules. Start is 110, stop is 1001.
_MSI = {
    "0": "100100100100", "1": "100100100110", "2": "100100110100", "3": "100100110110",
    "4": "100110100100", "5": "100110100110", "6": "100110110100", "7": "100110110110",
    "8": "110100100100", "9": "110100100110",
}
_MSI_REV = {pattern: digit for digit, pattern in _MSI.items()}

if len(_CODE93) != 48 or len(set(_CODE93)) != 48:
    raise RuntimeError("Code 93 對照表不完整")
if any(sum(int(part) for part in pattern) != 9 for pattern in _CODE93):
    raise RuntimeError("Code 93 條寬加總不是 9")


def _runs(binary: np.ndarray):
    runs = []
    current = int(binary[0])
    length = 1
    for value in binary[1:]:
        value = int(value)
        if value == current:
            length += 1
        else:
            runs.append([current, length])
            current = value
            length = 1
    runs.append([current, length])
    return runs


def _span(binary: np.ndarray):
    index = np.flatnonzero(binary)
    if len(index) == 0:
        return 0, 1
    return int(index[0]), int(index[-1])


def _paint(chunks: list[tuple[int, int]], module: int = 2, height: int = 80, quiet: int = 16) -> np.ndarray:
    pixels = [255] * (quiet * module)
    for black, width in chunks:
        pixels.extend(([0] if black else [255]) * (width * module))
    pixels.extend([255] * (quiet * module))
    row = np.asarray(pixels, np.uint8)
    return np.tile(row, (height, 1))


def _paint_bits(bits: str, module: int = 2, height: int = 80, quiet: int = 16) -> np.ndarray:
    return _paint([(1 if bit == "1" else 0, 1) for bit in bits], module, height, quiet)


def _ean_check(digits: str) -> str:
    total = sum(int(char) * (1 if index % 2 == 0 else 3) for index, char in enumerate(digits))
    return str((10 - (total % 10)) % 10)


def _upc_check(digits: str) -> str:
    total = sum(int(char) * (3 if index % 2 == 0 else 1) for index, char in enumerate(digits))
    return str((10 - (total % 10)) % 10)


def _digits(payload: str, name: str) -> str:
    if not payload.isdigit():
        raise ValueError(f"{name} 只能是數字")
    return payload


def render_ean8(payload: str, module: int = 2, height: int = 80, quiet: int = 16) -> np.ndarray:
    digits = _digits(payload.strip(), "EAN-8")
    if len(digits) == 7:
        digits += _ean_check(digits)
    elif len(digits) == 8:
        if _ean_check(digits[:7]) != digits[7]:
            raise ValueError("EAN-8 檢查碼不對")
    else:
        raise ValueError("EAN-8 要 7 或 8 位數字")
    bits = ["101"]
    bits.extend(L[char] for char in digits[:4])
    bits.append("01010")
    bits.extend(R[char] for char in digits[4:])
    bits.append("101")
    return _paint_bits("".join(bits), module, height, quiet)


def _sample(binary: np.ndarray, offset: float, module: float, count: int):
    bits = []
    for index in range(count):
        x = int(round(offset + (index + 0.5) * module))
        if x < 0 or x >= len(binary):
            return None
        bits.append("1" if binary[x] else "0")
    return "".join(bits)


def decode_ean8_row(binary: np.ndarray):
    runs = _runs(binary)
    offset = 0
    for start in range(len(runs) - 2):
        black, white, black2 = runs[start : start + 3]
        if [black[0], white[0], black2[0]] != [1, 0, 1]:
            offset += black[1]
            continue
        widths = [black[1], white[1], black2[1]]
        module = sum(widths) / 3.0
        if module < 1 or max(widths) > min(widths) * 1.8:
            offset += black[1]
            continue
        bits = _sample(binary, offset, module, 67)
        if bits is None or bits[:3] != "101" or bits[31:36] != "01010" or bits[64:] != "101":
            offset += black[1]
            continue
        left = []
        right = []
        ok = True
        for index in range(4):
            chunk = bits[3 + index * 7 : 10 + index * 7]
            if chunk not in _L_REV:
                ok = False
                break
            left.append(_L_REV[chunk])
        for index in range(4):
            chunk = bits[36 + index * 7 : 43 + index * 7]
            if chunk not in _R_REV:
                ok = False
                break
            right.append(_R_REV[chunk])
        code = "".join(left + right)
        if not ok or _ean_check(code[:7]) != code[7]:
            offset += black[1]
            continue
        return code, offset, offset + 67 * module
    return None


def _upce_body(six: str) -> str:
    last = six[5]
    if last in "012":
        return "0" + six[0] + six[1] + last + "0000" + six[2] + six[3] + six[4]
    if last == "3":
        return "0" + six[:3] + "00000" + six[3] + six[4]
    if last == "4":
        return "0" + six[:4] + "00000" + six[4]
    return "0" + six[:5] + "0000" + last


def render_upce(payload: str, module: int = 2, height: int = 80, quiet: int = 16) -> np.ndarray:
    digits = _digits(payload.strip(), "UPC-E")
    if len(digits) != 6:
        raise ValueError("UPC-E 要 6 位數字")
    check = _upc_check(_upce_body(digits))
    bits = ["101"]
    for index, flag in enumerate(_UPCE_PARITY[check]):
        bits.append((G if flag == "E" else L)[digits[index]])
    bits.append("010101")
    return _paint_bits("".join(bits), module, height, quiet)


def decode_upce_row(binary: np.ndarray):
    runs = _runs(binary)
    offset = 0
    for start in range(len(runs) - 2):
        black, white, black2 = runs[start : start + 3]
        if [black[0], white[0], black2[0]] != [1, 0, 1]:
            offset += black[1]
            continue
        widths = [black[1], white[1], black2[1]]
        module = sum(widths) / 3.0
        if module < 1 or max(widths) > min(widths) * 1.8:
            offset += black[1]
            continue
        bits = _sample(binary, offset, module, 51)
        if bits is None or bits[:3] != "101" or bits[45:] != "010101":
            offset += black[1]
            continue
        digits = []
        flags = []
        ok = True
        for index in range(6):
            chunk = bits[3 + index * 7 : 10 + index * 7]
            if chunk in _L_REV:
                digits.append(_L_REV[chunk])
                flags.append("O")
            elif chunk in _G_REV:
                digits.append(_G_REV[chunk])
                flags.append("E")
            else:
                ok = False
                break
        pattern = "".join(flags)
        code = "".join(digits)
        if not ok or pattern not in _UPCE_PARITY_REV:
            offset += black[1]
            continue
        if _upc_check(_upce_body(code)) != _UPCE_PARITY_REV[pattern]:
            offset += black[1]
            continue
        return code, offset, offset + 51 * module
    return None


def _wide_gap(pattern: str, gap: bool) -> list[tuple[int, int]]:
    chunks = []
    for index, flag in enumerate(pattern):
        chunks.append((1 if index % 2 == 0 else 0, 3 if flag == "1" else 1))
    if gap:
        chunks.append((0, 1))
    return chunks


def render_codabar(payload: str, module: int = 2, height: int = 80, quiet: int = 16) -> np.ndarray:
    text = payload.strip().upper()
    if len(text) < 3 or text[0] not in "ABCD" or text[-1] not in "ABCD":
        raise ValueError("Codabar 要以 A、B、C、D 當開頭和結尾")
    if any(char not in _CODABAR for char in text):
        raise ValueError("Codabar 只接受數字和 - $ : / . +")
    chunks = []
    for index, char in enumerate(text):
        chunks.extend(_wide_gap(_CODABAR[char], index != len(text) - 1))
    return _paint(chunks, module, height, quiet)


def _codabar_at(runs, index: int):
    if index + 7 > len(runs) or runs[index][0] != 1:
        return None
    window = runs[index : index + 7]
    for offset, run in enumerate(window):
        if run[0] != (1 if offset % 2 == 0 else 0):
            return None
    widths = [run[1] for run in window]
    narrow = float(np.median(sorted(widths)[:4]))
    if narrow < 1:
        return None
    bits = "".join("1" if width > narrow * 1.8 else "0" for width in widths)
    char = _CODABAR_REV.get(bits)
    if char is None:
        return None
    return char, index + 7


def decode_codabar_row(binary: np.ndarray):
    runs = _runs(binary)
    for start in range(len(runs)):
        symbol = _codabar_at(runs, start)
        if symbol is None or symbol[0] not in "ABCD":
            continue
        chars = [symbol[0]]
        cursor = symbol[1]
        while cursor < len(runs):
            if runs[cursor][0] == 0:
                cursor += 1
            nxt = _codabar_at(runs, cursor)
            if nxt is None:
                break
            chars.append(nxt[0])
            cursor = nxt[1]
            if nxt[0] in "ABCD":
                x0 = sum(run[1] for run in runs[:start])
                x1 = sum(run[1] for run in runs[:cursor])
                return "".join(chars), x0, x1
    return None


def render_code2of5(payload: str, module: int = 2, height: int = 80, quiet: int = 16) -> np.ndarray:
    digits = _digits(payload.strip(), "2/5 Interleaved")
    if not digits:
        raise ValueError("2/5 Interleaved 不能是空的")
    if len(digits) % 2:
        digits = "0" + digits
    chunks = [(1, 1), (0, 1), (1, 1), (0, 1)]
    for index in range(0, len(digits), 2):
        bars = _WIDE[digits[index]]
        spaces = _WIDE[digits[index + 1]]
        for bar, space in zip(bars, spaces):
            chunks.append((1, 3 if bar == "1" else 1))
            chunks.append((0, 3 if space == "1" else 1))
    chunks.extend([(1, 3), (0, 1), (1, 1)])
    return _paint(chunks, module, height, quiet)


def _narrow_of(widths) -> float:
    return float(sum(widths) / len(widths))


def _is_wide(width: float, narrow: float) -> bool:
    return width > narrow * 1.8


def decode_code2of5_row(binary: np.ndarray):
    runs = _runs(binary)
    if runs and runs[0][0] == 0:
        runs = runs[1:]
    if runs and runs[-1][0] == 0:
        runs = runs[:-1]
    for start in range(len(runs) - 3):
        window = runs[start : start + 4]
        if [run[0] for run in window] != [1, 0, 1, 0]:
            continue
        widths = [run[1] for run in window]
        if any(_is_wide(width, _narrow_of(widths)) for width in widths):
            continue
        narrow = _narrow_of(widths)
        rest = runs[start + 4 :]
        if len(rest) < 3 or (len(rest) - 3) % 10:
            continue
        body = rest[:-3]
        stop = rest[-3:]
        if [run[0] for run in stop] != [1, 0, 1]:
            continue
        if not (_is_wide(stop[0][1], narrow) and not _is_wide(stop[1][1], narrow) and not _is_wide(stop[2][1], narrow)):
            continue
        digits = []
        ok = True
        for index in range(0, len(body), 10):
            pair = body[index : index + 10]
            bars = "".join("1" if _is_wide(pair[pos][1], narrow) else "0" for pos in range(0, 10, 2))
            spaces = "".join("1" if _is_wide(pair[pos][1], narrow) else "0" for pos in range(1, 10, 2))
            if bars not in _WIDE_REV or spaces not in _WIDE_REV:
                ok = False
                break
            digits.append(_WIDE_REV[bars] + _WIDE_REV[spaces])
        if ok and digits:
            x0 = sum(run[1] for run in runs[:start])
            x1 = sum(run[1] for run in runs[: start + 4 + len(rest)])
            return "".join(digits), x0, x1
    return None


def render_industrial2of5(payload: str, module: int = 2, height: int = 80, quiet: int = 16) -> np.ndarray:
    digits = _digits(payload.strip(), "2/5 Industrial")
    if not digits:
        raise ValueError("2/5 Industrial 不能是空的")
    chunks = []
    for _ in range(2):
        chunks.extend([(1, 1), (0, 1)])
    for char in digits:
        for flag in _WIDE[char]:
            chunks.extend([(1, 3 if flag == "1" else 1), (0, 1)])
    chunks.extend([(1, 3), (0, 1), (1, 1)])
    return _paint(chunks, module, height, quiet)


def decode_industrial2of5_row(binary: np.ndarray):
    runs = _runs(binary)
    if runs and runs[0][0] == 0:
        runs = runs[1:]
    bars = [length for color, length in runs if color == 1]
    if len(bars) < 4:
        return None
    narrow = (bars[0] + bars[1]) / 2.0
    if _is_wide(bars[0], narrow) or _is_wide(bars[1], narrow):
        return None
    rest = bars[2:]
    if len(rest) < 2 or (len(rest) - 2) % 5:
        return None
    if not (_is_wide(rest[-2], narrow) and not _is_wide(rest[-1], narrow)):
        return None
    digits = []
    for index in range(0, len(rest) - 2, 5):
        bits = "".join("1" if _is_wide(width, narrow) else "0" for width in rest[index : index + 5])
        if bits not in _WIDE_REV:
            return None
        digits.append(_WIDE_REV[bits])
    if not digits:
        return None
    return "".join(digits), *_span(binary)


def _code93_checks(values: list[int]):
    def weighted(seq, period: int) -> int:
        total = 0
        weight = 1
        for value in reversed(seq):
            total += value * weight
            weight = 1 if weight == period else weight + 1
        return total % 47

    check_c = weighted(values, 20)
    check_k = weighted(values + [check_c], 15)
    return check_c, check_k


def render_code93(payload: str, module: int = 2, height: int = 80, quiet: int = 16) -> np.ndarray:
    payload = payload.upper()
    if not payload or any(char not in _CODE93_CHARS for char in payload):
        raise ValueError("Code 93 只接受數字、英文和 - . 空白 $ / + %。小寫會轉成大寫。")
    values = [_CODE93_CHARS.index(char) for char in payload]
    check_c, check_k = _code93_checks(values)
    chunks = []
    for value in [47, *values, check_c, check_k, 47]:
        black = True
        for width in _CODE93[value]:
            chunks.append((1 if black else 0, int(width)))
            black = not black
    chunks.append((1, 1))
    return _paint(chunks, module, height, quiet)


def _match_widths(widths, pattern: str) -> bool:
    expected = np.asarray([int(char) for char in pattern], np.float64)
    actual = np.asarray(widths, np.float64)
    total = float(actual.sum())
    if total <= 0 or len(expected) != len(actual):
        return False
    scaled = actual * (float(expected.sum()) / total)
    return float(np.max(np.abs(scaled - expected))) < 0.65


def decode_code93_row(binary: np.ndarray):
    runs = _runs(binary)
    for start in range(len(runs)):
        if runs[start][0] != 1 or start + 6 > len(runs):
            continue
        if not _match_widths([runs[start + offset][1] for offset in range(6)], _CODE93[47]):
            continue
        symbols = [47]
        cursor = start + 6
        while cursor + 6 <= len(runs) and runs[cursor][0] == 1:
            widths = [runs[cursor + offset][1] for offset in range(6)]
            matched = next((value for value, pattern in enumerate(_CODE93) if _match_widths(widths, pattern)), None)
            if matched is None:
                break
            symbols.append(matched)
            cursor += 6
            if matched == 47:
                break
        if len(symbols) < 4 or symbols[-1] != 47 or symbols[0] != 47:
            continue
        payload = symbols[1:-1]
        if len(payload) < 2:
            continue
        data, check_c, check_k = payload[:-2], payload[-2], payload[-1]
        if _code93_checks(data) != (check_c, check_k):
            continue
        if any(value >= len(_CODE93_CHARS) for value in data):
            continue
        x0 = sum(run[1] for run in runs[:start])
        x1 = sum(run[1] for run in runs[:cursor])
        return "".join(_CODE93_CHARS[value] for value in data), x0, x1
    return None


def render_msi(payload: str, module: int = 2, height: int = 80, quiet: int = 16) -> np.ndarray:
    digits = _digits(payload.strip(), "MSI")
    if not digits:
        raise ValueError("MSI 不能是空的")
    bits = "110" + "".join(_MSI[char] for char in digits) + "1001"
    return _paint_bits(bits, module, height, quiet)


def _modules(binary: np.ndarray) -> str:
    runs = _runs(binary)
    if runs and runs[0][0] == 0:
        runs = runs[1:]
    if runs and runs[-1][0] == 0:
        runs = runs[:-1]
    if not runs:
        return ""
    narrow = min(run[1] for run in runs)
    bits = []
    for color, length in runs:
        count = max(1, int(round(length / narrow)))
        bits.append(("1" if color else "0") * count)
    return "".join(bits)


def decode_msi_row(binary: np.ndarray):
    bits = _modules(binary)
    if not bits.startswith("110") or not bits.endswith("1001"):
        return None
    body = bits[3:-4]
    if len(body) % 12:
        return None
    digits = []
    for index in range(0, len(body), 12):
        char = _MSI_REV.get(body[index : index + 12])
        if char is None:
            return None
        digits.append(char)
    if not digits:
        return None
    return "".join(digits), *_span(binary)


def _row(image: np.ndarray) -> np.ndarray:
    return (image[image.shape[0] // 2] < 128).astype(np.uint8)


def _check_roundtrip() -> None:
    cases = [
        (render_ean8, decode_ean8_row, "5512345", "55123453"),
        (render_upce, decode_upce_row, "123456", "123456"),
        (render_codabar, decode_codabar_row, "A1234B", "A1234B"),
        (render_code2of5, decode_code2of5_row, "123456", "123456"),
        (render_industrial2of5, decode_industrial2of5_row, "12345", "12345"),
        (render_code93, decode_code93_row, "VISDEV", "VISDEV"),
        (render_msi, decode_msi_row, "123456", "123456"),
    ]
    for render, decode, payload, expected in cases:
        found = decode(_row(render(payload)))
        if found is None or found[0] != expected:
            raise RuntimeError(f"{render.__name__} 讀回 {None if found is None else found[0]}，要的是 {expected}")


_check_roundtrip()
