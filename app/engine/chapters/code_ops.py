"""Barcodes, QR codes, and text results.

1D codes are EAN-13, UPC-A, Code 39, and Code 128. The scanline reader is used
when OpenCV's barcode detector returns nothing. 2D codes are QR codes from
OpenCV. find_text stores the word boxes from the local OCR engine; it does not
read an MLP or OCR classifier file.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..objects import Image, Region, Result, XLD
from ..ocr import method_choices, normalize_method, read_text
from ..operators import P, expect_image, expect_region, operator, text

_CODE_TYPES = ["'auto'", "'EAN-13'", "'UPC-A'", "'Code 39'", "'Code 128'"]
_SYMBOLS = ["'QR Code'"]
_TEXT_MODE = ["'auto'", "'manual'"]
_TEXT_RESULT = ["'class'", "'confidence'", "'row'", "'column'"]

_L = {
    "0": "0001101", "1": "0011001", "2": "0010011", "3": "0111101", "4": "0100011",
    "5": "0110001", "6": "0101111", "7": "0111011", "8": "0110111", "9": "0001011",
}
_G = {
    "0": "0100111", "1": "0110011", "2": "0011011", "3": "0100001", "4": "0011101",
    "5": "0111001", "6": "0000101", "7": "0010001", "8": "0001001", "9": "0010111",
}
_R = {digit: "".join("1" if bit == "0" else "0" for bit in bits) for digit, bits in _L.items()}
_PARITY = {
    "0": "LLLLLL", "1": "LLGLGG", "2": "LLGGLG", "3": "LLGGGL", "4": "LGLLGG",
    "5": "LGGLLG", "6": "LGGGLL", "7": "LGLGLG", "8": "LGLGGL", "9": "LGGLGL",
}
_L_REV = {bits: digit for digit, bits in _L.items()}
_G_REV = {bits: digit for digit, bits in _G.items()}
_R_REV = {bits: digit for digit, bits in _R.items()}
_PARITY_REV = {pattern: digit for digit, pattern in _PARITY.items()}

# 1 means a wide element. Each symbol is bar, space, bar, space, bar, space, bar, space, bar.
_CODE39 = {
    "0": "000110100", "1": "100100001", "2": "001100001", "3": "101100000", "4": "000110001",
    "5": "100110000", "6": "001110000", "7": "000100101", "8": "100100100", "9": "001100100",
    "A": "100001001", "B": "001001001", "C": "101001000", "D": "000011001", "E": "100011000",
    "F": "001011000", "G": "000001101", "H": "100001100", "I": "001001100", "J": "000011100",
    "K": "100000011", "L": "001000011", "M": "101000010", "N": "000010011", "O": "100010010",
    "P": "001010010", "Q": "000000111", "R": "100000110", "S": "001000110", "T": "000010110",
    "U": "110000001", "V": "011000001", "W": "111000000", "X": "010010001", "Y": "110010000",
    "Z": "011010000", "-": "010000101", ".": "110000100", " ": "011000100", "$": "010101000",
    "/": "010100010", "+": "010001010", "%": "000101010", "*": "010010100",
}
_CODE39_REV = {pattern: char for char, pattern in _CODE39.items()}

# Six run widths, except the stop symbol which has seven. Values 103–105 start A/B/C.
_CODE128 = (
    "212222", "222122", "222221", "121223", "121322", "131222", "122213", "122312", "132212", "221213",
    "221312", "231212", "112232", "122132", "122231", "113222", "123122", "123221", "223211", "221132",
    "221231", "213212", "223112", "312131", "311222", "321122", "321221", "312212", "322112", "322211",
    "212123", "212321", "232121", "111323", "131123", "131321", "112313", "132113", "132311", "211313",
    "231113", "231311", "112133", "112331", "132131", "113123", "113321", "133121", "313121", "211331",
    "231131", "213113", "213311", "213131", "311123", "311321", "331121", "312113", "312311", "332111",
    "314111", "221411", "431111", "111224", "111422", "121124", "121421", "141122", "141221", "112214",
    "112412", "122114", "122411", "142112", "142211", "241211", "221114", "413111", "241112", "134111",
    "111242", "121142", "121241", "114212", "124112", "124211", "411212", "421112", "421211", "212141",
    "214121", "412121", "111143", "111341", "131141", "114113", "114311", "411113", "411311", "113141",
    "114131", "311141", "411131", "211412", "211214", "211232", "2331112",
)
if len(_CODE128) != 107 or len(set(_CODE128[:106])) != 106:
    raise RuntimeError("Code 128 對照表不完整")


@dataclass
class BarCodeModel:
    closed: bool = False

    def summary(self) -> str:
        return "條碼模型  已關閉" if self.closed else "條碼模型  EAN-13 / UPC-A / Code 39 / Code 128"


@dataclass
class DataCodeModel:
    symbol: str
    closed: bool = False

    def summary(self) -> str:
        state = "已關閉" if self.closed else self.symbol
        return f"二維碼模型  {state}"


@dataclass
class TextModel:
    method: str
    mode: str
    closed: bool = False

    def summary(self) -> str:
        return "文字模型  已關閉" if self.closed else f"文字模型  {self.method}"


@dataclass
class TextResult:
    width: int
    height: int
    words: list
    closed: bool = False

    def summary(self) -> str:
        return "文字結果  已關閉" if self.closed else f"文字結果  {len(self.words)} 個字塊"


def _token(value, name: str) -> str:
    return text(value, name).strip().strip("'\"")


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


def _decode_ean_row(binary: np.ndarray):
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
        bits = []
        for index in range(95):
            x = int(round(offset + (index + 0.5) * module))
            if x < 0 or x >= len(binary):
                bits = []
                break
            bits.append("1" if binary[x] else "0")
        if len(bits) != 95 or "".join(bits[:3]) != "101" or "".join(bits[45:50]) != "01010" or "".join(bits[92:]) != "101":
            offset += black[1]
            continue
        left = []
        flags = []
        ok = True
        for index in range(6):
            chunk = "".join(bits[3 + index * 7 : 10 + index * 7])
            if chunk in _L_REV:
                left.append(_L_REV[chunk])
                flags.append("L")
            elif chunk in _G_REV:
                left.append(_G_REV[chunk])
                flags.append("G")
            else:
                ok = False
                break
        right = []
        for index in range(6):
            chunk = "".join(bits[50 + index * 7 : 57 + index * 7])
            if chunk not in _R_REV:
                ok = False
                break
            right.append(_R_REV[chunk])
        pattern = "".join(flags)
        if not ok or pattern not in _PARITY_REV:
            offset += black[1]
            continue
        code = _PARITY_REV[pattern] + "".join(left) + "".join(right)
        nums = [int(char) for char in code]
        total = sum(nums[index] * (1 if index % 2 == 0 else 3) for index in range(12))
        if (10 - (total % 10)) % 10 != nums[12]:
            offset += black[1]
            continue
        return code, offset, offset + 95 * module
    return None


def _scan_ean(gray: np.ndarray):
    height, width = gray.shape
    found = {}
    for fraction in (0.35, 0.5, 0.65):
        row_index = min(height - 1, max(0, int(height * fraction)))
        row = gray[row_index]
        level = float(np.median(row))
        binary = row < level
        if int(binary.sum()) < 8:
            binary = row < 128
        decoded = _decode_ean_row(binary.astype(np.uint8))
        if decoded is None:
            continue
        code, x0, x1 = decoded
        band = found.setdefault(code, [row_index, row_index, x0, x1])
        band[0] = min(band[0], row_index)
        band[1] = max(band[1], row_index)
        band[2] = min(band[2], x0)
        band[3] = max(band[3], x1)
    hits = []
    for code, (top, bottom, x0, x1) in found.items():
        if top == bottom:
            top = max(0, top - height // 5)
            bottom = min(height - 1, bottom + height // 5)
        hits.append((code, top, bottom, x0, x1))
    return hits


def _bars(pattern: str, module: int, gap: bool) -> list[int]:
    pixels = []
    for index, flag in enumerate(pattern):
        width = (3 if flag == "1" else 1) * module
        pixels.extend(([0] if index % 2 == 0 else [255]) * width)
    if gap:
        pixels.extend([255] * module)
    return pixels


def render_code39(payload: str, module: int = 2, height: int = 120, quiet: int = 20) -> np.ndarray:
    text = payload.strip().upper()
    if not text or any(char not in _CODE39 or char == "*" for char in text):
        raise ValueError("Code 39 只接受數字、大寫英文和 - . 空白 $ / + %")
    pixels = [255] * (quiet * module)
    coded = f"*{text}*"
    for index, char in enumerate(coded):
        pixels.extend(_bars(_CODE39[char], module, index != len(coded) - 1))
    pixels.extend([255] * (quiet * module))
    row = np.asarray(pixels, np.uint8)
    return np.tile(row, (height, 1))


def render_code128(payload: str, module: int = 2, height: int = 120, quiet: int = 24) -> np.ndarray:
    if not payload or any(ord(char) < 32 or ord(char) > 126 for char in payload):
        raise ValueError("Code 128 這版用字元集 B，只接受可列印 ASCII")
    values = [104]
    values.extend(ord(char) - 32 for char in payload)
    checksum = values[0]
    for index, value in enumerate(values[1:], start=1):
        checksum += index * value
    values.append(checksum % 103)
    values.append(106)
    pixels = [255] * (quiet * module)
    for value in values:
        black = True
        for width in _CODE128[value]:
            pixels.extend(([0] if black else [255]) * (int(width) * module))
            black = not black
    pixels.extend([255] * (quiet * module))
    row = np.asarray(pixels, np.uint8)
    return np.tile(row, (height, 1))


def _code39_symbol(runs, index: int):
    if index + 9 > len(runs) or runs[index][0] != 1:
        return None
    window = runs[index : index + 9]
    for offset, run in enumerate(window):
        if run[0] != (1 if offset % 2 == 0 else 0):
            return None
    widths = np.asarray([run[1] for run in window], np.float64)
    order = np.argsort(widths)
    if widths[order[-3]] < widths[order[-4]] * 1.5:
        return None
    bits = ["0"] * 9
    for pos in order[-3:]:
        bits[int(pos)] = "1"
    char = _CODE39_REV.get("".join(bits))
    if char is None:
        return None
    return char, index + 9


def _decode_code39_row(binary: np.ndarray):
    runs = _runs(binary)
    for start in range(len(runs)):
        symbol = _code39_symbol(runs, start)
        if symbol is None or symbol[0] != "*":
            continue
        chars = []
        cursor = symbol[1]
        while cursor < len(runs):
            if runs[cursor][0] == 0:
                cursor += 1
            nxt = _code39_symbol(runs, cursor)
            if nxt is None:
                break
            cursor = nxt[1]
            if nxt[0] == "*":
                if not chars:
                    break
                x0 = sum(run[1] for run in runs[:start])
                x1 = sum(run[1] for run in runs[:cursor])
                return "".join(chars), x0, x1
            chars.append(nxt[0])
    return None


def _match_widths(widths, pattern: str) -> bool:
    expected = np.asarray([int(char) for char in pattern], np.float64)
    actual = np.asarray(widths, np.float64)
    total = float(actual.sum())
    if total <= 0 or len(expected) != len(actual):
        return False
    scaled = actual * (float(expected.sum()) / total)
    return float(np.max(np.abs(scaled - expected))) < 0.7


def _code128_text(start: int, values: list[int]):
    mode = {103: "A", 104: "B", 105: "C"}.get(start)
    if mode is None:
        return None
    chars = []
    for value in values:
        if mode == "B":
            if 0 <= value <= 94:
                chars.append(chr(value + 32))
            elif value == 99:
                mode = "C"
            else:
                return None
        elif mode == "C":
            if 0 <= value <= 99:
                chars.append(f"{value:02d}")
            elif value == 100:
                mode = "B"
            else:
                return None
        else:
            return None
    return "".join(chars)


def _decode_code128_row(binary: np.ndarray):
    runs = _runs(binary)
    for start in range(len(runs)):
        if runs[start][0] != 1 or start + 6 > len(runs):
            continue
        widths = [runs[start + offset][1] for offset in range(6)]
        starter = next((value for value in (103, 104, 105) if _match_widths(widths, _CODE128[value])), None)
        if starter is None:
            continue
        symbols = [starter]
        cursor = start + 6
        while cursor < len(runs) and runs[cursor][0] == 1:
            matched = None
            if cursor + 6 <= len(runs):
                six = [runs[cursor + offset][1] for offset in range(6)]
                for value in range(106):
                    if _match_widths(six, _CODE128[value]):
                        matched = value
                        break
            if matched is not None:
                symbols.append(matched)
                cursor += 6
                continue
            if cursor + 7 <= len(runs) and _match_widths([runs[cursor + offset][1] for offset in range(7)], _CODE128[106]):
                symbols.append(106)
                cursor += 7
                break
            break
        if len(symbols) < 4 or symbols[-1] != 106:
            continue
        payload = symbols[:-2]
        checksum = symbols[-2]
        total = payload[0]
        for index, value in enumerate(payload[1:], start=1):
            total += index * value
        if total % 103 != checksum:
            continue
        text = _code128_text(payload[0], payload[1:])
        if not text:
            continue
        x0 = sum(run[1] for run in runs[:start])
        x1 = sum(run[1] for run in runs[:cursor])
        return text, x0, x1
    return None


def _scan_decoder(gray: np.ndarray, decoder):
    height = gray.shape[0]
    found = {}
    for fraction in (0.35, 0.5, 0.65):
        row_index = min(height - 1, max(0, int(height * fraction)))
        row = gray[row_index]
        level = float(np.median(row))
        binary = (row < level).astype(np.uint8)
        if int(binary.sum()) < 8:
            binary = (row < 128).astype(np.uint8)
        decoded = decoder(binary)
        if decoded is None:
            continue
        code, x0, x1 = decoded
        band = found.setdefault(code, [row_index, row_index, x0, x1])
        band[0] = min(band[0], row_index)
        band[1] = max(band[1], row_index)
        band[2] = min(band[2], x0)
        band[3] = max(band[3], x1)
    hits = []
    for code, (top, bottom, x0, x1) in found.items():
        if top == bottom:
            top = max(0, top - height // 5)
            bottom = min(height - 1, bottom + height // 5)
        hits.append((code, top, bottom, x0, x1))
    return hits


def _opencv_barcodes(gray: np.ndarray):
    if not hasattr(cv2, "barcode"):
        return []
    try:
        ok, decoded, types, points = cv2.barcode.BarcodeDetector().detectAndDecodeWithType(gray)
    except cv2.error:
        return []
    if not ok:
        return []
    texts = list(decoded) if isinstance(decoded, (list, tuple)) else [str(decoded)]
    kind_names = list(types) if isinstance(types, (list, tuple)) else [str(types)]
    corners = np.asarray(points) if points is not None else None
    hits = []
    for index, raw in enumerate(texts):
        code = str(raw).strip()
        if not code:
            continue
        kind = str(kind_names[index]).upper() if index < len(kind_names) else ""
        box = None
        if corners is not None and len(corners) > index:
            pts = np.asarray(corners[index], np.float64).reshape(-1, 2)
            box = (float(pts[:, 1].min()), float(pts[:, 1].max()), float(pts[:, 0].min()), float(pts[:, 0].max()), kind)
        hits.append((code, kind, box))
    return hits


def _region(width: int, height: int, boxes: list) -> Region:
    masks = []
    for top, bottom, left, right in boxes:
        mask = np.zeros((height, width), np.uint8)
        r0 = max(0, min(height - 1, int(round(top))))
        r1 = max(0, min(height - 1, int(round(bottom))))
        c0 = max(0, min(width - 1, int(round(left))))
        c1 = max(0, min(width - 1, int(round(right))))
        if r1 < r0:
            r0, r1 = r1, r0
        if c1 < c0:
            c0, c1 = c1, c0
        mask[r0 : r1 + 1, c0 : c1 + 1] = 255
        masks.append(mask)
    return Region(width, height, masks, multi_color=len(masks) > 1)


def _select_code(code: str, kind: str, requested: str):
    digits = "".join(char for char in code if char.isdigit())
    if requested == "auto":
        return digits or code
    if requested == "EAN-13" and len(digits) == 13:
        return digits
    if requested == "UPC-A" and len(digits) == 13 and digits.startswith("0"):
        return digits[1:]
    if requested == "UPC-A" and len(digits) == 12:
        return digits
    if kind in ("EAN_13", "EAN13") and requested in ("auto", "EAN-13"):
        return digits
    if requested == "Code 39" and code.strip():
        return code.strip()
    if requested == "Code 128" and code.strip():
        return code.strip()
    return None


@operator(
    "條碼",
    [],
    [],
    [
        P("GenParamName", "這版不使用，留空字串", "''"),
        P("GenParamValue", "這版不使用，留空字串", "''"),
    ],
    [P("BarCodeHandle", "條碼模型")],
    "建立一維條碼模型",
    "能讀 EAN-13、UPC-A、Code 39 和 Code 128。Data Matrix 和 PDF417 這版不解。",
    "create_bar_code_model (BarCodeHandle)",
)
def create_bar_code_model(_name, _value, *, ctx):
    return Result([], [BarCodeModel()])


@operator(
    "條碼",
    [P("Image", "含條碼的影像")],
    [P("SymbolRegions", "條碼範圍")],
    [
        P("BarCodeHandle", "條碼模型"),
        P("CodeType", "auto、EAN-13、UPC-A、Code 39 或 Code 128", "'auto'", _CODE_TYPES),
    ],
    [P("DecodedDataStrings", "讀到的字串")],
    "尋找一維條碼",
    "EAN-13 和 UPC-A 先問 OpenCV，沒有結果再用掃描線。Code 39 與 Code 128 用掃描線，Code 128 這版依字元集 B 產生、也能讀字元集 C 的數字對。"
    "UPC-A 是開頭為 0 的 EAN-13，去掉第一個 0。auto 先找 EAN，沒有再找 Code 128 和 Code 39。找不到時區域是空的，字串是空 tuple。",
    "find_bar_code (Image, SymbolRegions, BarCodeHandle, 'auto', DecodedDataStrings)",
)
def find_bar_code(image, handle, code_type, *, ctx):
    picture = expect_image(image, "Image")
    if not isinstance(handle, BarCodeModel):
        raise TypeError("BarCodeHandle 必須是 create_bar_code_model 建立的模型")
    if handle.closed:
        raise RuntimeError("條碼模型已經 clear_bar_code_model")
    requested = _token(code_type, "CodeType").upper()
    aliases = {
        "AUTO": "auto", "EAN-13": "EAN-13", "EAN13": "EAN-13", "UPC-A": "UPC-A", "UPCA": "UPC-A",
        "CODE 39": "Code 39", "CODE39": "Code 39", "CODE 128": "Code 128", "CODE128": "Code 128",
    }
    if requested not in aliases:
        raise ValueError("CodeType 這版只接受 auto、EAN-13、UPC-A、Code 39 或 Code 128")
    requested = aliases[requested]
    gray = picture.gray()
    codes = []
    boxes = []
    seen = set()
    opencv_hits = [] if requested in ("Code 39", "Code 128") else _opencv_barcodes(gray)
    for code, kind, box in opencv_hits:
        chosen = _select_code(code, kind.upper(), requested)
        if not chosen or chosen in seen or box is None:
            continue
        seen.add(chosen)
        codes.append(chosen)
        boxes.append(box[:4])
    if requested in ("auto", "EAN-13", "UPC-A") and not codes:
        for code, top, bottom, left, right in _scan_ean(gray):
            chosen = _select_code(code, "", requested)
            if not chosen or chosen in seen:
                continue
            seen.add(chosen)
            codes.append(chosen)
            boxes.append((top, bottom, left, right))
    if requested == "Code 128" or (requested == "auto" and not codes):
        for code, top, bottom, left, right in _scan_decoder(gray, _decode_code128_row):
            if code in seen:
                continue
            seen.add(code)
            codes.append(code)
            boxes.append((top, bottom, left, right))
    if requested == "Code 39" or (requested == "auto" and not codes):
        for code, top, bottom, left, right in _scan_decoder(gray, _decode_code39_row):
            if code in seen:
                continue
            seen.add(code)
            codes.append(code)
            boxes.append((top, bottom, left, right))
    return Result([_region(picture.width, picture.height, boxes)], [codes])


@operator(
    "條碼",
    [],
    [],
    [P("BarCodeHandle", "條碼模型")],
    [],
    "關閉一維條碼模型",
    "關閉後不能再搜尋。",
    "clear_bar_code_model (BarCodeHandle)",
)
def clear_bar_code_model(handle, *, ctx):
    if not isinstance(handle, BarCodeModel):
        raise TypeError("BarCodeHandle 必須是條碼模型")
    handle.closed = True
    return Result([], [])


@operator(
    "條碼",
    [],
    [],
    [
        P("SymbolType", "這版只做 QR Code", "'QR Code'", _SYMBOLS),
        P("GenParamName", "這版不使用，留空字串", "''"),
        P("GenParamValue", "這版不使用，留空字串", "''"),
    ],
    [P("DataCodeHandle", "二維碼模型")],
    "建立二維碼模型",
    "只解 QR Code。Data Matrix、PDF417、Aztec 這版沒有解碼器。",
    "create_data_code_2d_model ('QR Code', DataCodeHandle)",
)
def create_data_code_2d_model(symbol_type, _name, _value, *, ctx):
    symbol = _token(symbol_type, "SymbolType")
    if symbol.lower().replace(" ", "") not in ("qrcode", "qr"):
        raise ValueError("SymbolType 這版只接受 QR Code")
    return Result([], [DataCodeModel("QR Code")])


def _decode_qr(gray: np.ndarray):
    detector = cv2.QRCodeDetector()
    for scale in (1, 2, 3):
        picture = gray if scale == 1 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        ok, decoded, points, _straight = detector.detectAndDecodeMulti(picture)
        if not ok or points is None:
            continue
        texts = [str(item).strip() for item in decoded]
        corners = np.asarray(points, np.float64)
        if corners.ndim == 2:
            corners = corners.reshape(1, -1, 2)
        hits = []
        for text_value, quad in zip(texts, corners):
            if not text_value:
                continue
            quad = quad.reshape(-1, 2) / scale
            hits.append((text_value, quad))
        if hits:
            return hits
    return []


@operator(
    "條碼",
    [P("Image", "含二維碼的影像")],
    [P("SymbolXLDs", "每個碼的外框")],
    [
        P("DataCodeHandle", "二維碼模型"),
        P("GenParamName", "這版不使用，留空字串", "''"),
        P("GenParamValue", "這版不使用，留空字串", "''"),
    ],
    [P("ResultHandles", "從 0 開始的結果編號"), P("DecodedDataStrings", "讀到的字串")],
    "尋找 QR Code",
    "太小的碼會先放大再解，外框座標會縮回原圖。找不到時輪廓和字串都是空的。",
    "find_data_code_2d (Image, SymbolXLDs, DataCodeHandle, ResultHandles, DecodedDataStrings)",
)
def find_data_code_2d(image, handle, _name, _value, *, ctx):
    picture = expect_image(image, "Image")
    if not isinstance(handle, DataCodeModel):
        raise TypeError("DataCodeHandle 必須是 create_data_code_2d_model 建立的模型")
    if handle.closed:
        raise RuntimeError("二維碼模型已經 clear_data_code_2d_model")
    hits = _decode_qr(picture.gray())
    contours = []
    codes = []
    for text_value, quad in hits:
        contour = [[float(point[0]), float(point[1])] for point in quad]
        if contour[0] != contour[-1]:
            contour.append(contour[0])
        contours.append(contour)
        codes.append(text_value)
    xld = XLD(picture.width, picture.height, contours, multi_color=len(contours) > 1)
    return Result([xld], [list(range(len(codes))), codes])


@operator(
    "條碼",
    [],
    [],
    [P("DataCodeHandle", "二維碼模型")],
    [],
    "關閉二維碼模型",
    "關閉後不能再搜尋。",
    "clear_data_code_2d_model (DataCodeHandle)",
)
def clear_data_code_2d_model(handle, *, ctx):
    if not isinstance(handle, DataCodeModel):
        raise TypeError("DataCodeHandle 必須是二維碼模型")
    handle.closed = True
    return Result([], [])


@operator(
    "文字",
    [],
    [],
    [
        P("Mode", "auto 或 manual。這版兩種都用同一個本機引擎", "'auto'", _TEXT_MODE),
        P("OCRClassifier", "auto 或目錄裡的引擎名稱。不讀 MLP 檔", "'auto'", method_choices()),
    ],
    [P("TextModel", "文字模型")],
    "建立文字模型",
    "這不是 HALCON 的 MLP 分類器。auto 只用已安裝的本機通用引擎，不會把影像送到雲端。指定引擎名稱時才用那個引擎。",
    "create_text_model_reader ('auto', 'auto', TextModel)",
)
def create_text_model_reader(mode, classifier, *, ctx):
    kind = _token(mode, "Mode").lower()
    if kind not in ("auto", "manual"):
        raise ValueError("Mode 只能是 auto 或 manual")
    name = _token(classifier, "OCRClassifier")
    if any(name.lower().endswith(suffix) for suffix in (".omc", ".occ", ".ocr")):
        raise ValueError("這版不讀 OCR 分類器檔。請用 auto 或目錄裡的引擎名稱。")
    return Result([], [TextModel(normalize_method(name), kind)])


@operator(
    "文字",
    [P("Image", "要找字的影像")],
    [],
    [P("TextModel", "文字模型")],
    [P("TextResultID", "文字結果")],
    "找出文字塊",
    "用文字模型指定的引擎找出每個字塊，並把讀到的字一起存起來。和 ocr_find 分開，結果要用 get_text_result 或 do_ocr_multi_class_mlp 取出。",
    "find_text (Image, TextModel, TextResultID)",
)
def find_text(image, model, *, ctx):
    picture = expect_image(image, "Image")
    if not isinstance(model, TextModel):
        raise TypeError("TextModel 必須是 create_text_model_reader 建立的模型")
    if model.closed:
        raise RuntimeError("文字模型已經 clear_text_model")
    _engine, hits = read_text(picture, model.method)
    words = []
    for hit in hits:
        points = np.asarray(hit.box, np.float64).reshape(-1, 2)
        if len(points) < 2 or not str(hit.text).strip():
            continue
        words.append(
            {
                "text": str(hit.text).strip(),
                "score": float(hit.score),
                "column": float(points[:, 0].mean()),
                "row": float(points[:, 1].mean()),
                "points": points,
            }
        )
    return Result([], [TextResult(picture.width, picture.height, words)])


@operator(
    "文字",
    [],
    [],
    [
        P("TextResultID", "文字結果"),
        P("ResultName", "class、confidence、row 或 column", "'class'", _TEXT_RESULT),
    ],
    [P("ResultValue", "結果")],
    "讀取文字結果",
    "class 是每個字塊的字。confidence 是 0 到 1。row 和 column 是字塊中心。",
    "get_text_result (TextResultID, 'class', ResultValue)",
)
def get_text_result(result, result_name, *, ctx):
    if not isinstance(result, TextResult):
        raise TypeError("TextResultID 必須是 find_text 的結果")
    if result.closed:
        raise RuntimeError("文字結果已經 clear_text_result")
    name = _token(result_name, "ResultName").lower()
    if name == "class":
        values = [item["text"] for item in result.words]
    elif name == "confidence":
        values = [item["score"] for item in result.words]
    elif name in ("row", "column"):
        values = [item[name] for item in result.words]
    else:
        raise ValueError("ResultName 只能是 class、confidence、row 或 column")
    return Result([], [values])


@operator(
    "文字",
    [P("Character", "要讀的區域"), P("Image", "灰階或彩色影像")],
    [],
    [P("TextModel", "文字模型")],
    [P("Class", "讀到的字"), P("Confidence", "信心")],
    "用文字模型讀區域裡的字",
    "這不是 MLP。它把區域外框裁下來，交給文字模型的引擎。一個區域可以是一個字或一行字。",
    "do_ocr_multi_class_mlp (Character, Image, TextModel, Class, Confidence)",
)
def do_ocr_multi_class_mlp(character, image, model, *, ctx):
    region = expect_region(character, "Character")
    picture = expect_image(image, "Image")
    if not isinstance(model, TextModel):
        raise TypeError("TextModel 必須是 create_text_model_reader 建立的模型")
    if model.closed:
        raise RuntimeError("文字模型已經 clear_text_model")
    mask = region.union_mask()
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Character 是空的區域")
    crop = picture.data[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]
    _engine, hits = read_text(Image(crop), model.method)
    if not hits:
        return Result([], ["", 0.0])
    words = [str(hit.text).strip() for hit in hits if str(hit.text).strip()]
    score = float(np.mean([hit.score for hit in hits])) if hits else 0.0
    return Result([], [" ".join(words), score])


@operator(
    "文字",
    [],
    [],
    [P("TextModel", "文字模型")],
    [],
    "關閉文字模型",
    "關閉後不能再 find_text。",
    "clear_text_model (TextModel)",
)
def clear_text_model(model, *, ctx):
    if not isinstance(model, TextModel):
        raise TypeError("TextModel 必須是文字模型")
    model.closed = True
    return Result([], [])


@operator(
    "文字",
    [],
    [],
    [P("TextResultID", "文字結果")],
    [],
    "關閉文字結果",
    "關閉後不能再讀結果。",
    "clear_text_result (TextResultID)",
)
def clear_text_result(result, *, ctx):
    if not isinstance(result, TextResult):
        raise TypeError("TextResultID 必須是文字結果")
    result.closed = True
    result.words = []
    return Result([], [])
