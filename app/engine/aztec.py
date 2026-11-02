"""Compact Aztec-style symbols that this program can print and read back.

The center is a 9 by 9 bullseye: black, white, black, white, black.
Data bits sit outside that bullseye in the same row-major order on both sides,
with a length and a checksum. The numbering is ours, shared by the encoder and
the decoder.
"""

from __future__ import annotations

import cv2
import numpy as np

_SIZES = (15, 19, 23, 27)


def _crc(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def _data_cells(size: int) -> list[tuple[int, int]]:
    center = size // 2
    cells = []
    for row in range(size):
        for col in range(size):
            if max(abs(row - center), abs(col - center)) >= 5:
                cells.append((row, col))
    return cells


def _payload_bits(payload: str, count: int) -> list[int]:
    raw = payload.encode("ascii")
    body = bytes([len(raw)]) + raw
    crc = _crc(body)
    blob = body + bytes((crc >> 8, crc & 0xFF))
    bits = []
    for byte in blob:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    if len(bits) > count:
        raise ValueError("這段文字放不進 Aztec")
    bits.extend([0] * (count - len(bits)))
    return bits


def _text(bits: list[int]) -> str:
    if len(bits) < 24:
        return ""

    def read_byte(start: int) -> int:
        value = 0
        for shift in range(8):
            value = (value << 1) | bits[start + shift]
        return value

    length = read_byte(0)
    if length < 1 or (1 + length + 2) * 8 > len(bits):
        return ""
    raw = bytes(read_byte(8 * (index + 1)) for index in range(length))
    crc = (read_byte(8 * (length + 1)) << 8) | read_byte(8 * (length + 2))
    if crc != _crc(bytes([length]) + raw):
        return ""
    try:
        return raw.decode("ascii")
    except UnicodeDecodeError:
        return ""


def _paint(size: int, bits: list[int]) -> np.ndarray:
    grid = np.full((size, size), 255, np.uint8)
    center = size // 2
    for row in range(size):
        for col in range(size):
            distance = max(abs(row - center), abs(col - center))
            if distance <= 4 and distance % 2 == 0:
                grid[row, col] = 0
    for bit, (row, col) in zip(bits, _data_cells(size)):
        if bit:
            grid[row, col] = 0
    return grid


def encode_modules(payload: str) -> np.ndarray:
    if payload == "":
        raise ValueError("Text 不能是空的")
    try:
        payload.encode("ascii")
    except UnicodeError as exc:
        raise ValueError("Aztec 這版只寫 ASCII") from exc
    for size in _SIZES:
        cells = _data_cells(size)
        try:
            bits = _payload_bits(payload, len(cells))
        except ValueError:
            continue
        return _paint(size, bits)
    raise ValueError("這段文字放不進 Aztec")


def _rings(gray: np.ndarray, cy: float, cx: float, module: float) -> bool:
    height, width = gray.shape[:2]
    expected = {0: True, 1: False, 2: True, 3: False, 4: True}
    for distance, black in expected.items():
        for dy, dx in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            if distance == 0 and (dy or dx):
                continue
            if distance and not dy and not dx:
                continue
            y = int(round(cy + dy * distance * module))
            x = int(round(cx + dx * distance * module))
            if y < 0 or x < 0 or y >= height or x >= width:
                return False
            if (int(gray[y, x]) < 128) != black:
                return False
    return True


def _read(gray: np.ndarray, cy: float, cx: float, module: float, size: int) -> tuple[str, list[list[float]]] | None:
    height, width = gray.shape[:2]
    half = size // 2
    bits = []
    for row, col in _data_cells(size):
        y = int(round(cy + (row - half) * module))
        x = int(round(cx + (col - half) * module))
        if y < 0 or x < 0 or y >= height or x >= width:
            return None
        bits.append(1 if int(gray[y, x]) < 128 else 0)
    text = _text(bits)
    if not text:
        return None
    span = size * module / 2
    left = cx - span
    top = cy - span
    quad = [
        [left, top],
        [cx + span, top],
        [cx + span, cy + span],
        [left, cy + span],
    ]
    return text, quad


def decode_image(gray: np.ndarray) -> list[tuple[str, list[list[float]]]]:
    picture = np.asarray(gray)
    if picture.ndim == 3:
        picture = picture[:, :, 0]
    dark = (picture < 128).astype(np.uint8)
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(dark, 4)
    hits = []
    for index in range(1, count):
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        if width < 1 or height < 1 or area < 0.6 * width * height:
            continue
        if max(width, height) > min(width, height) * 1.5:
            continue
        module = (width + height) / 2
        cy = float(centroids[index][1])
        cx = float(centroids[index][0])
        if not _rings(picture, cy, cx, module):
            continue
        for size in _SIZES:
            hit = _read(picture, cy, cx, module, size)
            if hit:
                hits.append(hit)
                break
    return hits
