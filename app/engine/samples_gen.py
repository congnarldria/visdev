from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .render import write_cv_image


def ensure_samples(root: Path):
    samples = root / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    (root / "workspace").mkdir(parents=True, exist_ok=True)
    write_parts(samples / "parts.png")
    write_stripes(samples / "stripes.png")
    write_shape_template(samples / "shape_template.png")
    write_shapes_scene(samples / "shapes_scene.png")
    write_part_template(samples / "part_template.png")
    write_part_scene(samples / "part_scene.png")
    (samples / "shape_match.hdv").write_text(SHAPE_MATCH_PROGRAM, encoding="utf-8")
    (samples / "shape_teach.hdv").write_text(SHAPE_TEACH_PROGRAM, encoding="utf-8")
    (samples / "ncc_locate.hdv").write_text(NCC_LOCATE_PROGRAM, encoding="utf-8")


def write_parts(path: Path):
    image = np.full((480, 640, 3), 28, np.uint8)
    noise = np.random.default_rng(3).integers(0, 16, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    cv2.circle(image, (100, 120), 30, (210, 210, 210), -1, lineType=cv2.LINE_AA)
    cv2.circle(image, (220, 120), 18, (200, 200, 200), -1, lineType=cv2.LINE_AA)
    cv2.circle(image, (340, 120), 6, (190, 190, 190), -1, lineType=cv2.LINE_AA)
    cv2.circle(image, (500, 390), 50, (230, 230, 230), -1, lineType=cv2.LINE_AA)
    cv2.rectangle(image, (80, 280), (160, 340), (205, 205, 205), -1)
    cv2.rectangle(image, (230, 250), (260, 370), (200, 200, 200), -1)
    speck = np.random.default_rng(1)
    for _ in range(10):
        x = int(speck.integers(20, 600))
        y = int(speck.integers(20, 440))
        image[y : y + 2, x : x + 2] = 170
    write_cv_image(path, image)


def write_stripes(path: Path):
    image = np.full((360, 640), 32, np.uint8)
    x = 36
    for index, width in enumerate((6, 14, 22, 10, 30, 8, 18)):
        image[:, x : x + width] = 220 if index % 2 == 0 else 90
        x += width + 36
    write_cv_image(path, image)


# (row, column, degrees). The side barb makes 180 degrees a different shape.
SHAPE_POSES = (
    (130.4, 160.6, 8.5),
    (125.0, 470.0, 31.0),
    (360.0, 170.0, -27.0),
    (350.0, 480.0, 64.0),
)

_ARROW = (
    (0.0, -46.0),
    (18.0, -14.0),
    (7.0, -14.0),
    (7.0, 18.0),
    (16.0, 18.0),
    (7.0, 28.0),
    (7.0, 42.0),
    (-7.0, 42.0),
    (-7.0, -14.0),
    (-18.0, -14.0),
)


def _draw_arrow(image: np.ndarray, row: float, col: float, degrees: float):
    radians = np.deg2rad(degrees)
    cosine = float(np.cos(radians))
    sine = float(np.sin(radians))
    polygon = []
    for point_col, point_row in _ARROW:
        x = col + point_col * cosine + point_row * sine
        y = row - point_col * sine + point_row * cosine
        polygon.append([int(round(x)), int(round(y))])
    cv2.fillPoly(image, [np.array(polygon, np.int32)], 214, lineType=cv2.LINE_AA)


def write_shape_template(path: Path):
    image = np.full((180, 180), 32, np.uint8)
    center = (image.shape[0] - 1) / 2.0
    _draw_arrow(image, center, center, 0.0)
    image = cv2.GaussianBlur(image, (3, 3), 0.7)
    write_cv_image(path, image)


def write_shapes_scene(path: Path):
    image = np.full((480, 640), 32, np.uint8)
    for row, col, degrees in SHAPE_POSES:
        _draw_arrow(image, row, col, degrees)
    image = cv2.GaussianBlur(image, (3, 3), 0.7)
    noise = np.random.default_rng(11).normal(0, 3.0, image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    write_cv_image(path, image)


SHAPE_MATCH_PROGRAM = """* 教導箭頭後做次像素形狀匹配，每個實例一種顏色
read_image (Scene, 'samples/shapes_scene.png')
read_image (Template, 'samples/shape_template.png')
teach_shape_model (Template, 15, 15, 164, 164, 0, rad(-180), rad(360), 0, 'use_polarity', 40, 10, ModelID)
find_shape_model (Scene, ModelID, rad(-180), rad(360), 0.7, 0, 0.4, 'interpolation', 0, 0.4, Row, Column, Angle, Score)
dev_display (Scene)
dev_set_line_width (2)
dev_display_shape_matching_results (ModelID, Row, Column, Angle)
Count := |Row|
Label := 'matches ' + Count
dev_set_color ('white')
dev_disp_text (Label, 24, 12)
"""


# (row, column, degrees). The side tooth makes 180 degrees a different shape.
PART_POSES = (
    (150.0, 160.0, 15.0),
    (160.0, 470.0, -35.0),
    (370.0, 180.0, 80.0),
    (360.0, 490.0, 155.0),
)

# Local (column, row). Row grows downward. The tooth is only on the right.
_KEY = (
    (0.0, -52.0),
    (18.0, -38.0),
    (18.0, -18.0),
    (8.0, -12.0),
    (8.0, 16.0),
    (24.0, 16.0),
    (24.0, 32.0),
    (8.0, 32.0),
    (8.0, 48.0),
    (-8.0, 48.0),
    (-8.0, -12.0),
    (-18.0, -18.0),
    (-18.0, -38.0),
)


def _draw_key(image: np.ndarray, row: float, col: float, degrees: float):
    radians = np.deg2rad(degrees)
    cosine = float(np.cos(radians))
    sine = float(np.sin(radians))
    polygon = []
    for point_col, point_row in _KEY:
        x = col + point_col * cosine + point_row * sine
        y = row - point_col * sine + point_row * cosine
        polygon.append([int(round(x)), int(round(y))])
    cv2.fillPoly(image, [np.array(polygon, np.int32)], 220, lineType=cv2.LINE_AA)


def write_part_template(path: Path):
    image = np.full((200, 200), 36, np.uint8)
    center = (image.shape[0] - 1) / 2.0
    _draw_key(image, center, center, 0.0)
    image = cv2.GaussianBlur(image, (3, 3), 0.6)
    write_cv_image(path, image)


def write_part_scene(path: Path):
    image = np.full((480, 640), 36, np.uint8)
    cv2.circle(image, (320, 250), 34, 180, 8, lineType=cv2.LINE_AA)
    for row, col, degrees in PART_POSES:
        _draw_key(image, row, col, degrees)
    image = cv2.GaussianBlur(image, (3, 3), 0.6)
    noise = np.random.default_rng(19).normal(0, 2.5, image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    write_cv_image(path, image)


SHAPE_TEACH_PROGRAM = """* 兩張圖：part_template.png 是要教導的鑰匙，part_scene.png 是搜尋畫面
* 教導框包住整支鑰匙。畫面裡有四支不同位置、不同角度的鑰匙，圓環不是目標
read_image (Template, 'samples/part_template.png')
read_image (Scene, 'samples/part_scene.png')
teach_shape_model (Template, 18, 18, 181, 181, 0, rad(-180), rad(360), 0, 'use_polarity', 40, 10, ModelID)
find_shape_model (Scene, ModelID, rad(-180), rad(360), 0.8, 0, 0.4, 'interpolation', 0, 0.5, Row, Column, Angle, Score)
dev_display (Scene)
dev_set_line_width (2)
dev_display_shape_matching_results (ModelID, Row, Column, Angle)
Count := |Row|
Label := '找到 ' + Count
dev_set_color ('white')
dev_disp_text (Label, 28, 16)
"""


NCC_LOCATE_PROGRAM = """* 教導同一支箭頭，再用 NCC 定位。只畫中心、方向箭頭和角度
read_image (Scene, 'samples/shapes_scene.png')
read_image (Template, 'samples/shape_template.png')
teach_ncc_model (Template, 15, 15, 164, 164, 0, rad(-180), rad(360), 0, 'use_polarity', ModelID)
find_ncc_model (Scene, ModelID, rad(-180), rad(360), 0.75, 0, 0.4, 'interpolation', 0, Row, Column, Angle, Score)
dev_display (Scene)
dev_set_line_width (2)
dev_display_ncc_matching_results (ModelID, Row, Column, Angle)
Count := |Row|
Label := 'ncc ' + Count
dev_set_color ('white')
dev_disp_text (Label, 24, 12)
"""
