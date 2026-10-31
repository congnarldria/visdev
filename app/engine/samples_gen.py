from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .render import write_cv_image


SAMPLE_ORDER = (
    "blob_inspect.hdv",
    "roi_check.hdv",
    "auto_threshold.hdv",
    "morph_region.hdv",
    "region_sets.hdv",
    "subpix_blob.hdv",
    "edge_amp.hdv",
    "xld_fit.hdv",
    "xld_circle.hdv",
    "image_filter.hdv",
    "tuple_stats.hdv",
    "hdevelop_batch1.hdv",
    "hdevelop_batch2.hdv",
    "hdevelop_batch3.hdv",
    "hdevelop_batch4.hdv",
    "hdevelop_batch5.hdv",
    "hdevelop_batch6.hdv",
    "hdevelop_batch7.hdv",
    "hdevelop_batch8.hdv",
    "hdevelop_batch9.hdv",
    "hdevelop_batch10.hdv",
    "hdevelop_batch11.hdv",
    "hdevelop_batch12.hdv",
    "hdevelop_batch13.hdv",
    "hdevelop_batch14.hdv",
    "hdevelop_batch15.hdv",
    "hdevelop_batch16.hdv",
    "hdevelop_batch17.hdv",
    "hdevelop_batch18.hdv",
    "hdevelop_batch19.hdv",
    "hdevelop_batch20.hdv",
    "hdevelop_batch21.hdv",
    "hdevelop_batch22.hdv",
    "hdevelop_batch23.hdv",
    "hdevelop_batch24.hdv",
    "hdevelop_batch25.hdv",
    "hdevelop_batch26.hdv",
    "hdevelop_batch27.hdv",
    "hdevelop_batch28.hdv",
    "hdevelop_batch29.hdv",
    "hdevelop_batch30.hdv",
    "hdevelop_batch31.hdv",
    "hdevelop_batch32.hdv",
    "hdevelop_batch33.hdv",
    "hdevelop_batch34.hdv",
    "hdevelop_batch35.hdv",
    "hdevelop_batch36.hdv",
    "hdevelop_batch37.hdv",
    "hdevelop_batch38.hdv",
    "hdevelop_batch39.hdv",
    "hdevelop_batch40.hdv",
    "hdevelop_batch41.hdv",
    "hdevelop_batch42.hdv",
    "hdevelop_batch43.hdv",
    "hdevelop_batch44.hdv",
    "hdevelop_batch45.hdv",
    "hdevelop_batch46.hdv",
    "hdevelop_batch47.hdv",
    "hdevelop_batch48.hdv",
    "hdevelop_batch49.hdv",
    "hdevelop_batch50.hdv",
    "hdevelop_batch51.hdv",
    "hdevelop_batch52.hdv",
    "hdevelop_batch53.hdv",
    "hdevelop_batch54.hdv",
    "ocr_read.hdv",
    "shape_match.hdv",
    "shape_teach.hdv",
    "ncc_locate.hdv",
    "gauge.hdv",
)


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
    write_ocr_label(samples / "ocr_label.png")
    write_measure_scene(samples / "measure_batch.png")
    write_codes(samples / "codes.png")
    write_code39(samples / "code39.png")
    (samples / "shape_match.hdv").write_text(SHAPE_MATCH_PROGRAM, encoding="utf-8")
    (samples / "shape_teach.hdv").write_text(SHAPE_TEACH_PROGRAM, encoding="utf-8")
    (samples / "ncc_locate.hdv").write_text(NCC_LOCATE_PROGRAM, encoding="utf-8")
    write_gauge(samples / "gauge.png")
    (samples / "gauge.hdv").write_text(GAUGE_PROGRAM, encoding="utf-8")
    for name, source in GENERATED_PROGRAMS.items():
        (samples / name).write_text(source, encoding="utf-8")


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


def write_ocr_label(path: Path):
    image = np.full((180, 720, 3), 255, np.uint8)
    cv2.putText(image, "VISDEV OCR 168", (28, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3, cv2.LINE_AA)
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


def write_gauge(path: Path):
    image = np.full((480, 640), 32, np.uint8)
    row, col = np.indices(image.shape)
    image[(col - 150) ** 2 + (row - 110) ** 2 <= 42 ** 2] = 230
    image[200:290, 360:500] = 210
    dist = np.hypot(col - 170, row - 340)
    delta = np.mod(np.arctan2(row - 340, col - 170) - 0.35, 2 * np.pi)
    image[(dist >= 24) & (dist <= 58) & (delta <= 1.8)] = 200
    write_cv_image(path, image)


GAUGE_PROGRAM = """* 2D 次像素量測：點、直線、圓、矩形、扇形、多邊形
read_image (Image, 'samples/gauge.png')
dev_set_line_width (2)
dev_display (Image)
gauge_point (Image, Mark, 110, 80, 110, 220, 1, 8, 'all', 'first', PRow, PCol, Amp, Dist)
dev_set_color ('white')
dev_display (Mark)
gauge_line (Image, Line, 200, 380, 200, 480, 12, 1, 8, 'all', 'best', LRow1, LCol1, LRow2, LCol2, LAngle, LUsed)
dev_set_color ('yellow')
dev_display (Line)
gauge_circle (Image, Circle, 110, 150, 42, 12, 1, 8, 'all', 'best', CRow, CCol, CRad, CUsed)
dev_set_color ('green')
dev_display (Circle)
gauge_rectangle (Image, Rect, 244.5, 429.5, 0, 70, 45, 12, 1, 8, 'all', 'best', RRow, RCol, RPhi, RL1, RL2, RUsed)
dev_set_color ('cyan')
dev_display (Rect)
gauge_wedge (Image, Wedge, 340, 170, 24, 58, 0.35, 1.8, 10, 1, 8, 'all', 'best', WRow, WCol, WR1, WR2, WStart, WExtent, WUsed)
dev_set_color ('magenta')
dev_display (Wedge)
gauge_polygon (Image, Poly, [200, 200, 290, 290], [360, 500, 500, 360], 12, 1, 8, 'all', 'best', 1, GRows, GCols, GUsed)
dev_set_color ('orange')
dev_display (Poly)
distance_pp (0, 0, 3, 4, Dist)
distance_pl (20, 0, 0, 10, 50, 10, DistLine)
distance_pc (CRow, CCol + CRad, CRow, CCol, CRad, DistCircle)
angle_ll (0, 0, 0, 10, 0, 0, 10, 0, RightAngle)
intersection_ll (0, 10, 50, 10, 20, 0, 20, 40, IRow, ICol)
"""


AUTO_THRESHOLD_PROGRAM = """* 自動二值化，再拆成連通物件
read_image (Image, 'samples/parts.png')
rgb1_to_gray (Image, Gray)
binary_threshold (Gray, Region, 'max_separability', 'light', Used)
opencv_threshold (Gray, CvRegion, 80, 'binary')
connection (Region, Parts, 1)
count_obj (Parts, Count)
dev_display (Gray)
dev_set_colored (16)
dev_set_draw ('fill')
dev_display (Parts)
Label := 'Otsu ' + Used + '  count ' + Count
dev_set_color ('white')
dev_disp_text (Label, 24, 12)
"""


MORPH_REGION_PROGRAM = """* 形態學：開運算去掉小點，閉運算補洞，再取邊界
read_image (Image, 'samples/parts.png')
rgb1_to_gray (Image, Gray)
threshold (Gray, Region, 60, 255)
opening_circle (Region, Opened, 2.5)
closing_circle (Opened, Closed, 4.5)
fill_up (Closed, Filled)
dilation_circle (Filled, Dilated, 3.5)
erosion_rectangle1 (Dilated, Eroded, 5, 5)
boundary (Eroded, Border, 'inner')
dev_display (Gray)
dev_set_draw ('margin')
dev_set_line_width (2)
dev_set_color ('green')
dev_display (Eroded)
dev_set_color ('yellow')
dev_display (Border)
"""


REGION_SETS_PROGRAM = """* 區域集合：矩形與圓的交集、差集、聯集
read_image (Image, 'samples/parts.png')
gen_rectangle1 (Rect, 70, 40, 240, 360)
gen_circle (Disk, 150, 180, 80)
intersection (Rect, Disk, Common)
difference (Rect, Disk, OnlyRect)
union2 (Rect, Disk, Both)
complement (Common, Outside)
dev_display (Image)
dev_set_draw ('margin')
dev_set_line_width (2)
dev_set_color ('cyan')
dev_display (Both)
dev_set_color ('yellow')
dev_display (Common)
dev_set_color ('red')
dev_display (OnlyRect)
"""


SUBPIX_BLOB_PROGRAM = """* 次像素輪廓，平滑後填回區域
read_image (Image, 'samples/parts.png')
rgb1_to_gray (Image, Gray)
threshold_sub_pix (Gray, Border, 80)
select_contours_xld (Border, Long, 'contlength', 30, 99999)
smooth_contours_xld (Long, Smooth, 5)
close_contours_xld (Smooth, Closed)
gen_region_contour_xld (Closed, Region, 'filled')
dev_display (Gray)
dev_set_line_width (2)
dev_set_color ('magenta')
dev_display (Closed)
dev_set_draw ('fill')
dev_set_color ('#66ccff')
dev_display (Region)
"""


XLD_FIT_PROGRAM = """* 條紋的次像素邊緣擬合成直線，並接上靠近的輪廓
read_image (Image, 'samples/stripes.png')
rgb1_to_gray (Image, Gray)
edges_sub_pix (Gray, Edges, 'canny', 1, 20, 40)
select_contours_xld (Edges, Long, 'contlength', 40, 99999)
fit_line_contour_xld (Long, Lines, Row1, Col1, Row2, Col2)
union_adjacent_contours_xld (Long, Joined, 12, 0.5)
sort_contours_xld (Joined, Sorted, 'character', 'true', 'row')
Count := |Row1|
dev_display (Gray)
dev_set_line_width (2)
dev_set_color ('green')
dev_display (Lines)
dev_set_color ('yellow')
dev_display (Sorted)
Label := 'lines ' + Count
dev_set_color ('white')
dev_disp_text (Label, 20, 12)
"""


XLD_CIRCLE_PROGRAM = """* 亮斑外輪廓擬合圓與橢圓
read_image (Image, 'samples/parts.png')
rgb1_to_gray (Image, Gray)
threshold (Gray, Region, 60, 255)
connection (Region, Parts, 1)
select_shape (Parts, Big, 'area', 'and', 400, 999999)
gen_contour_region_xld (Big, Contours)
fit_circle_contour_xld (Contours, Circles, Row, Column, Radius)
fit_ellipse_contour_xld (Contours, Ellipses, ERow, ECol, Phi, Radius1, Radius2)
length_xld (Contours, Length)
area_center_xld (Contours, Area, CRow, CCol)
dev_display (Gray)
dev_set_line_width (2)
dev_set_color ('green')
dev_display (Circles)
dev_set_color ('orange')
dev_display (Ellipses)
"""


IMAGE_FILTER_PROGRAM = """* 影像濾波：高斯、中值、邊緣強化、直方圖等化、反相
read_image (Image, 'samples/stripes.png')
rgb1_to_gray (Image, Gray)
gauss_image (Gray, Smooth, 5)
median_image (Gray, Median, 2)
mean_image (Gray, MeanImage, 5, 5)
emphasize (Gray, Sharp, 7, 7, 1)
scale_image (Sharp, Scaled, 1.2, 0)
equ_histo_image (Gray, Equ)
invert_image (Equ, Inverted)
dev_display (Sharp)
"""


TUPLE_STATS_PROGRAM = """* 連通物件面積的 tuple 長度、最小、最大、總和、平均
read_image (Image, 'samples/parts.png')
rgb1_to_gray (Image, Gray)
threshold (Gray, Region, 60, 255)
connection (Region, Parts, 1)
area_center (Parts, Area, Row, Column)
tuple_length (Area, Length)
tuple_min (Area, MinArea)
tuple_max (Area, MaxArea)
tuple_sum (Area, SumArea)
tuple_mean (Area, MeanArea)
dev_display (Gray)
dev_set_colored (16)
dev_display (Parts)
Label := 'n=' + Length + ' mean=' + MeanArea
dev_set_color ('white')
dev_disp_text (Label, 24, 12)
"""


def write_measure_scene(path: Path):
    image = np.full((240, 320), 30, np.uint8)
    cv2.rectangle(image, (40, 30), (100, 200), 210, -1)
    cv2.circle(image, (180, 70), 28, 220, -1, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (220, 160), (36, 22), 25, 0, 360, 200, -1, lineType=cv2.LINE_AA)
    write_cv_image(path, image)


def write_code39(path: Path):
    from .chapters.code_ops import render_code39

    write_cv_image(path, render_code39("VISDEV", module=3, height=140, quiet=24))


def write_codes(path: Path):
    canvas = np.full((280, 760), 255, np.uint8)
    qr = cv2.QRCodeEncoder.create().encode("VISDEV-3")
    qr = cv2.resize(qr, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
    canvas[40 : 40 + qr.shape[0], 36 : 36 + qr.shape[1]] = qr
    left = {
        "0": "0001101", "1": "0011001", "2": "0010011", "3": "0111101", "4": "0100011",
        "5": "0110001", "6": "0101111", "7": "0111011", "8": "0110111", "9": "0001011",
    }
    even = {
        "0": "0100111", "1": "0110011", "2": "0011011", "3": "0100001", "4": "0011101",
        "5": "0111001", "6": "0000101", "7": "0010001", "8": "0001001", "9": "0010111",
    }
    parity_table = {
        "0": "LLLLLL", "1": "LLGLGG", "2": "LLGGLG", "3": "LLGGGL", "4": "LGLLGG",
        "5": "LGGLLG", "6": "LGGGLL", "7": "LGLGLG", "8": "LGLGGL", "9": "LGGLGL",
    }
    digits12 = "400638133393"
    parity = parity_table[digits12[0]]
    nums = [int(char) for char in digits12]
    check = (10 - ((sum(nums[::2]) + 3 * sum(nums[1::2])) % 10)) % 10
    digits = digits12 + str(check)
    bits = "101"
    for index, flag in enumerate(parity):
        bits += (left if flag == "L" else even)[digits[index + 1]]
    bits += "01010"
    for index in range(6):
        bits += "".join("1" if bit == "0" else "0" for bit in left[digits[7 + index]])
    bits += "101"
    row = []
    for bit in bits:
        row.extend([0 if bit == "1" else 255] * 3)
    quiet = [255] * 36
    stripe = np.array(quiet + row + quiet, np.uint8)
    bar = np.tile(stripe, (150, 1))
    x0 = 280
    canvas[60 : 60 + bar.shape[0], x0 : x0 + bar.shape[1]] = bar
    write_cv_image(path, canvas)


HDEVELOP_BATCH3_PROGRAM = """* 矩陣、剛性變換、條碼和文字
create_matrix (2, 2, [4, 7, 2, 6], A)
invert_matrix (A, 'general', 0, Inv)
mult_matrix (A, Inv, 'AB', Eye)
get_value_matrix (Eye, 0, 0, Eye00)
vector_angle_to_rigid (10, 0, 0, 0, 10, rad(90), HomMat2D)
affine_trans_point_2d (HomMat2D, 10, 0, Row, Col)
create_pose (0, 0, 0, 0, 0, rad(90), 'Rp+T', 'gba', 'point', Pose)
pose_to_hom_mat3d (Pose, HomMat3D)
affine_trans_point_3d (HomMat3D, 1, 0, 0, Qx, Qy, Qz)
read_image (Codes, 'samples/codes.png')
create_bar_code_model ('', '', BarCode)
find_bar_code (Codes, BarRegions, BarCode, 'EAN-13', BarText)
create_data_code_2d_model ('QR Code', '', '', DataCode)
find_data_code_2d (Codes, CodeXld, DataCode, '', '', CodeId, CodeText)
read_image (Label, 'samples/ocr_label.png')
create_text_model_reader ('auto', 'auto', TextModel)
find_text (Label, TextModel, TextResult)
get_text_result (TextResult, 'class', Words)
dev_display (Codes)
dev_set_color ('green')
dev_display (CodeXld)
"""


HDEVELOP_BATCH2_PROGRAM = """* 量測、計量模型與縮放形狀
read_image (Image, 'samples/measure_batch.png')
rgb1_to_gray (Image, Gray)
gen_measure_rectangle2 (100, 70, 0, 50, 4, 320, 240, 'bilinear', Measure)
measure_pos (Gray, Measure, 1, 20, 'all', 'all', RowEdge, ColEdge, Amp, Dist)
measure_pairs (Gray, Measure, 1, 20, 'positive', 'all', RowA, ColA, AmpA, RowB, ColB, AmpB, Intra, Inter)
create_metrology_model (Metro)
set_metrology_model_image_size (Metro, 320, 240)
add_metrology_object_line_measure (Metro, 40, 40, 190, 40, 12, 3, 1, 20, LineIndex)
add_metrology_object_circle_measure (Metro, 70, 180, 28, 12, 3, 1, 20, CircleIndex)
add_metrology_object_ellipse_measure (Metro, 160, 220, rad(25), 36, 22, 12, 3, 1, 18, EllipseIndex)
apply_metrology_model (Gray, Metro)
get_metrology_object_result (Metro, LineIndex, 'all', 'result_type', 'column_begin', LineCol)
get_metrology_object_result (Metro, CircleIndex, 'all', 'result_type', 'all_param', CircleParam)
get_metrology_object_result (Metro, EllipseIndex, 'all', 'result_type', 'all_param', EllipseParam)
get_metrology_object_result_contour (Contour, Metro, 'all', 'all', 2)
crop_rectangle1 (Gray, Template, 40, 150, 100, 210)
create_scaled_shape_model (Template, 2, 0, rad(15), 0.15, 0.9, 1.1, 0.1, 'none', 'use_polarity', 30, 10, ModelID)
find_scaled_shape_model (Gray, ModelID, 0, rad(15), 0.85, 1.15, 0.4, 1, 0.5, 'least_squares', 0, 0.6, FoundRow, FoundCol, FoundAngle, FoundScale, FoundScore)
get_shape_model_params (ModelID, Levels, AngleStart, AngleExtent, AngleStep, ScaleMin, ScaleMax, ScaleStep, Metric, MinContrast)
dev_display (Gray)
dev_set_color ('green')
dev_display (Contour)
"""


HDEVELOP_BATCH1_PROGRAM = """* tuple、影像相加、平移和局部閾值
read_image (Image, 'samples/parts.png')
rgb1_to_gray (Image, Gray)
tuple_gen_sequence (1, 5, 2, Seq)
tuple_add (Seq, 10, Plus)
tuple_select (Plus, 0, First)
add_image (Gray, Gray, Sum, 0.5, 0)
gauss_image (Gray, Smooth, 5)
dyn_threshold (Gray, Smooth, Bright, 8, 'light')
hom_mat2d_identity (Id)
hom_mat2d_translate (Id, 12, 0, Mat)
affine_trans_image (Sum, Moved, Mat, 'nearest_neighbor', 'false')
dev_display (Moved)
dev_set_color ('red')
dev_disp_circle (120, 100, 30)
"""


OCR_READ_PROGRAM = """* 讀出圖上的文字，每個字塊一個外框
read_image (Image, 'samples/ocr_label.png')
ocr_find (Image, Words, 'auto', Text, Row, Column, Score)
dev_display (Image)
dev_set_line_width (2)
dev_set_color ('#e6b800')
dev_display (Words)
"""


HDEVELOP_BATCH4_PROGRAM = """* 檔案相機、Code 39、四元數和點雲
open_framegrabber ('File', 'samples/code39.png', 'gray', 0, 0, Acq)
grab_image (Image, Acq)
close_framegrabber (Acq)
create_bar_code_model (Bar)
find_bar_code (Image, Regions, Bar, 'Code 39', Text39)
axis_angle_to_quat (0, 0, 1, rad(90), Quat)
quat_rotate_point (Quat, 1, 0, 0, Qx, Qy, Qz)
gen_object_model_3d_from_points ([0, 1], [0, 0], [0, 0], Cloud)
hom_mat3d_identity (Id)
hom_mat3d_translate (Id, 0, 0, 5, Shift)
affine_trans_object_model_3d (Cloud, Shift, Moved)
get_object_model_3d_params (Moved, 'center', Center)
create_class_knn (1, Knn)
add_sample_class_knn (Knn, [0, 0], 1)
add_sample_class_knn (Knn, [10, 0], 2)
classify_class_knn (Knn, [0.2, 0], ClassId, Score)
dev_display (Image)
dev_set_color ('green')
dev_display (Regions)
"""


HDEVELOP_BATCH5_PROGRAM = """* 灰階侵蝕、橢圓和對比
gen_image_const (Image, 'byte', 160, 120)
gen_rectangle1 (Box, 30, 40, 90, 120)
paint_region (Box, Image, Painted, 200, 'fill')
gray_erosion_rect (Painted, Eroded, 5, 5)
scale_image_max (Eroded, Stretched)
gen_ellipse (Oval, 60, 80, 0, 30, 12)
distance_ps (0, 0, 0, 10, 0, 20, Dist)
projection_pl (5, 4, 0, 0, 0, 10, RowProj, ColProj)
dev_display (Stretched)
dev_set_color ('green')
dev_display (Oval)
"""


HDEVELOP_BATCH6_PROGRAM = """* 尺寸、灰階和頂帽
gen_image_const (Base, 'byte', 80, 60)
get_image_size (Base, Width, Height)
gen_rectangle1 (Box, 10, 10, 40, 50)
paint_region (Box, Base, Marked, 180, 'fill')
min_max_gray (Box, Marked, 0, MinGray, MaxGray, Range)
get_grayval (Marked, 20, 20, Gray)
get_region_points (Box, Rows, Columns)
tuple_length (Rows, Count)
gen_circle (SE, 20, 20, 2)
gray_tophat (Marked, SE, Peaks)
get_grayval (Peaks, 25, 30, Peak)
dev_display (Marked)
"""


HDEVELOP_BATCH7_PROGRAM = """* 斜坡、直徑和標準形
gen_image_gray_ramp (Ramp, 0, 2, 10, 0, 0, 40, 30)
get_grayval (Ramp, 4, 5, Gray)
copy_image (Ramp, Copied)
get_grayval (Copied, 4, 5, CopiedGray)
gen_rectangle1 (Wide, 2, 2, 12, 18)
gen_rectangle1 (Small, 2, 24, 6, 28)
concat_obj (Wide, Small, Both)
count_obj (Both, BothCount)
select_shape_std (Both, Biggest, 'max_area', 100)
area_center (Biggest, BigArea, BigRow, BigCol)
copy_obj (Both, First, 1, 1)
count_obj (First, FirstCount)
gray_histo (Wide, Ramp, Histo, Relative)
tuple_sum (Histo, HistoSum)
gray_projections (Wide, Ramp, 'simple', Hor, Vert)
tuple_select (Hor, 4, RowSum)
diameter_region (Wide, DRow1, DCol1, DRow2, DCol2, Diameter)
elliptic_axis (Wide, Ra, Rb, Phi)
orientation_region (Wide, Angle)
gen_region_line (Line, 0, 0, 10, 0)
test_region_point (Line, 5, 0, Inside)
test_region_point (Line, 5, 3, Outside)
gen_region_points (Dots, [1, 2], [3, 4])
area_center (Dots, DotArea, DotRow, DotCol)
crop_part (Ramp, Part, 0, 0, 10, 8)
get_image_size (Part, PartWidth, PartHeight)
deviation_image (Ramp, Dev, 5, 5)
get_grayval (Dev, 15, 20, DevGray)
gray_range_rect (Ramp, Span, 5, 5)
get_grayval (Span, 15, 20, SpanGray)
sobel_dir (Ramp, Amp, Dir, 'sum_sqrt', 3)
get_grayval (Amp, 15, 20, AmpGray)
get_grayval (Dir, 15, 20, DirGray)
gen_circle_contour_xld (Circle, 15, 20, 8, 0, rad(360), 'positive', 1)
length_xld (Circle, CircLen)
gen_rectangle2_contour_xld (Rect, 15, 20, 0, 10, 4)
length_xld (Rect, RectLen)
distance_ss (0, 0, 0, 10, 5, 0, 5, 10, SegDist)
distance_ss (0, 0, 10, 10, 0, 10, 10, 0, CrossDist)
dev_display (Ramp)
"""


HDEVELOP_BATCH8_PROGRAM = """* 對比拉伸、投影和箭頭
gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 64, 48)
scale_image_range (Ramp, Scaled, 0, 63)
get_grayval (Scaled, 0, 63, Top)
derivate_gauss (Ramp, Dx, 1.0, 'x')
derivate_gauss (Ramp, Dxx, 1.0, 'xx')
get_grayval (Dx, 24, 32, Deriv)
get_grayval (Dxx, 24, 32, Deriv2)
gen_image_proto (Ramp, Flat, 40)
texture_laws (Flat, Energy, 'el', 0, 5)
texture_laws (Flat, Level, 'll', 0, 5)
get_grayval (Energy, 24, 32, EdgeEnergy)
get_grayval (Level, 24, 32, LevelGray)
gen_rectangle1 (Outer, 8, 8, 32, 42)
gen_rectangle1 (Inner, 14, 16, 24, 30)
difference (Outer, Inner, Ring)
fill_up_shape (Ring, Filled, 'area', 1, 99999)
area_center (Ring, HoleArea, HoleRow, HoleCol)
area_center (Filled, FullArea, FullRow, FullCol)
gen_rectangle1 (Box, 10, 10, 30, 40)
distance_pr (Box, 10, 10, InsideMin, InsideMax)
distance_pr (Box, 0, 0, OutsideMin, OutsideMax)
angle_lx (0, 0, 0, 10, Angle)
line_position (0, 0, 0, 10, RowCenter, ColCenter, Length, Phi)
gen_contour_polygon_xld (Segment, [10, 10], [0, 20])
gen_parallel_contour_xld (Segment, Parallel, 'contour_normal', 4)
get_contour_xld (Parallel, ParRows, ParCols)
tuple_mean (ParRows, ParRow)
smallest_rectangle1_xld (Segment, Row1, Column1, Row2, Column2)
smallest_circle_xld (Segment, CircleRow, CircleCol, CircleRadius)
distance_cc (Segment, Parallel, 'point_to_segment', ContourMin, ContourMax)
gen_arrow_contour_xld (Arrow, 5, 5, 5, 25, 6, 3)
count_obj (Arrow, ArrowCount)
paint_xld (Arrow, Ramp, Painted, 200)
get_grayval (Painted, 5, 15, ArrowGray)
gen_image_const (Base, 'byte', 8, 8)
gen_image_proto (Base, Hue, 0)
gen_image_proto (Base, Sat, 255)
gen_image_proto (Base, Val, 255)
trans_to_rgb (Hue, Sat, Val, Red, Green, Blue, 'hsv')
get_grayval (Red, 1, 1, RedGray)
get_grayval (Green, 1, 1, GreenGray)
get_grayval (Blue, 1, 1, BlueGray)
gen_image_const (Dot, 'byte', 40, 30)
set_grayval (Dot, Marked, 10, 12, 255)
hom_vector_to_proj_hom_mat2d ([0, 0, 20, 20], [0, 30, 30, 0], [1, 1, 1, 1], [7, 7, 27, 27], [5, 35, 35, 5], [1, 1, 1, 1], HomMat)
projective_trans_pixel (HomMat, 10, 12, RowT, ColT)
projective_trans_image (Marked, Moved, HomMat, 'nearest_neighbor', 'false')
get_grayval (Moved, 17, 17, MovedGray)
get_grayval (Moved, 10, 12, OldGray)
dev_display (Scaled)
"""


HDEVELOP_BATCH9_PROGRAM = """* 對齊、角點和區域距離
gen_image_const (Image, 'byte', 80, 60)
gen_rectangle1 (Left, 10, 10, 25, 30)
gen_rectangle1 (Right, 10, 42, 25, 60)
distance_rr (Left, Right, DistMin, DistMax)
vector_to_hom_mat2d ([0, 0, 10], [0, 10, 0], [5, 5, 15], [3, 13, 3], Affine)
affine_trans_pixel (Affine, 2, 4, RowA, ColA)
vector_to_similarity ([0, 0], [0, 10], [0, 0], [0, 20], Similar)
affine_trans_pixel (Similar, 0, 10, RowS, ColS)
vector_to_rigid ([0, 0, 10], [0, 10, 0], [0, -10, 0], [0, 0, 10], Rigid)
affine_trans_pixel (Rigid, 0, 10, RowR, ColR)
hom_mat2d_identity (Id)
hom_mat2d_reflect (Id, 0, 0, 10, 0, Mirror)
affine_trans_pixel (Mirror, 0, 10, RowM, ColM)
paint_region (Left, Image, Block, 255, 'fill')
points_harris (Block, 1, 0.05, 0.04, CornerRows, CornerCols)
tuple_length (CornerRows, CornerCount)
tuple_min (CornerRows, CornerTop)
set_grayval (Image, Peak, 20, 20, 255)
local_max (Peak, Peaks)
test_region_point (Peaks, 20, 20, PeakInside)
gray_erosion_circle (Peak, Eroded, 1)
get_grayval (Eroded, 20, 20, ErodedGray)
gen_image_proto (Image, Flat, 80)
gray_erosion_circle (Flat, FlatErode, 2)
get_grayval (FlatErode, 30, 30, FlatGray)
get_region_contour (Left, CRows, CCols)
tuple_length (CRows, ContourCount)
gen_rectangle1 (Big, 4, 4, 20, 40)
gen_rectangle1 (Tiny, 40, 4, 44, 8)
concat_obj (Big, Tiny, Shapes)
select_shape_proto (Shapes, Big, Kept, 'area', 0.5, 1.5)
count_obj (Kept, KeptCount)
moments_region_2nd (Right, M11, M20, M02)
projective_trans_region (Left, Shifted, Affine, 'nearest_neighbor')
test_region_point (Shifted, 20, 23, MovedIn)
test_region_point (Shifted, 10, 10, MovedOut)
gen_contour_polygon_xld (Segment, [8, 8], [8, 18])
projective_trans_contour_xld (Segment, SegmentT, Affine)
get_contour_xld (SegmentT, TRows, TCols)
tuple_mean (TRows, MeanRow)
tuple_mean (TCols, MeanCol)
gen_region_line (Thin, 40, 5, 40, 50)
gen_contours_skeleton_xld (Thin, Skel)
length_xld (Skel, SkelLen)
convert_image_type (Image, Copied, 'byte')
get_grayval (Copied, 0, 0, Zero)
cfa_to_rgb (Image, Bayer, 'bayer_rg')
count_channels (Bayer, Channels)
dev_display (Block)
"""


HDEVELOP_BATCH10_PROGRAM = """* 次方、脊線和交點
gen_image_const (Image, 'byte', 80, 60)
gen_rectangle1 (Bar, 29, 8, 31, 72)
gen_rectangle1 (Hole, 20, 20, 24, 24)
gen_region_points (Dot, [10], [20])
gen_rectangle2_contour_xld (Rect, 30, 40, 0, 12, 5)
paint_region (Bar, Image, Ridges, 255, 'fill')
lines_gauss (Ridges, Lines, 1.2, 1, 5, 'light')
length_xld (Lines, LineLen)
tuple_max (LineLen, Longest)
gen_image_proto (Image, Bright, 100)
set_grayval (Bright, Dim, 15, 15, 0)
local_min (Dim, Mins)
test_region_point (Mins, 15, 15, DarkInside)
paint_region (Hole, Bright, Broken, 0, 'fill')
harmonic_interpolation (Broken, Hole, Filled)
get_grayval (Filled, 22, 22, FilledGray)
transpose_region (Dot, Flipped, 0, 0)
test_region_point (Flipped, 20, 10, FlippedInside)
smallest_rectangle2_xld (Rect, RectRow, RectCol, RectPhi, Len1, Len2)
points_harris (Ridges, 1, 0.05, 0.04, 'true', SubRows, SubCols)
tuple_length (SubRows, SubCount)
gen_image_proto (Image, Mid, 128)
pow_image (Mid, Powered, 2)
get_grayval (Powered, 10, 10, PowGray)
gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 40, 30)
rank_rect (Ramp, Ranked, 3, 3, 0)
get_grayval (Ramp, 10, 10, RampGray)
get_grayval (Ranked, 10, 10, RankLow)
intersection_segments (0, 0, 10, 10, 0, 10, 10, 0, CrossRow, CrossCol, Overlap)
intersection_line_circle (0, -10, 0, 10, 0, 0, 5, CircRow, CircCol)
tuple_length (CircCol, Hits)
distance_sl (0, 0, 0, 10, 5, 0, 5, 20, LineDist)
distance_sl (0, 0, 10, 10, 5, 0, 5, 10, CrossDist)
gen_image_const (LeftImage, 'byte', 8, 6)
gen_image_const (RightImage, 'byte', 4, 6)
tile_images (LeftImage, RightImage, Wide, 'horizontal')
get_image_size (Wide, WideW, WideH)
dev_display (Ridges)
"""


HDEVELOP_BATCH11_PROGRAM = """* 濾波、骨架端點和圓的交點
gen_image_const (Image, 'byte', 80, 60)
gen_rectangle1 (Box, 10, 10, 12, 20)
get_region_runs (Box, RunRow, RunBegin, RunEnd)
tuple_length (RunRow, RunCount)
tuple_min (RunRow, RunTop)
tuple_max (RunRow, RunBot)
tuple_min (RunBegin, RunLeft)
tuple_max (RunEnd, RunRight)
gen_region_runs (Rebuilt, RunRow, RunBegin, RunEnd)
area_center (Rebuilt, RebuiltArea, SkipR, SkipC)
gen_rectangle1 (Wide, 10, 10, 30, 40)
clip_region_rel (Wide, Clipped, 2, 2, 5, 5)
test_region_point (Clipped, 12, 15, ClipIn)
test_region_point (Clipped, 10, 10, ClipOut)
gen_circle (Disk, 30, 40, 12)
inner_circle (Disk, InRow, InCol, InRad)
gen_region_line (Horiz, 30, 10, 30, 70)
gen_region_line (Vert, 10, 40, 50, 40)
union2 (Horiz, Vert, Plus)
skeleton (Plus, Thin)
junctions_skeleton (Plus, Ends, Juncs)
area_center (Ends, EndArea, EndRow, EndCol)
area_center (Juncs, JuncArea, JuncRow, JuncCol)
test_region_point (Juncs, 30, 40, JuncInside)
gen_grid_region (Grid, 10, 10, 'lines', 80, 60)
test_region_point (Grid, 0, 0, GridOn)
test_region_point (Grid, 5, 5, GridOff)
gen_grid_region (Cells, 20, 20, 'rects', 80, 60)
count_obj (Cells, CellCount)
gen_contour_polygon_xld (Wiggle, [10, 12, 9, 11, 10], [5, 15, 25, 35, 45])
regress_contours_xld (Wiggle, FitLine, 'no', 0)
get_contour_xld (FitLine, FitRows, FitCols)
tuple_mean (FitRows, FitRow)
tuple_min (FitCols, FitColMin)
tuple_max (FitCols, FitColMax)
orientation_xld (Wiggle, Phi)
intersection_circles (0, 0, 5, 0, 6, 5, CircRow, CircCol)
tuple_length (CircCol, CircHits)
intersection_segment_circle (0, -10, 0, 10, 0, 0, 5, SegRow, SegCol)
tuple_length (SegCol, SegHits)
gen_image_proto (Image, Flat, 80)
bilateral_filter (Flat, Soft, 3, 20)
get_grayval (Soft, 10, 10, SoftGray)
gen_image_proto (Image, Mid, 128)
convol_image (Mid, Smoothed, 'smooth3', 0, 'continued')
get_grayval (Smoothed, 10, 10, SmoothGray)
convol_image (Mid, Edged, 'sobel_x', 0, 'continued')
get_grayval (Edged, 10, 10, EdgeGray)
gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 40, 30)
convol_image (Ramp, Same, [0, 0, 0, 0, 1, 0, 0, 0, 0], 3, 'continued')
get_grayval (Same, 10, 10, SameGray)
dev_display (Image)
"""


HDEVELOP_BATCH12_PROGRAM = """* 內接矩形、孔洞和圓角
gen_image_const (Image, 'byte', 80, 60)
gen_rectangle1 (Solid, 10, 10, 40, 50)
inner_rectangle1 (Solid, InR1, InC1, InR2, InC2)
gen_rectangle1 (Outer, 10, 10, 40, 40)
gen_rectangle1 (Hole, 20, 20, 30, 30)
difference (Outer, Hole, Frame)
area_holes (Frame, HoleArea)
gen_contour_region_xld (Frame, OuterOnly, 'border')
gen_contour_region_xld (Frame, WithHoles, 'border_holes')
count_obj (OuterOnly, OuterCount)
count_obj (WithHoles, HoleCount)
gen_rectangle1 (Left, 10, 10, 20, 20)
gen_rectangle1 (Right, 10, 15, 20, 30)
hamming_distance (Left, Right, HamDist, HamSim)
gen_rectangle1 (Sym, 10, 10, 20, 20)
moments_region_3rd (Sym, S21, S12, S03, S30)
gen_rectangle1 (Wing, 10, 10, 12, 40)
union2 (Sym, Wing, Heavy)
moments_region_3rd (Heavy, H21, H12, H03, H30)
gen_rectangle1 (BlobA, 10, 10, 30, 20)
gen_rectangle1 (BlobB, 10, 40, 30, 55)
union2 (BlobA, BlobB, Pair)
partition_dynamic (Pair, Parts, 18, 50)
count_obj (Parts, PartCount)
area_center (Parts, PartArea, PartRow, PartCol)
gen_ellipse_contour_xld (Oval, 30, 40, 0, 20, 8, 0, rad(360), 'positive', 1)
elliptic_axis_xld (Oval, Ra, Rb, AxisPhi)
gen_contour_polygon_rounded_xld (Round, [0, 0, 20], [0, 20, 20], 5, 1)
get_contour_xld (Round, RoundRows, RoundCols)
gen_contour_polygon_xld (Chain, [0, 1, 2, 3, 4], [10, 11, 12, 13, 14])
clip_end_points_contours_xld (Chain, Clipped, 'num_points', 1)
get_contour_xld (Clipped, ClipRows, ClipCols)
tuple_length (ClipRows, ClipCount)
tuple_min (ClipRows, ClipFirst)
tuple_max (ClipRows, ClipLast)
gen_contour_polygon_xld (Span, [10, 10, 10], [0, 20, 40])
clip_end_points_contours_xld (Span, Short, 'length', 10)
get_contour_xld (Short, ShortRows, ShortCols)
tuple_min (ShortCols, ShortMin)
tuple_max (ShortCols, ShortMax)
gen_contour_polygon_xld (DiagA, [0, 10], [0, 10])
gen_contour_polygon_xld (DiagB, [0, 10], [10, 0])
intersection_contours_xld (DiagA, DiagB, CrossRow, CrossCol)
dev_display (Image)
"""


HDEVELOP_BATCH13_PROGRAM = """* 差異、字元和轉角
gen_image_const (Image, 'byte', 80, 60)
gen_image_proto (Image, Flat, 100)
set_grayval (Flat, Changed, 20, 20, 180)
check_difference (Changed, Flat, Diff, 'abs', 30)
test_region_point (Diff, 20, 20, DiffIn)
test_region_point (Diff, 10, 10, DiffOut)
check_difference (Changed, Flat, Dark, 'dark', 30)
test_region_point (Dark, 20, 20, DarkIn)
set_grayval (Image, DotImage, 40, 40, 255)
dots_image (DotImage, Dots, 5, 'light', 40)
test_region_point (Dots, 40, 40, DotIn)
test_region_point (Dots, 10, 10, DotOut)
gen_image_proto (Image, Page, 180)
gen_rectangle1 (Glyph, 15, 15, 25, 25)
paint_region (Glyph, Page, Scene, 20, 'fill')
get_domain (Scene, Domain)
char_threshold (Scene, Domain, Chars, 2, 90, Used)
test_region_point (Chars, 20, 20, CharIn)
test_region_point (Chars, 5, 5, CharOut)
gen_rectangle1 (All, 0, 0, 59, 79)
cooc_feature_image (All, Flat, 8, '0', Energy, Correlation, Homogeneity, Contrast)
gen_image_proto (Image, Bright, 255)
gen_rectangle1 (Half, 0, 0, 59, 39)
paint_region (Half, Bright, Edge, 0, 'fill')
cooc_feature_image (All, Edge, 8, '0', EdgeEnergy, EdgeCorr, EdgeHomo, EdgeContrast)
gen_rectangle1 (Left, 10, 10, 20, 20)
gen_rectangle1 (Right, 15, 15, 25, 25)
symm_difference (Left, Right, Xor)
test_region_point (Xor, 12, 12, OnlyLeft)
test_region_point (Xor, 17, 17, Overlap)
test_region_point (Xor, 22, 22, OnlyRight)
concat_obj (Left, Right, Both)
get_region_index (Both, 15, 12, IndexIn)
get_region_index (Both, 5, 5, IndexOut)
get_region_index (Both, 18, 22, IndexSecond)
gen_rectangle1 (NearA, 10, 10, 30, 18)
gen_rectangle1 (NearB, 10, 30, 30, 40)
find_neighbors (NearA, NearB, Far1, Far2, 5)
find_neighbors (NearA, NearB, Close1, Close2, 15)
count_obj (Far1, FarCount)
count_obj (Close1, CloseCount)
gen_contour_polygon_xld (Straight, [10, 10, 10], [0, 10, 20])
get_contour_angle_xld (Straight, 'abs', StraightAngles)
tuple_max (StraightAngles, StraightMax)
gen_contour_polygon_xld (Corner, [0, 0, 10], [0, 10, 10])
get_contour_angle_xld (Corner, 'rel', TurnAngles)
tuple_max (TurnAngles, TurnMax)
gen_contour_polygon_xld (OpenLine, [0, 0], [0, 10])
test_closed_xld (OpenLine, OpenFlag)
gen_contour_polygon_xld (Loop, [0, 0, 10, 0], [0, 10, 10, 0])
test_closed_xld (Loop, ClosedFlag)
dev_display (Image)
"""


HDEVELOP_BATCH14_PROGRAM = """* 邊緣、高原和光流
gen_image_const (Image, 'byte', 80, 60)
gen_empty_region (Empty)
count_obj (Empty, EmptyCount)
gen_rectangle1 (Left, 10, 10, 20, 20)
gen_rectangle1 (Right, 10, 30, 20, 40)
concat_obj (Left, Right, Both)
select_region_point (Both, Picked, 15, 35)
count_obj (Picked, PickedCount)
test_region_point (Picked, 15, 12, PickedOld)
test_region_point (Picked, 15, 35, PickedNew)
gen_rectangle1 (Block, 20, 20, 24, 24)
paint_region (Block, Image, Blocks, 200, 'fill')
plateaus (Blocks, Flats, 3)
test_region_point (Flats, 22, 22, FlatIn)
test_region_point (Flats, 20, 20, FlatEdge)
test_region_point (Flats, 5, 5, FlatBack)
gen_image_proto (Image, Dark, 0)
gen_rectangle1 (RightHalf, 0, 40, 59, 79)
paint_region (RightHalf, Dark, EdgeImage, 255, 'fill')
edges_image (EdgeImage, Amp, Dir, 'sobel', 1, 'nms', 10, 30)
get_grayval (Amp, [30, 30, 30, 30, 30, 30, 30, 30], [36, 37, 38, 39, 40, 41, 42, 43], EdgeAmps)
tuple_max (EdgeAmps, EdgePeak)
get_grayval (Amp, 30, 10, FlatAmp)
get_grayval (Dir, [30, 30, 30, 30, 30, 30, 30, 30], [36, 37, 38, 39, 40, 41, 42, 43], EdgeDirs)
gen_rectangle1 (Seed, 5, 38, 18, 42)
close_edges (Seed, EdgeImage, Closed, 20)
test_region_point (Closed, [40, 40, 40], [38, 39, 40], ClosedFlags)
tuple_max (ClosedFlags, ClosedOn)
test_region_point (Closed, 40, 10, ClosedOff)
gen_image_proto (Image, Ground, 90)
gen_rectangle1 (ValleyL, 10, 8, 40, 22)
gen_rectangle1 (ValleyR, 10, 40, 40, 54)
paint_region (ValleyL, Ground, Wells, 30, 'fill')
paint_region (ValleyR, Wells, Wells, 30, 'fill')
watersheds_threshold (Wells, LowBasins, 40)
watersheds_threshold (Wells, HighBasins, 100)
count_obj (LowBasins, LowCount)
count_obj (HighBasins, HighCount)
gen_rectangle1 (CornerBox, 12, 12, 36, 50)
paint_region (CornerBox, Image, Corners, 255, 'fill')
points_foerstner (Corners, 1.5, 0.05, CornerRows, CornerCols)
tuple_length (CornerRows, CornerCount)
tuple_min (CornerRows, CornerTop)
gen_image_proto (Image, Plain, 20)
gen_circle (Spot, 30, 30, 16)
paint_region (Spot, Plain, SpotImage, 220, 'fill')
hom_mat2d_identity (Id)
hom_mat2d_translate (Id, 0, 4, Shift)
affine_trans_image (SpotImage, Moved, Shift, 'constant', 'false')
optical_flow_mg (SpotImage, Moved, RowFlow, ColFlow, 1)
get_grayval (ColFlow, 30, 30, FlowCol)
get_grayval (RowFlow, 30, 30, FlowRow)
dev_display (EdgeImage)
"""


HDEVELOP_BATCH15_PROGRAM = """* 距離、高通和角點
gen_image_const (Image, 'byte', 80, 60)
gen_rectangle1 (Box, 20, 20, 40, 50)
distance_transform (Box, DistNear, 'euclidean', 'true')
distance_transform (Box, DistCity, 'city-block', 'true')
distance_transform (Box, DistChess, 'chessboard', 'true')
distance_transform (Box, DistInside, 'euclidean', 'false')
get_grayval (DistNear, 30, 35, NearIn)
get_grayval (DistNear, 10, 35, NearOut)
get_grayval (DistCity, 10, 35, CityOut)
get_grayval (DistChess, 10, 35, ChessOut)
get_grayval (DistInside, 30, 35, InsideDist)
get_grayval (DistInside, 10, 35, InsideOut)
region_to_bin (Box, BinImage, 200, 40)
get_grayval (BinImage, 30, 35, BinIn)
get_grayval (BinImage, 5, 5, BinOut)
gen_image_proto (Image, Blank, 0)
gen_rectangle1 (Left, 8, 8, 22, 24)
gen_rectangle1 (Right, 8, 40, 22, 60)
paint_region (Left, Blank, Painted, 80, 'fill')
paint_region (Right, Painted, Painted, 180, 'fill')
concat_obj (Left, Right, Parts)
region_to_mean (Painted, Parts, MeanImage)
get_grayval (MeanImage, 15, 16, MeanLeft)
get_grayval (MeanImage, 15, 50, MeanRight)
get_grayval (MeanImage, 40, 16, MeanOut)
highpass_image (Image, HighFlat, 5, 5)
get_grayval (HighFlat, 30, 30, HighZero)
gen_image_proto (Image, Dark, 10)
set_grayval (Dark, Spike, 30, 40, 255)
highpass_image (Spike, HighSpike, 3, 3)
get_grayval (HighSpike, 30, 40, HighPeak)
get_grayval (HighSpike, 30, 10, HighBack)
median_rect (Spike, Clean, 3, 3)
get_grayval (Clean, 30, 40, MedGray)
gen_image_proto (Image, Field, 40)
set_grayval (Field, Noisy, 30, 12, 90)
gen_rectangle1 (Bright, 0, 45, 59, 79)
paint_region (Bright, Noisy, Scene, 255, 'fill')
anisotropic_diffusion (Scene, Smooth, 'perona-malik', 25, 0.2, 12)
get_grayval (Smooth, 30, 12, SpikeLeft)
get_grayval (Smooth, 30, 70, EdgeKeep)
get_grayval (Smooth, 30, 4, Quiet)
gen_image_proto (Image, Peak, 0)
set_grayval (Peak, Peak, 29, 39, 40)
set_grayval (Peak, Peak, 29, 40, 80)
set_grayval (Peak, Peak, 29, 41, 40)
set_grayval (Peak, Peak, 30, 39, 80)
set_grayval (Peak, Peak, 30, 40, 200)
set_grayval (Peak, Peak, 30, 41, 160)
set_grayval (Peak, Peak, 31, 39, 40)
set_grayval (Peak, Peak, 31, 40, 80)
set_grayval (Peak, Peak, 31, 41, 40)
local_max_sub_pix (Peak, 'facet', 1, 20, PeakRows, PeakCols)
tuple_length (PeakRows, PeakCount)
gen_circle (Spot, 30, 40, 10)
paint_region (Spot, Image, Corners, 255, 'fill')
points_lepetit (Corners, 40, FastRows, FastCols)
tuple_length (FastRows, FastCount)
tuple_min (FastRows, FastTop)
dev_display (DistNear)
"""


HDEVELOP_BATCH16_PROGRAM = """* 短行程、平面和熵
gen_image_const (Image, 'byte', 80, 60)
gen_rectangle1 (Thin, 10, 10, 30, 11)
gen_rectangle1 (Wide, 10, 10, 30, 20)
eliminate_runs (Thin, Gone, 3)
eliminate_runs (Wide, Kept, 3)
count_obj (Gone, GoneCount)
area_center (Kept, WideArea, WideRow, WideCol)
gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 80, 60)
gen_rectangle1 (PlaneBox, 10, 10, 19, 19)
moments_gray_plane (PlaneBox, Ramp, MRow, MCol, Alpha, Beta, Mean)
plane_deviation (PlaneBox, Ramp, Deviation)
area_center_gray (PlaneBox, Ramp, GrayArea, GrayRow, GrayCol)
gen_image_proto (Image, Flat, 40)
set_grayval (Flat, Bumpy, 15, 45, 200)
gen_rectangle1 (BumpBox, 10, 40, 19, 49)
plane_deviation (BumpBox, Bumpy, BumpDev)
gen_image_proto (Image, Blank, 0)
gen_rectangle1 (RightHalf, 10, 20, 29, 29)
paint_region (RightHalf, Blank, Split, 255, 'fill')
gen_rectangle1 (Both, 10, 10, 29, 29)
entropy_gray (Both, Split, Entropy)
bandpass_image (Image, BandFlat, 3, 9)
get_grayval (BandFlat, 30, 30, BandZero)
gen_rectangle1 (DotBox, 29, 14, 31, 16)
paint_region (DotBox, Blank, Dot, 255, 'fill')
bandpass_image (Dot, BandDot, 3, 9)
get_grayval (BandDot, 30, 15, DotPeak)
get_grayval (BandDot, 30, 50, DotBack)
gen_rectangle1 (Block, 36, 36, 54, 64)
paint_region (Block, Blank, WideGray, 255, 'fill')
bandpass_image (WideGray, BandWide, 3, 9)
get_grayval (BandWide, 45, 50, WideCenter)
gen_image_proto (Image, Valley, 80)
set_grayval (Valley, Valley, [29, 29, 29, 30, 30, 30, 31, 31, 31], [39, 40, 41, 39, 40, 41, 39, 40, 41], [160, 120, 160, 120, 0, 40, 160, 120, 160])
local_min_sub_pix (Valley, 'facet', 1, 40, ValleyRows, ValleyCols)
tuple_length (ValleyRows, ValleyCount)
hom_mat2d_identity (Id)
hom_mat2d_scale (Id, 2, 3, 0, 0, Scaled)
hom_mat2d_rotate (Scaled, 0.5, 0, 0, Turned)
hom_mat2d_translate (Turned, 4, -2, Moved)
hom_mat2d_to_affine_par (Moved, Sx, Sy, Phi, Theta, Tx, Ty)
dev_display (Ramp)
"""


HDEVELOP_BATCH17_PROGRAM = """* 對數、鞍點和修剪
gen_image_const (Image, 'byte', 80, 60)
gen_image_proto (Image, Dark, 0)
set_grayval (Dark, Mark, 30, 40, 81)
sqrt_image (Mark, Root)
get_grayval (Root, 30, 40, RootGray)
get_grayval (Root, 10, 10, RootZero)
set_grayval (Dark, Fifteen, 20, 20, 15)
set_grayval (Fifteen, Fifteen, 20, 21, 255)
log_image (Fifteen, LogScaled, 'scaled')
log_image (Fifteen, LogNatural, 'natural')
get_grayval (LogScaled, 20, 20, LogMid)
get_grayval (LogScaled, 20, 21, LogHigh)
get_grayval (LogScaled, 10, 10, LogZero)
get_grayval (LogNatural, 20, 21, LogRaw)
gen_image_proto (Image, Flat, 100)
shock_filter (Flat, Still, 0.4, 4, 'laplace', 0.8)
get_grayval (Still, 30, 30, StillGray)
gen_image_proto (Image, Step, 0)
gen_rectangle1 (Right, 0, 40, 59, 79)
paint_region (Right, Step, Edge, 255, 'fill')
gauss_image (Edge, Soft, 9)
shock_filter (Soft, Sharp, 0.5, 6, 'laplace', 0)
get_grayval (Soft, 30, 39, SoftLeft)
get_grayval (Soft, 30, 41, SoftRight)
get_grayval (Sharp, 30, 39, SharpLeft)
get_grayval (Sharp, 30, 41, SharpRight)
gen_image_proto (Image, Base, 128)
set_grayval (Base, Saddle, [29, 29, 29, 30, 30, 30, 31, 31, 31], [39, 40, 41, 39, 40, 41, 39, 40, 41], [128, 127, 128, 129, 128, 129, 128, 127, 128])
saddle_points_sub_pix (Saddle, 0, 0.2, SaddleRows, SaddleCols)
tuple_length (SaddleRows, SaddleCount)
add_noise_white (Flat, NoiseA, 20, 'uniform', 1)
add_noise_white (Flat, NoiseB, 20, 'uniform', 1)
add_noise_white (Flat, Quiet, 0, 'uniform')
get_grayval (NoiseA, [30, 30, 30, 30], [10, 20, 30, 40], NoiseVals)
get_grayval (NoiseB, [30, 30, 30, 30], [10, 20, 30, 40], NoiseAgain)
get_grayval (Quiet, 30, 30, QuietGray)
gen_rectangle1 (Box, 10, 20, 40, 55)
get_region_polygon (Box, 2, PolyRows, PolyCols)
tuple_length (PolyRows, PolyCount)
tuple_min (PolyRows, PolyTop)
tuple_max (PolyRows, PolyBottom)
gen_region_line (Line, 10, 10, 10, 30)
prune (Line, Short, 3)
area_center (Short, PruneArea, PruneRow, PruneCol)
gen_rectangle1 (Outer, 10, 10, 40, 40)
gen_rectangle1 (Inner, 20, 20, 30, 30)
difference (Outer, Inner, Frame)
connect_and_holes (Frame, Parts, Holes)
count_obj (Parts, PartCount)
count_obj (Holes, HoleCount)
area_center (Holes, HoleArea, HoleRow, HoleCol)
gen_rectangle1 (LeftBlob, 8, 8, 18, 18)
gen_rectangle1 (RightBlob, 8, 40, 18, 55)
union2 (LeftBlob, RightBlob, Pair)
connect_and_holes (Pair, PairParts, PairHoles)
count_obj (PairParts, PairCount)
count_obj (PairHoles, PairHoleCount)
dev_display (Sharp)
"""


HDEVELOP_BATCH18_PROGRAM = """* 特徵、距離和自動閾值
gen_image_const (Image, 'byte', 80, 60)
gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 80, 60)
gen_rectangle1 (Box, 10, 10, 19, 19)
gray_features (Box, Ramp, ['gray_min', 'gray_max', 'gray_mean'], GrayVals)
gen_contour_polygon_xld (Square, [10, 10, 30, 30, 10], [10, 30, 30, 10, 10])
xld_features (Square, 'contlength', SideLength)
xld_features (Square, 'area', SquareArea)
gen_contour_polygon_xld (Other, [10, 10, 30, 30, 10], [50, 70, 70, 50, 50])
concat_obj (Square, Other, BothContours)
select_xld_point (BothContours, Picked, 20, 20)
select_xld_point (BothContours, Missed, 5, 5)
count_obj (Picked, PickCount)
count_obj (Missed, MissCount)
distance_lr (Box, 0, 0, 0, 80, LineDist)
distance_sr (Box, 0, 0, 0, 80, SegDist)
distance_sr (Box, 0, 0, 0, 2, EndDist)
distance_lc (Square, 0, 0, 0, 80, ContDist)
gen_image_proto (Image, Dark, 20)
gen_rectangle1 (Bright, 40, 40, 55, 60)
paint_region (Bright, Dark, TwoTone, 200, 'fill')
auto_threshold (TwoTone, Modes, 2)
count_obj (Modes, ModeCount)
get_region_index (Modes, 5, 5, DarkIndex)
get_region_index (Modes, 48, 50, BrightIndex)
gen_rectangle1 (Block, 10, 40, 14, 44)
gen_region_points (Dot, 30, 30)
union2 (Block, Dot, Scene)
gen_region_points (Hit, 1, 1)
gen_region_points (Miss, [0, 0, 0, 1, 1, 2, 2, 2], [0, 1, 2, 0, 2, 0, 1, 2])
hit_or_miss (Scene, Hit, Miss, Found)
test_region_point (Found, 30, 30, DotIn)
test_region_point (Found, 12, 42, BlockIn)
dev_display (TwoTone)
"""


HDEVELOP_BATCH19_PROGRAM = """* 伽瑪、過零和扇形
gen_image_const (Image, 'byte', 80, 60)
gen_image_proto (Image, Flat, 100)
gen_image_proto (Image, Mid, 128)
gamma_image (Mid, Darker, 2, 0, 0, 255, 'false')
get_grayval (Darker, 10, 10, GammaGray)
binomial_filter (Flat, BinFlat, 3, 3)
get_grayval (BinFlat, 10, 10, BinFlatGray)
gen_image_proto (Image, Zero, 0)
set_grayval (Zero, Dot, 30, 40, 255)
binomial_filter (Dot, BinDot, 3, 3)
get_grayval (BinDot, 30, 40, BinCenter)
get_grayval (BinDot, 5, 5, BinFar)
isotropic_diffusion (Flat, IsoFlat, 1.2, 1)
get_grayval (IsoFlat, 10, 10, IsoGray)
isotropic_diffusion (Dot, IsoDot, 1, 1)
get_grayval (IsoDot, 30, 40, IsoCenter)
get_grayval (IsoDot, 5, 5, IsoFar)
gen_image_proto (Image, Base, 10)
gen_image_proto (Image, Bright, 200)
gen_rectangle1 (Box, 10, 10, 19, 19)
reduce_domain (Bright, Box, Source)
paint_gray (Source, Base, Mixed)
get_grayval (Mixed, 15, 15, PaintIn)
get_grayval (Mixed, 5, 5, PaintOut)
gen_image_proto (Image, Step, 100)
gen_rectangle1 (Right, 0, 21, 59, 79)
paint_region (Right, Step, Signed, 156, 'fill')
gen_rectangle1 (ZeroCol, 0, 20, 59, 20)
paint_region (ZeroCol, Signed, Signed, 128, 'fill')
zero_crossing (Signed, Cross)
test_region_point (Cross, 10, 20, OnLine)
test_region_point (Cross, 10, 19, OffLine)
area_center (Cross, CrossArea, CrossRow, CrossCol)
gen_image_proto (Image, Level, 128)
gen_rectangle1 (Hot, 10, 10, 14, 14)
gen_rectangle1 (Warm, 40, 40, 44, 44)
gen_region_points (Speckle, 30, 30)
gen_rectangle1 (Cold, 10, 50, 14, 54)
paint_region (Hot, Level, Level, 200, 'fill')
paint_region (Warm, Level, Level, 150, 'fill')
paint_region (Speckle, Level, Level, 200, 'fill')
paint_region (Cold, Level, Level, 40, 'fill')
dual_threshold (Level, Lobes, 4, 40, 20)
count_obj (Lobes, LobeCount)
get_region_index (Lobes, 12, 12, HotIndex)
get_region_index (Lobes, 42, 42, WarmIndex)
get_region_index (Lobes, 30, 30, SpeckIndex)
get_region_index (Lobes, 12, 52, ColdIndex)
gen_circle_sector (Sector, 30, 40, 10, 0, rad(90))
test_region_point (Sector, 30, 48, RightIn)
test_region_point (Sector, 36, 44, DownIn)
test_region_point (Sector, 22, 40, UpOut)
test_region_point (Sector, 30, 32, LeftOut)
gen_rectangle1 (Inner, 12, 12, 18, 18)
gen_rectangle1 (Outer, 8, 8, 24, 24)
test_subset_region (Inner, Outer, IsSubset)
test_subset_region (Outer, Inner, NotSubset)
test_equal_region (Inner, Inner, IsEqual)
test_equal_region (Inner, Outer, NotEqual)
dev_display (Mixed)
"""


HDEVELOP_BATCH20_PROGRAM = """* 指數、極座標和頂帽
gen_image_const (Image, 'byte', 80, 60)
gen_image_proto (Image, Zero, 0)
gen_image_proto (Image, Mid, 128)
gen_image_proto (Image, High, 255)
exp_image (Zero, ExpZero, 'scaled')
exp_image (Mid, ExpMid, 'scaled')
exp_image (High, ExpHigh, 'scaled')
exp_image (Zero, ExpOne, 'natural')
set_grayval (Zero, Five, 8, 8, 5)
exp_image (Five, ExpFive, 'natural')
get_grayval (ExpZero, 4, 4, ExpAtZero)
get_grayval (ExpMid, 4, 4, ExpAtMid)
get_grayval (ExpHigh, 4, 4, ExpAtHigh)
get_grayval (ExpOne, 4, 4, ExpNatural)
get_grayval (ExpFive, 8, 8, ExpAtFive)
gen_region_points (Dot, 30, 50)
polar_trans_region (Dot, PolarRegion, 30, 40, 0, rad(360), 0, 20, 40, 20)
test_region_point (PolarRegion, 10, 0, PolarIn)
test_region_point (PolarRegion, 0, 0, PolarOut)
gen_contour_polygon_xld (Spoke, [30, 40, 20], [50, 40, 40])
polar_trans_contour_xld (Spoke, SpokePolar, 30, 40, 0, rad(360), 0, 20, 40, 20)
get_contour_xld (SpokePolar, PolarRows, PolarCols)
gen_ellipse_sector (EllipseSector, 30, 40, 0, 16, 8, 0, rad(90))
test_region_point (EllipseSector, 30, 52, WideIn)
test_region_point (EllipseSector, 36, 40, ThinIn)
test_region_point (EllipseSector, 22, 40, UpOut)
test_region_point (EllipseSector, 30, 28, LeftOut)
test_region_point (EllipseSector, 42, 40, FarDown)
gen_rectangle1 (Block, 10, 10, 30, 30)
gen_region_points (Speck, 5, 50)
union2 (Block, Speck, Scene)
gen_circle (Kernel, 40, 40, 1)
top_hat (Scene, Kernel, Hat)
test_region_point (Hat, 5, 50, SpeckKept)
test_region_point (Hat, 20, 20, BlockGone)
gen_rectangle1 (LeftBar, 10, 10, 20, 20)
gen_rectangle1 (RightBar, 10, 22, 20, 32)
union2 (LeftBar, RightBar, Pair)
gen_rectangle1 (Box3, 0, 0, 2, 2)
bottom_hat (Pair, Box3, Gap)
test_region_point (Gap, 15, 21, GapFilled)
test_region_point (Gap, 15, 15, BodyKept)
gen_rectangle1 (PartA, 40, 10, 48, 18)
gen_rectangle1 (PartB, 40, 30, 48, 38)
concat_obj (PartA, PartB, Parts)
region_to_label (Parts, Labels)
get_grayval (Labels, 44, 14, LabelA)
get_grayval (Labels, 44, 34, LabelB)
get_grayval (Labels, 4, 4, LabelBack)
label_to_region (Labels, Restored)
count_obj (Restored, LabelCount)
get_region_index (Restored, 44, 14, IndexA)
get_region_index (Restored, 44, 34, IndexB)
gen_contour_polygon_xld (Near, [10, 10], [10, 30])
distance_sc (Near, 0, 0, 0, 80, LineDist)
distance_sc (Near, 0, 0, 0, 2, EndDist)
gen_contour_polygon_xld (Slash, [0, 10], [0, 10])
distance_sc (Slash, 0, 10, 10, 0, CrossDist)
dev_display (Labels)
"""


HDEVELOP_BATCH21_PROGRAM = """* 塗色、角點和交點
gen_image_const (Image, 'byte', 80, 60)
gen_image_proto (Image, Base, 10)
gen_rectangle1 (Window, 10, 10, 20, 20)
reduce_domain (Base, Window, Limited)
gen_rectangle1 (Big, 5, 5, 30, 30)
overpaint_region (Big, Limited, Painted, 200, 'fill')
get_grayval (Painted, 15, 15, InDomain)
get_grayval (Painted, 6, 6, OutDomain)
gen_image_proto (Image, Flat, 100)
corner_response (Flat, FlatCorner, 3, 0.04, 0.001)
get_grayval (FlatCorner, 20, 20, FlatResp)
gen_image_proto (Image, Board, 0)
gen_rectangle1 (Square, 10, 10, 30, 30)
paint_region (Square, Board, Board, 255, 'fill')
corner_response (Board, Corners, 3, 0.04, 0.001)
get_grayval (Corners, 10, 10, CornerResp)
get_grayval (Corners, 20, 10, EdgeResp)
get_grayval (Corners, 20, 20, InsideResp)
rank_image (Flat, Ranked, 1, 0)
get_grayval (Ranked, 12, 12, RankFlat)
gen_image_proto (Image, Dark, 0)
set_grayval (Dark, Spike, 30, 40, 255)
rank_image (Spike, RankMin, 1, 0)
rank_image (Spike, RankMax, 1, 4)
get_grayval (RankMin, 30, 40, RankHole)
get_grayval (RankMax, 30, 41, RankNeighbor)
get_grayval (RankMax, 5, 5, RankFar)
gen_contour_polygon_xld (Bar, [10, 10], [10, 30])
intersection_line_contour_xld (Bar, 0, 20, 40, 20, HitRow, HitCol)
intersection_segment_contour_xld (Bar, 0, 20, 15, 20, SegRow, SegCol)
intersection_segment_contour_xld (Bar, 0, 20, 5, 20, MissRow, MissCol)
MissCount := |MissRow|
gen_contour_polygon_xld (SquareXld, [10, 10, 30, 30, 10], [10, 30, 30, 10, 10])
shape_trans_xld (SquareXld, BoxXld, 'rectangle1')
xld_features (BoxXld, 'area', BoxArea)
xld_features (BoxXld, 'contlength', BoxLength)
gen_image_proto (Image, ZeroA, 0)
gen_image_proto (Image, ZeroB, 0)
set_grayval (ZeroA, OneA, 8, 8, 30)
set_grayval (ZeroB, OneB, 8, 8, 40)
histo_2dim (OneA, OneB, Histo)
get_grayval (Histo, 30, 40, PairCount)
get_grayval (Histo, 0, 0, BackCount)
get_grayval (Histo, 1, 1, EmptyCount)
dev_display (Painted)
"""


HDEVELOP_BATCH22_PROGRAM = """* 絕對值、方向和交線
gen_image_const (Image, 'byte', 80, 60)
abs_image (Image, AbsZero)
get_grayval (AbsZero, 4, 4, AbsAtZero)
gen_image_proto (Image, Mid, 128)
gen_image_proto (Image, Low, 100)
abs_image (Mid, AbsMid)
abs_image (Low, AbsLow)
get_grayval (AbsMid, 4, 4, AbsAtMid)
get_grayval (AbsLow, 4, 4, AbsAtLow)
gen_image_proto (Image, Right, 255)
gen_image_proto (Image, Down, 255)
gen_image_proto (Image, Level, 128)
atan2_image (Level, Right, DirRight)
atan2_image (Down, Level, DirDown)
get_grayval (DirRight, 4, 4, RightDir)
get_grayval (DirDown, 4, 4, DownDir)
gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 80, 60)
get_grayval_interpolated (Ramp, 10, 10.5, MidGray)
get_grayval_interpolated (Ramp, 10, 10.5, 'nearest', NearGray)
gen_image_proto (Image, Dest, 10)
gen_rectangle1 (Box, 10, 10, 19, 19)
reduce_domain (Dest, Box, Limited)
gen_image_proto (Image, Src, 200)
overpaint_gray (Limited, Src, Mixed)
get_grayval (Mixed, 15, 15, InPaint)
get_grayval (Mixed, 5, 5, OutPaint)
intersection_lines (0, 0, 0, 80, 0, 20, 40, 20, CrossRow, CrossCol, CrossHit)
intersection_lines (0, 0, 0, 80, 5, 0, 5, 80, ParRow, ParCol, ParHit)
gen_contour_polygon_xld (OnCircle, [30, 40], [50, 40])
gen_contour_polygon_xld (AtCenter, [30, 30], [40, 40])
dist_ellipse_contour_xld (OnCircle, 30, 40, 0, 10, 10, MaxOn)
dist_ellipse_contour_xld (AtCenter, 30, 40, 0, 10, 10, MaxOff)
gen_contour_polygon_xld (BoxXld, [10, 10, 30, 30, 10], [10, 30, 30, 10, 10])
gen_contour_polygon_xld (Bow, [10, 30, 30, 10, 10], [10, 30, 10, 30, 10])
concat_obj (BoxXld, Bow, Shapes)
test_self_intersection_xld (Shapes, SelfFlags)
critical_points_xld (BoxXld, 0.5, CritRow, CritCol)
CritCount := |CritRow|
gen_contour_polygon_xld (Straight, [10, 10, 10], [10, 20, 30])
critical_points_xld (Straight, 0.5, LineRow, LineCol)
LineCount := |LineRow|
dev_display (Mixed)
"""


HDEVELOP_BATCH23_PROGRAM = """* 普瑞維特、極座標和拆線
gen_image_const (Image, 'byte', 80, 60)
gen_rectangle1 (Half, 0, 40, 59, 79)
paint_region (Half, Image, Stepped, 40, 'fill')
prewitt_amp (Stepped, Prewitt)
kirsch_amp (Stepped, Kirsch)
roberts (Stepped, RobertsSqrt, 'sum_sqrt')
roberts (Stepped, RobertsSum, 'sum_abs')
get_grayval (Prewitt, 30, 39, PrewittEdge)
get_grayval (Prewitt, 30, 10, PrewittFlat)
get_grayval (Kirsch, 30, 39, KirschEdge)
get_grayval (Kirsch, 30, 10, KirschFlat)
get_grayval (RobertsSqrt, 30, 39, RobertsSqrtEdge)
get_grayval (RobertsSum, 30, 39, RobertsSumEdge)
intersection_segment_line (10, 0, 10, 40, 0, 20, 40, 20, HitRow, HitCol, HitFlag)
intersection_segment_line (10, 0, 10, 10, 0, 20, 40, 20, MissRow, MissCol, MissFlag)
gen_contour_polygon_xld (Corner, [35, 35], [50, 50])
gen_contour_polygon_xld (Center, [30, 30], [40, 40])
gen_contour_polygon_xld (Outside, [30, 30], [56, 56])
dist_rectangle2_contour_xld (Corner, 30, 40, 0, 10, 5, DistOn)
dist_rectangle2_contour_xld (Center, 30, 40, 0, 10, 5, DistCenter)
dist_rectangle2_contour_xld (Outside, 30, 40, 0, 10, 5, DistOut)
gen_contour_polygon_xld (BoxXld, [10, 10, 30, 30, 10], [10, 30, 30, 10, 10])
split_contours_xld (BoxXld, Pieces, 0.5)
count_obj (Pieces, PieceCount)
length_xld (Pieces, PieceLen)
gen_contour_polygon_xld (Straight, [10, 10, 10], [10, 20, 30])
split_contours_xld (Straight, OneLine, 0.5)
count_obj (OneLine, LinePieces)
gen_region_points (Dot, 30, 50)
polar_trans_region (Dot, PolarRegion, 30, 40, 0, rad(360), 0, 20, 40, 20)
polar_trans_region_inv (PolarRegion, BackRegion, 30, 40, 0, rad(360), 0, 20, 80, 60)
test_region_point (BackRegion, 30, 50, BackInside)
test_region_point (BackRegion, 30, 40, BackCenter)
set_grayval (Image, Marked, 30, 50, 255)
polar_trans_image_ext (Marked, Polar, 30, 40, 0, rad(360), 0, 20, 40, 20, 'nearest_neighbor')
polar_trans_image_inv (Polar, Back, 30, 40, 0, rad(360), 0, 20, 80, 60, 'nearest_neighbor')
get_grayval (Back, 30, 50, BackGray)
get_grayval (Back, 30, 40, CenterGray)
dev_display (Prewitt)
"""


HDEVELOP_BATCH24_PROGRAM = """* 羅賓遜、方位和交圓
gen_image_const (Image, 'byte', 80, 60)
gen_rectangle1 (Half, 0, 40, 59, 79)
paint_region (Half, Image, Stepped, 40, 'fill')
frei_amp (Stepped, Frei)
robinson_amp (Stepped, Robinson)
robinson_dir (Stepped, RobinsonDir)
get_grayval (Frei, 30, 39, FreiEdge)
get_grayval (Frei, 30, 10, FreiFlat)
get_grayval (Robinson, 30, 39, RobinsonEdge)
get_grayval (Robinson, 30, 10, RobinsonFlat)
get_grayval (RobinsonDir, 30, 39, RobinsonRight)
gen_rectangle1 (Left, 10, 5, 20, 15)
gen_rectangle1 (Mark, 10, 30, 20, 45)
gen_rectangle1 (Top, 0, 30, 4, 45)
gen_rectangle1 (Bottom, 40, 30, 50, 45)
concat_obj (Left, Top, Pair)
concat_obj (Pair, Bottom, Candidates)
select_region_spatial (Candidates, Mark, ChosenLeft, 'left')
select_region_spatial (Candidates, Mark, ChosenAbove, 'above')
select_region_spatial (Candidates, Mark, ChosenBelow, 'below')
select_region_spatial (Candidates, Mark, ChosenRight, 'right')
count_obj (ChosenLeft, LeftCount)
count_obj (ChosenAbove, AboveCount)
count_obj (ChosenBelow, BelowCount)
count_obj (ChosenRight, RightCount)
test_region_point (ChosenLeft, 15, 10, OnLeft)
test_region_point (ChosenLeft, 15, 35, OnMark)
gen_contour_polygon_xld (Spoke, [30, 40, 20], [50, 40, 40])
polar_trans_contour_xld (Spoke, PolarSpoke, 30, 40, 0, rad(360), 0, 20, 40, 20)
polar_trans_contour_xld_inv (PolarSpoke, BackSpoke, 30, 40, 0, rad(360), 0, 20, 40, 20)
get_contour_xld (BackSpoke, BackRow, BackCol)
gen_contour_polygon_xld (Across, [30, 30], [20, 60])
gen_contour_polygon_xld (MissLine, [0, 5], [0, 1])
intersection_circle_contour_xld (Across, 30, 40, 10, CircleRow, CircleCol)
intersection_circle_contour_xld (MissLine, 30, 40, 10, MissRow, MissCol)
MissCount := |MissRow|
gen_contour_polygon_xld (BoxXld, [10, 10, 30, 30, 10], [10, 30, 30, 10, 10])
contour_point_num_xld (BoxXld, BoxPoints)
contour_point_num_xld (Spoke, SpokePoints)
BoxCount := BoxPoints[1]
SpokeCount := SpokePoints[1]
gen_image_proto (Image, Flat, 100)
add_noise_salt_pepper (Flat, Quiet, 0, 1)
get_grayval (Quiet, 4, 4, QuietGray)
add_noise_salt_pepper (Flat, NoiseA, 100, 7)
add_noise_salt_pepper (Flat, NoiseB, 100, 7)
sub_image (NoiseA, NoiseB, Diff, 1, 0)
gen_rectangle1 (Full, 0, 0, 59, 79)
min_max_gray (Full, Diff, 0, DiffMin, DiffMax, DiffRange)
threshold (NoiseA, Salt, 255, 255)
threshold (NoiseA, Pepper, 0, 0)
area_center (Salt, SaltArea, SaltRow, SaltCol)
area_center (Pepper, PepperArea, PepperRow, PepperCol)
dev_display (Robinson)
"""


HDEVELOP_BATCH25_PROGRAM = """* 對照表、位元和金字塔
gen_image_const (Image, 'byte', 80, 60)
gen_image_proto (Image, Ten, 10)
gen_image_proto (Image, One, 1)
gen_image_proto (Image, Sixteen, 16)
gen_image_proto (Image, High, 240)
gen_image_proto (Image, Low, 15)
bit_not (Ten, NotImage)
bit_and (High, Low, AndImage)
bit_or (High, Low, OrImage)
bit_xor (High, Low, XorImage)
bit_lshift (One, ShiftImage, 3)
bit_rshift (Sixteen, UnshiftImage, 2)
get_grayval (NotImage, 4, 4, NotGray)
get_grayval (AndImage, 4, 4, AndGray)
get_grayval (OrImage, 4, 4, OrGray)
get_grayval (XorImage, 4, 4, XorGray)
get_grayval (ShiftImage, 4, 4, ShiftGray)
get_grayval (UnshiftImage, 4, 4, UnshiftGray)
tuple_gen_sequence (0, 255, 1, Seq)
tuple_sub (255, Seq, Inv)
lut_trans (Ten, LutImage, Inv)
get_grayval (LutImage, 4, 4, LutGray)
gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 80, 60)
gen_gauss_pyramid (Ramp, PlainPyr, 'none', 0.5)
gen_gauss_pyramid (Ramp, MinPyr, 'min', 0.5)
gen_gauss_pyramid (Ramp, MaxPyr, 'max', 0.5)
gen_gauss_pyramid (Ten, ConstPyr, 'constant', 0.5)
get_grayval (PlainPyr, 4, 3, PlainGray)
get_grayval (MinPyr, 4, 3, MinGray)
get_grayval (MaxPyr, 4, 3, MaxGray)
get_grayval (ConstPyr, 4, 3, ConstGray)
get_image_size (ConstPyr, PyrWidth, PyrHeight)
dev_display (ConstPyr)
"""


HDEVELOP_BATCH26_PROGRAM = """* 量測剖面和一維函數
gen_image_const (Image, 'byte', 80, 60)
gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 80, 60)
gen_measure_rectangle2 (30, 40, 0, 10, 0, 80, 60, 'bilinear', Rect)
measure_projection (Ramp, Rect, Profile)
ProfileCount := |Profile|
ProfileFirst := Profile[1]
ProfileMid := Profile[21]
ProfileLast := Profile[41]
measure_thresh (Ramp, Rect, 0, 40, 'all', ThreshRow, ThreshCol, ThreshDist)
translate_measure (Rect, 30, 50)
measure_projection (Ramp, Rect, Moved)
MovedFirst := Moved[1]
MovedLast := Moved[41]
gen_measure_arc (30, 40, 10, 0, rad(180), 0, 80, 60, 'bilinear', Arc)
measure_projection (Ramp, Arc, ArcProfile)
ArcCount := |ArcProfile|
ArcFirst := ArcProfile[1]
ArcMid := ArcProfile[32]
ArcLast := ArcProfile[63]
create_funct_1d_pairs ([0, 2], [10, 30], Curve)
get_y_value_funct_1d (Curve, 1, 'constant', MidY)
get_y_value_funct_1d (Curve, -1, 'constant', LowY)
get_y_value_funct_1d (Curve, 3, 'zero', HighY)
get_y_value_funct_1d (Curve, -1, 'mirror', MirrorY)
sample_funct_1d (Curve, 0, 2, 1, 'constant', Sampled)
S1 := Sampled[1]
S2 := Sampled[2]
S3 := Sampled[3]
create_funct_1d_pairs ([0, 2], [-1, 1], Line)
zero_crossings_funct_1d (Line, Zeros)
ZeroX := Zeros[1]
dev_display (Ramp)
"""


HDEVELOP_BATCH27_PROGRAM = """* 函數積分、導數和反函數
gen_image_const (Image, 'byte', 80, 60)
create_funct_1d_pairs ([0, 2], [10, 30], Curve)
integrate_funct_1d (Curve, 0, 2, Area)
derivate_funct_1d (Curve, 'second', Curve2)
get_y_value_funct_1d (Curve2, 1, 'constant', Flat2)
invert_funct_1d (Curve, Inv)
get_y_value_funct_1d (Inv, 20, 'constant', InvY)
create_funct_1d_pairs ([0, 2], [0, 2], Ident)
compose_funct_1d (Curve, Ident, 'constant', Mixed)
get_y_value_funct_1d (Mixed, 1, 'constant', MixedY)
create_funct_1d_array ([4, 8, 6], ArrayFn)
funct_1d_to_pairs (ArrayFn, ArrayX, ArrayY)
ArrayX0 := ArrayX[1]
ArrayY1 := ArrayY[2]
get_y_value_funct_1d (ArrayFn, 1, 'constant', ArrayMid)
create_funct_1d_pairs ([0, 1, 2], [0, 2, 2], Kink)
derivate_funct_1d (Kink, 'first', Slope)
get_y_value_funct_1d (Slope, 0, 'constant', Slope0)
get_y_value_funct_1d (Slope, 1, 'constant', Slope1)
get_y_value_funct_1d (Slope, 2, 'constant', Slope2)
derivate_funct_1d (Kink, 'second', Bend2)
get_y_value_funct_1d (Bend2, 1, 'constant', SecondMid)
create_funct_1d_pairs ([0, 1, 2, 3], [0, 2, -1, 0], Bend)
local_min_max_funct_1d (Bend, MinX, MaxX)
MinAt := MinX[1]
MaxAt := MaxX[1]
create_funct_1d_pairs ([0, 2], [-1, 1], Line)
abs_funct_1d (Line, AbsLine)
get_y_value_funct_1d (AbsLine, 0, 'constant', Abs0)
get_y_value_funct_1d (AbsLine, 1, 'constant', AbsMid)
get_y_value_funct_1d (AbsLine, 2, 'constant', Abs2)
dev_display (Image)
"""


HDEVELOP_BATCH28_PROGRAM = """* 傅立葉、平移和相位相關
gen_image_const (Image, 'byte', 80, 60)
gen_image_proto (Image, Ten, 10)
fft_image (Ten, Spec)
get_complex (Spec, 0, 0, DcReal, DcImag)
get_complex (Spec, 0, 1, AcReal, AcImag)
fftshift (Spec, 'center', Centered)
get_complex (Centered, 30, 40, ShiftReal, ShiftImag)
fft_image_inv (Back, Spec)
get_grayval (Back, 4, 4, BackGray)
scale_fft (Spec, 0.5, HalfSpec)
fft_image_inv (Half, HalfSpec)
get_grayval (Half, 4, 4, HalfGray)
gen_complex_const (1, 0, 80, 60, Ones)
mult_fft (Spec, Ones, Product)
get_complex (Product, 0, 0, ProdReal, ProdImag)
gen_image_const (Small, 'byte', 5, 5)
gen_image_proto (Small, Five, 10)
fft_image (Five, SmallSpec)
fftshift (SmallSpec, 'center', Mid)
fftshift (SmallSpec, 'corner', Corner)
get_complex (Mid, 2, 2, MidReal, MidImag)
get_complex (Mid, 0, 0, MidZero, MidZeroImag)
get_complex (Corner, 3, 3, CornerReal, CornerImag)
get_complex (Corner, 2, 2, CornerOff, CornerOffImag)
gen_image_const (Blank, 'byte', 32, 32)
set_grayval (Blank, Dots, 10, 12, 255)
set_grayval (Blank, Moved, 15, 14, 255)
phase_correlation (Dots, Moved, PeakRow, PeakCol, PeakScore)
dev_display (Dots)
"""


HDEVELOP_BATCH29_PROGRAM = """* 頻譜遮罩、卷積和能量
gen_image_const (Image, 'byte', 80, 60)
gen_image_proto (Image, Ten, 10)
fft_image (Ten, Spec)
conj_fft (Spec, Conj)
get_complex (Conj, 0, 0, ConjReal, ConjImag)
add_fft (Spec, Spec, SumSpec)
fft_image_inv (SumImage, SumSpec)
get_grayval (SumImage, 4, 4, SumGray)
sub_fft (Spec, Spec, ZeroSpec)
fft_image_inv (ZeroImage, ZeroSpec)
get_grayval (ZeroImage, 4, 4, ZeroGray)
mask_fft (Spec, 'dc', DcSpec)
fft_image_inv (DcImage, DcSpec)
get_grayval (DcImage, 4, 4, DcGray)
mask_fft (Spec, 'no_dc', HighSpec)
fft_image_inv (HighImage, HighSpec)
get_grayval (HighImage, 4, 4, HighGray)
mask_fft (Spec, 'real', RealSpec)
fft_image_inv (RealImage, RealSpec)
get_grayval (RealImage, 4, 4, RealGray)
mask_fft (Spec, 'imag', ImagSpec)
fft_image_inv (ImagImage, ImagSpec)
get_grayval (ImagImage, 4, 4, ImagGray)
fft_energy (Spec, Energy)
gen_complex_const (1, 0, 80, 60, Ones)
div_fft (Spec, Ones, Quot)
fft_image_inv (QuotImage, Quot)
get_grayval (QuotImage, 4, 4, QuotGray)
complex_to_image (RealByte, Spec, 'real')
get_grayval (RealByte, 0, 0, RealPeak)
get_grayval (RealByte, 0, 1, RealFlat)
complex_to_image (LogByte, Spec, 'log')
get_grayval (LogByte, 0, 0, LogPeak)
get_grayval (LogByte, 0, 1, LogFlat)
gen_image_const (Blank, 'byte', 32, 32)
set_grayval (Blank, Dot, 10, 12, 255)
gen_image_const (Kernel, 'byte', 4, 4)
set_grayval (Kernel, ShiftK, 2, 3, 1)
convol_fft (Dot, ShiftK, Shifted)
get_grayval (Shifted, 12, 15, MovedGray)
get_grayval (Shifted, 10, 12, OldGray)
dev_display (Shifted)
"""


HDEVELOP_BATCH30_PROGRAM = """* 相關、移頻和補邊
gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 80, 60)
fft_image (Ramp, RampSpec)
freq_filter (RampSpec, 'low', 0, LowSpec)
fft_image_inv (LowImage, LowSpec)
get_grayval (LowImage, 4, 4, LowGray)
get_grayval (LowImage, 4, 79, LowEdge)
freq_filter (RampSpec, 'high', 0, HighSpec)
fft_image_inv (HighImage, HighSpec)
get_grayval (HighImage, 4, 0, HighDark)
get_grayval (HighImage, 4, 79, HighBright)
freq_energy (RampSpec, 0, DcEnergy)
gen_image_const (Blank, 'byte', 32, 32)
set_grayval (Blank, Dot, 10, 12, 255)
shift_image (Dot, Rolled, 5, 2)
get_grayval (Rolled, 15, 14, RollGray)
get_grayval (Rolled, 10, 12, RollOld)
fft_image (Dot, DotSpec)
translate_fft (DotSpec, 5, 2, MovedSpec)
fft_image_inv (Moved, MovedSpec)
get_grayval (Moved, 15, 14, PhaseGray)
get_grayval (Moved, 10, 12, PhaseOld)
normalize_fft (DotSpec, Unit)
get_complex (Unit, 0, 0, UnitReal, UnitImag)
get_complex (Unit, 0, 1, SideReal, SideImag)
tuple_pow (SideReal, 2, SqR)
tuple_pow (SideImag, 2, SqI)
tuple_add (SqR, SqI, SumSq)
tuple_sqrt (SumSq, UnitMag)
gen_image_const (Kernel, 'byte', 4, 4)
set_grayval (Kernel, Mark, 2, 3, 255)
locate_template (Dot, Mark, MatchRow, MatchCol, MatchScore)
correlation_image (Dot, Mark, ScoreImage)
get_grayval (ScoreImage, 8, 9, ScorePeak)
get_grayval (ScoreImage, 12, 15, ScoreOff)
gen_image_const (Small, 'byte', 4, 4)
set_grayval (Small, Edge, 1, 0, 200)
pad_image (Edge, ConstPad, 1, 0, 0, 0, 0, 'constant')
get_grayval (ConstPad, 0, 0, ConstCorner)
get_grayval (ConstPad, 2, 0, ConstMoved)
get_image_size (ConstPad, ConstWidth, ConstHeight)
pad_image (Edge, MirrorPad, 1, 0, 0, 0, 0, 'mirror')
get_grayval (MirrorPad, 0, 0, MirrorCorner)
get_grayval (MirrorPad, 2, 0, MirrorMoved)
pad_image (Edge, CycPad, 0, 0, 0, 1, 0, 'cyclic')
get_grayval (CycPad, 1, 0, CycOld)
get_grayval (CycPad, 1, 4, CycWrap)
get_image_size (CycPad, CycWidth, CycHeight)
dev_display (ScoreImage)
"""


HDEVELOP_BATCH31_PROGRAM = """* 點雲、平面和包圍盒
gen_image_const (Image, 'byte', 80, 60)
gen_object_model_3d_from_points ([0, 2, 0, 2], [0, 0, 2, 2], [3, 3, 3, 3], Cloud)
fit_plane_object_model_3d (Cloud, Nx, Ny, Nz, PlaneDist)
distance_plane_object_model_3d (Cloud, Nx, Ny, Nz, PlaneDist, OnPlane)
OnDist := OnPlane[1]
PlaneCount := |OnPlane|
gen_object_model_3d_from_points (0, 0, 1, Below)
distance_plane_object_model_3d (Below, 0, 0, 1, 3, OffList)
OffDist := OffList[1]
gen_object_model_3d_from_points ([0, 3], [0, 0], [0, 0], Pair)
distance_point_object_model_3d (Pair, 0, 4, 0, NearDist)
translate_object_model_3d (Pair, 10, -1, 2, Moved)
get_object_model_3d_params (Moved, 'point_coord_x', MovedX)
get_object_model_3d_params (Moved, 'point_coord_y', MovedY)
get_object_model_3d_params (Moved, 'point_coord_z', MovedZ)
MovedX0 := MovedX[1]
MovedX1 := MovedX[2]
MovedY0 := MovedY[1]
MovedZ0 := MovedZ[1]
mirror_object_model_3d (Pair, 'x', Flipped)
get_object_model_3d_params (Flipped, 'point_coord_x', FlipX)
FlipX0 := FlipX[1]
FlipX1 := FlipX[2]
gen_object_model_3d_from_points ([0, 1, 2, 3], 0, 0, Line)
sample_object_model_3d (Line, 2, Kept)
get_object_model_3d_params (Kept, 'point_coord_x', KeptX)
KeptX0 := KeptX[1]
KeptX1 := KeptX[2]
get_object_model_3d_params (Kept, 'num_points', KeptCount)
gen_box_object_model_3d (0, 1, 2, 4, 5, 6, Box)
get_object_model_3d_params (Box, 'num_points', BoxCount)
smallest_bounding_box_object_model_3d (Box, MinX, MinY, MinZ, MaxX, MaxY, MaxZ)
dev_display (Image)
"""


HDEVELOP_BATCH32_PROGRAM = """* 點雲縮放、投影和主軸
gen_image_const (Image, 'byte', 80, 60)
gen_object_model_3d_from_points (2, 0, 0, One)
scale_object_model_3d (One, 3, 1, 1, Scaled)
get_object_model_3d_params (Scaled, 'point_coord_x', ScaleX)
scale_object_model_3d (One, 3, 1, 1, 2, 0, 0, Held)
get_object_model_3d_params (Held, 'point_coord_x', HeldX)
gen_object_model_3d_from_points (1, 0, 0, Arm)
rotate_object_model_3d (Arm, 'z', rad(90), Turned)
get_object_model_3d_params (Turned, 'point_coord_x', TurnX)
get_object_model_3d_params (Turned, 'point_coord_y', TurnY)
gen_object_model_3d_from_points (1, 2, 3, Tip)
project_object_model_3d (Tip, 'xy', Flat)
get_object_model_3d_params (Flat, 'point_coord_x', FlatX)
get_object_model_3d_params (Flat, 'point_coord_y', FlatY)
get_object_model_3d_params (Flat, 'point_coord_z', FlatZ)
gen_object_model_3d_from_points (0, 0, [3, 1], Layer)
select_plane_object_model_3d (Layer, 0, 0, 1, 3, 0.5, Kept)
get_object_model_3d_params (Kept, 'num_points', KeptCount)
get_object_model_3d_params (Kept, 'point_coord_z', KeptZ)
gen_object_model_3d_from_points ([0, 3], 0, 0, Pair)
max_distance_point_object_model_3d (Pair, 0, 4, 0, FarDist)
gen_object_model_3d_from_points ([0, 1, 2], 0, 0, Line)
gen_object_model_3d_from_points (1, 0, 0, Drop)
difference_object_model_3d (Line, Drop, 0, Rest)
get_object_model_3d_params (Rest, 'point_coord_x', RestX)
RestX0 := RestX[1]
RestX1 := RestX[2]
sort_object_model_3d (Line, 'x', 'descending', Ordered)
get_object_model_3d_params (Ordered, 'point_coord_x', OrderX)
OrderX0 := OrderX[1]
OrderX1 := OrderX[2]
gen_object_model_3d_from_points ([0, 2, 4], 0, 0, Shaft)
principal_axis_object_model_3d (Shaft, Ax, Ay, Az)
dev_display (Image)
"""


HDEVELOP_BATCH33_PROGRAM = """* 點雲交集、格子和主軸長度
gen_image_const (Image, 'byte', 80, 60)
gen_object_model_3d_from_points ([0, 1, 2], 0, 0, Line)
gen_object_model_3d_from_points (1, 0, 0, Hit)
intersect_object_model_3d (Line, Hit, 0, Common)
get_object_model_3d_points (Common, CommonX, CommonY, CommonZ)
gen_object_model_3d_from_points ([0, 2], 0, 0, Pair)
center_object_model_3d (Pair, Mid)
get_object_model_3d_points (Mid, MidX, MidY, MidZ)
MidX0 := MidX[1]
MidX1 := MidX[2]
gen_object_model_3d_from_points ([0, 2, 5], 0, 0, Wide)
clip_object_model_3d (Wide, 1, 0, 0, 3, 0, 0, Cut)
get_object_model_3d_points (Cut, CutX, CutY, CutZ)
gen_object_model_3d_from_points ([0, 0.5, 3], 0, 0, Dots)
voxel_object_model_3d (Dots, 2, 'point', Sparse)
get_object_model_3d_points (Sparse, SparseX, SparseY, SparseZ)
SparseX0 := SparseX[1]
SparseX1 := SparseX[2]
voxel_object_model_3d (Dots, 2, 'center', Cells)
get_object_model_3d_points (Cells, CellX, CellY, CellZ)
CellX0 := CellX[1]
CellX1 := CellX[2]
gen_object_model_3d_from_points ([0, 10], 0, 0, Far)
gen_object_model_3d_from_points ([0, 1], 0, 0, Near)
max_distance_object_model_3d (Far, Near, Gap)
gen_plane_object_model_3d (0, 0, 3, 2, 2, 1, Sheet)
get_object_model_3d_params (Sheet, 'num_points', SheetCount)
get_object_model_3d_points (Sheet, SheetX, SheetY, SheetZ)
SheetX0 := SheetX[1]
SheetX1 := SheetX[2]
SheetY2 := SheetY[3]
SheetZ0 := SheetZ[1]
gen_object_model_3d_from_points ([0, 2, 4], 0, 0, Shaft)
principal_spread_object_model_3d (Shaft, Spread1, Spread2, Spread3)
dev_display (Image)
"""


HDEVELOP_BATCH34_PROGRAM = """* 點雲和影像對齊
gen_image_const (Image, 'byte', 80, 60)
gen_rectangle1 (Rect, 10, 20, 12, 22)
gen_object_model_3d_from_region (Rect, RegionCloud)
get_object_model_3d_params (RegionCloud, 'num_points', RegionCount)
get_object_model_3d_points (RegionCloud, Rx, Ry, Rz)
Rx0 := Rx[1]
Ry0 := Ry[1]
Rz0 := Rz[1]
gen_region_object_model_3d (Back, RegionCloud, 80, 60)
area_center (Back, Area, AreaRow, AreaCol)
BackArea := Area[1]
gen_contour_polygon_xld (Contour, [10, 30], [20, 40])
gen_object_model_3d_from_xld (Contour, XldCloud)
get_object_model_3d_points (XldCloud, Xx, Xy, Xz)
Xx0 := Xx[1]
Xy0 := Xy[1]
Xx1 := Xx[2]
Xy1 := Xy[2]
gen_image_const (Tiny, 'byte', 2, 2)
set_grayval (Tiny, Marked, 0, 1, 40)
set_grayval (Marked, Marked2, 1, 0, 10)
gen_object_model_3d_from_image (Marked2, ImageCloud)
get_object_model_3d_params (ImageCloud, 'num_points', ImageCount)
get_object_model_3d_points (ImageCloud, PixX, PixY, PixZ)
PixZ1 := PixZ[2]
PixY2 := PixY[3]
PixZ2 := PixZ[3]
gen_object_model_3d_from_points (5, 6, 200, Spot)
paint_object_model_3d (Image, Painted, Spot, 0, 'z')
get_grayval (Painted, 6, 5, PaintZ)
paint_object_model_3d (Image, PaintedGray, Spot, 80, 'gray')
get_grayval (PaintedGray, 6, 5, PaintGray)
get_grayval_object_model_3d (Painted, Spot, ReadZ)
select_object_model_3d_region (Rect, RegionCloud, 'inside', Inside)
get_object_model_3d_params (Inside, 'num_points', InsideCount)
gen_object_model_3d_from_points (0, 0, 0, Origin)
select_object_model_3d_region (Rect, Origin, 'outside', Outside)
get_object_model_3d_params (Outside, 'num_points', OutsideCount)
select_object_model_3d_region (Rect, Origin, 'inside', Miss)
get_object_model_3d_params (Miss, 'num_points', MissCount)
gen_object_model_3d_from_points (1, 0, 0, Seed)
set_object_model_3d_z (Marked2, Seed, Lifted)
get_object_model_3d_points (Lifted, LiftX, LiftY, LiftZ)
dev_display (Painted)
"""


HDEVELOP_BATCH35_PROGRAM = """* 高度、法線和明暗
gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 80, 60)
fit_image_plane (Ramp, Alpha, Beta, Gamma)
sub_image_plane (Ramp, Flat, 128)
get_grayval (Flat, 0, 0, FlatCorner)
get_grayval (Flat, 30, 40, FlatMid)
image_plane_deviation (Ramp, Deviation)
image_plane_normal (Ramp, PlaneNx, PlaneNy, PlaneNz)
height_slope_image (Ramp, Slope, 1)
get_grayval (Slope, 30, 40, SlopeMid)
height_slope_image (Ramp, Slope2, 2)
get_grayval (Slope2, 30, 40, SlopeGentle)
gen_image_proto (Ramp, Plain, 0)
height_slope_image (Plain, FlatSlope, 1)
get_grayval (FlatSlope, 0, 0, SlopeZero)
height_normal_image (Ramp, Nx, Ny, Nz, 1)
get_grayval (Nx, 30, 40, NxMid)
get_grayval (Ny, 30, 40, NyMid)
get_grayval (Nz, 30, 40, NzMid)
height_normal_image (Plain, FNx, FNy, FNz, 1)
get_grayval (FNx, 10, 10, UpNx)
get_grayval (FNy, 10, 10, UpNy)
get_grayval (FNz, 10, 10, UpNz)
shade_height_field (Plain, Sky, 0, 90, 1, 'clamp')
get_grayval (Sky, 10, 10, SkyGray)
shade_height_field (Ramp, Shaded, 0, 90, 1, 'clamp')
get_grayval (Shaded, 30, 40, ShadeMid)
shade_height_field (Ramp, Dark, 0, 0, 1, 'clamp')
get_grayval (Dark, 30, 40, ShadeSide)
shade_height_field (Ramp, Both, 0, 90, 1, 'full')
get_grayval (Both, 30, 40, ShadeFull)
shade_height_field (Ramp, Back, 0, 0, 1, 'full')
get_grayval (Back, 30, 40, ShadeBack)
above_image_plane (Ramp, NoneAbove, 'above')
count_obj (NoneAbove, AboveCount)
gen_rectangle1 (Rect, 15, 20, 44, 59)
paint_region (Rect, Plain, Block, 200, 'fill')
shade_height_field (Block, Relief, 0, 20, 1, 'clamp')
get_grayval (Relief, 30, 40, ReliefMid)
gen_image_const (Bump, 'byte', 3, 3)
set_grayval (Bump, Peak, 1, 1, 9)
fit_image_plane (Peak, BumpA, BumpB, BumpG)
image_plane_deviation (Peak, BumpDev)
sub_image_plane (Peak, BumpFlat, 128)
get_grayval (BumpFlat, 1, 1, BumpMid)
get_grayval (BumpFlat, 0, 0, BumpCorner)
above_image_plane (Peak, Above, 'above')
area_center (Above, AboveArea, AboveRow, AboveCol)
AboveN := AboveArea[1]
AboveR := AboveRow[1]
AboveC := AboveCol[1]
above_image_plane (Peak, Below, 'below')
area_center (Below, BelowArea, BelowRow, BelowCol)
BelowN := BelowArea[1]
dev_display (Relief)
"""


HDEVELOP_BATCH36_PROGRAM = """* 參考平面、傾角和體積
gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 80, 60)
sub_plane_image (Ramp, Level, 0, 1, 0, 128)
get_grayval (Level, 0, 0, LevelCorner)
get_grayval (Level, 30, 40, LevelMid)
sub_plane_image (Ramp, Shifted, 0, 0, 10, 128)
get_grayval (Shifted, 0, 0, Shift0)
get_grayval (Shifted, 0, 40, Shift40)
gen_image_proto (Ramp, Base, 10)
add_plane_image (Base, Raised, 0, 1, 0)
get_grayval (Raised, 4, 6, RaisedGray)
add_plane_image (Base, Clipped, 0, 0, 300)
get_grayval (Clipped, 0, 0, ClipGray)
gen_image_plane (Ramp, Made, 0, 1, 5)
get_grayval (Made, 0, 0, Made0)
get_grayval (Made, 2, 3, Made3)
plane_gray (0, 1, 5, 2, 3, OneGray)
plane_gray (0, 1, 5, [0, 2], [3, 4], ManyGray)
Many0 := ManyGray[1]
Many1 := ManyGray[2]
plane_angle (0, 1, TiltX, AzX)
plane_angle (1, 0, TiltY, AzY)
plane_angle (0, 0, Tilt0, Az0)
plane_angle (0, -1, TiltBack, AzBack)
plane_angle (-1, 0, TiltUp, AzUp)
plane_angle (1, 1, TiltDiag, AzDiag)
band_image_plane (Ramp, OnPlane, 0, 1, 0, -1, 1)
area_center (OnPlane, BandArea, BandRow, BandCol)
BandN := BandArea[1]
band_image_plane (Ramp, Swapped, 0, 1, 0, 1, -1)
area_center (Swapped, SwapArea, SwapRow, SwapCol)
SwapN := SwapArea[1]
band_image_plane (Ramp, OffPlane, 0, 1, 0, 1, 2)
count_obj (OffPlane, OffCount)
volume_image_plane (Ramp, 0, 1, 0, 'above', VolAbove)
volume_image_plane (Ramp, 0, 1, 0, 'signed', VolSigned)
extreme_image_plane (Ramp, 0, 0, 0, 'high', HighRow, HighCol, HighZ)
extreme_image_plane (Ramp, 0, 0, 0, 'low', LowRow, LowCol, LowZ)
gen_image_const (Bump, 'byte', 3, 3)
set_grayval (Bump, Peak, 1, 1, 9)
sub_plane_image (Peak, Kept, 0, 0, 0, 128)
get_grayval (Kept, 1, 1, KeptMid)
get_grayval (Kept, 0, 0, KeptCorner)
volume_image_plane (Peak, 0, 0, 0, 'above', BumpAbove)
volume_image_plane (Peak, 0, 0, 0, 'below', BumpBelow)
volume_image_plane (Peak, 0, 0, 1, 'above', FitAbove)
volume_image_plane (Peak, 0, 0, 1, 'signed', FitSigned)
extreme_image_plane (Peak, 0, 0, 0, 'high', PeakRow, PeakCol, PeakZ)
dev_display (Made)
"""


HDEVELOP_BATCH37_PROGRAM = """* 區域裡的平面
gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 80, 60)
gen_rectangle1 (Part, 10, 20, 30, 50)
fit_plane_region (Part, Ramp, Alpha, Beta, Gamma)
deviation_plane_region (Part, Ramp, 0, 1, 0, PartDev)
mean_plane_region (Part, Ramp, 0, 1, 0, PartMean)
volume_plane_region (Part, Ramp, 0, 1, 0, 'above', PartVol)
extreme_plane_region (Part, Ramp, 0, 0, 0, 'high', HighRow, HighCol, HighZ)
extreme_plane_region (Part, Ramp, 0, 0, 0, 'low', LowRow, LowCol, LowZ)
band_plane_region (Part, Ramp, Band, 0, 1, 0, -1, 1)
area_center (Band, BandArea, BandRow, BandCol)
BandN := BandArea[1]
band_plane_region (Part, Ramp, Swapped, 0, 1, 0, 1, -1)
area_center (Swapped, SwapArea, SwapRow, SwapCol)
SwapN := SwapArea[1]
band_plane_region (Part, Ramp, Miss, 0, 1, 0, 1, 2)
count_obj (Miss, OffCount)
sub_plane_region (Part, Ramp, Leveled, 0, 1, 0, 128)
get_grayval (Leveled, 20, 35, InGray)
get_grayval (Leveled, 0, 40, OutGray)
fit_plane_region_inlier (Part, Ramp, 1, KeepA, KeepB, KeepG)
gen_image_const (Bump, 'byte', 3, 3)
set_grayval (Bump, Peak, 1, 1, 9)
gen_rectangle1 (All, 0, 0, 2, 2)
gen_region_points (Center, 1, 1)
difference (All, Center, Border)
fit_plane_region (Border, Peak, BorderA, BorderB, BorderG)
fit_plane_region (All, Peak, FlatA, FlatB, FlatG)
fit_plane_region_inlier (All, Peak, 2, InlierA, InlierB, InlierG)
deviation_plane_region (All, Peak, 0, 0, 0, DevAll)
mean_plane_region (All, Peak, 0, 0, 0, MeanAll)
volume_plane_region (All, Peak, 0, 0, 0, 'above', VolAll)
dev_display (Leveled)
"""


HDEVELOP_BATCH38_PROGRAM = """* 每個物件的平面
gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 80, 60)
gen_rectangle1 (BoxA, 10, 10, 20, 20)
gen_rectangle1 (BoxB, 10, 40, 20, 50)
gen_rectangle1 (BoxC, 40, 40, 42, 42)
concat_obj (BoxA, BoxB, Pair)
concat_obj (Pair, BoxC, Parts)
set_grayval (Ramp, Peak, [40, 40, 40, 41, 41, 41, 42, 42, 42], [40, 41, 42, 40, 41, 42, 40, 41, 42], [0, 0, 0, 0, 9, 0, 0, 0, 0])
fit_plane_objects (Parts, Peak, Alpha, Beta, Gamma)
Beta1 := Beta[1]
Beta3 := Beta[3]
Gamma3 := Gamma[3]
flatness_objects (Parts, Peak, Flatness)
Flat1 := Flatness[1]
Flat3 := Flatness[3]
mean_plane_objects (Parts, Peak, 0, 0, 0, Mean)
Mean1 := Mean[1]
Mean2 := Mean[2]
Mean3 := Mean[3]
volume_plane_objects (Parts, Peak, 0, 0, 0, 'above', Volume)
Vol1 := Volume[1]
Vol2 := Volume[2]
Vol3 := Volume[3]
extreme_plane_objects (Parts, Peak, 0, 0, 0, 'high', HighRow, HighCol, HighZ)
HighR1 := HighRow[1]
HighC1 := HighCol[1]
HighZ1 := HighZ[1]
HighC2 := HighCol[2]
HighR3 := HighRow[3]
HighC3 := HighCol[3]
HighZ3 := HighZ[3]
extreme_plane_objects (Parts, Peak, 0, 0, 0, 'low', LowRow, LowCol, LowZ)
LowC1 := LowCol[1]
LowZ1 := LowZ[1]
select_flatness (Parts, Peak, Warp, 1, 10)
count_obj (Warp, WarpCount)
area_center (Warp, WarpArea, WarpRow, WarpCol)
WarpN := WarpArea[1]
select_flatness (Parts, Peak, WarpSwap, 10, 1)
count_obj (WarpSwap, SwapCount)
select_flatness (Parts, Peak, Planar, 0, 0.001)
count_obj (Planar, PlanarCount)
select_plane_mean (Parts, Peak, Tall, 0, 0, 0, 40, 50)
area_center (Tall, TallArea, TallRow, TallCol)
TallC := TallCol[1]
level_objects (Parts, Peak, Leveled, 128)
get_grayval (Leveled, 15, 15, FlatGray)
get_grayval (Leveled, 41, 41, BumpGray)
get_grayval (Leveled, 40, 40, EdgeGray)
get_grayval (Leveled, 0, 5, OutGray)
dev_display (Leveled)
"""


HDEVELOP_BATCH39_PROGRAM = """* 抛物線和二次曲面
gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 80, 60)
gen_rectangle1 (Bowl, 20, 10, 22, 14)
set_grayval (Ramp, Peak, [20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22], [10, 11, 12, 13, 14, 10, 11, 12, 13, 14, 10, 11, 12, 13, 14], [0, 1, 4, 9, 16, 0, 1, 4, 9, 16, 0, 1, 4, 9, 16])
fit_quad_region (Bowl, Peak, A, B, C, D, E, F)
fit_quad_objects (Bowl, Peak, OA, OB, OC, OD, OE, OF)
OB1 := OB[1]
OE1 := OE[1]
OF1 := OF[1]
flatness_quad_objects (Bowl, Peak, QFlat)
QF1 := QFlat[1]
flatness_objects (Bowl, Peak, PFlat)
PF1 := PFlat[1]
deviation_quad_region (Bowl, Peak, 0, 1, 0, 0, -20, 100, Dev0)
deviation_quad_region (Bowl, Peak, 0, 1, 0, 0, -20, 101, Dev1)
quad_gray (0, 1, 0, 0, -20, 100, 21, 12, OneG)
quad_gray (0, 1, 0, 0, -20, 100, [20, 22], [10, 14], Many)
Many0 := Many[1]
Many1 := Many[2]
sub_quad_image (Peak, Subtracted, 0, 1, 0, 0, -20, 100, 128)
get_grayval (Subtracted, 21, 12, SubGray)
level_quad_objects (Bowl, Peak, Leveled, 128)
get_grayval (Leveled, 21, 12, LevelGray)
get_grayval (Leveled, 0, 30, OutGray)
select_quad_flatness (Bowl, Peak, Kept, 0, 0.001)
count_obj (Kept, KeptCount)
select_quad_flatness (Bowl, Peak, Swapped, 0.001, 0)
count_obj (Swapped, SwapCount)
select_quad_flatness (Bowl, Peak, Dropped, 1, 10)
count_obj (Dropped, DropCount)
dev_display (Leveled)
"""


HDEVELOP_BATCH40_PROGRAM = """* 兩張高度圖的差
gen_image_gray_ramp (Ref, 0, 1, 0, 0, 0, 80, 60)
gen_rectangle1 (BoxA, 10, 10, 12, 12)
gen_rectangle1 (BoxB, 30, 40, 32, 42)
gen_rectangle1 (BoxC, 50, 50, 52, 52)
concat_obj (BoxA, BoxB, Pair)
concat_obj (Pair, BoxC, Parts)
add_plane_image (Ref, Test, 0, 0, 0)
set_grayval (Test, Raised, [10, 10, 10, 11, 11, 11, 12, 12, 12], [10, 11, 12, 10, 11, 12, 10, 11, 12], [20, 21, 22, 20, 31, 22, 20, 21, 22])
set_grayval (Raised, Changed, 51, 51, 41)
mean_height_diff (Parts, Changed, Ref, UnionMean)
deviation_height_diff (Parts, Changed, Ref, UnionDev)
volume_height_diff (Parts, Changed, Ref, 'above', VolAbove)
volume_height_diff (Parts, Changed, Ref, 'below', VolBelow)
volume_height_diff (Parts, Changed, Ref, 'signed', VolSigned)
extreme_height_diff (Parts, Changed, Ref, 'high', HighRow, HighCol, HighDiff)
extreme_height_diff (Parts, Changed, Ref, 'low', LowRow, LowCol, LowDiff)
mean_height_diff_objects (Parts, Changed, Ref, ObjMean)
Mean1 := ObjMean[1]
Mean2 := ObjMean[2]
Mean3 := ObjMean[3]
extreme_height_diff_objects (Parts, Changed, Ref, 'high', ObjRow, ObjCol, ObjDiff)
HighR1 := ObjRow[1]
HighC1 := ObjCol[1]
HighD1 := ObjDiff[1]
HighR2 := ObjRow[2]
HighC2 := ObjCol[2]
select_height_diff (Parts, Changed, Ref, Tall, 10, 12)
count_obj (Tall, TallCount)
select_height_diff (Parts, Changed, Ref, TallSwap, 12, 10)
count_obj (TallSwap, SwapCount)
select_height_diff (Parts, Changed, Ref, Flat, -0.5, 0.5)
area_center (Flat, FlatArea, FlatRow, FlatCol)
FlatC := FlatCol[1]
band_height_diff (Parts, Changed, Ref, Peak, 15, 25)
area_center (Peak, PeakArea, PeakRow, PeakCol)
PeakN := PeakArea[1]
PeakR := PeakRow[1]
PeakC := PeakCol[1]
sub_image (Changed, Ref, DiffView, 1, 128)
get_grayval (DiffView, 11, 11, ViewHigh)
get_grayval (DiffView, 51, 51, ViewLow)
dev_display (DiffView)
"""


HDEVELOP_BATCH41_PROGRAM = """* 局部平均和粗糙度
gen_image_const (Image, 'byte', 80, 60)
gen_rectangle1 (Flat, 10, 10, 14, 14)
gen_rectangle1 (Spot, 29, 39, 31, 41)
concat_obj (Flat, Spot, Parts)
set_grayval (Image, Peak, 30, 40, 30)
sub_local_mean (Peak, Level, 3, 3)
get_grayval (Level, 30, 40, CenterGray)
get_grayval (Level, 30, 39, SideGray)
get_grayval (Level, 12, 12, FlatGray)
roughness_ra (Spot, Peak, 3, 3, SpotRa)
roughness_ra (Flat, Peak, 3, 3, FlatRa)
roughness_rq (Spot, Peak, 3, 3, SpotRq)
roughness_rz (Spot, Peak, 3, 3, Rp, Rv, Rz)
roughness_ra_objects (Parts, Peak, 3, 3, ObjRa)
Ra1 := ObjRa[1]
Ra2 := ObjRa[2]
select_roughness (Parts, Peak, Rough, 1, 10, 3, 3)
count_obj (Rough, RoughCount)
area_center (Rough, RoughArea, RoughRow, RoughCol)
RoughN := RoughArea[1]
select_roughness (Parts, Peak, RoughSwap, 10, 1, 3, 3)
count_obj (RoughSwap, SwapCount)
select_roughness (Parts, Peak, Smooth, 0, 0.001, 3, 3)
count_obj (Smooth, SmoothCount)
extreme_roughness (Spot, Peak, 3, 3, 'high', HighRow, HighCol, HighRes)
band_roughness (Spot, Peak, PeakPix, 20, 30, 3, 3)
area_center (PeakPix, PeakArea, PeakRow, PeakCol)
PeakN := PeakArea[1]
PeakR := PeakRow[1]
PeakC := PeakCol[1]
dev_display (Level)
"""


HDEVELOP_BATCH42_PROGRAM = """* 中值和局部凸起
gen_image_const (Image, 'byte', 80, 60)
gen_rectangle1 (Flat, 10, 10, 14, 14)
gen_rectangle1 (Spot, 29, 39, 31, 41)
concat_obj (Flat, Spot, Parts)
set_grayval (Image, Peak, 30, 40, 30)
sub_local_median (Peak, Level, 3)
get_grayval (Level, 30, 40, CenterGray)
get_grayval (Level, 30, 39, SideGray)
get_grayval (Level, 12, 12, FlatGray)
roughness_median_ra (Spot, Peak, 3, SpotRa)
roughness_median_ra (Flat, Peak, 3, FlatRa)
roughness_median_rq (Spot, Peak, 3, SpotRq)
roughness_median_rz (Spot, Peak, 3, Rp, Rv, Rz)
roughness_median_ra_objects (Parts, Peak, 3, ObjRa)
Ra1 := ObjRa[1]
Ra2 := ObjRa[2]
select_median_roughness (Parts, Peak, Rough, 1, 10, 3)
count_obj (Rough, RoughCount)
area_center (Rough, RoughArea, RoughRow, RoughCol)
RoughN := RoughArea[1]
select_median_roughness (Parts, Peak, RoughSwap, 10, 1, 3)
count_obj (RoughSwap, SwapCount)
select_median_roughness (Parts, Peak, Smooth, 0, 0.001, 3)
count_obj (Smooth, SmoothCount)
extreme_median_roughness (Spot, Peak, 3, 'high', HighRow, HighCol, HighRes)
extreme_median_roughness (Spot, Peak, 3, 'low', LowRow, LowCol, LowRes)
band_median_roughness (Spot, Peak, PeakPix, 20, 40, 3)
area_center (PeakPix, PeakArea, PeakRow, PeakCol)
PeakN := PeakArea[1]
PeakR := PeakRow[1]
PeakC := PeakCol[1]
dev_display (Level)
"""


HDEVELOP_BATCH43_PROGRAM = """* 比結構元素小的亮點和暗點
gen_image_const (Image, 'byte', 80, 60)
gen_image_proto (Image, Field, 100)
gen_rectangle1 (BrightBox, 19, 24, 21, 26)
gen_rectangle1 (DarkBox, 39, 49, 41, 51)
concat_obj (BrightBox, DarkBox, Parts)
set_grayval (Field, Marked, [20, 40], [25, 50], [130, 70])
gray_spot_white (Marked, White, 3)
get_grayval (White, 20, 25, WhitePeak)
get_grayval (White, 20, 24, WhiteSide)
get_grayval (White, 40, 50, WhitePit)
gray_spot_black (Marked, Black, 3)
get_grayval (Black, 40, 50, BlackPit)
get_grayval (Black, 40, 49, BlackSide)
get_grayval (Black, 20, 25, BlackPeak)
spot_volume (BrightBox, Marked, 'white', 3, VolWhite)
spot_volume (BrightBox, Marked, 'black', 3, VolBlackOnBright)
spot_mean (BrightBox, Marked, 'white', 3, MeanWhite)
spot_volume (DarkBox, Marked, 'black', 3, VolBlack)
spot_volume (DarkBox, Marked, 'white', 3, VolWhiteOnDark)
spot_volume_objects (Parts, Marked, 'white', 3, ObjWhite)
White1 := ObjWhite[1]
White2 := ObjWhite[2]
spot_volume_objects (Parts, Marked, 'black', 3, ObjBlack)
Black1 := ObjBlack[1]
Black2 := ObjBlack[2]
extreme_spot (BrightBox, Marked, 'white', 3, HighRow, HighCol, HighH)
extreme_spot (DarkBox, Marked, 'black', 3, PitRow, PitCol, PitH)
select_spot (Parts, Marked, Kept, 1, 100, 'white', 3)
count_obj (Kept, KeepCount)
select_spot (Parts, Marked, KeptSwap, 100, 1, 'white', 3)
count_obj (KeptSwap, SwapCount)
select_spot (Parts, Marked, DarkKept, 1, 100, 'black', 3)
count_obj (DarkKept, DarkCount)
band_spot (Parts, Marked, BrightPix, 20, 40, 'white', 3)
area_center (BrightPix, BrightArea, BrightRow, BrightCol)
BrightN := BrightArea[1]
BrightR := BrightRow[1]
BrightC := BrightCol[1]
band_spot (Parts, Marked, DarkPix, 20, 40, 'black', 3)
area_center (DarkPix, DarkArea, DarkRow, DarkCol)
DarkN := DarkArea[1]
DarkR := DarkRow[1]
DarkC := DarkCol[1]
dev_display (White)
"""


HDEVELOP_BATCH44_PROGRAM = """* 開運算和重建
gen_image_const (Image, 'byte', 80, 60)
gen_rectangle1 (Square, 20, 20, 26, 26)
gen_rectangle1 (Arm, 23, 27, 23, 36)
gen_rectangle1 (Speck, 45, 50, 45, 50)
concat_obj (Square, Arm, Body)
concat_obj (Body, Speck, BrightParts)
paint_region (BrightParts, Image, Bright, 200, 'fill')
gray_opening_rect (Bright, Opened, 3)
gray_opening_reconstruct (Bright, Restored, 3)
get_grayval (Opened, 23, 23, OpenSq)
get_grayval (Opened, 23, 30, OpenArm)
get_grayval (Opened, 45, 50, OpenSpeck)
get_grayval (Restored, 23, 23, RecSq)
get_grayval (Restored, 23, 30, RecArm)
get_grayval (Restored, 45, 50, RecSpeck)
region_opening_gain (Bright, ArmBack, 3)
area_center (ArmBack, GainArea, GainRow, GainCol)
GainN := GainArea[1]
GainR := GainRow[1]
GainC := GainCol[1]
volume_opening_gain (Arm, Bright, 3, GainVol)
gen_image_proto (Image, Field, 200)
paint_region (BrightParts, Field, Dark, 0, 'fill')
gray_closing_rect (Dark, Closed, 3)
gray_closing_reconstruct (Dark, DarkRestored, 3)
get_grayval (Closed, 23, 23, CloseSq)
get_grayval (Closed, 23, 30, CloseArm)
get_grayval (Closed, 45, 50, CloseHole)
get_grayval (DarkRestored, 23, 30, RecDarkArm)
get_grayval (DarkRestored, 45, 50, RecHole)
region_closing_gain (Dark, PitBack, 3)
area_center (PitBack, PitArea, PitRow, PitCol)
PitN := PitArea[1]
volume_closing_gain (Arm, Dark, 3, PitVol)
dev_display (Restored)
"""


HDEVELOP_BATCH45_PROGRAM = """* 區域有多厚
gen_image_const (Image, 'byte', 80, 60)
gen_rectangle1 (Fat, 10, 10, 14, 14)
gen_rectangle1 (Bar, 40, 10, 42, 30)
gen_rectangle1 (Left, 10, 50, 18, 58)
gen_rectangle1 (Bridge, 13, 59, 15, 69)
gen_rectangle1 (Right, 10, 70, 18, 78)
union2 (Left, Bridge, Joined)
union2 (Joined, Right, Bell)
concat_obj (Fat, Bar, Pair)
concat_obj (Pair, Bell, Parts)
thickness_max (Parts, 'chessboard', MaxRow, MaxCol, MaxDist)
MaxR1 := MaxRow[1]
MaxC1 := MaxCol[1]
MaxD1 := MaxDist[1]
MaxR2 := MaxRow[2]
MaxC2 := MaxCol[2]
MaxD2 := MaxDist[2]
MaxR3 := MaxRow[3]
MaxC3 := MaxCol[3]
MaxD3 := MaxDist[3]
thickness_neck (Parts, 'chessboard', NeckRow, NeckCol, NeckDist)
NeckR1 := NeckRow[1]
NeckC1 := NeckCol[1]
NeckD1 := NeckDist[1]
NeckR2 := NeckRow[2]
NeckC2 := NeckCol[2]
NeckD2 := NeckDist[2]
NeckR3 := NeckRow[3]
NeckC3 := NeckCol[3]
NeckD3 := NeckDist[3]
thickness_mean (Parts, 'chessboard', MeanDist)
Mean1 := MeanDist[1]
Mean2 := MeanDist[2]
neck_ratio (Parts, 'chessboard', Ratio)
Ratio1 := Ratio[1]
Ratio2 := Ratio[2]
Ratio3 := Ratio[3]
thickness_sum (Parts, 'chessboard', SumDist)
Sum1 := SumDist[1]
Sum2 := SumDist[2]
select_neck (Parts, Thick, 3, 3, 'chessboard')
count_obj (Thick, ThickCount)
select_neck (Parts, ThickSwap, 6, 3, 'chessboard')
count_obj (ThickSwap, SwapCount)
select_neck (Parts, Thin, 2, 2, 'chessboard')
count_obj (Thin, ThinCount)
band_thickness (Fat, Core, 3, 3, 'chessboard')
area_center (Core, CoreArea, CoreRow, CoreCol)
CoreN := CoreArea[1]
CoreR := CoreRow[1]
CoreC := CoreCol[1]
thickness_image (Bell, Dist, 'chessboard')
get_grayval (Dist, 14, 54, RawCore)
get_grayval (Dist, 14, 64, RawNeck)
get_grayval (Dist, 0, 0, RawOut)
scale_image (Dist, View, 40, 0)
get_grayval (View, 14, 54, ViewCore)
get_grayval (View, 14, 64, ViewNeck)
dev_display (View)
"""


HDEVELOP_BATCH46_PROGRAM = """* 兩個區域重疊多少
gen_image_const (Image, 'byte', 80, 60)
gen_rectangle1 (Wide, 10, 10, 19, 19)
gen_rectangle1 (Shift, 10, 15, 19, 24)
gen_rectangle1 (Small, 40, 40, 44, 44)
gen_rectangle1 (Big, 38, 38, 48, 48)
concat_obj (Wide, Small, Parts)
concat_obj (Shift, Big, Other)
area_intersection (Parts, Other, Inter)
Inter1 := Inter[1]
Inter2 := Inter[2]
area_union (Parts, Other, UnionArea)
Union1 := UnionArea[1]
Union2 := UnionArea[2]
area_difference (Parts, Other, Only)
Only1 := Only[1]
Only2 := Only[2]
region_iou (Parts, Other, IoU)
Iou1 := IoU[1]
Iou2 := IoU[2]
region_cover (Parts, Other, Cover)
Cover1 := Cover[1]
Cover2 := Cover[2]
region_inside (Parts, Other, Inside)
Inside1 := Inside[1]
Inside2 := Inside[2]
select_cover (Parts, Other, Kept, 0.9, 1)
count_obj (Kept, KeepCount)
area_center (Kept, KeepArea, KeepRow, KeepCol)
KeepN := KeepArea[1]
KeepR := KeepRow[1]
KeepC := KeepCol[1]
select_cover (Parts, Other, KeptSwap, 1, 0.9)
count_obj (KeptSwap, SwapCount)
select_cover (Parts, Other, Half, 0.4, 0.6)
count_obj (Half, HalfCount)
sym_difference (Parts, Other, Differ)
area_center (Differ, SymArea, SymRow, SymCol)
Sym1 := SymArea[1]
Sym2 := SymArea[2]
intersection (Wide, Shift, Overlap)
paint_region (Overlap, Image, Shown, 255, 'fill')
get_grayval (Shown, 12, 17, HitGray)
get_grayval (Shown, 12, 12, MissGray)
dev_display (Shown)
"""


HDEVELOP_BATCH47_PROGRAM = """* 兩個區域隔多遠
gen_image_const (Image, 'byte', 80, 60)
gen_rectangle1 (GapA, 10, 10, 19, 19)
gen_rectangle1 (GapB, 10, 30, 19, 39)
gen_rectangle1 (TouchA, 40, 10, 44, 14)
gen_rectangle1 (TouchB, 40, 15, 44, 19)
gen_rectangle1 (OverA, 50, 40, 54, 49)
gen_rectangle1 (OverB, 50, 45, 54, 54)
concat_obj (GapA, TouchA, Pair1)
concat_obj (Pair1, OverA, Parts)
concat_obj (GapB, TouchB, Pair2)
concat_obj (Pair2, OverB, Other)
distance_objects (Parts, Other, 'chessboard', MinDist)
Min1 := MinDist[1]
Min2 := MinDist[2]
Min3 := MinDist[3]
distance_objects_far (Parts, Other, 'chessboard', FarDist)
Far1 := FarDist[1]
Far2 := FarDist[2]
Far3 := FarDist[3]
nearest_object_point (Parts, Other, 'chessboard', NearRow1, NearCol1, NearRow2, NearCol2, NearDist)
NearR1 := NearRow1[1]
NearC1 := NearCol1[1]
NearR2 := NearRow2[1]
NearC2 := NearCol2[1]
NearD1 := NearDist[1]
TouchR := NearRow1[2]
TouchC := NearCol1[2]
TouchD := NearDist[2]
OverR := NearRow1[3]
OverC := NearCol1[3]
OverD := NearDist[3]
farthest_object_point (Parts, Other, 'chessboard', FarRow1, FarCol1, FarRow2, FarCol2, FarPoint)
FarR1 := FarRow1[1]
FarC1 := FarCol1[1]
FarP1 := FarPoint[1]
centroid_offset (Parts, Other, 'chessboard', OffRow, OffCol)
OffR1 := OffRow[1]
OffC1 := OffCol[1]
OffC2 := OffCol[2]
centroid_distance (Parts, Other, 'chessboard', CenterDist)
Center1 := CenterDist[1]
Center2 := CenterDist[2]
select_object_distance (Parts, Other, GapKept, 11, 11, 'chessboard')
count_obj (GapKept, GapCount)
select_object_distance (Parts, Other, GapSwap, 20, 11, 'chessboard')
count_obj (GapSwap, SwapCount)
select_object_distance (Parts, Other, TouchKept, 1, 1, 'chessboard')
count_obj (TouchKept, TouchCount)
select_object_distance (Parts, Other, OverKept, 0, 0, 'chessboard')
count_obj (OverKept, OverCount)
facing_region (Parts, Other, Edge, 'chessboard')
area_center (Edge, EdgeArea, EdgeRow, EdgeCol)
EdgeN := EdgeArea[1]
EdgeR := EdgeRow[1]
EdgeC := EdgeCol[1]
select_obj (Edge, GapEdge, 1)
paint_region (GapEdge, Image, Shown, 255, 'fill')
get_grayval (Shown, 10, 19, EdgeGray)
get_grayval (Shown, 10, 18, InsideGray)
dev_display (Shown)
"""


HDEVELOP_BATCH48_PROGRAM = """* 物件在參考的哪一側
gen_image_const (Image, 'byte', 80, 60)
gen_rectangle1 (Ref, 30, 30, 39, 39)
gen_rectangle1 (Right, 30, 50, 39, 59)
gen_rectangle1 (Left, 30, 22, 39, 26)
gen_rectangle1 (Below, 48, 30, 52, 39)
gen_rectangle1 (Above, 14, 30, 18, 39)
concat_obj (Right, Left, Pair)
concat_obj (Pair, Below, Three)
concat_obj (Three, Above, Parts)
region_angle (Parts, Ref, Angle)
Ang1 := Angle[1]
Ang2 := Angle[2]
Ang3 := Angle[3]
Ang4 := Angle[4]
select_side (Parts, Ref, RightKept, 'right')
count_obj (RightKept, RightCount)
select_side (Parts, Ref, LeftKept, 'left')
count_obj (LeftKept, LeftCount)
select_side (Parts, Ref, BelowKept, 'below')
count_obj (BelowKept, BelowCount)
select_side (Parts, Ref, AboveKept, 'above')
count_obj (AboveKept, AboveCount)
sort_distance (Parts, Ref, Near, 'near', 'city-block')
area_center (Near, NearArea, NearRow, NearCol)
NearR := NearRow[1]
NearC := NearCol[1]
sort_distance (Parts, Ref, FarSorted, 'far', 'city-block')
area_center (FarSorted, FarArea, FarRow, FarCol)
FarR := FarRow[1]
FarC := FarCol[1]
nearest_one (Parts, Ref, Closest, 'city-block')
area_center (Closest, CloseArea, CloseRow, CloseCol)
CloseN := CloseArea[1]
CloseR := CloseRow[1]
CloseC := CloseCol[1]
farthest_one (Parts, Ref, Farthest, 'city-block')
area_center (Farthest, FarMostArea, FarMostRow, FarMostCol)
FarN := FarMostArea[1]
FarMostR := FarMostRow[1]
FarMostC := FarMostCol[1]
gap_area (Parts, Ref, 'city-block', GapArea)
Gap1 := GapArea[1]
Gap2 := GapArea[2]
Gap3 := GapArea[3]
Gap4 := GapArea[4]
gap_width (Parts, Ref, 'city-block', GapWidth)
Wide1 := GapWidth[1]
Wide2 := GapWidth[2]
Wide3 := GapWidth[3]
Wide4 := GapWidth[4]
gap_region (Right, Ref, Gap, 'city-block')
area_center (Gap, GapOnly, GapRow, GapCol)
GapN := GapOnly[1]
paint_region (Gap, Image, Shown, 255, 'fill')
get_grayval (Shown, 34, 44, GapGray)
get_grayval (Shown, 34, 39, RefGray)
dev_display (Shown)
"""


HDEVELOP_BATCH49_PROGRAM = """* 空白分給最近的物件
gen_image_const (Image, 'byte', 80, 60)
gen_rectangle1 (Left, 10, 10, 19, 19)
gen_rectangle1 (Right, 10, 30, 19, 39)
concat_obj (Left, Right, Parts)
partition_nearest (Parts, Grown, 'city-block')
count_obj (Grown, GrownN)
area_center (Grown, GrownArea, GrownRow, GrownCol)
G1 := GrownArea[1]
G2 := GrownArea[2]
GR := GrownRow[1]
GC1 := GrownCol[1]
GC2 := GrownCol[2]
partition_area (Parts, 'city-block', Territory)
T1 := Territory[1]
T2 := Territory[2]
owner_index (Parts, [14, 14], [24, 25], 'city-block', GapOwn)
OwnL := GapOwn[1]
OwnR := GapOwn[2]
owner_index (Parts, [0, 59], [0, 79], 'city-block', Corners)
CornerL := Corners[1]
CornerR := Corners[2]
partition_border (Parts, Border, 'city-block')
count_obj (Border, BorderN)
area_center (Border, BorderArea, BorderRow, BorderCol)
B1 := BorderArea[1]
B2 := BorderArea[2]
BC1 := BorderCol[1]
BC2 := BorderCol[2]
select_territory (Parts, Big, 'large', 'city-block')
area_center (Big, BigArea, BigRow, BigCol)
BigN := BigArea[1]
BigC := BigCol[1]
select_territory (Parts, Small, 'small', 'city-block')
area_center (Small, SmallArea, SmallRow, SmallCol)
SmallN := SmallArea[1]
SmallC := SmallCol[1]
gen_rectangle1 (Strip, 0, 20, 59, 29)
partition_domain (Parts, Strip, Inside, 'city-block')
count_obj (Inside, InsideN)
area_center (Inside, InArea, InRow, InCol)
In1 := InArea[1]
In2 := InArea[2]
InC1 := InCol[1]
InC2 := InCol[2]
test_region_point (Inside, 14, 14, OffStrip)
test_region_point (Inside, 14, 22, OnStrip)
partition_label (Parts, Labels, 'city-block')
get_grayval (Labels, 14, 24, LabelL)
get_grayval (Labels, 14, 25, LabelR)
gen_region_points (DotA, 40, 10)
gen_region_points (DotB, 40, 12)
concat_obj (DotA, DotB, Dots)
tie_region (Dots, Ties, 'city-block')
count_obj (Ties, TieCount)
area_center (Ties, TieArea, TieRow, TieCol)
TieN := TieArea[1]
TieR := TieRow[1]
TieC := TieCol[1]
paint_region (Border, Image, Shown, 255, 'fill')
get_grayval (Shown, 14, 24, CutGray)
get_grayval (Shown, 14, 25, CutGray2)
get_grayval (Shown, 14, 23, SideGray)
dev_display (Shown)
"""


HDEVELOP_BATCH50_PROGRAM = """* 依特徵挑出物件
gen_image_const (Image, 'byte', 80, 60)
gen_rectangle1 (Big, 10, 10, 19, 19)
gen_rectangle1 (Tiny, 10, 30, 13, 33)
gen_rectangle1 (Mid, 30, 10, 35, 17)
gen_rectangle1 (Twin, 40, 40, 43, 43)
concat_obj (Big, Tiny, Pair)
concat_obj (Pair, Mid, Three)
concat_obj (Three, Twin, Parts)
sort_feature (Parts, Down, 'area', 'down')
area_center (Down, DownArea, DownRow, DownCol)
D1 := DownArea[1]
D2 := DownArea[2]
D3 := DownArea[3]
DR := DownRow[1]
DC := DownCol[1]
DC3 := DownCol[3]
sort_feature (Parts, Up, 'area', 'up')
area_center (Up, UpArea, UpRow, UpCol)
U1 := UpArea[1]
UR := UpRow[1]
UC := UpCol[1]
UC2 := UpCol[2]
select_top (Parts, Top, 2, 'area', 'down')
count_obj (Top, TopN)
area_center (Top, TopArea, TopRow, TopCol)
Top1 := TopArea[1]
Top2 := TopArea[2]
select_extreme (Parts, Largest, 'max', 'area')
area_center (Largest, LargeArea, LargeRow, LargeCol)
LargeN := LargeArea[1]
LargeR := LargeRow[1]
LargeC := LargeCol[1]
select_extreme (Parts, Smallest, 'min', 'area')
area_center (Smallest, SmallArea, SmallRow, SmallCol)
SmallN := SmallArea[1]
SmallR := SmallRow[1]
SmallC := SmallCol[1]
feature_rank (Parts, 'area', 'down', Rank)
R1 := Rank[1]
R2 := Rank[2]
R3 := Rank[3]
R4 := Rank[4]
select_rank (Parts, Second, 2, 'area', 'down')
area_center (Second, SecondArea, SecondRow, SecondCol)
SecN := SecondArea[1]
SecR := SecondRow[1]
select_rank (Parts, Picked, [1, 3], 'area', 'down')
area_center (Picked, PickArea, PickRow, PickCol)
Pick1 := PickArea[1]
Pick2 := PickArea[2]
PickC := PickCol[2]
feature_index (Parts, 'area', 'down', Index)
I1 := Index[1]
I2 := Index[2]
I3 := Index[3]
I4 := Index[4]
select_relative (Parts, Above, 50, 'area', 'above')
count_obj (Above, AboveN)
area_center (Above, AboveArea, AboveRow, AboveCol)
Ab1 := AboveArea[1]
Ab2 := AboveArea[2]
select_relative (Parts, Below, 50, 'area', 'below')
count_obj (Below, BelowN)
feature_range (Parts, 'area', Low, High)
paint_region (Largest, Image, Shown, 255, 'fill')
get_grayval (Shown, 14, 14, HitGray)
get_grayval (Shown, 11, 31, MissGray)
dev_display (Shown)
"""


HDEVELOP_BATCH51_PROGRAM = """* 補上缺掉的缺口
gen_image_const (Image, 'byte', 80, 60)
gen_rectangle1 (Block, 10, 10, 39, 39)
gen_rectangle1 (Bite, 20, 30, 29, 39)
difference (Block, Bite, Notched)
gen_rectangle1 (Plain, 45, 45, 54, 54)
gen_rectangle1 (CornerBlock, 10, 50, 29, 69)
gen_rectangle1 (CornerBite, 10, 65, 14, 69)
difference (CornerBlock, CornerBite, Corner)
concat_obj (Notched, Plain, Pair)
concat_obj (Pair, Corner, Parts)
region_defects (Parts, Bites, 1, 'city-block')
count_obj (Bites, BiteN)
area_center (Bites, BiteArea, BiteRow, BiteCol)
BA1 := BiteArea[1]
BA2 := BiteArea[2]
BR1 := BiteRow[1]
BC1 := BiteCol[1]
BR2 := BiteRow[2]
BC2 := BiteCol[2]
defect_count (Parts, 1, 'city-block', Counts)
C1 := Counts[1]
C2 := Counts[2]
C3 := Counts[3]
defect_area (Parts, 1, 'city-block', Areas)
DA1 := Areas[1]
DA2 := Areas[2]
defect_depth (Parts, 1, 'city-block', Depths)
DP1 := Depths[1]
DP2 := Depths[2]
defect_point (Parts, 1, 'city-block', PointRow, PointCol, PointDepth)
PR1 := PointRow[1]
PC1 := PointCol[1]
PD1 := PointDepth[1]
PR2 := PointRow[2]
PC2 := PointCol[2]
PD2 := PointDepth[2]
select_defect (Parts, Chipped, 2, 'city-block')
count_obj (Chipped, ChipN)
area_center (Chipped, ChipArea, ChipRow, ChipCol)
Chip1 := ChipArea[1]
Chip2 := ChipArea[2]
region_defects (Parts, DeepOnly, 6, 'city-block')
count_obj (DeepOnly, DeepN)
area_center (DeepOnly, DeepArea, DeepRow, DeepCol)
DeepA := DeepArea[1]
deepest_defect (Parts, Deepest, 'city-block')
area_center (Deepest, MostArea, MostRow, MostCol)
MostN := MostArea[1]
fill_defects (Parts, Fixed, 1, 'city-block')
area_center (Fixed, FixArea, FixRow, FixCol)
F1 := FixArea[1]
F2 := FixArea[2]
F3 := FixArea[3]
paint_region (Deepest, Image, Shown, 255, 'fill')
get_grayval (Shown, 24, 34, GapGray)
get_grayval (Shown, 14, 14, BodyGray)
dev_display (Shown)
"""


HDEVELOP_BATCH52_PROGRAM = """* 誰包住誰
gen_image_const (Image, 'byte', 80, 60)
gen_rectangle1 (Outer, 5, 5, 44, 44)
gen_rectangle1 (Hole, 15, 15, 34, 34)
difference (Outer, Hole, Frame)
gen_rectangle1 (Middle, 18, 18, 31, 31)
gen_rectangle1 (Tiny, 22, 22, 25, 25)
gen_rectangle1 (Out, 50, 50, 54, 54)
concat_obj (Frame, Middle, Pair)
concat_obj (Pair, Tiny, Three)
concat_obj (Three, Out, Parts)
select_inner (Parts, Inner)
count_obj (Inner, InnerN)
area_center (Inner, InnerArea, InnerRow, InnerCol)
In1 := InnerArea[1]
In2 := InnerArea[2]
select_outer (Parts, OuterKept)
count_obj (OuterKept, OuterN)
area_center (OuterKept, OuterArea, OuterRow, OuterCol)
Out1 := OuterArea[1]
Out2 := OuterArea[2]
select_container (Parts, Holders)
count_obj (Holders, HolderN)
area_center (Holders, HolderArea, HolderRow, HolderCol)
Hold1 := HolderArea[1]
Hold2 := HolderArea[2]
innermost (Parts, Leaf)
area_center (Leaf, LeafArea, LeafRow, LeafCol)
LeafN := LeafArea[1]
LeafR := LeafRow[1]
LeafC := LeafCol[1]
select_depth (Parts, Level1, 1)
area_center (Level1, L1Area, L1Row, L1Col)
MidN := L1Area[1]
MidR := L1Row[1]
select_depth (Parts, Level2, 2)
count_obj (Level2, DeepN)
select_depth (Parts, Level0, 0)
count_obj (Level0, TopN)
container_index (Parts, Holder)
H1 := Holder[1]
H2 := Holder[2]
H3 := Holder[3]
H4 := Holder[4]
inner_count (Parts, InsideCount)
K1 := InsideCount[1]
K2 := InsideCount[2]
K3 := InsideCount[3]
K4 := InsideCount[4]
nesting_depth (Parts, NestLevel)
D1 := NestLevel[1]
D2 := NestLevel[2]
D3 := NestLevel[3]
D4 := NestLevel[4]
paint_region (Leaf, Image, Shown, 255, 'fill')
get_grayval (Shown, 23, 23, LeafGray)
get_grayval (Shown, 10, 10, FrameGray)
dev_display (Shown)
"""


HDEVELOP_BATCH53_PROGRAM = """* 有沒有碰到影像邊緣
gen_image_const (Image, 'byte', 80, 60)
gen_rectangle1 (TouchTop, 0, 20, 9, 29)
gen_rectangle1 (TouchLeft, 20, 0, 29, 4)
gen_rectangle1 (Center, 25, 35, 34, 44)
gen_rectangle1 (NearRight, 40, 70, 49, 77)
concat_obj (TouchTop, TouchLeft, Pair)
concat_obj (Pair, Center, Three)
concat_obj (Three, NearRight, Parts)
select_border (Parts, Touched, 'touch')
count_obj (Touched, TouchN)
area_center (Touched, TouchArea, TouchRow, TouchCol)
Touch1 := TouchArea[1]
Touch2 := TouchArea[2]
select_border (Parts, Inside, 'inner')
count_obj (Inside, InnerN)
area_center (Inside, InnerArea, InnerRow, InnerCol)
Inner1 := InnerArea[1]
Inner2 := InnerArea[2]
select_border_side (Parts, TopSide, 'top')
count_obj (TopSide, TopN)
area_center (TopSide, TopArea, TopRow, TopCol)
TopA := TopArea[1]
select_border_side (Parts, RightSide, 'right')
count_obj (RightSide, RightN)
select_margin (Parts, Kept, 3)
count_obj (Kept, KeepN)
area_center (Kept, KeepArea, KeepRow, KeepCol)
KeepA := KeepArea[1]
select_border_range (Parts, Band, 1, 10)
count_obj (Band, BandN)
area_center (Band, BandArea, BandRow, BandCol)
BandA := BandArea[1]
farthest_border (Parts, Far)
area_center (Far, FarArea, FarRow, FarCol)
FarN := FarArea[1]
FarR := FarRow[1]
FarC := FarCol[1]
border_distance (Parts, Distance)
D1 := Distance[1]
D2 := Distance[2]
D3 := Distance[3]
D4 := Distance[4]
border_gap (Parts, GapTop, GapBottom, GapLeft, GapRight)
GT1 := GapTop[1]
GT2 := GapTop[2]
GT3 := GapTop[3]
GT4 := GapTop[4]
GB1 := GapBottom[1]
GB3 := GapBottom[3]
GB4 := GapBottom[4]
GL2 := GapLeft[2]
GL3 := GapLeft[3]
GR2 := GapRight[2]
GR4 := GapRight[4]
border_point (Parts, PointRow, PointCol, PointSide)
PR1 := PointRow[1]
PC1 := PointCol[1]
PS1 := PointSide[1]
PR2 := PointRow[2]
PC2 := PointCol[2]
PS2 := PointSide[2]
PR3 := PointRow[3]
PC3 := PointCol[3]
PS3 := PointSide[3]
PR4 := PointRow[4]
PC4 := PointCol[4]
PS4 := PointSide[4]
paint_region (Touched, Image, Shown, 255, 'fill')
get_grayval (Shown, 0, 20, EdgeGray)
get_grayval (Shown, 25, 35, MidGray)
dev_display (Shown)
"""


HDEVELOP_BATCH54_PROGRAM = """* 物件有沒有互相碰到
gen_image_const (Image, 'byte', 80, 60)
gen_rectangle1 (Block, 10, 10, 19, 19)
gen_rectangle1 (Beside, 10, 20, 19, 24)
gen_rectangle1 (Below, 20, 20, 24, 24)
gen_rectangle1 (Far, 40, 40, 44, 44)
concat_obj (Block, Beside, Pair)
concat_obj (Pair, Below, Three)
concat_obj (Three, Far, Parts)
neighbor_distance (Parts, 'chessboard', Distance)
D1 := Distance[1]
D2 := Distance[2]
D3 := Distance[3]
D4 := Distance[4]
neighbor_index (Parts, 'chessboard', NearIndex)
I1 := NearIndex[1]
I2 := NearIndex[2]
I3 := NearIndex[3]
I4 := NearIndex[4]
neighbor_point (Parts, 'chessboard', NearRow, NearCol, NearGap)
NR1 := NearRow[1]
NC1 := NearCol[1]
NG1 := NearGap[1]
NR2 := NearRow[2]
NC2 := NearCol[2]
NR4 := NearRow[4]
NC4 := NearCol[4]
NG4 := NearGap[4]
neighbor_count (Parts, 1, 'chessboard', TouchCount)
K1 := TouchCount[1]
K2 := TouchCount[2]
K3 := TouchCount[3]
K4 := TouchCount[4]
select_touching (Parts, Touched, 1, 'chessboard')
count_obj (Touched, TouchN)
area_center (Touched, TouchArea, TouchRow, TouchCol)
Touch1 := TouchArea[1]
Touch2 := TouchArea[2]
Touch3 := TouchArea[3]
select_isolated (Parts, Alone, 1, 'chessboard')
count_obj (Alone, AloneN)
area_center (Alone, AloneArea, AloneRow, AloneCol)
AloneA := AloneArea[1]
AloneR := AloneRow[1]
AloneC := AloneCol[1]
contact_region (Parts, Contact, 1, 'chessboard')
count_obj (Contact, ContactN)
area_center (Contact, ContactArea, ContactRow, ContactCol)
CA1 := ContactArea[1]
CA2 := ContactArea[2]
CA3 := ContactArea[3]
CR1 := ContactRow[1]
CC1 := ContactCol[1]
union_touching (Parts, Merged, 1, 'chessboard')
count_obj (Merged, MergeN)
area_center (Merged, MergeArea, MergeRow, MergeCol)
M1 := MergeArea[1]
M2 := MergeArea[2]
MR := MergeRow[1]
MC := MergeCol[1]
paint_region (Contact, Image, Shown, 255, 'fill')
get_grayval (Shown, 14, 19, TouchGray)
get_grayval (Shown, 14, 18, BodyGray)
get_grayval (Shown, 42, 42, FarGray)
dev_display (Shown)
"""


GENERATED_PROGRAMS = {
    "auto_threshold.hdv": AUTO_THRESHOLD_PROGRAM,
    "morph_region.hdv": MORPH_REGION_PROGRAM,
    "region_sets.hdv": REGION_SETS_PROGRAM,
    "subpix_blob.hdv": SUBPIX_BLOB_PROGRAM,
    "xld_fit.hdv": XLD_FIT_PROGRAM,
    "xld_circle.hdv": XLD_CIRCLE_PROGRAM,
    "image_filter.hdv": IMAGE_FILTER_PROGRAM,
    "tuple_stats.hdv": TUPLE_STATS_PROGRAM,
    "hdevelop_batch1.hdv": HDEVELOP_BATCH1_PROGRAM,
    "hdevelop_batch2.hdv": HDEVELOP_BATCH2_PROGRAM,
    "hdevelop_batch3.hdv": HDEVELOP_BATCH3_PROGRAM,
    "hdevelop_batch4.hdv": HDEVELOP_BATCH4_PROGRAM,
    "hdevelop_batch5.hdv": HDEVELOP_BATCH5_PROGRAM,
    "hdevelop_batch6.hdv": HDEVELOP_BATCH6_PROGRAM,
    "hdevelop_batch7.hdv": HDEVELOP_BATCH7_PROGRAM,
    "hdevelop_batch8.hdv": HDEVELOP_BATCH8_PROGRAM,
    "hdevelop_batch9.hdv": HDEVELOP_BATCH9_PROGRAM,
    "hdevelop_batch10.hdv": HDEVELOP_BATCH10_PROGRAM,
    "hdevelop_batch11.hdv": HDEVELOP_BATCH11_PROGRAM,
    "hdevelop_batch12.hdv": HDEVELOP_BATCH12_PROGRAM,
    "hdevelop_batch13.hdv": HDEVELOP_BATCH13_PROGRAM,
    "hdevelop_batch14.hdv": HDEVELOP_BATCH14_PROGRAM,
    "hdevelop_batch15.hdv": HDEVELOP_BATCH15_PROGRAM,
    "hdevelop_batch16.hdv": HDEVELOP_BATCH16_PROGRAM,
    "hdevelop_batch17.hdv": HDEVELOP_BATCH17_PROGRAM,
    "hdevelop_batch18.hdv": HDEVELOP_BATCH18_PROGRAM,
    "hdevelop_batch19.hdv": HDEVELOP_BATCH19_PROGRAM,
    "hdevelop_batch20.hdv": HDEVELOP_BATCH20_PROGRAM,
    "hdevelop_batch21.hdv": HDEVELOP_BATCH21_PROGRAM,
    "hdevelop_batch22.hdv": HDEVELOP_BATCH22_PROGRAM,
    "hdevelop_batch23.hdv": HDEVELOP_BATCH23_PROGRAM,
    "hdevelop_batch24.hdv": HDEVELOP_BATCH24_PROGRAM,
    "hdevelop_batch25.hdv": HDEVELOP_BATCH25_PROGRAM,
    "hdevelop_batch26.hdv": HDEVELOP_BATCH26_PROGRAM,
    "hdevelop_batch27.hdv": HDEVELOP_BATCH27_PROGRAM,
    "hdevelop_batch28.hdv": HDEVELOP_BATCH28_PROGRAM,
    "hdevelop_batch29.hdv": HDEVELOP_BATCH29_PROGRAM,
    "hdevelop_batch30.hdv": HDEVELOP_BATCH30_PROGRAM,
    "hdevelop_batch31.hdv": HDEVELOP_BATCH31_PROGRAM,
    "hdevelop_batch32.hdv": HDEVELOP_BATCH32_PROGRAM,
    "hdevelop_batch33.hdv": HDEVELOP_BATCH33_PROGRAM,
    "hdevelop_batch34.hdv": HDEVELOP_BATCH34_PROGRAM,
    "hdevelop_batch35.hdv": HDEVELOP_BATCH35_PROGRAM,
    "hdevelop_batch36.hdv": HDEVELOP_BATCH36_PROGRAM,
    "hdevelop_batch37.hdv": HDEVELOP_BATCH37_PROGRAM,
    "hdevelop_batch38.hdv": HDEVELOP_BATCH38_PROGRAM,
    "hdevelop_batch39.hdv": HDEVELOP_BATCH39_PROGRAM,
    "hdevelop_batch40.hdv": HDEVELOP_BATCH40_PROGRAM,
    "hdevelop_batch41.hdv": HDEVELOP_BATCH41_PROGRAM,
    "hdevelop_batch42.hdv": HDEVELOP_BATCH42_PROGRAM,
    "hdevelop_batch43.hdv": HDEVELOP_BATCH43_PROGRAM,
    "hdevelop_batch44.hdv": HDEVELOP_BATCH44_PROGRAM,
    "hdevelop_batch45.hdv": HDEVELOP_BATCH45_PROGRAM,
    "hdevelop_batch46.hdv": HDEVELOP_BATCH46_PROGRAM,
    "hdevelop_batch47.hdv": HDEVELOP_BATCH47_PROGRAM,
    "hdevelop_batch48.hdv": HDEVELOP_BATCH48_PROGRAM,
    "hdevelop_batch49.hdv": HDEVELOP_BATCH49_PROGRAM,
    "hdevelop_batch50.hdv": HDEVELOP_BATCH50_PROGRAM,
    "hdevelop_batch51.hdv": HDEVELOP_BATCH51_PROGRAM,
    "hdevelop_batch52.hdv": HDEVELOP_BATCH52_PROGRAM,
    "hdevelop_batch53.hdv": HDEVELOP_BATCH53_PROGRAM,
    "hdevelop_batch54.hdv": HDEVELOP_BATCH54_PROGRAM,
    "ocr_read.hdv": OCR_READ_PROGRAM,
}
