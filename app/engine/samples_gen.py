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
    "ocr_read.hdv": OCR_READ_PROGRAM,
}
