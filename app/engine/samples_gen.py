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
    "ViZion_hdevelop_batch1.viz",
    "ViZion_hdevelop_batch2.viz",
    "ViZion_hdevelop_batch3.viz",
    "ViZion_hdevelop_batch4.viz",
    "ViZion_hdevelop_batch5.viz",
    "ViZion_hdevelop_batch6.viz",
    "ViZion_hdevelop_batch7.viz",
    "ViZion_hdevelop_batch8.viz",
    "ViZion_hdevelop_batch9.viz",
    "ViZion_hdevelop_batch10.viz",
    "ViZion_hdevelop_batch11.viz",
    "ViZion_hdevelop_batch12.viz",
    "ViZion_hdevelop_batch13.viz",
    "ViZion_hdevelop_batch14.viz",
    "ViZion_hdevelop_batch15.viz",
    "ViZion_hdevelop_batch16.viz",
    "ViZion_hdevelop_batch17.viz",
    "ViZion_hdevelop_batch18.viz",
    "ViZion_hdevelop_batch19.viz",
    "ViZion_hdevelop_batch20.viz",
    "ViZion_hdevelop_batch21.viz",
    "ViZion_hdevelop_batch22.viz",
    "ViZion_hdevelop_batch23.viz",
    "ViZion_hdevelop_batch24.viz",
    "ViZion_hdevelop_batch25.viz",
    "ViZion_hdevelop_batch26.viz",
    "ViZion_hdevelop_batch27.viz",
    "ViZion_hdevelop_batch28.viz",
    "ViZion_hdevelop_batch29.viz",
    "ViZion_hdevelop_batch30.viz",
    "ViZion_hdevelop_batch31.viz",
    "ViZion_hdevelop_batch32.viz",
    "ViZion_hdevelop_batch33.viz",
    "ViZion_hdevelop_batch34.viz",
    "ViZion_hdevelop_batch35.viz",
    "ViZion_hdevelop_batch36.viz",
    "ViZion_hdevelop_batch37.viz",
    "ViZion_hdevelop_batch38.viz",
    "ViZion_hdevelop_batch39.viz",
    "ViZion_hdevelop_batch40.viz",
    "ViZion_hdevelop_batch41.viz",
    "ViZion_hdevelop_batch42.viz",
    "ViZion_hdevelop_batch43.viz",
    "ViZion_hdevelop_batch44.viz",
    "ViZion_hdevelop_batch45.viz",
    "ViZion_hdevelop_batch46.viz",
    "ViZion_hdevelop_batch47.viz",
    "ViZion_hdevelop_batch48.viz",
    "ViZion_hdevelop_batch49.viz",
    "ViZion_hdevelop_batch50.viz",
    "ViZion_hdevelop_batch51.viz",
    "ViZion_hdevelop_batch52.viz",
    "ViZion_hdevelop_batch53.viz",
    "ViZion_hdevelop_batch54.viz",
    "ViZion_hdevelop_batch55.viz",
    "ViZion_hdevelop_batch56.viz",
    "ViZion_hdevelop_batch57.viz",
    "ViZion_hdevelop_batch58.viz",
    "ViZion_hdevelop_batch59.viz",
    "ViZion_hdevelop_batch60.viz",
    "ViZion_hdevelop_batch61.viz",
    "ViZion_hdevelop_batch62.viz",
    "ViZion_hdevelop_batch63.viz",
    "ViZion_hdevelop_batch64.viz",
    "ViZion_hdevelop_batch65.viz",
    "ViZion_hdevelop_batch66.viz",
    "ViZion_hdevelop_batch67.viz",
    "ViZion_hdevelop_batch68.viz",
    "ViZion_hdevelop_batch69.viz",
    "ViZion_hdevelop_batch70.viz",
    "ViZion_hdevelop_batch71.viz",
    "ViZion_hdevelop_batch72.viz",
    "ViZion_hdevelop_batch73.viz",
    "ViZion_hdevelop_batch74.viz",
    "ViZion_hdevelop_batch75.viz",
    "ViZion_hdevelop_batch76.viz",
    "ViZion_hdevelop_batch77.viz",
    "ViZion_hdevelop_batch78.viz",
    "ViZion_hdevelop_batch79.viz",
    "ViZion_hdevelop_batch80.viz",
    "ViZion_hdevelop_batch81.viz",
    "ViZion_hdevelop_batch82.viz",
    "ViZion_hdevelop_batch83.viz",
    "ViZion_hdevelop_batch84.viz",
    "ViZion_hdevelop_batch85.viz",
    "ViZion_hdevelop_batch86.viz",
    "ViZion_hdevelop_batch87.viz",
    "ViZion_hdevelop_batch88.viz",
    "ViZion_hdevelop_batch89.viz",
    "ViZion_hdevelop_batch90.viz",
    "ViZion_hdevelop_batch91.viz",
    "ViZion_hdevelop_batch92.viz",
    "ViZion_hdevelop_batch93.viz",
    "ViZion_hdevelop_batch94.viz",
    "ViZion_hdevelop_batch95.viz",
    "ViZion_hdevelop_batch96.viz",
    "ViZion_hdevelop_batch97.viz",
    "ViZion_hdevelop_batch98.viz",
    "ViZion_hdevelop_batch99.viz",
    "ViZion_hdevelop_batch100.viz",
    "ViZion_hdevelop_batch101.viz",
    "ViZion_hdevelop_batch102.viz",
    "ViZion_hdevelop_batch103.viz",
    "ViZion_hdevelop_batch104.viz",
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
Vizion.read_image (Scene, 'samples/shapes_scene.png')
Vizion.read_image (Template, 'samples/shape_template.png')
Vizion.teach_shape_model (Template, 15, 15, 164, 164, 0, rad(-180), rad(360), 0, 'use_polarity', 40, 10, ModelID)
Vizion.find_shape_model (Scene, ModelID, rad(-180), rad(360), 0.7, 0, 0.4, 'interpolation', 0, 0.4, Row, Column, Angle, Score)
Vizion.dev_display (Scene)
Vizion.dev_set_line_width (2)
Vizion.dev_display_shape_matching_results (ModelID, Row, Column, Angle)
Count := |Row|
Label := 'matches ' + Count
Vizion.dev_set_color ('white')
Vizion.dev_disp_text (Label, 24, 12)
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
Vizion.read_image (Template, 'samples/part_template.png')
Vizion.read_image (Scene, 'samples/part_scene.png')
Vizion.teach_shape_model (Template, 18, 18, 181, 181, 0, rad(-180), rad(360), 0, 'use_polarity', 40, 10, ModelID)
Vizion.find_shape_model (Scene, ModelID, rad(-180), rad(360), 0.8, 0, 0.4, 'interpolation', 0, 0.5, Row, Column, Angle, Score)
Vizion.dev_display (Scene)
Vizion.dev_set_line_width (2)
Vizion.dev_display_shape_matching_results (ModelID, Row, Column, Angle)
Count := |Row|
Label := '找到 ' + Count
Vizion.dev_set_color ('white')
Vizion.dev_disp_text (Label, 28, 16)
"""


NCC_LOCATE_PROGRAM = """* 教導同一支箭頭，再用 NCC 定位。只畫中心、方向箭頭和角度
Vizion.read_image (Scene, 'samples/shapes_scene.png')
Vizion.read_image (Template, 'samples/shape_template.png')
Vizion.teach_ncc_model (Template, 15, 15, 164, 164, 0, rad(-180), rad(360), 0, 'use_polarity', ModelID)
Vizion.find_ncc_model (Scene, ModelID, rad(-180), rad(360), 0.75, 0, 0.4, 'interpolation', 0, Row, Column, Angle, Score)
Vizion.dev_display (Scene)
Vizion.dev_set_line_width (2)
Vizion.dev_display_ncc_matching_results (ModelID, Row, Column, Angle)
Count := |Row|
Label := 'ncc ' + Count
Vizion.dev_set_color ('white')
Vizion.dev_disp_text (Label, 24, 12)
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
Vizion.read_image (Image, 'samples/gauge.png')
Vizion.dev_set_line_width (2)
Vizion.dev_display (Image)
Vizion.gauge_point (Image, Mark, 110, 80, 110, 220, 1, 8, 'all', 'first', PRow, PCol, Amp, Dist)
Vizion.dev_set_color ('white')
Vizion.dev_display (Mark)
Vizion.gauge_line (Image, Line, 200, 380, 200, 480, 12, 1, 8, 'all', 'best', LRow1, LCol1, LRow2, LCol2, LAngle, LUsed)
Vizion.dev_set_color ('yellow')
Vizion.dev_display (Line)
Vizion.gauge_circle (Image, Circle, 110, 150, 42, 12, 1, 8, 'all', 'best', CRow, CCol, CRad, CUsed)
Vizion.dev_set_color ('green')
Vizion.dev_display (Circle)
Vizion.gauge_rectangle (Image, Rect, 244.5, 429.5, 0, 70, 45, 12, 1, 8, 'all', 'best', RRow, RCol, RPhi, RL1, RL2, RUsed)
Vizion.dev_set_color ('cyan')
Vizion.dev_display (Rect)
Vizion.gauge_wedge (Image, Wedge, 340, 170, 24, 58, 0.35, 1.8, 10, 1, 8, 'all', 'best', WRow, WCol, WR1, WR2, WStart, WExtent, WUsed)
Vizion.dev_set_color ('magenta')
Vizion.dev_display (Wedge)
Vizion.gauge_polygon (Image, Poly, [200, 200, 290, 290], [360, 500, 500, 360], 12, 1, 8, 'all', 'best', 1, GRows, GCols, GUsed)
Vizion.dev_set_color ('orange')
Vizion.dev_display (Poly)
Vizion.distance_pp (0, 0, 3, 4, Dist)
Vizion.distance_pl (20, 0, 0, 10, 50, 10, DistLine)
Vizion.distance_pc (CRow, CCol + CRad, CRow, CCol, CRad, DistCircle)
Vizion.angle_ll (0, 0, 0, 10, 0, 0, 10, 0, RightAngle)
Vizion.intersection_ll (0, 10, 50, 10, 20, 0, 20, 40, IRow, ICol)
"""


AUTO_THRESHOLD_PROGRAM = """* 自動二值化，再拆成連通物件
Vizion.read_image (Image, 'samples/parts.png')
Vizion.rgb1_to_gray (Image, Gray)
Vizion.binary_threshold (Gray, Region, 'max_separability', 'light', Used)
Vizion.opencv_threshold (Gray, CvRegion, 80, 'binary')
Vizion.connection (Region, Parts, 1)
Vizion.count_obj (Parts, Count)
Vizion.dev_display (Gray)
Vizion.dev_set_colored (16)
Vizion.dev_set_draw ('fill')
Vizion.dev_display (Parts)
Label := 'Otsu ' + Used + '  count ' + Count
Vizion.dev_set_color ('white')
Vizion.dev_disp_text (Label, 24, 12)
"""


MORPH_REGION_PROGRAM = """* 形態學：開運算去掉小點，閉運算補洞，再取邊界
Vizion.read_image (Image, 'samples/parts.png')
Vizion.rgb1_to_gray (Image, Gray)
Vizion.threshold (Gray, Region, 60, 255)
Vizion.opening_circle (Region, Opened, 2.5)
Vizion.closing_circle (Opened, Closed, 4.5)
Vizion.fill_up (Closed, Filled)
Vizion.dilation_circle (Filled, Dilated, 3.5)
Vizion.erosion_rectangle1 (Dilated, Eroded, 5, 5)
Vizion.boundary (Eroded, Border, 'inner')
Vizion.dev_display (Gray)
Vizion.dev_set_draw ('margin')
Vizion.dev_set_line_width (2)
Vizion.dev_set_color ('green')
Vizion.dev_display (Eroded)
Vizion.dev_set_color ('yellow')
Vizion.dev_display (Border)
"""


REGION_SETS_PROGRAM = """* 區域集合：矩形與圓的交集、差集、聯集
Vizion.read_image (Image, 'samples/parts.png')
Vizion.gen_rectangle1 (Rect, 70, 40, 240, 360)
Vizion.gen_circle (Disk, 150, 180, 80)
Vizion.intersection (Rect, Disk, Common)
Vizion.difference (Rect, Disk, OnlyRect)
Vizion.union2 (Rect, Disk, Both)
Vizion.complement (Common, Outside)
Vizion.dev_display (Image)
Vizion.dev_set_draw ('margin')
Vizion.dev_set_line_width (2)
Vizion.dev_set_color ('cyan')
Vizion.dev_display (Both)
Vizion.dev_set_color ('yellow')
Vizion.dev_display (Common)
Vizion.dev_set_color ('red')
Vizion.dev_display (OnlyRect)
"""


SUBPIX_BLOB_PROGRAM = """* 次像素輪廓，平滑後填回區域
Vizion.read_image (Image, 'samples/parts.png')
Vizion.rgb1_to_gray (Image, Gray)
Vizion.threshold_sub_pix (Gray, Border, 80)
Vizion.select_contours_xld (Border, Long, 'contlength', 30, 99999)
Vizion.smooth_contours_xld (Long, Smooth, 5)
Vizion.close_contours_xld (Smooth, Closed)
Vizion.gen_region_contour_xld (Closed, Region, 'filled')
Vizion.dev_display (Gray)
Vizion.dev_set_line_width (2)
Vizion.dev_set_color ('magenta')
Vizion.dev_display (Closed)
Vizion.dev_set_draw ('fill')
Vizion.dev_set_color ('#66ccff')
Vizion.dev_display (Region)
"""


XLD_FIT_PROGRAM = """* 條紋的次像素邊緣擬合成直線，並接上靠近的輪廓
Vizion.read_image (Image, 'samples/stripes.png')
Vizion.rgb1_to_gray (Image, Gray)
Vizion.edges_sub_pix (Gray, Edges, 'canny', 1, 20, 40)
Vizion.select_contours_xld (Edges, Long, 'contlength', 40, 99999)
Vizion.fit_line_contour_xld (Long, Lines, Row1, Col1, Row2, Col2)
Vizion.union_adjacent_contours_xld (Long, Joined, 12, 0.5)
Vizion.sort_contours_xld (Joined, Sorted, 'character', 'true', 'row')
Count := |Row1|
Vizion.dev_display (Gray)
Vizion.dev_set_line_width (2)
Vizion.dev_set_color ('green')
Vizion.dev_display (Lines)
Vizion.dev_set_color ('yellow')
Vizion.dev_display (Sorted)
Label := 'lines ' + Count
Vizion.dev_set_color ('white')
Vizion.dev_disp_text (Label, 20, 12)
"""


XLD_CIRCLE_PROGRAM = """* 亮斑外輪廓擬合圓與橢圓
Vizion.read_image (Image, 'samples/parts.png')
Vizion.rgb1_to_gray (Image, Gray)
Vizion.threshold (Gray, Region, 60, 255)
Vizion.connection (Region, Parts, 1)
Vizion.select_shape (Parts, Big, 'area', 'and', 400, 999999)
Vizion.gen_contour_region_xld (Big, Contours)
Vizion.fit_circle_contour_xld (Contours, Circles, Row, Column, Radius)
Vizion.fit_ellipse_contour_xld (Contours, Ellipses, ERow, ECol, Phi, Radius1, Radius2)
Vizion.length_xld (Contours, Length)
Vizion.area_center_xld (Contours, Area, CRow, CCol)
Vizion.dev_display (Gray)
Vizion.dev_set_line_width (2)
Vizion.dev_set_color ('green')
Vizion.dev_display (Circles)
Vizion.dev_set_color ('orange')
Vizion.dev_display (Ellipses)
"""


IMAGE_FILTER_PROGRAM = """* 影像濾波：高斯、中值、邊緣強化、直方圖等化、反相
Vizion.read_image (Image, 'samples/stripes.png')
Vizion.rgb1_to_gray (Image, Gray)
Vizion.gauss_image (Gray, Smooth, 5)
Vizion.median_image (Gray, Median, 2)
Vizion.mean_image (Gray, MeanImage, 5, 5)
Vizion.emphasize (Gray, Sharp, 7, 7, 1)
Vizion.scale_image (Sharp, Scaled, 1.2, 0)
Vizion.equ_histo_image (Gray, Equ)
Vizion.invert_image (Equ, Inverted)
Vizion.dev_display (Sharp)
"""


TUPLE_STATS_PROGRAM = """* 連通物件面積的 tuple 長度、最小、最大、總和、平均
Vizion.read_image (Image, 'samples/parts.png')
Vizion.rgb1_to_gray (Image, Gray)
Vizion.threshold (Gray, Region, 60, 255)
Vizion.connection (Region, Parts, 1)
Vizion.area_center (Parts, Area, Row, Column)
Vizion.tuple_length (Area, Length)
Vizion.tuple_min (Area, MinArea)
Vizion.tuple_max (Area, MaxArea)
Vizion.tuple_sum (Area, SumArea)
Vizion.tuple_mean (Area, MeanArea)
Vizion.dev_display (Gray)
Vizion.dev_set_colored (16)
Vizion.dev_display (Parts)
Label := 'n=' + Length + ' mean=' + MeanArea
Vizion.dev_set_color ('white')
Vizion.dev_disp_text (Label, 24, 12)
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
Vizion.create_matrix (2, 2, [4, 7, 2, 6], A)
Vizion.invert_matrix (A, 'general', 0, Inv)
Vizion.mult_matrix (A, Inv, 'AB', Eye)
Vizion.get_value_matrix (Eye, 0, 0, Eye00)
Vizion.vector_angle_to_rigid (10, 0, 0, 0, 10, rad(90), HomMat2D)
Vizion.affine_trans_point_2d (HomMat2D, 10, 0, Row, Col)
Vizion.create_pose (0, 0, 0, 0, 0, rad(90), 'Rp+T', 'gba', 'point', Pose)
Vizion.pose_to_hom_mat3d (Pose, HomMat3D)
Vizion.affine_trans_point_3d (HomMat3D, 1, 0, 0, Qx, Qy, Qz)
Vizion.read_image (Codes, 'samples/codes.png')
Vizion.create_bar_code_model ('', '', BarCode)
Vizion.find_bar_code (Codes, BarRegions, BarCode, 'EAN-13', BarText)
Vizion.create_data_code_2d_model ('QR Code', '', '', DataCode)
Vizion.find_data_code_2d (Codes, CodeXld, DataCode, '', '', CodeId, CodeText)
Vizion.read_image (Label, 'samples/ocr_label.png')
Vizion.create_text_model_reader ('auto', 'auto', TextModel)
Vizion.find_text (Label, TextModel, TextResult)
Vizion.get_text_result (TextResult, 'class', Words)
Vizion.dev_display (Codes)
Vizion.dev_set_color ('green')
Vizion.dev_display (CodeXld)
"""


HDEVELOP_BATCH2_PROGRAM = """* 量測、計量模型與縮放形狀
Vizion.read_image (Image, 'samples/measure_batch.png')
Vizion.rgb1_to_gray (Image, Gray)
Vizion.gen_measure_rectangle2 (100, 70, 0, 50, 4, 320, 240, 'bilinear', Measure)
Vizion.measure_pos (Gray, Measure, 1, 20, 'all', 'all', RowEdge, ColEdge, Amp, Dist)
Vizion.measure_pairs (Gray, Measure, 1, 20, 'positive', 'all', RowA, ColA, AmpA, RowB, ColB, AmpB, Intra, Inter)
Vizion.create_metrology_model (Metro)
Vizion.set_metrology_model_image_size (Metro, 320, 240)
Vizion.add_metrology_object_line_measure (Metro, 40, 40, 190, 40, 12, 3, 1, 20, LineIndex)
Vizion.add_metrology_object_circle_measure (Metro, 70, 180, 28, 12, 3, 1, 20, CircleIndex)
Vizion.add_metrology_object_ellipse_measure (Metro, 160, 220, rad(25), 36, 22, 12, 3, 1, 18, EllipseIndex)
Vizion.apply_metrology_model (Gray, Metro)
Vizion.get_metrology_object_result (Metro, LineIndex, 'all', 'result_type', 'column_begin', LineCol)
Vizion.get_metrology_object_result (Metro, CircleIndex, 'all', 'result_type', 'all_param', CircleParam)
Vizion.get_metrology_object_result (Metro, EllipseIndex, 'all', 'result_type', 'all_param', EllipseParam)
Vizion.get_metrology_object_result_contour (Contour, Metro, 'all', 'all', 2)
Vizion.crop_rectangle1 (Gray, Template, 40, 150, 100, 210)
Vizion.create_scaled_shape_model (Template, 2, 0, rad(15), 0.15, 0.9, 1.1, 0.1, 'none', 'use_polarity', 30, 10, ModelID)
Vizion.find_scaled_shape_model (Gray, ModelID, 0, rad(15), 0.85, 1.15, 0.4, 1, 0.5, 'least_squares', 0, 0.6, FoundRow, FoundCol, FoundAngle, FoundScale, FoundScore)
Vizion.get_shape_model_params (ModelID, Levels, AngleStart, AngleExtent, AngleStep, ScaleMin, ScaleMax, ScaleStep, Metric, MinContrast)
Vizion.dev_display (Gray)
Vizion.dev_set_color ('green')
Vizion.dev_display (Contour)
"""


HDEVELOP_BATCH1_PROGRAM = """* tuple、影像相加、平移和局部閾值
Vizion.read_image (Image, 'samples/parts.png')
Vizion.rgb1_to_gray (Image, Gray)
Vizion.tuple_gen_sequence (1, 5, 2, Seq)
Vizion.tuple_add (Seq, 10, Plus)
Vizion.tuple_select (Plus, 0, First)
Vizion.add_image (Gray, Gray, Sum, 0.5, 0)
Vizion.gauss_image (Gray, Smooth, 5)
Vizion.dyn_threshold (Gray, Smooth, Bright, 8, 'light')
Vizion.hom_mat2d_identity (Id)
Vizion.hom_mat2d_translate (Id, 12, 0, Mat)
Vizion.affine_trans_image (Sum, Moved, Mat, 'nearest_neighbor', 'false')
Vizion.dev_display (Moved)
Vizion.dev_set_color ('red')
Vizion.dev_disp_circle (120, 100, 30)
"""


OCR_READ_PROGRAM = """* 讀出圖上的文字，每個字塊一個外框
Vizion.read_image (Image, 'samples/ocr_label.png')
Vizion.ocr_find (Image, Words, 'auto', Text, Row, Column, Score)
Vizion.dev_display (Image)
Vizion.dev_set_line_width (2)
Vizion.dev_set_color ('#e6b800')
Vizion.dev_display (Words)
"""


HDEVELOP_BATCH4_PROGRAM = """* 檔案相機、Code 39、四元數和點雲
Vizion.open_framegrabber ('File', 'samples/code39.png', 'gray', 0, 0, Acq)
Vizion.grab_image (Image, Acq)
Vizion.close_framegrabber (Acq)
Vizion.create_bar_code_model (Bar)
Vizion.find_bar_code (Image, Regions, Bar, 'Code 39', Text39)
Vizion.axis_angle_to_quat (0, 0, 1, rad(90), Quat)
Vizion.quat_rotate_point (Quat, 1, 0, 0, Qx, Qy, Qz)
Vizion.gen_object_model_3d_from_points ([0, 1], [0, 0], [0, 0], Cloud)
Vizion.hom_mat3d_identity (Id)
Vizion.hom_mat3d_translate (Id, 0, 0, 5, Shift)
Vizion.affine_trans_object_model_3d (Cloud, Shift, Moved)
Vizion.get_object_model_3d_params (Moved, 'center', Center)
Vizion.create_class_knn (1, Knn)
Vizion.add_sample_class_knn (Knn, [0, 0], 1)
Vizion.add_sample_class_knn (Knn, [10, 0], 2)
Vizion.classify_class_knn (Knn, [0.2, 0], ClassId, Score)
Vizion.dev_display (Image)
Vizion.dev_set_color ('green')
Vizion.dev_display (Regions)
"""


HDEVELOP_BATCH5_PROGRAM = """* 灰階侵蝕、橢圓和對比
Vizion.gen_image_const (Image, 'byte', 160, 120)
Vizion.gen_rectangle1 (Box, 30, 40, 90, 120)
Vizion.paint_region (Box, Image, Painted, 200, 'fill')
Vizion.gray_erosion_rect (Painted, Eroded, 5, 5)
Vizion.scale_image_max (Eroded, Stretched)
Vizion.gen_ellipse (Oval, 60, 80, 0, 30, 12)
Vizion.distance_ps (0, 0, 0, 10, 0, 20, Dist)
Vizion.projection_pl (5, 4, 0, 0, 0, 10, RowProj, ColProj)
Vizion.dev_display (Stretched)
Vizion.dev_set_color ('green')
Vizion.dev_display (Oval)
"""


HDEVELOP_BATCH6_PROGRAM = """* 尺寸、灰階和頂帽
Vizion.gen_image_const (Base, 'byte', 80, 60)
Vizion.get_image_size (Base, Width, Height)
Vizion.gen_rectangle1 (Box, 10, 10, 40, 50)
Vizion.paint_region (Box, Base, Marked, 180, 'fill')
Vizion.min_max_gray (Box, Marked, 0, MinGray, MaxGray, Range)
Vizion.get_grayval (Marked, 20, 20, Gray)
Vizion.get_region_points (Box, Rows, Columns)
Vizion.tuple_length (Rows, Count)
Vizion.gen_circle (SE, 20, 20, 2)
Vizion.gray_tophat (Marked, SE, Peaks)
Vizion.get_grayval (Peaks, 25, 30, Peak)
Vizion.dev_display (Marked)
"""


HDEVELOP_BATCH7_PROGRAM = """* 斜坡、直徑和標準形
Vizion.gen_image_gray_ramp (Ramp, 0, 2, 10, 0, 0, 40, 30)
Vizion.get_grayval (Ramp, 4, 5, Gray)
Vizion.copy_image (Ramp, Copied)
Vizion.get_grayval (Copied, 4, 5, CopiedGray)
Vizion.gen_rectangle1 (Wide, 2, 2, 12, 18)
Vizion.gen_rectangle1 (Small, 2, 24, 6, 28)
Vizion.concat_obj (Wide, Small, Both)
Vizion.count_obj (Both, BothCount)
Vizion.select_shape_std (Both, Biggest, 'max_area', 100)
Vizion.area_center (Biggest, BigArea, BigRow, BigCol)
Vizion.copy_obj (Both, First, 1, 1)
Vizion.count_obj (First, FirstCount)
Vizion.gray_histo (Wide, Ramp, Histo, Relative)
Vizion.tuple_sum (Histo, HistoSum)
Vizion.gray_projections (Wide, Ramp, 'simple', Hor, Vert)
Vizion.tuple_select (Hor, 4, RowSum)
Vizion.diameter_region (Wide, DRow1, DCol1, DRow2, DCol2, Diameter)
Vizion.elliptic_axis (Wide, Ra, Rb, Phi)
Vizion.orientation_region (Wide, Angle)
Vizion.gen_region_line (Line, 0, 0, 10, 0)
Vizion.test_region_point (Line, 5, 0, Inside)
Vizion.test_region_point (Line, 5, 3, Outside)
Vizion.gen_region_points (Dots, [1, 2], [3, 4])
Vizion.area_center (Dots, DotArea, DotRow, DotCol)
Vizion.crop_part (Ramp, Part, 0, 0, 10, 8)
Vizion.get_image_size (Part, PartWidth, PartHeight)
Vizion.deviation_image (Ramp, Dev, 5, 5)
Vizion.get_grayval (Dev, 15, 20, DevGray)
Vizion.gray_range_rect (Ramp, Span, 5, 5)
Vizion.get_grayval (Span, 15, 20, SpanGray)
Vizion.sobel_dir (Ramp, Amp, Dir, 'sum_sqrt', 3)
Vizion.get_grayval (Amp, 15, 20, AmpGray)
Vizion.get_grayval (Dir, 15, 20, DirGray)
Vizion.gen_circle_contour_xld (Circle, 15, 20, 8, 0, rad(360), 'positive', 1)
Vizion.length_xld (Circle, CircLen)
Vizion.gen_rectangle2_contour_xld (Rect, 15, 20, 0, 10, 4)
Vizion.length_xld (Rect, RectLen)
Vizion.distance_ss (0, 0, 0, 10, 5, 0, 5, 10, SegDist)
Vizion.distance_ss (0, 0, 10, 10, 0, 10, 10, 0, CrossDist)
Vizion.dev_display (Ramp)
"""


HDEVELOP_BATCH8_PROGRAM = """* 對比拉伸、投影和箭頭
Vizion.gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 64, 48)
Vizion.scale_image_range (Ramp, Scaled, 0, 63)
Vizion.get_grayval (Scaled, 0, 63, Top)
Vizion.derivate_gauss (Ramp, Dx, 1.0, 'x')
Vizion.derivate_gauss (Ramp, Dxx, 1.0, 'xx')
Vizion.get_grayval (Dx, 24, 32, Deriv)
Vizion.get_grayval (Dxx, 24, 32, Deriv2)
Vizion.gen_image_proto (Ramp, Flat, 40)
Vizion.texture_laws (Flat, Energy, 'el', 0, 5)
Vizion.texture_laws (Flat, Level, 'll', 0, 5)
Vizion.get_grayval (Energy, 24, 32, EdgeEnergy)
Vizion.get_grayval (Level, 24, 32, LevelGray)
Vizion.gen_rectangle1 (Outer, 8, 8, 32, 42)
Vizion.gen_rectangle1 (Inner, 14, 16, 24, 30)
Vizion.difference (Outer, Inner, Ring)
Vizion.fill_up_shape (Ring, Filled, 'area', 1, 99999)
Vizion.area_center (Ring, HoleArea, HoleRow, HoleCol)
Vizion.area_center (Filled, FullArea, FullRow, FullCol)
Vizion.gen_rectangle1 (Box, 10, 10, 30, 40)
Vizion.distance_pr (Box, 10, 10, InsideMin, InsideMax)
Vizion.distance_pr (Box, 0, 0, OutsideMin, OutsideMax)
Vizion.angle_lx (0, 0, 0, 10, Angle)
Vizion.line_position (0, 0, 0, 10, RowCenter, ColCenter, Length, Phi)
Vizion.gen_contour_polygon_xld (Segment, [10, 10], [0, 20])
Vizion.gen_parallel_contour_xld (Segment, Parallel, 'contour_normal', 4)
Vizion.get_contour_xld (Parallel, ParRows, ParCols)
Vizion.tuple_mean (ParRows, ParRow)
Vizion.smallest_rectangle1_xld (Segment, Row1, Column1, Row2, Column2)
Vizion.smallest_circle_xld (Segment, CircleRow, CircleCol, CircleRadius)
Vizion.distance_cc (Segment, Parallel, 'point_to_segment', ContourMin, ContourMax)
Vizion.gen_arrow_contour_xld (Arrow, 5, 5, 5, 25, 6, 3)
Vizion.count_obj (Arrow, ArrowCount)
Vizion.paint_xld (Arrow, Ramp, Painted, 200)
Vizion.get_grayval (Painted, 5, 15, ArrowGray)
Vizion.gen_image_const (Base, 'byte', 8, 8)
Vizion.gen_image_proto (Base, Hue, 0)
Vizion.gen_image_proto (Base, Sat, 255)
Vizion.gen_image_proto (Base, Val, 255)
Vizion.trans_to_rgb (Hue, Sat, Val, Red, Green, Blue, 'hsv')
Vizion.get_grayval (Red, 1, 1, RedGray)
Vizion.get_grayval (Green, 1, 1, GreenGray)
Vizion.get_grayval (Blue, 1, 1, BlueGray)
Vizion.gen_image_const (Dot, 'byte', 40, 30)
Vizion.set_grayval (Dot, Marked, 10, 12, 255)
Vizion.hom_vector_to_proj_hom_mat2d ([0, 0, 20, 20], [0, 30, 30, 0], [1, 1, 1, 1], [7, 7, 27, 27], [5, 35, 35, 5], [1, 1, 1, 1], HomMat)
Vizion.projective_trans_pixel (HomMat, 10, 12, RowT, ColT)
Vizion.projective_trans_image (Marked, Moved, HomMat, 'nearest_neighbor', 'false')
Vizion.get_grayval (Moved, 17, 17, MovedGray)
Vizion.get_grayval (Moved, 10, 12, OldGray)
Vizion.dev_display (Scaled)
"""


HDEVELOP_BATCH9_PROGRAM = """* 對齊、角點和區域距離
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Left, 10, 10, 25, 30)
Vizion.gen_rectangle1 (Right, 10, 42, 25, 60)
Vizion.distance_rr (Left, Right, DistMin, DistMax)
Vizion.vector_to_hom_mat2d ([0, 0, 10], [0, 10, 0], [5, 5, 15], [3, 13, 3], Affine)
Vizion.affine_trans_pixel (Affine, 2, 4, RowA, ColA)
Vizion.vector_to_similarity ([0, 0], [0, 10], [0, 0], [0, 20], Similar)
Vizion.affine_trans_pixel (Similar, 0, 10, RowS, ColS)
Vizion.vector_to_rigid ([0, 0, 10], [0, 10, 0], [0, -10, 0], [0, 0, 10], Rigid)
Vizion.affine_trans_pixel (Rigid, 0, 10, RowR, ColR)
Vizion.hom_mat2d_identity (Id)
Vizion.hom_mat2d_reflect (Id, 0, 0, 10, 0, Mirror)
Vizion.affine_trans_pixel (Mirror, 0, 10, RowM, ColM)
Vizion.paint_region (Left, Image, Block, 255, 'fill')
Vizion.points_harris (Block, 1, 0.05, 0.04, CornerRows, CornerCols)
Vizion.tuple_length (CornerRows, CornerCount)
Vizion.tuple_min (CornerRows, CornerTop)
Vizion.set_grayval (Image, Peak, 20, 20, 255)
Vizion.local_max (Peak, Peaks)
Vizion.test_region_point (Peaks, 20, 20, PeakInside)
Vizion.gray_erosion_circle (Peak, Eroded, 1)
Vizion.get_grayval (Eroded, 20, 20, ErodedGray)
Vizion.gen_image_proto (Image, Flat, 80)
Vizion.gray_erosion_circle (Flat, FlatErode, 2)
Vizion.get_grayval (FlatErode, 30, 30, FlatGray)
Vizion.get_region_contour (Left, CRows, CCols)
Vizion.tuple_length (CRows, ContourCount)
Vizion.gen_rectangle1 (Big, 4, 4, 20, 40)
Vizion.gen_rectangle1 (Tiny, 40, 4, 44, 8)
Vizion.concat_obj (Big, Tiny, Shapes)
Vizion.select_shape_proto (Shapes, Big, Kept, 'area', 0.5, 1.5)
Vizion.count_obj (Kept, KeptCount)
Vizion.moments_region_2nd (Right, M11, M20, M02)
Vizion.projective_trans_region (Left, Shifted, Affine, 'nearest_neighbor')
Vizion.test_region_point (Shifted, 20, 23, MovedIn)
Vizion.test_region_point (Shifted, 10, 10, MovedOut)
Vizion.gen_contour_polygon_xld (Segment, [8, 8], [8, 18])
Vizion.projective_trans_contour_xld (Segment, SegmentT, Affine)
Vizion.get_contour_xld (SegmentT, TRows, TCols)
Vizion.tuple_mean (TRows, MeanRow)
Vizion.tuple_mean (TCols, MeanCol)
Vizion.gen_region_line (Thin, 40, 5, 40, 50)
Vizion.gen_contours_skeleton_xld (Thin, Skel)
Vizion.length_xld (Skel, SkelLen)
Vizion.convert_image_type (Image, Copied, 'byte')
Vizion.get_grayval (Copied, 0, 0, Zero)
Vizion.cfa_to_rgb (Image, Bayer, 'bayer_rg')
Vizion.count_channels (Bayer, Channels)
Vizion.dev_display (Block)
"""


HDEVELOP_BATCH10_PROGRAM = """* 次方、脊線和交點
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Bar, 29, 8, 31, 72)
Vizion.gen_rectangle1 (Hole, 20, 20, 24, 24)
Vizion.gen_region_points (Dot, [10], [20])
Vizion.gen_rectangle2_contour_xld (Rect, 30, 40, 0, 12, 5)
Vizion.paint_region (Bar, Image, Ridges, 255, 'fill')
Vizion.lines_gauss (Ridges, Lines, 1.2, 1, 5, 'light')
Vizion.length_xld (Lines, LineLen)
Vizion.tuple_max (LineLen, Longest)
Vizion.gen_image_proto (Image, Bright, 100)
Vizion.set_grayval (Bright, Dim, 15, 15, 0)
Vizion.local_min (Dim, Mins)
Vizion.test_region_point (Mins, 15, 15, DarkInside)
Vizion.paint_region (Hole, Bright, Broken, 0, 'fill')
Vizion.harmonic_interpolation (Broken, Hole, Filled)
Vizion.get_grayval (Filled, 22, 22, FilledGray)
Vizion.transpose_region (Dot, Flipped, 0, 0)
Vizion.test_region_point (Flipped, 20, 10, FlippedInside)
Vizion.smallest_rectangle2_xld (Rect, RectRow, RectCol, RectPhi, Len1, Len2)
Vizion.points_harris (Ridges, 1, 0.05, 0.04, 'true', SubRows, SubCols)
Vizion.tuple_length (SubRows, SubCount)
Vizion.gen_image_proto (Image, Mid, 128)
Vizion.pow_image (Mid, Powered, 2)
Vizion.get_grayval (Powered, 10, 10, PowGray)
Vizion.gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 40, 30)
Vizion.rank_rect (Ramp, Ranked, 3, 3, 0)
Vizion.get_grayval (Ramp, 10, 10, RampGray)
Vizion.get_grayval (Ranked, 10, 10, RankLow)
Vizion.intersection_segments (0, 0, 10, 10, 0, 10, 10, 0, CrossRow, CrossCol, Overlap)
Vizion.intersection_line_circle (0, -10, 0, 10, 0, 0, 5, CircRow, CircCol)
Vizion.tuple_length (CircCol, Hits)
Vizion.distance_sl (0, 0, 0, 10, 5, 0, 5, 20, LineDist)
Vizion.distance_sl (0, 0, 10, 10, 5, 0, 5, 10, CrossDist)
Vizion.gen_image_const (LeftImage, 'byte', 8, 6)
Vizion.gen_image_const (RightImage, 'byte', 4, 6)
Vizion.tile_images (LeftImage, RightImage, Wide, 'horizontal')
Vizion.get_image_size (Wide, WideW, WideH)
Vizion.dev_display (Ridges)
"""


HDEVELOP_BATCH11_PROGRAM = """* 濾波、骨架端點和圓的交點
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Box, 10, 10, 12, 20)
Vizion.get_region_runs (Box, RunRow, RunBegin, RunEnd)
Vizion.tuple_length (RunRow, RunCount)
Vizion.tuple_min (RunRow, RunTop)
Vizion.tuple_max (RunRow, RunBot)
Vizion.tuple_min (RunBegin, RunLeft)
Vizion.tuple_max (RunEnd, RunRight)
Vizion.gen_region_runs (Rebuilt, RunRow, RunBegin, RunEnd)
Vizion.area_center (Rebuilt, RebuiltArea, SkipR, SkipC)
Vizion.gen_rectangle1 (Wide, 10, 10, 30, 40)
Vizion.clip_region_rel (Wide, Clipped, 2, 2, 5, 5)
Vizion.test_region_point (Clipped, 12, 15, ClipIn)
Vizion.test_region_point (Clipped, 10, 10, ClipOut)
Vizion.gen_circle (Disk, 30, 40, 12)
Vizion.inner_circle (Disk, InRow, InCol, InRad)
Vizion.gen_region_line (Horiz, 30, 10, 30, 70)
Vizion.gen_region_line (Vert, 10, 40, 50, 40)
Vizion.union2 (Horiz, Vert, Plus)
Vizion.skeleton (Plus, Thin)
Vizion.junctions_skeleton (Plus, Ends, Juncs)
Vizion.area_center (Ends, EndArea, EndRow, EndCol)
Vizion.area_center (Juncs, JuncArea, JuncRow, JuncCol)
Vizion.test_region_point (Juncs, 30, 40, JuncInside)
Vizion.gen_grid_region (Grid, 10, 10, 'lines', 80, 60)
Vizion.test_region_point (Grid, 0, 0, GridOn)
Vizion.test_region_point (Grid, 5, 5, GridOff)
Vizion.gen_grid_region (Cells, 20, 20, 'rects', 80, 60)
Vizion.count_obj (Cells, CellCount)
Vizion.gen_contour_polygon_xld (Wiggle, [10, 12, 9, 11, 10], [5, 15, 25, 35, 45])
Vizion.regress_contours_xld (Wiggle, FitLine, 'no', 0)
Vizion.get_contour_xld (FitLine, FitRows, FitCols)
Vizion.tuple_mean (FitRows, FitRow)
Vizion.tuple_min (FitCols, FitColMin)
Vizion.tuple_max (FitCols, FitColMax)
Vizion.orientation_xld (Wiggle, Phi)
Vizion.intersection_circles (0, 0, 5, 0, 6, 5, CircRow, CircCol)
Vizion.tuple_length (CircCol, CircHits)
Vizion.intersection_segment_circle (0, -10, 0, 10, 0, 0, 5, SegRow, SegCol)
Vizion.tuple_length (SegCol, SegHits)
Vizion.gen_image_proto (Image, Flat, 80)
Vizion.bilateral_filter (Flat, Soft, 3, 20)
Vizion.get_grayval (Soft, 10, 10, SoftGray)
Vizion.gen_image_proto (Image, Mid, 128)
Vizion.convol_image (Mid, Smoothed, 'smooth3', 0, 'continued')
Vizion.get_grayval (Smoothed, 10, 10, SmoothGray)
Vizion.convol_image (Mid, Edged, 'sobel_x', 0, 'continued')
Vizion.get_grayval (Edged, 10, 10, EdgeGray)
Vizion.gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 40, 30)
Vizion.convol_image (Ramp, Same, [0, 0, 0, 0, 1, 0, 0, 0, 0], 3, 'continued')
Vizion.get_grayval (Same, 10, 10, SameGray)
Vizion.dev_display (Image)
"""


HDEVELOP_BATCH12_PROGRAM = """* 內接矩形、孔洞和圓角
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Solid, 10, 10, 40, 50)
Vizion.inner_rectangle1 (Solid, InR1, InC1, InR2, InC2)
Vizion.gen_rectangle1 (Outer, 10, 10, 40, 40)
Vizion.gen_rectangle1 (Hole, 20, 20, 30, 30)
Vizion.difference (Outer, Hole, Frame)
Vizion.area_holes (Frame, HoleArea)
Vizion.gen_contour_region_xld (Frame, OuterOnly, 'border')
Vizion.gen_contour_region_xld (Frame, WithHoles, 'border_holes')
Vizion.count_obj (OuterOnly, OuterCount)
Vizion.count_obj (WithHoles, HoleCount)
Vizion.gen_rectangle1 (Left, 10, 10, 20, 20)
Vizion.gen_rectangle1 (Right, 10, 15, 20, 30)
Vizion.hamming_distance (Left, Right, HamDist, HamSim)
Vizion.gen_rectangle1 (Sym, 10, 10, 20, 20)
Vizion.moments_region_3rd (Sym, S21, S12, S03, S30)
Vizion.gen_rectangle1 (Wing, 10, 10, 12, 40)
Vizion.union2 (Sym, Wing, Heavy)
Vizion.moments_region_3rd (Heavy, H21, H12, H03, H30)
Vizion.gen_rectangle1 (BlobA, 10, 10, 30, 20)
Vizion.gen_rectangle1 (BlobB, 10, 40, 30, 55)
Vizion.union2 (BlobA, BlobB, Pair)
Vizion.partition_dynamic (Pair, Parts, 18, 50)
Vizion.count_obj (Parts, PartCount)
Vizion.area_center (Parts, PartArea, PartRow, PartCol)
Vizion.gen_ellipse_contour_xld (Oval, 30, 40, 0, 20, 8, 0, rad(360), 'positive', 1)
Vizion.elliptic_axis_xld (Oval, Ra, Rb, AxisPhi)
Vizion.gen_contour_polygon_rounded_xld (Round, [0, 0, 20], [0, 20, 20], 5, 1)
Vizion.get_contour_xld (Round, RoundRows, RoundCols)
Vizion.gen_contour_polygon_xld (Chain, [0, 1, 2, 3, 4], [10, 11, 12, 13, 14])
Vizion.clip_end_points_contours_xld (Chain, Clipped, 'num_points', 1)
Vizion.get_contour_xld (Clipped, ClipRows, ClipCols)
Vizion.tuple_length (ClipRows, ClipCount)
Vizion.tuple_min (ClipRows, ClipFirst)
Vizion.tuple_max (ClipRows, ClipLast)
Vizion.gen_contour_polygon_xld (Span, [10, 10, 10], [0, 20, 40])
Vizion.clip_end_points_contours_xld (Span, Short, 'length', 10)
Vizion.get_contour_xld (Short, ShortRows, ShortCols)
Vizion.tuple_min (ShortCols, ShortMin)
Vizion.tuple_max (ShortCols, ShortMax)
Vizion.gen_contour_polygon_xld (DiagA, [0, 10], [0, 10])
Vizion.gen_contour_polygon_xld (DiagB, [0, 10], [10, 0])
Vizion.intersection_contours_xld (DiagA, DiagB, CrossRow, CrossCol)
Vizion.dev_display (Image)
"""


HDEVELOP_BATCH13_PROGRAM = """* 差異、字元和轉角
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_image_proto (Image, Flat, 100)
Vizion.set_grayval (Flat, Changed, 20, 20, 180)
Vizion.check_difference (Changed, Flat, Diff, 'abs', 30)
Vizion.test_region_point (Diff, 20, 20, DiffIn)
Vizion.test_region_point (Diff, 10, 10, DiffOut)
Vizion.check_difference (Changed, Flat, Dark, 'dark', 30)
Vizion.test_region_point (Dark, 20, 20, DarkIn)
Vizion.set_grayval (Image, DotImage, 40, 40, 255)
Vizion.dots_image (DotImage, Dots, 5, 'light', 40)
Vizion.test_region_point (Dots, 40, 40, DotIn)
Vizion.test_region_point (Dots, 10, 10, DotOut)
Vizion.gen_image_proto (Image, Page, 180)
Vizion.gen_rectangle1 (Glyph, 15, 15, 25, 25)
Vizion.paint_region (Glyph, Page, Scene, 20, 'fill')
Vizion.get_domain (Scene, Domain)
Vizion.char_threshold (Scene, Domain, Chars, 2, 90, Used)
Vizion.test_region_point (Chars, 20, 20, CharIn)
Vizion.test_region_point (Chars, 5, 5, CharOut)
Vizion.gen_rectangle1 (All, 0, 0, 59, 79)
Vizion.cooc_feature_image (All, Flat, 8, '0', Energy, Correlation, Homogeneity, Contrast)
Vizion.gen_image_proto (Image, Bright, 255)
Vizion.gen_rectangle1 (Half, 0, 0, 59, 39)
Vizion.paint_region (Half, Bright, Edge, 0, 'fill')
Vizion.cooc_feature_image (All, Edge, 8, '0', EdgeEnergy, EdgeCorr, EdgeHomo, EdgeContrast)
Vizion.gen_rectangle1 (Left, 10, 10, 20, 20)
Vizion.gen_rectangle1 (Right, 15, 15, 25, 25)
Vizion.symm_difference (Left, Right, Xor)
Vizion.test_region_point (Xor, 12, 12, OnlyLeft)
Vizion.test_region_point (Xor, 17, 17, Overlap)
Vizion.test_region_point (Xor, 22, 22, OnlyRight)
Vizion.concat_obj (Left, Right, Both)
Vizion.get_region_index (Both, 15, 12, IndexIn)
Vizion.get_region_index (Both, 5, 5, IndexOut)
Vizion.get_region_index (Both, 18, 22, IndexSecond)
Vizion.gen_rectangle1 (NearA, 10, 10, 30, 18)
Vizion.gen_rectangle1 (NearB, 10, 30, 30, 40)
Vizion.find_neighbors (NearA, NearB, Far1, Far2, 5)
Vizion.find_neighbors (NearA, NearB, Close1, Close2, 15)
Vizion.count_obj (Far1, FarCount)
Vizion.count_obj (Close1, CloseCount)
Vizion.gen_contour_polygon_xld (Straight, [10, 10, 10], [0, 10, 20])
Vizion.get_contour_angle_xld (Straight, 'abs', StraightAngles)
Vizion.tuple_max (StraightAngles, StraightMax)
Vizion.gen_contour_polygon_xld (Corner, [0, 0, 10], [0, 10, 10])
Vizion.get_contour_angle_xld (Corner, 'rel', TurnAngles)
Vizion.tuple_max (TurnAngles, TurnMax)
Vizion.gen_contour_polygon_xld (OpenLine, [0, 0], [0, 10])
Vizion.test_closed_xld (OpenLine, OpenFlag)
Vizion.gen_contour_polygon_xld (Loop, [0, 0, 10, 0], [0, 10, 10, 0])
Vizion.test_closed_xld (Loop, ClosedFlag)
Vizion.dev_display (Image)
"""


HDEVELOP_BATCH14_PROGRAM = """* 邊緣、高原和光流
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_empty_region (Empty)
Vizion.count_obj (Empty, EmptyCount)
Vizion.gen_rectangle1 (Left, 10, 10, 20, 20)
Vizion.gen_rectangle1 (Right, 10, 30, 20, 40)
Vizion.concat_obj (Left, Right, Both)
Vizion.select_region_point (Both, Picked, 15, 35)
Vizion.count_obj (Picked, PickedCount)
Vizion.test_region_point (Picked, 15, 12, PickedOld)
Vizion.test_region_point (Picked, 15, 35, PickedNew)
Vizion.gen_rectangle1 (Block, 20, 20, 24, 24)
Vizion.paint_region (Block, Image, Blocks, 200, 'fill')
Vizion.plateaus (Blocks, Flats, 3)
Vizion.test_region_point (Flats, 22, 22, FlatIn)
Vizion.test_region_point (Flats, 20, 20, FlatEdge)
Vizion.test_region_point (Flats, 5, 5, FlatBack)
Vizion.gen_image_proto (Image, Dark, 0)
Vizion.gen_rectangle1 (RightHalf, 0, 40, 59, 79)
Vizion.paint_region (RightHalf, Dark, EdgeImage, 255, 'fill')
Vizion.edges_image (EdgeImage, Amp, Dir, 'sobel', 1, 'nms', 10, 30)
Vizion.get_grayval (Amp, [30, 30, 30, 30, 30, 30, 30, 30], [36, 37, 38, 39, 40, 41, 42, 43], EdgeAmps)
Vizion.tuple_max (EdgeAmps, EdgePeak)
Vizion.get_grayval (Amp, 30, 10, FlatAmp)
Vizion.get_grayval (Dir, [30, 30, 30, 30, 30, 30, 30, 30], [36, 37, 38, 39, 40, 41, 42, 43], EdgeDirs)
Vizion.gen_rectangle1 (Seed, 5, 38, 18, 42)
Vizion.close_edges (Seed, EdgeImage, Closed, 20)
Vizion.test_region_point (Closed, [40, 40, 40], [38, 39, 40], ClosedFlags)
Vizion.tuple_max (ClosedFlags, ClosedOn)
Vizion.test_region_point (Closed, 40, 10, ClosedOff)
Vizion.gen_image_proto (Image, Ground, 90)
Vizion.gen_rectangle1 (ValleyL, 10, 8, 40, 22)
Vizion.gen_rectangle1 (ValleyR, 10, 40, 40, 54)
Vizion.paint_region (ValleyL, Ground, Wells, 30, 'fill')
Vizion.paint_region (ValleyR, Wells, Wells, 30, 'fill')
Vizion.watersheds_threshold (Wells, LowBasins, 40)
Vizion.watersheds_threshold (Wells, HighBasins, 100)
Vizion.count_obj (LowBasins, LowCount)
Vizion.count_obj (HighBasins, HighCount)
Vizion.gen_rectangle1 (CornerBox, 12, 12, 36, 50)
Vizion.paint_region (CornerBox, Image, Corners, 255, 'fill')
Vizion.points_foerstner (Corners, 1.5, 0.05, CornerRows, CornerCols)
Vizion.tuple_length (CornerRows, CornerCount)
Vizion.tuple_min (CornerRows, CornerTop)
Vizion.gen_image_proto (Image, Plain, 20)
Vizion.gen_circle (Spot, 30, 30, 16)
Vizion.paint_region (Spot, Plain, SpotImage, 220, 'fill')
Vizion.hom_mat2d_identity (Id)
Vizion.hom_mat2d_translate (Id, 0, 4, Shift)
Vizion.affine_trans_image (SpotImage, Moved, Shift, 'constant', 'false')
Vizion.optical_flow_mg (SpotImage, Moved, RowFlow, ColFlow, 1)
Vizion.get_grayval (ColFlow, 30, 30, FlowCol)
Vizion.get_grayval (RowFlow, 30, 30, FlowRow)
Vizion.dev_display (EdgeImage)
"""


HDEVELOP_BATCH15_PROGRAM = """* 距離、高通和角點
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Box, 20, 20, 40, 50)
Vizion.distance_transform (Box, DistNear, 'euclidean', 'true')
Vizion.distance_transform (Box, DistCity, 'city-block', 'true')
Vizion.distance_transform (Box, DistChess, 'chessboard', 'true')
Vizion.distance_transform (Box, DistInside, 'euclidean', 'false')
Vizion.get_grayval (DistNear, 30, 35, NearIn)
Vizion.get_grayval (DistNear, 10, 35, NearOut)
Vizion.get_grayval (DistCity, 10, 35, CityOut)
Vizion.get_grayval (DistChess, 10, 35, ChessOut)
Vizion.get_grayval (DistInside, 30, 35, InsideDist)
Vizion.get_grayval (DistInside, 10, 35, InsideOut)
Vizion.region_to_bin (Box, BinImage, 200, 40)
Vizion.get_grayval (BinImage, 30, 35, BinIn)
Vizion.get_grayval (BinImage, 5, 5, BinOut)
Vizion.gen_image_proto (Image, Blank, 0)
Vizion.gen_rectangle1 (Left, 8, 8, 22, 24)
Vizion.gen_rectangle1 (Right, 8, 40, 22, 60)
Vizion.paint_region (Left, Blank, Painted, 80, 'fill')
Vizion.paint_region (Right, Painted, Painted, 180, 'fill')
Vizion.concat_obj (Left, Right, Parts)
Vizion.region_to_mean (Painted, Parts, MeanImage)
Vizion.get_grayval (MeanImage, 15, 16, MeanLeft)
Vizion.get_grayval (MeanImage, 15, 50, MeanRight)
Vizion.get_grayval (MeanImage, 40, 16, MeanOut)
Vizion.highpass_image (Image, HighFlat, 5, 5)
Vizion.get_grayval (HighFlat, 30, 30, HighZero)
Vizion.gen_image_proto (Image, Dark, 10)
Vizion.set_grayval (Dark, Spike, 30, 40, 255)
Vizion.highpass_image (Spike, HighSpike, 3, 3)
Vizion.get_grayval (HighSpike, 30, 40, HighPeak)
Vizion.get_grayval (HighSpike, 30, 10, HighBack)
Vizion.median_rect (Spike, Clean, 3, 3)
Vizion.get_grayval (Clean, 30, 40, MedGray)
Vizion.gen_image_proto (Image, Field, 40)
Vizion.set_grayval (Field, Noisy, 30, 12, 90)
Vizion.gen_rectangle1 (Bright, 0, 45, 59, 79)
Vizion.paint_region (Bright, Noisy, Scene, 255, 'fill')
Vizion.anisotropic_diffusion (Scene, Smooth, 'perona-malik', 25, 0.2, 12)
Vizion.get_grayval (Smooth, 30, 12, SpikeLeft)
Vizion.get_grayval (Smooth, 30, 70, EdgeKeep)
Vizion.get_grayval (Smooth, 30, 4, Quiet)
Vizion.gen_image_proto (Image, Peak, 0)
Vizion.set_grayval (Peak, Peak, 29, 39, 40)
Vizion.set_grayval (Peak, Peak, 29, 40, 80)
Vizion.set_grayval (Peak, Peak, 29, 41, 40)
Vizion.set_grayval (Peak, Peak, 30, 39, 80)
Vizion.set_grayval (Peak, Peak, 30, 40, 200)
Vizion.set_grayval (Peak, Peak, 30, 41, 160)
Vizion.set_grayval (Peak, Peak, 31, 39, 40)
Vizion.set_grayval (Peak, Peak, 31, 40, 80)
Vizion.set_grayval (Peak, Peak, 31, 41, 40)
Vizion.local_max_sub_pix (Peak, 'facet', 1, 20, PeakRows, PeakCols)
Vizion.tuple_length (PeakRows, PeakCount)
Vizion.gen_circle (Spot, 30, 40, 10)
Vizion.paint_region (Spot, Image, Corners, 255, 'fill')
Vizion.points_lepetit (Corners, 40, FastRows, FastCols)
Vizion.tuple_length (FastRows, FastCount)
Vizion.tuple_min (FastRows, FastTop)
Vizion.dev_display (DistNear)
"""


HDEVELOP_BATCH16_PROGRAM = """* 短行程、平面和熵
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Thin, 10, 10, 30, 11)
Vizion.gen_rectangle1 (Wide, 10, 10, 30, 20)
Vizion.eliminate_runs (Thin, Gone, 3)
Vizion.eliminate_runs (Wide, Kept, 3)
Vizion.count_obj (Gone, GoneCount)
Vizion.area_center (Kept, WideArea, WideRow, WideCol)
Vizion.gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 80, 60)
Vizion.gen_rectangle1 (PlaneBox, 10, 10, 19, 19)
Vizion.moments_gray_plane (PlaneBox, Ramp, MRow, MCol, Alpha, Beta, Mean)
Vizion.plane_deviation (PlaneBox, Ramp, Deviation)
Vizion.area_center_gray (PlaneBox, Ramp, GrayArea, GrayRow, GrayCol)
Vizion.gen_image_proto (Image, Flat, 40)
Vizion.set_grayval (Flat, Bumpy, 15, 45, 200)
Vizion.gen_rectangle1 (BumpBox, 10, 40, 19, 49)
Vizion.plane_deviation (BumpBox, Bumpy, BumpDev)
Vizion.gen_image_proto (Image, Blank, 0)
Vizion.gen_rectangle1 (RightHalf, 10, 20, 29, 29)
Vizion.paint_region (RightHalf, Blank, Split, 255, 'fill')
Vizion.gen_rectangle1 (Both, 10, 10, 29, 29)
Vizion.entropy_gray (Both, Split, Entropy)
Vizion.bandpass_image (Image, BandFlat, 3, 9)
Vizion.get_grayval (BandFlat, 30, 30, BandZero)
Vizion.gen_rectangle1 (DotBox, 29, 14, 31, 16)
Vizion.paint_region (DotBox, Blank, Dot, 255, 'fill')
Vizion.bandpass_image (Dot, BandDot, 3, 9)
Vizion.get_grayval (BandDot, 30, 15, DotPeak)
Vizion.get_grayval (BandDot, 30, 50, DotBack)
Vizion.gen_rectangle1 (Block, 36, 36, 54, 64)
Vizion.paint_region (Block, Blank, WideGray, 255, 'fill')
Vizion.bandpass_image (WideGray, BandWide, 3, 9)
Vizion.get_grayval (BandWide, 45, 50, WideCenter)
Vizion.gen_image_proto (Image, Valley, 80)
Vizion.set_grayval (Valley, Valley, [29, 29, 29, 30, 30, 30, 31, 31, 31], [39, 40, 41, 39, 40, 41, 39, 40, 41], [160, 120, 160, 120, 0, 40, 160, 120, 160])
Vizion.local_min_sub_pix (Valley, 'facet', 1, 40, ValleyRows, ValleyCols)
Vizion.tuple_length (ValleyRows, ValleyCount)
Vizion.hom_mat2d_identity (Id)
Vizion.hom_mat2d_scale (Id, 2, 3, 0, 0, Scaled)
Vizion.hom_mat2d_rotate (Scaled, 0.5, 0, 0, Turned)
Vizion.hom_mat2d_translate (Turned, 4, -2, Moved)
Vizion.hom_mat2d_to_affine_par (Moved, Sx, Sy, Phi, Theta, Tx, Ty)
Vizion.dev_display (Ramp)
"""


HDEVELOP_BATCH17_PROGRAM = """* 對數、鞍點和修剪
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_image_proto (Image, Dark, 0)
Vizion.set_grayval (Dark, Mark, 30, 40, 81)
Vizion.sqrt_image (Mark, Root)
Vizion.get_grayval (Root, 30, 40, RootGray)
Vizion.get_grayval (Root, 10, 10, RootZero)
Vizion.set_grayval (Dark, Fifteen, 20, 20, 15)
Vizion.set_grayval (Fifteen, Fifteen, 20, 21, 255)
Vizion.log_image (Fifteen, LogScaled, 'scaled')
Vizion.log_image (Fifteen, LogNatural, 'natural')
Vizion.get_grayval (LogScaled, 20, 20, LogMid)
Vizion.get_grayval (LogScaled, 20, 21, LogHigh)
Vizion.get_grayval (LogScaled, 10, 10, LogZero)
Vizion.get_grayval (LogNatural, 20, 21, LogRaw)
Vizion.gen_image_proto (Image, Flat, 100)
Vizion.shock_filter (Flat, Still, 0.4, 4, 'laplace', 0.8)
Vizion.get_grayval (Still, 30, 30, StillGray)
Vizion.gen_image_proto (Image, Step, 0)
Vizion.gen_rectangle1 (Right, 0, 40, 59, 79)
Vizion.paint_region (Right, Step, Edge, 255, 'fill')
Vizion.gauss_image (Edge, Soft, 9)
Vizion.shock_filter (Soft, Sharp, 0.5, 6, 'laplace', 0)
Vizion.get_grayval (Soft, 30, 39, SoftLeft)
Vizion.get_grayval (Soft, 30, 41, SoftRight)
Vizion.get_grayval (Sharp, 30, 39, SharpLeft)
Vizion.get_grayval (Sharp, 30, 41, SharpRight)
Vizion.gen_image_proto (Image, Base, 128)
Vizion.set_grayval (Base, Saddle, [29, 29, 29, 30, 30, 30, 31, 31, 31], [39, 40, 41, 39, 40, 41, 39, 40, 41], [128, 127, 128, 129, 128, 129, 128, 127, 128])
Vizion.saddle_points_sub_pix (Saddle, 0, 0.2, SaddleRows, SaddleCols)
Vizion.tuple_length (SaddleRows, SaddleCount)
Vizion.add_noise_white (Flat, NoiseA, 20, 'uniform', 1)
Vizion.add_noise_white (Flat, NoiseB, 20, 'uniform', 1)
Vizion.add_noise_white (Flat, Quiet, 0, 'uniform')
Vizion.get_grayval (NoiseA, [30, 30, 30, 30], [10, 20, 30, 40], NoiseVals)
Vizion.get_grayval (NoiseB, [30, 30, 30, 30], [10, 20, 30, 40], NoiseAgain)
Vizion.get_grayval (Quiet, 30, 30, QuietGray)
Vizion.gen_rectangle1 (Box, 10, 20, 40, 55)
Vizion.get_region_polygon (Box, 2, PolyRows, PolyCols)
Vizion.tuple_length (PolyRows, PolyCount)
Vizion.tuple_min (PolyRows, PolyTop)
Vizion.tuple_max (PolyRows, PolyBottom)
Vizion.gen_region_line (Line, 10, 10, 10, 30)
Vizion.prune (Line, Short, 3)
Vizion.area_center (Short, PruneArea, PruneRow, PruneCol)
Vizion.gen_rectangle1 (Outer, 10, 10, 40, 40)
Vizion.gen_rectangle1 (Inner, 20, 20, 30, 30)
Vizion.difference (Outer, Inner, Frame)
Vizion.connect_and_holes (Frame, Parts, Holes)
Vizion.count_obj (Parts, PartCount)
Vizion.count_obj (Holes, HoleCount)
Vizion.area_center (Holes, HoleArea, HoleRow, HoleCol)
Vizion.gen_rectangle1 (LeftBlob, 8, 8, 18, 18)
Vizion.gen_rectangle1 (RightBlob, 8, 40, 18, 55)
Vizion.union2 (LeftBlob, RightBlob, Pair)
Vizion.connect_and_holes (Pair, PairParts, PairHoles)
Vizion.count_obj (PairParts, PairCount)
Vizion.count_obj (PairHoles, PairHoleCount)
Vizion.dev_display (Sharp)
"""


HDEVELOP_BATCH18_PROGRAM = """* 特徵、距離和自動閾值
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 80, 60)
Vizion.gen_rectangle1 (Box, 10, 10, 19, 19)
Vizion.gray_features (Box, Ramp, ['gray_min', 'gray_max', 'gray_mean'], GrayVals)
Vizion.gen_contour_polygon_xld (Square, [10, 10, 30, 30, 10], [10, 30, 30, 10, 10])
Vizion.xld_features (Square, 'contlength', SideLength)
Vizion.xld_features (Square, 'area', SquareArea)
Vizion.gen_contour_polygon_xld (Other, [10, 10, 30, 30, 10], [50, 70, 70, 50, 50])
Vizion.concat_obj (Square, Other, BothContours)
Vizion.select_xld_point (BothContours, Picked, 20, 20)
Vizion.select_xld_point (BothContours, Missed, 5, 5)
Vizion.count_obj (Picked, PickCount)
Vizion.count_obj (Missed, MissCount)
Vizion.distance_lr (Box, 0, 0, 0, 80, LineDist)
Vizion.distance_sr (Box, 0, 0, 0, 80, SegDist)
Vizion.distance_sr (Box, 0, 0, 0, 2, EndDist)
Vizion.distance_lc (Square, 0, 0, 0, 80, ContDist)
Vizion.gen_image_proto (Image, Dark, 20)
Vizion.gen_rectangle1 (Bright, 40, 40, 55, 60)
Vizion.paint_region (Bright, Dark, TwoTone, 200, 'fill')
Vizion.auto_threshold (TwoTone, Modes, 2)
Vizion.count_obj (Modes, ModeCount)
Vizion.get_region_index (Modes, 5, 5, DarkIndex)
Vizion.get_region_index (Modes, 48, 50, BrightIndex)
Vizion.gen_rectangle1 (Block, 10, 40, 14, 44)
Vizion.gen_region_points (Dot, 30, 30)
Vizion.union2 (Block, Dot, Scene)
Vizion.gen_region_points (Hit, 1, 1)
Vizion.gen_region_points (Miss, [0, 0, 0, 1, 1, 2, 2, 2], [0, 1, 2, 0, 2, 0, 1, 2])
Vizion.hit_or_miss (Scene, Hit, Miss, Found)
Vizion.test_region_point (Found, 30, 30, DotIn)
Vizion.test_region_point (Found, 12, 42, BlockIn)
Vizion.dev_display (TwoTone)
"""


HDEVELOP_BATCH19_PROGRAM = """* 伽瑪、過零和扇形
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_image_proto (Image, Flat, 100)
Vizion.gen_image_proto (Image, Mid, 128)
Vizion.gamma_image (Mid, Darker, 2, 0, 0, 255, 'false')
Vizion.get_grayval (Darker, 10, 10, GammaGray)
Vizion.binomial_filter (Flat, BinFlat, 3, 3)
Vizion.get_grayval (BinFlat, 10, 10, BinFlatGray)
Vizion.gen_image_proto (Image, Zero, 0)
Vizion.set_grayval (Zero, Dot, 30, 40, 255)
Vizion.binomial_filter (Dot, BinDot, 3, 3)
Vizion.get_grayval (BinDot, 30, 40, BinCenter)
Vizion.get_grayval (BinDot, 5, 5, BinFar)
Vizion.isotropic_diffusion (Flat, IsoFlat, 1.2, 1)
Vizion.get_grayval (IsoFlat, 10, 10, IsoGray)
Vizion.isotropic_diffusion (Dot, IsoDot, 1, 1)
Vizion.get_grayval (IsoDot, 30, 40, IsoCenter)
Vizion.get_grayval (IsoDot, 5, 5, IsoFar)
Vizion.gen_image_proto (Image, Base, 10)
Vizion.gen_image_proto (Image, Bright, 200)
Vizion.gen_rectangle1 (Box, 10, 10, 19, 19)
Vizion.reduce_domain (Bright, Box, Source)
Vizion.paint_gray (Source, Base, Mixed)
Vizion.get_grayval (Mixed, 15, 15, PaintIn)
Vizion.get_grayval (Mixed, 5, 5, PaintOut)
Vizion.gen_image_proto (Image, Step, 100)
Vizion.gen_rectangle1 (Right, 0, 21, 59, 79)
Vizion.paint_region (Right, Step, Signed, 156, 'fill')
Vizion.gen_rectangle1 (ZeroCol, 0, 20, 59, 20)
Vizion.paint_region (ZeroCol, Signed, Signed, 128, 'fill')
Vizion.zero_crossing (Signed, Cross)
Vizion.test_region_point (Cross, 10, 20, OnLine)
Vizion.test_region_point (Cross, 10, 19, OffLine)
Vizion.area_center (Cross, CrossArea, CrossRow, CrossCol)
Vizion.gen_image_proto (Image, Level, 128)
Vizion.gen_rectangle1 (Hot, 10, 10, 14, 14)
Vizion.gen_rectangle1 (Warm, 40, 40, 44, 44)
Vizion.gen_region_points (Speckle, 30, 30)
Vizion.gen_rectangle1 (Cold, 10, 50, 14, 54)
Vizion.paint_region (Hot, Level, Level, 200, 'fill')
Vizion.paint_region (Warm, Level, Level, 150, 'fill')
Vizion.paint_region (Speckle, Level, Level, 200, 'fill')
Vizion.paint_region (Cold, Level, Level, 40, 'fill')
Vizion.dual_threshold (Level, Lobes, 4, 40, 20)
Vizion.count_obj (Lobes, LobeCount)
Vizion.get_region_index (Lobes, 12, 12, HotIndex)
Vizion.get_region_index (Lobes, 42, 42, WarmIndex)
Vizion.get_region_index (Lobes, 30, 30, SpeckIndex)
Vizion.get_region_index (Lobes, 12, 52, ColdIndex)
Vizion.gen_circle_sector (Sector, 30, 40, 10, 0, rad(90))
Vizion.test_region_point (Sector, 30, 48, RightIn)
Vizion.test_region_point (Sector, 36, 44, DownIn)
Vizion.test_region_point (Sector, 22, 40, UpOut)
Vizion.test_region_point (Sector, 30, 32, LeftOut)
Vizion.gen_rectangle1 (Inner, 12, 12, 18, 18)
Vizion.gen_rectangle1 (Outer, 8, 8, 24, 24)
Vizion.test_subset_region (Inner, Outer, IsSubset)
Vizion.test_subset_region (Outer, Inner, NotSubset)
Vizion.test_equal_region (Inner, Inner, IsEqual)
Vizion.test_equal_region (Inner, Outer, NotEqual)
Vizion.dev_display (Mixed)
"""


HDEVELOP_BATCH20_PROGRAM = """* 指數、極座標和頂帽
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_image_proto (Image, Zero, 0)
Vizion.gen_image_proto (Image, Mid, 128)
Vizion.gen_image_proto (Image, High, 255)
Vizion.exp_image (Zero, ExpZero, 'scaled')
Vizion.exp_image (Mid, ExpMid, 'scaled')
Vizion.exp_image (High, ExpHigh, 'scaled')
Vizion.exp_image (Zero, ExpOne, 'natural')
Vizion.set_grayval (Zero, Five, 8, 8, 5)
Vizion.exp_image (Five, ExpFive, 'natural')
Vizion.get_grayval (ExpZero, 4, 4, ExpAtZero)
Vizion.get_grayval (ExpMid, 4, 4, ExpAtMid)
Vizion.get_grayval (ExpHigh, 4, 4, ExpAtHigh)
Vizion.get_grayval (ExpOne, 4, 4, ExpNatural)
Vizion.get_grayval (ExpFive, 8, 8, ExpAtFive)
Vizion.gen_region_points (Dot, 30, 50)
Vizion.polar_trans_region (Dot, PolarRegion, 30, 40, 0, rad(360), 0, 20, 40, 20)
Vizion.test_region_point (PolarRegion, 10, 0, PolarIn)
Vizion.test_region_point (PolarRegion, 0, 0, PolarOut)
Vizion.gen_contour_polygon_xld (Spoke, [30, 40, 20], [50, 40, 40])
Vizion.polar_trans_contour_xld (Spoke, SpokePolar, 30, 40, 0, rad(360), 0, 20, 40, 20)
Vizion.get_contour_xld (SpokePolar, PolarRows, PolarCols)
Vizion.gen_ellipse_sector (EllipseSector, 30, 40, 0, 16, 8, 0, rad(90))
Vizion.test_region_point (EllipseSector, 30, 52, WideIn)
Vizion.test_region_point (EllipseSector, 36, 40, ThinIn)
Vizion.test_region_point (EllipseSector, 22, 40, UpOut)
Vizion.test_region_point (EllipseSector, 30, 28, LeftOut)
Vizion.test_region_point (EllipseSector, 42, 40, FarDown)
Vizion.gen_rectangle1 (Block, 10, 10, 30, 30)
Vizion.gen_region_points (Speck, 5, 50)
Vizion.union2 (Block, Speck, Scene)
Vizion.gen_circle (Kernel, 40, 40, 1)
Vizion.top_hat (Scene, Kernel, Hat)
Vizion.test_region_point (Hat, 5, 50, SpeckKept)
Vizion.test_region_point (Hat, 20, 20, BlockGone)
Vizion.gen_rectangle1 (LeftBar, 10, 10, 20, 20)
Vizion.gen_rectangle1 (RightBar, 10, 22, 20, 32)
Vizion.union2 (LeftBar, RightBar, Pair)
Vizion.gen_rectangle1 (Box3, 0, 0, 2, 2)
Vizion.bottom_hat (Pair, Box3, Gap)
Vizion.test_region_point (Gap, 15, 21, GapFilled)
Vizion.test_region_point (Gap, 15, 15, BodyKept)
Vizion.gen_rectangle1 (PartA, 40, 10, 48, 18)
Vizion.gen_rectangle1 (PartB, 40, 30, 48, 38)
Vizion.concat_obj (PartA, PartB, Parts)
Vizion.region_to_label (Parts, Labels)
Vizion.get_grayval (Labels, 44, 14, LabelA)
Vizion.get_grayval (Labels, 44, 34, LabelB)
Vizion.get_grayval (Labels, 4, 4, LabelBack)
Vizion.label_to_region (Labels, Restored)
Vizion.count_obj (Restored, LabelCount)
Vizion.get_region_index (Restored, 44, 14, IndexA)
Vizion.get_region_index (Restored, 44, 34, IndexB)
Vizion.gen_contour_polygon_xld (Near, [10, 10], [10, 30])
Vizion.distance_sc (Near, 0, 0, 0, 80, LineDist)
Vizion.distance_sc (Near, 0, 0, 0, 2, EndDist)
Vizion.gen_contour_polygon_xld (Slash, [0, 10], [0, 10])
Vizion.distance_sc (Slash, 0, 10, 10, 0, CrossDist)
Vizion.dev_display (Labels)
"""


HDEVELOP_BATCH21_PROGRAM = """* 塗色、角點和交點
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_image_proto (Image, Base, 10)
Vizion.gen_rectangle1 (Window, 10, 10, 20, 20)
Vizion.reduce_domain (Base, Window, Limited)
Vizion.gen_rectangle1 (Big, 5, 5, 30, 30)
Vizion.overpaint_region (Big, Limited, Painted, 200, 'fill')
Vizion.get_grayval (Painted, 15, 15, InDomain)
Vizion.get_grayval (Painted, 6, 6, OutDomain)
Vizion.gen_image_proto (Image, Flat, 100)
Vizion.corner_response (Flat, FlatCorner, 3, 0.04, 0.001)
Vizion.get_grayval (FlatCorner, 20, 20, FlatResp)
Vizion.gen_image_proto (Image, Board, 0)
Vizion.gen_rectangle1 (Square, 10, 10, 30, 30)
Vizion.paint_region (Square, Board, Board, 255, 'fill')
Vizion.corner_response (Board, Corners, 3, 0.04, 0.001)
Vizion.get_grayval (Corners, 10, 10, CornerResp)
Vizion.get_grayval (Corners, 20, 10, EdgeResp)
Vizion.get_grayval (Corners, 20, 20, InsideResp)
Vizion.rank_image (Flat, Ranked, 1, 0)
Vizion.get_grayval (Ranked, 12, 12, RankFlat)
Vizion.gen_image_proto (Image, Dark, 0)
Vizion.set_grayval (Dark, Spike, 30, 40, 255)
Vizion.rank_image (Spike, RankMin, 1, 0)
Vizion.rank_image (Spike, RankMax, 1, 4)
Vizion.get_grayval (RankMin, 30, 40, RankHole)
Vizion.get_grayval (RankMax, 30, 41, RankNeighbor)
Vizion.get_grayval (RankMax, 5, 5, RankFar)
Vizion.gen_contour_polygon_xld (Bar, [10, 10], [10, 30])
Vizion.intersection_line_contour_xld (Bar, 0, 20, 40, 20, HitRow, HitCol)
Vizion.intersection_segment_contour_xld (Bar, 0, 20, 15, 20, SegRow, SegCol)
Vizion.intersection_segment_contour_xld (Bar, 0, 20, 5, 20, MissRow, MissCol)
MissCount := |MissRow|
Vizion.gen_contour_polygon_xld (SquareXld, [10, 10, 30, 30, 10], [10, 30, 30, 10, 10])
Vizion.shape_trans_xld (SquareXld, BoxXld, 'rectangle1')
Vizion.xld_features (BoxXld, 'area', BoxArea)
Vizion.xld_features (BoxXld, 'contlength', BoxLength)
Vizion.gen_image_proto (Image, ZeroA, 0)
Vizion.gen_image_proto (Image, ZeroB, 0)
Vizion.set_grayval (ZeroA, OneA, 8, 8, 30)
Vizion.set_grayval (ZeroB, OneB, 8, 8, 40)
Vizion.histo_2dim (OneA, OneB, Histo)
Vizion.get_grayval (Histo, 30, 40, PairCount)
Vizion.get_grayval (Histo, 0, 0, BackCount)
Vizion.get_grayval (Histo, 1, 1, EmptyCount)
Vizion.dev_display (Painted)
"""


HDEVELOP_BATCH22_PROGRAM = """* 絕對值、方向和交線
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.abs_image (Image, AbsZero)
Vizion.get_grayval (AbsZero, 4, 4, AbsAtZero)
Vizion.gen_image_proto (Image, Mid, 128)
Vizion.gen_image_proto (Image, Low, 100)
Vizion.abs_image (Mid, AbsMid)
Vizion.abs_image (Low, AbsLow)
Vizion.get_grayval (AbsMid, 4, 4, AbsAtMid)
Vizion.get_grayval (AbsLow, 4, 4, AbsAtLow)
Vizion.gen_image_proto (Image, Right, 255)
Vizion.gen_image_proto (Image, Down, 255)
Vizion.gen_image_proto (Image, Level, 128)
Vizion.atan2_image (Level, Right, DirRight)
Vizion.atan2_image (Down, Level, DirDown)
Vizion.get_grayval (DirRight, 4, 4, RightDir)
Vizion.get_grayval (DirDown, 4, 4, DownDir)
Vizion.gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 80, 60)
Vizion.get_grayval_interpolated (Ramp, 10, 10.5, MidGray)
Vizion.get_grayval_interpolated (Ramp, 10, 10.5, 'nearest', NearGray)
Vizion.gen_image_proto (Image, Dest, 10)
Vizion.gen_rectangle1 (Box, 10, 10, 19, 19)
Vizion.reduce_domain (Dest, Box, Limited)
Vizion.gen_image_proto (Image, Src, 200)
Vizion.overpaint_gray (Limited, Src, Mixed)
Vizion.get_grayval (Mixed, 15, 15, InPaint)
Vizion.get_grayval (Mixed, 5, 5, OutPaint)
Vizion.intersection_lines (0, 0, 0, 80, 0, 20, 40, 20, CrossRow, CrossCol, CrossHit)
Vizion.intersection_lines (0, 0, 0, 80, 5, 0, 5, 80, ParRow, ParCol, ParHit)
Vizion.gen_contour_polygon_xld (OnCircle, [30, 40], [50, 40])
Vizion.gen_contour_polygon_xld (AtCenter, [30, 30], [40, 40])
Vizion.dist_ellipse_contour_xld (OnCircle, 30, 40, 0, 10, 10, MaxOn)
Vizion.dist_ellipse_contour_xld (AtCenter, 30, 40, 0, 10, 10, MaxOff)
Vizion.gen_contour_polygon_xld (BoxXld, [10, 10, 30, 30, 10], [10, 30, 30, 10, 10])
Vizion.gen_contour_polygon_xld (Bow, [10, 30, 30, 10, 10], [10, 30, 10, 30, 10])
Vizion.concat_obj (BoxXld, Bow, Shapes)
Vizion.test_self_intersection_xld (Shapes, SelfFlags)
Vizion.critical_points_xld (BoxXld, 0.5, CritRow, CritCol)
CritCount := |CritRow|
Vizion.gen_contour_polygon_xld (Straight, [10, 10, 10], [10, 20, 30])
Vizion.critical_points_xld (Straight, 0.5, LineRow, LineCol)
LineCount := |LineRow|
Vizion.dev_display (Mixed)
"""


HDEVELOP_BATCH23_PROGRAM = """* 普瑞維特、極座標和拆線
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Half, 0, 40, 59, 79)
Vizion.paint_region (Half, Image, Stepped, 40, 'fill')
Vizion.prewitt_amp (Stepped, Prewitt)
Vizion.kirsch_amp (Stepped, Kirsch)
Vizion.roberts (Stepped, RobertsSqrt, 'sum_sqrt')
Vizion.roberts (Stepped, RobertsSum, 'sum_abs')
Vizion.get_grayval (Prewitt, 30, 39, PrewittEdge)
Vizion.get_grayval (Prewitt, 30, 10, PrewittFlat)
Vizion.get_grayval (Kirsch, 30, 39, KirschEdge)
Vizion.get_grayval (Kirsch, 30, 10, KirschFlat)
Vizion.get_grayval (RobertsSqrt, 30, 39, RobertsSqrtEdge)
Vizion.get_grayval (RobertsSum, 30, 39, RobertsSumEdge)
Vizion.intersection_segment_line (10, 0, 10, 40, 0, 20, 40, 20, HitRow, HitCol, HitFlag)
Vizion.intersection_segment_line (10, 0, 10, 10, 0, 20, 40, 20, MissRow, MissCol, MissFlag)
Vizion.gen_contour_polygon_xld (Corner, [35, 35], [50, 50])
Vizion.gen_contour_polygon_xld (Center, [30, 30], [40, 40])
Vizion.gen_contour_polygon_xld (Outside, [30, 30], [56, 56])
Vizion.dist_rectangle2_contour_xld (Corner, 30, 40, 0, 10, 5, DistOn)
Vizion.dist_rectangle2_contour_xld (Center, 30, 40, 0, 10, 5, DistCenter)
Vizion.dist_rectangle2_contour_xld (Outside, 30, 40, 0, 10, 5, DistOut)
Vizion.gen_contour_polygon_xld (BoxXld, [10, 10, 30, 30, 10], [10, 30, 30, 10, 10])
Vizion.split_contours_xld (BoxXld, Pieces, 0.5)
Vizion.count_obj (Pieces, PieceCount)
Vizion.length_xld (Pieces, PieceLen)
Vizion.gen_contour_polygon_xld (Straight, [10, 10, 10], [10, 20, 30])
Vizion.split_contours_xld (Straight, OneLine, 0.5)
Vizion.count_obj (OneLine, LinePieces)
Vizion.gen_region_points (Dot, 30, 50)
Vizion.polar_trans_region (Dot, PolarRegion, 30, 40, 0, rad(360), 0, 20, 40, 20)
Vizion.polar_trans_region_inv (PolarRegion, BackRegion, 30, 40, 0, rad(360), 0, 20, 80, 60)
Vizion.test_region_point (BackRegion, 30, 50, BackInside)
Vizion.test_region_point (BackRegion, 30, 40, BackCenter)
Vizion.set_grayval (Image, Marked, 30, 50, 255)
Vizion.polar_trans_image_ext (Marked, Polar, 30, 40, 0, rad(360), 0, 20, 40, 20, 'nearest_neighbor')
Vizion.polar_trans_image_inv (Polar, Back, 30, 40, 0, rad(360), 0, 20, 80, 60, 'nearest_neighbor')
Vizion.get_grayval (Back, 30, 50, BackGray)
Vizion.get_grayval (Back, 30, 40, CenterGray)
Vizion.dev_display (Prewitt)
"""


HDEVELOP_BATCH24_PROGRAM = """* 羅賓遜、方位和交圓
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Half, 0, 40, 59, 79)
Vizion.paint_region (Half, Image, Stepped, 40, 'fill')
Vizion.frei_amp (Stepped, Frei)
Vizion.robinson_amp (Stepped, Robinson)
Vizion.robinson_dir (Stepped, RobinsonDir)
Vizion.get_grayval (Frei, 30, 39, FreiEdge)
Vizion.get_grayval (Frei, 30, 10, FreiFlat)
Vizion.get_grayval (Robinson, 30, 39, RobinsonEdge)
Vizion.get_grayval (Robinson, 30, 10, RobinsonFlat)
Vizion.get_grayval (RobinsonDir, 30, 39, RobinsonRight)
Vizion.gen_rectangle1 (Left, 10, 5, 20, 15)
Vizion.gen_rectangle1 (Mark, 10, 30, 20, 45)
Vizion.gen_rectangle1 (Top, 0, 30, 4, 45)
Vizion.gen_rectangle1 (Bottom, 40, 30, 50, 45)
Vizion.concat_obj (Left, Top, Pair)
Vizion.concat_obj (Pair, Bottom, Candidates)
Vizion.select_region_spatial (Candidates, Mark, ChosenLeft, 'left')
Vizion.select_region_spatial (Candidates, Mark, ChosenAbove, 'above')
Vizion.select_region_spatial (Candidates, Mark, ChosenBelow, 'below')
Vizion.select_region_spatial (Candidates, Mark, ChosenRight, 'right')
Vizion.count_obj (ChosenLeft, LeftCount)
Vizion.count_obj (ChosenAbove, AboveCount)
Vizion.count_obj (ChosenBelow, BelowCount)
Vizion.count_obj (ChosenRight, RightCount)
Vizion.test_region_point (ChosenLeft, 15, 10, OnLeft)
Vizion.test_region_point (ChosenLeft, 15, 35, OnMark)
Vizion.gen_contour_polygon_xld (Spoke, [30, 40, 20], [50, 40, 40])
Vizion.polar_trans_contour_xld (Spoke, PolarSpoke, 30, 40, 0, rad(360), 0, 20, 40, 20)
Vizion.polar_trans_contour_xld_inv (PolarSpoke, BackSpoke, 30, 40, 0, rad(360), 0, 20, 40, 20)
Vizion.get_contour_xld (BackSpoke, BackRow, BackCol)
Vizion.gen_contour_polygon_xld (Across, [30, 30], [20, 60])
Vizion.gen_contour_polygon_xld (MissLine, [0, 5], [0, 1])
Vizion.intersection_circle_contour_xld (Across, 30, 40, 10, CircleRow, CircleCol)
Vizion.intersection_circle_contour_xld (MissLine, 30, 40, 10, MissRow, MissCol)
MissCount := |MissRow|
Vizion.gen_contour_polygon_xld (BoxXld, [10, 10, 30, 30, 10], [10, 30, 30, 10, 10])
Vizion.contour_point_num_xld (BoxXld, BoxPoints)
Vizion.contour_point_num_xld (Spoke, SpokePoints)
BoxCount := BoxPoints[1]
SpokeCount := SpokePoints[1]
Vizion.gen_image_proto (Image, Flat, 100)
Vizion.add_noise_salt_pepper (Flat, Quiet, 0, 1)
Vizion.get_grayval (Quiet, 4, 4, QuietGray)
Vizion.add_noise_salt_pepper (Flat, NoiseA, 100, 7)
Vizion.add_noise_salt_pepper (Flat, NoiseB, 100, 7)
Vizion.sub_image (NoiseA, NoiseB, Diff, 1, 0)
Vizion.gen_rectangle1 (Full, 0, 0, 59, 79)
Vizion.min_max_gray (Full, Diff, 0, DiffMin, DiffMax, DiffRange)
Vizion.threshold (NoiseA, Salt, 255, 255)
Vizion.threshold (NoiseA, Pepper, 0, 0)
Vizion.area_center (Salt, SaltArea, SaltRow, SaltCol)
Vizion.area_center (Pepper, PepperArea, PepperRow, PepperCol)
Vizion.dev_display (Robinson)
"""


HDEVELOP_BATCH25_PROGRAM = """* 對照表、位元和金字塔
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_image_proto (Image, Ten, 10)
Vizion.gen_image_proto (Image, One, 1)
Vizion.gen_image_proto (Image, Sixteen, 16)
Vizion.gen_image_proto (Image, High, 240)
Vizion.gen_image_proto (Image, Low, 15)
Vizion.bit_not (Ten, NotImage)
Vizion.bit_and (High, Low, AndImage)
Vizion.bit_or (High, Low, OrImage)
Vizion.bit_xor (High, Low, XorImage)
Vizion.bit_lshift (One, ShiftImage, 3)
Vizion.bit_rshift (Sixteen, UnshiftImage, 2)
Vizion.get_grayval (NotImage, 4, 4, NotGray)
Vizion.get_grayval (AndImage, 4, 4, AndGray)
Vizion.get_grayval (OrImage, 4, 4, OrGray)
Vizion.get_grayval (XorImage, 4, 4, XorGray)
Vizion.get_grayval (ShiftImage, 4, 4, ShiftGray)
Vizion.get_grayval (UnshiftImage, 4, 4, UnshiftGray)
Vizion.tuple_gen_sequence (0, 255, 1, Seq)
Vizion.tuple_sub (255, Seq, Inv)
Vizion.lut_trans (Ten, LutImage, Inv)
Vizion.get_grayval (LutImage, 4, 4, LutGray)
Vizion.gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 80, 60)
Vizion.gen_gauss_pyramid (Ramp, PlainPyr, 'none', 0.5)
Vizion.gen_gauss_pyramid (Ramp, MinPyr, 'min', 0.5)
Vizion.gen_gauss_pyramid (Ramp, MaxPyr, 'max', 0.5)
Vizion.gen_gauss_pyramid (Ten, ConstPyr, 'constant', 0.5)
Vizion.get_grayval (PlainPyr, 4, 3, PlainGray)
Vizion.get_grayval (MinPyr, 4, 3, MinGray)
Vizion.get_grayval (MaxPyr, 4, 3, MaxGray)
Vizion.get_grayval (ConstPyr, 4, 3, ConstGray)
Vizion.get_image_size (ConstPyr, PyrWidth, PyrHeight)
Vizion.dev_display (ConstPyr)
"""


HDEVELOP_BATCH26_PROGRAM = """* 量測剖面和一維函數
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 80, 60)
Vizion.gen_measure_rectangle2 (30, 40, 0, 10, 0, 80, 60, 'bilinear', Rect)
Vizion.measure_projection (Ramp, Rect, Profile)
ProfileCount := |Profile|
ProfileFirst := Profile[1]
ProfileMid := Profile[21]
ProfileLast := Profile[41]
Vizion.measure_thresh (Ramp, Rect, 0, 40, 'all', ThreshRow, ThreshCol, ThreshDist)
Vizion.translate_measure (Rect, 30, 50)
Vizion.measure_projection (Ramp, Rect, Moved)
MovedFirst := Moved[1]
MovedLast := Moved[41]
Vizion.gen_measure_arc (30, 40, 10, 0, rad(180), 0, 80, 60, 'bilinear', Arc)
Vizion.measure_projection (Ramp, Arc, ArcProfile)
ArcCount := |ArcProfile|
ArcFirst := ArcProfile[1]
ArcMid := ArcProfile[32]
ArcLast := ArcProfile[63]
Vizion.create_funct_1d_pairs ([0, 2], [10, 30], Curve)
Vizion.get_y_value_funct_1d (Curve, 1, 'constant', MidY)
Vizion.get_y_value_funct_1d (Curve, -1, 'constant', LowY)
Vizion.get_y_value_funct_1d (Curve, 3, 'zero', HighY)
Vizion.get_y_value_funct_1d (Curve, -1, 'mirror', MirrorY)
Vizion.sample_funct_1d (Curve, 0, 2, 1, 'constant', Sampled)
S1 := Sampled[1]
S2 := Sampled[2]
S3 := Sampled[3]
Vizion.create_funct_1d_pairs ([0, 2], [-1, 1], Line)
Vizion.zero_crossings_funct_1d (Line, Zeros)
ZeroX := Zeros[1]
Vizion.dev_display (Ramp)
"""


HDEVELOP_BATCH27_PROGRAM = """* 函數積分、導數和反函數
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.create_funct_1d_pairs ([0, 2], [10, 30], Curve)
Vizion.integrate_funct_1d (Curve, 0, 2, Area)
Vizion.derivate_funct_1d (Curve, 'second', Curve2)
Vizion.get_y_value_funct_1d (Curve2, 1, 'constant', Flat2)
Vizion.invert_funct_1d (Curve, Inv)
Vizion.get_y_value_funct_1d (Inv, 20, 'constant', InvY)
Vizion.create_funct_1d_pairs ([0, 2], [0, 2], Ident)
Vizion.compose_funct_1d (Curve, Ident, 'constant', Mixed)
Vizion.get_y_value_funct_1d (Mixed, 1, 'constant', MixedY)
Vizion.create_funct_1d_array ([4, 8, 6], ArrayFn)
Vizion.funct_1d_to_pairs (ArrayFn, ArrayX, ArrayY)
ArrayX0 := ArrayX[1]
ArrayY1 := ArrayY[2]
Vizion.get_y_value_funct_1d (ArrayFn, 1, 'constant', ArrayMid)
Vizion.create_funct_1d_pairs ([0, 1, 2], [0, 2, 2], Kink)
Vizion.derivate_funct_1d (Kink, 'first', Slope)
Vizion.get_y_value_funct_1d (Slope, 0, 'constant', Slope0)
Vizion.get_y_value_funct_1d (Slope, 1, 'constant', Slope1)
Vizion.get_y_value_funct_1d (Slope, 2, 'constant', Slope2)
Vizion.derivate_funct_1d (Kink, 'second', Bend2)
Vizion.get_y_value_funct_1d (Bend2, 1, 'constant', SecondMid)
Vizion.create_funct_1d_pairs ([0, 1, 2, 3], [0, 2, -1, 0], Bend)
Vizion.local_min_max_funct_1d (Bend, MinX, MaxX)
MinAt := MinX[1]
MaxAt := MaxX[1]
Vizion.create_funct_1d_pairs ([0, 2], [-1, 1], Line)
Vizion.abs_funct_1d (Line, AbsLine)
Vizion.get_y_value_funct_1d (AbsLine, 0, 'constant', Abs0)
Vizion.get_y_value_funct_1d (AbsLine, 1, 'constant', AbsMid)
Vizion.get_y_value_funct_1d (AbsLine, 2, 'constant', Abs2)
Vizion.dev_display (Image)
"""


HDEVELOP_BATCH28_PROGRAM = """* 傅立葉、平移和相位相關
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_image_proto (Image, Ten, 10)
Vizion.fft_image (Ten, Spec)
Vizion.get_complex (Spec, 0, 0, DcReal, DcImag)
Vizion.get_complex (Spec, 0, 1, AcReal, AcImag)
Vizion.fftshift (Spec, 'center', Centered)
Vizion.get_complex (Centered, 30, 40, ShiftReal, ShiftImag)
Vizion.fft_image_inv (Back, Spec)
Vizion.get_grayval (Back, 4, 4, BackGray)
Vizion.scale_fft (Spec, 0.5, HalfSpec)
Vizion.fft_image_inv (Half, HalfSpec)
Vizion.get_grayval (Half, 4, 4, HalfGray)
Vizion.gen_complex_const (1, 0, 80, 60, Ones)
Vizion.mult_fft (Spec, Ones, Product)
Vizion.get_complex (Product, 0, 0, ProdReal, ProdImag)
Vizion.gen_image_const (Small, 'byte', 5, 5)
Vizion.gen_image_proto (Small, Five, 10)
Vizion.fft_image (Five, SmallSpec)
Vizion.fftshift (SmallSpec, 'center', Mid)
Vizion.fftshift (SmallSpec, 'corner', Corner)
Vizion.get_complex (Mid, 2, 2, MidReal, MidImag)
Vizion.get_complex (Mid, 0, 0, MidZero, MidZeroImag)
Vizion.get_complex (Corner, 3, 3, CornerReal, CornerImag)
Vizion.get_complex (Corner, 2, 2, CornerOff, CornerOffImag)
Vizion.gen_image_const (Blank, 'byte', 32, 32)
Vizion.set_grayval (Blank, Dots, 10, 12, 255)
Vizion.set_grayval (Blank, Moved, 15, 14, 255)
Vizion.phase_correlation (Dots, Moved, PeakRow, PeakCol, PeakScore)
Vizion.dev_display (Dots)
"""


HDEVELOP_BATCH29_PROGRAM = """* 頻譜遮罩、卷積和能量
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_image_proto (Image, Ten, 10)
Vizion.fft_image (Ten, Spec)
Vizion.conj_fft (Spec, Conj)
Vizion.get_complex (Conj, 0, 0, ConjReal, ConjImag)
Vizion.add_fft (Spec, Spec, SumSpec)
Vizion.fft_image_inv (SumImage, SumSpec)
Vizion.get_grayval (SumImage, 4, 4, SumGray)
Vizion.sub_fft (Spec, Spec, ZeroSpec)
Vizion.fft_image_inv (ZeroImage, ZeroSpec)
Vizion.get_grayval (ZeroImage, 4, 4, ZeroGray)
Vizion.mask_fft (Spec, 'dc', DcSpec)
Vizion.fft_image_inv (DcImage, DcSpec)
Vizion.get_grayval (DcImage, 4, 4, DcGray)
Vizion.mask_fft (Spec, 'no_dc', HighSpec)
Vizion.fft_image_inv (HighImage, HighSpec)
Vizion.get_grayval (HighImage, 4, 4, HighGray)
Vizion.mask_fft (Spec, 'real', RealSpec)
Vizion.fft_image_inv (RealImage, RealSpec)
Vizion.get_grayval (RealImage, 4, 4, RealGray)
Vizion.mask_fft (Spec, 'imag', ImagSpec)
Vizion.fft_image_inv (ImagImage, ImagSpec)
Vizion.get_grayval (ImagImage, 4, 4, ImagGray)
Vizion.fft_energy (Spec, Energy)
Vizion.gen_complex_const (1, 0, 80, 60, Ones)
Vizion.div_fft (Spec, Ones, Quot)
Vizion.fft_image_inv (QuotImage, Quot)
Vizion.get_grayval (QuotImage, 4, 4, QuotGray)
Vizion.complex_to_image (RealByte, Spec, 'real')
Vizion.get_grayval (RealByte, 0, 0, RealPeak)
Vizion.get_grayval (RealByte, 0, 1, RealFlat)
Vizion.complex_to_image (LogByte, Spec, 'log')
Vizion.get_grayval (LogByte, 0, 0, LogPeak)
Vizion.get_grayval (LogByte, 0, 1, LogFlat)
Vizion.gen_image_const (Blank, 'byte', 32, 32)
Vizion.set_grayval (Blank, Dot, 10, 12, 255)
Vizion.gen_image_const (Kernel, 'byte', 4, 4)
Vizion.set_grayval (Kernel, ShiftK, 2, 3, 1)
Vizion.convol_fft (Dot, ShiftK, Shifted)
Vizion.get_grayval (Shifted, 12, 15, MovedGray)
Vizion.get_grayval (Shifted, 10, 12, OldGray)
Vizion.dev_display (Shifted)
"""


HDEVELOP_BATCH30_PROGRAM = """* 相關、移頻和補邊
Vizion.gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 80, 60)
Vizion.fft_image (Ramp, RampSpec)
Vizion.freq_filter (RampSpec, 'low', 0, LowSpec)
Vizion.fft_image_inv (LowImage, LowSpec)
Vizion.get_grayval (LowImage, 4, 4, LowGray)
Vizion.get_grayval (LowImage, 4, 79, LowEdge)
Vizion.freq_filter (RampSpec, 'high', 0, HighSpec)
Vizion.fft_image_inv (HighImage, HighSpec)
Vizion.get_grayval (HighImage, 4, 0, HighDark)
Vizion.get_grayval (HighImage, 4, 79, HighBright)
Vizion.freq_energy (RampSpec, 0, DcEnergy)
Vizion.gen_image_const (Blank, 'byte', 32, 32)
Vizion.set_grayval (Blank, Dot, 10, 12, 255)
Vizion.shift_image (Dot, Rolled, 5, 2)
Vizion.get_grayval (Rolled, 15, 14, RollGray)
Vizion.get_grayval (Rolled, 10, 12, RollOld)
Vizion.fft_image (Dot, DotSpec)
Vizion.translate_fft (DotSpec, 5, 2, MovedSpec)
Vizion.fft_image_inv (Moved, MovedSpec)
Vizion.get_grayval (Moved, 15, 14, PhaseGray)
Vizion.get_grayval (Moved, 10, 12, PhaseOld)
Vizion.normalize_fft (DotSpec, Unit)
Vizion.get_complex (Unit, 0, 0, UnitReal, UnitImag)
Vizion.get_complex (Unit, 0, 1, SideReal, SideImag)
Vizion.tuple_pow (SideReal, 2, SqR)
Vizion.tuple_pow (SideImag, 2, SqI)
Vizion.tuple_add (SqR, SqI, SumSq)
Vizion.tuple_sqrt (SumSq, UnitMag)
Vizion.gen_image_const (Kernel, 'byte', 4, 4)
Vizion.set_grayval (Kernel, Mark, 2, 3, 255)
Vizion.locate_template (Dot, Mark, MatchRow, MatchCol, MatchScore)
Vizion.correlation_image (Dot, Mark, ScoreImage)
Vizion.get_grayval (ScoreImage, 8, 9, ScorePeak)
Vizion.get_grayval (ScoreImage, 12, 15, ScoreOff)
Vizion.gen_image_const (Small, 'byte', 4, 4)
Vizion.set_grayval (Small, Edge, 1, 0, 200)
Vizion.pad_image (Edge, ConstPad, 1, 0, 0, 0, 0, 'constant')
Vizion.get_grayval (ConstPad, 0, 0, ConstCorner)
Vizion.get_grayval (ConstPad, 2, 0, ConstMoved)
Vizion.get_image_size (ConstPad, ConstWidth, ConstHeight)
Vizion.pad_image (Edge, MirrorPad, 1, 0, 0, 0, 0, 'mirror')
Vizion.get_grayval (MirrorPad, 0, 0, MirrorCorner)
Vizion.get_grayval (MirrorPad, 2, 0, MirrorMoved)
Vizion.pad_image (Edge, CycPad, 0, 0, 0, 1, 0, 'cyclic')
Vizion.get_grayval (CycPad, 1, 0, CycOld)
Vizion.get_grayval (CycPad, 1, 4, CycWrap)
Vizion.get_image_size (CycPad, CycWidth, CycHeight)
Vizion.dev_display (ScoreImage)
"""


HDEVELOP_BATCH31_PROGRAM = """* 點雲、平面和包圍盒
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_object_model_3d_from_points ([0, 2, 0, 2], [0, 0, 2, 2], [3, 3, 3, 3], Cloud)
Vizion.fit_plane_object_model_3d (Cloud, Nx, Ny, Nz, PlaneDist)
Vizion.distance_plane_object_model_3d (Cloud, Nx, Ny, Nz, PlaneDist, OnPlane)
OnDist := OnPlane[1]
PlaneCount := |OnPlane|
Vizion.gen_object_model_3d_from_points (0, 0, 1, Below)
Vizion.distance_plane_object_model_3d (Below, 0, 0, 1, 3, OffList)
OffDist := OffList[1]
Vizion.gen_object_model_3d_from_points ([0, 3], [0, 0], [0, 0], Pair)
Vizion.distance_point_object_model_3d (Pair, 0, 4, 0, NearDist)
Vizion.translate_object_model_3d (Pair, 10, -1, 2, Moved)
Vizion.get_object_model_3d_params (Moved, 'point_coord_x', MovedX)
Vizion.get_object_model_3d_params (Moved, 'point_coord_y', MovedY)
Vizion.get_object_model_3d_params (Moved, 'point_coord_z', MovedZ)
MovedX0 := MovedX[1]
MovedX1 := MovedX[2]
MovedY0 := MovedY[1]
MovedZ0 := MovedZ[1]
Vizion.mirror_object_model_3d (Pair, 'x', Flipped)
Vizion.get_object_model_3d_params (Flipped, 'point_coord_x', FlipX)
FlipX0 := FlipX[1]
FlipX1 := FlipX[2]
Vizion.gen_object_model_3d_from_points ([0, 1, 2, 3], 0, 0, Line)
Vizion.sample_object_model_3d (Line, 2, Kept)
Vizion.get_object_model_3d_params (Kept, 'point_coord_x', KeptX)
KeptX0 := KeptX[1]
KeptX1 := KeptX[2]
Vizion.get_object_model_3d_params (Kept, 'num_points', KeptCount)
Vizion.gen_box_object_model_3d (0, 1, 2, 4, 5, 6, Box)
Vizion.get_object_model_3d_params (Box, 'num_points', BoxCount)
Vizion.smallest_bounding_box_object_model_3d (Box, MinX, MinY, MinZ, MaxX, MaxY, MaxZ)
Vizion.dev_display (Image)
"""


HDEVELOP_BATCH32_PROGRAM = """* 點雲縮放、投影和主軸
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_object_model_3d_from_points (2, 0, 0, One)
Vizion.scale_object_model_3d (One, 3, 1, 1, Scaled)
Vizion.get_object_model_3d_params (Scaled, 'point_coord_x', ScaleX)
Vizion.scale_object_model_3d (One, 3, 1, 1, 2, 0, 0, Held)
Vizion.get_object_model_3d_params (Held, 'point_coord_x', HeldX)
Vizion.gen_object_model_3d_from_points (1, 0, 0, Arm)
Vizion.rotate_object_model_3d (Arm, 'z', rad(90), Turned)
Vizion.get_object_model_3d_params (Turned, 'point_coord_x', TurnX)
Vizion.get_object_model_3d_params (Turned, 'point_coord_y', TurnY)
Vizion.gen_object_model_3d_from_points (1, 2, 3, Tip)
Vizion.project_object_model_3d (Tip, 'xy', Flat)
Vizion.get_object_model_3d_params (Flat, 'point_coord_x', FlatX)
Vizion.get_object_model_3d_params (Flat, 'point_coord_y', FlatY)
Vizion.get_object_model_3d_params (Flat, 'point_coord_z', FlatZ)
Vizion.gen_object_model_3d_from_points (0, 0, [3, 1], Layer)
Vizion.select_plane_object_model_3d (Layer, 0, 0, 1, 3, 0.5, Kept)
Vizion.get_object_model_3d_params (Kept, 'num_points', KeptCount)
Vizion.get_object_model_3d_params (Kept, 'point_coord_z', KeptZ)
Vizion.gen_object_model_3d_from_points ([0, 3], 0, 0, Pair)
Vizion.max_distance_point_object_model_3d (Pair, 0, 4, 0, FarDist)
Vizion.gen_object_model_3d_from_points ([0, 1, 2], 0, 0, Line)
Vizion.gen_object_model_3d_from_points (1, 0, 0, Drop)
Vizion.difference_object_model_3d (Line, Drop, 0, Rest)
Vizion.get_object_model_3d_params (Rest, 'point_coord_x', RestX)
RestX0 := RestX[1]
RestX1 := RestX[2]
Vizion.sort_object_model_3d (Line, 'x', 'descending', Ordered)
Vizion.get_object_model_3d_params (Ordered, 'point_coord_x', OrderX)
OrderX0 := OrderX[1]
OrderX1 := OrderX[2]
Vizion.gen_object_model_3d_from_points ([0, 2, 4], 0, 0, Shaft)
Vizion.principal_axis_object_model_3d (Shaft, Ax, Ay, Az)
Vizion.dev_display (Image)
"""


HDEVELOP_BATCH33_PROGRAM = """* 點雲交集、格子和主軸長度
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_object_model_3d_from_points ([0, 1, 2], 0, 0, Line)
Vizion.gen_object_model_3d_from_points (1, 0, 0, Hit)
Vizion.intersect_object_model_3d (Line, Hit, 0, Common)
Vizion.get_object_model_3d_points (Common, CommonX, CommonY, CommonZ)
Vizion.gen_object_model_3d_from_points ([0, 2], 0, 0, Pair)
Vizion.center_object_model_3d (Pair, Mid)
Vizion.get_object_model_3d_points (Mid, MidX, MidY, MidZ)
MidX0 := MidX[1]
MidX1 := MidX[2]
Vizion.gen_object_model_3d_from_points ([0, 2, 5], 0, 0, Wide)
Vizion.clip_object_model_3d (Wide, 1, 0, 0, 3, 0, 0, Cut)
Vizion.get_object_model_3d_points (Cut, CutX, CutY, CutZ)
Vizion.gen_object_model_3d_from_points ([0, 0.5, 3], 0, 0, Dots)
Vizion.voxel_object_model_3d (Dots, 2, 'point', Sparse)
Vizion.get_object_model_3d_points (Sparse, SparseX, SparseY, SparseZ)
SparseX0 := SparseX[1]
SparseX1 := SparseX[2]
Vizion.voxel_object_model_3d (Dots, 2, 'center', Cells)
Vizion.get_object_model_3d_points (Cells, CellX, CellY, CellZ)
CellX0 := CellX[1]
CellX1 := CellX[2]
Vizion.gen_object_model_3d_from_points ([0, 10], 0, 0, Far)
Vizion.gen_object_model_3d_from_points ([0, 1], 0, 0, Near)
Vizion.max_distance_object_model_3d (Far, Near, Gap)
Vizion.gen_plane_object_model_3d (0, 0, 3, 2, 2, 1, Sheet)
Vizion.get_object_model_3d_params (Sheet, 'num_points', SheetCount)
Vizion.get_object_model_3d_points (Sheet, SheetX, SheetY, SheetZ)
SheetX0 := SheetX[1]
SheetX1 := SheetX[2]
SheetY2 := SheetY[3]
SheetZ0 := SheetZ[1]
Vizion.gen_object_model_3d_from_points ([0, 2, 4], 0, 0, Shaft)
Vizion.principal_spread_object_model_3d (Shaft, Spread1, Spread2, Spread3)
Vizion.dev_display (Image)
"""


HDEVELOP_BATCH34_PROGRAM = """* 點雲和影像對齊
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Rect, 10, 20, 12, 22)
Vizion.gen_object_model_3d_from_region (Rect, RegionCloud)
Vizion.get_object_model_3d_params (RegionCloud, 'num_points', RegionCount)
Vizion.get_object_model_3d_points (RegionCloud, Rx, Ry, Rz)
Rx0 := Rx[1]
Ry0 := Ry[1]
Rz0 := Rz[1]
Vizion.gen_region_object_model_3d (Back, RegionCloud, 80, 60)
Vizion.area_center (Back, Area, AreaRow, AreaCol)
BackArea := Area[1]
Vizion.gen_contour_polygon_xld (Contour, [10, 30], [20, 40])
Vizion.gen_object_model_3d_from_xld (Contour, XldCloud)
Vizion.get_object_model_3d_points (XldCloud, Xx, Xy, Xz)
Xx0 := Xx[1]
Xy0 := Xy[1]
Xx1 := Xx[2]
Xy1 := Xy[2]
Vizion.gen_image_const (Tiny, 'byte', 2, 2)
Vizion.set_grayval (Tiny, Marked, 0, 1, 40)
Vizion.set_grayval (Marked, Marked2, 1, 0, 10)
Vizion.gen_object_model_3d_from_image (Marked2, ImageCloud)
Vizion.get_object_model_3d_params (ImageCloud, 'num_points', ImageCount)
Vizion.get_object_model_3d_points (ImageCloud, PixX, PixY, PixZ)
PixZ1 := PixZ[2]
PixY2 := PixY[3]
PixZ2 := PixZ[3]
Vizion.gen_object_model_3d_from_points (5, 6, 200, Spot)
Vizion.paint_object_model_3d (Image, Painted, Spot, 0, 'z')
Vizion.get_grayval (Painted, 6, 5, PaintZ)
Vizion.paint_object_model_3d (Image, PaintedGray, Spot, 80, 'gray')
Vizion.get_grayval (PaintedGray, 6, 5, PaintGray)
Vizion.get_grayval_object_model_3d (Painted, Spot, ReadZ)
Vizion.select_object_model_3d_region (Rect, RegionCloud, 'inside', Inside)
Vizion.get_object_model_3d_params (Inside, 'num_points', InsideCount)
Vizion.gen_object_model_3d_from_points (0, 0, 0, Origin)
Vizion.select_object_model_3d_region (Rect, Origin, 'outside', Outside)
Vizion.get_object_model_3d_params (Outside, 'num_points', OutsideCount)
Vizion.select_object_model_3d_region (Rect, Origin, 'inside', Miss)
Vizion.get_object_model_3d_params (Miss, 'num_points', MissCount)
Vizion.gen_object_model_3d_from_points (1, 0, 0, Seed)
Vizion.set_object_model_3d_z (Marked2, Seed, Lifted)
Vizion.get_object_model_3d_points (Lifted, LiftX, LiftY, LiftZ)
Vizion.dev_display (Painted)
"""


HDEVELOP_BATCH35_PROGRAM = """* 高度、法線和明暗
Vizion.gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 80, 60)
Vizion.fit_image_plane (Ramp, Alpha, Beta, Gamma)
Vizion.sub_image_plane (Ramp, Flat, 128)
Vizion.get_grayval (Flat, 0, 0, FlatCorner)
Vizion.get_grayval (Flat, 30, 40, FlatMid)
Vizion.image_plane_deviation (Ramp, Deviation)
Vizion.image_plane_normal (Ramp, PlaneNx, PlaneNy, PlaneNz)
Vizion.height_slope_image (Ramp, Slope, 1)
Vizion.get_grayval (Slope, 30, 40, SlopeMid)
Vizion.height_slope_image (Ramp, Slope2, 2)
Vizion.get_grayval (Slope2, 30, 40, SlopeGentle)
Vizion.gen_image_proto (Ramp, Plain, 0)
Vizion.height_slope_image (Plain, FlatSlope, 1)
Vizion.get_grayval (FlatSlope, 0, 0, SlopeZero)
Vizion.height_normal_image (Ramp, Nx, Ny, Nz, 1)
Vizion.get_grayval (Nx, 30, 40, NxMid)
Vizion.get_grayval (Ny, 30, 40, NyMid)
Vizion.get_grayval (Nz, 30, 40, NzMid)
Vizion.height_normal_image (Plain, FNx, FNy, FNz, 1)
Vizion.get_grayval (FNx, 10, 10, UpNx)
Vizion.get_grayval (FNy, 10, 10, UpNy)
Vizion.get_grayval (FNz, 10, 10, UpNz)
Vizion.shade_height_field (Plain, Sky, 0, 90, 1, 'clamp')
Vizion.get_grayval (Sky, 10, 10, SkyGray)
Vizion.shade_height_field (Ramp, Shaded, 0, 90, 1, 'clamp')
Vizion.get_grayval (Shaded, 30, 40, ShadeMid)
Vizion.shade_height_field (Ramp, Dark, 0, 0, 1, 'clamp')
Vizion.get_grayval (Dark, 30, 40, ShadeSide)
Vizion.shade_height_field (Ramp, Both, 0, 90, 1, 'full')
Vizion.get_grayval (Both, 30, 40, ShadeFull)
Vizion.shade_height_field (Ramp, Back, 0, 0, 1, 'full')
Vizion.get_grayval (Back, 30, 40, ShadeBack)
Vizion.above_image_plane (Ramp, NoneAbove, 'above')
Vizion.count_obj (NoneAbove, AboveCount)
Vizion.gen_rectangle1 (Rect, 15, 20, 44, 59)
Vizion.paint_region (Rect, Plain, Block, 200, 'fill')
Vizion.shade_height_field (Block, Relief, 0, 20, 1, 'clamp')
Vizion.get_grayval (Relief, 30, 40, ReliefMid)
Vizion.gen_image_const (Bump, 'byte', 3, 3)
Vizion.set_grayval (Bump, Peak, 1, 1, 9)
Vizion.fit_image_plane (Peak, BumpA, BumpB, BumpG)
Vizion.image_plane_deviation (Peak, BumpDev)
Vizion.sub_image_plane (Peak, BumpFlat, 128)
Vizion.get_grayval (BumpFlat, 1, 1, BumpMid)
Vizion.get_grayval (BumpFlat, 0, 0, BumpCorner)
Vizion.above_image_plane (Peak, Above, 'above')
Vizion.area_center (Above, AboveArea, AboveRow, AboveCol)
AboveN := AboveArea[1]
AboveR := AboveRow[1]
AboveC := AboveCol[1]
Vizion.above_image_plane (Peak, Below, 'below')
Vizion.area_center (Below, BelowArea, BelowRow, BelowCol)
BelowN := BelowArea[1]
Vizion.dev_display (Relief)
"""


HDEVELOP_BATCH36_PROGRAM = """* 參考平面、傾角和體積
Vizion.gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 80, 60)
Vizion.sub_plane_image (Ramp, Level, 0, 1, 0, 128)
Vizion.get_grayval (Level, 0, 0, LevelCorner)
Vizion.get_grayval (Level, 30, 40, LevelMid)
Vizion.sub_plane_image (Ramp, Shifted, 0, 0, 10, 128)
Vizion.get_grayval (Shifted, 0, 0, Shift0)
Vizion.get_grayval (Shifted, 0, 40, Shift40)
Vizion.gen_image_proto (Ramp, Base, 10)
Vizion.add_plane_image (Base, Raised, 0, 1, 0)
Vizion.get_grayval (Raised, 4, 6, RaisedGray)
Vizion.add_plane_image (Base, Clipped, 0, 0, 300)
Vizion.get_grayval (Clipped, 0, 0, ClipGray)
Vizion.gen_image_plane (Ramp, Made, 0, 1, 5)
Vizion.get_grayval (Made, 0, 0, Made0)
Vizion.get_grayval (Made, 2, 3, Made3)
Vizion.plane_gray (0, 1, 5, 2, 3, OneGray)
Vizion.plane_gray (0, 1, 5, [0, 2], [3, 4], ManyGray)
Many0 := ManyGray[1]
Many1 := ManyGray[2]
Vizion.plane_angle (0, 1, TiltX, AzX)
Vizion.plane_angle (1, 0, TiltY, AzY)
Vizion.plane_angle (0, 0, Tilt0, Az0)
Vizion.plane_angle (0, -1, TiltBack, AzBack)
Vizion.plane_angle (-1, 0, TiltUp, AzUp)
Vizion.plane_angle (1, 1, TiltDiag, AzDiag)
Vizion.band_image_plane (Ramp, OnPlane, 0, 1, 0, -1, 1)
Vizion.area_center (OnPlane, BandArea, BandRow, BandCol)
BandN := BandArea[1]
Vizion.band_image_plane (Ramp, Swapped, 0, 1, 0, 1, -1)
Vizion.area_center (Swapped, SwapArea, SwapRow, SwapCol)
SwapN := SwapArea[1]
Vizion.band_image_plane (Ramp, OffPlane, 0, 1, 0, 1, 2)
Vizion.count_obj (OffPlane, OffCount)
Vizion.volume_image_plane (Ramp, 0, 1, 0, 'above', VolAbove)
Vizion.volume_image_plane (Ramp, 0, 1, 0, 'signed', VolSigned)
Vizion.extreme_image_plane (Ramp, 0, 0, 0, 'high', HighRow, HighCol, HighZ)
Vizion.extreme_image_plane (Ramp, 0, 0, 0, 'low', LowRow, LowCol, LowZ)
Vizion.gen_image_const (Bump, 'byte', 3, 3)
Vizion.set_grayval (Bump, Peak, 1, 1, 9)
Vizion.sub_plane_image (Peak, Kept, 0, 0, 0, 128)
Vizion.get_grayval (Kept, 1, 1, KeptMid)
Vizion.get_grayval (Kept, 0, 0, KeptCorner)
Vizion.volume_image_plane (Peak, 0, 0, 0, 'above', BumpAbove)
Vizion.volume_image_plane (Peak, 0, 0, 0, 'below', BumpBelow)
Vizion.volume_image_plane (Peak, 0, 0, 1, 'above', FitAbove)
Vizion.volume_image_plane (Peak, 0, 0, 1, 'signed', FitSigned)
Vizion.extreme_image_plane (Peak, 0, 0, 0, 'high', PeakRow, PeakCol, PeakZ)
Vizion.dev_display (Made)
"""


HDEVELOP_BATCH37_PROGRAM = """* 區域裡的平面
Vizion.gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 80, 60)
Vizion.gen_rectangle1 (Part, 10, 20, 30, 50)
Vizion.fit_plane_region (Part, Ramp, Alpha, Beta, Gamma)
Vizion.deviation_plane_region (Part, Ramp, 0, 1, 0, PartDev)
Vizion.mean_plane_region (Part, Ramp, 0, 1, 0, PartMean)
Vizion.volume_plane_region (Part, Ramp, 0, 1, 0, 'above', PartVol)
Vizion.extreme_plane_region (Part, Ramp, 0, 0, 0, 'high', HighRow, HighCol, HighZ)
Vizion.extreme_plane_region (Part, Ramp, 0, 0, 0, 'low', LowRow, LowCol, LowZ)
Vizion.band_plane_region (Part, Ramp, Band, 0, 1, 0, -1, 1)
Vizion.area_center (Band, BandArea, BandRow, BandCol)
BandN := BandArea[1]
Vizion.band_plane_region (Part, Ramp, Swapped, 0, 1, 0, 1, -1)
Vizion.area_center (Swapped, SwapArea, SwapRow, SwapCol)
SwapN := SwapArea[1]
Vizion.band_plane_region (Part, Ramp, Miss, 0, 1, 0, 1, 2)
Vizion.count_obj (Miss, OffCount)
Vizion.sub_plane_region (Part, Ramp, Leveled, 0, 1, 0, 128)
Vizion.get_grayval (Leveled, 20, 35, InGray)
Vizion.get_grayval (Leveled, 0, 40, OutGray)
Vizion.fit_plane_region_inlier (Part, Ramp, 1, KeepA, KeepB, KeepG)
Vizion.gen_image_const (Bump, 'byte', 3, 3)
Vizion.set_grayval (Bump, Peak, 1, 1, 9)
Vizion.gen_rectangle1 (All, 0, 0, 2, 2)
Vizion.gen_region_points (Center, 1, 1)
Vizion.difference (All, Center, Border)
Vizion.fit_plane_region (Border, Peak, BorderA, BorderB, BorderG)
Vizion.fit_plane_region (All, Peak, FlatA, FlatB, FlatG)
Vizion.fit_plane_region_inlier (All, Peak, 2, InlierA, InlierB, InlierG)
Vizion.deviation_plane_region (All, Peak, 0, 0, 0, DevAll)
Vizion.mean_plane_region (All, Peak, 0, 0, 0, MeanAll)
Vizion.volume_plane_region (All, Peak, 0, 0, 0, 'above', VolAll)
Vizion.dev_display (Leveled)
"""


HDEVELOP_BATCH38_PROGRAM = """* 每個物件的平面
Vizion.gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 80, 60)
Vizion.gen_rectangle1 (BoxA, 10, 10, 20, 20)
Vizion.gen_rectangle1 (BoxB, 10, 40, 20, 50)
Vizion.gen_rectangle1 (BoxC, 40, 40, 42, 42)
Vizion.concat_obj (BoxA, BoxB, Pair)
Vizion.concat_obj (Pair, BoxC, Parts)
Vizion.set_grayval (Ramp, Peak, [40, 40, 40, 41, 41, 41, 42, 42, 42], [40, 41, 42, 40, 41, 42, 40, 41, 42], [0, 0, 0, 0, 9, 0, 0, 0, 0])
Vizion.fit_plane_objects (Parts, Peak, Alpha, Beta, Gamma)
Beta1 := Beta[1]
Beta3 := Beta[3]
Gamma3 := Gamma[3]
Vizion.flatness_objects (Parts, Peak, Flatness)
Flat1 := Flatness[1]
Flat3 := Flatness[3]
Vizion.mean_plane_objects (Parts, Peak, 0, 0, 0, Mean)
Mean1 := Mean[1]
Mean2 := Mean[2]
Mean3 := Mean[3]
Vizion.volume_plane_objects (Parts, Peak, 0, 0, 0, 'above', Volume)
Vol1 := Volume[1]
Vol2 := Volume[2]
Vol3 := Volume[3]
Vizion.extreme_plane_objects (Parts, Peak, 0, 0, 0, 'high', HighRow, HighCol, HighZ)
HighR1 := HighRow[1]
HighC1 := HighCol[1]
HighZ1 := HighZ[1]
HighC2 := HighCol[2]
HighR3 := HighRow[3]
HighC3 := HighCol[3]
HighZ3 := HighZ[3]
Vizion.extreme_plane_objects (Parts, Peak, 0, 0, 0, 'low', LowRow, LowCol, LowZ)
LowC1 := LowCol[1]
LowZ1 := LowZ[1]
Vizion.select_flatness (Parts, Peak, Warp, 1, 10)
Vizion.count_obj (Warp, WarpCount)
Vizion.area_center (Warp, WarpArea, WarpRow, WarpCol)
WarpN := WarpArea[1]
Vizion.select_flatness (Parts, Peak, WarpSwap, 10, 1)
Vizion.count_obj (WarpSwap, SwapCount)
Vizion.select_flatness (Parts, Peak, Planar, 0, 0.001)
Vizion.count_obj (Planar, PlanarCount)
Vizion.select_plane_mean (Parts, Peak, Tall, 0, 0, 0, 40, 50)
Vizion.area_center (Tall, TallArea, TallRow, TallCol)
TallC := TallCol[1]
Vizion.level_objects (Parts, Peak, Leveled, 128)
Vizion.get_grayval (Leveled, 15, 15, FlatGray)
Vizion.get_grayval (Leveled, 41, 41, BumpGray)
Vizion.get_grayval (Leveled, 40, 40, EdgeGray)
Vizion.get_grayval (Leveled, 0, 5, OutGray)
Vizion.dev_display (Leveled)
"""


HDEVELOP_BATCH39_PROGRAM = """* 抛物線和二次曲面
Vizion.gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 80, 60)
Vizion.gen_rectangle1 (Bowl, 20, 10, 22, 14)
Vizion.set_grayval (Ramp, Peak, [20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22], [10, 11, 12, 13, 14, 10, 11, 12, 13, 14, 10, 11, 12, 13, 14], [0, 1, 4, 9, 16, 0, 1, 4, 9, 16, 0, 1, 4, 9, 16])
Vizion.fit_quad_region (Bowl, Peak, A, B, C, D, E, F)
Vizion.fit_quad_objects (Bowl, Peak, OA, OB, OC, OD, OE, OF)
OB1 := OB[1]
OE1 := OE[1]
OF1 := OF[1]
Vizion.flatness_quad_objects (Bowl, Peak, QFlat)
QF1 := QFlat[1]
Vizion.flatness_objects (Bowl, Peak, PFlat)
PF1 := PFlat[1]
Vizion.deviation_quad_region (Bowl, Peak, 0, 1, 0, 0, -20, 100, Dev0)
Vizion.deviation_quad_region (Bowl, Peak, 0, 1, 0, 0, -20, 101, Dev1)
Vizion.quad_gray (0, 1, 0, 0, -20, 100, 21, 12, OneG)
Vizion.quad_gray (0, 1, 0, 0, -20, 100, [20, 22], [10, 14], Many)
Many0 := Many[1]
Many1 := Many[2]
Vizion.sub_quad_image (Peak, Subtracted, 0, 1, 0, 0, -20, 100, 128)
Vizion.get_grayval (Subtracted, 21, 12, SubGray)
Vizion.level_quad_objects (Bowl, Peak, Leveled, 128)
Vizion.get_grayval (Leveled, 21, 12, LevelGray)
Vizion.get_grayval (Leveled, 0, 30, OutGray)
Vizion.select_quad_flatness (Bowl, Peak, Kept, 0, 0.001)
Vizion.count_obj (Kept, KeptCount)
Vizion.select_quad_flatness (Bowl, Peak, Swapped, 0.001, 0)
Vizion.count_obj (Swapped, SwapCount)
Vizion.select_quad_flatness (Bowl, Peak, Dropped, 1, 10)
Vizion.count_obj (Dropped, DropCount)
Vizion.dev_display (Leveled)
"""


HDEVELOP_BATCH40_PROGRAM = """* 兩張高度圖的差
Vizion.gen_image_gray_ramp (Ref, 0, 1, 0, 0, 0, 80, 60)
Vizion.gen_rectangle1 (BoxA, 10, 10, 12, 12)
Vizion.gen_rectangle1 (BoxB, 30, 40, 32, 42)
Vizion.gen_rectangle1 (BoxC, 50, 50, 52, 52)
Vizion.concat_obj (BoxA, BoxB, Pair)
Vizion.concat_obj (Pair, BoxC, Parts)
Vizion.add_plane_image (Ref, Test, 0, 0, 0)
Vizion.set_grayval (Test, Raised, [10, 10, 10, 11, 11, 11, 12, 12, 12], [10, 11, 12, 10, 11, 12, 10, 11, 12], [20, 21, 22, 20, 31, 22, 20, 21, 22])
Vizion.set_grayval (Raised, Changed, 51, 51, 41)
Vizion.mean_height_diff (Parts, Changed, Ref, UnionMean)
Vizion.deviation_height_diff (Parts, Changed, Ref, UnionDev)
Vizion.volume_height_diff (Parts, Changed, Ref, 'above', VolAbove)
Vizion.volume_height_diff (Parts, Changed, Ref, 'below', VolBelow)
Vizion.volume_height_diff (Parts, Changed, Ref, 'signed', VolSigned)
Vizion.extreme_height_diff (Parts, Changed, Ref, 'high', HighRow, HighCol, HighDiff)
Vizion.extreme_height_diff (Parts, Changed, Ref, 'low', LowRow, LowCol, LowDiff)
Vizion.mean_height_diff_objects (Parts, Changed, Ref, ObjMean)
Mean1 := ObjMean[1]
Mean2 := ObjMean[2]
Mean3 := ObjMean[3]
Vizion.extreme_height_diff_objects (Parts, Changed, Ref, 'high', ObjRow, ObjCol, ObjDiff)
HighR1 := ObjRow[1]
HighC1 := ObjCol[1]
HighD1 := ObjDiff[1]
HighR2 := ObjRow[2]
HighC2 := ObjCol[2]
Vizion.select_height_diff (Parts, Changed, Ref, Tall, 10, 12)
Vizion.count_obj (Tall, TallCount)
Vizion.select_height_diff (Parts, Changed, Ref, TallSwap, 12, 10)
Vizion.count_obj (TallSwap, SwapCount)
Vizion.select_height_diff (Parts, Changed, Ref, Flat, -0.5, 0.5)
Vizion.area_center (Flat, FlatArea, FlatRow, FlatCol)
FlatC := FlatCol[1]
Vizion.band_height_diff (Parts, Changed, Ref, Peak, 15, 25)
Vizion.area_center (Peak, PeakArea, PeakRow, PeakCol)
PeakN := PeakArea[1]
PeakR := PeakRow[1]
PeakC := PeakCol[1]
Vizion.sub_image (Changed, Ref, DiffView, 1, 128)
Vizion.get_grayval (DiffView, 11, 11, ViewHigh)
Vizion.get_grayval (DiffView, 51, 51, ViewLow)
Vizion.dev_display (DiffView)
"""


HDEVELOP_BATCH41_PROGRAM = """* 局部平均和粗糙度
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Flat, 10, 10, 14, 14)
Vizion.gen_rectangle1 (Spot, 29, 39, 31, 41)
Vizion.concat_obj (Flat, Spot, Parts)
Vizion.set_grayval (Image, Peak, 30, 40, 30)
Vizion.sub_local_mean (Peak, Level, 3, 3)
Vizion.get_grayval (Level, 30, 40, CenterGray)
Vizion.get_grayval (Level, 30, 39, SideGray)
Vizion.get_grayval (Level, 12, 12, FlatGray)
Vizion.roughness_ra (Spot, Peak, 3, 3, SpotRa)
Vizion.roughness_ra (Flat, Peak, 3, 3, FlatRa)
Vizion.roughness_rq (Spot, Peak, 3, 3, SpotRq)
Vizion.roughness_rz (Spot, Peak, 3, 3, Rp, Rv, Rz)
Vizion.roughness_ra_objects (Parts, Peak, 3, 3, ObjRa)
Ra1 := ObjRa[1]
Ra2 := ObjRa[2]
Vizion.select_roughness (Parts, Peak, Rough, 1, 10, 3, 3)
Vizion.count_obj (Rough, RoughCount)
Vizion.area_center (Rough, RoughArea, RoughRow, RoughCol)
RoughN := RoughArea[1]
Vizion.select_roughness (Parts, Peak, RoughSwap, 10, 1, 3, 3)
Vizion.count_obj (RoughSwap, SwapCount)
Vizion.select_roughness (Parts, Peak, Smooth, 0, 0.001, 3, 3)
Vizion.count_obj (Smooth, SmoothCount)
Vizion.extreme_roughness (Spot, Peak, 3, 3, 'high', HighRow, HighCol, HighRes)
Vizion.band_roughness (Spot, Peak, PeakPix, 20, 30, 3, 3)
Vizion.area_center (PeakPix, PeakArea, PeakRow, PeakCol)
PeakN := PeakArea[1]
PeakR := PeakRow[1]
PeakC := PeakCol[1]
Vizion.dev_display (Level)
"""


HDEVELOP_BATCH42_PROGRAM = """* 中值和局部凸起
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Flat, 10, 10, 14, 14)
Vizion.gen_rectangle1 (Spot, 29, 39, 31, 41)
Vizion.concat_obj (Flat, Spot, Parts)
Vizion.set_grayval (Image, Peak, 30, 40, 30)
Vizion.sub_local_median (Peak, Level, 3)
Vizion.get_grayval (Level, 30, 40, CenterGray)
Vizion.get_grayval (Level, 30, 39, SideGray)
Vizion.get_grayval (Level, 12, 12, FlatGray)
Vizion.roughness_median_ra (Spot, Peak, 3, SpotRa)
Vizion.roughness_median_ra (Flat, Peak, 3, FlatRa)
Vizion.roughness_median_rq (Spot, Peak, 3, SpotRq)
Vizion.roughness_median_rz (Spot, Peak, 3, Rp, Rv, Rz)
Vizion.roughness_median_ra_objects (Parts, Peak, 3, ObjRa)
Ra1 := ObjRa[1]
Ra2 := ObjRa[2]
Vizion.select_median_roughness (Parts, Peak, Rough, 1, 10, 3)
Vizion.count_obj (Rough, RoughCount)
Vizion.area_center (Rough, RoughArea, RoughRow, RoughCol)
RoughN := RoughArea[1]
Vizion.select_median_roughness (Parts, Peak, RoughSwap, 10, 1, 3)
Vizion.count_obj (RoughSwap, SwapCount)
Vizion.select_median_roughness (Parts, Peak, Smooth, 0, 0.001, 3)
Vizion.count_obj (Smooth, SmoothCount)
Vizion.extreme_median_roughness (Spot, Peak, 3, 'high', HighRow, HighCol, HighRes)
Vizion.extreme_median_roughness (Spot, Peak, 3, 'low', LowRow, LowCol, LowRes)
Vizion.band_median_roughness (Spot, Peak, PeakPix, 20, 40, 3)
Vizion.area_center (PeakPix, PeakArea, PeakRow, PeakCol)
PeakN := PeakArea[1]
PeakR := PeakRow[1]
PeakC := PeakCol[1]
Vizion.dev_display (Level)
"""


HDEVELOP_BATCH43_PROGRAM = """* 比結構元素小的亮點和暗點
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_image_proto (Image, Field, 100)
Vizion.gen_rectangle1 (BrightBox, 19, 24, 21, 26)
Vizion.gen_rectangle1 (DarkBox, 39, 49, 41, 51)
Vizion.concat_obj (BrightBox, DarkBox, Parts)
Vizion.set_grayval (Field, Marked, [20, 40], [25, 50], [130, 70])
Vizion.gray_spot_white (Marked, White, 3)
Vizion.get_grayval (White, 20, 25, WhitePeak)
Vizion.get_grayval (White, 20, 24, WhiteSide)
Vizion.get_grayval (White, 40, 50, WhitePit)
Vizion.gray_spot_black (Marked, Black, 3)
Vizion.get_grayval (Black, 40, 50, BlackPit)
Vizion.get_grayval (Black, 40, 49, BlackSide)
Vizion.get_grayval (Black, 20, 25, BlackPeak)
Vizion.spot_volume (BrightBox, Marked, 'white', 3, VolWhite)
Vizion.spot_volume (BrightBox, Marked, 'black', 3, VolBlackOnBright)
Vizion.spot_mean (BrightBox, Marked, 'white', 3, MeanWhite)
Vizion.spot_volume (DarkBox, Marked, 'black', 3, VolBlack)
Vizion.spot_volume (DarkBox, Marked, 'white', 3, VolWhiteOnDark)
Vizion.spot_volume_objects (Parts, Marked, 'white', 3, ObjWhite)
White1 := ObjWhite[1]
White2 := ObjWhite[2]
Vizion.spot_volume_objects (Parts, Marked, 'black', 3, ObjBlack)
Black1 := ObjBlack[1]
Black2 := ObjBlack[2]
Vizion.extreme_spot (BrightBox, Marked, 'white', 3, HighRow, HighCol, HighH)
Vizion.extreme_spot (DarkBox, Marked, 'black', 3, PitRow, PitCol, PitH)
Vizion.select_spot (Parts, Marked, Kept, 1, 100, 'white', 3)
Vizion.count_obj (Kept, KeepCount)
Vizion.select_spot (Parts, Marked, KeptSwap, 100, 1, 'white', 3)
Vizion.count_obj (KeptSwap, SwapCount)
Vizion.select_spot (Parts, Marked, DarkKept, 1, 100, 'black', 3)
Vizion.count_obj (DarkKept, DarkCount)
Vizion.band_spot (Parts, Marked, BrightPix, 20, 40, 'white', 3)
Vizion.area_center (BrightPix, BrightArea, BrightRow, BrightCol)
BrightN := BrightArea[1]
BrightR := BrightRow[1]
BrightC := BrightCol[1]
Vizion.band_spot (Parts, Marked, DarkPix, 20, 40, 'black', 3)
Vizion.area_center (DarkPix, DarkArea, DarkRow, DarkCol)
DarkN := DarkArea[1]
DarkR := DarkRow[1]
DarkC := DarkCol[1]
Vizion.dev_display (White)
"""


HDEVELOP_BATCH44_PROGRAM = """* 開運算和重建
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Square, 20, 20, 26, 26)
Vizion.gen_rectangle1 (Arm, 23, 27, 23, 36)
Vizion.gen_rectangle1 (Speck, 45, 50, 45, 50)
Vizion.concat_obj (Square, Arm, Body)
Vizion.concat_obj (Body, Speck, BrightParts)
Vizion.paint_region (BrightParts, Image, Bright, 200, 'fill')
Vizion.gray_opening_rect (Bright, Opened, 3)
Vizion.gray_opening_reconstruct (Bright, Restored, 3)
Vizion.get_grayval (Opened, 23, 23, OpenSq)
Vizion.get_grayval (Opened, 23, 30, OpenArm)
Vizion.get_grayval (Opened, 45, 50, OpenSpeck)
Vizion.get_grayval (Restored, 23, 23, RecSq)
Vizion.get_grayval (Restored, 23, 30, RecArm)
Vizion.get_grayval (Restored, 45, 50, RecSpeck)
Vizion.region_opening_gain (Bright, ArmBack, 3)
Vizion.area_center (ArmBack, GainArea, GainRow, GainCol)
GainN := GainArea[1]
GainR := GainRow[1]
GainC := GainCol[1]
Vizion.volume_opening_gain (Arm, Bright, 3, GainVol)
Vizion.gen_image_proto (Image, Field, 200)
Vizion.paint_region (BrightParts, Field, Dark, 0, 'fill')
Vizion.gray_closing_rect (Dark, Closed, 3)
Vizion.gray_closing_reconstruct (Dark, DarkRestored, 3)
Vizion.get_grayval (Closed, 23, 23, CloseSq)
Vizion.get_grayval (Closed, 23, 30, CloseArm)
Vizion.get_grayval (Closed, 45, 50, CloseHole)
Vizion.get_grayval (DarkRestored, 23, 30, RecDarkArm)
Vizion.get_grayval (DarkRestored, 45, 50, RecHole)
Vizion.region_closing_gain (Dark, PitBack, 3)
Vizion.area_center (PitBack, PitArea, PitRow, PitCol)
PitN := PitArea[1]
Vizion.volume_closing_gain (Arm, Dark, 3, PitVol)
Vizion.dev_display (Restored)
"""


HDEVELOP_BATCH45_PROGRAM = """* 區域有多厚
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Fat, 10, 10, 14, 14)
Vizion.gen_rectangle1 (Bar, 40, 10, 42, 30)
Vizion.gen_rectangle1 (Left, 10, 50, 18, 58)
Vizion.gen_rectangle1 (Bridge, 13, 59, 15, 69)
Vizion.gen_rectangle1 (Right, 10, 70, 18, 78)
Vizion.union2 (Left, Bridge, Joined)
Vizion.union2 (Joined, Right, Bell)
Vizion.concat_obj (Fat, Bar, Pair)
Vizion.concat_obj (Pair, Bell, Parts)
Vizion.thickness_max (Parts, 'chessboard', MaxRow, MaxCol, MaxDist)
MaxR1 := MaxRow[1]
MaxC1 := MaxCol[1]
MaxD1 := MaxDist[1]
MaxR2 := MaxRow[2]
MaxC2 := MaxCol[2]
MaxD2 := MaxDist[2]
MaxR3 := MaxRow[3]
MaxC3 := MaxCol[3]
MaxD3 := MaxDist[3]
Vizion.thickness_neck (Parts, 'chessboard', NeckRow, NeckCol, NeckDist)
NeckR1 := NeckRow[1]
NeckC1 := NeckCol[1]
NeckD1 := NeckDist[1]
NeckR2 := NeckRow[2]
NeckC2 := NeckCol[2]
NeckD2 := NeckDist[2]
NeckR3 := NeckRow[3]
NeckC3 := NeckCol[3]
NeckD3 := NeckDist[3]
Vizion.thickness_mean (Parts, 'chessboard', MeanDist)
Mean1 := MeanDist[1]
Mean2 := MeanDist[2]
Vizion.neck_ratio (Parts, 'chessboard', Ratio)
Ratio1 := Ratio[1]
Ratio2 := Ratio[2]
Ratio3 := Ratio[3]
Vizion.thickness_sum (Parts, 'chessboard', SumDist)
Sum1 := SumDist[1]
Sum2 := SumDist[2]
Vizion.select_neck (Parts, Thick, 3, 3, 'chessboard')
Vizion.count_obj (Thick, ThickCount)
Vizion.select_neck (Parts, ThickSwap, 6, 3, 'chessboard')
Vizion.count_obj (ThickSwap, SwapCount)
Vizion.select_neck (Parts, Thin, 2, 2, 'chessboard')
Vizion.count_obj (Thin, ThinCount)
Vizion.band_thickness (Fat, Core, 3, 3, 'chessboard')
Vizion.area_center (Core, CoreArea, CoreRow, CoreCol)
CoreN := CoreArea[1]
CoreR := CoreRow[1]
CoreC := CoreCol[1]
Vizion.thickness_image (Bell, Dist, 'chessboard')
Vizion.get_grayval (Dist, 14, 54, RawCore)
Vizion.get_grayval (Dist, 14, 64, RawNeck)
Vizion.get_grayval (Dist, 0, 0, RawOut)
Vizion.scale_image (Dist, View, 40, 0)
Vizion.get_grayval (View, 14, 54, ViewCore)
Vizion.get_grayval (View, 14, 64, ViewNeck)
Vizion.dev_display (View)
"""


HDEVELOP_BATCH46_PROGRAM = """* 兩個區域重疊多少
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Wide, 10, 10, 19, 19)
Vizion.gen_rectangle1 (Shift, 10, 15, 19, 24)
Vizion.gen_rectangle1 (Small, 40, 40, 44, 44)
Vizion.gen_rectangle1 (Big, 38, 38, 48, 48)
Vizion.concat_obj (Wide, Small, Parts)
Vizion.concat_obj (Shift, Big, Other)
Vizion.area_intersection (Parts, Other, Inter)
Inter1 := Inter[1]
Inter2 := Inter[2]
Vizion.area_union (Parts, Other, UnionArea)
Union1 := UnionArea[1]
Union2 := UnionArea[2]
Vizion.area_difference (Parts, Other, Only)
Only1 := Only[1]
Only2 := Only[2]
Vizion.region_iou (Parts, Other, IoU)
Iou1 := IoU[1]
Iou2 := IoU[2]
Vizion.region_cover (Parts, Other, Cover)
Cover1 := Cover[1]
Cover2 := Cover[2]
Vizion.region_inside (Parts, Other, Inside)
Inside1 := Inside[1]
Inside2 := Inside[2]
Vizion.select_cover (Parts, Other, Kept, 0.9, 1)
Vizion.count_obj (Kept, KeepCount)
Vizion.area_center (Kept, KeepArea, KeepRow, KeepCol)
KeepN := KeepArea[1]
KeepR := KeepRow[1]
KeepC := KeepCol[1]
Vizion.select_cover (Parts, Other, KeptSwap, 1, 0.9)
Vizion.count_obj (KeptSwap, SwapCount)
Vizion.select_cover (Parts, Other, Half, 0.4, 0.6)
Vizion.count_obj (Half, HalfCount)
Vizion.sym_difference (Parts, Other, Differ)
Vizion.area_center (Differ, SymArea, SymRow, SymCol)
Sym1 := SymArea[1]
Sym2 := SymArea[2]
Vizion.intersection (Wide, Shift, Overlap)
Vizion.paint_region (Overlap, Image, Shown, 255, 'fill')
Vizion.get_grayval (Shown, 12, 17, HitGray)
Vizion.get_grayval (Shown, 12, 12, MissGray)
Vizion.dev_display (Shown)
"""


HDEVELOP_BATCH47_PROGRAM = """* 兩個區域隔多遠
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (GapA, 10, 10, 19, 19)
Vizion.gen_rectangle1 (GapB, 10, 30, 19, 39)
Vizion.gen_rectangle1 (TouchA, 40, 10, 44, 14)
Vizion.gen_rectangle1 (TouchB, 40, 15, 44, 19)
Vizion.gen_rectangle1 (OverA, 50, 40, 54, 49)
Vizion.gen_rectangle1 (OverB, 50, 45, 54, 54)
Vizion.concat_obj (GapA, TouchA, Pair1)
Vizion.concat_obj (Pair1, OverA, Parts)
Vizion.concat_obj (GapB, TouchB, Pair2)
Vizion.concat_obj (Pair2, OverB, Other)
Vizion.distance_objects (Parts, Other, 'chessboard', MinDist)
Min1 := MinDist[1]
Min2 := MinDist[2]
Min3 := MinDist[3]
Vizion.distance_objects_far (Parts, Other, 'chessboard', FarDist)
Far1 := FarDist[1]
Far2 := FarDist[2]
Far3 := FarDist[3]
Vizion.nearest_object_point (Parts, Other, 'chessboard', NearRow1, NearCol1, NearRow2, NearCol2, NearDist)
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
Vizion.farthest_object_point (Parts, Other, 'chessboard', FarRow1, FarCol1, FarRow2, FarCol2, FarPoint)
FarR1 := FarRow1[1]
FarC1 := FarCol1[1]
FarP1 := FarPoint[1]
Vizion.centroid_offset (Parts, Other, 'chessboard', OffRow, OffCol)
OffR1 := OffRow[1]
OffC1 := OffCol[1]
OffC2 := OffCol[2]
Vizion.centroid_distance (Parts, Other, 'chessboard', CenterDist)
Center1 := CenterDist[1]
Center2 := CenterDist[2]
Vizion.select_object_distance (Parts, Other, GapKept, 11, 11, 'chessboard')
Vizion.count_obj (GapKept, GapCount)
Vizion.select_object_distance (Parts, Other, GapSwap, 20, 11, 'chessboard')
Vizion.count_obj (GapSwap, SwapCount)
Vizion.select_object_distance (Parts, Other, TouchKept, 1, 1, 'chessboard')
Vizion.count_obj (TouchKept, TouchCount)
Vizion.select_object_distance (Parts, Other, OverKept, 0, 0, 'chessboard')
Vizion.count_obj (OverKept, OverCount)
Vizion.facing_region (Parts, Other, Edge, 'chessboard')
Vizion.area_center (Edge, EdgeArea, EdgeRow, EdgeCol)
EdgeN := EdgeArea[1]
EdgeR := EdgeRow[1]
EdgeC := EdgeCol[1]
Vizion.select_obj (Edge, GapEdge, 1)
Vizion.paint_region (GapEdge, Image, Shown, 255, 'fill')
Vizion.get_grayval (Shown, 10, 19, EdgeGray)
Vizion.get_grayval (Shown, 10, 18, InsideGray)
Vizion.dev_display (Shown)
"""


HDEVELOP_BATCH48_PROGRAM = """* 物件在參考的哪一側
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Ref, 30, 30, 39, 39)
Vizion.gen_rectangle1 (Right, 30, 50, 39, 59)
Vizion.gen_rectangle1 (Left, 30, 22, 39, 26)
Vizion.gen_rectangle1 (Below, 48, 30, 52, 39)
Vizion.gen_rectangle1 (Above, 14, 30, 18, 39)
Vizion.concat_obj (Right, Left, Pair)
Vizion.concat_obj (Pair, Below, Three)
Vizion.concat_obj (Three, Above, Parts)
Vizion.region_angle (Parts, Ref, Angle)
Ang1 := Angle[1]
Ang2 := Angle[2]
Ang3 := Angle[3]
Ang4 := Angle[4]
Vizion.select_side (Parts, Ref, RightKept, 'right')
Vizion.count_obj (RightKept, RightCount)
Vizion.select_side (Parts, Ref, LeftKept, 'left')
Vizion.count_obj (LeftKept, LeftCount)
Vizion.select_side (Parts, Ref, BelowKept, 'below')
Vizion.count_obj (BelowKept, BelowCount)
Vizion.select_side (Parts, Ref, AboveKept, 'above')
Vizion.count_obj (AboveKept, AboveCount)
Vizion.sort_distance (Parts, Ref, Near, 'near', 'city-block')
Vizion.area_center (Near, NearArea, NearRow, NearCol)
NearR := NearRow[1]
NearC := NearCol[1]
Vizion.sort_distance (Parts, Ref, FarSorted, 'far', 'city-block')
Vizion.area_center (FarSorted, FarArea, FarRow, FarCol)
FarR := FarRow[1]
FarC := FarCol[1]
Vizion.nearest_one (Parts, Ref, Closest, 'city-block')
Vizion.area_center (Closest, CloseArea, CloseRow, CloseCol)
CloseN := CloseArea[1]
CloseR := CloseRow[1]
CloseC := CloseCol[1]
Vizion.farthest_one (Parts, Ref, Farthest, 'city-block')
Vizion.area_center (Farthest, FarMostArea, FarMostRow, FarMostCol)
FarN := FarMostArea[1]
FarMostR := FarMostRow[1]
FarMostC := FarMostCol[1]
Vizion.gap_area (Parts, Ref, 'city-block', GapArea)
Gap1 := GapArea[1]
Gap2 := GapArea[2]
Gap3 := GapArea[3]
Gap4 := GapArea[4]
Vizion.gap_width (Parts, Ref, 'city-block', GapWidth)
Wide1 := GapWidth[1]
Wide2 := GapWidth[2]
Wide3 := GapWidth[3]
Wide4 := GapWidth[4]
Vizion.gap_region (Right, Ref, Gap, 'city-block')
Vizion.area_center (Gap, GapOnly, GapRow, GapCol)
GapN := GapOnly[1]
Vizion.paint_region (Gap, Image, Shown, 255, 'fill')
Vizion.get_grayval (Shown, 34, 44, GapGray)
Vizion.get_grayval (Shown, 34, 39, RefGray)
Vizion.dev_display (Shown)
"""


HDEVELOP_BATCH49_PROGRAM = """* 空白分給最近的物件
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Left, 10, 10, 19, 19)
Vizion.gen_rectangle1 (Right, 10, 30, 19, 39)
Vizion.concat_obj (Left, Right, Parts)
Vizion.partition_nearest (Parts, Grown, 'city-block')
Vizion.count_obj (Grown, GrownN)
Vizion.area_center (Grown, GrownArea, GrownRow, GrownCol)
G1 := GrownArea[1]
G2 := GrownArea[2]
GR := GrownRow[1]
GC1 := GrownCol[1]
GC2 := GrownCol[2]
Vizion.partition_area (Parts, 'city-block', Territory)
T1 := Territory[1]
T2 := Territory[2]
Vizion.owner_index (Parts, [14, 14], [24, 25], 'city-block', GapOwn)
OwnL := GapOwn[1]
OwnR := GapOwn[2]
Vizion.owner_index (Parts, [0, 59], [0, 79], 'city-block', Corners)
CornerL := Corners[1]
CornerR := Corners[2]
Vizion.partition_border (Parts, Border, 'city-block')
Vizion.count_obj (Border, BorderN)
Vizion.area_center (Border, BorderArea, BorderRow, BorderCol)
B1 := BorderArea[1]
B2 := BorderArea[2]
BC1 := BorderCol[1]
BC2 := BorderCol[2]
Vizion.select_territory (Parts, Big, 'large', 'city-block')
Vizion.area_center (Big, BigArea, BigRow, BigCol)
BigN := BigArea[1]
BigC := BigCol[1]
Vizion.select_territory (Parts, Small, 'small', 'city-block')
Vizion.area_center (Small, SmallArea, SmallRow, SmallCol)
SmallN := SmallArea[1]
SmallC := SmallCol[1]
Vizion.gen_rectangle1 (Strip, 0, 20, 59, 29)
Vizion.partition_domain (Parts, Strip, Inside, 'city-block')
Vizion.count_obj (Inside, InsideN)
Vizion.area_center (Inside, InArea, InRow, InCol)
In1 := InArea[1]
In2 := InArea[2]
InC1 := InCol[1]
InC2 := InCol[2]
Vizion.test_region_point (Inside, 14, 14, OffStrip)
Vizion.test_region_point (Inside, 14, 22, OnStrip)
Vizion.partition_label (Parts, Labels, 'city-block')
Vizion.get_grayval (Labels, 14, 24, LabelL)
Vizion.get_grayval (Labels, 14, 25, LabelR)
Vizion.gen_region_points (DotA, 40, 10)
Vizion.gen_region_points (DotB, 40, 12)
Vizion.concat_obj (DotA, DotB, Dots)
Vizion.tie_region (Dots, Ties, 'city-block')
Vizion.count_obj (Ties, TieCount)
Vizion.area_center (Ties, TieArea, TieRow, TieCol)
TieN := TieArea[1]
TieR := TieRow[1]
TieC := TieCol[1]
Vizion.paint_region (Border, Image, Shown, 255, 'fill')
Vizion.get_grayval (Shown, 14, 24, CutGray)
Vizion.get_grayval (Shown, 14, 25, CutGray2)
Vizion.get_grayval (Shown, 14, 23, SideGray)
Vizion.dev_display (Shown)
"""


HDEVELOP_BATCH50_PROGRAM = """* 依特徵挑出物件
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Big, 10, 10, 19, 19)
Vizion.gen_rectangle1 (Tiny, 10, 30, 13, 33)
Vizion.gen_rectangle1 (Mid, 30, 10, 35, 17)
Vizion.gen_rectangle1 (Twin, 40, 40, 43, 43)
Vizion.concat_obj (Big, Tiny, Pair)
Vizion.concat_obj (Pair, Mid, Three)
Vizion.concat_obj (Three, Twin, Parts)
Vizion.sort_feature (Parts, Down, 'area', 'down')
Vizion.area_center (Down, DownArea, DownRow, DownCol)
D1 := DownArea[1]
D2 := DownArea[2]
D3 := DownArea[3]
DR := DownRow[1]
DC := DownCol[1]
DC3 := DownCol[3]
Vizion.sort_feature (Parts, Up, 'area', 'up')
Vizion.area_center (Up, UpArea, UpRow, UpCol)
U1 := UpArea[1]
UR := UpRow[1]
UC := UpCol[1]
UC2 := UpCol[2]
Vizion.select_top (Parts, Top, 2, 'area', 'down')
Vizion.count_obj (Top, TopN)
Vizion.area_center (Top, TopArea, TopRow, TopCol)
Top1 := TopArea[1]
Top2 := TopArea[2]
Vizion.select_extreme (Parts, Largest, 'max', 'area')
Vizion.area_center (Largest, LargeArea, LargeRow, LargeCol)
LargeN := LargeArea[1]
LargeR := LargeRow[1]
LargeC := LargeCol[1]
Vizion.select_extreme (Parts, Smallest, 'min', 'area')
Vizion.area_center (Smallest, SmallArea, SmallRow, SmallCol)
SmallN := SmallArea[1]
SmallR := SmallRow[1]
SmallC := SmallCol[1]
Vizion.feature_rank (Parts, 'area', 'down', Rank)
R1 := Rank[1]
R2 := Rank[2]
R3 := Rank[3]
R4 := Rank[4]
Vizion.select_rank (Parts, Second, 2, 'area', 'down')
Vizion.area_center (Second, SecondArea, SecondRow, SecondCol)
SecN := SecondArea[1]
SecR := SecondRow[1]
Vizion.select_rank (Parts, Picked, [1, 3], 'area', 'down')
Vizion.area_center (Picked, PickArea, PickRow, PickCol)
Pick1 := PickArea[1]
Pick2 := PickArea[2]
PickC := PickCol[2]
Vizion.feature_index (Parts, 'area', 'down', Index)
I1 := Index[1]
I2 := Index[2]
I3 := Index[3]
I4 := Index[4]
Vizion.select_relative (Parts, Above, 50, 'area', 'above')
Vizion.count_obj (Above, AboveN)
Vizion.area_center (Above, AboveArea, AboveRow, AboveCol)
Ab1 := AboveArea[1]
Ab2 := AboveArea[2]
Vizion.select_relative (Parts, Below, 50, 'area', 'below')
Vizion.count_obj (Below, BelowN)
Vizion.feature_range (Parts, 'area', Low, High)
Vizion.paint_region (Largest, Image, Shown, 255, 'fill')
Vizion.get_grayval (Shown, 14, 14, HitGray)
Vizion.get_grayval (Shown, 11, 31, MissGray)
Vizion.dev_display (Shown)
"""


HDEVELOP_BATCH51_PROGRAM = """* 補上缺掉的缺口
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Block, 10, 10, 39, 39)
Vizion.gen_rectangle1 (Bite, 20, 30, 29, 39)
Vizion.difference (Block, Bite, Notched)
Vizion.gen_rectangle1 (Plain, 45, 45, 54, 54)
Vizion.gen_rectangle1 (CornerBlock, 10, 50, 29, 69)
Vizion.gen_rectangle1 (CornerBite, 10, 65, 14, 69)
Vizion.difference (CornerBlock, CornerBite, Corner)
Vizion.concat_obj (Notched, Plain, Pair)
Vizion.concat_obj (Pair, Corner, Parts)
Vizion.region_defects (Parts, Bites, 1, 'city-block')
Vizion.count_obj (Bites, BiteN)
Vizion.area_center (Bites, BiteArea, BiteRow, BiteCol)
BA1 := BiteArea[1]
BA2 := BiteArea[2]
BR1 := BiteRow[1]
BC1 := BiteCol[1]
BR2 := BiteRow[2]
BC2 := BiteCol[2]
Vizion.defect_count (Parts, 1, 'city-block', Counts)
C1 := Counts[1]
C2 := Counts[2]
C3 := Counts[3]
Vizion.defect_area (Parts, 1, 'city-block', Areas)
DA1 := Areas[1]
DA2 := Areas[2]
Vizion.defect_depth (Parts, 1, 'city-block', Depths)
DP1 := Depths[1]
DP2 := Depths[2]
Vizion.defect_point (Parts, 1, 'city-block', PointRow, PointCol, PointDepth)
PR1 := PointRow[1]
PC1 := PointCol[1]
PD1 := PointDepth[1]
PR2 := PointRow[2]
PC2 := PointCol[2]
PD2 := PointDepth[2]
Vizion.select_defect (Parts, Chipped, 2, 'city-block')
Vizion.count_obj (Chipped, ChipN)
Vizion.area_center (Chipped, ChipArea, ChipRow, ChipCol)
Chip1 := ChipArea[1]
Chip2 := ChipArea[2]
Vizion.region_defects (Parts, DeepOnly, 6, 'city-block')
Vizion.count_obj (DeepOnly, DeepN)
Vizion.area_center (DeepOnly, DeepArea, DeepRow, DeepCol)
DeepA := DeepArea[1]
Vizion.deepest_defect (Parts, Deepest, 'city-block')
Vizion.area_center (Deepest, MostArea, MostRow, MostCol)
MostN := MostArea[1]
Vizion.fill_defects (Parts, Fixed, 1, 'city-block')
Vizion.area_center (Fixed, FixArea, FixRow, FixCol)
F1 := FixArea[1]
F2 := FixArea[2]
F3 := FixArea[3]
Vizion.paint_region (Deepest, Image, Shown, 255, 'fill')
Vizion.get_grayval (Shown, 24, 34, GapGray)
Vizion.get_grayval (Shown, 14, 14, BodyGray)
Vizion.dev_display (Shown)
"""


HDEVELOP_BATCH52_PROGRAM = """* 誰包住誰
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Outer, 5, 5, 44, 44)
Vizion.gen_rectangle1 (Hole, 15, 15, 34, 34)
Vizion.difference (Outer, Hole, Frame)
Vizion.gen_rectangle1 (Middle, 18, 18, 31, 31)
Vizion.gen_rectangle1 (Tiny, 22, 22, 25, 25)
Vizion.gen_rectangle1 (Out, 50, 50, 54, 54)
Vizion.concat_obj (Frame, Middle, Pair)
Vizion.concat_obj (Pair, Tiny, Three)
Vizion.concat_obj (Three, Out, Parts)
Vizion.select_inner (Parts, Inner)
Vizion.count_obj (Inner, InnerN)
Vizion.area_center (Inner, InnerArea, InnerRow, InnerCol)
In1 := InnerArea[1]
In2 := InnerArea[2]
Vizion.select_outer (Parts, OuterKept)
Vizion.count_obj (OuterKept, OuterN)
Vizion.area_center (OuterKept, OuterArea, OuterRow, OuterCol)
Out1 := OuterArea[1]
Out2 := OuterArea[2]
Vizion.select_container (Parts, Holders)
Vizion.count_obj (Holders, HolderN)
Vizion.area_center (Holders, HolderArea, HolderRow, HolderCol)
Hold1 := HolderArea[1]
Hold2 := HolderArea[2]
Vizion.innermost (Parts, Leaf)
Vizion.area_center (Leaf, LeafArea, LeafRow, LeafCol)
LeafN := LeafArea[1]
LeafR := LeafRow[1]
LeafC := LeafCol[1]
Vizion.select_depth (Parts, Level1, 1)
Vizion.area_center (Level1, L1Area, L1Row, L1Col)
MidN := L1Area[1]
MidR := L1Row[1]
Vizion.select_depth (Parts, Level2, 2)
Vizion.count_obj (Level2, DeepN)
Vizion.select_depth (Parts, Level0, 0)
Vizion.count_obj (Level0, TopN)
Vizion.container_index (Parts, Holder)
H1 := Holder[1]
H2 := Holder[2]
H3 := Holder[3]
H4 := Holder[4]
Vizion.inner_count (Parts, InsideCount)
K1 := InsideCount[1]
K2 := InsideCount[2]
K3 := InsideCount[3]
K4 := InsideCount[4]
Vizion.nesting_depth (Parts, NestLevel)
D1 := NestLevel[1]
D2 := NestLevel[2]
D3 := NestLevel[3]
D4 := NestLevel[4]
Vizion.paint_region (Leaf, Image, Shown, 255, 'fill')
Vizion.get_grayval (Shown, 23, 23, LeafGray)
Vizion.get_grayval (Shown, 10, 10, FrameGray)
Vizion.dev_display (Shown)
"""


HDEVELOP_BATCH53_PROGRAM = """* 有沒有碰到影像邊緣
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (TouchTop, 0, 20, 9, 29)
Vizion.gen_rectangle1 (TouchLeft, 20, 0, 29, 4)
Vizion.gen_rectangle1 (Center, 25, 35, 34, 44)
Vizion.gen_rectangle1 (NearRight, 40, 70, 49, 77)
Vizion.concat_obj (TouchTop, TouchLeft, Pair)
Vizion.concat_obj (Pair, Center, Three)
Vizion.concat_obj (Three, NearRight, Parts)
Vizion.select_border (Parts, Touched, 'touch')
Vizion.count_obj (Touched, TouchN)
Vizion.area_center (Touched, TouchArea, TouchRow, TouchCol)
Touch1 := TouchArea[1]
Touch2 := TouchArea[2]
Vizion.select_border (Parts, Inside, 'inner')
Vizion.count_obj (Inside, InnerN)
Vizion.area_center (Inside, InnerArea, InnerRow, InnerCol)
Inner1 := InnerArea[1]
Inner2 := InnerArea[2]
Vizion.select_border_side (Parts, TopSide, 'top')
Vizion.count_obj (TopSide, TopN)
Vizion.area_center (TopSide, TopArea, TopRow, TopCol)
TopA := TopArea[1]
Vizion.select_border_side (Parts, RightSide, 'right')
Vizion.count_obj (RightSide, RightN)
Vizion.select_margin (Parts, Kept, 3)
Vizion.count_obj (Kept, KeepN)
Vizion.area_center (Kept, KeepArea, KeepRow, KeepCol)
KeepA := KeepArea[1]
Vizion.select_border_range (Parts, Band, 1, 10)
Vizion.count_obj (Band, BandN)
Vizion.area_center (Band, BandArea, BandRow, BandCol)
BandA := BandArea[1]
Vizion.farthest_border (Parts, Far)
Vizion.area_center (Far, FarArea, FarRow, FarCol)
FarN := FarArea[1]
FarR := FarRow[1]
FarC := FarCol[1]
Vizion.border_distance (Parts, Distance)
D1 := Distance[1]
D2 := Distance[2]
D3 := Distance[3]
D4 := Distance[4]
Vizion.border_gap (Parts, GapTop, GapBottom, GapLeft, GapRight)
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
Vizion.border_point (Parts, PointRow, PointCol, PointSide)
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
Vizion.paint_region (Touched, Image, Shown, 255, 'fill')
Vizion.get_grayval (Shown, 0, 20, EdgeGray)
Vizion.get_grayval (Shown, 25, 35, MidGray)
Vizion.dev_display (Shown)
"""


HDEVELOP_BATCH54_PROGRAM = """* 物件有沒有互相碰到
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Block, 10, 10, 19, 19)
Vizion.gen_rectangle1 (Beside, 10, 20, 19, 24)
Vizion.gen_rectangle1 (Below, 20, 20, 24, 24)
Vizion.gen_rectangle1 (Far, 40, 40, 44, 44)
Vizion.concat_obj (Block, Beside, Pair)
Vizion.concat_obj (Pair, Below, Three)
Vizion.concat_obj (Three, Far, Parts)
Vizion.neighbor_distance (Parts, 'chessboard', Distance)
D1 := Distance[1]
D2 := Distance[2]
D3 := Distance[3]
D4 := Distance[4]
Vizion.neighbor_index (Parts, 'chessboard', NearIndex)
I1 := NearIndex[1]
I2 := NearIndex[2]
I3 := NearIndex[3]
I4 := NearIndex[4]
Vizion.neighbor_point (Parts, 'chessboard', NearRow, NearCol, NearGap)
NR1 := NearRow[1]
NC1 := NearCol[1]
NG1 := NearGap[1]
NR2 := NearRow[2]
NC2 := NearCol[2]
NR4 := NearRow[4]
NC4 := NearCol[4]
NG4 := NearGap[4]
Vizion.neighbor_count (Parts, 1, 'chessboard', TouchCount)
K1 := TouchCount[1]
K2 := TouchCount[2]
K3 := TouchCount[3]
K4 := TouchCount[4]
Vizion.select_touching (Parts, Touched, 1, 'chessboard')
Vizion.count_obj (Touched, TouchN)
Vizion.area_center (Touched, TouchArea, TouchRow, TouchCol)
Touch1 := TouchArea[1]
Touch2 := TouchArea[2]
Touch3 := TouchArea[3]
Vizion.select_isolated (Parts, Alone, 1, 'chessboard')
Vizion.count_obj (Alone, AloneN)
Vizion.area_center (Alone, AloneArea, AloneRow, AloneCol)
AloneA := AloneArea[1]
AloneR := AloneRow[1]
AloneC := AloneCol[1]
Vizion.contact_region (Parts, Contact, 1, 'chessboard')
Vizion.count_obj (Contact, ContactN)
Vizion.area_center (Contact, ContactArea, ContactRow, ContactCol)
CA1 := ContactArea[1]
CA2 := ContactArea[2]
CA3 := ContactArea[3]
CR1 := ContactRow[1]
CC1 := ContactCol[1]
Vizion.union_touching (Parts, Merged, 1, 'chessboard')
Vizion.count_obj (Merged, MergeN)
Vizion.area_center (Merged, MergeArea, MergeRow, MergeCol)
M1 := MergeArea[1]
M2 := MergeArea[2]
MR := MergeRow[1]
MC := MergeCol[1]
Vizion.paint_region (Contact, Image, Shown, 255, 'fill')
Vizion.get_grayval (Shown, 14, 19, TouchGray)
Vizion.get_grayval (Shown, 14, 18, BodyGray)
Vizion.get_grayval (Shown, 42, 42, FarGray)
Vizion.dev_display (Shown)
"""


HDEVELOP_BATCH55_PROGRAM = """* 哪個物件最像範本
* 用法：範本一個方塊。拿去比對同樣的方塊、較大的方塊、轉 20 度的方塊，和一根長條。
* 分數越小越像。搬家、旋轉、放大縮小都不計。i1 最常用。Score[1] 從 1 開始。
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Template, 10, 10, 19, 19)
Vizion.gen_rectangle1 (Same, 10, 40, 19, 49)
Vizion.gen_rectangle1 (Bigger, 40, 40, 54, 54)
Vizion.gen_rectangle2 (Tilted, 30, 60, rad(20), 6, 6)
Vizion.gen_rectangle1 (Bar, 30, 10, 34, 39)
Vizion.concat_obj (Same, Bigger, Pair)
Vizion.concat_obj (Pair, Tilted, Three)
Vizion.concat_obj (Three, Bar, Parts)
* 四個分數：同樣的方塊是 0，較大的方塊也接近 0，轉過的約 0.00179，長條約 3.46192
Vizion.shape_score (Parts, Template, 'i1', Score)
S1 := Score[1]
S2 := Score[2]
S3 := Score[3]
S4 := Score[4]
* 只要最像或最不像的那一個，不必自己找最小最大
Vizion.most_similar (Parts, Template, Best, 'i1')
Vizion.area_center (Best, BestArea, BestRow, BestCol)
BestN := BestArea[1]
BestR := BestRow[1]
BestC := BestCol[1]
Vizion.least_similar (Parts, Template, Worst, 'i1')
Vizion.area_center (Worst, WorstArea, WorstRow, WorstCol)
WorstN := WorstArea[1]
WorstR := WorstRow[1]
WorstC := WorstCol[1]
* near 最像的排前面，far 反過來。兩項都可省略，預設就是這樣
Vizion.sort_shape (Parts, Template, Near, 'near', 'i1')
Vizion.area_center (Near, NearArea, NearRow, NearCol)
Near1 := NearArea[1]
Near2 := NearArea[2]
NearC := NearCol[1]
Vizion.sort_shape (Parts, Template, FarSorted, 'far', 'i1')
Vizion.area_center (FarSorted, FarArea, FarRow, FarCol)
Far1 := FarArea[1]
FarR := FarRow[1]
* Rank 對齊原來的物件。Index 是由像到不像的原來編號，可接 select_obj
Vizion.shape_rank (Parts, Template, 'i1', Rank)
R1 := Rank[1]
R2 := Rank[2]
R3 := Rank[3]
R4 := Rank[4]
Vizion.shape_index (Parts, Template, 'i1', Index)
I1 := Index[1]
I2 := Index[2]
I3 := Index[3]
I4 := Index[4]
* 0.01 留下方塊和轉過的方塊，長條被去掉
Vizion.select_similar (Parts, Template, Kept, 0.01, 'i1')
Vizion.count_obj (Kept, KeepN)
Vizion.area_center (Kept, KeepArea, KeepRow, KeepCol)
Keep1 := KeepArea[1]
Keep2 := KeepArea[2]
Keep3 := KeepArea[3]
* 七個不變矩。方塊的 Hu1 約 0.16667。比像不像時，用上面的分數比較直接
Vizion.hu_moments (Template, Hu1, Hu2, Hu3, Hu4, Hu5, Hu6, Hu7)
H1 := Hu1[1]
Vizion.hu_moments (Same, Same1, Same2, Same3, Same4, Same5, Same6, Same7)
SameH := Same1[1]
Vizion.paint_region (Best, Image, Shown, 255, 'fill')
Vizion.get_grayval (Shown, 14, 44, LikeGray)
Vizion.get_grayval (Shown, 32, 24, BarGray)
Vizion.dev_display (Shown)
"""


HDEVELOP_BATCH56_PROGRAM = """* 把物件搬到範本上
* 用法：shape_score 看像不像。這裡把物件搬到範本中心，看蓋住多少、還差幾個像素。
* center 只搬位置，位移四捨五入成整數像素。方塊請用 center。
* rigid 再把長軸轉到和範本一樣。長條才用這個。長軸顛倒 180 度當成同一個方向。
* 正的列往下搬，正的行往右搬。DRow[1] 從 1 開始。Cover 為 1 是完全蓋住範本。
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Template, 10, 10, 19, 19)
Vizion.gen_rectangle1 (Shifted, 10, 40, 19, 49)
Vizion.gen_rectangle1 (Small, 40, 40, 45, 45)
Vizion.gen_rectangle1 (Bar, 30, 50, 34, 69)
Vizion.concat_obj (Shifted, Small, Pair)
Vizion.concat_obj (Pair, Bar, Parts)
* 要搬的量。右邊那塊方塊往左 30 像素，長條的列還有半個像素
Vizion.align_delta (Parts, Template, DRow, DColumn)
DR1 := DRow[1]
DC1 := DColumn[1]
DR2 := DRow[2]
DC2 := DColumn[2]
DR3 := DRow[3]
DC3 := DColumn[3]
* 搬完之後重疊幾個像素，以及蓋住範本的比例
Vizion.align_overlap (Parts, Template, 'center', Overlap)
O1 := Overlap[1]
O2 := Overlap[2]
O3 := Overlap[3]
Vizion.align_cover (Parts, Template, 'center', Cover)
C1 := Cover[1]
C2 := Cover[2]
C3 := Cover[3]
* 完全疊上是 0。小方塊差 64，長條差 100
Vizion.mismatch_area (Parts, Template, 'center', Miss)
MA1 := Miss[1]
MA2 := Miss[2]
MA3 := Miss[3]
Vizion.mismatch_region (Parts, Template, Mismatch, 'center')
Vizion.count_obj (Mismatch, MissN)
Vizion.area_center (Mismatch, MissArea, MissRow, MissCol)
MissA1 := MissArea[1]
MissA2 := MissArea[2]
* 0.95 只留下幾乎完全蓋住的那一塊，留下的仍是原來的位置
Vizion.select_aligned (Parts, Template, Kept, 0.95, 'center')
Vizion.count_obj (Kept, KeepN)
Vizion.area_center (Kept, KeepArea, KeepRow, KeepCol)
KeepA := KeepArea[1]
KeepC := KeepCol[1]
Vizion.align_region (Parts, Template, Moved, 'center')
Vizion.area_center (Moved, MoveArea, MoveRow, MoveCol)
Move1 := MoveArea[1]
MoveR := MoveRow[1]
MoveC := MoveCol[1]
* 直的長條轉到橫的長條上。center 只會交錯 25 像素，rigid 轉完重疊 100
Vizion.gen_rectangle1 (Wide, 50, 10, 54, 29)
Vizion.gen_rectangle1 (Tall, 10, 70, 29, 74)
Vizion.align_angle (Tall, Wide, DPhi)
Phi := DPhi[1]
Vizion.align_overlap (Tall, Wide, 'center', Cross)
CrossN := Cross[1]
Vizion.align_overlap (Tall, Wide, 'rigid', TurnedOver)
TurnN := TurnedOver[1]
Vizion.align_region (Tall, Wide, Turned, 'rigid')
Vizion.area_center (Turned, TurnArea, TurnRow, TurnCol)
TurnA := TurnArea[1]
TurnR := TurnRow[1]
TurnC := TurnCol[1]
Vizion.select_obj (Moved, First, 1)
Vizion.paint_region (First, Image, Shown, 255, 'fill')
Vizion.get_grayval (Shown, 14, 14, OnGray)
Vizion.get_grayval (Shown, 14, 44, OffGray)
Vizion.dev_display (Shown)
"""


HDEVELOP_BATCH57_PROGRAM = """* 物件排成幾列幾行
* 用法：很多相同的小物件排成陣列時，用這個看它在第幾列第幾行，並把缺掉的格子補出來。
* Gap 可省略。0 表示用物件高度：中心列差比高度大，就換下一列。列距比物件還近時，再自己填較小的 Gap。
* 行距用同一列裡最小的相鄰間距。某一格的間距大約是兩倍，中間就當缺一格。每一列都缺同一行時看不出來。
* 編號從 1 開始，Row[1] 對齊原來的第一個物件。
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (A, 10, 10, 13, 13)
Vizion.gen_rectangle1 (B, 10, 30, 13, 33)
Vizion.gen_rectangle1 (C, 10, 50, 13, 53)
Vizion.gen_rectangle1 (D, 30, 10, 33, 13)
Vizion.gen_rectangle1 (E, 30, 50, 33, 53)
Vizion.concat_obj (A, B, Pair)
Vizion.concat_obj (Pair, C, Three)
Vizion.concat_obj (Three, D, Four)
Vizion.concat_obj (Four, E, Parts)
* 上面三格、下面左右各一格。仍是 2 列 3 行，列距和行距都是 20
Vizion.grid_size (Parts, GridRows, GridCols)
Vizion.grid_pitch (Parts, RowPitch, ColPitch)
Vizion.grid_index (Parts, RowIndex, ColIndex)
R1 := RowIndex[1]
R4 := RowIndex[4]
R5 := RowIndex[5]
K1 := ColIndex[1]
K2 := ColIndex[2]
K3 := ColIndex[3]
K4 := ColIndex[4]
K5 := ColIndex[5]
* 由左上讀到右下。缺的中間格會跳過
Vizion.sort_grid (Parts, Ordered)
Vizion.area_center (Ordered, OrdArea, OrdRow, OrdCol)
S1C := OrdCol[1]
S3C := OrdCol[3]
S4C := OrdCol[4]
S5R := OrdRow[5]
* 缺的是第二列第二行，補一塊和別的物件一樣大的方塊
Vizion.missing_cell (Parts, Missing)
Vizion.count_obj (Missing, MissN)
Vizion.area_center (Missing, MissArea, MissRow, MissCol)
MissA := MissArea[1]
MissR := MissRow[1]
MissC := MissCol[1]
* 有物件就取出來。空格用 cell_center，不必真的有物件
Vizion.select_grid (Parts, Cell, 2, 3)
Vizion.area_center (Cell, CellArea, CellRow, CellCol)
CellA := CellArea[1]
CellR := CellRow[1]
CellC := CellCol[1]
Vizion.cell_center (Parts, 2, 2, HoleR, HoleC)
* 像素值是行號。第一行是 1，缺的格子是 0
Vizion.grid_label (Parts, Labeled)
Vizion.get_grayval (Labeled, 11, 11, G1)
Vizion.get_grayval (Labeled, 11, 31, G2)
Vizion.get_grayval (Labeled, 31, 51, G3)
Vizion.get_grayval (Labeled, 31, 31, G0)
Vizion.dev_display (Labeled)
"""


HDEVELOP_BATCH58_PROGRAM = """* 亮度偏到哪一邊
* 用法：看亮的部分相對形狀中心偏了哪邊。正的列是偏下，正的行是偏右。0 是正在中間。
* 方向 0 度向右，90 度向下，270 度向上。Balance 為 1 是在正中間，0 是全擠在邊緣。
* 右半邊較亮的方塊大約是 0.68，不是 0。center 表示幾乎沒偏。
* 較亮的一半用形狀中心切開。亮度在中間時，亮的一半和暗的一半都是整個物件。
Vizion.gen_image_const (Blank, 'byte', 80, 60)
Vizion.gen_rectangle1 (RightLeft, 10, 10, 29, 19)
Vizion.gen_rectangle1 (RightRight, 10, 20, 29, 29)
Vizion.paint_region (RightLeft, Blank, Step1, 50, 'fill')
Vizion.paint_region (RightRight, Step1, Step2, 200, 'fill')
Vizion.union2 (RightLeft, RightRight, RightBlock)
Vizion.gen_rectangle1 (Flat, 10, 40, 19, 59)
Vizion.paint_region (Flat, Step2, Step3, 100, 'fill')
Vizion.gen_rectangle1 (TopTop, 40, 40, 49, 59)
Vizion.gen_rectangle1 (TopBottom, 50, 40, 59, 59)
Vizion.paint_region (TopTop, Step3, Step4, 200, 'fill')
Vizion.paint_region (TopBottom, Step4, Picture, 50, 'fill')
Vizion.union2 (TopTop, TopBottom, TopBlock)
Vizion.concat_obj (RightBlock, Flat, Pair)
Vizion.concat_obj (Pair, TopBlock, Parts)
* 右邊較亮的方塊行偏 3。均勻的是 0。上面較亮的方塊列偏 -3
Vizion.gray_offset (Parts, Picture, DRow, DColumn)
DR1 := DRow[1]
DC1 := DColumn[1]
DR2 := DRow[2]
DC2 := DColumn[2]
DR3 := DRow[3]
DC3 := DColumn[3]
Vizion.gray_shift (Parts, Picture, Shift)
S1 := Shift[1]
S2 := Shift[2]
S3 := Shift[3]
Vizion.gray_direction (Parts, Picture, Phi)
P1 := Phi[1]
P2 := Phi[2]
P3 := Phi[3]
Vizion.bright_side (Parts, Picture, Side)
Side1 := Side[1]
Side2 := Side[2]
Side3 := Side[3]
Vizion.gray_balance (Parts, Picture, Balance)
B1 := Balance[1]
B2 := Balance[2]
B3 := Balance[3]
* 較亮的一半在右，中心行 24.5。較暗的一半在左，中心行 14.5
Vizion.bright_half (Parts, Picture, Bright)
Vizion.select_obj (Bright, BrightOne, 1)
Vizion.area_center (BrightOne, BrightArea, BrightRow, BrightCol)
BA := BrightArea[1]
BR := BrightRow[1]
BC := BrightCol[1]
Vizion.dark_half (Parts, Picture, Dark)
Vizion.select_obj (Dark, DarkOne, 1)
Vizion.area_center (DarkOne, DarkArea, DarkRow, DarkCol)
DA := DarkArea[1]
DR := DarkRow[1]
DC := DarkCol[1]
* 偏移至少 1 才留下偏右的。center 則是偏移小於 0.5
Vizion.select_bright (Parts, Picture, Kept, 'right', 1)
Vizion.count_obj (Kept, KeepN)
Vizion.area_center (Kept, KeepArea, KeepRow, KeepCol)
KeepA := KeepArea[1]
KeepC := KeepCol[1]
Vizion.select_bright (Parts, Picture, Middle, 'center', 0.5)
Vizion.count_obj (Middle, MidN)
Vizion.area_center (Middle, MidArea, MidRow, MidCol)
MidA := MidArea[1]
MidC := MidCol[1]
Vizion.get_grayval (Picture, 19, 14, LeftGray)
Vizion.get_grayval (Picture, 19, 24, RightGray)
Vizion.dev_display (Picture)
"""


HDEVELOP_BATCH59_PROGRAM = """* 邊緣清不清楚
* 用法：看物件最外一圈，和貼在外面的那一圈，灰階差多少。正的表示物件比較亮，負的表示外面比較亮。
* 0 是邊界兩邊一樣，看不出邊緣。差的絕對值愈大，邊緣愈清楚。門檻可省略，預設 10。
* inner 是物件自己的最外一像素，outer 是外面那一圈。物件只有一像素厚時，內圈就是整個物件。
Vizion.gen_image_const (Blank, 'byte', 80, 60)
Vizion.gen_rectangle1 (Sharp, 10, 10, 29, 29)
Vizion.paint_region (Sharp, Blank, Step1, 200, 'fill')
Vizion.gen_rectangle1 (FlatBig, 8, 38, 31, 61)
Vizion.paint_region (FlatBig, Step1, Step2, 100, 'fill')
Vizion.gen_rectangle1 (Flat, 10, 40, 29, 59)
Vizion.gen_rectangle1 (DarkBig, 38, 8, 61, 31)
Vizion.paint_region (DarkBig, Step2, Step3, 180, 'fill')
Vizion.gen_rectangle1 (Dark, 40, 10, 59, 29)
Vizion.paint_region (Dark, Step3, Picture, 20, 'fill')
Vizion.concat_obj (Sharp, Flat, Pair)
Vizion.concat_obj (Pair, Dark, Parts)
* 清楚的邊緣差 200。融在背景裡的是 0。外面比較亮的是 -160
Vizion.edge_contrast (Parts, Picture, Contrast)
C1 := Contrast[1]
C2 := Contrast[2]
C3 := Contrast[3]
Vizion.edge_abs (Parts, Picture, AbsContrast)
A1 := AbsContrast[1]
A2 := AbsContrast[2]
A3 := AbsContrast[3]
Vizion.inner_gray (Parts, Picture, InnerMean)
I1 := InnerMean[1]
I2 := InnerMean[2]
I3 := InnerMean[3]
Vizion.outer_gray (Parts, Picture, OuterMean)
O1 := OuterMean[1]
O2 := OuterMean[2]
O3 := OuterMean[3]
* 差至少 100 才留下。融在背景裡的用較小的上限抓
Vizion.select_edge (Parts, Picture, Kept, 100)
Vizion.count_obj (Kept, KeepN)
Vizion.area_center (Kept, KeepArea, KeepRow, KeepCol)
Keep1 := KeepRow[1]
Keep2 := KeepRow[2]
Vizion.select_soft (Parts, Picture, Soft, 10)
Vizion.count_obj (Soft, SoftN)
Vizion.area_center (Soft, SoftArea, SoftRow, SoftCol)
SoftA := SoftArea[1]
SoftC := SoftCol[1]
Vizion.strongest_edge (Parts, Picture, Best)
Vizion.area_center (Best, BestArea, BestRow, BestCol)
BestA := BestArea[1]
BestR := BestRow[1]
* 清楚那塊的內圈 76 像素，外圈 84 像素，中心仍在方塊中間
Vizion.edge_ring (Parts, Picture, InnerRing, 'inner')
Vizion.select_obj (InnerRing, InnerOne, 1)
Vizion.area_center (InnerOne, InnerArea, InnerRow, InnerCol)
InnerA := InnerArea[1]
InnerR := InnerRow[1]
InnerC := InnerCol[1]
Vizion.edge_ring (Parts, Picture, OuterRing, 'outer')
Vizion.select_obj (OuterRing, OuterOne, 1)
Vizion.area_center (OuterOne, OuterArea, OuterRow, OuterCol)
OuterA := OuterArea[1]
Vizion.get_grayval (Picture, 19, 19, SharpGray)
Vizion.get_grayval (Picture, 9, 19, OutsideGray)
Vizion.get_grayval (Picture, 49, 19, DarkGray)
Vizion.get_grayval (Picture, 39, 19, HaloGray)
Vizion.dev_display (Picture)
"""


HDEVELOP_BATCH60_PROGRAM = """* 邊緣哪裡最弱
* 用法：edge_contrast 是整圈平均。這裡只看最弱的一點，物件最外一像素減掉緊鄰的外側。
* 負的表示這一點外面比較亮。edge_floor 只看差多少，不管誰亮。0 是這一點兩邊一樣。
* 外側有好幾個時，先看上、下、左、右。一樣弱時留列最小、再來行最小的那一點。
* 門檻可省略。weak_count 和 select_broken 預設 30，select_closed 預設 10。
Vizion.gen_image_const (Blank, 'byte', 80, 60)
Vizion.gen_rectangle1 (Gap, 15, 15, 34, 34)
Vizion.paint_region (Gap, Blank, Step1, 200, 'fill')
Vizion.gen_rectangle1 (Patch, 20, 35, 24, 35)
Vizion.paint_region (Patch, Step1, Step2, 210, 'fill')
Vizion.gen_rectangle1 (Sharp, 15, 50, 34, 69)
Vizion.paint_region (Sharp, Step2, Step3, 200, 'fill')
Vizion.gen_rectangle1 (FlatBig, 36, 6, 59, 29)
Vizion.paint_region (FlatBig, Step3, Picture, 100, 'fill')
Vizion.gen_rectangle1 (Flat, 38, 8, 57, 27)
Vizion.concat_obj (Gap, Sharp, Pair)
Vizion.concat_obj (Pair, Flat, Parts)
* 缺口那點外面是 210，物件是 200，所以是 -10。清楚的整圈最弱也有 200。融在背景裡的是 0
Vizion.edge_min (Parts, Picture, Contrast)
M1 := Contrast[1]
M2 := Contrast[2]
M3 := Contrast[3]
Vizion.edge_floor (Parts, Picture, Floor)
F1 := Floor[1]
F2 := Floor[2]
F3 := Floor[3]
Vizion.edge_min_point (Parts, Picture, MinRow, MinCol)
MR1 := MinRow[1]
MC1 := MinCol[1]
MR2 := MinRow[2]
MC2 := MinCol[2]
MR3 := MinRow[3]
MC3 := MinCol[3]
Vizion.edge_min_side (Parts, Picture, Side)
Side1 := Side[1]
Side2 := Side[2]
Side3 := Side[3]
* 差不超過 30 的點。缺口有 5 個，清楚的是 0，融在背景裡的整圈都算
Vizion.weak_count (Parts, Picture, 30, WeakN)
N1 := WeakN[1]
N2 := WeakN[2]
N3 := WeakN[3]
Vizion.weak_edge (Parts, Picture, Weak, 30)
Vizion.count_obj (Weak, WeakObj)
Vizion.area_center (Weak, WeakArea, WeakRow, WeakCol)
WA1 := WeakArea[1]
WR1 := WeakRow[1]
WC1 := WeakCol[1]
WA2 := WeakArea[2]
* 最弱的地方也要達到 100，才算整圈都清楚。30 以下當成有斷掉
Vizion.select_closed (Parts, Picture, Closed, 100)
Vizion.count_obj (Closed, ClosedN)
Vizion.area_center (Closed, ClosedArea, ClosedRow, ClosedCol)
ClosedA := ClosedArea[1]
ClosedR := ClosedRow[1]
ClosedC := ClosedCol[1]
Vizion.select_broken (Parts, Picture, Broken, 30)
Vizion.count_obj (Broken, BrokenN)
Vizion.area_center (Broken, BrokenArea, BrokenRow, BrokenCol)
BR1 := BrokenRow[1]
BR2 := BrokenRow[2]
Vizion.get_grayval (Picture, 20, 34, EdgeGray)
Vizion.get_grayval (Picture, 20, 35, LeakGray)
Vizion.dev_display (Picture)
"""


HDEVELOP_BATCH61_PROGRAM = """* 物件裡有多少夠亮
* 用法：數物件裡面落在 Low 到 High 的像素。兩個門檻都含在內，寫反了會自動對調。
* 比例 1 是整個物件都合格，0 是完全沒有。分母是物件自己的像素。
* 最亮或最暗有很多點時，留列最小、再來行最小的那一點。peak_region 則把同樣亮的點全部留下。
* select_band 的比例可省略，預設 0.5。select_uniform 的 0 表示整個物件同一個灰階。
Vizion.gen_image_const (Blank, 'byte', 80, 60)
Vizion.gen_rectangle1 (ALeft, 10, 10, 19, 19)
Vizion.paint_region (ALeft, Blank, Step1, 40, 'fill')
Vizion.gen_rectangle1 (ARight, 10, 20, 19, 29)
Vizion.paint_region (ARight, Step1, Step2, 180, 'fill')
Vizion.union2 (ALeft, ARight, BlockA)
Vizion.gen_rectangle1 (BlockB, 30, 10, 39, 19)
Vizion.paint_region (BlockB, Step2, Step3, 200, 'fill')
Vizion.gen_rectangle1 (BlockC, 30, 40, 39, 59)
Vizion.paint_region (BlockC, Step3, Step4, 20, 'fill')
Vizion.gen_rectangle1 (Hot, 30, 40, 31, 41)
Vizion.paint_region (Hot, Step4, Picture, 255, 'fill')
Vizion.concat_obj (BlockA, BlockB, Pair)
Vizion.concat_obj (Pair, BlockC, Parts)
* 100 到 255：半邊較亮的有 100 個，整塊亮的有 100 個，小亮點只有 4 個
Vizion.band_count (Parts, Picture, 100, 255, Count)
N1 := Count[1]
N2 := Count[2]
N3 := Count[3]
Vizion.band_count (Parts, Picture, 255, 100, Swapped)
S1 := Swapped[1]
Vizion.band_ratio (Parts, Picture, 100, 255, Ratio)
R1 := Ratio[1]
R2 := Ratio[2]
R3 := Ratio[3]
Vizion.band_region (Parts, Picture, Band, 100, 255)
Vizion.area_center (Band, BandArea, BandRow, BandCol)
BA1 := BandArea[1]
BandC := BandCol[1]
BA3 := BandArea[3]
* 至少 80% 才留下。半邊亮的是 50%，不會留下
Vizion.select_band (Parts, Picture, Kept, 100, 255, 0.8)
Vizion.count_obj (Kept, KeepN)
Vizion.area_center (Kept, KeepArea, KeepRow, KeepCol)
KeepA := KeepArea[1]
KeepR := KeepRow[1]
KeepC := KeepCol[1]
Vizion.bright_point (Parts, Picture, BrightRow, BrightCol, BrightGray)
BR1 := BrightRow[1]
BC1 := BrightCol[1]
BG1 := BrightGray[1]
BR3 := BrightRow[3]
BC3 := BrightCol[3]
BG3 := BrightGray[3]
Vizion.dark_point (Parts, Picture, DarkRow, DarkCol, DarkGray)
DR1 := DarkRow[1]
DC1 := DarkCol[1]
DG1 := DarkGray[1]
DR2 := DarkRow[2]
DG2 := DarkGray[2]
DR3 := DarkRow[3]
DC3 := DarkCol[3]
DG3 := DarkGray[3]
* 最亮的像素。第三個物件只有 2×2 的 255
Vizion.peak_region (Parts, Picture, Peak)
Vizion.select_obj (Peak, HotPeak, 3)
Vizion.area_center (HotPeak, PeakArea, PeakRow, PeakCol)
PA := PeakArea[1]
PR := PeakRow[1]
PC := PeakCol[1]
Vizion.select_uniform (Parts, Picture, Flat, 0)
Vizion.count_obj (Flat, FlatN)
Vizion.area_center (Flat, FlatArea, FlatRow, FlatCol)
FlatA := FlatArea[1]
Vizion.get_grayval (Picture, 14, 14, DimGray)
Vizion.get_grayval (Picture, 14, 24, LitGray)
Vizion.get_grayval (Picture, 30, 40, HotGray)
Vizion.dev_display (Picture)
"""


HDEVELOP_BATCH62_PROGRAM = """* 這個物件主要是哪個灰階
* 用法：數出現最多次的灰階，不是最亮的那一點。最亮用 bright_point。
* 兩種一樣多時，留較暗的那個。第二個比的是個數，不是灰階的大小。
* 只有一種灰階時，第二個的個數是 0，灰階仍回傳那一個。差是 0。
* select_mode 的門檻含在內，寫反了會自動對調。select_mixed 可省略，預設第二種至少 1 個像素。
Vizion.gen_image_const (Blank, 'byte', 80, 60)
Vizion.gen_rectangle1 (ALeft, 10, 10, 19, 19)
Vizion.paint_region (ALeft, Blank, Step1, 40, 'fill')
Vizion.gen_rectangle1 (ARight, 10, 20, 19, 29)
Vizion.paint_region (ARight, Step1, Step2, 180, 'fill')
Vizion.union2 (ALeft, ARight, BlockA)
Vizion.gen_rectangle1 (BlockB, 30, 10, 39, 19)
Vizion.paint_region (BlockB, Step2, Step3, 200, 'fill')
Vizion.gen_rectangle1 (BlockC, 30, 40, 39, 59)
Vizion.paint_region (BlockC, Step3, Step4, 20, 'fill')
Vizion.gen_rectangle1 (Hot, 30, 40, 31, 41)
Vizion.paint_region (Hot, Step4, Picture, 255, 'fill')
Vizion.concat_obj (BlockA, BlockB, Pair)
Vizion.concat_obj (Pair, BlockC, Parts)
* 左右各半時，主要是較暗的 40。整塊 200 的沒有第二種。小亮點只有 4 個，主要仍是 20
Vizion.gray_mode (Parts, Picture, Mode)
G1 := Mode[1]
G2 := Mode[2]
G3 := Mode[3]
Vizion.mode_count (Parts, Picture, ModeN)
C1 := ModeN[1]
C2 := ModeN[2]
C3 := ModeN[3]
Vizion.mode_ratio (Parts, Picture, ModeR)
R1 := ModeR[1]
R2 := ModeR[2]
R3 := ModeR[3]
Vizion.second_mode (Parts, Picture, Second, SecondN)
H1 := Second[1]
H2 := Second[2]
H3 := Second[3]
S1 := SecondN[1]
S2 := SecondN[2]
S3 := SecondN[3]
Vizion.mode_gap (Parts, Picture, Gap)
D1 := Gap[1]
D2 := Gap[2]
D3 := Gap[3]
Vizion.mode_region (Parts, Picture, Main)
Vizion.area_center (Main, MainArea, MainRow, MainCol)
MA1 := MainArea[1]
MR1 := MainRow[1]
MC1 := MainCol[1]
MA3 := MainArea[3]
MR3 := MainRow[3]
MC3 := MainCol[3]
* 主要灰階在 150 到 255 的，只有整塊 200 的那一個
Vizion.select_mode (Parts, Picture, Kept, 150, 255)
Vizion.count_obj (Kept, KeepN)
Vizion.area_center (Kept, KeepArea, KeepRow, KeepCol)
KeepA := KeepArea[1]
KeepR := KeepRow[1]
KeepC := KeepCol[1]
Vizion.select_mode (Parts, Picture, SwapKept, 255, 150)
Vizion.count_obj (SwapKept, SwapN)
* 第二種至少 10 個像素。小亮點只有 4 個，不算混在一起
Vizion.select_mixed (Parts, Picture, Mixed, 10)
Vizion.count_obj (Mixed, MixN)
Vizion.area_center (Mixed, MixArea, MixRow, MixCol)
MixA := MixArea[1]
MixR := MixRow[1]
MixC := MixCol[1]
Vizion.dev_display (Picture)
"""


HDEVELOP_BATCH63_PROGRAM = """* 有幾條橫的、幾條直的
* 用法：看物件的剪影分成幾條。同一列上左右分開仍算同一條橫的，中間空一列才算斷開。
* 直的也一樣，上下分開仍是同一條，空一行才斷開。實心方塊是 1 條、0 個空隙。
* 空隙個數是條數減 1，不是空隙有幾列。最長的一樣長時，留上面或左邊的那一條。
* select_stripes 要比正好幾條。方向是下拉選單。
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (All, 10, 10, 29, 29)
Vizion.gen_rectangle1 (Gap1, 13, 10, 16, 29)
Vizion.gen_rectangle1 (Gap2, 20, 10, 23, 29)
Vizion.union2 (Gap1, Gap2, Gaps)
Vizion.difference (All, Gaps, Bars)
Vizion.gen_rectangle1 (Tall, 40, 10, 54, 29)
Vizion.gen_rectangle1 (Hole1, 40, 14, 54, 16)
Vizion.gen_rectangle1 (Hole2, 40, 21, 54, 23)
Vizion.union2 (Hole1, Hole2, Holes)
Vizion.difference (Tall, Holes, Verts)
Vizion.gen_rectangle1 (Solid, 10, 40, 19, 49)
Vizion.concat_obj (Bars, Verts, Pair)
Vizion.concat_obj (Pair, Solid, Parts)
* 橫條三條，最高 6 列，中間空兩次。直條只有 1 條，寬 20
Vizion.row_runs (Parts, RowN)
RN1 := RowN[1]
RN2 := RowN[2]
RN3 := RowN[3]
Vizion.col_runs (Parts, ColN)
CN1 := ColN[1]
CN2 := ColN[2]
CN3 := ColN[3]
Vizion.longest_rows (Parts, RowL)
RL1 := RowL[1]
RL2 := RowL[2]
RL3 := RowL[3]
Vizion.longest_cols (Parts, ColL)
CL1 := ColL[1]
CL2 := ColL[2]
CL3 := ColL[3]
Vizion.row_gaps (Parts, RowG)
RG1 := RowG[1]
RG2 := RowG[2]
Vizion.col_gaps (Parts, ColG)
CG2 := ColG[2]
CG3 := ColG[3]
* 正好 3 條橫的是第一塊。正好 3 條直的是第二塊
Vizion.select_stripes (Parts, Row3, 'row', 3)
Vizion.count_obj (Row3, RowKeep)
Vizion.area_center (Row3, RowArea, RowRow, RowCol)
RowA := RowArea[1]
RowR := RowRow[1]
RowC := RowCol[1]
Vizion.select_stripes (Parts, Col3, 'column', 3)
Vizion.count_obj (Col3, ColKeep)
Vizion.area_center (Col3, ColArea, ColRow, ColCol)
ColA := ColArea[1]
ColR := ColRow[1]
ColC := ColCol[1]
* 最高的橫條在下面，120 像素。最寬的直條在右邊，90 像素
Vizion.stripe_region (Parts, RowStripe, 'row')
Vizion.select_obj (RowStripe, Thick, 1)
Vizion.area_center (Thick, ThickArea, ThickRow, ThickCol)
TA := ThickArea[1]
TR := ThickRow[1]
TC := ThickCol[1]
Vizion.stripe_region (Parts, ColStripe, 'column')
Vizion.select_obj (ColStripe, Wide, 2)
Vizion.area_center (Wide, WideArea, WideRow, WideCol)
WA := WideArea[1]
WR := WideRow[1]
WC := WideCol[1]
Vizion.paint_region (Thick, Image, Shown, 255, 'fill')
Vizion.dev_display (Shown)
"""


HDEVELOP_BATCH64_PROGRAM = """* 左右對不對稱
* 用法：以物件外框的中心翻面。1 是完全對稱，0 是翻過去完全對不上。
* column 左右對調，row 上下對調，both 轉 180 度。方向是下拉選單，可省略時預設左右。
* 中心用外框，不是像素平均。缺一塊但外框的角還在，中心不會跑掉。整張圖翻面請用 mirror_region。
* 至少 0.95 算夠對稱，可以改。對不上的像素數用 symmetry_miss，0 也留著。
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Block, 10, 10, 29, 29)
Vizion.gen_rectangle1 (Bite, 18, 26, 21, 29)
Vizion.difference (Block, Bite, Notch)
Vizion.gen_rectangle1 (Square, 10, 50, 19, 59)
Vizion.concat_obj (Notch, Square, Parts)
* 右邊缺一塊。左右只剩 368/384。上下仍是 1。方塊是 1
Vizion.symmetry_score (Parts, 'column', ColScore)
HC1 := ColScore[1]
HC2 := ColScore[2]
Vizion.symmetry_score (Parts, 'row', RowScore)
HR1 := RowScore[1]
HR2 := RowScore[2]
Vizion.symmetry_score (Parts, 'both', BothScore)
HB1 := BothScore[1]
Vizion.symmetry_miss (Parts, 'column', ColMiss)
MC1 := ColMiss[1]
MC2 := ColMiss[2]
Vizion.symmetry_miss (Parts, 'row', RowMiss)
MR1 := RowMiss[1]
Vizion.symmetry_center (Parts, CenterRow, CenterCol)
CR1 := CenterRow[1]
CC1 := CenterCol[1]
CR2 := CenterRow[2]
CC2 := CenterCol[2]
* 對不上的是左邊那 16 個像素，對面正好是缺口
Vizion.asymmetry_region (Parts, Asym, 'column')
Vizion.count_obj (Asym, AsymN)
Vizion.area_center (Asym, AsymArea, AsymRow, AsymCol)
AA := AsymArea[1]
AR := AsymRow[1]
AC := AsymCol[1]
Vizion.symmetry_core (Parts, Core, 'column')
Vizion.select_obj (Core, CoreOne, 1)
Vizion.area_center (CoreOne, CoreArea, CoreRow, CoreCol)
CA := CoreArea[1]
* 0.99 只留下方塊。0.96 留下缺塊的那個
Vizion.select_symmetric (Parts, Kept, 'column', 0.99)
Vizion.count_obj (Kept, KeepN)
Vizion.area_center (Kept, KeepArea, KeepRow, KeepCol)
KeepA := KeepArea[1]
KeepR := KeepRow[1]
KeepC := KeepCol[1]
Vizion.select_asymmetric (Parts, Odd, 'column', 0.96)
Vizion.count_obj (Odd, OddN)
Vizion.area_center (Odd, OddArea, OddRow, OddCol)
OddA := OddArea[1]
OddR := OddRow[1]
OddC := OddCol[1]
Vizion.reflect_region (Notch, Flipped, 'column')
Vizion.paint_region (Flipped, Image, Shown, 255, 'fill')
Vizion.get_grayval (Shown, 18, 29, FlipGray)
Vizion.get_grayval (Shown, 18, 10, GoneGray)
Vizion.dev_display (Shown)
"""


HDEVELOP_BATCH65_PROGRAM = """* 四個端點在哪
* 用法：找真的有像素的端點，不是外框的空角。平邊時，上和下留最左邊的點，左和右留最上面的點。
* top_edge 是最上的一整列，不是只有那一點。下、左、右也一樣。
* 外框左上角缺一個像素時，最上點會往右移，最左點會往下移。
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Bar, 10, 10, 19, 29)
Vizion.gen_rectangle1 (Box, 30, 40, 39, 49)
Vizion.gen_rectangle1 (Dot, 30, 40, 30, 40)
Vizion.difference (Box, Dot, Cut)
Vizion.concat_obj (Bar, Cut, Parts)
Vizion.top_point (Parts, TopRow, TopCol)
TR1 := TopRow[1]
TC1 := TopCol[1]
TR2 := TopRow[2]
TC2 := TopCol[2]
Vizion.bottom_point (Parts, BottomRow, BottomCol)
BR1 := BottomRow[1]
BC1 := BottomCol[1]
Vizion.left_point (Parts, LeftRow, LeftCol)
LR1 := LeftRow[1]
LC1 := LeftCol[1]
LR2 := LeftRow[2]
LC2 := LeftCol[2]
Vizion.right_point (Parts, RightRow, RightCol)
RR1 := RightRow[1]
RC1 := RightCol[1]
RR2 := RightRow[2]
RC2 := RightCol[2]
* 長條的上邊有 20 個像素。缺角之後，上邊只剩 9 個，中心行是 45
Vizion.top_edge (Parts, TopEdge)
Vizion.area_center (TopEdge, TopArea, TopEdgeRow, TopEdgeCol)
TA1 := TopArea[1]
TER1 := TopEdgeRow[1]
TEC1 := TopEdgeCol[1]
TA2 := TopArea[2]
TEC2 := TopEdgeCol[2]
Vizion.left_edge (Parts, LeftEdge)
Vizion.select_obj (LeftEdge, LeftCut, 2)
Vizion.area_center (LeftCut, LeftArea, LeftEdgeRow, LeftEdgeCol)
LA := LeftArea[1]
LER := LeftEdgeRow[1]
LEC := LeftEdgeCol[1]
Vizion.right_edge (Parts, RightEdge)
Vizion.select_obj (RightEdge, RightBar, 1)
Vizion.area_center (RightBar, RightArea, RightEdgeRow, RightEdgeCol)
RA := RightArea[1]
RER := RightEdgeRow[1]
REC := RightEdgeCol[1]
Vizion.select_obj (TopEdge, TopBar, 1)
Vizion.paint_region (TopBar, Image, Shown, 255, 'fill')
Vizion.get_grayval (Shown, 10, 19, EdgeGray)
Vizion.get_grayval (Shown, 11, 19, BelowGray)
Vizion.dev_display (Shown)
"""


HDEVELOP_BATCH66_PROGRAM = """* 洞有幾個、有多大
* 用法：洞是被物件完全包住的空白。缺口連到外面或影像邊緣，就不算洞，也不會被 fill_object 補上。
* hole_ratio 是洞的像素除以補滿之後的像素。0 是實心。
* hole_region 把每個洞拆開。同一個物件裡，較上面的洞先出來。largest_hole 只留最大的，一樣大留較上面的。
* fill_object 每個物件自己補，物件數不變。fill_up 會先把全部合成一個。
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Outer, 10, 10, 39, 39)
Vizion.gen_rectangle1 (Bite, 20, 20, 29, 29)
Vizion.difference (Outer, Bite, Frame)
Vizion.gen_rectangle1 (Solid, 42, 10, 51, 19)
Vizion.gen_rectangle1 (Big, 10, 50, 49, 79)
Vizion.gen_rectangle1 (HoleA, 15, 55, 24, 64)
Vizion.gen_rectangle1 (HoleB, 30, 60, 34, 64)
Vizion.union2 (HoleA, HoleB, Bites)
Vizion.difference (Big, Bites, Twin)
Vizion.gen_rectangle1 (Cup, 42, 40, 56, 59)
Vizion.gen_rectangle1 (Mouth, 46, 50, 52, 59)
Vizion.difference (Cup, Mouth, Open)
Vizion.concat_obj (Frame, Solid, Two)
Vizion.concat_obj (Two, Twin, Parts)
Vizion.hole_count (Parts, Counts)
C1 := Counts[1]
C2 := Counts[2]
C3 := Counts[3]
Vizion.hole_count (Open, OpenCount)
OC := OpenCount[1]
Vizion.hole_ratio (Parts, Ratios)
R1 := Ratios[1]
R2 := Ratios[2]
R3 := Ratios[3]
Vizion.hole_region (Parts, Holes)
Vizion.count_obj (Holes, HoleN)
Vizion.select_obj (Holes, FirstHole, 1)
Vizion.area_center (FirstHole, FirstArea, FirstRow, FirstCol)
FA := FirstArea[1]
FR := FirstRow[1]
FC := FirstCol[1]
Vizion.select_obj (Holes, SmallHole, 3)
Vizion.area_center (SmallHole, SmallArea, SmallRow, SmallCol)
SA := SmallArea[1]
SR := SmallRow[1]
SC := SmallCol[1]
Vizion.largest_hole (Parts, BigHole)
Vizion.count_obj (BigHole, BigN)
Vizion.area_center (BigHole, BigArea, BigRow, BigCol)
BA := BigArea[2]
BR := BigRow[2]
BC := BigCol[2]
Vizion.select_holed (Parts, Holed, 2)
Vizion.count_obj (Holed, HoledN)
Vizion.area_center (Holed, HoledArea, HoledRow, HoledCol)
HA := HoledArea[1]
HRow := HoledRow[1]
HCol := HoledCol[1]
Vizion.select_solid (Parts, Solids)
Vizion.count_obj (Solids, SolidN)
Vizion.area_center (Solids, SolidArea, SolidRow, SolidCol)
SolidA := SolidArea[1]
SolidR := SolidRow[1]
SolidC := SolidCol[1]
Vizion.select_hole_ratio (Parts, Hollow, 0.11)
Vizion.count_obj (Hollow, HollowN)
Vizion.area_center (Hollow, HollowArea, HollowRow, HollowCol)
HollowA := HollowArea[1]
HollowR := HollowRow[1]
Vizion.fill_object (Frame, Filled)
Vizion.area_center (Filled, FillArea, FillRow, FillCol)
FillA := FillArea[1]
FillR := FillRow[1]
Vizion.fill_object (Open, OpenFilled)
Vizion.area_center (Open, OpenArea, OpenRow, OpenCol)
Vizion.area_center (OpenFilled, OpenFillArea, OpenFillRow, OpenFillCol)
OA := OpenArea[1]
OFA := OpenFillArea[1]
Vizion.select_obj (BigHole, OneHole, 1)
Vizion.paint_region (OneHole, Image, Shown, 255, 'fill')
Vizion.get_grayval (Shown, 24, 24, HoleGray)
Vizion.get_grayval (Shown, 10, 10, WallGray)
Vizion.dev_display (Shown)
"""


HDEVELOP_BATCH67_PROGRAM = """* 有沒有跑出這個方框
* 用法：方框的邊算在裡面。上要小於等於下，左要小於等於右。
* window_ratio 是方框裡的像素除以物件自己的像素。1 是整個在裡面，0 是整個在外面。
* window_depth 是跑出去最遠的距離。chessboard 列和行取較遠的那個，city-block 才把兩個相加。完全在裡面是 0。
* overflow_region 只留跑出去的像素。完全在裡面的物件不會留下。clip_region 是把外面裁掉。
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (InBox, 22, 22, 31, 31)
Vizion.gen_rectangle1 (Cross, 10, 30, 29, 39)
Vizion.gen_rectangle1 (Far, 45, 45, 54, 54)
Vizion.concat_obj (InBox, Cross, Two)
Vizion.concat_obj (Two, Far, Parts)
Vizion.window_inside (Parts, 20, 20, 39, 39, InArea)
IA1 := InArea[1]
IA2 := InArea[2]
IA3 := InArea[3]
Vizion.window_outside (Parts, 20, 20, 39, 39, OutArea)
OA1 := OutArea[1]
OA2 := OutArea[2]
OA3 := OutArea[3]
Vizion.window_ratio (Parts, 20, 20, 39, 39, Ratio)
R1 := Ratio[1]
R2 := Ratio[2]
R3 := Ratio[3]
Vizion.window_depth (Parts, 20, 20, 39, 39, Depth)
D1 := Depth[1]
D2 := Depth[2]
D3 := Depth[3]
Vizion.window_depth (Parts, 20, 20, 39, 39, 'city-block', City)
CD2 := City[2]
CD3 := City[3]
Vizion.overflow_region (Parts, Overflow, 20, 20, 39, 39)
Vizion.count_obj (Overflow, OverN)
Vizion.area_center (Overflow, OverArea, OverRow, OverCol)
OVA1 := OverArea[1]
OVR1 := OverRow[1]
OVC1 := OverCol[1]
OVA2 := OverArea[2]
OVR2 := OverRow[2]
Vizion.select_inside (Parts, Inside, 20, 20, 39, 39)
Vizion.count_obj (Inside, InsideN)
Vizion.area_center (Inside, InsideArea, InsideRow, InsideCol)
InA := InsideArea[1]
InR := InsideRow[1]
InC := InsideCol[1]
Vizion.select_outside (Parts, Outside, 20, 20, 39, 39)
Vizion.count_obj (Outside, OutsideN)
Vizion.area_center (Outside, OutsideArea, OutsideRow, OutsideCol)
OutA := OutsideArea[1]
OutR := OutsideRow[1]
Vizion.select_crossing (Parts, Crossing, 20, 20, 39, 39)
Vizion.count_obj (Crossing, CrossN)
Vizion.area_center (Crossing, CrossArea, CrossRow, CrossCol)
CA := CrossArea[1]
CR := CrossRow[1]
CC := CrossCol[1]
Vizion.select_obj (Overflow, FirstOver, 1)
Vizion.paint_region (FirstOver, Image, Shown, 255, 'fill')
Vizion.get_grayval (Shown, 14, 34, OutGray)
Vizion.get_grayval (Shown, 24, 34, InGray)
Vizion.dev_display (Shown)
"""


HDEVELOP_BATCH68_PROGRAM = """* 長軸朝哪
* 用法：0 沿著行往右，正角度在畫面上順時針，直的是 90 度。範圍在 -90 度到 90 度。
* orientation_region 和 elliptic_axis 現在也用二次矩。橫條是 0，直條是 90 度。
* 長度是兩端像素中心的距離再加 1。20 像素長、5 像素厚的長條，長度是 20、厚度是 5，倍數是 4。
* 接近正方形時角度會落在 0。major_ratio 接近 1 就沒有長軸，select_long 的 2 會把它去掉。
* select_axis 的 Tolerance 是度。15 表示離水平或垂直 15 度以內，diagonal 是兩邊都超過。
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Wide, 10, 10, 14, 29)
Vizion.gen_rectangle1 (Tall, 10, 50, 29, 54)
Vizion.gen_rectangle1 (Box, 40, 10, 49, 19)
Vizion.gen_rectangle2 (Tilt, 30, 60, rad(30), 12, 3)
Vizion.concat_obj (Wide, Tall, Two)
Vizion.concat_obj (Two, Box, Three)
Vizion.concat_obj (Three, Tilt, Parts)
Vizion.major_phi (Parts, Phi)
P1 := Phi[1]
P2 := Phi[2]
P3 := Phi[3]
P4 := Phi[4]
Vizion.major_length (Parts, Length)
L1 := Length[1]
L2 := Length[2]
L3 := Length[3]
L4 := Length[4]
Vizion.minor_length (Parts, Thick)
M1 := Thick[1]
M4 := Thick[4]
Vizion.major_ratio (Parts, Ratio)
R1 := Ratio[1]
R3 := Ratio[3]
R4 := Ratio[4]
Vizion.axis_ends (Parts, EndRow1, EndCol1, EndRow2, EndCol2)
ER1 := EndRow1[1]
EC1 := EndCol1[1]
ER2 := EndRow2[1]
EC2 := EndCol2[1]
VR1 := EndRow1[2]
VC1 := EndCol1[2]
VR2 := EndRow2[2]
VC2 := EndCol2[2]
TR1 := EndRow1[4]
TC1 := EndCol1[4]
TR2 := EndRow2[4]
TC2 := EndCol2[4]
Vizion.select_axis (Parts, Level, 'horizontal', 15)
Vizion.count_obj (Level, LevelN)
Vizion.select_axis (Parts, Upright, 'vertical', 15)
Vizion.count_obj (Upright, UprightN)
Vizion.area_center (Upright, UprightArea, UprightRow, UprightCol)
UA := UprightArea[1]
UR := UprightRow[1]
UC := UprightCol[1]
Vizion.select_axis (Parts, Slant, 'diagonal', 15)
Vizion.count_obj (Slant, SlantN)
Vizion.area_center (Slant, SlantArea, SlantRow, SlantCol)
SA := SlantArea[1]
SR := SlantRow[1]
SC := SlantCol[1]
Vizion.select_axis (Parts, Loose, 'horizontal', 30)
Vizion.count_obj (Loose, LooseN)
Vizion.select_long (Parts, Long, 2)
Vizion.count_obj (Long, LongN)
Vizion.area_center (Long, LongArea, LongRow, LongCol)
LA := LongArea[3]
LR := LongRow[3]
Vizion.axis_region (Parts, Axis)
Vizion.select_obj (Axis, Spine, 1)
Vizion.area_center (Spine, SpineArea, SpineRow, SpineCol)
SpineA := SpineArea[1]
SpineR := SpineRow[1]
SpineC := SpineCol[1]
Vizion.paint_region (Spine, Image, Shown, 255, 'fill')
Vizion.get_grayval (Shown, 12, 19, AxisGray)
Vizion.get_grayval (Shown, 11, 19, OffGray)
Vizion.dev_display (Shown)
"""


HDEVELOP_BATCH69_PROGRAM = """* 把邊緣銳利、平坦留著
* 用法：edge_enhance 用高斯，起伏小於 Threshold 就不加。平坦處的小雜訊不會變銳。
* emphasize 用方框平均、沒有門檻，小亮點也會被放大。彩色只調亮度，四個控制都可以不寫，預設 1.2、1、2、0。
* edge_band 只加強兩個尺度中間的邊緣。edge_masked 只在梯度大的地方加。edge_preserve 把平坦的雜訊收掉，邊緣留著。
* Limit 大於 0 時，單邊最多只加或減這個量。edge_residual 和 enhance_map 以 128 為沒變，比 128 大是變亮。
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (All, 0, 0, 59, 79)
Vizion.paint_region (All, Image, Base, 100, 'fill')
Vizion.gen_rectangle1 (Right, 0, 40, 59, 79)
Vizion.paint_region (Right, Base, Step, 180, 'fill')
Vizion.set_grayval (Step, Spiked, 30, 10, 106)
Vizion.edge_enhance (Step, Sharp, 1.2, 1, 8, 0)
Vizion.edge_enhance (Spiked, Gated, 1.2, 1, 8, 0)
Vizion.edge_enhance (Spiked, Open, 1.2, 1, 0, 0)
Vizion.edge_enhance (Step, Capped, 1.2, 3, 8, 12)
Vizion.edge_band (Step, Band, 0.8, 2.4, 1, 2)
Vizion.edge_masked (Step, Masked, 1.2, 1, 30)
Vizion.edge_preserve (Spiked, Kept, 1.2, 30)
Vizion.edge_residual (Step, Residual, 1.2, 8)
Vizion.enhance_map (Step, Sharp, Map)
Vizion.enhance_change (Step, Sharp, MeanS, MaxS)
Vizion.enhance_change (Step, Band, MeanB, MaxB)
Vizion.enhance_hot (Step, Sharp, Hot, 10)
Vizion.count_obj (Hot, HotN)
Vizion.area_center (Hot, HotArea, HotRow, HotCol)
HotA := HotArea[1]
HotR := HotRow[1]
HotC := HotCol[1]
Vizion.get_grayval (Sharp, 30, 5, Flat)
Vizion.get_grayval (Sharp, 30, 39, DarkSharp)
Vizion.get_grayval (Sharp, 30, 40, BrightSharp)
Vizion.get_grayval (Gated, 30, 10, SpikeGate)
Vizion.get_grayval (Open, 30, 10, SpikeOpen)
Vizion.get_grayval (Capped, 30, 39, DarkCap)
Vizion.get_grayval (Capped, 30, 40, BrightCap)
Vizion.get_grayval (Band, 30, 5, BandFlat)
Vizion.get_grayval (Band, 30, 39, DarkBand)
Vizion.get_grayval (Band, 30, 40, BrightBand)
Vizion.get_grayval (Masked, 30, 5, MaskFlat)
Vizion.get_grayval (Masked, 30, 39, DarkMask)
Vizion.get_grayval (Kept, 30, 10, SpikeKept)
Vizion.get_grayval (Kept, 30, 40, EdgeKept)
Vizion.get_grayval (Residual, 30, 5, ResFlat)
Vizion.get_grayval (Residual, 30, 39, ResDark)
Vizion.get_grayval (Map, 30, 40, MapBright)
Vizion.emphasize (Spiked, Box, 7, 7, 1)
Vizion.get_grayval (Box, 30, 10, SpikeBox)
Vizion.dev_display (Sharp)
"""


HDEVELOP_BATCH70_PROGRAM = """* 程序算完再接住錯誤
* procedure 寫在最外層。呼叫順序跟算子一樣：圖像輸入、圖像輸出、控制輸入、控制輸出。
* 程序裡的變數是自己的。這裡 Temp 在程序裡被改成 1，外面的 Temp 仍是 9。
* try 和 catch 之間出錯會進 catch，Exception 放錯誤文字，整支程式不會中止。
* 沒有出錯就跳過 catch，直接做 endtry 後面。
procedure box (: Region : Row1, Col1, Row2, Col2 :)
    Vizion.gen_rectangle1 (Region, Row1, Col1, Row2, Col2)
endprocedure
box (Blob, 10, 10, 29, 29)
Vizion.area_center (Blob, Area, Row, Column)
Temp := 9
procedure twice (: : Value : Result)
    Temp := 1
    Result := Value * 2
endprocedure
twice (21, Answer)
try
    Bad := Missing + 1
catch (Exception)
    Safe := 1
endtry
Kept := Answer + Safe
Vizion.dev_display (Blob)
"""


HDEVELOP_BATCH71_PROGRAM = """* 和黃金影像差在哪
* Golden 是標準灰階 100。Scene 有一塊 180，還有一點只差 10 的雜訊。
* diff_region 的 MinDiff 是 20，差不到 20 的雜訊不算缺陷。
* diff_amount 的 Mean 是整張平均絕對差，Maximum 是差最多的那一點。
* 兩張圖要一樣大。彩色會先轉成灰階再比。
Vizion.gen_image_const (Base, 'byte', 80, 60)
Vizion.gen_image_proto (Base, Golden, 100)
Vizion.gen_image_proto (Base, Scene, 100)
Vizion.gen_rectangle1 (Block, 10, 10, 19, 19)
Vizion.paint_region (Block, Scene, Scene, 180, 'fill')
Vizion.set_grayval (Scene, Scene, 30, 30, 110)
Vizion.diff_region (Scene, Golden, Defects, 20)
Vizion.diff_amount (Scene, Golden, Mean, Maximum)
Vizion.area_center (Defects, Area, Row, Column)
Vizion.dev_display (Defects)
"""


HDEVELOP_BATCH72_PROGRAM = """* 印一張 QR 再讀回來
* gen_qrcode 做出白底黑點，四周留白，find_data_code_2d 才讀得到。
* 先 create_data_code_2d_model。QR 的 SymbolType 是 QR Code。
* Decoded 是字串 tuple，第一個用 [1]。找不到時字串是空的。
* PDF417 和 Aztec 這版不讀。Data Matrix 用 gen_datamatrix。
Vizion.gen_qrcode (Code, 'VISDEV')
Vizion.create_data_code_2d_model ('QR Code', '', '', Handle)
Vizion.find_data_code_2d (Code, Symbol, Handle, '', '', Ids, Decoded)
Text := Decoded[1]
Vizion.dev_display (Code)
"""


HDEVELOP_BATCH73_PROGRAM = """* 暗的刮痕在哪、朝哪
* 細線用高斯導數的 Hessian 找脊線，再沿法線收到次像素中心。這就是改良後的 lines_gauss。
* dark 找比周圍暗的線，light 找亮線。平坦、大方塊的邊、抓錯明暗，都不會留下。
* ridge_response 只有中心有灰階，那個數字就是 lines_gauss 的強度門檻。
* ridge_region 是一像素寬。Phi 為 0 朝右，垂直是 1.5708。Length 是點距，不是面積。
Vizion.gen_image_const (Base, 'byte', 80, 60)
Vizion.gen_image_proto (Base, Sheet, 180)
Vizion.gen_region_line (Mark, 10, 40, 49, 40)
Vizion.paint_region (Mark, Sheet, Scene, 40, 'fill')
Vizion.ridge_region (Scene, Scratch, 1, 8, 'dark')
Vizion.area_center (Scratch, Area, Row, Column)
Vizion.ridge_response (Scene, Response, 1, 'dark')
Vizion.get_grayval (Response, 30, 40, Peak)
Vizion.get_grayval (Response, 30, 39, Side)
Vizion.ridge_stats (Scene, 1, 8, 'dark', Count, Length, Phi, CenterRow, CenterCol)
Vizion.ridge_stats (Scene, 1, 8, 'light', WrongCount, WrongLength, WrongPhi, WrongRow, WrongCol)
Vizion.ridge_stats (Sheet, 1, 8, 'dark', FlatCount, FlatLength, FlatPhi, FlatRow, FlatCol)
Vizion.gen_image_proto (Base, Dark, 0)
Vizion.gen_rectangle1 (Bar, 29, 8, 31, 72)
Vizion.paint_region (Bar, Dark, Bright, 255, 'fill')
Vizion.ridge_region (Bright, LightLine, 1.2, 5, 'light')
Vizion.area_center (LightLine, LightArea, LightRow, LightCol)
Vizion.lines_gauss (Bright, Lines, 1.2, 5, 20, 'light')
Vizion.length_xld (Lines, LineLen)
Vizion.tuple_max (LineLen, Longest)
Vizion.gen_rectangle1 (Block, 10, 10, 49, 49)
Vizion.paint_region (Block, Dark, Wide, 255, 'fill')
Vizion.ridge_stats (Wide, 1.2, 5, 'light', BlockCount, BlockLength, BlockPhi, BlockRow, BlockCol)
Vizion.dev_display (Scratch)
"""


HDEVELOP_BATCH74_PROGRAM = """* 邊緣在兩個像素中間
* edges_sub_pix 用高斯導數的梯度，沿法線做拋物線內插，不再描整數像素的外框。
* 左半灰階 40、右半 200，真正的邊界在行 39.5。edge_region 落在第 40 行那個整數像素。
* edge_response 的灰階就是門檻。黑白階躍大約 58，差 10 個灰階的方塊不會超過 20。
* Phi 為 0 朝右，垂直是 1.5708。Length 是次像素點距。
Vizion.gen_image_const (Base, 'byte', 80, 60)
Vizion.gen_image_proto (Base, Left, 40)
Vizion.gen_rectangle1 (Right, 0, 40, 59, 79)
Vizion.paint_region (Right, Left, Step, 200, 'fill')
Vizion.edge_region (Step, Edge, 1, 20)
Vizion.area_center (Edge, Area, Row, Column)
Vizion.edge_response (Step, Response, 1)
Vizion.get_grayval (Response, 30, 40, Peak)
Vizion.get_grayval (Response, 30, 39, Side)
Vizion.edge_stats (Step, 1, 20, Count, Length, Phi, CenterRow, CenterCol)
Vizion.edges_sub_pix (Step, Lines, 'canny', 1, 20, 40)
Vizion.length_xld (Lines, LineLen)
Vizion.tuple_max (LineLen, Longest)
Vizion.gen_image_proto (Base, Top, 40)
Vizion.gen_rectangle1 (Bottom, 30, 0, 59, 79)
Vizion.paint_region (Bottom, Top, Band, 200, 'fill')
Vizion.edge_stats (Band, 1, 20, AcrossCount, AcrossLength, AcrossPhi, AcrossRow, AcrossCol)
Vizion.gen_image_proto (Base, Plain, 100)
Vizion.gen_rectangle1 (Box, 20, 20, 39, 39)
Vizion.paint_region (Box, Plain, Soft, 110, 'fill')
Vizion.edge_stats (Soft, 1, 20, SoftCount, SoftLength, SoftPhi, SoftRow, SoftCol)
Vizion.edge_stats (Left, 1, 20, FlatCount, FlatLength, FlatPhi, FlatRow, FlatCol)
Vizion.dev_display (Edge)
"""


HDEVELOP_BATCH75_PROGRAM = """* 平移不到一個像素
* Vizion.phase_correlation (第一張, 第二張, 列, 行, 峰值) 回傳把第一張對到第二張要加的位移，可以是小數。
* 兩張完全一樣時，列和行都是 0，峰值接近 1。整數平移會收成整數，這裡是列 3、行 4。
* 不到一個像素時，用峰值旁邊兩個格子的比例算出來。影像外補黑，所以 1.5 會落在 1.50 附近，誤差在 0.02 以內。
* Vizion.align_translation (要拉的影像, 基準, 拉回後, 列, 行, 峰值) 量完就把影像平移回去。中間灰階 200 會回到原位。
Vizion.gen_image_const (Base, 'byte', 96, 80)
Vizion.gen_image_proto (Base, Sheet, 30)
Vizion.gen_rectangle1 (A, 18, 20, 33, 43)
Vizion.paint_region (A, Sheet, P1, 200, 'fill')
Vizion.gen_rectangle1 (B, 36, 40, 57, 69)
Vizion.paint_region (B, P1, P2, 90, 'fill')
Vizion.gen_rectangle1 (C, 22, 60, 29, 79)
Vizion.paint_region (C, P2, P3, 240, 'fill')
Vizion.gen_rectangle1 (D, 48, 12, 69, 27)
Vizion.paint_region (D, P3, Image, 150, 'fill')
Vizion.hom_mat2d_identity (Id)
Vizion.hom_mat2d_translate (Id, 1.5, 0, Down)
Vizion.affine_trans_image (Image, Moved, Down, 'bicubic', 'false')
Vizion.phase_correlation (Image, Moved, Row, Column, Score)
Vizion.phase_correlation (Image, Image, SameRow, SameCol, SameScore)
Vizion.hom_mat2d_translate (Id, 0, 1.5, Right)
Vizion.affine_trans_image (Image, Side, Right, 'bicubic', 'false')
Vizion.phase_correlation (Image, Side, SideRow, SideCol, SideScore)
Vizion.hom_mat2d_translate (Id, 3, 4, Far)
Vizion.affine_trans_image (Image, FarImage, Far, 'bicubic', 'false')
Vizion.phase_correlation (Image, FarImage, FarRow, FarCol, FarScore)
Vizion.align_translation (Moved, Image, Back, BackRow, BackCol, BackScore)
Vizion.get_grayval (Image, 25, 30, Orig)
Vizion.get_grayval (Back, 25, 30, Restored)
Vizion.dev_display (Back)
"""


HDEVELOP_BATCH76_PROGRAM = """* 有一個飛點也不歪
* fit_line_contour_xld 先擬合，再丟掉離直線超過約 1 像素的點，重擬到方向穩定。
* 線段只覆蓋留下來的點，飛點不會把線拉長，也不會把整條線抬起來。
* 四個點在列 10、行 0、10、20、30。中間飛點在列 18、行 15。直線仍是列 10，行從 0 到 30。
* 斜線上的飛點也一樣：點在 (0,0) 到 (30,30)，飛點 (15,28) 被丟掉，端點仍是這兩個角。
Vizion.gen_contour_polygon_xld (Edge, [10, 10, 10, 10, 18], [0, 10, 20, 30, 15])
Vizion.fit_line_contour_xld (Edge, Line, Row1, Col1, Row2, Col2)
Vizion.gen_contour_polygon_xld (Diag, [0, 10, 20, 30, 15], [0, 10, 20, 30, 28])
Vizion.fit_line_contour_xld (Diag, Line2, D1, C1, D2, C2)
Vizion.dev_display (Line)
"""


HDEVELOP_BATCH77_PROGRAM = """* 圓弧上有一個飛點
* fit_circle_contour_xld 用 Pratt 擬合。短圓弧不會像舊的代數法把半徑縮很小。
* 離圓超過約 1 像素的點會丟掉，再用剩下的點擬一次。點都在圓上時，圓心和半徑不變。
* 這九個點在圓心列 80、行 90、半徑 40 的圓弧上，正中間那個被往外推到行 138。
* 飛點丟掉之後，圓心仍是列 80、行 90，半徑仍是 40。
Vizion.gen_contour_polygon_xld (Arc, [60, 64.692663, 69.647238, 74.778952, 80, 85.221048, 90.352762, 95.307337, 100], [124.641016, 126.955181, 128.637033, 129.657794, 138, 129.657794, 128.637033, 126.955181, 124.641016])
Vizion.fit_circle_contour_xld (Arc, Circle, Row, Column, Radius)
Vizion.dev_display (Circle)
"""


HDEVELOP_BATCH78_PROGRAM = """* 邊界在半個像素
* gen_contour_region_xld 的 center 走在區域和外面之間的邊上，座標是 .5。
* border 仍停在最外圈的整數像素。方塊列 10 到 29、行 10 到 29 時，center 外框是 9.5 和 29.5，border 是 10 和 29。
* 有孔的框用 center 只畫外圈，不畫孔。外框在 9.5 和 40.5。
* 圓的像素中心在半徑裡面。center 再擬合圓會更靠近真正的半徑，border 會偏小。
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Box, 10, 10, 29, 29)
Vizion.gen_contour_region_xld (Box, Center, 'center')
Vizion.gen_contour_region_xld (Box, Border, 'border')
Vizion.count_obj (Center, CenterCount)
Vizion.get_contour_xld (Center, Rows, Cols)
Vizion.tuple_min (Rows, RowMin)
Vizion.tuple_max (Rows, RowMax)
Vizion.tuple_min (Cols, ColMin)
Vizion.tuple_max (Cols, ColMax)
Vizion.get_contour_xld (Border, BRows, BCols)
Vizion.tuple_min (BCols, BorderMin)
Vizion.tuple_max (BCols, BorderMax)
Vizion.gen_rectangle1 (Outer, 10, 10, 40, 40)
Vizion.gen_rectangle1 (Hole, 20, 20, 30, 30)
Vizion.difference (Outer, Hole, Frame)
Vizion.gen_contour_region_xld (Frame, FrameCenter, 'center')
Vizion.count_obj (FrameCenter, FrameCount)
Vizion.select_obj (FrameCenter, OuterLine, 1)
Vizion.get_contour_xld (OuterLine, FRows, FCols)
Vizion.tuple_min (FCols, FrameMin)
Vizion.tuple_max (FCols, FrameMax)
Vizion.gen_circle (Disk, 30, 40, 15)
Vizion.gen_contour_region_xld (Disk, DiskCenter, 'center')
Vizion.gen_contour_region_xld (Disk, DiskBorder, 'border')
Vizion.fit_circle_contour_xld (DiskCenter, CCircle, CRow, CCol, CRad)
Vizion.fit_circle_contour_xld (DiskBorder, BCircle, BRow, BCol, BRad)
Vizion.dev_display (Center)
"""


HDEVELOP_BATCH79_PROGRAM = """* 高斯權重不跟飛點
* regress_contours_xld 的 gauss 依每個點離直線的距離做高斯加權。Iterations 填 0 會做 4 次。
* no 每個點一樣重。四個點在列 10、行 0 到 30，飛點在列 22、行 15。
* no 整條被抬到列 12.4。gauss 仍是列 10，行從 0 到 30。
* 點本來就在直線上時，gauss 和 no 一樣，還是列 10。
Vizion.gen_contour_polygon_xld (Spike, [10, 10, 10, 10, 22], [0, 10, 20, 30, 15])
Vizion.regress_contours_xld (Spike, Fit, 'gauss', 0)
Vizion.get_contour_xld (Fit, Rows, Cols)
Vizion.tuple_min (Rows, RowMin)
Vizion.tuple_max (Rows, RowMax)
Vizion.tuple_min (Cols, ColMin)
Vizion.tuple_max (Cols, ColMax)
Vizion.regress_contours_xld (Spike, Plain, 'no', 0)
Vizion.get_contour_xld (Plain, PRows, PCols)
Vizion.tuple_mean (PRows, PlainMean)
Vizion.gen_contour_polygon_xld (Clean, [10, 10, 10], [0, 15, 30])
Vizion.regress_contours_xld (Clean, Same, 'gauss', 0)
Vizion.get_contour_xld (Same, CRows, CCols)
Vizion.tuple_min (CRows, CleanMin)
Vizion.tuple_max (CRows, CleanMax)
Vizion.dev_display (Fit)
"""


HDEVELOP_BATCH80_PROGRAM = """* 斜率 1 的導數是 129
* derivate_gauss 用可分離的高斯導數，不是先模糊再做 Sobel。
* 有號導數以 128 為 0。影像往右每一行灰階加 1 時，x 是 129，y 是 128。
* 往下每一列加 1 時，y 是 129，x 是 128。二階導數在直線斜坡上是 128。
* gradient 是振幅，不加 128，所以這個斜坡大約是 1。
Vizion.gen_image_gray_ramp (Ramp, 0, 1, 0, 0, 0, 64, 48)
Vizion.derivate_gauss (Ramp, Dx, 1, 'x')
Vizion.derivate_gauss (Ramp, Dy, 1, 'y')
Vizion.derivate_gauss (Ramp, Dxx, 1, 'xx')
Vizion.derivate_gauss (Ramp, Grad, 1, 'gradient')
Vizion.get_grayval (Dx, 24, 32, Deriv)
Vizion.get_grayval (Dy, 24, 32, ZeroY)
Vizion.get_grayval (Dxx, 24, 32, Deriv2)
Vizion.get_grayval (Grad, 24, 32, Amp)
Vizion.gen_image_gray_ramp (Down, 1, 0, 0, 0, 0, 64, 48)
Vizion.derivate_gauss (Down, Vy, 1, 'y')
Vizion.derivate_gauss (Down, Vx, 1, 'x')
Vizion.get_grayval (Vy, 24, 32, DerivY)
Vizion.get_grayval (Vx, 24, 32, ZeroX)
Vizion.dev_display (Dx)
"""


HDEVELOP_BATCH81_PROGRAM = """* 半個像素不是兩點的平均
* get_grayval_interpolated 的 bicubic 用周圍 16 個像素的三次內插。bilinear 只看 4 個。
* 量測矩形和圓弧選 bicubic 時，剖面用同一套。整數座標讀到像素本身。
* 行 10、列 10 的灰階是 200，旁邊是 0。讀列 10、行 10.5：bilinear 是 100，bicubic 是 112.5。
* 三次核在半像素的權重是 0.5625，200 乘 0.5625 就是 112.5。nearest 的 .5 會進到下一個像素，這裡是 0。
Vizion.gen_image_const (Base, 'byte', 32, 32)
Vizion.set_grayval (Base, Image, 10, 10, 200)
Vizion.get_grayval_interpolated (Image, 10, 10, 'bicubic', OnPixel)
Vizion.get_grayval_interpolated (Image, 10, 10.5, 'bicubic', Cubic)
Vizion.get_grayval_interpolated (Image, 10, 10.5, 'bilinear', Linear)
Vizion.get_grayval_interpolated (Image, 10, 10.5, 'nearest', Near)
Vizion.dev_display (Image)
"""


HDEVELOP_BATCH82_PROGRAM = """* 最小二乘把形狀對準
* find_shape_model 的 interpolation 只在分數曲面上做拋物線。least_squares 再沿邊緣法線修列、行和角度。
* 方塊的教導中心是列 44.5、行 59.5。整張圖往右移 0.4 像素後，真的中心在行 59.9。
* interpolation 會偏到大約 60.27。least_squares 落在 59.85，比較靠近 59.9。角度仍是 0。
Vizion.gen_image_const (Base, 'byte', 120, 90)
Vizion.gen_image_proto (Base, Sheet, 20)
Vizion.gen_rectangle1 (Box, 25, 35, 64, 84)
Vizion.paint_region (Box, Sheet, Image, 220, 'fill')
Vizion.teach_shape_model (Image, 20, 30, 69, 89, 1, 0, 0, 0, 'use_polarity', 30, 10, ModelID)
Vizion.hom_mat2d_identity (Id)
Vizion.hom_mat2d_translate (Id, 0, 0.4, Shift)
Vizion.affine_trans_image (Image, Moved, Shift, 'bicubic', 'false')
Vizion.find_shape_model (Moved, ModelID, 0, 0, 0.5, 1, 0.5, 'interpolation', 0, 0.8, IRow, ICol, IAng, IScore)
Vizion.find_shape_model (Moved, ModelID, 0, 0, 0.5, 1, 0.5, 'least_squares', 0, 0.8, LRow, LCol, LAng, LScore)
Vizion.dev_display (Moved)
"""


HDEVELOP_BATCH83_PROGRAM = """* 最小二乘的列更靠近 39.25
* interpolation 在整數峰周圍的分數上做 Keys 三次內插。平坦的峰留在整數，不再用三點拋物線。
* least_squares 仍從整數峰用 Lucas-Kanade，沿灰階梯度修列和行。角度維持搜尋結果。
* 教導框中心是列 39.5、行 47.5。往上 0.25、往右 0.35 之後，真的中心在列 39.25、行 47.85。
* interpolation 的列是 39.23。least_squares 的列更靠近 39.25。
Vizion.gen_image_const (Base, 'byte', 96, 80)
Vizion.gen_image_proto (Base, Sheet, 30)
Vizion.gen_rectangle1 (A, 18, 20, 33, 43)
Vizion.paint_region (A, Sheet, P1, 200, 'fill')
Vizion.gen_rectangle1 (B, 36, 40, 57, 69)
Vizion.paint_region (B, P1, P2, 90, 'fill')
Vizion.gen_rectangle1 (C, 22, 60, 29, 79)
Vizion.paint_region (C, P2, P3, 240, 'fill')
Vizion.gen_rectangle1 (D, 48, 12, 69, 27)
Vizion.paint_region (D, P3, Sharp, 150, 'fill')
Vizion.gauss_image (Sharp, Image, 5)
Vizion.teach_ncc_model (Image, 10, 10, 69, 85, 1, 0, 0, 0, 'use_polarity', ModelID)
Vizion.hom_mat2d_identity (Id)
Vizion.hom_mat2d_translate (Id, -0.25, 0.35, Shift)
Vizion.affine_trans_image (Image, Moved, Shift, 'bicubic', 'false')
Vizion.find_ncc_model (Moved, ModelID, 0, 0, 0.5, 1, 0.5, 'interpolation', 1, IRow, ICol, IAng, IScore)
Vizion.find_ncc_model (Moved, ModelID, 0, 0, 0.5, 1, 0.5, 'least_squares', 1, LRow, LCol, LAng, LScore)
Vizion.dev_display (Moved)
"""


HDEVELOP_BATCH84_PROGRAM = """* 卡尺把尖刺平均掉
* 直線計量的 MeasureLength2 是卡尺沿邊緣的半寬。0 只看中心一條線。大於 0 會把這個寬度裡的灰階平均，再找邊緣。
* 亮區從列 40 開始，真正的邊緣在 39.5。每支卡尺正中間那一行被往下拉到 45.5。
* 半寬 0 跟著尖刺走到 45.5。半寬 1 把尖刺平均掉，兩端都回到 39.5。圓和橢圓的卡尺同樣沿切向平均。
Vizion.gen_image_const (Base, 'byte', 140, 90)
Vizion.gen_image_proto (Base, Sheet, 30)
Vizion.gen_rectangle1 (Bar, 40, 0, 89, 139)
Vizion.paint_region (Bar, Sheet, Image, 190, 'fill')
Cols := [26, 29, 33, 36, 40, 43, 47, 51, 54, 58, 61, 65, 68, 72, 75, 79, 82, 86, 89, 93, 97, 100, 104, 107, 111, 114]
for Index := 1 to |Cols| by 1
Vizion.gen_rectangle1 (Spike, 40, Cols[Index], 45, Cols[Index])
Vizion.paint_region (Spike, Image, Image, 30, 'fill')
endfor
Vizion.create_metrology_model (Model)
Vizion.set_metrology_model_image_size (Model, 140, 90)
Vizion.add_metrology_object_line_measure (Model, 40, 20, 40, 120, 14, 0, 1, 10, Index0)
Vizion.add_metrology_object_line_measure (Model, 40, 20, 40, 120, 14, 1, 1, 10, Index1)
Vizion.apply_metrology_model (Image, Model)
Vizion.get_metrology_object_result (Model, 0, 'all', 'result_type', 'row_begin', ThinBegin)
Vizion.get_metrology_object_result (Model, 0, 'all', 'result_type', 'row_end', ThinEnd)
Vizion.get_metrology_object_result (Model, 1, 'all', 'result_type', 'row_begin', WideBegin)
Vizion.get_metrology_object_result (Model, 1, 'all', 'result_type', 'row_end', WideEnd)
Vizion.dev_display (Image)
"""


HDEVELOP_BATCH85_PROGRAM = """* 角點修進像素裡面
* points_foerstner 先在結構張量上找整數峰，再用周圍的高斯導數解加權最小二乘，把角點修到次像素。
* 方塊左上角的整數峰停在列 25、行 37。次像素把它拉到 24.13、36.13。
* 整張圖往下 0.4、往右 0.35 之後，整數峰還是 25、37。次像素跟到 24.41、36.39，比整數峰更靠近平移後的角。
Vizion.gen_image_const (Base, 'byte', 120, 90)
Vizion.gen_image_proto (Base, Sheet, 20)
Vizion.gen_rectangle1 (Box, 24, 36, 64, 84)
Vizion.paint_region (Box, Sheet, Image, 220, 'fill')
Vizion.points_foerstner (Image, 1.2, 0.04, Row, Col)
StillRow := 0
StillCol := 0
for Index := 1 to |Row| by 1
if (Row[Index] < 40 and Col[Index] < 50)
StillRow := Row[Index]
StillCol := Col[Index]
endif
endfor
Vizion.hom_mat2d_identity (Id)
Vizion.hom_mat2d_translate (Id, 0.4, 0.35, Shift)
Vizion.affine_trans_image (Image, Moved, Shift, 'bicubic', 'false')
Vizion.points_foerstner (Moved, 1.2, 0.04, MRow, MCol)
CornerRow := 0
CornerCol := 0
for Index := 1 to |MRow| by 1
if (MRow[Index] < 40 and MCol[Index] < 50)
CornerRow := MRow[Index]
CornerCol := MCol[Index]
endif
endfor
Vizion.dev_display (Moved)
"""


HDEVELOP_BATCH86_PROGRAM = """* 仿射不跟飛點走
* vector_to_hom_mat2d 先做最小平方，再丟掉殘差明顯偏大的點，重新擬合。剛好三個點時不會丟。相似變換和剛體也一樣。
* 四個角從列、行 (0,0)、(0,40)、(40,0)、(40,40) 平移到列加 2、行加 5。中心點本該到 (22,25)，卻被放到 (80,90)。
* 剔掉這個飛點之後，(0,0) 落到列 2、行 5，(10,15) 落到列 12、行 20。
Vizion.vector_to_hom_mat2d ([0, 0, 40, 40, 20], [0, 40, 0, 40, 20], [2, 2, 42, 42, 80], [5, 45, 5, 45, 90], Affine)
Vizion.affine_trans_pixel (Affine, 0, 0, Row0, Col0)
Vizion.affine_trans_pixel (Affine, 10, 15, Row1, Col1)
"""


HDEVELOP_BATCH87_PROGRAM = """* 投影不跟飛點走
* hom_vector_to_proj_hom_mat2d 從四點子集裡挑中位誤差最小的投影，再只用殘差小的點重擬合。剛好四個點時不會丟。
* 四個角從列、行 (0,0)、(0,40)、(40,0)、(40,40) 平移到列加 3、行加 4。中心點本該到 (23,24)，卻被放到 (90,80)。
* 剔掉這個飛點之後，(0,0) 落到列 3、行 4，(10,10) 落到列 13、行 14。
Vizion.hom_vector_to_proj_hom_mat2d ([0, 0, 40, 40, 20], [0, 40, 0, 40, 20], [1, 1, 1, 1, 1], [3, 3, 43, 43, 90], [4, 44, 4, 44, 80], [1, 1, 1, 1, 1], Hom)
Vizion.projective_trans_pixel (Hom, 0, 0, Row0, Col0)
Vizion.projective_trans_pixel (Hom, 10, 10, Row1, Col1)
"""


HDEVELOP_BATCH88_PROGRAM = """* 橢圓弧上有一個飛點
* fit_ellipse_contour_xld 在點很多、而且有一個點離開弧線時，會先找中位誤差小的橢圓，再丟掉那個飛點重擬合。剛好五個點時不會丟。
* 橢圓中心在列 80、行 90，長軸 40、短軸 18，長軸角度 0.4。弧只蓋住中間約 69 度，正中間那個點往右移 8 像素。
* 重擬合後中心仍是 80、90，兩軸仍是 40 和 18。
Vizion.gen_ellipse_contour_xld (Arc, 80, 90, 0.4, 40, 18, -0.6, 0.6, 'positive', 1.5)
Vizion.get_contour_xld (Arc, Row, Col)
Vizion.tuple_length (Row, Count)
Mid := round(Count / 2) - 1
Vizion.tuple_replace (Col, Col[Mid + 1] + 8, Mid, Col2)
Vizion.gen_contour_polygon_xld (Bumped, Row, Col2)
Vizion.fit_ellipse_contour_xld (Bumped, Ellipse, BRow, BCol, BPhi, BR1, BR2)
Vizion.dev_display (Bumped)
"""


HDEVELOP_BATCH89_PROGRAM = """* 矩形邊上有一個飛點
* fit_rectangle2_contour_xld 在點很多、而且有一個點離開矩形邊時，會先找中位誤差小、總誤差也小的矩形，再丟掉那個飛點重擬合。點很少時仍是最小外接矩形。
* 矩形中心在列 50、行 80，半長 36、半寬 18。上邊正中間那個點被拉到列 18。
* 最小外接矩形跟著飛點走到中心列 43、半寬 25。擬合仍是中心列 50、行 80，半長 36、半寬 18。
Rows := []
Cols := []
for Step := 0 to 12 by 1
Vizion.tuple_concat (Rows, 32, Rows)
Vizion.tuple_concat (Cols, 44 + Step * 6, Cols)
endfor
for Step := 1 to 6 by 1
Vizion.tuple_concat (Rows, 32 + Step * 6, Rows)
Vizion.tuple_concat (Cols, 116, Cols)
endfor
for Step := 1 to 12 by 1
Vizion.tuple_concat (Rows, 68, Rows)
Vizion.tuple_concat (Cols, 116 - Step * 6, Cols)
endfor
for Step := 1 to 5 by 1
Vizion.tuple_concat (Rows, 68 - Step * 6, Rows)
Vizion.tuple_concat (Cols, 44, Cols)
endfor
Vizion.tuple_replace (Rows, 18, 6, Rows2)
Vizion.gen_contour_polygon_xld (Bumped, Rows2, Cols)
Vizion.fit_rectangle2_contour_xld (Bumped, Rect, FRow, FCol, FPhi, FL1, FL2)
Vizion.smallest_rectangle2_xld (Bumped, SRow, SCol, SPhi, SL1, SL2)
Vizion.dev_display (Bumped)
"""


HDEVELOP_BATCH90_PROGRAM = """* 尖角不會掀起平面
* fit_plane_region_inlier 先從三點子集裡挑中位誤差小的平面，再留下差距不超過 MaxDist 的像素重擬合。每個點一開始就在門檻裡時，直接用全部像素。
* 3×3 的高度都是 0，左上角被設成 40。一次擬合會被掀成列、行斜率都是 -6.66667，常數 51.11111。門檻 2 時，這張斜面讓九個點全都超過門檻。
* 改從平的三點出發之後，平面回到 0、0、0。
Vizion.gen_image_const (Sheet, 'byte', 12, 10)
Vizion.gen_rectangle1 (Box, 2, 3, 4, 5)
Vizion.set_grayval (Sheet, Peak, 2, 3, 40)
Vizion.fit_plane_region (Box, Peak, RawA, RawB, RawG)
Vizion.fit_plane_region_inlier (Box, Peak, 2, Alpha, Beta, Gamma)
Vizion.dev_display (Peak)
"""


HDEVELOP_BATCH91_PROGRAM = """* 點雲平面不跟飛點走
* fit_plane_object_model_3d 在點比較多、而且有一個點離平面很遠時，會先從三點子集裡挑中位誤差小的平面，再丟掉那個飛點重擬合。剛好三個點，或每個點一開始就在 1 以內時，不會丟。
* 九個點在 z = 0 的 3×3 格子上，另一個點在 (0, 0, 8)。一次擬合的法向量大約是 (0.69, 0.69, 0.19)，飛點自己的距離只有 0.14，格子角落反而更遠。
* 丟掉飛點之後法向量是 (0, 0, 1)，平面常數是 0。
Vizion.gen_object_model_3d_from_points ([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], [0, 0, 0, 1, 1, 1, 2, 2, 2, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 8], Cloud)
Vizion.fit_plane_object_model_3d (Cloud, Nx, Ny, Nz, Dist)
"""


HDEVELOP_BATCH92_PROGRAM = """* 主軸不跟旁邊的點走
* principal_axis_object_model_3d 在點比較多、而且有一個點離主軸很遠時，會先從兩點子集裡挑中位誤差小的方向，再丟掉那個點重算。剛好兩個點，或每個點一開始就在 1 以內時，不會丟。整團點都還在門檻裡時，仍用全部點的主軸。
* 五個點在 x 軸上，從 0 到 8。另一個點在 (4, 12, 0)。一次擬合的主軸被轉成 y，那個點自己的垂直距離是 0，軸上的點反而離 2 到 4。
* 丟掉旁邊那個點之後，主軸回到 (1, 0, 0)。
Vizion.gen_object_model_3d_from_points ([0, 2, 4, 6, 8, 4], [0, 0, 0, 0, 0, 12], 0, Shaft)
Vizion.principal_axis_object_model_3d (Shaft, Ax, Ay, Az)
"""


HDEVELOP_BATCH93_PROGRAM = """* 直線法線不跟尖角歪
* gen_parallel_contour_xld 的 regression_normal 先把整條輪廓擬合成直線，再沿那條直線的右側法線平移。contour_normal 仍沿每個點自己的法線。
* 水平線在列 40，行從 10 到 90。正中間那個點被拉到列 52。往右側平移 4。
* contour_normal 把尖角兩旁的點推到行 28.85061 和 71.14939。regression_normal 讓這兩點留在行 30 和 70，列都是 44。
Vizion.gen_contour_polygon_xld (Edge, [40, 40, 52, 40, 40], [10, 30, 50, 70, 90])
Vizion.gen_parallel_contour_xld (Edge, Local, 'contour_normal', 4)
Vizion.get_contour_xld (Local, LRow, LCol)
Vizion.gen_parallel_contour_xld (Edge, Fitted, 'regression_normal', 4)
Vizion.get_contour_xld (Fitted, RRow, RCol)
Vizion.dev_display (Fitted)
"""


HDEVELOP_BATCH94_PROGRAM = """* Harris 角點修進像素裡面
* points_harris 用高斯導數的結構張量算 Harris 回應，不是方框加 Sobel。SubPixel 為 true 時，在整數峰上解加權最小二乘。
* 方塊左上角的整數峰停在列 25、行 37。次像素把它拉到 24.13、36.13。
* 整張圖往下 0.4、往右 0.35 之後，整數峰還是 25、37。次像素跟到 24.41、36.39，比整數峰更靠近平移後的角。
Vizion.gen_image_const (Base, 'byte', 120, 90)
Vizion.gen_image_proto (Base, Sheet, 20)
Vizion.gen_rectangle1 (Box, 24, 36, 64, 84)
Vizion.paint_region (Box, Sheet, Image, 220, 'fill')
Vizion.points_harris (Image, 1.2, 0.04, 0.04, 'false', IRow, ICol)
StillIRow := 0
StillICol := 0
for Index := 1 to |IRow| by 1
if (IRow[Index] < 40 and ICol[Index] < 50)
StillIRow := IRow[Index]
StillICol := ICol[Index]
endif
endfor
Vizion.points_harris (Image, 1.2, 0.04, 0.04, 'true', Row, Col)
StillRow := 0
StillCol := 0
for Index := 1 to |Row| by 1
if (Row[Index] < 40 and Col[Index] < 50)
StillRow := Row[Index]
StillCol := Col[Index]
endif
endfor
Vizion.hom_mat2d_identity (Id)
Vizion.hom_mat2d_translate (Id, 0.4, 0.35, Shift)
Vizion.affine_trans_image (Image, Moved, Shift, 'bicubic', 'false')
Vizion.points_harris (Moved, 1.2, 0.04, 0.04, 'false', MIRow, MICol)
MoveIRow := 0
MoveICol := 0
for Index := 1 to |MIRow| by 1
if (MIRow[Index] < 40 and MICol[Index] < 50)
MoveIRow := MIRow[Index]
MoveICol := MICol[Index]
endif
endfor
Vizion.points_harris (Moved, 1.2, 0.04, 0.04, 'true', MRow, MCol)
CornerRow := 0
CornerCol := 0
for Index := 1 to |MRow| by 1
if (MRow[Index] < 40 and MCol[Index] < 50)
CornerRow := MRow[Index]
CornerCol := MCol[Index]
endif
endfor
Vizion.dev_display (Moved)
"""


HDEVELOP_BATCH95_PROGRAM = """* 平滑不會把端點往裡縮
* smooth_contours_xld 沿輪廓用高斯權重平滑。開放輪廓在端點做奇數延拓，直線不會被端點的缺點往裡拉。閉合輪廓維持閉合。
* 直線在列 10，行從 0 到 40，窗寬 5。左右平均會把第一個點從行 0 拉到 10，最後一個點從 40 拉到 30。
* 奇數延拓之後，五個點仍是列 10，行 0、10、20、30、40。
Vizion.gen_contour_polygon_xld (Line, [10, 10, 10, 10, 10], [0, 10, 20, 30, 40])
Vizion.smooth_contours_xld (Line, Smooth, 5)
Vizion.get_contour_xld (Smooth, Row, Col)
Vizion.dev_display (Smooth)
"""


HDEVELOP_BATCH96_PROGRAM = """* 放大的半個像素是 112
* zoom_image_size 的 bicubic 用 16 點 Keys 三次內插。半個像素不是兩點的平均。bilinear 仍是平均。
* 4×4 圖裡只有列 1、行 1 是 200，其餘是 0。放大成寬 6、高 4 時，輸出列 1、行 1 正好落在原圖的半個像素上。
* 兩點平均是 100。以前的三次核會得到 119。現在是 112。原圖那個亮點在一倍縮放時仍是 200。
Vizion.gen_image_const (Base, 'byte', 4, 4)
Vizion.set_grayval (Base, Dot, 1, 1, 200)
Vizion.zoom_image_size (Dot, Zoomed, 6, 4, 'bicubic')
Vizion.get_grayval (Zoomed, 1, 1, Mid)
Vizion.zoom_image_size (Dot, Linear, 6, 4, 'bilinear')
Vizion.get_grayval (Linear, 1, 1, Average)
Vizion.zoom_image_factor (Dot, Same, 1, 1, 'bicubic')
Vizion.get_grayval (Same, 1, 1, Kept)
Vizion.dev_display (Zoomed)
"""


HDEVELOP_BATCH97_PROGRAM = """* 橫條是 0、直條是 90 度
* elliptic_axis 和 orientation_region 用像素的二次矩。OpenCV 的邊界擬合對橫條和直條都回 -90 度，所以不再用它。
* 寬 20、高 5 的長條，長軸半徑是 11.53256，短軸是 2.82843，角度是 0。直的那條角度是 1.5708。
* major_phi 和這兩個角度相同。
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (Wide, 10, 10, 14, 29)
Vizion.gen_rectangle1 (Tall, 10, 50, 29, 54)
Vizion.elliptic_axis (Wide, WideRa, WideRb, WidePhi)
Vizion.orientation_region (Wide, WideAngle)
WideP := WidePhi[1]
WideA := WideAngle[1]
WRa := WideRa[1]
WRb := WideRb[1]
Vizion.elliptic_axis (Tall, TallRa, TallRb, TallPhi)
Vizion.orientation_region (Tall, TallAngle)
TallP := TallPhi[1]
TallA := TallAngle[1]
Vizion.dev_display (Wide)
"""


HDEVELOP_BATCH98_PROGRAM = """* 拉回去的半個像素是 112
* 仿射和對齊拉回的 bicubic 都用 16 點 Keys 三次內插。64×64 只有列 32、行 32 是 200。
* 行減 0.5 之後，列 32、行 32 落在半個像素上，得到 112。bilinear 仍是 100。
* align_translation 把原圖對到這張往左半個像素的基準，同一個位置也是 112。
Vizion.gen_image_const (Base, 'byte', 64, 64)
Vizion.set_grayval (Base, Dot, 32, 32, 200)
Vizion.hom_mat2d_identity (Id)
Vizion.hom_mat2d_translate (Id, 0, -0.5, Left)
Vizion.affine_trans_image (Dot, Cubic, Left, 'bicubic', 'false')
Vizion.get_grayval (Cubic, 32, 32, Mid)
Vizion.affine_trans_image (Dot, Linear, Left, 'bilinear', 'false')
Vizion.get_grayval (Linear, 32, 32, Average)
Vizion.align_translation (Dot, Cubic, Aligned, Row, Column, Score)
Vizion.get_grayval (Aligned, 32, 32, Pulled)
Vizion.dev_display (Cubic)
"""


HDEVELOP_BATCH99_PROGRAM = """* 高斯導數把峰修到 30.33298
* 同一列上，行 30 是 200、行 31 是 100。亮度加權的中心在行 30.33333。
* facet 的 3×3 只看旁邊一格，停在 30.16667。gauss 用和 derivate_gauss 同一套導數，Sigma 是 3，峰在 30.33298。
Vizion.gen_image_const (Base, 'byte', 48, 48)
Vizion.set_grayval (Base, Image, 20, 30, 200)
Vizion.set_grayval (Image, Image, 20, 31, 100)
Vizion.local_max_sub_pix (Image, 'facet', 1, 5, FacetRow, FacetCol)
Vizion.local_max_sub_pix (Image, 'gauss', 3, 5, GaussRow, GaussCol)
FRow := FacetRow[1]
FCol := FacetCol[1]
GRow := GaussRow[1]
GCol := GaussCol[1]
Vizion.dev_display (Image)
"""


HDEVELOP_BATCH100_PROGRAM = """* 方塊的角修進像素裡面
* 方塊左上角在列 24、行 36。OpenCV 的 FAST 在這張圖上一個角都沒有。
* 圓周上連續 9 點夠亮或夠暗就留下，再把整數位置修進像素。左上角是列 24.09426、行 36.09426。
Vizion.gen_image_const (Base, 'byte', 90, 120)
Vizion.gen_image_proto (Base, Sheet, 20)
Vizion.gen_rectangle1 (Box, 24, 36, 64, 84)
Vizion.paint_region (Box, Sheet, Image, 220, 'fill')
Vizion.points_lepetit (Image, 30, 'true', Row, Column)
Best := 100
BRow := 0
BCol := 0
for Index := 1 to |Row| by 1
Dist := (Row[Index] - 24) * (Row[Index] - 24) + (Column[Index] - 36) * (Column[Index] - 36)
if (Dist < Best)
Best := Dist
BRow := Row[Index]
Vizion.dev_display (Image)
"""


HDEVELOP_BATCH101_PROGRAM = """* 尺度修到 1.07505
* 教導的尺度是 0.8、1.0、1.2。模板方塊高 30，場景方塊高 33，真正的尺度是 1.1。
* 不修尺度時停在 1。least_squares 沿邊緣法線把尺度修到 1.07505，比 1 更靠近 1.1。
Vizion.gen_image_const (Base, 'byte', 80, 80)
Vizion.gen_image_proto (Base, Sheet, 30)
Vizion.gen_rectangle1 (Box, 20, 20, 49, 49)
Vizion.paint_region (Box, Sheet, Template, 220, 'fill')
Vizion.gen_image_const (SceneBase, 'byte', 120, 120)
Vizion.gen_image_proto (SceneBase, SceneSheet, 30)
Vizion.gen_rectangle1 (Big, 24, 24, 56, 56)
Vizion.paint_region (Big, SceneSheet, Scene, 220, 'fill')
Vizion.create_scaled_shape_model (Template, 2, 0, 0, 0.2, 0.8, 1.2, 0.2, 'none', 'use_polarity', 30, 10, ModelID)
Vizion.find_scaled_shape_model (Scene, ModelID, 0, 0, 0.8, 1.2, 0.4, 1, 0.5, 'least_squares', 0, 0.6, Row, Column, Angle, Scale, Score)
Vizion.find_scaled_shape_model (Scene, ModelID, 0, 0, 0.8, 1.2, 0.4, 1, 0.5, 'none', 0, 0.6, NRow, NCol, NAngle, NScale, NScore)
S := Scale[1]
Still := NScale[1]
Vizion.dev_display (Scene)
"""


HDEVELOP_BATCH102_PROGRAM = """* 平滑平移的列是 39.23
* 教導框中心是列 39.5、行 47.5。圖案往上 0.25、往右 0.35，真的中心在列 39.25、行 47.85。
* 整數峰停在列 40。列 39 的分數更高，三點拋物線卻會停在 40。
* Keys 三次內插沿著這段分數把列修到 39.23、行修到 47.86。峰頂幾乎一樣高時，位置留在整數。
Vizion.gen_image_const (Base, 'byte', 96, 80)
Vizion.gen_image_proto (Base, Sheet, 30)
Vizion.gen_rectangle1 (A, 18, 20, 33, 43)
Vizion.paint_region (A, Sheet, P1, 200, 'fill')
Vizion.gen_rectangle1 (B, 36, 40, 57, 69)
Vizion.paint_region (B, P1, P2, 90, 'fill')
Vizion.gen_rectangle1 (C, 22, 60, 29, 79)
Vizion.paint_region (C, P2, P3, 240, 'fill')
Vizion.gen_rectangle1 (D, 48, 12, 69, 27)
Vizion.paint_region (D, P3, Sharp, 150, 'fill')
Vizion.gauss_image (Sharp, Image, 5)
Vizion.teach_ncc_model (Image, 10, 10, 69, 85, 1, 0, 0, 0, 'use_polarity', ModelID)
Vizion.hom_mat2d_identity (Id)
Vizion.hom_mat2d_translate (Id, -0.25, 0.35, Shift)
Vizion.affine_trans_image (Image, Moved, Shift, 'bicubic', 'false')
Vizion.find_ncc_model (Moved, ModelID, 0, 0, 0.5, 1, 0.5, 'none', 1, NRow, NCol, NAng, NScore)
Vizion.find_ncc_model (Moved, ModelID, 0, 0, 0.5, 1, 0.5, 'interpolation', 1, IRow, ICol, IAng, IScore)
Still := NRow[1]
SRow := IRow[1]
SCol := ICol[1]
Vizion.dev_display (Moved)
"""


HDEVELOP_BATCH103_PROGRAM = """* 印出 Data Matrix 再讀回 VISDEV
* gen_datamatrix 自己排 ECC200 方碼，放大後四周留白。
* SymbolType 用 Data Matrix。find_data_code_2d 讀回來仍是 VISDEV。
Vizion.gen_datamatrix (Code, 'VISDEV')
Vizion.create_data_code_2d_model ('Data Matrix', '', '', Handle)
Vizion.find_data_code_2d (Code, Symbol, Handle, '', '', Ids, Decoded)
Text := Decoded[1]
Vizion.dev_display (Code)
"""


HDEVELOP_BATCH104_PROGRAM = """* 輪廓 tuple 傳進程序仍是 HObject
* 幾條輪廓收成同一個物件，再傳進程序。count_obj 數的是條數。
* ho_XLDs[1] 是第一條。圖形視窗裡兩條顏色不同。
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_contour_polygon_xld (ho_A, [10, 10, 40], [10, 40, 40])
Vizion.gen_contour_polygon_xld (ho_B, [20, 50], [15, 55])
ho_XLDs := [ho_A, ho_B]
pass (ho_XLDs, ho_Kept, hv_Count)
Vizion.length_xld (ho_XLDs, Length)
First := Length[1]
ho_One := ho_XLDs[1]
Vizion.count_obj (ho_One, OneN)
Vizion.dev_display (ho_XLDs)
procedure pass (Contours : Kept : : Number)
    Vizion.count_obj (Contours, Number)
    Vizion.select_obj (Contours, Kept, 1)
endprocedure
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
    "ViZion_hdevelop_batch1.viz": HDEVELOP_BATCH1_PROGRAM,
    "ViZion_hdevelop_batch2.viz": HDEVELOP_BATCH2_PROGRAM,
    "ViZion_hdevelop_batch3.viz": HDEVELOP_BATCH3_PROGRAM,
    "ViZion_hdevelop_batch4.viz": HDEVELOP_BATCH4_PROGRAM,
    "ViZion_hdevelop_batch5.viz": HDEVELOP_BATCH5_PROGRAM,
    "ViZion_hdevelop_batch6.viz": HDEVELOP_BATCH6_PROGRAM,
    "ViZion_hdevelop_batch7.viz": HDEVELOP_BATCH7_PROGRAM,
    "ViZion_hdevelop_batch8.viz": HDEVELOP_BATCH8_PROGRAM,
    "ViZion_hdevelop_batch9.viz": HDEVELOP_BATCH9_PROGRAM,
    "ViZion_hdevelop_batch10.viz": HDEVELOP_BATCH10_PROGRAM,
    "ViZion_hdevelop_batch11.viz": HDEVELOP_BATCH11_PROGRAM,
    "ViZion_hdevelop_batch12.viz": HDEVELOP_BATCH12_PROGRAM,
    "ViZion_hdevelop_batch13.viz": HDEVELOP_BATCH13_PROGRAM,
    "ViZion_hdevelop_batch14.viz": HDEVELOP_BATCH14_PROGRAM,
    "ViZion_hdevelop_batch15.viz": HDEVELOP_BATCH15_PROGRAM,
    "ViZion_hdevelop_batch16.viz": HDEVELOP_BATCH16_PROGRAM,
    "ViZion_hdevelop_batch17.viz": HDEVELOP_BATCH17_PROGRAM,
    "ViZion_hdevelop_batch18.viz": HDEVELOP_BATCH18_PROGRAM,
    "ViZion_hdevelop_batch19.viz": HDEVELOP_BATCH19_PROGRAM,
    "ViZion_hdevelop_batch20.viz": HDEVELOP_BATCH20_PROGRAM,
    "ViZion_hdevelop_batch21.viz": HDEVELOP_BATCH21_PROGRAM,
    "ViZion_hdevelop_batch22.viz": HDEVELOP_BATCH22_PROGRAM,
    "ViZion_hdevelop_batch23.viz": HDEVELOP_BATCH23_PROGRAM,
    "ViZion_hdevelop_batch24.viz": HDEVELOP_BATCH24_PROGRAM,
    "ViZion_hdevelop_batch25.viz": HDEVELOP_BATCH25_PROGRAM,
    "ViZion_hdevelop_batch26.viz": HDEVELOP_BATCH26_PROGRAM,
    "ViZion_hdevelop_batch27.viz": HDEVELOP_BATCH27_PROGRAM,
    "ViZion_hdevelop_batch28.viz": HDEVELOP_BATCH28_PROGRAM,
    "ViZion_hdevelop_batch29.viz": HDEVELOP_BATCH29_PROGRAM,
    "ViZion_hdevelop_batch30.viz": HDEVELOP_BATCH30_PROGRAM,
    "ViZion_hdevelop_batch31.viz": HDEVELOP_BATCH31_PROGRAM,
    "ViZion_hdevelop_batch32.viz": HDEVELOP_BATCH32_PROGRAM,
    "ViZion_hdevelop_batch33.viz": HDEVELOP_BATCH33_PROGRAM,
    "ViZion_hdevelop_batch34.viz": HDEVELOP_BATCH34_PROGRAM,
    "ViZion_hdevelop_batch35.viz": HDEVELOP_BATCH35_PROGRAM,
    "ViZion_hdevelop_batch36.viz": HDEVELOP_BATCH36_PROGRAM,
    "ViZion_hdevelop_batch37.viz": HDEVELOP_BATCH37_PROGRAM,
    "ViZion_hdevelop_batch38.viz": HDEVELOP_BATCH38_PROGRAM,
    "ViZion_hdevelop_batch39.viz": HDEVELOP_BATCH39_PROGRAM,
    "ViZion_hdevelop_batch40.viz": HDEVELOP_BATCH40_PROGRAM,
    "ViZion_hdevelop_batch41.viz": HDEVELOP_BATCH41_PROGRAM,
    "ViZion_hdevelop_batch42.viz": HDEVELOP_BATCH42_PROGRAM,
    "ViZion_hdevelop_batch43.viz": HDEVELOP_BATCH43_PROGRAM,
    "ViZion_hdevelop_batch44.viz": HDEVELOP_BATCH44_PROGRAM,
    "ViZion_hdevelop_batch45.viz": HDEVELOP_BATCH45_PROGRAM,
    "ViZion_hdevelop_batch46.viz": HDEVELOP_BATCH46_PROGRAM,
    "ViZion_hdevelop_batch47.viz": HDEVELOP_BATCH47_PROGRAM,
    "ViZion_hdevelop_batch48.viz": HDEVELOP_BATCH48_PROGRAM,
    "ViZion_hdevelop_batch49.viz": HDEVELOP_BATCH49_PROGRAM,
    "ViZion_hdevelop_batch50.viz": HDEVELOP_BATCH50_PROGRAM,
    "ViZion_hdevelop_batch51.viz": HDEVELOP_BATCH51_PROGRAM,
    "ViZion_hdevelop_batch52.viz": HDEVELOP_BATCH52_PROGRAM,
    "ViZion_hdevelop_batch53.viz": HDEVELOP_BATCH53_PROGRAM,
    "ViZion_hdevelop_batch54.viz": HDEVELOP_BATCH54_PROGRAM,
    "ViZion_hdevelop_batch55.viz": HDEVELOP_BATCH55_PROGRAM,
    "ViZion_hdevelop_batch56.viz": HDEVELOP_BATCH56_PROGRAM,
    "ViZion_hdevelop_batch57.viz": HDEVELOP_BATCH57_PROGRAM,
    "ViZion_hdevelop_batch58.viz": HDEVELOP_BATCH58_PROGRAM,
    "ViZion_hdevelop_batch59.viz": HDEVELOP_BATCH59_PROGRAM,
    "ViZion_hdevelop_batch60.viz": HDEVELOP_BATCH60_PROGRAM,
    "ViZion_hdevelop_batch61.viz": HDEVELOP_BATCH61_PROGRAM,
    "ViZion_hdevelop_batch62.viz": HDEVELOP_BATCH62_PROGRAM,
    "ViZion_hdevelop_batch63.viz": HDEVELOP_BATCH63_PROGRAM,
    "ViZion_hdevelop_batch64.viz": HDEVELOP_BATCH64_PROGRAM,
    "ViZion_hdevelop_batch65.viz": HDEVELOP_BATCH65_PROGRAM,
    "ViZion_hdevelop_batch66.viz": HDEVELOP_BATCH66_PROGRAM,
    "ViZion_hdevelop_batch67.viz": HDEVELOP_BATCH67_PROGRAM,
    "ViZion_hdevelop_batch68.viz": HDEVELOP_BATCH68_PROGRAM,
    "ViZion_hdevelop_batch69.viz": HDEVELOP_BATCH69_PROGRAM,
    "ViZion_hdevelop_batch70.viz": HDEVELOP_BATCH70_PROGRAM,
    "ViZion_hdevelop_batch71.viz": HDEVELOP_BATCH71_PROGRAM,
    "ViZion_hdevelop_batch72.viz": HDEVELOP_BATCH72_PROGRAM,
    "ViZion_hdevelop_batch73.viz": HDEVELOP_BATCH73_PROGRAM,
    "ViZion_hdevelop_batch74.viz": HDEVELOP_BATCH74_PROGRAM,
    "ViZion_hdevelop_batch75.viz": HDEVELOP_BATCH75_PROGRAM,
    "ViZion_hdevelop_batch76.viz": HDEVELOP_BATCH76_PROGRAM,
    "ViZion_hdevelop_batch77.viz": HDEVELOP_BATCH77_PROGRAM,
    "ViZion_hdevelop_batch78.viz": HDEVELOP_BATCH78_PROGRAM,
    "ViZion_hdevelop_batch79.viz": HDEVELOP_BATCH79_PROGRAM,
    "ViZion_hdevelop_batch80.viz": HDEVELOP_BATCH80_PROGRAM,
    "ViZion_hdevelop_batch81.viz": HDEVELOP_BATCH81_PROGRAM,
    "ViZion_hdevelop_batch82.viz": HDEVELOP_BATCH82_PROGRAM,
    "ViZion_hdevelop_batch83.viz": HDEVELOP_BATCH83_PROGRAM,
    "ViZion_hdevelop_batch84.viz": HDEVELOP_BATCH84_PROGRAM,
    "ViZion_hdevelop_batch85.viz": HDEVELOP_BATCH85_PROGRAM,
    "ViZion_hdevelop_batch86.viz": HDEVELOP_BATCH86_PROGRAM,
    "ViZion_hdevelop_batch87.viz": HDEVELOP_BATCH87_PROGRAM,
    "ViZion_hdevelop_batch88.viz": HDEVELOP_BATCH88_PROGRAM,
    "ViZion_hdevelop_batch89.viz": HDEVELOP_BATCH89_PROGRAM,
    "ViZion_hdevelop_batch90.viz": HDEVELOP_BATCH90_PROGRAM,
    "ViZion_hdevelop_batch91.viz": HDEVELOP_BATCH91_PROGRAM,
    "ViZion_hdevelop_batch92.viz": HDEVELOP_BATCH92_PROGRAM,
    "ViZion_hdevelop_batch93.viz": HDEVELOP_BATCH93_PROGRAM,
    "ViZion_hdevelop_batch94.viz": HDEVELOP_BATCH94_PROGRAM,
    "ViZion_hdevelop_batch95.viz": HDEVELOP_BATCH95_PROGRAM,
    "ViZion_hdevelop_batch96.viz": HDEVELOP_BATCH96_PROGRAM,
    "ViZion_hdevelop_batch97.viz": HDEVELOP_BATCH97_PROGRAM,
    "ViZion_hdevelop_batch98.viz": HDEVELOP_BATCH98_PROGRAM,
    "ViZion_hdevelop_batch99.viz": HDEVELOP_BATCH99_PROGRAM,
    "ViZion_hdevelop_batch100.viz": HDEVELOP_BATCH100_PROGRAM,
    "ViZion_hdevelop_batch101.viz": HDEVELOP_BATCH101_PROGRAM,
    "ViZion_hdevelop_batch102.viz": HDEVELOP_BATCH102_PROGRAM,
    "ViZion_hdevelop_batch103.viz": HDEVELOP_BATCH103_PROGRAM,
    "ViZion_hdevelop_batch104.viz": HDEVELOP_BATCH104_PROGRAM,
    "ocr_read.hdv": OCR_READ_PROGRAM,
}
