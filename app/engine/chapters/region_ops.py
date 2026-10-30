"""Region morphology, geometry, and segmentation on the existing Region type."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from ..objects import Region, Result
from ..operators import P, as_list, expect_image, expect_region, operator, region_out, scalar, text, whole
from .image_ops import _mat_from, _opencv_forward


def _kernel(struct) -> np.ndarray:
    region = expect_region(struct, "StructElement")
    mask = region.union_mask()
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("結構元素是空的")
    kernel = (mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] > 0).astype(np.uint8)
    return kernel


def _with_kernel(region, struct, mode: str) -> Result:
    source = expect_region(region, "Region")
    kernel = _kernel(struct)
    mask = source.union_mask()
    if mode == "erode":
        out = cv2.erode(mask, kernel)
    elif mode == "dilate":
        out = cv2.dilate(mask, kernel)
    elif mode == "open":
        out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif mode == "close":
        out = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError(mode)
    return region_out(source.width, source.height, out)


_STRUCT = "結構元素只取形狀，原點放在它的外接框中心。這和 HALCON 用區域座標 (0,0) 當原點不同。"


@operator(
    "形態學",
    [P("Region", "輸入區域"), P("StructElement", "結構元素區域")],
    [P("RegionErosion", "侵蝕結果")],
    [],
    [],
    "用任意結構元素侵蝕",
    _STRUCT,
    "erosion1 (Region, StructElement, RegionErosion)",
)
def erosion1(region, struct_element, *, ctx):
    return _with_kernel(region, struct_element, "erode")


@operator(
    "形態學",
    [P("Region", "輸入區域"), P("StructElement", "結構元素區域")],
    [P("RegionDilation", "膨脹結果")],
    [],
    [],
    "用任意結構元素膨脹",
    _STRUCT,
    "dilation1 (Region, StructElement, RegionDilation)",
)
def dilation1(region, struct_element, *, ctx):
    return _with_kernel(region, struct_element, "dilate")


@operator(
    "形態學",
    [P("Region", "輸入區域"), P("StructElement", "結構元素區域")],
    [P("RegionOpening", "開運算結果")],
    [],
    [],
    "用任意結構元素做開運算",
    "先侵蝕再膨脹。" + _STRUCT,
    "opening (Region, StructElement, RegionOpening)",
)
def opening(region, struct_element, *, ctx):
    return _with_kernel(region, struct_element, "open")


@operator(
    "形態學",
    [P("Region", "輸入區域"), P("StructElement", "結構元素區域")],
    [P("RegionClosing", "閉運算結果")],
    [],
    [],
    "用任意結構元素做閉運算",
    "先膨脹再侵蝕。" + _STRUCT,
    "closing (Region, StructElement, RegionClosing)",
)
def closing(region, struct_element, *, ctx):
    return _with_kernel(region, struct_element, "close")


@operator(
    "形態學",
    [P("Region", "輸入區域")],
    [P("Skeleton", "骨架")],
    [],
    [],
    "區域骨架",
    "用形態學細化，留下一條大致在區域中線的骨架。結果是單一區域。",
    "skeleton (Region, Skeleton)",
)
def skeleton(region, *, ctx):
    source = expect_region(region, "Region")
    img = source.union_mask()
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(img):
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        skel = cv2.bitwise_or(skel, cv2.subtract(img, opened))
        eroded = cv2.erode(img, element)
        if cv2.countNonZero(eroded) == cv2.countNonZero(img):
            break
        img = eroded
    return region_out(source.width, source.height, skel)


def _shift_mask(mask: np.ndarray, drow: int, dcol: int) -> np.ndarray:
    height, width = mask.shape[:2]
    matrix = np.array([[1.0, 0.0, float(dcol)], [0.0, 1.0, float(drow)]], np.float64)
    return cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)


def _each(region: Region, fn) -> Result:
    objects = []
    for mask in region.objects:
        out = fn(mask)
        if int(cv2.countNonZero(out)):
            objects.append(out)
    return Result([Region(region.width, region.height, objects, multi_color=region.multi_color and len(objects) > 1)], [])


@operator(
    "區域",
    [P("Region", "輸入區域")],
    [P("RegionMoved", "平移後區域")],
    [P("Row", "往下的列數，負數往上"), P("Column", "往右的行數，負數往左")],
    [],
    "平移區域",
    "移出影像的部分會被切掉。每個物件各自平移。",
    "move_region (Region, RegionMoved, 10, 0)",
)
def move_region(region, row, column, *, ctx):
    source = expect_region(region, "Region")
    drow, dcol = whole(row, "Row"), whole(column, "Column")
    return _each(source, lambda mask: _shift_mask(mask, drow, dcol))


@operator(
    "區域",
    [P("Region", "輸入區域")],
    [P("RegionMirror", "鏡像後區域")],
    [P("Mode", "row、column 或 diagonal", choices=["'row'", "'column'", "'diagonal'"])],
    [],
    "區域鏡像",
    "row 上下對調，column 左右對調，diagonal 轉置。轉置後寬高交換。",
    "mirror_region (Region, RegionMirror, 'row')",
)
def mirror_region(region, mode, *, ctx):
    source = expect_region(region, "Region")
    kind = text(mode, "Mode").lower()
    if kind == "row":
        flip = lambda mask: cv2.flip(mask, 0)
        width, height = source.width, source.height
    elif kind == "column":
        flip = lambda mask: cv2.flip(mask, 1)
        width, height = source.width, source.height
    elif kind == "diagonal":
        flip = cv2.transpose
        width, height = source.height, source.width
    else:
        raise ValueError("Mode 只能是 row、column 或 diagonal")
    objects = [flip(mask) for mask in source.objects if int(cv2.countNonZero(mask))]
    return Result([Region(width, height, objects, multi_color=source.multi_color and len(objects) > 1)], [])


@operator(
    "區域",
    [P("Region", "輸入區域")],
    [P("RegionZoom", "縮放後區域")],
    [P("ScaleWidth", "寬度倍率"), P("ScaleHeight", "高度倍率")],
    [],
    "依倍率縮放區域",
    "ScaleWidth 乘行數，ScaleHeight 乘列數。用最近鄰，所以邊緣是像素級。",
    "zoom_region (Region, RegionZoom, 2, 2)",
)
def zoom_region(region, scale_width, scale_height, *, ctx):
    source = expect_region(region, "Region")
    sx, sy = scalar(scale_width, "ScaleWidth"), scalar(scale_height, "ScaleHeight")
    if sx <= 0 or sy <= 0:
        raise ValueError("縮放倍率必須大於 0")
    width = max(1, int(round(source.width * sx)))
    height = max(1, int(round(source.height * sy)))
    objects = []
    for mask in source.objects:
        resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        resized = np.where(resized > 0, 255, 0).astype(np.uint8)
        if int(cv2.countNonZero(resized)):
            objects.append(resized)
    return Result([Region(width, height, objects, multi_color=source.multi_color and len(objects) > 1)], [])


@operator(
    "區域",
    [P("Region", "輸入區域")],
    [P("RegionAffineTrans", "轉換後區域")],
    [P("HomMat2D", "仿射矩陣"), P("Interpolation", "nearest_neighbor 或 constant", "'nearest_neighbor'", ["'nearest_neighbor'", "'constant'"])],
    [],
    "用仿射矩陣轉換區域",
    "矩陣和 affine_trans_image 同一套：x 是列、y 是行。畫布大小不變，轉出去的部分切掉。區域只用最近鄰。",
    "affine_trans_region (Region, RegionAffineTrans, HomMat2D, 'nearest_neighbor')",
)
def affine_trans_region(region, hom_mat2d, interpolation, *, ctx):
    source = expect_region(region, "Region")
    kind = text(interpolation, "Interpolation").lower()
    if kind not in ("nearest_neighbor", "constant"):
        raise ValueError("區域轉換的 Interpolation 只能是 nearest_neighbor 或 constant")
    mapping = _opencv_forward(_mat_from(hom_mat2d))
    objects = []
    for mask in source.objects:
        warped = cv2.warpAffine(mask, mapping, (source.width, source.height), flags=cv2.INTER_NEAREST, borderValue=0)
        warped = np.where(warped > 0, 255, 0).astype(np.uint8)
        if int(cv2.countNonZero(warped)):
            objects.append(warped)
    return Result([Region(source.width, source.height, objects, multi_color=source.multi_color and len(objects) > 1)], [])


def _gray_domain(image):
    picture = expect_image(image, "Image")
    gray = picture.gray()
    domain = picture.domain if picture.domain is not None else np.full(gray.shape[:2], 255, np.uint8)
    return picture, gray, domain


@operator(
    "分割",
    [P("OrigImage", "原圖"), P("ThresholdImage", "比較用的平滑圖")],
    [P("RegionDynThresh", "局部閾值區域")],
    [
        P("Offset", "灰階差"),
        P("LightDark", "light、dark、equal 或 not_equal", choices=["'light'", "'dark'", "'equal'", "'not_equal'"]),
    ],
    [],
    "和另一張圖比較的局部閾值",
    "light：原圖 >= 平滑圖 + Offset。dark：原圖 <= 平滑圖 - Offset。equal：兩張差的絕對值不超過 Offset。not_equal 是 equal 的相反。",
    "dyn_threshold (Image, Smooth, Region, 5, 'light')",
)
def dyn_threshold(orig_image, threshold_image, offset, light_dark, *, ctx):
    left, gray, domain = _gray_domain(orig_image)
    right = expect_image(threshold_image, "ThresholdImage")
    if right.data.shape[:2] != gray.shape[:2]:
        raise ValueError("兩張影像的尺寸不同")
    other = right.gray().astype(np.int16)
    base = gray.astype(np.int16)
    delta = scalar(offset, "Offset")
    mode = text(light_dark, "LightDark").lower()
    if mode == "light":
        picked = base >= other + delta
    elif mode == "dark":
        picked = base <= other - delta
    elif mode == "equal":
        picked = np.abs(base - other) <= delta
    elif mode == "not_equal":
        picked = np.abs(base - other) > delta
    else:
        raise ValueError("LightDark 只能是 light、dark、equal 或 not_equal")
    mask = np.where(picked & (domain > 0), 255, 0).astype(np.uint8)
    return region_out(left.width, left.height, mask)


@operator(
    "分割",
    [P("Image", "輸入影像")],
    [P("Region", "局部閾值區域")],
    [
        P("MaskWidth", "局部視窗寬，奇數"),
        P("MaskHeight", "局部視窗高，奇數"),
        P("StdDevScale", "標準差倍率"),
        P("AbsThreshold", "至少要差這麼多灰階"),
        P("LightDark", "light、dark、equal 或 not_equal", choices=["'light'", "'dark'", "'equal'", "'not_equal'"]),
    ],
    [],
    "用局部平均和標準差做閾值",
    "門檻是 max(標準差 × StdDevScale, AbsThreshold)。light 取高於平均加門檻的像素，dark 取低於平均減門檻的像素。",
    "var_threshold (Image, Region, 15, 15, 0.2, 2, 'light')",
)
def var_threshold(image, mask_width, mask_height, std_dev_scale, abs_threshold, light_dark, *, ctx):
    picture, gray, domain = _gray_domain(image)
    wide = whole(mask_width, "MaskWidth")
    high = whole(mask_height, "MaskHeight")
    if wide < 1 or high < 1:
        raise ValueError("視窗寬高至少要 1")
    if wide % 2 == 0:
        wide += 1
    if high % 2 == 0:
        high += 1
    values = gray.astype(np.float32)
    mean = cv2.blur(values, (wide, high))
    deviation = np.sqrt(np.maximum(cv2.blur(values * values, (wide, high)) - mean * mean, 0))
    limit = np.maximum(deviation * scalar(std_dev_scale, "StdDevScale"), scalar(abs_threshold, "AbsThreshold"))
    mode = text(light_dark, "LightDark").lower()
    if mode == "light":
        picked = values >= mean + limit
    elif mode == "dark":
        picked = values <= mean - limit
    elif mode == "equal":
        picked = np.abs(values - mean) <= limit
    elif mode == "not_equal":
        picked = np.abs(values - mean) > limit
    else:
        raise ValueError("LightDark 只能是 light、dark、equal 或 not_equal")
    mask = np.where(picked & (domain > 0), 255, 0).astype(np.uint8)
    return region_out(picture.width, picture.height, mask)


@operator(
    "分割",
    [P("Image", "輸入影像")],
    [P("RegionHysteresis", "滯後閾值區域")],
    [P("Low", "低門檻"), P("High", "高門檻，種子"), P("MaxLength", "種子可以跨出去接低門檻的半徑")],
    [],
    "滯後閾值",
    "高門檻以上是種子，低門檻以上而且和種子八連通的像素留下。MaxLength 大於 0 時，種子先膨脹這個半徑再去接低門檻。這是距離膨脹，不是沿細線走的橋接。",
    "hysteresis_threshold (Image, Region, 20, 40, 0)",
)
def hysteresis_threshold(image, low, high, max_length, *, ctx):
    picture, gray, domain = _gray_domain(image)
    low_v = scalar(low, "Low")
    high_v = scalar(high, "High")
    if high_v < low_v:
        raise ValueError("High 不能小於 Low")
    length = scalar(max_length, "MaxLength")
    if length < 0:
        raise ValueError("MaxLength 不能是負的")
    low_mask = np.where((gray >= low_v) & (domain > 0), 255, 0).astype(np.uint8)
    seeds = np.where((gray >= high_v) & (domain > 0), 255, 0).astype(np.uint8)
    if length > 0:
        radius = max(1, int(round(length)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        seeds = cv2.dilate(seeds, kernel)
    count, labels = cv2.connectedComponents(low_mask, connectivity=8)
    keep = np.zeros(count, np.uint8)
    overlapping = np.unique(labels[seeds > 0])
    keep[overlapping] = 1
    keep[0] = 0
    mask = np.where(keep[labels] > 0, 255, 0).astype(np.uint8)
    return region_out(picture.width, picture.height, mask)


@operator(
    "分割",
    [P("Image", "輸入影像")],
    [P("Regions", "長出來的區域")],
    [P("Row", "種子列"), P("Column", "種子行"), P("Tolerance", "相鄰像素允許的灰階差"), P("MinSize", "小於這個面積就丟掉")],
    [],
    "從種子做區域成長",
    "只從給的種子開始。八鄰域裡，和已經長到的那個像素灰階差不超過 Tolerance 就加入。每個種子一塊區域，面積小於 MinSize 的丟掉。",
    "regiongrowing (Image, Regions, 120, 100, 10, 20)",
)
def regiongrowing(image, row, column, tolerance, min_size, *, ctx):
    picture, gray, domain = _gray_domain(image)
    rows = [int(round(float(item))) for item in as_list(row)]
    cols = [int(round(float(item))) for item in as_list(column)]
    if len(rows) != len(cols):
        raise ValueError("Row 和 Column 的長度要相同")
    if not rows:
        raise ValueError("至少要一個種子")
    allowed = scalar(tolerance, "Tolerance")
    minimum = whole(min_size, "MinSize")
    height, width = gray.shape[:2]
    visited = np.zeros(gray.shape, np.uint8)
    objects = []
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for seed_row, seed_col in zip(rows, cols):
        if not (0 <= seed_row < height and 0 <= seed_col < width):
            continue
        if visited[seed_row, seed_col] or domain[seed_row, seed_col] == 0:
            continue
        queue = deque([(seed_row, seed_col)])
        visited[seed_row, seed_col] = 1
        pixels = []
        while queue:
            r, c = queue.popleft()
            pixels.append((r, c))
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < height and 0 <= nc < width) or visited[nr, nc] or domain[nr, nc] == 0:
                    continue
                if abs(int(gray[nr, nc]) - int(gray[r, c])) <= allowed:
                    visited[nr, nc] = 1
                    queue.append((nr, nc))
        if len(pixels) < minimum:
            continue
        mask = np.zeros(gray.shape, np.uint8)
        rr, cc = zip(*pixels)
        mask[rr, cc] = 255
        objects.append(mask)
    return Result([Region(picture.width, picture.height, objects, multi_color=len(objects) > 1)], [])


@operator(
    "分割",
    [P("Image", "輸入影像")],
    [P("Basins", "集水區"), P("Watersheds", "分水嶺")],
    [],
    [],
    "灰階影像的分水嶺",
    "把灰階當成高度。每個局部最低點是一個集水區，物件用不同顏色。分水嶺是盆地交界，另外輸出成一個區域。平坦或雜訊多的圖會切得很碎。",
    "watersheds (Image, Basins, Watersheds)",
)
def watersheds(image, *, ctx):
    picture, gray, domain = _gray_domain(image)
    kernel = np.ones((3, 3), np.uint8)
    minima = cv2.compare(gray, cv2.erode(gray, kernel), cv2.CMP_EQ)
    minima = cv2.bitwise_and(minima, domain)
    count, markers = cv2.connectedComponents((minima > 0).astype(np.uint8))
    markers = markers.astype(np.int32)
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)
    markers = np.where(domain > 0, markers, 0)
    basins = []
    for label in range(1, count):
        mask = np.where(markers == label, 255, 0).astype(np.uint8)
        if int(cv2.countNonZero(mask)):
            basins.append(mask)
    lines = np.where(markers == -1, 255, 0).astype(np.uint8)
    return Result(
        [
            Region(picture.width, picture.height, basins, multi_color=len(basins) > 1),
            Region(picture.width, picture.height, [lines] if int(cv2.countNonZero(lines)) else []),
        ],
        [],
    )
