"""Region morphology, geometry, and segmentation on the existing Region type."""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

from ..objects import Image, Region, Result
from ..operators import (
    P,
    as_list,
    difference,
    expect_image,
    expect_region,
    halcon_components,
    REGION_FEATURE_CHOICES,
    object_feature,
    operator,
    equivalent_ellipse,
    region_measures,
    region_out,
    scalar,
    text,
    whole,
)
from .image_ops import _mat_from, _opencv_forward, _opencv_projective, _polar_inverse_maps, _proj_from


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
    "Vizion.erosion1 (Region, StructElement, RegionErosion)",
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
    "Vizion.dilation1 (Region, StructElement, RegionDilation)",
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
    "Vizion.opening (Region, StructElement, RegionOpening)",
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
    "Vizion.closing (Region, StructElement, RegionClosing)",
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
    "Vizion.skeleton (Region, Skeleton)",
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
    "Vizion.move_region (Region, RegionMoved, 10, 0)",
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
    "Vizion.mirror_region (Region, RegionMirror, 'row')",
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
    "Vizion.zoom_region (Region, RegionZoom, 2, 2)",
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
    "Vizion.affine_trans_region (Region, RegionAffineTrans, HomMat2D, 'nearest_neighbor')",
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
    "Vizion.dyn_threshold (Image, Smooth, Region, 5, 'light')",
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
    "Vizion.var_threshold (Image, Region, 15, 15, 0.2, 2, 'light')",
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
    "Vizion.hysteresis_threshold (Image, Region, 20, 40, 0)",
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
    "Vizion.regiongrowing (Image, Regions, 120, 100, 10, 20)",
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
def _watershed_markers(gray: np.ndarray, domain: np.ndarray):
    kernel = np.ones((3, 3), np.uint8)
    minima = cv2.compare(gray, cv2.erode(gray, kernel), cv2.CMP_EQ)
    minima = cv2.bitwise_and(minima, domain)
    count, markers = cv2.connectedComponents((minima > 0).astype(np.uint8))
    markers = markers.astype(np.int32)
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)
    return np.where(domain > 0, markers, 0), count


def watersheds(image, *, ctx):
    picture, gray, domain = _gray_domain(image)
    markers, count = _watershed_markers(gray, domain)
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


def _merge_basins(markers: np.ndarray, gray: np.ndarray, count: int, threshold: float) -> np.ndarray:
    parent = list(range(count))

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    positive = markers > 0
    mins = np.full(count, 255, np.int16)
    if int(positive.sum()):
        np.minimum.at(mins, markers[positive], gray[positive].astype(np.int16))
    padded = np.pad(markers, 1, constant_values=0)
    height, width = markers.shape
    shifts = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    neighbors = np.stack([padded[1 + drow : 1 + drow + height, 1 + dcol : 1 + dcol + width] for drow, dcol in shifts], axis=0)
    rows, cols = np.where(markers == -1)
    for row, col in zip(rows.tolist(), cols.tolist()):
        labels = {int(value) for value in neighbors[:, row, col] if value > 0}
        if len(labels) < 2:
            continue
        height_gray = int(gray[row, col])
        ordered = sorted(labels)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                if height_gray - int(min(mins[left], mins[right])) < threshold:
                    parent[find(right)] = find(left)
    for row, col in zip(rows.tolist(), cols.tolist()):
        labels = [int(value) for value in neighbors[:, row, col] if value > 0]
        if labels:
            markers[row, col] = min(labels, key=lambda label: (int(mins[label]), label))
    roots = [find(label) for label in range(count)]
    mapped = markers.copy()
    inside = mapped > 0
    if int(inside.sum()):
        mapped[inside] = np.take(np.asarray(roots), mapped[inside])
    return mapped


@operator(
    "分割",
    [P("Image", "輸入影像")],
    [P("Basins", "合併後的集水區")],
    [P("Threshold", "低過這個高度差就合併")],
    [],
    "把太淺的分水嶺合併",
    "先做 watersheds。兩個盆地交界的灰階，減去兩邊裡較低的盆地底，若小過 Threshold 就合成一個。交界像素會併進相鄰盆地。Threshold 0 幾乎不合併。彩色先轉成亮度。",
    "Vizion.watersheds_threshold (Image, Basins, 40)",
)
def watersheds_threshold(image, threshold, *, ctx):
    picture, gray, domain = _gray_domain(image)
    limit = scalar(threshold, "Threshold")
    if limit < 0:
        raise ValueError("Threshold 不能是負的")
    markers, count = _watershed_markers(gray, domain)
    mapped = _merge_basins(markers, gray, count, limit)
    objects = []
    for label in np.unique(mapped):
        if int(label) <= 0:
            continue
        mask = np.where(mapped == label, 255, 0).astype(np.uint8)
        if int(cv2.countNonZero(mask)):
            objects.append(mask)
    return Result([Region(picture.width, picture.height, objects, multi_color=len(objects) > 1)], [])


def _contours(mask: np.ndarray):
    found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]
    return contours


def _shape_mask(mask: np.ndarray, kind: str) -> np.ndarray:
    out = np.zeros_like(mask)
    if int(cv2.countNonZero(mask)) == 0:
        return out
    if kind == "rectangle1":
        ys, xs = np.where(mask > 0)
        out[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = 255
        return out
    contours = _contours(mask)
    if not contours:
        return out
    contour = max(contours, key=cv2.contourArea)
    if kind == "convex":
        hull = cv2.convexHull(np.vstack(contours))
        cv2.fillConvexPoly(out, hull, 255)
    elif kind == "outer_circle":
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        cv2.circle(out, (int(round(cx)), int(round(cy))), max(0, int(round(radius))), 255, -1)
    elif kind == "inner_circle":
        dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        _min, radius, _min_loc, center = cv2.minMaxLoc(dist)
        if radius > 0:
            cv2.circle(out, (int(center[0]), int(center[1])), int(math.floor(radius)), 255, -1)
    elif kind == "rectangle2":
        box = cv2.boxPoints(cv2.minAreaRect(contour))
        cv2.fillConvexPoly(out, np.int32(np.round(box)), 255)
    elif kind == "ellipse":
        moments = cv2.moments(mask)
        ra, rb, phi = equivalent_ellipse(moments)
        if moments["m00"] and ra >= 0.5:
            col = float(moments["m10"] / moments["m00"])
            row = float(moments["m01"] / moments["m00"])
            cv2.ellipse(out, ((col, row), (2.0 * ra, 2.0 * rb), math.degrees(phi)), 255, -1)
        elif contour is not None:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            cv2.circle(out, (int(round(cx)), int(round(cy))), max(0, int(round(radius))), 255, -1)
    return out


_SHAPES = ["'convex'", "'ellipse'", "'outer_circle'", "'inner_circle'", "'rectangle1'", "'rectangle2'"]


@operator(
    "區域",
    [P("Region", "輸入區域")],
    [P("RegionTrans", "轉換後的區域")],
    [P("Type", "convex、ellipse、outer_circle、inner_circle、rectangle1 或 rectangle2", "'convex'", _SHAPES)],
    [],
    "把區域換成規則形狀",
    "每個物件各自轉換。inner_circle 用距離轉換的最大內切圓，半徑無條件捨去。rectangle2 和 ellipse 用 OpenCV 的外接矩形與橢圓。沒有 inner_rectangle1。",
    "Vizion.shape_trans (Region, RegionTrans, 'rectangle1')",
)
def shape_trans(region, kind, *, ctx):
    source = expect_region(region, "Region")
    mode = text(kind, "Type").lower()
    if mode not in ("convex", "ellipse", "outer_circle", "inner_circle", "rectangle1", "rectangle2"):
        raise ValueError("Type 只能是 convex、ellipse、outer_circle、inner_circle、rectangle1 或 rectangle2")
    objects = []
    for mask in source.objects:
        shaped = _shape_mask(mask, mode)
        if int(cv2.countNonZero(shaped)):
            objects.append(shaped)
    return Result([Region(source.width, source.height, objects, multi_color=len(objects) > 1)], [])


def _region_box(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return 0, 0, 0, 0
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def _sorted_masks(objects: list, mode: str, order: str, axis: str) -> list:
    items = []
    for mask in objects:
        row1, col1, row2, col2 = _region_box(mask)
        items.append((mask, row1, col1, row2, col2))
    reverse = order == "false"

    def by_row(item):
        return (item[1], item[2])

    def by_col(item):
        return (item[2], item[1])

    if mode == "character" and axis == "row":
        items.sort(key=by_row)
        lines = []
        for item in items:
            placed = False
            for line in lines:
                top = min(entry[1] for entry in line)
                bottom = max(entry[3] for entry in line)
                overlap = min(item[3], bottom) - max(item[1], top) + 1
                height = max(1, min(item[3] - item[1] + 1, bottom - top + 1))
                if overlap >= 0.5 * height:
                    line.append(item)
                    placed = True
                    break
            if not placed:
                lines.append([item])
        if reverse:
            lines.reverse()
        ordered = []
        for line in lines:
            line.sort(key=lambda entry: entry[2], reverse=reverse)
            ordered.extend(line)
        return [item[0] for item in ordered]
    if mode == "character":
        items.sort(key=by_col)
        columns = []
        for item in items:
            placed = False
            for column in columns:
                left = min(entry[2] for entry in column)
                right = max(entry[4] for entry in column)
                overlap = min(item[4], right) - max(item[2], left) + 1
                width = max(1, min(item[4] - item[2] + 1, right - left + 1))
                if overlap >= 0.5 * width:
                    column.append(item)
                    placed = True
                    break
            if not placed:
                columns.append([item])
        if reverse:
            columns.reverse()
        ordered = []
        for column in columns:
            column.sort(key=lambda entry: entry[1], reverse=reverse)
            ordered.extend(column)
        return [item[0] for item in ordered]
    key = by_row if axis == "row" else by_col
    items.sort(key=key, reverse=reverse)
    return [item[0] for item in items]


@operator(
    "區域",
    [P("Regions", "物件集合")],
    [P("SortedRegions", "排序後的物件")],
    [
        P("SortMode", "character、upper_left 或 first_point", "'character'", ["'character'", "'upper_left'", "'first_point'"]),
        P("Order", "true 由小到大，false 反過來", "'true'", ["'true'", "'false'"]),
        P("RowOrCol", "先比列或先比行", "'row'", ["'row'", "'column'"]),
    ],
    [],
    "依位置排序區域",
    "character 會把同一列重疊超過一半的物件排成一行，再由左到右。upper_left 和 first_point 這版都用左上角。",
    "Vizion.sort_region (Regions, SortedRegions, 'character', 'true', 'row')",
)
def sort_region(region, sort_mode, order, row_or_col, *, ctx):
    source = expect_region(region, "Regions")
    mode = text(sort_mode, "SortMode").lower()
    direction = text(order, "Order").lower()
    axis = text(row_or_col, "RowOrCol").lower()
    if mode not in ("character", "upper_left", "first_point"):
        raise ValueError("SortMode 只能是 character、upper_left 或 first_point")
    if direction not in ("true", "false"):
        raise ValueError("Order 只能是 true 或 false")
    if axis not in ("row", "column"):
        raise ValueError("RowOrCol 只能是 row 或 column")
    objects = _sorted_masks(source.objects, mode, direction, axis)
    return Result([Region(source.width, source.height, objects, multi_color=len(objects) > 1)], [])


@operator(
    "區域",
    [P("Region", "輸入區域")],
    [P("RegionClipped", "裁切後的區域")],
    [P("Row1", "上"), P("Column1", "左"), P("Row2", "下"), P("Column2", "右")],
    [],
    "用矩形裁切區域",
    "矩形外的像素清掉。裁完是空的物件會去掉。",
    "Vizion.clip_region (Region, RegionClipped, 10, 10, 100, 100)",
)
def clip_region(region, row1, column1, row2, column2, *, ctx):
    source = expect_region(region, "Region")
    r1 = whole(row1, "Row1")
    c1 = whole(column1, "Column1")
    r2 = whole(row2, "Row2")
    c2 = whole(column2, "Column2")
    if r2 < r1 or c2 < c1:
        raise ValueError("矩形座標不正確")
    rr1 = max(0, min(source.height - 1, r1))
    cc1 = max(0, min(source.width - 1, c1))
    rr2 = max(0, min(source.height - 1, r2))
    cc2 = max(0, min(source.width - 1, c2))
    objects = []
    for mask in source.objects:
        clipped = np.zeros_like(mask)
        clipped[rr1 : rr2 + 1, cc1 : cc2 + 1] = mask[rr1 : rr2 + 1, cc1 : cc2 + 1]
        if int(cv2.countNonZero(clipped)):
            objects.append(clipped)
    return Result([Region(source.width, source.height, objects, multi_color=len(objects) > 1)], [])


def _canvas(ctx):
    height, width = ctx.canvas_size()
    return height, width


def _rect2_polygon(row: float, col: float, phi: float, length1: float, length2: float) -> np.ndarray:
    cosine = math.cos(phi)
    sine = math.sin(phi)
    along_col = length1 * cosine
    along_row = length1 * sine
    side_col = -length2 * sine
    side_row = length2 * cosine
    points = []
    for sign_along, sign_side in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        points.append([
            col + sign_along * along_col + sign_side * side_col,
            row + sign_along * along_row + sign_side * side_row,
        ])
    return np.int32(np.round(points))


@operator(
    "區域",
    [],
    [P("Rectangle", "旋轉矩形")],
    [
        P("Row", "中心列"),
        P("Column", "中心行"),
        P("Phi", "長邊角度，弳度。0 沿著行，正角度在畫面上順時針"),
        P("Length1", "角度方向的半長"),
        P("Length2", "垂直方向的半長"),
    ],
    [],
    "產生旋轉矩形區域",
    "尺寸對齊最近一張影像。請先讀圖或 gen_image_const。Phi 為 0 時 Length1 沿水平方向。",
    "Vizion.gen_rectangle2 (Rectangle, 100, 150, 0, 40, 15)",
)
def gen_rectangle2(row, column, phi, length1, length2, *, ctx):
    height, width = _canvas(ctx)
    a = scalar(length1, "Length1")
    b = scalar(length2, "Length2")
    if a < 0 or b < 0:
        raise ValueError("Length1 和 Length2 不能是負的")
    mask = np.zeros((height, width), np.uint8)
    polygon = _rect2_polygon(scalar(row, "Row"), scalar(column, "Column"), scalar(phi, "Phi"), a, b)
    cv2.fillConvexPoly(mask, polygon, 255)
    objects = [mask] if int(cv2.countNonZero(mask)) else []
    return Result([Region(width, height, objects)], [])


@operator(
    "區域",
    [],
    [P("Ellipse", "實心橢圓")],
    [
        P("Row", "中心列"),
        P("Column", "中心行"),
        P("Phi", "第一軸角度，弳度。0 沿著行"),
        P("Radius1", "第一軸半徑"),
        P("Radius2", "第二軸半徑"),
    ],
    [],
    "產生橢圓區域",
    "半徑會四捨五入成像素。正角度在畫面上順時針。尺寸對齊最近一張影像。",
    "Vizion.gen_ellipse (Ellipse, 120, 160, 0, 40, 20)",
)
def gen_ellipse(row, column, phi, radius1, radius2, *, ctx):
    height, width = _canvas(ctx)
    a = scalar(radius1, "Radius1")
    b = scalar(radius2, "Radius2")
    if a <= 0 or b <= 0:
        raise ValueError("半徑必須大於 0")
    mask = np.zeros((height, width), np.uint8)
    center = (whole(column, "Column"), whole(row, "Row"))
    axes = (max(1, int(round(a))), max(1, int(round(b))))
    cv2.ellipse(mask, center, axes, math.degrees(scalar(phi, "Phi")), 0, 360, 255, -1)
    objects = [mask] if int(cv2.countNonZero(mask)) else []
    return Result([Region(width, height, objects)], [])


@operator(
    "區域",
    [P("Region", "輸入區域")],
    [],
    [],
    [P("Rows", "前景的列"), P("Columns", "前景的行")],
    "讀出區域裡的像素座標",
    "依列再依行排列。多個物件會接在一起。空區域得到空 tuple。",
    "Vizion.get_region_points (Region, Rows, Columns)",
)
def get_region_points(region, *, ctx):
    source = expect_region(region, "Region")
    mask = source.union_mask()
    rows, cols = np.where(mask > 0)
    return Result([], [rows.astype(int).tolist(), cols.astype(int).tolist()])


@operator(
    "區域",
    [],
    [P("Region", "填滿的多邊形")],
    [P("Rows", "頂點的列"), P("Columns", "頂點的行")],
    [],
    "用多邊形產生區域",
    "至少三個點。尺寸對齊最近一張影像，請先讀圖或 gen_image_const。",
    "Vizion.gen_region_polygon (Region, [10, 10, 40], [10, 50, 30])",
)
def gen_region_polygon(rows, columns, *, ctx):
    height, width = _canvas(ctx)
    rr = [float(item) for item in as_list(rows)]
    cc = [float(item) for item in as_list(columns)]
    if len(rr) != len(cc) or len(rr) < 3:
        raise ValueError("Rows 和 Columns 至少要三個，而且長度要相同")
    points = np.int32(np.round(np.column_stack([cc, rr])))
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [points], 255)
    objects = [mask] if int(cv2.countNonZero(mask)) else []
    return Result([Region(width, height, objects)], [])


@operator(
    "區域",
    [P("Region", "輸入區域")],
    [P("Partitioned", "切成小塊的區域")],
    [P("Width", "每一塊的寬"), P("Height", "每一塊的高")],
    [],
    "把區域切成矩形格子",
    "沒有像素的格子會丟掉。寬高是每一格的像素數，不是格子的個數。",
    "Vizion.partition_rectangle (Region, Partitioned, 20, 20)",
)
def partition_rectangle(region, width, height, *, ctx):
    source = expect_region(region, "Region")
    tile_w = whole(width, "Width")
    tile_h = whole(height, "Height")
    if tile_w < 1 or tile_h < 1:
        raise ValueError("格子的寬高至少要 1")
    objects = []
    for mask in source.objects:
        for top in range(0, source.height, tile_h):
            for left in range(0, source.width, tile_w):
                if int(cv2.countNonZero(mask[top : top + tile_h, left : left + tile_w])) == 0:
                    continue
                piece = np.zeros_like(mask)
                piece[top : top + tile_h, left : left + tile_w] = mask[top : top + tile_h, left : left + tile_w]
                objects.append(piece)
    return Result([Region(source.width, source.height, objects, multi_color=len(objects) > 1)], [])


_GRAY_FEATURES = ["'mean'", "'deviation'", "'min'", "'max'"]


def _gray_stat(mask: np.ndarray, gray: np.ndarray, feature: str) -> float:
    values = gray[mask > 0]
    if values.size == 0:
        return 0.0
    if feature == "mean":
        return float(values.mean())
    if feature == "min":
        return float(values.min())
    if feature == "max":
        return float(values.max())
    return float(values.std())


@operator(
    "特徵",
    [P("Regions", "物件集合"), P("Image", "影像")],
    [P("SelectedRegions", "留下的物件")],
    [
        P("Features", "mean、deviation、min 或 max", "'mean'", _GRAY_FEATURES),
        P("Operation", "and 要全部成立，or 有一個成立", "'and'", ["'and'", "'or'"]),
        P("Min", "下限"),
        P("Max", "上限"),
    ],
    [],
    "依灰階留下物件",
    "一次可以給一個特徵，也可以給 tuple。下限和上限的個數要是 1，或和特徵一樣多。deviation 是母體標準差。",
    "Vizion.select_gray (Regions, Image, SelectedRegions, 'mean', 'and', 100, 255)",
)
def select_gray(region, image, features, operation, min_value, max_value, *, ctx):
    source = expect_region(region, "Regions")
    picture = expect_image(image, "Image")
    if (source.width, source.height) != (picture.width, picture.height):
        raise ValueError("影像和區域的尺寸不同")
    names = [text(item, "Features").lower() for item in as_list(features)]
    if not names or any(item not in ("mean", "deviation", "min", "max") for item in names):
        raise ValueError("Features 只能是 mean、deviation、min 或 max")
    mode = text(operation, "Operation").lower()
    if mode not in ("and", "or"):
        raise ValueError("Operation 只能是 and 或 or")
    lows = [float(item) for item in as_list(min_value)]
    highs = [float(item) for item in as_list(max_value)]
    if len(lows) == 1 and len(names) > 1:
        lows = lows * len(names)
    if len(highs) == 1 and len(names) > 1:
        highs = highs * len(names)
    if not (len(lows) == len(highs) == len(names)):
        raise ValueError("Min、Max 的個數要和 Features 相同")
    gray = picture.gray()
    kept = []
    for mask in source.objects:
        checks = [lows[index] <= _gray_stat(mask, gray, name) <= highs[index] for index, name in enumerate(names)]
        if all(checks) if mode == "and" else any(checks):
            kept.append(mask)
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "形態學",
    [P("Regions", "要長大的物件"), P("ForbiddenRegion", "不能進入的區域")],
    [P("RegionExpanded", "長大後的物件")],
    [P("Iterations", "長大幾次"), P("Mode", "這版只做 image", "'image'", ["'image'"])],
    [],
    "把區域往外長",
    "每次用 8 鄰域長一像素。不會進入 ForbiddenRegion，也不會讓物件互相覆蓋。Mode 這版只有 image。",
    "Vizion.expand_region (Regions, ForbiddenRegion, RegionExpanded, 2, 'image')",
)
def expand_region(region, forbidden, iterations, mode, *, ctx):
    source = expect_region(region, "Regions")
    blocked = expect_region(forbidden, "ForbiddenRegion")
    if (source.width, source.height) != (blocked.width, blocked.height):
        raise ValueError("兩個區域的尺寸不同")
    if text(mode, "Mode").lower() != "image":
        raise ValueError("Mode 這版只接受 image")
    steps = whole(iterations, "Iterations")
    if steps < 0:
        raise ValueError("Iterations 不能是負的")
    objects = [mask.copy() for mask in source.objects]
    wall = blocked.union_mask()
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(steps):
        occupied = wall.copy()
        for mask in objects:
            occupied = cv2.bitwise_or(occupied, mask)
        grown = []
        for mask in objects:
            dilated = cv2.dilate(mask, kernel)
            others = occupied.copy()
            others[mask > 0] = 0
            dilated[others > 0] = 0
            grown.append(dilated)
        objects = grown
    kept = [mask for mask in objects if int(cv2.countNonZero(mask))]
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


def _points_on_canvas(rows, columns, ctx):
    height, width = _canvas(ctx)
    rr = [int(round(float(item))) for item in as_list(rows)]
    cc = [int(round(float(item))) for item in as_list(columns)]
    if len(rr) != len(cc):
        raise ValueError("Rows 和 Columns 的長度要相同")
    if any(item < 0 or item >= height for item in rr) or any(item < 0 or item >= width for item in cc):
        raise ValueError("座標超出影像。請先讀圖或 gen_image_const")
    return height, width, rr, cc


@operator(
    "區域",
    [],
    [P("Region", "點組成的區域")],
    [P("Rows", "列"), P("Columns", "行")],
    [],
    "用像素點產生區域",
    "所有點合成一個區域。點不相連時，要用 connection 才會拆開。尺寸對齊最近一張影像。",
    "Vizion.gen_region_points (Region, [1, 2], [3, 4])",
)
def gen_region_points(rows, columns, *, ctx):
    height, width, rr, cc = _points_on_canvas(rows, columns, ctx)
    mask = np.zeros((height, width), np.uint8)
    for row, col in zip(rr, cc):
        mask[row, col] = 255
    objects = [mask] if int(cv2.countNonZero(mask)) else []
    return Result([Region(width, height, objects)], [])


@operator(
    "區域",
    [],
    [P("Region", "線段區域")],
    [
        P("BeginRow", "起點列"),
        P("BeginCol", "起點行"),
        P("EndRow", "終點列"),
        P("EndCol", "終點行"),
    ],
    [],
    "產生一像素寬的線段",
    "超出影像的部分會被裁掉。尺寸對齊最近一張影像。",
    "Vizion.gen_region_line (Region, 0, 0, 10, 0)",
)
def gen_region_line(begin_row, begin_col, end_row, end_col, *, ctx):
    height, width = _canvas(ctx)
    mask = np.zeros((height, width), np.uint8)
    start = (whole(begin_col, "BeginCol"), whole(begin_row, "BeginRow"))
    end = (whole(end_col, "EndCol"), whole(end_row, "EndRow"))
    cv2.line(mask, start, end, 255, 1)
    objects = [mask] if int(cv2.countNonZero(mask)) else []
    return Result([Region(width, height, objects)], [])


@operator(
    "區域",
    [P("Regions", "區域")],
    [],
    [P("Row", "列，可以是 tuple"), P("Column", "行，可以是 tuple")],
    [P("IsInside", "1 表示在區域裡")],
    "測試點是否在區域內",
    "多個物件合併判斷。座標會四捨五入。超出影像算不在裡面。一個點回傳 0 或 1，多個點回傳 tuple。",
    "Vizion.test_region_point (Regions, 5, 0, IsInside)",
)
def test_region_point(region, row, column, *, ctx):
    source = expect_region(region, "Regions")
    rows = [int(round(float(item))) for item in as_list(row)]
    cols = [int(round(float(item))) for item in as_list(column)]
    if len(rows) != len(cols) or not rows:
        raise ValueError("Row 和 Column 的長度要相同，而且不能是空的")
    mask = source.union_mask()
    flags = []
    for item_row, item_col in zip(rows, cols):
        inside = 0 <= item_row < source.height and 0 <= item_col < source.width and int(mask[item_row, item_col]) > 0
        flags.append(1 if inside else 0)
    if len(flags) == 1 and not isinstance(row, (list, tuple)):
        return Result([], [flags[0]])
    return Result([], [flags])


def _per_object(region: Region, reader):
    rows = []
    for mask in region.objects:
        rows.append(reader(mask))
    return rows


@operator(
    "特徵",
    [P("Regions", "物件集合")],
    [],
    [],
    [
        P("Row1", "直徑一端的列"),
        P("Column1", "直徑一端的行"),
        P("Row2", "另一端的列"),
        P("Column2", "另一端的行"),
        P("Diameter", "兩端距離"),
    ],
    "區域的最大直徑",
    "在凸包上找最遠的兩個點。每個物件一組，空物件的直徑是 0。",
    "Vizion.diameter_region (Regions, Row1, Column1, Row2, Column2, Diameter)",
)
def diameter_region(region, *, ctx):
    source = expect_region(region, "Regions")

    def measure(mask):
        contours, _hierarchy = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        hull = cv2.convexHull(max(contours, key=cv2.contourArea)).reshape(-1, 2).astype(np.float64)
        if len(hull) == 1:
            col, row = hull[0]
            return (float(row), float(col), float(row), float(col), 0.0)
        best = -1.0
        first = second = hull[0]
        for index, origin in enumerate(hull):
            delta = hull[index + 1 :] - origin
            if len(delta) == 0:
                continue
            dist = np.hypot(delta[:, 0], delta[:, 1])
            pick = int(np.argmax(dist))
            if float(dist[pick]) > best:
                best = float(dist[pick])
                first = origin
                second = hull[index + 1 + pick]
        return (float(first[1]), float(first[0]), float(second[1]), float(second[0]), best)

    values = _per_object(source, measure)
    columns = list(zip(*values)) if values else [()] * 5
    return Result([], [list(column) for column in columns])


@operator(
    "特徵",
    [P("Regions", "物件集合")],
    [],
    [],
    [P("Ra", "長軸半徑"), P("Rb", "短軸半徑"), P("Phi", "長軸角度，弳度")],
    "等效橢圓的兩軸",
    "用區域像素的二次矩，不是邊界擬合。0 沿著行往右，正角度在畫面上順時針。橫條是 0，直條是 1.5708。Ra、Rb 是等效橢圓的半徑。每個物件一組。",
    "Vizion.elliptic_axis (Regions, Ra, Rb, Phi)",
)
def elliptic_axis(region, *, ctx):
    source = expect_region(region, "Regions")
    values = _per_object(source, region_measures)
    return Result([], [[item["ra"] for item in values], [item["rb"] for item in values], [item["phi"] for item in values]])


@operator(
    "特徵",
    [P("Regions", "物件集合")],
    [],
    [],
    [P("Phi", "方向，弳度")],
    "區域的方向",
    "和 elliptic_axis 的 Phi 相同，用二次矩。橫條是 0，直條是 1.5708。每個物件一個角度。",
    "Vizion.orientation_region (Regions, Phi)",
)
def orientation_region(region, *, ctx):
    source = expect_region(region, "Regions")
    return Result([], [[region_measures(mask)["orientation"] for mask in source.objects]])


_STD_SHAPES = ["'max_area'", "'rectangle'", "'convex'"]


@operator(
    "特徵",
    [P("Regions", "物件集合")],
    [P("SelectedRegions", "留下的物件")],
    [
        P("Shape", "max_area、rectangle 或 convex", "'max_area'", _STD_SHAPES),
        P("Percent", "0 到 100"),
    ],
    [],
    "用標準形留下物件",
    "max_area 留下面積達到最大面積這個百分比的物件。rectangle 用矩形度，convex 用凸度，Percent 80 表示至少要有 0.8。",
    "Vizion.select_shape_std (Regions, SelectedRegions, 'max_area', 100)",
)
def select_shape_std(region, shape, percent, *, ctx):
    source = expect_region(region, "Regions")
    kind = text(shape, "Shape").lower()
    ratio = scalar(percent, "Percent")
    if ratio < 0 or ratio > 100:
        raise ValueError("Percent 要在 0 到 100")
    if kind == "max_area":
        areas = [object_feature(mask, "area") for mask in source.objects]
        limit = (max(areas) if areas else 0.0) * ratio / 100.0
        kept = [mask for mask, area in zip(source.objects, areas) if area + 1e-6 >= limit]
    elif kind == "rectangle":
        kept = [mask for mask in source.objects if object_feature(mask, "rectangularity") + 1e-6 >= ratio / 100.0]
    elif kind == "convex":
        kept = [mask for mask in source.objects if object_feature(mask, "convexity") + 1e-6 >= ratio / 100.0]
    else:
        raise ValueError("Shape 只能是 max_area、rectangle 或 convex")
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


def _hole_masks(mask: np.ndarray) -> list[np.ndarray]:
    work = np.where(mask > 0, 0, 255).astype(np.uint8)
    flood = np.zeros((work.shape[0] + 2, work.shape[1] + 2), np.uint8)
    height, width = work.shape
    for col in range(width):
        if work[0, col] == 255:
            cv2.floodFill(work, flood, (col, 0), 0)
        if work[height - 1, col] == 255:
            cv2.floodFill(work, flood, (col, height - 1), 0)
    for row in range(height):
        if work[row, 0] == 255:
            cv2.floodFill(work, flood, (0, row), 0)
        if work[row, width - 1] == 255:
            cv2.floodFill(work, flood, (width - 1, row), 0)
    holes = np.where(work == 255, 255, 0).astype(np.uint8)
    count, labels = cv2.connectedComponents(holes)
    return [np.where(labels == index, 255, 0).astype(np.uint8) for index in range(1, count)]


_FILL_FEATURES = ["'area'", "'circularity'", "'compactness'", "'ra'", "'rb'", "'anisometry'"]


@operator(
    "形態學",
    [P("Region", "輸入區域")],
    [P("RegionFillUp", "補上符合條件的孔洞")],
    [
        P("Feature", "用來判斷孔洞的特徵", "'area'", _FILL_FEATURES),
        P("Min", "下限，含"),
        P("Max", "上限，含"),
    ],
    [],
    "依特徵填孔洞",
    "只填不連到影像邊緣、而且特徵落在範圍裡的洞。孔洞面積含內部像素。結果是單一區域。",
    "Vizion.fill_up_shape (Region, RegionFillUp, 'area', 1, 99999)",
)
def fill_up_shape(region, feature, min_value, max_value, *, ctx):
    source = expect_region(region, "Region")
    low = scalar(min_value, "Min")
    high = scalar(max_value, "Max")
    mask = source.union_mask()
    filled = mask.copy()
    for hole in _hole_masks(mask):
        value = object_feature(hole, feature)
        if low <= value <= high:
            filled = cv2.bitwise_or(filled, hole)
    return region_out(source.width, source.height, filled)


@operator(
    "量測",
    [P("Region", "區域")],
    [],
    [P("Row", "列，可以是 tuple"), P("Column", "行，可以是 tuple")],
    [P("DistanceMin", "最近距離"), P("DistanceMax", "最遠距離")],
    "點到區域的最近和最遠距離",
    "點在區域裡時最近距離是 0。最遠距離量到凸包頂點。點必須在影像內。一個點回傳兩個數，多個點回傳 tuple。",
    "Vizion.distance_pr (Region, 0, 0, DistanceMin, DistanceMax)",
)
def distance_pr(region, row, column, *, ctx):
    source = expect_region(region, "Region")
    rows = [int(round(float(item))) for item in as_list(row)]
    cols = [int(round(float(item))) for item in as_list(column)]
    if len(rows) != len(cols) or not rows:
        raise ValueError("Row 和 Column 的長度要相同，而且不能是空的")
    if any(item < 0 or item >= source.height for item in rows) or any(item < 0 or item >= source.width for item in cols):
        raise ValueError("座標超出影像")
    mask = source.union_mask()
    if int(cv2.countNonZero(mask)) == 0:
        raise ValueError("區域是空的")
    found = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]
    hull = cv2.convexHull(np.vstack(contours)).reshape(-1, 2).astype(np.float64)
    nearest = cv2.distanceTransform(np.where(mask > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 5)
    mins, maxs = [], []
    for item_row, item_col in zip(rows, cols):
        mins.append(float(nearest[item_row, item_col]))
        maxs.append(float(np.max(np.hypot(hull[:, 0] - item_col, hull[:, 1] - item_row))))
    if len(mins) == 1 and not isinstance(row, (list, tuple)):
        return Result([], [mins[0], maxs[0]])
    return Result([], [mins, maxs])


def _hull_points(mask: np.ndarray) -> np.ndarray:
    found = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]
    if not contours:
        raise ValueError("區域是空的")
    return cv2.convexHull(np.vstack(contours)).reshape(-1, 2).astype(np.float64)


@operator(
    "量測",
    [P("Region1", "區域一"), P("Region2", "區域二")],
    [],
    [],
    [P("DistanceMin", "最近距離"), P("DistanceMax", "最遠距離")],
    "兩個區域的最近和最遠距離",
    "有重疊時最近距離是 0。最遠距離量兩個凸包的頂點。空區域會失敗。",
    "Vizion.distance_rr (Region1, Region2, DistanceMin, DistanceMax)",
)
def distance_rr(region1, region2, *, ctx):
    first = expect_region(region1, "Region1")
    second = expect_region(region2, "Region2")
    same_size = (first.width, first.height) == (second.width, second.height)
    if not same_size:
        raise ValueError("兩個區域的尺寸不同")
    mask1 = first.union_mask()
    mask2 = second.union_mask()
    if int(cv2.countNonZero(mask1)) == 0 or int(cv2.countNonZero(mask2)) == 0:
        raise ValueError("區域是空的")
    nearest = cv2.distanceTransform(np.where(mask2 > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 5)
    distance_min = float(nearest[mask1 > 0].min())
    hull1 = _hull_points(mask1)
    hull2 = _hull_points(mask2)
    delta_col = hull1[:, None, 0] - hull2[None, :, 0]
    delta_row = hull1[:, None, 1] - hull2[None, :, 1]
    return Result([], [distance_min, float(np.hypot(delta_col, delta_row).max())])


@operator(
    "區域",
    [P("Region", "輸入區域")],
    [],
    [],
    [P("Rows", "外輪廓的列"), P("Columns", "外輪廓的行")],
    "讀出區域外輪廓",
    "每個物件取最大的外輪廓，再依序接在一起。不含孔洞。空區域會失敗。",
    "Vizion.get_region_contour (Region, Rows, Columns)",
)
def get_region_contour(region, *, ctx):
    source = expect_region(region, "Region")
    rows, cols = [], []
    for mask in source.objects:
        found = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = found[0] if len(found) == 2 else found[1]
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
        cols.extend(int(item) for item in contour[:, 0])
        rows.extend(int(item) for item in contour[:, 1])
    if not rows:
        raise ValueError("區域是空的")
    return Result([], [rows, cols])


_PROTO_FEATURES = ["'area'", "'circularity'", "'compactness'", "'ra'", "'rb'", "'rectangularity'"]


@operator(
    "特徵",
    [P("Regions", "物件集合"), P("Pattern", "範本區域")],
    [P("SelectedRegions", "留下的物件")],
    [
        P("Feature", "用來比較的特徵", "'area'", _PROTO_FEATURES),
        P("Min", "比值下限"),
        P("Max", "比值上限"),
    ],
    [],
    "和範本比對形狀",
    "比值是物件特徵除以範本特徵。範本有多個物件時只用第一個。範本特徵是 0 會失敗。",
    "Vizion.select_shape_proto (Regions, Pattern, SelectedRegions, 'area', 0.5, 1.5)",
)
def select_shape_proto(region, pattern, feature, min_value, max_value, *, ctx):
    source = expect_region(region, "Regions")
    model = expect_region(pattern, "Pattern")
    if (source.width, source.height) != (model.width, model.height):
        raise ValueError("區域和範本的尺寸不同")
    if not model.objects:
        raise ValueError("範本是空的")
    reference = object_feature(model.objects[0], feature)
    if abs(reference) < 1e-9:
        raise ValueError("範本的特徵是 0")
    low = scalar(min_value, "Min")
    high = scalar(max_value, "Max")
    kept = []
    for mask in source.objects:
        ratio = object_feature(mask, feature) / reference
        if low <= ratio <= high:
            kept.append(mask)
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "特徵",
    [P("Regions", "物件集合")],
    [],
    [],
    [P("M11", "中心二次矩"), P("M20", "列方向的二次矩"), P("M02", "行方向的二次矩")],
    "區域的中心二次矩",
    "對重心計算，平移不會改變。M20 是列的散布，M02 是行的散布。每個物件一組。",
    "Vizion.moments_region_2nd (Regions, M11, M20, M02)",
)
def moments_region_2nd(region, *, ctx):
    source = expect_region(region, "Regions")
    m11, m20, m02 = [], [], []
    for mask in source.objects:
        moments = cv2.moments((mask > 0).astype(np.uint8))
        m11.append(float(moments["mu11"]))
        m20.append(float(moments["mu02"]))
        m02.append(float(moments["mu20"]))
    return Result([], [m11, m20, m02])


@operator(
    "區域",
    [P("Region", "輸入區域")],
    [P("RegionProjTrans", "轉換後區域")],
    [
        P("HomMat2D", "投影矩陣"),
        P("Interpolation", "nearest_neighbor 或 constant", "'nearest_neighbor'", ["'nearest_neighbor'", "'constant'"]),
    ],
    [],
    "用投影矩陣轉換區域",
    "和 projective_trans_image 同一套矩陣。6 個數會當成仿射。畫布大小不變，轉出去的部分切掉。區域只用最近鄰。",
    "Vizion.projective_trans_region (Region, RegionProjTrans, HomMat2D, 'nearest_neighbor')",
)
def projective_trans_region(region, hom_mat2d, interpolation, *, ctx):
    source = expect_region(region, "Region")
    kind = text(interpolation, "Interpolation").lower()
    if kind not in ("nearest_neighbor", "constant"):
        raise ValueError("區域轉換的 Interpolation 只能是 nearest_neighbor 或 constant")
    mapping = _opencv_projective(_proj_from(hom_mat2d))
    objects = []
    for mask in source.objects:
        warped = cv2.warpPerspective(mask, mapping, (source.width, source.height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped = np.where(warped > 0, 255, 0).astype(np.uint8)
        if int(cv2.countNonZero(warped)):
            objects.append(warped)
    return Result([Region(source.width, source.height, objects, multi_color=source.multi_color and len(objects) > 1)], [])


@operator(
    "區域",
    [P("Region", "輸入區域")],
    [P("Transposed", "轉置後的區域")],
    [P("Row", "對角軸通過的列"), P("Column", "對角軸通過的行")],
    [],
    "對一條斜軸做轉置",
    "點 (r, c) 會跑到 (Row + c − Column, Column + r − Row)。軸是 (0, 0) 時，列和行對調。轉出影像的點會被丟掉。",
    "Vizion.transpose_region (Region, Transposed, 0, 0)",
)
def transpose_region(region, row, column, *, ctx):
    source = expect_region(region, "Region")
    origin_row = scalar(row, "Row")
    origin_col = scalar(column, "Column")
    objects = []
    for mask in source.objects:
        rows, cols = np.where(mask > 0)
        new_rows = np.rint(origin_row + cols - origin_col).astype(int)
        new_cols = np.rint(origin_col + rows - origin_row).astype(int)
        keep = (new_rows >= 0) & (new_rows < source.height) & (new_cols >= 0) & (new_cols < source.width)
        moved = np.zeros_like(mask)
        moved[new_rows[keep], new_cols[keep]] = 255
        if int(cv2.countNonZero(moved)):
            objects.append(moved)
    return Result([Region(source.width, source.height, objects, multi_color=len(objects) > 1)], [])


@operator(
    "區域",
    [],
    [P("RegionGrid", "格線或格子")],
    [
        P("RowSteps", "列方向的間距"),
        P("ColumnSteps", "行方向的間距"),
        P("Type", "lines 是線，rects 是一格一個物件", "'lines'", ["'lines'", "'rects'"]),
        P("Width", "影像寬"),
        P("Height", "影像高"),
    ],
    [],
    "產生格線區域",
    "lines 把間距上的橫線和直線畫成一個區域。rects 每一格是獨立物件，最後一格不滿也會留下。間距至少是 1。",
    "Vizion.gen_grid_region (RegionGrid, 10, 10, 'lines', 80, 60)",
)
def gen_grid_region(row_steps, column_steps, kind, width, height, *, ctx):
    step_row = whole(row_steps, "RowSteps")
    step_col = whole(column_steps, "ColumnSteps")
    wide = whole(width, "Width")
    high = whole(height, "Height")
    if step_row < 1 or step_col < 1 or wide < 1 or high < 1:
        raise ValueError("間距和寬高至少要 1")
    pattern = text(kind, "Type").lower()
    if pattern == "lines":
        mask = np.zeros((high, wide), np.uint8)
        mask[::step_row, :] = 255
        mask[:, ::step_col] = 255
        objects = [mask]
    elif pattern == "rects":
        objects = []
        for row in range(0, high, step_row):
            for col in range(0, wide, step_col):
                mask = np.zeros((high, wide), np.uint8)
                mask[row : row + step_row, col : col + step_col] = 255
                objects.append(mask)
    else:
        raise ValueError("Type 只能是 lines 或 rects")
    return Result([Region(wide, high, objects, multi_color=len(objects) > 1)], [])


@operator(
    "區域",
    [P("Region", "細線區域")],
    [P("EndPoints", "端點"), P("JuncPoints", "交叉點")],
    [],
    [],
    "骨架的端點和交叉點",
    "沿著八個鄰居走一圈，0 變 1 的次數是 1 就是端點，3 次以上是交叉。這樣十字中心旁邊的點不會被算成交叉。請先用 skeleton。",
    "Vizion.junctions_skeleton (Region, EndPoints, JuncPoints)",
)
def junctions_skeleton(region, *, ctx):
    source = expect_region(region, "Region")
    on = (source.union_mask() > 0).astype(np.uint8)
    padded = np.pad(on, 1)
    height, width = on.shape
    shifts = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    neighbors = [padded[1 + drow : 1 + drow + height, 1 + dcol : 1 + dcol + width] for drow, dcol in shifts]
    ring = np.stack(neighbors + [neighbors[0]], axis=0)
    crossings = np.sum((ring[:-1] == 0) & (ring[1:] == 1), axis=0)
    ends = np.where((on > 0) & (crossings == 1), 255, 0).astype(np.uint8)
    joints = np.where((on > 0) & (crossings >= 3), 255, 0).astype(np.uint8)
    return Result(
        [
            Region(source.width, source.height, [ends] if int(cv2.countNonZero(ends)) else []),
            Region(source.width, source.height, [joints] if int(cv2.countNonZero(joints)) else []),
        ],
        [],
    )


@operator(
    "區域",
    [P("Region", "輸入區域")],
    [],
    [],
    [P("Row", "每一段的列"), P("ColumnBegin", "每一段的起始行"), P("ColumnEnd", "每一段的結束行，含")],
    "讀出區域的水平線段",
    "同一列上連續的像素合成一段。多個物件會先合併。沒有像素時得到空 tuple。",
    "Vizion.get_region_runs (Region, Row, ColumnBegin, ColumnEnd)",
)
def get_region_runs(region, *, ctx):
    source = expect_region(region, "Region")
    mask = source.union_mask() > 0
    rows, begins, ends = [], [], []
    for row, line in enumerate(mask):
        if not line.any():
            continue
        padded = np.pad(line, (1, 1))
        change = np.diff(padded.astype(np.int8))
        for start, stop in zip(np.where(change == 1)[0], np.where(change == -1)[0] - 1):
            rows.append(int(row))
            begins.append(int(start))
            ends.append(int(stop))
    return Result([], [rows, begins, ends])


@operator(
    "區域",
    [],
    [P("Region", "畫出來的區域")],
    [P("Row", "每一段的列"), P("ColumnBegin", "起始行"), P("ColumnEnd", "結束行，含")],
    [],
    "用水平線段畫區域",
    "三段的長度要相同。超出目前影像的部分會切掉。影像大小沿用已經建立的影像。",
    "Vizion.gen_region_runs (Region, Row, ColumnBegin, ColumnEnd)",
)
def gen_region_runs(row, column_begin, column_end, *, ctx):
    rows = [int(round(float(item))) for item in as_list(row)]
    begins = [int(round(float(item))) for item in as_list(column_begin)]
    ends = [int(round(float(item))) for item in as_list(column_end)]
    if not rows or len(rows) != len(begins) or len(rows) != len(ends):
        raise ValueError("Row、ColumnBegin、ColumnEnd 的長度要相同，而且不能是空的")
    height, width = _canvas(ctx)
    mask = np.zeros((height, width), np.uint8)
    for item_row, begin, end in zip(rows, begins, ends):
        if end < begin:
            begin, end = end, begin
        if item_row < 0 or item_row >= height:
            continue
        left = max(0, begin)
        right = min(width - 1, end)
        if right >= left:
            mask[item_row, left : right + 1] = 255
    return region_out(width, height, mask)


@operator(
    "區域",
    [P("Region", "輸入區域")],
    [P("RegionClipped", "裁切後區域")],
    [
        P("Top", "從外接矩形上方去掉的列數"),
        P("Bottom", "從下方去掉的列數"),
        P("Left", "從左方去掉的行數"),
        P("Right", "從右方去掉的行數"),
    ],
    [],
    "依外接矩形的邊距裁切",
    "每個物件用自己的外接矩形。四個邊距不能是負的。切完沒有像素就得到空區域。",
    "Vizion.clip_region_rel (Region, RegionClipped, 2, 2, 5, 5)",
)
def clip_region_rel(region, top, bottom, left, right, *, ctx):
    source = expect_region(region, "Region")
    margins = [whole(top, "Top"), whole(bottom, "Bottom"), whole(left, "Left"), whole(right, "Right")]
    if any(item < 0 for item in margins):
        raise ValueError("邊距不能是負的")
    cut_top, cut_bottom, cut_left, cut_right = margins
    objects = []
    for mask in source.objects:
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            continue
        row1 = int(ys.min()) + cut_top
        row2 = int(ys.max()) - cut_bottom
        col1 = int(xs.min()) + cut_left
        col2 = int(xs.max()) - cut_right
        if row2 < row1 or col2 < col1:
            continue
        clipped = np.zeros_like(mask)
        clipped[row1 : row2 + 1, col1 : col2 + 1] = mask[row1 : row2 + 1, col1 : col2 + 1]
        if int(cv2.countNonZero(clipped)):
            objects.append(clipped)
    return Result([Region(source.width, source.height, objects, multi_color=len(objects) > 1)], [])


@operator(
    "區域",
    [P("Region", "輸入區域")],
    [],
    [],
    [P("Row", "圓心列"), P("Column", "圓心行"), P("Radius", "半徑")],
    "區域裡最大的內切圓",
    "用距離轉換找離邊界最遠的點。有多個一樣大時，取掃描時遇到的第一個。空區域會失敗。多個物件會先合併。",
    "Vizion.inner_circle (Region, Row, Column, Radius)",
)
def inner_circle(region, *, ctx):
    source = expect_region(region, "Region")
    mask = source.union_mask()
    if int(cv2.countNonZero(mask)) == 0:
        raise ValueError("區域是空的")
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    index = int(np.argmax(distance))
    row, col = divmod(index, source.width)
    return Result([], [int(row), int(col), float(distance[row, col])])


def _largest_axis_rect(mask: np.ndarray):
    rows = (mask > 0).tolist()
    if not rows or not any(rows):
        return None
    width = len(rows[0])
    heights = [0] * width
    best_area = 0
    best = None
    for row_index, row in enumerate(rows):
        for col in range(width):
            heights[col] = heights[col] + 1 if row[col] else 0
        stack = [-1]
        for col in range(width + 1):
            current = heights[col] if col < width else 0
            while stack[-1] != -1 and heights[stack[-1]] > current:
                top = stack.pop()
                high = heights[top]
                left = stack[-1]
                wide = col - left - 1
                area = high * wide
                if area > best_area:
                    best_area = area
                    best = (row_index - high + 1, left + 1, row_index, col - 1)
            stack.append(col)
    return best


@operator(
    "區域",
    [P("Region", "輸入區域")],
    [],
    [],
    [P("Row1", "上"), P("Column1", "左"), P("Row2", "下"), P("Column2", "右")],
    "區域裡最大的軸對齊內接矩形",
    "矩形必須整塊都在區域裡。每個物件一組。有多個一樣大時，保留掃描時先遇到的那個。空區域會失敗。",
    "Vizion.inner_rectangle1 (Region, Row1, Column1, Row2, Column2)",
)
def inner_rectangle1(region, *, ctx):
    source = expect_region(region, "Region")
    if not source.objects:
        raise ValueError("區域是空的")
    row1, col1, row2, col2 = [], [], [], []
    for mask in source.objects:
        found = _largest_axis_rect(mask)
        if found is None:
            continue
        row1.append(int(found[0]))
        col1.append(int(found[1]))
        row2.append(int(found[2]))
        col2.append(int(found[3]))
    if not row1:
        raise ValueError("區域是空的")
    return Result([], [row1, col1, row2, col2])


@operator(
    "區域",
    [P("Regions1", "區域一"), P("Regions2", "區域二")],
    [],
    [],
    [P("Distance", "不一樣的像素數"), P("Similarity", "相似程度，0 到 1")],
    "兩個區域差幾個像素",
    "Distance 是只有一邊有的像素數。Similarity 是 1 減去 Distance 除以兩邊面積的和。多個物件會先合併。尺寸要相同。",
    "Vizion.hamming_distance (Regions1, Regions2, Distance, Similarity)",
)
def hamming_distance(region1, region2, *, ctx):
    first = expect_region(region1, "Regions1")
    second = expect_region(region2, "Regions2")
    if (first.width, first.height) != (second.width, second.height):
        raise ValueError("兩個區域的尺寸要相同")
    left = first.union_mask() > 0
    right = second.union_mask() > 0
    distance = int(np.count_nonzero(left ^ right))
    total = int(left.sum() + right.sum())
    similarity = 1.0 if total == 0 else 1.0 - distance / total
    return Result([], [distance, similarity])


@operator(
    "特徵",
    [P("Regions", "物件集合")],
    [],
    [],
    [
        P("M21", "列平方乘行"),
        P("M12", "列乘行平方"),
        P("M03", "行的三次矩"),
        P("M30", "列的三次矩"),
    ],
    "區域的中心三次矩",
    "對重心計算。M30 是列的偏斜，M03 是行的偏斜。左右對稱時 M03 接近 0。每個物件一組。",
    "Vizion.moments_region_3rd (Regions, M21, M12, M03, M30)",
)
def moments_region_3rd(region, *, ctx):
    source = expect_region(region, "Regions")
    m21, m12, m03, m30 = [], [], [], []
    for mask in source.objects:
        moments = cv2.moments((mask > 0).astype(np.uint8))
        m21.append(float(moments["mu12"]))
        m12.append(float(moments["mu21"]))
        m03.append(float(moments["mu30"]))
        m30.append(float(moments["mu03"]))
    return Result([], [m21, m12, m03, m30])


def _partition_ranges(projection: np.ndarray, start: int, end: int, distance: float, percent: float):
    low = max(1, int(round(distance * (1.0 - percent / 100.0))))
    high = max(low, int(round(distance * (1.0 + percent / 100.0))))
    ranges = []
    cursor = start
    for _ in range(end - start + 2):
        if cursor > end:
            break
        if end - cursor + 1 <= high:
            ranges.append((cursor, end))
            break
        window_a = min(end, cursor + low)
        window_b = min(end, cursor + high)
        if window_a > window_b:
            ranges.append((cursor, end))
            break
        cut = window_a + int(np.argmin(projection[window_a : window_b + 1]))
        if cut <= cursor:
            ranges.append((cursor, end))
            break
        ranges.append((cursor, cut - 1))
        cursor = cut
        if projection[cut] == 0:
            while cursor <= end and projection[cursor] == 0:
                cursor += 1
    return ranges


@operator(
    "區域",
    [P("Region", "輸入區域")],
    [P("Partitioned", "切開後的物件")],
    [P("Distance", "每一段大約的寬度"), P("Percent", "寬度可以差幾個百分點，0 到 100")],
    [],
    "依大約寬度切開區域",
    "沿著較長的那一邊切。會優先切在空隙上。實際寬度大約是 Distance 的上下 Percent%。每個物件分開切。這版沒有斜向切割。",
    "Vizion.partition_dynamic (Region, Partitioned, 20, 40)",
)
def partition_dynamic(region, distance, percent, *, ctx):
    source = expect_region(region, "Region")
    width_target = scalar(distance, "Distance")
    spread = scalar(percent, "Percent")
    if width_target < 1:
        raise ValueError("Distance 至少要 1")
    if not 0 <= spread <= 100:
        raise ValueError("Percent 要在 0 到 100")
    objects = []
    for mask in source.objects:
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            continue
        row1, row2 = int(ys.min()), int(ys.max())
        col1, col2 = int(xs.min()), int(xs.max())
        along_columns = (col2 - col1) >= (row2 - row1)
        projection = (mask > 0).sum(axis=0 if along_columns else 1)
        start, end = (col1, col2) if along_columns else (row1, row2)
        for begin, finish in _partition_ranges(projection, start, end, width_target, spread):
            piece = np.zeros_like(mask)
            if along_columns:
                piece[:, begin : finish + 1] = mask[:, begin : finish + 1]
            else:
                piece[begin : finish + 1, :] = mask[begin : finish + 1, :]
            if int(cv2.countNonZero(piece)):
                objects.append(piece)
    return Result([Region(source.width, source.height, objects, multi_color=len(objects) > 1)], [])


@operator(
    "特徵",
    [P("Regions", "物件集合")],
    [],
    [],
    [P("Area", "孔洞面積")],
    "孔洞的面積",
    "每個物件自己的孔。碰到影像邊緣的缺口不算孔洞。沒有孔時是 0。",
    "Vizion.area_holes (Regions, Area)",
)
def area_holes(region, *, ctx):
    source = expect_region(region, "Regions")
    areas = []
    for mask in source.objects:
        padded = np.pad(np.where(mask > 0, 255, 0).astype(np.uint8), 1)
        flood = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
        cv2.floodFill(padded, flood, (0, 0), 128)
        areas.append(int(np.count_nonzero(padded[1:-1, 1:-1] == 0)))
    return Result([], [areas])


@operator(
    "區域",
    [P("Regions", "物件集合")],
    [P("DestRegions", "含有這些點的物件")],
    [P("Row", "列，可以是 tuple"), P("Column", "行，可以是 tuple")],
    [],
    "留下包含指定點的物件",
    "只要碰到其中一個點就留下。點可以是 tuple。沒有物件含這些點時得到空區域。",
    "Vizion.select_region_point (Regions, DestRegions, 10, 20)",
)
def select_region_point(region, row, column, *, ctx):
    source = expect_region(region, "Regions")
    rows = [int(round(float(item))) for item in as_list(row)]
    cols = [int(round(float(item))) for item in as_list(column)]
    if len(rows) != len(cols) or not rows:
        raise ValueError("Row 和 Column 的長度要相同，而且不能是空的")
    kept = []
    for mask in source.objects:
        if any(
            0 <= item_row < source.height and 0 <= item_col < source.width and int(mask[item_row, item_col]) > 0
            for item_row, item_col in zip(rows, cols)
        ):
            kept.append(mask)
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [],
    [P("EmptyRegion", "空區域")],
    [],
    [],
    "產生空區域",
    "沿用目前影像的寬高，裡面沒有物件。還沒有影像時用 480×640。",
    "Vizion.gen_empty_region (EmptyRegion)",
)
def gen_empty_region(*, ctx):
    height, width = _canvas(ctx)
    return Result([Region(width, height, [])], [])


@operator(
    "輪廓",
    [P("Edges", "斷掉的邊緣"), P("Image", "原圖")],
    [P("RegionResult", "補上缺口之後")],
    [P("MinAmplitude", "梯度至少要這麼高")],
    [],
    "沿著高梯度補邊緣缺口",
    "Sobel 振幅夠高、而且和原本的邊緣連在一起的像素會補進去。平坦的地方不會被補上。這版沒有另外指定缺口長度。彩色先轉成亮度。",
    "Vizion.close_edges (Edges, Image, RegionResult, 20)",
)
def close_edges(edges, image, min_amplitude, *, ctx):
    seeds = expect_region(edges, "Edges")
    picture, gray, domain = _gray_domain(image)
    if (picture.width, picture.height) != (seeds.width, seeds.height):
        raise ValueError("邊緣和影像的尺寸不同")
    limit = scalar(min_amplitude, "MinAmplitude")
    if limit < 0:
        raise ValueError("MinAmplitude 不能是負的")
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    amplitude = cv2.magnitude(dx, dy)
    allowed = np.where((amplitude >= limit) & (domain > 0), 255, 0).astype(np.uint8)
    seed = cv2.bitwise_and(seeds.union_mask(), allowed)
    if int(cv2.countNonZero(seed)) == 0:
        return region_out(picture.width, picture.height, np.zeros_like(allowed))
    count, labels = cv2.connectedComponents(allowed, connectivity=8)
    keep = np.zeros(count, np.uint8)
    keep[np.unique(labels[seed > 0])] = 1
    keep[0] = 0
    mask = np.where(keep[labels] > 0, 255, 0).astype(np.uint8)
    return region_out(picture.width, picture.height, mask)


@operator(
    "區域",
    [P("Region1", "區域一"), P("Region2", "區域二")],
    [P("RegionDifference", "只有一邊有的像素")],
    [],
    [],
    "兩個區域的對稱差",
    "留下只屬於其中一邊的像素。重疊的部分會拿掉。多個物件會先合併。尺寸要相同。",
    "Vizion.symm_difference (Region1, Region2, RegionDifference)",
)
def symm_difference(region1, region2, *, ctx):
    first = expect_region(region1, "Region1")
    second = expect_region(region2, "Region2")
    if (first.width, first.height) != (second.width, second.height):
        raise ValueError("兩個區域的尺寸要相同")
    mask = np.where((first.union_mask() > 0) ^ (second.union_mask() > 0), 255, 0).astype(np.uint8)
    return region_out(first.width, first.height, mask)


@operator(
    "區域",
    [P("Regions", "物件集合")],
    [],
    [P("Row", "列，可以是 tuple"), P("Column", "行，可以是 tuple")],
    [P("Index", "物件編號，從 1 開始。沒有是 0")],
    "點落在第幾個物件",
    "重疊時回傳編號最小的那個。一個點回傳一個數字，多個點回傳 tuple。超出影像是 0。",
    "Vizion.get_region_index (Regions, 10, 20, Index)",
)
def get_region_index(region, row, column, *, ctx):
    source = expect_region(region, "Regions")
    rows = [int(round(float(item))) for item in as_list(row)]
    cols = [int(round(float(item))) for item in as_list(column)]
    if len(rows) != len(cols) or not rows:
        raise ValueError("Row 和 Column 的長度要相同，而且不能是空的")
    indices = []
    for item_row, item_col in zip(rows, cols):
        found = 0
        if 0 <= item_row < source.height and 0 <= item_col < source.width:
            for index, mask in enumerate(source.objects, start=1):
                if int(mask[item_row, item_col]) > 0:
                    found = index
                    break
        indices.append(found)
    if len(indices) == 1 and not isinstance(row, (list, tuple)):
        return Result([], [indices[0]])
    return Result([], [indices])


def _region_gap(mask, other, limit: float) -> bool:
    if int(cv2.countNonZero(other)) == 0 or int(cv2.countNonZero(mask)) == 0:
        return False
    distance = cv2.distanceTransform(np.where(mask > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 5)
    return float(distance[other > 0].min()) <= limit + 1e-6


@operator(
    "區域",
    [P("Regions1", "第一組"), P("Regions2", "第二組")],
    [P("Neighbors1", "第一組裡有鄰居的"), P("Neighbors2", "第二組裡有鄰居的")],
    [P("MaxDistance", "最遠距離，像素")],
    [],
    "找出距離夠近的物件",
    "某一邊的物件，只要和另一邊的某個物件距離不超過 MaxDistance，就會留下。距離 0 表示要重疊，挨著但沒重疊至少要 1。兩組的尺寸要相同。",
    "Vizion.find_neighbors (Regions1, Regions2, Neighbors1, Neighbors2, 5)",
)
def find_neighbors(region1, region2, max_distance, *, ctx):
    first = expect_region(region1, "Regions1")
    second = expect_region(region2, "Regions2")
    if (first.width, first.height) != (second.width, second.height):
        raise ValueError("兩個區域的尺寸要相同")
    limit = scalar(max_distance, "MaxDistance")
    if limit < 0:
        raise ValueError("MaxDistance 不能是負的")
    kept1 = [mask for mask in first.objects if any(_region_gap(mask, other, limit) for other in second.objects)]
    kept2 = [mask for mask in second.objects if any(_region_gap(other, mask, limit) for other in first.objects)]
    return Result(
        [
            Region(first.width, first.height, kept1, multi_color=len(kept1) > 1),
            Region(second.width, second.height, kept2, multi_color=len(kept2) > 1),
        ],
        [],
    )


def _drop_short_runs(mask: np.ndarray, limit: int) -> np.ndarray:
    height, width = mask.shape
    binary = mask > 0
    padded = np.pad(binary, ((0, 0), (1, 1)), constant_values=False)
    diff = padded[:, 1:].astype(np.int8) - padded[:, :-1].astype(np.int8)
    rows, starts = np.where(diff == 1)
    _end_rows, ends = np.where(diff == -1)
    kept = np.zeros((height, width), np.uint8)
    if len(rows) == 0:
        return kept
    keep = (ends - starts) > limit
    delta = np.zeros((height, width + 1), np.int16)
    np.add.at(delta, (rows[keep], starts[keep]), 1)
    np.add.at(delta, (rows[keep], ends[keep]), -1)
    kept[np.cumsum(delta, axis=1)[:, :width] > 0] = 255
    return kept


@operator(
    "區域",
    [P("Region", "輸入區域")],
    [P("RegionClipped", "去掉短行程之後")],
    [P("ElimShorter", "這個長度以下的水平行程會去掉")],
    [],
    "去掉太短的水平行程",
    "每一列連續的像素是一段。長度小於或等於 ElimShorter 的段會去掉。直的細線每一列都很短，也會被去掉。每個物件分開處理，整段被清掉的物件不會留下。",
    "Vizion.eliminate_runs (Region, RegionClipped, 3)",
)
def eliminate_runs(region, elim_shorter, *, ctx):
    source = expect_region(region, "Region")
    limit = whole(elim_shorter, "ElimShorter")
    if limit < 0:
        raise ValueError("ElimShorter 不能是負的")
    kept = []
    for mask in source.objects:
        clipped = _drop_short_runs(mask, limit)
        if int(cv2.countNonZero(clipped)):
            kept.append(clipped)
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


def _hole_objects(mask: np.ndarray) -> list[np.ndarray]:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    padded = np.pad(binary, 1)
    flood = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, flood, (0, 0), 128)
    holes = np.where(padded[1:-1, 1:-1] == 0, 255, 0).astype(np.uint8)
    if int(cv2.countNonZero(holes)) == 0:
        return []
    count, labels = cv2.connectedComponents(holes, connectivity=8)
    return [np.where(labels == label, 255, 0).astype(np.uint8) for label in range(1, count)]


@operator(
    "區域",
    [P("Region", "輸入區域")],
    [P("ConnectedRegions", "拆開的物件"), P("Holes", "這些物件裡的孔")],
    [],
    [],
    "拆開物件並取出孔洞",
    "先做八連通。每個物件裡被圍住、而且沒碰到影像邊緣的洞會另外變成物件。碰到邊緣的缺口不算孔。",
    "Vizion.connect_and_holes (Region, ConnectedRegions, Holes)",
)
def connect_and_holes(region, *, ctx):
    source = expect_region(region, "Region")
    components = halcon_components(source.union_mask())
    holes = []
    for mask in components:
        holes.extend(_hole_objects(mask))
    return Result(
        [
            Region(source.width, source.height, components, multi_color=len(components) > 1),
            Region(source.width, source.height, holes, multi_color=len(holes) > 1),
        ],
        [],
    )


@operator(
    "形態學",
    [P("Region", "輸入區域")],
    [P("RegionPrune", "修剪後")],
    [P("Iterations", "拿掉端點的次數")],
    [],
    "剪掉細線的端點",
    "每一輪拿掉只有一個鄰居的端點。迴圈沒有端點，不會變短。單獨一點也留著。次數 0 到 100。直的細線會從兩頭縮短。",
    "Vizion.prune (Region, RegionPrune, 3)",
)
def prune(region, iterations, *, ctx):
    source = expect_region(region, "Region")
    count = whole(iterations, "Iterations")
    if count < 0 or count > 100:
        raise ValueError("Iterations 要在 0 到 100")
    kernel = np.ones((3, 3), np.uint8)
    kernel[1, 1] = 0
    kept = []
    for mask in source.objects:
        binary = (mask > 0).astype(np.uint8)
        for _ in range(count):
            neighbors = cv2.filter2D(binary, cv2.CV_16S, kernel, borderType=cv2.BORDER_CONSTANT)
            binary[(binary == 1) & (neighbors == 1)] = 0
        if int(binary.sum()):
            kept.append(binary * 255)
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Region", "輸入區域")],
    [],
    [P("Tolerance", "近似誤差，像素")],
    [P("Row", "多邊形的列"), P("Column", "多邊形的行")],
    "把區域外輪廓收成多邊形",
    "每個物件只用最外圈。誤差愈大，點愈少。空區域會失敗。多個物件的點會接在同一個 tuple。",
    "Vizion.get_region_polygon (Region, 2, Row, Column)",
)
def get_region_polygon(region, tolerance, *, ctx):
    source = expect_region(region, "Region")
    limit = scalar(tolerance, "Tolerance")
    if limit < 0:
        raise ValueError("Tolerance 不能是負的")
    rows, cols = [], []
    for mask in source.objects:
        found = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = found[0] if len(found) == 2 else found[1]
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(contour, limit, True).reshape(-1, 2)
        cols.extend(int(item) for item in approx[:, 0])
        rows.extend(int(item) for item in approx[:, 1])
    if not rows:
        raise ValueError("區域是空的")
    return Result([], [rows, cols])


def _hitmiss_kernel(hit: Region, miss: Region) -> np.ndarray:
    if (hit.width, hit.height) != (miss.width, miss.height):
        raise ValueError("兩個結構元素的尺寸不同")
    foreground = hit.union_mask() > 0
    background = miss.union_mask() > 0
    if np.any(foreground & background):
        raise ValueError("前景和背景結構不能重疊")
    rows, cols = np.nonzero(foreground | background)
    if len(rows) == 0:
        raise ValueError("結構元素是空的")
    top, left, bottom, right = int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())
    kernel = np.zeros((bottom - top + 1, right - left + 1), np.int8)
    kernel[foreground[top : bottom + 1, left : right + 1]] = 1
    kernel[background[top : bottom + 1, left : right + 1]] = -1
    return kernel


@operator(
    "形態學",
    [P("Region", "輸入區域"), P("StructElement1", "必須是前景"), P("StructElement2", "必須是背景")],
    [P("RegionHitMiss", "對上的位置")],
    [],
    [],
    "命中不中",
    "前景結構要對上區域，背景結構要對上區域外。原點放在兩個結構合起來的外接框中心。兩者重疊會失敗。多個物件先合成再找。",
    "Vizion.hit_or_miss (Region, Foreground, Background, RegionHitMiss)",
)
def hit_or_miss(region, foreground, background, *, ctx):
    source = expect_region(region, "Region")
    kernel = _hitmiss_kernel(expect_region(foreground, "StructElement1"), expect_region(background, "StructElement2"))
    found = cv2.morphologyEx(source.union_mask(), cv2.MORPH_HITMISS, kernel)
    return region_out(source.width, source.height, found)


@operator(
    "區域",
    [],
    [P("CircleSector", "扇形")],
    [
        P("Row", "圓心列"),
        P("Column", "圓心行"),
        P("Radius", "半徑"),
        P("StartAngle", "起始角，弳度"),
        P("EndAngle", "結束角，弳度"),
    ],
    [],
    "產生扇形區域",
    "角度 0 朝右，正角度在畫面上順時針。從起始角順時針畫到結束角，包含圓心。兩角相同得到空區域，相差一整圈就是實心圓。半徑會四捨五入，尺寸對齊最近一張影像。",
    "Vizion.gen_circle_sector (CircleSector, 30, 40, 10, 0, rad(90))",
)
def gen_circle_sector(row, column, radius, start_angle, end_angle, *, ctx):
    height, width = _canvas(ctx)
    center = (whole(column, "Column"), whole(row, "Row"))
    size = whole(radius, "Radius")
    if size < 0:
        raise ValueError("半徑不能是負的")
    start = scalar(start_angle, "StartAngle")
    sweep = scalar(end_angle, "EndAngle") - start
    while sweep < 0:
        sweep += math.tau
    while sweep > math.tau:
        sweep -= math.tau
    mask = np.zeros((height, width), np.uint8)
    if sweep > 1e-8:
        if size == 0 or sweep >= math.tau - 1e-6:
            cv2.circle(mask, center, size, 255, -1)
        else:
            cv2.ellipse(
                mask,
                center,
                (size, size),
                0,
                math.degrees(start),
                math.degrees(start + sweep),
                255,
                -1,
            )
    objects = [mask] if int(cv2.countNonZero(mask)) else []
    return Result([Region(width, height, objects)], [])


def _region_bits(region, name: str) -> np.ndarray:
    source = expect_region(region, name)
    return source.union_mask() > 0


@operator(
    "區域",
    [P("Region1", "要檢查的區域"), P("Region2", "範圍")],
    [],
    [],
    [P("IsSubset", "1 表示完全落在範圍裡")],
    "區域是否被包含",
    "多個物件先合成再比較。Region1 的每個像素都在 Region2 裡就是 1。空區域算被包含。",
    "Vizion.test_subset_region (Region1, Region2, IsSubset)",
)
def test_subset_region(region1, region2, *, ctx):
    first = _region_bits(region1, "Region1")
    second = _region_bits(region2, "Region2")
    if first.shape != second.shape:
        raise ValueError("兩個區域的尺寸不同")
    outside = int(np.count_nonzero(first & ~second))
    return Result([], [0 if outside else 1])


@operator(
    "區域",
    [P("Region1", "第一個區域"), P("Region2", "第二個區域")],
    [],
    [],
    [P("IsEqual", "1 表示像素完全相同")],
    "兩個區域是否相同",
    "多個物件先合成再比較。空區域和空區域相同。",
    "Vizion.test_equal_region (Region1, Region2, IsEqual)",
)
def test_equal_region(region1, region2, *, ctx):
    first = _region_bits(region1, "Region1")
    second = _region_bits(region2, "Region2")
    if first.shape != second.shape:
        raise ValueError("兩個區域的尺寸不同")
    return Result([], [1 if np.array_equal(first, second) else 0])


def _angle_sweep(start_angle, end_angle, start_name: str, end_name: str) -> tuple[float, float]:
    start = scalar(start_angle, start_name)
    sweep = scalar(end_angle, end_name) - start
    while sweep < 0:
        sweep += math.tau
    while sweep > math.tau:
        sweep -= math.tau
    return start, sweep


def _polar_maps(row, column, angle_start, angle_end, radius_start, radius_end, width, height):
    wide = whole(width, "Width")
    high = whole(height, "Height")
    if wide < 1 or high < 1:
        raise ValueError("寬高至少要 1")
    angles = np.linspace(scalar(angle_start, "AngleStart"), scalar(angle_end, "AngleEnd"), wide, endpoint=False, dtype=np.float32)
    radii = np.linspace(scalar(radius_start, "RadiusStart"), scalar(radius_end, "RadiusEnd"), high, endpoint=False, dtype=np.float32)
    angle_grid, radius_grid = np.meshgrid(angles, radii)
    map_x = np.float32(scalar(column, "Column") + radius_grid * np.cos(angle_grid))
    map_y = np.float32(scalar(row, "Row") + radius_grid * np.sin(angle_grid))
    return wide, high, map_x, map_y


@operator(
    "幾何",
    [P("Region", "輸入區域")],
    [P("RegionPolar", "展開後的區域")],
    [
        P("Row", "極座標中心列"),
        P("Column", "極座標中心行"),
        P("AngleStart", "起始角，弳度"),
        P("AngleEnd", "結束角，弳度"),
        P("RadiusStart", "內半徑"),
        P("RadiusEnd", "外半徑"),
        P("Width", "角度方向的取樣數，變成寬度"),
        P("Height", "半徑方向的取樣數，變成高度"),
    ],
    [],
    "把區域展開成極座標",
    "角度 0 朝右，正角度順時針，取樣方式和 polar_trans_image_ext 相同。用最近鄰，每個物件各自展開。",
    "Vizion.polar_trans_region (Region, RegionPolar, 30, 40, 0, rad(360), 0, 20, 40, 20)",
)
def polar_trans_region(region, row, column, angle_start, angle_end, radius_start, radius_end, width, height, *, ctx):
    source = expect_region(region, "Region")
    wide, high, map_x, map_y = _polar_maps(row, column, angle_start, angle_end, radius_start, radius_end, width, height)
    objects = []
    for mask in source.objects:
        moved = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if int(cv2.countNonZero(moved)):
            objects.append(np.where(moved > 0, 255, 0).astype(np.uint8))
    return Result([Region(wide, high, objects, multi_color=len(objects) > 1)], [])


@operator(
    "區域",
    [],
    [P("EllipseSector", "橢圓扇形")],
    [
        P("Row", "中心列"),
        P("Column", "中心行"),
        P("Phi", "第一軸角度，弳度。0 沿著行"),
        P("Radius1", "第一軸半徑"),
        P("Radius2", "第二軸半徑"),
        P("StartAngle", "起始角，從第一軸量起"),
        P("EndAngle", "結束角，從第一軸量起"),
    ],
    [],
    "產生橢圓扇形",
    "角度 0 沿著第一軸，正角度順時針。兩角相同得到空區域，相差一整圈就是整個橢圓。半徑會四捨五入，尺寸對齊最近一張影像。",
    "Vizion.gen_ellipse_sector (EllipseSector, 30, 40, 0, 16, 8, 0, rad(90))",
)
def gen_ellipse_sector(row, column, phi, radius1, radius2, start_angle, end_angle, *, ctx):
    height, width = _canvas(ctx)
    first = scalar(radius1, "Radius1")
    second = scalar(radius2, "Radius2")
    if first <= 0 or second <= 0:
        raise ValueError("半徑必須大於 0")
    start, sweep = _angle_sweep(start_angle, end_angle, "StartAngle", "EndAngle")
    mask = np.zeros((height, width), np.uint8)
    center = (whole(column, "Column"), whole(row, "Row"))
    axes = (max(1, int(round(first))), max(1, int(round(second))))
    if sweep > 1e-8:
        if sweep >= math.tau - 1e-6:
            cv2.ellipse(mask, center, axes, math.degrees(scalar(phi, "Phi")), 0, 360, 255, -1)
        else:
            cv2.ellipse(
                mask,
                center,
                axes,
                math.degrees(scalar(phi, "Phi")),
                math.degrees(start),
                math.degrees(start + sweep),
                255,
                -1,
            )
    objects = [mask] if int(cv2.countNonZero(mask)) else []
    return Result([Region(width, height, objects)], [])


@operator(
    "形態學",
    [P("Region", "輸入區域"), P("StructElement", "結構元素區域")],
    [P("RegionTopHat", "開運算去掉的部分")],
    [],
    [],
    "頂帽",
    "原區域減去開運算。小亮塊會留下，大塊的內部不會。" + _STRUCT,
    "Vizion.top_hat (Region, StructElement, RegionTopHat)",
)
def top_hat(region, struct_element, *, ctx):
    opened = opening(region, struct_element, ctx=ctx).iconic[0]
    return difference(region, opened, ctx=ctx)


@operator(
    "形態學",
    [P("Region", "輸入區域"), P("StructElement", "結構元素區域")],
    [P("RegionBottomHat", "閉運算補上的部分")],
    [],
    [],
    "底帽",
    "閉運算減去原區域。比結構元素小的缺口會被補出來。" + _STRUCT,
    "Vizion.bottom_hat (Region, StructElement, RegionBottomHat)",
)
def bottom_hat(region, struct_element, *, ctx):
    closed = closing(region, struct_element, ctx=ctx).iconic[0]
    return difference(closed, region, ctx=ctx)


@operator(
    "區域",
    [P("Region", "物件")],
    [P("ImageLabel", "標籤影像")],
    [],
    [],
    "區域編成標籤",
    "背景是 0，物件從 1 開始。後面的物件蓋掉重疊的地方。超過 255 個物件會失敗。",
    "Vizion.region_to_label (Regions, ImageLabel)",
)
def region_to_label(region, *, ctx):
    source = expect_region(region, "Region")
    if len(source.objects) > 255:
        raise ValueError("物件超過 255 個，灰階影像放不下")
    label = np.zeros((source.height, source.width), np.uint8)
    for index, mask in enumerate(source.objects, start=1):
        label[mask > 0] = index
    return Result([Image(label)], [])


@operator(
    "區域",
    [P("ImageLabel", "標籤影像")],
    [P("Region", "每個非零灰階一個物件")],
    [],
    [],
    "標籤編成區域",
    "同一個灰階是一個物件，就算中間斷開。0 是背景。物件依灰階由小到大排列。彩色先轉成亮度，定義域外不算。",
    "Vizion.label_to_region (ImageLabel, Region)",
)
def label_to_region(image, *, ctx):
    picture = expect_image(image, "Image")
    gray = picture.gray()
    if picture.domain is not None:
        gray = gray.copy()
        gray[picture.domain == 0] = 0
    values = np.unique(gray)
    values = values[values > 0]
    objects = [np.where(gray == int(value), 255, 0).astype(np.uint8) for value in values]
    return Result([Region(picture.width, picture.height, objects, multi_color=len(objects) > 1)], [])


@operator(
    "幾何",
    [P("RegionPolar", "極座標區域")],
    [P("RegionXY", "還原後的區域")],
    [
        P("Row", "極座標中心列"),
        P("Column", "極座標中心行"),
        P("AngleStart", "起始角，弳度"),
        P("AngleEnd", "結束角，弳度"),
        P("RadiusStart", "內半徑"),
        P("RadiusEnd", "外半徑"),
        P("Width", "還原後的寬度"),
        P("Height", "還原後的高度"),
    ],
    [],
    "把極座標區域還原",
    "和 polar_trans_region 相反，用最近鄰。角度 0 朝右，正角度順時針。輸出尺寸是 Width 和 Height。",
    "Vizion.polar_trans_region_inv (RegionPolar, RegionXY, 30, 40, 0, rad(360), 0, 20, 80, 60)",
)
def polar_trans_region_inv(region, row, column, angle_start, angle_end, radius_start, radius_end, width, height, *, ctx):
    source = expect_region(region, "RegionPolar")
    wide, high, map_x, map_y = _polar_inverse_maps(
        source.width, source.height, row, column, angle_start, angle_end, radius_start, radius_end, width, height
    )
    objects = []
    for mask in source.objects:
        moved = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if int(cv2.countNonZero(moved)):
            objects.append(np.where(moved > 0, 255, 0).astype(np.uint8))
    return Result([Region(wide, high, objects, multi_color=len(objects) > 1)], [])


_SPATIAL = ["'left'", "'right'", "'above'", "'below'"]


def _spatial_hit(row1, col1, row2, col2, ref_row1, ref_col1, ref_row2, ref_col2, direction: str) -> bool:
    if direction == "left":
        return col2 < ref_col1
    if direction == "right":
        return col1 > ref_col2
    if direction == "above":
        return row2 < ref_row1
    return row1 > ref_row2


@operator(
    "區域",
    [P("Regions", "候選物件"), P("Reference", "參考物件")],
    [P("Selected", "方位符合的物件")],
    [P("Direction", "left、right、above 或 below", "'left'", _SPATIAL)],
    [],
    "用外接矩形挑方位",
    "left 是候選的右邊界在參考左邊界的左邊，邊貼著不算。above 是列比較小的那一側。和任何一個參考物件成立就留下。",
    "Vizion.select_region_spatial (Regions, Reference, Selected, 'left')",
)
def select_region_spatial(regions, reference, direction, *, ctx):
    source = expect_region(regions, "Regions")
    marks = expect_region(reference, "Reference")
    if (source.width, source.height) != (marks.width, marks.height):
        raise ValueError("兩個區域的尺寸要相同")
    kind = text(direction, "Direction").lower()
    if kind not in ("left", "right", "above", "below"):
        raise ValueError("Direction 只能是 left、right、above 或 below")
    boxes = [_region_box(mask) for mask in marks.objects if int(cv2.countNonZero(mask))]
    kept = []
    for mask in source.objects:
        if int(cv2.countNonZero(mask)) == 0:
            continue
        row1, col1, row2, col2 = _region_box(mask)
        if any(_spatial_hit(row1, col1, row2, col2, *box, kind) for box in boxes):
            kept.append(mask)
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


_THICK = ["'chessboard'", "'city-block'", "'euclidean'"]


def _metric_code(metric) -> tuple[int, int]:
    kind = text(metric, "Metric").lower()
    codes = {
        "chessboard": (cv2.DIST_C, 3),
        "city-block": (cv2.DIST_L1, 3),
        "euclidean": (cv2.DIST_L2, 5),
    }
    if kind not in codes:
        raise ValueError("Metric 只能是 chessboard、city-block 或 euclidean")
    return codes[kind]


def _object_distance(mask: np.ndarray, metric) -> np.ndarray | None:
    if int(cv2.countNonZero(mask)) == 0:
        return None
    flag, size = _metric_code(metric)
    return cv2.distanceTransform((mask > 0).astype(np.uint8), flag, size)


def _distance_objects(region, metric):
    source = expect_region(region, "Region")
    if source.count() == 0:
        raise ValueError("區域是空的")
    found = []
    for index, mask in enumerate(source.objects, start=1):
        dist = _object_distance(mask, metric)
        if dist is None:
            raise ValueError(f"第 {index} 個物件是空的")
        found.append(dist)
    return source, found


def _ridge(dist: np.ndarray) -> np.ndarray:
    padded = np.pad(dist, 1, mode="constant", constant_values=-1.0)
    keep = dist > 0
    high, wide = dist.shape
    for drow in (-1, 0, 1):
        for dcol in (-1, 0, 1):
            if drow == 0 and dcol == 0:
                continue
            keep = keep & (dist >= padded[1 + drow : 1 + drow + high, 1 + dcol : 1 + dcol + wide])
    if not np.any(keep):
        keep = dist >= float(dist.max())
    return keep


def _span_limits(low, high) -> tuple[float, float]:
    floor = scalar(low, "Min")
    ceiling = scalar(high, "Max")
    return (floor, ceiling) if floor <= ceiling else (ceiling, floor)


def _peak(dist: np.ndarray) -> tuple[int, int, float]:
    pick = int(np.argmax(dist))
    row, col = divmod(pick, dist.shape[1])
    return row, col, float(dist[row, col])


def _neck(dist: np.ndarray) -> tuple[int, int, float]:
    rows, cols = np.nonzero(_ridge(dist))
    values = dist[rows, cols]
    pick = int(np.argmin(values))
    return int(rows[pick]), int(cols[pick]), float(values[pick])


@operator(
    "區域",
    [P("Region", "區域")],
    [P("ImageDistance", "到區域外的距離")],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [],
    "把厚度畫成影像",
    "每個前景像素的值是到區域外的距離。chessboard 是棋盤步數，邊和角都算一步。區域外是 0。多個物件先合併。超過 255 會被截掉。",
    "Vizion.thickness_image (Region, ImageDistance, 'chessboard')",
)
def thickness_image(region, metric, *, ctx):
    source = expect_region(region, "Region")
    dist = _object_distance(source.union_mask(), metric)
    if dist is None:
        raise ValueError("區域是空的")
    data = np.clip(np.round(dist), 0, 255).astype(np.uint8)
    return Result([Image(data)], [])


@operator(
    "區域",
    [P("Region", "物件集合")],
    [],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [P("Row", "列"), P("Column", "行"), P("Distance", "到區域外的距離")],
    "每個物件離邊界最遠的點",
    "不會把物件合併。一樣遠時留列最小、再來行最小的那一點。一個物件也是 tuple。",
    "Vizion.thickness_max (Region, 'chessboard', Row, Column, Distance)",
)
def thickness_max(region, metric, *, ctx):
    _source, found = _distance_objects(region, metric)
    rows, cols, distances = [], [], []
    for dist in found:
        row, col, value = _peak(dist)
        rows.append(row)
        cols.append(col)
        distances.append(value)
    return Result([], [rows, cols, distances])


@operator(
    "區域",
    [P("Region", "物件集合")],
    [],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [P("Row", "列"), P("Column", "行"), P("Distance", "最細處的距離")],
    "每個物件最細的中線",
    "中線是距離不比鄰居小的像素。最細處是中線上距離最小的點。一樣細時留列最小、再來行最小的那一點。",
    "Vizion.thickness_neck (Region, 'chessboard', Row, Column, Distance)",
)
def thickness_neck(region, metric, *, ctx):
    _source, found = _distance_objects(region, metric)
    rows, cols, distances = [], [], []
    for dist in found:
        row, col, value = _neck(dist)
        rows.append(row)
        cols.append(col)
        distances.append(value)
    return Result([], [rows, cols, distances])


@operator(
    "區域",
    [P("Region", "物件集合")],
    [],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [P("Mean", "中線距離的平均")],
    "每個物件中線的平均厚度",
    "只平均中線上的距離，不會把整塊區域平均進來。一個物件也是 tuple。",
    "Vizion.thickness_mean (Region, 'chessboard', Mean)",
)
def thickness_mean(region, metric, *, ctx):
    _source, found = _distance_objects(region, metric)
    means = [float(dist[_ridge(dist)].mean()) for dist in found]
    return Result([], [means])


@operator(
    "區域",
    [P("Region", "物件集合")],
    [],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [P("Ratio", "最細除以最厚")],
    "每個物件有多細腰",
    "最細處的距離除以最厚處的距離。方塊和直條接近 1，中間變細的會比較小。",
    "Vizion.neck_ratio (Region, 'chessboard', Ratio)",
)
def neck_ratio(region, metric, *, ctx):
    _source, found = _distance_objects(region, metric)
    ratios = []
    for dist in found:
        _row, _col, peak = _peak(dist)
        _nrow, _ncol, neck = _neck(dist)
        ratios.append(float(neck / peak) if peak else 0.0)
    return Result([], [ratios])


@operator(
    "區域",
    [P("Region", "物件集合")],
    [],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [P("Sum", "距離總和")],
    "每個物件的距離總和",
    "把物件裡每個像素到區域外的距離加起來。一個物件也是 tuple。",
    "Vizion.thickness_sum (Region, 'chessboard', Sum)",
)
def thickness_sum(region, metric, *, ctx):
    _source, found = _distance_objects(region, metric)
    totals = [float(dist[dist > 0].sum()) for dist in found]
    return Result([], [totals])


@operator(
    "區域",
    [P("Region", "物件集合")],
    [P("Selected", "最細處落在範圍裡的物件")],
    [
        P("Min", "最細距離的下限"),
        P("Max", "最細距離的上限"),
        P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK),
    ],
    [],
    "用最細處篩選物件",
    "留下最細處的距離在下限和上限之間的物件，兩端都含在內。寫反了會自動對調。沒有符合的就得到空區域。",
    "Vizion.select_neck (Region, Selected, 2, 5, 'chessboard')",
)
def select_neck(region, low, high, metric, *, ctx):
    source, found = _distance_objects(region, metric)
    floor, ceiling = _span_limits(low, high)
    kept = []
    for mask, dist in zip(source.objects, found):
        _row, _col, neck = _neck(dist)
        if floor <= neck <= ceiling:
            kept.append(mask)
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Region", "區域")],
    [P("RegionBand", "距離落在範圍裡的像素")],
    [
        P("Min", "距離下限"),
        P("Max", "距離上限"),
        P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK),
    ],
    [],
    "留下離邊界一定距離的像素",
    "下限和上限含在內，寫反了會自動對調。多個物件先合併。區域外面的像素不會留下。",
    "Vizion.band_thickness (Region, RegionBand, 3, 5, 'chessboard')",
)
def band_thickness(region, low, high, metric, *, ctx):
    source = expect_region(region, "Region")
    dist = _object_distance(source.union_mask(), metric)
    if dist is None:
        raise ValueError("區域是空的")
    floor, ceiling = _span_limits(low, high)
    selected = (dist >= floor) & (dist <= ceiling)
    return region_out(source.width, source.height, selected.astype(np.uint8) * 255)


def _paired_masks(region1, region2):
    first = expect_region(region1, "Region1")
    second = expect_region(region2, "Region2")
    if (first.width, first.height) != (second.width, second.height):
        raise ValueError("兩個區域的尺寸要相同")
    if first.count() == 0 or second.count() == 0:
        raise ValueError("區域是空的")
    left = list(first.objects)
    right = list(second.objects)
    if len(left) == 1 and len(right) > 1:
        left = left * len(right)
    elif len(right) == 1 and len(left) > 1:
        right = right * len(left)
    elif len(left) != len(right):
        raise ValueError("兩個區域的物件數要相同，或其中一邊只有一個物件")
    pairs = []
    for index, (mask1, mask2) in enumerate(zip(left, right), start=1):
        one = mask1 > 0
        other = mask2 > 0
        if not one.any() or not other.any():
            raise ValueError(f"第 {index} 個物件是空的")
        pairs.append((mask1, one, other))
    return first, pairs


def _overlap_rows(region1, region2):
    source, pairs = _paired_masks(region1, region2)
    rows = []
    for one, other in ((left, right) for _mask, left, right in pairs):
        inter = int(np.count_nonzero(one & other))
        only = int(np.count_nonzero(one & ~other))
        union = int(np.count_nonzero(one | other))
        rows.append((inter, only, union, int(one.sum())))
    return source, pairs, rows


def _cover_rows(region1, region2):
    source, pairs, rows = _overlap_rows(region1, region2)
    return source, pairs, [inter / area if area else 0.0 for inter, _only, _union, area in rows]


@operator(
    "區域",
    [P("Region1", "第一組物件"), P("Region2", "第二組物件")],
    [],
    [],
    [P("Area", "重疊的像素數")],
    "兩組物件重疊幾個像素",
    "依順序配對。物件數相同，或其中一邊只有一個物件時，那個物件會配對另一邊的每一個。一個結果也是 tuple。",
    "Vizion.area_intersection (Region1, Region2, Area)",
)
def area_intersection(region1, region2, *, ctx):
    _source, _pairs, rows = _overlap_rows(region1, region2)
    return Result([], [[float(inter) for inter, _only, _union, _area in rows]])


@operator(
    "區域",
    [P("Region1", "第一組物件"), P("Region2", "第二組物件")],
    [],
    [],
    [P("Area", "聯集的像素數")],
    "兩組物件合併後有幾個像素",
    "聯集是任一邊有的像素。配對方式和重疊面積相同。",
    "Vizion.area_union (Region1, Region2, Area)",
)
def area_union(region1, region2, *, ctx):
    _source, _pairs, rows = _overlap_rows(region1, region2)
    return Result([], [[float(union) for _inter, _only, union, _area in rows]])


@operator(
    "區域",
    [P("Region1", "第一組物件"), P("Region2", "第二組物件")],
    [],
    [],
    [P("Area", "只在第一組裡的像素數")],
    "第一組比第二組多出來的像素",
    "只算第一組有、第二組沒有的像素。完全蓋住時是 0。",
    "Vizion.area_difference (Region1, Region2, Area)",
)
def area_difference(region1, region2, *, ctx):
    _source, _pairs, rows = _overlap_rows(region1, region2)
    return Result([], [[float(only) for _inter, only, _union, _area in rows]])


@operator(
    "區域",
    [P("Region1", "第一組物件"), P("Region2", "第二組物件")],
    [],
    [],
    [P("IoU", "重疊除以聯集")],
    "兩組物件的重疊比例",
    "重疊像素數除以聯集像素數。完全一樣是 1，完全沒碰到是 0。",
    "Vizion.region_iou (Region1, Region2, IoU)",
)
def region_iou(region1, region2, *, ctx):
    _source, _pairs, rows = _overlap_rows(region1, region2)
    return Result([], [[float(inter / union) if union else 0.0 for inter, _only, union, _area in rows]])


@operator(
    "區域",
    [P("Region1", "第一組物件"), P("Region2", "第二組物件")],
    [],
    [],
    [P("Cover", "第一組被蓋住的比例")],
    "第一組有多少落在第二組裡",
    "重疊像素數除以第一組的面積。第一組整個被蓋住是 1。這和調換兩組的結果不一樣。",
    "Vizion.region_cover (Region1, Region2, Cover)",
)
def region_cover(region1, region2, *, ctx):
    _source, _pairs, covers = _cover_rows(region1, region2)
    return Result([], [covers])


@operator(
    "區域",
    [P("Region1", "第一組物件"), P("Region2", "第二組物件")],
    [],
    [],
    [P("Inside", "1 表示整個落在第二組裡")],
    "第一組是否完全落在第二組裡",
    "第一組的每個像素都在第二組裡就是 1，否則是 0。",
    "Vizion.region_inside (Region1, Region2, Inside)",
)
def region_inside(region1, region2, *, ctx):
    _source, _pairs, rows = _overlap_rows(region1, region2)
    return Result([], [[1 if only == 0 else 0 for _inter, only, _union, _area in rows]])


@operator(
    "區域",
    [P("Region1", "要篩選的物件"), P("Region2", "對照區域")],
    [P("Selected", "被蓋住的比例落在範圍裡的物件")],
    [P("Min", "比例下限"), P("Max", "比例上限")],
    [],
    "用被蓋住的比例篩選物件",
    "留下第一組裡、被第二組蓋住的比例在下限和上限之間的物件，兩端都含在內。寫反了會自動對調。第二組若只有一個物件，會拿來比對每一個。沒有符合的就得到空區域。",
    "Vizion.select_cover (Region1, Region2, Selected, 0.9, 1)",
)
def select_cover(region1, region2, low, high, *, ctx):
    source, _pairs, covers = _cover_rows(region1, region2)
    if source.count() != len(covers):
        raise ValueError("篩選時第一組的物件會各自保留，請讓兩邊物件數相同，或第二組只有一個物件")
    floor, ceiling = _span_limits(low, high)
    kept = [mask for mask, cover in zip(source.objects, covers) if floor <= cover <= ceiling]
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Region1", "第一組物件"), P("Region2", "第二組物件")],
    [P("RegionDifference", "只在其中一邊的像素")],
    [],
    [],
    "留下只在其中一邊的像素",
    "兩邊都有或兩邊都沒有的像素不會留下。完全一樣就得到空區域。配對方式和重疊面積相同。",
    "Vizion.sym_difference (Region1, Region2, RegionDifference)",
)
def sym_difference(region1, region2, *, ctx):
    _source, pairs, _rows = _overlap_rows(region1, region2)
    kept = []
    width = height = 0
    for mask, one, other in pairs:
        height, width = one.shape
        xor = np.where(one ^ other, 255, 0).astype(np.uint8)
        if int(cv2.countNonZero(xor)):
            kept.append(xor)
    if width == 0:
        width, height = _source.width, _source.height
    return Result([Region(width, height, kept, multi_color=len(kept) > 1)], [])


def _toward(other: np.ndarray, metric) -> np.ndarray:
    flag, size = _metric_code(metric)
    seeds = np.where(other, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(seeds, flag, size)


def _pick_distance(dist: np.ndarray, mask: np.ndarray, far: bool) -> tuple[int, int, float]:
    work = np.where(mask, dist, -np.inf if far else np.inf)
    pick = int(np.argmax(work) if far else np.argmin(work))
    row, col = divmod(pick, dist.shape[1])
    return row, col, float(dist[row, col])


def _nearest_target(other: np.ndarray, row: int, col: int, metric) -> tuple[int, int]:
    flag, size = _metric_code(metric)
    seeds = np.full(other.shape, 255, np.uint8)
    seeds[row, col] = 0
    back = cv2.distanceTransform(seeds, flag, size)
    hit_row, hit_col, _value = _pick_distance(back, other, False)
    return hit_row, hit_col


def _pair_distance(region1, region2, metric):
    source, pairs = _paired_masks(region1, region2)
    found = [(one, other, _toward(other, metric)) for _mask, one, other in pairs]
    return source, found


def _center_delta(metric, drow: float, dcol: float) -> float:
    kind = text(metric, "Metric").lower()
    if kind == "chessboard":
        return float(max(abs(drow), abs(dcol)))
    if kind == "city-block":
        return float(abs(drow) + abs(dcol))
    if kind == "euclidean":
        return float(np.hypot(drow, dcol))
    raise ValueError("Metric 只能是 chessboard、city-block 或 euclidean")


def _centers(one: np.ndarray, other: np.ndarray) -> tuple[float, float, float, float]:
    rows, cols = np.nonzero(one)
    other_rows, other_cols = np.nonzero(other)
    return float(rows.mean()), float(cols.mean()), float(other_rows.mean()), float(other_cols.mean())


@operator(
    "區域",
    [P("Region1", "第一組物件"), P("Region2", "第二組物件")],
    [],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [P("Distance", "最近距離")],
    "每個物件到對照區域的最近距離",
    "不會把物件合併。重疊是 0。相鄰但沒重疊是 1。distance_rr 會先把物件合併，再量凸包，和這個不一樣。",
    "Vizion.distance_objects (Region1, Region2, 'chessboard', Distance)",
)
def distance_objects(region1, region2, metric, *, ctx):
    _source, found = _pair_distance(region1, region2, metric)
    return Result([], [[float(dist[one].min()) for one, _other, dist in found]])


@operator(
    "區域",
    [P("Region1", "第一組物件"), P("Region2", "第二組物件")],
    [],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [P("Distance", "第一組裡離對照區域最遠的距離")],
    "每個物件離對照區域最遠的點有多遠",
    "量的是第一組裡離第二組最遠的那個像素，不是兩個凸包的頂點。",
    "Vizion.distance_objects_far (Region1, Region2, 'chessboard', Distance)",
)
def distance_objects_far(region1, region2, metric, *, ctx):
    _source, found = _pair_distance(region1, region2, metric)
    return Result([], [[float(dist[one].max()) for one, _other, dist in found]])


@operator(
    "區域",
    [P("Region1", "第一組物件"), P("Region2", "第二組物件")],
    [],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [P("Row1", "第一組的列"), P("Column1", "第一組的行"), P("Row2", "第二組的列"), P("Column2", "第二組的行"), P("Distance", "距離")],
    "找出兩組物件最接近的像素",
    "一樣近時，第一組留列最小、再來行最小的那一點，第二組也一樣。重疊時兩個點是同一個像素，距離是 0。",
    "Vizion.nearest_object_point (Region1, Region2, 'chessboard', Row1, Column1, Row2, Column2, Distance)",
)
def nearest_object_point(region1, region2, metric, *, ctx):
    _source, found = _pair_distance(region1, region2, metric)
    row1, col1, row2, col2, distances = [], [], [], [], []
    for one, other, dist in found:
        row, col, value = _pick_distance(dist, one, False)
        hit_row, hit_col = _nearest_target(other, row, col, metric)
        row1.append(row)
        col1.append(col)
        row2.append(hit_row)
        col2.append(hit_col)
        distances.append(value)
    return Result([], [row1, col1, row2, col2, distances])


@operator(
    "區域",
    [P("Region1", "第一組物件"), P("Region2", "第二組物件")],
    [],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [P("Row1", "第一組的列"), P("Column1", "第一組的行"), P("Row2", "第二組的列"), P("Column2", "第二組的行"), P("Distance", "距離")],
    "找出第一組裡離對照區域最遠的像素",
    "一樣遠時留列最小、再來行最小的那一點。第二組回報離這個點最近的像素。",
    "Vizion.farthest_object_point (Region1, Region2, 'chessboard', Row1, Column1, Row2, Column2, Distance)",
)
def farthest_object_point(region1, region2, metric, *, ctx):
    _source, found = _pair_distance(region1, region2, metric)
    row1, col1, row2, col2, distances = [], [], [], [], []
    for one, other, dist in found:
        row, col, value = _pick_distance(dist, one, True)
        hit_row, hit_col = _nearest_target(other, row, col, metric)
        row1.append(row)
        col1.append(col)
        row2.append(hit_row)
        col2.append(hit_col)
        distances.append(value)
    return Result([], [row1, col1, row2, col2, distances])


@operator(
    "區域",
    [P("Region1", "第一組物件"), P("Region2", "第二組物件")],
    [],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [P("Row", "第二組中心減第一組中心的列"), P("Column", "第二組中心減第一組中心的行")],
    "兩個物件中心的位移",
    "中心是像素座標的平均。列或行是正的，表示第二組在第一組的下方或右方。",
    "Vizion.centroid_offset (Region1, Region2, 'chessboard', Row, Column)",
)
def centroid_offset(region1, region2, metric, *, ctx):
    _source, found = _pair_distance(region1, region2, metric)
    rows, cols = [], []
    for one, other, _dist in found:
        row1, _col1, row2, _col2 = _centers(one, other)
        rows.append(row2 - row1)
        cols.append(_col2 - _col1)
    return Result([], [rows, cols])


@operator(
    "區域",
    [P("Region1", "第一組物件"), P("Region2", "第二組物件")],
    [],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [P("Distance", "兩個中心的距離")],
    "兩個物件中心隔多遠",
    "用選的距離去量兩個中心。方塊的中心是對角的平均。",
    "Vizion.centroid_distance (Region1, Region2, 'chessboard', Distance)",
)
def centroid_distance(region1, region2, metric, *, ctx):
    _source, found = _pair_distance(region1, region2, metric)
    distances = []
    for one, other, _dist in found:
        row1, col1, row2, col2 = _centers(one, other)
        distances.append(_center_delta(metric, row2 - row1, col2 - col1))
    return Result([], [distances])


@operator(
    "區域",
    [P("Region1", "要篩選的物件"), P("Region2", "對照區域")],
    [P("Selected", "最近距離落在範圍裡的物件")],
    [
        P("Min", "距離下限"),
        P("Max", "距離上限"),
        P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK),
    ],
    [],
    "用最近距離篩選物件",
    "留下第一組裡、到第二組的最近距離在下限和上限之間的物件，兩端都含在內。寫反了會自動對調。沒有符合的就得到空區域。",
    "Vizion.select_object_distance (Region1, Region2, Selected, 0, 1, 'chessboard')",
)
def select_object_distance(region1, region2, low, high, metric, *, ctx):
    source, found = _pair_distance(region1, region2, metric)
    if source.count() != len(found):
        raise ValueError("篩選時第一組的物件會各自保留，請讓兩邊物件數相同，或第二組只有一個物件")
    floor, ceiling = _span_limits(low, high)
    kept = [mask for mask, (one, _other, dist) in zip(source.objects, found) if floor <= float(dist[one].min()) <= ceiling]
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Region1", "第一組物件"), P("Region2", "第二組物件")],
    [P("RegionEdge", "第一組裡最靠近第二組的像素")],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [],
    "留下每個物件朝向對照區域的那一側",
    "每個物件留下距離等於最近距離的像素。重疊時留下重疊的部分。結果依物件排列。",
    "Vizion.facing_region (Region1, Region2, RegionEdge, 'chessboard')",
)
def facing_region(region1, region2, metric, *, ctx):
    source, found = _pair_distance(region1, region2, metric)
    kept = []
    for one, _other, dist in found:
        limit = float(dist[one].min())
        edge = one & (dist <= limit + 1e-6)
        kept.append(edge.astype(np.uint8) * 255)
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


_SIDE = ["'right'", "'left'", "'below'", "'above'"]
_NEAR = ["'near'", "'far'"]


def _bearing(drow: float, dcol: float) -> float:
    if abs(drow) < 1e-9 and abs(dcol) < 1e-9:
        return 0.0
    deg = math.degrees(math.atan2(drow, dcol))
    if deg < 0:
        deg += 360.0
    return float(deg)


def _gap_fields(region1, region2, metric):
    source, found = _pair_distance(region1, region2, metric)
    gaps = []
    for one, other, dist in found:
        separation = float(dist[one].min())
        back = _toward(one, metric)
        space = ~(one | other) & (back + dist <= separation + 1e-6)
        gaps.append((separation, space))
    return source, gaps


@operator(
    "區域",
    [P("Region1", "物件"), P("Region2", "參考區域")],
    [],
    [],
    [P("Angle", "物件在參考的哪個方向，度")],
    "物件中心在參考中心的哪個方向",
    "0 度朝右，順時針往下。正下方是 90，正左是 180，正上方是 270。兩個中心重合時是 0。一個物件也是 tuple。",
    "Vizion.region_angle (Region1, Region2, Angle)",
)
def region_angle(region1, region2, *, ctx):
    _source, pairs = _paired_masks(region1, region2)
    angles = []
    for _mask, one, other in pairs:
        row1, col1, row2, col2 = _centers(one, other)
        angles.append(_bearing(row1 - row2, col1 - col2))
    return Result([], [angles])


@operator(
    "區域",
    [P("Region1", "要篩選的物件"), P("Region2", "參考區域")],
    [P("Selected", "落在指定一側的物件")],
    [P("Side", "right、left、below 或 above", None, _SIDE)],
    [],
    "留下參考區域某一側的物件",
    "用中心比較。right 是中心在參考的右方，below 是下方。中心對齊的不算。第二組只有一個物件時，會拿來比對每一個。",
    "Vizion.select_side (Region1, Region2, Selected, 'right')",
)
def select_side(region1, region2, side, *, ctx):
    source, pairs = _paired_masks(region1, region2)
    kind = text(side, "Side").lower()
    checks = {
        "right": lambda row1, col1, row2, col2: col1 > col2,
        "left": lambda row1, col1, row2, col2: col1 < col2,
        "below": lambda row1, col1, row2, col2: row1 > row2,
        "above": lambda row1, col1, row2, col2: row1 < row2,
    }
    if kind not in checks:
        raise ValueError("Side 只能是 right、left、below 或 above")
    if source.count() != len(pairs):
        raise ValueError("篩選時第一組的物件會各自保留，請讓兩邊物件數相同，或第二組只有一個物件")
    kept = []
    for mask, (_stored, one, other) in zip(source.objects, pairs):
        row1, col1, row2, col2 = _centers(one, other)
        if checks[kind](row1, col1, row2, col2):
            kept.append(mask)
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Region1", "要排序的物件"), P("Region2", "參考區域")],
    [P("Sorted", "按距離排好的物件")],
    [
        P("Order", "near 由近到遠，far 由遠到近", None, _NEAR),
        P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK),
    ],
    [],
    "按到參考區域的距離排序",
    "near 把最近的排前面。距離一樣時維持原來的順序。第二組只有一個物件時，會拿來比對每一個。",
    "Vizion.sort_distance (Region1, Region2, Sorted, 'near', 'chessboard')",
)
def sort_distance(region1, region2, order, metric, *, ctx):
    source, found = _pair_distance(region1, region2, metric)
    kind = text(order, "Order").lower()
    if kind not in ("near", "far"):
        raise ValueError("Order 只能是 near 或 far")
    if source.count() != len(found):
        raise ValueError("篩選時第一組的物件會各自保留，請讓兩邊物件數相同，或第二組只有一個物件")
    distances = np.array([float(dist[one].min()) for one, _other, dist in found], dtype=np.float64)
    rank = np.argsort(distances if kind == "near" else -distances, kind="stable")
    kept = [source.objects[int(index)] for index in rank]
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Region1", "物件"), P("Region2", "參考區域")],
    [P("Selected", "最近的那個物件")],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [],
    "留下離參考區域最近的物件",
    "一樣近時留原來排前面的那個。",
    "Vizion.nearest_one (Region1, Region2, Selected, 'chessboard')",
)
def nearest_one(region1, region2, metric, *, ctx):
    source, found = _pair_distance(region1, region2, metric)
    if source.count() != len(found):
        raise ValueError("篩選時第一組的物件會各自保留，請讓兩邊物件數相同，或第二組只有一個物件")
    best = 0
    best_distance = float(found[0][2][found[0][0]].min())
    for index, (one, _other, dist) in enumerate(found):
        value = float(dist[one].min())
        if value < best_distance:
            best = index
            best_distance = value
    return Result([Region(source.width, source.height, [source.objects[best]])], [])


@operator(
    "區域",
    [P("Region1", "物件"), P("Region2", "參考區域")],
    [P("Selected", "最遠的那個物件")],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [],
    "留下離參考區域最遠的物件",
    "比的是最近距離，所以整塊都離得比較遠的會被留下。一樣遠時留原來排前面的那個。",
    "Vizion.farthest_one (Region1, Region2, Selected, 'chessboard')",
)
def farthest_one(region1, region2, metric, *, ctx):
    source, found = _pair_distance(region1, region2, metric)
    if source.count() != len(found):
        raise ValueError("篩選時第一組的物件會各自保留，請讓兩邊物件數相同，或第二組只有一個物件")
    best = 0
    best_distance = float(found[0][2][found[0][0]].min())
    for index, (one, _other, dist) in enumerate(found):
        value = float(dist[one].min())
        if value > best_distance:
            best = index
            best_distance = value
    return Result([Region(source.width, source.height, [source.objects[best]])], [])


@operator(
    "區域",
    [P("Region1", "第一組物件"), P("Region2", "第二組物件")],
    [P("RegionGap", "兩組中間的空隙")],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [],
    "留下兩組物件中間的空隙",
    "空隙是兩邊都不在、而且到兩邊的距離加起來等於分隔距離的像素。碰到或重疊時沒有空隙。city-block 在左右並排的方塊之間會是一塊矩形。",
    "Vizion.gap_region (Region1, Region2, RegionGap, 'city-block')",
)
def gap_region(region1, region2, metric, *, ctx):
    source, gaps = _gap_fields(region1, region2, metric)
    kept = [space.astype(np.uint8) * 255 for _separation, space in gaps if np.any(space)]
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Region1", "第一組物件"), P("Region2", "第二組物件")],
    [],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [P("Area", "空隙的像素數")],
    "兩組物件中間的空隙有幾個像素",
    "碰到或重疊是 0。一個結果也是 tuple。",
    "Vizion.gap_area (Region1, Region2, 'city-block', Area)",
)
def gap_area(region1, region2, metric, *, ctx):
    _source, gaps = _gap_fields(region1, region2, metric)
    return Result([], [[float(np.count_nonzero(space)) for _separation, space in gaps]])


@operator(
    "區域",
    [P("Region1", "第一組物件"), P("Region2", "第二組物件")],
    [],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [P("Width", "中間空了幾步")],
    "兩組物件中間空了幾步",
    "最近距離減 1。碰到或重疊是 0。一個結果也是 tuple。",
    "Vizion.gap_width (Region1, Region2, 'city-block', Width)",
)
def gap_width(region1, region2, metric, *, ctx):
    _source, gaps = _gap_fields(region1, region2, metric)
    return Result([], [[float(max(0.0, separation - 1.0)) for separation, _space in gaps]])


_SIZE = ["'large'", "'small'"]


def _live_masks(region):
    source = expect_region(region, "Region")
    if source.count() == 0:
        raise ValueError("區域是空的")
    masks = []
    for index, mask in enumerate(source.objects, start=1):
        if int(cv2.countNonZero(mask)) == 0:
            raise ValueError(f"第 {index} 個物件是空的")
        masks.append(mask > 0)
    return source, masks


def _owners(region, metric, domain=None):
    source, masks = _live_masks(region)
    stack = np.stack([_toward(mask, metric) for mask in masks], axis=0)
    owner = np.argmin(stack, axis=0).astype(np.int16)
    if domain is not None:
        limit = expect_region(domain, "Domain")
        if (limit.width, limit.height) != (source.width, source.height):
            raise ValueError("兩個區域的尺寸要相同")
        allowed = limit.union_mask() > 0
        if not np.any(allowed):
            raise ValueError("範圍是空的")
        owner = np.where(allowed, owner, np.int16(-1))
    return source, owner, stack


def _territory_masks(source, owner: np.ndarray) -> list:
    kept = []
    for index in range(int(owner.max()) + 1 if owner.size and int(owner.max()) >= 0 else 0):
        piece = owner == index
        if np.any(piece):
            kept.append(piece.astype(np.uint8) * 255)
    return kept


def _border_mask(owner: np.ndarray) -> np.ndarray:
    mark = np.zeros(owner.shape, np.bool_)
    left, right = owner[:, :-1], owner[:, 1:]
    clash = (left >= 0) & (right >= 0) & (left != right)
    mark[:, :-1] |= clash
    mark[:, 1:] |= clash
    up, down = owner[:-1, :], owner[1:, :]
    clash = (up >= 0) & (down >= 0) & (up != down)
    mark[:-1, :] |= clash
    mark[1:, :] |= clash
    return mark


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Partitioned", "長大到交界的物件")],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [],
    "把每個像素分給最近的物件",
    "整張影像都會分完。距離相同時算排前面的物件。重疊的地方也是。地盤變成空的不會留下。distance_transform 會先把物件合成一塊，這個不會。",
    "Vizion.partition_nearest (Region, Partitioned, 'city-block')",
)
def partition_nearest(region, metric, *, ctx):
    source, owner, _stack = _owners(region, metric)
    kept = _territory_masks(source, owner)
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [P("Area", "每個物件分到的像素數")],
    "每個物件的地盤有幾個像素",
    "和物件一一對應，沒分到是 0。一個結果也是 tuple。",
    "Vizion.partition_area (Region, 'city-block', Area)",
)
def partition_area(region, metric, *, ctx):
    _source, owner, stack = _owners(region, metric)
    areas = [float(np.count_nonzero(owner == index)) for index in range(stack.shape[0])]
    return Result([], [areas])


@operator(
    "區域",
    [P("Region", "物件")],
    [P("ImageLabel", "地盤標籤影像")],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [],
    "把地盤畫成標籤影像",
    "1 是第一個物件的地盤。距離相同時標前面的號碼。超過 255 個物件會失敗。",
    "Vizion.partition_label (Region, ImageLabel, 'city-block')",
)
def partition_label(region, metric, *, ctx):
    source, owner, stack = _owners(region, metric)
    if stack.shape[0] > 255:
        raise ValueError("物件超過 255 個，灰階影像放不下")
    label = np.zeros(owner.shape, np.uint8)
    for index in range(stack.shape[0]):
        label[owner == index] = index + 1
    return Result([Image(label)], [])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [
        P("Row", "列，可以是 tuple"),
        P("Column", "行，可以是 tuple"),
        P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK),
    ],
    [P("Index", "從 1 開始的物件編號")],
    "這個點分給第幾個物件",
    "距離相同時是排前面的那個。一個點回傳一個數，多個點回傳 tuple。",
    "Vizion.owner_index (Region, 10, 20, 'city-block', Index)",
)
def owner_index(region, row, column, metric, *, ctx):
    source, owner, _stack = _owners(region, metric)
    rows = [int(round(float(item))) for item in as_list(row)]
    cols = [int(round(float(item))) for item in as_list(column)]
    if len(rows) != len(cols) or not rows:
        raise ValueError("Row 和 Column 的長度要相同，而且不能是空的")
    indexes = []
    for item_row, item_col in zip(rows, cols):
        if not (0 <= item_row < source.height and 0 <= item_col < source.width):
            raise ValueError("點超出影像範圍")
        indexes.append(float(int(owner[item_row, item_col]) + 1))
    if len(indexes) == 1 and not isinstance(row, (list, tuple)):
        return Result([], [indexes[0]])
    return Result([], [indexes])


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Border", "地盤交界")],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [],
    "留下地盤相接的那一層像素",
    "上下左右有別人的地盤才算。每個還有交界的物件各留一層。距離相同時交界算在前面的物件上。",
    "Vizion.partition_border (Region, Border, 'city-block')",
)
def partition_border(region, metric, *, ctx):
    source, owner, _stack = _owners(region, metric)
    mark = _border_mask(owner)
    kept = []
    for index in range(int(owner.max()) + 1 if owner.size and int(owner.max()) >= 0 else 0):
        piece = mark & (owner == index)
        if np.any(piece):
            kept.append(piece.astype(np.uint8) * 255)
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Selected", "地盤最大或最小的物件")],
    [
        P("Which", "large 最大，small 最小", "'large'", _SIZE),
        P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK),
    ],
    [],
    "留下地盤最大或最小的物件",
    "留下的是長大之後的地盤，不是原來的形狀。一樣大時留排前面的。",
    "Vizion.select_territory (Region, Selected, 'large', 'city-block')",
)
def select_territory(region, which, metric, *, ctx):
    source, owner, stack = _owners(region, metric)
    kind = text(which, "Which").lower()
    if kind not in ("large", "small"):
        raise ValueError("Which 只能是 large 或 small")
    areas = [int(np.count_nonzero(owner == index)) for index in range(stack.shape[0])]
    best = int(np.argmax(areas) if kind == "large" else np.argmin(areas))
    piece = owner == best
    kept = [piece.astype(np.uint8) * 255] if np.any(piece) else []
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Region", "物件"), P("Domain", "只在這個範圍裡分")],
    [P("Partitioned", "範圍內的地盤")],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [],
    "只把指定範圍分給最近的物件",
    "範圍外不屬於任何人。物件可以在範圍外面，仍然用它來算距離。沒分到像素的物件不會留下。",
    "Vizion.partition_domain (Region, Domain, Partitioned, 'city-block')",
)
def partition_domain(region, domain, metric, *, ctx):
    source, owner, _stack = _owners(region, metric, domain)
    kept = _territory_masks(source, owner)
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Ties", "兩個以上物件一樣近的像素")],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [],
    "留下距離打成平手的像素",
    "這些像素在分地盤時會算給排前面的物件。這裡把它們另外圈出來。沒有平手時區域是空的。",
    "Vizion.tie_region (Region, Ties, 'city-block')",
)
def tie_region(region, metric, *, ctx):
    source, _owner, stack = _owners(region, metric)
    best = np.min(stack, axis=0)
    tied = np.count_nonzero(np.abs(stack - best) <= 1e-6, axis=0) >= 2
    kept = [tied.astype(np.uint8) * 255] if np.any(tied) else []
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


_RANK_ORDER = ["'down'", "'up'"]
_EXTREME = ["'max'", "'min'"]
_BAND = ["'above'", "'below'"]


def _feature_values(region, feature):
    source = expect_region(region, "Region")
    if source.count() == 0:
        raise ValueError("區域是空的")
    name = text(feature, "Feature")
    values = np.array([object_feature(mask, name) for mask in source.objects], dtype=np.float64)
    return source, values


def _feature_order(values: np.ndarray, order) -> np.ndarray:
    kind = text(order, "Order").lower()
    if kind not in ("down", "up"):
        raise ValueError("Order 只能是 down 或 up")
    signed = values if kind == "up" else -values
    return np.argsort(signed, kind="stable")


def _take_objects(source, indexes: list):
    masks = [source.objects[index] for index in indexes]
    links = None
    if source.links is not None and len(source.links) == source.count():
        links = [source.links[index] for index in indexes]
    return Result([Region(source.width, source.height, masks, multi_color=len(masks) > 1, links=links)], [])


@operator(
    "特徵",
    [P("Region", "物件")],
    [P("Sorted", "依特徵排好的物件")],
    [
        P("Feature", "用來排序的特徵", "'area'", REGION_FEATURE_CHOICES),
        P("Order", "down 由大到小，up 由小到大", "'down'", _RANK_ORDER),
    ],
    [],
    "依特徵排序物件",
    "特徵和 region_features 相同。一樣大時維持原來的順序。sort_region 仍是依位置排。",
    "Vizion.sort_feature (Region, Sorted, 'area', 'down')",
)
def sort_feature(region, feature, order, *, ctx):
    source, values = _feature_values(region, feature)
    indexes = [int(index) for index in _feature_order(values, order)]
    return _take_objects(source, indexes)


@operator(
    "特徵",
    [P("Region", "物件")],
    [P("Selected", "排在前面的幾個物件")],
    [
        P("Count", "要留下幾個"),
        P("Feature", "用來排序的特徵", "'area'", REGION_FEATURE_CHOICES),
        P("Order", "down 由大到小，up 由小到大", "'down'", _RANK_ORDER),
    ],
    [],
    "留下特徵排在前面的幾個物件",
    "down 時最大的排前面。一樣大時原來排前面的先留下。",
    "Vizion.select_top (Region, Selected, 2, 'area', 'down')",
)
def select_top(region, count, feature, order, *, ctx):
    source, values = _feature_values(region, feature)
    keep = whole(count, "Count")
    if keep < 1:
        raise ValueError("Count 至少要 1")
    if keep > source.count():
        raise ValueError("Count 超出物件數")
    indexes = [int(index) for index in _feature_order(values, order)[:keep]]
    return _take_objects(source, indexes)


@operator(
    "特徵",
    [P("Region", "物件")],
    [P("Selected", "特徵最大或最小的物件")],
    [
        P("Which", "max 最大，min 最小", "'max'", _EXTREME),
        P("Feature", "用來比較的特徵", "'area'", REGION_FEATURE_CHOICES),
    ],
    [],
    "留下特徵最大或最小的物件",
    "一樣大時留原來排前面的那個。",
    "Vizion.select_extreme (Region, Selected, 'max', 'area')",
)
def select_extreme(region, which, feature, *, ctx):
    source, values = _feature_values(region, feature)
    kind = text(which, "Which").lower()
    if kind not in ("max", "min"):
        raise ValueError("Which 只能是 max 或 min")
    best = int(np.argmax(values) if kind == "max" else np.argmin(values))
    return _take_objects(source, [best])


@operator(
    "特徵",
    [P("Region", "物件")],
    [],
    [
        P("Feature", "用來排名的特徵", "'area'", REGION_FEATURE_CHOICES),
        P("Order", "down 最大是第 1 名，up 最小是第 1 名", "'down'", _RANK_ORDER),
    ],
    [P("Rank", "每個物件的名次")],
    "每個物件的名次",
    "名次從 1 開始，順序和原來的物件對齊。一樣大時，原來排前面的名次較小。一個結果也是 tuple。",
    "Vizion.feature_rank (Region, 'area', 'down', Rank)",
)
def feature_rank(region, feature, order, *, ctx):
    _source, values = _feature_values(region, feature)
    placed = _feature_order(values, order)
    rank = np.empty(len(values), dtype=np.float64)
    rank[placed] = np.arange(1, len(values) + 1)
    return Result([], [rank.tolist()])


@operator(
    "特徵",
    [P("Region", "物件")],
    [P("Selected", "指定名次的物件")],
    [
        P("Rank", "從 1 開始，可以是 tuple"),
        P("Feature", "用來排名的特徵", "'area'", REGION_FEATURE_CHOICES),
        P("Order", "down 最大是第 1 名，up 最小是第 1 名", "'down'", _RANK_ORDER),
    ],
    [],
    "留下指定名次的物件",
    "依你寫的名次順序留下。一樣大時，原來排前面的名次較小。",
    "Vizion.select_rank (Region, Selected, 2, 'area', 'down')",
)
def select_rank(region, rank, feature, order, *, ctx):
    source, values = _feature_values(region, feature)
    placed = _feature_order(values, order)
    wanted = []
    seen = set()
    for item in as_list(rank):
        pos = whole(item, "Rank")
        if pos < 1 or pos > source.count():
            raise ValueError("名次超出範圍")
        if pos in seen:
            raise ValueError("名次重複")
        seen.add(pos)
        wanted.append(int(placed[pos - 1]))
    return _take_objects(source, wanted)


@operator(
    "特徵",
    [P("Region", "物件")],
    [],
    [
        P("Feature", "用來排序的特徵", "'area'", REGION_FEATURE_CHOICES),
        P("Order", "down 由大到小，up 由小到大", "'down'", _RANK_ORDER),
    ],
    [P("Index", "排好後的原來編號")],
    "依特徵排序後的原來編號",
    "編號從 1 開始。可以拿去 select_obj。一樣大時維持原來的順序。一個結果也是 tuple。",
    "Vizion.feature_index (Region, 'area', 'down', Index)",
)
def feature_index(region, feature, order, *, ctx):
    source, values = _feature_values(region, feature)
    indexes = [float(int(index) + 1) for index in _feature_order(values, order)]
    return Result([], [indexes])


@operator(
    "特徵",
    [P("Region", "物件")],
    [P("Selected", "落在這一側的物件")],
    [
        P("Percent", "0 到 100 的百分位"),
        P("Feature", "用來比較的特徵", "'area'", REGION_FEATURE_CHOICES),
        P("Side", "above 大於等於，below 小於等於", "'above'", _BAND),
    ],
    [],
    "留下高於或低於百分位的物件",
    "50 是中位數。above 留下大於等於這個值的物件，below 留下小於等於的。維持原來的順序。",
    "Vizion.select_relative (Region, Selected, 50, 'area', 'above')",
)
def select_relative(region, percent, feature, side, *, ctx):
    source, values = _feature_values(region, feature)
    level = scalar(percent, "Percent")
    if not 0 <= level <= 100:
        raise ValueError("Percent 要在 0 到 100")
    kind = text(side, "Side").lower()
    if kind not in ("above", "below"):
        raise ValueError("Side 只能是 above 或 below")
    cutoff = float(np.percentile(values, level))
    if kind == "above":
        indexes = [index for index, value in enumerate(values) if value + 1e-9 >= cutoff]
    else:
        indexes = [index for index, value in enumerate(values) if value <= cutoff + 1e-9]
    return _take_objects(source, indexes)


@operator(
    "特徵",
    [P("Region", "物件")],
    [],
    [P("Feature", "要看的特徵", "'area'", REGION_FEATURE_CHOICES)],
    [P("Minimum", "最小的那個值"), P("Maximum", "最大的那個值")],
    "全部物件裡這個特徵的最小和最大",
    "看的是物件之間，不是一個物件裡的像素。",
    "Vizion.feature_range (Region, 'area', Minimum, Maximum)",
)
def feature_range(region, feature, *, ctx):
    _source, values = _feature_values(region, feature)
    return Result([], [float(np.min(values)), float(np.max(values))])


def _defect_parts(region, min_depth, metric):
    source = expect_region(region, "Region")
    if source.count() == 0:
        raise ValueError("區域是空的")
    limit = scalar(min_depth, "MinDepth")
    if limit < 0:
        raise ValueError("MinDepth 不能是負的")
    flag, size = _metric_code(metric)
    bites = []
    counts = []
    for index, mask in enumerate(source.objects):
        if int(cv2.countNonZero(mask)) == 0:
            raise ValueError(f"第 {index + 1} 個物件是空的")
        hull = _shape_mask(mask, "convex")
        missing = (hull > 0) & ~(mask > 0)
        seeds = np.where(hull > 0, 255, 0).astype(np.uint8)
        if not np.any(missing) or not np.any(seeds == 0):
            counts.append(0)
            continue
        dist = cv2.distanceTransform(seeds, flag, size)
        count, labels = cv2.connectedComponents(missing.astype(np.uint8), connectivity=8)
        kept = 0
        for label in range(1, count):
            piece = labels == label
            depth = float(dist[piece].max())
            if depth + 1e-6 < limit:
                continue
            kept += 1
            pick = int(np.argmax(np.where(piece, dist, -np.inf)))
            row, col = divmod(pick, dist.shape[1])
            bites.append((index, piece.astype(np.uint8) * 255, depth, int(row), int(col)))
        counts.append(kept)
    return source, bites, counts


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Defects", "凹進去的缺口")],
    [
        P("MinDepth", "淺過這個深度的不算", "1"),
        P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK),
    ],
    [],
    "留下凸包補不上的缺口",
    "每個缺口各自一個物件，用 8 連通。深度是缺口裡的像素到凸包外面的距離。缺一角時凸包是斜線，不會補回整個外接矩形。孔洞也算。比 MinDepth 淺的不算。",
    "Vizion.region_defects (Region, Defects, 1, 'city-block')",
)
def region_defects(region, min_depth, metric, *, ctx):
    source, bites, _counts = _defect_parts(region, min_depth, metric)
    kept = [mask for _index, mask, _depth, _row, _col in bites]
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [
        P("MinDepth", "淺過這個深度的不算", "1"),
        P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK),
    ],
    [P("Count", "每個物件有幾個缺口")],
    "每個物件有幾個缺口",
    "和物件一一對應，沒有缺口是 0。一個結果也是 tuple。",
    "Vizion.defect_count (Region, 1, 'city-block', Count)",
)
def defect_count(region, min_depth, metric, *, ctx):
    _source, _bites, counts = _defect_parts(region, min_depth, metric)
    return Result([], [[float(count) for count in counts]])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [
        P("MinDepth", "淺過這個深度的不算", "1"),
        P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK),
    ],
    [P("Area", "每個缺口的像素數")],
    "每個缺口有幾個像素",
    "順序和 region_defects 相同，不是依原物件對齊。沒有缺口時是空 tuple。",
    "Vizion.defect_area (Region, 1, 'city-block', Area)",
)
def defect_area(region, min_depth, metric, *, ctx):
    _source, bites, _counts = _defect_parts(region, min_depth, metric)
    return Result([], [[float(cv2.countNonZero(mask)) for _index, mask, _depth, _row, _col in bites]])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [
        P("MinDepth", "淺過這個深度的不算", "1"),
        P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK),
    ],
    [P("Depth", "每個缺口最深的距離")],
    "每個缺口有多深",
    "順序和 region_defects 相同。一個結果也是 tuple。",
    "Vizion.defect_depth (Region, 1, 'city-block', Depth)",
)
def defect_depth(region, min_depth, metric, *, ctx):
    _source, bites, _counts = _defect_parts(region, min_depth, metric)
    return Result([], [[depth for _index, _mask, depth, _row, _col in bites]])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [
        P("MinDepth", "淺過這個深度的不算", "1"),
        P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK),
    ],
    [P("Row", "最深的點的列"), P("Column", "最深的點的行"), P("Depth", "那個點的深度")],
    "每個缺口最深的那個點",
    "一樣深時留列最小、再來行最小的點。順序和 region_defects 相同。",
    "Vizion.defect_point (Region, 1, 'city-block', Row, Column, Depth)",
)
def defect_point(region, min_depth, metric, *, ctx):
    _source, bites, _counts = _defect_parts(region, min_depth, metric)
    rows = [float(row) for _index, _mask, _depth, row, _col in bites]
    cols = [float(col) for _index, _mask, _depth, _row, col in bites]
    depths = [depth for _index, _mask, depth, _row, _col in bites]
    return Result([], [rows, cols, depths])


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Selected", "有夠深缺口的物件")],
    [
        P("MinDepth", "淺過這個深度的不算", "1"),
        P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK),
    ],
    [],
    "留下有夠深缺口的物件",
    "留下的是原來的物件，不是缺口本身。維持原來的順序。",
    "Vizion.select_defect (Region, Selected, 1, 'city-block')",
)
def select_defect(region, min_depth, metric, *, ctx):
    source, _bites, counts = _defect_parts(region, min_depth, metric)
    kept = [mask for mask, count in zip(source.objects, counts) if count]
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Defect", "最深的那個缺口")],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [],
    "留下最深的那個缺口",
    "一樣深時留先出現的物件、再來列較小的缺口。沒有缺口時區域是空的。",
    "Vizion.deepest_defect (Region, Defect, 'city-block')",
)
def deepest_defect(region, metric, *, ctx):
    source, bites, _counts = _defect_parts(region, 0, metric)
    if not bites:
        return Result([Region(source.width, source.height, [])], [])
    best = 0
    for index, item in enumerate(bites):
        if item[2] > bites[best][2]:
            best = index
    return Result([Region(source.width, source.height, [bites[best][1]])], [])


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Filled", "缺口補上後的物件")],
    [
        P("MinDepth", "淺過這個深度的不補", "1"),
        P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK),
    ],
    [],
    "把夠深的缺口補平",
    "只補深度達到 MinDepth 的缺口，物件的數目和順序不變。沒有缺口的維持原樣。",
    "Vizion.fill_defects (Region, Filled, 1, 'city-block')",
)
def fill_defects(region, min_depth, metric, *, ctx):
    source, bites, _counts = _defect_parts(region, min_depth, metric)
    filled = [mask.copy() for mask in source.objects]
    for index, piece, _depth, _row, _col in bites:
        filled[index][piece > 0] = 255
    return Result([Region(source.width, source.height, filled, multi_color=len(filled) > 1)], [])


def _solid_mask(mask: np.ndarray) -> np.ndarray:
    work = np.where(mask > 0, 255, 0).astype(np.uint8)
    found = cv2.findContours(work.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]
    solid = np.zeros(work.shape, np.uint8)
    if contours:
        cv2.drawContours(solid, contours, -1, 255, thickness=-1)
    return solid > 0


def _mask_bounds(mask: np.ndarray):
    rows, cols = np.nonzero(mask)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _nesting(region):
    source = expect_region(region, "Region")
    if source.count() == 0:
        raise ValueError("區域是空的")
    masks = []
    solids = []
    solid_area = []
    object_area = []
    for index, mask in enumerate(source.objects):
        if int(cv2.countNonZero(mask)) == 0:
            raise ValueError(f"第 {index + 1} 個物件是空的")
        binary = mask > 0
        masks.append(binary)
        solid = _solid_mask(mask)
        solids.append(solid)
        solid_area.append(int(np.count_nonzero(solid)))
        object_area.append(int(np.count_nonzero(binary)))
    count = len(masks)
    contains = np.zeros((count, count), np.bool_)
    bounds = [_mask_bounds(item) for item in masks]
    solid_bounds = [_mask_bounds(item) for item in solids]
    for inner in range(count):
        inner_rows = bounds[inner]
        for outer in range(count):
            if inner == outer or solid_area[outer] <= object_area[inner]:
                continue
            top, left, bottom, right = solid_bounds[outer]
            if inner_rows[0] < top or inner_rows[1] < left or inner_rows[2] > bottom or inner_rows[3] > right:
                continue
            if not np.any(masks[inner] & ~solids[outer]):
                contains[outer, inner] = True
    holders = []
    for inner in range(count):
        options = np.flatnonzero(contains[:, inner])
        if len(options) == 0:
            holders.append(0)
            continue
        best = min(options, key=lambda item: (solid_area[int(item)], int(item)))
        holders.append(int(best) + 1)
    inside = contains.sum(axis=1).astype(np.int32)
    depth = contains.sum(axis=0).astype(np.int32)
    return source, holders, inside, depth


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Selected", "被別的物件包住的")],
    [],
    [],
    "留下被別的物件包住的物件",
    "先把對方的外輪廓填滿，孔洞裡也算裡面。自己不算。兩塊完全一樣大時互不算裡面。維持原來的順序。",
    "Vizion.select_inner (Region, Selected)",
)
def select_inner(region, *, ctx):
    source, _holders, _inside, depth = _nesting(region)
    indexes = [index for index, level in enumerate(depth) if int(level) > 0]
    return _take_objects(source, indexes)


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Selected", "沒有被包住的物件")],
    [],
    [],
    "留下沒有被包住的物件",
    "最外層，或單獨的一塊，都會留下。",
    "Vizion.select_outer (Region, Selected)",
)
def select_outer(region, *, ctx):
    source, _holders, _inside, depth = _nesting(region)
    indexes = [index for index, level in enumerate(depth) if int(level) == 0]
    return _take_objects(source, indexes)


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Selected", "裡面還有別的物件")],
    [],
    [],
    "留下裡面還有別的物件的物件",
    "直接包住或透過中間那層包住，都算。",
    "Vizion.select_container (Region, Selected)",
)
def select_container(region, *, ctx):
    source, _holders, inside, _depth = _nesting(region)
    indexes = [index for index, count in enumerate(inside) if int(count) > 0]
    return _take_objects(source, indexes)


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Selected", "被包住、而且裡面是空的")],
    [],
    [],
    "留下最裡面的物件",
    "它自己被別的物件包住，而且裡面沒有再包別的物件。",
    "Vizion.innermost (Region, Selected)",
)
def innermost(region, *, ctx):
    source, _holders, inside, depth = _nesting(region)
    indexes = [index for index, (level, count) in enumerate(zip(depth, inside)) if int(level) > 0 and int(count) == 0]
    return _take_objects(source, indexes)


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Selected", "這一層的物件")],
    [P("Depth", "0 是最外層")],
    [],
    "留下被包住幾層的物件",
    "0 是沒有被包住的。1 是只在一層裡面。數字是包住它的物件有幾個。",
    "Vizion.select_depth (Region, Selected, 1)",
)
def select_depth(region, depth, *, ctx):
    source, _holders, _inside, levels = _nesting(region)
    level = whole(depth, "Depth")
    if level < 0:
        raise ValueError("Depth 不能是負的")
    indexes = [index for index, item in enumerate(levels) if int(item) == level]
    return _take_objects(source, indexes)


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [],
    [P("Index", "包住它的最小物件，從 1 開始。沒有是 0")],
    "每個物件被哪一個包住",
    "有好幾層時，留面積最小的那一個。面積一樣留編號較小的。一個結果也是 tuple。",
    "Vizion.container_index (Region, Index)",
)
def container_index(region, *, ctx):
    _source, holders, _inside, _depth = _nesting(region)
    return Result([], [[float(item) for item in holders]])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [],
    [P("Count", "裡面有幾個物件")],
    "每個物件裡面有幾個物件",
    "直接或間接包住的都算。沒有是 0。一個結果也是 tuple。",
    "Vizion.inner_count (Region, Count)",
)
def inner_count(region, *, ctx):
    _source, _holders, inside, _depth = _nesting(region)
    return Result([], [[float(item) for item in inside]])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [],
    [P("Depth", "被幾層包住")],
    "每個物件被幾層包住",
    "沒有被包住是 0。一個結果也是 tuple。",
    "Vizion.nesting_depth (Region, Depth)",
)
def nesting_depth(region, *, ctx):
    _source, _holders, _inside, depth = _nesting(region)
    return Result([], [[float(item) for item in depth]])


_EDGE_MODE = ["'touch'", "'inner'"]
_EDGE = ["'top'", "'bottom'", "'left'", "'right'"]
_EDGE_RANK = {"top": 0, "bottom": 1, "left": 2, "right": 3}


def _border_facts(region):
    source = expect_region(region, "Region")
    if source.count() == 0:
        raise ValueError("區域是空的")
    height = source.height
    width = source.width
    facts = []
    for index, mask in enumerate(source.objects):
        if int(cv2.countNonZero(mask)) == 0:
            raise ValueError(f"第 {index + 1} 個物件是空的")
        rows, cols = np.nonzero(mask)
        top = int(rows.min())
        bottom = height - 1 - int(rows.max())
        left = int(cols.min())
        right = width - 1 - int(cols.max())
        distance = min(top, bottom, left, right)
        pixel = np.minimum(np.minimum(rows, height - 1 - rows), np.minimum(cols, width - 1 - cols))
        hit = int(np.flatnonzero(pixel == distance)[0])
        row = int(rows[hit])
        col = int(cols[hit])
        scores = (
            ("top", row),
            ("bottom", height - 1 - row),
            ("left", col),
            ("right", width - 1 - col),
        )
        side = min(scores, key=lambda item: (item[1], _EDGE_RANK[item[0]]))[0]
        facts.append((top, bottom, left, right, distance, row, col, side))
    return source, facts


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Selected", "碰到邊緣或完整落在裡面的物件")],
    [P("Mode", "touch 碰到邊緣，inner 完全在裡面", "'inner'", _EDGE_MODE)],
    [],
    "依有沒有碰到影像邊緣來篩選",
    "touch 留下碰到上、下、左或右邊緣的物件。inner 留下四邊都還有空隙的。",
    "Vizion.select_border (Region, Selected, 'inner')",
)
def select_border(region, mode, *, ctx):
    source, facts = _border_facts(region)
    kind = text(mode, "Mode").lower()
    if kind not in ("touch", "inner"):
        raise ValueError("Mode 只能是 touch 或 inner")
    indexes = [index for index, item in enumerate(facts) if (item[4] == 0) == (kind == "touch")]
    return _take_objects(source, indexes)


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Selected", "碰到這一邊的物件")],
    [P("Side", "top、bottom、left 或 right", None, _EDGE)],
    [],
    "留下碰到影像某一邊的物件",
    "一個物件可以同時碰到兩邊，例如左上角。",
    "Vizion.select_border_side (Region, Selected, 'top')",
)
def select_border_side(region, side, *, ctx):
    source, facts = _border_facts(region)
    kind = text(side, "Side").lower()
    slot = {"top": 0, "bottom": 1, "left": 2, "right": 3}
    if kind not in slot:
        raise ValueError("Side 只能是 top、bottom、left 或 right")
    indexes = [index for index, item in enumerate(facts) if item[slot[kind]] == 0]
    return _take_objects(source, indexes)


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Selected", "離邊緣夠遠的物件")],
    [P("Margin", "至少要離邊緣幾像素", "1")],
    [],
    "留下離影像邊緣夠遠的物件",
    "距離小於 Margin 的會去掉。1 會去掉碰到邊緣的。0 會全部留下。",
    "Vizion.select_margin (Region, Selected, 1)",
)
def select_margin(region, margin, *, ctx):
    source, facts = _border_facts(region)
    limit = scalar(margin, "Margin")
    if limit < 0:
        raise ValueError("Margin 不能是負的")
    indexes = [index for index, item in enumerate(facts) if item[4] + 1e-6 >= limit]
    return _take_objects(source, indexes)


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Selected", "落在這個距離範圍的物件")],
    [P("Low", "最小距離，含"), P("High", "最大距離，含")],
    [],
    "留下離邊緣的距離落在範圍裡的物件",
    "比的是到最近那一邊的距離。上下限寫反會自動對調。",
    "Vizion.select_border_range (Region, Selected, 1, 10)",
)
def select_border_range(region, low, high, *, ctx):
    source, facts = _border_facts(region)
    start = scalar(low, "Low")
    stop = scalar(high, "High")
    if start > stop:
        start, stop = stop, start
    indexes = [index for index, item in enumerate(facts) if start - 1e-6 <= item[4] <= stop + 1e-6]
    return _take_objects(source, indexes)


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Selected", "離邊緣最遠的物件")],
    [],
    [],
    "留下離影像邊緣最遠的物件",
    "一樣遠時留原來排前面的那個。",
    "Vizion.farthest_border (Region, Selected)",
)
def farthest_border(region, *, ctx):
    source, facts = _border_facts(region)
    best = 0
    for index, item in enumerate(facts):
        if item[4] > facts[best][4]:
            best = index
    return _take_objects(source, [best])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [],
    [P("Distance", "到最近邊緣的距離")],
    "每個物件離影像邊緣多遠",
    "碰到邊緣是 0。一個結果也是 tuple。",
    "Vizion.border_distance (Region, Distance)",
)
def border_distance(region, *, ctx):
    _source, facts = _border_facts(region)
    return Result([], [[float(item[4]) for item in facts]])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [],
    [
        P("Top", "離上邊"),
        P("Bottom", "離下邊"),
        P("Left", "離左邊"),
        P("Right", "離右邊"),
    ],
    "每個物件離影像四邊多遠",
    "0 表示那一邊已經碰到。每個結果都是 tuple。",
    "Vizion.border_gap (Region, Top, Bottom, Left, Right)",
)
def border_gap(region, *, ctx):
    _source, facts = _border_facts(region)
    return Result([], [[float(item[slot]) for item in facts] for slot in range(4)])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [],
    [P("Row", "最近點的列"), P("Column", "最近點的行"), P("Side", "top、bottom、left 或 right")],
    "每個物件最靠近影像邊緣的點",
    "一樣近時留列最小、再來行最小的點。同一個點離兩邊一樣近時，優先上、下、左、右。",
    "Vizion.border_point (Region, Row, Column, Side)",
)
def border_point(region, *, ctx):
    _source, facts = _border_facts(region)
    return Result(
        [],
        [
            [float(item[5]) for item in facts],
            [float(item[6]) for item in facts],
            [item[7] for item in facts],
        ],
    )


def _sibling_views(region, metric):
    source = expect_region(region, "Region")
    if source.count() == 0:
        raise ValueError("區域是空的")
    if source.count() < 2:
        raise ValueError("至少要有兩個物件")
    masks = []
    for index, mask in enumerate(source.objects):
        if int(cv2.countNonZero(mask)) == 0:
            raise ValueError(f"第 {index + 1} 個物件是空的")
        masks.append(mask > 0)
    fields = [_toward(mask, metric) for mask in masks]
    return source, masks, fields


def _gap_limit(max_distance):
    limit = scalar(max_distance, "MaxDistance")
    if limit < 0:
        raise ValueError("MaxDistance 不能是負的")
    return limit


def _closest_sibling(fields, mask, index):
    best_value = None
    best_index = 0
    best_row = 0
    best_col = 0
    width = mask.shape[1]
    for other, field in enumerate(fields):
        if other == index:
            continue
        work = np.where(mask, field, np.inf)
        pick = int(np.argmin(work))
        row, col = divmod(pick, width)
        value = float(field[row, col])
        if best_value is None or value < best_value - 1e-6 or (abs(value - best_value) <= 1e-6 and other < best_index):
            best_value = value
            best_index = other
            best_row = row
            best_col = col
    return best_index, best_row, best_col, best_value


def _neighbor_counts(fields, masks, limit):
    counts = []
    for index, mask in enumerate(masks):
        total = 0
        for other, field in enumerate(fields):
            if other != index and float(field[mask].min()) <= limit + 1e-6:
                total += 1
        counts.append(total)
    return counts


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [P("Distance", "到最近的另一個物件")],
    "每個物件離其他物件多遠",
    "重疊是 0，挨著但沒重疊是 1。斜角在 chessboard 是 1，city-block 是 2。至少要有兩個物件。一個結果也是 tuple。",
    "Vizion.neighbor_distance (Region, 'chessboard', Distance)",
)
def neighbor_distance(region, metric, *, ctx):
    _source, masks, fields = _sibling_views(region, metric)
    distances = [_closest_sibling(fields, mask, index)[3] for index, mask in enumerate(masks)]
    return Result([], [distances])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [P("Index", "最近的另一個物件，從 1 開始")],
    "每個物件最近的是哪一個",
    "一樣近時留編號較小的。一個結果也是 tuple。",
    "Vizion.neighbor_index (Region, 'chessboard', Index)",
)
def neighbor_index(region, metric, *, ctx):
    _source, masks, fields = _sibling_views(region, metric)
    indexes = [float(_closest_sibling(fields, mask, index)[0] + 1) for index, mask in enumerate(masks)]
    return Result([], [indexes])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK)],
    [P("Row", "最近點的列"), P("Column", "最近點的行"), P("Distance", "到那個物件的距離")],
    "每個物件最靠近其他物件的點",
    "一樣近時留列最小、再來行最小的點。有兩個物件一樣近時，留編號較小的那個。",
    "Vizion.neighbor_point (Region, 'chessboard', Row, Column, Distance)",
)
def neighbor_point(region, metric, *, ctx):
    _source, masks, fields = _sibling_views(region, metric)
    rows, cols, distances = [], [], []
    for index, mask in enumerate(masks):
        _other, row, col, value = _closest_sibling(fields, mask, index)
        rows.append(float(row))
        cols.append(float(col))
        distances.append(value)
    return Result([], [rows, cols, distances])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [
        P("MaxDistance", "多近才算鄰居", "1"),
        P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK),
    ],
    [P("Count", "鄰居有幾個")],
    "每個物件有幾個鄰居",
    "0 只算重疊。1 連挨著的也算。斜角要不要算，看距離的算法。一個結果也是 tuple。",
    "Vizion.neighbor_count (Region, 1, 'chessboard', Count)",
)
def neighbor_count(region, max_distance, metric, *, ctx):
    _source, masks, fields = _sibling_views(region, metric)
    counts = _neighbor_counts(fields, masks, _gap_limit(max_distance))
    return Result([], [[float(count) for count in counts]])


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Selected", "有鄰居的物件")],
    [
        P("MaxDistance", "多近才算鄰居", "1"),
        P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK),
    ],
    [],
    "留下有鄰居的物件",
    "留下的是原來的物件。維持原來的順序。",
    "Vizion.select_touching (Region, Selected, 1, 'chessboard')",
)
def select_touching(region, max_distance, metric, *, ctx):
    source, masks, fields = _sibling_views(region, metric)
    counts = _neighbor_counts(fields, masks, _gap_limit(max_distance))
    indexes = [index for index, count in enumerate(counts) if count]
    return _take_objects(source, indexes)


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Selected", "沒有鄰居的物件")],
    [
        P("MaxDistance", "多近才算鄰居", "1"),
        P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK),
    ],
    [],
    "留下沒有鄰居的物件",
    "在 MaxDistance 以內沒有別人的會留下。",
    "Vizion.select_isolated (Region, Selected, 1, 'chessboard')",
)
def select_isolated(region, max_distance, metric, *, ctx):
    source, masks, fields = _sibling_views(region, metric)
    counts = _neighbor_counts(fields, masks, _gap_limit(max_distance))
    indexes = [index for index, count in enumerate(counts) if count == 0]
    return _take_objects(source, indexes)


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Contact", "靠近其他物件的像素")],
    [
        P("MaxDistance", "多近才算碰到", "1"),
        P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK),
    ],
    [],
    "留下靠近其他物件的那一層像素",
    "每個有碰到的物件各留一層。完全沒碰到的不會留下。",
    "Vizion.contact_region (Region, Contact, 1, 'chessboard')",
)
def contact_region(region, max_distance, metric, *, ctx):
    source, masks, fields = _sibling_views(region, metric)
    limit = _gap_limit(max_distance)
    kept = []
    for index, mask in enumerate(masks):
        others = [field for item, field in enumerate(fields) if item != index]
        dist = others[0] if len(others) == 1 else np.min(np.stack(others, axis=0), axis=0)
        piece = mask & (dist <= limit + 1e-6)
        if np.any(piece):
            kept.append(piece.astype(np.uint8) * 255)
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Union", "碰到的物件合成一塊")],
    [
        P("MaxDistance", "多近就合成一塊", "1"),
        P("Metric", "chessboard、city-block 或 euclidean", "'chessboard'", _THICK),
    ],
    [],
    "把互相碰到的物件合成一塊",
    "透過中間的物件間接碰到也會合成同一塊。沒碰到的維持原樣。順序依原來最前面的那個。",
    "Vizion.union_touching (Region, Union, 1, 'chessboard')",
)
def union_touching(region, max_distance, metric, *, ctx):
    source, masks, fields = _sibling_views(region, metric)
    limit = _gap_limit(max_distance)
    parent = list(range(len(masks)))

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for index, mask in enumerate(masks):
        for other in range(index + 1, len(masks)):
            if float(fields[other][mask].min()) > limit + 1e-6:
                continue
            left, right = find(index), find(other)
            if left == right:
                continue
            if left < right:
                parent[right] = left
            else:
                parent[left] = right
    groups = {}
    for index in range(len(masks)):
        groups.setdefault(find(index), []).append(index)
    kept = []
    for root in sorted(groups, key=lambda item: min(groups[item])):
        piece = np.zeros(masks[0].shape, np.uint8)
        for index in groups[root]:
            piece[masks[index]] = 255
        kept.append(piece)
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


_SHAPE_METHOD = ["'i1'", "'i2'", "'i3'"]


def _shape_method(method):
    kind = text(method, "Method").lower()
    codes = {
        "i1": cv2.CONTOURS_MATCH_I1,
        "i2": cv2.CONTOURS_MATCH_I2,
        "i3": cv2.CONTOURS_MATCH_I3,
    }
    if kind not in codes:
        raise ValueError("Method 只能是 i1、i2 或 i3")
    return codes[kind]


def _outer_contour(mask, index: int, name: str):
    if int(cv2.countNonZero(mask)) == 0:
        raise ValueError(f"第 {index + 1} 個{name}是空的")
    work = np.where(mask > 0, 255, 0).astype(np.uint8)
    found = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]
    if not contours:
        raise ValueError(f"第 {index + 1} 個{name}太小，無法比較形狀")
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        raise ValueError(f"第 {index + 1} 個{name}太小，無法比較形狀")
    return contour


def _shape_jobs(region, pattern, method):
    source = expect_region(region, "Region")
    model = expect_region(pattern, "Pattern")
    if source.count() == 0 or model.count() == 0:
        raise ValueError("區域是空的")
    if model.count() != 1 and model.count() != source.count():
        raise ValueError("範本要只有一個物件，或兩邊物件數相同")
    code = _shape_method(method)
    models = [_outer_contour(mask, index, "範本") for index, mask in enumerate(model.objects)]
    if model.count() == 1:
        models = models * source.count()
    jobs = []
    for index, mask in enumerate(source.objects):
        jobs.append((mask, _outer_contour(mask, index, "物件"), models[index]))
    return source, jobs, code


def _shape_scores(jobs, code) -> np.ndarray:
    return np.array([float(cv2.matchShapes(contour, model, code, 0.0)) for _mask, contour, model in jobs], dtype=np.float64)


@operator(
    "區域",
    [P("Region", "要比較的物件"), P("Pattern", "範本")],
    [],
    [P("Method", "i1、i2 或 i3", "'i1'", _SHAPE_METHOD)],
    [P("Score", "越小越像")],
    "每個物件和範本有多像",
    "分數越小越像，0 是同一種輪廓。搬家、旋轉、放大縮小都不影響，所以大小不同的方塊分數接近 0，長條會大很多。i1 最常用，i2、i3 對細節更敏感。範本只有一個時會比對每一個物件；兩邊個數相同才逐一配對。Score[1] 是第一個，從 1 開始。",
    "Vizion.shape_score (Region, Pattern, 'i1', Score)",
)
def shape_score(region, pattern, method, *, ctx):
    _source, jobs, code = _shape_jobs(region, pattern, method)
    return Result([], [_shape_scores(jobs, code).tolist()])


@operator(
    "區域",
    [P("Region", "要比較的物件"), P("Pattern", "範本")],
    [P("Selected", "夠像的物件")],
    [
        P("MaxScore", "分數上限，含"),
        P("Method", "i1、i2 或 i3", "'i1'", _SHAPE_METHOD),
    ],
    [],
    "留下和範本夠像的物件",
    "先用 shape_score 看分數，再把 MaxScore 設在要留和要丟之間。0.01 留得下方塊和轉過的方塊，長條約 3.5 會被去掉。留下的仍是原來的順序。",
    "Vizion.select_similar (Region, Pattern, Selected, 0.1, 'i1')",
)
def select_similar(region, pattern, max_score, method, *, ctx):
    source, jobs, code = _shape_jobs(region, pattern, method)
    limit = scalar(max_score, "MaxScore")
    scores = _shape_scores(jobs, code)
    indexes = [index for index, score in enumerate(scores) if score <= limit + 1e-9]
    return _take_objects(source, indexes)


@operator(
    "區域",
    [P("Region", "要比較的物件"), P("Pattern", "範本")],
    [P("Selected", "最像的物件")],
    [P("Method", "i1、i2 或 i3", "'i1'", _SHAPE_METHOD)],
    [],
    "留下和範本最像的物件",
    "只要最像的那一個，不必自己找最小分數。一樣像時留原來排前面的。",
    "Vizion.most_similar (Region, Pattern, Selected, 'i1')",
)
def most_similar(region, pattern, method, *, ctx):
    source, jobs, code = _shape_jobs(region, pattern, method)
    scores = _shape_scores(jobs, code)
    best = int(np.argmin(scores))
    return _take_objects(source, [best])


@operator(
    "區域",
    [P("Region", "要比較的物件"), P("Pattern", "範本")],
    [P("Selected", "最不像的物件")],
    [P("Method", "i1、i2 或 i3", "'i1'", _SHAPE_METHOD)],
    [],
    "留下和範本最不像的物件",
    "只要最不像的那一個。一樣不像時留原來排前面的。",
    "Vizion.least_similar (Region, Pattern, Selected, 'i1')",
)
def least_similar(region, pattern, method, *, ctx):
    source, jobs, code = _shape_jobs(region, pattern, method)
    scores = _shape_scores(jobs, code)
    worst = int(np.argmax(scores))
    return _take_objects(source, [worst])


@operator(
    "區域",
    [P("Region", "要比較的物件"), P("Pattern", "範本")],
    [P("Sorted", "依相似度排好的物件")],
    [
        P("Order", "near 最像的在前，far 最不像的在前", "'near'", _NEAR),
        P("Method", "i1、i2 或 i3", "'i1'", _SHAPE_METHOD),
    ],
    [],
    "依和範本的相似度排序",
    "near 最像的排前面，far 反過來。方法和順序都可省略，預設 i1、最像的在前。一樣像時維持原來的順序。",
    "Vizion.sort_shape (Region, Pattern, Sorted, 'near', 'i1')",
)
def sort_shape(region, pattern, order, method, *, ctx):
    source, jobs, code = _shape_jobs(region, pattern, method)
    kind = text(order, "Order").lower()
    if kind not in ("near", "far"):
        raise ValueError("Order 只能是 near 或 far")
    scores = _shape_scores(jobs, code)
    rank = np.argsort(scores if kind == "near" else -scores, kind="stable")
    return _take_objects(source, [int(index) for index in rank])


@operator(
    "區域",
    [P("Region", "要比較的物件"), P("Pattern", "範本")],
    [],
    [P("Method", "i1、i2 或 i3", "'i1'", _SHAPE_METHOD)],
    [P("Rank", "1 是最像")],
    "每個物件和範本的名次",
    "名次對齊原來的物件。Rank[1] 是第一個物件，1 代表最像。一樣像時，原來排前面的名次較小。",
    "Vizion.shape_rank (Region, Pattern, 'i1', Rank)",
)
def shape_rank(region, pattern, method, *, ctx):
    _source, jobs, code = _shape_jobs(region, pattern, method)
    scores = _shape_scores(jobs, code)
    placed = np.argsort(scores, kind="stable")
    rank = np.empty(len(scores), dtype=np.float64)
    rank[placed] = np.arange(1, len(scores) + 1)
    return Result([], [rank.tolist()])


@operator(
    "區域",
    [P("Region", "要比較的物件"), P("Pattern", "範本")],
    [],
    [P("Method", "i1、i2 或 i3", "'i1'", _SHAPE_METHOD)],
    [P("Index", "由像到不像的原來編號")],
    "依相似度排序後的原來編號",
    "由像到不像列出原來的編號，從 1 開始。下一行可寫 Vizion.select_obj (Region, Picked, Index)。一樣像時維持原來的順序。",
    "Vizion.shape_index (Region, Pattern, 'i1', Index)",
)
def shape_index(region, pattern, method, *, ctx):
    _source, jobs, code = _shape_jobs(region, pattern, method)
    scores = _shape_scores(jobs, code)
    indexes = [float(int(index) + 1) for index in np.argsort(scores, kind="stable")]
    return Result([], [indexes])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [],
    [
        P("Hu1", "第 1 個不變矩"),
        P("Hu2", "第 2 個不變矩"),
        P("Hu3", "第 3 個不變矩"),
        P("Hu4", "第 4 個不變矩"),
        P("Hu5", "第 5 個不變矩"),
        P("Hu6", "第 6 個不變矩"),
        P("Hu7", "第 7 個不變矩"),
    ],
    "區域的七個 Hu 不變矩",
    "七個數描述形狀，不看位置、角度和大小。方塊的 Hu1 約 0.16667。要判斷像不像範本，用 shape_score 比較直接。Hu1[1] 是第一個物件。",
    "Vizion.hu_moments (Region, Hu1, Hu2, Hu3, Hu4, Hu5, Hu6, Hu7)",
)
def hu_moments(region, *, ctx):
    source = expect_region(region, "Region")
    if source.count() == 0:
        raise ValueError("區域是空的")
    values = [[] for _ in range(7)]
    for index, mask in enumerate(source.objects):
        contour = _outer_contour(mask, index, "物件")
        hu = cv2.HuMoments(cv2.moments(contour)).reshape(-1)
        for slot, item in enumerate(hu):
            values[slot].append(float(item))
    return Result([], values)


_ALIGN = ["'center'", "'rigid'"]


def _fold_phi(delta: float) -> float:
    while delta > math.pi / 2:
        delta -= math.pi
    while delta < -math.pi / 2:
        delta += math.pi
    return delta


def _pose_of(mask: np.ndarray):
    moments = cv2.moments(mask)
    if not moments["m00"]:
        return None
    row = float(moments["m01"] / moments["m00"])
    col = float(moments["m10"] / moments["m00"])
    mu20 = float(moments["mu20"])
    mu02 = float(moments["mu02"])
    mu11 = float(moments["mu11"])
    if mu20 + mu02 < 1e-6:
        phi = 0.0
    else:
        phi = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
    return row, col, phi


def _align_mode(mode) -> str:
    kind = text(mode, "Mode").lower()
    if kind not in ("center", "rigid"):
        raise ValueError("Mode 只能是 center 或 rigid")
    return kind


def _align_jobs(region, pattern):
    source = expect_region(region, "Region")
    model = expect_region(pattern, "Pattern")
    if source.count() == 0 or model.count() == 0:
        raise ValueError("區域是空的")
    if (source.width, source.height) != (model.width, model.height):
        raise ValueError("兩個區域的尺寸要相同")
    if model.count() != 1 and model.count() != source.count():
        raise ValueError("範本要只有一個物件，或兩邊物件數相同")
    patterns = []
    for index, mask in enumerate(model.objects):
        pose = _pose_of(mask)
        if pose is None:
            raise ValueError(f"第 {index + 1} 個範本是空的")
        patterns.append((mask > 0, pose))
    if model.count() == 1:
        patterns = patterns * source.count()
    jobs = []
    for index, mask in enumerate(source.objects):
        pose = _pose_of(mask)
        if pose is None:
            raise ValueError(f"第 {index + 1} 個物件是空的")
        pattern_mask, pattern_pose = patterns[index]
        drow = pattern_pose[0] - pose[0]
        dcol = pattern_pose[1] - pose[1]
        dphi = _fold_phi(pattern_pose[2] - pose[2])
        jobs.append((mask, pose, pattern_mask, pattern_pose, drow, dcol, dphi))
    return source, jobs


def _warp_rigid(mask, obj_row, obj_col, pat_row, pat_col, dphi) -> np.ndarray:
    cosine = math.cos(dphi)
    sine = math.sin(dphi)
    matrix = np.array(
        [
            [cosine, -sine, pat_col - cosine * obj_col + sine * obj_row],
            [sine, cosine, pat_row - sine * obj_col - cosine * obj_row],
        ],
        np.float64,
    )
    warped = cv2.warpAffine(mask, matrix, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)
    return np.where(warped > 0, 255, 0).astype(np.uint8)


def _placed(mask, pose, pattern_pose, drow, dcol, dphi, mode: str) -> np.ndarray:
    if mode == "center":
        return _shift_mask(mask, int(round(drow)), int(round(dcol)))
    return _warp_rigid(mask, pose[0], pose[1], pattern_pose[0], pattern_pose[1], dphi)


def _placed_stats(aligned: np.ndarray, pattern_mask: np.ndarray):
    both = (aligned > 0) & pattern_mask
    either = (aligned > 0) ^ pattern_mask
    overlap = int(np.count_nonzero(both))
    mismatch = int(np.count_nonzero(either))
    total = int(np.count_nonzero(pattern_mask))
    cover = float(overlap) / float(total) if total else 0.0
    return overlap, mismatch, cover, either


@operator(
    "區域",
    [P("Region", "要搬的物件"), P("Pattern", "範本")],
    [],
    [],
    [P("DRow", "還要往下搬的列"), P("DColumn", "還要往右搬的行")],
    "物件中心要搬多少才對上範本",
    "正的列往下，正的行往右。這是中心的差，還沒有四捨五入。範本只有一個時會拿來比對每一個。DRow[1] 是第一個物件，從 1 開始。",
    "Vizion.align_delta (Region, Pattern, DRow, DColumn)",
)
def align_delta(region, pattern, *, ctx):
    _source, jobs = _align_jobs(region, pattern)
    return Result([], [[item[4] for item in jobs], [item[5] for item in jobs]])


@operator(
    "區域",
    [P("Region", "要轉的物件"), P("Pattern", "範本")],
    [],
    [],
    [P("DPhi", "還要順時針轉的角度，弳度")],
    "物件的長軸要轉多少才和範本同向",
    "0 大致沿著行，正角度在畫面上順時針。長軸顛倒 180 度當成同一個方向，所以結果在 -90 度到 90 度之間。方塊的長軸不穩定，長條才適合看這個角度。",
    "Vizion.align_angle (Region, Pattern, DPhi)",
)
def align_angle(region, pattern, *, ctx):
    _source, jobs = _align_jobs(region, pattern)
    return Result([], [[item[6] for item in jobs]])


@operator(
    "區域",
    [P("Region", "要搬的物件"), P("Pattern", "範本")],
    [P("Aligned", "搬到範本上的物件")],
    [P("Mode", "center 只對齊中心，rigid 連長軸一起轉", "'center'", _ALIGN)],
    [],
    "把物件搬到範本的位置",
    "center 只搬中心，位移會四捨五入成整數像素，方塊請用這個。rigid 再把長軸轉成和範本一樣。移出影像的部分會被切掉。",
    "Vizion.align_region (Region, Pattern, Aligned, 'center')",
)
def align_region(region, pattern, mode, *, ctx):
    source, jobs = _align_jobs(region, pattern)
    kind = _align_mode(mode)
    kept = []
    for mask, pose, _pattern_mask, pattern_pose, drow, dcol, dphi in jobs:
        moved = _placed(mask, pose, pattern_pose, drow, dcol, dphi, kind)
        if int(cv2.countNonZero(moved)):
            kept.append(moved)
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Region", "要搬的物件"), P("Pattern", "範本")],
    [],
    [P("Mode", "center 只對齊中心，rigid 連長軸一起轉", "'center'", _ALIGN)],
    [P("Area", "搬過去之後和範本重疊的像素")],
    "搬到範本上之後重疊幾個像素",
    "先對齊再算交集。完全疊上時等於範本的面積。一個結果也是 tuple。",
    "Vizion.align_overlap (Region, Pattern, 'center', Area)",
)
def align_overlap(region, pattern, mode, *, ctx):
    _source, jobs = _align_jobs(region, pattern)
    kind = _align_mode(mode)
    areas = []
    for mask, pose, pattern_mask, pattern_pose, drow, dcol, dphi in jobs:
        moved = _placed(mask, pose, pattern_pose, drow, dcol, dphi, kind)
        areas.append(float(_placed_stats(moved, pattern_mask)[0]))
    return Result([], [areas])


@operator(
    "區域",
    [P("Region", "要搬的物件"), P("Pattern", "範本")],
    [],
    [P("Mode", "center 只對齊中心，rigid 連長軸一起轉", "'center'", _ALIGN)],
    [P("Cover", "蓋住範本的比例")],
    "搬過去之後蓋住範本多少",
    "1 是完全蓋住範本，0.95 是至少蓋住 95%。分母是範本的像素，不是物件自己的。",
    "Vizion.align_cover (Region, Pattern, 'center', Cover)",
)
def align_cover(region, pattern, mode, *, ctx):
    _source, jobs = _align_jobs(region, pattern)
    kind = _align_mode(mode)
    covers = []
    for mask, pose, pattern_mask, pattern_pose, drow, dcol, dphi in jobs:
        moved = _placed(mask, pose, pattern_pose, drow, dcol, dphi, kind)
        covers.append(_placed_stats(moved, pattern_mask)[2])
    return Result([], [covers])


@operator(
    "區域",
    [P("Region", "要搬的物件"), P("Pattern", "範本")],
    [],
    [P("Mode", "center 只對齊中心，rigid 連長軸一起轉", "'center'", _ALIGN)],
    [P("Area", "對齊後只有一邊有的像素")],
    "搬過去之後還差幾個像素",
    "物件多出來的和範本沒被蓋到的都算。完全疊上是 0。一個結果也是 tuple。",
    "Vizion.mismatch_area (Region, Pattern, 'center', Area)",
)
def mismatch_area(region, pattern, mode, *, ctx):
    _source, jobs = _align_jobs(region, pattern)
    kind = _align_mode(mode)
    areas = []
    for mask, pose, pattern_mask, pattern_pose, drow, dcol, dphi in jobs:
        moved = _placed(mask, pose, pattern_pose, drow, dcol, dphi, kind)
        areas.append(float(_placed_stats(moved, pattern_mask)[1]))
    return Result([], [areas])


@operator(
    "區域",
    [P("Region", "要搬的物件"), P("Pattern", "範本")],
    [P("Mismatch", "對齊後沒有重疊的像素")],
    [P("Mode", "center 只對齊中心，rigid 連長軸一起轉", "'center'", _ALIGN)],
    [],
    "留下對齊後還沒重疊的像素",
    "完全疊上的物件不會留下。要看差幾個像素，用 mismatch_area，0 也會留在 tuple 裡。",
    "Vizion.mismatch_region (Region, Pattern, Mismatch, 'center')",
)
def mismatch_region(region, pattern, mode, *, ctx):
    source, jobs = _align_jobs(region, pattern)
    kind = _align_mode(mode)
    kept = []
    for mask, pose, pattern_mask, pattern_pose, drow, dcol, dphi in jobs:
        moved = _placed(mask, pose, pattern_pose, drow, dcol, dphi, kind)
        either = _placed_stats(moved, pattern_mask)[3]
        if np.any(either):
            kept.append(either.astype(np.uint8) * 255)
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Region", "要搬的物件"), P("Pattern", "範本")],
    [P("Selected", "搬過去之後夠重合的物件")],
    [
        P("MinCover", "至少要蓋住範本的比例"),
        P("Mode", "center 只對齊中心，rigid 連長軸一起轉", "'center'", _ALIGN),
    ],
    [],
    "留下搬到範本上之後夠重合的物件",
    "先看 align_cover。1 表示要完全蓋住，0.95 表示至少 95%。留下的是原來的物件，不是搬完的結果。",
    "Vizion.select_aligned (Region, Pattern, Selected, 0.95, 'center')",
)
def select_aligned(region, pattern, min_cover, mode, *, ctx):
    source, jobs = _align_jobs(region, pattern)
    kind = _align_mode(mode)
    limit = scalar(min_cover, "MinCover")
    if limit < 0:
        raise ValueError("MinCover 不能是負的")
    indexes = []
    for index, (mask, pose, pattern_mask, pattern_pose, drow, dcol, dphi) in enumerate(jobs):
        moved = _placed(mask, pose, pattern_pose, drow, dcol, dphi, kind)
        if _placed_stats(moved, pattern_mask)[2] + 1e-9 >= limit:
            indexes.append(index)
    return _take_objects(source, indexes)


def _grid_view(region, gap):
    source = expect_region(region, "Region")
    if source.count() == 0:
        raise ValueError("區域是空的")
    limit = scalar(gap, "Gap")
    if limit < 0:
        raise ValueError("Gap 不能是負的")
    rows, cols, heights, widths = [], [], [], []
    for index, mask in enumerate(source.objects):
        moments = cv2.moments(mask)
        if not moments["m00"]:
            raise ValueError(f"第 {index + 1} 個物件是空的")
        rows.append(float(moments["m01"] / moments["m00"]))
        cols.append(float(moments["m10"] / moments["m00"]))
        _x, _y, width, height = cv2.boundingRect(mask)
        heights.append(int(height))
        widths.append(int(width))
    if limit == 0:
        limit = float(np.median(np.asarray(heights, dtype=np.float64)))
    order = sorted(range(len(rows)), key=lambda index: (rows[index], cols[index], index))
    clusters = []
    for index in order:
        if not clusters or rows[index] - clusters[-1]["anchor"] > limit:
            clusters.append({"members": [index], "anchor": rows[index]})
        else:
            clusters[-1]["members"].append(index)
    gaps = []
    for cluster in clusters:
        members = sorted(cluster["members"], key=lambda index: (cols[index], index))
        cluster["members"] = members
        cluster["row"] = float(np.median([rows[index] for index in members]))
        for left, right in zip(members, members[1:]):
            delta = cols[right] - cols[left]
            if delta > 1e-6:
                gaps.append(delta)
    pitch = float(min(gaps)) if gaps else 0.0
    origin = float(min(cols))
    col_index = [1] * len(rows)
    if pitch > 1e-6:
        for index, col in enumerate(cols):
            col_index[index] = int(round((col - origin) / pitch)) + 1
    row_index = [1] * len(rows)
    for number, cluster in enumerate(clusters, start=1):
        for index in cluster["members"]:
            row_index[index] = number
    if len(clusters) >= 2:
        deltas = [clusters[index + 1]["row"] - clusters[index]["row"] for index in range(len(clusters) - 1)]
        row_pitch = float(np.median(deltas))
    else:
        row_pitch = 0.0
    return {
        "source": source,
        "rows": rows,
        "cols": cols,
        "heights": heights,
        "widths": widths,
        "clusters": clusters,
        "row_index": row_index,
        "col_index": col_index,
        "row_pitch": row_pitch,
        "pitch": pitch,
        "origin": origin,
        "max_col": max(col_index) if col_index else 1,
    }


def _grid_slot(view, row, column):
    row_no = whole(row, "Row")
    col_no = whole(column, "Column")
    if row_no < 1 or col_no < 1 or row_no > len(view["clusters"]) or col_no > view["max_col"]:
        raise ValueError(f"第 {row_no} 列第 {col_no} 行超出格子範圍")
    cluster = view["clusters"][row_no - 1]
    found = [index for index in cluster["members"] if view["col_index"][index] == col_no]
    return row_no, col_no, cluster, found


def _cell_mask(view, center_row, center_col) -> np.ndarray:
    source = view["source"]
    height = int(np.median(np.asarray(view["heights"], dtype=np.float64)))
    width = int(np.median(np.asarray(view["widths"], dtype=np.float64)))
    row1 = int(round(center_row - (height - 1) / 2.0))
    col1 = int(round(center_col - (width - 1) / 2.0))
    row2 = row1 + height
    col2 = col1 + width
    mask = np.zeros((source.height, source.width), np.uint8)
    row1 = max(0, row1)
    col1 = max(0, col1)
    row2 = min(source.height, row2)
    col2 = min(source.width, col2)
    if row2 > row1 and col2 > col1:
        mask[row1:row2, col1:col2] = 255
    return mask


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [P("Gap", "中心列差超過這個值就換一列。0 用物件高度", "0")],
    [P("Row", "第幾列，從 1 開始"), P("Column", "這一列的第幾行，從 1 開始")],
    "每個物件在第幾列第幾行",
    "由上到下分列，由左到右分行。Gap 填 0 時，用物件高度當門檻，中心列差比高度大就換下一列。行距用最小的相鄰間距，約兩倍就當成中間缺一格。順序和原來的物件對齊，Row[1] 是第一個。",
    "Vizion.grid_index (Region, Row, Column)",
)
def grid_index(region, gap, *, ctx):
    view = _grid_view(region, gap)
    return Result([], [[float(item) for item in view["row_index"]], [float(item) for item in view["col_index"]]])


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Sorted", "由左上到右下排好的物件")],
    [P("Gap", "中心列差超過這個值就換一列。0 用物件高度", "0")],
    [],
    "依格子由左上到右下排序",
    "先上列再下列，同一列由左到右。缺掉的格子會跳過。Gap 可省略，0 表示用物件高度。",
    "Vizion.sort_grid (Region, Sorted)",
)
def sort_grid(region, gap, *, ctx):
    view = _grid_view(region, gap)
    indexes = [index for cluster in view["clusters"] for index in cluster["members"]]
    return _take_objects(view["source"], indexes)


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [P("Gap", "中心列差超過這個值就換一列。0 用物件高度", "0")],
    [P("Rows", "有幾列"), P("Columns", "有幾行，含缺格")],
    "物件排成幾列幾行",
    "行數含缺掉的格子。上面三格、下面左右各一格、中間空的，仍算 2 列 3 行。Gap 填 0 用物件高度。",
    "Vizion.grid_size (Region, Rows, Columns)",
)
def grid_size(region, gap, *, ctx):
    view = _grid_view(region, gap)
    return Result([], [float(len(view["clusters"])), float(view["max_col"])])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [P("Gap", "中心列差超過這個值就換一列。0 用物件高度", "0")],
    [P("RowPitch", "列距"), P("ColumnPitch", "行距")],
    "格子的列距和行距",
    "列距是相鄰兩列中心的差。行距是同一列裡最小的相鄰間距。只有一列或一行時，那個間距是 0。",
    "Vizion.grid_pitch (Region, RowPitch, ColumnPitch)",
)
def grid_pitch(region, gap, *, ctx):
    view = _grid_view(region, gap)
    return Result([], [view["row_pitch"], view["pitch"]])


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Missing", "缺掉的格子")],
    [P("Gap", "中心列差超過這個值就換一列。0 用物件高度", "0")],
    [],
    "補上排程中缺掉的格子",
    "某一列的間距大約是最小行距的兩倍，中間就補一格，大小用物件的中位數。每一列都缺同一行時看不出來，因為沒有較小的行距可當基準。",
    "Vizion.missing_cell (Region, Missing)",
)
def missing_cell(region, gap, *, ctx):
    view = _grid_view(region, gap)
    kept = []
    for cluster in view["clusters"]:
        present = {view["col_index"][index] for index in cluster["members"]}
        for column in range(1, view["max_col"] + 1):
            if column in present or view["pitch"] <= 1e-6:
                continue
            center_col = view["origin"] + (column - 1) * view["pitch"]
            piece = _cell_mask(view, cluster["row"], center_col)
            if int(cv2.countNonZero(piece)):
                kept.append(piece)
    return Result([Region(view["source"].width, view["source"].height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Selected", "這一格的物件")],
    [
        P("Row", "第幾列，從 1 開始"),
        P("Column", "第幾行，從 1 開始"),
        P("Gap", "中心列差超過這個值就換一列。0 用物件高度", "0"),
    ],
    [],
    "取出某一列某一行的物件",
    "編號從 1 開始。這一格是空的會報錯，空格用 missing_cell 看。同一個格子有多個物件時全部留下。",
    "Vizion.select_grid (Region, Selected, 2, 3)",
)
def select_grid(region, row, column, gap, *, ctx):
    view = _grid_view(region, gap)
    row_no, col_no, _cluster, found = _grid_slot(view, row, column)
    if not found:
        raise ValueError(f"第 {row_no} 列第 {col_no} 行沒有物件")
    return _take_objects(view["source"], found)


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [
        P("Row", "第幾列，從 1 開始"),
        P("Column", "第幾行，從 1 開始"),
        P("Gap", "中心列差超過這個值就換一列。0 用物件高度", "0"),
    ],
    [P("CenterRow", "這一格的中心列"), P("CenterColumn", "這一格的中心行")],
    "某一格的中心，沒有物件也算",
    "有物件就用它的中心。沒有物件就用格線算出來的位置，方便把相機或範本移過去。行列超出範圍才會報錯。",
    "Vizion.cell_center (Region, 2, 2, CenterRow, CenterColumn)",
)
def cell_center(region, row, column, gap, *, ctx):
    view = _grid_view(region, gap)
    _row_no, col_no, cluster, found = _grid_slot(view, row, column)
    if found:
        index = found[0]
        return Result([], [view["rows"][index], view["cols"][index]])
    center_col = view["origin"] + (col_no - 1) * view["pitch"]
    return Result([], [cluster["row"], center_col])


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Label", "行號圖")],
    [P("Gap", "中心列差超過這個值就換一列。0 用物件高度", "0")],
    [],
    "把行號畫成灰階",
    "像素值就是行號，從 1 開始。空白和缺掉的格子是 0。行數超過 255 放不進灰階圖。",
    "Vizion.grid_label (Region, Label)",
)
def grid_label(region, gap, *, ctx):
    view = _grid_view(region, gap)
    source = view["source"]
    if view["max_col"] > 255:
        raise ValueError("行數超過 255，灰階影像放不下")
    label = np.zeros((source.height, source.width), np.uint8)
    for index, mask in enumerate(source.objects):
        label[mask > 0] = view["col_index"][index]
    return Result([Image(label)], [])


_BRIGHT_SIDE = ["'right'", "'left'", "'below'", "'above'", "'center'"]


def _gray_balance(drow, dcol, row1, col1, row2, col2, grow, gcol):
    distance = math.hypot(drow, dcol)
    if distance < 1e-9:
        return 0.0, 1.0, "center"
    if abs(drow) > abs(dcol):
        side = "below" if drow > 0 else "above"
    elif abs(dcol) > abs(drow):
        side = "right" if dcol > 0 else "left"
    else:
        side = "below" if drow > 0 else "above"
    unit_row = drow / distance
    unit_col = dcol / distance
    limits = []
    if unit_row > 1e-12:
        limits.append((row2 - grow) / unit_row)
    elif unit_row < -1e-12:
        limits.append((row1 - grow) / unit_row)
    if unit_col > 1e-12:
        limits.append((col2 - gcol) / unit_col)
    elif unit_col < -1e-12:
        limits.append((col1 - gcol) / unit_col)
    reach = min(limits) if limits else distance
    balance = 1.0 - distance / reach if reach > 1e-12 else 0.0
    return distance, float(min(1.0, max(0.0, balance))), side


def _gray_facts(region, image):
    source = expect_region(region, "Region")
    picture = expect_image(image, "Image")
    gray = picture.gray().astype(np.float64)
    if gray.shape[0] != source.height or gray.shape[1] != source.width:
        raise ValueError("區域和影像的尺寸要相同")
    if source.count() == 0:
        raise ValueError("區域是空的")
    facts = []
    for index, mask in enumerate(source.objects):
        geo = cv2.moments(mask)
        if not geo["m00"]:
            raise ValueError(f"第 {index + 1} 個物件是空的")
        grow = float(geo["m01"] / geo["m00"])
        gcol = float(geo["m10"] / geo["m00"])
        bright = cv2.moments(np.where(mask > 0, gray, 0.0))
        total = float(bright["m00"])
        if total <= 1e-9:
            drow = dcol = 0.0
        else:
            drow = float(bright["m01"] / total) - grow
            dcol = float(bright["m10"] / total) - gcol
        _x, _y, width, height = cv2.boundingRect(mask)
        row1, col1 = float(_y), float(_x)
        distance, balance, side = _gray_balance(drow, dcol, row1, col1, row1 + height - 1, col1 + width - 1, grow, gcol)
        facts.append(
            {
                "mask": mask,
                "grow": grow,
                "gcol": gcol,
                "drow": drow,
                "dcol": dcol,
                "distance": distance,
                "balance": balance,
                "side": side,
                "phi": _bearing(drow, dcol),
            }
        )
    return source, facts


def _bright_side(side) -> str:
    kind = text(side, "Side").lower()
    if kind not in ("right", "left", "below", "above", "center"):
        raise ValueError("Side 只能是 right、left、below、above 或 center")
    return kind


def _half_mask(mask, grow, gcol, side, dark: bool) -> np.ndarray:
    if dark:
        side = {"right": "left", "left": "right", "below": "above", "above": "below", "center": "center"}[side]
    if side == "center":
        return mask.copy()
    height, width = mask.shape[:2]
    grid_col = np.arange(width, dtype=np.float64)[None, :]
    grid_row = np.arange(height, dtype=np.float64)[:, None]
    inside = mask > 0
    if side == "right":
        keep = inside & (grid_col > gcol)
    elif side == "left":
        keep = inside & (grid_col < gcol)
    elif side == "below":
        keep = inside & (grid_row > grow)
    else:
        keep = inside & (grid_row < grow)
    out = np.zeros_like(mask)
    out[keep] = 255
    return out


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [],
    [],
    [P("DRow", "亮度重心比形狀中心低多少"), P("DColumn", "亮度重心比形狀中心右多少")],
    "亮度重心偏離形狀中心多少",
    "正的列表示亮的部分偏下，正的行表示偏右。0 是亮度正在形狀中間。全黑當成在正中間。彩色先轉成亮度。DRow[1] 是第一個物件。",
    "Vizion.gray_offset (Region, Image, DRow, DColumn)",
)
def gray_offset(region, image, *, ctx):
    _source, facts = _gray_facts(region, image)
    return Result([], [[item["drow"] for item in facts], [item["dcol"] for item in facts]])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [],
    [],
    [P("Distance", "亮度重心離開形狀中心的距離")],
    "亮度重心偏了多遠",
    "就是偏移的直線距離。均勻或全黑是 0。一個結果也是 tuple。",
    "Vizion.gray_shift (Region, Image, Distance)",
)
def gray_shift(region, image, *, ctx):
    _source, facts = _gray_facts(region, image)
    return Result([], [[item["distance"] for item in facts]])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [],
    [],
    [P("Phi", "偏亮的方向，度。0 向右，順時針")],
    "亮度往哪個方向偏",
    "0 度向右，90 度向下，180 度向左，270 度向上。沒有偏移時是 0。",
    "Vizion.gray_direction (Region, Image, Phi)",
)
def gray_direction(region, image, *, ctx):
    _source, facts = _gray_facts(region, image)
    return Result([], [[item["phi"] for item in facts]])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [],
    [],
    [P("Side", "right、left、below、above 或 center")],
    "亮的部分在哪一側",
    "看偏移比較大的那個方向。列和行一樣大時先算上或下。幾乎沒偏就是 center。",
    "Vizion.bright_side (Region, Image, Side)",
)
def bright_side(region, image, *, ctx):
    _source, facts = _gray_facts(region, image)
    return Result([], [[item["side"] for item in facts]])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [],
    [],
    [P("Balance", "1 在正中間，0 全擠在邊緣")],
    "亮度有多靠中間",
    "1 是亮度重心和形狀中心重疊，0 是再偏就出了物件。方塊右半邊較亮時大約是 0.68，不是 0。",
    "Vizion.gray_balance (Region, Image, Balance)",
)
def gray_balance(region, image, *, ctx):
    _source, facts = _gray_facts(region, image)
    return Result([], [[item["balance"] for item in facts]])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [P("Half", "較亮的那一半")],
    [],
    [],
    "留下物件裡較亮的那一半",
    "用形狀中心切開，留下偏亮的那一側。亮度在中間時留下整個物件。先看 bright_side 再決定要不要用這個結果。",
    "Vizion.bright_half (Region, Image, Half)",
)
def bright_half(region, image, *, ctx):
    source, facts = _gray_facts(region, image)
    kept = []
    for item in facts:
        piece = _half_mask(item["mask"], item["grow"], item["gcol"], item["side"], False)
        if int(cv2.countNonZero(piece)):
            kept.append(piece)
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [P("Half", "較暗的那一半")],
    [],
    [],
    "留下物件裡較暗的那一半",
    "和 bright_half 相反的那一側。亮度在中間時同樣留下整個物件。",
    "Vizion.dark_half (Region, Image, Half)",
)
def dark_half(region, image, *, ctx):
    source, facts = _gray_facts(region, image)
    kept = []
    for item in facts:
        piece = _half_mask(item["mask"], item["grow"], item["gcol"], item["side"], True)
        if int(cv2.countNonZero(piece)):
            kept.append(piece)
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [P("Selected", "亮部在指定側的物件")],
    [
        P("Side", "right、left、below、above 或 center", None, _BRIGHT_SIDE),
        P("MinShift", "至少要偏這麼遠。center 則是不能超過", "0.5"),
    ],
    [],
    "留下亮部在某一側的物件",
    "先用 gray_offset 看偏了多少。right 這類方向要偏移至少 MinShift 才留下。center 則是偏移小於 MinShift 才算在中間。留下的是原來的物件。",
    "Vizion.select_bright (Region, Image, Selected, 'right', 1)",
)
def select_bright(region, image, side, min_shift, *, ctx):
    source, facts = _gray_facts(region, image)
    kind = _bright_side(side)
    limit = scalar(min_shift, "MinShift")
    if limit < 0:
        raise ValueError("MinShift 不能是負的")
    indexes = []
    for index, item in enumerate(facts):
        if kind == "center":
            matched = item["distance"] < limit + 1e-9
        else:
            matched = item["side"] == kind and item["distance"] + 1e-9 >= limit
        if matched:
            indexes.append(index)
    return _take_objects(source, indexes)


_RING = ["'inner'", "'outer'"]
_EDGE_KERNEL = np.ones((3, 3), np.uint8)


def _ring_kind(which) -> str:
    kind = text(which, "Which").lower()
    if kind not in ("inner", "outer"):
        raise ValueError("Which 只能是 inner 或 outer")
    return kind


def _edge_facts(region, image):
    source = expect_region(region, "Region")
    picture = expect_image(image, "Image")
    gray = picture.gray().astype(np.float64)
    if gray.shape[0] != source.height or gray.shape[1] != source.width:
        raise ValueError("區域和影像的尺寸要相同")
    if source.count() == 0:
        raise ValueError("區域是空的")
    facts = []
    for index, mask in enumerate(source.objects):
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        if not int(cv2.countNonZero(binary)):
            raise ValueError(f"第 {index + 1} 個物件是空的")
        eroded = cv2.erode(binary, _EDGE_KERNEL)
        dilated = cv2.dilate(binary, _EDGE_KERNEL)
        inner = (binary > 0) & (eroded == 0)
        outer = (dilated > 0) & (binary == 0)
        if not np.any(outer):
            raise ValueError(f"第 {index + 1} 個物件貼滿影像，沒有外側可比較")
        inner_mean = float(gray[inner].mean()) if np.any(inner) else float(gray[binary > 0].mean())
        outer_mean = float(gray[outer].mean())
        facts.append(
            {
                "mask": binary,
                "inner": inner,
                "outer": outer,
                "inner_mean": inner_mean,
                "outer_mean": outer_mean,
                "contrast": inner_mean - outer_mean,
            }
        )
    return source, facts


def _abs_limit(value, name) -> float:
    limit = scalar(value, name)
    if limit < 0:
        raise ValueError(f"{name} 不能是負的")
    return limit


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [],
    [],
    [P("Contrast", "內圈減外圈。正的表示物件比較亮")],
    "邊緣內外側差多少灰階",
    "內圈是物件最外一像素，外圈是貼在物件外面的一像素。正的表示物件比外面亮，負的表示外面比較亮。0 是這裡沒有邊緣。彩色先轉成亮度。Contrast[1] 是第一個物件。",
    "Vizion.edge_contrast (Region, Image, Contrast)",
)
def edge_contrast(region, image, *, ctx):
    _source, facts = _edge_facts(region, image)
    return Result([], [[item["contrast"] for item in facts]])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [],
    [],
    [P("Contrast", "內外圈灰階差的絕對值")],
    "邊緣有多清楚",
    "不管物件比較亮還是外面比較亮，只看差多少。0 是邊界兩邊一樣，看不出邊緣。",
    "Vizion.edge_abs (Region, Image, Contrast)",
)
def edge_abs(region, image, *, ctx):
    _source, facts = _edge_facts(region, image)
    return Result([], [[abs(item["contrast"]) for item in facts]])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [],
    [],
    [P("Mean", "內圈的平均灰階")],
    "物件最外一圈的平均灰階",
    "只看物件自己的最外一像素，不含外面。物件只有一像素厚時，這一圈就是整個物件。",
    "Vizion.inner_gray (Region, Image, Mean)",
)
def inner_gray(region, image, *, ctx):
    _source, facts = _edge_facts(region, image)
    return Result([], [[item["inner_mean"] for item in facts]])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [],
    [],
    [P("Mean", "外圈的平均灰階")],
    "貼在物件外面那一圈的平均灰階",
    "外圈是緊貼物件、不屬於物件的一像素。物件貼滿整張影像時沒有外圈。",
    "Vizion.outer_gray (Region, Image, Mean)",
)
def outer_gray(region, image, *, ctx):
    _source, facts = _edge_facts(region, image)
    return Result([], [[item["outer_mean"] for item in facts]])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [P("Selected", "邊緣夠清楚的物件")],
    [P("MinContrast", "內外圈至少要差這麼多", "10")],
    [],
    "留下邊緣清楚的物件",
    "用差的絕對值，物件較亮或外面較亮都算。先看 edge_abs 再決定門檻。10 表示兩邊至少差 10 個灰階。留下的是原來的物件。",
    "Vizion.select_edge (Region, Image, Selected, 10)",
)
def select_edge(region, image, min_contrast, *, ctx):
    source, facts = _edge_facts(region, image)
    limit = _abs_limit(min_contrast, "MinContrast")
    indexes = [index for index, item in enumerate(facts) if abs(item["contrast"]) + 1e-9 >= limit]
    return _take_objects(source, indexes)


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [P("Selected", "邊緣不明顯的物件")],
    [P("MaxContrast", "內外圈差不能超過", "10")],
    [],
    "留下邊緣不明顯的物件",
    "差的絕對值小於等於 MaxContrast 才留下。0 表示邊界兩邊完全一樣。用來抓沒有對到背景的物件。",
    "Vizion.select_soft (Region, Image, Selected, 10)",
)
def select_soft(region, image, max_contrast, *, ctx):
    source, facts = _edge_facts(region, image)
    limit = _abs_limit(max_contrast, "MaxContrast")
    indexes = [index for index, item in enumerate(facts) if abs(item["contrast"]) <= limit + 1e-9]
    return _take_objects(source, indexes)


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [P("Ring", "內圈或外圈")],
    [P("Which", "inner 物件最外一圈，outer 外面那一圈", "'inner'", _RING)],
    [],
    "取出邊緣的內圈或外圈",
    "inner 是物件自己的最外一像素，outer 是貼在外面的一像素。先看 edge_contrast 知道兩邊差多少，再用這個把那一圈畫出來。",
    "Vizion.edge_ring (Region, Image, Ring, 'inner')",
)
def edge_ring(region, image, which, *, ctx):
    source, facts = _edge_facts(region, image)
    kind = _ring_kind(which)
    kept = []
    for item in facts:
        piece = item["inner"] if kind == "inner" else item["outer"]
        if np.any(piece):
            kept.append(piece.astype(np.uint8) * 255)
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [P("Selected", "邊緣最清楚的物件")],
    [],
    [],
    "留下邊緣最清楚的那一個",
    "比的是內外圈灰階差的絕對值。一樣清楚時留原來排前面的。",
    "Vizion.strongest_edge (Region, Image, Selected)",
)
def strongest_edge(region, image, *, ctx):
    source, facts = _edge_facts(region, image)
    best = max(range(len(facts)), key=lambda index: (abs(facts[index]["contrast"]), -index))
    return _take_objects(source, [best])


_STEP_NAMES = ("above", "below", "left", "right")


def _neighbor_steps(binary: np.ndarray, gray: np.ndarray) -> np.ndarray:
    inside = binary > 0
    signed = np.full((4, inside.shape[0], inside.shape[1]), np.inf, dtype=np.float64)
    shifts = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for index, (drow, dcol) in enumerate(shifts):
        valid = np.zeros(inside.shape, dtype=bool)
        neighbor = np.zeros(gray.shape, dtype=np.float64)
        if drow == -1:
            valid[1:, :] = ~inside[:-1, :]
            neighbor[1:, :] = gray[:-1, :]
        elif drow == 1:
            valid[:-1, :] = ~inside[1:, :]
            neighbor[:-1, :] = gray[1:, :]
        elif dcol == -1:
            valid[:, 1:] = ~inside[:, :-1]
            neighbor[:, 1:] = gray[:, :-1]
        else:
            valid[:, :-1] = ~inside[:, 1:]
            neighbor[:, :-1] = gray[:, 1:]
        signed[index] = np.where(valid, gray - neighbor, np.inf)
    return signed


def _spot_facts(region, image):
    source = expect_region(region, "Region")
    picture = expect_image(image, "Image")
    gray = picture.gray().astype(np.float64)
    if gray.shape[0] != source.height or gray.shape[1] != source.width:
        raise ValueError("區域和影像的尺寸要相同")
    if source.count() == 0:
        raise ValueError("區域是空的")
    facts = []
    for index, mask in enumerate(source.objects):
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        if not int(cv2.countNonZero(binary)):
            raise ValueError(f"第 {index + 1} 個物件是空的")
        eroded = cv2.erode(binary, _EDGE_KERNEL)
        inner = (binary > 0) & (eroded == 0)
        signed = _neighbor_steps(binary, gray)
        direction = np.argmin(np.abs(signed), axis=0)
        chosen = np.take_along_axis(signed, direction[None, :, :], axis=0)[0]
        strength = np.abs(chosen)
        strength[~inner] = np.inf
        if not np.isfinite(strength).any():
            raise ValueError(f"第 {index + 1} 個物件沒有外側可比較")
        flat = int(np.argmin(strength))
        row, col = divmod(flat, source.width)
        facts.append(
            {
                "signed": float(chosen[row, col]),
                "floor": float(strength[row, col]),
                "row": float(row),
                "col": float(col),
                "side": _STEP_NAMES[int(direction[row, col])],
                "strength": strength,
            }
        )
    return source, facts


def _spot_limit(value, name) -> float:
    limit = scalar(value, name)
    if limit < 0:
        raise ValueError(f"{name} 不能是負的")
    return limit


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [],
    [],
    [P("Contrast", "最弱那一點的內減外。負的表示外面比較亮")],
    "邊緣最弱那一點差多少",
    "edge_contrast 是整圈的平均。這裡只看最弱的一點：物件最外一像素，減掉緊鄰的外側像素。負的表示這一點外面比較亮。外側有好幾個時，先看上、下、左、右。一樣弱時留列最小、再來行最小的那一點。Contrast[1] 是第一個物件。",
    "Vizion.edge_min (Region, Image, Contrast)",
)
def edge_min(region, image, *, ctx):
    _source, facts = _spot_facts(region, image)
    return Result([], [[item["signed"] for item in facts]])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [],
    [],
    [P("Floor", "最弱那一點的灰階差，不分誰亮")],
    "邊緣最弱的地方還差多少",
    "只看強度。0 表示這一點兩邊一樣，邊緣在這裡斷掉。整圈都很清楚時，這個數會接近 edge_abs。",
    "Vizion.edge_floor (Region, Image, Floor)",
)
def edge_floor(region, image, *, ctx):
    _source, facts = _spot_facts(region, image)
    return Result([], [[item["floor"] for item in facts]])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [],
    [],
    [P("Row", "最弱點的列"), P("Column", "最弱點的行")],
    "邊緣最弱的那一點在哪",
    "點在物件自己的最外一像素上，不是外面那一圈。一樣弱時留列最小、再來行最小的。",
    "Vizion.edge_min_point (Region, Image, Row, Column)",
)
def edge_min_point(region, image, *, ctx):
    _source, facts = _spot_facts(region, image)
    return Result([], [[item["row"] for item in facts], [item["col"] for item in facts]])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [],
    [],
    [P("Side", "弱點朝外的方向：above、below、left 或 right")],
    "最弱的那一點朝哪邊掉下去",
    "above 是緊鄰的外側在這個點的上方。外側有好幾個一樣弱時，先看上、下、左、右。",
    "Vizion.edge_min_side (Region, Image, Side)",
)
def edge_min_side(region, image, *, ctx):
    _source, facts = _spot_facts(region, image)
    return Result([], [[item["side"] for item in facts]])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [],
    [P("MaxContrast", "差不超過這個值就算弱點", "30")],
    [P("Count", "弱點有幾個像素")],
    "邊緣上有幾個弱點",
    "數的是物件最外圈裡，和外側的差不超過 MaxContrast 的像素。0 表示整圈都比這個門檻清楚。一個結果也是 tuple。",
    "Vizion.weak_count (Region, Image, 30, Count)",
)
def weak_count(region, image, max_contrast, *, ctx):
    _source, facts = _spot_facts(region, image)
    limit = _spot_limit(max_contrast, "MaxContrast")
    counts = [float(np.count_nonzero(item["strength"] <= limit + 1e-9)) for item in facts]
    return Result([], [counts])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [P("Weak", "弱點像素")],
    [P("MaxContrast", "差不超過這個值就算弱點", "30")],
    [],
    "留下邊緣上的弱點",
    "整圈都清楚的物件不會留下。要看每一個物件有幾個弱點，用 weak_count，0 也留在 tuple 裡。",
    "Vizion.weak_edge (Region, Image, Weak, 30)",
)
def weak_edge(region, image, max_contrast, *, ctx):
    source, facts = _spot_facts(region, image)
    limit = _spot_limit(max_contrast, "MaxContrast")
    kept = []
    for item in facts:
        piece = item["strength"] <= limit + 1e-9
        if np.any(piece):
            mask = np.zeros((source.height, source.width), np.uint8)
            mask[piece] = 255
            kept.append(mask)
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [P("Selected", "整圈都清楚的物件")],
    [P("MinContrast", "最弱的地方也至少要差這麼多", "10")],
    [],
    "留下整圈都清楚的物件",
    "看的是最弱那一點，不是整圈平均。最弱的地方也要達到 MinContrast 才留下。先用 edge_floor 看那個數字。",
    "Vizion.select_closed (Region, Image, Selected, 10)",
)
def select_closed(region, image, min_contrast, *, ctx):
    source, facts = _spot_facts(region, image)
    limit = _spot_limit(min_contrast, "MinContrast")
    indexes = [index for index, item in enumerate(facts) if item["floor"] + 1e-9 >= limit]
    return _take_objects(source, indexes)


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [P("Selected", "邊緣有斷掉的物件")],
    [P("MaxContrast", "最弱的地方不超過這個值", "30")],
    [],
    "留下邊緣有弱點的物件",
    "最弱那一點的差小於等於 MaxContrast 就留下。0 表示有一點兩邊完全一樣。留下的是原來的整個物件，不是只有弱點。",
    "Vizion.select_broken (Region, Image, Selected, 30)",
)
def select_broken(region, image, max_contrast, *, ctx):
    source, facts = _spot_facts(region, image)
    limit = _spot_limit(max_contrast, "MaxContrast")
    indexes = [index for index, item in enumerate(facts) if item["floor"] <= limit + 1e-9]
    return _take_objects(source, indexes)


def _gray_objects(region, image):
    source = expect_region(region, "Region")
    picture = expect_image(image, "Image")
    gray = picture.gray().astype(np.float64)
    if gray.shape[0] != source.height or gray.shape[1] != source.width:
        raise ValueError("區域和影像的尺寸要相同")
    if source.count() == 0:
        raise ValueError("區域是空的")
    objects = []
    for index, mask in enumerate(source.objects):
        inside = mask > 0
        if not np.any(inside):
            raise ValueError(f"第 {index + 1} 個物件是空的")
        values = gray[inside]
        objects.append({"inside": inside, "area": int(np.count_nonzero(inside)), "low": float(values.min()), "high": float(values.max())})
    return source, gray, objects


def _band_limits(low, high) -> tuple[float, float]:
    start = scalar(low, "Low")
    stop = scalar(high, "High")
    if start > stop:
        start, stop = stop, start
    return start, stop


def _in_band(inside: np.ndarray, gray: np.ndarray, low: float, high: float) -> np.ndarray:
    return inside & (gray >= low) & (gray <= high)


def _extreme_point(inside: np.ndarray, gray: np.ndarray, bright: bool):
    score = np.where(inside, gray, -np.inf if bright else np.inf)
    flat = int(np.argmax(score) if bright else np.argmin(score))
    row, col = divmod(flat, gray.shape[1])
    return float(row), float(col), float(gray[row, col])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [],
    [P("Low", "灰階下限，含"), P("High", "灰階上限，含")],
    [P("Count", "落在這個範圍的像素數")],
    "物件裡有幾個像素落在這個灰階",
    "下限和上限含在內，寫反了會自動對調。每個物件各自數。0 表示沒有像素落在這個範圍。彩色先轉成亮度。Count[1] 是第一個物件。",
    "Vizion.band_count (Region, Image, 100, 255, Count)",
)
def band_count(region, image, low, high, *, ctx):
    _source, gray, objects = _gray_objects(region, image)
    start, stop = _band_limits(low, high)
    counts = [float(np.count_nonzero(_in_band(item["inside"], gray, start, stop))) for item in objects]
    return Result([], [counts])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [],
    [P("Low", "灰階下限，含"), P("High", "灰階上限，含")],
    [P("Ratio", "合格像素佔物件的比例")],
    "合格像素佔物件多少",
    "1 是整個物件都落在這個灰階。0 是完全沒有。分母是物件自己的像素，不是整張圖。",
    "Vizion.band_ratio (Region, Image, 100, 255, Ratio)",
)
def band_ratio(region, image, low, high, *, ctx):
    _source, gray, objects = _gray_objects(region, image)
    start, stop = _band_limits(low, high)
    ratios = []
    for item in objects:
        count = float(np.count_nonzero(_in_band(item["inside"], gray, start, stop)))
        ratios.append(count / float(item["area"]))
    return Result([], [ratios])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [P("Band", "落在範圍裡的像素")],
    [P("Low", "灰階下限，含"), P("High", "灰階上限，含")],
    [],
    "留下物件裡落在這個灰階的像素",
    "同一個物件裡合格的像素仍算一個物件，中間可以有洞。完全沒有合格像素的物件不會留下。要看個數，用 band_count，0 也留在 tuple 裡。",
    "Vizion.band_region (Region, Image, Band, 100, 255)",
)
def band_region(region, image, low, high, *, ctx):
    source, gray, objects = _gray_objects(region, image)
    start, stop = _band_limits(low, high)
    kept = []
    for item in objects:
        piece = _in_band(item["inside"], gray, start, stop)
        if np.any(piece):
            mask = np.zeros((source.height, source.width), np.uint8)
            mask[piece] = 255
            kept.append(mask)
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [P("Selected", "合格比例夠高的物件")],
    [
        P("Low", "灰階下限，含"),
        P("High", "灰階上限，含"),
        P("MinRatio", "至少要佔物件的這個比例", "0.5"),
    ],
    [],
    "留下夠亮比例夠高的物件",
    "先用 band_ratio 看比例。1 是整個物件都要落在範圍裡，0.8 是至少 80%。留下的是原來的整個物件，不是只有合格的像素。",
    "Vizion.select_band (Region, Image, Selected, 100, 255, 0.8)",
)
def select_band(region, image, low, high, min_ratio, *, ctx):
    source, gray, objects = _gray_objects(region, image)
    start, stop = _band_limits(low, high)
    limit = scalar(min_ratio, "MinRatio")
    if limit < 0:
        raise ValueError("MinRatio 不能是負的")
    indexes = []
    for index, item in enumerate(objects):
        count = float(np.count_nonzero(_in_band(item["inside"], gray, start, stop)))
        if count / float(item["area"]) + 1e-9 >= limit:
            indexes.append(index)
    return _take_objects(source, indexes)


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [],
    [],
    [P("Row", "最亮點的列"), P("Column", "最亮點的行"), P("Gray", "這一點的灰階")],
    "物件裡最亮的那一點",
    "一樣亮時留列最小、再來行最小的。每個物件一點。Gray[1] 是第一個物件的灰階。",
    "Vizion.bright_point (Region, Image, Row, Column, Gray)",
)
def bright_point(region, image, *, ctx):
    _source, gray, objects = _gray_objects(region, image)
    rows, cols, values = [], [], []
    for item in objects:
        row, col, value = _extreme_point(item["inside"], gray, True)
        rows.append(row)
        cols.append(col)
        values.append(value)
    return Result([], [rows, cols, values])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [],
    [],
    [P("Row", "最暗點的列"), P("Column", "最暗點的行"), P("Gray", "這一點的灰階")],
    "物件裡最暗的那一點",
    "一樣暗時留列最小、再來行最小的。整個物件同一個灰階時，最暗點和最亮點是同一點。",
    "Vizion.dark_point (Region, Image, Row, Column, Gray)",
)
def dark_point(region, image, *, ctx):
    _source, gray, objects = _gray_objects(region, image)
    rows, cols, values = [], [], []
    for item in objects:
        row, col, value = _extreme_point(item["inside"], gray, False)
        rows.append(row)
        cols.append(col)
        values.append(value)
    return Result([], [rows, cols, values])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [P("Peak", "最亮的那些像素")],
    [],
    [],
    "留下物件裡最亮的像素",
    "只留等於這個物件最大灰階的像素。整個物件一樣亮時，就是整個物件。一樣亮的點全部留下，不只是第一點。",
    "Vizion.peak_region (Region, Image, Peak)",
)
def peak_region(region, image, *, ctx):
    source, gray, objects = _gray_objects(region, image)
    kept = []
    for item in objects:
        piece = item["inside"] & (gray == item["high"])
        mask = np.zeros((source.height, source.width), np.uint8)
        mask[piece] = 255
        kept.append(mask)
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [P("Selected", "灰階幾乎一樣的物件")],
    [P("MaxSpan", "最亮和最暗最多差這麼多", "0")],
    [],
    "留下灰階很均勻的物件",
    "0 是整個物件同一個灰階。10 表示最亮和最暗最多差 10。min_max_gray 是把全部物件合在一起，這裡是每個物件各自看。",
    "Vizion.select_uniform (Region, Image, Selected, 0)",
)
def select_uniform(region, image, max_span, *, ctx):
    source, _gray, objects = _gray_objects(region, image)
    limit = scalar(max_span, "MaxSpan")
    if limit < 0:
        raise ValueError("MaxSpan 不能是負的")
    indexes = [index for index, item in enumerate(objects) if item["high"] - item["low"] <= limit + 1e-9]
    return _take_objects(source, indexes)


def _gray_modes(inside: np.ndarray, gray: np.ndarray):
    hist = np.bincount(gray[inside].astype(np.int32), minlength=256)
    mode = int(np.argmax(hist))
    count = int(hist[mode])
    hist[mode] = 0
    second = int(np.argmax(hist))
    second_count = int(hist[second])
    if second_count == 0:
        second = mode
    return mode, count, second, second_count


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [],
    [],
    [P("Gray", "出現最多次的灰階")],
    "物件主要是哪個灰階",
    "數的是出現最多次的灰階，不是最亮的那一點。最亮用 bright_point。兩種灰階一樣多時，留較暗的那個。Gray[1] 是第一個物件。彩色先轉成亮度。",
    "Vizion.gray_mode (Region, Image, Gray)",
)
def gray_mode(region, image, *, ctx):
    _source, gray, objects = _gray_objects(region, image)
    values = [float(_gray_modes(item["inside"], gray)[0]) for item in objects]
    return Result([], [values])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [],
    [],
    [P("Count", "主要灰階有幾個像素")],
    "主要灰階佔了幾個像素",
    "和 gray_mode 是同一種灰階。一個結果也是 tuple。",
    "Vizion.mode_count (Region, Image, Count)",
)
def mode_count(region, image, *, ctx):
    _source, gray, objects = _gray_objects(region, image)
    values = [float(_gray_modes(item["inside"], gray)[1]) for item in objects]
    return Result([], [values])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [],
    [],
    [P("Ratio", "主要灰階佔物件的比例")],
    "主要灰階佔物件多少",
    "1 是整個物件同一個灰階。分母是物件自己的像素。",
    "Vizion.mode_ratio (Region, Image, Ratio)",
)
def mode_ratio(region, image, *, ctx):
    _source, gray, objects = _gray_objects(region, image)
    ratios = []
    for item in objects:
        _mode, count, _second, _second_count = _gray_modes(item["inside"], gray)
        ratios.append(float(count) / float(item["area"]))
    return Result([], [ratios])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [],
    [],
    [P("Gray", "個數次多的灰階"), P("Count", "這一種有幾個像素")],
    "第二多的是哪個灰階",
    "比的是個數，不是灰階的大小。一樣多時留較暗的。只有一種灰階時，個數是 0，灰階仍回傳那一個，才不會和黑色搞混。",
    "Vizion.second_mode (Region, Image, Gray, Count)",
)
def second_mode(region, image, *, ctx):
    _source, gray, objects = _gray_objects(region, image)
    values, counts = [], []
    for item in objects:
        _mode, _count, second, second_count = _gray_modes(item["inside"], gray)
        values.append(float(second))
        counts.append(float(second_count))
    return Result([], [values, counts])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [],
    [],
    [P("Gap", "第二種灰階減主要灰階")],
    "兩種主要灰階差多少",
    "正的表示第二種比較亮。只有一種灰階時是 0。",
    "Vizion.mode_gap (Region, Image, Gap)",
)
def mode_gap(region, image, *, ctx):
    _source, gray, objects = _gray_objects(region, image)
    gaps = []
    for item in objects:
        mode, _count, second, second_count = _gray_modes(item["inside"], gray)
        gaps.append(0.0 if second_count == 0 else float(second - mode))
    return Result([], [gaps])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [P("Main", "主要灰階的像素")],
    [],
    [],
    "留下主要灰階的像素",
    "等於 gray_mode 的那些像素。兩種一樣多時，留較暗的那一種。整個物件同一個灰階時，就是整個物件。",
    "Vizion.mode_region (Region, Image, Main)",
)
def mode_region(region, image, *, ctx):
    source, gray, objects = _gray_objects(region, image)
    kept = []
    for item in objects:
        mode, _count, _second, _second_count = _gray_modes(item["inside"], gray)
        piece = item["inside"] & (gray == mode)
        mask = np.zeros((source.height, source.width), np.uint8)
        mask[piece] = 255
        kept.append(mask)
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [P("Selected", "主要灰階落在範圍裡的物件")],
    [P("Low", "灰階下限，含"), P("High", "灰階上限，含")],
    [],
    "留下主要灰階在這個範圍的物件",
    "看的是出現最多次的灰階，不是平均，也不是最亮。下限和上限含在內，寫反了會自動對調。留下的是原來的整個物件。",
    "Vizion.select_mode (Region, Image, Selected, 150, 255)",
)
def select_mode(region, image, low, high, *, ctx):
    source, gray, objects = _gray_objects(region, image)
    start, stop = _band_limits(low, high)
    indexes = []
    for index, item in enumerate(objects):
        mode = _gray_modes(item["inside"], gray)[0]
        if start <= mode <= stop:
            indexes.append(index)
    return _take_objects(source, indexes)


@operator(
    "特徵",
    [P("Region", "物件"), P("Image", "灰階影像")],
    [P("Selected", "有兩種主要灰階的物件")],
    [P("MinCount", "第二種至少要有幾個像素", "1")],
    [],
    "留下灰階混在一起的物件",
    "第二多的灰階至少要有 MinCount 個像素。1 表示只要有第二種就算。0 會全部留下。整個物件同一個灰階時不會留下。",
    "Vizion.select_mixed (Region, Image, Selected, 1)",
)
def select_mixed(region, image, min_count, *, ctx):
    source, gray, objects = _gray_objects(region, image)
    limit = scalar(min_count, "MinCount")
    if limit < 0:
        raise ValueError("MinCount 不能是負的")
    indexes = []
    for index, item in enumerate(objects):
        _mode, _count, _second, second_count = _gray_modes(item["inside"], gray)
        if second_count + 1e-9 >= limit:
            indexes.append(index)
    return _take_objects(source, indexes)


_STRIPE = ["'row'", "'column'"]


def _stripe_axis(direction) -> str:
    kind = text(direction, "Direction").lower()
    if kind not in ("row", "column"):
        raise ValueError("Direction 只能是 row 或 column")
    return kind


def _occupied_runs(mask: np.ndarray, direction: str):
    occupied = np.any(mask > 0, axis=1 if direction == "row" else 0)
    padded = np.concatenate([np.array([False]), occupied, np.array([False])])
    change = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(change == 1)
    ends = np.flatnonzero(change == -1)
    return starts, ends - starts


def _stripe_facts(region):
    source = expect_region(region, "Region")
    if source.count() == 0:
        raise ValueError("區域是空的")
    facts = []
    for index, mask in enumerate(source.objects):
        if not int(cv2.countNonZero(mask)):
            raise ValueError(f"第 {index + 1} 個物件是空的")
        row_starts, row_lengths = _occupied_runs(mask, "row")
        col_starts, col_lengths = _occupied_runs(mask, "column")
        facts.append({"mask": mask, "row_starts": row_starts, "row_lengths": row_lengths, "col_starts": col_starts, "col_lengths": col_lengths})
    return source, facts


def _longest(starts: np.ndarray, lengths: np.ndarray):
    best = int(np.argmax(lengths))
    return int(starts[best]), int(starts[best] + lengths[best]), int(lengths[best])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [],
    [P("Count", "有幾條橫的")],
    "有幾條橫向的條帶",
    "同一列上只要有像素就算，左右分開仍是同一條。中間空一列才算斷開。實心方塊是 1。Count[1] 是第一個物件。",
    "Vizion.row_runs (Region, Count)",
)
def row_runs(region, *, ctx):
    _source, facts = _stripe_facts(region)
    return Result([], [[float(len(item["row_lengths"])) for item in facts]])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [],
    [P("Count", "有幾條直的")],
    "有幾條直向的條帶",
    "同一行上只要有像素就算，上下分開仍是同一條。中間空一行才算斷開。實心方塊是 1。",
    "Vizion.col_runs (Region, Count)",
)
def col_runs(region, *, ctx):
    _source, facts = _stripe_facts(region)
    return Result([], [[float(len(item["col_lengths"])) for item in facts]])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [],
    [P("Length", "最長的橫條有幾列")],
    "最長的橫條有多高",
    "單位是列數。一樣高時，這個數字不分辨是哪一條。要取出那一條，用 stripe_region。",
    "Vizion.longest_rows (Region, Length)",
)
def longest_rows(region, *, ctx):
    _source, facts = _stripe_facts(region)
    return Result([], [[float(int(item["row_lengths"].max())) for item in facts]])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [],
    [P("Length", "最長的直條有幾行")],
    "最長的直條有多寬",
    "單位是行數。一樣寬時，這個數字不分辨是哪一條。",
    "Vizion.longest_cols (Region, Length)",
)
def longest_cols(region, *, ctx):
    _source, facts = _stripe_facts(region)
    return Result([], [[float(int(item["col_lengths"].max())) for item in facts]])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [],
    [P("Count", "橫條之間有幾個空隙")],
    "橫條中間空了幾次",
    "是條數減 1，不是空隙的列數。實心方塊是 0。上下貼著影像邊緣也不算空隙。",
    "Vizion.row_gaps (Region, Count)",
)
def row_gaps(region, *, ctx):
    _source, facts = _stripe_facts(region)
    return Result([], [[float(max(0, len(item["row_lengths"]) - 1)) for item in facts]])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [],
    [P("Count", "直條之間有幾個空隙")],
    "直條中間空了幾次",
    "是條數減 1，不是空隙的行數。實心方塊是 0。",
    "Vizion.col_gaps (Region, Count)",
)
def col_gaps(region, *, ctx):
    _source, facts = _stripe_facts(region)
    return Result([], [[float(max(0, len(item["col_lengths"]) - 1)) for item in facts]])


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Selected", "正好有這麼多條的物件")],
    [
        P("Direction", "row 橫條，column 直條", None, _STRIPE),
        P("Count", "正好要有幾條"),
    ],
    [],
    "留下正好有幾條的物件",
    "row 看橫條，column 看直條。要比至少幾條，先用 row_runs 或 col_runs 自己比。Count 至少要 1。",
    "Vizion.select_stripes (Region, Selected, 'row', 3)",
)
def select_stripes(region, direction, count, *, ctx):
    source, facts = _stripe_facts(region)
    kind = _stripe_axis(direction)
    need = whole(count, "Count")
    if need < 1:
        raise ValueError("Count 至少要 1")
    indexes = []
    for index, item in enumerate(facts):
        found = len(item["row_lengths"] if kind == "row" else item["col_lengths"])
        if found == need:
            indexes.append(index)
    return _take_objects(source, indexes)


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Stripe", "最長的那一條")],
    [P("Direction", "row 橫條，column 直條", "'row'", _STRIPE)],
    [],
    "留下最長的那一條",
    "row 留下最高的橫條，column 留下最寬的直條。一樣長時留上面或左邊的那一條。留下的是原來的像素，不會把中間的空白補上。",
    "Vizion.stripe_region (Region, Stripe, 'row')",
)
def stripe_region(region, direction, *, ctx):
    source, facts = _stripe_facts(region)
    kind = _stripe_axis(direction)
    kept = []
    for item in facts:
        starts = item["row_starts"] if kind == "row" else item["col_starts"]
        lengths = item["row_lengths"] if kind == "row" else item["col_lengths"]
        start, stop, _length = _longest(starts, lengths)
        piece = np.zeros_like(item["mask"])
        if kind == "row":
            piece[start:stop] = item["mask"][start:stop]
        else:
            piece[:, start:stop] = item["mask"][:, start:stop]
        if int(cv2.countNonZero(piece)):
            kept.append(piece)
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


_MIRROR = ["'column'", "'row'", "'both'"]


def _mirror_axis(axis) -> str:
    kind = text(axis, "Axis").lower()
    if kind not in ("column", "row", "both"):
        raise ValueError("Axis 只能是 column、row 或 both")
    return kind


def _mirror_parts(mask: np.ndarray, axis: str):
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return None
    row1, row2 = int(rows.min()), int(rows.max())
    col1, col2 = int(cols.min()), int(cols.max())
    mirror_row = (row1 + row2) - rows if axis in ("row", "both") else rows
    mirror_col = (col1 + col2) - cols if axis in ("column", "both") else cols
    height, width = mask.shape[:2]
    inside = (mirror_row >= 0) & (mirror_row < height) & (mirror_col >= 0) & (mirror_col < width)
    partner = np.zeros(len(rows), dtype=bool)
    partner[inside] = mask[mirror_row[inside], mirror_col[inside]] > 0
    missing = np.zeros_like(mask)
    missing[rows[~partner], cols[~partner]] = 255
    core = np.zeros_like(mask)
    core[rows[partner], cols[partner]] = 255
    reflected = np.zeros_like(mask)
    reflected[mirror_row[inside], mirror_col[inside]] = 255
    return {
        "score": float(partner.sum()) / float(len(rows)),
        "miss": float((~partner).sum()),
        "missing": missing,
        "core": core,
        "reflected": reflected,
        "row": (row1 + row2) / 2.0,
        "col": (col1 + col2) / 2.0,
    }


def _mirror_facts(region, axis: str):
    source = expect_region(region, "Region")
    if source.count() == 0:
        raise ValueError("區域是空的")
    facts = []
    for index, mask in enumerate(source.objects):
        item = _mirror_parts(mask, axis)
        if item is None:
            raise ValueError(f"第 {index + 1} 個物件是空的")
        facts.append(item)
    return source, facts


def _score_limit(value, name) -> float:
    limit = scalar(value, name)
    if limit < 0:
        raise ValueError(f"{name} 不能是負的")
    return limit


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [P("Axis", "column 左右，row 上下，both 轉 180 度", "'column'", _MIRROR)],
    [P("Score", "1 是完全對稱")],
    "對不對稱",
    "以物件外框的中心翻面，看有多少像素翻過去還落在自己身上。1 是完全對稱。column 左右對調，row 上下對調，both 轉 180 度。中心用外框，不是像素平均，所以缺一塊但外框還在時，中心不會跑掉。Score[1] 是第一個物件。",
    "Vizion.symmetry_score (Region, 'column', Score)",
)
def symmetry_score(region, axis, *, ctx):
    _source, facts = _mirror_facts(region, _mirror_axis(axis))
    return Result([], [[item["score"] for item in facts]])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [P("Axis", "column 左右，row 上下，both 轉 180 度", "'column'", _MIRROR)],
    [P("Count", "翻過去對不到的像素")],
    "有幾個像素對不上",
    "這些像素的對面沒有東西。0 是完全對稱。一個結果也是 tuple。",
    "Vizion.symmetry_miss (Region, 'column', Count)",
)
def symmetry_miss(region, axis, *, ctx):
    _source, facts = _mirror_facts(region, _mirror_axis(axis))
    return Result([], [[item["miss"] for item in facts]])


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Asymmetry", "對不上的像素")],
    [P("Axis", "column 左右，row 上下，both 轉 180 度", "'column'", _MIRROR)],
    [],
    "留下對不上的像素",
    "完全對稱的物件不會留下。要看有幾個對不上，用 symmetry_miss，0 也留在 tuple 裡。",
    "Vizion.asymmetry_region (Region, Asymmetry, 'column')",
)
def asymmetry_region(region, axis, *, ctx):
    source, facts = _mirror_facts(region, _mirror_axis(axis))
    kept = [item["missing"] for item in facts if int(cv2.countNonZero(item["missing"]))]
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Core", "翻過去還在的像素")],
    [P("Axis", "column 左右，row 上下，both 轉 180 度", "'column'", _MIRROR)],
    [],
    "留下翻過去還對得上的像素",
    "完全對稱時就是整個物件。和 asymmetry_region 加起來是原來的物件。",
    "Vizion.symmetry_core (Region, Core, 'column')",
)
def symmetry_core(region, axis, *, ctx):
    source, facts = _mirror_facts(region, _mirror_axis(axis))
    kept = [item["core"] for item in facts if int(cv2.countNonZero(item["core"]))]
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Reflected", "以自己的中心翻面")],
    [P("Axis", "column 左右，row 上下，both 轉 180 度", "'column'", _MIRROR)],
    [],
    "以物件自己的中心翻面",
    "翻的是這個物件，不是整張圖。整張圖翻面用 mirror_region。翻出影像的部分會被切掉。",
    "Vizion.reflect_region (Region, Reflected, 'column')",
)
def reflect_region(region, axis, *, ctx):
    source, facts = _mirror_facts(region, _mirror_axis(axis))
    kept = [item["reflected"] for item in facts if int(cv2.countNonZero(item["reflected"]))]
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [],
    [P("Row", "外框中心列"), P("Column", "外框中心行")],
    "對稱用的中心在哪",
    "是外框兩邊的平均，不是像素平均。缺一角但外框的角還在時，這個中心不會移動。",
    "Vizion.symmetry_center (Region, Row, Column)",
)
def symmetry_center(region, *, ctx):
    _source, facts = _mirror_facts(region, "both")
    return Result([], [[item["row"] for item in facts], [item["col"] for item in facts]])


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Selected", "夠對稱的物件")],
    [
        P("Axis", "column 左右，row 上下，both 轉 180 度", "'column'", _MIRROR),
        P("MinScore", "至少要有多對稱", "0.95"),
    ],
    [],
    "留下夠對稱的物件",
    "1 是完全對稱。0.95 表示至少有 95% 的像素翻過去還在。先用 symmetry_score 看分數。留下的是原來的物件。",
    "Vizion.select_symmetric (Region, Selected, 'column', 0.95)",
)
def select_symmetric(region, axis, min_score, *, ctx):
    source, facts = _mirror_facts(region, _mirror_axis(axis))
    limit = _score_limit(min_score, "MinScore")
    indexes = [index for index, item in enumerate(facts) if item["score"] + 1e-9 >= limit]
    return _take_objects(source, indexes)


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Selected", "不夠對稱的物件")],
    [
        P("Axis", "column 左右，row 上下，both 轉 180 度", "'column'", _MIRROR),
        P("MaxScore", "對稱分數不能超過", "0.95"),
    ],
    [],
    "留下不夠對稱的物件",
    "分數小於等於 MaxScore 才留下。0.95 表示還不到 95% 就算不對稱。留下的是原來的物件。",
    "Vizion.select_asymmetric (Region, Selected, 'column', 0.95)",
)
def select_asymmetric(region, axis, max_score, *, ctx):
    source, facts = _mirror_facts(region, _mirror_axis(axis))
    limit = _score_limit(max_score, "MaxScore")
    indexes = [index for index, item in enumerate(facts) if item["score"] <= limit + 1e-9]
    return _take_objects(source, indexes)


def _end_facts(region):
    source = expect_region(region, "Region")
    if source.count() == 0:
        raise ValueError("區域是空的")
    facts = []
    for index, mask in enumerate(source.objects):
        rows, cols = np.nonzero(mask)
        if len(rows) == 0:
            raise ValueError(f"第 {index + 1} 個物件是空的")
        top = int(rows.min())
        bottom = int(rows.max())
        left = int(cols.min())
        right = int(cols.max())
        facts.append(
            {
                "mask": mask,
                "top": (float(top), float(cols[rows == top].min())),
                "bottom": (float(bottom), float(cols[rows == bottom].min())),
                "left": (float(rows[cols == left].min()), float(left)),
                "right": (float(rows[cols == right].min()), float(right)),
            }
        )
    return source, facts


def _end_line(mask: np.ndarray, kind: str) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    out = np.zeros_like(mask)
    if kind == "top":
        line = int(rows.min())
        out[line, cols[rows == line]] = 255
    elif kind == "bottom":
        line = int(rows.max())
        out[line, cols[rows == line]] = 255
    elif kind == "left":
        line = int(cols.min())
        out[rows[cols == line], line] = 255
    else:
        line = int(cols.max())
        out[rows[cols == line], line] = 255
    return out


def _point_result(region, kind: str):
    _source, facts = _end_facts(region)
    return Result([], [[item[kind][0] for item in facts], [item[kind][1] for item in facts]])


def _edge_result(region, kind: str):
    source, facts = _end_facts(region)
    kept = [_end_line(item["mask"], kind) for item in facts]
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [],
    [P("Row", "最上點的列"), P("Column", "最上點的行")],
    "最上的那一點",
    "列最小的像素。同一列有很多點時，留最左邊的。外框的左上角不一定有像素，這裡給的是真的有像素的點。Row[1] 是第一個物件。",
    "Vizion.top_point (Region, Row, Column)",
)
def top_point(region, *, ctx):
    return _point_result(region, "top")


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [],
    [P("Row", "最下點的列"), P("Column", "最下點的行")],
    "最下的那一點",
    "列最大的像素。同一列有很多點時，留最左邊的。",
    "Vizion.bottom_point (Region, Row, Column)",
)
def bottom_point(region, *, ctx):
    return _point_result(region, "bottom")


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [],
    [P("Row", "最左點的列"), P("Column", "最左點的行")],
    "最左的那一點",
    "行最小的像素。同一行有很多點時，留最上面的。",
    "Vizion.left_point (Region, Row, Column)",
)
def left_point(region, *, ctx):
    return _point_result(region, "left")


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [],
    [P("Row", "最右點的列"), P("Column", "最右點的行")],
    "最右的那一點",
    "行最大的像素。同一行有很多點時，留最上面的。",
    "Vizion.right_point (Region, Row, Column)",
)
def right_point(region, *, ctx):
    return _point_result(region, "right")


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Edge", "最上的那一列")],
    [],
    [],
    "留下最上的一整列",
    "不是只有 top_point 那一點。這一列上有的像素全部留下。要點用 top_point，要整條邊用這個。",
    "Vizion.top_edge (Region, Edge)",
)
def top_edge(region, *, ctx):
    return _edge_result(region, "top")


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Edge", "最下的那一列")],
    [],
    [],
    "留下最下的一整列",
    "這一列上有的像素全部留下。同一列最左邊的點是 bottom_point。",
    "Vizion.bottom_edge (Region, Edge)",
)
def bottom_edge(region, *, ctx):
    return _edge_result(region, "bottom")


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Edge", "最左的那一行")],
    [],
    [],
    "留下最左的一整行",
    "這一行上有的像素全部留下。最上面的點是 left_point。",
    "Vizion.left_edge (Region, Edge)",
)
def left_edge(region, *, ctx):
    return _edge_result(region, "left")


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Edge", "最右的那一行")],
    [],
    [],
    "留下最右的一整行",
    "這一行上有的像素全部留下。最上面的點是 right_point。",
    "Vizion.right_edge (Region, Edge)",
)
def right_edge(region, *, ctx):
    return _edge_result(region, "right")


def _enclosed_holes(mask: np.ndarray) -> list:
    if int(cv2.countNonZero(mask)) == 0:
        return []
    padded = np.pad(np.where(mask > 0, 255, 0).astype(np.uint8), 1)
    flood = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, flood, (0, 0), 128)
    holes = np.where(padded[1:-1, 1:-1] == 0, 255, 0).astype(np.uint8)
    count, labels = cv2.connectedComponents(holes, connectivity=8)
    pieces = []
    for label in range(1, int(count)):
        pieces.append(np.where(labels == label, 255, 0).astype(np.uint8))
    return pieces


def _hole_parts(region):
    source = expect_region(region, "Region")
    if source.count() == 0:
        raise ValueError("區域是空的")
    parts = []
    for index, mask in enumerate(source.objects):
        if int(cv2.countNonZero(mask)) == 0:
            raise ValueError(f"第 {index + 1} 個物件是空的")
        holes = _enclosed_holes(mask)
        hole_area = int(sum(cv2.countNonZero(piece) for piece in holes))
        solid = int(cv2.countNonZero(mask))
        parts.append(
            {
                "mask": mask,
                "holes": holes,
                "ratio": 0.0 if solid + hole_area == 0 else hole_area / (solid + hole_area),
            }
        )
    return source, parts


def _biggest_hole(pieces: list):
    best = None
    best_area = -1
    for piece in pieces:
        area = int(cv2.countNonZero(piece))
        if area > best_area:
            best = piece
            best_area = area
    return best


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [],
    [P("Count", "每個物件有幾個洞")],
    "數每個物件有幾個洞",
    "洞是被這個物件完全包住的空白。缺口連到外面，或連到影像邊緣，都不算。沒有洞是 0，0 也留在 tuple 裡。斜角碰到的兩塊空白算同一個洞。Count[1] 是第一個物件。",
    "Vizion.hole_count (Region, Count)",
)
def hole_count(region, *, ctx):
    _source, parts = _hole_parts(region)
    return Result([], [[len(item["holes"]) for item in parts]])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [],
    [P("Ratio", "洞占補滿後的比例")],
    "洞占整個物件多少",
    "洞的像素除以補滿之後的像素。0 是實心。數字愈大，洞愈大。缺口連到外面不算在裡面。Ratio[1] 是第一個物件。",
    "Vizion.hole_ratio (Region, Ratio)",
)
def hole_ratio(region, *, ctx):
    _source, parts = _hole_parts(region)
    return Result([], [[item["ratio"] for item in parts]])


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Holes", "一個洞一個物件")],
    [],
    [],
    "把每個洞拆開",
    "每個洞自己一個物件。順序是原來的物件順序，同一個物件裡較上面、再來較左邊的洞先出來。沒有洞的物件不會留下。要面積和中心再呼叫 area_center。",
    "Vizion.hole_region (Region, Holes)",
)
def hole_region(region, *, ctx):
    source, parts = _hole_parts(region)
    kept = [piece for item in parts for piece in item["holes"]]
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Hole", "每個物件最大的洞")],
    [],
    [],
    "留下每個物件最大的洞",
    "一個物件留一個洞。一樣大時留較上面、再來較左邊的。沒有洞的物件不會留下。",
    "Vizion.largest_hole (Region, Hole)",
)
def largest_hole(region, *, ctx):
    source, parts = _hole_parts(region)
    kept = [_biggest_hole(item["holes"]) for item in parts if item["holes"]]
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Filled", "補好洞的物件")],
    [],
    [],
    "每個物件自己補洞",
    "只補被完全包住的洞，物件數不變。缺口連到外面不會被補上。fill_up 會先把全部物件合成一個再補，這個不會。",
    "Vizion.fill_object (Region, Filled)",
)
def fill_object(region, *, ctx):
    source, parts = _hole_parts(region)
    kept = []
    for item in parts:
        filled = item["mask"].copy()
        for piece in item["holes"]:
            filled = cv2.bitwise_or(filled, piece)
        kept.append(filled)
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Selected", "有夠多洞的物件")],
    [P("MinCount", "至少幾個洞", "1")],
    [],
    "留下洞夠多的物件",
    "洞的個數達到 MinCount 才留下。1 表示至少要有一個洞。留下的是原來的物件，不是洞本身。",
    "Vizion.select_holed (Region, Selected, 1)",
)
def select_holed(region, min_count, *, ctx):
    number = scalar(min_count, "MinCount")
    if number < 0:
        raise ValueError("MinCount 不能是負的")
    need = int(round(number))
    source, parts = _hole_parts(region)
    indexes = [index for index, item in enumerate(parts) if len(item["holes"]) >= need]
    return _take_objects(source, indexes)


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Selected", "沒有洞的物件")],
    [],
    [],
    "留下沒有洞的物件",
    "一個洞都沒有才留下，包含缺口連到外面的物件。留下的是原來的物件。",
    "Vizion.select_solid (Region, Selected)",
)
def select_solid(region, *, ctx):
    source, parts = _hole_parts(region)
    indexes = [index for index, item in enumerate(parts) if not item["holes"]]
    return _take_objects(source, indexes)


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Selected", "夠空的物件")],
    [P("MinRatio", "洞至少占多少", "0.1")],
    [],
    "留下洞占得夠多的物件",
    "hole_ratio 達到 MinRatio 才留下。0.1 表示洞至少占補滿後的一成。留下的是原來的物件。",
    "Vizion.select_hole_ratio (Region, Selected, 0.1)",
)
def select_hole_ratio(region, min_ratio, *, ctx):
    limit = scalar(min_ratio, "MinRatio")
    if limit < 0:
        raise ValueError("MinRatio 不能是負的")
    source, parts = _hole_parts(region)
    indexes = [index for index, item in enumerate(parts) if item["ratio"] + 1e-9 >= limit]
    return _take_objects(source, indexes)


_WINDOW_METRIC = ["'chessboard'", "'city-block'"]


def _window_box(row1, column1, row2, column2):
    r1 = whole(row1, "Row1")
    c1 = whole(column1, "Column1")
    r2 = whole(row2, "Row2")
    c2 = whole(column2, "Column2")
    if r2 < r1 or c2 < c1:
        raise ValueError("矩形座標不正確")
    return r1, c1, r2, c2


def _window_metric(metric) -> str:
    name = text(metric, "Metric").lower()
    if name not in ("chessboard", "city-block"):
        raise ValueError("Metric 只能是 chessboard 或 city-block")
    return name


def _window_facts(region, row1, column1, row2, column2, metric: str):
    source = expect_region(region, "Region")
    if source.count() == 0:
        raise ValueError("區域是空的")
    r1, c1, r2, c2 = _window_box(row1, column1, row2, column2)
    facts = []
    for index, mask in enumerate(source.objects):
        rows, cols = np.nonzero(mask)
        if len(rows) == 0:
            raise ValueError(f"第 {index + 1} 個物件是空的")
        inside = (rows >= r1) & (rows <= r2) & (cols >= c1) & (cols <= c2)
        in_count = int(inside.sum())
        out_count = int(len(rows) - in_count)
        outside = np.zeros_like(mask)
        depth = 0
        if out_count:
            orows = rows[~inside]
            ocols = cols[~inside]
            outside[orows, ocols] = 255
            dr = np.maximum(np.maximum(r1 - orows, 0), orows - r2)
            dc = np.maximum(np.maximum(c1 - ocols, 0), ocols - c2)
            depth = int((dr + dc).max() if metric == "city-block" else np.maximum(dr, dc).max())
        facts.append(
            {
                "inside": in_count,
                "outside": out_count,
                "ratio": in_count / float(len(rows)),
                "depth": depth,
                "mask": outside,
            }
        )
    return source, facts


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [P("Row1", "方框上"), P("Column1", "方框左"), P("Row2", "方框下"), P("Column2", "方框右")],
    [P("Area", "方框裡的像素")],
    "方框裡有多少像素",
    "方框的邊算在裡面。上要小於等於下，左要小於等於右。整個在外面時是 0，0 也留在 tuple 裡。Area[1] 是第一個物件。clip_region 是把外面裁掉，這個只回報有多少留在裡面。",
    "Vizion.window_inside (Region, 20, 20, 39, 39, Area)",
)
def window_inside(region, row1, column1, row2, column2, *, ctx):
    _source, facts = _window_facts(region, row1, column1, row2, column2, "chessboard")
    return Result([], [[item["inside"] for item in facts]])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [P("Row1", "方框上"), P("Column1", "方框左"), P("Row2", "方框下"), P("Column2", "方框右")],
    [P("Area", "方框外的像素")],
    "方框外有多少像素",
    "跑出方框的像素數。完全在裡面時是 0。方框的邊算在裡面，不算跑出去。",
    "Vizion.window_outside (Region, 20, 20, 39, 39, Area)",
)
def window_outside(region, row1, column1, row2, column2, *, ctx):
    _source, facts = _window_facts(region, row1, column1, row2, column2, "chessboard")
    return Result([], [[item["outside"] for item in facts]])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [P("Row1", "方框上"), P("Column1", "方框左"), P("Row2", "方框下"), P("Column2", "方框右")],
    [P("Ratio", "留在方框裡的比例")],
    "有多少比例留在方框裡",
    "方框裡的像素除以物件自己的像素。1 是整個在裡面，0 是整個在外面。Ratio[1] 是第一個物件。",
    "Vizion.window_ratio (Region, 20, 20, 39, 39, Ratio)",
)
def window_ratio(region, row1, column1, row2, column2, *, ctx):
    _source, facts = _window_facts(region, row1, column1, row2, column2, "chessboard")
    return Result([], [[item["ratio"] for item in facts]])


@operator(
    "區域",
    [P("Region", "物件")],
    [],
    [
        P("Row1", "方框上"),
        P("Column1", "方框左"),
        P("Row2", "方框下"),
        P("Column2", "方框右"),
        P("Metric", "距離算法", "'chessboard'", _WINDOW_METRIC),
    ],
    [P("Depth", "跑出方框多遠")],
    "跑出方框多遠",
    "跑出去的像素裡，離方框最遠的那段。完全在裡面是 0。chessboard 是列和行取較遠的那個，斜著出去不會把兩個相加。city-block 才是列加行。方框的邊算在裡面。",
    "Vizion.window_depth (Region, 20, 20, 39, 39, 'chessboard', Depth)",
)
def window_depth(region, row1, column1, row2, column2, metric, *, ctx):
    _source, facts = _window_facts(region, row1, column1, row2, column2, _window_metric(metric))
    return Result([], [[item["depth"] for item in facts]])


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Outside", "跑出方框的像素")],
    [P("Row1", "方框上"), P("Column1", "方框左"), P("Row2", "方框下"), P("Column2", "方框右")],
    [],
    "留下跑出方框的像素",
    "只留方框外面的像素。完全在裡面的物件不會留下。整個在外面的物件會整塊留下。clip_region 是留下裡面的部分。",
    "Vizion.overflow_region (Region, Outside, 20, 20, 39, 39)",
)
def overflow_region(region, row1, column1, row2, column2, *, ctx):
    source, facts = _window_facts(region, row1, column1, row2, column2, "chessboard")
    kept = [item["mask"] for item in facts if item["outside"]]
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Selected", "整個在方框裡的物件")],
    [P("Row1", "方框上"), P("Column1", "方框左"), P("Row2", "方框下"), P("Column2", "方框右")],
    [],
    "留下整個在方框裡的物件",
    "一個像素都沒有跑出去才留下。留下的是原來的物件。方框的邊算在裡面。",
    "Vizion.select_inside (Region, Selected, 20, 20, 39, 39)",
)
def select_inside(region, row1, column1, row2, column2, *, ctx):
    source, facts = _window_facts(region, row1, column1, row2, column2, "chessboard")
    indexes = [index for index, item in enumerate(facts) if item["outside"] == 0]
    return _take_objects(source, indexes)


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Selected", "整個在方框外的物件")],
    [P("Row1", "方框上"), P("Column1", "方框左"), P("Row2", "方框下"), P("Column2", "方框右")],
    [],
    "留下整個在方框外的物件",
    "一個像素都沒有在方框裡才留下。留下的是原來的物件。",
    "Vizion.select_outside (Region, Selected, 20, 20, 39, 39)",
)
def select_outside(region, row1, column1, row2, column2, *, ctx):
    source, facts = _window_facts(region, row1, column1, row2, column2, "chessboard")
    indexes = [index for index, item in enumerate(facts) if item["inside"] == 0]
    return _take_objects(source, indexes)


@operator(
    "區域",
    [P("Region", "物件")],
    [P("Selected", "跨在方框邊上的物件")],
    [P("Row1", "方框上"), P("Column1", "方框左"), P("Row2", "方框下"), P("Column2", "方框右")],
    [],
    "留下跨在方框邊上的物件",
    "一部分在裡面、一部分在外面才留下。留下的是原來的物件，不是裁切後的形狀。",
    "Vizion.select_crossing (Region, Selected, 20, 20, 39, 39)",
)
def select_crossing(region, row1, column1, row2, column2, *, ctx):
    source, facts = _window_facts(region, row1, column1, row2, column2, "chessboard")
    indexes = [index for index, item in enumerate(facts) if item["inside"] and item["outside"]]
    return _take_objects(source, indexes)


_AXIS = ["'horizontal'", "'vertical'", "'diagonal'"]


def _axis_facts(region):
    source = expect_region(region, "Region")
    if source.count() == 0:
        raise ValueError("區域是空的")
    facts = []
    for index, mask in enumerate(source.objects):
        pose = _pose_of(mask)
        if pose is None:
            raise ValueError(f"第 {index + 1} 個物件是空的")
        row, col, phi = pose
        rows, cols = np.nonzero(mask)
        cosine = math.cos(phi)
        sine = math.sin(phi)
        along = cols.astype(np.float64) * cosine + rows.astype(np.float64) * sine
        across = -cols.astype(np.float64) * sine + rows.astype(np.float64) * cosine
        center_across = -col * sine + row * cosine
        dist = np.abs(across - center_across)
        spine = np.zeros_like(mask)
        spine[rows[dist <= float(dist.min()) + 1e-6], cols[dist <= float(dist.min()) + 1e-6]] = 255
        ends = []
        for target in (float(along.min()), float(along.max())):
            penalty = np.where(np.abs(along - target) <= 1e-6, dist, np.inf)
            chosen = int(np.lexsort((cols, rows, penalty))[0])
            ends.append((int(rows[chosen]), int(cols[chosen])))
        ends.sort()
        major = float(along.max() - along.min() + 1.0)
        minor = float(across.max() - across.min() + 1.0)
        facts.append(
            {
                "phi": float(phi),
                "major": major,
                "minor": minor,
                "ratio": major / minor,
                "ends": ends,
                "spine": spine,
            }
        )
    return source, facts


def _axis_direction(direction) -> str:
    kind = text(direction, "Direction").lower()
    if kind not in ("horizontal", "vertical", "diagonal"):
        raise ValueError("Direction 只能是 horizontal、vertical 或 diagonal")
    return kind


def _axis_match(phi: float, kind: str, tolerance: float) -> bool:
    horizontal = abs(phi) <= tolerance + 1e-9
    vertical = math.pi / 2 - abs(phi) <= tolerance + 1e-9
    if kind == "horizontal":
        return horizontal
    if kind == "vertical":
        return vertical
    return not horizontal and not vertical


@operator(
    "特徵",
    [P("Region", "物件")],
    [],
    [],
    [P("Phi", "長軸角度，弳度")],
    "長軸朝哪",
    "用像素的二次矩，和 orientation_region、elliptic_axis 的角度相同。0 沿著行往右，正角度在畫面上順時針，直的是 1.5708。範圍在 -90 度到 90 度。接近正方形時角度會落在 0，不代表它是橫的，請看 major_ratio。Phi[1] 是第一個物件。",
    "Vizion.major_phi (Region, Phi)",
)
def major_phi(region, *, ctx):
    _source, facts = _axis_facts(region)
    return Result([], [[item["phi"] for item in facts]])


@operator(
    "特徵",
    [P("Region", "物件")],
    [],
    [],
    [P("Length", "沿著長軸的長度")],
    "沿著長軸有多長",
    "兩端像素中心的距離再加 1。橫向 20 個像素、厚 5 的長條，長度是 20。斜的物件會帶小數。",
    "Vizion.major_length (Region, Length)",
)
def major_length(region, *, ctx):
    _source, facts = _axis_facts(region)
    return Result([], [[item["major"] for item in facts]])


@operator(
    "特徵",
    [P("Region", "物件")],
    [],
    [],
    [P("Length", "沿著短軸的厚度")],
    "沿著短軸有多厚",
    "和 major_length 同一把尺，只是換成短軸。厚 5 的長條是 5。",
    "Vizion.minor_length (Region, Length)",
)
def minor_length(region, *, ctx):
    _source, facts = _axis_facts(region)
    return Result([], [[item["minor"] for item in facts]])


@operator(
    "特徵",
    [P("Region", "物件")],
    [],
    [],
    [P("Ratio", "長除以厚")],
    "長軸是短軸的幾倍",
    "major_length 除以 minor_length。正方形是 1。20×5 的長條是 4。接近 1 時長軸不可靠。",
    "Vizion.major_ratio (Region, Ratio)",
)
def major_ratio(region, *, ctx):
    _source, facts = _axis_facts(region)
    return Result([], [[item["ratio"] for item in facts]])


@operator(
    "特徵",
    [P("Region", "物件")],
    [],
    [],
    [
        P("Row1", "一端的列"),
        P("Column1", "一端的行"),
        P("Row2", "另一端的列"),
        P("Column2", "另一端的行"),
    ],
    "長軸的兩個端點",
    "端點落在物件上，平的那一頭留最靠近軸線的像素。先給較上面的端點，同一列再給較左邊的。Row1[1] 是第一個物件。",
    "Vizion.axis_ends (Region, Row1, Column1, Row2, Column2)",
)
def axis_ends(region, *, ctx):
    _source, facts = _axis_facts(region)
    return Result(
        [],
        [
            [item["ends"][0][0] for item in facts],
            [item["ends"][0][1] for item in facts],
            [item["ends"][1][0] for item in facts],
            [item["ends"][1][1] for item in facts],
        ],
    )


@operator(
    "特徵",
    [P("Region", "物件")],
    [P("Axis", "長軸上的像素")],
    [],
    [],
    "留下長軸上的那一層像素",
    "每個物件留離長軸最近的那一層。橫條是中間那一列，直條是中間那一行。正方形會留下中間的兩列。",
    "Vizion.axis_region (Region, Axis)",
)
def axis_region(region, *, ctx):
    source, facts = _axis_facts(region)
    kept = [item["spine"] for item in facts]
    return Result([Region(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "特徵",
    [P("Region", "物件")],
    [P("Selected", "方向符合的物件")],
    [
        P("Direction", "要留的方向", "'horizontal'", _AXIS),
        P("Tolerance", "允許差幾度", "15"),
    ],
    [],
    "依長軸方向留下物件",
    "horizontal 離 0 度在 Tolerance 以內，vertical 離 90 度在 Tolerance 以內，diagonal 是兩邊都超過。15 度是預設。接近正方形時角度會是 0，要先看 major_ratio。留下的是原來的物件。",
    "Vizion.select_axis (Region, Selected, 'horizontal', 15)",
)
def select_axis(region, direction, tolerance, *, ctx):
    kind = _axis_direction(direction)
    degrees = scalar(tolerance, "Tolerance")
    if degrees < 0:
        raise ValueError("Tolerance 不能是負的")
    limit = math.radians(degrees)
    source, facts = _axis_facts(region)
    indexes = [index for index, item in enumerate(facts) if _axis_match(item["phi"], kind, limit)]
    return _take_objects(source, indexes)


@operator(
    "特徵",
    [P("Region", "物件")],
    [P("Selected", "夠細長的物件")],
    [P("MinRatio", "長至少是厚的幾倍", "2")],
    [],
    "留下夠細長的物件",
    "major_ratio 達到 MinRatio 才留下。2 表示至少是兩倍長。正方形是 1，不會留下。留下的是原來的物件。",
    "Vizion.select_long (Region, Selected, 2)",
)
def select_long(region, min_ratio, *, ctx):
    limit = scalar(min_ratio, "MinRatio")
    if limit < 0:
        raise ValueError("MinRatio 不能是負的")
    source, facts = _axis_facts(region)
    indexes = [index for index, item in enumerate(facts) if item["ratio"] + 1e-9 >= limit]
    return _take_objects(source, indexes)
