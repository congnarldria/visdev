"""Vision operators with HDevelop call order.

Iconic inputs, iconic outputs, control inputs, control outputs.
Add a new one with @operator; the name of the function is the name used in programs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .objects import Image, Region, Result, StopSignal, XLD
from .render import normalize_color, read_cv_image, to_uint8, write_cv_image

REGISTRY: dict[str, "Operator"] = {}
_ORDER = 0


@dataclass
class Param:
    name: str
    desc: str
    default: str | None = None


@dataclass
class Operator:
    name: str
    group: str
    iconic_in: list[Param]
    iconic_out: list[Param]
    control_in: list[Param]
    control_out: list[Param]
    short: str
    detail: str
    example: str
    fn: object
    order: int = 0

    def prototype(self) -> str:
        def join(params: list[Param]) -> str:
            return ", ".join(item.name for item in params)

        return (
            f"{self.name}({join(self.iconic_in)} : {join(self.iconic_out)} : "
            f"{join(self.control_in)} : {join(self.control_out)})"
        )

    def to_dict(self) -> dict:
        def pack(params: list[Param], role: str):
            return [{"name": item.name, "role": role, "desc": item.desc} for item in params]

        return {
            "name": self.name,
            "group": self.group,
            "short": self.short,
            "detail": self.detail,
            "prototype": self.prototype(),
            "example": self.example,
            "order": self.order,
            "params": pack(self.iconic_in, "圖像輸入")
            + pack(self.iconic_out, "圖像輸出")
            + pack(self.control_in, "控制輸入")
            + pack(self.control_out, "控制輸出"),
        }


def P(name: str, desc: str, default: str | None = None) -> Param:
    return Param(name, desc, default)


def operator(group, iconic_in, iconic_out, control_in, control_out, short, detail, example):
    def deco(fn):
        global _ORDER
        _ORDER += 1
        op = Operator(
            fn.__name__,
            group,
            iconic_in,
            iconic_out,
            control_in,
            control_out,
            short,
            detail,
            example,
            fn,
            _ORDER,
        )
        REGISTRY[fn.__name__] = op
        return fn

    return deco


def expect_image(value, name: str) -> Image:
    if not isinstance(value, Image):
        raise TypeError(f"{name} 必須是影像")
    return value


def expect_region(value, name: str) -> Region:
    if not isinstance(value, Region):
        raise TypeError(f"{name} 必須是區域")
    return value


def scalar(value, name: str) -> float:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{name} 需要單一數值，得到 {len(value)} 個")
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 不是數值：{value}") from exc


def whole(value, name: str) -> int:
    return int(round(scalar(value, name)))


def text(value, name: str) -> str:
    if isinstance(value, (list, tuple)):
        raise ValueError(f"{name} 需要字串")
    return str(value)


def same_size(a: Region, b: Region, what: str) -> None:
    if a.width != b.width or a.height != b.height:
        raise ValueError(f"{what}：兩個區域的尺寸不同")


def image_out(src: Image, data: np.ndarray) -> Result:
    domain = None if src.domain is None else src.domain.copy()
    return Result([Image(data, domain)], [])


def region_out(width: int, height: int, mask: np.ndarray) -> Result:
    objects = [mask] if int(cv2.countNonZero(mask)) else []
    return Result([Region(width, height, objects)], [])


def disk(radius: float) -> np.ndarray:
    if radius < 0:
        raise ValueError("半徑不能是負的")
    size = int(round(radius * 2))
    if size < 1:
        size = 1
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def rect_kernel(width: float, height: float) -> np.ndarray:
    wide = int(round(width))
    high = int(round(height))
    if wide < 1 or high < 1:
        raise ValueError("矩形寬高至少要 1")
    if wide % 2 == 0:
        wide += 1
    if high % 2 == 0:
        high += 1
    return cv2.getStructuringElement(cv2.MORPH_RECT, (wide, high))


def morph(region: Region, kernel: np.ndarray, mode: str) -> Result:
    mask = region.union_mask()
    if mode == "open":
        out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif mode == "close":
        out = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    elif mode == "erode":
        out = cv2.erode(mask, kernel)
    elif mode == "dilate":
        out = cv2.dilate(mask, kernel)
    else:
        raise ValueError(mode)
    return region_out(region.width, region.height, out)


def gray_bins(image: Image) -> tuple[list[int], int]:
    gray = image.gray()
    values = gray[image.domain > 0] if image.domain is not None else gray.reshape(-1)
    if values.size == 0:
        return [0] * 256, 0
    counts = np.bincount(values.astype(np.uint8), minlength=256)[:256]
    return [int(item) for item in counts], int(values.size)


def attach_histogram(ctx, image: Image, lo: int, hi: int, mask: np.ndarray, kind: str, **extra) -> None:
    bins, total = gray_bins(image)
    ctx.threshold_gray = image.gray().copy()
    ctx.threshold_domain = None if image.domain is None else image.domain.copy()
    ctx.threshold_mask = mask.copy()
    ctx.histogram = {
        "bins": bins,
        "min": int(lo),
        "max": int(hi),
        "kind": kind,
        "area": int(cv2.countNonZero(mask)),
        "total": total,
        "literal": False,
        "line": None,
        **extra,
    }


def preview_mask(gray: np.ndarray, domain: np.ndarray | None, kind: str, lo: int, hi: int, level: int, mode: str) -> np.ndarray:
    name = str(kind or "range")
    if name == "opencv":
        flag = cv2.THRESH_BINARY_INV if str(mode).lower() == "binary_inv" else cv2.THRESH_BINARY
        _used, mask = cv2.threshold(gray, int(level), 255, flag)
    elif name == "subpix":
        mask = np.where(gray.astype(np.float32) >= float(level), 255, 0).astype(np.uint8)
    elif name == "otsu":
        _used, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if str(mode).lower() == "dark":
            mask = cv2.bitwise_not(mask)
    elif int(lo) > int(hi):
        mask = np.zeros_like(gray)
    else:
        mask = cv2.inRange(gray, int(lo), int(hi))
    if domain is not None:
        mask = cv2.bitwise_and(mask, domain)
    return mask


REGION_FEATURES = [
    "area", "row", "column", "width", "height", "ratio",
    "circularity", "compactness", "contlength", "rectangularity", "convexity",
    "ra", "rb", "phi", "anisometry", "bulkiness", "struct_factor",
    "outer_radius", "max_diameter", "orientation",
    "rect2_phi", "rect2_len1", "rect2_len2",
    "holes_num", "area_holes", "roundness", "num_sides", "euler_number",
]

GRAY_FEATURES = [
    "gray_min", "gray_max", "gray_mean", "gray_deviation", "gray_median",
]

XLD_FEATURES = [
    "area", "row", "column", "width", "height", "ratio",
    "circularity", "compactness", "contlength", "rectangularity", "convexity",
    "ra", "rb", "phi", "anisometry", "bulkiness", "struct_factor",
    "outer_radius", "max_diameter", "orientation",
    "rect2_phi", "rect2_len1", "rect2_len2",
]


def _find_contours(mask: np.ndarray, mode: int, method: int):
    found = cv2.findContours(mask, mode, method)
    contours = found[0] if len(found) == 2 else found[1]
    hierarchy = found[1] if len(found) == 2 else found[2]
    return contours, hierarchy


def _sample_gray(gray: np.ndarray, x: float, y: float) -> float:
    height, width = gray.shape
    if x < 0 or y < 0 or x > width - 1 or y > height - 1:
        x = min(max(x, 0.0), width - 1)
        y = min(max(y, 0.0), height - 1)
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    dx = x - x0
    dy = y - y0
    top = float(gray[y0, x0]) * (1 - dx) + float(gray[y0, x1]) * dx
    bottom = float(gray[y1, x0]) * (1 - dx) + float(gray[y1, x1]) * dx
    return top * (1 - dy) + bottom * dy


def _axes(contour) -> tuple[float, float, float, float, float, float]:
    if contour is None or len(contour) < 3:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    _center, (rw, rh), angle = cv2.minAreaRect(contour)
    rect_len1 = float(max(rw, rh) / 2)
    rect_len2 = float(min(rw, rh) / 2)
    rect_phi = float(math.radians(angle))
    ra = rb = phi = 0.0
    if len(contour) >= 5:
        try:
            (_c, (ew, eh), eangle) = cv2.fitEllipse(contour)
            ra = float(max(ew, eh) / 2)
            rb = float(min(ew, eh) / 2)
            phi = float(math.radians(eangle))
        except cv2.error:
            ra = rect_len1
            rb = rect_len2
            phi = rect_phi
    else:
        ra = rect_len1
        rb = rect_len2
        phi = rect_phi
    return ra, rb, phi, rect_len1, rect_len2, rect_phi


def region_measures(mask: np.ndarray) -> dict[str, float]:
    area = int(cv2.countNonZero(mask))
    moments = cv2.moments(mask)
    if moments["m00"]:
        row = float(moments["m01"] / moments["m00"])
        col = float(moments["m10"] / moments["m00"])
    else:
        row = col = 0.0
    if area:
        _x, _y, width, height = cv2.boundingRect(mask)
    else:
        width = height = 0
    contours, _hier = _find_contours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0] if contours else None
    peri = float(cv2.arcLength(contour, True)) if contour is not None else 0.0
    circularity = 0.0 if peri <= 0 or area <= 0 else float(min(1.0, (4 * math.pi * area) / (peri * peri)))
    compactness = 0.0 if area <= 0 else float((peri * peri) / (4 * math.pi * area))
    ra, rb, phi, rect_len1, rect_len2, rect_phi = _axes(contour)
    rect_area = 0.0
    if contour is not None:
        _center, (rw, rh), _angle = cv2.minAreaRect(contour)
        rect_area = float(rw * rh)
    rectangularity = float(area) / rect_area if rect_area > 0 else 0.0
    hull_area = float(cv2.contourArea(cv2.convexHull(contour))) if contour is not None else 0.0
    convexity = float(area) / hull_area if hull_area > 0 else 0.0
    outer = 0.0
    diameter = 0.0
    if contour is not None:
        pts = contour.reshape(-1, 2).astype(np.float64)
        outer = float(np.max(np.hypot(pts[:, 0] - col, pts[:, 1] - row))) if len(pts) else 0.0
        hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float64)
        for i in range(len(hull)):
            delta = hull[i + 1 :] - hull[i]
            if len(delta):
                diameter = max(diameter, float(np.max(np.hypot(delta[:, 0], delta[:, 1]))))
    anisometry = ra / rb if rb > 1e-6 else 0.0
    bulkiness = (math.pi * ra * rb / area) if area > 0 else 0.0
    holes, hole_area = _hole_stats(mask)
    distances = []
    if contour is not None and area:
        pts = contour.reshape(-1, 2).astype(np.float64)
        distances = np.hypot(pts[:, 0] - col, pts[:, 1] - row)
    if len(distances):
        mean_dist = float(np.mean(distances))
        roundness = 0.0 if mean_dist <= 1e-6 else float(max(0.0, 1 - float(np.std(distances)) / mean_dist))
    else:
        roundness = 0.0
    sides = 0.0
    if contour is not None and peri > 0:
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        sides = float(len(approx))
    ratio = float(width) / float(height) if height else 0.0
    return {
        "area": float(area),
        "row": row,
        "column": col,
        "width": float(width),
        "height": float(height),
        "ratio": ratio,
        "circularity": circularity,
        "compactness": compactness,
        "contlength": peri,
        "rectangularity": rectangularity,
        "convexity": convexity,
        "ra": ra,
        "rb": rb,
        "phi": phi,
        "anisometry": anisometry,
        "bulkiness": bulkiness,
        "struct_factor": anisometry * bulkiness - 1,
        "outer_radius": outer,
        "max_diameter": diameter,
        "orientation": phi,
        "rect2_phi": rect_phi,
        "rect2_len1": rect_len1,
        "rect2_len2": rect_len2,
        "holes_num": float(holes),
        "area_holes": float(hole_area),
        "roundness": roundness,
        "num_sides": sides,
        "euler_number": float(1 - holes),
    }


def _hole_stats(mask: np.ndarray) -> tuple[int, float]:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, hierarchy = _find_contours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0, 0.0
    holes = 0
    area = 0.0
    for contour, link in zip(contours, hierarchy[0]):
        if int(link[3]) < 0:
            continue
        holes += 1
        area += abs(float(cv2.contourArea(contour)))
    return holes, area


def gray_measures(mask: np.ndarray, gray: np.ndarray | None) -> dict[str, float]:
    empty = {name: 0.0 for name in GRAY_FEATURES}
    if gray is None or gray.shape != mask.shape:
        return empty
    values = gray[mask > 0]
    if values.size == 0:
        return empty
    return {
        "gray_min": float(values.min()),
        "gray_max": float(values.max()),
        "gray_mean": float(values.mean()),
        "gray_deviation": float(values.std()),
        "gray_median": float(np.median(values)),
    }


def object_feature(mask: np.ndarray, feature: str) -> float:
    name = str(feature).strip().lower()
    values = region_measures(mask)
    if name not in values:
        known = ", ".join(REGION_FEATURES)
        raise ValueError(f"不支援的特徵 {feature}。可用 {known}")
    return values[name]


def xld_measures(points: list) -> dict[str, float]:
    if len(points) < 2:
        return {name: 0.0 for name in XLD_FEATURES}
    pts = np.array(points, np.float64).reshape(-1, 2)
    closed = np.linalg.norm(pts[0] - pts[-1]) < 1.5
    if closed and not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    segment = pts[1:] - pts[:-1]
    length = float(np.sum(np.hypot(segment[:, 0], segment[:, 1])))
    contour = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    area = abs(float(cv2.contourArea(contour)))
    moments = cv2.moments(contour)
    if moments["m00"]:
        row = float(moments["m01"] / moments["m00"])
        col = float(moments["m10"] / moments["m00"])
    else:
        row = float(np.mean(pts[:, 1]))
        col = float(np.mean(pts[:, 0]))
    x, y, width, height = cv2.boundingRect(contour)
    ra, rb, phi, rect_len1, rect_len2, rect_phi = _axes(contour)
    circularity = 0.0 if length <= 0 or area <= 0 else float(min(1.0, (4 * math.pi * area) / (length * length)))
    compactness = 0.0 if area <= 0 else float((length * length) / (4 * math.pi * area))
    hull = cv2.convexHull(contour)
    hull_area = abs(float(cv2.contourArea(hull)))
    convexity = area / hull_area if hull_area > 0 else 0.0
    _center, (rw, rh), _angle = cv2.minAreaRect(contour)
    rectangularity = area / float(rw * rh) if rw * rh > 0 else 0.0
    outer = float(np.max(np.hypot(pts[:, 0] - col, pts[:, 1] - row)))
    hull_pts = hull.reshape(-1, 2).astype(np.float64)
    diameter = 0.0
    for i in range(len(hull_pts)):
        delta = hull_pts[i + 1 :] - hull_pts[i]
        if len(delta):
            diameter = max(diameter, float(np.max(np.hypot(delta[:, 0], delta[:, 1]))))
    anisometry = ra / rb if rb > 1e-6 else 0.0
    bulkiness = (math.pi * ra * rb / area) if area > 0 else 0.0
    return {
        "area": area,
        "row": row,
        "column": col,
        "width": float(width),
        "height": float(height),
        "ratio": float(width) / float(height) if height else 0.0,
        "circularity": circularity,
        "compactness": compactness,
        "contlength": length,
        "rectangularity": rectangularity,
        "convexity": convexity,
        "ra": ra,
        "rb": rb,
        "phi": phi,
        "anisometry": anisometry,
        "bulkiness": bulkiness,
        "struct_factor": anisometry * bulkiness - 1,
        "outer_radius": outer,
        "max_diameter": diameter,
        "orientation": phi,
        "rect2_phi": rect_phi,
        "rect2_len1": rect_len1,
        "rect2_len2": rect_len2,
    }


def refine_polyline(points: list, gray: np.ndarray | None, level: float | None) -> list[list[float]]:
    pts = np.array(points, np.float64).reshape(-1, 2)
    if gray is None or len(pts) < 3:
        return [[float(x), float(y)] for x, y in pts]
    closed = np.linalg.norm(pts[0] - pts[-1]) < 1.5
    refined = []
    count = len(pts) - 1 if closed and len(pts) > 1 else len(pts)
    for index in range(count):
        point = pts[index]
        prev = pts[(index - 1) % count]
        nxt = pts[(index + 1) % count]
        tangent = nxt - prev
        norm = float(np.hypot(tangent[0], tangent[1]))
        if norm < 1e-6:
            refined.append([float(point[0]), float(point[1])])
            continue
        nx = -tangent[1] / norm
        ny = tangent[0] / norm
        samples = [_sample_gray(gray, point[0] + nx * t, point[1] + ny * t) for t in (-1.0, 0.0, 1.0)]
        offset = 0.0
        if level is not None:
            for (t0, g0), (t1, g1) in zip(((-1.0, samples[0]), (0.0, samples[1])), ((0.0, samples[1]), (1.0, samples[2]))):
                if (g0 - level) * (g1 - level) <= 0 and g0 != g1:
                    offset = t0 + (t1 - t0) * ((level - g0) / (g1 - g0))
                    break
        else:
            left, mid, right = samples
            denom = left - 2 * mid + right
            if abs(denom) > 1e-6:
                offset = max(-1.0, min(1.0, 0.5 * (left - right) / denom))
        refined.append([float(point[0] + nx * offset), float(point[1] + ny * offset)])
    if closed and refined:
        refined.append(refined[0])
    return refined


def subpixel_boundary(mask: np.ndarray, gray: np.ndarray | None, level: float | None, links: dict | None = None) -> list[list[list[float]]]:
    chains = []
    if links and links.get("points"):
        chains = [chain for chain in links["points"] if len(chain) >= 2]
    if not chains:
        contours, _hier = _find_contours(np.where(mask > 0, 255, 0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        chains = [[[int(x), int(y)] for x, y in contour.reshape(-1, 2)] for contour in contours if len(contour) >= 2]
    return [refine_polyline(chain, gray, level) for chain in chains]


def as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def fit_numbers(value, count: int, name: str) -> list[float]:
    nums = [float(item) for item in as_list(value)]
    if len(nums) == 1 and count > 1:
        nums = nums * count
    if len(nums) != count:
        raise ValueError(f"{name} 需要 {count} 個數值")
    return nums


def numbers_of(value) -> list[float]:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("空 tuple")
        return [float(item) for item in value]
    return [float(value)]


@operator(
    "檔案",
    [],
    [P("Image", "讀入的影像")],
    [P("FileName", "檔案路徑")],
    [],
    "讀取影像",
    "讀取灰階或彩色圖。相對路徑會依專案、samples、workspace 的順序尋找。路徑請用斜線，例如 samples/parts.png。",
    "read_image (Image, 'samples/parts.png')",
)
def read_image(filename, *, ctx):
    path = ctx.resolve_read_path(text(filename, "FileName"))
    data = read_cv_image(path)
    if data is None:
        raise RuntimeError(f"無法讀取影像 {path}")
    data = to_uint8(data)
    if data.ndim == 3 and data.shape[2] == 4:
        data = cv2.cvtColor(data, cv2.COLOR_BGRA2BGR)
    return Result([Image(data)], [])


@operator(
    "檔案",
    [P("Image", "要儲存的影像")],
    [],
    [P("FileName", "輸出路徑")],
    [],
    "儲存影像",
    "把影像寫到磁碟。相對路徑以專案目錄為起點。這版不使用 HALCON 的 Format 與 FillColor 參數。",
    "write_image (Image, 'workspace/out.png')",
)
def write_image(image, filename, *, ctx):
    image = expect_image(image, "Image")
    path = ctx.resolve_write_path(text(filename, "FileName"))
    if not write_cv_image(path, image.data):
        raise RuntimeError(f"無法寫入 {path}")
    ctx.log.append(f"已寫入 {path}")
    return Result([], [])


@operator(
    "灰階與濾波",
    [P("RGBImage", "彩色或灰階影像")],
    [P("GrayImage", "灰階影像")],
    [],
    [],
    "轉成灰階",
    "彩色影像轉成單通道。已經是灰階就直接複製。",
    "rgb1_to_gray (Image, Gray)",
)
def rgb1_to_gray(image, *, ctx):
    image = expect_image(image, "RGBImage")
    return image_out(image, image.gray())


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageInvert", "反相影像")],
    [],
    [],
    "灰階反相",
    "每個像素變成 255 減去原值。",
    "invert_image (Image, Inverted)",
)
def invert_image(image, *, ctx):
    image = expect_image(image, "Image")
    return image_out(image, cv2.bitwise_not(image.data))


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageScaled", "縮放後影像")],
    [P("Mult", "乘上的係數"), P("Add", "加上的偏移")],
    [],
    "線性縮放灰階",
    "輸出 = 輸入 × Mult + Add，超過 0 到 255 的值會被截斷。",
    "scale_image (Image, Scaled, 1.2, 0)",
)
def scale_image(image, mult, add, *, ctx):
    image = expect_image(image, "Image")
    out = image.data.astype(np.float32) * scalar(mult, "Mult") + scalar(add, "Add")
    return image_out(image, np.clip(out, 0, 255).astype(np.uint8))


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageGauss", "平滑後影像")],
    [P("Size", "奇數核大小，例如 3、5、7")],
    [],
    "高斯平滑",
    "Size 必須是大於等於 3 的奇數。",
    "gauss_image (Image, Smooth, 5)",
)
def gauss_image(image, size, *, ctx):
    image = expect_image(image, "Image")
    kernel = whole(size, "Size")
    if kernel < 3 or kernel % 2 == 0:
        raise ValueError("Size 必須是大於等於 3 的奇數")
    return image_out(image, cv2.GaussianBlur(image.data, (kernel, kernel), 0))


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageMedian", "中值濾波結果")],
    [P("Radius", "半徑，像素")],
    [],
    "中值濾波",
    "這版只收半徑。HALCON 的 MaskType 與 Margin 先省略。",
    "median_image (Image, Median, 2)",
)
def median_image(image, radius, *, ctx):
    image = expect_image(image, "Image")
    kernel = max(1, whole(radius, "Radius") * 2 + 1)
    if kernel % 2 == 0:
        kernel += 1
    return image_out(image, cv2.medianBlur(image.data, kernel))


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageMean", "均值濾波結果")],
    [P("MaskWidth", "遮罩寬"), P("MaskHeight", "遮罩高")],
    [],
    "均值濾波",
    "以矩形遮罩做平均。",
    "mean_image (Image, MeanImage, 5, 5)",
)
def mean_image(image, mask_width, mask_height, *, ctx):
    image = expect_image(image, "Image")
    width = max(1, whole(mask_width, "MaskWidth"))
    height = max(1, whole(mask_height, "MaskHeight"))
    return image_out(image, cv2.blur(image.data, (width, height)))


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageEmphasize", "強化後影像")],
    [P("MaskWidth", "遮罩寬"), P("MaskHeight", "遮罩高"), P("Factor", "強化係數")],
    [],
    "強化邊緣對比",
    "輸出 = 原圖 + (原圖 − 局部平均) × Factor。",
    "emphasize (Image, Sharp, 7, 7, 1)",
)
def emphasize(image, mask_width, mask_height, factor, *, ctx):
    image = expect_image(image, "Image")
    width = max(1, whole(mask_width, "MaskWidth") | 1)
    height = max(1, whole(mask_height, "MaskHeight") | 1)
    blur = cv2.blur(image.data, (width, height)).astype(np.float32)
    src = image.data.astype(np.float32)
    out = src + (src - blur) * scalar(factor, "Factor")
    return image_out(image, np.clip(out, 0, 255).astype(np.uint8))


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageEquHisto", "直方圖等化結果")],
    [],
    [],
    "直方圖等化",
    "灰階直接等化。彩色影像只等化亮度通道。",
    "equ_histo_image (Image, Equ)",
)
def equ_histo_image(image, *, ctx):
    image = expect_image(image, "Image")
    if image.channels == 1:
        out = cv2.equalizeHist(image.data)
    else:
        ycrcb = cv2.cvtColor(image.data, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return image_out(image, out)


@operator(
    "分割",
    [P("Image", "灰階或彩色影像")],
    [P("Region", "落入範圍的區域")],
    [P("MinGray", "最小灰階，含"), P("MaxGray", "最大灰階，含")],
    [],
    "HALCON 區間門檻",
    "這是 HALCON 的 threshold：灰階落在 MinGray 到 MaxGray（含兩端）才留下，結果是還沒拆開的單一區域。彩色影像會先轉灰階。有 domain 時只處理 domain 裡面的像素。執行後顯示灰階統計圖；兩端若是數字可以直接拖曳。拆物件請用 connection，可再設連通距離。",
    "threshold (Image, Region, 128, 255)",
)
def threshold(image, min_gray, max_gray, *, ctx):
    image = expect_image(image, "Image")
    lo = whole(min_gray, "MinGray")
    hi = whole(max_gray, "MaxGray")
    if lo > hi:
        mask = np.zeros((image.height, image.width), np.uint8)
    else:
        mask = cv2.inRange(image.gray(), lo, hi)
    if image.domain is not None:
        mask = cv2.bitwise_and(mask, image.domain)
    attach_histogram(ctx, image, lo, hi, mask, "range")
    return region_out(image.width, image.height, mask)


def smooth_histo_level(gray: np.ndarray) -> int:
    hist = np.bincount(gray.reshape(-1).astype(np.uint8), minlength=256).astype(np.float64)
    kernel = cv2.getGaussianKernel(15, 2).reshape(-1)
    smooth = np.convolve(hist, kernel, mode="same")
    peaks = [index for index in range(1, 255) if smooth[index] >= smooth[index - 1] and smooth[index] >= smooth[index + 1] and smooth[index] > 0]
    if len(peaks) < 2:
        used, _binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return int(round(float(used)))
    peaks.sort(key=lambda index: smooth[index], reverse=True)
    left, right = sorted(peaks[:2])
    valley = left + int(np.argmin(smooth[left : right + 1]))
    return int(valley)


@operator(
    "分割",
    [P("Image", "灰階或彩色影像")],
    [P("Region", "二值化區域")],
    [P("Method", "max_separability、smooth_histo、triangle、local_mean、local_gauss"), P("LightDark", "light 或 dark")],
    [P("UsedThreshold", "實際使用的門檻")],
    "自動門檻",
    "method 可以是 max_separability、smooth_histo、triangle、local_mean、local_gauss。light 保留亮處，dark 保留暗處。前三種會在統計圖標出門檻，局部方法用鄰域比較。",
    "binary_threshold (Image, Region, 'max_separability', 'light', UsedThreshold)",
)
def binary_threshold(image, method, light_dark, *, ctx):
    image = expect_image(image, "Image")
    method_name = text(method, "Method").lower()
    mode = text(light_dark, "LightDark").lower()
    if mode not in ("light", "dark"):
        raise ValueError("LightDark 只能是 light 或 dark")
    gray = image.gray()
    local = method_name in ("local_mean", "local_gauss")
    if method_name in ("max_separability", "otsu"):
        used, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method_name == "smooth_histo":
        used = smooth_histo_level(gray)
        _used, binary = cv2.threshold(gray, used, 255, cv2.THRESH_BINARY)
    elif method_name == "triangle":
        used, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    elif method_name == "local_mean":
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5)
        used = 5
    elif method_name == "local_gauss":
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
        used = 5
    else:
        raise ValueError("Method 可以是 max_separability、smooth_histo、triangle、local_mean、local_gauss")
    if mode == "dark":
        binary = cv2.bitwise_not(binary)
    if image.domain is not None:
        binary = cv2.bitwise_and(binary, image.domain)
    used_i = int(round(float(used)))
    if local:
        lo, hi = 0, 255
        kind = "local"
    elif mode == "light":
        lo, hi = min(255, used_i + 1), 255
        kind = "otsu" if method_name in ("max_separability", "otsu") else "auto"
    else:
        lo, hi = 0, used_i
        kind = "otsu" if method_name in ("max_separability", "otsu") else "auto"
    attach_histogram(ctx, image, lo, hi, binary, kind, threshold=used_i, lightDark=mode, method=method_name)
    result = region_out(image.width, image.height, binary)
    result.control.append(used_i)
    return result


@operator(
    "分割",
    [P("Image", "灰階或彩色影像")],
    [P("Region", "二值化區域")],
    [P("Threshold", "門檻，不含等於"), P("Method", "binary 或 binary_inv")],
    [],
    "OpenCV 二值化",
    "對應 cv2.threshold。binary 留下大於 Threshold 的像素，binary_inv 留下小於等於 Threshold 的像素。等於門檻的像素在 binary 不會留下。二值化結果會保留 OpenCV 的輪廓鏈表：外輪廓用 next/prev 串起來，孔洞用 child 掛在外輪廓下面。",
    "opencv_threshold (Image, Region, 128, 'binary')",
)
def opencv_threshold(image, thresh, method, *, ctx):
    image = expect_image(image, "Image")
    level = whole(thresh, "Threshold")
    if level < 0 or level > 255:
        raise ValueError("Threshold 要在 0 到 255")
    mode = text(method, "Method").lower()
    if mode == "binary":
        flag = cv2.THRESH_BINARY
        lo, hi = min(255, level + 1), 255
    elif mode == "binary_inv":
        flag = cv2.THRESH_BINARY_INV
        lo, hi = 0, level
    else:
        raise ValueError("Method 只能是 binary 或 binary_inv")
    _used, binary = cv2.threshold(image.gray(), level, 255, flag)
    if image.domain is not None:
        binary = cv2.bitwise_and(binary, image.domain)
    attach_histogram(ctx, image, lo, hi, binary, "opencv", threshold=level, lightDark=mode)
    return opencv_region(image.width, image.height, binary)


def subpix_contours(gray: np.ndarray, level: float, domain: np.ndarray | None = None) -> list[list[list[float]]]:
    mask = np.where(gray.astype(np.float32) >= float(level), 255, 0).astype(np.uint8)
    if domain is not None:
        mask = cv2.bitwise_and(mask, domain)
    lines = subpixel_boundary(mask, gray, float(level))
    return [line for line in lines if len(line) >= 3]


@operator(
    "分割",
    [P("Image", "灰階或彩色影像")],
    [P("Border", "次像素輪廓")],
    [P("Threshold", "灰階交界")],
    [],
    "次像素二值化",
    "沿著灰階等於 Threshold 的位置取出次像素輪廓，亮處留在輪廓內。結果是 XLD，不是整數像素的 region。多條輪廓用 16 色。門檻若是數字，可以在統計圖拖曳，也可以用次像素視窗調，那個視窗不會擋住主視窗。",
    "threshold_sub_pix (Image, Border, 128)",
)
def threshold_sub_pix(image, threshold, *, ctx):
    image = expect_image(image, "Image")
    level = scalar(threshold, "Threshold")
    if level < 0 or level > 255:
        raise ValueError("Threshold 要在 0 到 255")
    gray = image.gray()
    lines = subpix_contours(gray, level, image.domain)
    mask = np.where(gray.astype(np.float32) >= level, 255, 0).astype(np.uint8)
    if image.domain is not None:
        mask = cv2.bitwise_and(mask, image.domain)
    lo = min(255, int(round(level)))
    attach_histogram(ctx, image, lo, 255, mask, "subpix", threshold=level)
    return Result([XLD(image.width, image.height, lines, multi_color=True)], [])


def halcon_components(mask: np.ndarray) -> list[np.ndarray]:
    """HALCON connection：前景八連通，依掃描遇到的第一個像素排序。"""
    height, width = mask.shape
    foreground = mask > 0
    runs: list[tuple[int, int, int]] = []
    row_runs: list[list[int]] = []
    for row in range(height):
        line = foreground[row]
        indices: list[int] = []
        col = 0
        while col < width:
            if not line[col]:
                col += 1
                continue
            start = col
            while col < width and line[col]:
                col += 1
            indices.append(len(runs))
            runs.append((row, start, col - 1))
        row_runs.append(indices)
    if not runs:
        return []
    parent = list(range(len(runs)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for row in range(1, height):
        above = row_runs[row - 1]
        cursor = 0
        for index in row_runs[row]:
            _row, start, end = runs[index]
            low, high = start - 1, end + 1
            while cursor < len(above) and runs[above[cursor]][2] < low:
                cursor += 1
            probe = cursor
            while probe < len(above) and runs[above[probe]][1] <= high:
                union(index, above[probe])
                probe += 1
    groups: dict[int, list[tuple[int, int, int]]] = {}
    for index, run in enumerate(runs):
        groups.setdefault(find(index), []).append(run)
    objects = []
    for group in sorted(groups.values(), key=lambda item: (item[0][0], item[0][1])):
        component = np.zeros((height, width), np.uint8)
        for row, start, end in group:
            component[row, start : end + 1] = 255
        objects.append(component)
    return objects


def _boxes_within(left: tuple[int, int, int, int], right: tuple[int, int, int, int], distance: float) -> bool:
    gap_x = 0.0
    if left[2] < right[0]:
        gap_x = right[0] - left[2]
    elif right[2] < left[0]:
        gap_x = left[0] - right[2]
    gap_y = 0.0
    if left[3] < right[1]:
        gap_y = right[1] - left[3]
    elif right[3] < left[1]:
        gap_y = left[1] - right[3]
    return math.hypot(gap_x, gap_y) <= distance


def merge_within_distance(objects: list[np.ndarray], distance: float) -> list[np.ndarray]:
    """八連通之後，最近前景像素的距離不超過 distance 的物件再併起來。"""
    if distance < 2 or len(objects) < 2:
        return objects
    boxes = []
    for obj in objects:
        ys, xs = np.where(obj > 0)
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    parent = list(range(len(objects)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        if root_left < root_right:
            parent[root_right] = root_left
        else:
            parent[root_left] = root_right

    distances: dict[int, np.ndarray] = {}

    def distance_to(index: int) -> np.ndarray:
        cached = distances.get(index)
        if cached is None:
            inverse = np.where(objects[index] > 0, 0, 255).astype(np.uint8)
            cached = cv2.distanceTransform(inverse, cv2.DIST_L2, 5)
            distances[index] = cached
        return cached

    for left in range(len(objects)):
        for right in range(left + 1, len(objects)):
            if find(left) == find(right):
                continue
            if not _boxes_within(boxes[left], boxes[right], distance):
                continue
            ys, xs = np.where(objects[right] > 0)
            if float(distance_to(left)[ys, xs].min()) <= distance + 1e-3:
                union(left, right)
    grouped: dict[int, np.ndarray] = {}
    for index, obj in enumerate(objects):
        root = find(index)
        current = grouped.get(root)
        grouped[root] = obj if current is None else np.maximum(current, obj)
    return [grouped[index] for index in sorted(grouped)]


def opencv_components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    count, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=connectivity)
    objects = []
    for label in range(1, int(count)):
        objects.append(np.where(labels == label, 255, 0).astype(np.uint8))
    return objects


def opencv_link_list(mask: np.ndarray) -> dict:
    """OpenCV findContours 的鏈表。next/prev 是同一層，child/parent 是孔洞。索引從 0 開始，沒有則為 -1。"""
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    found = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = found[0] if len(found) == 2 else found[1]
    hierarchy = found[1] if len(found) == 2 else found[2]
    points: list[list[list[int]]] = []
    nxt: list[int] = []
    prev: list[int] = []
    child: list[int] = []
    parent: list[int] = []
    if hierarchy is not None:
        for contour, link in zip(contours, hierarchy[0]):
            flat = contour.reshape(-1, 2)
            points.append([[int(x), int(y)] for x, y in flat])
            nxt.append(int(link[0]))
            prev.append(int(link[1]))
            child.append(int(link[2]))
            parent.append(int(link[3]))
    return {"points": points, "next": nxt, "prev": prev, "child": child, "parent": parent}


def opencv_region(width: int, height: int, mask: np.ndarray, multi_color: bool = False) -> Result:
    if int(cv2.countNonZero(mask)) == 0:
        return Result([Region(width, height, [], multi_color=multi_color, links=[])], [])
    return Result([Region(width, height, [mask], multi_color=multi_color, links=[opencv_link_list(mask)])], [])


def connected_region(region: Region, objects: list[np.ndarray]) -> Result:
    return Result([Region(region.width, region.height, objects, multi_color=True)], [])


@operator(
    "分割",
    [P("Region", "輸入區域")],
    [P("ConnectedRegions", "個別連通物件")],
    [P("Distance", "連通距離，像素。1 是八連通，更大就把這個距離內的物件合併", "1")],
    [],
    "依距離連通",
    "先做八連通，斜角相接一定算同一個物件，不會拆得更碎。Distance 再用前景像素的最近距離：1 維持八連通，5 代表最近像素距離在 5 以內的物件併成一個。沒寫 Distance 時當作 1。每個物件一種顏色，16 色循環。",
    "connection (Region, ConnectedRegions, 1)",
)
def connection(region, distance=1, *, ctx):
    region = expect_region(region, "Region")
    dist = scalar(distance, "Distance")
    if dist < 0:
        raise ValueError("Distance 不能是負數")
    objects = merge_within_distance(halcon_components(region.union_mask()), dist)
    return connected_region(region, objects)


@operator(
    "分割",
    [P("Region", "輸入區域")],
    [P("ConnectedRegions", "個別連通物件")],
    [P("Connectivity", "4 或 8"), P("Distance", "連通距離，像素。1 不額外合併", "1")],
    [],
    "OpenCV 連通",
    "對應 cv2.connectedComponents。4 只算上下左右，8 連斜角也算。Distance 再把最近前景像素落在這個距離內的物件合併，1 表示不額外合併。每個物件保留自己的 OpenCV 輪廓鏈表。顯示時用 16 色。",
    "opencv_connection (Region, ConnectedRegions, 8, 1)",
)
def opencv_connection(region, connectivity, distance=1, *, ctx):
    region = expect_region(region, "Region")
    linked = whole(connectivity, "Connectivity")
    if linked not in (4, 8):
        raise ValueError("Connectivity 只能是 4 或 8")
    dist = scalar(distance, "Distance")
    if dist < 0:
        raise ValueError("Distance 不能是負數")
    objects = merge_within_distance(opencv_components(region.union_mask(), linked), dist)
    links = [opencv_link_list(mask) for mask in objects]
    return Result([Region(region.width, region.height, objects, multi_color=True, links=links)], [])


@operator(
    "分割",
    [P("Regions", "物件集合")],
    [P("SelectedRegions", "符合條件的物件")],
    [
        P("Features", "特徵名稱，或名稱 tuple"),
        P("Operation", "and 或 or"),
        P("Min", "下限，含"),
        P("Max", "上限，含"),
    ],
    [],
    "用形狀特徵篩選",
    "可用特徵和圖形視窗的勾選清單相同，包含 area、circularity、ra、rb、holes_num、euler_number 等。多個特徵時 Min/Max 也要是等長 tuple。",
    "select_shape (Regions, SelectedRegions, 'area', 'and', 150, 99999)",
)
def select_shape(region, features, operation, min_value, max_value, *, ctx):
    region = expect_region(region, "Regions")
    names = [str(item) for item in as_list(features)]
    mode = text(operation, "Operation").lower()
    if mode not in ("and", "or"):
        raise ValueError("Operation 只能是 and 或 or")
    lows = fit_numbers(min_value, len(names), "Min")
    highs = fit_numbers(max_value, len(names), "Max")
    kept = []
    kept_links = []
    linked = region.links if region.links is not None and len(region.links) == region.count() else None
    for index, mask in enumerate(region.objects):
        flags = []
        for name, low, high in zip(names, lows, highs):
            value = object_feature(mask, name)
            flags.append(low <= value <= high)
        if (all(flags) if mode == "and" else any(flags)):
            kept.append(mask)
            if linked is not None:
                kept_links.append(linked[index])
    return Result([Region(region.width, region.height, kept, multi_color=region.multi_color, links=kept_links or None)], [])


@operator(
    "分割",
    [P("Objects", "物件集合")],
    [P("ObjectSelected", "挑出的物件")],
    [P("Index", "從 1 開始的索引，可以是 tuple")],
    [],
    "按索引取物件",
    "索引從 1 開始，和 HDevelop 一樣。",
    "select_obj (Objects, ObjectSelected, 1)",
)
def select_obj(region, index, *, ctx):
    region = expect_region(region, "Objects")
    picked = []
    picked_links = []
    linked = region.links if region.links is not None and len(region.links) == region.count() else None
    for item in as_list(index):
        pos = int(round(float(item)))
        if pos < 1 or pos > region.count():
            raise IndexError(f"物件索引 {pos} 超出範圍 1..{region.count()}")
        picked.append(region.objects[pos - 1])
        if linked is not None:
            picked_links.append(linked[pos - 1])
    return Result([Region(region.width, region.height, picked, multi_color=region.multi_color, links=picked_links or None)], [])


@operator(
    "形態學",
    [P("Region", "輸入區域")],
    [P("RegionFillUp", "補滿孔洞後的區域")],
    [],
    [],
    "填滿孔洞",
    "填上區域內部的洞。結果是單一區域，要再拆開請呼叫 connection。",
    "fill_up (Region, RegionFillUp)",
)
def fill_up(region, *, ctx):
    region = expect_region(region, "Region")
    mask = region.union_mask()
    found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=-1)
    return region_out(region.width, region.height, filled)


@operator(
    "形態學",
    [P("Region", "輸入區域")],
    [P("RegionOpening", "開運算結果")],
    [P("Radius", "圓形結構元素半徑")],
    [],
    "圓形開運算",
    "作用在區域聯集上。結果是單一區域。",
    "opening_circle (Region, RegionOpening, 3.5)",
)
def opening_circle(region, radius, *, ctx):
    return morph(expect_region(region, "Region"), disk(scalar(radius, "Radius")), "open")


@operator(
    "形態學",
    [P("Region", "輸入區域")],
    [P("RegionClosing", "閉運算結果")],
    [P("Radius", "圓形結構元素半徑")],
    [],
    "圓形閉運算",
    "作用在區域聯集上。結果是單一區域。",
    "closing_circle (Region, RegionClosing, 3.5)",
)
def closing_circle(region, radius, *, ctx):
    return morph(expect_region(region, "Region"), disk(scalar(radius, "Radius")), "close")


@operator(
    "形態學",
    [P("Region", "輸入區域")],
    [P("RegionErosion", "侵蝕結果")],
    [P("Radius", "圓形結構元素半徑")],
    [],
    "圓形侵蝕",
    "作用在區域聯集上。結果是單一區域。",
    "erosion_circle (Region, RegionErosion, 2.5)",
)
def erosion_circle(region, radius, *, ctx):
    return morph(expect_region(region, "Region"), disk(scalar(radius, "Radius")), "erode")


@operator(
    "形態學",
    [P("Region", "輸入區域")],
    [P("RegionDilation", "膨脹結果")],
    [P("Radius", "圓形結構元素半徑")],
    [],
    "圓形膨脹",
    "作用在區域聯集上。結果是單一區域。",
    "dilation_circle (Region, RegionDilation, 2.5)",
)
def dilation_circle(region, radius, *, ctx):
    return morph(expect_region(region, "Region"), disk(scalar(radius, "Radius")), "dilate")


def morph_rect(region, width, height, mode: str) -> Result:
    kernel = rect_kernel(scalar(width, "Width"), scalar(height, "Height"))
    return morph(expect_region(region, "Region"), kernel, mode)


@operator(
    "形態學",
    [P("Region", "輸入區域")],
    [P("RegionErosion", "侵蝕結果")],
    [P("Width", "矩形寬"), P("Height", "矩形高")],
    [],
    "矩形侵蝕",
    "結構元素是實心矩形。偶數寬高會加 1，讓中心對齊像素。作用在區域聯集上，結果是單一區域。",
    "erosion_rectangle1 (Region, RegionErosion, 5, 5)",
)
def erosion_rectangle1(region, width, height, *, ctx):
    return morph_rect(region, width, height, "erode")


@operator(
    "形態學",
    [P("Region", "輸入區域")],
    [P("RegionDilation", "膨脹結果")],
    [P("Width", "矩形寬"), P("Height", "矩形高")],
    [],
    "矩形膨脹",
    "結構元素是實心矩形。偶數寬高會加 1。作用在區域聯集上，結果是單一區域。",
    "dilation_rectangle1 (Region, RegionDilation, 5, 5)",
)
def dilation_rectangle1(region, width, height, *, ctx):
    return morph_rect(region, width, height, "dilate")


@operator(
    "形態學",
    [P("Region", "輸入區域")],
    [P("RegionOpening", "開運算結果")],
    [P("Width", "矩形寬"), P("Height", "矩形高")],
    [],
    "矩形開運算",
    "先侵蝕再膨脹。作用在區域聯集上，結果是單一區域。",
    "opening_rectangle1 (Region, RegionOpening, 5, 5)",
)
def opening_rectangle1(region, width, height, *, ctx):
    return morph_rect(region, width, height, "open")


@operator(
    "形態學",
    [P("Region", "輸入區域")],
    [P("RegionClosing", "閉運算結果")],
    [P("Width", "矩形寬"), P("Height", "矩形高")],
    [],
    "矩形閉運算",
    "先膨脹再侵蝕。作用在區域聯集上，結果是單一區域。",
    "closing_rectangle1 (Region, RegionClosing, 5, 5)",
)
def closing_rectangle1(region, width, height, *, ctx):
    return morph_rect(region, width, height, "close")


@operator(
    "形態學",
    [P("Region", "輸入區域")],
    [P("RegionBorder", "邊界區域")],
    [P("BoundaryType", "inner 或 outer")],
    [],
    "區域邊界",
    "inner 是原區域減去侵蝕一圈，outer 是膨脹一圈再減去原區域。",
    "boundary (Region, RegionBorder, 'inner')",
)
def boundary(region, boundary_type, *, ctx):
    region = expect_region(region, "Region")
    kind = text(boundary_type, "BoundaryType").lower()
    mask = region.union_mask()
    kernel = np.ones((3, 3), np.uint8)
    if kind == "inner":
        out = cv2.subtract(mask, cv2.erode(mask, kernel))
    elif kind == "outer":
        out = cv2.subtract(cv2.dilate(mask, kernel), mask)
    else:
        raise ValueError("BoundaryType 只能是 inner 或 outer")
    return region_out(region.width, region.height, out)


@operator(
    "區域",
    [P("Region", "被減的區域"), P("Sub", "要減掉的區域")],
    [P("RegionDifference", "差集")],
    [],
    [],
    "區域相減",
    "留下只在第一個區域、不在第二個區域的像素。",
    "difference (Region, Sub, RegionDifference)",
)
def difference(region, sub, *, ctx):
    region = expect_region(region, "Region")
    sub = expect_region(sub, "Sub")
    same_size(region, sub, "difference")
    out = cv2.bitwise_and(region.union_mask(), cv2.bitwise_not(sub.union_mask()))
    return region_out(region.width, region.height, out)


@operator(
    "區域",
    [P("Region1", "區域一"), P("Region2", "區域二")],
    [P("RegionIntersection", "交集")],
    [],
    [],
    "區域交集",
    "留下兩個區域都有的像素。",
    "intersection (Region1, Region2, RegionIntersection)",
)
def intersection(region1, region2, *, ctx):
    region1 = expect_region(region1, "Region1")
    region2 = expect_region(region2, "Region2")
    same_size(region1, region2, "intersection")
    out = cv2.bitwise_and(region1.union_mask(), region2.union_mask())
    return region_out(region1.width, region1.height, out)


@operator(
    "區域",
    [P("Region1", "區域一"), P("Region2", "區域二")],
    [P("RegionUnion", "聯集")],
    [],
    [],
    "兩個區域的聯集",
    "結果是單一區域。",
    "union2 (Region1, Region2, RegionUnion)",
)
def union2(region1, region2, *, ctx):
    region1 = expect_region(region1, "Region1")
    region2 = expect_region(region2, "Region2")
    same_size(region1, region2, "union2")
    out = cv2.bitwise_or(region1.union_mask(), region2.union_mask())
    return region_out(region1.width, region1.height, out)


@operator(
    "區域",
    [P("Region", "物件集合")],
    [P("RegionUnion", "合併成一個區域")],
    [],
    [],
    "合併全部物件",
    "把 connection 拆開的物件合成一個區域。",
    "union1 (Region, RegionUnion)",
)
def union1(region, *, ctx):
    region = expect_region(region, "Region")
    return region_out(region.width, region.height, region.union_mask())


@operator(
    "區域",
    [P("Region", "輸入區域")],
    [P("RegionComplement", "補集")],
    [],
    [],
    "區域補集",
    "在同一張畫布上，把有像素和沒像素的地方對調。",
    "complement (Region, RegionComplement)",
)
def complement(region, *, ctx):
    region = expect_region(region, "Region")
    return region_out(region.width, region.height, cv2.bitwise_not(region.union_mask()))


@operator(
    "區域",
    [P("Image", "輸入影像"), P("Region", "要留下的範圍")],
    [P("ImageReduced", "套用 domain 的影像")],
    [],
    [],
    "用區域限制影像",
    "像素不會被裁掉，domain 以外的地方之後不會被 threshold 使用。顯示時 domain 外會變暗。",
    "reduce_domain (Image, Region, ImageReduced)",
)
def reduce_domain(image, region, *, ctx):
    image = expect_image(image, "Image")
    region = expect_region(region, "Region")
    if image.width != region.width or image.height != region.height:
        raise ValueError("影像和區域的尺寸不同")
    reduced = image.copy()
    reduced.domain = region.union_mask()
    return Result([reduced], [])


@operator(
    "區域",
    [P("Image", "輸入影像")],
    [P("Domain", "影像的 domain")],
    [],
    [],
    "取得 domain",
    "沒有做過 reduce_domain 時，domain 就是整張影像。",
    "get_domain (Image, Domain)",
)
def get_domain(image, *, ctx):
    image = expect_image(image, "Image")
    if image.domain is None:
        mask = np.full((image.height, image.width), 255, np.uint8)
    else:
        mask = image.domain.copy()
    return region_out(image.width, image.height, mask)


@operator(
    "區域",
    [P("Image", "輸入影像")],
    [P("ImagePart", "裁切後的影像")],
    [
        P("Row1", "上邊界，含"),
        P("Column1", "左邊界，含"),
        P("Row2", "下邊界，含"),
        P("Column2", "右邊界，含"),
    ],
    [],
    "用矩形裁切",
    "座標和 HALCON 一樣含頭含尾。超出影像的部分會被夾住。",
    "crop_rectangle1 (Image, ImagePart, 0, 0, 100, 100)",
)
def crop_rectangle1(image, row1, column1, row2, column2, *, ctx):
    image = expect_image(image, "Image")
    r1 = max(0, whole(row1, "Row1"))
    c1 = max(0, whole(column1, "Column1"))
    r2 = min(image.height - 1, whole(row2, "Row2"))
    c2 = min(image.width - 1, whole(column2, "Column2"))
    if r2 < r1 or c2 < c1:
        raise ValueError("裁切範圍是空的")
    crop = image.data[r1 : r2 + 1, c1 : c2 + 1].copy()
    domain = None if image.domain is None else image.domain[r1 : r2 + 1, c1 : c2 + 1].copy()
    return Result([Image(crop, domain)], [])


@operator(
    "區域",
    [],
    [P("Rectangle", "軸對齊矩形")],
    [
        P("Row1", "上"),
        P("Column1", "左"),
        P("Row2", "下"),
        P("Column2", "右"),
    ],
    [],
    "產生矩形區域",
    "尺寸對齊最近一張影像。請先 read_image，否則使用 480×640。",
    "gen_rectangle1 (Rectangle, 40, 40, 200, 300)",
)
def gen_rectangle1(row1, column1, row2, column2, *, ctx):
    height, width = ctx.canvas_size()
    r1 = whole(row1, "Row1")
    c1 = whole(column1, "Column1")
    r2 = whole(row2, "Row2")
    c2 = whole(column2, "Column2")
    if r2 < r1 or c2 < c1:
        raise ValueError("矩形座標不正確")
    if ctx.ref_shape is None and ctx.display.base is None:
        height = r2 + 1
        width = c2 + 1
        ctx.ref_shape = (height, width)
    mask = np.zeros((height, width), np.uint8)
    rr1 = max(0, min(height - 1, r1))
    cc1 = max(0, min(width - 1, c1))
    rr2 = max(0, min(height - 1, r2))
    cc2 = max(0, min(width - 1, c2))
    mask[rr1 : rr2 + 1, cc1 : cc2 + 1] = 255
    return Result([Region(width, height, [mask])], [])


@operator(
    "區域",
    [],
    [P("Circle", "實心圓")],
    [P("Row", "圓心列"), P("Column", "圓心行"), P("Radius", "半徑")],
    [],
    "產生圓形區域",
    "尺寸對齊最近一張影像。請先 read_image。",
    "gen_circle (Circle, 120, 160, 40)",
)
def gen_circle(row, column, radius, *, ctx):
    height, width = ctx.canvas_size()
    mask = np.zeros((height, width), np.uint8)
    center = (whole(column, "Column"), whole(row, "Row"))
    cv2.circle(mask, center, max(0, whole(radius, "Radius")), 255, -1)
    return Result([Region(width, height, [mask])], [])


@operator(
    "特徵",
    [P("Regions", "物件集合")],
    [],
    [],
    [P("Area", "面積"), P("Row", "中心列"), P("Column", "中心行")],
    "面積與重心",
    "每個物件各得到一個值，所以 Area、Row、Column 都是 tuple。索引從 1 開始。",
    "area_center (Regions, Area, Row, Column)",
)
def area_center(region, *, ctx):
    region = expect_region(region, "Regions")
    areas, rows, cols = [], [], []
    for mask in region.objects:
        area = int(cv2.countNonZero(mask))
        moments = cv2.moments(mask)
        if area == 0 or not moments["m00"]:
            continue
        areas.append(area)
        rows.append(float(moments["m01"] / moments["m00"]))
        cols.append(float(moments["m10"] / moments["m00"]))
    return Result([], [areas, rows, cols])


@operator(
    "特徵",
    [P("Regions", "物件集合")],
    [],
    [],
    [
        P("Row1", "上"),
        P("Column1", "左"),
        P("Row2", "下"),
        P("Column2", "右"),
    ],
    "最小軸對齊外接矩形",
    "每個物件一組座標，四個輸出都是 tuple。",
    "smallest_rectangle1 (Regions, Row1, Column1, Row2, Column2)",
)
def smallest_rectangle1(region, *, ctx):
    region = expect_region(region, "Regions")
    row1, col1, row2, col2 = [], [], [], []
    for mask in region.objects:
        if cv2.countNonZero(mask) == 0:
            continue
        x, y, width, height = cv2.boundingRect(mask)
        col1.append(int(x))
        row1.append(int(y))
        col2.append(int(x + width - 1))
        row2.append(int(y + height - 1))
    return Result([], [row1, col1, row2, col2])


def _oriented_rect(mask: np.ndarray):
    found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]
    if not contours:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    contour = max(contours, key=cv2.contourArea)
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
    if rw < rh:
        rw, rh = rh, rw
        angle += 90.0
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return float(cy), float(cx), math.radians(angle), float(rw) / 2.0, float(rh) / 2.0


@operator(
    "特徵",
    [P("Regions", "物件集合")],
    [],
    [],
    [
        P("Row", "中心列"),
        P("Column", "中心行"),
        P("Phi", "長邊角度，弳度"),
        P("Length1", "長邊的一半"),
        P("Length2", "短邊的一半"),
    ],
    "最小外接旋轉矩形",
    "Length1、Length2 是半長，和 HALCON 一樣。Phi 是長邊方向。",
    "smallest_rectangle2 (Regions, Row, Column, Phi, Length1, Length2)",
)
def smallest_rectangle2(region, *, ctx):
    region = expect_region(region, "Regions")
    rows, cols, phis, len1, len2 = [], [], [], [], []
    for mask in region.objects:
        if cv2.countNonZero(mask) == 0:
            continue
        row, col, phi, a, b = _oriented_rect(mask)
        rows.append(row)
        cols.append(col)
        phis.append(phi)
        len1.append(a)
        len2.append(b)
    return Result([], [rows, cols, phis, len1, len2])


@operator(
    "特徵",
    [P("Regions", "物件集合")],
    [],
    [],
    [P("Row", "圓心列"), P("Column", "圓心行"), P("Radius", "半徑")],
    "最小外接圓",
    "每個物件一組圓心與半徑。",
    "smallest_circle (Regions, Row, Column, Radius)",
)
def smallest_circle(region, *, ctx):
    region = expect_region(region, "Regions")
    rows, cols, radii = [], [], []
    for mask in region.objects:
        found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = found[0] if len(found) == 2 else found[1]
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        rows.append(float(cy))
        cols.append(float(cx))
        radii.append(float(radius))
    return Result([], [rows, cols, radii])


@operator(
    "特徵",
    [P("Regions", "物件集合"), P("Image", "用來量灰階的影像")],
    [],
    [],
    [P("Mean", "平均灰階"), P("Deviation", "標準差")],
    "區域灰階統計",
    "每個物件的平均亮度與標準差。",
    "intensity (Regions, Image, Mean, Deviation)",
)
def intensity(region, image, *, ctx):
    region = expect_region(region, "Regions")
    image = expect_image(image, "Image")
    gray = image.gray()
    means, devs = [], []
    for mask in region.objects:
        values = gray[mask > 0]
        if values.size == 0:
            means.append(0.0)
            devs.append(0.0)
            continue
        means.append(float(values.mean()))
        devs.append(float(values.std()))
    return Result([], [means, devs])


@operator(
    "特徵",
    [P("Objects", "物件集合")],
    [],
    [],
    [P("Number", "物件數")],
    "物件數量",
    "回傳單一整數，不是 tuple。",
    "count_obj (Objects, Number)",
)
def count_obj(region, *, ctx):
    region = expect_region(region, "Objects")
    return Result([], [region.count()])


@operator(
    "特徵",
    [P("Regions", "物件集合")],
    [],
    [P("Features", "一個特徵名稱，或名稱 tuple")],
    [P("Value", "特徵值 tuple")],
    "讀取區域特徵",
    "單一特徵時，Value 依物件排列。多個特徵時，先排完第一個特徵的全部物件，再排下一個。",
    "region_features (Regions, 'area', Value)",
)
def region_features(region, features, *, ctx):
    region = expect_region(region, "Regions")
    values = []
    for feature in as_list(features):
        for mask in region.objects:
            values.append(object_feature(mask, str(feature)))
    return Result([], [values])


@operator(
    "輪廓",
    [P("Image", "輸入影像")],
    [P("EdgeAmplitude", "梯度振幅")],
    [P("FilterType", "sum_abs 或 sum_sqrt"), P("Size", "奇數核大小")],
    [],
    "Sobel 振幅",
    "輸出是 8-bit。超過 255 的振幅會被截斷，所以後面的 threshold 要用 0 到 255 的門檻。",
    "sobel_amp (Image, EdgeAmplitude, 'sum_abs', 3)",
)
def sobel_amp(image, filter_type, size, *, ctx):
    image = expect_image(image, "Image")
    kernel = whole(size, "Size")
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("Size 必須是正奇數")
    gray = image.gray()
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=kernel)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=kernel)
    kind = text(filter_type, "FilterType").lower()
    if kind == "sum_sqrt":
        mag = cv2.magnitude(dx, dy)
    else:
        mag = np.abs(dx) + np.abs(dy)
    return image_out(image, np.clip(mag, 0, 255).astype(np.uint8))


@operator(
    "輪廓",
    [P("Image", "輸入影像")],
    [P("Edges", "邊緣輪廓")],
    [
        P("Filter", "目前接受任意名稱，內部用 Canny"),
        P("Alpha", "平滑程度"),
        P("Low", "低門檻"),
        P("High", "高門檻"),
    ],
    [],
    "萃取邊緣輪廓",
    "這是 Canny 近似，不是 HALCON 的次像素 edges_sub_pix。輪廓是 XLD，可用 dev_display 畫出。",
    "edges_sub_pix (Image, Edges, 'canny', 1.0, 20, 40)",
)
def edges_sub_pix(image, _filter, alpha, low, high, *, ctx):
    image = expect_image(image, "Image")
    sigma = max(0.4, scalar(alpha, "Alpha"))
    kernel = max(3, int(round(sigma * 6)) | 1)
    blur = cv2.GaussianBlur(image.gray(), (kernel, kernel), sigma)
    edges = cv2.Canny(blur, scalar(low, "Low"), scalar(high, "High"))
    if image.domain is not None:
        edges = cv2.bitwise_and(edges, image.domain)
    found = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = found[0] if len(found) == 2 else found[1]
    polylines = []
    for contour in contours:
        if len(contour) < 2:
            continue
        polylines.append([[float(point[0][0]), float(point[0][1])] for point in contour])
    return Result([XLD(image.width, image.height, polylines)], [])


@operator(
    "輪廓",
    [P("Regions", "區域")],
    [P("Contours", "區域外輪廓")],
    [],
    [],
    "區域轉成輪廓",
    "每個物件的外輪廓變成一條 XLD。",
    "gen_contour_region_xld (Regions, Contours)",
)
def gen_contour_region_xld(region, *, ctx):
    region = expect_region(region, "Regions")
    polylines = []
    for mask in region.objects:
        for line in _contour_lines(mask):
            polylines.append(line)
    return Result([XLD(region.width, region.height, polylines)], [])


def _contour_lines(mask: np.ndarray):
    found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = found[0] if len(found) == 2 else found[1]
    lines = []
    for contour in contours:
        if len(contour) < 2:
            continue
        lines.append([[float(point[0][0]), float(point[0][1])] for point in contour])
    return lines


def _pair(rows, cols):
    row_list = [float(item) for item in as_list(rows)]
    col_list = [float(item) for item in as_list(cols)]
    if len(row_list) == 1 and len(col_list) > 1:
        row_list = row_list * len(col_list)
    if len(col_list) == 1 and len(row_list) > 1:
        col_list = col_list * len(row_list)
    if len(row_list) != len(col_list):
        raise ValueError("Row 與 Column 的長度不同")
    return row_list, col_list


@operator(
    "輪廓",
    [],
    [P("Cross", "十字輪廓")],
    [
        P("Row", "列，可以是 tuple"),
        P("Column", "行，可以是 tuple"),
        P("Size", "十字的全長"),
        P("Angle", "旋轉角，弳度"),
    ],
    [],
    "產生十字標記",
    "用來標重心。Row、Column 通常直接接 area_center 的輸出。",
    "gen_cross_contour_xld (Cross, Row, Column, 16, 0)",
)
def gen_cross_contour_xld(row, column, size, angle, *, ctx):
    rows, cols = _pair(row, column)
    half = scalar(size, "Size") / 2.0
    theta = scalar(angle, "Angle")
    dx, dy = math.cos(theta) * half, math.sin(theta) * half
    px, py = -math.sin(theta) * half, math.cos(theta) * half
    height, width = ctx.canvas_size()
    contours = []
    for r, c in zip(rows, cols):
        contours.append([[c - dx, r - dy], [c + dx, r + dy]])
        contours.append([[c - px, r - py], [c + px, r + py]])
    return Result([XLD(width, height, contours)], [])


@operator(
    "顯示",
    [P("Object", "影像、區域或輪廓")],
    [],
    [],
    [],
    "顯示到圖形視窗",
    "顯示影像會清掉舊的疊加。顯示區域或輪廓會疊在目前影像上。",
    "dev_display (Object)",
)
def dev_display(obj, *, ctx):
    ctx.display_obj(obj)
    return Result([], [])


@operator(
    "顯示",
    [],
    [],
    [],
    [],
    "清空圖形視窗",
    "清掉目前的影像和疊加。",
    "dev_clear_window ()",
)
def dev_clear_window(*, ctx):
    ctx.display.clear()
    return Result([], [])


@operator(
    "顯示",
    [],
    [],
    [P("Color", "顏色名稱或 #RRGGBB")],
    [],
    "設定繪圖顏色",
    "可用 red、green、blue、yellow、cyan、magenta、white、orange，或 #RRGGBB。",
    "dev_set_color ('green')",
)
def dev_set_color(color, *, ctx):
    ctx.display.color = normalize_color(text(color, "Color"))
    return Result([], [])


@operator(
    "顯示",
    [],
    [],
    [P("DrawMode", "margin 或 fill")],
    [],
    "設定區域畫法",
    "margin 畫輪廓，fill 填色。",
    "dev_set_draw ('margin')",
)
def dev_set_draw(draw_mode, *, ctx):
    mode = text(draw_mode, "DrawMode").lower()
    if mode not in ("fill", "margin"):
        raise ValueError("DrawMode 只能是 fill 或 margin")
    ctx.display.draw = mode
    return Result([], [])


@operator(
    "顯示",
    [],
    [],
    [P("Width", "線寬，像素")],
    [],
    "設定線寬",
    "線寬是螢幕像素，放大影像時不會跟著變粗。",
    "dev_set_line_width (2)",
)
def dev_set_line_width(width, *, ctx):
    ctx.display.line_width = max(1, whole(width, "Width"))
    return Result([], [])


@operator(
    "顯示",
    [],
    [],
    [P("Text", "文字"), P("Row", "列"), P("Column", "行")],
    [],
    "在圖上寫字",
    "VisDev 擴充。HDevelop 要透過視窗 handle，這裡直接寫進目前的圖形視窗。",
    "dev_disp_text ('ok', 24, 12)",
)
def dev_disp_text(message, row, column, *, ctx):
    ctx.display.overlays.append(
        {
            "kind": "text",
            "text": text(message, "Text"),
            "row": scalar(row, "Row"),
            "col": scalar(column, "Column"),
            "color": ctx.display.color,
        }
    )
    return Result([], [])


@operator(
    "顯示",
    [],
    [],
    [],
    [],
    "暫停執行",
    "停在下一行，和中斷點一樣，可以再按執行或單步繼續。",
    "stop ()",
)
def stop(*, ctx):
    raise StopSignal()


@operator(
    "Tuple",
    [],
    [],
    [P("Tuple", "控制變數")],
    [P("Length", "長度")],
    "tuple 長度",
    "單一數值的長度是 1。程式裡也可以寫 |Tuple|。",
    "tuple_length (Tuple, Length)",
)
def tuple_length(value, *, ctx):
    return Result([], [len(value) if isinstance(value, list) else 1])


@operator(
    "Tuple",
    [],
    [],
    [P("Tuple", "數值 tuple")],
    [P("Min", "最小值")],
    "tuple 最小值",
    "回傳單一數值。",
    "tuple_min (Tuple, Min)",
)
def tuple_min(value, *, ctx):
    return Result([], [min(numbers_of(value))])


@operator(
    "Tuple",
    [],
    [],
    [P("Tuple", "數值 tuple")],
    [P("Max", "最大值")],
    "tuple 最大值",
    "回傳單一數值。",
    "tuple_max (Tuple, Max)",
)
def tuple_max(value, *, ctx):
    return Result([], [max(numbers_of(value))])


@operator(
    "Tuple",
    [],
    [],
    [P("Tuple", "數值 tuple")],
    [P("Sum", "總和")],
    "tuple 總和",
    "回傳單一數值。",
    "tuple_sum (Tuple, Sum)",
)
def tuple_sum(value, *, ctx):
    return Result([], [sum(numbers_of(value))])


@operator(
    "Tuple",
    [],
    [],
    [P("Tuple", "數值 tuple")],
    [P("Mean", "平均")],
    "tuple 平均",
    "回傳單一數值。",
    "tuple_mean (Tuple, Mean)",
)
def tuple_mean(value, *, ctx):
    nums = numbers_of(value)
    return Result([], [sum(nums) / len(nums)])


def _optional_number(value, name: str):
    if isinstance(value, str) and value.strip().lower() == "auto":
        return None
    return scalar(value, name)


def _teach_image(image: Image, row1, column1, row2, column2) -> Image:
    picture = expect_image(image, "Image")
    r1 = max(0, whole(row1, "Row1"))
    c1 = max(0, whole(column1, "Column1"))
    r2 = min(picture.height - 1, whole(row2, "Row2"))
    c2 = min(picture.width - 1, whole(column2, "Column2"))
    if r2 - r1 < 7 or c2 - c1 < 7:
        raise ValueError("教導框至少要 8×8，而且要落在影像內。")
    crop = picture.data[r1 : r2 + 1, c1 : c2 + 1].copy()
    domain = None if picture.domain is None else picture.domain[r1 : r2 + 1, c1 : c2 + 1].copy()
    return Image(crop, domain)


def _expect_model(value):
    from .shape_model import ShapeModel

    if not isinstance(value, ShapeModel):
        raise TypeError("ModelID 必須是 create_shape_model 建立的形狀模型")
    return value


@operator(
    "形狀匹配",
    [P("Template", "模板影像，可用 domain 限定範圍")],
    [],
    [
        P("NumLevels", "金字塔層數，0 為自動"),
        P("AngleStart", "起始角，弳度"),
        P("AngleExtent", "搜尋角度範圍，弳度"),
        P("AngleStep", "角度步進，0 為自動"),
        P("Metric", "use_polarity 或 ignore_local_polarity"),
        P("Contrast", "邊緣對比，或 auto"),
        P("MinContrast", "搜尋圖的最低對比，或 auto"),
    ],
    [P("ModelID", "形狀模型")],
    "建立形狀模型",
    "沿用梯度方向的形狀匹配。這就是教導：整張圖或 domain 成為模板，原點是 domain 重心；沒有 domain 時用影像中心。",
    "create_shape_model (Template, 0, rad(-180), rad(360), 0, 'use_polarity', 'auto', 'auto', ModelID)",
)
def create_shape_model(template, num_levels, angle_start, angle_extent, angle_step, metric, contrast, min_contrast, *, ctx):
    from .shape_model import create_shape_model as build_model

    image = expect_image(template, "Template")
    kind = text(metric, "Metric").lower()
    if kind not in ("use_polarity", "ignore_local_polarity"):
        raise ValueError("Metric 只能是 use_polarity 或 ignore_local_polarity")
    model = build_model(
        image,
        whole(num_levels, "NumLevels"),
        _optional_number(angle_step, "AngleStep") or 0,
        kind,
        _optional_number(contrast, "Contrast"),
        _optional_number(min_contrast, "MinContrast"),
    )
    model.angle_start = scalar(angle_start, "AngleStart")
    model.angle_extent = scalar(angle_extent, "AngleExtent")
    return Result([], [model])


@operator(
    "形狀匹配",
    [P("Image", "要教導的影像")],
    [],
    [
        P("Row1", "教導框上邊界"),
        P("Column1", "教導框左邊界"),
        P("Row2", "教導框下邊界"),
        P("Column2", "教導框右邊界"),
        P("NumLevels", "金字塔層數，0 為自動"),
        P("AngleStart", "起始角，弳度"),
        P("AngleExtent", "角度範圍，弳度"),
        P("AngleStep", "角度步進，0 為自動"),
        P("Metric", "use_polarity 或 ignore_local_polarity"),
        P("Contrast", "邊緣對比，或 auto"),
        P("MinContrast", "搜尋圖的最低對比，或 auto"),
    ],
    [P("ModelID", "形狀模型")],
    "教導形狀模型",
    "用矩形框出要記的圖案。模型原點是教導框中心，定位時的 Row、Column 就是這個中心。",
    "teach_shape_model (Image, 15, 15, 164, 164, 0, rad(-180), rad(360), 0, 'use_polarity', 40, 10, ModelID)",
)
def teach_shape_model(
    image,
    row1,
    column1,
    row2,
    column2,
    num_levels,
    angle_start,
    angle_extent,
    angle_step,
    metric,
    contrast,
    min_contrast,
    *,
    ctx,
):
    taught = _teach_image(image, row1, column1, row2, column2)
    return create_shape_model(taught, num_levels, angle_start, angle_extent, angle_step, metric, contrast, min_contrast, ctx=ctx)


@operator(
    "形狀匹配",
    [P("Image", "要搜尋的影像")],
    [],
    [
        P("ModelID", "形狀模型"),
        P("AngleStart", "起始角，弳度"),
        P("AngleExtent", "搜尋角度範圍，弳度"),
        P("MinScore", "最低分數，0 到 1"),
        P("NumMatches", "最多幾個，0 表示全部"),
        P("MaxOverlap", "允許重疊比例，0 到 1"),
        P("SubPixel", "none 或 interpolation"),
        P("NumLevels", "使用層數，0 表示模型全部層"),
        P("Greediness", "0 到 1，越大越快，也可能漏掉"),
    ],
    [P("Row", "列"), P("Column", "行"), P("Angle", "角度，弳度"), P("Score", "分數")],
    "尋找形狀模型",
    "金字塔由粗到細搜尋後，對分數曲面做拋物線內插，得到次像素的列、行與角度。",
    "find_shape_model (Image, ModelID, rad(-180), rad(360), 0.6, 0, 0.5, 'interpolation', 0, 0.8, Row, Column, Angle, Score)",
)
def find_shape_model(
    image,
    model,
    angle_start,
    angle_extent,
    min_score,
    num_matches,
    max_overlap,
    subpixel,
    num_levels,
    greediness,
    *,
    ctx,
):
    from .shape_model import find_shape_model as search

    picture = expect_image(image, "Image")
    shape = _expect_model(model)
    mode = text(subpixel, "SubPixel").lower()
    if mode not in ("none", "interpolation", "least_squares"):
        raise ValueError("SubPixel 只能是 none、interpolation 或 least_squares")
    rows, cols, angles, scores = search(
        picture,
        shape,
        scalar(angle_start, "AngleStart"),
        scalar(angle_extent, "AngleExtent"),
        scalar(min_score, "MinScore"),
        whole(num_matches, "NumMatches"),
        scalar(max_overlap, "MaxOverlap"),
        mode != "none",
        whole(num_levels, "NumLevels"),
        scalar(greediness, "Greediness"),
    )
    return Result([], [rows, cols, angles, scores])


@operator(
    "形狀匹配",
    [],
    [P("ModelContours", "模型輪廓")],
    [P("ModelID", "形狀模型"), P("Level", "1 是最細的一層")],
    [],
    "取出模型輪廓",
    "輪廓點已沿梯度做到次像素，座標在模板影像上。Level 1 是原始解析度。",
    "get_shape_model_contours (ModelContours, ModelID, 1)",
)
def get_shape_model_contours(model, level, *, ctx):
    shape = _expect_model(model)
    step = max(0, whole(level, "Level") - 1)
    scale = float(2**step)
    contours = []
    for chain in shape.contours:
        contours.append(
            [[shape.origin_col + point[0] / scale, shape.origin_row + point[1] / scale] for point in chain]
        )
    width = max(1, int(round(shape.width / scale)))
    height = max(1, int(round(shape.height / scale)))
    return Result([XLD(width, height, contours)], [])


@operator(
    "形狀匹配",
    [],
    [],
    [P("ModelID", "形狀模型"), P("Row", "找到的列"), P("Column", "找到的行"), P("Angle", "找到的角度")],
    [],
    "畫出匹配輪廓",
    "把模型的次像素輪廓剛性變換到每個匹配位置，並在輪廓旁標出角度到小數三位。每個實例使用 16 色循環中的下一個顏色。",
    "dev_display_shape_matching_results (ModelID, Row, Column, Angle)",
)
def dev_display_shape_matching_results(model, row, column, angle, *, ctx):
    from .ncc_model import format_degrees
    from .shape_model import color_for_match, transform_contours

    shape = _expect_model(model)
    rows, cols = _pair(row, column)
    angles = [float(item) for item in as_list(angle)]
    if len(angles) == 1 and len(rows) > 1:
        angles = angles * len(rows)
    if not (len(rows) == len(cols) == len(angles)):
        raise ValueError("Row、Column、Angle 的長度要相同")
    contours = []
    colors = []
    ctx.display.drop_provisional()
    for index, (match_row, match_col, match_angle) in enumerate(zip(rows, cols, angles)):
        color = color_for_match(index)
        for chain in transform_contours(shape, match_row, match_col, match_angle):
            contours.append(chain)
            colors.append(color)
        ctx.display.overlays.append(
            {
                "kind": "text",
                "text": format_degrees(match_angle),
                "row": match_row - 18,
                "col": match_col,
                "color": color,
            }
        )
    height, width = ctx.canvas_size()
    if contours:
        ctx.display.overlays.append(
            {
                "kind": "xld",
                "xld": XLD(width, height, contours),
                "color": colors[0],
                "colors": colors,
                "width": max(ctx.display.line_width, 2),
                "provisional": False,
            }
        )
        ctx.ref_shape = (height, width)
    return Result([], [])


@operator(
    "NCC定位",
    [P("Template", "已裁好的教導圖，可用 domain")],
    [],
    [
        P("NumLevels", "金字塔層數，0 為自動"),
        P("AngleStart", "起始角，弳度"),
        P("AngleExtent", "角度範圍，弳度"),
        P("AngleStep", "角度步進，0 為自動"),
        P("Metric", "use_polarity 或 ignore_global_polarity"),
    ],
    [P("ModelID", "NCC 模型")],
    "建立 NCC 模型",
    "灰階正規相關的教導。原點是 domain 重心；沒有 domain 時用影像中心。",
    "create_ncc_model (Template, 0, rad(-180), rad(360), 0, 'use_polarity', ModelID)",
)
def create_ncc_model(template, num_levels, angle_start, angle_extent, angle_step, metric, *, ctx):
    from .ncc_model import create_ncc_model as build_model

    image = expect_image(template, "Template")
    kind = text(metric, "Metric").lower()
    if kind not in ("use_polarity", "ignore_global_polarity"):
        raise ValueError("Metric 只能是 use_polarity 或 ignore_global_polarity")
    model = build_model(image, whole(num_levels, "NumLevels"), _optional_number(angle_step, "AngleStep") or 0, kind)
    model.angle_start = scalar(angle_start, "AngleStart")
    model.angle_extent = scalar(angle_extent, "AngleExtent")
    return Result([], [model])


@operator(
    "NCC定位",
    [P("Image", "要教導的影像")],
    [],
    [
        P("Row1", "教導框上邊界"),
        P("Column1", "教導框左邊界"),
        P("Row2", "教導框下邊界"),
        P("Column2", "教導框右邊界"),
        P("NumLevels", "金字塔層數，0 為自動"),
        P("AngleStart", "起始角，弳度"),
        P("AngleExtent", "角度範圍，弳度"),
        P("AngleStep", "角度步進，0 為自動"),
        P("Metric", "use_polarity 或 ignore_global_polarity"),
    ],
    [P("ModelID", "NCC 模型")],
    "教導 NCC 模型",
    "用矩形框出要記的灰階圖案。模型原點是教導框中心。",
    "teach_ncc_model (Image, 15, 15, 164, 164, 0, rad(-180), rad(360), 0, 'use_polarity', ModelID)",
)
def teach_ncc_model(
    image,
    row1,
    column1,
    row2,
    column2,
    num_levels,
    angle_start,
    angle_extent,
    angle_step,
    metric,
    *,
    ctx,
):
    taught = _teach_image(image, row1, column1, row2, column2)
    return create_ncc_model(taught, num_levels, angle_start, angle_extent, angle_step, metric, ctx=ctx)


@operator(
    "NCC定位",
    [P("Image", "要搜尋的影像")],
    [],
    [
        P("ModelID", "NCC 模型"),
        P("AngleStart", "起始角，弳度"),
        P("AngleExtent", "搜尋角度範圍，弳度"),
        P("MinScore", "最低分數，0 到 1"),
        P("NumMatches", "最多幾個，0 表示全部"),
        P("MaxOverlap", "允許重疊比例，0 到 1"),
        P("SubPixel", "none 或 interpolation"),
        P("NumLevels", "使用層數，0 表示模型全部層"),
    ],
    [P("Row", "中心列"), P("Column", "中心行"), P("Angle", "角度，弳度"), P("Score", "分數")],
    "NCC 定位",
    "金字塔正規相關搜尋後，對分數曲面做拋物線內插，得到次像素的中心與角度。",
    "find_ncc_model (Image, ModelID, rad(-180), rad(360), 0.7, 0, 0.4, 'interpolation', 0, Row, Column, Angle, Score)",
)
def find_ncc_model(
    image,
    model,
    angle_start,
    angle_extent,
    min_score,
    num_matches,
    max_overlap,
    subpixel,
    num_levels,
    *,
    ctx,
):
    from .ncc_model import NccModel
    from .ncc_model import find_ncc_model as search

    picture = expect_image(image, "Image")
    if not isinstance(model, NccModel):
        raise TypeError("ModelID 必須是 teach_ncc_model 或 create_ncc_model 建立的 NCC 模型")
    mode = text(subpixel, "SubPixel").lower()
    if mode not in ("none", "interpolation", "least_squares"):
        raise ValueError("SubPixel 只能是 none、interpolation 或 least_squares")
    rows, cols, angles, scores = search(
        picture,
        model,
        scalar(angle_start, "AngleStart"),
        scalar(angle_extent, "AngleExtent"),
        scalar(min_score, "MinScore"),
        whole(num_matches, "NumMatches"),
        scalar(max_overlap, "MaxOverlap"),
        mode != "none",
        whole(num_levels, "NumLevels"),
    )
    return Result([], [rows, cols, angles, scores])


@operator(
    "NCC定位",
    [],
    [],
    [P("ModelID", "NCC 模型"), P("Row", "中心列"), P("Column", "中心行"), P("Angle", "角度")],
    [],
    "畫出 NCC 定位",
    "只畫中心十字、方向箭頭，以及箭頭旁的角度。每個實例使用 16 色循環中的下一個顏色。角度 0 指向右方，逆時針為正，顯示到小數三位。",
    "dev_display_ncc_matching_results (ModelID, Row, Column, Angle)",
)
def dev_display_ncc_matching_results(model, row, column, angle, *, ctx):
    from .ncc_model import NccModel, angle_label, color_for_match, marker_contours

    if not isinstance(model, NccModel):
        raise TypeError("ModelID 必須是 teach_ncc_model 或 create_ncc_model 建立的 NCC 模型")
    rows, cols = _pair(row, column)
    angles = [float(item) for item in as_list(angle)]
    if len(angles) == 1 and len(rows) > 1:
        angles = angles * len(rows)
    if not (len(rows) == len(cols) == len(angles)):
        raise ValueError("Row、Column、Angle 的長度要相同")
    length = max(28.0, min(model.width, model.height) * 0.55)
    contours = []
    colors = []
    ctx.display.drop_provisional()
    for index, (match_row, match_col, match_angle) in enumerate(zip(rows, cols, angles)):
        color = color_for_match(index)
        for chain in marker_contours(match_row, match_col, match_angle, length):
            contours.append(chain)
            colors.append(color)
        label, label_row, label_col = angle_label(match_row, match_col, match_angle, length)
        ctx.display.overlays.append(
            {
                "kind": "text",
                "text": label,
                "row": label_row,
                "col": label_col,
                "color": color,
            }
        )
    height, width = ctx.canvas_size()
    if contours:
        ctx.display.overlays.append(
            {
                "kind": "xld",
                "xld": XLD(width, height, contours),
                "color": colors[0],
                "colors": colors,
                "width": max(ctx.display.line_width, 2),
                "provisional": False,
            }
        )
        ctx.ref_shape = (height, width)
    return Result([], [])


@operator(
    "顯示",
    [],
    [],
    [P("NumColors", "顏色數，最多 16")],
    [],
    "用多色顯示輪廓或區域",
    "之後 dev_display 的每一條輪廓、每一個區域物件會輪流使用這組顏色。單一顏色請用 dev_set_color。形狀匹配結果固定使用 16 色。",
    "dev_set_colored (16)",
)
def dev_set_colored(num_colors, *, ctx):
    count = whole(num_colors, "NumColors")
    ctx.display.colored = 0 if count <= 1 else min(count, 16)
    return Result([], [])


GROUP_ORDER = ["檔案", "灰階與濾波", "分割", "形態學", "區域", "特徵", "輪廓", "形狀匹配", "NCC定位", "顯示", "Tuple"]


def list_operators() -> list[dict]:
    def sort_key(op: Operator):
        try:
            group_index = GROUP_ORDER.index(op.group)
        except ValueError:
            group_index = 99
        return (group_index, op.order)

    return [op.to_dict() for op in sorted(REGISTRY.values(), key=sort_key)]
