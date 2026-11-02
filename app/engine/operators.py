"""Vision operators with HDevelop call order.

Iconic inputs, iconic outputs, control inputs, control outputs.
Add a new one with @operator; the name of the function is the name used in programs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from . import gauge
from .objects import Image, Region, Result, StopSignal, XLD, coerce_iconic
from .ocr import describe_catalog, method_choices, read_text
from .render import normalize_color, read_cv_image, to_uint8, write_cv_image

REGISTRY: dict[str, "Operator"] = {}
_ORDER = 0
# 程式裡的呼叫是 Vizion.算子名 (...)。C# 以後用 using Vizion.*，名稱對得上這一個前綴。
CALL_PREFIX = "Vizion."


def operator_bare(name: str) -> str:
    if name.lower().startswith(CALL_PREFIX.lower()):
        return name.split(".", 1)[1]
    return name


def _prefix_example(bare: str, example: str) -> str:
    public = CALL_PREFIX + bare
    if example.startswith(public):
        return example
    head = example.lstrip()
    if not head.startswith(bare):
        return example
    rest = head[len(bare):]
    if rest[:1].isspace() or rest.startswith("("):
        return example[: len(example) - len(head)] + public + rest
    return example


@dataclass
class Param:
    name: str
    desc: str
    default: str | None = None
    choices: list[str] | None = None


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
    note: str = ""

    def prototype(self) -> str:
        def join(params: list[Param]) -> str:
            return ", ".join(item.name for item in params)

        return (
            f"{self.name}({join(self.iconic_in)} : {join(self.iconic_out)} : "
            f"{join(self.control_in)} : {join(self.control_out)})"
        )

    def to_dict(self) -> dict:
        def pack(params: list[Param], role: str):
            return [
                {
                    "name": item.name,
                    "role": role,
                    "desc": item.desc,
                    "default": item.default,
                    "choices": item.choices or [],
                }
                for item in params
            ]

        return {
            "name": self.name,
            "group": self.group,
            "short": self.short,
            "detail": self.detail,
            "purpose": self.short,
            "principle": self.principle(),
            "setup": self.setup_text(),
            "prototype": self.prototype(),
            "example": self.example,
            "note": self.note,
            "order": self.order,
            "params": pack(self.iconic_in, "圖像輸入")
            + pack(self.iconic_out, "圖像輸出")
            + pack(self.control_in, "控制輸入")
            + pack(self.control_out, "控制輸出"),
        }

    def principle(self) -> str:
        head = GROUP_PRINCIPLE.get(self.group, "")
        body = self.detail.strip()
        if head and body:
            return head + "\n" + body
        return head or body

    def setup_text(self) -> str:
        blocks = (
            ("圖像輸入", self.iconic_in, "填已經有的影像、區域或輪廓變數名稱。"),
            ("圖像輸出", self.iconic_out, "填新變數名稱，執行後結果放在這裡。"),
            ("控制輸入", self.control_in, "可寫數字、變數或算式。字串要加引號。"),
            ("控制輸出", self.control_out, "填新變數名稱，數值或字串會放在這裡。"),
        )
        filled = [(title, params, intro) for title, params, intro in blocks if params]
        if not filled:
            return "這個算子沒有參數，照範例呼叫即可。"
        lines = []
        for title, params, intro in filled:
            lines.append(f"{title}：{intro}")
            for item in params:
                line = f"{item.name}：{item.desc}"
                if item.default is not None:
                    line += f"。沒填時用 {item.default}"
                if item.choices:
                    shown = "、".join(choice.strip("'\"") for choice in item.choices[:8])
                    if len(item.choices) > 8:
                        shown += f" 等 {len(item.choices)} 項"
                    line += f"。可選 {shown}"
                lines.append(line)
        return "\n".join(lines)


def P(name: str, desc: str, default: str | None = None, choices: list[str] | None = None) -> Param:
    return Param(name, desc, default, choices)


def operator(group, iconic_in, iconic_out, control_in, control_out, short, detail, example, note=""):
    def deco(fn):
        global _ORDER
        _ORDER += 1
        op = Operator(
            CALL_PREFIX + fn.__name__,
            group,
            iconic_in,
            iconic_out,
            control_in,
            control_out,
            short,
            detail,
            _prefix_example(fn.__name__, example),
            fn,
            _ORDER,
            note.strip(),
        )
        REGISTRY[fn.__name__] = op
        return fn

    return deco


def expect_image(value, name: str) -> Image:
    if not isinstance(value, Image):
        raise TypeError(f"{name} 必須是影像")
    return value


def expect_region(value, name: str) -> Region:
    if isinstance(value, list):
        value = coerce_iconic(value)
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
    ctx.threshold_region = None


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
    "row1", "column1", "row2", "column2", "connect_num",
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

ODD_SIZES = ["3", "5", "7", "9", "11", "13", "15"]
REGION_FEATURE_CHOICES = ["'" + name + "'" for name in REGION_FEATURES]
XLD_FEATURE_CHOICES = ["'" + name + "'" for name in XLD_FEATURES]
GRAY_FEATURE_CHOICES = ["'" + name + "'" for name in GRAY_FEATURES]
COLOR_CHOICES = [
    "'black'", "'white'", "'red'", "'green'", "'blue'", "'yellow'",
    "'cyan'", "'magenta'", "'orange'", "'coral'", "'gold'", "'pink'", "'gray'",
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


def equivalent_ellipse(moments: dict) -> tuple[float, float, float]:
    """Radii and major-axis angle of the equal-moment ellipse. Not a contour fit."""
    area = float(moments["m00"])
    if area <= 0:
        return 0.0, 0.0, 0.0
    mu20 = float(moments["mu20"])
    mu02 = float(moments["mu02"])
    mu11 = float(moments["mu11"])
    spread = math.hypot(mu20 - mu02, 2.0 * mu11)
    major = max(0.0, (mu20 + mu02 + spread) / area)
    minor = max(0.0, (mu20 + mu02 - spread) / area)
    if mu20 + mu02 < 1e-6:
        phi = 0.0
    else:
        phi = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
    return math.sqrt(2.0 * major), math.sqrt(2.0 * minor), phi


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
        box_x, box_y, width, height = cv2.boundingRect(mask)
        connect_num = float(cv2.connectedComponents((mask > 0).astype(np.uint8))[0] - 1)
    else:
        box_x = box_y = width = height = 0
        connect_num = 0.0
    contours, _hier = _find_contours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0] if contours else None
    peri = float(cv2.arcLength(contour, True)) if contour is not None else 0.0
    circularity = 0.0 if peri <= 0 or area <= 0 else float(min(1.0, (4 * math.pi * area) / (peri * peri)))
    compactness = 0.0 if area <= 0 else float((peri * peri) / (4 * math.pi * area))
    ra, rb, phi = equivalent_ellipse(moments)
    rect_len1 = rect_len2 = rect_phi = 0.0
    rect_area = 0.0
    if contour is not None and len(contour) >= 3:
        _center, (rw, rh), angle = cv2.minAreaRect(contour)
        rect_area = float(rw * rh)
        rect_len1 = float(max(rw, rh) / 2)
        rect_len2 = float(min(rw, rh) / 2)
        rect_phi = float(math.radians(angle))
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
        "row1": float(box_y),
        "column1": float(box_x),
        "row2": float(box_y + height - 1) if height else 0.0,
        "column2": float(box_x + width - 1) if width else 0.0,
        "connect_num": connect_num,
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
    name = str(feature).strip().lower().strip("'\"")
    name = {
        "center_row": "row",
        "center_column": "column",
        "center_col": "column",
    }.get(name, name)
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
    "Vizion.read_image (Image, 'samples/parts.png')",
)
def read_image(filename, *, ctx):
    path = ctx.resolve_read_path(text(filename, "FileName"))
    data = read_cv_image(path)
    if data is None:
        raise RuntimeError(f"無法讀取影像 {path}")
    data = to_uint8(data)
    if data.ndim == 3 and data.shape[2] == 4:
        data = cv2.cvtColor(data, cv2.COLOR_BGRA2BGR)
    ctx.image_stamp = int(getattr(ctx, "image_stamp", 0)) + 1
    ctx.threshold_gray = None
    ctx.threshold_domain = None
    ctx.threshold_mask = None
    ctx.threshold_region = None
    ctx.histogram = None
    ctx.preview_region = None
    return Result([Image(data)], [])


@operator(
    "檔案",
    [P("Image", "要儲存的影像")],
    [],
    [P("FileName", "輸出路徑")],
    [],
    "儲存影像",
    "把影像寫到磁碟。相對路徑以專案目錄為起點。這版不使用 HALCON 的 Format 與 FillColor 參數。",
    "Vizion.write_image (Image, 'workspace/out.png')",
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
    "Vizion.rgb1_to_gray (Image, Gray)",
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
    "Vizion.invert_image (Image, Inverted)",
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
    "Vizion.scale_image (Image, Scaled, 1.2, 0)",
)
def scale_image(image, mult, add, *, ctx):
    image = expect_image(image, "Image")
    out = image.data.astype(np.float32) * scalar(mult, "Mult") + scalar(add, "Add")
    return image_out(image, np.clip(out, 0, 255).astype(np.uint8))


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageGauss", "平滑後影像")],
    [P("Size", "奇數核大小", choices=ODD_SIZES)],
    [],
    "高斯平滑",
    "Size 必須是大於等於 3 的奇數。",
    "Vizion.gauss_image (Image, Smooth, 5)",
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
    "Vizion.median_image (Image, Median, 2)",
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
    "Vizion.mean_image (Image, MeanImage, 5, 5)",
)
def mean_image(image, mask_width, mask_height, *, ctx):
    image = expect_image(image, "Image")
    width = max(1, whole(mask_width, "MaskWidth"))
    height = max(1, whole(mask_height, "MaskHeight"))
    return image_out(image, cv2.blur(image.data, (width, height)))


def _box_mean(values: np.ndarray, radius: int) -> np.ndarray:
    window = radius * 2 + 1
    return cv2.blur(values, (window, window))


def _guided_channels(src: np.ndarray, guide: np.ndarray, radius: int, eps: float) -> np.ndarray:
    mean_g = _box_mean(guide, radius)
    mean_s = _box_mean(src, radius)
    variance = np.maximum(_box_mean(guide * guide, radius) - mean_g * mean_g, 0.0)
    if src.ndim == 2:
        covariance = _box_mean(guide * src, radius) - mean_g * mean_s
        coeff = covariance / (variance + eps)
        offset = mean_s - coeff * mean_g
        return _box_mean(coeff, radius) * guide + _box_mean(offset, radius)
    guide_c = guide[:, :, None]
    mean_g_c = mean_g[:, :, None]
    covariance = _box_mean(guide_c * src, radius) - mean_g_c * mean_s
    coeff = covariance / (variance[:, :, None] + eps)
    offset = mean_s - coeff * mean_g_c
    return _box_mean(coeff, radius) * guide_c + _box_mean(offset, radius)


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像，同時當引導圖")],
    [P("ImageGuided", "保留邊緣的平滑結果")],
    [P("Radius", "半徑，像素", "8"), P("Eps", "平滑強度，灰階平方", "800")],
    [],
    "引導濾波",
    "He、Sun、Tang 2010：用影像自己當引導。平坦處的係數接近 0，輸出變成局部平均；階梯邊緣的變異大，係數接近 1，兩邊的灰階不會被拉到中間。Eps 是加在變異上的常數，灰階 0 到 255 時常用幾百到幾千。彩色影像用亮度當引導，三個通道一起濾。",
    "Vizion.guided_filter (Image, Filtered, 8, 800)",
    "新算子。依 He、Sun、Tang 2010 做邊緣保留平滑。不是既有的均值或高斯。",
)
def guided_filter(image, radius, eps, *, ctx):
    image = expect_image(image, "Image")
    span = whole(radius, "Radius")
    regular = scalar(eps, "Eps")
    if span < 1:
        raise ValueError("Radius 至少要 1")
    if regular <= 0:
        raise ValueError("Eps 要大於 0")
    guide = image.gray().astype(np.float32)
    src = image.data.astype(np.float32)
    out = np.clip(np.round(_guided_channels(src, guide, span, regular)), 0, 255).astype(np.uint8)
    return image_out(image, out)


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageEmphasize", "強化後影像")],
    [P("MaskWidth", "遮罩寬"), P("MaskHeight", "遮罩高"), P("Factor", "強化係數")],
    [],
    "強化邊緣對比",
    "輸出 = 原圖 + (原圖 − 局部平均) × Factor。",
    "Vizion.emphasize (Image, Sharp, 7, 7, 1)",
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
    "Vizion.equ_histo_image (Image, Equ)",
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
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageClahe", "局部拉開對比的影像")],
    [P("ClipLimit", "對比上限，愈大愈用力", "8"), P("Tiles", "每邊切幾塊", "8")],
    [],
    "局部直方圖",
    "Zuiderveld 1994：把影像切成格子，每一格各自做直方圖等化，再把相鄰格子的結果混起來。暗角和亮區裡的小差距會一起被拉開。ClipLimit 限制單一灰階被放大的程度，避免雜訊被拉得太狠。Tiles 是每邊的格數。彩色影像只處理亮度。",
    "Vizion.emphasize_clahe (Image, Emphasized, 8, 8)",
    "新算子。依 Zuiderveld 1994 把每一小格的對比拉開。不是整張圖一起等化。",
)
def emphasize_clahe(image, clip_limit, tiles, *, ctx):
    image = expect_image(image, "Image")
    clip = scalar(clip_limit, "ClipLimit")
    count = whole(tiles, "Tiles")
    if clip <= 0:
        raise ValueError("ClipLimit 要大於 0")
    if count < 2:
        raise ValueError("Tiles 至少要 2")
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(count, count))
    if image.channels == 1:
        out = clahe.apply(image.data)
    else:
        ycrcb = cv2.cvtColor(image.data, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
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
    "Vizion.threshold (Image, Region, 128, 255)",
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
    [
        P("Method", "max_separability、smooth_histo、triangle、local_mean、local_gauss", choices=["'max_separability'", "'smooth_histo'", "'triangle'", "'local_mean'", "'local_gauss'"]),
        P("LightDark", "light 或 dark", choices=["'light'", "'dark'"]),
    ],
    [P("UsedThreshold", "實際使用的門檻")],
    "自動門檻",
    "method 可以是 max_separability、smooth_histo、triangle、local_mean、local_gauss。light 保留亮處，dark 保留暗處。前三種會在統計圖標出門檻，局部方法用鄰域比較。",
    "Vizion.binary_threshold (Image, Region, 'max_separability', 'light', UsedThreshold)",
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
    [P("Region", "比局部門檻暗的像素")],
    [
        P("MaskSize", "局部視窗，奇數", "15"),
        P("K", "對比係數，愈大愈挑細的暗處", "0.5"),
    ],
    [],
    "Sauvola 局部門檻",
    "Sauvola 與 Pietikäinen 2000：門檻是局部平均乘上 1 加 K 乘（局部標準差除以 128 再減 1）。比這個門檻暗的像素留下。光照斜坡上的細暗線會留下，一大塊均勻的暗區內部不會，因為那裡的標準差很小。K 常用 0.2 到 0.5。彩色影像先轉灰階。有 domain 時只處理 domain 裡面的像素。",
    "Vizion.sauvola (Image, Region, 15, 0.5)",
    "新算子。依 Sauvola 與 Pietikäinen 2000 用局部平均和標準差分黑白。不是固定門檻。",
)
def sauvola(image, mask_size, k, *, ctx):
    image = expect_image(image, "Image")
    size = whole(mask_size, "MaskSize")
    factor = scalar(k, "K")
    if size < 3 or size % 2 == 0:
        raise ValueError("MaskSize 要是 3 以上的奇數")
    if not 0 < factor <= 1:
        raise ValueError("K 要大於 0，而且不大於 1")
    gray = image.gray().astype(np.float32)
    mean = cv2.blur(gray, (size, size))
    mean_sq = cv2.blur(gray * gray, (size, size))
    deviation = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    level = mean * (1.0 + factor * (deviation / 128.0 - 1.0))
    mask = np.where(gray < level, np.uint8(255), np.uint8(0))
    if image.domain is not None:
        mask = cv2.bitwise_and(mask, image.domain)
    return region_out(image.width, image.height, mask)


@operator(
    "分割",
    [P("Image", "灰階或彩色影像")],
    [P("Region", "二值化區域")],
    [P("Threshold", "門檻，不含等於"), P("Method", "binary 或 binary_inv", choices=["'binary'", "'binary_inv'"])],
    [],
    "OpenCV 二值化",
    "對應 cv2.threshold。binary 留下大於 Threshold 的像素，binary_inv 留下小於等於 Threshold 的像素。等於門檻的像素在 binary 不會留下。二值化結果會保留 OpenCV 的輪廓鏈表：外輪廓用 next/prev 串起來，孔洞用 child 掛在外輪廓下面。",
    "Vizion.opencv_threshold (Image, Region, 128, 'binary')",
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


def subpix_parts(gray: np.ndarray, level: float, domain: np.ndarray | None = None) -> tuple[np.ndarray, list[tuple[np.ndarray, list]]]:
    """八連通拆開，每個連通物件自己的次像素輪廓。和 connection 距離 1 同一套。"""
    mask = np.where(gray.astype(np.float32) >= float(level), 255, 0).astype(np.uint8)
    if domain is not None:
        mask = cv2.bitwise_and(mask, domain)
    if int(cv2.countNonZero(mask)) == 0:
        return mask, []
    parts = []
    for component in halcon_components(mask):
        lines = [line for line in subpixel_boundary(component, gray, float(level)) if len(line) >= 3]
        if lines:
            parts.append((component, lines))
    return mask, parts


def subpix_contours(gray: np.ndarray, level: float, domain: np.ndarray | None = None) -> list[list[list[float]]]:
    _mask, parts = subpix_parts(gray, level, domain)
    return [line for _component, lines in parts for line in lines]


@operator(
    "分割",
    [P("Image", "灰階或彩色影像")],
    [P("Border", "次像素輪廓")],
    [P("Threshold", "灰階交界")],
    [],
    "次像素二值化",
    "沿著灰階等於 Threshold 的位置取出次像素輪廓，亮處留在輪廓內。結果是 XLD。會自動用八連通拆開，每個物件一種顏色，點一個只框那一個，和 connection 距離 1 一樣。門檻若是數字，可以在統計圖拖曳，也可以用次像素視窗調，那個視窗不會擋住主視窗。",
    "Vizion.threshold_sub_pix (Image, Border, 128)",
)
def threshold_sub_pix(image, threshold, *, ctx):
    image = expect_image(image, "Image")
    level = scalar(threshold, "Threshold")
    if level < 0 or level > 255:
        raise ValueError("Threshold 要在 0 到 255")
    gray = image.gray()
    mask, parts = subpix_parts(gray, level, image.domain)
    lines = []
    groups = []
    for index, (_component, part) in enumerate(parts):
        for line in part:
            lines.append(line)
            groups.append(index)
    lo = min(255, int(round(level)))
    attach_histogram(ctx, image, lo, 255, mask, "subpix", threshold=level)
    ctx.threshold_region = Region(image.width, image.height, [component for component, _part in parts], multi_color=len(parts) > 1)
    ctx.preview_region = ctx.threshold_region
    return Result([XLD(image.width, image.height, lines, multi_color=True, groups=groups)], [])


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
    "Vizion.connection (Region, ConnectedRegions, 1)",
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
    [P("Connectivity", "4 或 8", choices=["8", "4"]), P("Distance", "連通距離，像素。1 不額外合併", "1")],
    [],
    "OpenCV 連通",
    "對應 cv2.connectedComponents。4 只算上下左右，8 連斜角也算。Distance 再把最近前景像素落在這個距離內的物件合併，1 表示不額外合併。每個物件保留自己的 OpenCV 輪廓鏈表。顯示時用 16 色。",
    "Vizion.opencv_connection (Region, ConnectedRegions, 8, 1)",
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
        P("Features", "單一特徵。多個特徵請直接寫 tuple", choices=REGION_FEATURE_CHOICES),
        P("Operation", "and 或 or", choices=["'and'", "'or'"]),
        P("Min", "下限，含"),
        P("Max", "上限，含"),
    ],
    [],
    "用形狀特徵篩選",
    "可用特徵和圖形視窗的勾選清單相同，包含 area、circularity、ra、rb、holes_num、euler_number 等。多個特徵時 Min/Max 也要是等長 tuple。",
    "Vizion.select_shape (Regions, SelectedRegions, 'area', 'and', 150, 99999)",
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
    "索引從 1 開始。區域取出物件，輪廓取出其中一條。可以傳 tuple 一次取多條。",
    "Vizion.select_obj (Objects, ObjectSelected, 1)",
)
def select_obj(obj, index, *, ctx):
    if isinstance(obj, XLD):
        picked = []
        for item in as_list(index):
            pos = int(round(float(item)))
            if pos < 1 or pos > len(obj.contours):
                raise IndexError(f"輪廓索引 {pos} 超出範圍 1..{len(obj.contours)}")
            picked.append(obj.contours[pos - 1])
        return Result([XLD(obj.width, obj.height, picked, multi_color=len(picked) > 1)], [])
    region = expect_region(obj, "Objects")
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


def _same_links(region: Region):
    if region.links is not None and len(region.links) == region.count():
        return list(region.links)
    return None


@operator(
    "區域",
    [P("Objects1", "第一組"), P("Objects2", "第二組")],
    [P("ObjectsConcat", "接在一起的物件")],
    [],
    [],
    "把兩組物件接起來",
    "兩邊都要是區域，或都要是輪廓。區域的寬高必須相同。原來的物件不會被合併成一塊。",
    "Vizion.concat_obj (Objects1, Objects2, ObjectsConcat)",
)
def concat_obj(first, second, *, ctx):
    if isinstance(first, Region) and isinstance(second, Region):
        if (first.width, first.height) != (second.width, second.height):
            raise ValueError("兩個區域的尺寸不同")
        links1 = _same_links(first)
        links2 = _same_links(second)
        links = (links1 + links2) if links1 is not None and links2 is not None else None
        objects = list(first.objects) + list(second.objects)
        return Result([Region(first.width, first.height, objects, multi_color=len(objects) > 1, links=links)], [])
    if isinstance(first, XLD) and isinstance(second, XLD):
        contours = list(first.contours) + list(second.contours)
        return Result([XLD(max(first.width, second.width), max(first.height, second.height), contours, multi_color=len(contours) > 1)], [])
    raise ValueError("concat_obj 的兩邊要同是區域，或同是輪廓")


@operator(
    "區域",
    [P("Objects", "物件集合")],
    [P("CopiedObjects", "複製出來的物件")],
    [P("Index", "從 1 開始"), P("NumObj", "要複製幾個")],
    [],
    "複製連續的幾個物件",
    "索引從 1 開始。區域和輪廓都可以。超出範圍會失敗。",
    "Vizion.copy_obj (Objects, CopiedObjects, 1, 1)",
)
def copy_obj(obj, index, number, *, ctx):
    start = whole(index, "Index")
    count = whole(number, "NumObj")
    if start < 1 or count < 1:
        raise ValueError("Index 從 1 開始，NumObj 至少要 1")
    if isinstance(obj, XLD):
        end = start - 1 + count
        if end > len(obj.contours):
            raise IndexError(f"輪廓索引超出範圍 1..{len(obj.contours)}")
        picked = obj.contours[start - 1 : end]
        return Result([XLD(obj.width, obj.height, picked, multi_color=len(picked) > 1)], [])
    region = expect_region(obj, "Objects")
    end = start - 1 + count
    if end > region.count():
        raise IndexError(f"物件索引超出範圍 1..{region.count()}")
    links = _same_links(region)
    picked_links = links[start - 1 : end] if links is not None else None
    picked = region.objects[start - 1 : end]
    return Result([Region(region.width, region.height, picked, multi_color=region.multi_color, links=picked_links)], [])


def _object_positions(value, count: int, kind: str) -> list[int]:
    items = as_list(value)
    if not items:
        raise ValueError("Index 不能是空的")
    found = []
    for item in items:
        pos = int(round(float(item)))
        if pos < 1 or pos > count:
            raise IndexError(f"{kind}索引 {pos} 超出範圍 1..{count}")
        if pos not in found:
            found.append(pos)
    return found


def _splice_objects(source, start: int, remove: int, incoming):
    if isinstance(source, XLD):
        if not isinstance(incoming, XLD):
            raise ValueError("插入的物件也要是輪廓")
        contours = list(source.contours)
        contours[start : start + remove] = list(incoming.contours)
        return Result([XLD(source.width, source.height, contours, multi_color=len(contours) > 1)], [])
    region = expect_region(source, "Objects")
    added = expect_region(incoming, "ObjectsInsert")
    if (region.width, region.height) != (added.width, added.height):
        raise ValueError("兩個區域的尺寸不同")
    objects = list(region.objects)
    objects[start : start + remove] = list(added.objects)
    return Result([Region(region.width, region.height, objects, multi_color=len(objects) > 1)], [])


@operator(
    "區域",
    [P("Objects", "物件集合")],
    [P("ObjectsReduced", "拿掉之後的物件")],
    [P("Index", "從 1 開始，可以是 tuple")],
    [],
    "拿掉指定物件",
    "索引從 1 開始。其餘物件的順序不變。區域和輪廓都可以。",
    "Vizion.remove_obj (Objects, ObjectsReduced, 2)",
)
def remove_obj(obj, index, *, ctx):
    if isinstance(obj, XLD):
        positions = _object_positions(index, len(obj.contours), "輪廓")
        drop = set(positions)
        kept = [contour for pos, contour in enumerate(obj.contours, start=1) if pos not in drop]
        return Result([XLD(obj.width, obj.height, kept, multi_color=len(kept) > 1)], [])
    region = expect_region(obj, "Objects")
    positions = _object_positions(index, region.count(), "物件")
    drop = set(positions)
    kept = [mask for pos, mask in enumerate(region.objects, start=1) if pos not in drop]
    return Result([Region(region.width, region.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "區域",
    [P("Objects", "物件集合"), P("ObjectsReplace", "換上去的物件")],
    [P("ObjectsReplaced", "替換後的物件")],
    [P("Index", "從 1 開始")],
    [],
    "替換一個物件",
    "索引從 1 開始。那個位置換成新的物件，後面的順序跟著往後。區域和輪廓都可以。",
    "Vizion.replace_obj (Objects, ObjectsReplace, ObjectsReplaced, 2)",
)
def replace_obj(obj, incoming, index, *, ctx):
    kind = "輪廓" if isinstance(obj, XLD) else "物件"
    count = len(obj.contours) if isinstance(obj, XLD) else expect_region(obj, "Objects").count()
    pos = _object_positions(index, count, kind)
    if len(pos) != 1:
        raise ValueError("Index 只能是一個")
    return _splice_objects(obj, pos[0] - 1, 1, incoming)


@operator(
    "區域",
    [P("Objects", "物件集合"), P("ObjectsInsert", "要插入的物件")],
    [P("ObjectsExtended", "插入後的物件")],
    [P("Index", "從 1 開始，可以是物件數加 1")],
    [],
    "在指定位置插入物件",
    "索引從 1 開始。新物件放在這個位置，原來的往後移。索引等於物件數加 1 時接在最後。",
    "Vizion.insert_obj (Objects, ObjectsInsert, ObjectsExtended, 1)",
)
def insert_obj(obj, incoming, index, *, ctx):
    if isinstance(obj, XLD):
        count = len(obj.contours)
        kind = "輪廓"
    else:
        count = expect_region(obj, "Objects").count()
        kind = "物件"
    pos = whole(index, "Index")
    if pos < 1 or pos > count + 1:
        raise IndexError(f"{kind}索引 {pos} 超出範圍 1..{count + 1}")
    return _splice_objects(obj, pos - 1, 0, incoming)


@operator(
    "形態學",
    [P("Region", "輸入區域")],
    [P("RegionFillUp", "補滿孔洞後的區域")],
    [],
    [],
    "填滿孔洞",
    "填上區域內部的洞。結果是單一區域，要再拆開請呼叫 connection。",
    "Vizion.fill_up (Region, RegionFillUp)",
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
    "Vizion.opening_circle (Region, RegionOpening, 3.5)",
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
    "Vizion.closing_circle (Region, RegionClosing, 3.5)",
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
    "Vizion.erosion_circle (Region, RegionErosion, 2.5)",
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
    "Vizion.dilation_circle (Region, RegionDilation, 2.5)",
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
    "Vizion.erosion_rectangle1 (Region, RegionErosion, 5, 5)",
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
    "Vizion.dilation_rectangle1 (Region, RegionDilation, 5, 5)",
)
def dilation_rectangle1(region, width, height, *, ctx):
    return morph_rect(region, width, height, "dilate")


@operator(
    "形態學",
    [P("Region", "輸入區域")],
    [P("RegionRank", "排序濾波後的區域")],
    [P("Width", "視窗寬"), P("Height", "視窗高"), P("Rank", "視窗裡至少要幾個前景像素")],
    [],
    "區域的排序濾波",
    "視窗裡的前景像素達到 Rank 就留下。Rank 為 1 時和同樣奇數寬高的 dilation_rectangle1 相同。Rank 等於寬乘高時和 erosion_rectangle1 相同。作用在區域聯集上。",
    "Vizion.rank_region (Region, RegionRank, 3, 3, 1)",
)
def rank_region(region, width, height, rank, *, ctx):
    source = expect_region(region, "Region")
    wide = whole(width, "Width")
    high = whole(height, "Height")
    need = whole(rank, "Rank")
    if wide < 1 or high < 1:
        raise ValueError("Width 和 Height 至少要 1")
    if need < 1 or need > wide * high:
        raise ValueError("Rank 要在 1 到 Width 乘 Height")
    mask = (source.union_mask() > 0).astype(np.uint8)
    counts = cv2.boxFilter(mask, cv2.CV_32S, (wide, high), normalize=False, borderType=cv2.BORDER_CONSTANT)
    out = np.where(counts >= need, np.uint8(255), np.uint8(0))
    return region_out(source.width, source.height, out)


@operator(
    "形態學",
    [P("Region", "輸入區域")],
    [P("RegionOpening", "開運算結果")],
    [P("Width", "矩形寬"), P("Height", "矩形高")],
    [],
    "矩形開運算",
    "先侵蝕再膨脹。作用在區域聯集上，結果是單一區域。",
    "Vizion.opening_rectangle1 (Region, RegionOpening, 5, 5)",
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
    "Vizion.closing_rectangle1 (Region, RegionClosing, 5, 5)",
)
def closing_rectangle1(region, width, height, *, ctx):
    return morph_rect(region, width, height, "close")


@operator(
    "形態學",
    [P("Region", "輸入區域")],
    [P("RegionBorder", "邊界區域")],
    [P("BoundaryType", "inner 或 outer", choices=["'inner'", "'outer'"])],
    [],
    "區域邊界",
    "inner 是原區域減去侵蝕一圈，outer 是膨脹一圈再減去原區域。",
    "Vizion.boundary (Region, RegionBorder, 'inner')",
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
    "Vizion.difference (Region, Sub, RegionDifference)",
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
    "Vizion.intersection (Region1, Region2, RegionIntersection)",
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
    "Vizion.union2 (Region1, Region2, RegionUnion)",
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
    "Vizion.union1 (Region, RegionUnion)",
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
    "Vizion.complement (Region, RegionComplement)",
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
    "像素不會被裁掉。已經有定義域時會和這個區域取交集；change_domain 才是整段換掉。定義域以外之後不會被 threshold 使用，顯示時也會變暗。",
    "Vizion.reduce_domain (Image, Region, ImageReduced)",
)
def reduce_domain(image, region, *, ctx):
    image = expect_image(image, "Image")
    region = expect_region(region, "Region")
    if image.width != region.width or image.height != region.height:
        raise ValueError("影像和區域的尺寸不同")
    reduced = image.copy()
    mask = region.union_mask()
    if image.domain is not None:
        mask = cv2.bitwise_and(mask, image.domain)
    reduced.domain = mask
    return Result([reduced], [])


@operator(
    "區域",
    [P("Image", "輸入影像")],
    [P("Domain", "影像的 domain")],
    [],
    [],
    "取得 domain",
    "沒有做過 reduce_domain 時，domain 就是整張影像。",
    "Vizion.get_domain (Image, Domain)",
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
    "Vizion.crop_rectangle1 (Image, ImagePart, 0, 0, 100, 100)",
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
    "Vizion.gen_rectangle1 (Rectangle, 40, 40, 200, 300)",
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
    "Vizion.gen_circle (Circle, 120, 160, 40)",
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
    "Vizion.area_center (Regions, Area, Row, Column)",
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
    "Vizion.smallest_rectangle1 (Regions, Row1, Column1, Row2, Column2)",
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
    "Vizion.smallest_rectangle2 (Regions, Row, Column, Phi, Length1, Length2)",
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
    "Vizion.smallest_circle (Regions, Row, Column, Radius)",
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
    "Vizion.intensity (Regions, Image, Mean, Deviation)",
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


def _gray_feature(values: np.ndarray, name: str) -> float:
    feature = str(name).strip().lower().strip("'\"")
    if feature not in GRAY_FEATURES:
        known = ", ".join(GRAY_FEATURES)
        raise ValueError(f"不支援的灰階特徵 {name}。可用 {known}")
    if values.size == 0:
        return 0.0
    if feature == "gray_min":
        return float(values.min())
    if feature == "gray_max":
        return float(values.max())
    if feature == "gray_mean":
        return float(values.mean())
    if feature == "gray_deviation":
        return float(values.std())
    return float(np.median(values))


@operator(
    "特徵",
    [P("Regions", "物件"), P("Image", "灰階影像")],
    [],
    [P("Features", "灰階特徵，可以是 tuple", choices=GRAY_FEATURE_CHOICES)],
    [P("Value", "特徵值")],
    "讀取灰階特徵",
    "可選 gray_min、gray_max、gray_mean、gray_deviation、gray_median。單一特徵時依物件排列。多個特徵時先排完第一個特徵的全部物件。標準差和 intensity 一樣。彩色先轉成亮度。",
    "Vizion.gray_features (Regions, Image, 'gray_mean', Value)",
)
def gray_features(region, image, features, *, ctx):
    source = expect_region(region, "Regions")
    picture = expect_image(image, "Image")
    if (picture.width, picture.height) != (source.width, source.height):
        raise ValueError("影像和區域的尺寸不同")
    gray = picture.gray()
    names = as_list(features)
    if not names:
        raise ValueError("Features 不能是空的")
    values = []
    for feature in names:
        for mask in source.objects:
            values.append(_gray_feature(gray[mask > 0], feature))
    return Result([], [values])


@operator(
    "特徵",
    [P("Objects", "物件集合")],
    [],
    [],
    [P("Number", "物件數")],
    "物件數量",
    "區域回傳物件數，輪廓回傳輪廓數。結果是單一整數。",
    "Vizion.count_obj (Objects, Number)",
)
def count_obj(obj, *, ctx):
    if isinstance(obj, XLD):
        return Result([], [len(obj.contours)])
    region = expect_region(obj, "Objects")
    return Result([], [region.count()])


@operator(
    "特徵",
    [P("Regions", "物件集合")],
    [],
    [P("Features", "單一特徵。多個特徵請直接寫 tuple", choices=REGION_FEATURE_CHOICES)],
    [P("Value", "特徵值 tuple")],
    "讀取區域特徵",
    "單一特徵時，Value 依物件排列。多個特徵時，先排完第一個特徵的全部物件，再排下一個。",
    "Vizion.region_features (Regions, 'area', Value)",
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
    [P("FilterType", "sum_abs 或 sum_sqrt", choices=["'sum_abs'", "'sum_sqrt'"]), P("Size", "奇數核大小", choices=ODD_SIZES)],
    [],
    "Sobel 振幅",
    "輸出是 8-bit。超過 255 的振幅會被截斷，所以後面的 threshold 要用 0 到 255 的門檻。",
    "Vizion.sobel_amp (Image, EdgeAmplitude, 'sum_abs', 3)",
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


def _boundary_center_lines(mask: np.ndarray) -> list:
    """Outer contour on the edges between region pixels and the background."""
    foreground = np.asarray(mask) > 0
    if not np.any(foreground):
        return []
    padded = np.pad(foreground, 1)
    core = padded[1:-1, 1:-1]
    sides = (
        (core & ~padded[1:-1, :-2], (-0.5, 0.5, -0.5, -0.5)),
        (core & ~padded[:-2, 1:-1], (-0.5, -0.5, 0.5, -0.5)),
        (core & ~padded[1:-1, 2:], (0.5, -0.5, 0.5, 0.5)),
        (core & ~padded[2:, 1:-1], (0.5, 0.5, -0.5, 0.5)),
    )
    links: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for selected, (col0, row0, col1, row1) in sides:
        rows, cols = np.nonzero(selected)
        for row, col in zip(rows.tolist(), cols.tolist()):
            start = (int(round((col + col0) * 2)), int(round((row + row0) * 2)))
            end = (int(round((col + col1) * 2)), int(round((row + row1) * 2)))
            links.setdefault(start, []).append(end)
    lines = []
    while links:
        start = min(links)
        ends = links[start]
        nxt = min(ends)
        ends.remove(nxt)
        if not ends:
            del links[start]
        loop = [start, nxt]
        while nxt != start and nxt in links:
            here = nxt
            ends = links[here]
            nxt = min(ends)
            ends.remove(nxt)
            if not ends:
                del links[here]
            loop.append(nxt)
        if loop[-1] != start or len(loop) < 5:
            continue
        area = 0.0
        for left, right in zip(loop, loop[1:]):
            area += left[0] * right[1] - right[0] * left[1]
        if area <= 0:
            continue
        lines.append([[point[0] / 2, point[1] / 2] for point in loop])
    return lines


@operator(
    "輪廓",
    [P("Regions", "區域")],
    [P("Contours", "區域外輪廓")],
    [P("Mode", "border、border_holes 或 center", "'border'", ["'border'", "'border_holes'", "'center'"])],
    [],
    "區域轉成輪廓",
    "border 只要外輪廓，點在區域最外圈的整數像素上。border_holes 連孔洞一起。center 走在區域和外面之間的邊上，座標帶 .5，方塊會比整數外框再出去半個像素。center 不畫孔洞。",
    "Vizion.gen_contour_region_xld (Regions, Contours, 'border')",
)
def gen_contour_region_xld(region, mode, *, ctx):
    region = expect_region(region, "Regions")
    kind = text(mode, "Mode").lower()
    if kind not in ("border", "border_holes", "center"):
        raise ValueError("Mode 只能是 border、border_holes 或 center")
    polylines = []
    retrieval = cv2.RETR_CCOMP if kind == "border_holes" else cv2.RETR_EXTERNAL
    for mask in region.objects:
        if kind == "center":
            polylines.extend(_boundary_center_lines(mask))
        else:
            polylines.extend(_contour_lines(mask, retrieval))
    return Result([XLD(region.width, region.height, polylines, multi_color=len(polylines) > 1)], [])


def _contour_lines(mask: np.ndarray, retrieval=cv2.RETR_EXTERNAL):
    found = cv2.findContours(mask, retrieval, cv2.CHAIN_APPROX_NONE)
    contours = found[0] if len(found) == 2 else found[1]
    lines = []
    for contour in contours:
        if len(contour) < 2:
            continue
        lines.append([[float(point[0][0]), float(point[0][1])] for point in contour])
    return lines


def expect_xld(value, name: str) -> XLD:
    if isinstance(value, list):
        value = coerce_iconic(value)
    if not isinstance(value, XLD):
        raise TypeError(f"{name} 必須是輪廓")
    return value


def _points(contour) -> list[list[float]]:
    return [[float(point[0]), float(point[1])] for point in contour]


def _closed_poly(points, tol: float = 1.5) -> bool:
    if len(points) < 3:
        return False
    return math.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) < tol


def _unit(a, b) -> tuple[float, float]:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = math.hypot(dx, dy) or 1.0
    return dx / length, dy / length


def _angle_of(left, right) -> float:
    dot = max(-1.0, min(1.0, left[0] * right[0] + left[1] * right[1]))
    return math.acos(dot)


def _turn_angle(a, b, c) -> float:
    incoming = _unit(a, b)
    outgoing = _unit(b, c)
    return _angle_of(incoming, outgoing)


def _xld_box(xld: XLD) -> tuple[int, int]:
    width = int(xld.width or 0)
    height = int(xld.height or 0)
    for contour in xld.contours:
        for col, row in contour:
            width = max(width, int(math.ceil(float(col))) + 1)
            height = max(height, int(math.ceil(float(row))) + 1)
    return max(width, 1), max(height, 1)


def _pack_xld(width: int, height: int, contours: list) -> XLD:
    clean = [item for item in contours if len(item) >= 2]
    return XLD(width, height, clean, multi_color=len(clean) > 1)


def _fit_line(points: list) -> tuple[list[float], list[float]]:
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    if len(pts) < 2:
        raise ValueError("直線擬合至少要 2 個點")
    weights = np.ones(len(pts), np.float64)
    center = pts.mean(axis=0)
    _u, _s, vt = np.linalg.svd(pts - center, full_matrices=False)
    direction = vt[0]
    for _step in range(4):
        normal = np.array([-direction[1], direction[0]])
        distance = np.abs((pts - center) @ normal)
        typical = float(np.median(distance))
        if typical < 1e-8 and float(distance.max()) < 1e-6:
            break
        limit = 2.5 * typical if typical > 0.4 else 1.0
        weights = (distance <= limit).astype(np.float64)
        if float(weights.sum()) < 2:
            break
        center = (weights[:, None] * pts).sum(axis=0) / float(weights.sum())
        shifted = (pts - center) * np.sqrt(weights)[:, None]
        _u, _s, vt = np.linalg.svd(shifted, full_matrices=False)
        direction = vt[0]
    keep = weights > 0.05
    if int(np.count_nonzero(keep)) < 2:
        keep = np.ones(len(pts), dtype=bool)
    projection = (pts[keep] - center) @ direction
    start = center + direction * float(projection.min())
    end = center + direction * float(projection.max())
    return [float(start[0]), float(start[1])], [float(end[0]), float(end[1])]


def _pratt_circle(pts: np.ndarray) -> tuple[float, float, float]:
    count = len(pts)
    centroid = pts.mean(axis=0)
    shifted_col = pts[:, 0] - centroid[0]
    shifted_row = pts[:, 1] - centroid[1]
    square = shifted_col * shifted_col + shifted_row * shifted_row
    moment_col = float(np.dot(shifted_col, shifted_col) / count)
    moment_row = float(np.dot(shifted_row, shifted_row) / count)
    moment_cross = float(np.dot(shifted_col, shifted_row) / count)
    moment_col_sq = float(np.dot(shifted_col, square) / count)
    moment_row_sq = float(np.dot(shifted_row, square) / count)
    moment_sq = float(np.dot(square, square) / count)
    moment_sum = moment_col + moment_row
    covariance = moment_col * moment_row - moment_cross * moment_cross
    cubic = 4 * moment_sum
    quad = -3 * moment_sum * moment_sum - moment_sq
    linear = (
        moment_sq * moment_sum
        + 4 * covariance * moment_sum
        - moment_col_sq * moment_col_sq
        - moment_row_sq * moment_row_sq
        - moment_sum ** 3
    )
    constant = (
        moment_col_sq * moment_col_sq * moment_row
        + moment_row_sq * moment_row_sq * moment_col
        - moment_sq * covariance
        - 2 * moment_col_sq * moment_row_sq * moment_cross
        + moment_sum * moment_sum * covariance
    )
    root = 0.0
    value = 1e20
    for _step in range(20):
        previous = value
        value = constant + root * (linear + root * (quad + root * cubic))
        if abs(value) > abs(previous):
            root = 0.0
            break
        slope = linear + root * (2 * quad + 3 * root * cubic)
        if abs(slope) < 1e-18:
            break
        updated = root - value / slope
        if updated < 0:
            root = 0.0
            break
        if abs(updated) > 1e-12 and abs((updated - root) / updated) < 1e-12:
            root = updated
            break
        root = updated
    determinant = root * root - root * moment_sum + covariance
    if abs(determinant) < 1e-15:
        raise ValueError("這些點擬合不出圓")
    center_col = (moment_col_sq * (moment_row - root) - moment_row_sq * moment_cross) / determinant / 2
    center_row = (moment_row_sq * (moment_col - root) - moment_col_sq * moment_cross) / determinant / 2
    radius2 = center_col * center_col + center_row * center_row + moment_sum + 2 * root
    if radius2 <= 1e-8:
        raise ValueError("這些點擬合不出圓")
    return float(center_row + centroid[1]), float(center_col + centroid[0]), math.sqrt(radius2)


def _fit_circle(points: list) -> tuple[float, float, float]:
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    if len(pts) < 3:
        raise ValueError("圓擬合至少要 3 個點")
    row, col, radius = _pratt_circle(pts)
    keep = np.ones(len(pts), dtype=bool)
    for _step in range(4):
        distance = np.abs(np.hypot(pts[:, 0] - col, pts[:, 1] - row) - radius)
        typical = float(np.median(distance))
        if typical < 1e-8 and float(distance.max()) < 1e-6:
            break
        limit = 2.5 * typical if typical > 0.4 else 1.0
        keep = distance <= limit
        if int(np.count_nonzero(keep)) < 3:
            break
        row, col, radius = _pratt_circle(pts[keep])
    return row, col, radius


def _sample_circle(row: float, col: float, radius: float, count: int = 72) -> list[list[float]]:
    points = []
    for index in range(count + 1):
        angle = 2 * math.pi * index / count
        points.append([col + radius * math.cos(angle), row + radius * math.sin(angle)])
    return points


def _raw_ellipse(points: np.ndarray):
    try:
        (col, row), (width, height), angle = cv2.fitEllipse(points.astype(np.float32).reshape(-1, 1, 2))
    except cv2.error as exc:
        raise ValueError("這些點擬合不出橢圓") from exc
    return float(col), float(row), float(width), float(height), float(angle)


def _ellipse_ray_error(points: np.ndarray, ellipse) -> np.ndarray:
    center_col, center_row, width, height, angle = ellipse
    major = max(width / 2.0, 1e-6)
    minor = max(height / 2.0, 1e-6)
    radians = math.radians(angle)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    dx = points[:, 0] - center_col
    dy = points[:, 1] - center_row
    along = dx * cosine + dy * sine
    across = -dx * sine + dy * cosine
    radius = np.hypot(along, across)
    span = np.hypot(minor * along, major * across)
    on_ellipse = np.full(radius.shape, min(major, minor), dtype=np.float64)
    usable = span > 1e-9
    on_ellipse[usable] = (major * minor) * radius[usable] / span[usable]
    return np.abs(radius - on_ellipse)


def _robust_ellipse(points: np.ndarray):
    base = _raw_ellipse(points)
    if len(points) < 6:
        return base
    error = _ellipse_ray_error(points, base)
    if float(np.max(error)) <= 1.0:
        return base
    generator = np.random.default_rng(0)
    best = base
    best_error = error
    best_score = float(np.median(error))
    for _turn in range(48):
        index = generator.choice(len(points), 6, replace=False)
        try:
            trial = _raw_ellipse(points[index])
        except ValueError:
            continue
        trial_error = _ellipse_ray_error(points, trial)
        score = float(np.median(trial_error))
        if score < best_score:
            best = trial
            best_error = trial_error
            best_score = score
    limit = 2.5 * best_score if best_score > 0.4 else 1.0
    keep = best_error <= limit + 1e-9
    if int(np.count_nonzero(keep)) >= 5 and not bool(np.all(keep)):
        try:
            return _raw_ellipse(points[keep])
        except ValueError:
            return best
    return best


def _fit_ellipse(points: list) -> tuple[float, float, float, float, float]:
    cloud = np.asarray(points, np.float64).reshape(-1, 2)
    if len(cloud) < 5:
        raise ValueError("橢圓擬合至少要 5 個點")
    col, row, width, height, angle = _robust_ellipse(cloud)
    if width >= height:
        phi = math.radians(angle)
        radius1 = width / 2.0
        radius2 = height / 2.0
    else:
        phi = math.radians(angle + 90.0)
        radius1 = height / 2.0
        radius2 = width / 2.0
    return float(row), float(col), phi, radius1, radius2


def _sample_ellipse(row: float, col: float, phi: float, radius1: float, radius2: float, count: int = 72) -> list:
    points = []
    cosine = math.cos(phi)
    sine = math.sin(phi)
    for index in range(count + 1):
        angle = 2 * math.pi * index / count
        ex = radius1 * math.cos(angle)
        ey = radius2 * math.sin(angle)
        points.append([col + ex * cosine - ey * sine, row + ex * sine + ey * cosine])
    return points


def _sample_rectangle(row: float, col: float, phi: float, length1: float, length2: float) -> list:
    cosine = math.cos(phi)
    sine = math.sin(phi)
    corners = []
    for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1), (1, 1)):
        ex = sx * length1
        ey = sy * length2
        corners.append([col + ex * cosine - ey * sine, row + ex * sine + ey * cosine])
    return corners


def _junction(tail, head, max_dist: float, max_angle: float, collinear: bool) -> bool:
    dist = math.hypot(tail[-1][0] - head[0][0], tail[-1][1] - head[0][1])
    if dist > max_dist:
        return False
    if len(tail) < 2 or len(head) < 2:
        return True
    leave = _unit(tail[-2], tail[-1])
    enter = _unit(head[0], head[1])
    if _angle_of(leave, enter) > max_angle:
        return False
    if collinear and dist > 1e-3:
        gap = _unit(tail[-1], head[0])
        if _angle_of(leave, gap) > max_angle or _angle_of(gap, enter) > max_angle:
            return False
    return True


def _join_contours(contours: list, max_dist: float, max_angle: float, collinear: bool) -> list:
    open_chains = []
    closed = []
    for contour in contours:
        points = _points(contour)
        if len(points) < 2:
            continue
        if _closed_poly(points):
            closed.append(points)
        else:
            open_chains.append(points)
    changed = True
    while changed and len(open_chains) > 1:
        changed = False
        best = None
        for i, left in enumerate(open_chains):
            for j, right in enumerate(open_chains):
                if i == j:
                    continue
                options = (
                    (left, right, False, False),
                    (left, right[::-1], False, True),
                    (left[::-1], right, True, False),
                    (left[::-1], right[::-1], True, True),
                )
                for tail, head, _rev_left, _rev_right in options:
                    if not _junction(tail, head, max_dist, max_angle, collinear):
                        continue
                    dist = math.hypot(tail[-1][0] - head[0][0], tail[-1][1] - head[0][1])
                    if best is None or dist < best[0]:
                        best = (dist, i, j, tail, head)
        if best is None:
            break
        _dist, i, j, tail, head = best
        if math.hypot(tail[-1][0] - head[0][0], tail[-1][1] - head[0][1]) < 0.5:
            merged = tail + head[1:]
        else:
            merged = tail + head
        nxt = []
        for index, chain in enumerate(open_chains):
            if index == i or index == j:
                continue
            nxt.append(chain)
        if _closed_poly(merged):
            closed.append(merged)
        else:
            nxt.append(merged)
        open_chains = nxt
        changed = True
    return closed + open_chains


def _segment_lines(points: list, epsilon: float) -> list:
    closed = _closed_poly(points)
    contour = np.asarray(points, np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(contour, max(0.2, epsilon), closed)
    poly = [[float(point[0][0]), float(point[0][1])] for point in approx]
    pieces = []
    count = len(poly)
    span = count if closed else count - 1
    for index in range(span):
        start = poly[index]
        end = poly[(index + 1) % count]
        if math.hypot(start[0] - end[0], start[1] - end[1]) < 0.5:
            continue
        pieces.append([start, end])
    return pieces or [points]


def _segment_angle(points: list, threshold: float, min_length: float) -> list:
    if len(points) < 3:
        return [points]
    closed = _closed_poly(points)
    work = points[:-1] if closed else list(points)
    count = len(work)
    cuts = []
    for index in range(count):
        if not closed and (index == 0 or index == count - 1):
            continue
        angle = _turn_angle(work[(index - 1) % count], work[index], work[(index + 1) % count])
        if angle >= threshold:
            cuts.append(index)
    if not cuts:
        return [points]
    pieces = []
    ordered = cuts + [cuts[0] + count] if closed else [0] + cuts + [count - 1]
    if not closed:
        ordered = [0] + cuts + [count - 1]
    else:
        ordered = cuts + [cuts[0] + count]
    seen = []
    for index in ordered:
        if not seen or seen[-1] != index:
            seen.append(index)
    for start, end in zip(seen, seen[1:]):
        piece = [work[index % count] for index in range(start, end + 1)]
        length = 0.0
        for left, right in zip(piece, piece[1:]):
            length += math.hypot(right[0] - left[0], right[1] - left[1])
        if length >= min_length and len(piece) >= 2:
            pieces.append(piece)
    return pieces or [points]


def _smooth_poly(points: list, width: int) -> list:
    width = max(3, int(width) | 1)
    half = width // 2
    closed = _closed_poly(points)
    src = points[:-1] if closed else list(points)
    count = len(src)
    if count < 3:
        return points
    sigma = max(half / 2.0, 1e-6)
    offsets = list(range(-half, half + 1))
    weights = [math.exp(-0.5 * (offset / sigma) ** 2) for offset in offsets]

    def extended(pos: int) -> list[float]:
        if closed:
            return src[pos % count]
        if pos < 0:
            mirror = min(-pos, count - 1)
            origin = src[0]
            other = src[mirror]
            return [2 * origin[0] - other[0], 2 * origin[1] - other[1]]
        if pos >= count:
            mirror = max(2 * (count - 1) - pos, 0)
            origin = src[-1]
            other = src[mirror]
            return [2 * origin[0] - other[0], 2 * origin[1] - other[1]]
        return src[pos]

    smoothed = []
    for index in range(count):
        acc_col = acc_row = acc_weight = 0.0
        for offset, weight in zip(offsets, weights):
            col_value, row_value = extended(index + offset)
            acc_col += weight * col_value
            acc_row += weight * row_value
            acc_weight += weight
        smoothed.append([acc_col / acc_weight, acc_row / acc_weight])
    if closed:
        smoothed.append(smoothed[0][:])
    return smoothed


def _xld_feature_name(feature: str) -> str:
    name = str(feature).strip().lower().strip("'\"")
    alias = {"contour_length": "contlength", "length": "contlength", "center_row": "row", "center_column": "column"}
    name = alias.get(name, name)
    if name not in XLD_FEATURES:
        known = ", ".join(XLD_FEATURES)
        raise ValueError(f"不支援的輪廓特徵 {feature}。可用 {known}")
    return name


@operator(
    "輪廓",
    [],
    [P("Contour", "一條輪廓")],
    [P("Row", "列座標 tuple"), P("Col", "行座標 tuple")],
    [],
    "用點產生輪廓",
    "Row 與 Col 等長。點的順序就是輪廓方向。",
    "Vizion.gen_contour_polygon_xld (Contour, [10, 10, 40], [10, 40, 40])",
)
def gen_contour_polygon_xld(row, col, *, ctx):
    rows, cols = _pair(row, col)
    if len(rows) < 2:
        raise ValueError("至少要兩個點")
    height, width = ctx.canvas_size()
    points = [[col_value, row_value] for row_value, col_value in zip(rows, cols)]
    for col_value, row_value in points:
        width = max(width, int(math.ceil(col_value)) + 1)
        height = max(height, int(math.ceil(row_value)) + 1)
    return Result([XLD(width, height, [points])], [])


@operator(
    "輪廓",
    [P("Contour", "只含一條輪廓的 XLD")],
    [],
    [],
    [P("Row", "列座標"), P("Col", "行座標")],
    "讀出輪廓點",
    "這條 XLD 只能有一條輪廓。多條時先用 select_obj 取出。",
    "Vizion.get_contour_xld (Contour, Row, Col)",
)
def get_contour_xld(contour, *, ctx):
    xld = expect_xld(contour, "Contour")
    if len(xld.contours) != 1:
        raise ValueError(f"get_contour_xld 只能讀一條輪廓，目前有 {len(xld.contours)} 條。請先 select_obj")
    points = _points(xld.contours[0])
    return Result([], [[point[1] for point in points], [point[0] for point in points]])


@operator(
    "輪廓",
    [P("Contours", "輸入輪廓")],
    [P("Lines", "擬合後的直線段")],
    [],
    [
        P("RowBegin", "起點列"),
        P("ColBegin", "起點行"),
        P("RowEnd", "終點列"),
        P("ColEnd", "終點行"),
    ],
    "輪廓擬合直線",
    "每條輪廓先擬合，再丟掉離直線超過門檻的飛點，重擬到方向穩定。線段只覆蓋留下來的點，不會被飛點拉長。點都在直線上時，結果和最小二乘一樣。點不夠的輪廓會跳過。",
    "Vizion.fit_line_contour_xld (Contours, Lines, RowBegin, ColBegin, RowEnd, ColEnd)",
)
def fit_line_contour_xld(contours, *, ctx):
    xld = expect_xld(contours, "Contours")
    lines = []
    row1, col1, row2, col2 = [], [], [], []
    for contour in xld.contours:
        points = _points(contour)
        if len(points) < 2:
            continue
        start, end = _fit_line(points)
        lines.append([start, end])
        col1.append(start[0])
        row1.append(start[1])
        col2.append(end[0])
        row2.append(end[1])
    if not lines:
        raise ValueError("沒有可以擬合直線的輪廓")
    return Result([_pack_xld(xld.width, xld.height, lines)], [row1, col1, row2, col2])


@operator(
    "輪廓",
    [P("Contours", "輸入輪廓")],
    [P("Circles", "擬合後的圓")],
    [],
    [P("Row", "圓心列"), P("Column", "圓心行"), P("Radius", "半徑")],
    "輪廓擬合圓",
    "每條至少 3 點的輪廓用 Pratt 擬合圓，短圓弧不會把半徑縮太小。離圓超過約 1 像素的飛點會丟掉再擬一次，圓心不會被拉走。點都在圓上時，圓心和半徑不變。點不夠或排成直線的輪廓會跳過。",
    "Vizion.fit_circle_contour_xld (Contours, Circles, Row, Column, Radius)",
)
def fit_circle_contour_xld(contours, *, ctx):
    xld = expect_xld(contours, "Contours")
    circles = []
    rows, cols, radii = [], [], []
    for contour in xld.contours:
        points = _points(contour)
        if len(points) < 3:
            continue
        try:
            row, col, radius = _fit_circle(points)
        except ValueError:
            continue
        circles.append(_sample_circle(row, col, radius))
        rows.append(row)
        cols.append(col)
        radii.append(radius)
    if not circles:
        raise ValueError("沒有可以擬合圓的輪廓")
    return Result([_pack_xld(xld.width, xld.height, circles)], [rows, cols, radii])


@operator(
    "輪廓",
    [P("Contours", "輸入輪廓")],
    [P("Ellipses", "擬合後的橢圓")],
    [],
    [
        P("Row", "中心列"),
        P("Column", "中心行"),
        P("Phi", "長軸方向，弳度"),
        P("Radius1", "長軸半徑"),
        P("Radius2", "短軸半徑"),
    ],
    "輪廓擬合橢圓",
    "每條至少 5 點的輪廓擬合一個橢圓。點比較多、而且有一個離弧線很遠時，會先找中位誤差小的橢圓，再丟掉那個飛點重擬合。Phi 是長軸方向。",
    "Vizion.fit_ellipse_contour_xld (Contours, Ellipses, Row, Column, Phi, Radius1, Radius2)",
)
def fit_ellipse_contour_xld(contours, *, ctx):
    xld = expect_xld(contours, "Contours")
    ellipses = []
    rows, cols, phis, radius1, radius2 = [], [], [], [], []
    for contour in xld.contours:
        points = _points(contour)
        if len(points) < 5:
            continue
        row, col, phi, major, minor = _fit_ellipse(points)
        ellipses.append(_sample_ellipse(row, col, phi, major, minor))
        rows.append(row)
        cols.append(col)
        phis.append(phi)
        radius1.append(major)
        radius2.append(minor)
    if not ellipses:
        raise ValueError("沒有可以擬合橢圓的輪廓")
    return Result([_pack_xld(xld.width, xld.height, ellipses)], [rows, cols, phis, radius1, radius2])


def _rectangle_of(points: np.ndarray):
    (col, row), (width, height), angle = cv2.minAreaRect(points.astype(np.float32).reshape(-1, 1, 2))
    if width >= height:
        phi = math.radians(float(angle))
        half1 = float(width) / 2.0
        half2 = float(height) / 2.0
    else:
        phi = math.radians(float(angle) + 90.0)
        half1 = float(height) / 2.0
        half2 = float(width) / 2.0
    return float(row), float(col), phi, half1, half2


def _rectangle_boundary_error(points: np.ndarray, rectangle) -> np.ndarray:
    row, col, phi, half1, half2 = rectangle
    cosine = math.cos(phi)
    sine = math.sin(phi)
    dx = points[:, 0] - col
    dy = points[:, 1] - row
    along = dx * cosine + dy * sine
    across = -dx * sine + dy * cosine
    outside_along = np.abs(along) - max(half1, 0.0)
    outside_across = np.abs(across) - max(half2, 0.0)
    outside = np.hypot(np.maximum(outside_along, 0.0), np.maximum(outside_across, 0.0))
    inside = np.minimum(np.maximum(outside_along, outside_across), 0.0)
    return np.abs(outside + inside)


def _robust_rectangle(points: list):
    cloud = np.asarray(points, np.float64).reshape(-1, 2)
    base = _rectangle_of(cloud)
    if len(cloud) < 8:
        return base
    error = _rectangle_boundary_error(cloud, base)
    if float(np.max(error)) <= 1.0:
        return base
    generator = np.random.default_rng(0)
    best = base
    best_error = error
    best_score = float(np.median(error))
    best_total = float(np.sum(error))
    for _turn in range(48):
        index = generator.choice(len(cloud), 8, replace=False)
        trial = _rectangle_of(cloud[index])
        if trial[3] < 1.0 or trial[4] < 1.0:
            continue
        trial_error = _rectangle_boundary_error(cloud, trial)
        score = float(np.median(trial_error))
        total = float(np.sum(trial_error))
        if score < best_score - 1e-9 or (abs(score - best_score) <= 1e-9 and total < best_total - 1e-6):
            best = trial
            best_error = trial_error
            best_score = score
            best_total = total
    limit = 2.5 * best_score if best_score > 0.4 else 1.0
    keep = best_error <= limit + 1e-9
    if int(np.count_nonzero(keep)) >= 3 and not bool(np.all(keep)):
        return _rectangle_of(cloud[keep])
    return best


@operator(
    "輪廓",
    [P("Contours", "輸入輪廓")],
    [P("Rectangles", "擬合後的旋轉矩形")],
    [],
    [
        P("Row", "中心列"),
        P("Column", "中心行"),
        P("Phi", "長邊方向，弳度"),
        P("Length1", "半長"),
        P("Length2", "半寬"),
    ],
    "輪廓擬合旋轉矩形",
    "點比較多、而且有一個離開矩形邊時，會先找中位誤差小的矩形，再丟掉那個飛點重擬合。點很少時仍是最小外接矩形。Phi 是長邊方向。",
    "Vizion.fit_rectangle2_contour_xld (Contours, Rectangles, Row, Column, Phi, Length1, Length2)",
)
def fit_rectangle2_contour_xld(contours, *, ctx):
    xld = expect_xld(contours, "Contours")
    rectangles = []
    rows, cols, phis, length1, length2 = [], [], [], [], []
    for contour in xld.contours:
        points = _points(contour)
        if len(points) < 3:
            continue
        row, col, phi, half1, half2 = _robust_rectangle(points)
        rectangles.append(_sample_rectangle(row, col, phi, half1, half2))
        rows.append(row)
        cols.append(col)
        phis.append(phi)
        length1.append(half1)
        length2.append(half2)
    if not rectangles:
        raise ValueError("沒有可以擬合矩形的輪廓")
    return Result([_pack_xld(xld.width, xld.height, rectangles)], [rows, cols, phis, length1, length2])


@operator(
    "輪廓",
    [P("Contours", "輸入輪廓")],
    [P("UnionContours", "接起來的輪廓")],
    [P("MaxDist", "端點最大距離，像素", "10"), P("MaxAngle", "方向差上限，弳度", "0.5")],
    [],
    "把靠近的輪廓接起來",
    "端點距離在 MaxDist 以內、而且方向差不超過 MaxAngle，就接成同一條。已經閉合的輪廓不會被接走。",
    "Vizion.union_adjacent_contours_xld (Contours, UnionContours, 10, 0.5)",
)
def union_adjacent_contours_xld(contours, max_dist=10, max_angle=0.5, *, ctx):
    xld = expect_xld(contours, "Contours")
    joined = _join_contours(xld.contours, scalar(max_dist, "MaxDist"), scalar(max_angle, "MaxAngle"), False)
    return Result([_pack_xld(xld.width, xld.height, joined)], [])


@operator(
    "輪廓",
    [P("Contours", "輸入輪廓")],
    [P("UnionContours", "共線接起來的輪廓")],
    [P("MaxDist", "端點最大距離，像素", "30"), P("MaxAngle", "共線角度上限，弳度", "0.26")],
    [],
    "把共線的輪廓接起來",
    "除了端點靠近，中間空隙也要和兩條輪廓的方向一致，才會接成一條直線。",
    "Vizion.union_collinear_contours_xld (Contours, UnionContours, 30, 0.26)",
)
def union_collinear_contours_xld(contours, max_dist=30, max_angle=0.26, *, ctx):
    xld = expect_xld(contours, "Contours")
    joined = _join_contours(xld.contours, scalar(max_dist, "MaxDist"), scalar(max_angle, "MaxAngle"), True)
    return Result([_pack_xld(xld.width, xld.height, joined)], [])


@operator(
    "輪廓",
    [P("Contours", "輸入輪廓")],
    [P("ContoursSplit", "切開後的輪廓")],
    [P("Mode", "lines 或 angle", "'lines'", ["'lines'", "'angle'"]), P("MaxDist", "lines 是像素誤差，angle 是轉角弳度", "2")],
    [],
    "把輪廓截成多段",
    "lines 用多邊形近似，每一邊變成獨立線段。angle 在轉角超過門檻的地方切開，轉角點會同時屬於前後兩段。",
    "Vizion.segment_contours_xld (Contours, ContoursSplit, 'lines', 2)",
)
def segment_contours_xld(contours, mode="'lines'", max_dist=2, *, ctx):
    xld = expect_xld(contours, "Contours")
    kind = text(mode, "Mode").lower()
    limit = scalar(max_dist, "MaxDist")
    pieces = []
    for contour in xld.contours:
        points = _points(contour)
        if kind == "angle":
            pieces.extend(_segment_angle(points, limit, 1.0))
        elif kind == "lines":
            pieces.extend(_segment_lines(points, limit))
        else:
            raise ValueError("Mode 只能是 lines 或 angle")
    return Result([_pack_xld(xld.width, xld.height, pieces)], [])


@operator(
    "輪廓",
    [P("Contours", "輸入輪廓")],
    [P("SelectedContours", "符合條件的輪廓")],
    [P("Feature", "輪廓特徵", choices=XLD_FEATURE_CHOICES), P("Min", "下限，含"), P("Max", "上限，含")],
    [],
    "用特徵篩選輪廓",
    "特徵名稱和特徵視窗的 XLD 清單相同，例如 contlength、circularity、row。contour_length 也接受。",
    "Vizion.select_contours_xld (Contours, SelectedContours, 'contlength', 20, 99999)",
)
def select_contours_xld(contours, feature, min_value, max_value, *, ctx):
    xld = expect_xld(contours, "Contours")
    name = _xld_feature_name(feature)
    low = scalar(min_value, "Min")
    high = scalar(max_value, "Max")
    kept = []
    for contour in xld.contours:
        value = xld_measures(_points(contour)).get(name, 0.0)
        if low <= value <= high:
            kept.append(_points(contour))
    return Result([_pack_xld(xld.width, xld.height, kept)], [])


@operator(
    "輪廓",
    [P("Contours", "輸入輪廓")],
    [P("SmoothedContours", "平滑後的輪廓")],
    [P("NumPoints", "平滑點數，奇數", "5")],
    [],
    "平滑輪廓",
    "沿輪廓用高斯權重平滑。開放輪廓在端點做奇數延拓，直線的端點不會往裡縮。閉合輪廓維持閉合。NumPoints 是奇數窗寬。",
    "Vizion.smooth_contours_xld (Contours, SmoothedContours, 5)",
)
def smooth_contours_xld(contours, num_points=5, *, ctx):
    xld = expect_xld(contours, "Contours")
    smoothed = [_smooth_poly(_points(contour), whole(num_points, "NumPoints")) for contour in xld.contours]
    return Result([_pack_xld(xld.width, xld.height, smoothed)], [])


@operator(
    "輪廓",
    [P("Contours", "輸入輪廓")],
    [P("ClosedContours", "閉合後的輪廓")],
    [],
    [],
    "閉合輪廓",
    "還沒接回起點的輪廓，會把第一個點補到最後。",
    "Vizion.close_contours_xld (Contours, ClosedContours)",
)
def close_contours_xld(contours, *, ctx):
    xld = expect_xld(contours, "Contours")
    closed = []
    for contour in xld.contours:
        points = _points(contour)
        if len(points) >= 2 and not _closed_poly(points):
            points = points + [points[0][:]]
        closed.append(points)
    return Result([_pack_xld(xld.width, xld.height, closed)], [])


@operator(
    "輪廓",
    [P("Contours", "輸入輪廓")],
    [P("SortedContours", "排序後的輪廓")],
    [
        P("SortMode", "character 或 upper_left", "'character'", ["'character'", "'upper_left'"]),
        P("Order", "true 由小到大，false 相反", "'true'", ["'true'", "'false'"]),
        P("RowOrCol", "row 或 column", "'row'", ["'row'", "'column'"]),
    ],
    [],
    "排序輪廓",
    "character 先依列再依行。upper_left 只看 RowOrCol 指定的那一軸。順序用每條輪廓的中心。",
    "Vizion.sort_contours_xld (Contours, SortedContours, 'character', 'true', 'row')",
)
def sort_contours_xld(contours, sort_mode="'character'", order="'true'", row_or_col="'row'", *, ctx):
    xld = expect_xld(contours, "Contours")
    mode = text(sort_mode, "SortMode").lower()
    ascending = text(order, "Order").lower() not in ("false", "0", "desc")
    axis = text(row_or_col, "RowOrCol").lower()
    ranked = []
    for contour in xld.contours:
        points = _points(contour)
        if len(points) < 2:
            continue
        pts = np.asarray(points, np.float64)
        center_col = float(np.mean(pts[:, 0]))
        center_row = float(np.mean(pts[:, 1]))
        if mode == "character":
            key = (center_row, center_col)
        elif axis == "column":
            key = (center_col, center_row)
        else:
            key = (center_row, center_col)
        ranked.append((key, points))
    ranked.sort(key=lambda item: item[0], reverse=not ascending)
    return Result([_pack_xld(xld.width, xld.height, [item[1] for item in ranked])], [])


@operator(
    "輪廓",
    [P("Contour", "輸入輪廓")],
    [P("Region", "轉成的區域")],
    [P("Mode", "filled 填滿，或 margin 只畫線", "'filled'", ["'filled'", "'margin'"])],
    [],
    "輪廓轉成區域",
    "filled 把輪廓內部填滿。margin 只留下輪廓線。每一條輪廓變成一個區域物件。",
    "Vizion.gen_region_contour_xld (Contour, Region, 'filled')",
)
def gen_region_contour_xld(contour, mode="'filled'", *, ctx):
    xld = expect_xld(contour, "Contour")
    kind = text(mode, "Mode").lower()
    if kind not in ("filled", "margin"):
        raise ValueError("Mode 只能是 filled 或 margin")
    width, height = _xld_box(xld)
    masks = []
    for contour_points in xld.contours:
        points = _points(contour_points)
        if len(points) < 2:
            continue
        poly = np.round(np.asarray(points, np.float64)).astype(np.int32).reshape(-1, 1, 2)
        mask = np.zeros((height, width), np.uint8)
        if kind == "margin":
            cv2.polylines(mask, [poly], _closed_poly(points), 255, 1)
        else:
            if len(points) < 3:
                cv2.polylines(mask, [poly], False, 255, 1)
            else:
                cv2.fillPoly(mask, [poly], 255)
        if int(cv2.countNonZero(mask)):
            masks.append(mask)
    return Result([Region(width, height, masks, multi_color=len(masks) > 1)], [])


@operator(
    "輪廓",
    [P("Contours", "輸入輪廓")],
    [],
    [],
    [P("Length", "每條輪廓的長度")],
    "輪廓長度",
    "每一條輪廓一個長度，結果是 tuple。",
    "Vizion.length_xld (Contours, Length)",
)
def length_xld(contours, *, ctx):
    xld = expect_xld(contours, "Contours")
    lengths = []
    for contour in xld.contours:
        points = _points(contour)
        length = 0.0
        for left, right in zip(points, points[1:]):
            length += math.hypot(right[0] - left[0], right[1] - left[1])
        lengths.append(length)
    return Result([], [lengths])


@operator(
    "輪廓",
    [P("Contours", "輸入輪廓")],
    [],
    [],
    [P("Area", "圍成的面積"), P("Row", "中心列"), P("Column", "中心行")],
    "輪廓面積與中心",
    "用輪廓圍成的多邊形計算。開放輪廓會先看成接到起點。每一條一個值。",
    "Vizion.area_center_xld (Contours, Area, Row, Column)",
)
def area_center_xld(contours, *, ctx):
    xld = expect_xld(contours, "Contours")
    areas, rows, cols = [], [], []
    for contour in xld.contours:
        values = xld_measures(_points(contour))
        areas.append(values["area"])
        rows.append(values["row"])
        cols.append(values["column"])
    return Result([], [areas, rows, cols])


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
    "Vizion.gen_cross_contour_xld (Cross, Row, Column, 16, 0)",
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
    "Vizion.dev_display (Object)",
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
    "Vizion.dev_clear_window ()",
)
def dev_clear_window(*, ctx):
    ctx.display.clear()
    return Result([], [])


@operator(
    "顯示",
    [],
    [],
    [P("Color", "顏色名稱。自訂 #RRGGBB 寫在程式行，點兩下時會留在選單", choices=COLOR_CHOICES)],
    [],
    "設定繪圖顏色",
    "可用 red、green、blue、yellow、cyan、magenta、white、orange，或 #RRGGBB。",
    "Vizion.dev_set_color ('green')",
)
def dev_set_color(color, *, ctx):
    ctx.display.color = normalize_color(text(color, "Color"))
    return Result([], [])


@operator(
    "顯示",
    [],
    [],
    [P("DrawMode", "margin 或 fill", choices=["'margin'", "'fill'"])],
    [],
    "設定區域畫法",
    "margin 畫輪廓，fill 填色。",
    "Vizion.dev_set_draw ('margin')",
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
    [P("Width", "線寬 1 到 10")],
    [],
    "設定線寬",
    "輪廓線寬是 1 到 10 的螢幕像素，放大影像時不會跟著變粗。",
    "Vizion.dev_set_line_width (2)",
)
def dev_set_line_width(width, *, ctx):
    value = whole(width, "Width")
    if value < 1 or value > 10:
        raise ValueError("線寬要在 1 到 10")
    ctx.display.line_width = value
    return Result([], [])


@operator(
    "顯示",
    [],
    [],
    [P("Text", "文字"), P("Row", "列"), P("Column", "行")],
    [],
    "在圖上寫字",
    "VisDev 擴充。HDevelop 要透過視窗 handle，這裡直接寫進目前的圖形視窗。",
    "Vizion.dev_disp_text ('ok', 24, 12)",
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


def _blank_control(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _optional_scalar(value, name: str) -> float | None:
    if _blank_control(value):
        return None
    return scalar(value, name)


def _ocr_domain(image: Image, row1, column1, row2, column2, row, column, radius, radius1, radius2, phi) -> np.ndarray | None:
    rect = (
        _optional_scalar(row1, "Row1"),
        _optional_scalar(column1, "Column1"),
        _optional_scalar(row2, "Row2"),
        _optional_scalar(column2, "Column2"),
    )
    center = (_optional_scalar(row, "Row"), _optional_scalar(column, "Column"))
    circle = _optional_scalar(radius, "Radius")
    ellipse = (_optional_scalar(radius1, "Radius1"), _optional_scalar(radius2, "Radius2"))
    use_rect = all(item is not None for item in rect)
    use_ellipse = all(item is not None for item in center) and all(item is not None for item in ellipse)
    use_circle = all(item is not None for item in center) and circle is not None
    if any(item is not None for item in rect) and not use_rect:
        raise ValueError("矩形搜尋範圍要四個邊界都填，或全部留空")
    if (circle is not None or any(item is not None for item in ellipse)) and any(item is None for item in center):
        raise ValueError("圓和橢圓要一起填中心的列、行")
    if any(item is not None for item in ellipse) and not use_ellipse:
        raise ValueError("橢圓要同時填兩個半徑，或兩個都留空")
    chosen = [name for name, used in (("矩形", use_rect), ("橢圓", use_ellipse), ("圓", use_circle)) if used]
    if len(chosen) > 1:
        raise ValueError("搜尋範圍一次只用一種圖形：" + "、".join(chosen))
    if not chosen:
        return None
    mask = np.zeros((image.height, image.width), np.uint8)
    if use_rect:
        r1, c1, r2, c2 = rect
        cv2.rectangle(mask, (int(round(min(c1, c2))), int(round(min(r1, r2)))), (int(round(max(c1, c2))), int(round(max(r1, r2)))), 255, -1)
    elif use_ellipse:
        angle = 0.0 if _blank_control(phi) else math.degrees(scalar(phi, "Phi"))
        axes = (max(1, int(round(ellipse[0]))), max(1, int(round(ellipse[1]))))
        cv2.ellipse(mask, (int(round(center[1])), int(round(center[0]))), axes, angle, 0, 360, 255, -1)
    else:
        cv2.circle(mask, (int(round(center[1])), int(round(center[0]))), max(1, int(round(circle))), 255, -1)
    return mask


@operator(
    "文字",
    [P("Image", "灰階或彩色影像")],
    [P("Words", "每個字塊的外框")],
    [
        P("Method", "目錄中的引擎名稱。auto 只用已安裝的本機通用引擎，不會把影像送到雲端", "'auto'", method_choices()),
        P("Row1", "搜尋矩形的上邊界。四個邊界都留空字串就不用矩形", "''"),
        P("Column1", "搜尋矩形的左邊界", "''"),
        P("Row2", "搜尋矩形的下邊界", "''"),
        P("Column2", "搜尋矩形的右邊界", "''"),
        P("Row", "圓或橢圓的中心列。不用就留空字串", "''"),
        P("Column", "圓或橢圓的中心行", "''"),
        P("Radius", "圓半徑。不用圓就留空字串", "''"),
        P("Radius1", "橢圓沿行方向的半徑", "''"),
        P("Radius2", "橢圓沿列方向的半徑", "''"),
        P("Phi", "橢圓角度，弳度。0 是軸對齊", "''"),
    ],
    [P("Text", "讀到的字"), P("Row", "字塊中心列"), P("Column", "字塊中心行"), P("Score", "信心，0 到 1")],
    "讀出影像上的文字",
    "搜尋範圍可以空白，表示整張圖。參數窗開著時，在圖上按右鍵拉矩形、圓或橢圓，確認後填進這個範圍，只讀範圍裡的字。座標仍是整張圖的列、行。" + describe_catalog(),
    "Vizion.ocr_find (Image, Words, 'auto', Text, Row, Column, Score)",
)
def ocr_find(image, method, row1, column1, row2, column2, row, column, radius, radius1, radius2, phi, *, ctx):
    image = expect_image(image, "Image")
    roi = _ocr_domain(image, row1, column1, row2, column2, row, column, radius, radius1, radius2, phi)
    if roi is not None:
        domain = roi if image.domain is None else cv2.bitwise_and(image.domain, roi)
        image = Image(image.data, domain)
    _used, hits = read_text(image, text(method, "Method"))
    if ctx.display.base is None:
        ctx.display_obj(image)
    contours = []
    strings, rows, cols, scores = [], [], [], []
    for hit in hits:
        xs = [point[0] for point in hit.box]
        ys = [point[1] for point in hit.box]
        row = sum(ys) / len(ys)
        col = sum(xs) / len(xs)
        strings.append(hit.text)
        rows.append(row)
        cols.append(col)
        scores.append(hit.score)
        contours.append(hit.box)
        ctx.display.overlays.append(
            {
                "kind": "text",
                "text": hit.text,
                "row": min(ys),
                "col": min(xs),
                "color": "#ffe14a",
            }
        )
    words = XLD(image.width, image.height, contours, multi_color=len(contours) > 1)
    return Result([words], [strings, rows, cols, scores])


@operator(
    "顯示",
    [],
    [],
    [],
    [],
    "暫停執行",
    "停在下一行，和中斷點一樣，可以再按執行或單步繼續。",
    "Vizion.stop ()",
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
    "Vizion.tuple_length (Tuple, Length)",
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
    "Vizion.tuple_min (Tuple, Min)",
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
    "Vizion.tuple_max (Tuple, Max)",
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
    "Vizion.tuple_sum (Tuple, Sum)",
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
    "Vizion.tuple_mean (Tuple, Mean)",
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


def _match_numbers(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [float(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


def _with_origin(rows, cols, model, scales=None):
    if getattr(model, "cleared", False):
        raise RuntimeError("模型已經 clear，不能再搜尋")
    ref_row = getattr(model, "ref_row", None)
    ref_col = getattr(model, "ref_col", None)
    if ref_row is None or ref_col is None:
        return rows, cols
    row_list = _match_numbers(rows)
    col_list = _match_numbers(cols)
    if not row_list:
        return [], []
    if scales is None:
        scale_list = [1.0] * len(row_list)
    else:
        scale_list = _match_numbers(scales)
        if len(scale_list) == 1 and len(row_list) > 1:
            scale_list = scale_list * len(row_list)
    shift_row = float(ref_row) - float(model.origin_row)
    shift_col = float(ref_col) - float(model.origin_col)
    return (
        [row + shift_row * scale for row, scale in zip(row_list, scale_list)],
        [col + shift_col * scale for col, scale in zip(col_list, scale_list)],
    )


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
        P("Metric", "use_polarity 或 ignore_local_polarity", choices=["'use_polarity'", "'ignore_local_polarity'"]),
        P("Contrast", "邊緣對比，或 auto"),
        P("MinContrast", "搜尋圖的最低對比，或 auto"),
    ],
    [P("ModelID", "形狀模型")],
    "建立形狀模型",
    "建立時抽出邊緣點，做成金字塔，並依角度步進預先轉好每一層。原點是 domain 重心；沒有 domain 時用影像中心。",
    "Vizion.create_shape_model (Template, 0, rad(-180), rad(360), 0, 'use_polarity', 'auto', 'auto', ModelID)",
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
        scalar(angle_start, "AngleStart"),
        scalar(angle_extent, "AngleExtent"),
    )
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
        P("Metric", "use_polarity 或 ignore_local_polarity", choices=["'use_polarity'", "'ignore_local_polarity'"]),
        P("Contrast", "邊緣對比，或 auto"),
        P("MinContrast", "搜尋圖的最低對比，或 auto"),
    ],
    [P("ModelID", "形狀模型")],
    "教導形狀模型",
    "用矩形框出要記的圖案。模型原點是教導框中心，定位時的 Row、Column 就是這個中心。",
    "Vizion.teach_shape_model (Image, 15, 15, 164, 164, 0, rad(-180), rad(360), 0, 'use_polarity', 40, 10, ModelID)",
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
        P("SubPixel", "none、interpolation 或 least_squares", choices=["'none'", "'interpolation'", "'least_squares'"]),
        P("NumLevels", "使用層數，0 表示模型全部層"),
        P("Greediness", "0 到 1，越大越快，也可能漏掉"),
    ],
    [P("Row", "列"), P("Column", "行"), P("Angle", "角度，弳度"), P("Score", "分數")],
    "尋找形狀模型",
    "用建立時算好的旋轉模型，從最粗層全圖搜尋，再只在候選附近細化。interpolation 用分數曲面的拋物線。least_squares 再沿邊緣法線把列、行和角度做最小二乘。",
    "Vizion.find_shape_model (Image, ModelID, rad(-180), rad(360), 0.6, 0, 0.5, 'interpolation', 0, 0.8, Row, Column, Angle, Score)",
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
    from .shape_model import bind_search_cancel, find_shape_model as search, release_search_cancel

    picture = expect_image(image, "Image")
    shape = _expect_model(model)
    mode = text(subpixel, "SubPixel").lower()
    if mode not in ("none", "interpolation", "least_squares"):
        raise ValueError("SubPixel 只能是 none、interpolation 或 least_squares")
    token = bind_search_cancel(lambda: bool(getattr(ctx, "cancel", False)))
    try:
        rows, cols, angles, scores = search(
            picture,
            shape,
            scalar(angle_start, "AngleStart"),
            scalar(angle_extent, "AngleExtent"),
            scalar(min_score, "MinScore"),
            whole(num_matches, "NumMatches"),
            scalar(max_overlap, "MaxOverlap"),
            mode,
            whole(num_levels, "NumLevels"),
            scalar(greediness, "Greediness"),
        )
    finally:
        release_search_cancel(token)
    rows, cols = _with_origin(rows, cols, shape)
    return Result([], [rows, cols, angles, scores])


@operator(
    "形狀匹配",
    [],
    [P("ModelContours", "模型輪廓")],
    [P("ModelID", "形狀模型"), P("Level", "1 是最細的一層")],
    [],
    "取出模型輪廓",
    "輪廓點已沿梯度做到次像素，座標在模板影像上。Level 1 是原始解析度。",
    "Vizion.get_shape_model_contours (ModelContours, ModelID, 1)",
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
    "Vizion.dev_display_shape_matching_results (ModelID, Row, Column, Angle)",
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
        P("Metric", "use_polarity 或 ignore_global_polarity", choices=["'use_polarity'", "'ignore_global_polarity'"]),
    ],
    [P("ModelID", "NCC 模型")],
    "建立 NCC 模型",
    "建立時做成灰階金字塔，並依角度步進預先轉好每一層的模板。原點是 domain 重心；沒有 domain 時用影像中心。",
    "Vizion.create_ncc_model (Template, 0, rad(-180), rad(360), 0, 'use_polarity', ModelID)",
)
def create_ncc_model(template, num_levels, angle_start, angle_extent, angle_step, metric, *, ctx):
    from .ncc_model import create_ncc_model as build_model

    image = expect_image(template, "Template")
    kind = text(metric, "Metric").lower()
    if kind not in ("use_polarity", "ignore_global_polarity"):
        raise ValueError("Metric 只能是 use_polarity 或 ignore_global_polarity")
    model = build_model(
        image,
        whole(num_levels, "NumLevels"),
        _optional_number(angle_step, "AngleStep") or 0,
        kind,
        scalar(angle_start, "AngleStart"),
        scalar(angle_extent, "AngleExtent"),
    )
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
        P("Metric", "use_polarity 或 ignore_global_polarity", choices=["'use_polarity'", "'ignore_global_polarity'"]),
    ],
    [P("ModelID", "NCC 模型")],
    "教導 NCC 模型",
    "用矩形框出要記的灰階圖案。模型原點是教導框中心。",
    "Vizion.teach_ncc_model (Image, 15, 15, 164, 164, 0, rad(-180), rad(360), 0, 'use_polarity', ModelID)",
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
        P("SubPixel", "none、interpolation 或 least_squares", choices=["'none'", "'interpolation'", "'least_squares'"]),
        P("NumLevels", "使用層數，0 表示模型全部層"),
    ],
    [P("Row", "中心列"), P("Column", "中心行"), P("Angle", "角度，弳度"), P("Score", "分數")],
    "NCC 定位",
    "用建立時算好的旋轉模板，從最粗層做正規相關，再只在候選附近細化。interpolation 在整數峰周圍的分數上做 Keys 三次內插，平坦的峰留在整數。least_squares 仍從整數峰用 Lucas-Kanade 沿灰階梯度修列和行，角度維持搜尋結果。",
    "Vizion.find_ncc_model (Image, ModelID, rad(-180), rad(360), 0.7, 0, 0.4, 'interpolation', 0, Row, Column, Angle, Score)",
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
        mode,
        whole(num_levels, "NumLevels"),
    )
    rows, cols = _with_origin(rows, cols, model)
    return Result([], [rows, cols, angles, scores])


@operator(
    "NCC定位",
    [],
    [],
    [P("ModelID", "NCC 模型"), P("Row", "中心列"), P("Column", "中心行"), P("Angle", "角度")],
    [],
    "畫出 NCC 定位",
    "只畫中心十字、方向箭頭，以及箭頭旁的角度。每個實例使用 16 色循環中的下一個顏色。角度 0 指向右方，逆時針為正，顯示到小數三位。",
    "Vizion.dev_display_ncc_matching_results (ModelID, Row, Column, Angle)",
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
    "Vizion.dev_set_colored (16)",
)
def dev_set_colored(num_colors, *, ctx):
    count = whole(num_colors, "NumColors")
    ctx.display.colored = 0 if count <= 1 else min(count, 16)
    return Result([], [])


def _gauge_source(image):
    src = expect_image(image, "Image")
    return src, src.gray(), src.domain


def _gauge_view(image: Image, chains: list) -> XLD:
    clean = [chain for chain in chains if len(chain) >= 2]
    if not clean:
        raise ValueError("沒有可顯示的量測圖形")
    return XLD(image.width, image.height, clean)


def _gauge_cross(column: float, row: float, size: float = 8) -> list:
    return [
        [[column - size, row], [column + size, row]],
        [[column, row - size], [column, row + size]],
    ]


_GAUGE_TAIL = [
    P("Sigma", "剖面平滑，單位是像素", "1"),
    P("Threshold", "梯度門檻，灰階／像素", "8"),
    P("Transition", "all、positive 由暗到亮、negative 由亮到暗", "'all'", ["'all'", "'positive'", "'negative'"]),
    P("Select", "all 每一個、best 最強、first 第一個、last 最後一個", "'best'", ["'all'", "'best'", "'first'", "'last'"]),
]


@operator(
    "量測",
    [P("Image", "灰階或彩色影像")],
    [P("Cross", "量到的點，畫成十字")],
    [P("Row1", "搜尋起點的列"), P("Column1", "搜尋起點的行"), P("Row2", "搜尋終點的列"), P("Column2", "搜尋終點的行"), *_GAUGE_TAIL],
    [P("Row", "邊緣的列"), P("Column", "邊緣的行"), P("Amplitude", "梯度，沿搜尋方向"), P("Distance", "離起點的像素距離")],
    "點量規：沿一條線找次像素邊緣",
    "在起點到終點的灰階剖面上找梯度峰，再用拋物線內插到次像素。positive 是沿搜尋方向由暗到亮，negative 相反，all 兩者都收。Select 用 all 會回傳每一個邊緣。這組量規是剖面次像素邊緣再擬合幾何，不是商業量測函式庫的程式。",
    "Vizion.gauge_point (Image, Cross, 110, 80, 110, 220, 1, 8, 'all', 'first', Row, Column, Amplitude, Distance)",
)
def gauge_point(image, row1, column1, row2, column2, sigma, threshold, transition, select, *, ctx):
    src, gray, domain = _gauge_source(image)
    hits = gauge.measure_point(
        gray, domain, scalar(row1, "Row1"), scalar(column1, "Column1"), scalar(row2, "Row2"), scalar(column2, "Column2"),
        scalar(sigma, "Sigma"), scalar(threshold, "Threshold"), text(transition, "Transition"), text(select, "Select"),
    )
    chains = []
    for hit in hits:
        chains.extend(_gauge_cross(hit.column, hit.row))
    return Result(
        [_gauge_view(src, chains)],
        [[hit.row for hit in hits], [hit.column for hit in hits], [hit.amplitude for hit in hits], [hit.distance for hit in hits]],
    )


@operator(
    "量測",
    [P("Image", "灰階或彩色影像")],
    [P("Line", "擬合後的直線")],
    [
        P("Row1", "名義直線起點的列"), P("Column1", "名義直線起點的行"),
        P("Row2", "名義直線終點的列"), P("Column2", "名義直線終點的行"),
        P("Tolerance", "垂直於直線的搜尋半長，像素"),
        *_GAUGE_TAIL,
    ],
    [P("Row1", "擬合起點的列"), P("Column1", "擬合起點的行"), P("Row2", "擬合終點的列"), P("Column2", "擬合終點的行"), P("Angle", "直線方向，弳度"), P("Used", "用來擬合的點數")],
    "直線量規：沿名義直線找邊再擬合",
    "在直線上放多條垂直搜尋線，每條取一個次像素邊緣，再擬合直線。Select 為 all 時，每條搜尋線仍取梯度最強的點。角度 0 指向右方，列增加的方向為正。",
    "Vizion.gauge_line (Image, Line, 200, 380, 200, 480, 12, 1, 8, 'all', 'best', Row1, Column1, Row2, Column2, Angle, Used)",
)
def gauge_line(image, row1, column1, row2, column2, tolerance, sigma, threshold, transition, select, *, ctx):
    src, gray, domain = _gauge_source(image)
    found = gauge.measure_line(
        gray, domain, scalar(row1, "Row1"), scalar(column1, "Column1"), scalar(row2, "Row2"), scalar(column2, "Column2"),
        scalar(tolerance, "Tolerance"), scalar(sigma, "Sigma"), scalar(threshold, "Threshold"), text(transition, "Transition"), text(select, "Select"),
    )
    return Result(
        [_gauge_view(src, [found.contour])],
        [found.row1, found.column1, found.row2, found.column2, found.angle, found.used],
    )


@operator(
    "量測",
    [P("Image", "灰階或彩色影像")],
    [P("Circle", "擬合後的圓")],
    [
        P("Row", "名義圓心的列"), P("Column", "名義圓心的行"), P("Radius", "名義半徑"),
        P("Tolerance", "沿半徑的搜尋半長，像素"),
        *_GAUGE_TAIL,
    ],
    [P("Row", "圓心列"), P("Column", "圓心行"), P("Radius", "半徑"), P("Used", "用來擬合的點數")],
    "圓量規：沿半徑找邊再擬合圓",
    "繞名義圓一圈放搜尋線，由內往外取次像素邊緣，再做幾何圓擬合。名義圓心要落在真實圓心的 Tolerance 之內。",
    "Vizion.gauge_circle (Image, Circle, 110, 150, 42, 12, 1, 8, 'all', 'best', Row, Column, Radius, Used)",
)
def gauge_circle(image, row, column, radius, tolerance, sigma, threshold, transition, select, *, ctx):
    src, gray, domain = _gauge_source(image)
    found = gauge.measure_circle(
        gray, domain, scalar(row, "Row"), scalar(column, "Column"), scalar(radius, "Radius"),
        scalar(tolerance, "Tolerance"), scalar(sigma, "Sigma"), scalar(threshold, "Threshold"), text(transition, "Transition"), text(select, "Select"),
    )
    return Result([_gauge_view(src, [found.contour])], [found.row, found.column, found.radius, found.used])


@operator(
    "量測",
    [P("Image", "灰階或彩色影像")],
    [P("Rectangle", "擬合後的矩形")],
    [
        P("Row", "名義中心的列"), P("Column", "名義中心的行"), P("Phi", "長軸方向，弳度"),
        P("Length1", "沿 Phi 的半長"), P("Length2", "垂直 Phi 的半長"),
        P("Tolerance", "垂直於各邊的搜尋半長，像素"),
        *_GAUGE_TAIL,
    ],
    [P("Row", "中心列"), P("Column", "中心行"), P("Phi", "長軸方向，弳度"), P("Length1", "半長"), P("Length2", "半寬"), P("Used", "用來擬合的點數")],
    "矩形量規：四邊各找邊再交成矩形",
    "Phi 為 0 時 Length1 沿著行方向，Length2 沿著列方向。四邊各自擬合直線，交點決定矩形。",
    "Vizion.gauge_rectangle (Image, Rectangle, 244.5, 429.5, 0, 70, 45, 12, 1, 8, 'all', 'best', Row, Column, Phi, Length1, Length2, Used)",
)
def gauge_rectangle(image, row, column, phi, length1, length2, tolerance, sigma, threshold, transition, select, *, ctx):
    src, gray, domain = _gauge_source(image)
    found = gauge.measure_rectangle(
        gray, domain,
        scalar(row, "Row"), scalar(column, "Column"), scalar(phi, "Phi"), scalar(length1, "Length1"), scalar(length2, "Length2"),
        scalar(tolerance, "Tolerance"), scalar(sigma, "Sigma"), scalar(threshold, "Threshold"), text(transition, "Transition"), text(select, "Select"),
    )
    return Result(
        [_gauge_view(src, [found.contour])],
        [found.row, found.column, found.phi, found.length1, found.length2, found.used],
    )


@operator(
    "量測",
    [P("Image", "灰階或彩色影像")],
    [P("Wedge", "擬合後的扇形")],
    [
        P("Row", "名義圓心的列"), P("Column", "名義圓心的行"),
        P("Radius1", "內半徑"), P("Radius2", "外半徑"),
        P("AngleStart", "起始角，弳度"), P("AngleExtent", "張角，弳度，逆時針為正"),
        P("Tolerance", "搜尋半長，像素"),
        *_GAUGE_TAIL,
    ],
    [
        P("Row", "圓心列"), P("Column", "圓心行"), P("Radius1", "內半徑"), P("Radius2", "外半徑"),
        P("AngleStart", "起始角，弳度"), P("AngleExtent", "張角，弳度"), P("Used", "用來擬合的點數"),
    ],
    "扇形量規：內弧、外弧和兩條半徑",
    "角度 0 指向右方，列增加的方向為正。張角為正時，角度沿這個方向增加。四段邊緣都要找得到。",
    "Vizion.gauge_wedge (Image, Wedge, 340, 170, 24, 58, 0.35, 1.8, 10, 1, 8, 'all', 'best', Row, Column, Radius1, Radius2, AngleStart, AngleExtent, Used)",
)
def gauge_wedge(
    image, row, column, radius1, radius2, angle_start, angle_extent, tolerance, sigma, threshold, transition, select, *, ctx
):
    src, gray, domain = _gauge_source(image)
    found = gauge.measure_wedge(
        gray, domain,
        scalar(row, "Row"), scalar(column, "Column"), scalar(radius1, "Radius1"), scalar(radius2, "Radius2"),
        scalar(angle_start, "AngleStart"), scalar(angle_extent, "AngleExtent"),
        scalar(tolerance, "Tolerance"), scalar(sigma, "Sigma"), scalar(threshold, "Threshold"), text(transition, "Transition"), text(select, "Select"),
    )
    return Result(
        [_gauge_view(src, [found.contour])],
        [found.row, found.column, found.radius1, found.radius2, found.angle_start, found.angle_extent, found.used],
    )


@operator(
    "量測",
    [P("Image", "灰階或彩色影像")],
    [P("Polygon", "擬合後的多邊形")],
    [
        P("Rows", "名義頂點的列"), P("Columns", "名義頂點的行"),
        P("Tolerance", "垂直於各邊的搜尋半長，像素"),
        *_GAUGE_TAIL,
        P("Closed", "1 封閉，0 開放", "1", ["1", "0"]),
    ],
    [P("Rows", "擬合頂點的列"), P("Columns", "擬合頂點的行"), P("Used", "用來擬合的點數")],
    "多邊形量規：每邊找邊，相鄰邊相交成頂點",
    "頂點請沿邊界順序給。首尾座標相同時會當成封閉。開放多邊形的兩個端點是投影到擬合直線上的位置。",
    "Vizion.gauge_polygon (Image, Polygon, [200, 200, 290, 290], [360, 500, 500, 360], 12, 1, 8, 'all', 'best', 1, Rows, Columns, Used)",
)
def gauge_polygon(image, rows, columns, tolerance, sigma, threshold, transition, select, closed, *, ctx):
    src, gray, domain = _gauge_source(image)
    found = gauge.measure_polygon(
        gray, domain, numbers_of(rows), numbers_of(columns),
        scalar(tolerance, "Tolerance"), scalar(sigma, "Sigma"), scalar(threshold, "Threshold"), text(transition, "Transition"), text(select, "Select"),
        whole(closed, "Closed") != 0,
    )
    return Result([_gauge_view(src, [found.contour])], [found.rows, found.columns, found.used])


@operator(
    "量測",
    [],
    [],
    [P("Row1", "第一點的列"), P("Column1", "第一點的行"), P("Row2", "第二點的列"), P("Column2", "第二點的行")],
    [P("Distance", "歐氏距離，像素")],
    "兩點距離",
    "座標是像素。列往下、行往右。",
    "Vizion.distance_pp (0, 0, 3, 4, Distance)",
)
def distance_pp(row1, column1, row2, column2, *, ctx):
    return Result([], [gauge.distance_pp(scalar(row1, "Row1"), scalar(column1, "Column1"), scalar(row2, "Row2"), scalar(column2, "Column2"))])


@operator(
    "量測",
    [],
    [],
    [
        P("Row", "點的列"), P("Column", "點的行"),
        P("Row1", "直線起點的列"), P("Column1", "直線起點的行"),
        P("Row2", "直線終點的列"), P("Column2", "直線終點的行"),
    ],
    [P("Distance", "點到無限長直線的距離，像素")],
    "點到直線的距離",
    "直線用兩個點決定，距離是垂直距離，不是到線段端點的距離。",
    "Vizion.distance_pl (20, 0, 0, 10, 50, 10, Distance)",
)
def distance_pl(row, column, row1, column1, row2, column2, *, ctx):
    return Result([], [gauge.distance_pl(
        scalar(row, "Row"), scalar(column, "Column"),
        scalar(row1, "Row1"), scalar(column1, "Column1"), scalar(row2, "Row2"), scalar(column2, "Column2"),
    )])


def _project_line(row, column, row1, column1, row2, column2):
    drow = row2 - row1
    dcol = column2 - column1
    length2 = drow * drow + dcol * dcol
    if length2 < 1e-12:
        return row1, column1, 0.0
    scale = ((row - row1) * drow + (column - column1) * dcol) / length2
    return row1 + scale * drow, column1 + scale * dcol, scale


@operator(
    "量測",
    [],
    [],
    [
        P("Row", "點的列"), P("Column", "點的行"),
        P("Row1", "線段起點的列"), P("Column1", "線段起點的行"),
        P("Row2", "線段終點的列"), P("Column2", "線段終點的行"),
    ],
    [P("Distance", "點到線段的距離，像素")],
    "點到線段的距離",
    "垂足落在線段外時，距離改量到較近的端點。和 distance_pl 的無限長直線不同。",
    "Vizion.distance_ps (0, 0, 0, 10, 0, 20, Distance)",
)
def distance_ps(row, column, row1, column1, row2, column2, *, ctx):
    py = scalar(row, "Row")
    px = scalar(column, "Column")
    r1 = scalar(row1, "Row1")
    c1 = scalar(column1, "Column1")
    r2 = scalar(row2, "Row2")
    c2 = scalar(column2, "Column2")
    proj_row, proj_col, scale = _project_line(py, px, r1, c1, r2, c2)
    if scale < 0:
        proj_row, proj_col = r1, c1
    elif scale > 1:
        proj_row, proj_col = r2, c2
    return Result([], [math.hypot(py - proj_row, px - proj_col)])


def _line_distances(rows, cols, row1: float, column1: float, row2: float, column2: float, segment: bool) -> np.ndarray:
    rows = np.asarray(rows, np.float64).reshape(-1)
    cols = np.asarray(cols, np.float64).reshape(-1)
    drow = row2 - row1
    dcol = column2 - column1
    length = math.hypot(drow, dcol)
    if length < 1e-6:
        raise ValueError("直線的兩個點重疊")
    if not segment:
        return np.abs((cols - column1) * (drow / length) - (rows - row1) * (dcol / length))
    scale = ((rows - row1) * drow + (cols - column1) * dcol) / (length * length)
    scale = np.clip(scale, 0.0, 1.0)
    return np.hypot(rows - (row1 + scale * drow), cols - (column1 + scale * dcol))


def _region_line_distance(region, row1, column1, row2, column2, segment: bool) -> list[float]:
    source = expect_region(region, "Region")
    r1, c1 = scalar(row1, "Row1"), scalar(column1, "Column1")
    r2, c2 = scalar(row2, "Row2"), scalar(column2, "Column2")
    distances = []
    for mask in source.objects:
        rows, cols = np.nonzero(mask)
        if len(rows) == 0:
            continue
        distances.append(float(_line_distances(rows, cols, r1, c1, r2, c2, segment).min()))
    if not distances:
        raise ValueError("區域是空的")
    return distances


@operator(
    "量測",
    [P("Region", "區域")],
    [],
    [
        P("Row1", "直線起點的列"), P("Column1", "直線起點的行"),
        P("Row2", "直線終點的列"), P("Column2", "直線終點的行"),
    ],
    [P("Distance", "區域到無限長直線的最短距離")],
    "區域到直線的距離",
    "每個物件一個距離，量的是區域裡最近的像素到無限長直線的垂直距離。",
    "Vizion.distance_lr (Region, 0, 0, 0, 100, Distance)",
)
def distance_lr(region, row1, column1, row2, column2, *, ctx):
    return Result([], [_region_line_distance(region, row1, column1, row2, column2, False)])


@operator(
    "量測",
    [P("Region", "區域")],
    [],
    [
        P("Row1", "線段起點的列"), P("Column1", "線段起點的行"),
        P("Row2", "線段終點的列"), P("Column2", "線段終點的行"),
    ],
    [P("Distance", "區域到線段的最短距離")],
    "區域到線段的距離",
    "垂足落在線段外時，改量到較近的端點。每個物件一個距離。",
    "Vizion.distance_sr (Region, 0, 0, 0, 100, Distance)",
)
def distance_sr(region, row1, column1, row2, column2, *, ctx):
    return Result([], [_region_line_distance(region, row1, column1, row2, column2, True)])


@operator(
    "量測",
    [P("Contour", "輪廓")],
    [],
    [
        P("Row1", "直線起點的列"), P("Column1", "直線起點的行"),
        P("Row2", "直線終點的列"), P("Column2", "直線終點的行"),
    ],
    [P("Distance", "輪廓到無限長直線的最短距離")],
    "輪廓到直線的距離",
    "每條輪廓一個距離，用輪廓上的點量到無限長直線。",
    "Vizion.distance_lc (Contour, 0, 0, 0, 100, Distance)",
)
def distance_lc(contour, row1, column1, row2, column2, *, ctx):
    source = expect_xld(contour, "Contour")
    r1, c1 = scalar(row1, "Row1"), scalar(column1, "Column1")
    r2, c2 = scalar(row2, "Row2"), scalar(column2, "Column2")
    distances = []
    for item in source.contours:
        points = _points(item)
        if len(points) < 2:
            continue
        cols = [point[0] for point in points]
        rows = [point[1] for point in points]
        distances.append(float(_line_distances(rows, cols, r1, c1, r2, c2, False).min()))
    if not distances:
        raise ValueError("輪廓是空的")
    return Result([], [distances])


def _edges_cross(row1, column1, row2, column2, er1, ec1, er2, ec2) -> bool:
    def side(orow, ocol, arow, acol, brow, bcol):
        return (arow - orow) * (bcol - ocol) - (acol - ocol) * (brow - orow)

    left = side(row1, column1, row2, column2, er1, ec1)
    right = side(row1, column1, row2, column2, er2, ec2)
    start = side(er1, ec1, er2, ec2, row1, column1)
    end = side(er1, ec1, er2, ec2, row2, column2)
    return bool(np.any((left * right < 0) & (start * end < 0)))


def _points_to_edges(rows, cols, er1, ec1, er2, ec2) -> float:
    rows = np.asarray(rows, np.float64).reshape(-1, 1)
    cols = np.asarray(cols, np.float64).reshape(-1, 1)
    drow = er2 - er1
    dcol = ec2 - ec1
    length2 = drow * drow + dcol * dcol
    scale = np.zeros((len(rows), len(er1)), np.float64)
    nonzero = length2 > 1e-12
    proj = (rows - er1) * drow + (cols - ec1) * dcol
    scale[:, nonzero] = proj[:, nonzero] / length2[nonzero]
    scale = np.clip(scale, 0.0, 1.0)
    return float(np.hypot(rows - (er1 + scale * drow), cols - (ec1 + scale * dcol)).min())


def _polyline_segment_distance(rows, cols, row1, column1, row2, column2) -> float:
    best = float(_line_distances(rows, cols, row1, column1, row2, column2, True).min())
    if len(np.asarray(rows).reshape(-1)) < 2:
        return best
    er1 = np.asarray(rows, np.float64).reshape(-1)[:-1]
    ec1 = np.asarray(cols, np.float64).reshape(-1)[:-1]
    er2 = np.asarray(rows, np.float64).reshape(-1)[1:]
    ec2 = np.asarray(cols, np.float64).reshape(-1)[1:]
    if _edges_cross(row1, column1, row2, column2, er1, ec1, er2, ec2):
        return 0.0
    return min(best, _points_to_edges([row1, row2], [column1, column2], er1, ec1, er2, ec2))


@operator(
    "量測",
    [P("Contour", "輪廓")],
    [],
    [
        P("Row1", "線段起點的列"), P("Column1", "線段起點的行"),
        P("Row2", "線段終點的列"), P("Column2", "線段終點的行"),
    ],
    [P("Distance", "輪廓到線段的最短距離")],
    "輪廓到線段的距離",
    "每條輪廓一個距離。折線和線段相交時是 0，否則量端點到另一段的距離。不會自動把輪廓封閉。",
    "Vizion.distance_sc (Contour, 0, 0, 0, 100, Distance)",
)
def distance_sc(contour, row1, column1, row2, column2, *, ctx):
    source = expect_xld(contour, "Contour")
    r1, c1 = scalar(row1, "Row1"), scalar(column1, "Column1")
    r2, c2 = scalar(row2, "Row2"), scalar(column2, "Column2")
    distances = []
    for item in source.contours:
        points = _points(item)
        if len(points) < 2:
            continue
        cols = [point[0] for point in points]
        rows = [point[1] for point in points]
        distances.append(_polyline_segment_distance(rows, cols, r1, c1, r2, c2))
    if not distances:
        raise ValueError("輪廓是空的")
    return Result([], [distances])


def _segment_gap(row1, column1, row2, column2, row3, column3, row4, column4) -> float:
    def point_gap(row, column, start_row, start_col, end_row, end_col) -> float:
        proj_row, proj_col, scale = _project_line(row, column, start_row, start_col, end_row, end_col)
        if scale < 0:
            proj_row, proj_col = start_row, start_col
        elif scale > 1:
            proj_row, proj_col = end_row, end_col
        return math.hypot(row - proj_row, column - proj_col)

    def cross(origin, left, right) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (right[0] - origin[0])

    first = (row1, column1)
    second = (row2, column2)
    third = (row3, column3)
    fourth = (row4, column4)
    if cross(first, second, third) * cross(first, second, fourth) < 0 and cross(third, fourth, first) * cross(third, fourth, second) < 0:
        return 0.0
    return min(
        point_gap(row1, column1, row3, column3, row4, column4),
        point_gap(row2, column2, row3, column3, row4, column4),
        point_gap(row3, column3, row1, column1, row2, column2),
        point_gap(row4, column4, row1, column1, row2, column2),
    )


@operator(
    "量測",
    [],
    [],
    [
        P("Row1", "第一條起點的列"), P("Column1", "第一條起點的行"),
        P("Row2", "第一條終點的列"), P("Column2", "第一條終點的行"),
        P("Row3", "第二條起點的列"), P("Column3", "第二條起點的行"),
        P("Row4", "第二條終點的列"), P("Column4", "第二條終點的行"),
    ],
    [P("Distance", "兩條線段的最短距離")],
    "兩條線段的距離",
    "線段相交或端點落在另一條上時，距離是 0。這不是無限長直線。",
    "Vizion.distance_ss (0, 0, 0, 10, 5, 0, 5, 10, Distance)",
)
def distance_ss(row1, column1, row2, column2, row3, column3, row4, column4, *, ctx):
    return Result([], [_segment_gap(
        scalar(row1, "Row1"), scalar(column1, "Column1"), scalar(row2, "Row2"), scalar(column2, "Column2"),
        scalar(row3, "Row3"), scalar(column3, "Column3"), scalar(row4, "Row4"), scalar(column4, "Column4"),
    )])


@operator(
    "量測",
    [],
    [],
    [
        P("Row", "點的列"), P("Column", "點的行"),
        P("Row1", "直線起點的列"), P("Column1", "直線起點的行"),
        P("Row2", "直線終點的列"), P("Column2", "直線終點的行"),
    ],
    [P("RowProj", "垂足的列"), P("ColProj", "垂足的行")],
    "點在直線上的垂足",
    "直線是無限長的。兩個端點重合時，垂足就是那個點。",
    "Vizion.projection_pl (5, 4, 0, 0, 0, 10, RowProj, ColProj)",
)
def projection_pl(row, column, row1, column1, row2, column2, *, ctx):
    proj_row, proj_col, _scale = _project_line(
        scalar(row, "Row"), scalar(column, "Column"),
        scalar(row1, "Row1"), scalar(column1, "Column1"), scalar(row2, "Row2"), scalar(column2, "Column2"),
    )
    return Result([], [proj_row, proj_col])


@operator(
    "量測",
    [],
    [],
    [
        P("Row", "點的列"), P("Column", "點的行"),
        P("CenterRow", "圓心列"), P("CenterCol", "圓心行"), P("Radius", "半徑"),
    ],
    [P("Distance", "點到圓周的距離，像素")],
    "點到圓周的距離",
    "點在圓內時距離仍是正的，表示離圓周多遠。",
    "Vizion.distance_pc (110, 160, 110, 150, 42, Distance)",
)
def distance_pc(row, column, center_row, center_col, radius, *, ctx):
    return Result([], [gauge.distance_pc(
        scalar(row, "Row"), scalar(column, "Column"),
        scalar(center_row, "CenterRow"), scalar(center_col, "CenterCol"), scalar(radius, "Radius"),
    )])


@operator(
    "量測",
    [],
    [],
    [
        P("Row1", "第一條線起點的列"), P("Column1", "第一條線起點的行"),
        P("Row2", "第一條線終點的列"), P("Column2", "第一條線終點的行"),
        P("Row3", "第二條線起點的列"), P("Column3", "第二條線起點的行"),
        P("Row4", "第二條線終點的列"), P("Column4", "第二條線終點的行"),
    ],
    [P("Angle", "夾角，0 到 π/2 弳度")],
    "兩直線的夾角",
    "取較小的那個夾角，平行時為 0，垂直時為 π/2。方向相反的兩條線視為平行。",
    "Vizion.angle_ll (0, 0, 0, 10, 0, 0, 10, 0, Angle)",
)
def angle_ll(row1, column1, row2, column2, row3, column3, row4, column4, *, ctx):
    return Result([], [gauge.angle_ll(
        scalar(row1, "Row1"), scalar(column1, "Column1"), scalar(row2, "Row2"), scalar(column2, "Column2"),
        scalar(row3, "Row3"), scalar(column3, "Column3"), scalar(row4, "Row4"), scalar(column4, "Column4"),
    )])


@operator(
    "量測",
    [],
    [],
    [
        P("Row1", "第一條線起點的列"), P("Column1", "第一條線起點的行"),
        P("Row2", "第一條線終點的列"), P("Column2", "第一條線終點的行"),
        P("Row3", "第二條線起點的列"), P("Column3", "第二條線起點的行"),
        P("Row4", "第二條線終點的列"), P("Column4", "第二條線終點的行"),
    ],
    [P("Row", "交點的列"), P("Column", "交點的行")],
    "兩直線的交點",
    "兩條線視為無限長。平行時會報錯。",
    "Vizion.intersection_ll (0, 10, 50, 10, 20, 0, 20, 40, Row, Column)",
)
def intersection_ll(row1, column1, row2, column2, row3, column3, row4, column4, *, ctx):
    row, column = gauge.intersection_ll(
        scalar(row1, "Row1"), scalar(column1, "Column1"), scalar(row2, "Row2"), scalar(column2, "Column2"),
        scalar(row3, "Row3"), scalar(column3, "Column3"), scalar(row4, "Row4"), scalar(column4, "Column4"),
    )
    return Result([], [row, column])


def _line_phi(row1, column1, row2, column2) -> float:
    drow = row2 - row1
    dcol = column2 - column1
    if abs(drow) < 1e-12 and abs(dcol) < 1e-12:
        raise ValueError("兩個點重合，沒有方向")
    return math.atan2(drow, dcol)


@operator(
    "量測",
    [],
    [],
    [
        P("Row1", "起點列"), P("Column1", "起點行"),
        P("Row2", "終點列"), P("Column2", "終點行"),
    ],
    [P("Angle", "和水平方向的夾角，弳度")],
    "直線的方向",
    "水平朝右是 0。正角度在畫面上順時針，朝下是 π/2。範圍是 −π 到 π。",
    "Vizion.angle_lx (0, 0, 0, 10, Angle)",
)
def angle_lx(row1, column1, row2, column2, *, ctx):
    return Result([], [_line_phi(scalar(row1, "Row1"), scalar(column1, "Column1"), scalar(row2, "Row2"), scalar(column2, "Column2"))])


def _fold_orientation(phi: float) -> float:
    while phi > math.pi / 2:
        phi -= math.pi
    while phi < -math.pi / 2:
        phi += math.pi
    return phi


@operator(
    "量測",
    [],
    [],
    [
        P("Row1", "起點列"), P("Column1", "起點行"),
        P("Row2", "終點列"), P("Column2", "終點行"),
    ],
    [P("Phi", "無向直線的角度，弳度")],
    "無向直線的方向",
    "把 angle_lx 收到 −π/2 到 π/2。朝右和朝左都是 0，朝下是 π/2。",
    "Vizion.line_orientation (0, 0, 0, 10, Phi)",
)
def line_orientation(row1, column1, row2, column2, *, ctx):
    phi = _line_phi(scalar(row1, "Row1"), scalar(column1, "Column1"), scalar(row2, "Row2"), scalar(column2, "Column2"))
    return Result([], [_fold_orientation(phi)])


@operator(
    "量測",
    [],
    [],
    [
        P("Row1", "起點列"), P("Column1", "起點行"),
        P("Row2", "終點列"), P("Column2", "終點行"),
    ],
    [
        P("RowCenter", "中點列"), P("ColCenter", "中點行"),
        P("Length", "長度"), P("Phi", "方向，弳度"),
    ],
    "直線的中點、長度和方向",
    "方向和 angle_lx 相同。",
    "Vizion.line_position (0, 0, 0, 10, RowCenter, ColCenter, Length, Phi)",
)
def line_position(row1, column1, row2, column2, *, ctx):
    r1 = scalar(row1, "Row1")
    c1 = scalar(column1, "Column1")
    r2 = scalar(row2, "Row2")
    c2 = scalar(column2, "Column2")
    return Result([], [(r1 + r2) / 2.0, (c1 + c2) / 2.0, math.hypot(r2 - r1, c2 - c1), _line_phi(r1, c1, r2, c2)])


def _cross(row_a: float, col_a: float, row_b: float, col_b: float) -> float:
    return row_a * col_b - col_a * row_b


def _on_segment(row, column, row1, column1, row2, column2) -> bool:
    _proj_row, _proj_col, scale = _project_line(row, column, row1, column1, row2, column2)
    if scale < -1e-6 or scale > 1 + 1e-6:
        return False
    if scale < 0:
        return math.hypot(row - row1, column - column1) <= 1e-4
    if scale > 1:
        return math.hypot(row - row2, column - column2) <= 1e-4
    foot_row = row1 + scale * (row2 - row1)
    foot_col = column1 + scale * (column2 - column1)
    return math.hypot(row - foot_row, column - foot_col) <= 1e-4


@operator(
    "量測",
    [],
    [],
    [
        P("Row1", "第一條起點的列"), P("Column1", "第一條起點的行"),
        P("Row2", "第一條終點的列"), P("Column2", "第一條終點的行"),
        P("Row3", "第二條起點的列"), P("Column3", "第二條起點的行"),
        P("Row4", "第二條終點的列"), P("Column4", "第二條終點的行"),
    ],
    [P("Row", "交點列"), P("Column", "交點行"), P("IsOverlapping", "1 表示有交點")],
    "兩條線段的交點",
    "只算線段，不算無限長直線。沒有交點時座標是 0，IsOverlapping 是 0。端點落在另一條上也算相交。",
    "Vizion.intersection_segments (0, 0, 10, 10, 0, 10, 10, 0, Row, Column, IsOverlapping)",
)
def intersection_segments(row1, column1, row2, column2, row3, column3, row4, column4, *, ctx):
    r1 = scalar(row1, "Row1")
    c1 = scalar(column1, "Column1")
    r2 = scalar(row2, "Row2")
    c2 = scalar(column2, "Column2")
    r3 = scalar(row3, "Row3")
    c3 = scalar(column3, "Column3")
    r4 = scalar(row4, "Row4")
    c4 = scalar(column4, "Column4")
    drow1, dcol1 = r2 - r1, c2 - c1
    drow2, dcol2 = r4 - r3, c4 - c3
    den = _cross(drow1, dcol1, drow2, dcol2)
    if abs(den) >= 1e-9:
        t = _cross(r3 - r1, c3 - c1, drow2, dcol2) / den
        u = _cross(r3 - r1, c3 - c1, drow1, dcol1) / den
        if -1e-6 <= t <= 1 + 1e-6 and -1e-6 <= u <= 1 + 1e-6:
            return Result([], [r1 + t * drow1, c1 + t * dcol1, 1])
        return Result([], [0, 0, 0])
    for row, col in ((r1, c1), (r2, c2)):
        if _on_segment(row, col, r3, c3, r4, c4):
            return Result([], [row, col, 1])
    for row, col in ((r3, c3), (r4, c4)):
        if _on_segment(row, col, r1, c1, r2, c2):
            return Result([], [row, col, 1])
    return Result([], [0, 0, 0])


@operator(
    "量測",
    [],
    [],
    [
        P("Row1", "第一條起點的列"), P("Column1", "第一條起點的行"),
        P("Row2", "第一條終點的列"), P("Column2", "第一條終點的行"),
        P("Row3", "第二條起點的列"), P("Column3", "第二條起點的行"),
        P("Row4", "第二條終點的列"), P("Column4", "第二條終點的行"),
    ],
    [P("Row", "交點列"), P("Column", "交點行"), P("IsOverlapping", "1 表示有交點")],
    "兩條無限長直線的交點",
    "平行時沒有交點，座標是 0，IsOverlapping 是 0。這不是線段。",
    "Vizion.intersection_lines (0, 0, 0, 80, 0, 20, 40, 20, Row, Column, IsOverlapping)",
)
def intersection_lines(row1, column1, row2, column2, row3, column3, row4, column4, *, ctx):
    r1, c1 = scalar(row1, "Row1"), scalar(column1, "Column1")
    r2, c2 = scalar(row2, "Row2"), scalar(column2, "Column2")
    r3, c3 = scalar(row3, "Row3"), scalar(column3, "Column3")
    r4, c4 = scalar(row4, "Row4"), scalar(column4, "Column4")
    drow1, dcol1 = r2 - r1, c2 - c1
    drow2, dcol2 = r4 - r3, c4 - c3
    if math.hypot(drow1, dcol1) < 1e-9 or math.hypot(drow2, dcol2) < 1e-9:
        raise ValueError("直線的兩個點重疊")
    den = _cross(drow1, dcol1, drow2, dcol2)
    if abs(den) < 1e-9:
        return Result([], [0, 0, 0])
    along = _cross(r3 - r1, c3 - c1, drow2, dcol2) / den
    return Result([], [r1 + along * drow1, c1 + along * dcol1, 1])


@operator(
    "量測",
    [],
    [],
    [
        P("Row1", "線段起點的列"), P("Column1", "線段起點的行"),
        P("Row2", "線段終點的列"), P("Column2", "線段終點的行"),
        P("Row3", "直線起點的列"), P("Column3", "直線起點的行"),
        P("Row4", "直線終點的列"), P("Column4", "直線終點的行"),
    ],
    [P("Row", "交點列"), P("Column", "交點行"), P("IsOverlapping", "1 表示交在線段上")],
    "線段和無限長直線的交點",
    "第一組兩點是線段，第二組是無限長直線。交點不在線段上、或兩線平行時，座標是 0，IsOverlapping 是 0。",
    "Vizion.intersection_segment_line (10, 0, 10, 40, 0, 20, 40, 20, Row, Column, IsOverlapping)",
)
def intersection_segment_line(row1, column1, row2, column2, row3, column3, row4, column4, *, ctx):
    r1, c1 = scalar(row1, "Row1"), scalar(column1, "Column1")
    r2, c2 = scalar(row2, "Row2"), scalar(column2, "Column2")
    r3, c3 = scalar(row3, "Row3"), scalar(column3, "Column3")
    r4, c4 = scalar(row4, "Row4"), scalar(column4, "Column4")
    drow1, dcol1 = r2 - r1, c2 - c1
    drow2, dcol2 = r4 - r3, c4 - c3
    if math.hypot(drow1, dcol1) < 1e-9 or math.hypot(drow2, dcol2) < 1e-9:
        raise ValueError("直線的兩個點重疊")
    den = _cross(drow1, dcol1, drow2, dcol2)
    if abs(den) < 1e-9:
        return Result([], [0, 0, 0])
    along = _cross(r3 - r1, c3 - c1, drow2, dcol2) / den
    if along < -1e-9 or along > 1.0 + 1e-9:
        return Result([], [0, 0, 0])
    return Result([], [r1 + along * drow1, c1 + along * dcol1, 1])


@operator(
    "量測",
    [],
    [],
    [
        P("Row1", "直線起點的列"), P("Column1", "直線起點的行"),
        P("Row2", "直線終點的列"), P("Column2", "直線終點的行"),
        P("Row", "圓心列"), P("Column", "圓心行"), P("Radius", "半徑"),
    ],
    [P("RowIntersect", "交點列"), P("ColumnIntersect", "交點行")],
    "無限長直線和圓的交點",
    "沒有交點時得到空 tuple。相切時只有一個點。直線由兩個點決定。",
    "Vizion.intersection_line_circle (0, -10, 0, 10, 0, 0, 5, Row, Column)",
)
def intersection_line_circle(row1, column1, row2, column2, row, column, radius, *, ctx):
    r1 = scalar(row1, "Row1")
    c1 = scalar(column1, "Column1")
    r2 = scalar(row2, "Row2")
    c2 = scalar(column2, "Column2")
    center_row = scalar(row, "Row")
    center_col = scalar(column, "Column")
    rad = scalar(radius, "Radius")
    if rad < 0:
        raise ValueError("半徑不能是負的")
    drow, dcol = r2 - r1, c2 - c1
    quad_a = drow * drow + dcol * dcol
    if quad_a < 1e-12:
        raise ValueError("直線的兩個點不能重合")
    rel_row, rel_col = r1 - center_row, c1 - center_col
    quad_b = 2.0 * (rel_row * drow + rel_col * dcol)
    quad_c = rel_row * rel_row + rel_col * rel_col - rad * rad
    disc = quad_b * quad_b - 4.0 * quad_a * quad_c
    if disc < -1e-6:
        return Result([], [[], []])
    if disc < 0:
        disc = 0.0
    root = math.sqrt(disc)
    times = [(-quad_b - root) / (2.0 * quad_a)]
    if root > 1e-6:
        times.append((-quad_b + root) / (2.0 * quad_a))
    rows = [r1 + item * drow for item in times]
    cols = [c1 + item * dcol for item in times]
    return Result([], [rows, cols])


@operator(
    "量測",
    [],
    [],
    [
        P("Row1", "線段起點的列"), P("Column1", "線段起點的行"),
        P("Row2", "線段終點的列"), P("Column2", "線段終點的行"),
        P("Row3", "直線起點的列"), P("Column3", "直線起點的行"),
        P("Row4", "直線終點的列"), P("Column4", "直線終點的行"),
    ],
    [P("Distance", "線段到無限長直線的距離")],
    "線段到直線的距離",
    "線段穿過直線時距離是 0。否則是兩個端點裡較近的那個垂直距離。",
    "Vizion.distance_sl (0, 0, 0, 10, 5, 0, 5, 20, Distance)",
)
def distance_sl(row1, column1, row2, column2, row3, column3, row4, column4, *, ctx):
    r1 = scalar(row1, "Row1")
    c1 = scalar(column1, "Column1")
    r2 = scalar(row2, "Row2")
    c2 = scalar(column2, "Column2")
    r3 = scalar(row3, "Row3")
    c3 = scalar(column3, "Column3")
    r4 = scalar(row4, "Row4")
    c4 = scalar(column4, "Column4")
    drow, dcol = r4 - r3, c4 - c3
    length = math.hypot(drow, dcol)
    if length < 1e-12:
        raise ValueError("直線的兩個點不能重合")

    def side(row: float, col: float) -> float:
        return _cross(drow, dcol, row - r3, col - c3) / length

    side1 = side(r1, c1)
    side2 = side(r2, c2)
    if side1 * side2 <= 0:
        return Result([], [0.0])
    return Result([], [min(abs(side1), abs(side2))])


def _circle_hits(row1, col1, row2, col2, center_row, center_col, radius):
    if radius < 0:
        raise ValueError("半徑不能是負的")
    drow, dcol = row2 - row1, col2 - col1
    quad_a = drow * drow + dcol * dcol
    if quad_a < 1e-12:
        raise ValueError("兩個點不能重合")
    rel_row, rel_col = row1 - center_row, col1 - center_col
    quad_b = 2.0 * (rel_row * drow + rel_col * dcol)
    quad_c = rel_row * rel_row + rel_col * rel_col - radius * radius
    disc = quad_b * quad_b - 4.0 * quad_a * quad_c
    if disc < -1e-6:
        return []
    root = math.sqrt(max(disc, 0.0))
    times = [(-quad_b - root) / (2.0 * quad_a)]
    if root > 1e-6:
        times.append((-quad_b + root) / (2.0 * quad_a))
    return [(item, row1 + item * drow, col1 + item * dcol) for item in times]


@operator(
    "量測",
    [],
    [],
    [
        P("Row1", "第一個圓心的列"), P("Column1", "第一個圓心的行"), P("Radius1", "第一個半徑"),
        P("Row2", "第二個圓心的列"), P("Column2", "第二個圓心的行"), P("Radius2", "第二個半徑"),
    ],
    [P("Row", "交點列"), P("Column", "交點行")],
    "兩個圓的交點",
    "沒有交點時得到空 tuple。相切時只有一個點。兩圓完全重合會失敗。",
    "Vizion.intersection_circles (0, 0, 5, 0, 6, 5, Row, Column)",
)
def intersection_circles(row1, column1, radius1, row2, column2, radius2, *, ctx):
    center1_row = scalar(row1, "Row1")
    center1_col = scalar(column1, "Column1")
    rad1 = scalar(radius1, "Radius1")
    center2_row = scalar(row2, "Row2")
    center2_col = scalar(column2, "Column2")
    rad2 = scalar(radius2, "Radius2")
    if rad1 < 0 or rad2 < 0:
        raise ValueError("半徑不能是負的")
    drow = center2_row - center1_row
    dcol = center2_col - center1_col
    dist = math.hypot(drow, dcol)
    if dist < 1e-9:
        if abs(rad1 - rad2) < 1e-6:
            raise ValueError("兩圓重合，交點無限多")
        return Result([], [[], []])
    if dist > rad1 + rad2 + 1e-6 or dist < abs(rad1 - rad2) - 1e-6:
        return Result([], [[], []])
    along = (rad1 * rad1 - rad2 * rad2 + dist * dist) / (2.0 * dist)
    height = math.sqrt(max(rad1 * rad1 - along * along, 0.0))
    mid_row = center1_row + along * drow / dist
    mid_col = center1_col + along * dcol / dist
    if height < 1e-6:
        return Result([], [[mid_row], [mid_col]])
    off_row = -height * dcol / dist
    off_col = height * drow / dist
    return Result([], [[mid_row + off_row, mid_row - off_row], [mid_col + off_col, mid_col - off_col]])


@operator(
    "量測",
    [],
    [],
    [
        P("Row1", "線段起點的列"), P("Column1", "線段起點的行"),
        P("Row2", "線段終點的列"), P("Column2", "線段終點的行"),
        P("Row", "圓心列"), P("Column", "圓心行"), P("Radius", "半徑"),
    ],
    [P("RowIntersect", "交點列"), P("ColumnIntersect", "交點行")],
    "線段和圓的交點",
    "只算線段，不算無限長直線。沒有交點時得到空 tuple。",
    "Vizion.intersection_segment_circle (0, -10, 0, 10, 0, 0, 5, Row, Column)",
)
def intersection_segment_circle(row1, column1, row2, column2, row, column, radius, *, ctx):
    hits = _circle_hits(
        scalar(row1, "Row1"),
        scalar(column1, "Column1"),
        scalar(row2, "Row2"),
        scalar(column2, "Column2"),
        scalar(row, "Row"),
        scalar(column, "Column"),
        scalar(radius, "Radius"),
    )
    kept = [item for item in hits if -1e-6 <= item[0] <= 1 + 1e-6]
    return Result([], [[item[1] for item in kept], [item[2] for item in kept]])


GROUP_ORDER = [
    "檔案", "系統", "灰階與濾波", "幾何", "分割", "形態學", "區域", "特徵", "輪廓", "量測",
    "文字", "條碼", "相機", "通訊", "標定", "三維", "分類", "形狀匹配", "NCC定位", "顯示", "矩陣", "Tuple",
]

GROUP_PRINCIPLE = {
    "檔案": "影像以 8 位元讀寫。文字檔的相對路徑從專案目錄起算。",
    "系統": "讀這台電腦的時間、處理器數量和單調秒數。等待會停住這一步，最長 30 秒。",
    "灰階與濾波": "對每個像素做灰階運算，位置不變，輸出仍是影像。",
    "幾何": "用齊次矩陣描述平移、旋轉、縮放或投影，再套到點、區域或影像。",
    "分割": "依灰階門檻或相連的像素，把前景和背景分開。",
    "形態學": "用結構元素膨脹、侵蝕、開運算或閉運算，改變區域外形。",
    "區域": "對區域做集合、選取、移動或產生，不改影像的灰階。",
    "特徵": "由區域的面積、外框、矩或灰階統計，算出一個數值。",
    "輪廓": "把邊界或擬合結果存成 XLD 點列，點的順序是欄、列。",
    "量測": "沿一條線或一個形狀取出灰階剖面，再決定邊緣落在哪個位置。",
    "文字": "先在影像上找出字的位置，再把筆畫讀成字串。",
    "條碼": "依條碼或二維碼的模組規則畫出符號，或從影像解回字串。",
    "相機": "打開相機裝置，抓一張畫面進來。",
    "通訊": "用 Windows API 開關序列埠，並用本機 socket 連線讀寫位元組。loopback 只在這支程式裡把寫出的資料讀回來。",
    "三維": "對點雲或高度圖計算平面、距離或曲面。",
    "分類": "用樣本的特徵，把物件分成事先定好的類別。",
    "形狀匹配": "把教導過的邊緣和場景比對，找出位置、角度和尺度。",
    "NCC定位": "用正規化互相關，在灰階圖上找教導過的樣板。",
    "顯示": "把影像、區域或輪廓畫到畫布，不改已經算出的數值。",
    "矩陣": "對數值矩陣做建立、運算或分解。",
    "Tuple": "對控制用的數值或字串序列做索引、篩選和運算。tuple 算子的索引從 0 開始。",
}


def list_operators() -> list[dict]:
    def sort_key(op: Operator):
        try:
            group_index = GROUP_ORDER.index(op.group)
        except ValueError:
            group_index = 99
        return (group_index, op.order)

    return [op.to_dict() for op in sorted(REGISTRY.values(), key=sort_key)]


from . import chapters as _chapters  # noqa: E402,F401
