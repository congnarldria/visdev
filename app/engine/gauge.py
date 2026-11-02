"""2D subpixel gauges.

Point, line, circle, rectangle, wedge and polygon gauges sample a gray
profile, pick a gradient peak, and refine it with a parabola. Geometry
helpers measure distance, angle and intersection in pixel coordinates.
Angles are radians. This is an OpenCV-style measurement set, not a copy
of a commercial gauge library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_TRANSITIONS = {
    "all": "all",
    "any": "all",
    "全部": "all",
    "所有": "all",
    "positive": "positive",
    "pos": "positive",
    "+": "positive",
    "dark_to_light": "positive",
    "由暗到亮": "positive",
    "negative": "negative",
    "neg": "negative",
    "-": "negative",
    "light_to_dark": "negative",
    "由亮到暗": "negative",
}

_SELECTS = {
    "all": "all",
    "any": "all",
    "全部": "all",
    "所有": "all",
    "first": "first",
    "第一": "first",
    "第一個": "first",
    "last": "last",
    "最後": "last",
    "最後一個": "last",
    "best": "best",
    "strongest": "best",
    "最強": "best",
    "最大": "best",
}


@dataclass
class Hit:
    row: float
    column: float
    amplitude: float
    distance: float


@dataclass
class LineFit:
    row1: float
    column1: float
    row2: float
    column2: float
    angle: float
    used: int
    contour: list


@dataclass
class CircleFit:
    row: float
    column: float
    radius: float
    used: int
    contour: list


@dataclass
class RectFit:
    row: float
    column: float
    phi: float
    length1: float
    length2: float
    used: int
    contour: list


@dataclass
class WedgeFit:
    row: float
    column: float
    radius1: float
    radius2: float
    angle_start: float
    angle_extent: float
    used: int
    contour: list


@dataclass
class PolygonFit:
    rows: list
    columns: list
    used: int
    contour: list


def measure_point(gray, domain, row1, col1, row2, col2, sigma, threshold, transition, select) -> list[Hit]:
    gray, domain = _prepare(gray, domain)
    transition = _choice(transition, _TRANSITIONS, "Transition")
    select = _choice(select, _SELECTS, "Select")
    sigma = _number("Sigma", sigma, 0)
    threshold = _number("Threshold", threshold, 0)
    hits = _pick(
        _locate(gray, domain, float(col1), float(row1), float(col2), float(row2), sigma, threshold, transition),
        select,
    )
    if not hits:
        raise ValueError("點量規找不到邊緣。請加長搜尋段、放寬 Transition，或降低 Threshold")
    return hits


def measure_line(gray, domain, row1, col1, row2, col2, tolerance, sigma, threshold, transition, select, width=0.0) -> LineFit:
    gray, domain = _prepare(gray, domain)
    transition = _choice(transition, _TRANSITIONS, "Transition")
    select = _choice(select, _SELECTS, "Select")
    tolerance = _number("Tolerance", tolerance, 1)
    sigma = _number("Sigma", sigma, 0)
    threshold = _number("Threshold", threshold, 0)
    x0, y0, x1, y1 = float(col1), float(row1), float(col2), float(row2)
    dx, dy, length = _unit(x1 - x0, y1 - y0, "直線量規至少要 2 像素")
    sx0, sy0, sx1, sy1 = _shrink(x0, y0, x1, y1, min(4.0, length * 0.08))
    points = _along(gray, domain, sx0, sy0, sx1, sy1, tolerance, sigma, threshold, transition, select, width)
    if len(points) < 2:
        raise ValueError("直線量規找不到足夠的邊緣")
    origin, direction, kept = _refit_line(points, dx, dy)
    start = _project(origin, direction, x0, y0)
    end = _project(origin, direction, x1, y1)
    return LineFit(
        float(start[1]),
        float(start[0]),
        float(end[1]),
        float(end[0]),
        math.atan2(direction[1], direction[0]),
        len(kept),
        [[start[0], start[1]], [end[0], end[1]]],
    )


def measure_circle(gray, domain, row, col, radius, tolerance, sigma, threshold, transition, select, width=0.0) -> CircleFit:
    gray, domain = _prepare(gray, domain)
    transition = _choice(transition, _TRANSITIONS, "Transition")
    select = _choice(select, _SELECTS, "Select")
    tolerance = _number("Tolerance", tolerance, 1)
    sigma = _number("Sigma", sigma, 0)
    threshold = _number("Threshold", threshold, 0)
    row = float(row)
    col = float(col)
    radius = _number("Radius", radius, 1)
    points = _circle_points(gray, domain, row, col, radius, tolerance, sigma, threshold, transition, select, width)
    if len(points) < 3:
        raise ValueError("圓量規找不到足夠的邊緣")
    fitted_row, fitted_col, fitted_radius, kept = _refit_circle(points, row, col, radius, tolerance)
    return CircleFit(fitted_row, fitted_col, fitted_radius, len(kept), _arc(fitted_row, fitted_col, fitted_radius, 0.0, math.tau, 96))


def measure_rectangle(
    gray, domain, row, col, phi, length1, length2, tolerance, sigma, threshold, transition, select
) -> RectFit:
    gray, domain = _prepare(gray, domain)
    transition = _choice(transition, _TRANSITIONS, "Transition")
    select = _choice(select, _SELECTS, "Select")
    tolerance = _number("Tolerance", tolerance, 1)
    sigma = _number("Sigma", sigma, 0)
    threshold = _number("Threshold", threshold, 0)
    row = float(row)
    col = float(col)
    phi = float(phi)
    length1 = _number("Length1", length1, 1)
    length2 = _number("Length2", length2, 1)
    corners, _u, _v = _rect_corners(row, col, phi, length1, length2)
    lines = []
    used = 0
    for index, (start, end) in enumerate(_sides(corners)):
        points = _side_points(gray, domain, start, end, tolerance, sigma, threshold, transition, select)
        if len(points) < 2:
            raise ValueError(f"矩形的第 {index + 1} 邊找不到邊緣")
        direction = _unit(end[0] - start[0], end[1] - start[1], "矩形的邊長度是 0")
        origin, direction_xy, kept = _refit_line(points, direction[0], direction[1])
        lines.append((origin, direction_xy))
        used += len(kept)
    fitted = []
    for index in range(4):
        point = _intersect_lines(lines[(index - 1) % 4], lines[index])
        if point is None:
            raise ValueError("矩形的四邊擬合後沒有交點")
        fitted.append(point)
    center = (
        sum(point[0] for point in fitted) / 4,
        sum(point[1] for point in fitted) / 4,
    )
    u_vec = ((fitted[0][0] + fitted[3][0]) / 2 - center[0], (fitted[0][1] + fitted[3][1]) / 2 - center[1])
    v_vec = ((fitted[0][0] + fitted[1][0]) / 2 - center[0], (fitted[0][1] + fitted[1][1]) / 2 - center[1])
    nominal = (math.cos(phi), math.sin(phi))
    if u_vec[0] * nominal[0] + u_vec[1] * nominal[1] < 0:
        u_vec = (-u_vec[0], -u_vec[1])
    contour = [[point[0], point[1]] for point in fitted]
    contour.append(contour[0])
    return RectFit(
        float(center[1]),
        float(center[0]),
        math.atan2(u_vec[1], u_vec[0]),
        math.hypot(u_vec[0], u_vec[1]),
        math.hypot(v_vec[0], v_vec[1]),
        used,
        contour,
    )


def measure_wedge(
    gray, domain, row, col, radius1, radius2, angle_start, angle_extent, tolerance, sigma, threshold, transition, select
) -> WedgeFit:
    gray, domain = _prepare(gray, domain)
    transition = _choice(transition, _TRANSITIONS, "Transition")
    select = _choice(select, _SELECTS, "Select")
    tolerance = _number("Tolerance", tolerance, 1)
    sigma = _number("Sigma", sigma, 0)
    threshold = _number("Threshold", threshold, 0)
    row = float(row)
    col = float(col)
    radius1 = _number("Radius1", radius1, 1)
    radius2 = _number("Radius2", radius2, 0)
    if radius2 <= radius1:
        raise ValueError("Radius2 必須大於 Radius1")
    start = float(angle_start)
    extent = float(angle_extent)
    if abs(extent) < 0.05 or abs(extent) > math.tau - 0.05:
        raise ValueError("AngleExtent 要介於 ±0.05 和 ±6.2 弳度之間")
    outer = _arc_points(gray, domain, row, col, radius2, start, extent, tolerance, sigma, threshold, transition, select, 0.08)
    inner = _arc_points(gray, domain, row, col, radius1, start, extent, tolerance, sigma, threshold, transition, select, 0.08)
    if len(outer) < 3:
        raise ValueError("扇形的外弧找不到足夠的邊緣")
    if len(inner) < 3:
        raise ValueError("扇形的內弧找不到足夠的邊緣")
    begin = _radial_points(gray, domain, row, col, radius1, radius2, start, tolerance, sigma, threshold, transition, select)
    end = _radial_points(
        gray, domain, row, col, radius1, radius2, start + extent, tolerance, sigma, threshold, transition, select
    )
    if len(begin) < 2:
        raise ValueError("扇形的起始邊找不到邊緣")
    if len(end) < 2:
        raise ValueError("扇形的結束邊找不到邊緣")
    fitted_row, fitted_col, _radius, kept = _refit_circle(outer, row, col, radius2, tolerance)
    radius_out = _median_radius(kept, fitted_row, fitted_col)
    radius_in = _median_radius(inner, fitted_row, fitted_col)
    angle0 = _mean_angle(begin, fitted_row, fitted_col)
    angle1 = _mean_angle(end, fitted_row, fitted_col)
    if extent >= 0:
        sweep = (angle1 - angle0) % math.tau
    else:
        sweep = -((angle0 - angle1) % math.tau)
    contour = _arc(fitted_row, fitted_col, radius_out, angle0, sweep, 48)
    contour += _arc(fitted_row, fitted_col, radius_in, angle0, sweep, 32)[::-1]
    contour.append(contour[0])
    return WedgeFit(
        fitted_row,
        fitted_col,
        radius_in,
        radius_out,
        angle0,
        sweep,
        len(kept) + len(inner) + len(begin) + len(end),
        contour,
    )


def measure_polygon(gray, domain, rows, cols, tolerance, sigma, threshold, transition, select, closed) -> PolygonFit:
    gray, domain = _prepare(gray, domain)
    transition = _choice(transition, _TRANSITIONS, "Transition")
    select = _choice(select, _SELECTS, "Select")
    tolerance = _number("Tolerance", tolerance, 1)
    sigma = _number("Sigma", sigma, 0)
    threshold = _number("Threshold", threshold, 0)
    rows = [float(item) for item in rows]
    cols = [float(item) for item in cols]
    if len(rows) != len(cols):
        raise ValueError("Rows 和 Columns 的個數要相同")
    if len(rows) >= 2 and abs(rows[0] - rows[-1]) < 1e-3 and abs(cols[0] - cols[-1]) < 1e-3:
        rows = rows[:-1]
        cols = cols[:-1]
        closed = True
    if len(rows) < (3 if closed else 2):
        raise ValueError("封閉多邊形至少要 3 個頂點" if closed else "多邊形至少要 2 個頂點")
    points = list(zip(cols, rows))
    edge_count = len(points) if closed else len(points) - 1
    lines = []
    used = 0
    for index in range(edge_count):
        start = points[index]
        end = points[(index + 1) % len(points)]
        found = _side_points(gray, domain, start, end, tolerance, sigma, threshold, transition, select)
        if len(found) < 2:
            raise ValueError(f"多邊形的第 {index + 1} 邊找不到邊緣")
        direction = _unit(end[0] - start[0], end[1] - start[1], "多邊形有一邊的長度是 0")
        origin, direction_xy, kept = _refit_line(found, direction[0], direction[1])
        lines.append((origin, direction_xy, start, end))
        used += len(kept)
    vertices = []
    if closed:
        for index in range(edge_count):
            point = _intersect_lines(lines[(index - 1) % edge_count][:2], lines[index][:2])
            vertices.append(point if point is not None else points[index])
    else:
        vertices.append(_project(lines[0][0], lines[0][1], points[0][0], points[0][1]))
        for index in range(1, edge_count):
            point = _intersect_lines(lines[index - 1][:2], lines[index][:2])
            vertices.append(point if point is not None else points[index])
        vertices.append(_project(lines[-1][0], lines[-1][1], points[-1][0], points[-1][1]))
    contour = [[point[0], point[1]] for point in vertices]
    if closed:
        contour.append(contour[0])
    return PolygonFit([point[1] for point in vertices], [point[0] for point in vertices], used, contour)


def distance_pp(row1, col1, row2, col2) -> float:
    return math.hypot(float(row2) - float(row1), float(col2) - float(col1))


def distance_pl(row, col, row1, col1, row2, col2) -> float:
    dx, dy, length = _unit(float(col2) - float(col1), float(row2) - float(row1), "直線的兩個點重疊")
    return abs((float(col) - float(col1)) * dy - (float(row) - float(row1)) * dx)


def distance_pc(row, col, center_row, center_col, radius) -> float:
    return abs(math.hypot(float(row) - float(center_row), float(col) - float(center_col)) - abs(float(radius)))


def angle_ll(row1, col1, row2, col2, row3, col3, row4, col4) -> float:
    ax, ay, _na = _unit(float(col2) - float(col1), float(row2) - float(row1), "第一條線的兩個點重疊")
    bx, by, _nb = _unit(float(col4) - float(col3), float(row4) - float(row3), "第二條線的兩個點重疊")
    dot = abs(ax * bx + ay * by)
    return math.acos(min(1.0, max(0.0, dot)))


def intersection_ll(row1, col1, row2, col2, row3, col3, row4, col4) -> tuple[float, float]:
    d1 = (float(col2) - float(col1), float(row2) - float(row1))
    d2 = (float(col4) - float(col3), float(row4) - float(row3))
    det = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(det) < 1e-8:
        raise ValueError("兩條線平行，沒有交點")
    t = ((float(col3) - float(col1)) * d2[1] - (float(row3) - float(row1)) * d2[0]) / det
    return float(row1) + d1[1] * t, float(col1) + d1[0] * t


def _prepare(gray, domain):
    array = np.asarray(gray)
    if array.ndim != 2:
        raise ValueError("量測需要灰階影像")
    if domain is not None and tuple(np.asarray(domain).shape[:2]) != tuple(array.shape[:2]):
        domain = None
    return array, None if domain is None else np.asarray(domain)


def _choice(value, table: dict, name: str) -> str:
    raw = str(value).strip().strip("'\"")
    found = table.get(raw.lower()) or table.get(raw)
    if found is None:
        raise ValueError(f"{name} 不認得 {value}")
    return found


def _number(name: str, value, minimum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 不是數值") from exc
    if number < minimum:
        raise ValueError(f"{name} 不能小於 {minimum}")
    return number


def _unit(dx: float, dy: float, message: str) -> tuple[float, float, float]:
    length = math.hypot(dx, dy)
    if length < 1e-6:
        raise ValueError(message)
    return dx / length, dy / length, length


def _sample(gray, domain, x: float, y: float) -> float:
    height, width = gray.shape
    if x < 0 or y < 0 or x > width - 1 or y > height - 1:
        return math.nan
    if domain is not None:
        ix = min(max(int(round(x)), 0), width - 1)
        iy = min(max(int(round(y)), 0), height - 1)
        if domain[iy, ix] <= 0:
            return math.nan
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    dx = x - x0
    dy = y - y0
    top = float(gray[y0, x0]) * (1 - dx) + float(gray[y0, x1]) * dx
    bottom = float(gray[y1, x0]) * (1 - dx) + float(gray[y1, x1]) * dx
    return top * (1 - dy) + bottom * dy


def _cubic_kernel(offset: np.ndarray) -> np.ndarray:
    span = np.abs(np.asarray(offset, np.float64))
    cube = span * span * span
    square = span * span
    return np.where(
        span <= 1.0,
        1.5 * cube - 2.5 * square + 1.0,
        np.where(span < 2.0, -0.5 * cube + 2.5 * square - 4.0 * span + 2.0, 0.0),
    )


def sample_bicubic(planes: np.ndarray, rows, cols, *, border: str = "replicate") -> np.ndarray:
    """Keys cubic, a = -0.5. rows and cols may be any shape. planes is gray or color.

    replicate clamps the 16 taps to the image. constant treats anything outside as 0.
    """
    rows = np.asarray(rows, np.float64)
    cols = np.asarray(cols, np.float64)
    shape = rows.shape
    flat_rows = rows.reshape(-1)
    flat_cols = cols.reshape(-1)
    height, width = planes.shape[:2]
    row0 = np.floor(flat_rows).astype(np.int32)
    col0 = np.floor(flat_cols).astype(np.int32)
    extra = planes.shape[2:]
    total = np.zeros((flat_rows.size,) + extra, np.float64)
    for row_offset in range(-1, 3):
        row_weight = _cubic_kernel(row_offset - (flat_rows - row0))
        raw_row = row0 + row_offset
        row_index = np.clip(raw_row, 0, height - 1)
        for col_offset in range(-1, 3):
            col_weight = _cubic_kernel(col_offset - (flat_cols - col0))
            raw_col = col0 + col_offset
            col_index = np.clip(raw_col, 0, width - 1)
            picked = planes[row_index, col_index].astype(np.float64)
            weight = row_weight * col_weight
            if border == "constant":
                inside = (raw_row >= 0) & (raw_row < height) & (raw_col >= 0) & (raw_col < width)
                weight = weight * inside.astype(np.float64)
            if picked.ndim > 1:
                weight = weight.reshape(weight.shape + (1,) * (picked.ndim - 1))
            total += picked * weight
    return total.reshape(shape + extra)


def _smooth(values: np.ndarray, sigma_px: float, step: float) -> np.ndarray:
    if sigma_px <= 0.05 or len(values) < 5:
        return values
    sigma = max(sigma_px / step, 0.25)
    radius = min(max(1, int(round(sigma * 3))), (len(values) - 1) // 2)
    kernel_x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (kernel_x / sigma) ** 2)
    kernel /= kernel.sum()
    filled = np.where(np.isfinite(values), values, 0.0)
    weight = np.where(np.isfinite(values), 1.0, 0.0)
    den = np.convolve(weight, kernel, mode="same")
    out = np.full_like(values, np.nan)
    ok = den > 0.5
    out[ok] = np.convolve(filled, kernel, mode="same")[ok] / den[ok]
    return out


def _locate(gray, domain, x0, y0, x1, y1, sigma, threshold, transition, width=0.0) -> list[Hit]:
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1.5:
        return []
    step = 0.5
    count = int(length / step) + 1
    ts = np.linspace(0.0, length, count)
    dx = (x1 - x0) / length
    dy = (y1 - y0) / length
    xs = x0 + dx * ts
    ys = y0 + dy * ts
    half = max(0, int(round(float(width))))
    if half == 0:
        values = np.array([_sample(gray, domain, float(x), float(y)) for x, y in zip(xs, ys)], dtype=np.float64)
    else:
        tangent_x, tangent_y = -dy, dx
        values = np.empty(count, np.float64)
        for index, (x, y) in enumerate(zip(xs, ys)):
            samples = [
                _sample(gray, domain, float(x + tangent_x * offset), float(y + tangent_y * offset))
                for offset in range(-half, half + 1)
            ]
            finite = [item for item in samples if math.isfinite(item)]
            values[index] = float(np.mean(finite)) if finite else math.nan
    values = _smooth(values, sigma, step)
    grad = np.full(count, np.nan)
    for index in range(1, count - 1):
        if np.isfinite(values[index - 1]) and np.isfinite(values[index + 1]):
            grad[index] = (values[index + 1] - values[index - 1]) / (2 * step)
    spacing = float(ts[1] - ts[0])
    hits = []
    for index in range(2, count - 2):
        left, mid, right = grad[index - 1], grad[index], grad[index + 1]
        if not np.isfinite([left, mid, right]).all():
            continue
        positive = mid >= threshold and mid >= left and mid > right
        negative = mid <= -threshold and mid <= left and mid < right
        if transition == "positive" and not positive:
            continue
        if transition == "negative" and not negative:
            continue
        if transition == "all" and not (positive or negative):
            continue
        denom = left - 2 * mid + right
        delta = 0.0 if abs(denom) < 1e-9 else max(-1.0, min(1.0, 0.5 * (left - right) / denom))
        at = index + delta
        i0 = int(math.floor(at))
        i1 = min(i0 + 1, count - 1)
        frac = at - i0
        hits.append(
            Hit(
                float(ys[i0] * (1 - frac) + ys[i1] * frac),
                float(xs[i0] * (1 - frac) + xs[i1] * frac),
                float(mid),
                float(at * spacing),
            )
        )
    return hits


def _pick(hits: list[Hit], select: str) -> list[Hit]:
    if not hits:
        return []
    ordered = sorted(hits, key=lambda item: item.distance)
    if select == "all":
        return ordered
    if select == "first":
        return [ordered[0]]
    if select == "last":
        return [ordered[-1]]
    return [max(ordered, key=lambda item: abs(item.amplitude))]


def _along(gray, domain, x0, y0, x1, y1, tolerance, sigma, threshold, transition, select, width=0.0) -> list:
    dx, dy, length = _unit(x1 - x0, y1 - y0, "量測段至少要 2 像素")
    nx, ny = -dy, dx
    count = int(min(56, max(8, round(length / 3.5))))
    choice = "best" if select == "all" else select
    points = []
    for index in range(count):
        t = length * (index + 0.5) / count
        cx = x0 + dx * t
        cy = y0 + dy * t
        found = _pick(
            _locate(
                gray,
                domain,
                cx - nx * tolerance,
                cy - ny * tolerance,
                cx + nx * tolerance,
                cy + ny * tolerance,
                sigma,
                threshold,
                transition,
                width,
            ),
            choice,
        )
        points.extend([hit.column, hit.row] for hit in found)
    return points


def _shrink(x0, y0, x1, y1, pixels: float):
    dx, dy, length = _unit(x1 - x0, y1 - y0, "量測段至少要 2 像素")
    if length <= pixels * 2 + 2:
        return x0, y0, x1, y1
    return x0 + dx * pixels, y0 + dy * pixels, x1 - dx * pixels, y1 - dy * pixels


def _inset_segment(start, end, fraction=0.12):
    cut = math.hypot(end[0] - start[0], end[1] - start[1]) * fraction
    return _shrink(start[0], start[1], end[0], end[1], cut)


def _side_points(gray, domain, start, end, tolerance, sigma, threshold, transition, select) -> list:
    x0, y0, x1, y1 = _inset_segment(start, end)
    return _along(gray, domain, x0, y0, x1, y1, tolerance, sigma, threshold, transition, select)


def _svd_line(points, nx, ny):
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    center = pts.mean(axis=0)
    _u, singular, vt = np.linalg.svd(pts - center, full_matrices=False)
    if float(singular[0]) < 1e-6:
        raise ValueError("邊緣點擠在一起，無法擬合直線")
    direction = vt[0]
    if direction[0] * nx + direction[1] * ny < 0:
        direction = -direction
    return center, direction


def _line_inliers(points, origin, direction, limit):
    kept = []
    for point in points:
        dist = abs((point[0] - origin[0]) * direction[1] - (point[1] - origin[1]) * direction[0])
        if dist <= limit:
            kept.append(point)
    return kept


def _refit_line(points, nx, ny):
    origin, direction = _svd_line(points, nx, ny)
    kept = _line_inliers(points, origin, direction, 2.0)
    if len(kept) >= 2 and len(kept) < len(points):
        origin, direction = _svd_line(kept, nx, ny)
        return origin, direction, kept
    return origin, direction, points


def _project(origin, direction, x, y):
    t = (x - origin[0]) * direction[0] + (y - origin[1]) * direction[1]
    return (float(origin[0] + direction[0] * t), float(origin[1] + direction[1] * t))


def _fit_circle(points, guess_row, guess_col, guess_radius):
    xy = np.asarray(points, np.float64).reshape(-1, 2)
    center = np.array([guess_col, guess_row], np.float64)
    radius = float(guess_radius)
    for _ in range(12):
        delta = xy - center
        dist = np.maximum(np.hypot(delta[:, 0], delta[:, 1]), 1e-6)
        residual = dist - radius
        jac = np.column_stack([-delta[:, 0] / dist, -delta[:, 1] / dist, -np.ones(len(xy))])
        step, *_rest = np.linalg.lstsq(jac, residual, rcond=None)
        if not np.isfinite(step).all():
            break
        move = float(np.hypot(step[0], step[1]))
        if move > 6:
            step *= 6 / move
        center = center - step[:2]
        radius = max(0.5, float(radius - step[2]))
        if move < 1e-4 and abs(float(step[2])) < 1e-4:
            break
    return float(center[1]), float(center[0]), float(radius)


def _clamp_center(row, col, nominal_row, nominal_col, tolerance):
    dr = row - nominal_row
    dc = col - nominal_col
    dist = math.hypot(dr, dc)
    if dist <= tolerance or dist < 1e-9:
        return row, col
    scale = tolerance / dist
    return nominal_row + dr * scale, nominal_col + dc * scale


def _circle_inliers(points, row, col, radius, limit):
    kept = []
    for point in points:
        if abs(math.hypot(point[0] - col, point[1] - row) - radius) <= limit:
            kept.append(point)
    return kept


def _refit_circle(points, row, col, radius, tolerance):
    fitted_row, fitted_col, fitted_radius = _fit_circle(points, row, col, radius)
    kept = _circle_inliers(points, fitted_row, fitted_col, fitted_radius, 2.0)
    if len(kept) >= 3 and len(kept) < len(points):
        fitted_row, fitted_col, fitted_radius = _fit_circle(kept, fitted_row, fitted_col, fitted_radius)
    else:
        kept = points
    fitted_row, fitted_col = _clamp_center(fitted_row, fitted_col, row, col, tolerance)
    fitted_radius = min(max(fitted_radius, radius - tolerance), radius + tolerance)
    return fitted_row, fitted_col, fitted_radius, kept


def _circle_points(gray, domain, row, col, radius, tolerance, sigma, threshold, transition, select, width=0.0):
    count = int(min(72, max(24, round(math.tau * radius / 4))))
    choice = "best" if select == "all" else select
    points = []
    inner = max(0.0, radius - tolerance)
    outer = radius + tolerance
    for index in range(count):
        angle = math.tau * index / count
        dx = math.cos(angle)
        dy = math.sin(angle)
        found = _pick(
            _locate(
                gray,
                domain,
                col + dx * inner,
                row + dy * inner,
                col + dx * outer,
                row + dy * outer,
                sigma,
                threshold,
                transition,
                width,
            ),
            choice,
        )
        points.extend([hit.column, hit.row] for hit in found)
    return points


def _arc_points(gray, domain, row, col, radius, start, extent, tolerance, sigma, threshold, transition, select, margin):
    count = int(min(48, max(8, round(abs(extent) * radius / 4))))
    choice = "best" if select == "all" else select
    points = []
    inner = max(0.0, radius - tolerance)
    outer = radius + tolerance
    for index in range(count):
        angle = start + extent * (margin + (1 - 2 * margin) * (index + 0.5) / count)
        dx = math.cos(angle)
        dy = math.sin(angle)
        found = _pick(
            _locate(
                gray,
                domain,
                col + dx * inner,
                row + dy * inner,
                col + dx * outer,
                row + dy * outer,
                sigma,
                threshold,
                transition,
            ),
            choice,
        )
        points.extend([hit.column, hit.row] for hit in found)
    return points


def _radial_points(gray, domain, row, col, radius1, radius2, angle, tolerance, sigma, threshold, transition, select):
    span = radius2 - radius1
    near = radius1 + span * 0.15
    far = radius2 - span * 0.15
    dx = math.cos(angle)
    dy = math.sin(angle)
    tx, ty = -dy, dx
    return _along_tangent(
        gray, domain, row, col, near, far, dx, dy, tx, ty, tolerance, sigma, threshold, transition, select
    )


def _along_tangent(gray, domain, row, col, near, far, dx, dy, tx, ty, tolerance, sigma, threshold, transition, select):
    length = far - near
    count = int(min(24, max(6, round(length / 3.5))))
    choice = "best" if select == "all" else select
    points = []
    for index in range(count):
        radius = near + length * (index + 0.5) / count
        cx = col + dx * radius
        cy = row + dy * radius
        found = _pick(
            _locate(
                gray,
                domain,
                cx - tx * tolerance,
                cy - ty * tolerance,
                cx + tx * tolerance,
                cy + ty * tolerance,
                sigma,
                threshold,
                transition,
            ),
            choice,
        )
        points.extend([hit.column, hit.row] for hit in found)
    return points


def _rect_corners(row, col, phi, length1, length2):
    u = (math.cos(phi), math.sin(phi))
    v = (-math.sin(phi), math.cos(phi))
    def pt(su, sv):
        return (col + su * length1 * u[0] + sv * length2 * v[0], row + su * length1 * u[1] + sv * length2 * v[1])

    return [pt(1, 1), pt(-1, 1), pt(-1, -1), pt(1, -1)], u, v


def _sides(corners):
    return [(corners[index], corners[(index + 1) % 4]) for index in range(4)]


def _intersect_lines(line_a, line_b):
    origin_a, direction_a = line_a
    origin_b, direction_b = line_b
    det = direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]
    if abs(det) < 1e-8:
        return None
    t = ((origin_b[0] - origin_a[0]) * direction_b[1] - (origin_b[1] - origin_a[1]) * direction_b[0]) / det
    return (float(origin_a[0] + direction_a[0] * t), float(origin_a[1] + direction_a[1] * t))


def _median_radius(points, row, col) -> float:
    dists = [math.hypot(point[0] - col, point[1] - row) for point in points]
    return float(np.median(np.asarray(dists, np.float64)))


def _mean_angle(points, row, col) -> float:
    sine = 0.0
    cosine = 0.0
    for point in points:
        angle = math.atan2(point[1] - row, point[0] - col)
        sine += math.sin(angle)
        cosine += math.cos(angle)
    return math.atan2(sine, cosine)


def _arc(row, col, radius, start, extent, count) -> list:
    count = max(8, int(count))
    pts = []
    for index in range(count + 1):
        angle = start + extent * index / count
        pts.append([col + radius * math.cos(angle), row + radius * math.sin(angle)])
    return pts
