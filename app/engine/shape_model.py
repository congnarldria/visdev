"""Gradient-orientation shape matching with sub-pixel pose and contours.

The search follows the shape-based matching idea used in industrial vision:
edge points vote by how well their gradient direction agrees with the image.
A pyramid finds the pose, then a parabolic fit on the score surface yields a
sub-pixel row, column, and angle. Contour points are located to sub-pixel
along the gradient before that pose is applied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

cv2.setNumThreads(1)

MATCH_COLORS = [
    "#ff3b30",
    "#00e05a",
    "#3d8bff",
    "#ffe14a",
    "#2ce6e6",
    "#f04ad4",
    "#ff8a1e",
    "#c6ff4a",
    "#7aa2ff",
    "#ff6a9a",
    "#46f0c8",
    "#e6c14a",
    "#b388ff",
    "#ff7a3c",
    "#9be7ff",
    "#ffffff",
]


@dataclass
class _Level:
    rows: np.ndarray
    cols: np.ndarray
    gx: np.ndarray
    gy: np.ndarray


@dataclass
class ShapeModel:
    levels: list[_Level]
    contours: list
    origin_row: float
    origin_col: float
    width: int
    height: int
    radius: float
    num_levels: int
    metric: str
    contrast: float
    min_contrast: float
    angle_step: float
    num_points: int = 0
    angle_start: float = -math.pi
    angle_extent: float = math.tau
    native: object = None

    def summary(self) -> str:
        return f"形狀模型  已教導  {self.num_points} 點  {len(self.contours)} 條輪廓  {self.num_levels} 層"


def _gray(image) -> np.ndarray:
    return image.gray()


def _gradients(gray: np.ndarray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return gx, gy, mag


def _sample(image: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    height, width = image.shape
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < width) & (y1 < height)
    wx = (xs - x0).astype(np.float32)
    wy = (ys - y0).astype(np.float32)
    x0c = np.clip(x0, 0, width - 1)
    x1c = np.clip(x1, 0, width - 1)
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y1, 0, height - 1)
    value = (
        image[y0c, x0c] * (1 - wx) * (1 - wy)
        + image[y0c, x1c] * wx * (1 - wy)
        + image[y1c, x0c] * (1 - wx) * wy
        + image[y1c, x1c] * wx * wy
    )
    return value.astype(np.float32), valid


def _nms(mag: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    angle = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0
    padded = np.pad(mag, 1, mode="constant")
    center = padded[1:-1, 1:-1]
    north = padded[0:-2, 1:-1]
    south = padded[2:, 1:-1]
    west = padded[1:-1, 0:-2]
    east = padded[1:-1, 2:]
    north_west = padded[0:-2, 0:-2]
    north_east = padded[0:-2, 2:]
    south_west = padded[2:, 0:-2]
    south_east = padded[2:, 2:]
    horizontal = ((angle < 22.5) | (angle >= 157.5)) & (center >= west) & (center >= east)
    diagonal = (angle >= 22.5) & (angle < 67.5) & (center >= north_west) & (center >= south_east)
    vertical = (angle >= 67.5) & (angle < 112.5) & (center >= north) & (center >= south)
    other = (angle >= 112.5) & (angle < 157.5) & (center >= north_east) & (center >= south_west)
    keep = horizontal | diagonal | vertical | other
    out = np.zeros_like(mag)
    out[keep] = center[keep]
    return out


def _subpixel_edges(gray: np.ndarray, contrast: float, domain: np.ndarray | None):
    gx, gy, mag = _gradients(gray)
    if domain is not None:
        mag = mag.copy()
        mag[domain == 0] = 0
    ridge = _nms(mag, gx, gy)
    ys, xs = np.where(ridge >= contrast)
    if len(xs) == 0:
        empty = np.zeros((0, 2), np.float32)
        empty_index = np.zeros((0,), np.int32)
        return empty, empty, ridge, empty_index, empty_index
    ux = np.zeros(len(xs), np.float32)
    uy = np.zeros(len(xs), np.float32)
    strength = ridge[ys, xs]
    good = strength > 1e-3
    ux[good] = gx[ys[good], xs[good]] / strength[good]
    uy[good] = gy[ys[good], xs[good]] / strength[good]
    base_x = xs.astype(np.float32)
    base_y = ys.astype(np.float32)
    before, _ = _sample(mag, base_x - ux, base_y - uy)
    after, _ = _sample(mag, base_x + ux, base_y + uy)
    denom = before - 2 * strength + after
    delta = np.zeros(len(xs), np.float32)
    stable = np.abs(denom) > 1e-4
    delta[stable] = 0.5 * (before[stable] - after[stable]) / denom[stable]
    delta = np.clip(delta, -0.8, 0.8)
    points = np.stack([base_x + delta * ux, base_y + delta * uy], axis=1)
    directions = np.stack([ux, uy], axis=1)
    return points.astype(np.float32), directions.astype(np.float32), ridge, xs.astype(np.int32), ys.astype(np.int32)


def _chains(ridge: np.ndarray, points: np.ndarray, ys_xs_points: dict) -> list:
    ys, xs = np.where(ridge > 0)
    pixels = {(int(x), int(y)) for x, y in zip(xs, ys)}
    if not pixels:
        return []
    neighbors = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    used = set()
    chains = []
    for seed in pixels:
        if seed in used:
            continue
        chain = [seed]
        used.add(seed)
        for forward in (False, True):
            current = chain[-1] if forward else chain[0]
            previous = None
            while True:
                cx, cy = current
                found = []
                for dx, dy in neighbors:
                    nxt = (cx + dx, cy + dy)
                    if nxt in pixels and nxt not in used:
                        found.append(nxt)
                if not found:
                    break
                if previous is not None and len(found) > 1:
                    vx, vy = cx - previous[0], cy - previous[1]
                    nxt = max(found, key=lambda item: (item[0] - cx) * vx + (item[1] - cy) * vy)
                else:
                    nxt = found[0]
                used.add(nxt)
                if forward:
                    chain.append(nxt)
                else:
                    chain.insert(0, nxt)
                previous, current = current, nxt
        if len(chain) < 6:
            continue
        poly = []
        for x, y in chain:
            located = ys_xs_points.get((x, y))
            poly.append(located if located is not None else [float(x), float(y)])
        if math.hypot(poly[0][0] - poly[-1][0], poly[0][1] - poly[-1][1]) <= 2.5:
            poly.append(poly[0])
        chains.append(poly)
    return chains


def _auto_contrast(mag: np.ndarray, domain: np.ndarray | None) -> float:
    values = mag[domain > 0] if domain is not None else mag.ravel()
    if values.size == 0:
        return 20.0
    strong = values[values > 8]
    if strong.size == 0:
        return 12.0
    return float(max(12.0, np.percentile(strong, 75)))


def _subsample(rows, cols, gx, gy, limit: int):
    count = len(rows)
    if count <= limit:
        return rows, cols, gx, gy
    step = int(math.ceil(count / limit))
    return rows[::step], cols[::step], gx[::step], gy[::step]


def _rotate(cols, rows, gx, gy, angle: float):
    cosine = math.cos(angle)
    sine = math.sin(angle)
    next_cols = cols * cosine + rows * sine
    next_rows = -cols * sine + rows * cosine
    next_gx = gx * cosine + gy * sine
    next_gy = -gx * sine + gy * cosine
    return next_cols, next_rows, next_gx, next_gy


def _match_backend():
    try:
        from . import visdev_match
    except ImportError:
        return None
    return visdev_match


def create_shape_model(
    image, num_levels, angle_step, metric: str, contrast, min_contrast, angle_start=-math.pi, angle_extent=math.tau
) -> ShapeModel:
    gray = _gray(image)
    domain = image.domain
    gx, gy, mag = _gradients(gray)
    if contrast is None:
        contrast = _auto_contrast(mag, domain)
    if min_contrast is None:
        min_contrast = max(5.0, contrast * 0.35)
    points, directions, ridge, pixel_x, pixel_y = _subpixel_edges(gray, float(contrast), domain)
    if len(points) < 12:
        raise ValueError("模板邊緣點太少。請降低 Contrast，或確認模板有清楚輪廓。")
    if domain is not None and int(cv2.countNonZero(domain)):
        ys, xs = np.where(domain > 0)
        origin_row = float(ys.mean())
        origin_col = float(xs.mean())
    else:
        origin_row = (gray.shape[0] - 1) / 2.0
        origin_col = (gray.shape[1] - 1) / 2.0
    lookup = {
        (int(x), int(y)): [float(px), float(py)]
        for x, y, px, py in zip(pixel_x, pixel_y, points[:, 0], points[:, 1])
    }
    raw_chains = _chains(ridge >= float(contrast), points, lookup)
    contours = []
    for chain in raw_chains:
        relative = [[point[0] - origin_col, point[1] - origin_row] for point in chain]
        contours.append(relative)
    if not contours:
        contours = [[[float(x) - origin_col, float(y) - origin_row]] for x, y in points[:: max(1, len(points) // 80)]]
        contours = [chain for chain in contours if len(chain) >= 2]

    height, width = gray.shape
    if not num_levels:
        num_levels = 1
        probe_h, probe_w = height, width
        while min(probe_h, probe_w) >= 72 and num_levels < 5:
            probe_h //= 2
            probe_w //= 2
            num_levels += 1
    num_levels = max(1, min(int(num_levels), 5))
    rel_col = points[:, 0] - origin_col
    rel_row = points[:, 1] - origin_row
    base_row, base_col, base_gx, base_gy = _subsample(rel_row, rel_col, directions[:, 0], directions[:, 1], 360)
    radius = float(np.max(np.hypot(base_col, base_row))) if len(base_row) else 1.0
    levels = []
    for index in range(num_levels):
        scale = float(2**index)
        limit = 360 if index == 0 else 160
        level_row, level_col, level_gx, level_gy = _subsample(
            base_row / scale, base_col / scale, base_gx, base_gy, limit
        )
        levels.append(
            _Level(
                level_row.astype(np.float32),
                level_col.astype(np.float32),
                level_gx.astype(np.float32),
                level_gy.astype(np.float32),
            )
        )
    total_points = int(len(base_row))
    if angle_step is None or angle_step == 0:
        angle_step = max(np.deg2rad(0.6), 1.15 / max(radius, 1.0))
    model = ShapeModel(
        levels=levels,
        contours=contours,
        origin_row=origin_row,
        origin_col=origin_col,
        width=int(width),
        height=int(height),
        radius=max(radius, 1.0),
        num_levels=len(levels),
        metric=metric,
        contrast=float(contrast),
        min_contrast=float(min_contrast),
        angle_step=float(angle_step),
        num_points=total_points,
        angle_start=float(angle_start),
        angle_extent=float(angle_extent),
    )
    backend = _match_backend()
    if backend is not None:
        packed = [
            (
                np.ascontiguousarray(level.rows, dtype=np.float32),
                np.ascontiguousarray(level.cols, dtype=np.float32),
                np.ascontiguousarray(level.gx, dtype=np.float32),
                np.ascontiguousarray(level.gy, dtype=np.float32),
            )
            for level in model.levels
        ]
        model.native = backend.shape_create(
            packed,
            float(angle_start),
            float(angle_extent),
            float(model.angle_step),
            float(model.min_contrast),
            1 if metric == "ignore_local_polarity" else 0,
            float(model.radius),
        )
    return model


def _shift_to_origin(source: np.ndarray, row_offset: int, col_offset: int) -> np.ndarray:
    destination = np.zeros_like(source)
    height, width = source.shape
    if row_offset >= 0:
        src_y, dst_y, rows = row_offset, 0, height - row_offset
    else:
        src_y, dst_y, rows = 0, -row_offset, height + row_offset
    if col_offset >= 0:
        src_x, dst_x, cols = col_offset, 0, width - col_offset
    else:
        src_x, dst_x, cols = 0, -col_offset, width + col_offset
    if rows <= 0 or cols <= 0:
        return destination
    destination[dst_y : dst_y + rows, dst_x : dst_x + cols] = source[src_y : src_y + rows, src_x : src_x + cols]
    return destination


def _score_map(unit_x, unit_y, valid, level: _Level, angle: float, metric: str, greediness: float):
    cols, rows, gx, gy = _rotate(level.cols, level.rows, level.gx, level.gy, angle)
    count = len(cols)
    keep = count
    if greediness > 0 and count > 30:
        keep = max(30, int(round(count * (1.0 - 0.55 * greediness))))
        step = max(1, int(math.ceil(count / keep)))
        cols, rows, gx, gy = cols[::step], rows[::step], gx[::step], gy[::step]
    acc = np.zeros(unit_x.shape, np.float32)
    seen = np.zeros(unit_x.shape, np.float32)
    polarity = metric != "ignore_local_polarity"
    for col, row, vx, vy in zip(cols, rows, gx, gy):
        contrib = (unit_x * float(vx) + unit_y * float(vy)) * valid
        if not polarity:
            contrib = np.abs(contrib)
        iy = int(round(float(row)))
        ix = int(round(float(col)))
        acc += _shift_to_origin(contrib, iy, ix)
        seen += _shift_to_origin(valid, iy, ix)
    score = np.zeros_like(acc)
    enough = seen >= max(8.0, 0.42 * len(cols))
    score[enough] = acc[enough] / seen[enough]
    return score


def _local_peaks(score: np.ndarray, minimum: float, limit: int):
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(score, kernel)
    mask = (score >= minimum) & (score >= dilated - 1e-6)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    order = np.argsort(score[ys, xs])[::-1]
    peaks = []
    for index in order[: limit * 4]:
        peaks.append((float(ys[index]), float(xs[index]), float(score[ys[index], xs[index]])))
        if len(peaks) >= limit:
            break
    return peaks


def _score_at(field_x, field_y, field_m, level: _Level, row: float, col: float, angle: float, metric: str, min_contrast: float, scale_factor: float = 1.0):
    cols, rows, gx, gy = _rotate(level.cols, level.rows, level.gx, level.gy, angle)
    if scale_factor != 1.0:
        cols = cols * scale_factor
        rows = rows * scale_factor
    sample_x = col + cols
    sample_y = row + rows
    image_x, valid_x = _sample(field_x, sample_x, sample_y)
    image_y, valid_y = _sample(field_y, sample_x, sample_y)
    image_m, valid_m = _sample(field_m, sample_x, sample_y)
    valid = valid_x & valid_y & valid_m & (image_m >= min_contrast)
    if int(valid.sum()) < max(8, int(0.4 * len(cols))):
        return 0.0
    dots = (image_x[valid] * gx[valid] + image_y[valid] * gy[valid]) / np.maximum(image_m[valid], 1e-6)
    if metric == "ignore_local_polarity":
        dots = np.abs(dots)
    return float(dots.mean())


def _solve_rows(design, target) -> np.ndarray:
    """Least squares by Householder QR. The step does not depend on BLAS threads."""
    matrix = [[float(value) for value in row] for row in np.asarray(design, np.float64)]
    vector = [float(value) for value in np.asarray(target, np.float64)]
    rows = len(matrix)
    columns = len(matrix[0]) if rows else 0
    for col in range(columns):
        norm = math.sqrt(sum(matrix[row][col] * matrix[row][col] for row in range(col, rows)))
        if norm < 1e-18:
            continue
        sign = 1.0 if matrix[col][col] >= 0.0 else -1.0
        reflector = [0.0] * rows
        reflector[col] = matrix[col][col] + sign * norm
        for row in range(col + 1, rows):
            reflector[row] = matrix[row][col]
        weight = sum(value * value for value in reflector[col:])
        if weight < 1e-18:
            continue
        for other in range(col, columns):
            dot = sum(reflector[row] * matrix[row][other] for row in range(col, rows))
            beta = 2.0 * dot / weight
            for row in range(col, rows):
                matrix[row][other] -= beta * reflector[row]
        dot = sum(reflector[row] * vector[row] for row in range(col, rows))
        beta = 2.0 * dot / weight
        for row in range(col, rows):
            vector[row] -= beta * reflector[row]
    answer = [0.0] * columns
    for row in range(columns - 1, -1, -1):
        if abs(matrix[row][row]) < 1e-18:
            continue
        total = vector[row] - sum(matrix[row][col] * answer[col] for col in range(row + 1, columns))
        answer[row] = total / matrix[row][row]
    return np.array(answer, np.float64)


def _parabola(left: float, center: float, right: float) -> float:
    denom = left - 2 * center + right
    if abs(denom) < 1e-6 or center < left or center < right:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denom, -0.75, 0.75))


def _subpixel_mode(subpixel) -> str:
    if subpixel is True or subpixel == "interpolation":
        return "interpolation"
    if subpixel is False or subpixel in ("none", 0):
        return "none"
    if subpixel == "least_squares":
        return "least_squares"
    return "interpolation"


def _least_squares_pose(magnitude, level: _Level, row: float, col: float, angle: float, min_contrast: float):
    step = max(1, int(math.ceil(len(level.cols) / 80)))
    point_col = level.cols[::step]
    point_row = level.rows[::step]
    grad_x = level.gx[::step]
    grad_y = level.gy[::step]
    pose = (float(row), float(col), float(angle))
    for _turn in range(4):
        row, col, angle = pose
        cosine = math.cos(angle)
        sine = math.sin(angle)
        rotated_col = point_col * cosine + point_row * sine
        rotated_row = -point_col * sine + point_row * cosine
        normal_col = grad_x * cosine + grad_y * sine
        normal_row = -grad_x * sine + grad_y * cosine
        sample_col = col + rotated_col
        sample_row = row + rotated_row
        center, valid_center = _sample(magnitude, sample_col, sample_row)
        before, valid_before = _sample(magnitude, sample_col - normal_col, sample_row - normal_row)
        after, valid_after = _sample(magnitude, sample_col + normal_col, sample_row + normal_row)
        denom = before - 2.0 * center + after
        keep = (
            (center >= before)
            & (center >= after)
            & (np.abs(denom) > 1e-3)
            & valid_center
            & valid_before
            & valid_after
            & (center >= min_contrast)
        )
        if int(np.count_nonzero(keep)) < 8:
            break
        offset = np.clip(0.5 * (before[keep] - after[keep]) / denom[keep], -0.6, 0.6)
        normal_col = normal_col[keep]
        normal_row = normal_row[keep]
        rotated_col = rotated_col[keep]
        rotated_row = rotated_row[keep]
        design = np.column_stack(
            [normal_col, normal_row, normal_col * rotated_row - normal_row * rotated_col]
        )
        delta = _solve_rows(design, offset)
        dcol = float(np.clip(delta[0], -0.6, 0.6))
        drow = float(np.clip(delta[1], -0.6, 0.6))
        dangle = float(np.clip(delta[2], -math.radians(1.5), math.radians(1.5)))
        if abs(dcol) + abs(drow) + abs(dangle) < 1e-4:
            break
        pose = (row + drow, col + dcol, angle + dangle)
    return pose


def _scale_window(model, scale: float, search_min: float, search_max: float) -> tuple[float, float]:
    taught = sorted(item[0] for item in model.scales)
    index = min(range(len(taught)), key=lambda item: abs(taught[item] - scale))
    low = high = taught[index]
    if index > 0:
        low = 0.5 * (taught[index - 1] + taught[index])
    if index + 1 < len(taught):
        high = 0.5 * (taught[index] + taught[index + 1])
    low = max(low, float(search_min))
    high = min(high, float(search_max))
    if high < low:
        return float(scale), float(scale)
    return low, high


def _least_squares_scale(magnitude, level: _Level, row: float, col: float, angle: float, scale: float, low: float, high: float, min_contrast: float):
    """Move scale between the neighboring taught steps. Points already belong to `scale`."""
    if high - low < 1e-6 or scale <= 1e-6:
        return float(row), float(col), float(angle), float(scale)
    step = max(1, int(math.ceil(len(level.cols) / 80)))
    point_col = level.cols[::step]
    point_row = level.rows[::step]
    grad_x = level.gx[::step]
    grad_y = level.gy[::step]
    factor = 1.0
    pose = (float(row), float(col), float(angle), factor)
    for _turn in range(4):
        row, col, angle, factor = pose
        cosine = math.cos(angle)
        sine = math.sin(angle)
        base_col = point_col * cosine + point_row * sine
        base_row = -point_col * sine + point_row * cosine
        rotated_col = base_col * factor
        rotated_row = base_row * factor
        normal_col = grad_x * cosine + grad_y * sine
        normal_row = -grad_x * sine + grad_y * cosine
        sample_col = col + rotated_col
        sample_row = row + rotated_row
        center, valid_center = _sample(magnitude, sample_col, sample_row)
        before, valid_before = _sample(magnitude, sample_col - normal_col, sample_row - normal_row)
        after, valid_after = _sample(magnitude, sample_col + normal_col, sample_row + normal_row)
        denom = before - 2.0 * center + after
        keep = (
            (center >= before)
            & (center >= after)
            & (np.abs(denom) > 1e-3)
            & valid_center
            & valid_before
            & valid_after
            & (center >= min_contrast)
        )
        if int(np.count_nonzero(keep)) < 8:
            break
        offset = np.clip(0.5 * (before[keep] - after[keep]) / denom[keep], -0.6, 0.6)
        normal_col = normal_col[keep]
        normal_row = normal_row[keep]
        rotated_col = rotated_col[keep]
        rotated_row = rotated_row[keep]
        radial = base_col[keep] * normal_col + base_row[keep] * normal_row
        design = np.column_stack(
            [normal_col, normal_row, normal_col * rotated_row - normal_row * rotated_col, radial]
        )
        delta = _solve_rows(design, offset)
        dcol = float(np.clip(delta[0], -0.6, 0.6))
        drow = float(np.clip(delta[1], -0.6, 0.6))
        dangle = float(np.clip(delta[2], -math.radians(1.5), math.radians(1.5)))
        dfactor = float(np.clip(delta[3], -0.08, 0.08))
        if abs(dcol) + abs(drow) + abs(dangle) + abs(dfactor) < 1e-4:
            break
        factor = min(max(factor + dfactor, low / scale), high / scale)
        pose = (row + drow, col + dcol, angle + dangle, factor)
    row, col, angle, factor = pose
    return row, col, angle, float(scale * factor)


def _refine_poses(image, model: ShapeModel, rows, cols, angles, scores):
    gray = np.ascontiguousarray(_gray(image), dtype=np.float32)
    field_x, field_y, magnitude = _gradients(gray)
    level = model.levels[0]
    refined_rows, refined_cols, refined_angles, refined_scores = [], [], [], []
    for row, col, angle, score in zip(_flat(rows), _flat(cols), _flat(angles), _flat(scores)):
        old = _score_at(field_x, field_y, magnitude, level, row, col, angle, model.metric, model.min_contrast)
        nrow, ncol, nangle = _least_squares_pose(magnitude, level, row, col, angle, model.min_contrast)
        new = _score_at(field_x, field_y, magnitude, level, nrow, ncol, nangle, model.metric, model.min_contrast)
        if new + 1e-4 >= old:
            refined_rows.append(nrow)
            refined_cols.append(ncol)
            refined_angles.append(nangle)
            refined_scores.append(new if new > 0 else float(score))
        else:
            refined_rows.append(float(row))
            refined_cols.append(float(col))
            refined_angles.append(float(angle))
            refined_scores.append(float(score))
    return refined_rows, refined_cols, refined_angles, refined_scores


def _angle_step_at(model: ShapeModel, level_index: int) -> float:
    scale = 2**level_index
    radius = max(model.radius / scale, 1.0)
    widest = 2.1 / radius
    return float(min(max(model.angle_step * scale, model.angle_step), max(model.angle_step, widest)))


def _angles(start: float, extent: float, step: float) -> list[float]:
    if extent <= 0 or step <= 0:
        return [start]
    values = []
    angle = start
    stop = start + extent
    guard = 0
    while angle < stop - 1e-6 and guard < 2000:
        values.append(angle)
        angle += step
        guard += 1
    if not values:
        values.append(start)
    return values


def _angle_gap(left: float, right: float) -> float:
    return abs((left - right + math.pi) % (2 * math.pi) - math.pi)


def _suppress(candidates, min_dist: float, maximum: int, angle_gate: float | None = None):
    chosen = []
    for row, col, angle, score in candidates:
        blocked = False
        for other in chosen:
            if math.hypot(row - other[0], col - other[1]) >= min_dist:
                continue
            if angle_gate is not None and _angle_gap(angle, other[2]) > angle_gate:
                continue
            blocked = True
            break
        if blocked:
            continue
        chosen.append((row, col, angle, score))
        if maximum and len(chosen) >= maximum:
            break
    return chosen


def find_shape_model(
    image,
    model: ShapeModel,
    angle_start: float,
    angle_extent: float,
    min_score: float,
    num_matches: int,
    max_overlap: float,
    subpixel: bool,
    num_levels: int,
    greediness: float,
):
    mode = _subpixel_mode(subpixel)
    backend = _match_backend()
    if backend is not None and getattr(model, "native", None) is not None:
        levels_used = model.num_levels if not num_levels else max(1, min(int(num_levels), model.num_levels))
        gray = np.ascontiguousarray(_gray(image), dtype=np.float32)
        pyramid = [gray]
        for _ in range(1, levels_used):
            pyramid.append(np.ascontiguousarray(cv2.pyrDown(pyramid[-1])))
        rows, cols, angles, scores = backend.shape_find(
            model.native,
            pyramid,
            float(angle_start),
            float(angle_extent),
            float(min_score),
            int(num_matches),
            float(max_overlap),
            1 if mode != "none" else 0,
            float(greediness),
        )
        if mode == "least_squares":
            return _refine_poses(image, model, rows, cols, angles, scores)
        return rows, cols, angles, scores
    gray = _gray(image)
    levels_used = model.num_levels if not num_levels else max(1, min(int(num_levels), model.num_levels))
    pyramid = [gray]
    for _ in range(1, levels_used):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    fields = []
    for level_gray in pyramid:
        gx, gy, mag = _gradients(level_gray)
        valid = (mag >= model.min_contrast).astype(np.float32)
        unit_x = np.zeros_like(gx)
        unit_y = np.zeros_like(gy)
        usable = mag >= 1e-3
        unit_x[usable] = gx[usable] / mag[usable]
        unit_y[usable] = gy[usable] / mag[usable]
        fields.append((unit_x, unit_y, valid, gx, gy, mag))

    coarse = levels_used - 1
    step = _angle_step_at(model, coarse)
    coarse_angles = _angles(angle_start, angle_extent, step)
    coarse_min = min_score * (0.72 if greediness < 0.85 else 0.82)
    found = []
    unit_x, unit_y, valid, _gx, _gy, _mag = fields[coarse]
    level = model.levels[coarse]
    for angle in coarse_angles:
        score = _score_map(unit_x, unit_y, valid, level, angle, model.metric, greediness)
        for row, col, value in _local_peaks(score, coarse_min, 40):
            found.append((row, col, angle, value))
    found.sort(key=lambda item: item[3], reverse=True)
    found = _suppress(found, max(4.0, model.radius / (2**coarse) * 0.45), 80, np.deg2rad(20))

    for level_index in range(coarse - 1, -1, -1):
        refined = []
        scale_row = pyramid[level_index].shape[0] / pyramid[level_index + 1].shape[0]
        scale_col = pyramid[level_index].shape[1] / pyramid[level_index + 1].shape[1]
        field_x, field_y, _valid, raw_x, raw_y, mag = fields[level_index]
        level = model.levels[level_index]
        angle_window = _angle_step_at(model, level_index + 1)
        fine_step = max(model.angle_step, angle_window / 2.0)
        for row, col, angle, _score in found:
            best = (0.0, row * scale_row, col * scale_col, angle)
            for angle_try in (angle - angle_window, angle, angle + angle_window):
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        rr = row * scale_row + dy
                        cc = col * scale_col + dx
                        value = _score_at(raw_x, raw_y, mag, level, rr, cc, angle_try, model.metric, model.min_contrast)
                        if value > best[0]:
                            best = (value, rr, cc, angle_try)
            if best[0] >= min_score * 0.8:
                nudge = _local_angle(raw_x, raw_y, mag, level, best[1], best[2], best[3], fine_step, model)
                refined.append(nudge)
        refined.sort(key=lambda item: item[3], reverse=True)
        found = _suppress(
            [(item[0], item[1], item[2], item[3]) for item in refined],
            max(3.0, model.radius / (2**level_index) * 0.55),
            80,
            np.deg2rad(20),
        )
        if not found:
            break

    poses = []
    if found and mode != "none":
        raw_x, raw_y, mag = fields[0][3], fields[0][4], fields[0][5]
        level = model.levels[0]
        for row, col, angle, score in found:
            row, col, angle, score = _subpixel(raw_x, raw_y, mag, level, row, col, angle, model)
            if score >= min_score:
                poses.append((row, col, angle, score))
    else:
        poses = [item for item in found if item[3] >= min_score]
    poses.sort(key=lambda item: item[3], reverse=True)
    min_dist = model.radius * max(0.2, 1.0 - max_overlap)
    poses = _suppress(poses, min_dist, num_matches)
    rows = [item[0] for item in poses]
    cols = [item[1] for item in poses]
    angles = [item[2] for item in poses]
    scores = [item[3] for item in poses]
    if mode == "least_squares" and rows:
        return _refine_poses(image, model, rows, cols, angles, scores)
    return rows, cols, angles, scores


def _local_angle(raw_x, raw_y, mag, level, row, col, angle, step, model: ShapeModel):
    best_score = -1.0
    best = (row, col, angle, 0.0)
    for candidate in (angle - step, angle, angle + step):
        value = _score_at(raw_x, raw_y, mag, level, row, col, candidate, model.metric, model.min_contrast)
        if value > best_score:
            best_score = value
            best = (row, col, candidate, value)
    return best


def _subpixel(raw_x, raw_y, mag, level, row, col, angle, model: ShapeModel):
    step = max(model.angle_step, np.deg2rad(0.8))

    def value(rr, cc, aa):
        return _score_at(raw_x, raw_y, mag, level, rr, cc, aa, model.metric, model.min_contrast)

    for _ in range(2):
        row_delta = _parabola(value(row - 1, col, angle), value(row, col, angle), value(row + 1, col, angle))
        col_delta = _parabola(value(row, col - 1, angle), value(row, col, angle), value(row, col + 1, angle))
        angle_delta = _parabola(value(row, col, angle - step), value(row, col, angle), value(row, col, angle + step))
        row += row_delta
        col += col_delta
        angle += angle_delta * step
        step *= 0.5
    score = value(row, col, angle)
    return row, col, angle, score


def transform_contours(model: ShapeModel, row: float, col: float, angle: float) -> list:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    transformed = []
    for chain in model.contours:
        poly = []
        for point_col, point_row in chain:
            x = col + point_col * cosine + point_row * sine
            y = row - point_col * sine + point_row * cosine
            poly.append([float(x), float(y)])
        if len(poly) >= 2:
            transformed.append(poly)
    return transformed


def color_for_match(index: int) -> str:
    return MATCH_COLORS[index % len(MATCH_COLORS)]


def _flat(value) -> list[float]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [float(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


def _scale_values(scale_min: float, scale_max: float, scale_step: float) -> tuple[list[float], float]:
    low = float(scale_min)
    high = float(scale_max)
    if high + 1e-9 < low:
        raise ValueError("ScaleMax 不能小於 ScaleMin")
    if high - low < 1e-9:
        return [low], 0.0
    step = float(scale_step)
    if step <= 0:
        step = max((high - low) / 6.0, 0.02)
    count = int(math.floor((high - low) / step + 1e-6)) + 1
    if count > 8:
        step = (high - low) / 7.0
        count = 8
    values = [low + step * index for index in range(count)]
    if values[-1] > high:
        values[-1] = high
    return values, step


def _resize_image(image, scale: float):
    from .objects import Image as Picture

    height, width = image.gray().shape
    resized_w = int(round(width * scale))
    resized_h = int(round(height * scale))
    if resized_w < 8 or resized_h < 8:
        raise ValueError("縮放後小於 8×8")
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    data = cv2.resize(image.data, (resized_w, resized_h), interpolation=interpolation)
    domain = None
    if image.domain is not None and int(cv2.countNonZero(image.domain)):
        domain = cv2.resize(image.domain, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
        domain = np.where(domain > 127, 255, 0).astype(np.uint8)
    return Picture(data, domain)


@dataclass
class ScaledShapeModel:
    scales: list
    origin_row: float
    origin_col: float
    angle_start: float
    angle_extent: float
    angle_step: float
    scale_min: float
    scale_max: float
    scale_step: float
    metric: str
    optimization: str
    num_levels: int

    def summary(self) -> str:
        return (
            f"縮放形狀模型  {len(self.scales)} 個尺度  "
            f"{self.scale_min:.3f}–{self.scale_max:.3f}"
        )


def create_scaled_shape_model(
    image,
    num_levels,
    angle_step,
    scale_min,
    scale_max,
    scale_step,
    optimization: str,
    metric: str,
    contrast,
    min_contrast,
    angle_start: float,
    angle_extent: float,
) -> ScaledShapeModel:
    values, used_step = _scale_values(scale_min, scale_max, scale_step)
    built = []
    for scale in values:
        try:
            resized = _resize_image(image, scale)
            model = create_shape_model(
                resized,
                num_levels,
                angle_step,
                metric,
                contrast,
                min_contrast,
                angle_start,
                angle_extent,
            )
        except ValueError:
            continue
        built.append((float(scale), model))
    if not built:
        raise ValueError("這些尺度都做不出形狀模型。請加大模板，或放寬 Contrast。")
    first = built[0][1]
    if image.domain is not None and int(cv2.countNonZero(image.domain)):
        ys, xs = np.where(image.domain > 0)
        origin_row = float(ys.mean())
        origin_col = float(xs.mean())
    else:
        origin_row = (image.gray().shape[0] - 1) / 2.0
        origin_col = (image.gray().shape[1] - 1) / 2.0
    return ScaledShapeModel(
        scales=built,
        origin_row=origin_row,
        origin_col=origin_col,
        angle_start=float(angle_start),
        angle_extent=float(angle_extent),
        angle_step=float(first.angle_step),
        scale_min=float(values[0]),
        scale_max=float(values[-1]),
        scale_step=float(used_step),
        metric=metric,
        optimization=optimization,
        num_levels=int(first.num_levels),
    )


def find_scaled_shape_model(
    image,
    model: ScaledShapeModel,
    angle_start: float,
    angle_extent: float,
    scale_min: float,
    scale_max: float,
    min_score: float,
    num_matches: int,
    max_overlap: float,
    subpixel: bool,
    num_levels: int,
    greediness: float,
):
    if getattr(model, "cleared", False):
        raise RuntimeError("模型已經 clear，不能再搜尋")
    mode = _subpixel_mode(subpixel)
    # Which taught scale wins is decided without pose least squares. That step
    # walks off the peak and the following scale fit no longer reaches 1.07505.
    child_subpixel = "none" if mode == "least_squares" else subpixel
    pool = []
    wanted = int(num_matches) if num_matches else 8
    for scale, child in model.scales:
        if scale < float(scale_min) - 1e-6 or scale > float(scale_max) + 1e-6:
            continue
        rows, cols, angles, scores = find_shape_model(
            image,
            child,
            angle_start,
            angle_extent,
            min_score,
            wanted,
            max_overlap,
            child_subpixel,
            num_levels,
            greediness,
        )
        for row, col, angle, score in zip(_flat(rows), _flat(cols), _flat(angles), _flat(scores)):
            pool.append((score, row, col, angle, scale, float(child.radius)))
    pool.sort(key=lambda item: item[0], reverse=True)
    chosen = []
    limit = max(0.0, min(1.0, float(max_overlap)))
    for item in pool:
        _score, row, col, _angle, _scale, radius = item
        blocked = False
        for other in chosen:
            gap = math.hypot(row - other[1], col - other[2])
            if gap < max(8.0, (1.0 - limit) * max(radius, other[5])):
                blocked = True
                break
        if blocked:
            continue
        chosen.append(item)
        if num_matches and len(chosen) >= int(num_matches):
            break
    if not chosen:
        return [], [], [], [], []
    if mode == "least_squares":
        gray = np.ascontiguousarray(_gray(image), dtype=np.float32)
        field_x, field_y, magnitude = _gradients(gray)
        refined = []
        for score, row, col, angle, scale, radius in chosen:
            child = next(item[1] for item in model.scales if abs(item[0] - scale) < 1e-6)
            found_rows, found_cols, found_angles, _found_scores = find_shape_model(
                image,
                child,
                angle_start,
                angle_extent,
                min_score,
                1,
                max_overlap,
                "interpolation",
                num_levels,
                greediness,
            )
            if found_rows:
                row = float(_flat(found_rows)[0])
                col = float(_flat(found_cols)[0])
                angle = float(_flat(found_angles)[0])
            low, high = _scale_window(model, scale, scale_min, scale_max)
            nrow, ncol, nangle, nscale = _least_squares_scale(
                magnitude, child.levels[0], row, col, angle, scale, low, high, child.min_contrast
            )
            factor = nscale / scale if scale else 1.0
            old_same = _score_at(
                field_x, field_y, magnitude, child.levels[0], row, col, angle, model.metric, child.min_contrast, 1.0
            )
            new_score = _score_at(
                field_x, field_y, magnitude, child.levels[0], nrow, ncol, nangle, model.metric, child.min_contrast, factor
            )
            if new_score + 1e-4 >= old_same:
                refined.append((new_score if new_score > 0 else score, nrow, ncol, nangle, nscale, radius))
            else:
                refined.append((score, row, col, angle, scale, radius))
        chosen = refined
    return (
        [item[1] for item in chosen],
        [item[2] for item in chosen],
        [item[3] for item in chosen],
        [item[4] for item in chosen],
        [item[0] for item in chosen],
    )
