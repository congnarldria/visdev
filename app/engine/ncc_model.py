"""Normalized cross-correlation matching.

The template is taught from an image, optionally a rectangle. Its origin is the
center of that taught region. A pyramid of matchTemplate searches rotation and
translation; a parabolic fit then gives a sub-pixel row, column, and angle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .shape_model import MATCH_COLORS, color_for_match


@dataclass
class NccModel:
    gray: np.ndarray
    mask: np.ndarray
    origin_row: float
    origin_col: float
    width: int
    height: int
    radius: float
    num_levels: int
    angle_step: float
    metric: str

    def summary(self) -> str:
        return f"NCC 模型  已教導  {self.width}×{self.height}  {self.num_levels} 層"


def create_ncc_model(image, num_levels: int, angle_step, metric: str) -> NccModel:
    gray = image.gray()
    if image.domain is not None and int(cv2.countNonZero(image.domain)):
        mask = np.where(image.domain > 0, 255, 0).astype(np.uint8)
        ys, xs = np.where(mask > 0)
        if len(xs) < 64:
            raise ValueError("教導區域太小。請把教導框拉大，或確認 domain 有內容。")
        top, bottom = int(ys.min()), int(ys.max())
        left, right = int(xs.min()), int(xs.max())
        gray = gray[top : bottom + 1, left : right + 1]
        mask = mask[top : bottom + 1, left : right + 1]
    else:
        mask = np.full(gray.shape, 255, np.uint8)
    if min(gray.shape) < 8:
        raise ValueError("教導框至少要 8×8。")
    height, width = gray.shape
    origin_row = (height - 1) / 2.0
    origin_col = (width - 1) / 2.0
    radius = float(max(np.hypot(origin_col, origin_row), 1.0))
    if not num_levels:
        num_levels = 1
        probe_h, probe_w = height, width
        while min(probe_h, probe_w) >= 48 and num_levels < 4:
            probe_h //= 2
            probe_w //= 2
            num_levels += 1
    num_levels = max(1, min(int(num_levels), 4))
    if angle_step is None or angle_step == 0:
        angle_step = max(np.deg2rad(0.5), 0.9 / radius)
    return NccModel(
        gray=gray.astype(np.float32),
        mask=mask,
        origin_row=origin_row,
        origin_col=origin_col,
        width=int(width),
        height=int(height),
        radius=radius,
        num_levels=num_levels,
        angle_step=float(angle_step),
        metric=metric,
    )


def _pyramid(gray: np.ndarray, levels: int):
    images = [gray.astype(np.float32)]
    for _ in range(1, levels):
        images.append(cv2.pyrDown(images[-1]))
    return images


def _template_at(model: NccModel, level: int):
    gray = model.gray
    mask = model.mask
    for _ in range(level):
        gray = cv2.pyrDown(gray)
        mask = cv2.pyrDown(mask)
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    origin_row = (gray.shape[0] - 1) / 2.0
    origin_col = (gray.shape[1] - 1) / 2.0
    return gray, mask, origin_row, origin_col


def _points(gray: np.ndarray, mask: np.ndarray, origin_row: float, origin_col: float, limit: int):
    ys, xs = np.where(mask > 0)
    if len(xs) > limit:
        step = int(math.ceil(len(xs) / limit))
        ys, xs = ys[::step], xs[::step]
    rel_col = xs.astype(np.float32) - np.float32(origin_col)
    rel_row = ys.astype(np.float32) - np.float32(origin_row)
    values = gray[ys, xs].astype(np.float32)
    return rel_col, rel_row, values


def _rotate_patch(gray: np.ndarray, mask: np.ndarray, angle: float):
    height, width = gray.shape
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, np.rad2deg(angle), 1.0)
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    rotated = corners @ matrix[:, :2].T + matrix[:, 2]
    min_xy = rotated.min(axis=0)
    max_xy = rotated.max(axis=0)
    matrix = matrix.copy()
    matrix[0, 2] -= float(min_xy[0])
    matrix[1, 2] -= float(min_xy[1])
    new_w = max(1, int(math.ceil(float(max_xy[0] - min_xy[0]))) + 1)
    new_h = max(1, int(math.ceil(float(max_xy[1] - min_xy[1]))) + 1)
    warped = cv2.warpAffine(gray, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)
    warped_mask = cv2.warpAffine(mask, matrix, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)
    origin_col = float(matrix[0, 0] * center[0] + matrix[0, 1] * center[1] + matrix[0, 2])
    origin_row = float(matrix[1, 0] * center[0] + matrix[1, 1] * center[1] + matrix[1, 2])
    return warped, warped_mask, origin_row, origin_col


def _sample(image: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    height, width = image.shape
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < width) & (y1 < height)
    safe_x0 = np.clip(x0, 0, width - 1)
    safe_y0 = np.clip(y0, 0, height - 1)
    safe_x1 = np.clip(x1, 0, width - 1)
    safe_y1 = np.clip(y1, 0, height - 1)
    wx = (xs - x0).astype(np.float32)
    wy = (ys - y0).astype(np.float32)
    top = image[safe_y0, safe_x0] * (1 - wx) + image[safe_y0, safe_x1] * wx
    bottom = image[safe_y1, safe_x0] * (1 - wx) + image[safe_y1, safe_x1] * wx
    return (top * (1 - wy) + bottom * wy).astype(np.float32), valid


def _ncc(left: np.ndarray, right: np.ndarray) -> float:
    left = left - float(left.mean())
    right = right - float(right.mean())
    denom = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    if denom < 1e-6:
        return 0.0
    return float(np.dot(left, right) / denom)


def score_at(scene, rel_col, rel_row, values, row: float, col: float, angle: float, metric: str) -> float:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    sample_x = col + rel_col * cosine + rel_row * sine
    sample_y = row - rel_col * sine + rel_row * cosine
    sampled, valid = _sample(scene, sample_x, sample_y)
    if int(valid.sum()) < max(16, int(0.65 * len(values))):
        return 0.0
    score = _ncc(values[valid], sampled[valid])
    if metric == "ignore_global_polarity":
        score = abs(score)
    return score


def _parabola(left: float, center: float, right: float) -> float:
    denom = left - 2 * center + right
    if abs(denom) < 1e-6 or center < left or center < right:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denom, -0.75, 0.75))


def _angles(start: float, extent: float, step: float) -> list[float]:
    if extent <= 0 or step <= 0:
        return [start]
    values = []
    angle = start
    stop = start + extent
    guard = 0
    while angle < stop - 1e-6 and guard < 400:
        values.append(angle)
        angle += step
        guard += 1
    return values or [start]


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


def _peaks(score: np.ndarray, minimum: float, limit: int):
    if score.size == 0:
        return []
    dilated = cv2.dilate(score, np.ones((5, 5), np.uint8))
    mask = (score >= minimum) & (score >= dilated - 1e-6) & np.isfinite(score)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    order = np.argsort(score[ys, xs])[::-1]
    found = []
    for index in order[: limit * 3]:
        found.append((float(ys[index]), float(xs[index]), float(score[ys[index], xs[index]])))
        if len(found) >= limit:
            break
    return found


def _coarse_step(model: NccModel, level: int) -> float:
    scale = float(2**level)
    radius = max(model.radius / scale, 1.0)
    widest = 2.2 / radius
    step = min(max(model.angle_step * scale, model.angle_step), max(model.angle_step, widest))
    return float(max(step, model.angle_step))


def find_ncc_model(
    image,
    model: NccModel,
    angle_start: float,
    angle_extent: float,
    min_score: float,
    num_matches: int,
    max_overlap: float,
    subpixel: bool,
    num_levels: int,
):
    gray = image.gray().astype(np.float32)
    levels_used = model.num_levels if not num_levels else max(1, min(int(num_levels), model.num_levels))
    pyramid = _pyramid(gray, levels_used)
    coarse = levels_used - 1
    step = _coarse_step(model, coarse)
    level_gray, level_mask, _origin_row, _origin_col = _template_at(model, coarse)
    found = []
    scene = pyramid[coarse]
    for angle in _angles(angle_start, angle_extent, step):
        warped, warped_mask, origin_row, origin_col = _rotate_patch(level_gray, level_mask, angle)
        if warped.shape[0] >= scene.shape[0] or warped.shape[1] >= scene.shape[1]:
            continue
        if int(cv2.countNonZero(warped_mask)) < 16:
            continue
        try:
            result = cv2.matchTemplate(scene, warped, cv2.TM_CCOEFF_NORMED, mask=warped_mask)
        except cv2.error:
            result = cv2.matchTemplate(scene, warped, cv2.TM_CCOEFF_NORMED)
        for peak_row, peak_col, value in _peaks(result, min_score * 0.72, 24):
            found.append((peak_row + origin_row, peak_col + origin_col, angle, value))
    found.sort(key=lambda item: item[3], reverse=True)
    found = _suppress(found, max(4.0, model.radius / (2**coarse) * 0.45), 48, np.deg2rad(18))

    for level_index in range(coarse - 1, -1, -1):
        refined = []
        scale_row = pyramid[level_index].shape[0] / pyramid[level_index + 1].shape[0]
        scale_col = pyramid[level_index].shape[1] / pyramid[level_index + 1].shape[1]
        patch, patch_mask, origin_row, origin_col = _template_at(model, level_index)
        rel_col, rel_row, values = _points(patch, patch_mask, origin_row, origin_col, 700)
        scene = pyramid[level_index]
        window = _coarse_step(model, level_index + 1)
        for row, col, angle, _score in found:
            best = (0.0, row * scale_row, col * scale_col, angle)
            for angle_try in (angle - window, angle, angle + window):
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        rr = row * scale_row + dy
                        cc = col * scale_col + dx
                        value = score_at(scene, rel_col, rel_row, values, rr, cc, angle_try, model.metric)
                        if value > best[0]:
                            best = (value, rr, cc, angle_try)
            if best[0] >= min_score * 0.75:
                refined.append((best[1], best[2], best[3], best[0]))
        refined.sort(key=lambda item: item[3], reverse=True)
        found = _suppress(refined, max(3.0, model.radius / (2**level_index) * 0.5), 48, np.deg2rad(18))
        if not found:
            break

    poses = []
    if found and subpixel:
        rel_col, rel_row, values = _points(model.gray, model.mask, model.origin_row, model.origin_col, 2200)
        scene = pyramid[0]
        step = max(model.angle_step, np.deg2rad(0.6))
        for row, col, angle, _score in found:
            for _ in range(2):
                def value(rr, cc, aa, step=step):
                    return score_at(scene, rel_col, rel_row, values, rr, cc, aa, model.metric)

                row += _parabola(value(row - 1, col, angle), value(row, col, angle), value(row + 1, col, angle))
                col += _parabola(value(row, col - 1, angle), value(row, col, angle), value(row, col + 1, angle))
                angle += _parabola(value(row, col, angle - step), value(row, col, angle), value(row, col, angle + step)) * step
                step *= 0.5
            score = score_at(scene, rel_col, rel_row, values, row, col, angle, model.metric)
            if score >= min_score:
                poses.append((row, col, angle, score))
    else:
        poses = [item for item in found if item[3] >= min_score]
    poses.sort(key=lambda item: item[3], reverse=True)
    min_dist = model.radius * max(0.25, 1.0 - max_overlap)
    poses = _suppress(poses, min_dist, num_matches)
    return (
        [item[0] for item in poses],
        [item[1] for item in poses],
        [item[2] for item in poses],
        [item[3] for item in poses],
    )


def marker_contours(row: float, col: float, angle: float, length: float) -> list:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    arm = max(8.0, length * 0.22)
    tip_col = col + length * cosine
    tip_row = row - length * sine
    head = length * 0.28
    wing = head * 0.65
    neck_col = tip_col - head * cosine
    neck_row = tip_row + head * sine
    return [
        [[col - arm, row], [col + arm, row]],
        [[col, row - arm], [col, row + arm]],
        [[col, row], [tip_col, tip_row]],
        [
            [neck_col + wing * sine, neck_row + wing * cosine],
            [tip_col, tip_row],
            [neck_col - wing * sine, neck_row - wing * cosine],
        ],
    ]


def format_degrees(angle: float) -> str:
    degrees = (math.degrees(float(angle)) + 180.0) % 360.0 - 180.0
    return f"{degrees:.3f}°"


def angle_label(row: float, col: float, angle: float, length: float) -> tuple[str, float, float]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    text_col = col + length * cosine + 8 * sine
    text_row = row - length * sine + 8 * cosine
    return format_degrees(angle), text_row, text_col


__all__ = [
    "MATCH_COLORS",
    "NccModel",
    "angle_label",
    "color_for_match",
    "create_ncc_model",
    "find_ncc_model",
    "format_degrees",
    "marker_contours",
]
