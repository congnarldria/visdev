"""Normalized cross-correlation matching.

The template is taught from an image, optionally a rectangle. Its origin is the
center of that taught region. A pyramid of matchTemplate searches rotation and
translation. interpolation then places the peak with a Keys cubic (a = -0.5)
on the score samples around that integer. A flat score top stays on the integer
instead of sliding with a tiny slope. least_squares still starts from the
integer peak and refines row and column with Lucas-Kanade on the gray residual,
keeping the step only when the NCC score does not fall.
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
    angle_start: float = -math.pi
    angle_extent: float = math.tau
    native: object = None

    def summary(self) -> str:
        return f"NCC 模型  已教導  {self.width}×{self.height}  {self.num_levels} 層"


def _match_backend():
    try:
        from . import visdev_match
    except ImportError:
        return None
    return visdev_match


def create_ncc_model(image, num_levels: int, angle_step, metric: str, angle_start=-math.pi, angle_extent=math.tau) -> NccModel:
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
        while min(probe_h, probe_w) >= 28 and num_levels < 5:
            probe_h //= 2
            probe_w //= 2
            num_levels += 1
    num_levels = max(1, min(int(num_levels), 4))
    if angle_step is None or angle_step == 0:
        angle_step = max(np.deg2rad(0.5), 0.9 / radius)
    model = NccModel(
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
        angle_start=float(angle_start),
        angle_extent=float(angle_extent),
    )
    backend = _match_backend()
    if backend is not None:
        model.native = backend.ncc_create(
            np.ascontiguousarray(model.gray, dtype=np.float32),
            np.ascontiguousarray(model.mask),
            int(model.num_levels),
            float(angle_start),
            float(angle_extent),
            float(model.angle_step),
            1 if metric == "ignore_global_polarity" else 0,
        )
    return model


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


def _keys_weight(distance: float) -> float:
    span = abs(distance)
    if span < 1.0:
        return ((1.5 * span - 2.5) * span * span) + 1.0
    if span < 2.0:
        return ((-0.5 * span + 2.5) * span - 4.0) * span + 2.0
    return 0.0


def _keys_peak(samples: np.ndarray) -> float:
    """Offset, in samples, of the Keys cubic maximum on five neighboring scores.

    Index 2 is the integer under test. A flat top returns 0 so a tiny slope
    cannot drag the peak. The maximum of a tied top stays on the sample nearest
    that integer.
    """
    values = np.asarray(samples, dtype=np.float64)
    peak = float(values.max())
    if peak <= 0.0 or peak - float(values.min()) < 1e-4:
        return 0.0
    tied = [int(index) for index in np.flatnonzero(values >= peak - 1e-5)]
    best = min(tied, key=lambda index: (abs(index - 2), index))
    if best == 0 or best == 4:
        return float(best - 2)
    center = float(values[best])
    if min(center - float(values[best - 1]), center - float(values[best + 1])) < 2e-3:
        return float(best - 2)
    best_pos = float(best)
    best_value = -1e9
    position = float(best - 1)
    while position <= best + 1.0 + 1e-9:
        value = sum(_keys_weight(position - index) * float(values[index]) for index in range(5))
        if value > best_value + 1e-8:
            best_value = value
            best_pos = position
        position += 0.01
    return best_pos - 2.0


def _integer_offsets(gray: np.ndarray, mask: np.ndarray, angle: float, limit: int = 768):
    """Same point set the native matcher uses for an integer-pixel score.

    Relative coordinates are rounded to the nearest pixel, so the best
    placement is an integer row and column. An even-sized box then sits on
    40 rather than on the half-pixel center 39.5.
    """
    height, width = gray.shape
    stride = 1
    while stride < 32 and (height // stride) * (width // stride) > limit:
        stride += 1
    origin_row = (height - 1) * 0.5
    origin_col = (width - 1) * 0.5
    dx = []
    dy = []
    values = []
    for row in range(0, height, stride):
        for col in range(0, width, stride):
            if mask[row, col] == 0:
                continue
            dx.append(col - origin_col)
            dy.append(row - origin_row)
            values.append(float(gray[row, col]))
    if len(values) < 16:
        return None
    rel_col = np.asarray(dx, np.float64)
    rel_row = np.asarray(dy, np.float64)
    samples = np.asarray(values, np.float64)
    if angle:
        cosine = math.cos(angle)
        sine = math.sin(angle)
        rotated_col = rel_col * cosine + rel_row * sine
        rotated_row = -rel_col * sine + rel_row * cosine
        rel_col, rel_row = rotated_col, rotated_row
    samples = samples - float(samples.mean())
    norm = float(np.sqrt(np.dot(samples, samples)))
    return np.rint(rel_col).astype(np.int32), np.rint(rel_row).astype(np.int32), samples, norm


def _integer_score(scene: np.ndarray, ix, iy, samples, norm: float, row: int, col: int, ignore: bool) -> float:
    rows = row + iy
    cols = col + ix
    height, width = scene.shape
    if int(rows.min()) < 0 or int(cols.min()) < 0 or int(rows.max()) >= height or int(cols.max()) >= width:
        return -1.0
    pixels = scene[rows, cols].astype(np.float64)
    dot = float(np.dot(samples, pixels))
    energy = float(np.dot(pixels, pixels) - (float(pixels.sum()) ** 2) / len(pixels))
    if energy < 1e-6 or norm < 1e-6:
        return 0.0
    score = dot / (norm * math.sqrt(energy))
    if ignore:
        score = abs(score)
    return float(max(-1.0, min(1.0, score)))


def _snap_integer(scene: np.ndarray, model: NccModel, row: float, col: float, angle: float):
    packed = _integer_offsets(model.gray, model.mask, angle)
    if packed is None:
        return float(row), float(col), 0.0
    ix, iy, samples, norm = packed
    ignore = model.metric == "ignore_global_polarity"
    base_row = int(round(row))
    base_col = int(round(col))
    best = (-2.0, base_row, base_col)
    for d_row in range(-4, 5):
        for d_col in range(-4, 5):
            score = _integer_score(scene, ix, iy, samples, norm, base_row + d_row, base_col + d_col, ignore)
            if score > best[0]:
                best = (score, base_row + d_row, base_col + d_col)
    return float(best[1]), float(best[2]), best[0]


def _axis_peak(value_at, center: float, spacing: float = 1.0) -> float:
    offsets = (-2, -1, 0, 1, 2)

    def samples(origin: float) -> np.ndarray:
        return np.array([value_at(origin + spacing * offset) for offset in offsets], dtype=np.float64)

    window = samples(center)
    offset = _keys_peak(window)
    if abs(offset) > 1.0:
        center = center + math.copysign(spacing, offset)
        offset = float(np.clip(_keys_peak(samples(center)), -1.0, 1.0))
    return center + offset * spacing


def _interpolate_ncc(image, model: NccModel, rows, cols, angles, scores, min_score: float):
    scene = image.gray().astype(np.float32)
    rel_col, rel_row, values = _points(model.gray, model.mask, model.origin_row, model.origin_col, 2200)
    step = max(float(model.angle_step), math.radians(0.6))
    refine_angle = float(model.angle_extent) > 1e-6
    poses = []
    for row, col, angle, _score in zip(rows, cols, angles, scores):
        row = _axis_peak(lambda rr, col=col, angle=angle: score_at(scene, rel_col, rel_row, values, rr, col, angle, model.metric), float(row))
        col = _axis_peak(lambda cc, row=row, angle=angle: score_at(scene, rel_col, rel_row, values, row, cc, angle, model.metric), float(col))
        if refine_angle:
            angle = _axis_peak(
                lambda aa, row=row, col=col: score_at(scene, rel_col, rel_row, values, row, col, aa, model.metric),
                float(angle),
                step,
            )
        else:
            angle = float(angle)
        score = score_at(scene, rel_col, rel_row, values, row, col, angle, model.metric)
        if score >= min_score:
            poses.append((row, col, angle, score))
    return poses


def _ncc_mode(subpixel) -> str:
    if subpixel is True or subpixel == "interpolation":
        return "interpolation"
    if subpixel is False or subpixel in ("none", 0):
        return "none"
    if subpixel == "least_squares":
        return "least_squares"
    return "interpolation"


def _lucas_kanade(scene, rel_col, rel_row, values, row: float, col: float, angle: float, metric: str):
    step = max(1, int(math.ceil(len(values) / 500)))
    point_col = rel_col[::step]
    point_row = rel_row[::step]
    template = values[::step].astype(np.float64)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    offset_col = point_col * cosine + point_row * sine
    offset_row = -point_col * sine + point_row * cosine
    pose = (float(row), float(col))
    best = score_at(scene, rel_col, rel_row, values, pose[0], pose[1], angle, metric)
    for _turn in range(10):
        row, col = pose
        sampled, valid = _sample(scene, col + offset_col, row + offset_row)
        right, valid_right = _sample(scene, col + offset_col + 0.5, row + offset_row)
        left, valid_left = _sample(scene, col + offset_col - 0.5, row + offset_row)
        down, valid_down = _sample(scene, col + offset_col, row + offset_row + 0.5)
        up, valid_up = _sample(scene, col + offset_col, row + offset_row - 0.5)
        keep = valid & valid_right & valid_left & valid_down & valid_up
        if int(np.count_nonzero(keep)) < 16:
            break
        reference = template[keep]
        image = sampled[keep].astype(np.float64)
        slope_col = (right[keep] - left[keep]).astype(np.float64)
        slope_row = (down[keep] - up[keep]).astype(np.float64)
        residual = (reference - float(reference.mean())) - (image - float(image.mean()))
        jac_col = slope_col - float(slope_col.mean())
        jac_row = slope_row - float(slope_row.mean())
        delta, *_rest = np.linalg.lstsq(np.column_stack([jac_col, jac_row]), residual, rcond=None)
        accepted = False
        scale = 1.0
        while scale >= 0.1:
            cand_col = col + float(np.clip(delta[0] * scale, -0.6, 0.6))
            cand_row = row + float(np.clip(delta[1] * scale, -0.6, 0.6))
            cand = score_at(scene, rel_col, rel_row, values, cand_row, cand_col, angle, metric)
            if cand + 1e-6 >= best:
                pose = (cand_row, cand_col)
                best = cand
                accepted = True
                break
            scale *= 0.5
        if not accepted or abs(pose[0] - row) + abs(pose[1] - col) < 1e-3:
            break
    return pose[0], pose[1], float(angle), best


def _refine_ncc(image, model: NccModel, rows, cols, angles, scores):
    scene = image.gray().astype(np.float32)
    rel_col, rel_row, values = _points(model.gray, model.mask, model.origin_row, model.origin_col, 800)
    refined_rows, refined_cols, refined_angles, refined_scores = [], [], [], []
    for row, col, angle, score in zip(rows, cols, angles, scores):
        nrow, ncol, nangle, new = _lucas_kanade(
            scene, rel_col, rel_row, values, float(row), float(col), float(angle), model.metric
        )
        moved = abs(nrow - float(row)) + abs(ncol - float(col)) > 1e-3
        refined_rows.append(nrow if moved else float(row))
        refined_cols.append(ncol if moved else float(col))
        refined_angles.append(nangle)
        refined_scores.append(new if moved else float(score))
    return refined_rows, refined_cols, refined_angles, refined_scores


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
    mode = _ncc_mode(subpixel)
    backend = _match_backend()
    if backend is not None and getattr(model, "native", None) is not None:
        gray = np.ascontiguousarray(image.gray())
        rows, cols, angles, scores = backend.ncc_find(
            model.native,
            gray,
            float(angle_start),
            float(angle_extent),
            float(min_score),
            int(num_matches),
            float(max_overlap),
            0,
            int(num_levels),
        )
        if mode == "least_squares":
            return _refine_ncc(image, model, rows, cols, angles, scores)
        if mode == "interpolation" and rows:
            poses = _interpolate_ncc(image, model, rows, cols, angles, scores, float(min_score))
            return (
                [item[0] for item in poses],
                [item[1] for item in poses],
                [item[2] for item in poses],
                [item[3] for item in poses],
            )
        return rows, cols, angles, scores
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

    if found:
        full = image.gray().astype(np.float32)
        snapped = []
        for row, col, angle, score in found:
            integer_row, integer_col, integer_score = _snap_integer(full, model, row, col, angle)
            snapped.append((integer_row, integer_col, angle, integer_score if integer_score > 0 else score))
        found = snapped

    if found and mode == "interpolation":
        poses = _interpolate_ncc(
            image,
            model,
            [item[0] for item in found],
            [item[1] for item in found],
            [item[2] for item in found],
            [item[3] for item in found],
            float(min_score),
        )
    else:
        poses = [item for item in found if item[3] >= min_score]
    poses.sort(key=lambda item: item[3], reverse=True)
    min_dist = model.radius * max(0.25, 1.0 - max_overlap)
    poses = _suppress(poses, min_dist, num_matches)
    rows = [item[0] for item in poses]
    cols = [item[1] for item in poses]
    angles = [item[2] for item in poses]
    scores = [item[3] for item in poses]
    if mode == "least_squares" and rows:
        return _refine_ncc(image, model, rows, cols, angles, scores)
    return rows, cols, angles, scores


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
