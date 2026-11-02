"""XLD clipping, polygon approximation, and affine transform."""

from __future__ import annotations

import math

import numpy as np
import cv2

from ..objects import Image, Region, Result, XLD
from ..operators import (
    P,
    XLD_FEATURE_CHOICES,
    _sample_rectangle,
    _fit_line,
    _xld_feature_name,
    as_list,
    equivalent_ellipse,
    expect_image,
    expect_region,
    expect_xld,
    image_out,
    operator,
    scalar,
    text,
    whole,
    xld_measures,
)
from .image_ops import _apply_proj, _mat_from, _proj_from
from .region_ops import skeleton


def _inside(point, row1, col1, row2, col2) -> bool:
    col, row = float(point[0]), float(point[1])
    return row1 <= row <= row2 and col1 <= col <= col2


@operator(
    "輪廓",
    [P("Contours", "輸入輪廓")],
    [P("ClippedContours", "裁切後輪廓")],
    [P("Row1", "上"), P("Column1", "左"), P("Row2", "下"), P("Column2", "右")],
    [],
    "用矩形裁切輪廓",
    "落在矩形外的點會被拿掉。一條輪廓若被裁成幾段，就變成幾條。少於兩個點的片段丟掉。",
    "Vizion.clip_contours_xld (Contours, Clipped, 0, 0, 100, 100)",
)
def clip_contours_xld(contours, row1, column1, row2, column2, *, ctx):
    source = expect_xld(contours, "Contours")
    top, left = scalar(row1, "Row1"), scalar(column1, "Column1")
    bottom, right = scalar(row2, "Row2"), scalar(column2, "Column2")
    if bottom < top or right < left:
        raise ValueError("矩形的右下角必須在左上角的右下方")
    pieces = []
    for contour in source.contours:
        current = []
        for point in contour:
            if _inside(point, top, left, bottom, right):
                current.append([float(point[0]), float(point[1])])
            elif len(current) >= 2:
                pieces.append(current)
                current = []
            else:
                current = []
        if len(current) >= 2:
            pieces.append(current)
    return Result([XLD(source.width, source.height, pieces, multi_color=len(pieces) > 1)], [])


@operator(
    "輪廓",
    [P("Contours", "輸入輪廓")],
    [P("Polygons", "近似後的多邊形")],
    [P("Type", "ramer 或 douglas，都是 Douglas-Peucker", "'ramer'", ["'ramer'", "'douglas'"]), P("Alpha", "允許偏離原輪廓的最大距離")],
    [],
    "把輪廓近似成多邊形",
    "ramer 和 douglas 都是 Douglas-Peucker。Alpha 是頂點可以離開原輪廓的最大距離，單位是像素。",
    "Vizion.gen_polygons_xld (Contours, Polygons, 'ramer', 2)",
)
def gen_polygons_xld(contours, kind, alpha, *, ctx):
    source = expect_xld(contours, "Contours")
    mode = text(kind, "Type").lower()
    if mode not in ("ramer", "douglas"):
        raise ValueError("Type 只能是 ramer 或 douglas")
    epsilon = scalar(alpha, "Alpha")
    if epsilon < 0:
        raise ValueError("Alpha 不能是負的")
    pieces = []
    for contour in source.contours:
        if len(contour) < 2:
            continue
        pts = np.asarray(contour, np.float32).reshape(-1, 1, 2)
        closed = np.linalg.norm(pts[0, 0] - pts[-1, 0]) < 1.5
        approx = cv2.approxPolyDP(pts, epsilon, bool(closed))
        if len(approx) >= 2:
            pieces.append([[float(point[0][0]), float(point[0][1])] for point in approx])
    return Result([XLD(source.width, source.height, pieces, multi_color=len(pieces) > 1)], [])


@operator(
    "輪廓",
    [P("Contours", "輸入輪廓")],
    [P("ContoursAffineTrans", "轉換後輪廓")],
    [P("HomMat2D", "仿射矩陣")],
    [],
    "用仿射矩陣轉換輪廓",
    "和 affine_trans_pixel 同一套矩陣：x 是列、y 是行。點的存放順序仍是先行、再列。",
    "Vizion.affine_trans_contour_xld (Contours, ContoursAffineTrans, HomMat2D)",
)
def affine_trans_contour_xld(contours, hom_mat2d, *, ctx):
    source = expect_xld(contours, "Contours")
    matrix = _mat_from(hom_mat2d)
    pieces = []
    width = source.width
    height = source.height
    for contour in source.contours:
        moved = []
        for col, row in contour:
            x = float(row)
            y = float(col)
            row2 = float(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2])
            col2 = float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2])
            moved.append([col2, row2])
            width = max(width, int(np.ceil(col2)) + 1)
            height = max(height, int(np.ceil(row2)) + 1)
        if len(moved) >= 2:
            pieces.append(moved)
    return Result([XLD(max(width, 1), max(height, 1), pieces, multi_color=len(pieces) > 1)], [])


def _fit_canvas(ctx, contours: list) -> XLD:
    height, width = ctx.canvas_size()
    for contour in contours:
        for col, row in contour:
            width = max(width, int(math.ceil(float(col))) + 1)
            height = max(height, int(math.ceil(float(row))) + 1)
    return XLD(max(width, 1), max(height, 1), contours, multi_color=len(contours) > 1)


def _arc(row, col, radius1, radius2, phi, start, end, resolution, order) -> list:
    if radius1 <= 0 or radius2 <= 0:
        raise ValueError("半徑必須大於 0")
    if resolution <= 0:
        raise ValueError("Resolution 必須大於 0")
    mode = text(order, "PointOrder").lower()
    if mode not in ("positive", "negative"):
        raise ValueError("PointOrder 只能是 positive 或 negative")
    span = float(end) - float(start)
    while span <= 1e-9:
        span += 2 * math.pi
    steps = int(math.ceil(max(radius1, radius2) * span / resolution))
    steps = min(max(steps, 2), 4096)
    angles = np.linspace(float(start), float(start) + span, steps + 1)
    ex = radius1 * np.cos(angles)
    ey = radius2 * np.sin(angles)
    cosine = math.cos(phi)
    sine = math.sin(phi)
    cols = col + ex * cosine - ey * sine
    rows = row + ex * sine + ey * cosine
    points = [[float(item_col), float(item_row)] for item_col, item_row in zip(cols, rows)]
    if mode == "negative":
        points.reverse()
    return points


_POINT_ORDER = ["'positive'", "'negative'"]


@operator(
    "輪廓",
    [],
    [P("ContCircle", "圓或圓弧")],
    [
        P("Row", "圓心列"),
        P("Column", "圓心行"),
        P("Radius", "半徑"),
        P("StartPhi", "起始角，弳度"),
        P("EndPhi", "結束角，弳度"),
        P("PointOrder", "positive 或 negative", "'positive'", _POINT_ORDER),
        P("Resolution", "相鄰點的大約間距", "1"),
    ],
    [],
    "產生圓或圓弧輪廓",
    "角度 0 朝右，正角度在畫面上順時針。結束角小於等於起始角時會繞過一圈。negative 會把點序反過來。",
    "Vizion.gen_circle_contour_xld (ContCircle, 15, 20, 8, 0, rad(360), 'positive', 1)",
)
def gen_circle_contour_xld(row, column, radius, start_phi, end_phi, point_order, resolution, *, ctx):
    radius_value = scalar(radius, "Radius")
    points = _arc(
        scalar(row, "Row"), scalar(column, "Column"), radius_value, radius_value, 0.0,
        scalar(start_phi, "StartPhi"), scalar(end_phi, "EndPhi"), scalar(resolution, "Resolution"), point_order,
    )
    return Result([_fit_canvas(ctx, [points])], [])


@operator(
    "輪廓",
    [],
    [P("ContEllipse", "橢圓或橢圓弧")],
    [
        P("Row", "中心列"),
        P("Column", "中心行"),
        P("Phi", "第一軸角度，弳度"),
        P("Radius1", "第一軸半徑"),
        P("Radius2", "第二軸半徑"),
        P("StartPhi", "起始角，弳度"),
        P("EndPhi", "結束角，弳度"),
        P("PointOrder", "positive 或 negative", "'positive'", _POINT_ORDER),
        P("Resolution", "相鄰點的大約間距", "1"),
    ],
    [],
    "產生橢圓輪廓",
    "角度和 gen_ellipse 相同。點距是大約值，最長軸用來決定點數。",
    "Vizion.gen_ellipse_contour_xld (ContEllipse, 15, 20, 0, 12, 6, 0, rad(360), 'positive', 1)",
)
def gen_ellipse_contour_xld(row, column, phi, radius1, radius2, start_phi, end_phi, point_order, resolution, *, ctx):
    points = _arc(
        scalar(row, "Row"), scalar(column, "Column"), scalar(radius1, "Radius1"), scalar(radius2, "Radius2"), scalar(phi, "Phi"),
        scalar(start_phi, "StartPhi"), scalar(end_phi, "EndPhi"), scalar(resolution, "Resolution"), point_order,
    )
    return Result([_fit_canvas(ctx, [points])], [])


@operator(
    "輪廓",
    [],
    [P("Rectangle", "封閉矩形輪廓")],
    [
        P("Row", "中心列"),
        P("Column", "中心行"),
        P("Phi", "長邊角度，弳度。0 沿著行"),
        P("Length1", "半長"),
        P("Length2", "半寬"),
    ],
    [],
    "產生旋轉矩形輪廓",
    "Phi 0 時長邊是水平的。正角度在畫面上順時針。輪廓是封閉的五個點。",
    "Vizion.gen_rectangle2_contour_xld (Rectangle, 15, 20, 0, 10, 4)",
)
def gen_rectangle2_contour_xld(row, column, phi, length1, length2, *, ctx):
    long_side = scalar(length1, "Length1")
    short_side = scalar(length2, "Length2")
    if long_side < 0 or short_side < 0:
        raise ValueError("長度不能是負的")
    points = _sample_rectangle(scalar(row, "Row"), scalar(column, "Column"), scalar(phi, "Phi"), long_side, short_side)
    return Result([_fit_canvas(ctx, [points])], [])


def _right_offset(points: list, distance: float) -> list:
    pts = np.asarray(points, np.float64)
    if len(pts) < 2:
        return []
    forward = np.zeros_like(pts)
    forward[:-1] = pts[1:] - pts[:-1]
    forward[-1] = forward[-2]
    backward = np.zeros_like(pts)
    backward[1:] = pts[1:] - pts[:-1]
    backward[0] = backward[1]
    tangent = forward + backward
    dc = tangent[:, 0]
    dr = tangent[:, 1]
    norm = np.maximum(np.hypot(dc, dr), 1e-9)
    shifted = np.column_stack([pts[:, 0] + distance * (-dr) / norm, pts[:, 1] + distance * dc / norm])
    return shifted.tolist()


def _regression_offset(points: list, distance: float) -> list:
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return []
    start, end = _fit_line(pts.tolist())
    direction = np.array([end[0] - start[0], end[1] - start[1]], np.float64)
    travel = pts[-1] - pts[0]
    if float(np.hypot(travel[0], travel[1])) < 1e-9 and len(pts) >= 3:
        travel = pts[len(pts) // 2] - pts[0]
    if float(direction @ travel) < 0:
        direction = -direction
    length = float(np.hypot(direction[0], direction[1]))
    if length < 1e-9:
        return _right_offset(points, distance)
    dc = direction[0] / length
    dr = direction[1] / length
    moved = np.column_stack([pts[:, 0] + distance * (-dr), pts[:, 1] + distance * dc])
    return moved.tolist()


def _as_points(contour) -> np.ndarray:
    pts = np.asarray(contour, np.float64).reshape(-1, 2)
    if len(pts) == 0:
        raise ValueError("輪廓是空的")
    return pts


@operator(
    "輪廓",
    [P("Contours", "輸入輪廓")],
    [P("ParallelContours", "平移後的輪廓")],
    [
        P("Mode", "contour_normal 或 regression_normal", "'contour_normal'", ["'contour_normal'", "'regression_normal'"]),
        P("Distance", "平移距離，像素。正值在行進方向的右側"),
    ],
    [],
    "把輪廓沿法線平移",
    "正距離在畫面上是行進方向的右側。contour_normal 沿每個點自己的法線。regression_normal 先把整條輪廓擬合成直線，再沿那條直線的右側法線平移，所以直線上的一個尖角不會把旁邊的點往側面推。",
    "Vizion.gen_parallel_contour_xld (Contours, ParallelContours, 'contour_normal', 4)",
)
def gen_parallel_contour_xld(contours, mode, distance, *, ctx):
    source = expect_xld(contours, "Contours")
    kind = text(mode, "Mode").lower()
    if kind not in ("contour_normal", "regression_normal"):
        raise ValueError("Mode 只能是 contour_normal 或 regression_normal")
    gap = scalar(distance, "Distance")
    pieces = []
    for contour in source.contours:
        moved = _regression_offset(contour, gap) if kind == "regression_normal" else _right_offset(contour, gap)
        if len(moved) >= 2:
            pieces.append(moved)
    if not pieces:
        raise ValueError("沒有可以平移的輪廓")
    return Result([_fit_canvas(ctx, pieces)], [])


@operator(
    "輪廓",
    [P("XLD", "輪廓")],
    [],
    [],
    [P("Row1", "上"), P("Column1", "左"), P("Row2", "下"), P("Column2", "右")],
    "輪廓的正外包矩形",
    "每一條輪廓一組。座標是點的最小和最大列、行。",
    "Vizion.smallest_rectangle1_xld (XLD, Row1, Column1, Row2, Column2)",
)
def smallest_rectangle1_xld(contours, *, ctx):
    source = expect_xld(contours, "XLD")
    if not source.contours:
        raise ValueError("沒有輪廓")
    row1, col1, row2, col2 = [], [], [], []
    for contour in source.contours:
        pts = _as_points(contour)
        row1.append(float(np.min(pts[:, 1])))
        col1.append(float(np.min(pts[:, 0])))
        row2.append(float(np.max(pts[:, 1])))
        col2.append(float(np.max(pts[:, 0])))
    return Result([], [row1, col1, row2, col2])


@operator(
    "輪廓",
    [P("XLD", "輪廓")],
    [],
    [],
    [P("Row", "圓心列"), P("Column", "圓心行"), P("Radius", "半徑")],
    "輪廓的最小外包圓",
    "每一條輪廓一個圓。",
    "Vizion.smallest_circle_xld (XLD, Row, Column, Radius)",
)
def smallest_circle_xld(contours, *, ctx):
    source = expect_xld(contours, "XLD")
    if not source.contours:
        raise ValueError("沒有輪廓")
    rows, cols, radii = [], [], []
    for contour in source.contours:
        pts = _as_points(contour).astype(np.float32).reshape(-1, 1, 2)
        (col, row), radius = cv2.minEnclosingCircle(pts)
        rows.append(float(row))
        cols.append(float(col))
        radii.append(float(radius))
    return Result([], [rows, cols, radii])


@operator(
    "輪廓",
    [],
    [P("Arrow", "箭頭輪廓")],
    [
        P("Row1", "起點列"),
        P("Column1", "起點行"),
        P("Row2", "終點列"),
        P("Column2", "終點行"),
        P("HeadLength", "箭頭長度", "8"),
        P("HeadWidth", "箭頭寬度", "4"),
    ],
    [],
    "產生箭頭輪廓",
    "包含箭身和兩邊的箭頭。不填長度和寬度時用 8 和 4。",
    "Vizion.gen_arrow_contour_xld (Arrow, 5, 5, 5, 25, 6, 3)",
)
def gen_arrow_contour_xld(row1, column1, row2, column2, head_length, head_width, *, ctx):
    start = np.array([scalar(column1, "Column1"), scalar(row1, "Row1")], np.float64)
    end = np.array([scalar(column2, "Column2"), scalar(row2, "Row2")], np.float64)
    delta = end - start
    length = float(np.hypot(delta[0], delta[1]))
    if length < 1e-6:
        raise ValueError("箭頭的兩點不能重合")
    direction = delta / length
    normal = np.array([-direction[1], direction[0]])
    head = min(max(0.0, scalar(head_length, "HeadLength")), length)
    half = max(0.0, scalar(head_width, "HeadWidth")) / 2.0
    base = end - direction * head
    left = base + normal * half
    right = base - normal * half
    pieces = [
        [start.tolist(), end.tolist()],
        [left.tolist(), end.tolist()],
        [right.tolist(), end.tolist()],
    ]
    return Result([_fit_canvas(ctx, pieces)], [])


def _point_segment_min(points: np.ndarray, contour: np.ndarray) -> float:
    if len(contour) < 2:
        delta = points - contour[:1]
        return float(np.min(np.hypot(delta[:, 0], delta[:, 1])))
    start = contour[:-1]
    stop = contour[1:]
    edge = stop - start
    length2 = np.maximum(np.sum(edge * edge, axis=1), 1e-12)
    best = float("inf")
    for begin in range(0, len(points), 256):
        chunk = points[begin : begin + 256]
        offset = chunk[:, None, :] - start[None, :, :]
        scale = np.clip(np.sum(offset * edge, axis=2) / length2, 0.0, 1.0)
        proj = start[None, :, :] + scale[:, :, None] * edge
        dist = np.hypot(chunk[:, None, 0] - proj[:, :, 0], chunk[:, None, 1] - proj[:, :, 1])
        best = min(best, float(dist.min()))
    return best


@operator(
    "輪廓",
    [P("Contour1", "第一條輪廓"), P("Contour2", "第二條輪廓")],
    [],
    [P("Mode", "point_to_point 或 point_to_segment", "'point_to_segment'", ["'point_to_point'", "'point_to_segment'"])],
    [P("DistanceMin", "最近距離"), P("DistanceMax", "最遠的點距")],
    "兩條輪廓的距離",
    "最近距離在 point_to_segment 時量到線段，point_to_point 只量頂點。最遠距離都是頂點之間的最大距離。多條輪廓會先接在一起再量。",
    "Vizion.distance_cc (Contour1, Contour2, 'point_to_segment', DistanceMin, DistanceMax)",
)
def distance_cc(contour1, contour2, mode, *, ctx):
    first = expect_xld(contour1, "Contour1")
    second = expect_xld(contour2, "Contour2")
    kind = text(mode, "Mode").lower()
    if kind not in ("point_to_point", "point_to_segment"):
        raise ValueError("Mode 只能是 point_to_point 或 point_to_segment")
    if not first.contours or not second.contours:
        raise ValueError("輪廓是空的")
    left = np.vstack([_as_points(item) for item in first.contours])
    right = np.vstack([_as_points(item) for item in second.contours])
    delta = left[:, None, :] - right[None, :, :]
    pairwise = np.hypot(delta[:, :, 0], delta[:, :, 1])
    farthest = float(pairwise.max())
    if kind == "point_to_point":
        return Result([], [float(pairwise.min()), farthest])
    nearest = min(_point_segment_min(left, right), _point_segment_min(right, left))
    return Result([], [nearest, farthest])


@operator(
    "輪廓",
    [P("XLD", "要畫的輪廓"), P("Image", "底圖")],
    [P("ImageResult", "畫完的影像")],
    [P("Grayval", "灰階。彩色可給紅、綠、藍", "255")],
    [],
    "把輪廓畫進影像",
    "線寬 1 像素。不填灰階時用 255。超出影像的部分會被裁掉。這會產生新影像。",
    "Vizion.paint_xld (XLD, Image, ImageResult, 200)",
)
def paint_xld(contours, image, grayval, *, ctx):
    source = expect_xld(contours, "XLD")
    picture = expect_image(image, "Image")
    data = picture.data.copy()
    color = _gray_color(picture, grayval, "Grayval")
    for contour in source.contours:
        if len(contour) < 2:
            continue
        pts = np.int32(np.round(np.asarray(contour, np.float64))).reshape(-1, 1, 2)
        cv2.polylines(data, [pts], False, color, 1, cv2.LINE_8)
    return image_out(picture, data)


def _gray_color(image: Image, grayval, name: str):
    values = [int(np.clip(round(float(item)), 0, 255)) for item in as_list(grayval)]
    if not values:
        raise ValueError(f"{name} 不能是空的")
    if image.data.ndim == 2:
        if len(values) != 1:
            raise ValueError("灰階圖只能填一個灰階")
        return int(values[0])
    if len(values) == 1:
        return (values[0], values[0], values[0])
    if len(values) != 3:
        raise ValueError("彩色圖要一個灰階，或紅、綠、藍三個數")
    red, green, blue = values
    return (int(blue), int(green), int(red))


@operator(
    "輪廓",
    [P("Contours", "輸入輪廓")],
    [P("ContoursProjTrans", "轉換後輪廓")],
    [P("HomMat2D", "投影矩陣")],
    [],
    "用投影矩陣轉換輪廓",
    "和 projective_trans_pixel 同一套矩陣。6 個數會當成仿射。點的存放順序仍是先行、再列。",
    "Vizion.projective_trans_contour_xld (Contours, ContoursProjTrans, HomMat2D)",
)
def projective_trans_contour_xld(contours, hom_mat2d, *, ctx):
    source = expect_xld(contours, "Contours")
    matrix = _proj_from(hom_mat2d)
    pieces = []
    for contour in source.contours:
        if len(contour) < 2:
            continue
        cols = [float(point[0]) for point in contour]
        rows = [float(point[1]) for point in contour]
        moved_rows, moved_cols = _apply_proj(matrix, rows, cols)
        if not isinstance(moved_rows, list):
            moved_rows, moved_cols = [moved_rows], [moved_cols]
        pieces.append([[float(col), float(row)] for row, col in zip(moved_rows, moved_cols)])
    if not pieces:
        raise ValueError("沒有可以轉換的輪廓")
    return Result([_fit_canvas(ctx, pieces)], [])


@operator(
    "輪廓",
    [P("Region", "輸入區域")],
    [P("Contours", "骨架輪廓")],
    [],
    [],
    "把區域骨架轉成輪廓",
    "先做形態學骨架，再把細線收成輪廓。交叉處可能被拆成幾段。",
    "Vizion.gen_contours_skeleton_xld (Region, Contours)",
)
def gen_contours_skeleton_xld(region, *, ctx):
    skel = skeleton(region, ctx=ctx).iconic[0]
    pieces = []
    width, height = skel.width, skel.height
    for mask in skel.objects:
        found = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = found[0] if len(found) == 2 else found[1]
        for contour in contours:
            if len(contour) < 2:
                continue
            pieces.append([[float(point[0][0]), float(point[0][1])] for point in contour])
    if not pieces:
        raise ValueError("骨架沒有可連成線的點")
    return Result([XLD(width, height, pieces, multi_color=len(pieces) > 1)], [])


def _sample_field(field: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return cv2.remap(
        field,
        cols.astype(np.float32),
        rows.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _gauss_deriv_kernels(sigma: float):
    radius = max(1, int(math.ceil(sigma * 4.0)))
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    variance = sigma * sigma
    smooth = np.exp(-0.5 * axis * axis / variance)
    smooth /= smooth.sum()
    first = (-axis / variance) * smooth
    second = ((axis * axis / (variance * variance)) - (1.0 / variance)) * smooth
    return [item.astype(np.float32) for item in (smooth, first, second)]


def _sep(image, across, along):
    return cv2.sepFilter2D(image, cv2.CV_32F, across, along, borderType=cv2.BORDER_REFLECT_101)


def _ridge_fields(picture: Image, sigma: float, kind: str):
    """Steger's line detector: Gaussian-derivative Hessian, subpixel along the normal."""
    if sigma <= 0:
        raise ValueError("Sigma 必須大於 0")
    if kind not in ("light", "dark", "both"):
        raise ValueError("LightDark 只能是 light、dark 或 both")
    smooth, first, second = _gauss_deriv_kernels(sigma)
    gray = picture.gray().astype(np.float32)
    dx = _sep(gray, first, smooth)
    dy = _sep(gray, smooth, first)
    dxx = _sep(gray, second, smooth)
    dyy = _sep(gray, smooth, second)
    dxy = _sep(gray, first, first)
    trace = dxx + dyy
    det = dxx * dyy - dxy * dxy
    disc = np.sqrt(np.maximum(trace * trace * 0.25 - det, 0.0))
    larger = trace * 0.5 + disc
    smaller = trace * 0.5 - disc
    light = np.maximum(-smaller, 0.0)
    dark = np.maximum(larger, 0.0)
    if kind == "light":
        use_light = np.ones(gray.shape, np.bool_)
    elif kind == "dark":
        use_light = np.zeros(gray.shape, np.bool_)
    else:
        use_light = light >= dark
    strength = np.where(use_light, light, dark)
    curvature = np.where(use_light, smaller, larger)
    other = np.where(use_light, larger, smaller)
    left = dxx - curvature
    right = dyy - curvature
    prefer_left = np.abs(left) >= np.abs(right)
    normal_x = np.where(prefer_left, -dxy, right)
    normal_y = np.where(prefer_left, left, -dxy)
    norm = np.hypot(normal_x, normal_y)
    pointed = norm > 1e-6
    normal_x = np.divide(normal_x, norm, out=np.zeros_like(normal_x), where=pointed)
    normal_y = np.divide(normal_y, norm, out=np.zeros_like(normal_y), where=pointed)
    slope = normal_x * dx + normal_y * dy
    shift = np.divide(-slope, curvature, out=np.full_like(strength, 9.0), where=np.abs(curvature) > 1e-6)
    rows, cols = np.mgrid[0 : picture.height, 0 : picture.width]
    ahead = _sample_field(strength, rows + normal_y, cols + normal_x)
    behind = _sample_field(strength, rows - normal_y, cols - normal_x)
    linear = np.abs(other) <= 0.5 * np.maximum(strength, 1e-6)
    center = pointed & linear & (np.abs(shift) <= 0.5) & (strength >= ahead) & (strength >= behind) & (strength > 0)
    if picture.domain is not None:
        center &= picture.domain > 0
    return strength, normal_x, normal_y, shift, center


def _polarity(value, name: str) -> str:
    kind = text(value, name).lower()
    if kind not in ("light", "dark", "both"):
        raise ValueError(f"{name} 只能是 light、dark 或 both")
    return kind


def _link_ridge(mask: np.ndarray, shift, normal_x, normal_y) -> list[list]:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    points = {(int(row), int(col)) for row, col in zip(ys, xs)}
    neighbors = {}
    for row, col in points:
        found = []
        for d_row in (-1, 0, 1):
            for d_col in (-1, 0, 1):
                if d_row == 0 and d_col == 0:
                    continue
                nxt = (row + d_row, col + d_col)
                if nxt in points:
                    found.append(nxt)
        neighbors[(row, col)] = found

    def edge(a, b):
        return (a, b) if a <= b else (b, a)

    used = set()
    chains = []

    def walk(start, nxt):
        key = edge(start, nxt)
        if key in used:
            return
        chain = [start]
        previous, current = start, nxt
        used.add(key)
        while True:
            chain.append(current)
            choices = [item for item in neighbors[current] if edge(current, item) not in used]
            if len(choices) != 1:
                break
            previous, current = current, choices[0]
            used.add(edge(previous, current))
        if len(chain) >= 2:
            chains.append(chain)

    starts = [point for point, items in neighbors.items() if len(items) != 2]
    if not starts:
        starts = [min(points)]
    starts.sort()
    for start in starts:
        for nxt in neighbors[start]:
            walk(start, nxt)
    polylines = []
    for chain in chains:
        polyline = []
        for row, col in chain:
            moved = float(shift[row, col])
            polyline.append([float(col) + moved * float(normal_x[row, col]), float(row) + moved * float(normal_y[row, col])])
        polylines.append(polyline)
    return polylines


def _line_facts(polylines: list[list]):
    if not polylines:
        return 0, 0.0, 0.0, 0.0, 0.0
    best = None
    for polyline in polylines:
        length = 0.0
        for left, right in zip(polyline, polyline[1:]):
            length += math.hypot(right[0] - left[0], right[1] - left[1])
        rows = [point[1] for point in polyline]
        cols = [point[0] for point in polyline]
        mean_row = float(sum(rows) / len(rows))
        mean_col = float(sum(cols) / len(cols))
        rank = (length, -mean_row, -mean_col)
        if best is None or rank > best[0]:
            best = (rank, polyline, length, mean_row, mean_col)
    _rank, polyline, length, mean_row, mean_col = best
    phi = math.atan2(polyline[-1][1] - polyline[0][1], polyline[-1][0] - polyline[0][0])
    if phi <= -math.pi / 2:
        phi += math.pi
    elif phi > math.pi / 2:
        phi -= math.pi
    return len(polylines), length, phi, mean_row, mean_col


@operator(
    "輪廓",
    [P("Image", "輸入影像")],
    [P("Lines", "脊線輪廓")],
    [
        P("Sigma", "平滑標準差"),
        P("Low", "中心強度的低門檻"),
        P("High", "能當種子的高門檻"),
        P("LightDark", "light 亮線、dark 暗線或 both", "'light'", ["'light'", "'dark'", "'both'"]),
    ],
    [],
    "抽出亮線或暗線",
    "用可分離的高斯導數做 Hessian，沿法線把脊線收到次像素。0 朝右，垂直是 1.5708。"
    "Low 以上的中心才會相連，其中要有一點達到 High 才留下。交叉處分成幾段，不補角。"
    "平坦處和大方塊的邊緣不會當成細線。彩色先轉成亮度。Sigma 要大於 0，High 不能小於 Low。",
    "Vizion.lines_gauss (Image, Lines, 1.2, 1, 5, 'light')",
)
def lines_gauss(image, sigma, low, high, light_dark, *, ctx):
    picture = expect_image(image, "Image")
    low_value = scalar(low, "Low")
    high_value = scalar(high, "High")
    if high_value < low_value:
        raise ValueError("High 不能小於 Low")
    strength, normal_x, normal_y, shift, center = _ridge_fields(picture, scalar(sigma, "Sigma"), _polarity(light_dark, "LightDark"))
    allowed = center & (strength >= low_value)
    seeds = center & (strength >= high_value)
    if not np.any(seeds):
        return Result([XLD(picture.width, picture.height, [])], [])
    labels = cv2.connectedComponents(np.where(allowed, 255, 0).astype(np.uint8), connectivity=8)[1]
    keep_ids = np.unique(labels[seeds])
    keep_ids = keep_ids[keep_ids > 0]
    keep = np.isin(labels, keep_ids)
    pieces = _link_ridge(keep, shift, normal_x, normal_y)
    return Result([XLD(picture.width, picture.height, pieces, multi_color=len(pieces) > 1)], [])


_RIDGE_POLARITY = ["'dark'", "'light'", "'both'"]


@operator(
    "輪廓",
    [P("Image", "輸入影像")],
    [P("Response", "脊線強度，0 是沒有")],
    [
        P("Sigma", "平滑標準差", "1"),
        P("LightDark", "dark 暗線、light 亮線或 both", "'dark'", _RIDGE_POLARITY),
    ],
    [],
    "把細線強度畫成灰階",
    "只有脊線中心有值，旁邊和平坦處是 0。數字是 Hessian 曲率的絕對值，四捨五入到 0 到 255。"
    "用來決定 lines_gauss 的 Low 和 High：中心灰階就是強度。彩色先轉成亮度。",
    "Vizion.ridge_response (Image, Response, 1, 'dark')",
)
def ridge_response(image, sigma, light_dark, *, ctx):
    picture = expect_image(image, "Image")
    strength, _normal_x, _normal_y, _shift, center = _ridge_fields(picture, scalar(sigma, "Sigma"), _polarity(light_dark, "LightDark"))
    out = np.zeros(picture.gray().shape, np.uint8)
    if np.any(center):
        out[center] = np.clip(np.round(strength[center]), 0, 255).astype(np.uint8)
    return image_out(picture, out)


def _frangi_scale(gray: np.ndarray, sigma: float, dark: bool) -> np.ndarray:
    smooth, first, second = _gauss_deriv_kernels(sigma)
    scale = sigma * sigma
    dxx = _sep(gray, second, smooth) * scale
    dyy = _sep(gray, smooth, second) * scale
    dxy = _sep(gray, first, first) * scale
    trace = dxx + dyy
    det = dxx * dyy - dxy * dxy
    disc = np.sqrt(np.maximum(trace * trace * 0.25 - det, 0.0))
    high = trace * 0.5 + disc
    low = trace * 0.5 - disc
    swap = np.abs(high) < np.abs(low)
    first_eigen = np.where(swap, high, low)
    second_eigen = np.where(swap, low, high)
    ratio = np.divide(np.abs(first_eigen), np.abs(second_eigen), out=np.zeros_like(first_eigen), where=np.abs(second_eigen) > 1e-8)
    magnitude = np.sqrt(first_eigen * first_eigen + second_eigen * second_eigen)
    peak = float(np.max(magnitude))
    if peak < 1e-8:
        return np.zeros(gray.shape, np.float32)
    half = 0.5 * peak
    vessel = np.exp(-(ratio * ratio) / 0.5) * (1.0 - np.exp(-(magnitude * magnitude) / (2.0 * half * half)))
    sign = second_eigen > 0 if dark else second_eigen < 0
    return np.where(sign, vessel, 0.0).astype(np.float32)


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("Response", "細線加強後的灰階")],
    [
        P("Sigma", "最小尺度，像素", "1"),
        P("SigmaMax", "最大尺度。和 Sigma 相同就只做一層", "1"),
        P("LightDark", "dark 暗線或 light 亮線", "'dark'", ["'dark'", "'light'"]),
    ],
    [],
    "用 Frangi 把細線加強",
    "1998 年 Frangi 用 Hessian 的兩個特徵值分辨細線和方塊。細線接近 255，方塊內部和平坦處接近 0。"
    "Sigma 對一像素寬的線，更粗的線把 SigmaMax 拉大，中間再取一層，留下最亮的那層。彩色先轉成亮度。",
    "Vizion.frangi (Image, Response, 1.2, 1.2, 'dark')",
    "新算子。依 Frangi 1998 把暗線加強，方塊內部不會亮起來。不是 lines_gauss。",
)
def frangi(image, sigma, sigma_max, light_dark, *, ctx):
    picture = expect_image(image, "Image")
    start = scalar(sigma, "Sigma")
    stop = scalar(sigma_max, "SigmaMax")
    if start <= 0 or stop <= 0:
        raise ValueError("Sigma 和 SigmaMax 必須大於 0")
    if stop < start:
        raise ValueError("SigmaMax 不能小於 Sigma")
    kind = text(light_dark, "LightDark").lower()
    if kind not in ("dark", "light"):
        raise ValueError("LightDark 只能是 dark 或 light")
    gray = picture.gray().astype(np.float32)
    if abs(stop - start) < 1e-6:
        scales = [start]
    else:
        scales = [start, (start + stop) / 2.0, stop]
    response = np.maximum.reduce([_frangi_scale(gray, item, kind == "dark") for item in scales])
    return image_out(picture, np.clip(np.round(response * 255.0), 0, 255).astype(np.uint8))


@operator(
    "輪廓",
    [P("Image", "輸入影像")],
    [P("Region", "細線中心")],
    [
        P("Sigma", "平滑標準差", "1"),
        P("MinStrength", "中心至少要多強", "8"),
        P("LightDark", "dark 暗線、light 亮線或 both", "'dark'", _RIDGE_POLARITY),
    ],
    [],
    "細線中心變成區域",
    "一像素寬，每條不相連的線是一個物件。面積就是這條線上的點數。"
    "大方塊邊緣和只差幾個灰階的雜訊不會留下。MinStrength 不能是負的，0 表示只要有中心就算。",
    "Vizion.ridge_region (Image, Region, 1, 8, 'dark')",
)
def ridge_region(image, sigma, minimum, light_dark, *, ctx):
    picture = expect_image(image, "Image")
    floor = scalar(minimum, "MinStrength")
    if floor < 0:
        raise ValueError("MinStrength 不能是負的")
    strength, _normal_x, _normal_y, _shift, center = _ridge_fields(picture, scalar(sigma, "Sigma"), _polarity(light_dark, "LightDark"))
    mask = np.where(center & (strength >= floor), 255, 0).astype(np.uint8)
    count, labels = cv2.connectedComponents(mask, connectivity=8)
    objects = [np.where(labels == index, 255, 0).astype(np.uint8) for index in range(1, count)]
    return Result([Region(picture.width, picture.height, objects, multi_color=len(objects) > 1)], [])


@operator(
    "輪廓",
    [P("Image", "輸入影像")],
    [],
    [
        P("Sigma", "平滑標準差", "1"),
        P("MinStrength", "中心至少要多強", "8"),
        P("LightDark", "dark 暗線、light 亮線或 both", "'dark'", _RIDGE_POLARITY),
    ],
    [
        P("Count", "有幾條"),
        P("Length", "最長那條的長度"),
        P("Phi", "最長那條的方向，0 朝右"),
        P("Row", "最長那條的中心列"),
        P("Column", "最長那條的中心行"),
    ],
    "細線有幾條、朝哪",
    "Count 不算孤立的一個點。Length 是相鄰中心的距離和，不是面積。"
    "Phi 收在 -1.5708 到 1.5708，水平是 0，垂直是 1.5708。沒有線時五個數都是 0。"
    "同一條線從哪一頭起算都得到同一個方向。",
    "Vizion.ridge_stats (Image, 1, 8, 'dark', Count, Length, Phi, Row, Column)",
)
def ridge_stats(image, sigma, minimum, light_dark, *, ctx):
    picture = expect_image(image, "Image")
    floor = scalar(minimum, "MinStrength")
    if floor < 0:
        raise ValueError("MinStrength 不能是負的")
    strength, normal_x, normal_y, shift, center = _ridge_fields(picture, scalar(sigma, "Sigma"), _polarity(light_dark, "LightDark"))
    pieces = _link_ridge(center & (strength >= floor), shift, normal_x, normal_y)
    count, length, phi, row, column = _line_facts(pieces)
    return Result([], [count, length, phi, row, column])


def _edge_fields(picture: Image, sigma: float, sigma_name: str = "Alpha"):
    """Subpixel step edge: Gaussian gradient, parabolic peak along the normal."""
    if sigma <= 0:
        raise ValueError(f"{sigma_name} 必須大於 0")
    smooth, first, _second = _gauss_deriv_kernels(sigma)
    gray = picture.gray().astype(np.float32)
    dx = _sep(gray, first, smooth)
    dy = _sep(gray, smooth, first)
    magnitude = np.hypot(dx, dy)
    scale = np.maximum(magnitude, 1e-6)
    normal_x = dx / scale
    normal_y = dy / scale
    rows, cols = np.mgrid[0 : picture.height, 0 : picture.width]
    ahead = _sample_field(magnitude, rows + normal_y, cols + normal_x)
    behind = _sample_field(magnitude, rows - normal_y, cols - normal_x)
    denom = behind + ahead - 2.0 * magnitude
    shift = np.divide(
        0.5 * (behind - ahead),
        denom,
        out=np.full_like(magnitude, 9.0),
        where=np.abs(denom) > 1e-6,
    )
    center = (magnitude >= ahead) & (magnitude >= behind) & (magnitude > 0) & (shift > -0.5) & (shift <= 0.5)
    if picture.domain is not None:
        center &= picture.domain > 0
    return magnitude, normal_x, normal_y, shift, center


@operator(
    "輪廓",
    [P("Image", "輸入影像")],
    [P("Edges", "次像素邊緣")],
    [
        P("Filter", "canny。用高斯導數，不是整數 Canny", "'canny'", ["'canny'"]),
        P("Alpha", "平滑標準差"),
        P("Low", "梯度低門檻"),
        P("High", "能當種子的高門檻"),
    ],
    [],
    "萃取次像素邊緣",
    "高斯導數的梯度在法線上做拋物線內插，邊緣落在強度最高的次像素，而不是整數像素的外框。"
    "Low 以上才相連，其中要有一點達到 High。黑白階躍在 Sigma 1 時強度大約 60，門檻 20 和 40 留得住明顯的邊。"
    "差幾個灰階的雜訊不會留下。彩色先轉成亮度。Alpha 要大於 0，High 不能小於 Low。",
    "Vizion.edges_sub_pix (Image, Edges, 'canny', 1.0, 20, 40)",
)
def edges_sub_pix(image, kind, alpha, low, high, *, ctx):
    picture = expect_image(image, "Image")
    if text(kind, "Filter").lower() != "canny":
        raise ValueError("Filter 這版是 canny")
    low_value = scalar(low, "Low")
    high_value = scalar(high, "High")
    if high_value < low_value:
        raise ValueError("High 不能小於 Low")
    magnitude, normal_x, normal_y, shift, center = _edge_fields(picture, scalar(alpha, "Alpha"))
    allowed = center & (magnitude >= low_value)
    seeds = center & (magnitude >= high_value)
    if not np.any(seeds):
        return Result([XLD(picture.width, picture.height, [])], [])
    labels = cv2.connectedComponents(np.where(allowed, 255, 0).astype(np.uint8), connectivity=8)[1]
    keep_ids = np.unique(labels[seeds])
    keep_ids = keep_ids[keep_ids > 0]
    keep = np.isin(labels, keep_ids)
    pieces = _link_ridge(keep, shift, normal_x, normal_y)
    return Result([XLD(picture.width, picture.height, pieces, multi_color=len(pieces) > 1)], [])


_ZERNIKE_MASKS: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _zernike_masks(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cached = _ZERNIKE_MASKS.get(size)
    if cached is not None:
        return cached
    radius = size / 2.0
    samples = 5
    real11 = np.zeros((size, size), np.float64)
    imag11 = np.zeros((size, size), np.float64)
    moment20 = np.zeros((size, size), np.float64)
    sub_area = (1.0 / radius) ** 2 / (samples * samples)
    center = (size - 1) / 2.0
    for row in range(size):
        for col in range(size):
            total_x = 0.0
            total_y = 0.0
            total_r = 0.0
            for sy in range(samples):
                for sx in range(samples):
                    yy = row + (sy + 0.5) / samples - center
                    xx = col + (sx + 0.5) / samples - center
                    x = xx / radius
                    y = -yy / radius
                    if x * x + y * y > 1.0:
                        continue
                    total_x += x
                    total_y += y
                    total_r += 2.0 * (x * x + y * y) - 1.0
            real11[row, col] = total_x * sub_area
            imag11[row, col] = (-total_y) * sub_area
            moment20[row, col] = total_r * sub_area
    masks = (real11, imag11, moment20)
    _ZERNIKE_MASKS[size] = masks
    return masks


@operator(
    "輪廓",
    [P("Image", "灰階或彩色影像")],
    [P("Edges", "次像素邊緣")],
    [P("Size", "奇數視窗", "7"), P("MinAmp", "階躍至少要多少灰階", "30")],
    [P("Row", "邊緣列"), P("Column", "邊緣行")],
    "Zernike 次像素邊緣",
    "Ghosal 與 Mehrotra 1993：在單位圓上算 1 階和 2 階 Zernike 矩，得到邊緣離像素中心的距離和法線方向。視窗裡的階躍要大於 MinAmp，而且邊緣要落在中心像素附近。彩色先轉成亮度。",
    "Vizion.edges_zernike (Image, Edges, 7, 30, Row, Column)",
    "新算子。依 Ghosal 與 Mehrotra 1993 用 Zernike 矩把邊緣定到次像素。不是 measure_pos。",
)
def edges_zernike(image, size, min_amp, *, ctx):
    picture = expect_image(image, "Image")
    window = whole(size, "Size")
    amplitude = scalar(min_amp, "MinAmp")
    if window < 5 or window % 2 == 0:
        raise ValueError("Size 要是 5 以上的奇數")
    if amplitude <= 0:
        raise ValueError("MinAmp 要大於 0")
    gray = picture.gray().astype(np.float64)
    real11, imag11, moment20 = _zernike_masks(window)
    moment_real = cv2.filter2D(gray, cv2.CV_64F, real11, borderType=cv2.BORDER_REPLICATE)
    moment_imag = cv2.filter2D(gray, cv2.CV_64F, imag11, borderType=cv2.BORDER_REPLICATE)
    moment_20 = cv2.filter2D(gray, cv2.CV_64F, moment20, borderType=cv2.BORDER_REPLICATE)
    strength = np.hypot(moment_real, moment_imag)
    distance = np.zeros(strength.shape, np.float64)
    usable = strength > 1e-8
    distance[usable] = moment_20[usable] / strength[usable]
    limit = 1.0 / math.sqrt(2.0)
    inside = usable & (np.abs(distance) < limit)
    height_term = np.maximum(1.0 - distance * distance, 1e-8) ** 1.5
    step = np.zeros(strength.shape, np.float64)
    step[inside] = 3.0 * strength[inside] / (2.0 * height_term[inside])
    angle = np.arctan2(moment_imag, moment_real)
    margin = window // 2
    keep = np.zeros(step.shape, np.bool_)
    height, width = step.shape
    rows_i, cols_i = np.nonzero(inside)
    for row, col in zip(rows_i.tolist(), cols_i.tolist()):
        if row < margin or col < margin or row >= height - margin or col >= width - margin:
            continue
        if step[row, col] < amplitude:
            continue
        step_row = int(round(math.sin(angle[row, col])))
        step_col = int(round(math.cos(angle[row, col])))
        if step_row == 0 and step_col == 0:
            continue
        ahead_row = row + step_row
        ahead_col = col + step_col
        back_row = row - step_row
        back_col = col - step_col
        ahead = strength[ahead_row, ahead_col] if 0 <= ahead_row < height and 0 <= ahead_col < width else 0.0
        back = strength[back_row, back_col] if 0 <= back_row < height and 0 <= back_col < width else 0.0
        if strength[row, col] >= ahead and strength[row, col] > back:
            keep[row, col] = True
    labels = cv2.connectedComponents(np.where(keep, 255, 0).astype(np.uint8), connectivity=8)[1]
    radius = window / 2.0
    pieces = []
    out_rows = []
    out_cols = []
    for label in range(1, int(labels.max()) + 1):
        points_row, points_col = np.nonzero(labels == label)
        if points_row.size == 0:
            continue
        shift = distance[points_row, points_col] * radius
        phi = angle[points_row, points_col]
        sub_col = points_col.astype(np.float64) + shift * np.cos(phi)
        sub_row = points_row.astype(np.float64) + shift * np.sin(phi)
        tangent_col = float(np.mean(-np.sin(phi)))
        tangent_row = float(np.mean(np.cos(phi)))
        order = np.argsort(sub_col * tangent_col + sub_row * tangent_row)
        contour = []
        for index in order.tolist():
            contour.append([float(sub_col[index]), float(sub_row[index])])
            out_cols.append(float(sub_col[index]))
            out_rows.append(float(sub_row[index]))
        pieces.append(contour)
    edges = XLD(picture.width, picture.height, pieces, multi_color=len(pieces) > 1)
    return Result([edges], [out_rows, out_cols])


@operator(
    "輪廓",
    [P("Image", "輸入影像")],
    [P("Response", "邊緣強度，0 是沒有")],
    [P("Alpha", "平滑標準差", "1")],
    [],
    "把邊緣強度畫成灰階",
    "只有次像素邊緣所在的那個整數像素有值。數字是梯度大小，四捨五入到 0 到 255，用來決定 Low 和 High。"
    "彩色先轉成亮度。",
    "Vizion.edge_response (Image, Response, 1)",
)
def edge_response(image, alpha, *, ctx):
    picture = expect_image(image, "Image")
    magnitude, _normal_x, _normal_y, _shift, center = _edge_fields(picture, scalar(alpha, "Alpha"))
    out = np.zeros(picture.gray().shape, np.uint8)
    if np.any(center):
        out[center] = np.clip(np.round(magnitude[center]), 0, 255).astype(np.uint8)
    return image_out(picture, out)


@operator(
    "輪廓",
    [P("Image", "輸入影像")],
    [P("Region", "邊緣像素")],
    [
        P("Alpha", "平滑標準差", "1"),
        P("MinAmplitude", "梯度至少要多大", "20"),
    ],
    [],
    "邊緣變成區域",
    "區域畫在梯度峰值的整數像素上，次像素位置要看 edge_stats。"
    "不相連的邊各自一個物件。MinAmplitude 不能是負的。",
    "Vizion.edge_region (Image, Region, 1, 20)",
)
def edge_region(image, alpha, minimum, *, ctx):
    picture = expect_image(image, "Image")
    floor = scalar(minimum, "MinAmplitude")
    if floor < 0:
        raise ValueError("MinAmplitude 不能是負的")
    magnitude, _normal_x, _normal_y, _shift, center = _edge_fields(picture, scalar(alpha, "Alpha"))
    mask = np.where(center & (magnitude >= floor), 255, 0).astype(np.uint8)
    count, labels = cv2.connectedComponents(mask, connectivity=8)
    objects = [np.where(labels == index, 255, 0).astype(np.uint8) for index in range(1, count)]
    return Result([Region(picture.width, picture.height, objects, multi_color=len(objects) > 1)], [])


@operator(
    "輪廓",
    [P("Image", "輸入影像")],
    [],
    [
        P("Alpha", "平滑標準差", "1"),
        P("MinAmplitude", "梯度至少要多大", "20"),
    ],
    [
        P("Count", "有幾條"),
        P("Length", "最長那條的長度"),
        P("Phi", "最長那條的方向，0 朝右"),
        P("Row", "最長那條的次像素列"),
        P("Column", "最長那條的次像素行"),
    ],
    "邊緣在哪、朝哪",
    "Row 和 Column 是次像素，可以落在兩個整數像素中間。Phi 收在 -1.5708 到 1.5708，垂直是 1.5708。"
    "沒有邊時五個數都是 0。孤立的一個點不算一條。",
    "Vizion.edge_stats (Image, 1, 20, Count, Length, Phi, Row, Column)",
)
def edge_stats(image, alpha, minimum, *, ctx):
    picture = expect_image(image, "Image")
    floor = scalar(minimum, "MinAmplitude")
    if floor < 0:
        raise ValueError("MinAmplitude 不能是負的")
    magnitude, normal_x, normal_y, shift, center = _edge_fields(picture, scalar(alpha, "Alpha"))
    pieces = _link_ridge(center & (magnitude >= floor), shift, normal_x, normal_y)
    count, length, phi, row, column = _line_facts(pieces)
    return Result([], [count, length, phi, row, column])


@operator(
    "輪廓",
    [P("XLD", "輪廓")],
    [],
    [],
    [
        P("Row", "中心列"),
        P("Column", "中心行"),
        P("Phi", "長邊角度，弳度"),
        P("Length1", "半長"),
        P("Length2", "半寬"),
    ],
    "輪廓的最小旋轉外接矩形",
    "每一條輪廓一組。長邊角度收在 −90 到 90 度。點太少時會失敗。",
    "Vizion.smallest_rectangle2_xld (XLD, Row, Column, Phi, Length1, Length2)",
)
def smallest_rectangle2_xld(contours, *, ctx):
    source = expect_xld(contours, "XLD")
    if not source.contours:
        raise ValueError("沒有輪廓")
    rows, cols, phis, long_sides, short_sides = [], [], [], [], []
    for contour in source.contours:
        pts = _as_points(contour).astype(np.float32).reshape(-1, 1, 2)
        if len(pts) < 3:
            raise ValueError("旋轉矩形至少要 3 個點")
        (col, row), (width, height), angle = cv2.minAreaRect(pts)
        if width < height:
            width, height = height, width
            angle += 90.0
        while angle >= 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0
        rows.append(float(row))
        cols.append(float(col))
        phis.append(math.radians(float(angle)))
        long_sides.append(float(width) / 2.0)
        short_sides.append(float(height) / 2.0)
    return Result([], [rows, cols, phis, long_sides, short_sides])


def _axis_phi(direction_col: float, direction_row: float) -> float:
    phi = math.atan2(direction_row, direction_col)
    if phi > math.pi / 2:
        phi -= math.pi
    elif phi <= -math.pi / 2:
        phi += math.pi
    return phi


def _weighted_axis(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if float(weights.sum()) < 1e-8:
        raise ValueError("點太少，不能擬合直線")
    center = (weights[:, None] * points).sum(axis=0) / float(weights.sum())
    centered = (points - center) * np.sqrt(weights)[:, None]
    if float(np.max(np.abs(centered))) < 1e-9:
        raise ValueError("點都重疊，不能擬合直線")
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return center, vt[0]


def _segment(points: np.ndarray, center: np.ndarray, direction: np.ndarray, keep: np.ndarray) -> list[list[float]]:
    if int(np.count_nonzero(keep)) < 2:
        keep = np.ones(len(points), dtype=bool)
    offset = (points[keep] - center) @ direction
    start = center + direction * float(offset.min())
    end = center + direction * float(offset.max())
    return [[float(start[0]), float(start[1])], [float(end[0]), float(end[1])]]


def _gauss_line(points: np.ndarray, iterations: int) -> list[list[float]]:
    weights = np.ones(len(points), np.float64)
    passes = iterations if iterations > 0 else 4
    center, direction = _weighted_axis(points, weights)
    for _step in range(passes):
        distance = np.abs((points - center) @ np.array([-direction[1], direction[0]]))
        if float(distance.max()) < 1e-8:
            break
        deviation = float(np.median(np.abs(distance - np.median(distance))))
        sigma = 1.4826 * deviation if deviation > 0.05 else 1.0
        weights = np.exp(-0.5 * (distance / sigma) ** 2)
        center, direction = _weighted_axis(points, weights)
    return _segment(points, center, direction, weights > 0.05)


def _line_through(points: np.ndarray, mode: str, iterations: int) -> list[list[float]]:
    if len(points) < 2:
        raise ValueError("點太少，不能擬合直線")
    if mode == "gauss":
        return _gauss_line(points, iterations)
    current = points
    rounds = iterations if mode == "drop" else 0
    for step in range(rounds + 1):
        if len(current) < 2:
            raise ValueError("點太少，不能擬合直線")
        weights = np.ones(len(current), np.float64)
        center, direction = _weighted_axis(current, weights)
        centered = current - center
        distance = np.abs(centered[:, 0] * direction[1] - centered[:, 1] * direction[0])
        if mode == "drop" and step < rounds and len(current) > 4:
            limit = max(float(np.median(distance)) * 2.0, 0.5)
            kept = current[distance <= limit]
            if len(kept) >= 2 and len(kept) < len(current):
                current = kept
                continue
        return _segment(current, center, direction, np.ones(len(current), dtype=bool))
    raise ValueError("不能擬合直線")


@operator(
    "輪廓",
    [P("Contours", "輸入輪廓")],
    [P("RegressContours", "擬合成直線段")],
    [
        P("Mode", "no、drop 或 gauss", "'no'", ["'no'", "'drop'", "'gauss'"]),
        P("Iterations", "加權或剔除的次數，0 代表 gauss 做 4 次", "0"),
    ],
    [],
    "把輪廓擬合成直線段",
    "每條輪廓換成它在直線上的投影範圍。no 每個點一樣重。gauss 依離直線的距離做高斯加權，飛點幾乎沒有影響力，Iterations 為 0 時做 4 次。drop 會把離直線較遠的點拿掉。點都在直線上時，三種結果一樣。",
    "Vizion.regress_contours_xld (Contours, RegressContours, 'no', 0)",
)
def regress_contours_xld(contours, mode, iterations, *, ctx):
    source = expect_xld(contours, "Contours")
    if not source.contours:
        raise ValueError("沒有輪廓")
    kind = text(mode, "Mode").lower()
    if kind not in ("no", "drop", "gauss"):
        raise ValueError("Mode 只能是 no、drop 或 gauss")
    rounds = whole(iterations, "Iterations")
    if rounds < 0:
        raise ValueError("Iterations 不能是負的")
    pieces = [_line_through(_as_points(contour), kind, rounds) for contour in source.contours]
    return Result([XLD(source.width, source.height, pieces, multi_color=len(pieces) > 1)], [])


@operator(
    "輪廓",
    [P("Contours", "輸入輪廓")],
    [],
    [],
    [P("Phi", "長軸角度，弳度，−π/2 到 π/2")],
    "輪廓的方向",
    "用點的主軸，0 是向右，正值在畫面上往下轉。不是填滿後的區域方向。一條輪廓一個角度。",
    "Vizion.orientation_xld (Contours, Phi)",
)
def orientation_xld(contours, *, ctx):
    source = expect_xld(contours, "Contours")
    if not source.contours:
        raise ValueError("沒有輪廓")
    angles = []
    for contour in source.contours:
        points = _as_points(contour)
        center = points.mean(axis=0)
        centered = points - center
        if float(np.max(np.abs(centered))) < 1e-9:
            raise ValueError("點都重疊，沒有方向")
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        angles.append(_axis_phi(float(vt[0, 0]), float(vt[0, 1])))
    return Result([], [angles])


@operator(
    "輪廓",
    [P("XLD", "輪廓")],
    [],
    [],
    [P("Ra", "長軸"), P("Rb", "短軸"), P("Phi", "長軸角度，弳度")],
    "用輪廓點估計橢圓",
    "Ra 和 Rb 是 2 倍的標準差，不是區域版 elliptic_axis 的半徑。點如果只在邊界上，半徑會和填滿的橢圓不同。0 是向右。每條輪廓一組。",
    "Vizion.elliptic_axis_xld (XLD, Ra, Rb, Phi)",
)
def elliptic_axis_xld(contours, *, ctx):
    source = expect_xld(contours, "XLD")
    if not source.contours:
        raise ValueError("沒有輪廓")
    long_axes, short_axes, angles = [], [], []
    for contour in source.contours:
        points = _as_points(contour)
        center = points.mean(axis=0)
        centered = points - center
        if len(points) < 2 or float(np.max(np.abs(centered))) < 1e-9:
            raise ValueError("點太少，不能估計橢圓")
        cov = centered.T @ centered / len(points)
        values, vectors = np.linalg.eigh(cov)
        order = int(np.argmax(values))
        major = math.sqrt(max(float(values[order]), 0.0))
        minor = math.sqrt(max(float(values[1 - order]), 0.0))
        direction = vectors[:, order]
        long_axes.append(2.0 * major)
        short_axes.append(2.0 * minor)
        angles.append(_axis_phi(float(direction[0]), float(direction[1])))
    return Result([], [long_axes, short_axes, angles])


def _fillet(previous, current, following, radius: float):
    backward = previous - current
    forward = following - current
    length_back = float(np.hypot(backward[0], backward[1]))
    length_forward = float(np.hypot(forward[0], forward[1]))
    if length_back < 1e-6 or length_forward < 1e-6:
        return None
    unit_back = backward / length_back
    unit_forward = forward / length_forward
    cosine = float(np.clip(np.dot(unit_back, unit_forward), -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-3 or abs(math.pi - angle) < 1e-3:
        return None
    half = angle / 2.0
    inset = min(radius / math.tan(half), length_back * 0.5, length_forward * 0.5)
    if inset < 1e-6:
        return None
    used = inset * math.tan(half)
    start = current + unit_back * inset
    stop = current + unit_forward * inset
    bisector = unit_back + unit_forward
    bisector = bisector / np.hypot(bisector[0], bisector[1])
    center = current + bisector * (used / math.sin(half))
    angle_start = math.atan2(float(start[1] - center[1]), float(start[0] - center[0]))
    angle_stop = math.atan2(float(stop[1] - center[1]), float(stop[0] - center[0]))
    sweep = (angle_stop - angle_start + math.pi) % (2.0 * math.pi) - math.pi
    steps = max(2, int(math.ceil(abs(sweep) * used)))
    arc = []
    for step in range(steps + 1):
        theta = angle_start + sweep * step / steps
        arc.append(np.array([center[0] + used * math.cos(theta), center[1] + used * math.sin(theta)]))
    return arc


@operator(
    "輪廓",
    [],
    [P("Contour", "圓角後的輪廓")],
    [
        P("Row", "折線的列"),
        P("Col", "折線的行"),
        P("Radius", "圓角半徑"),
        P("Sampling", "圓弧上相鄰點的大約間距", "1"),
    ],
    [],
    "把折線的轉角修成圓弧",
    "首尾同一個點時當成封閉。半徑 0 就是原來的折線。圓弧太大會縮到線段的一半，避免把相鄰轉角疊在一起。",
    "Vizion.gen_contour_polygon_rounded_xld (Contour, [0, 0, 20], [0, 20, 20], 5, 1)",
)
def gen_contour_polygon_rounded_xld(row, col, radius, sampling, *, ctx):
    rows = [float(item) for item in as_list(row)]
    cols = [float(item) for item in as_list(col)]
    if len(rows) != len(cols) or len(rows) < 2:
        raise ValueError("Row 和 Col 的長度要相同，而且至少兩個點")
    curve = scalar(radius, "Radius")
    step = scalar(sampling, "Sampling")
    if curve < 0 or step <= 0:
        raise ValueError("半徑不能是負的，Sampling 要大於 0")
    points = [np.array([col_value, row_value], np.float64) for row_value, col_value in zip(rows, cols)]
    closed = float(np.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1])) < 1e-6
    core = points[:-1] if closed and len(points) > 2 else points
    if curve == 0 or len(core) < 3:
        rounded = points
    else:
        count = len(core)
        rounded = []

        def emit(point):
            if not rounded or float(np.hypot(point[0] - rounded[-1][0], point[1] - rounded[-1][1])) > 1e-6:
                rounded.append(point)

        if not closed:
            emit(core[0])
        corner_count = count if closed else count - 2
        for index in range(corner_count):
            vertex = index if closed else index + 1
            arc = _fillet(core[(vertex - 1) % count], core[vertex], core[(vertex + 1) % count], curve)
            if arc is None:
                emit(core[vertex])
            else:
                gap = max(1, int(round(step)))
                for point in arc[::gap]:
                    emit(point)
                emit(arc[-1])
        if not closed:
            emit(core[-1])
        elif rounded:
            emit(rounded[0])
    height, width = ctx.canvas_size()
    stored = [[float(point[0]), float(point[1])] for point in rounded]
    for col_value, row_value in stored:
        width = max(width, int(math.ceil(col_value)) + 1)
        height = max(height, int(math.ceil(row_value)) + 1)
    return Result([XLD(width, height, [stored])], [])


def _trim_end(points: list, amount: float) -> list:
    if amount <= 0 or len(points) < 2:
        return points
    remain = amount
    index = 0
    while index < len(points) - 1 and remain > 1e-9:
        step = math.hypot(points[index + 1][0] - points[index][0], points[index + 1][1] - points[index][1])
        if step > remain + 1e-9:
            scale = remain / step if step else 0.0
            landed = [
                points[index][0] + scale * (points[index + 1][0] - points[index][0]),
                points[index][1] + scale * (points[index + 1][1] - points[index][1]),
            ]
            return [landed] + points[index + 1 :]
        remain -= step
        index += 1
    return [points[-1]]


@operator(
    "輪廓",
    [P("Contours", "輸入輪廓")],
    [P("ClippedContours", "去掉兩端之後")],
    [
        P("Mode", "num_points 去掉點數，length 去掉長度", "'num_points'", ["'num_points'", "'length'"]),
        P("Length", "兩端各去掉多少"),
    ],
    [],
    "去掉輪廓的兩端",
    "num_points 是點數，length 是沿著線的像素長度。剩下少於 2 個點就丟掉這條輪廓。封閉輪廓也是從序列兩端切，不會改成從中間切。",
    "Vizion.clip_end_points_contours_xld (Contours, ClippedContours, 'num_points', 1)",
)
def clip_end_points_contours_xld(contours, mode, length, *, ctx):
    source = expect_xld(contours, "Contours")
    kind = text(mode, "Mode").lower()
    amount = scalar(length, "Length")
    if amount < 0:
        raise ValueError("Length 不能是負的")
    if kind not in ("num_points", "length"):
        raise ValueError("Mode 只能是 num_points 或 length")
    pieces = []
    for contour in source.contours:
        points = [[float(point[0]), float(point[1])] for point in contour]
        if kind == "num_points":
            count = whole(length, "Length")
            if len(points) - 2 * count >= 2:
                pieces.append(points[count : len(points) - count])
            continue
        trimmed = _trim_end(points, amount)
        trimmed = _trim_end(list(reversed(trimmed)), amount)
        trimmed.reverse()
        if len(trimmed) >= 2:
            pieces.append(trimmed)
    return Result([XLD(source.width, source.height, pieces, multi_color=len(pieces) > 1)], [])


def _contour_cross(first, second):
    row1, col1 = first[0][1], first[0][0]
    row2, col2 = first[1][1], first[1][0]
    row3, col3 = second[0][1], second[0][0]
    row4, col4 = second[1][1], second[1][0]
    drow1, dcol1 = row2 - row1, col2 - col1
    drow2, dcol2 = row4 - row3, col4 - col3
    den = drow1 * dcol2 - dcol1 * drow2
    if abs(den) < 1e-9:
        return None
    factor1 = ((row3 - row1) * dcol2 - (col3 - col1) * drow2) / den
    factor2 = ((row3 - row1) * dcol1 - (col3 - col1) * drow1) / den
    if -1e-6 <= factor1 <= 1 + 1e-6 and -1e-6 <= factor2 <= 1 + 1e-6:
        return (row1 + factor1 * drow1, col1 + factor1 * dcol1)
    return None


@operator(
    "輪廓",
    [P("Contour1", "第一組輪廓"), P("Contour2", "第二組輪廓")],
    [],
    [],
    [P("Row", "交點列"), P("Column", "交點行")],
    "兩組輪廓的交點",
    "每一段和另一組的每一段互相比。沒有交點時得到空 tuple。很長的輪廓會比較慢。靠得很近的交點會合成一個。",
    "Vizion.intersection_contours_xld (Contour1, Contour2, Row, Column)",
)
def intersection_contours_xld(contour1, contour2, *, ctx):
    first = expect_xld(contour1, "Contour1")
    second = expect_xld(contour2, "Contour2")
    hits = []
    for left in first.contours:
        for right in second.contours:
            for index in range(len(left) - 1):
                for other in range(len(right) - 1):
                    point = _contour_cross(left[index : index + 2], right[other : other + 2])
                    if point is None:
                        continue
                    if any(math.hypot(point[0] - old[0], point[1] - old[1]) < 0.5 for old in hits):
                        continue
                    hits.append(point)
    return Result([], [[item[0] for item in hits], [item[1] for item in hits]])


def _turn(previous: float, following: float) -> float:
    delta = following - previous
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta


@operator(
    "輪廓",
    [P("Contour", "輪廓")],
    [],
    [P("Mode", "abs 是方向，rel 是轉角", "'abs'", ["'abs'", "'rel'"])],
    [P("Angle", "每個點一個角度，弳度")],
    "輪廓每個點的角度",
    "abs 的 0 是向右，正值在畫面上往下轉。rel 是這一點轉了多少，直線接近 0，端點是 0。多條輪廓會依序接在同一個 tuple。",
    "Vizion.get_contour_angle_xld (Contour, 'abs', Angle)",
)
def get_contour_angle_xld(contour, mode, *, ctx):
    source = expect_xld(contour, "Contour")
    kind = text(mode, "Mode").lower()
    if kind not in ("abs", "rel"):
        raise ValueError("Mode 只能是 abs 或 rel")
    angles = []
    for item in source.contours:
        points = _as_points(item)
        if len(points) < 2:
            continue
        directions = []
        for index in range(len(points) - 1):
            dcol = float(points[index + 1, 0] - points[index, 0])
            drow = float(points[index + 1, 1] - points[index, 1])
            directions.append(math.atan2(drow, dcol))
        if kind == "abs":
            angles.extend(directions)
            angles.append(directions[-1])
            continue
        angles.append(0.0)
        for index in range(len(directions) - 1):
            angles.append(_turn(directions[index], directions[index + 1]))
        angles.append(0.0)
    return Result([], [angles])


@operator(
    "輪廓",
    [P("XLD", "輪廓")],
    [],
    [],
    [P("IsClosed", "1 表示首尾接在一起")],
    "輪廓是否封閉",
    "首尾距離小於 1.5 像素就算封閉。每條輪廓一個 0 或 1。",
    "Vizion.test_closed_xld (XLD, IsClosed)",
)
def test_closed_xld(contours, *, ctx):
    source = expect_xld(contours, "XLD")
    flags = []
    for contour in source.contours:
        points = _as_points(contour)
        if len(points) < 3:
            flags.append(0)
            continue
        gap = float(np.hypot(points[0, 0] - points[-1, 0], points[0, 1] - points[-1, 1]))
        flags.append(1 if gap <= 1.5 else 0)
    return Result([], [flags])


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [],
    [P("Features", "輪廓特徵，可以是 tuple", choices=XLD_FEATURE_CHOICES)],
    [P("Value", "特徵值")],
    "讀取輪廓特徵",
    "名稱和 select_contours_xld 相同，例如 contlength、area、circularity。單一特徵時依輪廓排列。多個特徵時先排完第一個特徵的全部輪廓。",
    "Vizion.xld_features (Contours, 'contlength', Value)",
)
def xld_features(contours, features, *, ctx):
    source = expect_xld(contours, "Contours")
    names = [_xld_feature_name(item) for item in as_list(features)]
    if not names:
        raise ValueError("Features 不能是空的")
    values = []
    for name in names:
        for contour in source.contours:
            values.append(float(xld_measures(contour).get(name, 0.0)))
    return Result([], [values])


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [P("DestContours", "包含這些點的輪廓")],
    [P("Row", "列，可以是 tuple"), P("Column", "行，可以是 tuple")],
    [],
    "留下包含指定點的輪廓",
    "把點序當成多邊形。點在內部或邊上就留下，碰到其中一個點即可。少於 3 個點的輪廓不會被選到。",
    "Vizion.select_xld_point (Contours, DestContours, 20, 20)",
)
def select_xld_point(contours, row, column, *, ctx):
    source = expect_xld(contours, "Contours")
    rows = [float(item) for item in as_list(row)]
    cols = [float(item) for item in as_list(column)]
    if not rows or len(rows) != len(cols):
        raise ValueError("Row 和 Column 的長度要相同，而且不能是空的")
    kept = []
    for contour in source.contours:
        points = _as_points(contour)
        if len(points) < 3:
            continue
        polygon = points.astype(np.float32).reshape(-1, 1, 2)
        if any(cv2.pointPolygonTest(polygon, (float(col), float(item_row)), False) >= 0 for item_row, col in zip(rows, cols)):
            kept.append(points.tolist())
    return Result([XLD(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [P("ContoursPolar", "展開後的輪廓")],
    [
        P("Row", "極座標中心列"),
        P("Column", "極座標中心行"),
        P("AngleStart", "起始角，弳度"),
        P("AngleEnd", "結束角，弳度"),
        P("RadiusStart", "內半徑"),
        P("RadiusEnd", "外半徑"),
        P("Width", "角度方向的寬度"),
        P("Height", "半徑方向的高度"),
    ],
    [],
    "把輪廓轉成極座標",
    "角度 0 朝右，正角度順時針。點會線性對到寬和高，可以落在影像外面。小於起始角的角度會加一整圈，用來對上 0 到 360 度的展開圖。",
    "Vizion.polar_trans_contour_xld (Contour, ContourPolar, 30, 40, 0, rad(360), 0, 20, 40, 20)",
)
def polar_trans_contour_xld(contours, row, column, angle_start, angle_end, radius_start, radius_end, width, height, *, ctx):
    source = expect_xld(contours, "Contours")
    wide = whole(width, "Width")
    high = whole(height, "Height")
    if wide < 1 or high < 1:
        raise ValueError("寬高至少要 1")
    center_row = scalar(row, "Row")
    center_col = scalar(column, "Column")
    start = scalar(angle_start, "AngleStart")
    span = scalar(angle_end, "AngleEnd") - start
    inner = scalar(radius_start, "RadiusStart")
    radial = scalar(radius_end, "RadiusEnd") - inner
    if abs(span) < 1e-9 or abs(radial) < 1e-9:
        raise ValueError("角度範圍和半徑範圍都不能是 0")
    kept = []
    for contour in source.contours:
        points = _as_points(contour)
        delta_row = points[:, 1] - center_row
        delta_col = points[:, 0] - center_col
        angle = np.arctan2(delta_row, delta_col)
        if span > 0:
            angle = np.where(angle < start, angle + math.tau, angle)
        else:
            angle = np.where(angle > start, angle - math.tau, angle)
        polar_col = (angle - start) / span * wide
        polar_row = (np.hypot(delta_row, delta_col) - inner) / radial * high
        kept.append(np.stack([polar_col, polar_row], axis=1).tolist())
    return Result([XLD(wide, high, kept, multi_color=len(kept) > 1)], [])


def _contour_intersections(points: np.ndarray, row1: float, column1: float, row2: float, column2: float, segment: bool):
    if len(points) < 2:
        return [], []
    start = points[:-1]
    stop = points[1:]
    start_row, start_col = start[:, 1], start[:, 0]
    stop_row, stop_col = stop[:, 1], stop[:, 0]
    line_row = row2 - row1
    line_col = column2 - column1
    edge_row = stop_row - start_row
    edge_col = stop_col - start_col
    den = line_row * edge_col - line_col * edge_row
    valid = np.abs(den) > 1e-9
    origin_row = start_row - row1
    origin_col = start_col - column1
    along = np.zeros(len(den), np.float64)
    across = np.zeros(len(den), np.float64)
    along[valid] = (origin_row[valid] * edge_col[valid] - origin_col[valid] * edge_row[valid]) / den[valid]
    across[valid] = (origin_row[valid] * line_col - origin_col[valid] * line_row) / den[valid]
    hit = valid & (across >= -1e-6) & (across <= 1.0 + 1e-6)
    if segment:
        hit &= (along >= -1e-6) & (along <= 1.0 + 1e-6)
    if not np.any(hit):
        return [], []
    found = np.stack([row1 + along[hit] * line_row, column1 + along[hit] * line_col], axis=1)
    _, order = np.unique(np.round(found, 6), axis=0, return_index=True)
    found = found[np.sort(order)]
    return found[:, 0].tolist(), found[:, 1].tolist()


def _all_intersections(contours, row1, column1, row2, column2, segment: bool):
    source = expect_xld(contours, "Contour")
    r1, c1 = scalar(row1, "Row1"), scalar(column1, "Column1")
    r2, c2 = scalar(row2, "Row2"), scalar(column2, "Column2")
    if math.hypot(r2 - r1, c2 - c1) < 1e-9:
        raise ValueError("直線的兩個點重疊")
    rows, cols = [], []
    for contour in source.contours:
        hit_rows, hit_cols = _contour_intersections(_as_points(contour), r1, c1, r2, c2, segment)
        rows.extend(hit_rows)
        cols.extend(hit_cols)
    return Result([], [rows, cols])


@operator(
    "量測",
    [P("Contour", "輪廓")],
    [],
    [
        P("Row1", "直線起點的列"), P("Column1", "直線起點的行"),
        P("Row2", "直線終點的列"), P("Column2", "直線終點的行"),
    ],
    [P("Row", "交點列"), P("Column", "交點行")],
    "無限長直線和輪廓的交點",
    "用輪廓上相鄰的點組成折線。沒有交點時得到空 tuple。平行或重疊的這版不回傳。不會自動把輪廓封閉。",
    "Vizion.intersection_line_contour_xld (Contour, 0, 20, 40, 20, Row, Column)",
)
def intersection_line_contour_xld(contour, row1, column1, row2, column2, *, ctx):
    return _all_intersections(contour, row1, column1, row2, column2, False)


@operator(
    "量測",
    [P("Contour", "輪廓")],
    [],
    [
        P("Row1", "線段起點的列"), P("Column1", "線段起點的行"),
        P("Row2", "線段終點的列"), P("Column2", "線段終點的行"),
    ],
    [P("Row", "交點列"), P("Column", "交點行")],
    "線段和輪廓的交點",
    "只算線段碰到折線的地方。延長出去的交點不算。沒有交點時得到空 tuple。",
    "Vizion.intersection_segment_contour_xld (Contour, 0, 20, 15, 20, Row, Column)",
)
def intersection_segment_contour_xld(contour, row1, column1, row2, column2, *, ctx):
    return _all_intersections(contour, row1, column1, row2, column2, True)


_XLD_SHAPES = ["'rectangle1'", "'rectangle2'", "'convex'", "'outer_circle'", "'ellipse'"]


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [P("ContoursTrans", "換成規則外形")],
    [P("Type", "rectangle1、rectangle2、convex、outer_circle 或 ellipse", "'rectangle1'", _XLD_SHAPES)],
    [],
    "把輪廓換成規則外形",
    "rectangle1 是正外接矩形。rectangle2 和 ellipse 用 OpenCV 的外接矩形與橢圓。outer_circle 是最小外包圓。點不夠擬合橢圓時改成圓。",
    "Vizion.shape_trans_xld (Contours, ContoursTrans, 'rectangle1')",
)
def shape_trans_xld(contours, kind, *, ctx):
    source = expect_xld(contours, "Contours")
    mode = text(kind, "Type").lower()
    if mode not in ("rectangle1", "rectangle2", "convex", "outer_circle", "ellipse"):
        raise ValueError("Type 只能是 rectangle1、rectangle2、convex、outer_circle 或 ellipse")
    kept = []
    for contour in source.contours:
        points = _as_points(contour)
        if len(points) < 2:
            continue
        shaped = _xld_shape(points, mode)
        if len(shaped) >= 2:
            kept.append(shaped)
    return Result([XLD(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


def _xld_shape(points: np.ndarray, mode: str) -> list:
    if mode == "rectangle1":
        row1, col1 = float(points[:, 1].min()), float(points[:, 0].min())
        row2, col2 = float(points[:, 1].max()), float(points[:, 0].max())
        return [[col1, row1], [col2, row1], [col2, row2], [col1, row2], [col1, row1]]
    cloud = points.astype(np.float32)
    if mode == "convex":
        hull = cv2.convexHull(cloud).reshape(-1, 2)
        if len(hull) == 0:
            return []
        if not np.allclose(hull[0], hull[-1]):
            hull = np.vstack([hull, hull[0]])
        return hull.tolist()
    if mode == "rectangle2":
        box = cv2.boxPoints(cv2.minAreaRect(cloud))
        return np.vstack([box, box[0]]).tolist()
    if mode == "ellipse" and len(points) >= 5:
        try:
            (center_col, center_row), axes, angle = cv2.fitEllipse(cloud)
            return _ellipse_poly(center_row, center_col, axes[1] / 2.0, axes[0] / 2.0, math.radians(angle))
        except cv2.error:
            pass
    (center_col, center_row), radius = cv2.minEnclosingCircle(cloud)
    return _ellipse_poly(center_row, center_col, radius, radius, 0.0)


def _ellipse_poly(row: float, col: float, radius_row: float, radius_col: float, phi: float) -> list:
    angles = np.linspace(0.0, math.tau, 64, endpoint=False)
    cosine, sine = math.cos(phi), math.sin(phi)
    flat_col = radius_col * np.cos(angles)
    flat_row = radius_row * np.sin(angles)
    cols = col + cosine * flat_col - sine * flat_row
    rows = row + sine * flat_col + cosine * flat_row
    poly = np.stack([cols, rows], axis=1)
    return np.vstack([poly, poly[0]]).tolist()


def _ellipse_point_distance(cols, rows, center_row, center_col, phi, radius1, radius2) -> np.ndarray:
    cosine, sine = math.cos(phi), math.sin(phi)
    delta_col = np.asarray(cols, np.float64) - center_col
    delta_row = np.asarray(rows, np.float64) - center_row
    local_a = cosine * delta_col + sine * delta_row
    local_b = -sine * delta_col + cosine * delta_row
    unit_a = local_a / radius1
    unit_b = local_b / radius2
    norm = np.hypot(unit_a, unit_b)
    distance = np.empty(norm.shape, np.float64)
    origin = norm <= 1e-12
    distance[origin] = min(radius1, radius2)
    usable = ~origin
    scale = np.zeros(norm.shape, np.float64)
    scale[usable] = 1.0 / norm[usable]
    on_a = unit_a * scale * radius1
    on_b = unit_b * scale * radius2
    distance[usable] = np.hypot(local_a[usable] - on_a[usable], local_b[usable] - on_b[usable])
    return distance


@operator(
    "量測",
    [P("Contours", "輪廓")],
    [],
    [
        P("Row", "橢圓中心列"),
        P("Column", "橢圓中心行"),
        P("Phi", "第一軸角度，弳度。0 沿著行"),
        P("Radius1", "第一軸半徑"),
        P("Radius2", "第二軸半徑"),
    ],
    [P("MaxDistance", "最大距離")],
    "輪廓到橢圓的最大距離",
    "每條輪廓一個距離。用徑向投影，正圓是準的。很扁的橢圓和真正的最短距離可能有一點差。Phi 為 0 時 Radius1 沿著行。",
    "Vizion.dist_ellipse_contour_xld (Contour, 30, 40, 0, 10, 10, MaxDistance)",
)
def dist_ellipse_contour_xld(contours, row, column, phi, radius1, radius2, *, ctx):
    source = expect_xld(contours, "Contours")
    center_row, center_col = scalar(row, "Row"), scalar(column, "Column")
    angle = scalar(phi, "Phi")
    major, minor = scalar(radius1, "Radius1"), scalar(radius2, "Radius2")
    if major <= 0 or minor <= 0:
        raise ValueError("半徑必須大於 0")
    distances = []
    for contour in source.contours:
        points = _as_points(contour)
        if len(points) == 0:
            continue
        gap = _ellipse_point_distance(points[:, 0], points[:, 1], center_row, center_col, angle, major, minor)
        distances.append(float(gap.max()))
    if not distances:
        raise ValueError("輪廓是空的")
    return Result([], [distances])


def _segment_crosses(row1, column1, row2, column2, er1, ec1, er2, ec2) -> bool:
    line_row = row2 - row1
    line_col = column2 - column1
    edge_row = er2 - er1
    edge_col = ec2 - ec1
    den = line_row * edge_col - line_col * edge_row
    valid = np.abs(den) > 1e-9
    origin_row = er1 - row1
    origin_col = ec1 - column1
    along = np.zeros(np.shape(den), np.float64)
    across = np.zeros(np.shape(den), np.float64)
    along[valid] = (origin_row[valid] * edge_col[valid] - origin_col[valid] * edge_row[valid]) / den[valid]
    across[valid] = (origin_row[valid] * line_col - origin_col[valid] * line_row) / den[valid]
    near = 1e-6
    on_line = valid & (along >= -near) & (along <= 1.0 + near) & (across >= -near) & (across <= 1.0 + near)
    shared_end = ((np.abs(along) <= near) | (np.abs(along - 1.0) <= near)) & (
        (np.abs(across) <= near) | (np.abs(across - 1.0) <= near)
    )
    return bool(np.any(on_line & ~shared_end))


def _crosses_self(points: np.ndarray) -> int:
    if len(points) < 4:
        return 0
    closed = float(np.hypot(points[0, 0] - points[-1, 0], points[0, 1] - points[-1, 1])) <= 1.5
    start = points[:-1]
    stop = points[1:]
    count = len(start)
    for index in range(count):
        last = count - 1 if closed and index == 0 else count
        follow = np.arange(index + 2, last)
        if len(follow) == 0:
            continue
        if _segment_crosses(
            float(start[index, 1]),
            float(start[index, 0]),
            float(stop[index, 1]),
            float(stop[index, 0]),
            stop[follow, 1],
            stop[follow, 0],
            start[follow, 1],
            start[follow, 0],
        ):
            return 1
    return 0


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [],
    [],
    [P("IsSelfIntersecting", "1 表示折線自己交叉")],
    "輪廓是否自交",
    "相鄰邊共用的端點不算交叉。封閉輪廓的首尾兩邊也不算。每條輪廓一個 0 或 1。",
    "Vizion.test_self_intersection_xld (Contours, IsSelfIntersecting)",
)
def test_self_intersection_xld(contours, *, ctx):
    source = expect_xld(contours, "Contours")
    return Result([], [[_crosses_self(_as_points(contour)) for contour in source.contours]])


def _curvature_points(points: np.ndarray, limit: float):
    closed = len(points) >= 3 and float(np.hypot(points[0, 0] - points[-1, 0], points[0, 1] - points[-1, 1])) <= 1.5
    pts = points[:-1] if closed and np.allclose(points[0], points[-1]) else points
    if len(pts) < 3:
        return np.zeros((0, 2))
    if closed:
        previous = np.roll(pts, 1, axis=0)
        current = pts
        following = np.roll(pts, -1, axis=0)
    else:
        previous, current, following = pts[:-2], pts[1:-1], pts[2:]
    first = current - previous
    second = following - current
    cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    dot = first[:, 0] * second[:, 0] + first[:, 1] * second[:, 1]
    turn = np.abs(np.arctan2(cross, dot))
    if len(turn) == 1:
        left = np.zeros(1)
        right = np.zeros(1)
    elif closed:
        left = np.roll(turn, 1)
        right = np.roll(turn, -1)
    else:
        left = np.concatenate([[0.0], turn[:-1]])
        right = np.concatenate([turn[1:], [0.0]])
    keep = (turn + 1e-9 >= left) & (turn + 1e-9 >= right) & (turn >= limit)
    return current[keep]


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [],
    [P("Threshold", "最小轉角，弳度", "0.5")],
    [P("Row", "轉角的列"), P("Column", "轉角的行")],
    "留下轉角大的輪廓點",
    "用相鄰三點的轉角。比 Threshold 大，而且不比左右轉角小的點會留下。封閉輪廓的首尾接點只算一次。這是折線頂點，不是曲線擬合。",
    "Vizion.critical_points_xld (Contours, 0.5, Row, Column)",
)
def critical_points_xld(contours, threshold, *, ctx):
    source = expect_xld(contours, "Contours")
    limit = scalar(threshold, "Threshold")
    if limit < 0:
        raise ValueError("Threshold 不能是負的")
    rows, cols = [], []
    for contour in source.contours:
        found = _curvature_points(_as_points(contour), limit)
        rows.extend(float(item) for item in found[:, 1])
        cols.extend(float(item) for item in found[:, 0])
    return Result([], [rows, cols])


def _rectangle2_distance(cols, rows, center_row, center_col, phi, length1, length2) -> np.ndarray:
    cosine, sine = math.cos(phi), math.sin(phi)
    delta_col = np.asarray(cols, np.float64) - center_col
    delta_row = np.asarray(rows, np.float64) - center_row
    along = np.abs(delta_col * cosine + delta_row * sine)
    side = np.abs(-delta_col * sine + delta_row * cosine)
    outside_along = np.maximum(along - length1, 0.0)
    outside_side = np.maximum(side - length2, 0.0)
    outside = np.hypot(outside_along, outside_side)
    inside = np.minimum(length1 - along, length2 - side)
    return np.where((outside_along == 0.0) & (outside_side == 0.0), inside, outside)


@operator(
    "量測",
    [P("Contours", "輪廓")],
    [],
    [
        P("Row", "矩形中心列"),
        P("Column", "矩形中心行"),
        P("Phi", "長邊角度，弳度。0 沿著行"),
        P("Length1", "角度方向的半長"),
        P("Length2", "垂直方向的半長"),
    ],
    [P("MaxDistance", "最大距離")],
    "輪廓到旋轉矩形邊界的最大距離",
    "每條輪廓一個距離。在矩形裡面是到最近一邊的距離，在外面是到矩形的距離。Phi 為 0 時 Length1 沿著行。",
    "Vizion.dist_rectangle2_contour_xld (Contour, 30, 40, 0, 10, 5, MaxDistance)",
)
def dist_rectangle2_contour_xld(contours, row, column, phi, length1, length2, *, ctx):
    source = expect_xld(contours, "Contours")
    center_row, center_col = scalar(row, "Row"), scalar(column, "Column")
    angle = scalar(phi, "Phi")
    half_along, half_side = scalar(length1, "Length1"), scalar(length2, "Length2")
    if half_along <= 0 or half_side <= 0:
        raise ValueError("半長必須大於 0")
    distances = []
    for contour in source.contours:
        points = _as_points(contour)
        if len(points) == 0:
            continue
        gap = _rectangle2_distance(points[:, 0], points[:, 1], center_row, center_col, angle, half_along, half_side)
        distances.append(float(gap.max()))
    if not distances:
        raise ValueError("輪廓是空的")
    return Result([], [distances])


def _split_edges(contour, epsilon: float) -> list:
    if len(contour) < 2:
        return []
    points = np.asarray(contour, np.float32).reshape(-1, 1, 2)
    closed = float(np.hypot(points[0, 0, 0] - points[-1, 0, 0], points[0, 0, 1] - points[-1, 0, 1])) <= 1.5
    approx = cv2.approxPolyDP(points, epsilon, closed)
    simplified = [[float(point[0][0]), float(point[0][1])] for point in approx]
    if len(simplified) >= 2 and math.hypot(simplified[0][0] - simplified[-1][0], simplified[0][1] - simplified[-1][1]) <= 1.5:
        simplified = simplified[:-1]
    if len(simplified) < 2:
        return []
    count = len(simplified)
    last = count if closed and count >= 3 else count - 1
    return [[simplified[index], simplified[(index + 1) % count]] for index in range(last)]


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [P("SplitContours", "拆開的邊")],
    [P("Weight", "允許偏離的像素距離", "0.5")],
    [],
    "把輪廓拆成一段一段",
    "先用 Douglas-Peucker 留下轉折，再把每條邊拆成獨立輪廓。封閉輪廓會接回起點。Weight 是頂點可以離開原輪廓的最大距離。",
    "Vizion.split_contours_xld (Contours, SplitContours, 0.5)",
)
def split_contours_xld(contours, weight, *, ctx):
    source = expect_xld(contours, "Contours")
    epsilon = scalar(weight, "Weight")
    if epsilon < 0:
        raise ValueError("Weight 不能是負的")
    pieces = []
    for contour in source.contours:
        pieces.extend(_split_edges(contour, epsilon))
    return Result([XLD(source.width, source.height, pieces, multi_color=len(pieces) > 1)], [])


@operator(
    "輪廓",
    [P("ContoursPolar", "極座標輪廓")],
    [P("ContoursXY", "還原後的輪廓")],
    [
        P("Row", "極座標中心列"),
        P("Column", "極座標中心行"),
        P("AngleStart", "起始角，弳度"),
        P("AngleEnd", "結束角，弳度"),
        P("RadiusStart", "內半徑"),
        P("RadiusEnd", "外半徑"),
        P("Width", "角度方向的取樣數"),
        P("Height", "半徑方向的取樣數"),
    ],
    [],
    "把極座標輪廓還原",
    "參數和 polar_trans_contour_xld 相同。寬和高是極座標圖的取樣數。角度 0 朝右，正角度順時針。",
    "Vizion.polar_trans_contour_xld_inv (ContoursPolar, ContoursXY, 30, 40, 0, rad(360), 0, 20, 40, 20)",
)
def polar_trans_contour_xld_inv(contours, row, column, angle_start, angle_end, radius_start, radius_end, width, height, *, ctx):
    source = expect_xld(contours, "ContoursPolar")
    wide = whole(width, "Width")
    high = whole(height, "Height")
    if wide < 1 or high < 1:
        raise ValueError("寬高至少要 1")
    center_row = scalar(row, "Row")
    center_col = scalar(column, "Column")
    start = scalar(angle_start, "AngleStart")
    span = scalar(angle_end, "AngleEnd") - start
    inner = scalar(radius_start, "RadiusStart")
    radial = scalar(radius_end, "RadiusEnd") - inner
    if abs(span) < 1e-9 or abs(radial) < 1e-9:
        raise ValueError("角度範圍和半徑範圍都不能是 0")
    kept = []
    for contour in source.contours:
        points = _as_points(contour)
        if len(points) == 0:
            continue
        angle = start + points[:, 0] / wide * span
        radius = inner + points[:, 1] / high * radial
        cols = center_col + radius * np.cos(angle)
        rows = center_row + radius * np.sin(angle)
        kept.append(np.stack([cols, rows], axis=1).tolist())
    return Result([_fit_canvas(ctx, kept)], [])


def _circle_hits(points: np.ndarray, center_row: float, center_col: float, radius: float):
    if len(points) < 2:
        return [], []
    start = points[:-1]
    stop = points[1:]
    delta_col = stop[:, 0] - start[:, 0]
    delta_row = stop[:, 1] - start[:, 1]
    offset_col = start[:, 0] - center_col
    offset_row = start[:, 1] - center_row
    quad_a = delta_row * delta_row + delta_col * delta_col
    quad_b = 2.0 * (offset_row * delta_row + offset_col * delta_col)
    quad_c = offset_row * offset_row + offset_col * offset_col - radius * radius
    valid = quad_a > 1e-12
    disc = quad_b * quad_b - 4.0 * quad_a * quad_c
    valid = valid & (disc >= -1e-8)
    root = np.sqrt(np.maximum(disc, 0.0))
    found = []
    for sign in (-1.0, 1.0):
        along = np.full(quad_a.shape, np.nan)
        along[valid] = (-quad_b[valid] + sign * root[valid]) / (2.0 * quad_a[valid])
        keep = np.isfinite(along) & (along >= -1e-6) & (along <= 1.0 + 1e-6)
        if np.any(keep):
            rows = start[keep, 1] + along[keep] * delta_row[keep]
            cols = start[keep, 0] + along[keep] * delta_col[keep]
            found.append(np.stack([rows, cols], axis=1))
    if not found:
        return [], []
    stacked = np.vstack(found)
    _, order = np.unique(np.round(stacked, 6), axis=0, return_index=True)
    stacked = stacked[np.sort(order)]
    return stacked[:, 0].tolist(), stacked[:, 1].tolist()


@operator(
    "量測",
    [P("Contour", "輪廓")],
    [],
    [P("Row", "圓心列"), P("Column", "圓心行"), P("Radius", "半徑")],
    [P("RowIntersect", "交點列"), P("ColumnIntersect", "交點行")],
    "輪廓和圓的交點",
    "不把輪廓自動封閉。沒有交點時得到空 tuple。相切只有一個點。",
    "Vizion.intersection_circle_contour_xld (Contour, 30, 40, 10, Row, Column)",
)
def intersection_circle_contour_xld(contour, row, column, radius, *, ctx):
    source = expect_xld(contour, "Contour")
    center_row, center_col = scalar(row, "Row"), scalar(column, "Column")
    rad = scalar(radius, "Radius")
    if rad < 0:
        raise ValueError("半徑不能是負的")
    rows, cols = [], []
    for item in source.contours:
        hit_rows, hit_cols = _circle_hits(_as_points(item), center_row, center_col, rad)
        rows.extend(hit_rows)
        cols.extend(hit_cols)
    return Result([], [rows, cols])


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [],
    [],
    [P("Length", "每條輪廓的點數")],
    "輪廓的點數",
    "每一條輪廓一個數。封閉輪廓若終點和起點重複，那個點也算進去。",
    "Vizion.contour_point_num_xld (Contours, Length)",
)
def contour_point_num_xld(contours, *, ctx):
    source = expect_xld(contours, "Contours")
    return Result([], [[len(_as_points(contour)) for contour in source.contours]])


def _xld_feature_list(contours, feature: str):
    source = expect_xld(contours, "Contours")
    return Result([], [[float(xld_measures(contour).get(feature, 0.0)) for contour in source.contours]])


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [],
    [],
    [P("Circularity", "圓度")],
    "輪廓圓度",
    "和 xld_features 的 circularity 相同。點序仍是先行、再列。每一條輪廓一個數。",
    "Vizion.circularity_xld (Contours, Circularity)",
)
def circularity_xld(contours, *, ctx):
    return _xld_feature_list(contours, "circularity")


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [],
    [],
    [P("Compactness", "緊實度")],
    "輪廓緊實度",
    "和 xld_features 的 compactness 相同。每一條輪廓一個數。",
    "Vizion.compactness_xld (Contours, Compactness)",
)
def compactness_xld(contours, *, ctx):
    return _xld_feature_list(contours, "compactness")


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [],
    [],
    [P("Convexity", "凸度")],
    "輪廓凸度",
    "和 xld_features 的 convexity 相同。沒有凹角時是 1。",
    "Vizion.convexity_xld (Contours, Convexity)",
)
def convexity_xld(contours, *, ctx):
    return _xld_feature_list(contours, "convexity")


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [],
    [],
    [P("Eccentricity", "偏心率")],
    "輪廓偏心率",
    "和 xld_features 的 anisometry 相同，長軸除以短軸。",
    "Vizion.eccentricity_xld (Contours, Eccentricity)",
)
def eccentricity_xld(contours, *, ctx):
    return _xld_feature_list(contours, "anisometry")


@operator(
    "輪廓",
    [P("Image", "灰階或彩色影像"), P("Contours", "輪廓")],
    [],
    [],
    [P("Grayval", "輪廓點上的灰階")],
    "讀輪廓點的灰階",
    "座標四捨五入後和 get_grayval 讀同一個像素。點的順序就是輪廓方向。超出影像會失敗。彩色圖依點排紅、綠、藍。",
    "Vizion.get_grayval_contour_xld (Image, Contours, Grayval)",
)
def get_grayval_contour_xld(image, contours, *, ctx):
    picture = expect_image(image, "Image")
    source = expect_xld(contours, "Contours")
    spots = []
    for contour in source.contours:
        for point in contour:
            row = int(round(float(point[1])))
            col = int(round(float(point[0])))
            spots.append((row, col))
    if not spots:
        return Result([], [[]])
    if any(row < 0 or row >= picture.height or col < 0 or col >= picture.width for row, col in spots):
        raise ValueError("座標超出影像")
    if picture.data.ndim == 2:
        return Result([], [[int(picture.data[row, col]) for row, col in spots]])
    if picture.data.shape[2] != 3:
        raise ValueError("get_grayval_contour_xld 只接受灰階或三通道彩色")
    values = []
    for row, col in spots:
        blue, green, red = (int(item) for item in picture.data[row, col, :3])
        values.extend([red, green, blue])
    return Result([], [values])


def _contour_points(contours) -> list[list[tuple[float, float]]]:
    source = expect_xld(contours, "Contours")
    groups = []
    for contour in source.contours:
        groups.append([(float(point[0]), float(point[1])) for point in contour])
    if not groups:
        raise ValueError("輪廓是空的")
    return groups


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [],
    [],
    [
        P("Row1", "較遠的一個點的列"),
        P("Column1", "較遠的一個點的行"),
        P("Row2", "另一個點的列"),
        P("Column2", "另一個點的行"),
        P("Diameter", "這兩點的距離"),
    ],
    "輪廓上最遠的兩點",
    "每一條輪廓各找距離最大的兩個頂點。距離是這兩點的直線長度。",
    "Vizion.diameter_xld (Contours, Row1, Column1, Row2, Column2, Diameter)",
    "新算子。點 (0,0) 和 (4,3) 的距離是 5。",
)
def diameter_xld(contours, *, ctx):
    rows1, cols1, rows2, cols2, diameters = [], [], [], [], []
    for points in _contour_points(contours):
        if len(points) < 2:
            raise ValueError("輪廓至少要兩個點")
        best = -1.0
        pair = (points[0], points[1])
        for index, first in enumerate(points):
            for second in points[index + 1 :]:
                distance = math.hypot(first[0] - second[0], first[1] - second[1])
                if distance > best:
                    best = distance
                    pair = (first, second)
        rows1.append(pair[0][1])
        cols1.append(pair[0][0])
        rows2.append(pair[1][1])
        cols2.append(pair[1][0])
        diameters.append(best)
    return Result([], [rows1, cols1, rows2, cols2, diameters])


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [],
    [],
    [P("Height", "列方向的跨距"), P("Width", "行方向的跨距"), P("Ratio", "高除以寬")],
    "輪廓外框的高寬比",
    "高是最大列減最小列，寬是最大行減最小行。比是高除以寬。寬是 0 時會失敗。",
    "Vizion.height_width_ratio_xld (Contours, Height, Width, Ratio)",
    "新算子。列跨 4、行跨 3 時，比是 4/3。",
)
def height_width_ratio_xld(contours, *, ctx):
    heights, widths, ratios = [], [], []
    for points in _contour_points(contours):
        rows = [point[1] for point in points]
        cols = [point[0] for point in points]
        height = max(rows) - min(rows)
        width = max(cols) - min(cols)
        if width <= 1e-12:
            raise ValueError("輪廓的寬是 0，不能算高寬比")
        heights.append(height)
        widths.append(width)
        ratios.append(height / width)
    return Result([], [heights, widths, ratios])


def _unique_vertices(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) >= 2 and abs(points[0][0] - points[-1][0]) < 1e-9 and abs(points[0][1] - points[-1][1]) < 1e-9:
        return points[:-1]
    return points


def _polygon_area_center(points: list[tuple[float, float]]):
    cleaned = _unique_vertices(points)
    if len(cleaned) < 3:
        if not cleaned:
            return 0.0, 0.0, 0.0
        return 0.0, sum(point[1] for point in cleaned) / len(cleaned), sum(point[0] for point in cleaned) / len(cleaned)
    signed = 0.0
    col_moment = 0.0
    row_moment = 0.0
    for index, (col, row) in enumerate(cleaned):
        next_col, next_row = cleaned[(index + 1) % len(cleaned)]
        cross = col * next_row - next_col * row
        signed += cross
        col_moment += (col + next_col) * cross
        row_moment += (row + next_row) * cross
    signed *= 0.5
    if abs(signed) < 1e-12:
        return 0.0, sum(point[1] for point in cleaned) / len(cleaned), sum(point[0] for point in cleaned) / len(cleaned)
    return abs(signed), row_moment / (6.0 * signed), col_moment / (6.0 * signed)


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [],
    [],
    [P("M20", "行座標平方和"), P("M02", "列座標平方和"), P("M11", "列乘行的和")],
    "輪廓頂點的二階矩",
    "只用頂點，不把邊補點。起點若在最後重複一次，不會算兩次。每一條輪廓各一組。",
    "Vizion.moments_xld (Contours, M20, M02, M11)",
    "新算子。正方形 (0,0)、(2,0)、(2,2)、(0,2) 的 M20 是 8，M02 是 8，M11 是 4。",
)
def moments_xld(contours, *, ctx):
    m20, m02, m11 = [], [], []
    for points in _contour_points(contours):
        vertices = _unique_vertices(points)
        m20.append(sum(point[0] * point[0] for point in vertices))
        m02.append(sum(point[1] * point[1] for point in vertices))
        m11.append(sum(point[0] * point[1] for point in vertices))
    return Result([], [m20, m02, m11])


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [],
    [],
    [P("Area", "多邊形面積"), P("Row", "重心列"), P("Column", "重心行")],
    "把輪廓頂點當成多邊形",
    "面積用鞋帶公式。重心是這塊多邊形的重心，不是頂點的平均，除非面積是 0。每一條輪廓各一組。",
    "Vizion.area_center_points_xld (Contours, Area, Row, Column)",
    "新算子。邊長 2 的正方形面積是 4，重心在列 1、行 1。",
)
def area_center_points_xld(contours, *, ctx):
    areas, rows, cols = [], [], []
    for points in _contour_points(contours):
        area, row, col = _polygon_area_center(points)
        areas.append(area)
        rows.append(row)
        cols.append(col)
    return Result([], [areas, rows, cols])


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [],
    [],
    [P("Rectangularity", "面積除以正外框")],
    "輪廓有多像正放的矩形",
    "多邊形面積除以軸對齊外框的面積。正好填滿外框時是 1。外框有一邊是 0 時結果是 0。",
    "Vizion.rectangularity_xld (Contours, Rectangularity)",
    "新算子。邊長 2 的正方形是 1。",
)
def rectangularity_xld(contours, *, ctx):
    values = []
    for points in _contour_points(contours):
        area, _row, _col = _polygon_area_center(points)
        rows = [point[1] for point in points]
        cols = [point[0] for point in points]
        span = (max(rows) - min(rows)) * (max(cols) - min(cols))
        values.append(0.0 if span <= 1e-12 else area / span)
    return Result([], [values])


def _clip_segment(start, end, col1, row1, col2, row2):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    t0, t1 = 0.0, 1.0
    for direction, slack in ((-dx, start[0] - col1), (dx, col2 - start[0]), (-dy, start[1] - row1), (dy, row2 - start[1])):
        if abs(direction) < 1e-12:
            if slack < 0:
                return None
            continue
        cut = slack / direction
        if direction < 0:
            if cut > t1:
                return None
            if cut > t0:
                t0 = cut
        else:
            if cut < t0:
                return None
            if cut < t1:
                t1 = cut
    return (start[0] + t0 * dx, start[1] + t0 * dy), (start[0] + t1 * dx, start[1] + t1 * dy)


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [P("Cropped", "裁進矩形的輪廓")],
    [P("Row1", "上"), P("Column1", "左"), P("Row2", "下"), P("Column2", "右")],
    [],
    "把輪廓裁進一個矩形",
    "矩形外的線段會切掉。切到邊界的地方補上交點。整段都在外面就丟掉。",
    "Vizion.crop_contours_xld (Contours, Cropped, -1, 2, 1, 8)",
    "新算子。從行 0 畫到行 10 的水平線，裁進行 2 到 8 之後，兩端是 2 和 8。",
)
def crop_contours_xld(contours, row1, column1, row2, column2, *, ctx):
    source = expect_xld(contours, "Contours")
    top = scalar(row1, "Row1")
    left = scalar(column1, "Column1")
    bottom = scalar(row2, "Row2")
    right = scalar(column2, "Column2")
    if bottom < top or right < left:
        raise ValueError("矩形座標不正確")
    pieces = []
    for contour in source.contours:
        points = [(float(point[0]), float(point[1])) for point in contour]
        current = []
        for index in range(len(points) - 1):
            clipped = _clip_segment(points[index], points[index + 1], left, top, right, bottom)
            if clipped is None:
                if len(current) >= 2:
                    pieces.append([[point[0], point[1]] for point in current])
                current = []
                continue
            if not current or abs(current[-1][0] - clipped[0][0]) > 1e-6 or abs(current[-1][1] - clipped[0][1]) > 1e-6:
                current.append(clipped[0])
            current.append(clipped[1])
        if len(current) >= 2:
            pieces.append([[point[0], point[1]] for point in current])
    return Result([XLD(source.width, source.height, pieces, multi_color=len(pieces) > 1)], [])


@operator(
    "輪廓",
    [P("Contours", "封閉輪廓")],
    [],
    [P("Row", "列"), P("Column", "行")],
    [P("IsInside", "在裡面是 1")],
    "點在不在輪廓裡",
    "輪廓會自動閉合。點在邊界上也算裡面。每一個點得到 0 或 1。",
    "Vizion.test_xld_point (Contours, 2, 2, IsInside)",
    "新算子。正方形裡的 (2,2) 是 1，外面的 (6,6) 是 0。",
)
def test_xld_point(contours, row, column, *, ctx):
    rows = [float(item) for item in as_list(row)]
    cols = [float(item) for item in as_list(column)]
    if len(rows) != len(cols) or not rows:
        raise ValueError("Row 和 Column 的長度要相同，而且不能是空的")
    polygons = []
    for points in _contour_points(contours):
        vertices = _unique_vertices(points)
        if len(vertices) >= 3:
            polygons.append(np.array(vertices, np.float32))
    flags = []
    for item_row, item_col in zip(rows, cols):
        inside = False
        for polygon in polygons:
            if cv2.pointPolygonTest(polygon, (item_col, item_row), False) >= 0:
                inside = True
                break
        flags.append(1 if inside else 0)
    if len(flags) == 1 and not isinstance(row, (list, tuple)):
        return Result([], [flags[0]])
    return Result([], [flags])


@operator(
    "輪廓",
    [P("Polygons", "多邊形輪廓")],
    [P("PolygonsAffineTrans", "轉換後的多邊形")],
    [P("HomMat2D", "仿射矩陣")],
    [],
    "用仿射矩陣轉換多邊形",
    "和輪廓同一套矩陣：x 是列、y 是行。點的存放順序仍是先行、再列。",
    "Vizion.affine_trans_polygon_xld (Polygons, PolygonsAffineTrans, HomMat2D)",
    "新算子。平移列 1、行 2 之後，原來的 (0,0) 變成列 1、行 2。",
)
def affine_trans_polygon_xld(polygons, hom_mat2d, *, ctx):
    return affine_trans_contour_xld(polygons, hom_mat2d, ctx=ctx)


@operator(
    "輪廓",
    [P("Polygons", "多邊形輪廓")],
    [],
    [],
    [
        P("Row1", "起點列"),
        P("Col1", "起點行"),
        P("Row2", "終點列"),
        P("Col2", "終點行"),
        P("Length", "這一段的長度"),
        P("Phi", "這一段的角度，弳度"),
    ],
    "把多邊形拆成一段一段的線",
    "相鄰兩個頂點是一條線。角度 0 朝右，正角度在畫面上順時針。每一條輪廓的每一段各一組。",
    "Vizion.get_lines_xld (Polygons, Row1, Col1, Row2, Col2, Length, Phi)",
    "新算子。從行 0 畫到行 10 的水平線，長度是 10，角度是 0。",
)
def get_lines_xld(polygons, *, ctx):
    rows1, cols1, rows2, cols2, lengths, angles = [], [], [], [], [], []
    for points in _contour_points(polygons):
        if len(points) < 2:
            raise ValueError("輪廓至少要兩個點")
        for start, end in zip(points, points[1:]):
            dcol = end[0] - start[0]
            drow = end[1] - start[1]
            rows1.append(start[1])
            cols1.append(start[0])
            rows2.append(end[1])
            cols2.append(end[0])
            lengths.append(math.hypot(dcol, drow))
            angles.append(math.atan2(drow, dcol))
    return Result([], [rows1, cols1, rows2, cols2, lengths, angles])


@operator(
    "輪廓",
    [],
    [],
    [
        P("Angle", "要算的角度，弳度。多個角度請直接寫 tuple"),
        P("Row", "圓心列"),
        P("Column", "圓心行"),
        P("Phi", "長軸角度，弳度"),
        P("Radius1", "長軸半徑"),
        P("Radius2", "短軸半徑"),
    ],
    [P("RowPoint", "點的列"), P("ColPoint", "點的行")],
    "算出橢圓上的點",
    "角度 0 在長軸上。Phi 為 0 時長軸朝右，正角度在畫面上順時針。",
    "Vizion.get_points_ellipse ([0, 1.5708], 10, 20, 0, 4, 2, RowPoint, ColPoint)",
    "新算子。圓心 (10,20)、長軸 4、短軸 2、角度 0 時，角度 0 的點是列 10 行 24，90 度是列 12 行 20。",
)
def get_points_ellipse(angle, row, column, phi, radius1, radius2, *, ctx):
    angles = [float(item) for item in as_list(angle)]
    if not angles:
        raise ValueError("至少要一個角度")
    center_row = scalar(row, "Row")
    center_col = scalar(column, "Column")
    major = scalar(radius1, "Radius1")
    minor = scalar(radius2, "Radius2")
    if major <= 0 or minor <= 0:
        raise ValueError("半徑必須大於 0")
    cosine = math.cos(scalar(phi, "Phi"))
    sine = math.sin(scalar(phi, "Phi"))
    rows, cols = [], []
    for item in angles:
        ex = major * math.cos(item)
        ey = minor * math.sin(item)
        cols.append(center_col + ex * cosine - ey * sine)
        rows.append(center_row + ex * sine + ey * cosine)
    return Result([], [rows, cols])


_SHAPE_OP = ["'and'", "'or'"]


@operator(
    "輪廓",
    [P("Contours", "輸入輪廓")],
    [P("Selected", "符合條件的輪廓")],
    [
        P("Features", "特徵。多個特徵請直接寫 tuple", choices=XLD_FEATURE_CHOICES),
        P("Operation", "and 或 or", "'and'", _SHAPE_OP),
        P("Min", "下限，含"),
        P("Max", "上限，含"),
    ],
    [],
    "用形狀特徵篩選輪廓",
    "特徵名稱和特徵視窗的輪廓清單相同。and 要每一個特徵都在範圍裡，or 有一個符合就留下。Min 和 Max 的個數要和特徵一樣，或只給一個數套用到全部。",
    "Vizion.select_shape_xld (Contours, Selected, 'contlength', 'and', 5, 100)",
    "新算子。長度 10 和長度 2 的兩條線，範圍 5 到 100 只留下 1 條。",
)
def select_shape_xld(contours, features, operation, min_value, max_value, *, ctx):
    source = expect_xld(contours, "Contours")
    names = [_xld_feature_name(item) for item in as_list(features)]
    if not names:
        raise ValueError("至少要一個特徵")
    mode = text(operation, "Operation").strip().strip("'\"").lower()
    if mode not in ("and", "or"):
        raise ValueError("Operation 只能是 and 或 or")
    lows = [float(item) for item in as_list(min_value)]
    highs = [float(item) for item in as_list(max_value)]
    if len(lows) == 1 and len(names) > 1:
        lows = lows * len(names)
    if len(highs) == 1 and len(names) > 1:
        highs = highs * len(names)
    if len(lows) != len(names) or len(highs) != len(names):
        raise ValueError("Min 和 Max 的個數要和特徵一樣")
    kept = []
    for points in _contour_points(source):
        measures = xld_measures([[point[0], point[1]] for point in points])
        hits = [lows[index] <= measures.get(names[index], 0.0) <= highs[index] for index in range(len(names))]
        if (mode == "and" and all(hits)) or (mode == "or" and any(hits)):
            kept.append([[point[0], point[1]] for point in points])
    return Result([XLD(source.width, source.height, kept, multi_color=len(kept) > 1)], [])


@operator(
    "輪廓",
    [P("Region", "區域"), P("Contours", "輪廓")],
    [P("Intersection", "落在區域裡的輪廓")],
    [],
    [],
    "輪廓和區域相交",
    "線段會切到區域外框。切完的中點若還在區域裡就留下。整段都在外面就丟掉。",
    "Vizion.intersection_region_contour_xld (Region, Contours, Intersection)",
    "新算子。水平線從行 0 到行 10，區域只蓋住第 0 列的行 2 到 8，剩下的兩端是 2 和 8。",
)
def intersection_region_contour_xld(region, contours, *, ctx):
    area = expect_region(region, "Region")
    source = expect_xld(contours, "Contours")
    mask = area.union_mask()
    ys, xs = np.where(mask > 0)
    pieces = []
    if len(xs):
        left, top, right, bottom = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
        for contour in source.contours:
            points = [(float(point[0]), float(point[1])) for point in contour]
            current = []
            for index in range(len(points) - 1):
                clipped = _clip_segment(points[index], points[index + 1], left, top, right, bottom)
                if clipped is None:
                    if len(current) >= 2:
                        pieces.append([[point[0], point[1]] for point in current])
                    current = []
                    continue
                mid_col = (clipped[0][0] + clipped[1][0]) / 2
                mid_row = (clipped[0][1] + clipped[1][1]) / 2
                row_i = int(round(mid_row))
                col_i = int(round(mid_col))
                inside = 0 <= row_i < mask.shape[0] and 0 <= col_i < mask.shape[1] and mask[row_i, col_i] > 0
                if not inside:
                    if len(current) >= 2:
                        pieces.append([[point[0], point[1]] for point in current])
                    current = []
                    continue
                if not current or abs(current[-1][0] - clipped[0][0]) > 1e-6 or abs(current[-1][1] - clipped[0][1]) > 1e-6:
                    current.append(clipped[0])
                current.append(clipped[1])
            if len(current) >= 2:
                pieces.append([[point[0], point[1]] for point in current])
    width = max(area.width, source.width)
    height = max(area.height, source.height)
    return Result([XLD(width, height, pieces, multi_color=len(pieces) > 1)], [])


def _segment_distance(point, start, end) -> float:
    dcol = end[0] - start[0]
    drow = end[1] - start[1]
    length2 = dcol * dcol + drow * drow
    if length2 <= 1e-18:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    scale = ((point[0] - start[0]) * dcol + (point[1] - start[1]) * drow) / length2
    scale = min(1.0, max(0.0, scale))
    return math.hypot(point[0] - (start[0] + scale * dcol), point[1] - (start[1] + scale * drow))


_DISTANCE_MODE = ["'point_to_segment'", "'point_to_point'"]


@operator(
    "輪廓",
    [P("ContourFrom", "要量的輪廓"), P("ContourTo", "量到這條輪廓")],
    [],
    [P("Mode", "point_to_segment 或 point_to_point", "'point_to_segment'", _DISTANCE_MODE)],
    [P("Distance", "每個點的距離")],
    "每個點到另一條輪廓的距離",
    "結果和第一條輪廓的頂點一一對應。point_to_segment 量到線段，point_to_point 只量頂點。",
    "Vizion.distance_contours_xld (ContourFrom, ContourTo, 'point_to_segment', Distance)",
    "新算子。兩條平行水平線相隔 3 列時，每個點的距離都是 3。",
)
def distance_contours_xld(contour_from, contour_to, mode, *, ctx):
    source = expect_xld(contour_from, "ContourFrom")
    target = expect_xld(contour_to, "ContourTo")
    kind = text(mode, "Mode").strip().strip("'\"").lower()
    if kind not in ("point_to_segment", "point_to_point"):
        raise ValueError("Mode 只能是 point_to_segment 或 point_to_point")
    if not source.contours or not target.contours:
        raise ValueError("輪廓是空的")
    targets = [(float(point[0]), float(point[1])) for contour in target.contours for point in contour]
    distances = []
    for contour in source.contours:
        for point in contour:
            item = (float(point[0]), float(point[1]))
            if kind == "point_to_point" or len(targets) == 1:
                distances.append(min(math.hypot(item[0] - other[0], item[1] - other[1]) for other in targets))
                continue
            best = min(_segment_distance(item, start, end) for start, end in zip(targets, targets[1:]))
            distances.append(best)
    return Result([], [distances])


def _cloud_moments(points: list[tuple[float, float]]) -> dict:
    cleaned = _unique_vertices(points)
    if len(cleaned) < 2:
        return {"m00": 0.0, "mu20": 0.0, "mu02": 0.0, "mu11": 0.0}
    cols = np.array([point[0] for point in cleaned], np.float64)
    rows = np.array([point[1] for point in cleaned], np.float64)
    cols -= cols.mean()
    rows -= rows.mean()
    return {
        "m00": float(len(cleaned)),
        "mu20": float((cols * cols).sum()),
        "mu02": float((rows * rows).sum()),
        "mu11": float((rows * cols).sum()),
    }


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [],
    [],
    [P("Phi", "長軸角度，弳度")],
    "把頂點當成點雲的方向",
    "每個頂點權重相同，不看線段長短。0 沿著行往右，正角度在畫面上順時針。水平的點角度是 0。",
    "Vizion.orientation_points_xld (Contours, Phi)",
    "新算子。四個角 (0,0)、(0,6)、(2,6)、(2,0) 的角度是 0。",
)
def orientation_points_xld(contours, *, ctx):
    values = []
    for points in _contour_points(contours):
        _ra, _rb, phi = equivalent_ellipse(_cloud_moments(points))
        values.append(phi)
    return Result([], [values])


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [],
    [],
    [P("Ra", "長軸半徑"), P("Rb", "短軸半徑"), P("Phi", "長軸角度，弳度")],
    "把頂點當成點雲的等效橢圓",
    "每個頂點權重相同。0 沿著行往右，正角度在畫面上順時針。",
    "Vizion.elliptic_axis_points_xld (Contours, Ra, Rb, Phi)",
    "新算子。四個角 (0,0)、(0,6)、(2,6)、(2,0) 的長軸是 6，短軸是 2，角度是 0。",
)
def elliptic_axis_points_xld(contours, *, ctx):
    ra, rb, phi = [], [], []
    for points in _contour_points(contours):
        major, minor, angle = equivalent_ellipse(_cloud_moments(points))
        ra.append(major)
        rb.append(minor)
        phi.append(angle)
    return Result([], [ra, rb, phi])


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [],
    [],
    [P("Anisometry", "長軸除以短軸")],
    "把頂點當成點雲的扁度",
    "扁度是等效橢圓的長軸除以短軸。圓形接近 1。短軸是 0 時結果是 0。",
    "Vizion.eccentricity_points_xld (Contours, Anisometry)",
    "新算子。四個角 (0,0)、(0,6)、(2,6)、(2,0) 的扁度是 3。",
)
def eccentricity_points_xld(contours, *, ctx):
    values = []
    for points in _contour_points(contours):
        major, minor, _phi = equivalent_ellipse(_cloud_moments(points))
        values.append(0.0 if minor <= 1e-12 else major / minor)
    return Result([], [values])


@operator(
    "輪廓",
    [P("Polygon", "多邊形")],
    [],
    [],
    [P("Row", "頂點列"), P("Col", "頂點行"), P("Length", "每一邊的長度"), P("Phi", "每一邊的角度，弳度")],
    "讀出多邊形的頂點和每一邊",
    "頂點照順序。每一邊的角度 0 朝右，正角度在畫面上順時針。最後一點若和起點相同，不會多算一邊。",
    "Vizion.get_polygon_xld (Polygon, Row, Col, Length, Phi)",
    "新算子。從行 0 畫到行 10 的水平線，邊長是 10，角度是 0。",
)
def get_polygon_xld(polygon, *, ctx):
    rows, cols, lengths, angles = [], [], [], []
    for points in _contour_points(polygon):
        vertices = _unique_vertices(points)
        if len(vertices) < 2:
            raise ValueError("多邊形至少要兩個點")
        rows.extend(point[1] for point in vertices)
        cols.extend(point[0] for point in vertices)
        for start, end in zip(vertices, vertices[1:]):
            dcol = end[0] - start[0]
            drow = end[1] - start[1]
            lengths.append(math.hypot(dcol, drow))
            angles.append(math.atan2(drow, dcol))
    return Result([], [rows, cols, lengths, angles])


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [],
    [],
    [
        P("Length", "迴歸線上兩端的距離"),
        P("Phi", "迴歸線角度，弳度"),
        P("Row", "平均列"),
        P("Column", "平均行"),
        P("Deviation", "點到迴歸線的均方根"),
    ],
    "輪廓的迴歸線",
    "每個頂點權重相同。角度 0 朝右。點都在同一條直線上時，偏差是 0。每一條輪廓各一組。",
    "Vizion.get_regress_params_xld (Contours, Length, Phi, Row, Column, Deviation)",
    "新算子。從行 0 到行 10 的水平線，長度是 10，角度是 0，平均行是 5，偏差是 0。",
)
def get_regress_params_xld(contours, *, ctx):
    lengths, angles, rows, cols, deviations = [], [], [], [], []
    for points in _contour_points(contours):
        vertices = _unique_vertices(points)
        if len(vertices) < 2:
            raise ValueError("輪廓至少要兩個點")
        cloud = np.array(vertices, np.float64)
        center = cloud.mean(axis=0)
        shifted = cloud - center
        _u, _s, vt = np.linalg.svd(shifted, full_matrices=False)
        direction = vt[0]
        if direction[0] < 0 or (abs(direction[0]) < 1e-12 and direction[1] < 0):
            direction = -direction
        projected = shifted @ direction
        span = float(projected.max() - projected.min())
        normal = np.array([-direction[1], direction[0]])
        deviation = float(np.sqrt(np.mean((shifted @ normal) ** 2)))
        lengths.append(span)
        angles.append(math.atan2(float(direction[1]), float(direction[0])))
        cols.append(float(center[0]))
        rows.append(float(center[1]))
        deviations.append(deviation)
    return Result([], [lengths, angles, rows, cols, deviations])


def _ellipse_ray_distance(row, col, center_row, center_col, phi, radius1, radius2) -> float:
    cosine = math.cos(phi)
    sine = math.sin(phi)
    dcol = col - center_col
    drow = row - center_row
    local_col = dcol * cosine + drow * sine
    local_row = -dcol * sine + drow * cosine
    norm = math.hypot(local_col, local_row)
    if norm <= 1e-12:
        return min(radius1, radius2)
    scale = ((local_col / radius1) ** 2 + (local_row / radius2) ** 2) ** 0.5
    return abs(norm - norm / scale)


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [],
    [
        P("Row", "圓心列"),
        P("Column", "圓心行"),
        P("Phi", "長軸角度，弳度"),
        P("Radius1", "長軸半徑"),
        P("Radius2", "短軸半徑"),
    ],
    [P("Distance", "每個頂點沿圓心方向到邊界的距離")],
    "每個頂點到橢圓邊界的距離",
    "距離沿著圓心和這個點的方向量。點在邊界上是 0。圓心本身用較短的那一軸。",
    "Vizion.dist_ellipse_contour_points_xld (Contours, 0, 0, 0, 4, 4, Distance)",
    "新算子。半徑 4 的圓，圓心的距離是 4，圓周上的點是 0。",
)
def dist_ellipse_contour_points_xld(contours, row, column, phi, radius1, radius2, *, ctx):
    center_row = scalar(row, "Row")
    center_col = scalar(column, "Column")
    angle = scalar(phi, "Phi")
    major = scalar(radius1, "Radius1")
    minor = scalar(radius2, "Radius2")
    if major <= 0 or minor <= 0:
        raise ValueError("半徑必須大於 0")
    distances = []
    for points in _contour_points(contours):
        for col, item_row in points:
            distances.append(_ellipse_ray_distance(item_row, col, center_row, center_col, angle, major, minor))
    return Result([], [distances])


def _rectangle_distance(row, col, center_row, center_col, phi, length1, length2) -> float:
    cosine = math.cos(phi)
    sine = math.sin(phi)
    dcol = col - center_col
    drow = row - center_row
    local_col = abs(dcol * cosine + drow * sine)
    local_row = abs(-dcol * sine + drow * cosine)
    outside_col = local_col - length1
    outside_row = local_row - length2
    if outside_col <= 0 and outside_row <= 0:
        return min(-outside_col, -outside_row)
    outside_col = max(outside_col, 0.0)
    outside_row = max(outside_row, 0.0)
    return math.hypot(outside_col, outside_row)


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [],
    [
        P("Row", "中心列"),
        P("Column", "中心行"),
        P("Phi", "長軸角度，弳度"),
        P("Length1", "沿長軸的一半"),
        P("Length2", "沿短軸的一半"),
    ],
    [P("Distance", "每個頂點到矩形邊界的距離")],
    "每個頂點到矩形邊界的距離",
    "點在矩形裡面時，距離是到最近一邊。點在邊界上是 0。Phi 為 0 時長軸朝右。",
    "Vizion.dist_rectangle2_contour_points_xld (Contours, 0, 0, 0, 4, 2, Distance)",
    "新算子。半長 4、半寬 2 的矩形，中心的距離是 2，右邊界上的點是 0。",
)
def dist_rectangle2_contour_points_xld(contours, row, column, phi, length1, length2, *, ctx):
    center_row = scalar(row, "Row")
    center_col = scalar(column, "Column")
    angle = scalar(phi, "Phi")
    half_long = scalar(length1, "Length1")
    half_short = scalar(length2, "Length2")
    if half_long < 0 or half_short < 0:
        raise ValueError("長度不能是負的")
    distances = []
    for points in _contour_points(contours):
        for col, item_row in points:
            distances.append(_rectangle_distance(item_row, col, center_row, center_col, angle, half_long, half_short))
    return Result([], [distances])


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [],
    [],
    [P("M20", "行座標平方和"), P("M02", "列座標平方和"), P("M11", "列乘行的和")],
    "把頂點當成點雲的二階矩",
    "每個頂點權重相同，不看邊長。起點若在最後重複一次，不會算兩次。每一條輪廓各一組。",
    "Vizion.moments_points_xld (Contours, M20, M02, M11)",
    "新算子。正方形 (0,0)、(2,0)、(2,2)、(0,2) 的 M20 是 8，M02 是 8，M11 是 4。",
)
def moments_points_xld(contours, *, ctx):
    return moments_xld(contours, ctx=ctx)


@operator(
    "輪廓",
    [P("Contours", "輪廓")],
    [],
    [P("P", "行座標的次方"), P("Q", "列座標的次方")],
    [P("Moment", "頂點的矩")],
    "頂點的任意次矩",
    "每一個頂點把行的 P 次方乘列的 Q 次方，再加總。起點若在最後重複一次，不會算兩次。",
    "Vizion.moments_any_points_xld (Contours, 2, 0, Moment)",
    "新算子。正方形 (0,0)、(2,0)、(2,2)、(0,2) 的 M20 是 8。",
)
def moments_any_points_xld(contours, power_col, power_row, *, ctx):
    power_c = whole(power_col, "P")
    power_r = whole(power_row, "Q")
    if power_c < 0 or power_r < 0:
        raise ValueError("次方不能是負的")
    values = []
    for points in _contour_points(contours):
        vertices = _unique_vertices(points)
        total = 0.0
        for col, row in vertices:
            total += (col ** power_c) * (row ** power_r)
        values.append(total)
    return Result([], [values])


@operator(
    "輪廓",
    [P("Contours", "輪廓"), P("Image", "灰階影像")],
    [P("LocalMax", "落在局部亮處的輪廓")],
    [],
    [],
    "留下落在局部最亮處的輪廓",
    "每個頂點的灰階都要大於或等於八鄰域，這條輪廓才留下。有一個頂點旁邊更亮就丟掉。彩色先轉成亮度。",
    "Vizion.local_max_contours_xld (Contours, Image, LocalMax)",
    "新算子。中間一列是 9、上面一列旁邊是 5 時，只留下中間那條。",
)
def local_max_contours_xld(contours, image, *, ctx):
    source = expect_xld(contours, "Contours")
    picture = expect_image(image, "Image")
    gray = picture.gray()
    kept = []
    neighbors = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    for points in _contour_points(source):
        vertices = _unique_vertices(points)
        bright = True
        for col, row in vertices:
            item_row = int(round(row))
            item_col = int(round(col))
            if item_row < 0 or item_col < 0 or item_row >= gray.shape[0] or item_col >= gray.shape[1]:
                bright = False
                break
            value = int(gray[item_row, item_col])
            for drow, dcol in neighbors:
                nrow, ncol = item_row + drow, item_col + dcol
                if 0 <= nrow < gray.shape[0] and 0 <= ncol < gray.shape[1] and int(gray[nrow, ncol]) > value:
                    bright = False
                    break
            if not bright:
                break
        if bright and len(vertices) >= 2:
            kept.append([[point[0], point[1]] for point in vertices])
    return Result([XLD(source.width, source.height, kept, multi_color=len(kept) > 1)], [])
