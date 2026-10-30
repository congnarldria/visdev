"""XLD clipping, polygon approximation, and affine transform."""

from __future__ import annotations

import numpy as np
import cv2

from ..objects import Result, XLD
from ..operators import P, expect_xld, operator, scalar, text
from .image_ops import _mat_from


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
    "clip_contours_xld (Contours, Clipped, 0, 0, 100, 100)",
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
    "gen_polygons_xld (Contours, Polygons, 'ramer', 2)",
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
    "affine_trans_contour_xld (Contours, ContoursAffineTrans, HomMat2D)",
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
