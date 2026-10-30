"""Display operators that draw on the current canvas."""

from __future__ import annotations

import math

from ..objects import Result, XLD
from ..operators import P, as_list, operator, scalar


def _show(ctx, contours: list) -> Result:
    height, width = ctx.canvas_size()
    ctx.display_obj(XLD(width, height, contours, multi_color=len(contours) > 1))
    return Result([], [])


def _pairs(row, column):
    rows = [float(item) for item in as_list(row)]
    cols = [float(item) for item in as_list(column)]
    if len(rows) == 1 and len(cols) > 1:
        rows = rows * len(cols)
    if len(cols) == 1 and len(rows) > 1:
        cols = cols * len(rows)
    if len(rows) != len(cols):
        raise ValueError("Row 和 Column 的長度不同")
    return rows, cols


@operator(
    "顯示",
    [],
    [],
    [P("Row", "圓心列"), P("Column", "圓心行"), P("Radius", "半徑")],
    [],
    "在圖上畫圓",
    "疊在目前影像上，用現在的顏色和線寬。Row、Column 可以是 tuple。",
    "dev_disp_circle (120, 100, 30)",
)
def dev_disp_circle(row, column, radius, *, ctx):
    rows, cols = _pairs(row, column)
    rad = scalar(radius, "Radius")
    if rad < 0:
        raise ValueError("半徑不能是負的")
    contours = []
    for center_row, center_col in zip(rows, cols):
        points = []
        for step in range(73):
            angle = 2 * math.pi * step / 72
            points.append([center_col + rad * math.cos(angle), center_row + rad * math.sin(angle)])
        contours.append(points)
    return _show(ctx, contours)


@operator(
    "顯示",
    [],
    [],
    [P("Row1", "上"), P("Column1", "左"), P("Row2", "下"), P("Column2", "右")],
    [],
    "在圖上畫軸對齊矩形",
    "疊在目前影像上，用現在的顏色和線寬。",
    "dev_disp_rectangle1 (40, 20, 120, 180)",
)
def dev_disp_rectangle1(row1, column1, row2, column2, *, ctx):
    top, left = scalar(row1, "Row1"), scalar(column1, "Column1")
    bottom, right = scalar(row2, "Row2"), scalar(column2, "Column2")
    contour = [[left, top], [right, top], [right, bottom], [left, bottom], [left, top]]
    return _show(ctx, [contour])


@operator(
    "顯示",
    [],
    [],
    [P("Row1", "起點列"), P("Column1", "起點行"), P("Row2", "終點列"), P("Column2", "終點行")],
    [],
    "在圖上畫直線",
    "疊在目前影像上，用現在的顏色和線寬。",
    "dev_disp_line (10, 10, 100, 80)",
)
def dev_disp_line(row1, column1, row2, column2, *, ctx):
    contour = [
        [scalar(column1, "Column1"), scalar(row1, "Row1")],
        [scalar(column2, "Column2"), scalar(row2, "Row2")],
    ]
    return _show(ctx, [contour])
