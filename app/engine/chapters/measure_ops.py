"""1D measure, 2D metrology, scaled shape search, and model queries.

Names and argument order follow HDevelop. The profile is the same gradient
peak used by the gauge operators. A metrology object keeps one fitted result.
Scaled shape search builds a normal shape model at each discrete scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .. import gauge
from ..ncc_model import NccModel
from ..objects import Result, XLD
from ..operators import P, _optional_number, as_list, expect_image, operator, scalar, text, whole
from ..shape_model import ScaledShapeModel, ShapeModel
from ..shape_model import create_scaled_shape_model as build_scaled_shape_model
from ..shape_model import find_scaled_shape_model as search_scaled_shape_model

_TRANSITION = ["'all'", "'positive'", "'negative'"]
_SELECT = ["'all'", "'first'", "'last'", "'best'"]
_INTERP = ["'nearest_neighbor'", "'bilinear'", "'bicubic'"]
_OPTIMIZATION = [
    "'auto'",
    "'none'",
    "'point_reduction_low'",
    "'point_reduction_medium'",
    "'point_reduction_high'",
]
_METRIC = ["'use_polarity'", "'ignore_local_polarity'"]
_SUBPIXEL = ["'none'", "'interpolation'", "'least_squares'"]
_INSTANCE = ["'all'", "0"]
_RESULT_NAME = ["'result_type'"]
_RESULT_VALUE = [
    "'all_param'",
    "'row'",
    "'column'",
    "'radius'",
    "'phi'",
    "'radius1'",
    "'radius2'",
    "'row_begin'",
    "'column_begin'",
    "'row_end'",
    "'column_end'",
]
_PARAM_KEYS = {
    "line": ("row_begin", "column_begin", "row_end", "column_end"),
    "circle": ("row", "column", "radius"),
    "ellipse": ("row", "column", "phi", "radius1", "radius2"),
}


@dataclass
class MeasureRectangle2:
    row: float
    column: float
    phi: float
    length1: float
    length2: float
    width: int
    height: int
    interpolation: str
    closed: bool = False

    def summary(self) -> str:
        state = "已關閉" if self.closed else "使用中"
        return f"量測矩形  {state}  中心 {self.row:.1f},{self.column:.1f}  半長 {self.length1:.1f}"


@dataclass
class MeasureArc:
    row: float
    column: float
    radius: float
    angle_start: float
    angle_extent: float
    annulus: float
    width: int
    height: int
    interpolation: str
    closed: bool = False

    def summary(self) -> str:
        state = "已關閉" if self.closed else "使用中"
        return f"量測弧  {state}  中心 {self.row:.1f},{self.column:.1f}  半徑 {self.radius:.1f}"


@dataclass
class Funct1D:
    x: np.ndarray
    y: np.ndarray

    def summary(self) -> str:
        return f"一維函數  {len(self.x)} 點  x {float(self.x[0]):.4g}…{float(self.x[-1]):.4g}"


@dataclass
class MetrologyObject:
    kind: str
    nominal: dict
    measure_length1: float
    measure_length2: float
    sigma: float
    threshold: float
    result: dict | None = None


@dataclass
class MetrologyModel:
    width: int = 0
    height: int = 0
    objects: list = field(default_factory=list)
    closed: bool = False

    def summary(self) -> str:
        state = "已關閉" if self.closed else "使用中"
        return f"計量模型  {state}  {len(self.objects)} 個物件"


def _token(value, name: str) -> str:
    return text(value, name).strip().strip("'\"").lower()


def _expect_measure(value) -> MeasureRectangle2:
    if not isinstance(value, MeasureRectangle2):
        raise TypeError("MeasureHandle 必須是 gen_measure_rectangle2 建立的量測矩形")
    if value.closed:
        raise RuntimeError("量測物件已經 close_measure")
    return value


def _expect_metro(value) -> MetrologyModel:
    if not isinstance(value, MetrologyModel):
        raise TypeError("MetrologyHandle 必須是 create_metrology_model 建立的計量模型")
    if value.closed:
        raise RuntimeError("計量模型已經 clear_metrology_model")
    return value


def _expect_ncc(value) -> NccModel:
    if not isinstance(value, NccModel):
        raise TypeError("ModelID 必須是 NCC 模型")
    if getattr(value, "cleared", False):
        raise RuntimeError("模型已經 clear，不能再使用")
    return value


def _expect_shape(value):
    if not isinstance(value, (ShapeModel, ScaledShapeModel)):
        raise TypeError("ModelID 必須是形狀模型或縮放形狀模型")
    if getattr(value, "cleared", False):
        raise RuntimeError("模型已經 clear，不能再使用")
    return value


def _check_size(image, width: int, height: int, what: str) -> None:
    if width and height and (image.width != width or image.height != height):
        raise ValueError(f"影像是 {image.width}×{image.height}，{what} 是 {width}×{height}")


def _sample_at(gray, domain, column: float, row: float, interpolation: str) -> float:
    if interpolation == "nearest_neighbor":
        column = round(column)
        row = round(row)
    if interpolation == "bicubic":
        height, width = gray.shape
        if column < 0 or row < 0 or column > width - 1 or row > height - 1:
            return math.nan
        if domain is not None:
            ix = min(max(int(round(column)), 0), width - 1)
            iy = min(max(int(round(row)), 0), height - 1)
            if domain[iy, ix] <= 0:
                return math.nan
        return float(gauge.sample_bicubic(gray, row, column))
    return gauge._sample(gray, domain, column, row)


def _profile_hits(gray, domain, row1, col1, row2, col2, length2, sigma, threshold, transition, interpolation):
    real = math.hypot(col2 - col1, row2 - row1)
    if real < 1.5:
        return []
    step = 0.5
    count = int(real / step) + 1
    dx = (col2 - col1) / real
    dy = (row2 - row1) / real
    normal_col = -dy
    normal_row = dx
    half = max(0, int(round(length2)))
    profile = np.full(count, np.nan, np.float64)
    for index, distance in enumerate(np.linspace(0.0, real, count)):
        column = col1 + dx * distance
        row = row1 + dy * distance
        samples = []
        for offset in range(-half, half + 1):
            value = _sample_at(gray, domain, column + normal_col * offset, row + normal_row * offset, interpolation)
            if math.isfinite(value):
                samples.append(value)
        if samples:
            profile[index] = float(np.mean(samples))
    band = np.vstack([profile, profile, profile])
    hits = gauge._locate(band, None, 0.0, 1.0, float(count - 1), 1.0, sigma, threshold, transition)
    mapped = []
    scale = real / max(count - 1, 1)
    for hit in hits:
        distance = float(hit.distance) * scale
        mapped.append(
            gauge.Hit(
                float(row1 + dy * distance),
                float(col1 + dx * distance),
                float(hit.amplitude),
                distance,
            )
        )
    return mapped


def _axis(handle: MeasureRectangle2):
    sine = math.sin(handle.phi)
    cosine = math.cos(handle.phi)
    row1 = handle.row - handle.length1 * sine
    col1 = handle.column - handle.length1 * cosine
    row2 = handle.row + handle.length1 * sine
    col2 = handle.column + handle.length1 * cosine
    return row1, col1, row2, col2


def _edges(image, handle: MeasureRectangle2, sigma, threshold, transition):
    _check_size(image, handle.width, handle.height, "量測矩形")
    gray = image.gray()
    row1, col1, row2, col2 = _axis(handle)
    kind = gauge._choice(transition, gauge._TRANSITIONS, "Transition")
    return _profile_hits(
        gray,
        image.domain,
        row1,
        col1,
        row2,
        col2,
        handle.length2,
        scalar(sigma, "Sigma"),
        scalar(threshold, "Threshold"),
        kind,
        handle.interpolation,
    ), kind


def _pair_edges(hits, transition: str):
    ordered = sorted(hits, key=lambda item: item.distance)
    pairs = []
    index = 0
    while index < len(ordered) - 1:
        first, second = ordered[index], ordered[index + 1]
        opposite = first.amplitude * second.amplitude < 0
        wanted = transition == "all" or (transition == "positive" and first.amplitude > 0) or (
            transition == "negative" and first.amplitude < 0
        )
        if opposite and wanted:
            pairs.append((first, second))
            index += 2
        else:
            index += 1
    return pairs


def _pick_pairs(pairs, select: str):
    if not pairs or select == "all":
        return pairs
    if select == "first":
        return pairs[:1]
    if select == "last":
        return pairs[-1:]
    return [max(pairs, key=lambda item: abs(item[0].amplitude))]


def _column(values) -> list[float]:
    return [float(item) for item in values]


def _origin_of(model):
    row = model.ref_row if getattr(model, "ref_row", None) is not None else model.origin_row
    column = model.ref_col if getattr(model, "ref_col", None) is not None else model.origin_col
    return float(row), float(column)


def _indices(value, count: int, name: str) -> list[int]:
    if isinstance(value, str) and _token(value, name) == "all":
        return list(range(count))
    if isinstance(value, (list, tuple)):
        picked = [int(round(float(item))) for item in value]
    else:
        try:
            picked = [int(round(float(value)))]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 要是 all 或從 0 開始的整數") from exc
    for index in picked:
        if index < 0 or index >= count:
            raise IndexError(f"{name} {index} 超出範圍 0..{count - 1}")
    return picked


def _result_values(obj: MetrologyObject, mode: str) -> list[float]:
    if obj.result is None:
        raise RuntimeError("這個計量物件還沒有結果。請先 apply_metrology_model")
    keys = _PARAM_KEYS[obj.kind]
    if mode in ("", "all_param", "all"):
        return [float(obj.result[key]) for key in keys]
    if mode not in keys:
        known = "、".join(keys)
        raise ValueError(f"{obj.kind} 沒有結果 {mode}。可用 {known} 或 all_param")
    return [float(obj.result[mode])]


def _line_contour(row1, col1, row2, col2, resolution: float):
    length = math.hypot(col2 - col1, row2 - row1)
    count = max(1, int(math.ceil(length / max(resolution, 0.5))))
    return [
        [col1 + (col2 - col1) * index / count, row1 + (row2 - row1) * index / count]
        for index in range(count + 1)
    ]


def _circle_contour(row, col, radius, resolution: float):
    count = max(12, int(math.ceil(2 * math.pi * max(radius, 1.0) / max(resolution, 0.5))))
    return [
        [col + radius * math.cos(2 * math.pi * index / count), row + radius * math.sin(2 * math.pi * index / count)]
        for index in range(count + 1)
    ]


def _ellipse_contour(row, col, phi, radius1, radius2, resolution: float):
    perimeter = 2 * math.pi * math.sqrt((radius1 * radius1 + radius2 * radius2) / 2.0)
    count = max(16, int(math.ceil(max(perimeter, 1.0) / max(resolution, 0.5))))
    points = []
    for index in range(count + 1):
        theta = 2 * math.pi * index / count
        points.append(
            [
                col + radius1 * math.cos(theta) * math.cos(phi) - radius2 * math.sin(theta) * math.sin(phi),
                row + radius1 * math.cos(theta) * math.sin(phi) + radius2 * math.sin(theta) * math.cos(phi),
            ]
        )
    return points


def _contour_of(obj: MetrologyObject, resolution: float):
    if obj.result is None:
        raise RuntimeError("這個計量物件還沒有結果。請先 apply_metrology_model")
    result = obj.result
    if obj.kind == "line":
        return _line_contour(result["row_begin"], result["column_begin"], result["row_end"], result["column_end"], resolution)
    if obj.kind == "circle":
        return _circle_contour(result["row"], result["column"], result["radius"], resolution)
    return _ellipse_contour(result["row"], result["column"], result["phi"], result["radius1"], result["radius2"], resolution)


def _ellipse_points(gray, domain, row, col, phi, radius1, radius2, length1, sigma, threshold, transition, length2=0.0):
    points = []
    steps = 36
    for index in range(steps):
        theta = index * math.tau / steps
        edge_row = row + radius1 * math.cos(theta) * math.sin(phi) + radius2 * math.sin(theta) * math.cos(phi)
        edge_col = col + radius1 * math.cos(theta) * math.cos(phi) - radius2 * math.sin(theta) * math.sin(phi)
        normal_col = edge_col - col
        normal_row = edge_row - row
        norm = math.hypot(normal_col, normal_row) or 1.0
        normal_col /= norm
        normal_row /= norm
        hits = gauge._locate(
            gray,
            domain,
            edge_col - normal_col * length1,
            edge_row - normal_row * length1,
            edge_col + normal_col * length1,
            edge_row + normal_row * length1,
            sigma,
            threshold,
            transition,
            length2,
        )
        if not hits:
            continue
        best = max(hits, key=lambda item: abs(item.amplitude))
        points.append((float(best.column), float(best.row)))
    return points


@operator(
    "量測",
    [],
    [],
    [
        P("Row", "量測矩形中心列"),
        P("Column", "量測矩形中心行"),
        P("Phi", "長軸角度，弳度。0 指向右方"),
        P("Length1", "沿長軸的半長，像素"),
        P("Length2", "沿短軸的半寬，像素。剖面會在這個寬度裡平均"),
        P("Width", "影像寬"),
        P("Height", "影像高"),
        P("Interpolation", "剖面取樣", "'bilinear'", _INTERP),
    ],
    [P("MeasureHandle", "量測矩形")],
    "建立量測矩形",
    "剖面沿 Phi 從中心往兩側各 Length1。Phi 為 0 時沿行方向。positive 是沿著這個方向由暗到亮。"
    "nearest_neighbor 會把取樣座標取整。bilinear 看四個像素。bicubic 用周圍 16 個像素的三次內插。",
    "Vizion.gen_measure_rectangle2 (120, 100, 0, 40, 5, 640, 480, 'bilinear', MeasureHandle)",
)
def gen_measure_rectangle2(row, column, phi, length1, length2, width, height, interpolation, *, ctx):
    mode = _token(interpolation, "Interpolation")
    if mode not in ("nearest_neighbor", "bilinear", "bicubic"):
        raise ValueError("Interpolation 只能是 nearest_neighbor、bilinear 或 bicubic")
    half_length = scalar(length1, "Length1")
    half_width = scalar(length2, "Length2")
    if half_length < 1 or half_width < 0:
        raise ValueError("Length1 至少要 1，Length2 不能是負的")
    handle = MeasureRectangle2(
        scalar(row, "Row"),
        scalar(column, "Column"),
        scalar(phi, "Phi"),
        half_length,
        half_width,
        whole(width, "Width"),
        whole(height, "Height"),
        mode,
    )
    if handle.width < 2 or handle.height < 2:
        raise ValueError("Width 和 Height 至少要 2")
    return Result([], [handle])


@operator(
    "量測",
    [P("Image", "灰階或彩色影像")],
    [],
    [
        P("MeasureHandle", "量測矩形"),
        P("Sigma", "平滑，像素"),
        P("Threshold", "梯度門檻，灰階"),
        P("Transition", "all、positive 或 negative", "'all'", _TRANSITION),
        P("Select", "all、first、last 或 best", "'all'", _SELECT),
    ],
    [
        P("RowEdge", "邊緣列"),
        P("ColumnEdge", "邊緣行"),
        P("Amplitude", "梯度。正值是由暗到亮"),
        P("Distance", "第一個是離起點的距離，其後是相鄰邊緣的間距"),
    ],
    "沿量測矩形找邊緣",
    "這是 measure_pos。找不到邊緣時回傳空 tuple，不會中斷。Select 的 best 是振幅最大的一個，HALCON 的 measure_pos 沒有這個選項。"
    "gauge_line 仍是另一套簡化量規，不會被這個算子取代。",
    "Vizion.measure_pos (Image, MeasureHandle, 1, 20, 'all', 'all', RowEdge, ColumnEdge, Amplitude, Distance)",
)
def measure_pos(image, handle, sigma, threshold, transition, select, *, ctx):
    picture = expect_image(image, "Image")
    rect = _expect_measure(handle)
    hits, _kind = _edges(picture, rect, sigma, threshold, transition)
    chosen = gauge._pick(hits, gauge._choice(select, gauge._SELECTS, "Select"))
    if not chosen:
        return Result([], [[], [], [], []])
    distances = [chosen[0].distance]
    for previous, current in zip(chosen, chosen[1:]):
        distances.append(current.distance - previous.distance)
    return Result(
        [],
        [
            _column(item.row for item in chosen),
            _column(item.column for item in chosen),
            _column(item.amplitude for item in chosen),
            _column(distances),
        ],
    )


@operator(
    "量測",
    [P("Image", "灰階或彩色影像")],
    [],
    [
        P("MeasureHandle", "量測矩形"),
        P("Sigma", "平滑，像素"),
        P("Threshold", "梯度門檻，灰階"),
        P("Transition", "一對裡第一個邊緣的方向", "'all'", _TRANSITION),
        P("Select", "all、first、last 或 best", "'all'", _SELECT),
    ],
    [
        P("RowEdgeFirst", "第一個邊緣的列"),
        P("ColumnEdgeFirst", "第一個邊緣的行"),
        P("AmplitudeFirst", "第一個邊緣的梯度"),
        P("RowEdgeSecond", "第二個邊緣的列"),
        P("ColumnEdgeSecond", "第二個邊緣的行"),
        P("AmplitudeSecond", "第二個邊緣的梯度"),
        P("IntraDistance", "這一對兩個邊緣的距離"),
        P("InterDistance", "第一對是離起點的距離，其後是前一對終點到這一對起點"),
    ],
    "沿量測矩形找成對邊緣",
    "一對的兩個邊緣梯度方向相反。Transition 決定這一對的第一個邊緣。找不到時回傳空 tuple。",
    "Vizion.measure_pairs (Image, MeasureHandle, 1, 20, 'positive', 'first', Row1, Col1, Amp1, Row2, Col2, Amp2, Intra, Inter)",
)
def measure_pairs(image, handle, sigma, threshold, transition, select, *, ctx):
    picture = expect_image(image, "Image")
    rect = _expect_measure(handle)
    kind = gauge._choice(transition, gauge._TRANSITIONS, "Transition")
    hits, _found = _edges(picture, rect, sigma, threshold, "all")
    pairs = _pick_pairs(_pair_edges(hits, kind), gauge._choice(select, gauge._SELECTS, "Select"))
    if not pairs:
        return Result([], [[], [], [], [], [], [], [], []])
    intra = []
    inter = []
    previous = None
    for first, second in pairs:
        intra.append(second.distance - first.distance)
        inter.append(first.distance if previous is None else first.distance - previous)
        previous = second.distance
    return Result(
        [],
        [
            _column(item[0].row for item in pairs),
            _column(item[0].column for item in pairs),
            _column(item[0].amplitude for item in pairs),
            _column(item[1].row for item in pairs),
            _column(item[1].column for item in pairs),
            _column(item[1].amplitude for item in pairs),
            _column(intra),
            _column(inter),
        ],
    )


@operator(
    "量測",
    [],
    [],
    [P("MeasureHandle", "要關閉的量測矩形")],
    [],
    "關閉量測矩形",
    "關閉後再呼叫 measure_pos 或 measure_pairs 會失敗。變數還在，只是不能再量。",
    "Vizion.close_measure (MeasureHandle)",
)
def close_measure(handle, *, ctx):
    if not isinstance(handle, (MeasureRectangle2, MeasureArc)):
        raise TypeError("MeasureHandle 必須是 gen_measure_rectangle2 或 gen_measure_arc 建立的量測")
    if handle.closed:
        raise RuntimeError("量測物件已經 close_measure")
    handle.closed = True
    return Result([], [])


@operator(
    "量測",
    [],
    [],
    [],
    [P("MetrologyHandle", "計量模型")],
    "建立計量模型",
    "模型先是空的。接著設定影像大小，加入直線、圓或橢圓，再 apply。",
    "Vizion.create_metrology_model (MetrologyHandle)",
)
def create_metrology_model(*, ctx):
    return Result([], [MetrologyModel()])


@operator(
    "量測",
    [],
    [],
    [
        P("MetrologyHandle", "計量模型"),
        P("Width", "影像寬"),
        P("Height", "影像高"),
    ],
    [],
    "設定計量模型的影像大小",
    "apply 時影像尺寸必須和這裡相同。",
    "Vizion.set_metrology_model_image_size (MetrologyHandle, 640, 480)",
)
def set_metrology_model_image_size(handle, width, height, *, ctx):
    model = _expect_metro(handle)
    model.width = whole(width, "Width")
    model.height = whole(height, "Height")
    if model.width < 2 or model.height < 2:
        raise ValueError("Width 和 Height 至少要 2")
    return Result([], [])


def _add_object(handle, kind: str, nominal: dict, length1, length2, sigma, threshold) -> int:
    model = _expect_metro(handle)
    obj = MetrologyObject(
        kind,
        nominal,
        scalar(length1, "MeasureLength1"),
        scalar(length2, "MeasureLength2"),
        scalar(sigma, "MeasureSigma"),
        scalar(threshold, "MeasureThreshold"),
    )
    if obj.measure_length1 < 1 or obj.measure_length2 < 0 or obj.sigma < 0 or obj.threshold < 0:
        raise ValueError("MeasureLength1 至少要 1，其餘量測參數不能是負的")
    model.objects.append(obj)
    return len(model.objects) - 1


@operator(
    "量測",
    [],
    [],
    [
        P("MetrologyHandle", "計量模型"),
        P("RowBegin", "名義直線的起點列"),
        P("ColumnBegin", "名義直線的起點行"),
        P("RowEnd", "名義直線的終點列"),
        P("ColumnEnd", "名義直線的終點行"),
        P("MeasureLength1", "垂直於直線的搜尋半長"),
        P("MeasureLength2", "沿直線的卡尺半寬。剖面會在這個寬度裡平均"),
        P("MeasureSigma", "平滑，像素"),
        P("MeasureThreshold", "梯度門檻"),
        P("GenParamName", "這版不使用，留空字串", "''"),
        P("GenParamValue", "這版不使用，留空字串", "''"),
    ],
    [P("Index", "物件索引，從 0 開始")],
    "加入直線計量物件",
    "名義線兩側各搜尋 MeasureLength1。每支卡尺再沿直線把 MeasureLength2 寬度的灰階平均，壓掉沿邊緣的雜點。",
    "Vizion.add_metrology_object_line_measure (MetrologyHandle, 40, 40, 200, 40, 12, 3, 1, 20, Index)",
)
def add_metrology_object_line_measure(
    handle, row_begin, column_begin, row_end, column_end, length1, length2, sigma, threshold, _name, _value, *, ctx
):
    index = _add_object(
        handle,
        "line",
        {
            "row_begin": scalar(row_begin, "RowBegin"),
            "column_begin": scalar(column_begin, "ColumnBegin"),
            "row_end": scalar(row_end, "RowEnd"),
            "column_end": scalar(column_end, "ColumnEnd"),
        },
        length1,
        length2,
        sigma,
        threshold,
    )
    return Result([], [index])


@operator(
    "量測",
    [],
    [],
    [
        P("MetrologyHandle", "計量模型"),
        P("Row", "名義圓心列"),
        P("Column", "名義圓心行"),
        P("Radius", "名義半徑"),
        P("MeasureLength1", "沿半徑方向的搜尋半長"),
        P("MeasureLength2", "沿切向的卡尺半寬。剖面會在這個寬度裡平均"),
        P("MeasureSigma", "平滑，像素"),
        P("MeasureThreshold", "梯度門檻"),
        P("GenParamName", "這版不使用，留空字串", "''"),
        P("GenParamValue", "這版不使用，留空字串", "''"),
    ],
    [P("Index", "物件索引，從 0 開始")],
    "加入圓計量物件",
    "沿名義圓取样，再把找到的邊緣重擬合。每支卡尺沿切向平均 MeasureLength2。",
    "Vizion.add_metrology_object_circle_measure (MetrologyHandle, 120, 100, 30, 12, 3, 1, 20, Index)",
)
def add_metrology_object_circle_measure(handle, row, column, radius, length1, length2, sigma, threshold, _name, _value, *, ctx):
    index = _add_object(
        handle,
        "circle",
        {"row": scalar(row, "Row"), "column": scalar(column, "Column"), "radius": scalar(radius, "Radius")},
        length1,
        length2,
        sigma,
        threshold,
    )
    return Result([], [index])


@operator(
    "量測",
    [],
    [],
    [
        P("MetrologyHandle", "計量模型"),
        P("Row", "名義橢圓中心列"),
        P("Column", "名義橢圓中心行"),
        P("Phi", "Radius1 那一軸的角度，弳度。0 指向右方"),
        P("Radius1", "第一軸的半長"),
        P("Radius2", "第二軸的半長"),
        P("MeasureLength1", "沿法向的搜尋半長"),
        P("MeasureLength2", "沿切向的卡尺半寬。剖面會在這個寬度裡平均"),
        P("MeasureSigma", "平滑，像素"),
        P("MeasureThreshold", "梯度門檻"),
        P("GenParamName", "這版不使用，留空字串", "''"),
        P("GenParamValue", "這版不使用，留空字串", "''"),
    ],
    [P("Index", "物件索引，從 0 開始")],
    "加入橢圓計量物件",
    "沿名義橢圓的法向找邊緣，每支卡尺沿切向平均 MeasureLength2，再用這些點擬合。擬合出來的 Phi 可能和輸入差 90 度，因為兩軸可以對調。",
    "Vizion.add_metrology_object_ellipse_measure (MetrologyHandle, 160, 220, rad(25), 36, 22, 12, 3, 1, 20, Index)",
)
def add_metrology_object_ellipse_measure(
    handle, row, column, phi, radius1, radius2, length1, length2, sigma, threshold, _name, _value, *, ctx
):
    index = _add_object(
        handle,
        "ellipse",
        {
            "row": scalar(row, "Row"),
            "column": scalar(column, "Column"),
            "phi": scalar(phi, "Phi"),
            "radius1": scalar(radius1, "Radius1"),
            "radius2": scalar(radius2, "Radius2"),
        },
        length1,
        length2,
        sigma,
        threshold,
    )
    return Result([], [index])


@operator(
    "量測",
    [P("Image", "要量測的影像")],
    [],
    [P("MetrologyHandle", "計量模型")],
    [],
    "套用量測並擬合",
    "每個物件各自擬合。直線回傳兩端點，圓回傳圓心和半徑，橢圓回傳中心、角度和兩軸。某個物件找不到邊緣時，整個 apply 失敗。",
    "Vizion.apply_metrology_model (Image, MetrologyHandle)",
)
def apply_metrology_model(image, handle, *, ctx):
    picture = expect_image(image, "Image")
    model = _expect_metro(handle)
    if not model.objects:
        raise RuntimeError("計量模型還沒有物件")
    _check_size(picture, model.width, model.height, "計量模型")
    gray = picture.gray()
    for index, obj in enumerate(model.objects):
        try:
            if obj.kind == "line":
                fit = gauge.measure_line(
                    gray,
                    picture.domain,
                    obj.nominal["row_begin"],
                    obj.nominal["column_begin"],
                    obj.nominal["row_end"],
                    obj.nominal["column_end"],
                    obj.measure_length1,
                    obj.sigma,
                    obj.threshold,
                    "all",
                    "all",
                    obj.measure_length2,
                )
                obj.result = {
                    "row_begin": fit.row1,
                    "column_begin": fit.column1,
                    "row_end": fit.row2,
                    "column_end": fit.column2,
                }
            elif obj.kind == "circle":
                fit = gauge.measure_circle(
                    gray,
                    picture.domain,
                    obj.nominal["row"],
                    obj.nominal["column"],
                    obj.nominal["radius"],
                    obj.measure_length1,
                    obj.sigma,
                    obj.threshold,
                    "all",
                    "all",
                    obj.measure_length2,
                )
                obj.result = {"row": fit.row, "column": fit.column, "radius": fit.radius}
            else:
                points = _ellipse_points(
                    gray,
                    picture.domain,
                    obj.nominal["row"],
                    obj.nominal["column"],
                    obj.nominal["phi"],
                    obj.nominal["radius1"],
                    obj.nominal["radius2"],
                    obj.measure_length1,
                    obj.sigma,
                    obj.threshold,
                    "all",
                    obj.measure_length2,
                )
                if len(points) < 5:
                    raise ValueError("橢圓量規找不到足夠的邊緣")
                (center_col, center_row), (width, height), angle = cv2.fitEllipse(
                    np.asarray(points, np.float32).reshape(-1, 1, 2)
                )
                obj.result = {
                    "row": float(center_row),
                    "column": float(center_col),
                    "phi": math.radians(float(angle)),
                    "radius1": float(width) / 2.0,
                    "radius2": float(height) / 2.0,
                }
        except ValueError as exc:
            raise RuntimeError(f"計量物件 {index} 失敗：{exc}") from exc
    return Result([], [])


@operator(
    "量測",
    [],
    [],
    [
        P("MetrologyHandle", "計量模型"),
        P("Index", "從 0 開始的索引，或 all"),
        P("Instance", "這版每個物件只有一個結果", "'all'", _INSTANCE),
        P("GenParamName", "結果種類", "'result_type'", _RESULT_NAME),
        P("GenParamValue", "all_param 或單一參數名", "'all_param'", _RESULT_VALUE),
    ],
    [P("Parameter", "結果數值")],
    "讀取計量結果",
    "直線的 all_param 是起點列、起點行、終點列、終點行。圓是列、行、半徑。橢圓是列、行、角度、Radius1、Radius2。"
    "索引從 0 開始。多個物件會依索引接成一條 tuple。",
    "Vizion.get_metrology_object_result (MetrologyHandle, 0, 'all', 'result_type', 'all_param', Parameter)",
)
def get_metrology_object_result(handle, index, instance, gen_name, gen_value, *, ctx):
    model = _expect_metro(handle)
    chosen = _indices(index, len(model.objects), "Index")
    which = _token(instance, "Instance")
    if which not in ("all", "0"):
        raise ValueError("Instance 這版只接受 all 或 0")
    mode_name = _token(gen_name, "GenParamName")
    if mode_name not in ("", "result_type"):
        raise ValueError("GenParamName 目前只做 result_type")
    mode = _token(gen_value, "GenParamValue")
    values = []
    for item in chosen:
        values.extend(_result_values(model.objects[item], mode))
    return Result([], [values])


@operator(
    "量測",
    [],
    [P("Contour", "擬合後的輪廓")],
    [
        P("MetrologyHandle", "計量模型"),
        P("Index", "從 0 開始的索引，或 all"),
        P("Instance", "這版每個物件只有一個結果", "'all'", _INSTANCE),
        P("Resolution", "輪廓點距，像素", "1.5"),
    ],
    [],
    "把計量結果畫成輪廓",
    "直線、圓、橢圓各自變成一條 XLD。點距是 Resolution。",
    "Vizion.get_metrology_object_result_contour (Contour, MetrologyHandle, 'all', 'all', 1.5)",
)
def get_metrology_object_result_contour(handle, index, instance, resolution, *, ctx):
    model = _expect_metro(handle)
    chosen = _indices(index, len(model.objects), "Index")
    which = _token(instance, "Instance")
    if which not in ("all", "0"):
        raise ValueError("Instance 這版只接受 all 或 0")
    spacing = scalar(resolution, "Resolution")
    if spacing <= 0:
        raise ValueError("Resolution 必須大於 0")
    contours = [_contour_of(model.objects[item], spacing) for item in chosen]
    width = model.width or 1
    height = model.height or 1
    return Result([XLD(width, height, contours, multi_color=len(contours) > 1)], [])


@operator(
    "量測",
    [],
    [],
    [P("MetrologyHandle", "要清掉的計量模型")],
    [],
    "清除計量模型",
    "物件和結果都會清掉。之後不能再 apply 或讀結果。",
    "Vizion.clear_metrology_model (MetrologyHandle)",
)
def clear_metrology_model(handle, *, ctx):
    model = handle if isinstance(handle, MetrologyModel) else None
    if model is None:
        raise TypeError("MetrologyHandle 必須是 create_metrology_model 建立的計量模型")
    model.objects.clear()
    model.closed = True
    return Result([], [])


@operator(
    "形狀匹配",
    [P("Template", "模板影像，可用 domain")],
    [],
    [
        P("NumLevels", "金字塔層數，0 為自動"),
        P("AngleStart", "起始角，弳度"),
        P("AngleExtent", "角度範圍，弳度"),
        P("AngleStep", "角度步進，0 為自動"),
        P("ScaleMin", "最小尺度"),
        P("ScaleMax", "最大尺度"),
        P("ScaleStep", "尺度步進，0 為自動"),
        P("Optimization", "點抽稀。這版都會接受，但不會再減少點", "'none'", _OPTIMIZATION),
        P("Metric", "use_polarity 或 ignore_local_polarity", "'use_polarity'", _METRIC),
        P("Contrast", "邊緣對比，或 auto"),
        P("MinContrast", "搜尋圖的最低對比，或 auto"),
    ],
    [P("ModelID", "縮放形狀模型")],
    "建立可縮放的形狀模型",
    "每個尺度各建一個形狀模型，最多 8 個。尺度是模板相對原圖的大小：0.5 代表把模板縮小一半再搜尋。"
    "這不是在金字塔裡連續縮放，數值不會和 HALCON 的同一個模型檔相同。",
    "Vizion.create_scaled_shape_model (Template, 2, 0, rad(20), 0.1, 0.9, 1.1, 0.1, 'none', 'use_polarity', 'auto', 'auto', ModelID)",
)
def create_scaled_shape_model(
    template,
    num_levels,
    angle_start,
    angle_extent,
    angle_step,
    scale_min,
    scale_max,
    scale_step,
    optimization,
    metric,
    contrast,
    min_contrast,
    *,
    ctx,
):
    image = expect_image(template, "Template")
    kind = _token(metric, "Metric")
    if kind not in ("use_polarity", "ignore_local_polarity"):
        raise ValueError("Metric 只能是 use_polarity 或 ignore_local_polarity")
    option = _token(optimization, "Optimization")
    allowed = {item.strip("'") for item in _OPTIMIZATION}
    if option not in allowed:
        raise ValueError("Optimization 只能是 " + "、".join(sorted(allowed)))
    model = build_scaled_shape_model(
        image,
        whole(num_levels, "NumLevels"),
        _optional_number(angle_step, "AngleStep") or 0,
        scalar(scale_min, "ScaleMin"),
        scalar(scale_max, "ScaleMax"),
        scalar(scale_step, "ScaleStep"),
        option,
        kind,
        _optional_number(contrast, "Contrast"),
        _optional_number(min_contrast, "MinContrast"),
        scalar(angle_start, "AngleStart"),
        scalar(angle_extent, "AngleExtent"),
    )
    return Result([], [model])


@operator(
    "形狀匹配",
    [P("Image", "要搜尋的影像")],
    [],
    [
        P("ModelID", "縮放形狀模型"),
        P("AngleStart", "起始角，弳度"),
        P("AngleExtent", "角度範圍，弳度"),
        P("ScaleMin", "這次搜尋的最小尺度"),
        P("ScaleMax", "這次搜尋的最大尺度"),
        P("MinScore", "最低分數，0 到 1"),
        P("NumMatches", "最多幾個，0 表示全部"),
        P("MaxOverlap", "允許重疊比例，0 到 1"),
        P("SubPixel", "none、interpolation 或 least_squares", "'interpolation'", _SUBPIXEL),
        P("NumLevels", "使用層數，0 表示模型全部層"),
        P("Greediness", "0 到 1，越大越快"),
    ],
    [
        P("Row", "列"),
        P("Column", "行"),
        P("Angle", "角度，弳度"),
        P("Scale", "匹配到的尺度"),
        P("Score", "分數"),
    ],
    "尋找可縮放的形狀模型",
    "只搜尋建立時算過、而且落在 ScaleMin 到 ScaleMax 裡的那些尺度。interpolation 用分數曲面的拋物線。least_squares 再沿邊緣法線把列、行、角度和尺度一起做最小二乘，尺度可以落在兩個教過的倍率中間。",
    "Vizion.find_scaled_shape_model (Image, ModelID, 0, rad(20), 0.9, 1.1, 0.5, 1, 0.5, 'interpolation', 0, 0.7, Row, Column, Angle, Scale, Score)",
)
def find_scaled_shape_model(
    image,
    model,
    angle_start,
    angle_extent,
    scale_min,
    scale_max,
    min_score,
    num_matches,
    max_overlap,
    subpixel,
    num_levels,
    greediness,
    *,
    ctx,
):
    picture = expect_image(image, "Image")
    if not isinstance(model, ScaledShapeModel):
        raise TypeError("ModelID 必須是 create_scaled_shape_model 建立的模型")
    mode = _token(subpixel, "SubPixel")
    if mode not in ("none", "interpolation", "least_squares"):
        raise ValueError("SubPixel 只能是 none、interpolation 或 least_squares")
    rows, cols, angles, scales, scores = search_scaled_shape_model(
        picture,
        model,
        scalar(angle_start, "AngleStart"),
        scalar(angle_extent, "AngleExtent"),
        scalar(scale_min, "ScaleMin"),
        scalar(scale_max, "ScaleMax"),
        scalar(min_score, "MinScore"),
        whole(num_matches, "NumMatches"),
        scalar(max_overlap, "MaxOverlap"),
        mode,
        whole(num_levels, "NumLevels"),
        scalar(greediness, "Greediness"),
    )
    from ..operators import _with_origin

    rows, cols = _with_origin(rows, cols, model, scales)
    return Result([], [rows, cols, angles, scales, scores])


@operator(
    "形狀匹配",
    [],
    [],
    [P("ModelID", "形狀模型或縮放形狀模型")],
    [
        P("NumLevels", "層數"),
        P("AngleStart", "起始角"),
        P("AngleExtent", "角度範圍"),
        P("AngleStep", "角度步進"),
        P("ScaleMin", "最小尺度。一般形狀模型是 1"),
        P("ScaleMax", "最大尺度。一般形狀模型是 1"),
        P("ScaleStep", "尺度步進。一般形狀模型是 0"),
        P("Metric", "極性"),
        P("MinContrast", "最低對比"),
    ],
    "讀取形狀模型參數",
    "一般形狀模型的尺度是 1。縮放模型回傳建立時實際用的尺度範圍。",
    "Vizion.get_shape_model_params (ModelID, NumLevels, AngleStart, AngleExtent, AngleStep, ScaleMin, ScaleMax, ScaleStep, Metric, MinContrast)",
)
def get_shape_model_params(model, *, ctx):
    shape = _expect_shape(model)
    if isinstance(shape, ScaledShapeModel):
        child = shape.scales[0][1]
        values = [
            shape.num_levels,
            shape.angle_start,
            shape.angle_extent,
            shape.angle_step,
            shape.scale_min,
            shape.scale_max,
            shape.scale_step,
            shape.metric,
            child.min_contrast,
        ]
    else:
        values = [
            shape.num_levels,
            shape.angle_start,
            shape.angle_extent,
            shape.angle_step,
            1.0,
            1.0,
            0.0,
            shape.metric,
            shape.min_contrast,
        ]
    return Result([], values)


@operator(
    "形狀匹配",
    [],
    [],
    [P("ModelID", "形狀模型或縮放形狀模型")],
    [P("Row", "原點列"), P("Column", "原點行")],
    "讀取形狀模型原點",
    "沒有改過時，原點是教導區域的重心；沒有 domain 時是影像中心。",
    "Vizion.get_shape_model_origin (ModelID, Row, Column)",
)
def get_shape_model_origin(model, *, ctx):
    row, column = _origin_of(_expect_shape(model))
    return Result([], [row, column])


@operator(
    "形狀匹配",
    [],
    [],
    [
        P("ModelID", "形狀模型或縮放形狀模型"),
        P("Row", "新的原點列，用模板座標"),
        P("Column", "新的原點行，用模板座標"),
    ],
    [],
    "設定形狀模型原點",
    "之後 find 回傳的是這個原點在搜尋圖上的位置。縮放模型會依匹配到的尺度放大這段偏移。",
    "Vizion.set_shape_model_origin (ModelID, 0, 0)",
)
def set_shape_model_origin(model, row, column, *, ctx):
    shape = _expect_shape(model)
    shape.ref_row = scalar(row, "Row")
    shape.ref_col = scalar(column, "Column")
    return Result([], [])


@operator(
    "形狀匹配",
    [],
    [],
    [P("ModelID", "要清掉的形狀模型")],
    [],
    "清除形狀模型",
    "清掉後不能再搜尋。變數還在。",
    "Vizion.clear_shape_model (ModelID)",
)
def clear_shape_model(model, *, ctx):
    shape = _expect_shape(model)
    shape.cleared = True
    shape.native = None
    if isinstance(shape, ScaledShapeModel):
        for _scale, child in shape.scales:
            child.cleared = True
            child.native = None
    return Result([], [])


@operator(
    "NCC定位",
    [],
    [],
    [P("ModelID", "NCC 模型")],
    [
        P("NumLevels", "層數"),
        P("AngleStart", "起始角"),
        P("AngleExtent", "角度範圍"),
        P("AngleStep", "角度步進"),
        P("Metric", "極性"),
    ],
    "讀取 NCC 模型參數",
    "回傳建立時實際使用的層數、角度和極性。",
    "Vizion.get_ncc_model_params (ModelID, NumLevels, AngleStart, AngleExtent, AngleStep, Metric)",
)
def get_ncc_model_params(model, *, ctx):
    ncc = _expect_ncc(model)
    return Result([], [ncc.num_levels, ncc.angle_start, ncc.angle_extent, ncc.angle_step, ncc.metric])


@operator(
    "NCC定位",
    [],
    [],
    [P("ModelID", "NCC 模型")],
    [P("Row", "原點列"), P("Column", "原點行")],
    "讀取 NCC 模型原點",
    "預設是教導區域的中心。",
    "Vizion.get_ncc_model_origin (ModelID, Row, Column)",
)
def get_ncc_model_origin(model, *, ctx):
    row, column = _origin_of(_expect_ncc(model))
    return Result([], [row, column])


@operator(
    "NCC定位",
    [],
    [],
    [
        P("ModelID", "NCC 模型"),
        P("Row", "新的原點列，用模板座標"),
        P("Column", "新的原點行，用模板座標"),
    ],
    [],
    "設定 NCC 模型原點",
    "之後 find_ncc_model 回傳的是這個原點在搜尋圖上的位置。",
    "Vizion.set_ncc_model_origin (ModelID, 0, 0)",
)
def set_ncc_model_origin(model, row, column, *, ctx):
    ncc = _expect_ncc(model)
    ncc.ref_row = scalar(row, "Row")
    ncc.ref_col = scalar(column, "Column")
    return Result([], [])


@operator(
    "NCC定位",
    [],
    [],
    [P("ModelID", "要清掉的 NCC 模型")],
    [],
    "清除 NCC 模型",
    "清掉後不能再搜尋。變數還在。",
    "Vizion.clear_ncc_model (ModelID)",
)
def clear_ncc_model(model, *, ctx):
    ncc = _expect_ncc(model)
    ncc.cleared = True
    ncc.native = None
    return Result([], [])


def _open_measure(handle):
    if not isinstance(handle, (MeasureRectangle2, MeasureArc)):
        raise TypeError("MeasureHandle 必須是 gen_measure_rectangle2 或 gen_measure_arc 建立的量測")
    if handle.closed:
        raise RuntimeError("量測物件已經 close_measure")
    return handle


def _measure_path(handle):
    if isinstance(handle, MeasureRectangle2):
        row1, col1, row2, col2 = _axis(handle)
        real = math.hypot(col2 - col1, row2 - row1)
        if real < 1e-9:
            raise ValueError("量測長度是 0")
        count = max(int(real / 0.5) + 1, 2)
        blend = np.linspace(0.0, 1.0, count)
        rows = row1 + (row2 - row1) * blend
        cols = col1 + (col2 - col1) * blend
        normal_row = np.full(count, (col2 - col1) / real)
        normal_col = np.full(count, -((row2 - row1) / real))
        return rows, cols, real * blend, normal_row, normal_col, handle.length2, handle.interpolation
    length = abs(handle.radius * handle.angle_extent)
    if length < 1e-9:
        raise ValueError("弧長是 0")
    count = max(int(length / 0.5) + 1, 2)
    blend = np.linspace(0.0, 1.0, count)
    angles = handle.angle_start + handle.angle_extent * blend
    rows = handle.row + handle.radius * np.sin(angles)
    cols = handle.column + handle.radius * np.cos(angles)
    return rows, cols, length * blend, np.sin(angles), np.cos(angles), handle.annulus, handle.interpolation


def _gather(gray: np.ndarray, domain, cols: np.ndarray, rows: np.ndarray, interpolation: str) -> np.ndarray:
    if interpolation == "nearest_neighbor":
        cols = np.rint(cols)
        rows = np.rint(rows)
    height, width = gray.shape
    inside = (cols >= 0) & (rows >= 0) & (cols <= width - 1) & (rows <= height - 1)
    if interpolation == "bicubic":
        values = gauge.sample_bicubic(gray, rows, cols).astype(np.float64, copy=True)
    else:
        x0 = np.floor(cols).astype(np.int32)
        y0 = np.floor(rows).astype(np.int32)
        x0s = np.clip(x0, 0, width - 1)
        y0s = np.clip(y0, 0, height - 1)
        x1 = np.clip(x0s + 1, 0, width - 1)
        y1 = np.clip(y0s + 1, 0, height - 1)
        dx = cols - x0
        dy = rows - y0
        top = gray[y0s, x0s].astype(np.float64) * (1.0 - dx) + gray[y0s, x1].astype(np.float64) * dx
        bottom = gray[y1, x0s].astype(np.float64) * (1.0 - dx) + gray[y1, x1].astype(np.float64) * dx
        values = top * (1.0 - dy) + bottom * dy
    values[~inside] = np.nan
    if domain is not None:
        ix = np.clip(np.rint(cols).astype(np.int32), 0, width - 1)
        iy = np.clip(np.rint(rows).astype(np.int32), 0, height - 1)
        values[domain[iy, ix] <= 0] = np.nan
    return values


def _mean_profile(image, handle):
    _check_size(image, handle.width, handle.height, "量測")
    rows, cols, distances, normal_row, normal_col, thickness, interpolation = _measure_path(handle)
    half = max(0, int(round(thickness)))
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    sample_rows = rows[None, :] + normal_row[None, :] * offsets[:, None]
    sample_cols = cols[None, :] + normal_col[None, :] * offsets[:, None]
    gathered = _gather(image.gray(), image.domain, sample_cols, sample_rows, interpolation)
    finite = np.isfinite(gathered)
    counts = finite.sum(axis=0)
    total = np.where(finite, gathered, 0.0).sum(axis=0)
    profile = np.full(rows.shape, np.nan, np.float64)
    ok = counts > 0
    profile[ok] = total[ok] / counts[ok]
    return profile, rows, cols, distances


def _crossings(profile, threshold, rows, cols, distances):
    if len(profile) < 2:
        return [], [], []
    previous = profile[:-1]
    current = profile[1:]
    finite = np.isfinite(previous) & np.isfinite(current)
    hit = finite & (
        ((previous < threshold) & (current >= threshold)) | ((previous >= threshold) & (current < threshold))
    )
    if not np.any(hit):
        return [], [], []
    delta = current - previous
    blend = np.zeros(previous.shape, np.float64)
    usable = np.abs(delta) > 1e-12
    blend[usable] = (threshold - previous[usable]) / delta[usable]
    blend = np.clip(blend, 0.0, 1.0)
    index = np.flatnonzero(hit)
    blend = blend[index]
    found_rows = rows[index] * (1.0 - blend) + rows[index + 1] * blend
    found_cols = cols[index] * (1.0 - blend) + cols[index + 1] * blend
    found_dist = distances[index] * (1.0 - blend) + distances[index + 1] * blend
    return found_rows.tolist(), found_cols.tolist(), found_dist.tolist()


@operator(
    "量測",
    [],
    [],
    [
        P("Row", "圓心列"),
        P("Column", "圓心行"),
        P("Radius", "半徑"),
        P("AngleStart", "起始角，弳度。0 朝右"),
        P("AngleExtent", "掃過的角度，正值順時針"),
        P("AnnulusRadius", "沿半徑方向的半寬"),
        P("Width", "影像寬"),
        P("Height", "影像高"),
        P("Interpolation", "剖面取樣", "'bilinear'", _INTERP),
    ],
    [P("MeasureHandle", "量測弧")],
    "建立量測弧",
    "角度 0 朝右，正角度順時針。剖面沿弧長每 0.5 像素取一點，AnnulusRadius 是垂直方向要平均的半寬。",
    "Vizion.gen_measure_arc (30, 40, 10, 0, rad(180), 0, 80, 60, 'bilinear', MeasureHandle)",
)
def gen_measure_arc(row, column, radius, angle_start, angle_extent, annulus, width, height, interpolation, *, ctx):
    mode = _token(interpolation, "Interpolation")
    if mode not in ("nearest_neighbor", "bilinear", "bicubic"):
        raise ValueError("Interpolation 只能是 nearest_neighbor、bilinear 或 bicubic")
    rad = scalar(radius, "Radius")
    extent = scalar(angle_extent, "AngleExtent")
    thick = scalar(annulus, "AnnulusRadius")
    if rad <= 0:
        raise ValueError("半徑必須大於 0")
    if abs(extent) < 1e-9:
        raise ValueError("AngleExtent 不能是 0")
    if thick < 0:
        raise ValueError("AnnulusRadius 不能是負的")
    wide, high = whole(width, "Width"), whole(height, "Height")
    if wide < 2 or high < 2:
        raise ValueError("Width 和 Height 至少要 2")
    return Result([], [MeasureArc(scalar(row, "Row"), scalar(column, "Column"), rad, scalar(angle_start, "AngleStart"), extent, thick, wide, high, mode)])


@operator(
    "量測",
    [P("Image", "灰階或彩色影像")],
    [],
    [P("MeasureHandle", "量測矩形或量測弧")],
    [P("GrayValues", "沿量測方向的灰階")],
    "讀出量測剖面",
    "直線或弧都沿中心每 0.5 像素取一點，垂直方向用 AnnulusRadius 或 Length2 平均。彩色先轉成亮度。超出影像的點是空值。",
    "Vizion.measure_projection (Image, MeasureHandle, GrayValues)",
)
def measure_projection(image, handle, *, ctx):
    picture = expect_image(image, "Image")
    profile, _, _, _ = _mean_profile(picture, _open_measure(handle))
    return Result([], [[float(item) for item in profile]])


_THRESH_SELECT = ["'all'", "'first'", "'last'"]


@operator(
    "量測",
    [P("Image", "灰階或彩色影像")],
    [],
    [
        P("MeasureHandle", "量測矩形或量測弧"),
        P("Sigma", "平滑，像素。0 表示不平滑"),
        P("Threshold", "灰階門檻"),
        P("Select", "all、first 或 last", "'all'", _THRESH_SELECT),
    ],
    [P("RowThresh", "過門檻的列"), P("ColumnThresh", "過門檻的行"), P("Distance", "離起點的距離")],
    "剖面灰階過門檻的位置",
    "看的是灰階本身，不是梯度。由小到大或由大到小經過 Threshold 都算。Sigma 為 0 不平滑。找不到時得到空 tuple。",
    "Vizion.measure_thresh (Image, MeasureHandle, 0, 40, 'all', RowThresh, ColumnThresh, Distance)",
)
def measure_thresh(image, handle, sigma, threshold, select, *, ctx):
    picture = expect_image(image, "Image")
    spread = scalar(sigma, "Sigma")
    level = scalar(threshold, "Threshold")
    if spread < 0:
        raise ValueError("Sigma 不能是負的")
    choice = _token(select, "Select")
    if choice not in ("all", "first", "last"):
        raise ValueError("Select 只能是 all、first 或 last")
    profile, rows, cols, distances = _mean_profile(picture, _open_measure(handle))
    smoothed = gauge._smooth(profile, spread, 0.5)
    found_rows, found_cols, found_dist = _crossings(smoothed, level, rows, cols, distances)
    if choice == "first":
        found_rows, found_cols, found_dist = found_rows[:1], found_cols[:1], found_dist[:1]
    elif choice == "last":
        found_rows, found_cols, found_dist = found_rows[-1:], found_cols[-1:], found_dist[-1:]
    return Result([], [found_rows, found_cols, found_dist])


@operator(
    "量測",
    [],
    [],
    [P("MeasureHandle", "量測矩形或量測弧"), P("Row", "新的中心列"), P("Column", "新的中心行")],
    [],
    "移動量測中心",
    "直接改這個量測的中心，方向、長度和半徑不變。原來的變數就是移動後的量測。",
    "Vizion.translate_measure (MeasureHandle, 30, 50)",
)
def translate_measure(handle, row, column, *, ctx):
    measure = _open_measure(handle)
    measure.row = scalar(row, "Row")
    measure.column = scalar(column, "Column")
    return Result([], [])


def _expect_funct(value) -> Funct1D:
    if not isinstance(value, Funct1D):
        raise TypeError("Function 必須是 create_funct_1d_pairs 建立的函數")
    return value


def _funct_numbers(x_value, y_value) -> tuple[np.ndarray, np.ndarray]:
    xs = np.asarray([float(item) for item in as_list(x_value)], np.float64)
    ys = np.asarray([float(item) for item in as_list(y_value)], np.float64)
    if xs.size < 2 or xs.size != ys.size:
        raise ValueError("X 和 Y 至少要兩個點，而且長度要相同")
    if np.any(np.diff(xs) <= 0):
        raise ValueError("X 要嚴格遞增")
    return xs, ys


_BORDERS = ["'constant'", "'zero'", "'mirror'", "'cyclic'", "'error'"]


def _interp_inside(x: np.ndarray, y: np.ndarray, query: np.ndarray) -> np.ndarray:
    if query.size == 0:
        return query
    index = np.clip(np.searchsorted(x, query, side="right") - 1, 0, len(x) - 2)
    left = x[index]
    span = x[index + 1] - left
    blend = (query - left) / span
    return y[index] * (1.0 - blend) + y[index + 1] * blend


def _map_border(x: np.ndarray, query: np.ndarray, border: str) -> np.ndarray:
    span = float(x[-1] - x[0])
    if span <= 0:
        raise ValueError("X 範圍是 0")
    rel = (query - float(x[0])) / span
    if border == "cyclic":
        rel = np.mod(rel, 1.0)
    else:
        rel = np.abs(rel) % 2.0
        rel = np.where(rel > 1.0, 2.0 - rel, rel)
    return float(x[0]) + rel * span


def _funct_at(funct: Funct1D, query, border: str) -> np.ndarray:
    if border not in ("constant", "zero", "mirror", "cyclic", "error"):
        raise ValueError("Border 只能是 constant、zero、mirror、cyclic 或 error")
    x, y = funct.x, funct.y
    asked = np.atleast_1d(np.asarray(query, np.float64))
    below = asked < x[0]
    above = asked > x[-1]
    outside = below | above
    inside = ~outside
    out = np.empty(asked.shape, np.float64)
    if border == "error" and np.any(outside):
        raise ValueError("X 超出函數範圍")
    if border == "zero":
        out[outside] = 0.0
    elif border == "constant":
        out[below] = y[0]
        out[above] = y[-1]
    elif np.any(outside):
        out[outside] = _interp_inside(x, y, _map_border(x, asked[outside], border))
    out[inside] = _interp_inside(x, y, asked[inside])
    return out


@operator(
    "量測",
    [],
    [],
    [P("X", "自變數，要嚴格遞增"), P("Y", "函數值")],
    [P("Function", "一維函數")],
    "用點建立一維函數",
    "X 和 Y 等長，至少兩點，X 要一個比一個大。之後可以用線性內插讀值。",
    "Vizion.create_funct_1d_pairs ([0, 2], [10, 30], Function)",
)
def create_funct_1d_pairs(x_value, y_value, *, ctx):
    xs, ys = _funct_numbers(x_value, y_value)
    return Result([], [Funct1D(xs, ys)])


@operator(
    "量測",
    [],
    [],
    [
        P("Function", "一維函數"),
        P("X", "要讀的自變數，可以是 tuple"),
        P("Border", "超出範圍時 constant、zero、mirror、cyclic 或 error", "'constant'", _BORDERS),
    ],
    [P("Y", "內插後的函數值")],
    "讀一維函數的值",
    "範圍裡用線性內插。constant 沿用端點，zero 在外面是 0，error 會失敗。一個數回傳一個數。",
    "Vizion.get_y_value_funct_1d (Function, 1, 'constant', Y)",
)
def get_y_value_funct_1d(function, x_value, border, *, ctx):
    funct = _expect_funct(function)
    kind = _token(border, "Border")
    values = _funct_at(funct, [float(item) for item in as_list(x_value)], kind)
    if len(values) == 1 and not isinstance(x_value, (list, tuple)):
        return Result([], [float(values[0])])
    return Result([], [values.tolist()])


@operator(
    "量測",
    [],
    [],
    [
        P("Function", "一維函數"),
        P("XMin", "起始 X，含"),
        P("XMax", "結束 X，含"),
        P("XStep", "間隔，必須大於 0"),
        P("Border", "超出範圍時 constant、zero、mirror、cyclic 或 error", "'constant'", _BORDERS),
    ],
    [P("Y", "取樣後的函數值")],
    "等間隔取樣一維函數",
    "從 XMin 走到不超過 XMax，間隔 XStep。結果是 Y 的 tuple。",
    "Vizion.sample_funct_1d (Function, 0, 2, 1, 'constant', Y)",
)
def sample_funct_1d(function, x_min, x_max, x_step, border, *, ctx):
    funct = _expect_funct(function)
    start, stop, step = scalar(x_min, "XMin"), scalar(x_max, "XMax"), scalar(x_step, "XStep")
    if step <= 0:
        raise ValueError("XStep 必須大於 0")
    if stop < start:
        raise ValueError("XMax 不能小於 XMin")
    count = int(math.floor((stop - start) / step + 1e-9)) + 1
    xs = start + np.arange(count, dtype=np.float64) * step
    values = _funct_at(funct, xs, _token(border, "Border"))
    return Result([], [values.tolist()])


@operator(
    "量測",
    [],
    [],
    [P("Function", "一維函數")],
    [P("ZeroCrossings", "函數為 0 的 X")],
    "找一維函數的零點",
    "剛好是 0 的點會留下。兩點之間跨過 0 時，用線性內插。沒有零點時得到空 tuple。",
    "Vizion.zero_crossings_funct_1d (Function, ZeroCrossings)",
)
def zero_crossings_funct_1d(function, *, ctx):
    funct = _expect_funct(function)
    xs, ys = funct.x, funct.y
    hits = []
    for index in range(len(xs) - 1):
        left, right = float(ys[index]), float(ys[index + 1])
        if left == 0.0 and (index == 0 or float(ys[index - 1]) != 0.0):
            hits.append(float(xs[index]))
        if left * right < 0.0:
            blend = -left / (right - left)
            hits.append(float(xs[index] + blend * (xs[index + 1] - xs[index])))
    if float(ys[-1]) == 0.0 and float(ys[-2]) != 0.0:
        hits.append(float(xs[-1]))
    return Result([], [hits])


def _knot_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    slopes = np.diff(y) / np.diff(x)
    out = np.empty(len(x), np.float64)
    out[0] = slopes[0]
    out[-1] = slopes[-1]
    if len(x) > 2:
        out[1:-1] = 0.5 * (slopes[:-1] + slopes[1:])
    return out


@operator(
    "量測",
    [],
    [],
    [P("Y", "函數值，至少兩個")],
    [P("Function", "一維函數")],
    "用一串值建立函數",
    "X 自動是 0、1、2，一直到最後一個點。Y 至少要兩個數。",
    "Vizion.create_funct_1d_array ([4, 8, 6], Function)",
)
def create_funct_1d_array(y_value, *, ctx):
    ys = np.asarray([float(item) for item in as_list(y_value)], np.float64)
    if ys.size < 2:
        raise ValueError("Y 至少要兩個數")
    return Result([], [Funct1D(np.arange(ys.size, dtype=np.float64), ys)])


@operator(
    "量測",
    [],
    [],
    [P("Function", "一維函數")],
    [P("X", "節點的自變數"), P("Y", "節點的函數值")],
    "讀出函數的節點",
    "把建立函數時的點原樣讀出來。X 從 0 開始編號的函數也看得到那些 X。",
    "Vizion.funct_1d_to_pairs (Function, X, Y)",
)
def funct_1d_to_pairs(function, *, ctx):
    funct = _expect_funct(function)
    return Result([], [funct.x.tolist(), funct.y.tolist()])


@operator(
    "量測",
    [],
    [],
    [P("Function", "一維函數"), P("XMin", "積分起點，含"), P("XMax", "積分終點，含")],
    [P("Integral", "梯形積分")],
    "對一維函數積分",
    "把函數看成節點之間的直線，用梯形積分。起點和終點都要落在函數的 X 範圍裡。",
    "Vizion.integrate_funct_1d (Function, 0, 2, Integral)",
)
def integrate_funct_1d(function, x_min, x_max, *, ctx):
    funct = _expect_funct(function)
    start, stop = scalar(x_min, "XMin"), scalar(x_max, "XMax")
    if stop < start:
        raise ValueError("XMax 不能小於 XMin")
    xs, ys = funct.x, funct.y
    if start < float(xs[0]) or stop > float(xs[-1]):
        raise ValueError("積分範圍要在函數的 X 裡面")
    total = 0.0
    for index in range(len(xs) - 1):
        left, right = float(xs[index]), float(xs[index + 1])
        if right <= start or left >= stop:
            continue
        begin, end = max(left, start), min(right, stop)
        span = right - left
        y_begin = float(ys[index]) + (float(ys[index + 1]) - float(ys[index])) * (begin - left) / span
        y_end = float(ys[index]) + (float(ys[index + 1]) - float(ys[index])) * (end - left) / span
        total += (y_begin + y_end) * 0.5 * (end - begin)
    return Result([], [total])


_DERIV = ["'first'", "'second'"]


@operator(
    "量測",
    [],
    [],
    [P("Function", "一維函數"), P("Mode", "first 或 second", "'first'", _DERIV)],
    [P("Derivative", "導數函數")],
    "對一維函數求導",
    "導數記在原來的節點上。端點用單側斜率，中間點用左右兩段斜率的平均。second 是再導一次。節點之間仍是直線，所以中間讀到的不是該段的固定斜率。",
    "Vizion.derivate_funct_1d (Function, 'first', Derivative)",
)
def derivate_funct_1d(function, mode, *, ctx):
    funct = _expect_funct(function)
    kind = _token(mode, "Mode")
    if kind not in ("first", "second"):
        raise ValueError("Mode 只能是 first 或 second")
    slope = _knot_derivative(funct.x, funct.y)
    if kind == "second":
        slope = _knot_derivative(funct.x, slope)
    return Result([], [Funct1D(funct.x.copy(), slope)])


@operator(
    "量測",
    [],
    [],
    [P("Function", "一維函數")],
    [P("FunctionInverted", "反函數")],
    "求一維函數的反函數",
    "Y 必須嚴格遞增或嚴格遞減。反函數的 X 是原來的 Y，Y 是原來的 X。",
    "Vizion.invert_funct_1d (Function, FunctionInverted)",
)
def invert_funct_1d(function, *, ctx):
    funct = _expect_funct(function)
    delta = np.diff(funct.y)
    if not (np.all(delta > 0) or np.all(delta < 0)):
        raise ValueError("Y 必須嚴格遞增或嚴格遞減")
    order = np.argsort(funct.y, kind="stable")
    return Result([], [Funct1D(funct.y[order].copy(), funct.x[order].copy())])


@operator(
    "量測",
    [],
    [],
    [P("Function", "一維函數")],
    [P("MinX", "局部最小的 X"), P("MaxX", "局部最大的 X")],
    "找函數的局部高低點",
    "只看中間的節點，端點不算。要比左右鄰居都低才是最小，都高才是最大。沒有時得到空 tuple。",
    "Vizion.local_min_max_funct_1d (Function, MinX, MaxX)",
)
def local_min_max_funct_1d(function, *, ctx):
    funct = _expect_funct(function)
    xs, ys = funct.x, funct.y
    minima, maxima = [], []
    for index in range(1, len(xs) - 1):
        value = float(ys[index])
        left, right = float(ys[index - 1]), float(ys[index + 1])
        if value < left and value < right:
            minima.append(float(xs[index]))
        elif value > left and value > right:
            maxima.append(float(xs[index]))
    return Result([], [minima, maxima])


@operator(
    "量測",
    [],
    [],
    [P("Function", "一維函數")],
    [P("FunctionAbs", "絕對值函數")],
    "取一維函數的絕對值",
    "每個節點取絕對值。兩點之間跨過 0 時，會補上那個零點，折線才不會把正負直接連起來。",
    "Vizion.abs_funct_1d (Function, FunctionAbs)",
)
def abs_funct_1d(function, *, ctx):
    funct = _expect_funct(function)
    xs, ys = funct.x, funct.y
    out_x = [float(xs[0])]
    out_y = [abs(float(ys[0]))]
    for index in range(len(xs) - 1):
        left, right = float(ys[index]), float(ys[index + 1])
        if left * right < 0.0:
            blend = -left / (right - left)
            out_x.append(float(xs[index] + blend * (xs[index + 1] - xs[index])))
            out_y.append(0.0)
        out_x.append(float(xs[index + 1]))
        out_y.append(abs(right))
    return Result([], [Funct1D(np.asarray(out_x, np.float64), np.asarray(out_y, np.float64))])


@operator(
    "量測",
    [],
    [],
    [
        P("Outer", "外層函數"),
        P("Inner", "內層函數"),
        P("Border", "內層結果超出外層時 constant、zero、mirror、cyclic 或 error", "'constant'", _BORDERS),
    ],
    [P("FunctionComposed", "合成函數")],
    "合成兩個一維函數",
    "先算 Inner，再把結果送進 Outer。新函數的節點沿用 Inner 的 X。Outer 在中間的轉折如果沒落在這些 X 上，不會被保留。",
    "Vizion.compose_funct_1d (Outer, Inner, 'constant', FunctionComposed)",
)
def compose_funct_1d(outer, inner, border, *, ctx):
    outer_fn = _expect_funct(outer)
    inner_fn = _expect_funct(inner)
    values = _funct_at(outer_fn, inner_fn.y, _token(border, "Border"))
    return Result([], [Funct1D(inner_fn.x.copy(), values)])
