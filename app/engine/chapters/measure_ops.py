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
from ..operators import P, _optional_number, expect_image, operator, scalar, text, whole
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


def _ellipse_points(gray, domain, row, col, phi, radius1, radius2, length1, sigma, threshold, transition):
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
    "nearest_neighbor 會把取樣座標取整。bicubic 這版仍用雙線性。",
    "gen_measure_rectangle2 (120, 100, 0, 40, 5, 640, 480, 'bilinear', MeasureHandle)",
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
    "measure_pos (Image, MeasureHandle, 1, 20, 'all', 'all', RowEdge, ColumnEdge, Amplitude, Distance)",
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
    "measure_pairs (Image, MeasureHandle, 1, 20, 'positive', 'first', Row1, Col1, Amp1, Row2, Col2, Amp2, Intra, Inter)",
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
    "close_measure (MeasureHandle)",
)
def close_measure(handle, *, ctx):
    rect = handle if isinstance(handle, MeasureRectangle2) else None
    if rect is None:
        raise TypeError("MeasureHandle 必須是 gen_measure_rectangle2 建立的量測矩形")
    rect.closed = True
    return Result([], [])


@operator(
    "量測",
    [],
    [],
    [],
    [P("MetrologyHandle", "計量模型")],
    "建立計量模型",
    "模型先是空的。接著設定影像大小，加入直線、圓或橢圓，再 apply。",
    "create_metrology_model (MetrologyHandle)",
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
    "set_metrology_model_image_size (MetrologyHandle, 640, 480)",
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
        P("MeasureLength2", "沿直線的卡尺半寬。這版有存下來，搜尋仍用中心剖面"),
        P("MeasureSigma", "平滑，像素"),
        P("MeasureThreshold", "梯度門檻"),
        P("GenParamName", "這版不使用，留空字串", "''"),
        P("GenParamValue", "這版不使用，留空字串", "''"),
    ],
    [P("Index", "物件索引，從 0 開始")],
    "加入直線計量物件",
    "名義線兩側各搜尋 MeasureLength1。MeasureLength2 不會沿切向平均，這一點和 HALCON 不同。",
    "add_metrology_object_line_measure (MetrologyHandle, 40, 40, 200, 40, 12, 3, 1, 20, Index)",
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
        P("MeasureLength2", "沿切向的卡尺半寬。這版有存下來，搜尋仍用中心剖面"),
        P("MeasureSigma", "平滑，像素"),
        P("MeasureThreshold", "梯度門檻"),
        P("GenParamName", "這版不使用，留空字串", "''"),
        P("GenParamValue", "這版不使用，留空字串", "''"),
    ],
    [P("Index", "物件索引，從 0 開始")],
    "加入圓計量物件",
    "沿名義圓取样，再把找到的邊緣重擬合。MeasureLength2 沒有用來平均切向寬度。",
    "add_metrology_object_circle_measure (MetrologyHandle, 120, 100, 30, 12, 3, 1, 20, Index)",
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
        P("MeasureLength2", "沿切向的卡尺半寬。這版有存下來，搜尋仍用中心射線"),
        P("MeasureSigma", "平滑，像素"),
        P("MeasureThreshold", "梯度門檻"),
        P("GenParamName", "這版不使用，留空字串", "''"),
        P("GenParamValue", "這版不使用，留空字串", "''"),
    ],
    [P("Index", "物件索引，從 0 開始")],
    "加入橢圓計量物件",
    "沿名義橢圓的法向找邊緣，再用這些點擬合。擬合出來的 Phi 可能和輸入差 90 度，因為兩軸可以對調。",
    "add_metrology_object_ellipse_measure (MetrologyHandle, 160, 220, rad(25), 36, 22, 12, 3, 1, 20, Index)",
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
    "apply_metrology_model (Image, MetrologyHandle)",
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
    "get_metrology_object_result (MetrologyHandle, 0, 'all', 'result_type', 'all_param', Parameter)",
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
    "get_metrology_object_result_contour (Contour, MetrologyHandle, 'all', 'all', 1.5)",
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
    "clear_metrology_model (MetrologyHandle)",
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
    "create_scaled_shape_model (Template, 2, 0, rad(20), 0.1, 0.9, 1.1, 0.1, 'none', 'use_polarity', 'auto', 'auto', ModelID)",
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
    "只搜尋建立時算過、而且落在 ScaleMin 到 ScaleMax 裡的那些尺度。least_squares 這版和 interpolation 一樣做分數曲面的拋物線內插。",
    "find_scaled_shape_model (Image, ModelID, 0, rad(20), 0.9, 1.1, 0.5, 1, 0.5, 'interpolation', 0, 0.7, Row, Column, Angle, Scale, Score)",
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
        mode != "none",
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
    "get_shape_model_params (ModelID, NumLevels, AngleStart, AngleExtent, AngleStep, ScaleMin, ScaleMax, ScaleStep, Metric, MinContrast)",
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
    "get_shape_model_origin (ModelID, Row, Column)",
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
    "set_shape_model_origin (ModelID, 0, 0)",
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
    "clear_shape_model (ModelID)",
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
    "get_ncc_model_params (ModelID, NumLevels, AngleStart, AngleExtent, AngleStep, Metric)",
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
    "get_ncc_model_origin (ModelID, Row, Column)",
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
    "set_ncc_model_origin (ModelID, 0, 0)",
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
    "clear_ncc_model (ModelID)",
)
def clear_ncc_model(model, *, ctx):
    ncc = _expect_ncc(model)
    ncc.cleared = True
    ncc.native = None
    return Result([], [])
