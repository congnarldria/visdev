"""Camera calibration through OpenCV.

A chessboard is three numbers: inner corner rows, columns, and the square size.
A .cpd or .descr file is a HALCON plate description: rows, columns, and the
mark coordinates. Circle plates are found with findCirclesGrid. Each camera
keeps its own views and is calibrated on its own. Parameters are fx, fy, cx,
cy, k1, k2, p1, p2, k3.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..chapters.pose_ops import _angles_gba, _pose_matrix, _pose_numbers
from ..objects import Image, Result, XLD
from ..operators import P, as_list, expect_image, operator, scalar, text, whole

_SETUPS = ["'calibration_object'"]
_ITEMS = ["'camera'", "'calib_obj_pose'", "'model'"]
_DATA = ["'params'", "'pose'", "'error'", "'num_poses'", "'image_size'"]


@dataclass
class PlateDescr:
    kind: str
    rows: int
    cols: int
    square: float
    marks: np.ndarray | None = None
    polarity: str = "light"


@dataclass
class CalibData:
    rows: int | None = None
    cols: int | None = None
    square: float = 1.0
    observations: list = field(default_factory=list)
    observation_cameras: list = field(default_factory=list)
    width: int = 0
    height: int = 0
    camera: list | None = None
    cameras: list | None = None
    poses: list = field(default_factory=list)
    error: float | None = None
    observation_errors: list = field(default_factory=list)
    observation_projected: list = field(default_factory=list)
    observation_images: list = field(default_factory=list)
    closed: bool = False
    num_cameras: int = 1
    plate: str = "chessboard"
    polarity: str = "light"
    marks: np.ndarray | None = None

    def summary(self) -> str:
        if self.closed:
            return "標定  已關閉"
        points = f"{self.rows}x{self.cols}" if self.rows else "尚未設定標定板"
        kind = "圓點" if self.plate == "circles" else "棋盤"
        counted = f"標定  {kind} {points}  {self.num_cameras} 台相機  {len(self.observations)} 組"
        if self.error is None:
            return counted
        return f"{counted}  誤差 {self.error:.3f} 像素"


def _alive(model, name: str) -> CalibData:
    if not isinstance(model, CalibData):
        raise TypeError(f"{name} 必須是 create_calib_data 建立的標定")
    if model.closed:
        raise RuntimeError("標定已經 clear_calib_data")
    return model


def _object_points(model: CalibData) -> np.ndarray:
    if model.marks is not None:
        return model.marks
    if not model.rows or not model.cols:
        raise RuntimeError("請先 set_calib_data_calib_object")
    grid = np.zeros((model.rows * model.cols, 3), np.float64)
    columns = np.tile(np.arange(model.cols), model.rows)
    rows = np.repeat(np.arange(model.rows), model.cols)
    grid[:, 0] = columns * model.square
    grid[:, 1] = rows * model.square
    return grid


def parse_plate_description(source: str) -> PlateDescr:
    """Read a HALCON .cpd or .descr plate file. Mark coordinates stay in meters."""
    rows = None
    cols = None
    zee = 0.0
    polarity = "light"
    marks: list[tuple[float, float, float]] = []
    finder_count = 0
    finders_read = 0
    reading_finders = False
    for line in source.splitlines():
        raw = line.strip()
        if not raw:
            continue
        if raw.startswith("#"):
            if "calibration marks" in raw.lower():
                reading_finders = False
            continue
        parts = raw.split()
        key = parts[0].lower()
        if key == "r" and len(parts) == 2:
            rows = int(float(parts[1]))
            reading_finders = False
            continue
        if key == "c" and len(parts) == 2:
            cols = int(float(parts[1]))
            reading_finders = False
            continue
        if key == "z" and len(parts) == 2:
            zee = float(parts[1])
            reading_finders = False
            continue
        if key == "p" and len(parts) >= 2:
            word = parts[1].lower()
            polarity = "dark" if word.startswith("dark") else "light"
            reading_finders = False
            continue
        if key == "f" and len(parts) == 2:
            finder_count = int(float(parts[1]))
            reading_finders = finder_count > 0
            finders_read = 0
            continue
        if key in ("o", "i", "t"):
            reading_finders = False
            continue
        if reading_finders and len(parts) >= 2 and finders_read < finder_count:
            finders_read += 1
            if finders_read >= finder_count:
                reading_finders = False
            continue
        if len(parts) >= 3:
            marks.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if rows is None or cols is None:
        raise ValueError("描述檔缺少列數或行數")
    if rows < 2 or cols < 2:
        raise ValueError("標定板至少要 2x2 個標記")
    if marks and len(marks) != rows * cols:
        raise ValueError(f"描述檔有 {len(marks)} 個標記，列數乘行數是 {rows * cols}")
    points = None
    square = 1.0
    if marks:
        ordered = sorted(marks, key=lambda item: (-item[1], item[0]))
        points = np.zeros((len(ordered), 3), np.float64)
        for index, (x_value, y_value, _radius) in enumerate(ordered):
            points[index, 0] = x_value
            points[index, 1] = y_value
            points[index, 2] = zee
        gaps = np.diff(points[:cols, 0]) if cols > 1 else np.array([1.0])
        square = float(np.median(np.abs(gaps))) if gaps.size else 1.0
    return PlateDescr("circles", rows, cols, square, points, polarity)


def _looks_like_plate_file(raw: str) -> bool:
    lowered = raw.lower()
    return lowered.endswith(".cpd") or lowered.endswith(".descr") or "/" in raw or "\\" in raw


def _load_plate_file(ctx, raw: str) -> PlateDescr:
    path = Path(raw).expanduser()
    roots = []
    base = getattr(ctx, "root", None)
    if path.is_absolute():
        roots.append(path)
    elif base is not None:
        roots.extend([Path(base) / path, Path(base) / "samples" / path, Path(base) / "workspace" / path])
    else:
        roots.append(path)
    chosen = next((item for item in roots if item.is_file()), None)
    if chosen is None:
        raise ValueError(f"讀不到描述檔 {raw}")
    return parse_plate_description(chosen.read_text(encoding="utf-8", errors="replace"))


def _parse_descr(value, ctx) -> PlateDescr:
    raw = text(value, "CalibObjDescr").strip().strip("'\"")
    if not _looks_like_plate_file(raw):
        numeric = raw.replace("x", ",").replace("X", ",").replace("，", ",")
        parts = [item.strip() for item in numeric.split(",") if item.strip()]
        if len(parts) == 3:
            try:
                rows = int(round(float(parts[0])))
                cols = int(round(float(parts[1])))
                square = float(parts[2])
            except ValueError:
                rows = cols = 0
                square = 0.0
            else:
                if rows < 2 or cols < 2 or square <= 0:
                    raise ValueError("棋盤至少要 2x2 個內角點，邊長要大於 0")
                return PlateDescr("chessboard", rows, cols, square, None, "light")
    return _load_plate_file(ctx, raw)


def _camera_matrix(params: list[float]):
    fx, fy, cx, cy = params[:4]
    dist = np.asarray(params[4:9], np.float64)
    matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    return matrix, dist


def _pose_from_vectors(rvec, tvec) -> list:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    rx, ry, rz = _angles_gba(rotation)
    translation = np.asarray(tvec, np.float64).reshape(3)
    return [float(translation[0]), float(translation[1]), float(translation[2]), float(rx), float(ry), float(rz), 0]


@operator(
    "標定",
    [],
    [],
    [
        P("CalibSetup", "這版只做一塊標定板", "'calibration_object'", _SETUPS),
        P("NumCameras", "相機數。一台填 1。兩台填 2，每張影像的 CameraIdx 再用 0 或 1", "1"),
        P("NumCalibObjects", "標定物數，這版必須是 1", "1"),
    ],
    [P("CalibDataID", "標定資料")],
    "建立標定資料",
    "可以有多台相機，標定物仍是一塊。每台相機用自己的影像各自估計 fx、fy、cx、cy、k1、k2、p1、p2、k3。姿態在該相機的座標裡，不是相機彼此的相對姿態。",
    "Vizion.create_calib_data ('calibration_object', 1, 1, CalibDataID)",
)
def create_calib_data(setup, num_cameras, num_objects, *, ctx):
    kind = text(setup, "CalibSetup").strip().strip("'\"").lower()
    if kind != "calibration_object":
        raise ValueError("CalibSetup 這版只接受 calibration_object")
    cameras = whole(num_cameras, "NumCameras")
    objects = whole(num_objects, "NumCalibObjects")
    if cameras < 1:
        raise ValueError("相機數至少要 1")
    if objects != 1:
        raise ValueError("這版只能一塊標定板")
    return Result([], [CalibData(num_cameras=cameras)])


@operator(
    "標定",
    [],
    [],
    [
        P("CalibDataID", "標定資料"),
        P("CalibObjIdx", "標定物編號，從 0 開始", "0"),
        P("CalibObjDescr", "棋盤寫 '5,7,10'：內角點 5 列、7 行、方格邊長 10。描述檔寫路徑，例如 'samples/calplate_4x5.cpd'", "'5,7,10'", browse="plate"),
    ],
    [],
    "設定標定板",
    "三個數是棋盤內角點的列數、行數和方格邊長，例如 '5,7,10'。副檔名是 .cpd 或 .descr 時讀 HALCON 描述檔：r、c 是標記的列數和行數，後面每行是標記的 x、y、半徑，單位是公尺。圓點板用 findCirclesGrid 來找。",
    "Vizion.set_calib_data_calib_object (CalibDataID, 0, '5,7,10')",
)
def set_calib_data_calib_object(model, index, descr, *, ctx):
    calib = _alive(model, "CalibDataID")
    if whole(index, "CalibObjIdx") != 0:
        raise ValueError("CalibObjIdx 這版只能是 0")
    plate = _parse_descr(descr, ctx)
    calib.rows = plate.rows
    calib.cols = plate.cols
    calib.square = plate.square
    calib.plate = plate.kind
    calib.polarity = plate.polarity
    calib.marks = plate.marks
    calib.observations.clear()
    calib.observation_cameras.clear()
    calib.observation_errors.clear()
    calib.observation_projected.clear()
    calib.observation_images.clear()
    calib.poses.clear()
    calib.camera = None
    calib.cameras = None
    calib.error = None
    calib.width = 0
    calib.height = 0
    return Result([], [])


def _camera_index(calib: CalibData, camera_idx, object_idx) -> int:
    camera = whole(camera_idx, "CameraIdx")
    if camera < 0 or camera >= calib.num_cameras:
        raise ValueError(f"相機編號要在 0 到 {calib.num_cameras - 1}")
    if whole(object_idx, "CalibObjIdx") != 0:
        raise ValueError("CalibObjIdx 這版只能是 0")
    return camera


def _store_observation(calib: CalibData, rows: np.ndarray, cols: np.ndarray, width: int, height: int, camera: int = 0, image: Image | None = None):
    expected = calib.rows * calib.cols
    if len(rows) != expected:
        raise ValueError(f"點數要是 {expected}")
    image_points = np.column_stack([cols, rows]).astype(np.float64)
    stored_width = int(width)
    stored_height = int(height)
    calib.observations.append((_object_points(calib), image_points, stored_width, stored_height))
    calib.observation_cameras.append(int(camera))
    calib.observation_errors.append(None)
    calib.observation_projected.append(None)
    calib.observation_images.append(image)
    calib.width = max(calib.width, stored_width)
    calib.height = max(calib.height, stored_height)
    return len(calib.observations) - 1


@operator(
    "標定",
    [P("Image", "含棋盤格的影像")],
    [],
    [
        P("CalibDataID", "標定資料"),
        P("CameraIdx", "第幾台相機，從 0 開始。只有一台就填 0", "0"),
        P("CalibObjIdx", "標定物編號", "0"),
        P("GenParamName", "這版不使用", "''"),
        P("GenParamValue", "這版不使用", "''"),
    ],
    [],
    "在影像裡找棋盤格",
    "棋盤用內角點，圓點板用 findCirclesGrid。找到的點會畫在這張圖上。找不到會失敗，已經放進去的影像還留著。CameraIdx 是第幾台相機。calibrate_cameras 每一台至少要三張，標完會把每張的點和誤差畫出來。",
    "Vizion.find_calib_object (Image, CalibDataID, 0, 0, '', '')",
)
def find_calib_object(image, model, camera_idx, object_idx, _name, _value, *, ctx):
    picture = expect_image(image, "Image")
    calib = _alive(model, "CalibDataID")
    camera = _camera_index(calib, camera_idx, object_idx)
    _object_points(calib)
    gray = picture.gray()
    if calib.plate == "circles":
        points = _circle_grid(gray, calib.cols, calib.rows, calib.polarity)
    else:
        points = _grid_corners(gray, calib.rows, calib.cols).reshape(-1, 2)
    index = _store_observation(calib, points[:, 1], points[:, 0], picture.width, picture.height, camera, picture)
    _show_found_marks(ctx, calib, index)
    return Result([], [])


@operator(
    "標定",
    [],
    [],
    [
        P("CalibDataID", "標定資料"),
        P("CameraIdx", "第幾台相機，從 0 開始。只有一台就填 0", "0"),
        P("CalibObjIdx", "標定物編號", "0"),
        P("Row", "角點的列"),
        P("Column", "角點的行"),
    ],
    [P("Index", "這一組的編號"), P("Pose", "尚未標定時是 0")],
    "直接寫入一組角點",
    "點數必須等於列數乘行數。順序是列優先：先走完一行的行座標，再換下一列。影像座標的行是 x、列是 y。",
    "Vizion.set_calib_data_observ_points (CalibDataID, 0, 0, Rows, Columns, Index, Pose)",
)
def set_calib_data_observ_points(model, camera_idx, object_idx, row, column, *, ctx):
    calib = _alive(model, "CalibDataID")
    camera = _camera_index(calib, camera_idx, object_idx)
    rows = np.asarray([float(item) for item in as_list(row)], np.float64)
    cols = np.asarray([float(item) for item in as_list(column)], np.float64)
    if len(rows) != len(cols):
        raise ValueError("Row 和 Column 的長度要相同")
    width = int(math.ceil(float(cols.max()) + 1)) if len(cols) else 1
    height = int(math.ceil(float(rows.max()) + 1)) if len(rows) else 1
    index = _store_observation(calib, rows, cols, max(width, 1), max(height, 1), camera)
    return Result([], [index, [0, 0, 0, 0, 0, 0, 0]])


def _fit_camera(observations, width: int, height: int):
    object_points = [item[0].reshape(-1, 1, 3).astype(np.float32) for item in observations]
    image_points = [item[1].reshape(-1, 1, 2).astype(np.float32) for item in observations]
    size = (max(width, 2), max(height, 2))
    error, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, size, None, None)
    dist = np.asarray(dist, np.float64).reshape(-1)
    padded = list(dist[:5]) + [0.0] * (5 - len(dist[:5]))
    params = [
        float(matrix[0, 0]),
        float(matrix[1, 1]),
        float(matrix[0, 2]),
        float(matrix[1, 2]),
        *[float(item) for item in padded],
    ]
    poses = [_pose_from_vectors(rvec, tvec) for rvec, tvec in zip(rvecs, tvecs)]
    views = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, matrix, dist)
        detected = np.asarray(img, np.float64).reshape(-1, 2)
        pred = np.asarray(projected, np.float64).reshape(-1, 2)
        delta = pred - detected
        rms = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
        views.append((rms, pred))
    return float(error), params, poses, views


def _log(ctx, message: str) -> None:
    if ctx is None:
        return
    log = getattr(ctx, "log", None)
    if isinstance(log, list):
        log.append(message)


def _view_label(camera: int, ordinal: int, count: int, error) -> str:
    label = f"相機 {camera} · 第 {ordinal} 組 · {count} 點"
    if error is None:
        return label
    return f"{label} · 誤差 {error:.3f} 像素"


def _error_line(calib: CalibData, owners: list[int]) -> str:
    ordinals: dict[int, int] = defaultdict(int)
    parts = []
    for index, camera in enumerate(owners):
        ordinals[camera] += 1
        error = calib.observation_errors[index] if index < len(calib.observation_errors) else None
        if error is None:
            continue
        parts.append(f"相機 {camera} 第 {ordinals[camera]} 組 {error:.3f}")
    mean = f"{calib.error:.3f}" if calib.error is not None else "?"
    if not parts:
        return f"標定誤差 {mean} 像素"
    return f"標定誤差 {mean} 像素。" + "、".join(parts)


def _cross_contours(points, half: float) -> list:
    contours = []
    for x_value, y_value in np.asarray(points, np.float64).reshape(-1, 2):
        x_value = float(x_value)
        y_value = float(y_value)
        contours.append([[x_value - half, y_value], [x_value + half, y_value]])
        contours.append([[x_value, y_value - half], [x_value, y_value + half]])
    return contours


def _segment_contours(detected, projected) -> list:
    contours = []
    found = np.asarray(detected, np.float64).reshape(-1, 2)
    pred = np.asarray(projected, np.float64).reshape(-1, 2)
    for (x0, y0), (x1, y1) in zip(found, pred):
        contours.append([[float(x0), float(y0)], [float(x1), float(y1)]])
    return contours


def _outlier_mask(detected, projected):
    """Points that sit far from the rest of this view. None when nothing stands out."""
    if projected is None:
        return None
    found = np.asarray(detected, np.float64).reshape(-1, 2)
    pred = np.asarray(projected, np.float64).reshape(-1, 2)
    if len(found) != len(pred) or len(found) < 4:
        return None
    errors = np.hypot(pred[:, 0] - found[:, 0], pred[:, 1] - found[:, 1])
    median = float(np.median(errors))
    limit = max(median * 3.0, median + 1.0)
    mask = errors >= limit - 1e-9
    count = int(np.count_nonzero(mask))
    if count <= 0 or count >= len(errors):
        return None
    return mask


def _with_outlier_note(label: str, detected, projected) -> str:
    mask = _outlier_mask(detected, projected)
    if mask is None:
        return label
    return f"{label} · 紅色 {int(np.count_nonzero(mask))} 點"


def _append_xld(ctx, width: int, height: int, contours: list, color: str) -> None:
    if not contours:
        return
    ctx.display.overlays.append(
        {
            "kind": "xld",
            "xld": XLD(width, height, contours),
            "color": color,
            "colors": None,
            "width": 2,
            "provisional": False,
        }
    )


def _add_mark_overlays(ctx, width: int, height: int, detected, projected, texts: list[tuple[str, float]]) -> None:
    half = max(4.0, round(min(width, height) * 0.012))
    found = np.asarray(detected, np.float64).reshape(-1, 2)
    mask = _outlier_mask(found, projected)
    if mask is None:
        _append_xld(ctx, width, height, _cross_contours(found, half), "#3dff7a")
        if projected is not None:
            _append_xld(ctx, width, height, _segment_contours(found, projected), "#ffb020")
    else:
        keep = np.flatnonzero(~mask)
        bad = np.flatnonzero(mask)
        pred = None if projected is None else np.asarray(projected, np.float64).reshape(-1, 2)
        _append_xld(ctx, width, height, _cross_contours(found[keep], half), "#3dff7a")
        _append_xld(ctx, width, height, _cross_contours(found[bad], half), "#ff3b30")
        if pred is not None:
            _append_xld(ctx, width, height, _segment_contours(found[keep], pred[keep]), "#ffb020")
            _append_xld(ctx, width, height, _segment_contours(found[bad], pred[bad]), "#ff3b30")
    for text_value, row in texts:
        ctx.display.overlays.append(
            {"kind": "text", "text": text_value, "row": row, "col": 8, "color": "#ffe14a"}
        )


def _show_found_marks(ctx, calib: CalibData, index: int) -> None:
    if ctx is None or not callable(getattr(ctx, "display_obj", None)):
        return
    image = calib.observation_images[index] if index < len(calib.observation_images) else None
    if image is None:
        return
    camera = calib.observation_cameras[index] if index < len(calib.observation_cameras) else 0
    ordinal = sum(1 for item in calib.observation_cameras[: index + 1] if item == camera)
    detected = np.asarray(calib.observations[index][1], np.float64).reshape(-1, 2)
    ctx.display_obj(image)
    _add_mark_overlays(
        ctx,
        image.width,
        image.height,
        detected,
        None,
        [(_view_label(int(camera), ordinal, len(detected), None), 18)],
    )


def _paste(canvas: np.ndarray, image: Image, row: int) -> None:
    patch = image.data
    if canvas.ndim == 3 and patch.ndim == 2:
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    elif canvas.ndim == 2 and patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    height, width = patch.shape[:2]
    canvas[row : row + height, :width] = patch


def _stack_panels(panels: list) -> tuple[Image, list, list]:
    band = 28
    gap = 8
    width = max(image.width for _index, image, *_rest in panels)
    height = sum(image.height + band for _index, image, *_rest in panels) + gap * (len(panels) - 1)
    gray = all(image.channels == 1 for _index, image, *_rest in panels)
    canvas = np.full((height, width), 16, np.uint8) if gray else np.full((height, width, 3), 16, np.uint8)
    marks = []
    row = 0
    for index, image, label, detected, projected in panels:
        top = row
        text_row = row + 20
        row += band
        _paste(canvas, image, row)
        bottom = row + image.height
        marks.append((index, label, text_row, detected, projected, row, top, bottom))
        row = bottom + gap
    picture = Image(canvas)
    scale = min(1.0, 1200 / picture.width, 1600 / picture.height)
    if scale < 0.999:
        resized = cv2.resize(
            picture.data,
            (max(1, int(round(picture.width * scale))), max(1, int(round(picture.height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        picture = Image(resized)
    placed = []
    bounds = []
    for index, label, text_row, detected, projected, origin, top, bottom in marks:
        found = np.asarray(detected, np.float64).reshape(-1, 2).copy()
        found[:, 0] *= scale
        found[:, 1] = (found[:, 1] + origin) * scale
        pred = None
        if projected is not None:
            pred = np.asarray(projected, np.float64).reshape(-1, 2).copy()
            pred[:, 0] *= scale
            pred[:, 1] = (pred[:, 1] + origin) * scale
        placed.append((label, text_row * scale, found, pred))
        bounds.append({"index": int(index), "top": float(top * scale), "bottom": float(bottom * scale)})
    return picture, placed, bounds


def _view_parts(calib: CalibData, owners: list[int]) -> list:
    ordinals: dict[int, int] = defaultdict(int)
    panels = []
    for index, camera in enumerate(owners):
        ordinals[int(camera)] += 1
        image = calib.observation_images[index] if index < len(calib.observation_images) else None
        if image is None:
            continue
        detected = np.asarray(calib.observations[index][1], np.float64).reshape(-1, 2)
        projected = calib.observation_projected[index] if index < len(calib.observation_projected) else None
        error = calib.observation_errors[index] if index < len(calib.observation_errors) else None
        label = _with_outlier_note(
            _view_label(int(camera), ordinals[int(camera)], len(detected), error),
            detected,
            projected,
        )
        panels.append((index, image, label, detected, projected))
    return panels


def _remember_board(ctx, calib: CalibData, owners: list[int]) -> None:
    if ctx is None:
        return
    ctx.calib_board = calib
    ctx.calib_owners = [int(item) for item in owners]


def _present_calibration(ctx, calib: CalibData, owners: list[int], *, log: bool = True) -> None:
    if log:
        _log(ctx, _error_line(calib, owners))
    _remember_board(ctx, calib, owners)
    if ctx is None or not callable(getattr(ctx, "display_obj", None)):
        return
    panels = _view_parts(calib, owners)
    if not panels:
        return
    if len(panels) == 1:
        _index, image, label, detected, projected = panels[0]
        texts = []
        if calib.error is not None:
            texts.append((f"平均誤差 {calib.error:.3f} 像素", 18))
            texts.append((label, 38))
        else:
            texts.append((label, 18))
        ctx.display_obj(image)
        _add_mark_overlays(ctx, image.width, image.height, detected, projected, texts)
        return
    picture, placed, bounds = _stack_panels(panels)
    ctx.display_obj(picture)
    for label, row, detected, projected in placed:
        _add_mark_overlays(ctx, picture.width, picture.height, detected, projected, [(label, row)])
    if calib.error is not None and ctx.display.overlays:
        for item in ctx.display.overlays:
            if item.get("kind") == "text":
                item["text"] = f"平均誤差 {calib.error:.3f} 像素 · {item['text']}"
                break
    ctx.display.calib_panels = bounds
    ctx.display.calib_back = False
    ctx.display.calib_note = "點某一張可看原圖、點和誤差。再點一下那張會回到全部。"


def _show_calib_index(ctx, calib: CalibData, owners: list[int], index: int) -> None:
    panels = _view_parts(calib, owners)
    chosen = next((panel for panel in panels if panel[0] == int(index)), None)
    if chosen is None or ctx is None or not callable(getattr(ctx, "display_obj", None)):
        return
    _index, image, label, detected, projected = chosen
    texts = [(label, 18)]
    if calib.error is not None:
        texts = [(f"平均誤差 {calib.error:.3f} 像素", 18), (label, 38)]
    texts.append(("再點一下回到全部", texts[-1][1] + 20))
    ctx.display_obj(image)
    _add_mark_overlays(ctx, image.width, image.height, detected, projected, texts)
    ctx.display.calib_panels = None
    ctx.display.calib_back = True
    ctx.display.calib_note = f"{label}。再點一下回到全部。"


def show_saved_calibration(ctx, index: int | None) -> None:
    calib = getattr(ctx, "calib_board", None)
    owners = getattr(ctx, "calib_owners", None)
    if calib is None or not owners:
        return
    if index is None:
        _present_calibration(ctx, calib, owners, log=False)
        return
    _show_calib_index(ctx, calib, owners, int(index))


def _record_fit(calib: CalibData, indexed: list, width: int, height: int):
    error, params, poses, views = _fit_camera([item[1] for item in indexed], width, height)
    if len(calib.poses) != len(calib.observations):
        calib.poses = [[0, 0, 0, 0, 0, 0, 0] for _ in calib.observations]
    for (index, _observation), pose, (rms, projected) in zip(indexed, poses, views):
        calib.poses[index] = pose
        calib.observation_errors[index] = rms
        calib.observation_projected[index] = projected
    return error, params


@operator(
    "標定",
    [],
    [],
    [P("CalibDataID", "標定資料")],
    [P("Error", "平均重投影誤差，像素")],
    "標定相機",
    "每一台有影像的相機至少要三組點。誤差是各相機重投影誤差的平均。算完會把每張圖的點、重投影位置和這張的誤差畫在圖形視窗。點總覽裡的某一張會改看那張原圖，偏差特別大的點用紅色。變數視窗也看得到平均誤差。姿態在該張影像所屬相機的座標：x 向右、y 向下、z 朝前。",
    "Vizion.calibrate_cameras (CalibDataID, Error)",
)
def calibrate_cameras(model, *, ctx):
    calib = _alive(model, "CalibDataID")
    if len(calib.observations) < 3:
        raise RuntimeError("至少要三組標定點")
    owners = calib.observation_cameras or [0] * len(calib.observations)
    if len(owners) < len(calib.observations):
        owners = owners + [0] * (len(calib.observations) - len(owners))
    grouped: dict[int, list] = defaultdict(list)
    for index, observation in enumerate(calib.observations):
        grouped[int(owners[index])].append((index, observation))
    calib.poses = [[0, 0, 0, 0, 0, 0, 0] for _ in calib.observations]
    if calib.num_cameras == 1:
        indexed = list(enumerate(calib.observations))
        error, params = _record_fit(calib, indexed, calib.width, calib.height)
        calib.camera = params
        calib.cameras = [params]
        calib.error = error
        _present_calibration(ctx, calib, [int(item) for item in owners])
        return Result([], [error])
    cameras: list = [None] * calib.num_cameras
    errors = []
    for camera, items in sorted(grouped.items()):
        if camera < 0 or camera >= calib.num_cameras:
            raise ValueError(f"相機編號要在 0 到 {calib.num_cameras - 1}")
        if len(items) < 3:
            raise RuntimeError(f"相機 {camera} 至少要三組標定點")
        width = max(item[1][2] for item in items)
        height = max(item[1][3] for item in items)
        error, params = _record_fit(calib, items, width, height)
        cameras[camera] = params
        errors.append(error)
    calib.cameras = cameras
    calib.camera = cameras[0]
    calib.error = float(sum(errors) / len(errors))
    _present_calibration(ctx, calib, [int(item) for item in owners])
    return Result([], [calib.error])


@operator(
    "標定",
    [],
    [],
    [
        P("CalibDataID", "標定資料"),
        P("ItemType", "camera、calib_obj_pose 或 model", "'camera'", _ITEMS),
        P("ItemIdx", "camera 時是第幾台相機，從 0 開始，一台就填 0。讀姿態或影像尺寸時是放進去的順序", "0"),
        P("DataName", "params、pose、error、num_poses 或 image_size", "'params'", _DATA),
    ],
    [P("DataValue", "結果")],
    "讀出標定結果",
    "camera/params 仍是九個數：fx、fy、cx、cy、k1、k2、p1、p2、k3，不含影像尺寸。calib_obj_pose/pose 是七個數的姿態，編號是放進去的順序。model/error 是重投影誤差。model/num_poses 是已經放進這份標定的影像張數。model/image_size 是該張影像的寬和高，編號對應放進去的順序，還沒標定也能讀。",
    "Vizion.get_calib_data (CalibDataID, 'camera', 0, 'params', CamParam)",
)
def get_calib_data(model, item_type, item_idx, data_name, *, ctx):
    calib = _alive(model, "CalibDataID")
    kind = text(item_type, "ItemType").strip().strip("'\"").lower()
    name = text(data_name, "DataName").strip().strip("'\"").lower()
    index = whole(item_idx, "ItemIdx")
    if kind == "camera" and name == "params":
        if index < 0 or index >= calib.num_cameras:
            raise ValueError(f"相機編號要在 0 到 {calib.num_cameras - 1}")
        params = None
        if calib.cameras is not None and index < len(calib.cameras):
            params = calib.cameras[index]
        elif index == 0:
            params = calib.camera
        if params is None:
            raise RuntimeError("還沒有 calibrate_cameras")
        return Result([], [params])
    if kind == "calib_obj_pose" and name == "pose":
        if index < 0 or index >= len(calib.poses):
            raise ValueError("這個編號還沒有姿態")
        return Result([], [calib.poses[index]])
    if kind == "model" and name == "error":
        if calib.error is None:
            raise RuntimeError("還沒有 calibrate_cameras")
        return Result([], [calib.error])
    if kind == "model" and name == "num_poses":
        if index != 0:
            raise ValueError("這版的編號只能是 0")
        return Result([], [len(calib.observations)])
    if kind == "model" and name == "image_size":
        if index < 0 or index >= len(calib.observations):
            raise ValueError("這個編號還沒有影像")
        _points, _image, width, height = calib.observations[index]
        return Result([], [[int(width), int(height)]])
    raise ValueError("這組 ItemType 和 DataName 沒有對應的結果")


@operator(
    "標定",
    [],
    [],
    [
        P("CamParam", "fx、fy、cx、cy、k1、k2、p1、p2、k3"),
        P("Pose", "平面在相機座標的姿態"),
        P("Rows", "影像列"),
        P("Cols", "影像行"),
        P("Scale", "世界座標再乘上的比例", "1"),
    ],
    [P("X", "平面上的 x"), P("Y", "平面上的 y")],
    "把影像點投到 Z 為 0 的平面",
    "CamParam 是 OpenCV 的九個數。Pose 用同一套相機座標。平面是標定物的 Z=0。",
    "Vizion.image_points_to_world_plane (CamParam, Pose, Rows, Cols, 1, X, Y)",
)
def image_points_to_world_plane(cam_param, pose, rows, cols, scale, *, ctx):
    nums = [float(item) for item in as_list(cam_param)]
    if len(nums) != 9:
        raise ValueError("CamParam 需要 9 個數：fx、fy、cx、cy、k1、k2、p1、p2、k3")
    matrix, dist = _camera_matrix(nums)
    pose_matrix = _pose_matrix(_pose_numbers(pose))
    rotation = pose_matrix[:3, :3]
    translation = pose_matrix[:3, 3]
    rr = np.asarray([float(item) for item in as_list(rows)], np.float64)
    cc = np.asarray([float(item) for item in as_list(cols)], np.float64)
    if len(rr) != len(cc) or len(rr) == 0:
        raise ValueError("Rows 和 Cols 的長度要相同，而且不能是空的")
    pixels = np.column_stack([cc, rr]).reshape(-1, 1, 2).astype(np.float64)
    normal = cv2.undistortPoints(pixels, matrix, dist).reshape(-1, 2)
    plane = rotation.T[2]
    xs = []
    ys = []
    factor = scalar(scale, "Scale")
    for x_value, y_value in normal:
        direction = np.array([x_value, y_value, 1.0], np.float64)
        denom = float(plane @ direction)
        if abs(denom) < 1e-12:
            raise ValueError("這條視線和平面平行")
        depth = float(plane @ translation) / denom
        world = rotation.T @ (depth * direction - translation)
        xs.append(float(world[0]) * factor)
        ys.append(float(world[1]) * factor)
    def pack(values):
        return values[0] if len(values) == 1 else values
    return Result([], [pack(xs), pack(ys)])


def _nine(value, name: str) -> list[float]:
    nums = [float(item) for item in as_list(value)]
    if len(nums) != 9:
        raise ValueError(f"{name} 需要 9 個數：fx、fy、cx、cy、k1、k2、p1、p2、k3")
    return nums


def _matrices(nums: list[float]):
    matrix = np.array([[nums[0], 0, nums[2]], [0, nums[1], nums[3]], [0, 0, 1]], np.float64)
    distortion = np.asarray(nums[4:9], np.float64)
    return matrix, distortion


@operator(
    "標定",
    [],
    [],
    [
        P("Mode", "fixed 把 k1 設成 Kappa，none 清掉全部畸變", "'none'", ["'none'", "'fixed'"]),
        P("CamParamIn", "原本的九個相機參數"),
        P("Kappa", "fixed 時寫入 k1", "0"),
    ],
    [P("CamParamOut", "新的九個相機參數")],
    "改相機的畸變參數",
    "參數是 OpenCV 的九個數，不是 HALCON 的 division 模型。none 把五個畸變都清成 0。fixed 只留下 k1，其餘畸變是 0。焦距和主點不變。",
    "Vizion.change_radial_distortion_cam_par ('none', CamParamIn, 0, CamParamOut)",
)
def change_radial_distortion_cam_par(mode, cam_param_in, kappa, *, ctx):
    kind = text(mode, "Mode").strip().strip("'\"").lower()
    nums = _nine(cam_param_in, "CamParamIn")
    if kind == "none":
        out = nums[:4] + [0.0, 0.0, 0.0, 0.0, 0.0]
    elif kind == "fixed":
        out = nums[:4] + [scalar(kappa, "Kappa"), 0.0, 0.0, 0.0, 0.0]
    else:
        raise ValueError("Mode 只能是 none 或 fixed")
    return Result([], [out])


@operator(
    "標定",
    [P("Image", "有畸變的影像")],
    [P("ImageRectified", "校正後的影像")],
    [P("CamParamIn", "原本的相機參數"), P("CamParamOut", "希望的相機參數")],
    [],
    "用相機參數校正影像",
    "整張圖都做，沒有 Region 參數。CamParamOut 的畸變通常先用 change_radial_distortion_cam_par 清成 0。邊緣補 0。",
    "Vizion.change_radial_distortion_image (Image, ImageRectified, CamParamIn, CamParamOut)",
)
def change_radial_distortion_image(image, cam_param_in, cam_param_out, *, ctx):
    picture = expect_image(image, "Image")
    camera_in, distortion = _matrices(_nine(cam_param_in, "CamParamIn"))
    camera_out, _ignored = _matrices(_nine(cam_param_out, "CamParamOut"))
    width, height = picture.width, picture.height
    map1, map2 = cv2.initUndistortRectifyMap(camera_in, distortion, None, camera_out, (width, height), cv2.CV_16SC2)
    corrected = cv2.remap(picture.data, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    domain = picture.domain
    if domain is not None:
        moved = cv2.remap(domain, map1, map2, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        domain = np.where(moved > 0, 255, 0).astype(np.uint8)
    return Result([Image(corrected, domain)], [])


@operator(
    "標定",
    [],
    [],
    [P("CalibDataID", "標定資料")],
    [],
    "關閉標定資料",
    "關閉後不能再找角點或讀結果。",
    "Vizion.clear_calib_data (CalibDataID)",
)
def clear_calib_data(model, *, ctx):
    calib = _alive(model, "CalibDataID")
    calib.observations.clear()
    calib.observation_cameras.clear()
    calib.observation_errors.clear()
    calib.observation_projected.clear()
    calib.observation_images.clear()
    calib.poses.clear()
    calib.camera = None
    calib.cameras = None
    calib.width = 0
    calib.height = 0
    calib.closed = True
    return Result([], [])


_GRID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _project_path(ctx, value, label: str) -> Path:
    raw = str(value).strip().strip("'\"")
    if not raw:
        raise ValueError(f"{label} 不能是空的")
    path = Path(raw)
    if not path.is_absolute():
        path = Path(ctx.root) / path
    return path


def _write_cam_par(path: Path, nums: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(" ".join(f"{item:.10g}" for item in nums), encoding="utf-8")


def _read_cam_par(path: Path) -> list[float]:
    if not path.is_file():
        raise ValueError(f"讀不到校正檔 {path}")
    nums = [float(item) for item in path.read_text(encoding="utf-8").split()]
    if len(nums) != 9:
        raise ValueError("校正檔要有 9 個數")
    return nums


def _circle_grid(gray: np.ndarray, cols: int, rows: int, polarity: str) -> np.ndarray:
    pattern = (int(cols), int(rows))
    view = gray if polarity == "dark" else 255 - gray
    found, corners = cv2.findCirclesGrid(view, pattern, flags=cv2.CALIB_CB_SYMMETRIC_GRID)
    if not found or corners is None:
        found, corners = cv2.findCirclesGrid(view, pattern, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
    if not found or corners is None:
        raise RuntimeError("找不到標定標記")
    return np.asarray(corners, np.float64).reshape(-1, 2)


def _grid_corners(gray: np.ndarray, rows: int, cols: int) -> np.ndarray:
    pattern = (cols, rows)
    found, corners = False, None
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, pattern)
    if not found:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not found or corners is None:
        raise RuntimeError("找不到棋盤格")
    corners = np.asarray(corners, np.float32).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    return corners


@operator(
    "標定",
    [P("Image", "一張棋盤照片")],
    [],
    [
        P("CornerRows", "內角點列數"),
        P("CornerCols", "內角點行數"),
        P("Square", "一格的邊長"),
        P("CamParFile", "要寫出的校正檔"),
    ],
    [P("Error", "重投影誤差")],
    "一張棋盤校正畸變",
    "只拍一張。主點固定在影像中心，只估焦距和 k1。校正檔是一行九個數：fx、fy、cx、cy、k1、k2、p1、p2、k3。",
    "Vizion.calibrate_grid (Image, 6, 8, 10, 'workspace/grid_cam.txt', Error)",
    "新算子。一張棋盤照片就寫出校正檔。不是多張標定板。",
)
def calibrate_grid(image, rows, cols, square, cam_file, *, ctx):
    picture = expect_image(image, "Image")
    corner_rows = whole(rows, "CornerRows")
    corner_cols = whole(cols, "CornerCols")
    size = scalar(square, "Square")
    if corner_rows < 2 or corner_cols < 2 or size <= 0:
        raise ValueError("棋盤至少要 2×2 個內角點，格子邊長要大於 0")
    gray = picture.gray()
    corners = _grid_corners(gray, corner_rows, corner_cols)
    height, width = gray.shape[:2]
    focal = float(width)
    camera = np.array(
        [[focal, 0, (width - 1) / 2.0], [0, focal, (height - 1) / 2.0], [0, 0, 1]],
        np.float64,
    )
    distortion = np.zeros((5, 1), np.float64)
    objects = np.zeros((corner_rows * corner_cols, 3), np.float32)
    for row in range(corner_rows):
        for col in range(corner_cols):
            objects[row * corner_cols + col, 0] = col * size
            objects[row * corner_cols + col, 1] = row * size
    flags = (
        cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
    )
    error, camera, distortion, _rvecs, _tvecs = cv2.calibrateCamera(
        [objects],
        [corners],
        (width, height),
        camera,
        distortion,
        flags=flags,
    )
    flat = distortion.reshape(-1)
    nums = [
        float(camera[0, 0]),
        float(camera[1, 1]),
        float(camera[0, 2]),
        float(camera[1, 2]),
        float(flat[0]),
        float(flat[1]),
        float(flat[2]),
        float(flat[3]),
        float(flat[4]),
    ]
    _write_cam_par(_project_path(ctx, cam_file, "CamParFile"), nums)
    return Result([], [float(error)])


@operator(
    "標定",
    [],
    [],
    [
        P("SourceDir", "原始圖資料夾"),
        P("CamParFile", "校正檔"),
        P("DestDir", "校正後要放的資料夾"),
    ],
    [P("Count", "校正了幾張")],
    "用校正檔拉直整個資料夾",
    "只讀資料夾最上一層的 png、jpg、bmp、tif。每張都用同一份校正檔，畸變清成 0 後另存，檔名不變。",
    "Vizion.correct_image_files ('samples/distorted_originals', 'workspace/grid_cam.txt', 'workspace/corrected_parts', Count)",
    "新算子。讀完校正檔，把資料夾裡每張原圖拉直後另存。",
)
def correct_image_files(source_dir, cam_file, dest_dir, *, ctx):
    source = _project_path(ctx, source_dir, "SourceDir")
    if not source.is_dir():
        raise ValueError(f"找不到目錄 {source}")
    matrix, distortion = _matrices(_read_cam_par(_project_path(ctx, cam_file, "CamParFile")))
    dest = _project_path(ctx, dest_dir, "DestDir")
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(source.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _GRID_SUFFIXES:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        corrected = cv2.undistort(image, matrix, distortion)
        if not cv2.imwrite(str(dest / path.name), corrected):
            raise ValueError(f"寫不出 {dest / path.name}")
        count += 1
    return Result([], [count])
