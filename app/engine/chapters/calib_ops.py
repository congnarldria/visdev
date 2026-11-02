"""Checkerboard camera calibration through OpenCV.

The calibration object is not an MVTec description file. Descr is three
numbers: inner corner rows, inner corner columns, and the square size.
Camera parameters are fx, fy, cx, cy, k1, k2, p1, p2, k3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..chapters.pose_ops import _angles_gba, _pose_matrix, _pose_numbers
from ..objects import Image, Result
from ..operators import P, as_list, expect_image, operator, scalar, text, whole

_SETUPS = ["'calibration_object'"]
_ITEMS = ["'camera'", "'calib_obj_pose'", "'model'"]
_DATA = ["'params'", "'pose'", "'error'", "'num_poses'"]


@dataclass
class CalibData:
    rows: int | None = None
    cols: int | None = None
    square: float = 1.0
    observations: list = field(default_factory=list)
    width: int = 0
    height: int = 0
    camera: list | None = None
    poses: list = field(default_factory=list)
    error: float | None = None
    closed: bool = False

    def summary(self) -> str:
        if self.closed:
            return "標定  已關閉"
        points = f"{self.rows}x{self.cols}" if self.rows else "尚未設定棋盤"
        return f"標定  {points}  {len(self.observations)} 組"


def _alive(model, name: str) -> CalibData:
    if not isinstance(model, CalibData):
        raise TypeError(f"{name} 必須是 create_calib_data 建立的標定")
    if model.closed:
        raise RuntimeError("標定已經 clear_calib_data")
    return model


def _object_points(model: CalibData) -> np.ndarray:
    if not model.rows or not model.cols:
        raise RuntimeError("請先 set_calib_data_calib_object")
    grid = np.zeros((model.rows * model.cols, 3), np.float64)
    columns = np.tile(np.arange(model.cols), model.rows)
    rows = np.repeat(np.arange(model.rows), model.cols)
    grid[:, 0] = columns * model.square
    grid[:, 1] = rows * model.square
    return grid


def _parse_descr(value) -> tuple[int, int, float]:
    raw = text(value, "CalibObjDescr").replace("x", ",").replace("X", ",").replace("，", ",")
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError("標定物請寫內角點列數,行數,方格邊長，例如 '5,7,10'。這版不讀描述檔。")
    rows = int(round(float(parts[0])))
    cols = int(round(float(parts[1])))
    square = float(parts[2])
    if rows < 2 or cols < 2 or square <= 0:
        raise ValueError("棋盤至少要 2x2 個內角點，邊長要大於 0")
    return rows, cols, square


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
        P("CalibSetup", "這版只做單一棋盤", "'calibration_object'", _SETUPS),
        P("NumCameras", "相機數，這版必須是 1", "1"),
        P("NumCalibObjects", "標定物數，這版必須是 1", "1"),
    ],
    [P("CalibDataID", "標定資料")],
    "建立標定資料",
    "只支援一台相機和一塊棋盤。相機參數是 OpenCV 的 fx、fy、cx、cy、k1、k2、p1、p2、k3，不是 HALCON 的 area_scan_division。",
    "Vizion.create_calib_data ('calibration_object', 1, 1, CalibDataID)",
)
def create_calib_data(setup, num_cameras, num_objects, *, ctx):
    kind = text(setup, "CalibSetup").strip().strip("'\"").lower()
    if kind != "calibration_object":
        raise ValueError("CalibSetup 這版只接受 calibration_object")
    if whole(num_cameras, "NumCameras") != 1 or whole(num_objects, "NumCalibObjects") != 1:
        raise ValueError("這版只能一台相機、一塊標定板")
    return Result([], [CalibData()])


@operator(
    "標定",
    [],
    [],
    [
        P("CalibDataID", "標定資料"),
        P("CalibObjIdx", "標定物編號，從 0 開始", "0"),
        P("CalibObjDescr", "內角點列數,行數,邊長", "'5,7,10'"),
    ],
    [],
    "設定棋盤格",
    "三個數是內角點的列數、行數和方格邊長，例如 '5,7,10'。不是 MVTec 的 .cpd 或 .descr。列數是直向的角點數。",
    "Vizion.set_calib_data_calib_object (CalibDataID, 0, '5,7,10')",
)
def set_calib_data_calib_object(model, index, descr, *, ctx):
    calib = _alive(model, "CalibDataID")
    if whole(index, "CalibObjIdx") != 0:
        raise ValueError("CalibObjIdx 這版只能是 0")
    rows, cols, square = _parse_descr(descr)
    calib.rows = rows
    calib.cols = cols
    calib.square = square
    calib.observations.clear()
    calib.poses.clear()
    calib.camera = None
    calib.error = None
    return Result([], [])


def _store_observation(calib: CalibData, rows: np.ndarray, cols: np.ndarray, width: int, height: int):
    expected = calib.rows * calib.cols
    if len(rows) != expected:
        raise ValueError(f"點數要是 {expected}")
    image_points = np.column_stack([cols, rows]).astype(np.float64)
    calib.observations.append((_object_points(calib), image_points))
    calib.width = max(calib.width, int(width))
    calib.height = max(calib.height, int(height))
    return len(calib.observations) - 1


@operator(
    "標定",
    [P("Image", "含棋盤格的影像")],
    [],
    [
        P("CalibDataID", "標定資料"),
        P("CameraIdx", "相機編號", "0"),
        P("CalibObjIdx", "標定物編號", "0"),
        P("GenParamName", "這版不使用", "''"),
        P("GenParamValue", "這版不使用", "''"),
    ],
    [],
    "在影像裡找棋盤格",
    "用 OpenCV 找內角點。找不到會失敗，已經放進去的影像還留著。同一塊板、同一台相機可以從不同角度再找，每一張都加進同一份標定。calibrate_cameras 至少要三張。",
    "Vizion.find_calib_object (Image, CalibDataID, 0, 0, '', '')",
)
def find_calib_object(image, model, camera_idx, object_idx, _name, _value, *, ctx):
    picture = expect_image(image, "Image")
    calib = _alive(model, "CalibDataID")
    if whole(camera_idx, "CameraIdx") != 0 or whole(object_idx, "CalibObjIdx") != 0:
        raise ValueError("這版的相機和標定物編號都只能是 0")
    _object_points(calib)
    gray = picture.gray()
    pattern = (calib.cols, calib.rows)
    found = False
    corners = None
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
    points = corners.reshape(-1, 2)
    _store_observation(calib, points[:, 1], points[:, 0], picture.width, picture.height)
    return Result([], [])


@operator(
    "標定",
    [],
    [],
    [
        P("CalibDataID", "標定資料"),
        P("CameraIdx", "相機編號", "0"),
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
    if whole(camera_idx, "CameraIdx") != 0 or whole(object_idx, "CalibObjIdx") != 0:
        raise ValueError("這版的相機和標定物編號都只能是 0")
    rows = np.asarray([float(item) for item in as_list(row)], np.float64)
    cols = np.asarray([float(item) for item in as_list(column)], np.float64)
    if len(rows) != len(cols):
        raise ValueError("Row 和 Column 的長度要相同")
    width = int(math.ceil(float(cols.max()) + 1)) if len(cols) else 1
    height = int(math.ceil(float(rows.max()) + 1)) if len(rows) else 1
    index = _store_observation(calib, rows, cols, max(width, 1), max(height, 1))
    return Result([], [index, [0, 0, 0, 0, 0, 0, 0]])


@operator(
    "標定",
    [],
    [],
    [P("CalibDataID", "標定資料")],
    [P("Error", "平均重投影誤差，像素")],
    "標定相機",
    "至少要三組角點。誤差是 OpenCV 的平均重投影誤差。姿態在相機座標：x 向右、y 向下、z 朝前。",
    "Vizion.calibrate_cameras (CalibDataID, Error)",
)
def calibrate_cameras(model, *, ctx):
    calib = _alive(model, "CalibDataID")
    if len(calib.observations) < 3:
        raise RuntimeError("至少要三組標定點")
    object_points = [item[0].reshape(-1, 1, 3).astype(np.float32) for item in calib.observations]
    image_points = [item[1].reshape(-1, 1, 2).astype(np.float32) for item in calib.observations]
    size = (max(calib.width, 2), max(calib.height, 2))
    error, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, size, None, None)
    dist = np.asarray(dist, np.float64).reshape(-1)
    padded = list(dist[:5]) + [0.0] * (5 - len(dist[:5]))
    calib.camera = [
        float(matrix[0, 0]),
        float(matrix[1, 1]),
        float(matrix[0, 2]),
        float(matrix[1, 2]),
        *[float(item) for item in padded],
    ]
    calib.poses = [_pose_from_vectors(rvec, tvec) for rvec, tvec in zip(rvecs, tvecs)]
    calib.error = float(error)
    return Result([], [float(error)])


@operator(
    "標定",
    [],
    [],
    [
        P("CalibDataID", "標定資料"),
        P("ItemType", "camera、calib_obj_pose 或 model", "'camera'", _ITEMS),
        P("ItemIdx", "編號，從 0 開始", "0"),
        P("DataName", "params、pose 或 error", "'params'", _DATA),
    ],
    [P("DataValue", "結果")],
    "讀出標定結果",
    "camera/params 是九個數：fx、fy、cx、cy、k1、k2、p1、p2、k3。calib_obj_pose/pose 是七個數的姿態，編號是放進去的順序。model/error 是重投影誤差。model/num_poses 是已經放進這份標定的影像張數，還沒標定也能讀。",
    "Vizion.get_calib_data (CalibDataID, 'camera', 0, 'params', CamParam)",
)
def get_calib_data(model, item_type, item_idx, data_name, *, ctx):
    calib = _alive(model, "CalibDataID")
    kind = text(item_type, "ItemType").strip().strip("'\"").lower()
    name = text(data_name, "DataName").strip().strip("'\"").lower()
    index = whole(item_idx, "ItemIdx")
    if kind == "camera" and name == "params":
        if calib.camera is None:
            raise RuntimeError("還沒有 calibrate_cameras")
        return Result([], [calib.camera])
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
    calib.poses.clear()
    calib.camera = None
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
