"""2D rigid transforms and 3D poses stored as number tuples.

HomMat2D stays six numbers, with x as the row and y as the column.
HomMat3D is twelve numbers, the first three rows of a 3×4 matrix:
[r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz].
A pose is [Tx, Ty, Tz, Rx, Ry, Rz, 0]. Angles are radians.
"""

from __future__ import annotations

import math

import numpy as np

from ..chapters.image_ops import _mat_from, _pack, _rotation, _tidy
from ..objects import Result
from ..operators import P, as_list, operator, scalar

_ORDER = ["'Rp+T'", "'R(p-T)'"]
_ROTATION = ["'gba'"]
_VIEW = ["'point'", "'coordinate_system'"]
_AXIS = ["'x'", "'y'", "'z'"]


def _token(value, name: str) -> str:
    return str(value).strip().strip("'\"")


def _rx(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], np.float64)


def _ry(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], np.float64)


def _rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], np.float64)


def _rotation_gba(rx: float, ry: float, rz: float) -> np.ndarray:
    return _rx(rx) @ _ry(ry) @ _rz(rz)


def _angles_gba(rotation: np.ndarray) -> tuple[float, float, float]:
    ry = math.asin(max(-1.0, min(1.0, float(rotation[0, 2]))))
    cy = math.cos(ry)
    if abs(cy) > 1e-8:
        rx = math.atan2(float(-rotation[1, 2]), float(rotation[2, 2]))
        rz = math.atan2(float(-rotation[0, 1]), float(rotation[0, 0]))
    else:
        rx = 0.0
        rz = math.atan2(float(rotation[1, 0]), float(rotation[1, 1]))
    return rx, ry, rz


def _mat3_from(value) -> np.ndarray:
    nums = [float(item) for item in as_list(value)]
    if len(nums) != 12:
        raise ValueError("HomMat3D 需要 12 個數：每一列四個，共三列")
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, :] = nums[0:4]
    matrix[1, :] = nums[4:8]
    matrix[2, :] = nums[8:12]
    return matrix


def _pack3(matrix: np.ndarray) -> list:
    return [_tidy(float(matrix[row, col])) for row in range(3) for col in range(4)]


def _pose_numbers(value) -> list[float]:
    nums = [float(item) for item in as_list(value)]
    if len(nums) != 7:
        raise ValueError("Pose 需要 7 個數：Tx、Ty、Tz、Rx、Ry、Rz、型別")
    return nums


def _pose_matrix(pose: list[float]) -> np.ndarray:
    code = int(round(pose[6]))
    rotation = _rotation_gba(pose[3], pose[4], pose[5])
    translation = np.array(pose[:3], np.float64)
    if code == 1:
        translation = -rotation @ translation
    elif code == 2:
        translation = -rotation.T @ translation
        rotation = rotation.T
    elif code != 0:
        raise ValueError("Pose 型別只接受 0、1 或 2")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def _check_pose_order(order: str, rotation: str, view: str) -> None:
    if order not in ("Rp+T", "R(p-T)"):
        raise ValueError("OrderOfTransform 只能是 Rp+T 或 R(p-T)")
    if rotation.lower() != "gba":
        raise ValueError("OrderOfRotation 這版只做 gba")
    if view not in ("point", "coordinate_system"):
        raise ValueError("ViewOfTransform 只能是 point 或 coordinate_system")


def _store_pose(order: str, rotation_order: str, view: str, tx, ty, tz, rx, ry, rz) -> list:
    _check_pose_order(order, rotation_order, view)
    rotation = _rotation_gba(rx, ry, rz)
    translation = np.array([tx, ty, tz], np.float64)
    if order == "R(p-T)":
        translation = -rotation @ translation
    if view == "coordinate_system":
        translation = -rotation.T @ translation
        rotation = rotation.T
        rx, ry, rz = _angles_gba(rotation)
    return [_tidy(float(translation[0])), _tidy(float(translation[1])), _tidy(float(translation[2])), _tidy(rx), _tidy(ry), _tidy(rz), 0]


@operator(
    "幾何",
    [],
    [],
    [
        P("Row1", "起點列"),
        P("Column1", "起點行"),
        P("Angle1", "起點方向，弳度"),
        P("Row2", "終點列"),
        P("Column2", "終點行"),
        P("Angle2", "終點方向，弳度"),
    ],
    [P("HomMat2D", "剛性變換")],
    "用兩個點和角度建立剛性變換",
    "把 (Row1, Column1, Angle1) 送到 (Row2, Column2, Angle2)。座標和 HomMat2D 一樣，x 是列、y 是行。正角度是逆時針。",
    "vector_angle_to_rigid (10, 0, 0, 0, 10, rad(90), HomMat2D)",
)
def vector_angle_to_rigid(row1, column1, angle1, row2, column2, angle2, *, ctx):
    r1 = scalar(row1, "Row1")
    c1 = scalar(column1, "Column1")
    r2 = scalar(row2, "Row2")
    c2 = scalar(column2, "Column2")
    delta = scalar(angle2, "Angle2") - scalar(angle1, "Angle1")
    rotation = _rotation(delta)
    tx = r2 - (rotation[0, 0] * r1 + rotation[0, 1] * c1)
    ty = c2 - (rotation[1, 0] * r1 + rotation[1, 1] * c1)
    rotation = rotation.copy()
    rotation[0, 2] = tx
    rotation[1, 2] = ty
    return Result([], [_pack(rotation)])


@operator(
    "幾何",
    [],
    [],
    [
        P("TransX", "平移 x"),
        P("TransY", "平移 y"),
        P("TransZ", "平移 z"),
        P("RotX", "繞 x 的角度，弳度"),
        P("RotY", "繞 y 的角度，弳度"),
        P("RotZ", "繞 z 的角度，弳度"),
        P("OrderOfTransform", "Rp+T 或 R(p-T)", "'Rp+T'", _ORDER),
        P("OrderOfRotation", "這版只做 gba", "'gba'", _ROTATION),
        P("ViewOfTransform", "point 或 coordinate_system", "'point'", _VIEW),
    ],
    [P("Pose", "七個數：Tx Ty Tz Rx Ry Rz 0")],
    "建立姿態",
    "旋轉順序是 Rx×Ry×Rz。存下來的平移一律換成 Rp+T、point，所以型別碼是 0。"
    "R(p-T) 會把平移改成旋轉之後的等效平移。coordinate_system 存的是點變換的反變換。",
    "create_pose (0, 0, 0, 0, 0, rad(90), 'Rp+T', 'gba', 'point', Pose)",
)
def create_pose(tx, ty, tz, rx, ry, rz, order, rotation, view, *, ctx):
    pose = _store_pose(
        _token(order, "OrderOfTransform"),
        _token(rotation, "OrderOfRotation"),
        _token(view, "ViewOfTransform"),
        scalar(tx, "TransX"),
        scalar(ty, "TransY"),
        scalar(tz, "TransZ"),
        scalar(rx, "RotX"),
        scalar(ry, "RotY"),
        scalar(rz, "RotZ"),
    )
    return Result([], [pose])


@operator(
    "幾何",
    [],
    [],
    [
        P("Pose", "原本的姿態"),
        P("OrderOfTransform", "要轉成的平移順序", "'Rp+T'", _ORDER),
        P("OrderOfRotation", "這版只做 gba", "'gba'", _ROTATION),
        P("ViewOfTransform", "point 或 coordinate_system", "'point'", _VIEW),
    ],
    [P("PoseOut", "轉換後的姿態")],
    "轉換姿態的表示法",
    "輸入必須是型別碼 0。Rp+T 加 point 仍是 0。R(p-T) 的型別碼是 1。coordinate_system 的型別碼是 2。"
    "pose_to_hom_mat3d 看得懂這三個型別碼。",
    "convert_pose_type (Pose, 'Rp+T', 'gba', 'point', PoseOut)",
)
def convert_pose_type(pose, order, rotation, view, *, ctx):
    nums = _pose_numbers(pose)
    if int(round(nums[6])) != 0:
        raise ValueError("輸入的 Pose 必須是型別碼 0。請先用 create_pose 正規化。")
    matrix = _pose_matrix(nums)
    kind = _token(order, "OrderOfTransform")
    rotation_order = _token(rotation, "OrderOfRotation")
    which = _token(view, "ViewOfTransform")
    _check_pose_order(kind, rotation_order, which)
    linear = matrix[:3, :3].copy()
    translation = matrix[:3, 3].copy()
    if which == "coordinate_system":
        linear = linear.T
        translation = -linear @ matrix[:3, 3]
    rx, ry, rz = _angles_gba(linear)
    code = 2 if which == "coordinate_system" and kind == "Rp+T" else 0
    if kind == "R(p-T)":
        translation = -linear.T @ translation
        code = 1
    stored = [_tidy(float(item)) for item in translation] + [_tidy(rx), _tidy(ry), _tidy(rz), code]
    return Result([], [stored])


@operator(
    "幾何",
    [],
    [],
    [P("Pose", "姿態")],
    [P("HomMat3D", "12 個數的三維矩陣")],
    "姿態轉成三維矩陣",
    "點的新座標是 R×p+T。十二個數是三列，每列四個。",
    "pose_to_hom_mat3d (Pose, HomMat3D)",
)
def pose_to_hom_mat3d(pose, *, ctx):
    return Result([], [_pack3(_pose_matrix(_pose_numbers(pose)))])


@operator(
    "幾何",
    [],
    [],
    [P("HomMat3D", "三維矩陣")],
    [P("Pose", "姿態")],
    "三維矩陣轉成姿態",
    "只接受剛性變換，旋轉用 gba 拆開。平移是 Rp+T。",
    "hom_mat3d_to_pose (HomMat3D, Pose)",
)
def hom_mat3d_to_pose(hom_mat3d, *, ctx):
    matrix = _mat3_from(hom_mat3d)
    rotation = matrix[:3, :3]
    if abs(np.linalg.det(rotation) - 1) > 1e-4 or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4):
        raise ValueError("這個矩陣不是剛性變換，不能轉成姿態")
    rx, ry, rz = _angles_gba(rotation)
    pose = _store_pose("Rp+T", "gba", "point", float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3]), rx, ry, rz)
    return Result([], [pose])


@operator(
    "幾何",
    [],
    [],
    [P("Pose", "姿態")],
    [P("PoseInvert", "反姿態")],
    "姿態取反",
    "反過來的點變換。仍是型別碼 0。",
    "pose_invert (Pose, PoseInvert)",
)
def pose_invert(pose, *, ctx):
    matrix = np.linalg.inv(_pose_matrix(_pose_numbers(pose)))
    rx, ry, rz = _angles_gba(matrix[:3, :3])
    stored = _store_pose("Rp+T", "gba", "point", float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3]), rx, ry, rz)
    return Result([], [stored])


@operator(
    "幾何",
    [],
    [],
    [P("PoseLeft", "後套用的姿態"), P("PoseRight", "先套用的姿態")],
    [P("PoseCompose", "合成姿態")],
    "合成兩個姿態",
    "先做右邊，再做左邊。",
    "pose_compose (PoseLeft, PoseRight, PoseCompose)",
)
def pose_compose(left, right, *, ctx):
    matrix = _pose_matrix(_pose_numbers(left)) @ _pose_matrix(_pose_numbers(right))
    rx, ry, rz = _angles_gba(matrix[:3, :3])
    stored = _store_pose("Rp+T", "gba", "point", float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3]), rx, ry, rz)
    return Result([], [stored])


@operator(
    "幾何",
    [],
    [],
    [],
    [P("HomMat3D", "單位矩陣")],
    "三維單位矩陣",
    "十二個數。平移是 0，旋轉是單位矩陣。",
    "hom_mat3d_identity (HomMat3D)",
)
def hom_mat3d_identity(*, ctx):
    return Result([], [_pack3(np.eye(4))])


@operator(
    "幾何",
    [],
    [],
    [P("HomMat3D", "原本的矩陣"), P("Tx", "x"), P("Ty", "y"), P("Tz", "z")],
    [P("HomMat3DTranslate", "平移後的矩陣")],
    "在固定座標系加上三維平移",
    "先做原來的矩陣，再平移。",
    "hom_mat3d_translate (HomMat3D, 1, 0, 0, HomMat3DTranslate)",
)
def hom_mat3d_translate(hom_mat3d, tx, ty, tz, *, ctx):
    matrix = _mat3_from(hom_mat3d)
    piece = np.eye(4)
    piece[:3, 3] = [scalar(tx, "Tx"), scalar(ty, "Ty"), scalar(tz, "Tz")]
    return Result([], [_pack3(piece @ matrix)])


@operator(
    "幾何",
    [],
    [],
    [
        P("HomMat3D", "原本的矩陣"),
        P("Phi", "角度，弳度"),
        P("Axis", "旋轉軸", "'z'", _AXIS),
        P("Px", "中心 x"),
        P("Py", "中心 y"),
        P("Pz", "中心 z"),
    ],
    [P("HomMat3DRotate", "旋轉後的矩陣")],
    "繞一軸旋轉",
    "先做原來的矩陣，再繞指定的軸轉。軸是 x、y 或 z。",
    "hom_mat3d_rotate (HomMat3D, rad(90), 'z', 0, 0, 0, HomMat3DRotate)",
)
def hom_mat3d_rotate(hom_mat3d, phi, axis, px, py, pz, *, ctx):
    kind = _token(axis, "Axis").lower()
    angle = scalar(phi, "Phi")
    if kind == "x":
        rotation = _rx(angle)
    elif kind == "y":
        rotation = _ry(angle)
    elif kind == "z":
        rotation = _rz(angle)
    else:
        raise ValueError("Axis 只能是 x、y 或 z")
    center = np.array([scalar(px, "Px"), scalar(py, "Py"), scalar(pz, "Pz")], np.float64)
    piece = np.eye(4)
    piece[:3, :3] = rotation
    piece[:3, 3] = center - rotation @ center
    return Result([], [_pack3(piece @ _mat3_from(hom_mat3d))])


@operator(
    "幾何",
    [],
    [],
    [P("HomMat3DLeft", "後套用的矩陣"), P("HomMat3DRight", "先套用的矩陣")],
    [P("HomMat3DCompose", "合成矩陣")],
    "兩個三維矩陣相乘",
    "先做右邊，再做左邊。",
    "hom_mat3d_compose (HomMat3DLeft, HomMat3DRight, HomMat3DCompose)",
)
def hom_mat3d_compose(left, right, *, ctx):
    return Result([], [_pack3(_mat3_from(left) @ _mat3_from(right))])


@operator(
    "幾何",
    [],
    [],
    [P("HomMat3D", "矩陣")],
    [P("HomMat3DInvert", "反矩陣")],
    "三維矩陣取反",
    "不可逆時會失敗。",
    "hom_mat3d_invert (HomMat3D, HomMat3DInvert)",
)
def hom_mat3d_invert(hom_mat3d, *, ctx):
    try:
        inverse = np.linalg.inv(_mat3_from(hom_mat3d))
    except np.linalg.LinAlgError as exc:
        raise ValueError("這個三維矩陣不能取反") from exc
    return Result([], [_pack3(inverse)])


@operator(
    "幾何",
    [],
    [],
    [P("HomMat3D", "矩陣"), P("Px", "x"), P("Py", "y"), P("Pz", "z")],
    [P("Qx", "轉換後的 x"), P("Qy", "轉換後的 y"), P("Qz", "轉換後的 z")],
    "用三維矩陣轉換點",
    "這是 x、y、z，不是影像的列和行。可以傳 tuple。",
    "affine_trans_point_3d (HomMat3D, 1, 0, 0, Qx, Qy, Qz)",
)
def affine_trans_point_3d(hom_mat3d, px, py, pz, *, ctx):
    matrix = _mat3_from(hom_mat3d)
    xs = as_list(px)
    ys = as_list(py)
    zs = as_list(pz)
    if len(xs) == 1 and max(len(ys), len(zs)) > 1:
        xs = xs * max(len(ys), len(zs))
    if len(ys) == 1 and max(len(xs), len(zs)) > 1:
        ys = ys * max(len(xs), len(zs))
    if len(zs) == 1 and max(len(xs), len(ys)) > 1:
        zs = zs * max(len(xs), len(ys))
    if not (len(xs) == len(ys) == len(zs)):
        raise ValueError("Px、Py、Pz 的長度要相同")
    out = []
    for x, y, z in zip(xs, ys, zs):
        point = matrix @ np.array([float(x), float(y), float(z), 1.0])
        out.append(point[:3] / point[3] if abs(point[3]) > 1e-12 else point[:3])
    qx = [_tidy(float(item[0])) for item in out]
    qy = [_tidy(float(item[1])) for item in out]
    qz = [_tidy(float(item[2])) for item in out]
    if len(qx) == 1 and not isinstance(px, (list, tuple)):
        return Result([], [qx[0], qy[0], qz[0]])
    return Result([], [qx, qy, qz])


def _quat_of(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        nums = [float(item) for item in value.reshape(-1)]
    else:
        nums = [float(item) for item in as_list(value)]
    if len(nums) != 4:
        raise ValueError("Quaternion 需要 4 個數：x、y、z、w")
    return np.asarray(nums, np.float64)


def _unit_quat(value) -> np.ndarray:
    quat = _quat_of(value)
    length = float(np.linalg.norm(quat))
    if length < 1e-12:
        raise ValueError("四元數長度是 0")
    return quat / length


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.array(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        np.float64,
    )


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = _unit_quat(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        np.float64,
    )


def _matrix_to_quat(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
    return _unit_quat([x, y, z, w])


def _quat_list(quat: np.ndarray) -> list:
    unit = _unit_quat(quat)
    return [_tidy(float(item)) for item in unit]


@operator(
    "幾何",
    [],
    [],
    [P("AxisX", "旋轉軸 x"), P("AxisY", "旋轉軸 y"), P("AxisZ", "旋轉軸 z"), P("Angle", "旋轉角，弳度")],
    [P("Quaternion", "四元數 x、y、z、w")],
    "用軸和角度建立四元數",
    "四元數是 [x, y, z, w]。軸不必先歸一化。正角度和 pose 的 gba 一樣。",
    "axis_angle_to_quat (0, 0, 1, rad(90), Quaternion)",
)
def axis_angle_to_quat(axis_x, axis_y, axis_z, angle, *, ctx):
    axis = np.array([scalar(axis_x, "AxisX"), scalar(axis_y, "AxisY"), scalar(axis_z, "AxisZ")], np.float64)
    length = float(np.linalg.norm(axis))
    if length < 1e-12:
        raise ValueError("旋轉軸長度是 0")
    axis /= length
    half = scalar(angle, "Angle") * 0.5
    scale = math.sin(half)
    quat = np.array([axis[0] * scale, axis[1] * scale, axis[2] * scale, math.cos(half)], np.float64)
    return Result([], [_quat_list(quat)])


@operator(
    "幾何",
    [],
    [],
    [P("QuaternionLeft", "左邊的四元數"), P("QuaternionRight", "右邊的四元數")],
    [P("QuaternionComposed", "合成後的四元數")],
    "合成兩個四元數",
    "先套用右邊，再套用左邊，和 pose_compose 一樣。",
    "quat_compose (QuaternionLeft, QuaternionRight, QuaternionComposed)",
)
def quat_compose(left, right, *, ctx):
    return Result([], [_quat_list(_quat_mul(_unit_quat(left), _unit_quat(right)))])


@operator(
    "幾何",
    [],
    [],
    [P("Quaternion", "四元數")],
    [P("Conjugate", "共軛")],
    "四元數共軛",
    "向量部分變號。單位四元數的共軛就是反旋轉。",
    "quat_conjugate (Quaternion, Conjugate)",
)
def quat_conjugate(quaternion, *, ctx):
    quat = _quat_of(quaternion)
    quat[:3] *= -1
    return Result([], [_quat_list(quat)])


@operator(
    "幾何",
    [],
    [],
    [P("Quaternion", "四元數")],
    [P("Normalized", "單位四元數")],
    "把四元數歸一化",
    "長度變成 1。長度是 0 時會失敗。",
    "quat_normalize (Quaternion, Normalized)",
)
def quat_normalize(quaternion, *, ctx):
    return Result([], [_quat_list(_quat_of(quaternion))])


@operator(
    "幾何",
    [],
    [],
    [P("Quaternion", "四元數")],
    [P("Inverse", "反四元數")],
    "四元數取反",
    "用共軛除以長度平方。單位四元數的反元素就是共軛。",
    "quat_invert (Quaternion, Inverse)",
)
def quat_invert(quaternion, *, ctx):
    quat = _quat_of(quaternion)
    length2 = float(np.dot(quat, quat))
    if length2 < 1e-12:
        raise ValueError("四元數長度是 0")
    quat = quat.copy()
    quat[:3] *= -1
    return Result([], [_quat_list(quat / length2)])


@operator(
    "幾何",
    [],
    [],
    [P("Quaternion", "四元數")],
    [P("HomMat3D", "旋轉矩陣，平移是 0")],
    "四元數轉成三維矩陣",
    "只含旋轉，平移是 0。",
    "quat_to_hom_mat3d (Quaternion, HomMat3D)",
)
def quat_to_hom_mat3d(quaternion, *, ctx):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_to_matrix(quaternion)
    return Result([], [_pack3(matrix)])


@operator(
    "幾何",
    [],
    [],
    [P("Pose", "姿態，平移會被忽略")],
    [P("Quaternion", "旋轉四元數")],
    "姿態的旋轉轉成四元數",
    "只用 Rx、Ry、Rz。平移不放進四元數。",
    "pose_to_quat (Pose, Quaternion)",
)
def pose_to_quat(pose, *, ctx):
    rotation = _pose_matrix(_pose_numbers(pose))[:3, :3]
    return Result([], [_quat_list(_matrix_to_quat(rotation))])


@operator(
    "幾何",
    [],
    [],
    [P("Quaternion", "四元數")],
    [P("Pose", "平移為 0 的姿態")],
    "四元數轉成姿態",
    "平移是 0，型別碼是 0，旋轉順序是 gba。",
    "quat_to_pose (Quaternion, Pose)",
)
def quat_to_pose(quaternion, *, ctx):
    rx, ry, rz = _angles_gba(_quat_to_matrix(quaternion))
    return Result([], [[0, 0, 0, _tidy(rx), _tidy(ry), _tidy(rz), 0]])


@operator(
    "幾何",
    [],
    [],
    [P("Quaternion", "四元數"), P("Px", "x"), P("Py", "y"), P("Pz", "z")],
    [P("Qx", "旋轉後的 x"), P("Qy", "旋轉後的 y"), P("Qz", "旋轉後的 z")],
    "用四元數旋轉點",
    "這是 x、y、z，不是影像的列和行。可以傳 tuple。",
    "quat_rotate_point (Quaternion, 1, 0, 0, Qx, Qy, Qz)",
)
def quat_rotate_point(quaternion, px, py, pz, *, ctx):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_to_matrix(quaternion)
    return affine_trans_point_3d( _pack3(matrix), px, py, pz, ctx=ctx)
