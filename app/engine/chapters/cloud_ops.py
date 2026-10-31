"""Point-cloud object models.

Points are Nx3 arrays of x, y, z. This is not a surface model: there is no
mesh, no 3D matching, and no HALCON file format.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import cv2
import numpy as np

from ..chapters.pose_ops import _mat3_from
from ..objects import Image, Region, Result
from ..operators import (
    P,
    as_list,
    expect_image,
    expect_region,
    expect_xld,
    image_out,
    operator,
    region_out,
    scalar,
    whole,
)

_PARAMS = ["'num_points'", "'center'", "'diameter'", "'point_coord_x'", "'point_coord_y'", "'point_coord_z'"]
_AXES = ["'point_coord_x'", "'point_coord_y'", "'point_coord_z'"]


@dataclass
class ObjectModel3D:
    points: np.ndarray
    closed: bool = False

    def summary(self) -> str:
        if self.closed:
            return "三維點雲  已關閉"
        return f"三維點雲  {len(self.points)} 點"


def _alive(model, name: str) -> ObjectModel3D:
    if not isinstance(model, ObjectModel3D):
        raise TypeError(f"{name} 必須是三維點雲")
    if model.closed:
        raise RuntimeError("點雲已經 clear_object_model_3d")
    return model


def _coords(x_value, y_value, z_value) -> np.ndarray:
    xs = [float(item) for item in as_list(x_value)]
    ys = [float(item) for item in as_list(y_value)]
    zs = [float(item) for item in as_list(z_value)]
    count = max(len(xs), len(ys), len(zs))
    if len(xs) == 1 and count > 1:
        xs = xs * count
    if len(ys) == 1 and count > 1:
        ys = ys * count
    if len(zs) == 1 and count > 1:
        zs = zs * count
    if not xs or not (len(xs) == len(ys) == len(zs)):
        raise ValueError("X、Y、Z 的長度要相同，而且不能是空的")
    return np.column_stack([xs, ys, zs]).astype(np.float64)


def _one(values: list):
    return values[0] if len(values) == 1 else values


def _diameter(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    if len(points) <= 800:
        best = 0.0
        for start in range(0, len(points), 200):
            block = points[start : start + 200]
            diff = block[:, None, :] - points[None, :, :]
            best = max(best, float(np.sqrt((diff * diff).sum(axis=2)).max()))
        return best
    span = points.max(axis=0) - points.min(axis=0)
    return float(np.linalg.norm(span))


def _min_distance(left: np.ndarray, right: np.ndarray) -> float:
    best = float("inf")
    for start in range(0, len(left), 256):
        block = left[start : start + 256]
        for other in range(0, len(right), 512):
            diff = block[:, None, :] - right[None, other : other + 512, :]
            best = min(best, float(np.sqrt((diff * diff).sum(axis=2)).min()))
    return best


@operator(
    "三維",
    [],
    [],
    [P("X", "點的 x"), P("Y", "點的 y"), P("Z", "點的 z")],
    [P("ObjectModel3D", "點雲")],
    "用點座標建立點雲",
    "這是點，不是曲面，也沒有網格。X、Y、Z 可以是 tuple。",
    "Vizion.gen_object_model_3d_from_points ([0, 1], [0, 0], [0, 0], ObjectModel3D)",
)
def gen_object_model_3d_from_points(x_value, y_value, z_value, *, ctx):
    return Result([], [ObjectModel3D(_coords(x_value, y_value, z_value))])


@operator(
    "三維",
    [],
    [],
    [
        P("ObjectModel3D", "點雲"),
        P("ParamName", "num_points、center、diameter 或 point_coord_*", "'num_points'", _PARAMS),
    ],
    [P("ParamValue", "結果")],
    "讀取點雲參數",
    "center 是三個數。diameter 在 800 點以內是最遠兩點的距離，更多點時是包圍盒對角線，比較快。",
    "Vizion.get_object_model_3d_params (ObjectModel3D, 'center', Center)",
)
def get_object_model_3d_params(model, param_name, *, ctx):
    cloud = _alive(model, "ObjectModel3D")
    name = str(param_name).strip().strip("'\"").lower()
    points = cloud.points
    if name == "num_points":
        return Result([], [int(len(points))])
    if name == "center":
        center = points.mean(axis=0) if len(points) else np.zeros(3)
        return Result([], [[float(center[0]), float(center[1]), float(center[2])]])
    if name == "diameter":
        return Result([], [_diameter(points)])
    axis = {"point_coord_x": 0, "point_coord_y": 1, "point_coord_z": 2}.get(name)
    if axis is None:
        raise ValueError("ParamName 只能是 num_points、center、diameter 或 point_coord_x/y/z")
    return Result([], [_one([float(item) for item in points[:, axis]])])


@operator(
    "三維",
    [],
    [],
    [P("ObjectModel3D", "點雲"), P("HomMat3D", "三維矩陣")],
    [P("ObjectModel3DAffineTrans", "轉換後的新點雲")],
    "用三維矩陣轉換點雲",
    "原來的點雲不會被改掉。點是 x、y、z。",
    "Vizion.affine_trans_object_model_3d (ObjectModel3D, HomMat3D, ObjectModel3DAffineTrans)",
)
def affine_trans_object_model_3d(model, hom_mat3d, *, ctx):
    cloud = _alive(model, "ObjectModel3D")
    matrix = _mat3_from(hom_mat3d)
    hom = np.concatenate([cloud.points, np.ones((len(cloud.points), 1))], axis=1)
    moved = (matrix @ hom.T).T
    scale = moved[:, 3:4]
    scale[np.abs(scale) < 1e-12] = 1
    return Result([], [ObjectModel3D(moved[:, :3] / scale)])


@operator(
    "三維",
    [],
    [],
    [
        P("ObjectModel3D", "點雲"),
        P("ParamName", "要用哪一軸篩", "'point_coord_z'", _AXES),
        P("MinValue", "下限"),
        P("MaxValue", "上限"),
    ],
    [P("ObjectModel3DThresholded", "留下的點")],
    "按座標留下點",
    "上下限都包含。沒有點落在範圍裡會得到空點雲。",
    "Vizion.select_points_object_model_3d (ObjectModel3D, 'point_coord_z', 0, 10, Kept)",
)
def select_points_object_model_3d(model, param_name, min_value, max_value, *, ctx):
    cloud = _alive(model, "ObjectModel3D")
    name = str(param_name).strip().strip("'\"").lower()
    axis = {"point_coord_x": 0, "point_coord_y": 1, "point_coord_z": 2}.get(name)
    if axis is None:
        raise ValueError("ParamName 只能是 point_coord_x、point_coord_y 或 point_coord_z")
    low = scalar(min_value, "MinValue")
    high = scalar(max_value, "MaxValue")
    column = cloud.points[:, axis]
    kept = cloud.points[(column >= low) & (column <= high)]
    return Result([], [ObjectModel3D(kept.copy())])


@operator(
    "三維",
    [],
    [],
    [P("ObjectModel3D1", "第一個點雲"), P("ObjectModel3D2", "第二個點雲")],
    [P("ObjectModel3DUnion", "合併後的新點雲")],
    "合併兩個點雲",
    "只是把點接在一起，不做法向，也不去掉重複點。",
    "Vizion.union_object_model_3d (ObjectModel3D1, ObjectModel3D2, ObjectModel3DUnion)",
)
def union_object_model_3d(first, second, *, ctx):
    left = _alive(first, "ObjectModel3D1")
    right = _alive(second, "ObjectModel3D2")
    return Result([], [ObjectModel3D(np.vstack([left.points, right.points]))])


@operator(
    "三維",
    [],
    [],
    [P("ObjectModel3D1", "第一個點雲"), P("ObjectModel3D2", "第二個點雲")],
    [P("Distance", "最近兩點的距離")],
    "兩個點雲的最短距離",
    "這是點到點的最短距離，沒有姿態搜尋，也不是曲面距離。",
    "Vizion.distance_object_model_3d (ObjectModel3D1, ObjectModel3D2, Distance)",
)
def distance_object_model_3d(first, second, *, ctx):
    left = _alive(first, "ObjectModel3D1")
    right = _alive(second, "ObjectModel3D2")
    if len(left.points) == 0 or len(right.points) == 0:
        raise ValueError("空點雲沒有距離")
    return Result([], [_min_distance(left.points, right.points)])


@operator(
    "三維",
    [],
    [],
    [P("ObjectModel3D", "點雲")],
    [],
    "關閉點雲",
    "關閉後不能再讀或轉換。",
    "Vizion.clear_object_model_3d (ObjectModel3D)",
)
def clear_object_model_3d(model, *, ctx):
    cloud = _alive(model, "ObjectModel3D")
    cloud.points = np.zeros((0, 3), np.float64)
    cloud.closed = True
    return Result([], [])


def _points(model, name: str) -> np.ndarray:
    cloud = _alive(model, name)
    if len(cloud.points) == 0:
        raise ValueError("點雲是空的")
    return cloud.points


def _low_high(first, second) -> tuple[float, float]:
    start, stop = scalar(first, "Min"), scalar(second, "Max")
    return (start, stop) if start <= stop else (stop, start)


@operator(
    "三維",
    [],
    [],
    [P("X1", "較小或較大的 x"), P("Y1", "y"), P("Z1", "z"), P("X2", "另一個 x"), P("Y2", "y"), P("Z2", "z")],
    [P("ObjectModel3D", "八個角的點雲")],
    "用盒子的兩個角建立點雲",
    "只放八個角，不是填滿的盒子。兩個角誰大誰小都可以，會自動排好。",
    "Vizion.gen_box_object_model_3d (0, 1, 2, 4, 5, 6, ObjectModel3D)",
)
def gen_box_object_model_3d(x1, y1, z1, x2, y2, z2, *, ctx):
    xs, ys, zs = _low_high(x1, x2), _low_high(y1, y2), _low_high(z1, z2)
    corners = np.array([(x, y, z) for x in xs for y in ys for z in zs], np.float64)
    return Result([], [ObjectModel3D(corners)])


@operator(
    "三維",
    [],
    [],
    [P("ObjectModel3D", "點雲")],
    [
        P("MinX", "最小 x"),
        P("MinY", "最小 y"),
        P("MinZ", "最小 z"),
        P("MaxX", "最大 x"),
        P("MaxY", "最大 y"),
        P("MaxZ", "最大 z"),
    ],
    "點雲的軸對齊包圍盒",
    "各軸分開取最小和最大，不是轉正後的最小盒子。空點雲不能算。",
    "Vizion.smallest_bounding_box_object_model_3d (ObjectModel3D, MinX, MinY, MinZ, MaxX, MaxY, MaxZ)",
)
def smallest_bounding_box_object_model_3d(model, *, ctx):
    points = _points(model, "ObjectModel3D")
    low, high = points.min(axis=0), points.max(axis=0)
    return Result([], [float(item) for item in (*low, *high)])


@operator(
    "三維",
    [],
    [],
    [P("ObjectModel3D", "點雲"), P("Tx", "x 位移"), P("Ty", "y 位移"), P("Tz", "z 位移")],
    [P("ObjectModel3DTranslated", "平移後的新點雲")],
    "平移點雲",
    "原來的點雲不會被改掉。",
    "Vizion.translate_object_model_3d (ObjectModel3D, 10, -1, 2, ObjectModel3DTranslated)",
)
def translate_object_model_3d(model, tx, ty, tz, *, ctx):
    cloud = _alive(model, "ObjectModel3D")
    offset = np.array([scalar(tx, "Tx"), scalar(ty, "Ty"), scalar(tz, "Tz")], np.float64)
    return Result([], [ObjectModel3D(cloud.points + offset)])


_MIRROR = ["'x'", "'y'", "'z'"]


@operator(
    "三維",
    [],
    [],
    [P("ObjectModel3D", "點雲"), P("Axis", "要翻轉的軸", "'x'", _MIRROR)],
    [P("ObjectModel3DMirrored", "翻轉後的新點雲")],
    "對一個軸鏡射點雲",
    "x 是把 x 變號，y 和 z 一樣。原來的點雲不會被改掉。",
    "Vizion.mirror_object_model_3d (ObjectModel3D, 'x', ObjectModel3DMirrored)",
)
def mirror_object_model_3d(model, axis, *, ctx):
    cloud = _alive(model, "ObjectModel3D")
    name = str(axis).strip().strip("'\"").lower()
    index = {"x": 0, "y": 1, "z": 2}.get(name)
    if index is None:
        raise ValueError("Axis 只能是 x、y 或 z")
    points = cloud.points.copy()
    points[:, index] *= -1
    return Result([], [ObjectModel3D(points)])


def _plane_svd(points: np.ndarray):
    center = points.mean(axis=0)
    singular, vh = np.linalg.svd(points - center, full_matrices=False)[1:]
    if len(singular) < 2 or singular[-2] <= 1e-8 * max(float(singular[0]), 1.0):
        raise ValueError("這些點排不成一個平面")
    normal = vh[-1].astype(np.float64)
    biggest = int(np.argmax(np.abs(normal)))
    if normal[biggest] < 0:
        normal = -normal
    return normal, float(normal @ center)


def _cloud_plane_error(points: np.ndarray, normal: np.ndarray, dist: float) -> np.ndarray:
    return np.abs(points @ normal - dist)


def _robust_cloud_plane(points: np.ndarray):
    normal, dist = _plane_svd(points)
    if len(points) <= 3:
        return normal, dist
    error = _cloud_plane_error(points, normal, dist)
    if float(np.max(error)) <= 1.0:
        return normal, dist
    count = len(points)
    if count <= 16:
        draws = itertools.combinations(range(count), 3)
    else:
        generator = np.random.default_rng(0)
        draws = (tuple(int(item) for item in generator.choice(count, 3, replace=False)) for _turn in range(80))
    best_normal = normal
    best_dist = dist
    best_error = error
    best_score = float(np.median(error))
    best_total = float(np.sum(error))
    for index in draws:
        chosen = list(index)
        try:
            trial_normal, trial_dist = _plane_svd(points[chosen])
        except ValueError:
            continue
        trial_error = _cloud_plane_error(points, trial_normal, trial_dist)
        score = float(np.median(trial_error))
        total = float(np.sum(trial_error))
        if score < best_score - 1e-9 or (abs(score - best_score) <= 1e-9 and total < best_total - 1e-6):
            best_normal = trial_normal
            best_dist = trial_dist
            best_error = trial_error
            best_score = score
            best_total = total
    limit = 2.5 * best_score if best_score > 0.4 else 1.0
    keep = best_error <= limit + 1e-9
    if int(np.count_nonzero(keep)) < 3 or bool(np.all(keep)):
        return best_normal, best_dist
    return _plane_svd(points[keep])


@operator(
    "三維",
    [],
    [],
    [P("ObjectModel3D", "點雲")],
    [P("Nx", "法向量 x"), P("Ny", "法向量 y"), P("Nz", "法向量 z"), P("Dist", "平面常數")],
    "用點擬合平面",
    "平面是 Nx·x + Ny·y + Nz·z = Dist，法向量長度是 1。點比較多、而且有一個離平面很遠時，會先從三點子集裡挑中位誤差小的平面，再丟掉那個飛點重擬合。剛好三個點，或每個點一開始就在 1 以內時，不會丟。至少要三個不在同一直線上的點。法向量最大的那個分量會取正號。",
    "Vizion.fit_plane_object_model_3d (ObjectModel3D, Nx, Ny, Nz, Dist)",
)
def fit_plane_object_model_3d(model, *, ctx):
    points = _points(model, "ObjectModel3D")
    if len(points) < 3:
        raise ValueError("至少要三個點")
    normal, dist = _robust_cloud_plane(points)
    return Result([], [float(normal[0]), float(normal[1]), float(normal[2]), dist])


@operator(
    "三維",
    [],
    [],
    [
        P("ObjectModel3D", "點雲"),
        P("Nx", "法向量 x"),
        P("Ny", "法向量 y"),
        P("Nz", "法向量 z"),
        P("Dist", "平面常數"),
    ],
    [P("Distances", "每個點的有號距離")],
    "點到平面的距離",
    "距離是 (Nx·x + Ny·y + Nz·z - Dist) 再除以法向量長度，所以法向量不必是單位長。平面法向量的那一側是正的。",
    "Vizion.distance_plane_object_model_3d (ObjectModel3D, 0, 0, 1, 3, Distances)",
)
def distance_plane_object_model_3d(model, nx, ny, nz, dist, *, ctx):
    points = _points(model, "ObjectModel3D")
    normal = np.array([scalar(nx, "Nx"), scalar(ny, "Ny"), scalar(nz, "Nz")], np.float64)
    length = float(np.linalg.norm(normal))
    if length < 1e-12:
        raise ValueError("法向量不能是 0")
    signed = (points @ normal - scalar(dist, "Dist")) / length
    return Result([], [signed.tolist()])


@operator(
    "三維",
    [],
    [],
    [P("ObjectModel3D", "點雲"), P("Step", "每隔幾個點留一個，從第 1 個開始")],
    [P("ObjectModel3DSampled", "留下的新點雲")],
    "按間隔抽點",
    "Step 為 1 會留下全部。Step 為 2 留下第 1、3、5 個點。順序不變，原來的點雲也不改。",
    "Vizion.sample_object_model_3d (ObjectModel3D, 2, ObjectModel3DSampled)",
)
def sample_object_model_3d(model, step, *, ctx):
    cloud = _alive(model, "ObjectModel3D")
    stride = whole(step, "Step")
    if stride < 1:
        raise ValueError("Step 至少要 1")
    return Result([], [ObjectModel3D(cloud.points[::stride].copy())])


@operator(
    "三維",
    [],
    [],
    [P("ObjectModel3D", "點雲"), P("X", "查詢點 x"), P("Y", "查詢點 y"), P("Z", "查詢點 z")],
    [P("Distance", "最近距離")],
    "點到點雲的最近距離",
    "是到最近那個點的直線距離，不是到曲面的距離。空點雲不能算。",
    "Vizion.distance_point_object_model_3d (ObjectModel3D, 0, 4, 0, Distance)",
)
def distance_point_object_model_3d(model, x_value, y_value, z_value, *, ctx):
    points = _points(model, "ObjectModel3D")
    query = np.array([scalar(x_value, "X"), scalar(y_value, "Y"), scalar(z_value, "Z")], np.float64)
    diff = points - query
    return Result([], [float(np.sqrt((diff * diff).sum(axis=1)).min())])


def _axis_index(value, name: str) -> int:
    token = str(value).strip().strip("'\"").lower()
    index = {"x": 0, "y": 1, "z": 2}.get(token)
    if index is None:
        raise ValueError(f"{name} 只能是 x、y 或 z")
    return index


def _copy_cloud(points: np.ndarray) -> ObjectModel3D:
    return ObjectModel3D(np.array(points, np.float64, copy=True))


_AXES_XYZ = ["'x'", "'y'", "'z'"]


@operator(
    "三維",
    [],
    [],
    [
        P("ObjectModel3D", "點雲"),
        P("Sx", "x 倍率"),
        P("Sy", "y 倍率"),
        P("Sz", "z 倍率"),
        P("Cx", "縮放中心 x", "0"),
        P("Cy", "縮放中心 y", "0"),
        P("Cz", "縮放中心 z", "0"),
    ],
    [P("ObjectModel3DScaled", "縮放後的新點雲")],
    "縮放點雲",
    "預設以原點為中心。倍率可以是負的，0 會把該軸壓到中心上。原來的點雲不會被改掉。",
    "Vizion.scale_object_model_3d (ObjectModel3D, 2, 1, 1, ObjectModel3DScaled)",
)
def scale_object_model_3d(model, sx, sy, sz, cx, cy, cz, *, ctx):
    cloud = _alive(model, "ObjectModel3D")
    center = np.array([scalar(cx, "Cx"), scalar(cy, "Cy"), scalar(cz, "Cz")], np.float64)
    gain = np.array([scalar(sx, "Sx"), scalar(sy, "Sy"), scalar(sz, "Sz")], np.float64)
    return Result([], [ObjectModel3D((cloud.points - center) * gain + center)])


@operator(
    "三維",
    [],
    [],
    [
        P("ObjectModel3D", "點雲"),
        P("Axis", "旋轉軸", "'z'", _AXES_XYZ),
        P("Angle", "角度，弳度。正值用右手定則"),
    ],
    [P("ObjectModel3DRotated", "旋轉後的新點雲")],
    "繞軸旋轉點雲",
    "繞過原點。正角度用右手定則，繞 z 時 x 軸轉向 y 軸。原來的點雲不會被改掉。",
    "Vizion.rotate_object_model_3d (ObjectModel3D, 'z', rad(90), ObjectModel3DRotated)",
)
def rotate_object_model_3d(model, axis, angle, *, ctx):
    cloud = _alive(model, "ObjectModel3D")
    index = _axis_index(axis, "Axis")
    turn = scalar(angle, "Angle")
    cosine, sine = float(np.cos(turn)), float(np.sin(turn))
    points = cloud.points.copy()
    first, second = (index + 1) % 3, (index + 2) % 3
    a = points[:, first].copy()
    b = points[:, second].copy()
    points[:, first] = cosine * a - sine * b
    points[:, second] = sine * a + cosine * b
    return Result([], [ObjectModel3D(points)])


_PROJECT = ["'xy'", "'xz'", "'yz'"]


@operator(
    "三維",
    [],
    [],
    [P("ObjectModel3D", "點雲"), P("Plane", "投到哪個座標面", "'xy'", _PROJECT)],
    [P("ObjectModel3DProjected", "投影後的新點雲")],
    "把點投到座標面",
    "xy 把 z 變成 0，xz 把 y 變成 0，yz 把 x 變成 0。原來的點雲不會被改掉。",
    "Vizion.project_object_model_3d (ObjectModel3D, 'xy', ObjectModel3DProjected)",
)
def project_object_model_3d(model, plane, *, ctx):
    cloud = _alive(model, "ObjectModel3D")
    name = str(plane).strip().strip("'\"").lower()
    drop = {"xy": 2, "xz": 1, "yz": 0}.get(name)
    if drop is None:
        raise ValueError("Plane 只能是 xy、xz 或 yz")
    points = cloud.points.copy()
    points[:, drop] = 0
    return Result([], [ObjectModel3D(points)])


@operator(
    "三維",
    [],
    [],
    [
        P("ObjectModel3D", "點雲"),
        P("Nx", "法向量 x"),
        P("Ny", "法向量 y"),
        P("Nz", "法向量 z"),
        P("Dist", "平面常數"),
        P("MaxDist", "兩側都算的最大距離"),
    ],
    [P("ObjectModel3DSelected", "留下的點")],
    "留下靠近平面的點",
    "距離用絕對值，平面兩側都算。法向量會先除以長度。沒有點落在範圍裡會得到空點雲。",
    "Vizion.select_plane_object_model_3d (ObjectModel3D, 0, 0, 1, 3, 0.5, ObjectModel3DSelected)",
)
def select_plane_object_model_3d(model, nx, ny, nz, dist, max_dist, *, ctx):
    cloud = _alive(model, "ObjectModel3D")
    limit = scalar(max_dist, "MaxDist")
    if limit < 0:
        raise ValueError("MaxDist 不能是負的")
    normal = np.array([scalar(nx, "Nx"), scalar(ny, "Ny"), scalar(nz, "Nz")], np.float64)
    length = float(np.linalg.norm(normal))
    if length < 1e-12:
        raise ValueError("法向量不能是 0")
    if len(cloud.points) == 0:
        return Result([], [ObjectModel3D(np.zeros((0, 3), np.float64))])
    signed = np.abs((cloud.points @ normal - scalar(dist, "Dist")) / length)
    return Result([], [ObjectModel3D(cloud.points[signed <= limit].copy())])


@operator(
    "三維",
    [],
    [],
    [P("ObjectModel3D", "點雲"), P("X", "查詢點 x"), P("Y", "查詢點 y"), P("Z", "查詢點 z")],
    [P("Distance", "最遠距離")],
    "點到點雲的最遠距離",
    "是到最遠那個點的直線距離。空點雲不能算。",
    "Vizion.max_distance_point_object_model_3d (ObjectModel3D, 0, 4, 0, Distance)",
)
def max_distance_point_object_model_3d(model, x_value, y_value, z_value, *, ctx):
    points = _points(model, "ObjectModel3D")
    query = np.array([scalar(x_value, "X"), scalar(y_value, "Y"), scalar(z_value, "Z")], np.float64)
    diff = points - query
    return Result([], [float(np.sqrt((diff * diff).sum(axis=1)).max())])


def _near_mask(left: np.ndarray, right: np.ndarray, epsilon: float) -> np.ndarray:
    hit = np.zeros(len(left), dtype=bool)
    limit = epsilon * epsilon
    for start in range(0, len(left), 256):
        block = left[start : start + 256]
        near = np.zeros(len(block), dtype=bool)
        for other in range(0, len(right), 512):
            diff = block[:, None, :] - right[None, other : other + 512, :]
            near |= (diff * diff).sum(axis=2).min(axis=1) <= limit
        hit[start : start + len(block)] = near
    return hit


@operator(
    "三維",
    [],
    [],
    [
        P("ObjectModel3D", "要留下的點雲"),
        P("ObjectModel3DRemove", "要去掉的點"),
        P("Epsilon", "視為同一個點的距離"),
    ],
    [P("ObjectModel3DDifference", "去掉之後的新點雲")],
    "從點雲去掉靠近的點",
    "第一個點雲裡，凡是和第二個點雲距離不大於 Epsilon 的點都會刪掉。Epsilon 為 0 表示座標要完全一樣。順序保留。",
    "Vizion.difference_object_model_3d (ObjectModel3D, ObjectModel3DRemove, 0, ObjectModel3DDifference)",
)
def difference_object_model_3d(model, remove, epsilon, *, ctx):
    left = _alive(model, "ObjectModel3D")
    right = _alive(remove, "ObjectModel3DRemove")
    gap = scalar(epsilon, "Epsilon")
    if gap < 0:
        raise ValueError("Epsilon 不能是負的")
    if len(left.points) == 0 or len(right.points) == 0:
        return Result([], [_copy_cloud(left.points)])
    keep = ~_near_mask(left.points, right.points, gap)
    return Result([], [ObjectModel3D(left.points[keep].copy())])


_SORT_ORDER = ["'ascending'", "'descending'"]


@operator(
    "三維",
    [],
    [],
    [
        P("ObjectModel3D", "點雲"),
        P("Axis", "用哪一軸排序", "'x'", _AXES_XYZ),
        P("Order", "ascending 由小到大，descending 由大到小", "'ascending'", _SORT_ORDER),
    ],
    [P("ObjectModel3DSorted", "排序後的新點雲")],
    "依座標排序點雲",
    "相同座標維持原來的先後。原來的點雲不會被改掉。",
    "Vizion.sort_object_model_3d (ObjectModel3D, 'x', 'ascending', ObjectModel3DSorted)",
)
def sort_object_model_3d(model, axis, order, *, ctx):
    cloud = _alive(model, "ObjectModel3D")
    index = _axis_index(axis, "Axis")
    name = str(order).strip().strip("'\"").lower()
    if name not in ("ascending", "descending"):
        raise ValueError("Order 只能是 ascending 或 descending")
    arranged = np.argsort(cloud.points[:, index], kind="stable")
    if name == "descending":
        arranged = arranged[::-1]
    return Result([], [ObjectModel3D(cloud.points[arranged].copy())])


def _axis_svd(points: np.ndarray):
    center = points.mean(axis=0)
    singular, vh = np.linalg.svd(points - center, full_matrices=False)[1:]
    if float(singular[0]) <= 1e-12:
        raise ValueError("這些點沒有方向")
    axis = vh[0].astype(np.float64).copy()
    biggest = int(np.argmax(np.abs(axis)))
    if axis[biggest] < 0:
        axis = -axis
    return axis, center


def _axis_error(points: np.ndarray, axis: np.ndarray, center: np.ndarray) -> np.ndarray:
    delta = points - center
    parallel = (delta @ axis)[:, None] * axis
    return np.linalg.norm(delta - parallel, axis=1)


def _robust_axis(points: np.ndarray) -> np.ndarray:
    axis, center = _axis_svd(points)
    if len(points) <= 2:
        return axis
    error = _axis_error(points, axis, center)
    if float(np.max(error)) <= 1.0:
        return axis
    count = len(points)
    if count <= 16:
        draws = itertools.combinations(range(count), 2)
    else:
        generator = np.random.default_rng(0)
        draws = (tuple(int(item) for item in generator.choice(count, 2, replace=False)) for _turn in range(80))
    best_axis = axis
    best_error = error
    best_score = float(np.median(error))
    best_total = float(np.sum(error))
    for index in draws:
        chosen = list(index)
        try:
            trial_axis, trial_center = _axis_svd(points[chosen])
        except ValueError:
            continue
        trial_error = _axis_error(points, trial_axis, trial_center)
        score = float(np.median(trial_error))
        total = float(np.sum(trial_error))
        if score < best_score - 1e-9 or (abs(score - best_score) <= 1e-9 and total < best_total - 1e-6):
            best_axis = trial_axis
            best_error = trial_error
            best_score = score
            best_total = total
    limit = 2.5 * best_score if best_score > 0.4 else 1.0
    keep = best_error <= limit + 1e-9
    if int(np.count_nonzero(keep)) < 2 or bool(np.all(keep)):
        return axis
    return _axis_svd(points[keep])[0]


@operator(
    "三維",
    [],
    [],
    [P("ObjectModel3D", "點雲")],
    [P("Ax", "主軸 x"), P("Ay", "主軸 y"), P("Az", "主軸 z")],
    "點雲拉得最長的方向",
    "這是變異數最大的方向，長度是 1。點比較多、而且有一個點離這條軸很遠時，會先從兩點子集裡挑中位誤差小的方向，再丟掉那個點重算。剛好兩個點，或每個點一開始就在 1 以內時，不會丟。最大的那個分量會取正號。至少要兩個不重疊的點。",
    "Vizion.principal_axis_object_model_3d (ObjectModel3D, Ax, Ay, Az)",
)
def principal_axis_object_model_3d(model, *, ctx):
    points = _points(model, "ObjectModel3D")
    if len(points) < 2:
        raise ValueError("至少要兩個點")
    axis = _robust_axis(points)
    return Result([], [float(axis[0]), float(axis[1]), float(axis[2])])


def _empty_cloud() -> ObjectModel3D:
    return ObjectModel3D(np.zeros((0, 3), np.float64))


@operator(
    "三維",
    [],
    [],
    [
        P("ObjectModel3D", "要留下的點雲"),
        P("ObjectModel3DNear", "用來比對的點"),
        P("Epsilon", "視為同一個點的距離"),
    ],
    [P("ObjectModel3DIntersection", "留下的新點雲")],
    "留下兩邊都有的點",
    "第一個點雲裡，和第二個點雲距離不大於 Epsilon 的點會留下。Epsilon 為 0 表示座標要完全一樣。任一邊是空的，結果就是空的。",
    "Vizion.intersect_object_model_3d (ObjectModel3D, ObjectModel3DNear, 0, ObjectModel3DIntersection)",
)
def intersect_object_model_3d(model, other, epsilon, *, ctx):
    left = _alive(model, "ObjectModel3D")
    right = _alive(other, "ObjectModel3DNear")
    gap = scalar(epsilon, "Epsilon")
    if gap < 0:
        raise ValueError("Epsilon 不能是負的")
    if len(left.points) == 0 or len(right.points) == 0:
        return Result([], [_empty_cloud()])
    keep = _near_mask(left.points, right.points, gap)
    return Result([], [ObjectModel3D(left.points[keep].copy())])


@operator(
    "三維",
    [],
    [],
    [P("ObjectModel3D", "點雲")],
    [P("ObjectModel3DCentered", "中心移到原點的新點雲")],
    "把點雲中心移到原點",
    "每個座標都減掉平均值。空點雲不能算。原來的點雲不會被改掉。",
    "Vizion.center_object_model_3d (ObjectModel3D, ObjectModel3DCentered)",
)
def center_object_model_3d(model, *, ctx):
    points = _points(model, "ObjectModel3D")
    return Result([], [ObjectModel3D(points - points.mean(axis=0))])


@operator(
    "三維",
    [],
    [],
    [
        P("ObjectModel3D", "點雲"),
        P("X1", "x 的一端"),
        P("Y1", "y 的一端"),
        P("Z1", "z 的一端"),
        P("X2", "x 的另一端"),
        P("Y2", "y 的另一端"),
        P("Z2", "z 的另一端"),
    ],
    [P("ObjectModel3DClipped", "留下的點")],
    "用盒子裁點雲",
    "六個邊界都包含。兩端誰大誰小都可以。沒有點落在裡面會得到空點雲。",
    "Vizion.clip_object_model_3d (ObjectModel3D, 1, 0, 0, 3, 0, 0, ObjectModel3DClipped)",
)
def clip_object_model_3d(model, x1, y1, z1, x2, y2, z2, *, ctx):
    cloud = _alive(model, "ObjectModel3D")
    bounds = (_low_high(x1, x2), _low_high(y1, y2), _low_high(z1, z2))
    if len(cloud.points) == 0:
        return Result([], [_empty_cloud()])
    keep = np.ones(len(cloud.points), dtype=bool)
    for axis, (low, high) in enumerate(bounds):
        column = cloud.points[:, axis]
        keep &= (column >= low) & (column <= high)
    return Result([], [ObjectModel3D(cloud.points[keep].copy())])


_VOXEL = ["'point'", "'center'"]


@operator(
    "三維",
    [],
    [],
    [
        P("ObjectModel3D", "點雲"),
        P("Size", "格子邊長，必須大於 0"),
        P("Mode", "point 留原來的點，center 改成格子中心", "'point'", _VOXEL),
    ],
    [P("ObjectModel3DReduced", "簡化後的新點雲")],
    "用格子簡化點雲",
    "每個格子只留一個點。point 留最先出現的那個點，座標不變。center 改成該格的中心。格子從 0 往正負兩邊量。",
    "Vizion.voxel_object_model_3d (ObjectModel3D, 2, 'point', ObjectModel3DReduced)",
)
def voxel_object_model_3d(model, size, mode, *, ctx):
    cloud = _alive(model, "ObjectModel3D")
    step = scalar(size, "Size")
    if step <= 0:
        raise ValueError("Size 必須大於 0")
    kind = str(mode).strip().strip("'\"").lower()
    if kind not in ("point", "center"):
        raise ValueError("Mode 只能是 point 或 center")
    if len(cloud.points) == 0:
        return Result([], [_empty_cloud()])
    cells = np.floor(cloud.points / step).astype(np.int64)
    _, first = np.unique(cells, axis=0, return_index=True)
    first.sort()
    if kind == "point":
        kept = cloud.points[first].copy()
    else:
        kept = (cells[first].astype(np.float64) + 0.5) * step
    return Result([], [ObjectModel3D(kept)])


def _farthest_near(left: np.ndarray, right: np.ndarray) -> float:
    best = 0.0
    for start in range(0, len(left), 256):
        block = left[start : start + 256]
        nearest = np.full(len(block), np.inf)
        for other in range(0, len(right), 512):
            diff = block[:, None, :] - right[None, other : other + 512, :]
            nearest = np.minimum(nearest, np.sqrt((diff * diff).sum(axis=2)).min(axis=1))
        best = max(best, float(nearest.max()))
    return best


@operator(
    "三維",
    [],
    [],
    [P("ObjectModel3D", "要量的點雲"), P("ObjectModel3DRef", "參考點雲")],
    [P("Distance", "最遠的那個最近距離")],
    "點雲到另一個點雲的最大間隙",
    "對第一個點雲的每個點，先找它到第二個點雲的最近距離，再取其中最大的。方向不能對調。空點雲不能算。",
    "Vizion.max_distance_object_model_3d (ObjectModel3D, ObjectModel3DRef, Distance)",
)
def max_distance_object_model_3d(model, reference, *, ctx):
    left = _points(model, "ObjectModel3D")
    right = _points(reference, "ObjectModel3DRef")
    return Result([], [_farthest_near(left, right)])


@operator(
    "三維",
    [],
    [],
    [P("ObjectModel3D", "點雲")],
    [P("X", "所有 x"), P("Y", "所有 y"), P("Z", "所有 z")],
    "讀出點雲的座標",
    "一個點時各回傳一個數，多個點時是 tuple。空點雲得到三個空 tuple。",
    "Vizion.get_object_model_3d_points (ObjectModel3D, X, Y, Z)",
)
def get_object_model_3d_points(model, *, ctx):
    cloud = _alive(model, "ObjectModel3D")
    if len(cloud.points) == 0:
        return Result([], [[], [], []])
    axes = [[float(item) for item in cloud.points[:, axis]] for axis in range(3)]
    return Result([], [_one(item) for item in axes])


@operator(
    "三維",
    [],
    [],
    [
        P("X0", "第一點的 x"),
        P("Y0", "第一點的 y"),
        P("Z", "整片的 z"),
        P("Rows", "y 方向的點數"),
        P("Cols", "x 方向的點數"),
        P("Step", "點距，必須大於 0"),
    ],
    [P("ObjectModel3D", "平面上的點")],
    "在平面上產生格子點",
    "z 都一樣。先沿 x 排，再換下一列 y。這是點，不是填滿的面。",
    "Vizion.gen_plane_object_model_3d (0, 0, 3, 2, 2, 1, ObjectModel3D)",
)
def gen_plane_object_model_3d(x0, y0, z_value, rows, cols, step, *, ctx):
    high = whole(rows, "Rows")
    wide = whole(cols, "Cols")
    gap = scalar(step, "Step")
    if high < 1 or wide < 1:
        raise ValueError("Rows 和 Cols 至少要 1")
    if gap <= 0:
        raise ValueError("Step 必須大於 0")
    cc = np.tile(np.arange(wide, dtype=np.float64), high)
    rr = np.repeat(np.arange(high, dtype=np.float64), wide)
    points = np.column_stack(
        [scalar(x0, "X0") + cc * gap, scalar(y0, "Y0") + rr * gap, np.full(high * wide, scalar(z_value, "Z"))]
    )
    return Result([], [ObjectModel3D(points)])


@operator(
    "三維",
    [],
    [],
    [P("ObjectModel3D", "點雲")],
    [P("Spread1", "最長主軸的平方和"), P("Spread2", "第二主軸"), P("Spread3", "最短主軸")],
    "主軸上的平方和",
    "先把中心移到原點，再沿三個主軸算座標平方的總和，由大到小。全部重疊時三個都是 0。至少要兩個點。",
    "Vizion.principal_spread_object_model_3d (ObjectModel3D, Spread1, Spread2, Spread3)",
)
def principal_spread_object_model_3d(model, *, ctx):
    points = _points(model, "ObjectModel3D")
    if len(points) < 2:
        raise ValueError("至少要兩個點")
    centered = points - points.mean(axis=0)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    spread = np.zeros(3, np.float64)
    spread[: len(singular)] = singular ** 2
    return Result([], [float(item) for item in spread])


def _pixel_index(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cols = np.rint(points[:, 0]).astype(np.int64)
    rows = np.rint(points[:, 1]).astype(np.int64)
    return cols, rows


@operator(
    "三維",
    [P("Region", "區域")],
    [],
    [],
    [P("ObjectModel3D", "點雲")],
    "用區域的像素建立點雲",
    "每個前景像素一點。x 是行，y 是列，z 是 0。多個物件會合成一個點雲。沒有像素就得到空點雲。",
    "Vizion.gen_object_model_3d_from_region (Region, ObjectModel3D)",
)
def gen_object_model_3d_from_region(region, *, ctx):
    mask = expect_region(region, "Region").union_mask()
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return Result([], [_empty_cloud()])
    points = np.column_stack([cols, rows, np.zeros(len(rows))]).astype(np.float64)
    return Result([], [ObjectModel3D(points)])


@operator(
    "三維",
    [P("Image", "灰階或彩色影像")],
    [],
    [],
    [P("ObjectModel3D", "點雲")],
    "用影像的像素建立點雲",
    "x 是行，y 是列，z 是灰階。有定義域時只取定義域裡的像素，沒有就整張。彩色先轉成亮度。",
    "Vizion.gen_object_model_3d_from_image (Image, ObjectModel3D)",
)
def gen_object_model_3d_from_image(image, *, ctx):
    picture = expect_image(image, "Image")
    gray = picture.gray()
    if picture.domain is None:
        rows, cols = np.mgrid[0 : picture.height, 0 : picture.width]
        rows, cols = rows.ravel(), cols.ravel()
    else:
        rows, cols = np.nonzero(picture.domain)
    if len(rows) == 0:
        return Result([], [_empty_cloud()])
    points = np.column_stack([cols, rows, gray[rows, cols]]).astype(np.float64)
    return Result([], [ObjectModel3D(points)])


@operator(
    "三維",
    [P("Contours", "輪廓")],
    [],
    [],
    [P("ObjectModel3D", "點雲")],
    "用輪廓的點建立點雲",
    "x 是行，y 是列，z 是 0。多條輪廓依序接在一起。沒有點就得到空點雲。",
    "Vizion.gen_object_model_3d_from_xld (Contours, ObjectModel3D)",
)
def gen_object_model_3d_from_xld(contours, *, ctx):
    xld = expect_xld(contours, "Contours")
    chunks = []
    for contour in xld.contours:
        pts = np.asarray(contour, np.float64).reshape(-1, 2)
        if len(pts) == 0:
            continue
        chunks.append(np.column_stack([pts[:, 0], pts[:, 1], np.zeros(len(pts))]))
    if not chunks:
        return Result([], [_empty_cloud()])
    return Result([], [ObjectModel3D(np.vstack(chunks))])


@operator(
    "三維",
    [],
    [P("Region", "點畫成的區域")],
    [P("ObjectModel3D", "點雲"), P("Width", "區域寬"), P("Height", "區域高")],
    [],
    "把點雲畫成區域",
    "x 四捨五入成行，y 四捨五入成列。超出寬高的點略過，z 不使用。同一個像素只算一次。",
    "Vizion.gen_region_object_model_3d (Region, ObjectModel3D, 80, 60)",
)
def gen_region_object_model_3d(model, width, height, *, ctx):
    cloud = _alive(model, "ObjectModel3D")
    wide, high = whole(width, "Width"), whole(height, "Height")
    if wide < 1 or high < 1:
        raise ValueError("寬和高至少要 1")
    mask = np.zeros((high, wide), np.uint8)
    if len(cloud.points):
        cols, rows = _pixel_index(cloud.points)
        inside = (cols >= 0) & (cols < wide) & (rows >= 0) & (rows < high)
        mask[rows[inside], cols[inside]] = 255
    return region_out(wide, high, mask)


_PAINT = ["'gray'", "'z'"]


@operator(
    "三維",
    [P("Image", "要畫上去的影像")],
    [P("ImageResult", "畫完的新影像")],
    [
        P("ObjectModel3D", "點雲"),
        P("Gray", "gray 模式的灰階", "255"),
        P("Mode", "gray 用固定灰階，z 用點的 z", "'gray'", _PAINT),
    ],
    [],
    "把點雲畫進影像",
    "x 四捨五入成行，y 四捨五入成列。超出影像的點略過。同一個像素後來的點蓋掉前面的。z 模式會把高度裁到 0 到 255。定義域不改。",
    "Vizion.paint_object_model_3d (Image, ImageResult, ObjectModel3D, 255, 'gray')",
)
def paint_object_model_3d(image, model, gray, mode, *, ctx):
    picture = expect_image(image, "Image")
    cloud = _alive(model, "ObjectModel3D")
    kind = str(mode).strip().strip("'\"").lower()
    if kind not in ("gray", "z"):
        raise ValueError("Mode 只能是 gray 或 z")
    data = picture.data.copy()
    if len(cloud.points) == 0:
        return image_out(picture, data)
    cols, rows = _pixel_index(cloud.points)
    inside = (cols >= 0) & (cols < picture.width) & (rows >= 0) & (rows < picture.height)
    if kind == "gray":
        values = np.full(len(cloud.points), int(np.clip(round(scalar(gray, "Gray")), 0, 255)))
    else:
        values = np.clip(np.round(cloud.points[:, 2]), 0, 255).astype(np.int64)
    data[rows[inside], cols[inside]] = values[inside]
    return image_out(picture, data)


@operator(
    "三維",
    [P("Image", "灰階或彩色影像")],
    [],
    [P("ObjectModel3D", "點雲")],
    [P("Grayval", "每個點讀到的灰階")],
    "在點的位置讀灰階",
    "x 四捨五入成行，y 四捨五入成列。有一個點在影像外面就失敗。一個點回傳一個數。彩色先轉成亮度。",
    "Vizion.get_grayval_object_model_3d (Image, ObjectModel3D, Grayval)",
)
def get_grayval_object_model_3d(image, model, *, ctx):
    picture = expect_image(image, "Image")
    points = _points(model, "ObjectModel3D")
    cols, rows = _pixel_index(points)
    if np.any((cols < 0) | (cols >= picture.width) | (rows < 0) | (rows >= picture.height)):
        raise ValueError("座標超出影像")
    values = [float(item) for item in picture.gray()[rows, cols]]
    return Result([], [_one(values)])


_SIDE = ["'inside'", "'outside'"]


@operator(
    "三維",
    [P("Region", "區域")],
    [],
    [P("ObjectModel3D", "點雲"), P("Mode", "inside 或 outside", "'inside'", _SIDE)],
    [P("ObjectModel3DSelected", "留下的點")],
    "用區域篩點",
    "x 四捨五入成行，y 四捨五入成列。inside 留下落在區域裡的點。超出這張區域範圍的點算 outside。",
    "Vizion.select_object_model_3d_region (Region, ObjectModel3D, 'inside', ObjectModel3DSelected)",
)
def select_object_model_3d_region(region, model, mode, *, ctx):
    cloud = _alive(model, "ObjectModel3D")
    source = expect_region(region, "Region")
    kind = str(mode).strip().strip("'\"").lower()
    if kind not in ("inside", "outside"):
        raise ValueError("Mode 只能是 inside 或 outside")
    if len(cloud.points) == 0:
        return Result([], [_empty_cloud()])
    cols, rows = _pixel_index(cloud.points)
    inside_canvas = (cols >= 0) & (cols < source.width) & (rows >= 0) & (rows < source.height)
    foreground = np.zeros(len(cloud.points), dtype=bool)
    if np.any(inside_canvas):
        foreground[inside_canvas] = source.union_mask()[rows[inside_canvas], cols[inside_canvas]] > 0
    keep = foreground if kind == "inside" else ~foreground
    return Result([], [ObjectModel3D(cloud.points[keep].copy())])


@operator(
    "三維",
    [P("Image", "灰階或彩色影像")],
    [],
    [P("ObjectModel3D", "點雲")],
    [P("ObjectModel3DZ", "z 改成灰階的新點雲")],
    "用影像灰階當高度",
    "x 和 y 不變。z 改成該像素的灰階。x 四捨五入成行，y 四捨五入成列。有一個點在影像外面就失敗。彩色先轉成亮度。",
    "Vizion.set_object_model_3d_z (Image, ObjectModel3D, ObjectModel3DZ)",
)
def set_object_model_3d_z(image, model, *, ctx):
    picture = expect_image(image, "Image")
    points = _points(model, "ObjectModel3D").copy()
    cols, rows = _pixel_index(points)
    if np.any((cols < 0) | (cols >= picture.width) | (rows < 0) | (rows >= picture.height)):
        raise ValueError("座標超出影像")
    points[:, 2] = picture.gray()[rows, cols]
    return Result([], [ObjectModel3D(points)])


def _choice(value, name: str, allowed: set[str]) -> str:
    kind = str(value).strip().strip("'\"").lower()
    if kind not in allowed:
        joined = "、".join(sorted(allowed))
        raise ValueError(f"{name} 只能是 {joined}")
    return kind


def _height(image) -> tuple[Image, np.ndarray]:
    picture = expect_image(image, "Image")
    return picture, picture.gray().astype(np.float64)


def _fit_height(height: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    high, wide = height.shape
    rows, cols = np.mgrid[0:high, 0:wide]
    design = np.column_stack([rows.ravel(), cols.ravel(), np.ones(height.size, np.float64)])
    alpha, beta, gamma = np.linalg.lstsq(design, height.ravel(), rcond=None)[0]
    plane = alpha * rows + beta * cols + gamma
    return plane, float(alpha), float(beta), float(gamma)


def _height_grad(height: np.ndarray, scale) -> tuple[np.ndarray, np.ndarray]:
    factor = scalar(scale, "Scale")
    if factor <= 0:
        raise ValueError("Scale 要大於 0")
    slope_x = cv2.Sobel(height, cv2.CV_64F, 1, 0, ksize=3) / 8.0 / factor
    slope_y = cv2.Sobel(height, cv2.CV_64F, 0, 1, ksize=3) / 8.0 / factor
    return slope_x, slope_y


def _unit_normal(slope_x: np.ndarray, slope_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = np.sqrt(slope_x * slope_x + slope_y * slope_y + 1.0)
    return -slope_x / length, -slope_y / length, 1.0 / length


def _signed_byte(component: np.ndarray) -> np.ndarray:
    return np.clip(np.round(128.0 + component * 127.0), 0, 255).astype(np.uint8)


def _with_domain(picture: Image, data: np.ndarray) -> Image:
    domain = None if picture.domain is None else picture.domain.copy()
    return Image(data, domain)


@operator(
    "三維",
    [P("Image", "高度圖，灰階就是 z")],
    [],
    [],
    [P("Alpha", "列的係數"), P("Beta", "行的係數"), P("Gamma", "常數")],
    "擬合高度平面",
    "灰階 = Alpha × 列 + Beta × 行 + Gamma。彩色先轉成亮度。用整張圖，不看定義域。",
    "Vizion.fit_image_plane (Image, Alpha, Beta, Gamma)",
)
def fit_image_plane(image, *, ctx):
    _picture, height = _height(image)
    _plane, alpha, beta, gamma = _fit_height(height)
    return Result([], [alpha, beta, gamma])


@operator(
    "三維",
    [P("Image", "高度圖")],
    [P("ImageFlat", "減掉平面後的影像")],
    [P("Offset", "平面變成這個灰階", "128")],
    [],
    "減掉擬合平面",
    "每個像素減掉最佳平面，再加上 Offset。完美的斜面會變成一整片 Offset。超出 0 到 255 會裁切。定義域不改。",
    "Vizion.sub_image_plane (Image, ImageFlat, 128)",
)
def sub_image_plane(image, offset, *, ctx):
    picture, height = _height(image)
    plane = _fit_height(height)[0]
    shift = scalar(offset, "Offset")
    data = np.clip(np.round(height - plane + shift), 0, 255).astype(np.uint8)
    return image_out(picture, data)


@operator(
    "三維",
    [P("Image", "高度圖")],
    [],
    [],
    [P("Deviation", "離開平面的均方根")],
    "高度離開平面多少",
    "先減掉最佳平面，再取均方根。完美平面是 0。彩色先轉成亮度。",
    "Vizion.image_plane_deviation (Image, Deviation)",
)
def image_plane_deviation(image, *, ctx):
    _picture, height = _height(image)
    plane = _fit_height(height)[0]
    return Result([], [float(np.sqrt(np.mean((height - plane) ** 2)))])


@operator(
    "三維",
    [P("Image", "高度圖")],
    [],
    [],
    [P("Nx", "法向量 x"), P("Ny", "法向量 y"), P("Nz", "法向量 z")],
    "擬合平面的法線",
    "長度是 1，z 維持正的，所以朝上。x 是行，y 是列。平坦的圖是 (0, 0, 1)。",
    "Vizion.image_plane_normal (Image, Nx, Ny, Nz)",
)
def image_plane_normal(image, *, ctx):
    _picture, height = _height(image)
    _plane, alpha, beta, _gamma = _fit_height(height)
    normal = np.array([-beta, -alpha, 1.0], np.float64)
    normal /= np.linalg.norm(normal)
    return Result([], [float(item) for item in normal])


@operator(
    "三維",
    [P("Image", "高度圖")],
    [P("ImageSlope", "坡度，單位是度")],
    [P("Scale", "幾個灰階才算一個像素的坡", "1")],
    [],
    "高度的坡度",
    "和正上方的夾角。平坦是 0。行每加 1、灰階也加 1 時，中間是 45 度。Scale 越大看起來越平。邊緣用反射邊界。",
    "Vizion.height_slope_image (Image, ImageSlope, 1)",
)
def height_slope_image(image, scale, *, ctx):
    picture, height = _height(image)
    slope_x, slope_y = _height_grad(height, scale)
    degrees = np.degrees(np.arctan2(np.hypot(slope_x, slope_y), 1.0))
    data = np.clip(np.round(degrees), 0, 255).astype(np.uint8)
    return image_out(picture, data)


@operator(
    "三維",
    [P("Image", "高度圖")],
    [P("Nx", "法線 x"), P("Ny", "法線 y"), P("Nz", "法線 z")],
    [P("Scale", "幾個灰階才算一個像素的坡", "1")],
    [],
    "每個像素的法線",
    "分量乘 127 再加 128。平坦是 (128, 128, 255)，z 朝上。x 是行，y 是列。Scale 越大法線越接近正上方。",
    "Vizion.height_normal_image (Image, Nx, Ny, Nz, 1)",
)
def height_normal_image(image, scale, *, ctx):
    picture, height = _height(image)
    normal_x, normal_y, normal_z = _unit_normal(*_height_grad(height, scale))
    images = [_with_domain(picture, _signed_byte(item)) for item in (normal_x, normal_y, normal_z)]
    return Result(images, [])


_SHADE = ["'clamp'", "'full'"]


@operator(
    "三維",
    [P("Image", "高度圖")],
    [P("ImageShade", "明暗圖")],
    [
        P("Azimuth", "光源方位。0 是 +行，順時針為正", "0"),
        P("Elevation", "光源仰角。90 是正上方", "90"),
        P("Scale", "幾個灰階才算一個像素的坡", "1"),
        P("Mode", "clamp 背光變 0，full 以 128 為零", "'clamp'", _SHADE),
    ],
    [],
    "依高度畫明暗",
    "平坦又從正上方打光，clamp 是 255。方位 0 沿 +行，順時針轉到 +列。full 把正對光源畫成接近 255，背光畫成接近 0，128 是側光。定義域不改。",
    "Vizion.shade_height_field (Image, ImageShade, 0, 90, 1, 'clamp')",
)
def shade_height_field(image, azimuth, elevation, scale, mode, *, ctx):
    picture, height = _height(image)
    kind = _choice(mode, "Mode", {"clamp", "full"})
    normal_x, normal_y, normal_z = _unit_normal(*_height_grad(height, scale))
    az = np.deg2rad(scalar(azimuth, "Azimuth"))
    el = np.deg2rad(scalar(elevation, "Elevation"))
    light = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], np.float64)
    length = float(np.linalg.norm(light))
    if length < 1e-12:
        raise ValueError("光源方向不能是 0")
    light /= length
    facing = normal_x * light[0] + normal_y * light[1] + normal_z * light[2]
    if kind == "clamp":
        data = np.clip(np.round(255.0 * np.maximum(facing, 0.0)), 0, 255).astype(np.uint8)
    else:
        data = np.clip(np.round(128.0 + 127.0 * facing), 0, 255).astype(np.uint8)
    return image_out(picture, data)


_ABOVE = ["'above'", "'below'"]


@operator(
    "三維",
    [P("Image", "高度圖")],
    [P("Region", "高出或低於平面的像素")],
    [P("Mode", "above 或 below", "'above'", _ABOVE)],
    [],
    "留下高出或低於平面的像素",
    "先擬合平面。above 留下比平面高出 0.000001 以上的像素，below 留下更低的。剛好貼在平面上的不算。結果是一個區域。",
    "Vizion.above_image_plane (Image, Region, 'above')",
)
def above_image_plane(image, mode, *, ctx):
    picture, height = _height(image)
    kind = _choice(mode, "Mode", {"above", "below"})
    residual = height - _fit_height(height)[0]
    if kind == "above":
        mask = residual > 1e-6
    else:
        mask = residual < -1e-6
    return region_out(picture.width, picture.height, mask.astype(np.uint8) * 255)


def _known_plane(height: np.ndarray, alpha, beta, gamma) -> np.ndarray:
    rows, cols = np.mgrid[0 : height.shape[0], 0 : height.shape[1]]
    return scalar(alpha, "Alpha") * rows + scalar(beta, "Beta") * cols + scalar(gamma, "Gamma")


def _byte_plane(values: np.ndarray) -> np.ndarray:
    return np.clip(np.round(values), 0, 255).astype(np.uint8)


def _row_col(row, column) -> tuple[list[float], list[float]]:
    rows = [float(item) for item in as_list(row)]
    cols = [float(item) for item in as_list(column)]
    count = max(len(rows), len(cols))
    if len(rows) == 1 and count > 1:
        rows = rows * count
    if len(cols) == 1 and count > 1:
        cols = cols * count
    if not rows or len(rows) != len(cols):
        raise ValueError("Row 和 Column 的長度要相同，而且不能是空的")
    return rows, cols


@operator(
    "三維",
    [P("Image", "高度圖")],
    [P("ImageFlat", "減掉平面後的影像")],
    [
        P("Alpha", "列的係數"),
        P("Beta", "行的係數"),
        P("Gamma", "常數"),
        P("Offset", "平面變成這個灰階", "128"),
    ],
    [],
    "減掉指定的平面",
    "用你給的係數，不重新擬合，所以局部的凸起會留著。灰階減去 Alpha × 列 + Beta × 行 + Gamma，再加上 Offset。超出 0 到 255 會裁切。定義域不改。",
    "Vizion.sub_plane_image (Image, ImageFlat, 0, 1, 0, 128)",
)
def sub_plane_image(image, alpha, beta, gamma, offset, *, ctx):
    picture, height = _height(image)
    plane = _known_plane(height, alpha, beta, gamma)
    data = _byte_plane(height - plane + scalar(offset, "Offset"))
    return image_out(picture, data)


@operator(
    "三維",
    [P("Image", "高度圖")],
    [P("ImageTilted", "加上平面後的影像")],
    [P("Alpha", "列的係數"), P("Beta", "行的係數"), P("Gamma", "常數")],
    [],
    "加上指定的平面",
    "每個像素加上 Alpha × 列 + Beta × 行 + Gamma。超出 0 到 255 會裁切。定義域不改。",
    "Vizion.add_plane_image (Image, ImageTilted, 0, 1, 0)",
)
def add_plane_image(image, alpha, beta, gamma, *, ctx):
    picture, height = _height(image)
    data = _byte_plane(height + _known_plane(height, alpha, beta, gamma))
    return image_out(picture, data)


@operator(
    "三維",
    [P("Image", "用來決定尺寸的影像")],
    [P("ImagePlane", "平面影像")],
    [P("Alpha", "列的係數"), P("Beta", "行的係數"), P("Gamma", "常數")],
    [],
    "用係數畫出平面",
    "尺寸跟輸入一樣，原點在左上角。灰階 = Alpha × 列 + Beta × 行 + Gamma，再夾到 0 到 255。係數可以直接接 fit_image_plane。",
    "Vizion.gen_image_plane (Image, ImagePlane, 0, 1, 0)",
)
def gen_image_plane(image, alpha, beta, gamma, *, ctx):
    picture, height = _height(image)
    data = _byte_plane(_known_plane(np.zeros_like(height), alpha, beta, gamma))
    return Result([Image(data)], [])


@operator(
    "三維",
    [],
    [],
    [
        P("Alpha", "列的係數"),
        P("Beta", "行的係數"),
        P("Gamma", "常數"),
        P("Row", "列"),
        P("Column", "行"),
    ],
    [P("Gray", "平面在該點的值")],
    "計算平面上某一點的高度",
    "值是 Alpha × 列 + Beta × 行 + Gamma，不裁切。Row 和 Column 可以是 tuple。一個點回傳一個數。",
    "Vizion.plane_gray (0, 1, 5, 2, 3, Gray)",
)
def plane_gray(alpha, beta, gamma, row, column, *, ctx):
    rows, cols = _row_col(row, column)
    a, b, g = scalar(alpha, "Alpha"), scalar(beta, "Beta"), scalar(gamma, "Gamma")
    values = [a * item_row + b * item_col + g for item_row, item_col in zip(rows, cols)]
    return Result([], [_one(values)])


@operator(
    "三維",
    [],
    [],
    [P("Alpha", "列的係數"), P("Beta", "行的係數")],
    [P("Tilt", "和正上方的夾角，單位是度"), P("Azimuth", "爬升方向，單位是度")],
    "平面的傾角和爬升方向",
    "傾角 0 是平坦。爬升方向 0 是往 +行，順時針轉到 +列，範圍 0 到 360。沒有坡度時方向是 0。",
    "Vizion.plane_angle (0, 1, Tilt, Azimuth)",
)
def plane_angle(alpha, beta, *, ctx):
    rise_row = scalar(alpha, "Alpha")
    rise_col = scalar(beta, "Beta")
    slope = float(np.hypot(rise_row, rise_col))
    tilt = float(np.degrees(np.arctan2(slope, 1.0)))
    if slope <= 1e-12:
        azimuth = 0.0
    else:
        azimuth = float(np.degrees(np.arctan2(rise_row, rise_col)))
        if azimuth < 0.0:
            azimuth += 360.0
    return Result([], [tilt, azimuth])


@operator(
    "三維",
    [P("Image", "高度圖")],
    [P("Region", "落在範圍裡的像素")],
    [
        P("Alpha", "列的係數"),
        P("Beta", "行的係數"),
        P("Gamma", "常數"),
        P("Min", "高出平面的下限"),
        P("Max", "高出平面的上限"),
    ],
    [],
    "留下和指定平面相差在範圍內的像素",
    "比較的是灰階減去平面。下限和上限含在內，寫反了會自動對調。結果是一個區域。",
    "Vizion.band_image_plane (Image, Region, 0, 1, 0, -1, 1)",
)
def band_image_plane(image, alpha, beta, gamma, low, high, *, ctx):
    picture, height = _height(image)
    residual = height - _known_plane(height, alpha, beta, gamma)
    floor = scalar(low, "Min")
    ceiling = scalar(high, "Max")
    if ceiling < floor:
        floor, ceiling = ceiling, floor
    mask = (residual >= floor) & (residual <= ceiling)
    return region_out(picture.width, picture.height, mask.astype(np.uint8) * 255)


_VOLUME = ["'above'", "'below'", "'signed'"]


@operator(
    "三維",
    [P("Image", "高度圖")],
    [],
    [
        P("Alpha", "列的係數"),
        P("Beta", "行的係數"),
        P("Gamma", "常數"),
        P("Mode", "above、below 或 signed", "'above'", _VOLUME),
    ],
    [P("Volume", "灰階差的總和")],
    "把高出或低於平面的灰階加起來",
    "above 只加高出的部分，below 只加低於的部分，兩邊都記成正的。signed 高出是正、低於是負。單位是灰階，一個像素一份。",
    "Vizion.volume_image_plane (Image, 0, 0, 0, 'above', Volume)",
)
def volume_image_plane(image, alpha, beta, gamma, mode, *, ctx):
    _picture, height = _height(image)
    kind = _choice(mode, "Mode", {"above", "below", "signed"})
    residual = height - _known_plane(height, alpha, beta, gamma)
    if kind == "above":
        total = float(np.maximum(residual, 0.0).sum())
    elif kind == "below":
        total = float(np.maximum(-residual, 0.0).sum())
    else:
        total = float(residual.sum())
    return Result([], [total])


_EXTREME = ["'high'", "'low'"]


@operator(
    "三維",
    [P("Image", "高度圖")],
    [],
    [
        P("Alpha", "列的係數"),
        P("Beta", "行的係數"),
        P("Gamma", "常數"),
        P("Mode", "high 或 low", "'high'", _EXTREME),
    ],
    [P("Row", "列"), P("Column", "行"), P("Residual", "高出平面多少")],
    "找離指定平面最高或最低的像素",
    "比的是灰階減去平面。一樣高時留列最小、再來行最小的那一點。",
    "Vizion.extreme_image_plane (Image, 0, 0, 0, 'high', Row, Column, Residual)",
)
def extreme_image_plane(image, alpha, beta, gamma, mode, *, ctx):
    picture, height = _height(image)
    kind = _choice(mode, "Mode", {"high", "low"})
    residual = height - _known_plane(height, alpha, beta, gamma)
    flat = residual.ravel()
    index = int(np.argmax(flat) if kind == "high" else np.argmin(flat))
    row, column = divmod(index, picture.width)
    return Result([], [int(row), int(column), float(flat[index])])


def _region_samples(region, image):
    picture = expect_image(image, "Image")
    area = expect_region(region, "Region")
    if (picture.width, picture.height) != (area.width, area.height):
        raise ValueError("區域和影像的尺寸要相同")
    mask = area.union_mask()
    if picture.domain is not None:
        mask = cv2.bitwise_and(mask, picture.domain)
    rows, cols = np.nonzero(mask)
    values = picture.gray()[rows, cols].astype(np.float64)
    return picture, mask, rows, cols, values


def _need_pixels(rows: np.ndarray) -> None:
    if len(rows) == 0:
        raise ValueError("區域是空的")


def _fit_pixels(rows: np.ndarray, cols: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    if len(rows) < 3:
        raise ValueError("至少要三個像素")
    centered = np.column_stack([rows - rows.mean(), cols - cols.mean()]).astype(np.float64)
    singular = np.linalg.svd(centered, compute_uv=False)
    limit = 1e-8 * max(float(singular[0]), 1.0)
    if len(singular) < 2 or float(singular[-1]) <= limit:
        raise ValueError("這些像素排不成一個平面")
    design = np.column_stack([rows, cols, np.ones(len(rows), np.float64)])
    alpha, beta, gamma = np.linalg.lstsq(design, values.astype(np.float64), rcond=None)[0]
    return float(alpha), float(beta), float(gamma)


def _plane_error(rows, cols, values, plane) -> np.ndarray:
    alpha, beta, gamma = plane
    return np.abs(values - (alpha * rows + beta * cols + gamma))


def _robust_plane(rows, cols, values, max_dist: float):
    base = _fit_pixels(rows, cols, values)
    error = _plane_error(rows, cols, values, base)
    if float(np.max(error)) <= max_dist + 1e-9:
        return base
    count = len(rows)
    if count <= 16:
        draws = itertools.combinations(range(count), 3)
    else:
        generator = np.random.default_rng(0)
        draws = (tuple(int(item) for item in generator.choice(count, 3, replace=False)) for _turn in range(80))
    best = base
    best_error = error
    best_score = float(np.median(error))
    best_total = float(np.sum(error))
    for index in draws:
        chosen = list(index)
        try:
            trial = _fit_pixels(rows[chosen], cols[chosen], values[chosen])
        except ValueError:
            continue
        trial_error = _plane_error(rows, cols, values, trial)
        score = float(np.median(trial_error))
        total = float(np.sum(trial_error))
        if score < best_score - 1e-9 or (abs(score - best_score) <= 1e-9 and total < best_total - 1e-6):
            best = trial
            best_error = trial_error
            best_score = score
            best_total = total
    keep = best_error <= max_dist + 1e-9
    if int(np.count_nonzero(keep)) < 3:
        raise ValueError("留下的像素少於三個")
    if bool(np.all(keep)):
        return best
    return _fit_pixels(rows[keep], cols[keep], values[keep])


def _residual_at(rows, cols, values, alpha, beta, gamma) -> np.ndarray:
    return values - (
        scalar(alpha, "Alpha") * rows + scalar(beta, "Beta") * cols + scalar(gamma, "Gamma")
    )


@operator(
    "三維",
    [P("Region", "要擬合的區域"), P("Image", "高度圖")],
    [],
    [],
    [P("Alpha", "列的係數"), P("Beta", "行的係數"), P("Gamma", "常數")],
    "用區域裡的像素擬合平面",
    "只用量區域裡的像素，背景不會把平面拉走。多個物件會合併。有定義域時再和定義域取交集。至少要三個排得成面的像素。彩色先轉成亮度。",
    "Vizion.fit_plane_region (Region, Image, Alpha, Beta, Gamma)",
)
def fit_plane_region(region, image, *, ctx):
    _picture, _mask, rows, cols, values = _region_samples(region, image)
    _need_pixels(rows)
    alpha, beta, gamma = _fit_pixels(rows, cols, values)
    return Result([], [alpha, beta, gamma])


@operator(
    "三維",
    [P("Region", "要擬合的區域"), P("Image", "高度圖")],
    [],
    [P("MaxDist", "像素可以離開平面的最大差距", "1")],
    [P("Alpha", "列的係數"), P("Beta", "行的係數"), P("Gamma", "常數")],
    "排除離群像素後再擬合平面",
    "先從三點子集裡挑中位誤差小的平面，再留下差距不超過 MaxDist 的像素重擬合。差距含等於 MaxDist。每個點一開始就在門檻裡時，直接用全部像素。留下的像素少於三個就失敗。",
    "Vizion.fit_plane_region_inlier (Region, Image, 1, Alpha, Beta, Gamma)",
)
def fit_plane_region_inlier(region, image, max_dist, *, ctx):
    _picture, _mask, rows, cols, values = _region_samples(region, image)
    _need_pixels(rows)
    dist = scalar(max_dist, "MaxDist")
    if dist < 0:
        raise ValueError("MaxDist 不能是負的")
    alpha, beta, gamma = _robust_plane(rows, cols, values, dist)
    return Result([], [alpha, beta, gamma])


@operator(
    "三維",
    [P("Region", "區域"), P("Image", "高度圖")],
    [],
    [P("Alpha", "列的係數"), P("Beta", "行的係數"), P("Gamma", "常數")],
    [P("Deviation", "均方根")],
    "區域裡離開平面的均方根",
    "只算區域裡的像素。均方根是差的平方平均再開根號。完美貼合是 0。",
    "Vizion.deviation_plane_region (Region, Image, 0, 1, 0, Deviation)",
)
def deviation_plane_region(region, image, alpha, beta, gamma, *, ctx):
    _picture, _mask, rows, cols, values = _region_samples(region, image)
    _need_pixels(rows)
    residual = _residual_at(rows, cols, values, alpha, beta, gamma)
    return Result([], [float(np.sqrt(np.mean(residual ** 2)))])


@operator(
    "三維",
    [P("Region", "區域"), P("Image", "高度圖")],
    [],
    [P("Alpha", "列的係數"), P("Beta", "行的係數"), P("Gamma", "常數")],
    [P("Mean", "平均差距")],
    "區域裡高出平面的平均",
    "每個像素先減去平面，再取平均。高出是正、低於是負。完美貼合是 0。",
    "Vizion.mean_plane_region (Region, Image, 0, 1, 0, Mean)",
)
def mean_plane_region(region, image, alpha, beta, gamma, *, ctx):
    _picture, _mask, rows, cols, values = _region_samples(region, image)
    _need_pixels(rows)
    residual = _residual_at(rows, cols, values, alpha, beta, gamma)
    return Result([], [float(residual.mean())])


@operator(
    "三維",
    [P("Region", "區域"), P("Image", "高度圖")],
    [],
    [
        P("Alpha", "列的係數"),
        P("Beta", "行的係數"),
        P("Gamma", "常數"),
        P("Mode", "above、below 或 signed", "'above'", _VOLUME),
    ],
    [P("Volume", "灰階差的總和")],
    "把區域裡高出或低於平面的灰階加起來",
    "above 只加高出的部分，below 只加低於的部分，兩邊都記成正的。signed 高出是正、低於是負。背景不算。",
    "Vizion.volume_plane_region (Region, Image, 0, 0, 0, 'above', Volume)",
)
def volume_plane_region(region, image, alpha, beta, gamma, mode, *, ctx):
    _picture, _mask, rows, cols, values = _region_samples(region, image)
    _need_pixels(rows)
    kind = _choice(mode, "Mode", {"above", "below", "signed"})
    residual = _residual_at(rows, cols, values, alpha, beta, gamma)
    if kind == "above":
        total = float(np.maximum(residual, 0.0).sum())
    elif kind == "below":
        total = float(np.maximum(-residual, 0.0).sum())
    else:
        total = float(residual.sum())
    return Result([], [total])


@operator(
    "三維",
    [P("Region", "區域"), P("Image", "高度圖")],
    [],
    [
        P("Alpha", "列的係數"),
        P("Beta", "行的係數"),
        P("Gamma", "常數"),
        P("Mode", "high 或 low", "'high'", _EXTREME),
    ],
    [P("Row", "列"), P("Column", "行"), P("Residual", "高出平面多少")],
    "找區域裡離平面最高或最低的像素",
    "只在區域裡找。一樣高時留列最小、再來行最小的那一點。",
    "Vizion.extreme_plane_region (Region, Image, 0, 0, 0, 'high', Row, Column, Residual)",
)
def extreme_plane_region(region, image, alpha, beta, gamma, mode, *, ctx):
    _picture, _mask, rows, cols, values = _region_samples(region, image)
    _need_pixels(rows)
    kind = _choice(mode, "Mode", {"high", "low"})
    residual = _residual_at(rows, cols, values, alpha, beta, gamma)
    index = int(np.argmax(residual) if kind == "high" else np.argmin(residual))
    return Result([], [int(rows[index]), int(cols[index]), float(residual[index])])


@operator(
    "三維",
    [P("Region", "區域"), P("Image", "高度圖")],
    [P("RegionBand", "落在範圍裡的像素")],
    [
        P("Alpha", "列的係數"),
        P("Beta", "行的係數"),
        P("Gamma", "常數"),
        P("Min", "高出平面的下限"),
        P("Max", "高出平面的上限"),
    ],
    [],
    "留下區域裡和平面相差在範圍內的像素",
    "比較的是灰階減去平面。下限和上限含在內，寫反了會自動對調。區域外面的像素不會留下。",
    "Vizion.band_plane_region (Region, Image, RegionBand, 0, 1, 0, -1, 1)",
)
def band_plane_region(region, image, alpha, beta, gamma, low, high, *, ctx):
    picture, mask, _rows, _cols, _values = _region_samples(region, image)
    floor = scalar(low, "Min")
    ceiling = scalar(high, "Max")
    if ceiling < floor:
        floor, ceiling = ceiling, floor
    residual = picture.gray().astype(np.float64) - _known_plane(picture.gray(), alpha, beta, gamma)
    selected = (mask > 0) & (residual >= floor) & (residual <= ceiling)
    return region_out(picture.width, picture.height, selected.astype(np.uint8) * 255)


@operator(
    "三維",
    [P("Region", "要修平的區域"), P("Image", "高度圖")],
    [P("ImageFlat", "區域裡減掉平面的影像")],
    [
        P("Alpha", "列的係數"),
        P("Beta", "行的係數"),
        P("Gamma", "常數"),
        P("Offset", "平面變成這個灰階", "128"),
    ],
    [],
    "只在區域裡減掉指定平面",
    "區域裡變成灰階減去平面再加 Offset。區域外維持原來的亮度。彩色先轉成亮度。超出 0 到 255 會裁切。定義域不改。",
    "Vizion.sub_plane_region (Region, Image, ImageFlat, 0, 1, 0, 128)",
)
def sub_plane_region(region, image, alpha, beta, gamma, offset, *, ctx):
    picture, _mask, rows, cols, values = _region_samples(region, image)
    data = picture.gray().astype(np.float64)
    if len(rows):
        data[rows, cols] = _residual_at(rows, cols, values, alpha, beta, gamma) + scalar(offset, "Offset")
    return image_out(picture, _byte_plane(data))


def _object_samples(region, image):
    picture = expect_image(image, "Image")
    area = expect_region(region, "Region")
    if (picture.width, picture.height) != (area.width, area.height):
        raise ValueError("區域和影像的尺寸要相同")
    if area.count() == 0:
        raise ValueError("區域是空的")
    gray = picture.gray()
    samples = []
    for index, mask in enumerate(area.objects, start=1):
        used = mask if picture.domain is None else cv2.bitwise_and(mask, picture.domain)
        rows, cols = np.nonzero(used)
        if len(rows) == 0:
            raise ValueError(f"第 {index} 個物件是空的")
        samples.append((index, rows, cols, gray[rows, cols].astype(np.float64)))
    return picture, area, samples


def _fit_object(index: int, rows, cols, values) -> tuple[float, float, float]:
    try:
        return _fit_pixels(rows, cols, values)
    except ValueError as exc:
        raise ValueError(f"第 {index} 個物件：{exc}") from exc


def _span(low, high, low_name: str, high_name: str) -> tuple[float, float]:
    floor = scalar(low, low_name)
    ceiling = scalar(high, high_name)
    if ceiling < floor:
        floor, ceiling = ceiling, floor
    return floor, ceiling


def _pick_region(area: Region, indexes: list[int]) -> Region:
    masks = [area.objects[index] for index in indexes]
    links = None
    if area.links is not None and len(area.links) == area.count():
        links = [area.links[index] for index in indexes]
    return Region(area.width, area.height, masks, multi_color=len(masks) > 1, links=links)


@operator(
    "三維",
    [P("Region", "物件集合"), P("Image", "高度圖")],
    [],
    [],
    [P("Alpha", "每個物件的列係數"), P("Beta", "每個物件的行係數"), P("Gamma", "每個物件的常數")],
    "每個物件各自擬合平面",
    "不會把物件合併。結果依物件排列，一個物件也是 tuple。排不成平面時會指出是第幾個物件。彩色先轉成亮度。有定義域時再和定義域取交集。",
    "Vizion.fit_plane_objects (Region, Image, Alpha, Beta, Gamma)",
)
def fit_plane_objects(region, image, *, ctx):
    _picture, _area, samples = _object_samples(region, image)
    alphas, betas, gammas = [], [], []
    for index, rows, cols, values in samples:
        alpha, beta, gamma = _fit_object(index, rows, cols, values)
        alphas.append(alpha)
        betas.append(beta)
        gammas.append(gamma)
    return Result([], [alphas, betas, gammas])


@operator(
    "三維",
    [P("Region", "物件集合"), P("Image", "高度圖")],
    [],
    [],
    [P("Flatness", "每個物件離開自己平面的均方根")],
    "每個物件有多不平",
    "各自擬合平面後，算殘差的均方根。平的物件是 0。結果依物件排列。",
    "Vizion.flatness_objects (Region, Image, Flatness)",
)
def flatness_objects(region, image, *, ctx):
    _picture, _area, samples = _object_samples(region, image)
    scores = []
    for index, rows, cols, values in samples:
        alpha, beta, gamma = _fit_object(index, rows, cols, values)
        residual = values - (alpha * rows + beta * cols + gamma)
        scores.append(float(np.sqrt(np.mean(residual ** 2))))
    return Result([], [scores])


@operator(
    "三維",
    [P("Region", "物件集合"), P("Image", "高度圖")],
    [],
    [P("Alpha", "列的係數"), P("Beta", "行的係數"), P("Gamma", "常數")],
    [P("Mean", "每個物件的平均差距")],
    "每個物件高出指定平面多少",
    "用同一組係數，不重新擬合。高出是正、低於是負。結果依物件排列。",
    "Vizion.mean_plane_objects (Region, Image, 0, 0, 0, Mean)",
)
def mean_plane_objects(region, image, alpha, beta, gamma, *, ctx):
    _picture, _area, samples = _object_samples(region, image)
    means = []
    for _index, rows, cols, values in samples:
        means.append(float(_residual_at(rows, cols, values, alpha, beta, gamma).mean()))
    return Result([], [means])


@operator(
    "三維",
    [P("Region", "物件集合"), P("Image", "高度圖")],
    [],
    [
        P("Alpha", "列的係數"),
        P("Beta", "行的係數"),
        P("Gamma", "常數"),
        P("Mode", "above、below 或 signed", "'above'", _VOLUME),
    ],
    [P("Volume", "每個物件的灰階差總和")],
    "每個物件高出或低於平面的總和",
    "above 只加高出的部分，below 只加低於的部分，兩邊都記成正的。signed 高出是正、低於是負。結果依物件排列。",
    "Vizion.volume_plane_objects (Region, Image, 0, 0, 0, 'above', Volume)",
)
def volume_plane_objects(region, image, alpha, beta, gamma, mode, *, ctx):
    _picture, _area, samples = _object_samples(region, image)
    kind = _choice(mode, "Mode", {"above", "below", "signed"})
    totals = []
    for _index, rows, cols, values in samples:
        residual = _residual_at(rows, cols, values, alpha, beta, gamma)
        if kind == "above":
            totals.append(float(np.maximum(residual, 0.0).sum()))
        elif kind == "below":
            totals.append(float(np.maximum(-residual, 0.0).sum()))
        else:
            totals.append(float(residual.sum()))
    return Result([], [totals])


@operator(
    "三維",
    [P("Region", "物件集合"), P("Image", "高度圖")],
    [],
    [
        P("Alpha", "列的係數"),
        P("Beta", "行的係數"),
        P("Gamma", "常數"),
        P("Mode", "high 或 low", "'high'", _EXTREME),
    ],
    [P("Row", "每個物件的列"), P("Column", "每個物件的行"), P("Residual", "高出平面多少")],
    "每個物件裡離平面最高或最低的像素",
    "一樣高時留該物件裡列最小、再來行最小的那一點。三個結果都依物件排列。",
    "Vizion.extreme_plane_objects (Region, Image, 0, 0, 0, 'high', Row, Column, Residual)",
)
def extreme_plane_objects(region, image, alpha, beta, gamma, mode, *, ctx):
    _picture, _area, samples = _object_samples(region, image)
    kind = _choice(mode, "Mode", {"high", "low"})
    out_rows, out_cols, out_values = [], [], []
    for _index, rows, cols, values in samples:
        residual = _residual_at(rows, cols, values, alpha, beta, gamma)
        pick = int(np.argmax(residual) if kind == "high" else np.argmin(residual))
        out_rows.append(int(rows[pick]))
        out_cols.append(int(cols[pick]))
        out_values.append(float(residual[pick]))
    return Result([], [out_rows, out_cols, out_values])


@operator(
    "三維",
    [P("Region", "物件集合"), P("Image", "高度圖")],
    [P("Selected", "平坦度落在範圍裡的物件")],
    [P("Min", "均方根下限"), P("Max", "均方根上限")],
    [],
    "用每個物件自己的平坦度篩選",
    "各自擬合後，留下均方根在下限和上限之間的物件，兩端都含在內。寫反了會自動對調。沒有符合的就得到空區域。",
    "Vizion.select_flatness (Region, Image, Selected, 0, 1)",
)
def select_flatness(region, image, low, high, *, ctx):
    picture, area, samples = _object_samples(region, image)
    floor, ceiling = _span(low, high, "Min", "Max")
    kept = []
    for index, rows, cols, values in samples:
        alpha, beta, gamma = _fit_object(index, rows, cols, values)
        residual = values - (alpha * rows + beta * cols + gamma)
        score = float(np.sqrt(np.mean(residual ** 2)))
        if floor <= score <= ceiling:
            kept.append(index - 1)
    return Result([_pick_region(area, kept)], [])


@operator(
    "三維",
    [P("Region", "物件集合"), P("Image", "高度圖")],
    [P("Selected", "平均高度落在範圍裡的物件")],
    [
        P("Alpha", "列的係數"),
        P("Beta", "行的係數"),
        P("Gamma", "常數"),
        P("Min", "平均差距下限"),
        P("Max", "平均差距上限"),
    ],
    [],
    "用每個物件高出指定平面的平均來篩選",
    "用同一組係數。留下平均差距在下限和上限之間的物件，兩端都含在內。寫反了會自動對調。",
    "Vizion.select_plane_mean (Region, Image, Selected, 0, 0, 0, 0, 10)",
)
def select_plane_mean(region, image, alpha, beta, gamma, low, high, *, ctx):
    _picture, area, samples = _object_samples(region, image)
    floor, ceiling = _span(low, high, "Min", "Max")
    kept = []
    for _index, rows, cols, values in samples:
        score = float(_residual_at(rows, cols, values, alpha, beta, gamma).mean())
        if floor <= score <= ceiling:
            kept.append(_index - 1)
    return Result([_pick_region(area, kept)], [])


@operator(
    "三維",
    [P("Region", "物件集合"), P("Image", "高度圖")],
    [P("ImageFlat", "每個物件各自修平的影像")],
    [P("Offset", "平面變成這個灰階", "128")],
    [],
    "每個物件減掉自己的平面",
    "各自擬合，物件裡變成殘差再加 Offset。物件外維持原來的亮度。物件重疊時，後面的蓋掉前面的。彩色先轉成亮度。定義域不改。",
    "Vizion.level_objects (Region, Image, ImageFlat, 128)",
)
def level_objects(region, image, offset, *, ctx):
    picture, _area, samples = _object_samples(region, image)
    data = picture.gray().astype(np.float64)
    shift = scalar(offset, "Offset")
    for index, rows, cols, values in samples:
        alpha, beta, gamma = _fit_object(index, rows, cols, values)
        residual = values - (alpha * rows + beta * cols + gamma)
        data[rows, cols] = residual + shift
    return image_out(picture, _byte_plane(data))


def _quad_design(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    rr = rows.astype(np.float64)
    cc = cols.astype(np.float64)
    return np.column_stack([rr * rr, cc * cc, rr * cc, rr, cc, np.ones(len(rr))])


def _fit_quad(rows: np.ndarray, cols: np.ndarray, values: np.ndarray) -> tuple[float, float, float, float, float, float]:
    if len(rows) < 6:
        raise ValueError("至少要六個像素")
    design = _quad_design(rows, cols)
    singular = np.linalg.svd(design, compute_uv=False)
    limit = 1e-8 * max(float(singular[0]), 1.0)
    if len(singular) < 6 or float(singular[5]) <= limit:
        raise ValueError("這些像素排不成一個二次曲面")
    coeff = np.linalg.lstsq(design, values.astype(np.float64), rcond=None)[0]
    return tuple(float(item) for item in coeff)


def _fit_quad_object(index: int, rows, cols, values):
    try:
        return _fit_quad(rows, cols, values)
    except ValueError as exc:
        raise ValueError(f"第 {index} 個物件：{exc}") from exc


def _quad_coeff(a, b, c, d, e, f) -> tuple[float, float, float, float, float, float]:
    names = ("A", "B", "C", "D", "E", "F")
    return tuple(scalar(value, name) for value, name in zip((a, b, c, d, e, f), names))


def _quad_at(rows, cols, coeff) -> np.ndarray:
    aa, bb, cc, dd, ee, ff = coeff
    rr = np.asarray(rows, np.float64)
    col = np.asarray(cols, np.float64)
    return aa * rr * rr + bb * col * col + cc * rr * col + dd * rr + ee * col + ff


_QUAD = "灰階 = A×列² + B×行² + C×列×行 + D×列 + E×行 + F。彩色先轉成亮度。"


@operator(
    "三維",
    [P("Region", "區域"), P("Image", "高度圖")],
    [],
    [],
    [
        P("A", "列²"),
        P("B", "行²"),
        P("C", "列×行"),
        P("D", "列"),
        P("E", "行"),
        P("F", "常數"),
    ],
    "用區域裡的像素擬合二次曲面",
    _QUAD + "至少要六個排得成曲面的像素。多個物件會合併。有定義域時再和定義域取交集。",
    "Vizion.fit_quad_region (Region, Image, A, B, C, D, E, F)",
)
def fit_quad_region(region, image, *, ctx):
    _picture, _mask, rows, cols, values = _region_samples(region, image)
    _need_pixels(rows)
    return Result([], list(_fit_quad(rows, cols, values)))


@operator(
    "三維",
    [P("Region", "物件集合"), P("Image", "高度圖")],
    [],
    [],
    [
        P("A", "每個物件的列²"),
        P("B", "每個物件的行²"),
        P("C", "每個物件的列×行"),
        P("D", "每個物件的列"),
        P("E", "每個物件的行"),
        P("F", "每個物件的常數"),
    ],
    "每個物件各自擬合二次曲面",
    "不會把物件合併。結果依物件排列，一個物件也是 tuple。排不成曲面時會指出是第幾個物件。",
    "Vizion.fit_quad_objects (Region, Image, A, B, C, D, E, F)",
)
def fit_quad_objects(region, image, *, ctx):
    _picture, _area, samples = _object_samples(region, image)
    columns = [[] for _ in range(6)]
    for index, rows, cols, values in samples:
        for bucket, item in zip(columns, _fit_quad_object(index, rows, cols, values)):
            bucket.append(item)
    return Result([], columns)


@operator(
    "三維",
    [P("Region", "物件集合"), P("Image", "高度圖")],
    [],
    [],
    [P("Flatness", "每個物件離開自己二次曲面的均方根")],
    "每個物件離開二次曲面多少",
    "各自擬合二次曲面後，算殘差的均方根。抛物線和圓柱面可以是 0，即使它們對平面並不平。",
    "Vizion.flatness_quad_objects (Region, Image, Flatness)",
)
def flatness_quad_objects(region, image, *, ctx):
    _picture, _area, samples = _object_samples(region, image)
    scores = []
    for index, rows, cols, values in samples:
        coeff = _fit_quad_object(index, rows, cols, values)
        residual = values - _quad_at(rows, cols, coeff)
        scores.append(float(np.sqrt(np.mean(residual ** 2))))
    return Result([], [scores])


@operator(
    "三維",
    [P("Region", "區域"), P("Image", "高度圖")],
    [],
    [
        P("A", "列²"),
        P("B", "行²"),
        P("C", "列×行"),
        P("D", "列"),
        P("E", "行"),
        P("F", "常數"),
    ],
    [P("Deviation", "均方根")],
    "區域裡離開指定二次曲面的均方根",
    "用你給的係數，不重新擬合。完美貼合是 0。",
    "Vizion.deviation_quad_region (Region, Image, 0, 1, 0, 0, 0, 0, Deviation)",
)
def deviation_quad_region(region, image, a, b, c, d, e, f, *, ctx):
    _picture, _mask, rows, cols, values = _region_samples(region, image)
    _need_pixels(rows)
    residual = values - _quad_at(rows, cols, _quad_coeff(a, b, c, d, e, f))
    return Result([], [float(np.sqrt(np.mean(residual ** 2)))])


@operator(
    "三維",
    [],
    [],
    [
        P("A", "列²"),
        P("B", "行²"),
        P("C", "列×行"),
        P("D", "列"),
        P("E", "行"),
        P("F", "常數"),
        P("Row", "列"),
        P("Column", "行"),
    ],
    [P("Gray", "曲面在該點的值")],
    "計算二次曲面上某一點的高度",
    _QUAD + "不裁切。Row 和 Column 可以是 tuple。一個點回傳一個數。",
    "Vizion.quad_gray (0, 1, 0, 0, -20, 100, 21, 12, Gray)",
)
def quad_gray(a, b, c, d, e, f, row, column, *, ctx):
    rows, cols = _row_col(row, column)
    surface = _quad_at(rows, cols, _quad_coeff(a, b, c, d, e, f))
    return Result([], [_one([float(item) for item in surface])])


@operator(
    "三維",
    [P("Image", "高度圖")],
    [P("ImageFlat", "減掉曲面後的影像")],
    [
        P("A", "列²"),
        P("B", "行²"),
        P("C", "列×行"),
        P("D", "列"),
        P("E", "行"),
        P("F", "常數"),
        P("Offset", "曲面變成這個灰階", "128"),
    ],
    [],
    "減掉指定的二次曲面",
    "整張圖都減，不重新擬合。再加上 Offset。超出 0 到 255 會裁切。定義域不改。",
    "Vizion.sub_quad_image (Image, ImageFlat, 0, 1, 0, 0, 0, 0, 128)",
)
def sub_quad_image(image, a, b, c, d, e, f, offset, *, ctx):
    picture, height = _height(image)
    rows, cols = np.mgrid[0 : picture.height, 0 : picture.width]
    surface = _quad_at(rows, cols, _quad_coeff(a, b, c, d, e, f))
    data = _byte_plane(height - surface + scalar(offset, "Offset"))
    return image_out(picture, data)


@operator(
    "三維",
    [P("Region", "物件集合"), P("Image", "高度圖")],
    [P("ImageFlat", "每個物件各自修平的影像")],
    [P("Offset", "曲面變成這個灰階", "128")],
    [],
    "每個物件減掉自己的二次曲面",
    "各自擬合。物件裡變成殘差再加 Offset，物件外維持原來的亮度。物件重疊時，後面的蓋掉前面的。定義域不改。",
    "Vizion.level_quad_objects (Region, Image, ImageFlat, 128)",
)
def level_quad_objects(region, image, offset, *, ctx):
    picture, _area, samples = _object_samples(region, image)
    data = picture.gray().astype(np.float64)
    shift = scalar(offset, "Offset")
    for index, rows, cols, values in samples:
        coeff = _fit_quad_object(index, rows, cols, values)
        data[rows, cols] = values - _quad_at(rows, cols, coeff) + shift
    return image_out(picture, _byte_plane(data))


@operator(
    "三維",
    [P("Region", "物件集合"), P("Image", "高度圖")],
    [P("Selected", "二次曲面平坦度落在範圍裡的物件")],
    [P("Min", "均方根下限"), P("Max", "均方根上限")],
    [],
    "用二次曲面的平坦度篩選物件",
    "各自擬合後，留下均方根在下限和上限之間的物件，兩端都含在內。寫反了會自動對調。沒有符合的就得到空區域。",
    "Vizion.select_quad_flatness (Region, Image, Selected, 0, 1)",
)
def select_quad_flatness(region, image, low, high, *, ctx):
    _picture, area, samples = _object_samples(region, image)
    floor, ceiling = _span(low, high, "Min", "Max")
    kept = []
    for index, rows, cols, values in samples:
        coeff = _fit_quad_object(index, rows, cols, values)
        residual = values - _quad_at(rows, cols, coeff)
        score = float(np.sqrt(np.mean(residual ** 2)))
        if floor <= score <= ceiling:
            kept.append(index - 1)
    return Result([_pick_region(area, kept)], [])


def _height_delta(image, reference):
    picture = expect_image(image, "Image")
    other = expect_image(reference, "Reference")
    if (picture.width, picture.height) != (other.width, other.height):
        raise ValueError("兩張影像的尺寸要相同")
    delta = picture.gray().astype(np.float64) - other.gray().astype(np.float64)
    domain = picture.domain
    if other.domain is not None:
        domain = other.domain if domain is None else cv2.bitwise_and(domain, other.domain)
    return picture, delta, domain


def _diff_area(region, image, reference):
    picture, delta, domain = _height_delta(image, reference)
    area = expect_region(region, "Region")
    if (picture.width, picture.height) != (area.width, area.height):
        raise ValueError("區域和影像的尺寸要相同")
    mask = area.union_mask()
    if domain is not None:
        mask = cv2.bitwise_and(mask, domain)
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError("區域是空的")
    return area, mask, rows, cols, delta[rows, cols], delta


def _diff_objects(region, image, reference):
    picture, delta, domain = _height_delta(image, reference)
    area = expect_region(region, "Region")
    if (picture.width, picture.height) != (area.width, area.height):
        raise ValueError("區域和影像的尺寸要相同")
    if area.count() == 0:
        raise ValueError("區域是空的")
    samples = []
    for index, mask in enumerate(area.objects, start=1):
        used = mask if domain is None else cv2.bitwise_and(mask, domain)
        rows, cols = np.nonzero(used)
        if len(rows) == 0:
            raise ValueError(f"第 {index} 個物件是空的")
        samples.append((index, rows, cols, delta[rows, cols]))
    return area, samples


def _diff_total(values: np.ndarray, kind: str) -> float:
    if kind == "above":
        return float(np.maximum(values, 0.0).sum())
    if kind == "below":
        return float(np.maximum(-values, 0.0).sum())
    return float(values.sum())


def _diff_extreme(rows, cols, values, kind: str):
    pick = int(np.argmax(values) if kind == "high" else np.argmin(values))
    return int(rows[pick]), int(cols[pick]), float(values[pick])


@operator(
    "三維",
    [P("Region", "區域"), P("Image", "要量的高度圖"), P("Reference", "參考高度圖")],
    [],
    [],
    [P("Mean", "平均差")],
    "區域裡兩張高度圖的平均差",
    "差是第一張減去參考圖，不裁切。高出是正、低於是負。完全相同是 0。彩色先轉成亮度。兩張圖都有定義域時先取交集。",
    "Vizion.mean_height_diff (Region, Image, Reference, Mean)",
)
def mean_height_diff(region, image, reference, *, ctx):
    _area, _mask, _rows, _cols, values, _delta = _diff_area(region, image, reference)
    return Result([], [float(values.mean())])


@operator(
    "三維",
    [P("Region", "區域"), P("Image", "要量的高度圖"), P("Reference", "參考高度圖")],
    [],
    [],
    [P("Deviation", "差的均方根")],
    "區域裡兩張高度圖相差的均方根",
    "先算第一張減去參考圖，再取平方平均的根號。完全相同是 0。整片都高一個常數時，均方根就是那個常數。",
    "Vizion.deviation_height_diff (Region, Image, Reference, Deviation)",
)
def deviation_height_diff(region, image, reference, *, ctx):
    _area, _mask, _rows, _cols, values, _delta = _diff_area(region, image, reference)
    return Result([], [float(np.sqrt(np.mean(values ** 2)))])


@operator(
    "三維",
    [P("Region", "區域"), P("Image", "要量的高度圖"), P("Reference", "參考高度圖")],
    [],
    [P("Mode", "above、below 或 signed", "'above'", _VOLUME)],
    [P("Volume", "差的總和")],
    "把區域裡高出或低於參考圖的差加起來",
    "above 只加高出的部分，below 只加低於的部分，兩邊都記成正的。signed 高出是正、低於是負。",
    "Vizion.volume_height_diff (Region, Image, Reference, 'above', Volume)",
)
def volume_height_diff(region, image, reference, mode, *, ctx):
    _area, _mask, _rows, _cols, values, _delta = _diff_area(region, image, reference)
    kind = _choice(mode, "Mode", {"above", "below", "signed"})
    return Result([], [_diff_total(values, kind)])


@operator(
    "三維",
    [P("Region", "區域"), P("Image", "要量的高度圖"), P("Reference", "參考高度圖")],
    [],
    [P("Mode", "high 或 low", "'high'", _EXTREME)],
    [P("Row", "列"), P("Column", "行"), P("Diff", "該點的差")],
    "找區域裡和參考圖相差最大或最小的像素",
    "差是第一張減去參考圖。一樣大時留列最小、再來行最小的那一點。",
    "Vizion.extreme_height_diff (Region, Image, Reference, 'high', Row, Column, Diff)",
)
def extreme_height_diff(region, image, reference, mode, *, ctx):
    _area, _mask, rows, cols, values, _delta = _diff_area(region, image, reference)
    kind = _choice(mode, "Mode", {"high", "low"})
    row, column, diff = _diff_extreme(rows, cols, values, kind)
    return Result([], [row, column, diff])


@operator(
    "三維",
    [P("Region", "物件集合"), P("Image", "要量的高度圖"), P("Reference", "參考高度圖")],
    [],
    [],
    [P("Mean", "每個物件的平均差")],
    "每個物件和參考圖的平均差",
    "不會把物件合併。差是第一張減去參考圖。結果依物件排列，一個物件也是 tuple。",
    "Vizion.mean_height_diff_objects (Region, Image, Reference, Mean)",
)
def mean_height_diff_objects(region, image, reference, *, ctx):
    _area, samples = _diff_objects(region, image, reference)
    return Result([], [[float(values.mean()) for _index, _rows, _cols, values in samples]])


@operator(
    "三維",
    [P("Region", "物件集合"), P("Image", "要量的高度圖"), P("Reference", "參考高度圖")],
    [],
    [P("Mode", "high 或 low", "'high'", _EXTREME)],
    [P("Row", "每個物件的列"), P("Column", "每個物件的行"), P("Diff", "每個物件的差")],
    "每個物件裡和參考圖相差最大或最小的像素",
    "一樣大時留該物件裡列最小、再來行最小的那一點。三個結果都依物件排列。",
    "Vizion.extreme_height_diff_objects (Region, Image, Reference, 'high', Row, Column, Diff)",
)
def extreme_height_diff_objects(region, image, reference, mode, *, ctx):
    _area, samples = _diff_objects(region, image, reference)
    kind = _choice(mode, "Mode", {"high", "low"})
    rows, cols, diffs = [], [], []
    for _index, item_rows, item_cols, values in samples:
        row, column, diff = _diff_extreme(item_rows, item_cols, values, kind)
        rows.append(row)
        cols.append(column)
        diffs.append(diff)
    return Result([], [rows, cols, diffs])


@operator(
    "三維",
    [P("Region", "物件集合"), P("Image", "要量的高度圖"), P("Reference", "參考高度圖")],
    [P("Selected", "平均差落在範圍裡的物件")],
    [P("Min", "平均差下限"), P("Max", "平均差上限")],
    [],
    "用和參考圖的平均差篩選物件",
    "留下平均差在下限和上限之間的物件，兩端都含在內。寫反了會自動對調。沒有符合的就得到空區域。",
    "Vizion.select_height_diff (Region, Image, Reference, Selected, -1, 1)",
)
def select_height_diff(region, image, reference, low, high, *, ctx):
    area, samples = _diff_objects(region, image, reference)
    floor, ceiling = _span(low, high, "Min", "Max")
    kept = []
    for index, _rows, _cols, values in samples:
        if floor <= float(values.mean()) <= ceiling:
            kept.append(index - 1)
    return Result([_pick_region(area, kept)], [])


@operator(
    "三維",
    [P("Region", "區域"), P("Image", "要量的高度圖"), P("Reference", "參考高度圖")],
    [P("RegionBand", "差落在範圍裡的像素")],
    [P("Min", "差的下限"), P("Max", "差的上限")],
    [],
    "留下和參考圖相差在範圍內的像素",
    "差是第一張減去參考圖。下限和上限含在內，寫反了會自動對調。區域外面的像素不會留下。",
    "Vizion.band_height_diff (Region, Image, Reference, RegionBand, -1, 1)",
)
def band_height_diff(region, image, reference, low, high, *, ctx):
    picture, delta, domain = _height_delta(image, reference)
    area = expect_region(region, "Region")
    if (picture.width, picture.height) != (area.width, area.height):
        raise ValueError("區域和影像的尺寸要相同")
    floor, ceiling = _span(low, high, "Min", "Max")
    mask = area.union_mask()
    if domain is not None:
        mask = cv2.bitwise_and(mask, domain)
    selected = (mask > 0) & (delta >= floor) & (delta <= ceiling)
    return region_out(picture.width, picture.height, selected.astype(np.uint8) * 255)


_ODD = ["3", "5", "7", "9"]


def _odd_window(width, height) -> tuple[int, int]:
    wide = whole(width, "MaskWidth")
    high = whole(height, "MaskHeight")
    if wide < 1 or high < 1 or wide % 2 == 0 or high % 2 == 0:
        raise ValueError("遮罩的寬和高要是正的奇數")
    return wide, high


def _local_residual(image, width, height):
    picture = expect_image(image, "Image")
    wide, high = _odd_window(width, height)
    gray = picture.gray().astype(np.float64)
    residual = gray - cv2.blur(gray, (wide, high))
    return picture, residual


def _rough_area(region, image, width, height):
    picture, residual = _local_residual(image, width, height)
    area = expect_region(region, "Region")
    if (picture.width, picture.height) != (area.width, area.height):
        raise ValueError("區域和影像的尺寸要相同")
    mask = area.union_mask()
    if picture.domain is not None:
        mask = cv2.bitwise_and(mask, picture.domain)
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError("區域是空的")
    return picture, area, mask, rows, cols, residual[rows, cols], residual


def _rough_objects(region, image, width, height):
    picture, residual = _local_residual(image, width, height)
    area = expect_region(region, "Region")
    if (picture.width, picture.height) != (area.width, area.height):
        raise ValueError("區域和影像的尺寸要相同")
    if area.count() == 0:
        raise ValueError("區域是空的")
    samples = []
    for index, mask in enumerate(area.objects, start=1):
        used = mask if picture.domain is None else cv2.bitwise_and(mask, picture.domain)
        rows, cols = np.nonzero(used)
        if len(rows) == 0:
            raise ValueError(f"第 {index} 個物件是空的")
        samples.append((index, rows, cols, residual[rows, cols]))
    return area, samples


@operator(
    "三維",
    [P("Image", "高度圖")],
    [P("ImageFlat", "減掉局部平均的影像")],
    [
        P("MaskWidth", "遮罩寬，奇數", "3", _ODD),
        P("MaskHeight", "遮罩高，奇數", "3", _ODD),
    ],
    [],
    "減掉局部平均",
    "每個像素減去同尺寸遮罩的平均，再加 128。128 代表和周圍一樣高。高於周圍比較亮，低於周圍比較暗。邊緣用反射邊界。彩色先轉成亮度。定義域不改。",
    "Vizion.sub_local_mean (Image, ImageFlat, 3, 3)",
)
def sub_local_mean(image, width, height, *, ctx):
    picture, residual = _local_residual(image, width, height)
    data = np.clip(np.round(128.0 + residual), 0, 255).astype(np.uint8)
    return image_out(picture, data)


@operator(
    "三維",
    [P("Region", "區域"), P("Image", "高度圖")],
    [],
    [
        P("MaskWidth", "遮罩寬，奇數", "3", _ODD),
        P("MaskHeight", "遮罩高，奇數", "3", _ODD),
    ],
    [P("Ra", "平均絕對偏差")],
    "區域裡的平均粗糙度",
    "先減去局部平均，再取絕對值的平均。平坦是 0。遮罩要是正的奇數。",
    "Vizion.roughness_ra (Region, Image, 3, 3, Ra)",
)
def roughness_ra(region, image, width, height, *, ctx):
    _picture, _area, _mask, _rows, _cols, values, _residual = _rough_area(region, image, width, height)
    return Result([], [float(np.mean(np.abs(values)))])


@operator(
    "三維",
    [P("Region", "區域"), P("Image", "高度圖")],
    [],
    [
        P("MaskWidth", "遮罩寬，奇數", "3", _ODD),
        P("MaskHeight", "遮罩高，奇數", "3", _ODD),
    ],
    [P("Rq", "均方根粗糙度")],
    "區域裡的均方根粗糙度",
    "先減去局部平均，再取平方平均的根號。平坦是 0。",
    "Vizion.roughness_rq (Region, Image, 3, 3, Rq)",
)
def roughness_rq(region, image, width, height, *, ctx):
    _picture, _area, _mask, _rows, _cols, values, _residual = _rough_area(region, image, width, height)
    return Result([], [float(np.sqrt(np.mean(values ** 2)))])


@operator(
    "三維",
    [P("Region", "區域"), P("Image", "高度圖")],
    [],
    [
        P("MaskWidth", "遮罩寬，奇數", "3", _ODD),
        P("MaskHeight", "遮罩高，奇數", "3", _ODD),
    ],
    [P("Rp", "最高過量"), P("Rv", "最低缺口"), P("Rz", "最高減最低")],
    "區域裡的峰谷粗糙度",
    "Rp 是比局部平均高出最多的值，Rv 是低於最多的值，通常是負的。Rz 是 Rp 減 Rv。平坦時三個都是 0。",
    "Vizion.roughness_rz (Region, Image, 3, 3, Rp, Rv, Rz)",
)
def roughness_rz(region, image, width, height, *, ctx):
    _picture, _area, _mask, _rows, _cols, values, _residual = _rough_area(region, image, width, height)
    peak = float(values.max())
    valley = float(values.min())
    return Result([], [peak, valley, peak - valley])


@operator(
    "三維",
    [P("Region", "物件集合"), P("Image", "高度圖")],
    [],
    [
        P("MaskWidth", "遮罩寬，奇數", "3", _ODD),
        P("MaskHeight", "遮罩高，奇數", "3", _ODD),
    ],
    [P("Ra", "每個物件的平均絕對偏差")],
    "每個物件的平均粗糙度",
    "不會把物件合併。平坦的物件是 0。結果依物件排列，一個物件也是 tuple。",
    "Vizion.roughness_ra_objects (Region, Image, 3, 3, Ra)",
)
def roughness_ra_objects(region, image, width, height, *, ctx):
    _area, samples = _rough_objects(region, image, width, height)
    return Result([], [[float(np.mean(np.abs(values))) for _index, _rows, _cols, values in samples]])


@operator(
    "三維",
    [P("Region", "物件集合"), P("Image", "高度圖")],
    [P("Selected", "粗糙度落在範圍裡的物件")],
    [
        P("Min", "Ra 下限"),
        P("Max", "Ra 上限"),
        P("MaskWidth", "遮罩寬，奇數", "3", _ODD),
        P("MaskHeight", "遮罩高，奇數", "3", _ODD),
    ],
    [],
    "用平均粗糙度篩選物件",
    "留下 Ra 在下限和上限之間的物件，兩端都含在內。寫反了會自動對調。沒有符合的就得到空區域。",
    "Vizion.select_roughness (Region, Image, Selected, 1, 10, 3, 3)",
)
def select_roughness(region, image, low, high, width, height, *, ctx):
    area, samples = _rough_objects(region, image, width, height)
    floor, ceiling = _span(low, high, "Min", "Max")
    kept = []
    for index, _rows, _cols, values in samples:
        if floor <= float(np.mean(np.abs(values))) <= ceiling:
            kept.append(index - 1)
    return Result([_pick_region(area, kept)], [])


@operator(
    "三維",
    [P("Region", "區域"), P("Image", "高度圖")],
    [],
    [
        P("MaskWidth", "遮罩寬，奇數", "3", _ODD),
        P("MaskHeight", "遮罩高，奇數", "3", _ODD),
        P("Mode", "high 或 low", "'high'", _EXTREME),
    ],
    [P("Row", "列"), P("Column", "行"), P("Residual", "比局部平均高多少")],
    "找區域裡比局部平均最高或最低的像素",
    "一樣高時留列最小、再來行最小的那一點。",
    "Vizion.extreme_roughness (Region, Image, 3, 3, 'high', Row, Column, Residual)",
)
def extreme_roughness(region, image, width, height, mode, *, ctx):
    _picture, _area, _mask, rows, cols, values, _residual = _rough_area(region, image, width, height)
    kind = _choice(mode, "Mode", {"high", "low"})
    pick = int(np.argmax(values) if kind == "high" else np.argmin(values))
    return Result([], [int(rows[pick]), int(cols[pick]), float(values[pick])])


@operator(
    "三維",
    [P("Region", "區域"), P("Image", "高度圖")],
    [P("RegionBand", "偏差落在範圍裡的像素")],
    [
        P("Min", "偏差下限"),
        P("Max", "偏差上限"),
        P("MaskWidth", "遮罩寬，奇數", "3", _ODD),
        P("MaskHeight", "遮罩高，奇數", "3", _ODD),
    ],
    [],
    "留下和局部平均相差在範圍內的像素",
    "比較的是灰階減去局部平均。下限和上限含在內，寫反了會自動對調。區域外面的像素不會留下。",
    "Vizion.band_roughness (Region, Image, RegionBand, 20, 30, 3, 3)",
)
def band_roughness(region, image, low, high, width, height, *, ctx):
    picture, residual = _local_residual(image, width, height)
    area = expect_region(region, "Region")
    if (picture.width, picture.height) != (area.width, area.height):
        raise ValueError("區域和影像的尺寸要相同")
    floor, ceiling = _span(low, high, "Min", "Max")
    mask = area.union_mask()
    if picture.domain is not None:
        mask = cv2.bitwise_and(mask, picture.domain)
    selected = (mask > 0) & (residual >= floor) & (residual <= ceiling)
    return region_out(picture.width, picture.height, selected.astype(np.uint8) * 255)


def _median_size(size) -> int:
    wide = whole(size, "MaskSize")
    if wide < 1 or wide % 2 == 0:
        raise ValueError("遮罩邊長要是正的奇數")
    return wide


def _median_residual(image, size):
    picture = expect_image(image, "Image")
    ksize = _median_size(size)
    gray = picture.gray()
    smooth = gray if ksize == 1 else cv2.medianBlur(gray, ksize)
    residual = gray.astype(np.float64) - smooth.astype(np.float64)
    return picture, residual


def _median_area(region, image, size):
    picture, residual = _median_residual(image, size)
    area = expect_region(region, "Region")
    if (picture.width, picture.height) != (area.width, area.height):
        raise ValueError("區域和影像的尺寸要相同")
    mask = area.union_mask()
    if picture.domain is not None:
        mask = cv2.bitwise_and(mask, picture.domain)
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError("區域是空的")
    return picture, area, rows, cols, residual[rows, cols], residual


def _median_objects(region, image, size):
    picture, residual = _median_residual(image, size)
    area = expect_region(region, "Region")
    if (picture.width, picture.height) != (area.width, area.height):
        raise ValueError("區域和影像的尺寸要相同")
    if area.count() == 0:
        raise ValueError("區域是空的")
    samples = []
    for index, mask in enumerate(area.objects, start=1):
        used = mask if picture.domain is None else cv2.bitwise_and(mask, picture.domain)
        rows, cols = np.nonzero(used)
        if len(rows) == 0:
            raise ValueError(f"第 {index} 個物件是空的")
        samples.append((index, residual[rows, cols]))
    return area, samples


@operator(
    "三維",
    [P("Image", "高度圖")],
    [P("ImageFlat", "減掉局部中值的影像")],
    [P("MaskSize", "正方形遮罩的邊長，奇數", "3", _ODD)],
    [],
    "減掉局部中值",
    "每個像素減去正方形遮罩的中值，再加 128。128 代表和周圍的中值一樣。3 是 3×3，不是半徑。單點凸起不會抹到旁邊。邊緣用複製邊界。彩色先轉成亮度。定義域不改。",
    "Vizion.sub_local_median (Image, ImageFlat, 3)",
)
def sub_local_median(image, size, *, ctx):
    picture, residual = _median_residual(image, size)
    data = np.clip(np.round(128.0 + residual), 0, 255).astype(np.uint8)
    return image_out(picture, data)


@operator(
    "三維",
    [P("Region", "區域"), P("Image", "高度圖")],
    [],
    [P("MaskSize", "正方形遮罩的邊長，奇數", "3", _ODD)],
    [P("Ra", "平均絕對偏差")],
    "區域裡相對中值的平均粗糙度",
    "先減去局部中值，再取絕對值的平均。平坦是 0。邊長要是正的奇數，3 是 3×3。",
    "Vizion.roughness_median_ra (Region, Image, 3, Ra)",
)
def roughness_median_ra(region, image, size, *, ctx):
    _picture, _area, _rows, _cols, values, _residual = _median_area(region, image, size)
    return Result([], [float(np.mean(np.abs(values)))])


@operator(
    "三維",
    [P("Region", "區域"), P("Image", "高度圖")],
    [],
    [P("MaskSize", "正方形遮罩的邊長，奇數", "3", _ODD)],
    [P("Rq", "均方根粗糙度")],
    "區域裡相對中值的均方根粗糙度",
    "先減去局部中值，再取平方平均的根號。平坦是 0。",
    "Vizion.roughness_median_rq (Region, Image, 3, Rq)",
)
def roughness_median_rq(region, image, size, *, ctx):
    _picture, _area, _rows, _cols, values, _residual = _median_area(region, image, size)
    return Result([], [float(np.sqrt(np.mean(values ** 2)))])


@operator(
    "三維",
    [P("Region", "區域"), P("Image", "高度圖")],
    [],
    [P("MaskSize", "正方形遮罩的邊長，奇數", "3", _ODD)],
    [P("Rp", "最高過量"), P("Rv", "最低缺口"), P("Rz", "最高減最低")],
    "區域裡相對中值的峰谷粗糙度",
    "Rp 是比局部中值高出最多的值，Rv 是低於最多的值。Rz 是 Rp 減 Rv。平坦時三個都是 0。",
    "Vizion.roughness_median_rz (Region, Image, 3, Rp, Rv, Rz)",
)
def roughness_median_rz(region, image, size, *, ctx):
    _picture, _area, _rows, _cols, values, _residual = _median_area(region, image, size)
    peak = float(values.max())
    valley = float(values.min())
    return Result([], [peak, valley, peak - valley])


@operator(
    "三維",
    [P("Region", "物件集合"), P("Image", "高度圖")],
    [],
    [P("MaskSize", "正方形遮罩的邊長，奇數", "3", _ODD)],
    [P("Ra", "每個物件的平均絕對偏差")],
    "每個物件相對中值的平均粗糙度",
    "不會把物件合併。平坦的物件是 0。結果依物件排列，一個物件也是 tuple。",
    "Vizion.roughness_median_ra_objects (Region, Image, 3, Ra)",
)
def roughness_median_ra_objects(region, image, size, *, ctx):
    _area, samples = _median_objects(region, image, size)
    return Result([], [[float(np.mean(np.abs(values))) for _index, values in samples]])


@operator(
    "三維",
    [P("Region", "物件集合"), P("Image", "高度圖")],
    [P("Selected", "粗糙度落在範圍裡的物件")],
    [
        P("Min", "Ra 下限"),
        P("Max", "Ra 上限"),
        P("MaskSize", "正方形遮罩的邊長，奇數", "3", _ODD),
    ],
    [],
    "用相對中值的粗糙度篩選物件",
    "留下 Ra 在下限和上限之間的物件，兩端都含在內。寫反了會自動對調。沒有符合的就得到空區域。",
    "Vizion.select_median_roughness (Region, Image, Selected, 1, 10, 3)",
)
def select_median_roughness(region, image, low, high, size, *, ctx):
    area, samples = _median_objects(region, image, size)
    floor, ceiling = _span(low, high, "Min", "Max")
    kept = [index - 1 for index, values in samples if floor <= float(np.mean(np.abs(values))) <= ceiling]
    return Result([_pick_region(area, kept)], [])


@operator(
    "三維",
    [P("Region", "區域"), P("Image", "高度圖")],
    [],
    [
        P("MaskSize", "正方形遮罩的邊長，奇數", "3", _ODD),
        P("Mode", "high 或 low", "'high'", _EXTREME),
    ],
    [P("Row", "列"), P("Column", "行"), P("Residual", "比局部中值高多少")],
    "找區域裡比局部中值最高或最低的像素",
    "一樣高時留列最小、再來行最小的那一點。",
    "Vizion.extreme_median_roughness (Region, Image, 3, 'high', Row, Column, Residual)",
)
def extreme_median_roughness(region, image, size, mode, *, ctx):
    _picture, _area, rows, cols, values, _residual = _median_area(region, image, size)
    kind = _choice(mode, "Mode", {"high", "low"})
    pick = int(np.argmax(values) if kind == "high" else np.argmin(values))
    return Result([], [int(rows[pick]), int(cols[pick]), float(values[pick])])


@operator(
    "三維",
    [P("Region", "區域"), P("Image", "高度圖")],
    [P("RegionBand", "偏差落在範圍裡的像素")],
    [
        P("Min", "偏差下限"),
        P("Max", "偏差上限"),
        P("MaskSize", "正方形遮罩的邊長，奇數", "3", _ODD),
    ],
    [],
    "留下和局部中值相差在範圍內的像素",
    "比較的是灰階減去局部中值。下限和上限含在內，寫反了會自動對調。區域外面的像素不會留下。",
    "Vizion.band_median_roughness (Region, Image, RegionBand, 20, 40, 3)",
)
def band_median_roughness(region, image, low, high, size, *, ctx):
    picture, residual = _median_residual(image, size)
    area = expect_region(region, "Region")
    if (picture.width, picture.height) != (area.width, area.height):
        raise ValueError("區域和影像的尺寸要相同")
    floor, ceiling = _span(low, high, "Min", "Max")
    mask = area.union_mask()
    if picture.domain is not None:
        mask = cv2.bitwise_and(mask, picture.domain)
    selected = (mask > 0) & (residual >= floor) & (residual <= ceiling)
    return region_out(picture.width, picture.height, selected.astype(np.uint8) * 255)


_SPOT = ["'white'", "'black'"]


def _spot_size(size) -> int:
    wide = whole(size, "MaskSize")
    if wide < 1 or wide % 2 == 0:
        raise ValueError("遮罩邊長要是正的奇數")
    return wide


def _spot_hat(image, size, mode):
    picture = expect_image(image, "Image")
    kind = _choice(mode, "Mode", {"white", "black"})
    ksize = _spot_size(size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    op = cv2.MORPH_TOPHAT if kind == "white" else cv2.MORPH_BLACKHAT
    hat = cv2.morphologyEx(picture.gray(), op, kernel)
    return picture, kind, hat


def _spot_area(region, image, size, mode):
    picture, kind, hat = _spot_hat(image, size, mode)
    area = expect_region(region, "Region")
    if (picture.width, picture.height) != (area.width, area.height):
        raise ValueError("區域和影像的尺寸要相同")
    mask = area.union_mask()
    if picture.domain is not None:
        mask = cv2.bitwise_and(mask, picture.domain)
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError("區域是空的")
    return picture, area, rows, cols, hat[rows, cols].astype(np.float64), hat


def _spot_objects(region, image, size, mode):
    picture, _kind, hat = _spot_hat(image, size, mode)
    area = expect_region(region, "Region")
    if (picture.width, picture.height) != (area.width, area.height):
        raise ValueError("區域和影像的尺寸要相同")
    if area.count() == 0:
        raise ValueError("區域是空的")
    samples = []
    for index, mask in enumerate(area.objects, start=1):
        used = mask if picture.domain is None else cv2.bitwise_and(mask, picture.domain)
        rows, cols = np.nonzero(used)
        if len(rows) == 0:
            raise ValueError(f"第 {index} 個物件是空的")
        samples.append((index, hat[rows, cols].astype(np.float64)))
    return area, samples


@operator(
    "形態學",
    [P("Image", "影像")],
    [P("ImageWhite", "小亮點")],
    [P("MaskSize", "正方形遮罩的邊長，奇數", "3", _ODD)],
    [],
    "抓比遮罩小的亮點",
    "原圖減去正方形開運算。比遮罩小的亮點會留下，大塊平坦的地方是 0。3 是 3×3。彩色先轉成亮度。定義域不改。",
    "Vizion.gray_spot_white (Image, ImageWhite, 3)",
)
def gray_spot_white(image, size, *, ctx):
    picture, _kind, hat = _spot_hat(image, size, "white")
    return image_out(picture, hat)


@operator(
    "形態學",
    [P("Image", "影像")],
    [P("ImageBlack", "小暗點")],
    [P("MaskSize", "正方形遮罩的邊長，奇數", "3", _ODD)],
    [],
    "抓比遮罩小的暗點",
    "正方形閉運算減去原圖。比遮罩小的暗點會留下，大塊平坦的地方是 0。3 是 3×3。彩色先轉成亮度。定義域不改。",
    "Vizion.gray_spot_black (Image, ImageBlack, 3)",
)
def gray_spot_black(image, size, *, ctx):
    picture, _kind, hat = _spot_hat(image, size, "black")
    return image_out(picture, hat)


@operator(
    "形態學",
    [P("Region", "區域"), P("Image", "影像")],
    [],
    [
        P("Mode", "white 或 black", "'white'", _SPOT),
        P("MaskSize", "正方形遮罩的邊長，奇數", "3", _ODD),
    ],
    [P("Volume", "亮點或暗點的灰階總和")],
    "區域裡小亮點或小暗點的總量",
    "white 加總比開運算更亮的部分，black 加總比閉運算更暗的部分。平坦是 0。",
    "Vizion.spot_volume (Region, Image, 'white', 3, Volume)",
)
def spot_volume(region, image, mode, size, *, ctx):
    _picture, _area, _rows, _cols, values, _hat = _spot_area(region, image, size, mode)
    return Result([], [float(values.sum())])


@operator(
    "形態學",
    [P("Region", "區域"), P("Image", "影像")],
    [],
    [
        P("Mode", "white 或 black", "'white'", _SPOT),
        P("MaskSize", "正方形遮罩的邊長，奇數", "3", _ODD),
    ],
    [P("Mean", "平均強度")],
    "區域裡小亮點或小暗點的平均",
    "把頂帽或底帽的灰階在區域裡取平均。平坦是 0。",
    "Vizion.spot_mean (Region, Image, 'white', 3, Mean)",
)
def spot_mean(region, image, mode, size, *, ctx):
    _picture, _area, _rows, _cols, values, _hat = _spot_area(region, image, size, mode)
    return Result([], [float(values.mean())])


@operator(
    "形態學",
    [P("Region", "物件集合"), P("Image", "影像")],
    [],
    [
        P("Mode", "white 或 black", "'white'", _SPOT),
        P("MaskSize", "正方形遮罩的邊長，奇數", "3", _ODD),
    ],
    [P("Volume", "每個物件的灰階總和")],
    "每個物件的小亮點或小暗點總量",
    "不會把物件合併。平坦的物件是 0。結果依物件排列，一個物件也是 tuple。",
    "Vizion.spot_volume_objects (Region, Image, 'white', 3, Volume)",
)
def spot_volume_objects(region, image, mode, size, *, ctx):
    _area, samples = _spot_objects(region, image, size, mode)
    return Result([], [[float(values.sum()) for _index, values in samples]])


@operator(
    "形態學",
    [P("Region", "物件集合"), P("Image", "影像")],
    [P("Selected", "總量落在範圍裡的物件")],
    [
        P("Min", "總量下限"),
        P("Max", "總量上限"),
        P("Mode", "white 或 black", "'white'", _SPOT),
        P("MaskSize", "正方形遮罩的邊長，奇數", "3", _ODD),
    ],
    [],
    "用小亮點或小暗點的總量篩選物件",
    "留下總量在下限和上限之間的物件，兩端都含在內。寫反了會自動對調。沒有符合的就得到空區域。",
    "Vizion.select_spot (Region, Image, Selected, 1, 100, 'white', 3)",
)
def select_spot(region, image, low, high, mode, size, *, ctx):
    area, samples = _spot_objects(region, image, size, mode)
    floor, ceiling = _span(low, high, "Min", "Max")
    kept = [index - 1 for index, values in samples if floor <= float(values.sum()) <= ceiling]
    return Result([_pick_region(area, kept)], [])


@operator(
    "形態學",
    [P("Region", "區域"), P("Image", "影像")],
    [],
    [
        P("Mode", "white 或 black", "'white'", _SPOT),
        P("MaskSize", "正方形遮罩的邊長，奇數", "3", _ODD),
    ],
    [P("Row", "列"), P("Column", "行"), P("Height", "這一點比周圍突出多少")],
    "找區域裡最亮的小亮點或最暗的小暗點",
    "white 找頂帽最高的像素，black 找底帽最高的像素。一樣高時留列最小、再來行最小的那一點。",
    "Vizion.extreme_spot (Region, Image, 'white', 3, Row, Column, Height)",
)
def extreme_spot(region, image, mode, size, *, ctx):
    _picture, _area, rows, cols, values, _hat = _spot_area(region, image, size, mode)
    pick = int(np.argmax(values))
    return Result([], [int(rows[pick]), int(cols[pick]), float(values[pick])])


@operator(
    "形態學",
    [P("Region", "區域"), P("Image", "影像")],
    [P("RegionBand", "強度落在範圍裡的像素")],
    [
        P("Min", "強度下限"),
        P("Max", "強度上限"),
        P("Mode", "white 或 black", "'white'", _SPOT),
        P("MaskSize", "正方形遮罩的邊長，奇數", "3", _ODD),
    ],
    [],
    "留下小亮點或小暗點落在範圍內的像素",
    "white 看頂帽，black 看底帽。下限和上限含在內，寫反了會自動對調。區域外面的像素不會留下。",
    "Vizion.band_spot (Region, Image, RegionBand, 20, 40, 'white', 3)",
)
def band_spot(region, image, low, high, mode, size, *, ctx):
    picture, _kind, hat = _spot_hat(image, size, mode)
    area = expect_region(region, "Region")
    if (picture.width, picture.height) != (area.width, area.height):
        raise ValueError("區域和影像的尺寸要相同")
    floor, ceiling = _span(low, high, "Min", "Max")
    mask = area.union_mask()
    if picture.domain is not None:
        mask = cv2.bitwise_and(mask, picture.domain)
    values = hat.astype(np.float64)
    selected = (mask > 0) & (values >= floor) & (values <= ceiling)
    return region_out(picture.width, picture.height, selected.astype(np.uint8) * 255)


_LINK = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


def _square_kernel(size) -> np.ndarray:
    ksize = _spot_size(size)
    return cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))


def _grow(mask: np.ndarray, marker: np.ndarray, upward: bool) -> np.ndarray:
    current = marker
    kernel = _LINK
    limit = int(mask.shape[0] + mask.shape[1])
    for _step in range(limit):
        moved = cv2.dilate(current, kernel) if upward else cv2.erode(current, kernel)
        nxt = np.minimum(moved, mask) if upward else np.maximum(moved, mask)
        if np.array_equal(nxt, current):
            return nxt
        current = nxt
    raise ValueError("重建沒有收斂")


def _opening_pair(image, size):
    picture = expect_image(image, "Image")
    gray = picture.gray()
    kernel = _square_kernel(size)
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    restored = _grow(gray, cv2.erode(gray, kernel), True)
    return picture, opened, restored


def _closing_pair(image, size):
    picture = expect_image(image, "Image")
    gray = picture.gray()
    kernel = _square_kernel(size)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    restored = _grow(gray, cv2.dilate(gray, kernel), False)
    return picture, closed, restored


def _gain_values(region, picture, delta: np.ndarray):
    area = expect_region(region, "Region")
    if (picture.width, picture.height) != (area.width, area.height):
        raise ValueError("區域和影像的尺寸要相同")
    mask = area.union_mask()
    if picture.domain is not None:
        mask = cv2.bitwise_and(mask, picture.domain)
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError("區域是空的")
    return delta[rows, cols]


@operator(
    "形態學",
    [P("Image", "影像")],
    [P("ImageOpening", "開運算後的影像")],
    [P("MaskSize", "正方形遮罩的邊長，奇數", "3", _ODD)],
    [],
    "正方形開運算",
    "比遮罩細的亮部會消失，不會再沿原圖長回來。比遮罩大的平坦方塊會留下來。3 是 3×3。彩色先轉成亮度。定義域不改。",
    "Vizion.gray_opening_rect (Image, ImageOpening, 3)",
)
def gray_opening_rect(image, size, *, ctx):
    picture, opened, _restored = _opening_pair(image, size)
    return image_out(picture, opened)


@operator(
    "形態學",
    [P("Image", "影像")],
    [P("ImageClosing", "閉運算後的影像")],
    [P("MaskSize", "正方形遮罩的邊長，奇數", "3", _ODD)],
    [],
    "正方形閉運算",
    "比遮罩細的暗部會被填上，不會再沿原圖長回來。比遮罩大的平坦暗塊會留下來。3 是 3×3。彩色先轉成亮度。定義域不改。",
    "Vizion.gray_closing_rect (Image, ImageClosing, 3)",
)
def gray_closing_rect(image, size, *, ctx):
    picture, closed, _restored = _closing_pair(image, size)
    return image_out(picture, closed)


@operator(
    "形態學",
    [P("Image", "影像")],
    [P("ImageRestored", "重建後的影像")],
    [P("MaskSize", "正方形遮罩的邊長，奇數", "3", _ODD)],
    [],
    "開運算後沿著原圖長回亮部",
    "先用正方形侵蝕，比遮罩細的亮部會消失。再沿著原圖膨脹：還連著留下的大塊的細亮部會長回來。單獨的小點沒有起點，不會回來。彩色先轉成亮度。定義域不改。",
    "Vizion.gray_opening_reconstruct (Image, ImageRestored, 3)",
)
def gray_opening_reconstruct(image, size, *, ctx):
    picture, _opened, restored = _opening_pair(image, size)
    return image_out(picture, restored)


@operator(
    "形態學",
    [P("Image", "影像")],
    [P("ImageRestored", "重建後的影像")],
    [P("MaskSize", "正方形遮罩的邊長，奇數", "3", _ODD)],
    [],
    "閉運算後沿著原圖長回暗部",
    "先用正方形膨脹，比遮罩細的暗部會被填上。再沿著原圖侵蝕：還連著留下的大塊暗區的細暗部會長回來。單獨的小洞沒有起點，會被填上。彩色先轉成亮度。定義域不改。",
    "Vizion.gray_closing_reconstruct (Image, ImageRestored, 3)",
)
def gray_closing_reconstruct(image, size, *, ctx):
    picture, _closed, restored = _closing_pair(image, size)
    return image_out(picture, restored)


@operator(
    "形態學",
    [P("Image", "影像")],
    [P("RegionGain", "重建長回來的亮部")],
    [P("MaskSize", "正方形遮罩的邊長，奇數", "3", _ODD)],
    [],
    "留下重建比開運算更亮的像素",
    "這些像素被正方形開運算削掉，但因為連著比較大的亮塊，重建時又長了回來。",
    "Vizion.region_opening_gain (Image, RegionGain, 3)",
)
def region_opening_gain(image, size, *, ctx):
    picture, opened, restored = _opening_pair(image, size)
    selected = restored.astype(np.int16) > opened.astype(np.int16)
    if picture.domain is not None:
        selected &= picture.domain > 0
    return region_out(picture.width, picture.height, selected.astype(np.uint8) * 255)


@operator(
    "形態學",
    [P("Image", "影像")],
    [P("RegionGain", "閉運算填上的細縫")],
    [P("MaskSize", "正方形遮罩的邊長，奇數", "3", _ODD)],
    [],
    "留下閉運算比重建更亮的像素",
    "這些像素被正方形閉運算填亮，但因為連著比較大的暗塊，重建時又回到原來的暗色。",
    "Vizion.region_closing_gain (Image, RegionGain, 3)",
)
def region_closing_gain(image, size, *, ctx):
    picture, closed, restored = _closing_pair(image, size)
    selected = closed.astype(np.int16) > restored.astype(np.int16)
    if picture.domain is not None:
        selected &= picture.domain > 0
    return region_out(picture.width, picture.height, selected.astype(np.uint8) * 255)


@operator(
    "形態學",
    [P("Region", "區域"), P("Image", "影像")],
    [],
    [P("MaskSize", "正方形遮罩的邊長，奇數", "3", _ODD)],
    [P("Volume", "長回來的灰階總和")],
    "區域裡重建比開運算多出來的灰階",
    "每個像素是重建減去開運算，再加總。沒有長回來的部分是 0。",
    "Vizion.volume_opening_gain (Region, Image, 3, Volume)",
)
def volume_opening_gain(region, image, size, *, ctx):
    picture, opened, restored = _opening_pair(image, size)
    delta = restored.astype(np.float64) - opened.astype(np.float64)
    return Result([], [float(_gain_values(region, picture, delta).sum())])


@operator(
    "形態學",
    [P("Region", "區域"), P("Image", "影像")],
    [],
    [P("MaskSize", "正方形遮罩的邊長，奇數", "3", _ODD)],
    [P("Volume", "被填上的灰階總和")],
    "區域裡閉運算比重建多填上的灰階",
    "每個像素是閉運算減去重建，再加總。細縫若連著大塊暗區，這個差就是被填上的量。",
    "Vizion.volume_closing_gain (Region, Image, 3, Volume)",
)
def volume_closing_gain(region, image, size, *, ctx):
    picture, closed, restored = _closing_pair(image, size)
    delta = closed.astype(np.float64) - restored.astype(np.float64)
    return Result([], [float(_gain_values(region, picture, delta).sum())])
