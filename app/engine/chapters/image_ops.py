"""Image arithmetic, channels, and 2D affine geometry.

HomMat2D is stored row by row as six numbers. For images, x is the row and y
is the column, matching HALCON's iconic convention: Tx shifts rows, Ty shifts columns.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..objects import Image, Result
from ..operators import P, as_list, expect_image, image_out, operator, scalar, text, whole


def _tidy(value: float):
    if math.isfinite(value) and abs(value - round(value)) < 1e-8:
        return int(round(value))
    return value


def _combine(image1, image2, fn, mult: float, add: float, what: str) -> Result:
    left = expect_image(image1, "Image1")
    right = expect_image(image2, "Image2")
    if left.data.shape != right.data.shape:
        raise ValueError(f"{what}：兩張影像的尺寸或通道數不同")
    out = fn(left.data.astype(np.float32), right.data.astype(np.float32)) * mult + add
    out = np.clip(out, 0, 255).astype(np.uint8)
    domain = _and_domain(left, right)
    if domain is not None:
        out[domain == 0] = 0
    return Result([Image(out, domain)], [])


def _and_domain(left: Image, right: Image):
    if left.domain is None and right.domain is None:
        return None
    height, width = left.data.shape[:2]
    first = left.domain if left.domain is not None else np.full((height, width), 255, np.uint8)
    second = right.domain if right.domain is not None else np.full((height, width), 255, np.uint8)
    return cv2.bitwise_and(first, second)


def _domain_of(image: Image):
    if image.domain is None:
        return None
    return image.domain.copy()


_INTERP = {
    "nearest_neighbor": cv2.INTER_NEAREST,
    "constant": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
}


def _interpolation(value) -> int:
    name = text(value, "Interpolation").lower()
    if name not in _INTERP:
        raise ValueError("Interpolation 只能是 nearest_neighbor、bilinear、bicubic 或 constant")
    return _INTERP[name]


def _warp(image: Image, matrix: np.ndarray, width: int, height: int, interpolation: int) -> Image:
    data = cv2.warpAffine(
        image.data,
        matrix,
        (width, height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    domain = None
    if image.domain is not None:
        warped = cv2.warpAffine(image.domain, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
        domain = np.where(warped > 0, 255, 0).astype(np.uint8)
        data = data.copy()
        data[domain == 0] = 0
    return Image(data, domain)


def _mat_from(value) -> np.ndarray:
    nums = [float(item) for item in as_list(value)]
    if len(nums) == 9:
        nums = nums[:6]
    if len(nums) != 6:
        raise ValueError("HomMat2D 需要 6 個數，依序是第一列三個、第二列三個。9 個數時忽略最後一列")
    return np.array([[nums[0], nums[1], nums[2]], [nums[3], nums[4], nums[5]], [0.0, 0.0, 1.0]], np.float64)


def _pack(matrix: np.ndarray) -> list:
    return [_tidy(float(matrix[0, 0])), _tidy(float(matrix[0, 1])), _tidy(float(matrix[0, 2])),
            _tidy(float(matrix[1, 0])), _tidy(float(matrix[1, 1])), _tidy(float(matrix[1, 2]))]


def _opencv_forward(matrix: np.ndarray) -> np.ndarray:
    # This OpenCV build moves pixels by the matrix (column, row) -> (column', row').
    return np.array(
        [
            [matrix[1, 1], matrix[1, 0], matrix[1, 2]],
            [matrix[0, 1], matrix[0, 0], matrix[0, 2]],
        ],
        np.float64,
    )


def _apply_points(matrix: np.ndarray, rows, cols):
    out_rows = []
    out_cols = []
    for row, col in zip(as_list(rows), as_list(cols)):
        x = float(row)
        y = float(col)
        out_rows.append(_tidy(float(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2])))
        out_cols.append(_tidy(float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2])))
    if len(out_rows) == 1 and not isinstance(rows, (list, tuple)):
        return out_rows[0], out_cols[0]
    return out_rows, out_cols


def _channel_count(image: Image) -> int:
    return 1 if image.data.ndim == 2 else int(image.data.shape[2])


def _plane(image: Image, channel: int) -> np.ndarray:
    if image.data.ndim == 2:
        if channel != 1:
            raise ValueError("灰階影像只有通道 1")
        return image.data
    # HALCON channel 1 is red. OpenCV stores BGR.
    index = {1: 2, 2: 1, 3: 0}.get(channel)
    if index is None or channel > image.data.shape[2]:
        raise ValueError(f"通道 {channel} 不存在")
    return image.data[:, :, index]


@operator(
    "灰階與濾波",
    [P("Image1", "第一張影像"), P("Image2", "第二張影像")],
    [P("ImageResult", "相加結果")],
    [P("Mult", "乘上的係數"), P("Add", "再加上的值")],
    [],
    "兩張影像相加",
    "每個像素是 (灰階1 + 灰階2) × Mult + Add，超過 0 到 255 會截斷。兩張圖的尺寸和通道數要相同。domain 取交集，外面補 0。",
    "add_image (Image1, Image2, ImageResult, 0.5, 0)",
)
def add_image(image1, image2, mult, add, *, ctx):
    return _combine(image1, image2, lambda a, b: a + b, scalar(mult, "Mult"), scalar(add, "Add"), "add_image")


@operator(
    "灰階與濾波",
    [P("ImageMinuend", "被減影像"), P("ImageSubtrahend", "減去的影像")],
    [P("ImageSub", "相減結果")],
    [P("Mult", "乘上的係數"), P("Add", "再加上的值")],
    [],
    "兩張影像相減",
    "每個像素是 (被減影像 - 減數影像) × Mult + Add，超過 0 到 255 會截斷。",
    "sub_image (Image1, Image2, ImageSub, 1, 128)",
)
def sub_image(image_minuend, image_subtrahend, mult, add, *, ctx):
    return _combine(image_minuend, image_subtrahend, lambda a, b: a - b, scalar(mult, "Mult"), scalar(add, "Add"), "sub_image")


@operator(
    "灰階與濾波",
    [P("Image1", "第一張影像"), P("Image2", "第二張影像")],
    [P("ImageResult", "相乘結果")],
    [P("Mult", "乘上的係數"), P("Add", "再加上的值")],
    [],
    "兩張影像相乘",
    "每個像素是 (灰階1 × 灰階2) × Mult + Add。灰階直接相乘會很大，Mult 通常要小於 1。",
    "mult_image (Image1, Image2, ImageResult, 0.005, 0)",
)
def mult_image(image1, image2, mult, add, *, ctx):
    return _combine(image1, image2, lambda a, b: a * b, scalar(mult, "Mult"), scalar(add, "Add"), "mult_image")


@operator(
    "灰階與濾波",
    [P("Image1", "分子影像"), P("Image2", "分母影像")],
    [P("ImageResult", "相除結果")],
    [P("Mult", "乘上的係數"), P("Add", "再加上的值")],
    [],
    "兩張影像相除",
    "每個像素是 (灰階1 / 灰階2) × Mult + Add。分母是 0 的像素結果是 0。",
    "div_image (Image1, Image2, ImageResult, 1, 0)",
)
def div_image(image1, image2, mult, add, *, ctx):
    def divide(a, b):
        out = np.zeros_like(a)
        np.divide(a, b, out=out, where=b != 0)
        return out

    return _combine(image1, image2, divide, scalar(mult, "Mult"), scalar(add, "Add"), "div_image")


@operator(
    "灰階與濾波",
    [P("Image1", "第一張影像"), P("Image2", "第二張影像")],
    [P("ImageAbsDiff", "絕對差")],
    [P("Mult", "乘上的係數")],
    [],
    "兩張影像的絕對差",
    "每個像素是 |灰階1 - 灰階2| × Mult，超過 255 會截斷。",
    "abs_diff_image (Image1, Image2, ImageAbsDiff, 1)",
)
def abs_diff_image(image1, image2, mult, *, ctx):
    return _combine(image1, image2, lambda a, b: np.abs(a - b), scalar(mult, "Mult"), 0.0, "abs_diff_image")


@operator(
    "灰階與濾波",
    [P("Image1", "第一張影像"), P("Image2", "第二張影像")],
    [P("ImageMin", "每個像素的較小值")],
    [],
    [],
    "每個像素取較小灰階",
    "兩張圖的尺寸和通道數要相同。",
    "min_image (Image1, Image2, ImageMin)",
)
def min_image(image1, image2, *, ctx):
    return _combine(image1, image2, np.minimum, 1.0, 0.0, "min_image")


@operator(
    "灰階與濾波",
    [P("Image1", "第一張影像"), P("Image2", "第二張影像")],
    [P("ImageMax", "每個像素的較大值")],
    [],
    [],
    "每個像素取較大灰階",
    "兩張圖的尺寸和通道數要相同。",
    "max_image (Image1, Image2, ImageMax)",
)
def max_image(image1, image2, *, ctx):
    return _combine(image1, image2, np.maximum, 1.0, 0.0, "max_image")


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [],
    [],
    [P("Channels", "通道數")],
    "通道數",
    "灰階是 1。彩色是 3，通道 1 紅、2 綠、3 藍。",
    "count_channels (Image, Channels)",
)
def count_channels(image, *, ctx):
    return Result([], [_channel_count(expect_image(image, "Image"))])


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageChannel", "單通道影像")],
    [P("Channel", "通道編號，從 1 開始")],
    [],
    "取出一個通道",
    "彩色圖的通道 1 是紅、2 是綠、3 是藍。影像在內部以 BGR 存放，這個編號仍跟 HALCON 相同。",
    "access_channel (Image, Red, 1)",
)
def access_channel(image, channel, *, ctx):
    picture = expect_image(image, "Image")
    plane = _plane(picture, whole(channel, "Channel"))
    return image_out(picture, plane.copy())


@operator(
    "灰階與濾波",
    [P("Image1", "紅通道"), P("Image2", "綠通道"), P("Image3", "藍通道")],
    [P("MultiChannelImage", "彩色影像")],
    [],
    [],
    "三張灰階合成彩色",
    "三張都必須是相同尺寸的灰階圖。順序是紅、綠、藍。",
    "compose3 (Red, Green, Blue, ImageRGB)",
)
def compose3(image1, image2, image3, *, ctx):
    planes = [expect_image(item, name) for item, name in ((image1, "Image1"), (image2, "Image2"), (image3, "Image3"))]
    if any(item.data.ndim != 2 for item in planes):
        raise ValueError("compose3 的三張輸入都要是灰階")
    if any(item.data.shape != planes[0].data.shape for item in planes):
        raise ValueError("三張灰階的尺寸要相同")
    merged = cv2.merge([planes[2].data, planes[1].data, planes[0].data])
    domain = planes[0].domain
    for item in planes[1:]:
        if domain is None:
            domain = item.domain
        elif item.domain is not None:
            domain = cv2.bitwise_and(domain, item.domain)
    return Result([Image(merged, None if domain is None else domain.copy())], [])


@operator(
    "灰階與濾波",
    [P("MultiChannelImage", "彩色影像")],
    [P("Image1", "紅通道"), P("Image2", "綠通道"), P("Image3", "藍通道")],
    [],
    [],
    "彩色拆成紅綠藍",
    "灰階影像不能拆。通道順序是紅、綠、藍。",
    "decompose3 (ImageRGB, Red, Green, Blue)",
)
def decompose3(image, *, ctx):
    picture = expect_image(image, "MultiChannelImage")
    if _channel_count(picture) != 3:
        raise ValueError("decompose3 需要三通道彩色影像")
    red, green, blue = _plane(picture, 1), _plane(picture, 2), _plane(picture, 3)
    domain = _domain_of(picture)
    return Result([Image(red.copy(), domain), Image(green.copy(), None if domain is None else domain.copy()), Image(blue.copy(), None if domain is None else domain.copy())], [])


@operator(
    "灰階與濾波",
    [P("Image1", "原本的影像"), P("Image2", "要接上的通道")],
    [P("ImageExtended", "多一個通道的影像")],
    [],
    [],
    "在後面加一個通道",
    "第二張必須是相同尺寸的灰階。彩色圖再加一通道後，access_channel 的紅綠藍編號不再適用。",
    "append_channel (Image, Gray, Extended)",
)
def append_channel(image1, image2, *, ctx):
    base = expect_image(image1, "Image1")
    extra = expect_image(image2, "Image2")
    if extra.data.ndim != 2 or extra.data.shape[:2] != base.data.shape[:2]:
        raise ValueError("加上的通道必須是相同尺寸的灰階")
    if base.data.ndim == 2:
        merged = cv2.merge([base.data, extra.data])
    else:
        merged = np.dstack([base.data, extra.data])
    return image_out(base, merged)


@operator(
    "幾何",
    [P("Image", "輸入影像")],
    [P("ImageMirror", "鏡像後影像")],
    [P("Mode", "row、column 或 diagonal", choices=["'row'", "'column'", "'diagonal'"])],
    [],
    "影像鏡像",
    "row 上下對調，column 左右對調，diagonal 轉置。轉置後寬高會交換。",
    "mirror_image (Image, ImageMirror, 'column')",
)
def mirror_image(image, mode, *, ctx):
    picture = expect_image(image, "Image")
    kind = text(mode, "Mode").lower()
    if kind == "row":
        data = cv2.flip(picture.data, 0)
        domain = None if picture.domain is None else cv2.flip(picture.domain, 0)
    elif kind == "column":
        data = cv2.flip(picture.data, 1)
        domain = None if picture.domain is None else cv2.flip(picture.domain, 1)
    elif kind == "diagonal":
        data = cv2.transpose(picture.data)
        domain = None if picture.domain is None else cv2.transpose(picture.domain)
    else:
        raise ValueError("Mode 只能是 row、column 或 diagonal")
    return Result([Image(data, domain)], [])


@operator(
    "幾何",
    [P("Image", "輸入影像")],
    [P("ImageRotate", "旋轉後影像")],
    [P("Phi", "角度，弳度"), P("Interpolation", "nearest_neighbor、bilinear、bicubic 或 constant", choices=["'nearest_neighbor'", "'bilinear'", "'bicubic'", "'constant'"])],
    [],
    "繞影像中心旋轉",
    "正角度在畫面上逆時針。畫布大小不變，轉出去的部分切掉，空出來的地方補 0。constant 用最近鄰取樣。",
    "rotate_image (Image, ImageRotate, rad(15), 'bilinear')",
)
def rotate_image(image, phi, interpolation, *, ctx):
    picture = expect_image(image, "Image")
    angle = math.degrees(scalar(phi, "Phi"))
    height, width = picture.data.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return Result([_warp(picture, matrix, width, height, _interpolation(interpolation))], [])


@operator(
    "幾何",
    [P("Image", "輸入影像")],
    [P("ImageZoomed", "縮放後影像")],
    [
        P("ScaleWidth", "寬度倍率"),
        P("ScaleHeight", "高度倍率"),
        P("Interpolation", "nearest_neighbor、bilinear、bicubic 或 constant", choices=["'nearest_neighbor'", "'bilinear'", "'bicubic'", "'constant'"]),
    ],
    [],
    "依倍率縮放影像",
    "ScaleWidth 乘行數，ScaleHeight 乘列數。小於等於 0 會報錯。",
    "zoom_image_factor (Image, ImageZoomed, 0.5, 0.5, 'bilinear')",
)
def zoom_image_factor(image, scale_width, scale_height, interpolation, *, ctx):
    picture = expect_image(image, "Image")
    sx = scalar(scale_width, "ScaleWidth")
    sy = scalar(scale_height, "ScaleHeight")
    if sx <= 0 or sy <= 0:
        raise ValueError("縮放倍率必須大於 0")
    width = max(1, int(round(picture.width * sx)))
    height = max(1, int(round(picture.height * sy)))
    data = cv2.resize(picture.data, (width, height), interpolation=_interpolation(interpolation))
    domain = None
    if picture.domain is not None:
        domain = cv2.resize(picture.domain, (width, height), interpolation=cv2.INTER_NEAREST)
        domain = np.where(domain > 0, 255, 0).astype(np.uint8)
    return Result([Image(data, domain)], [])


@operator(
    "幾何",
    [P("Image", "輸入影像")],
    [P("ImageZoom", "縮放後影像")],
    [
        P("Width", "新的寬度"),
        P("Height", "新的高度"),
        P("Interpolation", "nearest_neighbor、bilinear、bicubic 或 constant", choices=["'nearest_neighbor'", "'bilinear'", "'bicubic'", "'constant'"]),
    ],
    [],
    "縮放到指定尺寸",
    "寬高用像素，至少是 1。",
    "zoom_image_size (Image, ImageZoom, 320, 240, 'bilinear')",
)
def zoom_image_size(image, width, height, interpolation, *, ctx):
    picture = expect_image(image, "Image")
    wide = whole(width, "Width")
    high = whole(height, "Height")
    if wide < 1 or high < 1:
        raise ValueError("寬高至少要 1")
    data = cv2.resize(picture.data, (wide, high), interpolation=_interpolation(interpolation))
    domain = None
    if picture.domain is not None:
        domain = cv2.resize(picture.domain, (wide, high), interpolation=cv2.INTER_NEAREST)
        domain = np.where(domain > 0, 255, 0).astype(np.uint8)
    return Result([Image(data, domain)], [])


@operator(
    "幾何",
    [],
    [],
    [],
    [P("HomMat2D", "單位矩陣，6 個數")],
    "建立單位仿射矩陣",
    "用在影像時，前三個數作用在列，後三個數作用在行。x 是列、y 是行。",
    "hom_mat2d_identity (HomMat2D)",
)
def hom_mat2d_identity(*, ctx):
    return Result([], [[1, 0, 0, 0, 1, 0]])


def _translate(matrix: np.ndarray, tx: float, ty: float, local: bool) -> np.ndarray:
    shift = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], np.float64)
    return matrix @ shift if local else shift @ matrix


def _around(matrix: np.ndarray, linear: np.ndarray, px: float, py: float, local: bool) -> np.ndarray:
    to_origin = np.array([[1.0, 0.0, -px], [0.0, 1.0, -py], [0.0, 0.0, 1.0]])
    back = np.array([[1.0, 0.0, px], [0.0, 1.0, py], [0.0, 0.0, 1.0]])
    piece = back @ linear @ to_origin
    return matrix @ piece if local else piece @ matrix


@operator(
    "幾何",
    [],
    [],
    [P("HomMat2D", "原本的矩陣"), P("Tx", "沿 x 平移。影像上是列"), P("Ty", "沿 y 平移。影像上是行")],
    [P("HomMat2DTranslate", "平移後的矩陣")],
    "在固定座標系加上平移",
    "用在影像時 Tx 加在列、Ty 加在行。這是全域平移，先做原來的矩陣，再平移。",
    "hom_mat2d_translate (HomMat2D, 20, 0, HomMat2DTranslate)",
)
def hom_mat2d_translate(hom_mat2d, tx, ty, *, ctx):
    return Result([], [_pack(_translate(_mat_from(hom_mat2d), scalar(tx, "Tx"), scalar(ty, "Ty"), False))])


@operator(
    "幾何",
    [],
    [],
    [P("HomMat2D", "原本的矩陣"), P("Tx", "沿局部 x 平移"), P("Ty", "沿局部 y 平移")],
    [P("HomMat2DTranslate", "平移後的矩陣")],
    "在矩陣自己的座標系加上平移",
    "先平移，再做原來的矩陣。影像上 x 仍是列、y 仍是行。",
    "hom_mat2d_translate_local (HomMat2D, 20, 0, HomMat2DTranslate)",
)
def hom_mat2d_translate_local(hom_mat2d, tx, ty, *, ctx):
    return Result([], [_pack(_translate(_mat_from(hom_mat2d), scalar(tx, "Tx"), scalar(ty, "Ty"), True))])


def _rotation(phi: float) -> np.ndarray:
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@operator(
    "幾何",
    [],
    [],
    [P("HomMat2D", "原本的矩陣"), P("Phi", "角度，弳度"), P("Px", "旋轉中心 x。影像上是列"), P("Py", "旋轉中心 y。影像上是行")],
    [P("HomMat2DRotate", "旋轉後的矩陣")],
    "繞一個點旋轉",
    "正角度在 x 向右、y 向下的座標裡逆時針。用在影像時，點的座標是 (列, 行)。這是全域旋轉。",
    "hom_mat2d_rotate (HomMat2D, rad(90), 0, 0, HomMat2DRotate)",
)
def hom_mat2d_rotate(hom_mat2d, phi, px, py, *, ctx):
    matrix = _around(_mat_from(hom_mat2d), _rotation(scalar(phi, "Phi")), scalar(px, "Px"), scalar(py, "Py"), False)
    return Result([], [_pack(matrix)])


@operator(
    "幾何",
    [],
    [],
    [P("HomMat2D", "原本的矩陣"), P("Phi", "角度，弳度"), P("Px", "旋轉中心 x"), P("Py", "旋轉中心 y")],
    [P("HomMat2DRotate", "旋轉後的矩陣")],
    "在局部座標系繞點旋轉",
    "先繞局部的點旋轉，再做原來的矩陣。",
    "hom_mat2d_rotate_local (HomMat2D, rad(90), 0, 0, HomMat2DRotate)",
)
def hom_mat2d_rotate_local(hom_mat2d, phi, px, py, *, ctx):
    matrix = _around(_mat_from(hom_mat2d), _rotation(scalar(phi, "Phi")), scalar(px, "Px"), scalar(py, "Py"), True)
    return Result([], [_pack(matrix)])


def _scale(sx: float, sy: float) -> np.ndarray:
    return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])


@operator(
    "幾何",
    [],
    [],
    [P("HomMat2D", "原本的矩陣"), P("Sx", "x 倍率"), P("Sy", "y 倍率"), P("Px", "縮放中心 x"), P("Py", "縮放中心 y")],
    [P("HomMat2DScale", "縮放後的矩陣")],
    "繞一個點縮放",
    "用在影像時 Sx 放大列方向、Sy 放大行方向。倍率 0 會報錯。",
    "hom_mat2d_scale (HomMat2D, 2, 2, 0, 0, HomMat2DScale)",
)
def hom_mat2d_scale(hom_mat2d, sx, sy, px, py, *, ctx):
    scale_x, scale_y = scalar(sx, "Sx"), scalar(sy, "Sy")
    if scale_x == 0 or scale_y == 0:
        raise ValueError("縮放倍率不能是 0")
    matrix = _around(_mat_from(hom_mat2d), _scale(scale_x, scale_y), scalar(px, "Px"), scalar(py, "Py"), False)
    return Result([], [_pack(matrix)])


@operator(
    "幾何",
    [],
    [],
    [P("HomMat2D", "原本的矩陣"), P("Sx", "x 倍率"), P("Sy", "y 倍率"), P("Px", "縮放中心 x"), P("Py", "縮放中心 y")],
    [P("HomMat2DScale", "縮放後的矩陣")],
    "在局部座標系繞點縮放",
    "先做局部縮放，再做原來的矩陣。",
    "hom_mat2d_scale_local (HomMat2D, 2, 2, 0, 0, HomMat2DScale)",
)
def hom_mat2d_scale_local(hom_mat2d, sx, sy, px, py, *, ctx):
    scale_x, scale_y = scalar(sx, "Sx"), scalar(sy, "Sy")
    if scale_x == 0 or scale_y == 0:
        raise ValueError("縮放倍率不能是 0")
    matrix = _around(_mat_from(hom_mat2d), _scale(scale_x, scale_y), scalar(px, "Px"), scalar(py, "Py"), True)
    return Result([], [_pack(matrix)])


@operator(
    "幾何",
    [],
    [],
    [
        P("HomMat2D", "原本的矩陣"),
        P("Theta", "傾斜角，弳度"),
        P("Axis", "沿哪一軸傾斜", choices=["'x'", "'y'"]),
        P("Px", "中心 x"),
        P("Py", "中心 y"),
    ],
    [P("HomMat2DSlant", "傾斜後的矩陣")],
    "繞一個點做剪切",
    "axis 為 x 時，y 加上 tan(Theta) 乘上相對 x。axis 為 y 時反過來。這是全域剪切。",
    "hom_mat2d_slant (HomMat2D, rad(10), 'x', 0, 0, HomMat2DSlant)",
)
def hom_mat2d_slant(hom_mat2d, theta, axis, px, py, *, ctx):
    shear = math.tan(scalar(theta, "Theta"))
    direction = text(axis, "Axis").lower()
    if direction == "x":
        linear = np.array([[1.0, 0.0, 0.0], [shear, 1.0, 0.0], [0.0, 0.0, 1.0]])
    elif direction == "y":
        linear = np.array([[1.0, shear, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    else:
        raise ValueError("Axis 只能是 x 或 y")
    matrix = _around(_mat_from(hom_mat2d), linear, scalar(px, "Px"), scalar(py, "Py"), False)
    return Result([], [_pack(matrix)])


@operator(
    "幾何",
    [],
    [],
    [P("HomMat2DLeft", "後套用的矩陣"), P("HomMat2DRight", "先套用的矩陣")],
    [P("HomMat2DCompose", "合成矩陣")],
    "兩個矩陣相乘",
    "先做右邊，再做左邊。點的新座標是 Left × Right × 點。",
    "hom_mat2d_compose (HomMat2DLeft, HomMat2DRight, HomMat2DCompose)",
)
def hom_mat2d_compose(hom_mat2d_left, hom_mat2d_right, *, ctx):
    return Result([], [_pack(_mat_from(hom_mat2d_left) @ _mat_from(hom_mat2d_right))])


@operator(
    "幾何",
    [],
    [],
    [P("HomMat2D", "原本的矩陣")],
    [P("HomMat2DInvert", "反矩陣")],
    "仿射矩陣取反",
    "反矩陣再套用一次就回到原點。行列式太接近 0 時會報錯。",
    "hom_mat2d_invert (HomMat2D, HomMat2DInvert)",
)
def hom_mat2d_invert(hom_mat2d, *, ctx):
    matrix = _mat_from(hom_mat2d)
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("這個矩陣不能反轉")
    return Result([], [_pack(np.linalg.inv(matrix))])


@operator(
    "幾何",
    [],
    [],
    [P("HomMat2D", "矩陣")],
    [P("Determinant", "左上 2×2 的行列式")],
    "仿射矩陣的行列式",
    "看線性部分的面積倍率。負值表示有翻轉。",
    "hom_mat2d_determinant (HomMat2D, Determinant)",
)
def hom_mat2d_determinant(hom_mat2d, *, ctx):
    matrix = _mat_from(hom_mat2d)
    return Result([], [_tidy(float(np.linalg.det(matrix[:2, :2])))])


@operator(
    "幾何",
    [],
    [],
    [P("HomMat2D", "矩陣"), P("Row", "列"), P("Col", "行")],
    [P("RowTrans", "轉換後的列"), P("ColTrans", "轉換後的行")],
    "用矩陣轉換影像座標",
    "列是 x、行是 y。Row 和 Col 可以是 tuple，兩個輸出會一起變長。",
    "affine_trans_pixel (HomMat2D, 10, 20, RowTrans, ColTrans)",
)
def affine_trans_pixel(hom_mat2d, row, col, *, ctx):
    rows, cols = _apply_points(_mat_from(hom_mat2d), row, col)
    return Result([], [rows, cols])


@operator(
    "幾何",
    [],
    [],
    [P("HomMat2D", "矩陣"), P("Px", "x。用在影像時填列"), P("Py", "y。用在影像時填行")],
    [P("Qx", "轉換後的 x"), P("Qy", "轉換後的 y")],
    "用矩陣轉換點",
    "和 affine_trans_pixel 用同一套矩陣。用在影像時 Px 填列、Py 填行，Qx 是新的列、Qy 是新的行。",
    "affine_trans_point_2d (HomMat2D, 10, 20, Qx, Qy)",
)
def affine_trans_point_2d(hom_mat2d, px, py, *, ctx):
    qx, qy = _apply_points(_mat_from(hom_mat2d), px, py)
    return Result([], [qx, qy])


@operator(
    "幾何",
    [P("Image", "輸入影像")],
    [P("ImageAffineTrans", "轉換後影像")],
    [
        P("HomMat2D", "仿射矩陣"),
        P("Interpolation", "nearest_neighbor、bilinear、bicubic 或 constant", choices=["'nearest_neighbor'", "'bilinear'", "'bicubic'", "'constant'"]),
        P("AdaptImageSize", "true 會把畫布加大到裝得下", choices=["'true'", "'false'"]),
    ],
    [],
    "用仿射矩陣轉換影像",
    "矩陣的 x 是列、y 是行。false 保持原圖大小，轉出去的部分切掉。true 會平移畫布，讓四個角都落在新圖裡。影像外補 0。",
    "affine_trans_image (Image, ImageAffineTrans, HomMat2D, 'bilinear', 'false')",
)
def affine_trans_image(image, hom_mat2d, interpolation, adapt_image_size, *, ctx):
    picture = expect_image(image, "Image")
    matrix = _mat_from(hom_mat2d)
    height, width = picture.data.shape[:2]
    adapt = text(adapt_image_size, "AdaptImageSize").lower() in ("true", "1")
    shift_row = shift_col = 0.0
    out_w, out_h = width, height
    if adapt:
        corners = []
        for row, col in ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)):
            corners.append(_apply_points(matrix, row, col))
        rows = [item[0] for item in corners]
        cols = [item[1] for item in corners]
        shift_row = -min(rows)
        shift_col = -min(cols)
        out_h = max(1, int(math.floor(max(rows) + shift_row)) + 1)
        out_w = max(1, int(math.floor(max(cols) + shift_col)) + 1)
        matrix = _translate(matrix, shift_row, shift_col, False)
    return Result([_warp(picture, _opencv_forward(matrix), out_w, out_h, _interpolation(interpolation))], [])
