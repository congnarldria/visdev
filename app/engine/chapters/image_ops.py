"""Image arithmetic, channels, and 2D affine geometry.

HomMat2D is stored row by row as six numbers. For images, x is the row and y
is the column, matching HALCON's iconic convention: Tx shifts rows, Ty shifts columns.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..objects import Image, Region, Result
from ..operators import ODD_SIZES, P, as_list, expect_image, expect_region, image_out, operator, region_out, scalar, text, whole


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


def _rect_mask(height: int, width: int) -> np.ndarray:
    if height < 1 or width < 1:
        raise ValueError("遮罩的高和寬至少要 1")
    return cv2.getStructuringElement(cv2.MORPH_RECT, (width, height))


def _gray_morph(image, mask_height, mask_width, mode: str) -> Result:
    picture = expect_image(image, "Image")
    kernel = _rect_mask(whole(mask_height, "MaskHeight"), whole(mask_width, "MaskWidth"))
    work = picture.data
    if mode in ("erode", "open") and picture.domain is not None:
        work = picture.data.copy()
        work[picture.domain == 0] = 255
    if mode == "erode":
        out = cv2.erode(work, kernel)
    elif mode == "dilate":
        out = cv2.dilate(work, kernel)
    elif mode == "open":
        out = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel)
    else:
        out = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)
    if picture.domain is not None:
        out = out.copy()
        out[picture.domain == 0] = 0
    return image_out(picture, out)


def _bind_gray(name, mode, short, detail):
    def fn(image, mask_height, mask_width, *, ctx):
        return _gray_morph(image, mask_height, mask_width, mode)

    fn.__name__ = name
    operator(
        "形態學",
        [P("Image", "輸入影像")],
        [P("ImageMorph", "結果影像")],
        [P("MaskHeight", "遮罩高度，像素"), P("MaskWidth", "遮罩寬度，像素")],
        [],
        short,
        detail,
        f"{name} (Image, ImageMorph, 5, 5)",
    )(fn)
    return fn


_bind_gray(
    "gray_erosion_rect",
    "erode",
    "灰階矩形侵蝕",
    "每個像素換成遮罩裡最暗的值。偶數遮罩的原點在中心偏左上，可能和 HALCON 差一像素。定義域外先當成 255，避免邊界被拉暗。",
)
_bind_gray(
    "gray_dilation_rect",
    "dilate",
    "灰階矩形膨脹",
    "每個像素換成遮罩裡最亮的值。定義域外維持 0。",
)
_bind_gray(
    "gray_opening_rect",
    "open",
    "灰階矩形開運算",
    "先侵蝕再膨脹，用來去掉比遮罩小的亮點。",
)
_bind_gray(
    "gray_closing_rect",
    "close",
    "灰階矩形閉運算",
    "先膨脹再侵蝕，用來補上比遮罩小的暗洞。",
)


def _circle_gray(image, radius, mode: str):
    picture = expect_image(image, "Image")
    size = int(round(scalar(radius, "Radius")))
    if size < 0:
        raise ValueError("半徑不能是負的")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size * 2 + 1, size * 2 + 1))
    work = picture.data
    if mode in ("erode", "open") and picture.domain is not None:
        work = picture.data.copy()
        work[picture.domain == 0] = 255
    if mode == "erode":
        out = cv2.erode(work, kernel)
    elif mode == "dilate":
        out = cv2.dilate(work, kernel)
    elif mode == "open":
        out = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel)
    else:
        out = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)
    if picture.domain is not None:
        out = out.copy()
        out[picture.domain == 0] = 0
    return image_out(picture, out)


def _bind_circle(name, mode, short, detail):
    def fn(image, radius, *, ctx):
        return _circle_gray(image, radius, mode)

    fn.__name__ = name
    operator(
        "形態學",
        [P("Image", "輸入影像")],
        [P("ImageMorph", "結果影像")],
        [P("Radius", "圓形遮罩半徑")],
        [],
        short,
        detail,
        f"{name} (Image, ImageMorph, 2)",
    )(fn)
    return fn


_bind_circle("gray_erosion_circle", "erode", "灰階圓形侵蝕", "結構元素是實心圓。半徑會四捨五入。定義域外先當成 255，避免邊界被拉暗。")
_bind_circle("gray_dilation_circle", "dilate", "灰階圓形膨脹", "結構元素是實心圓。定義域外維持 0。")
_bind_circle("gray_opening_circle", "open", "灰階圓形開運算", "先用圓侵蝕再膨脹，去掉比圓小的亮點。")
_bind_circle("gray_closing_circle", "close", "灰階圓形閉運算", "先用圓膨脹再侵蝕，補上比圓小的暗洞。")


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageScaleMax", "拉到 0 到 255 的影像")],
    [],
    [],
    "把對比拉滿",
    "定義域裡的最小值變成 0、最大值變成 255。整張一樣亮時輸出 0。彩色圖每個通道分開拉。",
    "scale_image_max (Image, ImageScaleMax)",
)
def scale_image_max(image, *, ctx):
    picture = expect_image(image, "Image")
    data = picture.data.astype(np.float32)
    domain = _domain_of(picture)

    def stretch(plane, valid):
        if valid is None or int(valid.sum()) == 0:
            values = plane.reshape(-1)
        else:
            values = plane[valid > 0]
        lo = float(values.min()) if values.size else 0.0
        hi = float(values.max()) if values.size else 0.0
        if hi <= lo:
            return np.zeros(plane.shape, np.uint8)
        out = (plane - lo) * (255.0 / (hi - lo))
        return np.clip(out, 0, 255).astype(np.uint8)

    if data.ndim == 2:
        out = stretch(data, domain)
    else:
        out = np.dstack([stretch(data[:, :, index], domain) for index in range(data.shape[2])])
    if domain is not None:
        out = out.copy()
        out[domain == 0] = 0
    return image_out(picture, out)


@operator(
    "檔案",
    [],
    [P("Image", "新影像")],
    [
        P("Type", "這版只做 byte", "'byte'", ["'byte'"]),
        P("Width", "寬度"),
        P("Height", "高度"),
    ],
    [],
    "建立全黑影像",
    "Type 只能是 byte。影像裡每個像素都是 0。",
    "gen_image_const (Image, 'byte', 640, 480)",
)
def gen_image_const(kind, width, height, *, ctx):
    if text(kind, "Type").lower() != "byte":
        raise ValueError("Type 這版只接受 byte")
    wide = whole(width, "Width")
    high = whole(height, "Height")
    if wide < 1 or high < 1:
        raise ValueError("寬高至少要 1")
    return Result([Image(np.zeros((high, wide), np.uint8))], [])


@operator(
    "灰階與濾波",
    [P("Image", "用來決定尺寸的影像")],
    [P("ImageCleared", "填滿後的影像")],
    [P("Grayval", "填入的灰階，0 到 255")],
    [],
    "用固定灰階填滿一張新影像",
    "尺寸跟輸入相同。定義域是整張圖，不沿用輸入的定義域。",
    "gen_image_proto (Image, ImageCleared, 0)",
)
def gen_image_proto(image, grayval, *, ctx):
    picture = expect_image(image, "Image")
    value = int(np.clip(round(scalar(grayval, "Grayval")), 0, 255))
    data = np.full_like(picture.data, value)
    return Result([Image(data)], [])


@operator(
    "區域",
    [P("Region", "要畫上去的區域"), P("Image", "底圖")],
    [P("ImageResult", "畫完的影像")],
    [
        P("Grayval", "填入的灰階。彩色可給三個數，順序是紅、綠、藍"),
        P("Type", "fill 填滿，margin 只畫邊", "'fill'", ["'fill'", "'margin'"]),
    ],
    [],
    "把區域畫進影像",
    "區域和影像的寬高要相同。margin 用一像素的外框。",
    "paint_region (Region, Image, ImageResult, 255, 'fill')",
)
def paint_region(region, image, grayval, kind, *, ctx):
    picture = expect_image(image, "Image")
    area = expect_region(region, "Region")
    if (area.width, area.height) != (picture.width, picture.height):
        raise ValueError("區域和影像的尺寸要相同")
    mode = text(kind, "Type").lower()
    if mode not in ("fill", "margin"):
        raise ValueError("Type 只能是 fill 或 margin")
    mask = area.union_mask()
    if mode == "margin" and int(cv2.countNonZero(mask)):
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        mask = cv2.subtract(mask, cv2.erode(mask, kernel))
    data = picture.data.copy()
    values = [float(item) for item in as_list(grayval)]
    if data.ndim == 2:
        if len(values) != 1:
            raise ValueError("灰階圖的 Grayval 只能有一個數")
        data[mask > 0] = int(np.clip(round(values[0]), 0, 255))
    else:
        if len(values) == 1:
            values = values * 3
        if len(values) != 3:
            raise ValueError("彩色圖的 Grayval 請給一個數，或依紅、綠、藍給三個數")
        stored = [values[2], values[1], values[0]]
        for index, value in enumerate(stored):
            plane = data[:, :, index]
            plane[mask > 0] = int(np.clip(round(value), 0, 255))
            data[:, :, index] = plane
    return image_out(picture, data)


_COLOR_SPACES = ["'hsv'", "'lab'", "'yuv'"]


@operator(
    "灰階與濾波",
    [P("ImageRed", "紅"), P("ImageGreen", "綠"), P("ImageBlue", "藍")],
    [P("ImageResult1", "第一結果"), P("ImageResult2", "第二結果"), P("ImageResult3", "第三結果")],
    [P("ColorSpace", "hsv、lab 或 yuv", "'hsv'", _COLOR_SPACES)],
    [],
    "把紅綠藍轉成其他色彩空間",
    "三張輸入都要是相同尺寸的灰階。hsv 的 H 是 0 到 180。lab 的 a、b 以 128 為 0。這是 OpenCV 的數值，不是 HALCON 的實數範圍。",
    "trans_from_rgb (Red, Green, Blue, H, S, V, 'hsv')",
)
def trans_from_rgb(image_red, image_green, image_blue, color_space, *, ctx):
    planes = [expect_image(item, name) for item, name in ((image_red, "ImageRed"), (image_green, "ImageGreen"), (image_blue, "ImageBlue"))]
    if any(item.data.ndim != 2 for item in planes) or any(item.data.shape != planes[0].data.shape for item in planes):
        raise ValueError("三張輸入都要是相同尺寸的灰階")
    space = text(color_space, "ColorSpace").lower()
    codes = {"hsv": cv2.COLOR_RGB2HSV, "lab": cv2.COLOR_RGB2LAB, "yuv": cv2.COLOR_RGB2YUV}
    if space not in codes:
        raise ValueError("ColorSpace 只能是 hsv、lab 或 yuv")
    converted = cv2.cvtColor(cv2.merge([planes[0].data, planes[1].data, planes[2].data]), codes[space])
    domain = _domain_of(planes[0])
    outputs = [Image(converted[:, :, index].copy(), None if domain is None else domain.copy()) for index in range(3)]
    return Result(outputs, [])


def _odd_size(value, name: str) -> int:
    size = whole(value, name)
    if size < 1:
        raise ValueError(f"{name} 至少要 1")
    return size | 1


def _se_kernel(struct) -> np.ndarray:
    area = expect_region(struct, "StructElement")
    mask = area.union_mask()
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("結構元素是空的")
    return (mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] > 0).astype(np.uint8)


@operator(
    "特徵",
    [P("Image", "輸入影像")],
    [],
    [],
    [P("Width", "寬度"), P("Height", "高度")],
    "讀取影像寬高",
    "寬是行數，高是列數。",
    "get_image_size (Image, Width, Height)",
)
def get_image_size(image, *, ctx):
    picture = expect_image(image, "Image")
    return Result([], [picture.width, picture.height])


@operator(
    "特徵",
    [P("Image", "輸入影像")],
    [],
    [P("Row", "列"), P("Column", "行")],
    [P("Grayval", "灰階")],
    "讀取像素灰階",
    "座標會四捨五入到像素，超出影像會失敗。灰階圖一個點回傳一個數。彩色圖回傳紅、綠、藍。多個點時灰階是 tuple，彩色是依點排好的紅、綠、藍。",
    "get_grayval (Image, 10, 20, Grayval)",
)
def get_grayval(image, row, column, *, ctx):
    picture = expect_image(image, "Image")
    rows = [int(round(float(item))) for item in as_list(row)]
    cols = [int(round(float(item))) for item in as_list(column)]
    if len(rows) != len(cols) or not rows:
        raise ValueError("Row 和 Column 的長度要相同，而且不能是空的")
    if any(item < 0 or item >= picture.height for item in rows) or any(item < 0 or item >= picture.width for item in cols):
        raise ValueError("座標超出影像")
    if picture.data.ndim == 2:
        values = [int(picture.data[r, c]) for r, c in zip(rows, cols)]
        if len(values) == 1 and not isinstance(row, (list, tuple)):
            return Result([], [values[0]])
        return Result([], [values])
    if picture.data.shape[2] != 3:
        raise ValueError("get_grayval 只接受灰階或三通道彩色")
    values = []
    for r, c in zip(rows, cols):
        blue, green, red = (int(item) for item in picture.data[r, c, :3])
        values.extend([red, green, blue])
    return Result([], [values])


@operator(
    "特徵",
    [P("Regions", "要統計的區域"), P("Image", "影像")],
    [],
    [P("Percent", "兩邊合計要忽略的百分比", "0")],
    [P("Min", "最小灰階"), P("Max", "最大灰階"), P("Range", "最大減最小")],
    "區域裡的最小和最大灰階",
    "Percent 0 用真正的最小和最大。Percent 10 會丟掉最暗和最亮各 5%。多個物件合併成一個範圍。彩色影像先轉成亮度。",
    "min_max_gray (Regions, Image, 0, Min, Max, Range)",
)
def min_max_gray(region, image, percent, *, ctx):
    picture = expect_image(image, "Image")
    area = expect_region(region, "Regions")
    if (picture.width, picture.height) != (area.width, area.height):
        raise ValueError("影像和區域的尺寸不同")
    ratio = scalar(percent, "Percent")
    if ratio < 0 or ratio > 100:
        raise ValueError("Percent 要在 0 到 100")
    gray = picture.gray()
    mask = area.union_mask()
    if picture.domain is not None:
        mask = cv2.bitwise_and(mask, picture.domain)
    values = gray[mask > 0]
    if values.size == 0:
        raise ValueError("區域是空的")
    if ratio == 0:
        low = float(values.min())
        high = float(values.max())
    else:
        low = float(np.percentile(values, ratio / 2.0))
        high = float(np.percentile(values, 100.0 - ratio / 2.0))
    return Result([], [low, high, high - low])


@operator(
    "區域",
    [P("Image", "輸入影像")],
    [P("ImageFull", "定義域恢復成整張圖")],
    [],
    [],
    "恢復整張定義域",
    "像素不變，只拿掉 reduce_domain 或 change_domain 設過的範圍。",
    "full_domain (Image, ImageFull)",
)
def full_domain(image, *, ctx):
    picture = expect_image(image, "Image").copy()
    picture.domain = None
    return Result([picture], [])


@operator(
    "區域",
    [P("Image", "輸入影像")],
    [P("ImagePart", "裁到定義域的影像")],
    [],
    [],
    "裁切到定義域",
    "沒有限制定義域時，結果就是原圖。定義域是空的會失敗。",
    "crop_domain (Image, ImagePart)",
)
def crop_domain(image, *, ctx):
    picture = expect_image(image, "Image")
    if picture.domain is None:
        return Result([picture.copy()], [])
    ys, xs = np.where(picture.domain > 0)
    if len(ys) == 0:
        raise ValueError("定義域是空的")
    r1, r2 = int(ys.min()), int(ys.max())
    c1, c2 = int(xs.min()), int(xs.max())
    data = picture.data[r1 : r2 + 1, c1 : c2 + 1].copy()
    domain = picture.domain[r1 : r2 + 1, c1 : c2 + 1].copy()
    return Result([Image(data, domain)], [])


@operator(
    "區域",
    [P("Image", "輸入影像"), P("Region", "新的定義域")],
    [P("ImageNew", "換過定義域的影像")],
    [],
    [],
    "換成新的定義域",
    "直接換成這個區域，不和舊的定義域做交集。reduce_domain 會和現有定義域取交集。",
    "change_domain (Image, Region, ImageNew)",
)
def change_domain(image, region, *, ctx):
    picture = expect_image(image, "Image").copy()
    area = expect_region(region, "Region")
    if (picture.width, picture.height) != (area.width, area.height):
        raise ValueError("影像和區域的尺寸不同")
    picture.domain = area.union_mask()
    return Result([picture], [])


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageIlluminate", "校正後影像")],
    [P("MaskWidth", "局部平均的寬"), P("MaskHeight", "局部平均的高"), P("Factor", "0 完全拉平，1 保持原圖")],
    [],
    "拉平緩慢的照明變化",
    "輸出 = (原圖 − 局部平均) + 局部平均 × Factor + 整張平均 × (1 − Factor)。Factor 1 就是原圖，Factor 0 去掉局部明暗。偶數遮罩會加 1 變成奇數。這和 emphasize 不同，emphasize 是把高頻加回原圖。",
    "illuminate (Image, ImageIlluminate, 31, 31, 0)",
)
def illuminate(image, mask_width, mask_height, factor, *, ctx):
    picture = expect_image(image, "Image")
    width = _odd_size(mask_width, "MaskWidth")
    height = _odd_size(mask_height, "MaskHeight")
    amount = scalar(factor, "Factor")
    src = picture.data.astype(np.float32)
    domain = picture.domain

    def correct(plane):
        if domain is None:
            fill = float(plane.mean())
            work = plane
        else:
            valid = domain > 0
            fill = float(plane[valid].mean()) if int(valid.sum()) else 0.0
            work = plane.copy()
            work[~valid] = fill
        blur = cv2.blur(work, (width, height))
        return (plane - blur) + blur * amount + fill * (1.0 - amount)

    if src.ndim == 2:
        out = correct(src)
    else:
        out = np.dstack([correct(src[:, :, index]) for index in range(src.shape[2])])
    out = np.clip(out, 0, 255).astype(np.uint8)
    if domain is not None:
        out = out.copy()
        out[domain == 0] = 0
    return image_out(picture, out)


def _gray_hat(image, struct, mode: str) -> Result:
    picture = expect_image(image, "Image")
    kernel = _se_kernel(struct)
    op = cv2.MORPH_TOPHAT if mode == "top" else cv2.MORPH_BLACKHAT
    out = cv2.morphologyEx(picture.data, op, kernel)
    if picture.domain is not None:
        out = out.copy()
        out[picture.domain == 0] = 0
    return image_out(picture, out)


@operator(
    "形態學",
    [P("Image", "輸入影像"), P("StructElement", "結構元素")],
    [P("ImageTopHat", "比開運算更亮的部分")],
    [],
    [],
    "灰階頂帽",
    "原圖減去開運算，用來抓比結構元素小的亮點。結構元素的原點是它外框的中心，和區域形態學一樣。定義域外當成 0，邊緣可能有假亮點。",
    "gray_tophat (Image, StructElement, ImageTopHat)",
)
def gray_tophat(image, struct_element, *, ctx):
    return _gray_hat(image, struct_element, "top")


@operator(
    "形態學",
    [P("Image", "輸入影像"), P("StructElement", "結構元素")],
    [P("ImageBotHat", "比閉運算更暗的部分")],
    [],
    [],
    "灰階底帽",
    "閉運算減去原圖，用來抓比結構元素小的暗點。原點是結構元素外框的中心。",
    "gray_bothat (Image, StructElement, ImageBotHat)",
)
def gray_bothat(image, struct_element, *, ctx):
    return _gray_hat(image, struct_element, "bottom")


_LAPLACE_TYPES = ["'absolute'", "'signed'"]
_LAPLACE_MASKS = ["'n_4'", "'n_8'"]


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageLaplace", "拉普拉斯結果")],
    [
        P("ResultType", "absolute 取絕對值，signed 以 128 為 0", "'absolute'", _LAPLACE_TYPES),
        P("MaskSize", "3、5 或 7", "3"),
        P("FilterMask", "n_4 或 n_8", "'n_4'", _LAPLACE_MASKS),
    ],
    [],
    "拉普拉斯",
    "3×3 的 n_4 是上下左右，n_8 再包含對角。5 和 7 用 OpenCV 的拉普拉斯，兩種 FilterMask 相同。signed 會加上 128 再夾到 0 到 255。",
    "laplace (Image, ImageLaplace, 'absolute', 3, 'n_4')",
)
def laplace(image, result_type, mask_size, filter_mask, *, ctx):
    picture = expect_image(image, "Image")
    kind = text(result_type, "ResultType").lower()
    if kind not in ("absolute", "signed"):
        raise ValueError("ResultType 只能是 absolute 或 signed")
    size = whole(mask_size, "MaskSize")
    if size not in (3, 5, 7):
        raise ValueError("MaskSize 只能是 3、5 或 7")
    mode = text(filter_mask, "FilterMask").lower()
    if mode not in ("n_4", "n_8"):
        raise ValueError("FilterMask 只能是 n_4 或 n_8")
    src = picture.data.astype(np.float32)
    if size == 3 and mode == "n_4":
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
        response = cv2.filter2D(src, cv2.CV_32F, kernel)
    elif size == 3:
        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float32)
        response = cv2.filter2D(src, cv2.CV_32F, kernel)
    else:
        response = cv2.Laplacian(src, cv2.CV_32F, ksize=size)
    if kind == "absolute":
        out = np.clip(np.abs(response), 0, 255)
    else:
        out = np.clip(response + 128, 0, 255)
    return image_out(picture, out.astype(np.uint8))


@operator(
    "幾何",
    [P("Image", "輸入影像")],
    [P("PolarTransImage", "展開後的影像")],
    [
        P("Row", "極座標中心列"),
        P("Column", "極座標中心行"),
        P("AngleStart", "起始角，弳度"),
        P("AngleEnd", "結束角，弳度"),
        P("RadiusStart", "內半徑"),
        P("RadiusEnd", "外半徑"),
        P("Width", "角度方向的取樣數，變成寬度"),
        P("Height", "半徑方向的取樣數，變成高度"),
        P("Interpolation", "nearest_neighbor、bilinear、bicubic 或 constant", "'bilinear'", ["'nearest_neighbor'", "'bilinear'", "'bicubic'", "'constant'"]),
    ],
    [],
    "把環形區域展開成長方形",
    "角度 0 朝右，正角度在畫面上順時針。寬度是角度取樣，高度是半徑取樣。影像外補 0。",
    "polar_trans_image_ext (Image, PolarTransImage, 60, 60, 0, rad(360), 10, 28, 64, 16, 'bilinear')",
)
def polar_trans_image_ext(image, row, column, angle_start, angle_end, radius_start, radius_end, width, height, interpolation, *, ctx):
    picture = expect_image(image, "Image")
    wide = whole(width, "Width")
    high = whole(height, "Height")
    if wide < 1 or high < 1:
        raise ValueError("寬高至少要 1")
    angles = np.linspace(scalar(angle_start, "AngleStart"), scalar(angle_end, "AngleEnd"), wide, endpoint=False, dtype=np.float32)
    radii = np.linspace(scalar(radius_start, "RadiusStart"), scalar(radius_end, "RadiusEnd"), high, endpoint=False, dtype=np.float32)
    angle_grid, radius_grid = np.meshgrid(angles, radii)
    map_x = np.float32(scalar(column, "Column") + radius_grid * np.cos(angle_grid))
    map_y = np.float32(scalar(row, "Row") + radius_grid * np.sin(angle_grid))
    data = cv2.remap(picture.data, map_x, map_y, _interpolation(interpolation), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    domain = None
    if picture.domain is not None:
        moved = cv2.remap(picture.domain, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        domain = np.where(moved > 0, 255, 0).astype(np.uint8)
    return Result([Image(data, domain)], [])


@operator(
    "檔案",
    [P("Image", "輸入影像")],
    [P("ImageCopy", "複本")],
    [],
    [],
    "複製影像",
    "像素和定義域都複製一份。之後改其中一張，不會影響另一張。",
    "copy_image (Image, ImageCopy)",
)
def copy_image(image, *, ctx):
    return Result([expect_image(image, "Image").copy()], [])


@operator(
    "灰階與濾波",
    [],
    [P("ImageGrayRamp", "斜坡影像")],
    [
        P("Alpha", "列方向每像素增加的灰階"),
        P("Beta", "行方向每像素增加的灰階"),
        P("Gamma", "基準點的灰階"),
        P("Row", "基準列"),
        P("Column", "基準行"),
        P("Width", "寬度"),
        P("Height", "高度"),
    ],
    [],
    "建立灰階斜坡",
    "灰階 = Alpha × (列 − Row) + Beta × (行 − Column) + Gamma，再夾到 0 到 255。這版只做 byte。",
    "gen_image_gray_ramp (Ramp, 0, 2, 10, 0, 0, 40, 30)",
)
def gen_image_gray_ramp(alpha, beta, gamma, row, column, width, height, *, ctx):
    wide = whole(width, "Width")
    high = whole(height, "Height")
    if wide < 1 or high < 1:
        raise ValueError("寬高至少要 1")
    rows = np.arange(high, dtype=np.float32)[:, None]
    cols = np.arange(wide, dtype=np.float32)[None, :]
    plane = scalar(alpha, "Alpha") * (rows - scalar(row, "Row"))
    plane = plane + scalar(beta, "Beta") * (cols - scalar(column, "Column"))
    plane = plane + scalar(gamma, "Gamma")
    data = np.clip(np.round(plane), 0, 255).astype(np.uint8)
    return Result([Image(data)], [])


@operator(
    "區域",
    [P("Image", "輸入影像")],
    [P("ImagePart", "裁切後的影像")],
    [
        P("Row", "左上角的列"),
        P("Column", "左上角的行"),
        P("Width", "寬度"),
        P("Height", "高度"),
    ],
    [],
    "依寬高裁切",
    "矩形必須完全落在影像裡，所以結果的寬高就是指定的寬高。crop_rectangle1 是用兩個對角。",
    "crop_part (Image, ImagePart, 0, 0, 10, 8)",
)
def crop_part(image, row, column, width, height, *, ctx):
    picture = expect_image(image, "Image")
    top = whole(row, "Row")
    left = whole(column, "Column")
    wide = whole(width, "Width")
    high = whole(height, "Height")
    if wide < 1 or high < 1:
        raise ValueError("寬高至少要 1")
    if top < 0 or left < 0 or top + high > picture.height or left + wide > picture.width:
        raise ValueError("裁切範圍超出影像")
    data = picture.data[top : top + high, left : left + wide].copy()
    domain = None if picture.domain is None else picture.domain[top : top + high, left : left + wide].copy()
    return Result([Image(data, domain)], [])


def _local_gray(image, mask_width, mask_height):
    picture = expect_image(image, "Image")
    width = whole(mask_width, "MaskWidth")
    height = whole(mask_height, "MaskHeight")
    if width < 1 or height < 1:
        raise ValueError("遮罩寬高至少要 1")
    return picture, picture.gray(), width, height


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageDeviation", "局部標準差")],
    [P("Width", "遮罩寬"), P("Height", "遮罩高")],
    [],
    "局部標準差",
    "用矩形平均算出每個像素附近的標準差，再四捨五入成 0 到 255。彩色先轉成亮度。偶數遮罩可用，原點可能和 HALCON 差 1 像素。",
    "deviation_image (Image, ImageDeviation, 5, 5)",
)
def deviation_image(image, width, height, *, ctx):
    picture, gray, wide, high = _local_gray(image, width, height)
    src = gray.astype(np.float32)
    mean = cv2.blur(src, (wide, high))
    square = cv2.blur(src * src, (wide, high))
    deviation = np.sqrt(np.maximum(square - mean * mean, 0))
    return Result([Image(np.clip(np.round(deviation), 0, 255).astype(np.uint8), None if picture.domain is None else picture.domain.copy())], [])


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageResult", "局部最大減最小")],
    [P("MaskHeight", "遮罩高"), P("MaskWidth", "遮罩寬")],
    [],
    "矩形範圍的灰階差",
    "每個像素輸出附近最大值減最小值。彩色先轉成亮度。用來抓局部對比。",
    "gray_range_rect (Image, ImageResult, 5, 5)",
)
def gray_range_rect(image, mask_height, mask_width, *, ctx):
    picture, gray, wide, high = _local_gray(image, mask_width, mask_height)
    kernel = np.ones((high, wide), np.uint8)
    span = cv2.subtract(cv2.dilate(gray, kernel), cv2.erode(gray, kernel))
    return Result([Image(span, None if picture.domain is None else picture.domain.copy())], [])


@operator(
    "輪廓",
    [P("Image", "輸入影像")],
    [P("EdgeAmplitude", "梯度振幅"), P("EdgeDirection", "梯度方向")],
    [P("FilterType", "sum_abs 或 sum_sqrt", "'sum_sqrt'", ["'sum_abs'", "'sum_sqrt'"]), P("Size", "奇數核大小", "3", ODD_SIZES)],
    [],
    "Sobel 振幅和方向",
    "方向編成 0 到 255。朝右是 128，正角度在畫面上順時針增加，轉半圈回到 0。振幅超過 255 會被截斷。彩色先轉成亮度。",
    "sobel_dir (Image, EdgeAmplitude, EdgeDirection, 'sum_sqrt', 3)",
)
def sobel_dir(image, filter_type, size, *, ctx):
    picture = expect_image(image, "Image")
    kernel = whole(size, "Size")
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("Size 必須是正奇數")
    gray = picture.gray()
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=kernel)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=kernel)
    kind = text(filter_type, "FilterType").lower()
    if kind == "sum_sqrt":
        amplitude = cv2.magnitude(dx, dy)
    elif kind == "sum_abs":
        amplitude = np.abs(dx) + np.abs(dy)
    else:
        raise ValueError("FilterType 只能是 sum_abs 或 sum_sqrt")
    angle = np.arctan2(dy, dx)
    direction = np.clip(np.round((angle + math.pi) / (2 * math.pi) * 255), 0, 255).astype(np.uint8)
    domain = None if picture.domain is None else picture.domain.copy()
    amp = Image(np.clip(amplitude, 0, 255).astype(np.uint8), domain)
    return Result([amp, Image(direction, None if domain is None else domain.copy())], [])


@operator(
    "特徵",
    [P("Regions", "要統計的區域"), P("Image", "影像")],
    [],
    [],
    [P("AbsoluteHisto", "256 格的像素數"), P("RelativeHisto", "每一格占的比例")],
    "灰階直方圖",
    "固定 256 格，從 0 到 255。多個物件合併計算。彩色先轉成亮度。空區域得到全 0。",
    "gray_histo (Regions, Image, AbsoluteHisto, RelativeHisto)",
)
def gray_histo(region, image, *, ctx):
    picture = expect_image(image, "Image")
    area = expect_region(region, "Regions")
    if (picture.width, picture.height) != (area.width, area.height):
        raise ValueError("影像和區域的尺寸不同")
    mask = area.union_mask()
    if picture.domain is not None:
        mask = cv2.bitwise_and(mask, picture.domain)
    values = picture.gray()[mask > 0]
    counts = np.bincount(values.astype(np.int32), minlength=256)[:256]
    total = int(counts.sum())
    relative = (counts / total).tolist() if total else [0.0] * 256
    return Result([], [counts.astype(int).tolist(), relative])


@operator(
    "特徵",
    [P("Regions", "要投影的區域"), P("Image", "影像")],
    [],
    [P("Mode", "這版只做 simple", "'simple'", ["'simple'"])],
    [P("HorProjection", "每一列的灰階和"), P("VertProjection", "每一行的灰階和")],
    "灰階投影",
    "水平投影的長度等於影像高度，垂直投影的長度等於寬度。區域外的像素不計入。彩色先轉成亮度。",
    "gray_projections (Regions, Image, 'simple', HorProjection, VertProjection)",
)
def gray_projections(region, image, mode, *, ctx):
    picture = expect_image(image, "Image")
    area = expect_region(region, "Regions")
    if (picture.width, picture.height) != (area.width, area.height):
        raise ValueError("影像和區域的尺寸不同")
    if text(mode, "Mode").lower() != "simple":
        raise ValueError("Mode 這版只接受 simple")
    mask = area.union_mask() > 0
    if picture.domain is not None:
        mask = mask & (picture.domain > 0)
    values = np.where(mask, picture.gray().astype(np.float64), 0.0)
    return Result([], [values.sum(axis=1).tolist(), values.sum(axis=0).tolist()])


def _proj_from(value) -> np.ndarray:
    nums = [float(item) for item in as_list(value)]
    if len(nums) == 6:
        nums = nums + [0.0, 0.0, 1.0]
    if len(nums) != 9:
        raise ValueError("投影矩陣要 9 個數。6 個數會補成仿射")
    return np.array(nums, np.float64).reshape(3, 3)


def _pack_proj(matrix: np.ndarray) -> list:
    return [_tidy(float(item)) for item in matrix.reshape(-1)]


def _opencv_projective(matrix: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [matrix[1, 1], matrix[1, 0], matrix[1, 2]],
            [matrix[0, 1], matrix[0, 0], matrix[0, 2]],
            [matrix[2, 1], matrix[2, 0], matrix[2, 2]],
        ],
        np.float64,
    )


def _apply_proj(matrix: np.ndarray, rows, cols):
    out_rows = []
    out_cols = []
    for row, col in zip(as_list(rows), as_list(cols)):
        x = float(row)
        y = float(col)
        weight = float(matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2])
        if abs(weight) < 1e-12:
            raise ValueError("投影後的權重是 0")
        out_rows.append(_tidy(float(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / weight))
        out_cols.append(_tidy(float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / weight))
    if len(out_rows) == 1 and not isinstance(rows, (list, tuple)):
        return out_rows[0], out_cols[0]
    return out_rows, out_cols


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageScaled", "拉伸後影像")],
    [P("Min", "對應到 0 的灰階"), P("Max", "對應到 255 的灰階")],
    [],
    "把一段灰階拉開到 0 到 255",
    "小於 Min 變成 0，大於 Max 變成 255。Min 和 Max 相同時整張變成 0。彩色三個通道用同一組上下限。",
    "scale_image_range (Image, ImageScaled, 0, 63)",
)
def scale_image_range(image, min_gray, max_gray, *, ctx):
    picture = expect_image(image, "Image")
    low = scalar(min_gray, "Min")
    high = scalar(max_gray, "Max")
    src = picture.data.astype(np.float32)
    if abs(high - low) < 1e-9:
        out = np.zeros_like(picture.data)
    else:
        out = np.clip((src - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
    return image_out(picture, out)


@operator(
    "灰階與濾波",
    [P("Image1", "第一通道"), P("Image2", "第二通道"), P("Image3", "第三通道")],
    [P("ImageRed", "紅"), P("ImageGreen", "綠"), P("ImageBlue", "藍")],
    [P("ColorSpace", "hsv、lab 或 yuv", "'hsv'", _COLOR_SPACES)],
    [],
    "把其他色彩空間轉回紅綠藍",
    "輸入範圍要和 trans_from_rgb 相同。hsv 的 H 是 0 到 180，lab 的 a、b 以 128 為 0。",
    "trans_to_rgb (H, S, V, Red, Green, Blue, 'hsv')",
)
def trans_to_rgb(image1, image2, image3, color_space, *, ctx):
    planes = [expect_image(item, name) for item, name in ((image1, "Image1"), (image2, "Image2"), (image3, "Image3"))]
    if any(item.data.ndim != 2 for item in planes) or any(item.data.shape != planes[0].data.shape for item in planes):
        raise ValueError("三張輸入都要是相同尺寸的灰階")
    space = text(color_space, "ColorSpace").lower()
    codes = {"hsv": cv2.COLOR_HSV2RGB, "lab": cv2.COLOR_LAB2RGB, "yuv": cv2.COLOR_YUV2RGB}
    if space not in codes:
        raise ValueError("ColorSpace 只能是 hsv、lab 或 yuv")
    converted = cv2.cvtColor(cv2.merge([planes[0].data, planes[1].data, planes[2].data]), codes[space])
    domain = _domain_of(planes[0])
    outputs = [Image(converted[:, :, index].copy(), None if domain is None else domain.copy()) for index in range(3)]
    return Result(outputs, [])


_DERIV_COMPONENTS = ["'none'", "'x'", "'y'", "'xx'", "'yy'", "'xy'", "'gradient'", "'laplace'"]


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("DerivGauss", "結果影像")],
    [
        P("Sigma", "高斯標準差，像素"),
        P("Component", "none、x、y、xx、yy、xy、gradient 或 laplace", "'gradient'", _DERIV_COMPONENTS),
    ],
    [],
    "高斯導數",
    "x 是行方向，y 是列方向。none 只做平滑，gradient 是振幅。其餘是有號導數，128 代表 0。尺度是 OpenCV 的 Sobel，不是 HALCON 正規化後的導數。彩色先轉成亮度。",
    "derivate_gauss (Image, DerivGauss, 1.0, 'x')",
)
def derivate_gauss(image, sigma, component, *, ctx):
    picture = expect_image(image, "Image")
    spread = scalar(sigma, "Sigma")
    if spread <= 0:
        raise ValueError("Sigma 必須大於 0")
    kind = text(component, "Component").lower()
    if kind not in ("none", "x", "y", "xx", "yy", "xy", "gradient", "laplace"):
        raise ValueError("Component 不支援")
    kernel = min(31, max(3, int(round(spread * 6)) | 1))
    smooth = cv2.GaussianBlur(picture.gray().astype(np.float32), (kernel, kernel), spread)

    def sobel(order_x: int, order_y: int) -> np.ndarray:
        return cv2.Sobel(smooth, cv2.CV_32F, order_x, order_y, ksize=3)

    signed = kind not in ("none", "gradient")
    if kind == "none":
        response = smooth
    elif kind == "x":
        response = sobel(1, 0)
    elif kind == "y":
        response = sobel(0, 1)
    elif kind == "xx":
        response = sobel(2, 0)
    elif kind == "yy":
        response = sobel(0, 2)
    elif kind == "xy":
        response = sobel(1, 1)
    elif kind == "gradient":
        response = cv2.magnitude(sobel(1, 0), sobel(0, 1))
    else:
        response = cv2.Laplacian(smooth, cv2.CV_32F, ksize=3)
    if signed:
        out = np.clip(np.round(response + 128), 0, 255)
    else:
        out = np.clip(np.round(response), 0, 255)
    return Result([Image(out.astype(np.uint8))], [])


_LAWS3 = {
    "l": np.array([1, 2, 1], np.float32),
    "e": np.array([-1, 0, 1], np.float32),
    "s": np.array([-1, 2, -1], np.float32),
}
_LAWS5 = {
    "l": np.array([1, 4, 6, 4, 1], np.float32),
    "e": np.array([-1, -2, 0, 2, 1], np.float32),
    "s": np.array([-1, 0, 2, 0, -1], np.float32),
    "r": np.array([1, -4, 6, -4, 1], np.float32),
    "w": np.array([-1, 2, 0, -2, 1], np.float32),
}
_LAWS_TYPES = ["'ll'", "'le'", "'el'", "'ee'", "'ss'", "'ls'", "'sl'", "'rr'"]


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageTexture", "紋理能量")],
    [
        P("FilterType", "兩個字母，例如 el", "'el'", _LAWS_TYPES),
        P("Shift", "再除以 2 的這個次方", "0"),
        P("FilterSize", "3 或 5", "5", ["3", "5"]),
    ],
    [],
    "Laws 紋理濾波",
    "第一個字母沿列，第二個沿行。l 是平滑，e 是邊緣，s 是斑點，r 和 w 只在 5 點核。結果取絕對值並除以核的權重和，所以 ll 大致保留原灰階，常數圖的 el 是 0。",
    "texture_laws (Image, ImageTexture, 'el', 0, 5)",
)
def texture_laws(image, filter_type, shift, filter_size, *, ctx):
    picture = expect_image(image, "Image")
    size = whole(filter_size, "FilterSize")
    kernels = {3: _LAWS3, 5: _LAWS5}.get(size)
    if kernels is None:
        raise ValueError("FilterSize 只能是 3 或 5")
    name = text(filter_type, "FilterType").lower()
    if len(name) != 2 or any(item not in kernels for item in name):
        known = "、".join(sorted(kernels))
        raise ValueError(f"FilterType 要兩個字母，這個尺寸可用 {known}")
    places = whole(shift, "Shift")
    if places < 0:
        raise ValueError("Shift 不能是負的")
    vertical = kernels[name[0]]
    horizontal = kernels[name[1]]
    response = cv2.sepFilter2D(picture.gray().astype(np.float32), cv2.CV_32F, horizontal, vertical)
    scale = float(np.abs(horizontal).sum() * np.abs(vertical).sum()) * float(2**places)
    out = np.clip(np.round(np.abs(response) / scale), 0, 255).astype(np.uint8)
    return Result([Image(out)], [])


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageResult", "改過像素的新影像")],
    [P("Row", "列"), P("Column", "行"), P("Grayval", "灰階。彩色可給紅、綠、藍")],
    [],
    "改指定像素的灰階",
    "會產生一張新影像，不改原來那張。一個灰階可以套到所有點。彩色給三個數時依序是紅、綠、藍。",
    "set_grayval (Image, ImageResult, 10, 12, 255)",
)
def set_grayval(image, row, column, grayval, *, ctx):
    picture = expect_image(image, "Image")
    rows = [int(round(float(item))) for item in as_list(row)]
    cols = [int(round(float(item))) for item in as_list(column)]
    if len(rows) != len(cols) or not rows:
        raise ValueError("Row 和 Column 的長度要相同，而且不能是空的")
    if any(item < 0 or item >= picture.height for item in rows) or any(item < 0 or item >= picture.width for item in cols):
        raise ValueError("座標超出影像")
    data = picture.data.copy()
    values = [int(np.clip(round(float(item)), 0, 255)) for item in as_list(grayval)]
    if data.ndim == 2:
        if len(values) == 1:
            values = values * len(rows)
        if len(values) != len(rows):
            raise ValueError("Grayval 要一個數，或和點一樣多")
        for item_row, item_col, value in zip(rows, cols, values):
            data[item_row, item_col] = value
    else:
        if len(values) == 1:
            paint = [(values[0], values[0], values[0])] * len(rows)
        elif len(values) == 3:
            red, green, blue = values
            paint = [(blue, green, red)] * len(rows)
        elif len(values) == 3 * len(rows):
            paint = []
            for index in range(len(rows)):
                red, green, blue = values[index * 3 : index * 3 + 3]
                paint.append((blue, green, red))
        else:
            raise ValueError("彩色圖的 Grayval 數量不對")
        for item_row, item_col, color in zip(rows, cols, paint):
            data[item_row, item_col] = color
    return image_out(picture, data)


@operator(
    "幾何",
    [],
    [],
    [
        P("Px", "來源的 x，影像上是列"),
        P("Py", "來源的 y，影像上是行"),
        P("Pw", "來源權重，一般填 1"),
        P("Qx", "目標的 x"),
        P("Qy", "目標的 y"),
        P("Qw", "目標權重，一般填 1"),
    ],
    [P("HomMat2D", "3×3 投影矩陣，九個數")],
    "用點對點估計投影矩陣",
    "至少四個點。x 是列、y 是行，和 HomMat2D 相同。權重不是 1 時先除以權重。點共線會失敗。",
    "hom_vector_to_proj_hom_mat2d ([0,0,20,20], [0,30,30,0], [1,1,1,1], [7,7,27,27], [5,35,35,5], [1,1,1,1], HomMat2D)",
)
def hom_vector_to_proj_hom_mat2d(px, py, pw, qx, qy, qw, *, ctx):
    def cartesian(xs, ys, ws, label: str):
        xx = [float(item) for item in as_list(xs)]
        yy = [float(item) for item in as_list(ys)]
        ww = [float(item) for item in as_list(ws)]
        if not xx or not (len(xx) == len(yy) == len(ww)):
            raise ValueError(f"{label} 的 x、y、權重長度要相同")
        if any(abs(item) < 1e-12 for item in ww):
            raise ValueError(f"{label} 的權重不能是 0")
        return np.column_stack([np.array(yy) / np.array(ww), np.array(xx) / np.array(ww)]).astype(np.float32)

    src = cartesian(px, py, pw, "來源")
    dst = cartesian(qx, qy, qw, "目標")
    if len(src) != len(dst):
        raise ValueError("來源和目標的點數要相同")
    if len(src) < 4:
        raise ValueError("至少要四個點")
    if len(src) == 4:
        mapped = cv2.getPerspectiveTransform(src, dst)
    else:
        mapped, _status = cv2.findHomography(src, dst, 0)
    if mapped is None:
        raise ValueError("這些點估計不出投影矩陣")
    matrix = np.array(
        [
            [mapped[1, 1], mapped[1, 0], mapped[1, 2]],
            [mapped[0, 1], mapped[0, 0], mapped[0, 2]],
            [mapped[2, 1], mapped[2, 0], mapped[2, 2]],
        ],
        np.float64,
    )
    return Result([], [_pack_proj(matrix)])


@operator(
    "幾何",
    [],
    [],
    [P("HomMat2D", "投影矩陣"), P("Row", "列"), P("Col", "行")],
    [P("RowTrans", "轉換後的列"), P("ColTrans", "轉換後的行")],
    "用投影矩陣轉換像素",
    "6 個數會當成仿射。9 個數是完整投影。一個點回傳兩個數，多個點回傳 tuple。",
    "projective_trans_pixel (HomMat2D, 10, 12, RowTrans, ColTrans)",
)
def projective_trans_pixel(hom_mat2d, row, col, *, ctx):
    rows, cols = _apply_proj(_proj_from(hom_mat2d), row, col)
    return Result([], [rows, cols])


@operator(
    "幾何",
    [P("Image", "輸入影像")],
    [P("TransImage", "轉換後影像")],
    [
        P("HomMat2D", "投影矩陣"),
        P("Interpolation", "nearest_neighbor、bilinear、bicubic 或 constant", "'bilinear'", ["'nearest_neighbor'", "'bilinear'", "'bicubic'", "'constant'"]),
        P("AdaptImageSize", "true 會把畫布加大到裝得下", "'false'", ["'true'", "'false'"]),
    ],
    [],
    "用投影矩陣轉換影像",
    "和 projective_trans_pixel 同一套矩陣。false 保持原圖大小。true 會平移畫布，讓四個角落進新圖，座標原點跟著變。影像外補 0。",
    "projective_trans_image (Image, TransImage, HomMat2D, 'nearest_neighbor', 'false')",
)
def projective_trans_image(image, hom_mat2d, interpolation, adapt_image_size, *, ctx):
    picture = expect_image(image, "Image")
    matrix = _proj_from(hom_mat2d)
    height, width = picture.height, picture.width
    adapt = text(adapt_image_size, "AdaptImageSize").lower() in ("true", "1")
    out_h, out_w = height, width
    if adapt:
        corners = []
        for row, col in ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)):
            corners.append(_apply_proj(matrix, row, col))
        rows = [float(item[0]) for item in corners]
        cols = [float(item[1]) for item in corners]
        shift_row = -min(rows)
        shift_col = -min(cols)
        out_h = max(1, int(math.floor(max(rows) + shift_row)) + 1)
        out_w = max(1, int(math.floor(max(cols) + shift_col)) + 1)
        shift = np.array([[1.0, 0.0, shift_row], [0.0, 1.0, shift_col], [0.0, 0.0, 1.0]], np.float64)
        matrix = shift @ matrix
    mapped = _opencv_projective(matrix)
    data = cv2.warpPerspective(picture.data, mapped, (out_w, out_h), flags=_interpolation(interpolation), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    domain = None
    if picture.domain is not None:
        warped = cv2.warpPerspective(picture.domain, mapped, (out_w, out_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        domain = np.where(warped > 0, 255, 0).astype(np.uint8)
        data = data.copy()
        data[domain == 0] = 0
    return Result([Image(data, domain)], [])


@operator(
    "分割",
    [],
    [],
    [P("Histogram", "灰階直方圖"), P("Sigma", "平滑的標準差，單位是灰階格")],
    [P("MinThresh", "每一段的下限"), P("MaxThresh", "每一段的上限")],
    "從直方圖找出灰階區間",
    "先把直方圖做高斯平滑，再以波谷切開。沒有波谷時就是 0 到 255 一段。空的區間會丟掉。",
    "histo_to_thresh (Histogram, 2, MinThresh, MaxThresh)",
)
def histo_to_thresh(histogram, sigma, *, ctx):
    counts = np.array([float(item) for item in as_list(histogram)], np.float64)
    if counts.size < 3:
        raise ValueError("直方圖至少要 3 格")
    spread = scalar(sigma, "Sigma")
    if spread < 0:
        raise ValueError("Sigma 不能是負的")
    if spread > 0:
        kernel = min(counts.size if counts.size % 2 else counts.size - 1, max(3, int(round(spread * 6)) | 1))
        smooth = cv2.GaussianBlur(counts.reshape(1, -1), (kernel, 1), spread).ravel()
    else:
        smooth = counts
    valleys = []
    index = 1
    while index < len(smooth) - 1:
        if smooth[index] < smooth[index - 1]:
            end = index
            while end + 1 < len(smooth) and smooth[end + 1] <= smooth[end]:
                end += 1
            if end + 1 < len(smooth) and smooth[end + 1] > smooth[end]:
                valleys.append((index + end) // 2)
            index = end + 1
        else:
            index += 1
    starts = [0] + [item + 1 for item in valleys]
    ends = valleys + [len(counts) - 1]
    low, high = [], []
    for start, end in zip(starts, ends):
        if end < start or float(counts[start : end + 1].sum()) <= 0:
            continue
        low.append(int(start))
        high.append(int(end))
    if not low:
        low, high = [0], [len(counts) - 1]
    return Result([], [low, high])


def _point_pairs(px, py, qx, qy):
    src_r = [float(item) for item in as_list(px)]
    src_c = [float(item) for item in as_list(py)]
    dst_r = [float(item) for item in as_list(qx)]
    dst_c = [float(item) for item in as_list(qy)]
    if not src_r or not (len(src_r) == len(src_c) == len(dst_r) == len(dst_c)):
        raise ValueError("來源和目標的點數要相同，而且不能是空的")
    return np.column_stack([src_r, src_c]), np.column_stack([dst_r, dst_c])


def _affine_lstsq(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    design = np.zeros((len(src) * 2, 6), np.float64)
    target = np.zeros(len(src) * 2, np.float64)
    for index, ((row, col), (row2, col2)) in enumerate(zip(src, dst)):
        design[index * 2] = [row, col, 1, 0, 0, 0]
        design[index * 2 + 1] = [0, 0, 0, row, col, 1]
        target[index * 2] = row2
        target[index * 2 + 1] = col2
    solved, *_rest = np.linalg.lstsq(design, target, rcond=None)
    return np.array([[solved[0], solved[1], solved[2]], [solved[3], solved[4], solved[5]], [0.0, 0.0, 1.0]])


def _similarity_lstsq(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    design = np.zeros((len(src) * 2, 4), np.float64)
    target = np.zeros(len(src) * 2, np.float64)
    for index, ((row, col), (row2, col2)) in enumerate(zip(src, dst)):
        design[index * 2] = [row, -col, 1, 0]
        design[index * 2 + 1] = [col, row, 0, 1]
        target[index * 2] = row2
        target[index * 2 + 1] = col2
    solved, *_rest = np.linalg.lstsq(design, target, rcond=None)
    cosine, sine, tx, ty = (float(item) for item in solved)
    return np.array([[cosine, -sine, tx], [sine, cosine, ty], [0.0, 0.0, 1.0]])


def _rigid_lstsq(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    centered_src = src - src_mean
    centered_dst = dst - dst_mean
    covariance = centered_src.T @ centered_dst
    left, _values, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t = right_t.copy()
        right_t[-1, :] *= -1
        rotation = right_t.T @ left.T
    shift = dst_mean - rotation @ src_mean
    return np.array(
        [[rotation[0, 0], rotation[0, 1], shift[0]], [rotation[1, 0], rotation[1, 1], shift[1]], [0.0, 0.0, 1.0]]
    )


@operator(
    "幾何",
    [],
    [],
    [P("Px", "來源 x，影像上是列"), P("Py", "來源 y，影像上是行"), P("Qx", "目標 x"), P("Qy", "目標 y")],
    [P("HomMat2D", "仿射矩陣")],
    "用點對點估計仿射矩陣",
    "至少三個點。用最小平方，不剔除離群點。x 是列、y 是行。點共線時結果可能不穩定。",
    "vector_to_hom_mat2d ([0, 0, 10], [0, 10, 0], [5, 5, 15], [3, 13, 3], HomMat2D)",
)
def vector_to_hom_mat2d(px, py, qx, qy, *, ctx):
    src, dst = _point_pairs(px, py, qx, qy)
    if len(src) < 3:
        raise ValueError("仿射至少要三個點")
    return Result([], [_pack(_affine_lstsq(src, dst))])


@operator(
    "幾何",
    [],
    [],
    [P("Px", "來源 x，影像上是列"), P("Py", "來源 y，影像上是行"), P("Qx", "目標 x"), P("Qy", "目標 y")],
    [P("HomMat2D", "相似變換")],
    "用點對點估計相似變換",
    "至少兩個點。包含平移、旋轉和等比縮放，沒有剪切。用最小平方。",
    "vector_to_similarity ([0, 0], [0, 10], [0, 0], [0, 20], HomMat2D)",
)
def vector_to_similarity(px, py, qx, qy, *, ctx):
    src, dst = _point_pairs(px, py, qx, qy)
    if len(src) < 2:
        raise ValueError("相似變換至少要兩個點")
    return Result([], [_pack(_similarity_lstsq(src, dst))])


@operator(
    "幾何",
    [],
    [],
    [P("Px", "來源 x，影像上是列"), P("Py", "來源 y，影像上是行"), P("Qx", "目標 x"), P("Qy", "目標 y")],
    [P("HomMat2D", "剛體變換")],
    "用點對點估計剛體變換",
    "至少兩個點。只做平移和旋轉，不縮放。",
    "vector_to_rigid ([0, 0, 10], [0, 10, 0], [0, -10, 0], [0, 0, 10], HomMat2D)",
)
def vector_to_rigid(px, py, qx, qy, *, ctx):
    src, dst = _point_pairs(px, py, qx, qy)
    if len(src) < 2:
        raise ValueError("剛體變換至少要兩個點")
    return Result([], [_pack(_rigid_lstsq(src, dst))])


@operator(
    "幾何",
    [],
    [],
    [
        P("HomMat2D", "原本的矩陣"),
        P("Px", "鏡射軸上一點的 x，影像上是列"),
        P("Py", "鏡射軸上一點的 y，影像上是行"),
        P("Qx", "鏡射軸另一點的 x"),
        P("Qy", "鏡射軸另一點的 y"),
    ],
    [P("HomMat2DReflect", "加上鏡射後的矩陣")],
    "對一條軸做鏡射",
    "鏡射加在原本矩陣之後。兩點決定軸。正的行會被翻到軸的另一側。",
    "hom_mat2d_reflect (HomMat2D, 0, 0, 10, 0, HomMat2DReflect)",
)
def hom_mat2d_reflect(hom_mat2d, px, py, qx, qy, *, ctx):
    row1 = scalar(px, "Px")
    col1 = scalar(py, "Py")
    row2 = scalar(qx, "Qx")
    col2 = scalar(qy, "Qy")
    drow = row2 - row1
    dcol = col2 - col1
    if math.hypot(drow, dcol) < 1e-9:
        raise ValueError("鏡射軸的兩個點不能重合")
    phi = math.atan2(dcol, drow)
    flip = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    linear = _rotation(phi) @ flip @ _rotation(-phi)
    piece = _around(np.eye(3), linear, row1, col1, False)
    return Result([], [_pack(piece @ _mat_from(hom_mat2d))])


@operator(
    "特徵",
    [P("Image", "輸入影像")],
    [],
    [
        P("Sigma", "平滑尺度", "1"),
        P("Threshold", "相對最大回應的門檻，0 到 1", "0.05"),
        P("Alpha", "Harris 的 k", "0.04"),
        P("SubPixel", "true 會把角點修到次像素", "'false'", ["'true'", "'false'"]),
    ],
    [P("Row", "角點列"), P("Column", "角點行")],
    "Harris 角點",
    "Threshold 是最大回應的比例，不是絕對灰階。角點依回應由強到弱排列。SubPixel 為 true 時用 cornerSubPix 再修一次。不填時角點是整數像素。平坦或沒有角時得到空 tuple。",
    "points_harris (Image, 1, 0.05, 0.04, 'false', Row, Column)",
)
def points_harris(image, sigma, threshold, alpha, subpixel, *, ctx):
    picture = expect_image(image, "Image")
    spread = scalar(sigma, "Sigma")
    ratio = scalar(threshold, "Threshold")
    score = scalar(alpha, "Alpha")
    if spread <= 0:
        raise ValueError("Sigma 必須大於 0")
    if not 0 <= ratio <= 1:
        raise ValueError("Threshold 要在 0 到 1")
    block = max(3, int(round(spread * 2)) | 1)
    response = cv2.cornerHarris(picture.gray().astype(np.float32), block, 3, score)
    peak = response == cv2.dilate(response, np.ones((3, 3), np.uint8))
    limit = float(response.max()) * ratio if response.size else 0.0
    peak = peak & (response >= limit) & (response > 0)
    if picture.domain is not None:
        peak = peak & (picture.domain > 0)
    coords = np.argwhere(peak)
    if len(coords) == 0:
        return Result([], [[], []])
    order = np.argsort(response[coords[:, 0], coords[:, 1]])[::-1]
    coords = coords[order]
    refine = text(subpixel, "SubPixel").lower()
    if refine not in ("true", "false", "1", "0"):
        raise ValueError("SubPixel 只能是 true 或 false")
    if refine in ("true", "1"):
        gray = picture.gray()
        corners = np.column_stack([coords[:, 1], coords[:, 0]]).astype(np.float32).reshape(-1, 1, 2)
        cv2.cornerSubPix(
            gray,
            corners,
            (3, 3),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
        )
        return Result([], [corners[:, 0, 1].tolist(), corners[:, 0, 0].tolist()])
    return Result([], [coords[:, 0].astype(int).tolist(), coords[:, 1].astype(int).tolist()])


@operator(
    "分割",
    [P("Image", "輸入影像")],
    [P("LocalMaxima", "局部極大")],
    [P("MinGray", "最低灰階", "0")],
    [],
    "局部極大",
    "3×3 裡比周圍高的像素。比遮罩還大的平坦高台，內部不會被標出來。不填 MinGray 時用 0。彩色先轉成亮度。",
    "local_max (Image, LocalMaxima, 1)",
)
def local_max(image, min_gray, *, ctx):
    picture = expect_image(image, "Image")
    gray = picture.gray()
    kernel = np.ones((3, 3), np.uint8)
    peaks = (gray == cv2.dilate(gray, kernel)) & (gray > cv2.erode(gray, kernel))
    floor = scalar(min_gray, "MinGray")
    if floor > 0:
        peaks = peaks & (gray >= floor)
    if picture.domain is not None:
        peaks = peaks & (picture.domain > 0)
    return region_out(picture.width, picture.height, np.where(peaks, 255, 0).astype(np.uint8))


_BAYER = ["'bayer_rg'", "'bayer_gr'", "'bayer_bg'", "'bayer_gb'"]


@operator(
    "灰階與濾波",
    [P("CFAImage", "Bayer 灰階圖")],
    [P("RGBImage", "彩色影像")],
    [P("CFAType", "bayer_rg、bayer_gr、bayer_bg 或 bayer_gb", "'bayer_rg'", _BAYER)],
    [],
    "把 Bayer 圖轉成彩色",
    "bayer_rg 表示左上是紅、右上是綠，bayer_bg 表示左上是藍。名稱是像素排列，不是 OpenCV 常數的名字。結果的通道 1 是紅。邊緣像素比較不可靠。",
    "cfa_to_rgb (CFAImage, RGBImage, 'bayer_rg')",
)
def cfa_to_rgb(image, cfa_type, *, ctx):
    picture = expect_image(image, "CFAImage")
    if picture.data.ndim != 2:
        raise ValueError("CFA 輸入要是灰階圖")
    kind = text(cfa_type, "CFAType").lower().replace(" ", "").replace("-", "")
    codes = {
        "bayer_rg": cv2.COLOR_BAYER_BG2BGR,
        "bayer_gr": cv2.COLOR_BAYER_GB2BGR,
        "bayer_bg": cv2.COLOR_BAYER_RG2BGR,
        "bayer_gb": cv2.COLOR_BAYER_GR2BGR,
    }
    code = codes.get(kind)
    if code is None:
        raise ValueError("CFAType 只能是 bayer_rg、bayer_gr、bayer_bg 或 bayer_gb")
    color = cv2.cvtColor(picture.data, code)
    domain = None if picture.domain is None else picture.domain.copy()
    return Result([Image(color, domain)], [])


@operator(
    "檔案",
    [P("Image", "輸入影像")],
    [P("ImageConverted", "轉換後影像")],
    [P("Type", "這版只做 byte", "'byte'", ["'byte'"])],
    [],
    "轉換影像型別",
    "這版的影像本來就是 0 到 255 的 byte，所以只複製一份。其他型別會失敗。",
    "convert_image_type (Image, ImageConverted, 'byte')",
)
def convert_image_type(image, kind, *, ctx):
    if text(kind, "Type").lower() != "byte":
        raise ValueError("Type 這版只接受 byte")
    return Result([expect_image(image, "Image").copy()], [])


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImagePow", "次方後影像")],
    [P("Exponent", "次方，必須大於 0")],
    [],
    "灰階次方",
    "先把灰階除以 255，做次方，再乘回 255。大於 1 會讓中間調變暗。彩色每個通道分開算。",
    "pow_image (Image, ImagePow, 2)",
)
def pow_image(image, exponent, *, ctx):
    picture = expect_image(image, "Image")
    power = scalar(exponent, "Exponent")
    if power <= 0:
        raise ValueError("Exponent 必須大於 0")
    src = picture.data.astype(np.float32) / 255.0
    out = np.clip(np.round(np.power(src, power) * 255.0), 0, 255).astype(np.uint8)
    return image_out(picture, out)


@operator(
    "分割",
    [P("Image", "輸入影像")],
    [P("LocalMinima", "局部極小")],
    [P("MaxGray", "最高灰階", "255")],
    [],
    "局部極小",
    "3×3 裡比周圍暗的像素。比遮罩還大的平坦暗區，內部不會被標出來。不填 MaxGray 時用 255。彩色先轉成亮度。",
    "local_min (Image, LocalMinima, 255)",
)
def local_min(image, max_gray, *, ctx):
    picture = expect_image(image, "Image")
    gray = picture.gray()
    kernel = np.ones((3, 3), np.uint8)
    valleys = (gray == cv2.erode(gray, kernel)) & (gray < cv2.dilate(gray, kernel))
    ceiling = scalar(max_gray, "MaxGray")
    if ceiling < 255:
        valleys = valleys & (gray <= ceiling)
    if picture.domain is not None:
        valleys = valleys & (picture.domain > 0)
    return region_out(picture.width, picture.height, np.where(valleys, 255, 0).astype(np.uint8))


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageRank", "排序濾波結果")],
    [P("MaskWidth", "遮罩寬"), P("MaskHeight", "遮罩高"), P("Rank", "0 是最暗，最後一個是最亮")],
    [],
    "矩形排序濾波",
    "每個像素換成遮罩裡排第 Rank 的灰階，從 0 開始。偶數遮罩會加 1，最大是 7。邊界用邊緣的像素補上。Rank 0 接近侵蝕，最後一個接近膨脹。彩色先轉成亮度。",
    "rank_rect (Image, ImageRank, 3, 3, 0)",
)
def rank_rect(image, mask_width, mask_height, rank, *, ctx):
    picture = expect_image(image, "Image")
    wide = whole(mask_width, "MaskWidth") | 1
    high = whole(mask_height, "MaskHeight") | 1
    if wide < 1 or high < 1 or wide > 7 or high > 7:
        raise ValueError("遮罩寬高要是 1 到 7")
    index = whole(rank, "Rank")
    count = wide * high
    if index < 0 or index >= count:
        raise ValueError(f"Rank 要在 0 到 {count - 1}")
    gray = picture.gray().astype(np.float32)
    padded = np.pad(gray, ((high // 2, high // 2), (wide // 2, wide // 2)), mode="edge")
    shifts = [padded[row : row + gray.shape[0], col : col + gray.shape[1]] for row in range(high) for col in range(wide)]
    stacked = np.stack(shifts, axis=0)
    chosen = np.partition(stacked, index, axis=0)[index]
    return Result([Image(np.clip(np.round(chosen), 0, 255).astype(np.uint8))], [])


@operator(
    "灰階與濾波",
    [P("Image1", "第一張"), P("Image2", "第二張")],
    [P("TiledImage", "接起來的影像")],
    [P("Mode", "horizontal 左右接，vertical 上下接", "'horizontal'", ["'horizontal'", "'vertical'"])],
    [],
    "把兩張影像接起來",
    "這版一次兩張，不是一個影像 tuple。尺寸不同時，不足的地方補 0。通道數要相同。",
    "tile_images (Image1, Image2, TiledImage, 'horizontal')",
)
def tile_images(image1, image2, mode, *, ctx):
    first = expect_image(image1, "Image1")
    second = expect_image(image2, "Image2")
    if first.data.ndim != second.data.ndim or first.channels != second.channels:
        raise ValueError("兩張影像的通道數要相同")
    direction = text(mode, "Mode").lower()
    left = first.data
    right = second.data
    if direction == "horizontal":
        height = max(left.shape[0], right.shape[0])
        def fit(data):
            if data.shape[0] == height:
                return data
            pad = height - data.shape[0]
            return np.pad(data, ((0, pad), (0, 0)) if data.ndim == 2 else ((0, pad), (0, 0), (0, 0)))
        tiled = np.concatenate([fit(left), fit(right)], axis=1)
    elif direction == "vertical":
        width = max(left.shape[1], right.shape[1])
        def fit(data):
            if data.shape[1] == width:
                return data
            pad = width - data.shape[1]
            return np.pad(data, ((0, 0), (0, pad)) if data.ndim == 2 else ((0, 0), (0, pad), (0, 0)))
        tiled = np.concatenate([fit(left), fit(right)], axis=0)
    else:
        raise ValueError("Mode 只能是 horizontal 或 vertical")
    return Result([Image(tiled)], [])


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像"), P("Region", "要修補的區域")],
    [P("ImageInpainted", "修補後影像")],
    [],
    [],
    "修補區域裡的像素",
    "用周圍像素補上區域裡的洞。這是 OpenCV 的 Telea 方法，不是調和函數，所以和 HALCON 的數值不同。區域和影像的尺寸要相同。",
    "harmonic_interpolation (Image, Region, ImageInpainted)",
)
def harmonic_interpolation(image, region, *, ctx):
    picture = expect_image(image, "Image")
    area = expect_region(region, "Region")
    if (picture.width, picture.height) != (area.width, area.height):
        raise ValueError("影像和區域的尺寸不同")
    mask = area.union_mask()
    if int(cv2.countNonZero(mask)) == 0:
        return Result([picture.copy()], [])
    filled = cv2.inpaint(picture.data, mask, 3, cv2.INPAINT_TELEA)
    return image_out(picture, filled)


_CONVOL_PRESETS = {
    "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32),
    "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32),
    "laplace": np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32),
    "smooth3": np.ones((3, 3), np.float32) / 9.0,
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32),
}
_CONVOL_BORDERS = {
    "continued": cv2.BORDER_REPLICATE,
    "cyclic": cv2.BORDER_WRAP,
    "mirror": cv2.BORDER_REFLECT_101,
}


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageBilateral", "濾波後影像")],
    [P("SigmaSpatial", "空間標準差，像素"), P("SigmaRange", "灰階標準差")],
    [],
    "雙邊濾波",
    "平滑時會保留灰階差很大的邊緣。兩個 Sigma 都要大於 0，灰階差用 0 到 255。這版沒有聯合影像。彩色每個通道一起濾。",
    "bilateral_filter (Image, ImageBilateral, 3, 20)",
)
def bilateral_filter(image, sigma_spatial, sigma_range, *, ctx):
    picture = expect_image(image, "Image")
    spatial = scalar(sigma_spatial, "SigmaSpatial")
    band = scalar(sigma_range, "SigmaRange")
    if spatial <= 0 or band <= 0:
        raise ValueError("SigmaSpatial 和 SigmaRange 都要大於 0")
    filtered = cv2.bilateralFilter(picture.data, 0, band, spatial)
    return image_out(picture, filtered)


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageResult", "卷積結果")],
    [
        P("FilterMask", "sobel_x、sobel_y、laplace、smooth3、sharpen，或一排係數"),
        P("MaskWidth", "係數是數字時的寬度。用名稱時填 0", "0"),
        P("Margin", "continued、cyclic 或 mirror", "'continued'", ["'continued'", "'cyclic'", "'mirror'"]),
    ],
    [],
    "用自訂濾鏡做卷積",
    "名稱可用 sobel_x、sobel_y、laplace、smooth3、sharpen。係數總和是 0 時，結果會加 128，128 代表 0。其他係數直接裁到 0 到 255，不會自動除以總和。彩色先轉成亮度。",
    "convol_image (Image, ImageResult, 'smooth3', 0, 'continued')",
)
def convol_image(image, kernel, mask_width, margin, *, ctx):
    picture = expect_image(image, "Image")
    border_name = text(margin, "Margin").lower()
    if border_name not in _CONVOL_BORDERS:
        raise ValueError("Margin 只能是 continued、cyclic 或 mirror")
    if isinstance(kernel, str):
        name = kernel.lower()
        if name not in _CONVOL_PRESETS:
            raise ValueError("不認識的濾鏡。可用 sobel_x、sobel_y、laplace、smooth3、sharpen，或給一串數字")
        weights = _CONVOL_PRESETS[name]
    else:
        values = [float(item) for item in as_list(kernel)]
        wide = whole(mask_width, "MaskWidth")
        if wide < 1 or len(values) % wide != 0:
            raise ValueError("MaskWidth 要能整除係數個數")
        high = len(values) // wide
        if wide > 15 or high > 15:
            raise ValueError("濾鏡最大是 15")
        weights = np.asarray(values, np.float32).reshape(high, wide)
    offset = 128.0 if abs(float(weights.sum())) < 1e-6 else 0.0
    filtered = cv2.filter2D(picture.gray().astype(np.float32), cv2.CV_32F, weights, borderType=_CONVOL_BORDERS[border_name])
    return Result([Image(np.clip(np.round(filtered + offset), 0, 255).astype(np.uint8))], [])


def _direction_byte(dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    angle = np.arctan2(dy, dx)
    return np.clip(np.round((angle + math.pi) / (2 * math.pi) * 255), 0, 255).astype(np.uint8)


def _suppress_gradient(magnitude: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    angle = np.arctan2(dy, dx)
    rows, cols = np.mgrid[0 : magnitude.shape[0], 0 : magnitude.shape[1]]
    ahead_col = (cols + np.cos(angle)).astype(np.float32)
    ahead_row = (rows + np.sin(angle)).astype(np.float32)
    behind_col = (cols - np.cos(angle)).astype(np.float32)
    behind_row = (rows - np.sin(angle)).astype(np.float32)
    ahead = cv2.remap(magnitude, ahead_col, ahead_row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    behind = cv2.remap(magnitude, behind_col, behind_row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return (magnitude >= ahead) & (magnitude >= behind)


@operator(
    "輪廓",
    [P("Image", "輸入影像")],
    [P("Amplitude", "邊緣振幅"), P("Direction", "邊緣方向")],
    [
        P("Filter", "canny 或 sobel", "'sobel'", ["'canny'", "'sobel'"]),
        P("Alpha", "平滑標準差"),
        P("NMS", "nms 會把邊緣變細，none 保留整張梯度", "'nms'", ["'nms'", "'none'"]),
        P("Low", "遲滯低門檻"),
        P("High", "遲滯高門檻"),
    ],
    [],
    "抽出邊緣振幅和方向",
    "方向和 sobel_dir 一樣，朝右是 128。canny 本身就有變細和遲滯，NMS 只對 sobel 有作用。none 時 Low、High 不用，整張圖都有振幅。沒有邊緣的地方振幅和方向都是 0。彩色先轉成亮度。",
    "edges_image (Image, Amplitude, Direction, 'sobel', 1, 'nms', 10, 30)",
)
def edges_image(image, filter_name, alpha, nms, low, high, *, ctx):
    picture = expect_image(image, "Image")
    kind = text(filter_name, "Filter").lower()
    if kind not in ("canny", "sobel"):
        raise ValueError("Filter 只能是 canny 或 sobel")
    mode = text(nms, "NMS").lower()
    if mode not in ("nms", "none"):
        raise ValueError("NMS 只能是 nms 或 none")
    spread = scalar(alpha, "Alpha")
    if spread <= 0:
        raise ValueError("Alpha 必須大於 0")
    low_value = scalar(low, "Low")
    high_value = scalar(high, "High")
    kernel = min(31, max(3, int(round(spread * 6)) | 1))
    smooth = cv2.GaussianBlur(picture.gray(), (kernel, kernel), spread)
    dx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(dx, dy)
    direction = _direction_byte(dy, dx)
    if kind == "canny":
        if high_value < low_value:
            raise ValueError("High 不能小於 Low")
        edges = cv2.Canny(smooth, low_value, high_value)
    elif mode == "none":
        edges = np.full(magnitude.shape, 255, np.uint8)
    else:
        if high_value < low_value:
            raise ValueError("High 不能小於 Low")
        ridge = _suppress_gradient(magnitude, dy, dx) & (magnitude >= low_value)
        seeds = ((magnitude >= high_value) & ridge).astype(np.uint8)
        allowed = np.where(ridge, 255, 0).astype(np.uint8)
        if int(seeds.sum()) == 0:
            edges = np.zeros(magnitude.shape, np.uint8)
        else:
            _count, labels = cv2.connectedComponents(allowed, connectivity=8)
            seed_labels = np.unique(labels[seeds > 0])
            seed_labels = seed_labels[seed_labels > 0]
            edges = np.where(np.isin(labels, seed_labels), 255, 0).astype(np.uint8)
    if picture.domain is not None:
        edges = cv2.bitwise_and(edges, picture.domain)
    amplitude = np.where(edges > 0, np.clip(magnitude, 0, 255), 0).astype(np.uint8)
    direction = np.where(edges > 0, direction, 0)
    domain = None if picture.domain is None else picture.domain.copy()
    return Result([Image(amplitude, domain), Image(direction, None if domain is None else domain.copy())], [])


@operator(
    "分割",
    [P("Image", "輸入影像")],
    [P("Plateaus", "平坦區塊")],
    [P("Size", "鄰域邊長，奇數", "3")],
    [],
    "找出平坦的灰階塊",
    "鄰域裡每個像素都一樣才算。大片相同背景的內部也會被標出來，單點尖峰不會。偶數會加 1，最大 15。彩色先轉成亮度。",
    "plateaus (Image, Plateaus, 3)",
)
def plateaus(image, size, *, ctx):
    picture = expect_image(image, "Image")
    side = whole(size, "Size") | 1
    if side < 3 or side > 15:
        raise ValueError("Size 要在 3 到 15")
    gray = picture.gray()
    kernel = np.ones((side, side), np.uint8)
    flat = (gray == cv2.dilate(gray, kernel)) & (gray == cv2.erode(gray, kernel))
    if picture.domain is not None:
        flat = flat & (picture.domain > 0)
    return region_out(picture.width, picture.height, np.where(flat, 255, 0).astype(np.uint8))


@operator(
    "特徵",
    [P("Image", "輸入影像")],
    [],
    [P("Sigma", "平滑尺度"), P("Threshold", "相對最大強度，0 到 1")],
    [P("Row", "角點列"), P("Column", "角點行")],
    "Förstner 角點",
    "用結構張量。圓度要超過 0.5，強度要超過最大強度的 Threshold。角點依強度由強到弱。這版沒有輸出共變異數。平坦或沒有角時得到空 tuple。彩色先轉成亮度。",
    "points_foerstner (Image, 1.5, 0.05, Row, Column)",
)
def points_foerstner(image, sigma, threshold, *, ctx):
    picture = expect_image(image, "Image")
    spread = scalar(sigma, "Sigma")
    ratio = scalar(threshold, "Threshold")
    if spread <= 0:
        raise ValueError("Sigma 必須大於 0")
    if not 0 <= ratio <= 1:
        raise ValueError("Threshold 要在 0 到 1")
    kernel = min(31, max(3, int(round(spread * 6)) | 1))
    smooth = cv2.GaussianBlur(picture.gray().astype(np.float32), (kernel, kernel), spread)
    ix = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    sxx = cv2.GaussianBlur(ix * ix, (kernel, kernel), spread)
    syy = cv2.GaussianBlur(iy * iy, (kernel, kernel), spread)
    sxy = cv2.GaussianBlur(ix * iy, (kernel, kernel), spread)
    det = sxx * syy - sxy * sxy
    trace = sxx + syy
    valid = trace > 1e-6
    strength = np.zeros(trace.shape, np.float32)
    roundness = np.zeros(trace.shape, np.float32)
    strength[valid] = det[valid] / trace[valid]
    roundness[valid] = 4.0 * det[valid] / (trace[valid] * trace[valid])
    peak = strength == cv2.dilate(strength, np.ones((3, 3), np.uint8))
    limit = float(strength.max()) * ratio if strength.size else 0.0
    peak = peak & (roundness >= 0.5) & (strength >= limit) & (strength > 0)
    if picture.domain is not None:
        peak = peak & (picture.domain > 0)
    coords = np.argwhere(peak)
    if len(coords) == 0:
        return Result([], [[], []])
    order = np.argsort(strength[coords[:, 0], coords[:, 1]])[::-1]
    coords = coords[order]
    return Result([], [coords[:, 0].astype(int).tolist(), coords[:, 1].astype(int).tolist()])


@operator(
    "灰階與濾波",
    [P("Image1", "前一張"), P("Image2", "後一張")],
    [P("RowFlow", "列方向的位移"), P("ColFlow", "行方向的位移")],
    [P("Scale", "一個灰階代表幾像素", "1")],
    [],
    "用光流估計位移",
    "這是 Farneback，不是 HALCON 的 multigrid。128 代表沒有移動，比 128 大是往下或往右。超過範圍會截成 0 或 255。兩張圖的寬高要相同。彩色先轉成亮度。",
    "optical_flow_mg (Image1, Image2, RowFlow, ColFlow, 1)",
)
def optical_flow_mg(image1, image2, scale, *, ctx):
    first = expect_image(image1, "Image1")
    second = expect_image(image2, "Image2")
    if first.data.shape[:2] != second.data.shape[:2]:
        raise ValueError("兩張影像的寬高要相同")
    step = scalar(scale, "Scale")
    if step <= 0:
        raise ValueError("Scale 必須大於 0")
    flow = cv2.calcOpticalFlowFarneback(first.gray(), second.gray(), None, 0.5, 3, 21, 3, 5, 1.2, 0)
    row_flow = np.clip(np.round(128.0 + flow[:, :, 1] / step), 0, 255).astype(np.uint8)
    col_flow = np.clip(np.round(128.0 + flow[:, :, 0] / step), 0, 255).astype(np.uint8)
    return Result([Image(row_flow), Image(col_flow)], [])


_DIFF_MODES = ["'abs'", "'light'", "'dark'"]
_DOT_TYPES = ["'light'", "'dark'", "'all'"]
_COOC_DIRECTIONS = ["'0'", "'45'", "'90'", "'135'"]


@operator(
    "分割",
    [P("Image", "要比對的影像"), P("Pattern", "參考影像")],
    [P("Region", "差太多的像素")],
    [
        P("Mode", "abs 絕對差，light 比參考亮，dark 比參考暗", "'abs'", _DIFF_MODES),
        P("Threshold", "超過這個灰階差才留下"),
    ],
    [],
    "找出和參考影像不一樣的地方",
    "abs 是兩邊相差超過門檻。light 是比參考亮，dark 是比參考暗。彩色先轉成亮度。有定義域時只比定義域裡面。",
    "check_difference (Image, Pattern, Region, 'abs', 20)",
)
def check_difference(image, pattern, mode, threshold, *, ctx):
    picture = expect_image(image, "Image")
    reference = expect_image(pattern, "Pattern")
    if picture.data.shape[:2] != reference.data.shape[:2]:
        raise ValueError("兩張影像的寬高要相同")
    kind = text(mode, "Mode").lower()
    if kind not in ("abs", "light", "dark"):
        raise ValueError("Mode 只能是 abs、light 或 dark")
    limit = scalar(threshold, "Threshold")
    if limit < 0:
        raise ValueError("Threshold 不能是負的")
    diff = picture.gray().astype(np.int16) - reference.gray().astype(np.int16)
    if kind == "light":
        mask = diff > limit
    elif kind == "dark":
        mask = -diff > limit
    else:
        mask = np.abs(diff) > limit
    if picture.domain is not None:
        mask = mask & (picture.domain > 0)
    if reference.domain is not None:
        mask = mask & (reference.domain > 0)
    return region_out(picture.width, picture.height, np.where(mask, 255, 0).astype(np.uint8))


@operator(
    "分割",
    [P("Image", "輸入影像")],
    [P("DotRegion", "點狀區域")],
    [
        P("Diameter", "點的大約直徑"),
        P("FilterType", "light 亮點、dark 暗點或 all", "'light'", _DOT_TYPES),
        P("PixelShift", "要比周圍高或低幾個灰階"),
    ],
    [],
    "找出小亮點或小暗點",
    "用頂帽運算把小點從背景分離，再靠 PixelShift 門檻。直徑會變成奇數，範圍 3 到 31。彩色先轉成亮度。",
    "dots_image (Image, DotRegion, 5, 'light', 30)",
)
def dots_image(image, diameter, filter_type, pixel_shift, *, ctx):
    picture = expect_image(image, "Image")
    size = whole(diameter, "Diameter") | 1
    if size < 3 or size > 31:
        raise ValueError("Diameter 要在 3 到 31")
    kind = text(filter_type, "FilterType").lower()
    if kind not in ("light", "dark", "all"):
        raise ValueError("FilterType 只能是 light、dark 或 all")
    shift = scalar(pixel_shift, "PixelShift")
    if shift < 0:
        raise ValueError("PixelShift 不能是負的")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    gray = picture.gray()
    mask = np.zeros(gray.shape, np.uint8)
    if kind in ("light", "all"):
        bright = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        mask = np.maximum(mask, np.where(bright >= shift, 255, 0).astype(np.uint8))
    if kind in ("dark", "all"):
        dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        mask = np.maximum(mask, np.where(dark >= shift, 255, 0).astype(np.uint8))
    if picture.domain is not None:
        mask[picture.domain == 0] = 0
    return region_out(picture.width, picture.height, mask)


def _secondary_peak(smooth: np.ndarray, peak: int):
    best_index = None
    best_value = 0.0
    floor = float(smooth[peak]) * 0.01
    for index in range(1, 255):
        if abs(index - peak) < 8:
            continue
        value = float(smooth[index])
        if value > smooth[index - 1] and value > smooth[index + 1] and value > best_value and value > floor:
            best_value = value
            best_index = index
    return best_index


@operator(
    "分割",
    [P("Image", "輸入影像"), P("HistoRegion", "用來統計的區域")],
    [P("Characters", "字元區域")],
    [P("Sigma", "直方圖平滑"), P("Percent", "0 到 100，100 最靠近波谷")],
    [P("Threshold", "實際用的門檻")],
    "用直方圖把字元分開",
    "先找最高的背景波峰，再找另一側的字元波峰，門檻放在兩個波峰之間。Percent 100 就用波谷，比較低時門檻會往背景波峰靠。找不到第二個波峰時改用 Otsu。直方圖只用 HistoRegion，分類套到整張圖。",
    "char_threshold (Image, HistoRegion, Characters, 2, 90, Threshold)",
)
def char_threshold(image, histo_region, sigma, percent, *, ctx):
    picture = expect_image(image, "Image")
    area = expect_region(histo_region, "HistoRegion")
    if (picture.width, picture.height) != (area.width, area.height):
        raise ValueError("影像和區域的尺寸不同")
    spread = scalar(sigma, "Sigma")
    ratio = scalar(percent, "Percent")
    if spread <= 0:
        raise ValueError("Sigma 必須大於 0")
    if not 0 <= ratio <= 100:
        raise ValueError("Percent 要在 0 到 100")
    gray = picture.gray()
    samples = gray[area.union_mask() > 0]
    if samples.size == 0:
        raise ValueError("HistoRegion 是空的")
    hist = np.bincount(samples, minlength=256).astype(np.float64)
    radius = max(1, int(round(spread * 3)))
    kernel = cv2.getGaussianKernel(radius * 2 + 1, spread).reshape(-1)
    smooth = np.convolve(hist, kernel, mode="same")
    peak = int(np.argmax(smooth))
    secondary = _secondary_peak(smooth, peak)
    if secondary is None:
        limit, _mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        limit = int(limit)
        dark = int(np.count_nonzero(gray <= limit))
        bright = gray.size - dark
        chosen = gray <= limit if dark <= bright else gray >= limit
    else:
        low, high = sorted((peak, secondary))
        valley = low + int(np.argmin(smooth[low : high + 1]))
        limit = int(round(valley + (peak - valley) * (100.0 - ratio) / 100.0))
        limit = int(np.clip(limit, 0, 255))
        chosen = gray <= limit if secondary < peak else gray >= limit
    if picture.domain is not None:
        chosen = chosen & (picture.domain > 0)
    result = region_out(picture.width, picture.height, np.where(chosen, 255, 0).astype(np.uint8))
    return Result(result.iconic, [limit])


def _cooc_numbers(matrix: np.ndarray):
    total = float(matrix.sum())
    if total <= 0:
        return 0.0, 0.0, 0.0, 0.0
    probability = matrix / total
    rows, cols = np.indices(probability.shape)
    energy = float(np.sum(probability * probability))
    contrast = float(np.sum((rows - cols) ** 2 * probability))
    homogeneity = float(np.sum(probability / (1.0 + np.abs(rows - cols))))
    mean_row = float(np.sum(rows * probability))
    mean_col = float(np.sum(cols * probability))
    std_row = float(np.sqrt(np.sum((rows - mean_row) ** 2 * probability)))
    std_col = float(np.sqrt(np.sum((cols - mean_col) ** 2 * probability)))
    if std_row < 1e-12 or std_col < 1e-12:
        correlation = 0.0
    else:
        correlation = float(np.sum((rows - mean_row) * (cols - mean_col) * probability) / (std_row * std_col))
    return energy, correlation, homogeneity, contrast


@operator(
    "特徵",
    [P("Regions", "物件"), P("Image", "灰階影像")],
    [],
    [
        P("LdGray", "灰階要量化成幾級，2 到 32", "8"),
        P("Direction", "0、45、90 或 135 度", "'0'", _COOC_DIRECTIONS),
    ],
    [P("Energy", "能量"), P("Correlation", "相關"), P("Homogeneity", "均質"), P("Contrast", "對比")],
    "灰階共生矩陣的特徵",
    "0 度是同一列的下一個行，90 度是下一列。只算這一個方向。灰階完全一樣時 Correlation 是 0。每個物件一組。彩色先轉成亮度。",
    "cooc_feature_image (Regions, Image, 8, '0', Energy, Correlation, Homogeneity, Contrast)",
)
def cooc_feature_image(region, image, levels, direction, *, ctx):
    picture = expect_image(image, "Image")
    source = expect_region(region, "Regions")
    if (picture.width, picture.height) != (source.width, source.height):
        raise ValueError("影像和區域的尺寸不同")
    bins = whole(levels, "LdGray")
    if bins < 2 or bins > 32:
        raise ValueError("LdGray 要在 2 到 32")
    angle = text(direction, "Direction")
    shifts = {"0": (0, 1), "45": (1, 1), "90": (1, 0), "135": (1, -1)}
    if angle not in shifts:
        raise ValueError("Direction 只能是 0、45、90 或 135")
    drow, dcol = shifts[angle]
    gray = np.clip(np.round(picture.gray().astype(np.float32) * (bins - 1) / 255.0), 0, bins - 1).astype(np.int32)
    height, width = gray.shape
    row_a = slice(max(0, -drow), height - max(0, drow))
    col_a = slice(max(0, -dcol), width - max(0, dcol))
    row_b = slice(max(0, drow), height - max(0, -drow))
    col_b = slice(max(0, dcol), width - max(0, -dcol))
    energy, correlation, homogeneity, contrast = [], [], [], []
    for mask in source.objects:
        valid = (mask[row_a, col_a] > 0) & (mask[row_b, col_b] > 0)
        pairs = gray[row_a, col_a][valid] * bins + gray[row_b, col_b][valid]
        matrix = np.bincount(pairs, minlength=bins * bins).reshape(bins, bins).astype(np.float64)
        values = _cooc_numbers(matrix)
        energy.append(values[0])
        correlation.append(values[1])
        homogeneity.append(values[2])
        contrast.append(values[3])
    return Result([], [energy, correlation, homogeneity, contrast])


_DISTANCE_METRICS = ["'euclidean'", "'city-block'", "'chessboard'"]
_YES_NO = ["'true'", "'false'"]


def _odd_side(value, name: str, limit: int) -> int:
    side = whole(value, name) | 1
    if side < 1 or side > limit:
        raise ValueError(f"{name} 要在 1 到 {limit}")
    return side


@operator(
    "區域",
    [P("Region", "前景區域")],
    [P("Distance", "距離影像")],
    [
        P("Metric", "euclidean、city-block 或 chessboard", "'euclidean'", _DISTANCE_METRICS),
        P("Foreground", "true 時區域是距離 0，false 時量區域到背景的距離", "'true'", _YES_NO),
    ],
    [],
    "算到區域邊界的距離",
    "畫布沿用區域的寬高，不另外指定。距離超過 255 會截成 255。多個物件先合成一塊。這是 OpenCV 的距離轉換。",
    "distance_transform (Region, Distance, 'euclidean', 'true')",
)
def distance_transform(region, metric, foreground, *, ctx):
    source = expect_region(region, "Region")
    kind = text(metric, "Metric").lower()
    codes = {
        "euclidean": (cv2.DIST_L2, getattr(cv2, "DIST_MASK_PRECISE", cv2.DIST_MASK_5)),
        "city-block": (cv2.DIST_L1, cv2.DIST_MASK_3),
        "chessboard": (cv2.DIST_C, cv2.DIST_MASK_3),
    }
    if kind not in codes:
        raise ValueError("Metric 只能是 euclidean、city-block 或 chessboard")
    near = text(foreground, "Foreground").lower()
    if near not in ("true", "false"):
        raise ValueError("Foreground 只能是 true 或 false")
    mask = source.union_mask()
    if near == "true":
        seeds = np.where(mask > 0, 0, 255).astype(np.uint8)
    else:
        seeds = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not np.any(seeds == 0):
        return Result([Image(np.full(seeds.shape, 255, np.uint8))], [])
    distance_type, mask_size = codes[kind]
    raw = cv2.distanceTransform(seeds, distance_type, mask_size)
    return Result([Image(np.clip(np.round(raw), 0, 255).astype(np.uint8))], [])


@operator(
    "區域",
    [P("Region", "前景區域")],
    [P("BinImage", "二值影像")],
    [P("ForegroundGray", "區域裡的灰階"), P("BackgroundGray", "區域外的灰階")],
    [],
    "把區域畫成影像",
    "多個物件先合成一塊。灰階超出 0 到 255 會被截掉。畫布沿用區域的寬高。",
    "region_to_bin (Region, BinImage, 255, 0)",
)
def region_to_bin(region, foreground_gray, background_gray, *, ctx):
    source = expect_region(region, "Region")
    foreground = int(np.clip(round(scalar(foreground_gray, "ForegroundGray")), 0, 255))
    background = int(np.clip(round(scalar(background_gray, "BackgroundGray")), 0, 255))
    image = np.full((source.height, source.width), background, np.uint8)
    image[source.union_mask() > 0] = foreground
    return Result([Image(image)], [])


@operator(
    "灰階與濾波",
    [P("Image", "灰階來源"), P("Regions", "要填平均的物件")],
    [P("ImageMean", "每個物件填上自己的平均")],
    [],
    [],
    "用區域平均灰階填色",
    "每個物件用自己的平均灰階填滿，物件外是 0。物件重疊時，後面的蓋掉前面的。彩色先轉成亮度，結果是灰階。尺寸要相同。",
    "region_to_mean (Image, Regions, ImageMean)",
)
def region_to_mean(image, region, *, ctx):
    picture = expect_image(image, "Image")
    source = expect_region(region, "Regions")
    if (picture.width, picture.height) != (source.width, source.height):
        raise ValueError("影像和區域的尺寸不同")
    gray = picture.gray()
    out = np.zeros(gray.shape, np.uint8)
    for mask in source.objects:
        pixels = gray[mask > 0]
        if pixels.size == 0:
            continue
        out[mask > 0] = int(round(float(pixels.mean())))
    return Result([Image(out)], [])


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("Highpass", "高通結果")],
    [P("Width", "遮罩寬"), P("Height", "遮罩高")],
    [],
    "減去局部平均",
    "128 代表和周圍一樣，比 128 大是比周圍亮。偶數遮罩會加 1，最大 31。邊界用鏡射。彩色先轉成亮度。",
    "highpass_image (Image, Highpass, 5, 5)",
)
def highpass_image(image, width, height, *, ctx):
    picture = expect_image(image, "Image")
    wide = _odd_side(width, "Width", 31)
    high = _odd_side(height, "Height", 31)
    gray = picture.gray().astype(np.float32)
    mean = cv2.blur(gray, (wide, high))
    out = np.clip(np.round(128.0 + gray - mean), 0, 255).astype(np.uint8)
    return image_out(picture, out)


_DIFFUSION_MODES = ["'perona-malik'", "'parabolic'"]


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageAniso", "擴散後影像")],
    [
        P("Mode", "perona-malik 或 parabolic", "'perona-malik'", _DIFFUSION_MODES),
        P("Contrast", "對比尺度，邊緣比這個陡就比較不會被抹掉"),
        P("Theta", "每步的時間，最大 0.25"),
        P("Iterations", "次數，1 到 40"),
    ],
    [],
    "保邊的擴散平滑",
    "這是四鄰域的 Perona-Malik。parabolic 用 1/(1+(梯度/Contrast)²)。最外一圈像素保持不動。彩色先轉成亮度。",
    "anisotropic_diffusion (Image, ImageAniso, 'perona-malik', 20, 0.2, 10)",
)
def anisotropic_diffusion(image, mode, contrast, theta, iterations, *, ctx):
    picture = expect_image(image, "Image")
    kind = text(mode, "Mode").lower()
    if kind not in ("perona-malik", "parabolic"):
        raise ValueError("Mode 只能是 perona-malik 或 parabolic")
    edge = scalar(contrast, "Contrast")
    step = scalar(theta, "Theta")
    count = whole(iterations, "Iterations")
    if edge <= 0:
        raise ValueError("Contrast 必須大於 0")
    if step <= 0 or step > 0.25:
        raise ValueError("Theta 要大於 0，而且不能超過 0.25")
    if count < 1 or count > 40:
        raise ValueError("Iterations 要在 1 到 40")
    work = picture.gray().astype(np.float32)
    scale = edge * edge
    for _ in range(count):
        center = work[1:-1, 1:-1]
        north = work[:-2, 1:-1] - center
        south = work[2:, 1:-1] - center
        west = work[1:-1, :-2] - center
        east = work[1:-1, 2:] - center
        if kind == "perona-malik":
            north_c = np.exp(-(north * north) / scale)
            south_c = np.exp(-(south * south) / scale)
            west_c = np.exp(-(west * west) / scale)
            east_c = np.exp(-(east * east) / scale)
        else:
            north_c = 1.0 / (1.0 + (north * north) / scale)
            south_c = 1.0 / (1.0 + (south * south) / scale)
            west_c = 1.0 / (1.0 + (west * west) / scale)
            east_c = 1.0 / (1.0 + (east * east) / scale)
        updated = center + step * (north_c * north + south_c * south + west_c * west + east_c * east)
        work[1:-1, 1:-1] = updated
    return image_out(picture, np.clip(np.round(work), 0, 255).astype(np.uint8))


_MEDIAN_MARGINS = ["'continued'", "'mirror'", "'cyclic'"]


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageMedian", "中值結果")],
    [
        P("MaskWidth", "遮罩寬"),
        P("MaskHeight", "遮罩高"),
        P("Margin", "continued、mirror 或 cyclic", "'continued'", _MEDIAN_MARGINS),
    ],
    [],
    "矩形中值濾波",
    "偶數遮罩會加 1，最大是 7。continued 用邊緣像素補，mirror 鏡射，cyclic 循環。彩色先轉成亮度。",
    "median_rect (Image, ImageMedian, 3, 3, 'continued')",
)
def median_rect(image, mask_width, mask_height, margin, *, ctx):
    picture = expect_image(image, "Image")
    wide = _odd_side(mask_width, "MaskWidth", 7)
    high = _odd_side(mask_height, "MaskHeight", 7)
    border = text(margin, "Margin").lower()
    pads = {"continued": "edge", "mirror": "reflect", "cyclic": "wrap"}
    if border not in pads:
        raise ValueError("Margin 只能是 continued、mirror 或 cyclic")
    gray = picture.gray()
    padded = np.pad(gray, ((high // 2, high // 2), (wide // 2, wide // 2)), mode=pads[border])
    shifts = [padded[row : row + gray.shape[0], col : col + gray.shape[1]] for row in range(high) for col in range(wide)]
    index = (wide * high) // 2
    chosen = np.partition(np.stack(shifts, axis=0), index, axis=0)[index]
    return image_out(picture, chosen.astype(np.uint8))


_FACET_OFFSETS = tuple((drow, dcol) for drow in (-1, 0, 1) for dcol in (-1, 0, 1))


def _facet_pinv() -> np.ndarray:
    ys, xs = np.mgrid[-1:2, -1:2]
    dx = xs.ravel().astype(np.float64)
    dy = ys.ravel().astype(np.float64)
    design = np.column_stack([dx * dx, dy * dy, dx * dy, dx, dy, np.ones(9)])
    return np.linalg.pinv(design)


_FACET_PINV = _facet_pinv()


@operator(
    "特徵",
    [P("Image", "輸入影像")],
    [],
    [
        P("Filter", "facet 直接用灰階，gauss 先平滑", "'facet'", ["'facet'", "'gauss'"]),
        P("Sigma", "gauss 的平滑尺度"),
        P("Threshold", "峰值至少要這個灰階"),
    ],
    [P("Row", "次像素列"), P("Column", "次像素行")],
    "次像素局部極大",
    "facet 用 3×3 二次曲面，位移最多 1 像素。gauss 會先做高斯平滑，Sigma 必須大於 0。平坦高台不會被標出來。沒有峰值時得到空 tuple。彩色先轉成亮度。",
    "local_max_sub_pix (Image, 'facet', 1, 20, Row, Column)",
)
def local_max_sub_pix(image, filter_name, sigma, threshold, *, ctx):
    picture = expect_image(image, "Image")
    kind = text(filter_name, "Filter").lower()
    if kind not in ("facet", "gauss"):
        raise ValueError("Filter 只能是 facet 或 gauss")
    spread = scalar(sigma, "Sigma")
    floor = scalar(threshold, "Threshold")
    smooth = picture.gray().astype(np.float32)
    if kind == "gauss":
        if spread <= 0:
            raise ValueError("Sigma 必須大於 0")
        kernel = min(31, max(3, int(round(spread * 6)) | 1))
        smooth = cv2.GaussianBlur(smooth, (kernel, kernel), spread)
    peaks = (smooth == cv2.dilate(smooth, np.ones((3, 3), np.uint8))) & (smooth > cv2.erode(smooth, np.ones((3, 3), np.uint8)))
    peaks &= smooth >= floor
    return Result([], list(_subpix_from_mask(smooth, peaks, picture.domain, maxima=True)))


def _subpix_from_mask(smooth, mask, domain, *, maxima: bool):
    peaks = mask.copy()
    peaks[0, :] = False
    peaks[-1, :] = False
    peaks[:, 0] = False
    peaks[:, -1] = False
    if domain is not None:
        peaks &= domain > 0
    coords = np.argwhere(peaks)
    if len(coords) == 0:
        return [], []
    peak_rows = coords[:, 0]
    peak_cols = coords[:, 1]
    patches = np.stack([smooth[peak_rows + drow, peak_cols + dcol] for drow, dcol in _FACET_OFFSETS], axis=1)
    coeff = patches @ _FACET_PINV.T
    quad_x, quad_y, cross, linear_x, linear_y = (coeff[:, index] for index in range(5))
    disc = 4.0 * quad_x * quad_y - cross * cross
    if maxima:
        valid = (disc > 1e-6) & (quad_x < 0) & (quad_y < 0)
    else:
        valid = (disc > 1e-6) & (quad_x > 0) & (quad_y > 0)
    shift_x = np.zeros(len(coords), np.float64)
    shift_y = np.zeros(len(coords), np.float64)
    shift_x[valid] = (cross[valid] * linear_y[valid] - 2.0 * quad_y[valid] * linear_x[valid]) / disc[valid]
    shift_y[valid] = (cross[valid] * linear_x[valid] - 2.0 * quad_x[valid] * linear_y[valid]) / disc[valid]
    found_rows = peak_rows.astype(np.float64) + np.clip(shift_y, -1.0, 1.0)
    found_cols = peak_cols.astype(np.float64) + np.clip(shift_x, -1.0, 1.0)
    strength = smooth[peak_rows, peak_cols]
    order = np.argsort(strength)
    if maxima:
        order = order[::-1]
    return found_rows[order].tolist(), found_cols[order].tolist()


@operator(
    "特徵",
    [P("Image", "輸入影像")],
    [],
    [
        P("Filter", "facet 直接用灰階，gauss 先平滑", "'facet'", ["'facet'", "'gauss'"]),
        P("Sigma", "gauss 的平滑尺度"),
        P("MaxGray", "谷底不能高過這個灰階"),
    ],
    [P("Row", "次像素列"), P("Column", "次像素行")],
    "次像素局部極小",
    "和 local_max_sub_pix 同一種 3×3 曲面，只是找谷底。平坦的低地不會被標出來。沒有谷底時得到空 tuple。彩色先轉成亮度。",
    "local_min_sub_pix (Image, 'facet', 1, 40, Row, Column)",
)
def local_min_sub_pix(image, filter_name, sigma, max_gray, *, ctx):
    picture = expect_image(image, "Image")
    kind = text(filter_name, "Filter").lower()
    if kind not in ("facet", "gauss"):
        raise ValueError("Filter 只能是 facet 或 gauss")
    spread = scalar(sigma, "Sigma")
    ceiling = scalar(max_gray, "MaxGray")
    smooth = picture.gray().astype(np.float32)
    if kind == "gauss":
        if spread <= 0:
            raise ValueError("Sigma 必須大於 0")
        kernel = min(31, max(3, int(round(spread * 6)) | 1))
        smooth = cv2.GaussianBlur(smooth, (kernel, kernel), spread)
    valleys = (smooth == cv2.erode(smooth, np.ones((3, 3), np.uint8))) & (smooth < cv2.dilate(smooth, np.ones((3, 3), np.uint8)))
    valleys &= smooth <= ceiling
    return Result([], list(_subpix_from_mask(smooth, valleys, picture.domain, maxima=False)))


@operator(
    "特徵",
    [P("Image", "輸入影像")],
    [],
    [
        P("MinGray", "圓周上至少要差這麼多"),
        P("NonMax", "true 會抑制旁邊較弱的點", "'true'", _YES_NO),
    ],
    [P("Row", "角點列"), P("Column", "角點行")],
    "FAST 角點",
    "這是 OpenCV 的 FAST，圓周半徑固定是 3，沒有次像素。MinGray 是亮度差。直角的大方塊常常不夠彎，不會被標出來；圓或比較尖的角比較穩。沒有角點時得到空 tuple。彩色先轉成亮度。",
    "points_lepetit (Image, 30, Row, Column)",
)
def points_lepetit(image, min_gray, nonmax, *, ctx):
    picture = expect_image(image, "Image")
    difference = whole(min_gray, "MinGray")
    if difference < 1 or difference > 255:
        raise ValueError("MinGray 要在 1 到 255")
    keep = text(nonmax, "NonMax").lower()
    if keep not in ("true", "false"):
        raise ValueError("NonMax 只能是 true 或 false")
    detector = cv2.FastFeatureDetector_create(difference, nonmaxSuppression=keep == "true")
    points = list(detector.detect(np.ascontiguousarray(picture.gray()), None))
    if picture.domain is not None:
        domain = picture.domain
        points = [point for point in points if domain[int(round(point.pt[1])), int(round(point.pt[0]))] > 0]
    if not points:
        return Result([], [[], []])
    points.sort(key=lambda point: point.response, reverse=True)
    rows = [float(point.pt[1]) for point in points]
    cols = [float(point.pt[0]) for point in points]
    return Result([], [rows, cols])


def _same_gray_region(image: Image, region, image_name: str, region_name: str):
    picture = expect_image(image, image_name)
    source = expect_region(region, region_name)
    if (picture.width, picture.height) != (source.width, source.height):
        raise ValueError("影像和區域的尺寸不同")
    return picture, source


def _gray_plane(mask: np.ndarray, gray: np.ndarray):
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    rr = rows.astype(np.float64)
    cc = cols.astype(np.float64)
    values = gray[rows, cols].astype(np.float64)
    center_row = float(rr.mean())
    center_col = float(cc.mean())
    mean = float(values.mean())
    design = np.column_stack([rr - center_row, cc - center_col])
    coeff, *_rest = np.linalg.lstsq(design, values - mean, rcond=None)
    alpha = float(coeff[0]) if len(coeff) else 0.0
    beta = float(coeff[1]) if len(coeff) > 1 else 0.0
    residual = values - mean - design @ np.array([alpha, beta])
    deviation = float(np.sqrt(np.mean(residual * residual)))
    return center_row, center_col, alpha, beta, mean, deviation


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageBandpass", "帶通結果")],
    [P("Inner", "較小的正方形邊長"), P("Outer", "較大的正方形邊長")],
    [],
    "留下兩個尺度之間的變化",
    "小遮罩的平均減去大遮罩的平均。128 代表沒有變化，比 128 大是中間比外圈亮。偶數會加 1，最大 31，而且 Inner 要比 Outer 小。彩色先轉成亮度。",
    "bandpass_image (Image, ImageBandpass, 3, 9)",
)
def bandpass_image(image, inner, outer, *, ctx):
    picture = expect_image(image, "Image")
    small = _odd_side(inner, "Inner", 31)
    large = _odd_side(outer, "Outer", 31)
    if small >= large:
        raise ValueError("Inner 要比 Outer 小")
    gray = picture.gray().astype(np.float32)
    narrow = cv2.blur(gray, (small, small))
    wide = cv2.blur(gray, (large, large))
    out = np.clip(np.round(128.0 + narrow - wide), 0, 255).astype(np.uint8)
    return image_out(picture, out)


@operator(
    "特徵",
    [P("Regions", "物件"), P("Image", "灰階影像")],
    [],
    [P("Base", "2 是 bit，e 是自然對數", "'2'", ["'2'", "'e'"])],
    [P("Entropy", "熵")],
    "灰階分布的熵",
    "每個物件一組。灰階完全一樣時是 0。兩種灰階各一半、底數 2 時是 1。彩色先轉成亮度。",
    "entropy_gray (Regions, Image, Entropy)",
)
def entropy_gray(region, image, base, *, ctx):
    picture, source = _same_gray_region(image, region, "Image", "Regions")
    kind = text(base, "Base").lower()
    if kind not in ("2", "e"):
        raise ValueError("Base 只能是 2 或 e")
    gray = picture.gray()
    values = []
    for mask in source.objects:
        pixels = gray[mask > 0]
        if pixels.size == 0:
            values.append(0.0)
            continue
        counts = np.bincount(pixels.astype(np.uint8), minlength=256).astype(np.float64)
        probability = counts[counts > 0] / float(pixels.size)
        entropy = float(-(probability * np.log2(probability)).sum())
        if kind == "e":
            entropy *= math.log(2.0)
        values.append(entropy)
    return Result([], [values])


@operator(
    "特徵",
    [P("Regions", "物件"), P("Image", "灰階影像")],
    [],
    [],
    [
        P("MRow", "區域重心列"),
        P("MCol", "區域重心行"),
        P("Alpha", "列方向的斜率"),
        P("Beta", "行方向的斜率"),
        P("Mean", "平均灰階"),
    ],
    "用平面配灰階",
    "灰階大約是 Mean + Alpha×(列-MRow) + Beta×(行-MCol)。重心是區域的面積中心，不是亮度中心。每個物件一組。彩色先轉成亮度。",
    "moments_gray_plane (Regions, Image, MRow, MCol, Alpha, Beta, Mean)",
)
def moments_gray_plane(region, image, *, ctx):
    picture, source = _same_gray_region(image, region, "Image", "Regions")
    gray = picture.gray()
    rows, cols, alphas, betas, means = [], [], [], [], []
    for mask in source.objects:
        center_row, center_col, alpha, beta, mean, _deviation = _gray_plane(mask, gray)
        rows.append(center_row)
        cols.append(center_col)
        alphas.append(alpha)
        betas.append(beta)
        means.append(mean)
    return Result([], [rows, cols, alphas, betas, means])


@operator(
    "特徵",
    [P("Regions", "物件"), P("Image", "灰階影像")],
    [],
    [],
    [P("Deviation", "和平面的均方根誤差")],
    "灰階離開平面多遠",
    "平面和 moments_gray_plane 一樣。完全是斜坡或平面時接近 0。每個物件一個值。彩色先轉成亮度。",
    "plane_deviation (Regions, Image, Deviation)",
)
def plane_deviation(region, image, *, ctx):
    picture, source = _same_gray_region(image, region, "Image", "Regions")
    gray = picture.gray()
    deviations = [_gray_plane(mask, gray)[5] for mask in source.objects]
    return Result([], [deviations])


@operator(
    "特徵",
    [P("Regions", "物件"), P("Image", "灰階影像")],
    [],
    [],
    [P("Area", "灰階總和"), P("Row", "亮度重心列"), P("Column", "亮度重心行")],
    "灰階加權的面積和重心",
    "Area 是區域裡的灰階加總，不是像素數。灰階總和是 0 時，列和行回 0。每個物件一組。彩色先轉成亮度。",
    "area_center_gray (Regions, Image, Area, Row, Column)",
)
def area_center_gray(region, image, *, ctx):
    picture, source = _same_gray_region(image, region, "Image", "Regions")
    gray = picture.gray().astype(np.float64)
    areas, rows, cols = [], [], []
    for mask in source.objects:
        pixels = gray[mask > 0]
        total = float(pixels.sum()) if pixels.size else 0.0
        areas.append(total)
        if total <= 0 or pixels.size == 0:
            rows.append(0.0)
            cols.append(0.0)
            continue
        yy, xx = np.nonzero(mask)
        rows.append(float((yy * gray[yy, xx]).sum() / total))
        cols.append(float((xx * gray[yy, xx]).sum() / total))
    return Result([], [areas, rows, cols])


@operator(
    "幾何",
    [],
    [],
    [P("HomMat2D", "仿射矩陣")],
    [
        P("Sx", "x 倍率"),
        P("Sy", "y 倍率"),
        P("Phi", "旋轉角，弳度"),
        P("Theta", "沿 x 的剪切角，弳度"),
        P("Tx", "平移 x"),
        P("Ty", "平移 y"),
    ],
    "把仿射矩陣拆成參數",
    "重建順序是先縮放、再沿 x 剪切、再旋轉、最後平移。x 是列、y 是行。有翻轉時 Sx 可能是負的。倍率是 0 時會失敗。",
    "hom_mat2d_to_affine_par (HomMat2D, Sx, Sy, Phi, Theta, Tx, Ty)",
)
def hom_mat2d_to_affine_par(hom_mat2d, *, ctx):
    matrix = _mat_from(hom_mat2d)
    horizontal = float(matrix[0, 0])
    vertical_from_col = float(matrix[0, 1])
    vertical = float(matrix[1, 0])
    diagonal = float(matrix[1, 1])
    scale_y = math.hypot(vertical_from_col, diagonal)
    if scale_y < 1e-12:
        raise ValueError("這個矩陣的 y 倍率是 0，不能拆開")
    cos_phi = diagonal / scale_y
    sin_phi = -vertical_from_col / scale_y
    phi = math.atan2(sin_phi, cos_phi)
    scale_x = horizontal * cos_phi + vertical * sin_phi
    if abs(scale_x) < 1e-12:
        raise ValueError("這個矩陣的 x 倍率是 0，不能拆開")
    theta = math.atan((vertical * cos_phi - horizontal * sin_phi) / scale_x)
    return Result([], [
        _tidy(scale_x),
        _tidy(scale_y),
        _tidy(phi),
        _tidy(theta),
        _tidy(float(matrix[0, 2])),
        _tidy(float(matrix[1, 2])),
    ])


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageSqrt", "開根號後")],
    [],
    [],
    "灰階開根號",
    "每個通道分開開根號，再四捨五入成 byte。0 還是 0，81 變成 9，255 變成 16。",
    "sqrt_image (Image, ImageSqrt)",
)
def sqrt_image(image, *, ctx):
    picture = expect_image(image, "Image")
    out = np.clip(np.round(np.sqrt(picture.data.astype(np.float32))), 0, 255).astype(np.uint8)
    return image_out(picture, out)


_LOG_MODES = ["'scaled'", "'natural'"]


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageLog", "對數後")],
    [P("Mode", "scaled 拉回 0 到 255，natural 是自然對數四捨五入", "'scaled'", _LOG_MODES)],
    [],
    "灰階取對數",
    "scaled 用 ln(灰階+1)／ln(256) 乘上 255，0 還是 0、255 還是 255。natural 是 round(ln(灰階))，0 維持 0，所以 255 只會變成 6。每個通道分開算。",
    "log_image (Image, ImageLog, 'scaled')",
)
def log_image(image, mode, *, ctx):
    picture = expect_image(image, "Image")
    kind = text(mode, "Mode").lower()
    if kind not in ("scaled", "natural"):
        raise ValueError("Mode 只能是 scaled 或 natural")
    src = picture.data.astype(np.float32)
    if kind == "scaled":
        logged = np.log(src + 1.0) / math.log(256.0) * 255.0
    else:
        logged = np.zeros(src.shape, np.float32)
        positive = src >= 1
        logged[positive] = np.log(src[positive])
    return image_out(picture, np.clip(np.round(logged), 0, 255).astype(np.uint8))


_SHOCK_MODES = ["'laplace'", "'canny'"]


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("Sharpened", "變陡之後")],
    [
        P("Theta", "每步的強度，最大 1"),
        P("Iterations", "次數，1 到 20"),
        P("Mode", "laplace 或 canny", "'laplace'", _SHOCK_MODES),
        P("Sigma", "先平滑的尺度，0 表示不再平滑"),
    ],
    [],
    "把邊緣變陡",
    "亮區的拉普拉斯是負的，會再變亮；暗區再變暗。canny 改看梯度方向的二階導數。完全平的地方不會變。彩色先轉成亮度。",
    "shock_filter (Image, Sharpened, 0.5, 5, 'laplace', 0.8)",
)
def shock_filter(image, theta, iterations, mode, sigma, *, ctx):
    picture = expect_image(image, "Image")
    step = scalar(theta, "Theta")
    count = whole(iterations, "Iterations")
    kind = text(mode, "Mode").lower()
    spread = scalar(sigma, "Sigma")
    if step <= 0 or step > 1:
        raise ValueError("Theta 要大於 0，而且不能超過 1")
    if count < 1 or count > 20:
        raise ValueError("Iterations 要在 1 到 20")
    if kind not in ("laplace", "canny"):
        raise ValueError("Mode 只能是 laplace 或 canny")
    if spread < 0:
        raise ValueError("Sigma 不能是負的")
    work = picture.gray().astype(np.float32)
    for _ in range(count):
        if spread == 0:
            smooth = work
        else:
            kernel = min(31, max(3, int(round(spread * 6)) | 1))
            smooth = cv2.GaussianBlur(work, (kernel, kernel), spread)
        gx = np.zeros(smooth.shape, np.float32)
        gy = np.zeros(smooth.shape, np.float32)
        gx[:, 1:-1] = (smooth[:, 2:] - smooth[:, :-2]) * 0.5
        gy[1:-1, :] = (smooth[2:, :] - smooth[:-2, :]) * 0.5
        guide = np.zeros(smooth.shape, np.float32)
        if kind == "laplace":
            guide[1:-1, 1:-1] = (
                smooth[1:-1, 2:] + smooth[1:-1, :-2] + smooth[2:, 1:-1] + smooth[:-2, 1:-1] - 4.0 * smooth[1:-1, 1:-1]
            )
        else:
            center = smooth[1:-1, 1:-1]
            gxx = smooth[1:-1, 2:] - 2.0 * center + smooth[1:-1, :-2]
            gyy = smooth[2:, 1:-1] - 2.0 * center + smooth[:-2, 1:-1]
            gxy = 0.25 * (smooth[2:, 2:] - smooth[2:, :-2] - smooth[:-2, 2:] + smooth[:-2, :-2])
            magnitude2 = gx[1:-1, 1:-1] ** 2 + gy[1:-1, 1:-1] ** 2
            directional = np.zeros(center.shape, np.float32)
            valid = magnitude2 > 1e-6
            ix = gx[1:-1, 1:-1]
            iy = gy[1:-1, 1:-1]
            directional[valid] = (ix[valid] ** 2 * gxx[valid] + 2.0 * ix[valid] * iy[valid] * gxy[valid] + iy[valid] ** 2 * gyy[valid]) / magnitude2[valid]
            guide[1:-1, 1:-1] = directional
        work = work - step * np.sign(guide) * np.sqrt(gx * gx + gy * gy)
    return image_out(picture, np.clip(np.round(work), 0, 255).astype(np.uint8))


_NOISE_TYPES = ["'uniform'", "'gaussian'"]


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageNoise", "加過雜訊")],
    [
        P("Amp", "uniform 是上下限，gaussian 是標準差"),
        P("Type", "uniform 或 gaussian", "'uniform'", _NOISE_TYPES),
        P("Seed", "同一個種子得到同一張雜訊", "1"),
    ],
    [],
    "加上白雜訊",
    "結果會截成 0 到 255。Amp 是 0 時影像不變。同一個 Seed 每次都一樣，方便重現。彩色每個通道分開加。",
    "add_noise_white (Image, ImageNoise, 20, 'uniform', 1)",
)
def add_noise_white(image, amp, noise_type, seed, *, ctx):
    picture = expect_image(image, "Image")
    amount = scalar(amp, "Amp")
    kind = text(noise_type, "Type").lower()
    if amount < 0:
        raise ValueError("Amp 不能是負的")
    if kind not in ("uniform", "gaussian"):
        raise ValueError("Type 只能是 uniform 或 gaussian")
    rng = np.random.default_rng(whole(seed, "Seed"))
    if amount == 0:
        noise = np.zeros(picture.data.shape, np.float32)
    elif kind == "uniform":
        noise = rng.uniform(-amount, amount, picture.data.shape).astype(np.float32)
    else:
        noise = rng.normal(0.0, amount, picture.data.shape).astype(np.float32)
    out = np.clip(np.round(picture.data.astype(np.float32) + noise), 0, 255).astype(np.uint8)
    return image_out(picture, out)


@operator(
    "特徵",
    [P("Image", "輸入影像")],
    [],
    [P("Sigma", "平滑尺度，0 表示不平滑"), P("Threshold", "曲率乘積至少要這麼大")],
    [P("Row", "鞍點列"), P("Column", "鞍點行")],
    "次像素鞍點",
    "一個方向往上彎、另一個方向往下彎，而且彎曲中心落在這個像素裡。平坦的斜坡不會被標出來。依曲率由強到弱排列。沒有鞍點時得到空 tuple。彩色先轉成亮度。",
    "saddle_points_sub_pix (Image, 0, 0.2, Row, Column)",
)
def saddle_points_sub_pix(image, sigma, threshold, *, ctx):
    picture = expect_image(image, "Image")
    spread = scalar(sigma, "Sigma")
    limit = scalar(threshold, "Threshold")
    if spread < 0:
        raise ValueError("Sigma 不能是負的")
    if limit < 0:
        raise ValueError("Threshold 不能是負的")
    gray = picture.gray().astype(np.float32)
    if spread > 0:
        kernel = min(31, max(3, int(round(spread * 6)) | 1))
        gray = cv2.GaussianBlur(gray, (kernel, kernel), spread)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return Result([], [[], []])
    center = gray[1:-1, 1:-1]
    curve_row = gray[2:, 1:-1] - 2.0 * center + gray[:-2, 1:-1]
    curve_col = gray[1:-1, 2:] - 2.0 * center + gray[1:-1, :-2]
    candidates = (curve_row * curve_col) < 0
    if picture.domain is not None:
        candidates &= picture.domain[1:-1, 1:-1] > 0
    ys, xs = np.where(candidates)
    if len(ys) == 0:
        return Result([], [[], []])
    peak_rows = ys + 1
    peak_cols = xs + 1
    patches = np.stack([gray[peak_rows + drow, peak_cols + dcol] for drow, dcol in _FACET_OFFSETS], axis=1)
    coeff = patches @ _FACET_PINV.T
    quad_x, quad_y, cross, linear_x, linear_y = (coeff[:, index] for index in range(5))
    disc = 4.0 * quad_x * quad_y - cross * cross
    shift_x = np.zeros(len(ys), np.float64)
    shift_y = np.zeros(len(ys), np.float64)
    solvable = np.abs(disc) > 1e-6
    shift_x[solvable] = (cross[solvable] * linear_y[solvable] - 2.0 * quad_y[solvable] * linear_x[solvable]) / disc[solvable]
    shift_y[solvable] = (cross[solvable] * linear_x[solvable] - 2.0 * quad_x[solvable] * linear_y[solvable]) / disc[solvable]
    strength = np.abs(quad_x * quad_y)
    valid = solvable & (quad_x * quad_y < 0) & (strength >= limit) & (np.abs(shift_x) <= 0.5) & (np.abs(shift_y) <= 0.5)
    if not np.any(valid):
        return Result([], [[], []])
    found_rows = peak_rows[valid].astype(np.float64) + np.clip(shift_y[valid], -0.5, 0.5)
    found_cols = peak_cols[valid].astype(np.float64) + np.clip(shift_x[valid], -0.5, 0.5)
    order = np.argsort(strength[valid])[::-1]
    return Result([], [found_rows[order].tolist(), found_cols[order].tolist()])


@operator(
    "分割",
    [P("Image", "輸入影像")],
    [P("Regions", "每個灰階區間一個物件")],
    [P("Sigma", "直方圖平滑尺度")],
    [],
    "依直方圖的谷底分割",
    "先做灰階直方圖，用 Sigma 平滑後，在谷底切開。每個有像素的區間變成一個物件，暗的在前面。Sigma 0 不平滑。彩色先轉成亮度。",
    "auto_threshold (Image, Regions, 2)",
)
def auto_threshold(image, sigma, *, ctx):
    picture = expect_image(image, "Image")
    gray = picture.gray()
    if picture.domain is None:
        samples = gray.reshape(-1)
    else:
        samples = gray[picture.domain > 0]
    if samples.size == 0:
        return Result([Region(picture.width, picture.height, [])], [])
    counts = np.bincount(samples.astype(np.uint8), minlength=256)
    lows, highs = histo_to_thresh(counts.tolist(), sigma, ctx=ctx).control
    objects = []
    for low, high in zip(as_list(lows), as_list(highs)):
        mask = ((gray >= int(low)) & (gray <= int(high))).astype(np.uint8) * 255
        if picture.domain is not None:
            mask = cv2.bitwise_and(mask, picture.domain)
        if int(cv2.countNonZero(mask)):
            objects.append(mask)
    return Result([Region(picture.width, picture.height, objects, multi_color=len(objects) > 1)], [])


def _binomial_kernel(size: int) -> np.ndarray:
    degree = size - 1
    coeff = np.array([math.comb(degree, index) for index in range(size)], np.float64)
    return (coeff / coeff.sum()).astype(np.float32)


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageGamma", "調整後的影像")],
    [
        P("Gamma", "大於 0"),
        P("Offset", "公式加上的灰階", "0"),
        P("Threshold", "小於此值的灰階保持原樣", "0"),
        P("MaxGray", "公式的最高灰階", "255"),
        P("Encode", "true 用 1/Gamma，false 直接用 Gamma", "'true'", ["'true'", "'false'"]),
    ],
    [],
    "伽瑪校正",
    "小於 Threshold 的灰階不變。其餘先對齊 Threshold 和 MaxGray，再做次方。Encode 為 true 時次方是 1/Gamma，會把中間調變亮；false 時次方就是 Gamma。彩色每個通道各自做。",
    "gamma_image (Image, ImageGamma, 2)",
)
def gamma_image(image, gamma, offset, threshold, max_gray, encode, *, ctx):
    picture = expect_image(image, "Image")
    power = scalar(gamma, "Gamma")
    if power <= 0:
        raise ValueError("Gamma 必須大於 0")
    floor = scalar(threshold, "Threshold")
    ceiling = scalar(max_gray, "MaxGray")
    if ceiling <= floor:
        raise ValueError("MaxGray 必須大於 Threshold")
    mode = text(encode, "Encode").lower()
    if mode not in ("true", "false"):
        raise ValueError("Encode 只能是 true 或 false")
    exponent = (1.0 / power) if mode == "true" else power
    src = picture.data.astype(np.float32)
    below = src < floor
    ratio = np.clip((src - floor) / (ceiling - floor), 0.0, 1.0)
    mapped = scalar(offset, "Offset") + (ceiling - scalar(offset, "Offset")) * np.power(ratio, exponent)
    out = np.where(below, src, mapped)
    return image_out(picture, np.clip(np.round(out), 0, 255).astype(np.uint8))


@operator(
    "灰階與濾波",
    [P("ImageSource", "要塗上去的灰階"), P("ImageDestination", "被塗的影像")],
    [P("MixedImage", "塗完的影像")],
    [],
    [],
    "把一張影像的灰階塗進另一張",
    "只塗來源定義域裡的像素。來源沒有定義域就整張塗上。來源是灰階、目的是彩色時，三個通道都塗上同一個亮度。結果沿用目的影像的定義域。",
    "paint_gray (ImageSource, ImageDestination, MixedImage)",
)
def paint_gray(source_image, dest_image, *, ctx):
    source = expect_image(source_image, "ImageSource")
    dest = expect_image(dest_image, "ImageDestination")
    if (source.width, source.height) != (dest.width, dest.height):
        raise ValueError("兩張影像的尺寸不同")
    out = dest.data.copy()
    mask = np.ones((source.height, source.width), np.bool_) if source.domain is None else source.domain > 0
    if source.data.ndim == dest.data.ndim and source.channels == dest.channels:
        out[mask] = source.data[mask]
    elif source.data.ndim == 2 and dest.data.ndim == 3:
        out[mask] = source.gray()[mask, None]
    else:
        raise ValueError("來源和目的的通道數要相同，或來源是灰階、目的是彩色")
    return image_out(dest, out)


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageBinomial", "平滑後的影像")],
    [
        P("MaskWidth", "奇數寬度", "3", ["1"] + ODD_SIZES),
        P("MaskHeight", "奇數高度", "3", ["1"] + ODD_SIZES),
    ],
    [],
    "二項係數平滑",
    "遮罩係數是二項式，先沿水平再沿垂直，邊界用邊緣像素延伸。寬和高要是 1 到 31 的奇數。彩色每個通道各自做。",
    "binomial_filter (Image, ImageBinomial, 3, 3)",
)
def binomial_filter(image, mask_width, mask_height, *, ctx):
    picture = expect_image(image, "Image")
    wide = whole(mask_width, "MaskWidth")
    high = whole(mask_height, "MaskHeight")
    if wide < 1 or high < 1 or wide > 31 or high > 31 or wide % 2 == 0 or high % 2 == 0:
        raise ValueError("遮罩寬高要是 1 到 31 的奇數")
    blurred = cv2.sepFilter2D(
        picture.data.astype(np.float32),
        cv2.CV_32F,
        _binomial_kernel(wide),
        _binomial_kernel(high),
        borderType=cv2.BORDER_REPLICATE,
    )
    return image_out(picture, np.clip(np.round(blurred), 0, 255).astype(np.uint8))


def _zero_marks(signed: np.ndarray) -> np.ndarray:
    marks = np.zeros(signed.shape, np.bool_)
    left = signed[:, :-1].astype(np.int32)
    right = signed[:, 1:].astype(np.int32)
    change = (left * right < 0) | ((left == 0) ^ (right == 0))
    marks[:, :-1] |= change & (np.abs(left) <= np.abs(right))
    marks[:, 1:] |= change & (np.abs(right) <= np.abs(left))
    up = signed[:-1, :].astype(np.int32)
    down = signed[1:, :].astype(np.int32)
    change = (up * down < 0) | ((up == 0) ^ (down == 0))
    marks[:-1, :] |= change & (np.abs(up) <= np.abs(down))
    marks[1:, :] |= change & (np.abs(down) <= np.abs(up))
    return marks


@operator(
    "分割",
    [P("Image", "輸入影像")],
    [P("RegionCrossing", "過零的位置")],
    [],
    [],
    "灰階過零",
    "灰階 128 當作 0。相鄰像素一正一負，或其中一個剛好是 128 時，留下離 0 較近的那一側。彩色先轉成亮度。",
    "zero_crossing (Image, RegionCrossing)",
)
def zero_crossing(image, *, ctx):
    picture = expect_image(image, "Image")
    marks = _zero_marks(picture.gray().astype(np.int16) - 128)
    if picture.domain is not None:
        marks &= picture.domain > 0
    return region_out(picture.width, picture.height, np.where(marks, 255, 0).astype(np.uint8))


@operator(
    "分割",
    [P("Image", "輸入影像")],
    [P("RegionCrossings", "留下的區塊")],
    [P("MinSize", "最小面積"), P("MinGray", "區塊裡至少要有一個這麼強的像素"), P("Threshold", "候選像素的最低強度")],
    [],
    "雙閾值",
    "灰階 128 當作 0。先留下比 Threshold 更亮或更暗的像素，再用 8 連通分成區塊。區塊裡要有像素超過 MinGray，而且面積不小於 MinSize。彩色先轉成亮度。",
    "dual_threshold (Image, RegionCrossings, 4, 40, 20)",
)
def dual_threshold(image, min_size, min_gray, threshold, *, ctx):
    picture = expect_image(image, "Image")
    gray = picture.gray()
    limit = scalar(threshold, "Threshold")
    strong = scalar(min_gray, "MinGray")
    if limit < 0 or strong < 0:
        raise ValueError("Threshold 和 MinGray 不能是負的")
    least = whole(min_size, "MinSize")
    if least < 0:
        raise ValueError("MinSize 不能是負的")
    hot = gray > 128 + limit
    cold = gray < 128 - limit
    candidates = hot | cold
    if picture.domain is not None:
        candidates &= picture.domain > 0
    peak = (gray > 128 + strong) | (gray < 128 - strong)
    count, labels = cv2.connectedComponents(candidates.astype(np.uint8), connectivity=8)
    objects = []
    for index in range(1, count):
        component = labels == index
        if int(component.sum()) < least or not np.any(peak & component):
            continue
        objects.append(np.where(component, 255, 0).astype(np.uint8))
    return Result([Region(picture.width, picture.height, objects, multi_color=len(objects) > 1)], [])


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageIso", "平滑後的影像")],
    [P("Sigma", "每一次的標準差"), P("Iterations", "次數，1 到 40")],
    [],
    "等向擴散",
    "用一次高斯完成，標準差是 Sigma 乘上次數的平方根。邊界用邊緣像素延伸。彩色每個通道各自做。",
    "isotropic_diffusion (Image, ImageIso, 1.2, 1)",
)
def isotropic_diffusion(image, sigma, iterations, *, ctx):
    picture = expect_image(image, "Image")
    spread = scalar(sigma, "Sigma")
    count = whole(iterations, "Iterations")
    if spread <= 0:
        raise ValueError("Sigma 必須大於 0")
    if count < 1 or count > 40:
        raise ValueError("次數要在 1 到 40")
    blurred = cv2.GaussianBlur(picture.data, (0, 0), spread * math.sqrt(count), 0, cv2.BORDER_REPLICATE)
    return image_out(picture, np.clip(np.round(blurred), 0, 255).astype(np.uint8))


def _paint_values(picture: Image, mask: np.ndarray, grayval, kind: str) -> np.ndarray:
    mode = text(kind, "Type").lower()
    if mode not in ("fill", "margin"):
        raise ValueError("Type 只能是 fill 或 margin")
    paint = mask
    if mode == "margin" and int(cv2.countNonZero(paint)):
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        paint = cv2.subtract(paint, cv2.erode(paint, kernel))
    data = picture.data.copy()
    values = [float(item) for item in as_list(grayval)]
    if data.ndim == 2:
        if len(values) != 1:
            raise ValueError("灰階圖的 Grayval 只能有一個數")
        data[paint > 0] = int(np.clip(round(values[0]), 0, 255))
    else:
        if len(values) == 1:
            values = values * 3
        if len(values) != 3:
            raise ValueError("彩色圖的 Grayval 請給一個數，或依紅、綠、藍給三個數")
        stored = [values[2], values[1], values[0]]
        for index, value in enumerate(stored):
            plane = data[:, :, index]
            plane[paint > 0] = int(np.clip(round(value), 0, 255))
            data[:, :, index] = plane
    return data


@operator(
    "區域",
    [P("Region", "要畫上去的區域"), P("Image", "底圖")],
    [P("ImageResult", "畫完的影像")],
    [
        P("Grayval", "填入的灰階。彩色可給三個數，順序是紅、綠、藍"),
        P("Type", "fill 填滿，margin 只畫邊", "'fill'", ["'fill'", "'margin'"]),
    ],
    [],
    "只在定義域裡塗色",
    "和 paint_region 一樣填色，但定義域外面不改。結果的定義域維持原樣。沒有定義域時就整張畫。",
    "overpaint_region (Region, Image, ImageResult, 200, 'fill')",
)
def overpaint_region(region, image, grayval, kind, *, ctx):
    picture = expect_image(image, "Image")
    area = expect_region(region, "Region")
    if (area.width, area.height) != (picture.width, picture.height):
        raise ValueError("區域和影像的尺寸要相同")
    mask = area.union_mask()
    if picture.domain is not None:
        mask = cv2.bitwise_and(mask, picture.domain)
    return image_out(picture, _paint_values(picture, mask, grayval, kind))


@operator(
    "特徵",
    [P("Image1", "決定列的影像"), P("Image2", "決定行的影像")],
    [P("Histo2Dim", "256×256 的次數圖")],
    [],
    [],
    "二維灰階直方圖",
    "列是第一張的灰階，行是第二張的灰階。同一個位置出現幾次就寫進那個像素，超過 255 記成 255。兩張圖的定義域取交集。彩色先轉成亮度。",
    "histo_2dim (Image1, Image2, Histo2Dim)",
)
def histo_2dim(image1, image2, *, ctx):
    first = expect_image(image1, "Image1")
    second = expect_image(image2, "Image2")
    if (first.width, first.height) != (second.width, second.height):
        raise ValueError("兩張影像的尺寸不同")
    gray1 = first.gray()
    gray2 = second.gray()
    keep = np.ones(gray1.shape, np.bool_)
    if first.domain is not None:
        keep &= first.domain > 0
    if second.domain is not None:
        keep &= second.domain > 0
    counts = np.zeros(256 * 256, np.int32)
    bins = gray1[keep].astype(np.int32) * 256 + gray2[keep].astype(np.int32)
    np.add.at(counts, bins, 1)
    return Result([Image(np.clip(counts, 0, 255).astype(np.uint8).reshape(256, 256))], [])


@operator(
    "特徵",
    [P("Image", "輸入影像")],
    [P("ImageCorner", "角點回應")],
    [
        P("Size", "平滑視窗，3、5 或 7", "3", ["3", "5", "7"]),
        P("Weight", "Harris 的 k，常用 0.04", "0.04"),
        P("Scale", "回應除以這個數再加到 128", "0.001"),
    ],
    [],
    "Harris 角點回應",
    "128 表示沒有回應。角點比 128 亮，直線邊緣比 128 暗。Scale 用 0.001 時，明顯的直角大約會到 200 上下。彩色先轉成亮度。",
    "corner_response (Image, ImageCorner, 3, 0.04, 0.001)",
)
def corner_response(image, size, weight, scale, *, ctx):
    picture = expect_image(image, "Image")
    block = whole(size, "Size")
    if block not in (3, 5, 7):
        raise ValueError("Size 只能是 3、5 或 7")
    harris = scalar(weight, "Weight")
    step = scalar(scale, "Scale")
    if harris < 0:
        raise ValueError("Weight 不能是負的")
    if step <= 0:
        raise ValueError("Scale 必須大於 0")
    response = cv2.cornerHarris(picture.gray(), block, 3, harris)
    mapped = np.clip(np.round(128.0 + response / step), 0, 255).astype(np.uint8)
    return image_out(picture, mapped)


def _disk_offsets(radius: int) -> list[tuple[int, int]]:
    found = []
    for row in range(-radius, radius + 1):
        for col in range(-radius, radius + 1):
            if row * row + col * col <= radius * radius:
                found.append((row + radius, col + radius))
    return found


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageRank", "排序濾波結果")],
    [
        P("Radius", "圓的半徑，1 到 5", "1", ["1", "2", "3", "4", "5"]),
        P("Rank", "0 是最暗，最後一個是最亮"),
    ],
    [],
    "圓形排序濾波",
    "每個像素換成圓裡面排第 Rank 的灰階，從 0 開始。半徑 1 的圓有 5 個像素，Rank 4 就是最亮。邊界用邊緣的像素補上。彩色先轉成亮度。",
    "rank_image (Image, ImageRank, 1, 0)",
)
def rank_image(image, radius, rank, *, ctx):
    picture = expect_image(image, "Image")
    rad = whole(radius, "Radius")
    if rad < 1 or rad > 5:
        raise ValueError("半徑要是 1 到 5")
    offsets = _disk_offsets(rad)
    index = whole(rank, "Rank")
    if index < 0 or index >= len(offsets):
        raise ValueError(f"Rank 要在 0 到 {len(offsets) - 1}")
    gray = picture.gray().astype(np.float32)
    padded = np.pad(gray, rad, mode="edge")
    height, width = gray.shape
    stacked = np.stack([padded[row : row + height, col : col + width] for row, col in offsets], axis=0)
    chosen = np.partition(stacked, index, axis=0)[index]
    return Result([Image(np.clip(np.round(chosen), 0, 255).astype(np.uint8))], [])


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageExp", "指數之後")],
    [P("Mode", "scaled 是對數的反運算，natural 是 e 的次方", "'scaled'", ["'scaled'", "'natural'"])],
    [],
    "指數",
    "scaled 用 256 的次方再減 1，剛好是 scaled 對數的反運算：0 還是 0，128 約是 15，255 還是 255。natural 是 e 的灰階次方，超過 5 就會飽和。彩色每個通道各自做。",
    "exp_image (Image, ImageExp, 'scaled')",
)
def exp_image(image, mode, *, ctx):
    picture = expect_image(image, "Image")
    kind = text(mode, "Mode").lower()
    if kind not in ("scaled", "natural"):
        raise ValueError("Mode 只能是 scaled 或 natural")
    src = picture.data.astype(np.float64)
    if kind == "scaled":
        mapped = np.power(256.0, src / 255.0) - 1.0
    else:
        mapped = np.exp(src)
    return image_out(picture, np.clip(np.round(mapped), 0, 255).astype(np.uint8))


@operator(
    "灰階與濾波",
    [P("ImageDestination", "被塗的影像"), P("ImageSource", "要塗上去的灰階")],
    [P("MixedImage", "塗完的影像")],
    [],
    [],
    "在定義域裡塗上另一張影像",
    "只改目的影像定義域裡的像素，來源的定義域不影響。沒有定義域就整張塗上。結果沿用目的定義域。來源是灰階、目的是彩色時，三個通道都塗上同一個亮度。",
    "overpaint_gray (ImageDestination, ImageSource, MixedImage)",
)
def overpaint_gray(dest_image, source_image, *, ctx):
    dest = expect_image(dest_image, "ImageDestination")
    source = expect_image(source_image, "ImageSource")
    if (source.width, source.height) != (dest.width, dest.height):
        raise ValueError("兩張影像的尺寸不同")
    out = dest.data.copy()
    mask = np.ones((dest.height, dest.width), np.bool_) if dest.domain is None else dest.domain > 0
    if source.data.ndim == dest.data.ndim and source.channels == dest.channels:
        out[mask] = source.data[mask]
    elif source.data.ndim == 2 and dest.data.ndim == 3:
        out[mask] = source.gray()[mask, None]
    elif dest.data.ndim == 2 and source.data.ndim == 3:
        out[mask] = source.gray()[mask]
    else:
        raise ValueError("兩張影像的通道數要相同，或其中一張是灰階")
    return image_out(dest, out)


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageAbs", "離 128 的距離")],
    [],
    [],
    "對 128 取絕對值",
    "把 128 當作 0。平坦的 128 變成 0，0 變成 128，255 變成 127。彩色每個通道各自做。",
    "abs_image (Image, ImageAbs)",
)
def abs_image(image, *, ctx):
    picture = expect_image(image, "Image")
    signed = picture.data.astype(np.int16) - 128
    return image_out(picture, np.clip(np.abs(signed), 0, 255).astype(np.uint8))


@operator(
    "灰階與濾波",
    [P("ImageY", "列方向，128 是 0"), P("ImageX", "行方向，128 是 0")],
    [P("ImageDir", "方向")],
    [],
    [],
    "兩個方向圖的角度",
    "128 是朝右，和 sobel_dir 一樣。正角度在畫面上順時針，朝下約是 191，朝左是 255。兩個方向都是 128 時記成朝右。彩色先轉成亮度。",
    "atan2_image (ImageY, ImageX, ImageDir)",
)
def atan2_image(image_y, image_x, *, ctx):
    vertical = expect_image(image_y, "ImageY")
    horizontal = expect_image(image_x, "ImageX")
    if (vertical.width, vertical.height) != (horizontal.width, horizontal.height):
        raise ValueError("兩張影像的尺寸不同")
    angle = np.arctan2(vertical.gray().astype(np.float32) - 128.0, horizontal.gray().astype(np.float32) - 128.0)
    direction = np.clip(np.round((angle + math.pi) / (2.0 * math.pi) * 255.0), 0, 255).astype(np.uint8)
    return image_out(vertical, direction)


def _sample_gray(picture: Image, rows: np.ndarray, cols: np.ndarray, nearest: bool) -> np.ndarray:
    if picture.data.ndim == 2:
        planes = picture.data
    else:
        planes = picture.data[:, :, :3]
    height, width = picture.height, picture.width
    if nearest:
        row_i = np.floor(rows + 0.5).astype(np.int32)
        col_i = np.floor(cols + 0.5).astype(np.int32)
        if np.any((row_i < 0) | (col_i < 0) | (row_i >= height) | (col_i >= width)):
            raise ValueError("座標超出影像")
        return planes[row_i, col_i]
    if np.any((rows < 0) | (cols < 0) | (rows > height - 1) | (cols > width - 1)):
        raise ValueError("座標超出影像")
    row0 = np.floor(rows).astype(np.int32)
    col0 = np.floor(cols).astype(np.int32)
    row1 = np.minimum(row0 + 1, height - 1)
    col1 = np.minimum(col0 + 1, width - 1)
    dr = (rows - row0).astype(np.float64)
    dc = (cols - col0).astype(np.float64)
    top = planes[row0, col0].astype(np.float64) * (1.0 - dc) + planes[row0, col1].astype(np.float64) * dc
    bottom = planes[row1, col0].astype(np.float64) * (1.0 - dc) + planes[row1, col1].astype(np.float64) * dc
    return top * (1.0 - dr) + bottom * dr


@operator(
    "特徵",
    [P("Image", "輸入影像")],
    [],
    [
        P("Row", "列，可以是小數或 tuple"),
        P("Column", "行，可以是小數或 tuple"),
        P("Interpolation", "bilinear 或 nearest", "'bilinear'", ["'bilinear'", "'nearest'"]),
    ],
    [P("Grayval", "灰階")],
    "用內插讀灰階",
    "bilinear 看周圍四個像素。nearest 四捨五入到像素，.5 往大的整數靠。超出影像會失敗。一個點的灰階圖回傳一個數。彩色回傳紅、綠、藍。",
    "get_grayval_interpolated (Image, 10, 10.5, Grayval)",
)
def get_grayval_interpolated(image, row, column, interpolation, *, ctx):
    picture = expect_image(image, "Image")
    rows = np.asarray([float(item) for item in as_list(row)], np.float64)
    cols = np.asarray([float(item) for item in as_list(column)], np.float64)
    if rows.size == 0 or rows.size != cols.size:
        raise ValueError("Row 和 Column 的長度要相同，而且不能是空的")
    mode = text(interpolation, "Interpolation").lower()
    if mode not in ("bilinear", "nearest"):
        raise ValueError("Interpolation 只能是 bilinear 或 nearest")
    sampled = _sample_gray(picture, rows, cols, mode == "nearest")
    if sampled.ndim == 1:
        values = [float(item) if mode == "bilinear" else int(item) for item in sampled]
        if len(values) == 1 and not isinstance(row, (list, tuple)):
            return Result([], [values[0]])
        return Result([], [values])
    values = []
    for pixel in sampled:
        blue, green, red = (float(item) for item in pixel[:3])
        values.extend([red, green, blue])
    return Result([], [values])


_PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
_PREWITT_Y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32)
_KIRSCH = np.array(
    [
        [[5, 5, 5], [-3, 0, -3], [-3, -3, -3]],
        [[5, 5, -3], [5, 0, -3], [-3, -3, -3]],
        [[5, -3, -3], [5, 0, -3], [5, -3, -3]],
        [[-3, -3, -3], [5, 0, -3], [5, 5, -3]],
        [[-3, -3, -3], [-3, 0, -3], [5, 5, 5]],
        [[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]],
        [[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]],
        [[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]],
    ],
    np.float32,
)


def _filter_amp(gray: np.ndarray, kernels: list[np.ndarray], scale: float) -> np.ndarray:
    planes = [cv2.filter2D(gray, cv2.CV_32F, kernel, borderType=cv2.BORDER_REPLICATE) for kernel in kernels]
    strength = np.max(np.stack(planes, axis=0), axis=0) if len(planes) > 2 else np.hypot(planes[0], planes[1])
    return np.clip(np.round(strength / scale), 0, 255).astype(np.uint8)


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageEdgeAmp", "邊緣強度")],
    [],
    [],
    "Prewitt 邊緣強度",
    "3×3 Prewitt。垂直或水平的完整階梯除以 3 之後，階梯兩側都是階梯高度，平坦處是 0。超過 255 會飽和。彩色先轉成亮度。",
    "prewitt_amp (Image, ImageEdgeAmp)",
)
def prewitt_amp(image, *, ctx):
    picture = expect_image(image, "Image")
    gray = picture.gray().astype(np.float32)
    return image_out(picture, _filter_amp(gray, [_PREWITT_X, _PREWITT_Y], 3.0))


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageEdgeAmp", "邊緣強度")],
    [],
    [],
    "Kirsch 邊緣強度",
    "八個方向取最大，再除以 15。垂直或水平階梯的強度等於階梯高度，平坦處是 0。彩色先轉成亮度。",
    "kirsch_amp (Image, ImageEdgeAmp)",
)
def kirsch_amp(image, *, ctx):
    picture = expect_image(image, "Image")
    gray = picture.gray().astype(np.float32)
    return image_out(picture, _filter_amp(gray, list(_KIRSCH), 15.0))


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageRoberts", "Roberts 邊緣")],
    [P("FilterType", "sum_sqrt 是斜邊，sum_abs 是兩個差分相加", "'sum_sqrt'", ["'sum_sqrt'", "'sum_abs'"])],
    [],
    "Roberts 邊緣",
    "用右下兩個像素做差分。sum_sqrt 是兩個差分的斜邊，sum_abs 是絕對值相加。最後一列和最後一行是 0。彩色先轉成亮度。",
    "roberts (Image, ImageRoberts, 'sum_sqrt')",
)
def roberts(image, filter_type, *, ctx):
    picture = expect_image(image, "Image")
    mode = text(filter_type, "FilterType").lower()
    if mode not in ("sum_sqrt", "sum_abs"):
        raise ValueError("FilterType 只能是 sum_sqrt 或 sum_abs")
    gray = picture.gray().astype(np.float32)
    first = np.zeros_like(gray)
    second = np.zeros_like(gray)
    first[:-1, :-1] = gray[:-1, :-1] - gray[1:, 1:]
    second[:-1, :-1] = gray[:-1, 1:] - gray[1:, :-1]
    strength = np.hypot(first, second) if mode == "sum_sqrt" else np.abs(first) + np.abs(second)
    return image_out(picture, np.clip(np.round(strength), 0, 255).astype(np.uint8))


def _polar_inverse_maps(polar_width, polar_height, row, column, angle_start, angle_end, radius_start, radius_end, width, height):
    wide = whole(width, "Width")
    high = whole(height, "Height")
    if wide < 1 or high < 1 or polar_width < 1 or polar_height < 1:
        raise ValueError("寬高至少要 1")
    angle0 = scalar(angle_start, "AngleStart")
    angle1 = scalar(angle_end, "AngleEnd")
    radius0 = scalar(radius_start, "RadiusStart")
    radius1 = scalar(radius_end, "RadiusEnd")
    span = angle1 - angle0
    radial = radius1 - radius0
    if abs(span) < 1e-12 or abs(radial) < 1e-12:
        raise ValueError("角度或半徑的範圍不能是 0")
    rows = np.arange(high, dtype=np.float64)[:, None]
    cols = np.arange(wide, dtype=np.float64)[None, :]
    delta_row = rows - scalar(row, "Row")
    delta_col = cols - scalar(column, "Column")
    radius = np.hypot(delta_row, delta_col)
    angle = np.arctan2(delta_row, delta_col)
    if span > 0:
        angle = np.where(angle < angle0 - 1e-6, angle + 2.0 * math.pi, angle)
    else:
        angle = np.where(angle > angle0 + 1e-6, angle - 2.0 * math.pi, angle)
    map_x = ((angle - angle0) / span * polar_width).astype(np.float32)
    map_y = ((radius - radius0) / radial * polar_height).astype(np.float32)
    return wide, high, map_x, map_y


@operator(
    "幾何",
    [P("PolarImage", "極座標影像")],
    [P("XYTransImage", "還原後的影像")],
    [
        P("Row", "極座標中心列"),
        P("Column", "極座標中心行"),
        P("AngleStart", "起始角，弳度"),
        P("AngleEnd", "結束角，弳度"),
        P("RadiusStart", "內半徑"),
        P("RadiusEnd", "外半徑"),
        P("Width", "還原後的寬度"),
        P("Height", "還原後的高度"),
        P("Interpolation", "nearest_neighbor、bilinear、bicubic 或 constant", "'bilinear'", ["'nearest_neighbor'", "'bilinear'", "'bicubic'", "'constant'"]),
    ],
    [],
    "把極座標影像還原",
    "和 polar_trans_image_ext 相反。角度 0 朝右，正角度順時針。寬高是還原後的直角座標尺寸，會變成之後區域的參考大小。",
    "polar_trans_image_inv (PolarImage, XYTransImage, 30, 40, 0, rad(360), 0, 20, 80, 60, 'nearest_neighbor')",
)
def polar_trans_image_inv(image, row, column, angle_start, angle_end, radius_start, radius_end, width, height, interpolation, *, ctx):
    picture = expect_image(image, "PolarImage")
    wide, high, map_x, map_y = _polar_inverse_maps(
        picture.width, picture.height, row, column, angle_start, angle_end, radius_start, radius_end, width, height
    )
    data = cv2.remap(picture.data, map_x, map_y, _interpolation(interpolation), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    domain = None
    if picture.domain is not None:
        moved = cv2.remap(picture.domain, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        domain = np.where(moved > 0, 255, 0).astype(np.uint8)
    return Result([Image(data, domain)], [])


_FREI_SCALE = np.float32(1.0 / (2.0 * math.sqrt(2.0)))
_FREI = _FREI_SCALE * np.array(
    [
        [[1, math.sqrt(2), 1], [0, 0, 0], [-1, -math.sqrt(2), -1]],
        [[1, 0, -1], [math.sqrt(2), 0, -math.sqrt(2)], [1, 0, -1]],
        [[0, -1, math.sqrt(2)], [1, 0, -1], [-math.sqrt(2), 1, 0]],
        [[math.sqrt(2), -1, 0], [-1, 0, 1], [0, 1, -math.sqrt(2)]],
    ],
    np.float32,
)
_ROBINSON = np.array(
    [
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        [[2, 1, 0], [1, 0, -1], [0, -1, -2]],
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        [[0, -1, -2], [1, 0, -1], [2, 1, 0]],
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
    ],
    np.float32,
)


def _compass(gray: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    return np.stack(
        [cv2.filter2D(gray, cv2.CV_32F, kernel, borderType=cv2.BORDER_REPLICATE) for kernel in kernels],
        axis=0,
    )


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageEdgeAmp", "邊緣強度")],
    [],
    [],
    "Frei-Chen 邊緣強度",
    "四個邊緣遮罩的能量。平坦是 0。高度 40 的垂直階梯是 49。彩色先轉成亮度。",
    "frei_amp (Image, ImageEdgeAmp)",
)
def frei_amp(image, *, ctx):
    picture = expect_image(image, "Image")
    energy = np.sum(np.square(_compass(picture.gray().astype(np.float32), _FREI)), axis=0)
    return image_out(picture, np.clip(np.round(np.sqrt(energy)), 0, 255).astype(np.uint8))


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageEdgeAmp", "邊緣強度")],
    [],
    [],
    "Robinson 邊緣強度",
    "八個方向取絕對值最大，不另外縮小。高度 40 的垂直階梯是 160。超過 255 會飽和。彩色先轉成亮度。",
    "robinson_amp (Image, ImageEdgeAmp)",
)
def robinson_amp(image, *, ctx):
    picture = expect_image(image, "Image")
    strength = np.max(np.abs(_compass(picture.gray().astype(np.float32), _ROBINSON)), axis=0)
    return image_out(picture, np.clip(np.round(strength), 0, 255).astype(np.uint8))


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageDir", "邊緣方向")],
    [],
    [],
    "Robinson 邊緣方向",
    "和 sobel_dir 同一套：朝右是 128，正角度順時針。平坦時沒有方向，也記成朝右。要搭配 robinson_amp 才知道那裡有沒有邊緣。彩色先轉成亮度。",
    "robinson_dir (Image, ImageDir)",
)
def robinson_dir(image, *, ctx):
    picture = expect_image(image, "Image")
    planes = _compass(picture.gray().astype(np.float32), _ROBINSON)
    index = np.argmax(planes, axis=0)
    best = np.max(planes, axis=0)
    angle = index.astype(np.float64) * (math.pi / 4.0)
    angle = np.where(best < 0, angle + math.pi, angle)
    angle = np.arctan2(np.sin(angle), np.cos(angle))
    direction = np.clip(np.round((angle + math.pi) / (2.0 * math.pi) * 255.0), 0, 255).astype(np.uint8)
    return image_out(picture, direction)


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageNoise", "加了椒鹽的影像")],
    [P("Percent", "要改掉的像素百分比，0 到 100", "5"), P("Seed", "亂數種子，相同就重現", "0")],
    [],
    "加上椒鹽雜訊",
    "Percent 0 影像不變。100 時每個像素不是 255 就是 0，兩邊數量一樣。同一個 Seed 每次都一樣。彩色三個通道一起變。",
    "add_noise_salt_pepper (Image, ImageNoise, 5, 0)",
)
def add_noise_salt_pepper(image, percent, seed, *, ctx):
    picture = expect_image(image, "Image")
    ratio = scalar(percent, "Percent")
    if ratio < 0 or ratio > 100:
        raise ValueError("Percent 要在 0 到 100")
    out = picture.data.copy()
    count = picture.height * picture.width
    chosen = int(round(ratio / 100.0 * count))
    if chosen > 0:
        rng = np.random.default_rng(whole(seed, "Seed"))
        picked = rng.choice(count, size=chosen, replace=False)
        flat = out.reshape(count, -1)
        flat[picked[: chosen // 2]] = 255
        flat[picked[chosen // 2 :]] = 0
    return image_out(picture, out)


def _bit_pair(left, right):
    first = expect_image(left, "Image1")
    second = expect_image(right, "Image2")
    if first.data.shape != second.data.shape:
        raise ValueError("兩張影像的尺寸或通道數不同")
    return first, second


@operator(
    "灰階與濾波",
    [P("Image1", "第一張"), P("Image2", "第二張")],
    [P("ImageAnd", "位元且")],
    [],
    [],
    "兩張影像做位元且",
    "每個像素、每個通道各自做 AND。8 位元，超過的位元不保留。定義域沿用第一張。",
    "bit_and (Image1, Image2, ImageAnd)",
)
def bit_and(image1, image2, *, ctx):
    first, second = _bit_pair(image1, image2)
    return image_out(first, np.bitwise_and(first.data, second.data))


@operator(
    "灰階與濾波",
    [P("Image1", "第一張"), P("Image2", "第二張")],
    [P("ImageOr", "位元或")],
    [],
    [],
    "兩張影像做位元或",
    "每個像素、每個通道各自做 OR。定義域沿用第一張。",
    "bit_or (Image1, Image2, ImageOr)",
)
def bit_or(image1, image2, *, ctx):
    first, second = _bit_pair(image1, image2)
    return image_out(first, np.bitwise_or(first.data, second.data))


@operator(
    "灰階與濾波",
    [P("Image1", "第一張"), P("Image2", "第二張")],
    [P("ImageXor", "位元互斥或")],
    [],
    [],
    "兩張影像做位元互斥或",
    "每個像素、每個通道各自做 XOR。定義域沿用第一張。",
    "bit_xor (Image1, Image2, ImageXor)",
)
def bit_xor(image1, image2, *, ctx):
    first, second = _bit_pair(image1, image2)
    return image_out(first, np.bitwise_xor(first.data, second.data))


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageNot", "位元否定")],
    [],
    [],
    "影像做位元否定",
    "8 位元的 NOT。0 變成 255，10 變成 245。彩色每個通道各自做。",
    "bit_not (Image, ImageNot)",
)
def bit_not(image, *, ctx):
    picture = expect_image(image, "Image")
    return image_out(picture, np.bitwise_not(picture.data))


_SHIFTS = ["0", "1", "2", "3", "4", "5", "6", "7"]


def _shift_image(image, shift, left: bool):
    picture = expect_image(image, "Image")
    amount = whole(shift, "Shift")
    if amount < 0:
        raise ValueError("Shift 不能是負的")
    if amount == 0:
        return image_out(picture, picture.data.copy())
    if amount >= 8:
        return image_out(picture, np.zeros_like(picture.data))
    moved = np.left_shift(picture.data, amount) if left else np.right_shift(picture.data, amount)
    return image_out(picture, moved)


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageShift", "左移後的影像")],
    [P("Shift", "左移的位數", "1", _SHIFTS)],
    [],
    "灰階左移",
    "每個通道乘上 2 的 Shift 次方，只留 8 個位元。1 左移 3 位是 8。移 8 位以上變成 0。",
    "bit_lshift (Image, ImageShift, 1)",
)
def bit_lshift(image, shift, *, ctx):
    return _shift_image(image, shift, True)


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageShift", "右移後的影像")],
    [P("Shift", "右移的位數", "1", _SHIFTS)],
    [],
    "灰階右移",
    "每個通道除以 2 的 Shift 次方，小數直接丟掉。16 右移 2 位是 4。",
    "bit_rshift (Image, ImageShift, 1)",
)
def bit_rshift(image, shift, *, ctx):
    return _shift_image(image, shift, False)


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImageLut", "對照後的影像")],
    [P("Lut", "256 個輸出灰階，索引是原來的灰階")],
    [],
    "用對照表換灰階",
    "Lut 要剛好 256 個數，依序對應灰階 0 到 255。超過 0 到 255 會先夾住。彩色每個通道各自查表。",
    "lut_trans (Image, ImageLut, Lut)",
)
def lut_trans(image, lut, *, ctx):
    picture = expect_image(image, "Image")
    table = np.array([int(round(float(item))) for item in as_list(lut)], np.int32)
    if table.size != 256:
        raise ValueError("Lut 要 256 個數，對應灰階 0 到 255")
    table = np.clip(table, 0, 255).astype(np.uint8)
    return image_out(picture, table[picture.data])


_PYRAMID = ["'constant'", "'none'", "'min'", "'max'"]


def _pyramid_half(data: np.ndarray, mode: str) -> np.ndarray:
    if data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError("影像至少要 2×2")
    if mode == "none":
        return np.ascontiguousarray(data[::2, ::2])
    if mode == "constant":
        return cv2.pyrDown(data)
    high = data.shape[0] // 2 * 2
    wide = data.shape[1] // 2 * 2
    block = data[:high, :wide]
    if data.ndim == 2:
        parts = block.reshape(high // 2, 2, wide // 2, 2)
        axes = (1, 3)
    else:
        parts = block.reshape(high // 2, 2, wide // 2, 2, data.shape[2])
        axes = (1, 3)
    return parts.min(axis=axes) if mode == "min" else parts.max(axis=axes)


@operator(
    "灰階與濾波",
    [P("Image", "輸入影像")],
    [P("ImagePyramid", "縮成一半的影像")],
    [
        P("Mode", "constant 是高斯，none 是隔點取，min 和 max 是 2×2 的最小或最大", "'constant'", _PYRAMID),
        P("Scale", "只能是 0.5", "0.5"),
    ],
    [],
    "做一層一半大小的金字塔",
    "constant 用高斯再取樣。none 取偶數行列。min 和 max 看每個 2×2。這版只縮成一半，不保留定義域。結果會變成之後區域的參考尺寸。",
    "gen_gauss_pyramid (Image, ImagePyramid, 'constant', 0.5)",
)
def gen_gauss_pyramid(image, mode, scale, *, ctx):
    picture = expect_image(image, "Image")
    kind = text(mode, "Mode").lower()
    if kind not in ("constant", "none", "min", "max"):
        raise ValueError("Mode 只能是 constant、none、min 或 max")
    if abs(scalar(scale, "Scale") - 0.5) > 1e-9:
        raise ValueError("這版的金字塔只縮成一半")
    return Result([Image(_pyramid_half(picture.data, kind))], [])


class ComplexImage:
    def __init__(self, data: np.ndarray):
        self.data = np.asarray(data, np.complex128)

    def summary(self) -> str:
        height, width = self.data.shape[:2]
        return f"複數影像  {width}×{height}"


def _expect_complex(value) -> ComplexImage:
    if not isinstance(value, ComplexImage):
        raise TypeError("Complex 必須是 fft_image 或 gen_complex_const 建立的複數影像")
    return value


def _spectrum_mode(value, name: str) -> str:
    kind = text(value, name).strip().strip("'\"").lower()
    return kind


@operator(
    "灰階與濾波",
    [P("Image", "灰階或彩色影像")],
    [],
    [],
    [P("Complex", "複數頻譜")],
    "做傅立葉轉換",
    "正向不除以點數，低頻在左上角。彩色先轉成亮度。反轉換會再除以點數，灰階可以回到原來的值。",
    "fft_image (Image, Complex)",
)
def fft_image(image, *, ctx):
    picture = expect_image(image, "Image")
    return Result([], [ComplexImage(np.fft.fft2(picture.gray().astype(np.float64)))])


@operator(
    "灰階與濾波",
    [],
    [P("Image", "回到空間的灰階影像")],
    [P("Complex", "複數頻譜")],
    [],
    "傅立葉反轉換",
    "除以點數後四捨五入，再裁到 0 到 255。結果的尺寸就是頻譜的尺寸，也會變成之後區域的參考尺寸。",
    "fft_image_inv (Image, Complex)",
)
def fft_image_inv(complex_image, *, ctx):
    spectrum = _expect_complex(complex_image)
    spatial = np.fft.ifft2(spectrum.data).real
    return Result([Image(np.clip(np.round(spatial), 0, 255).astype(np.uint8))], [])


@operator(
    "灰階與濾波",
    [],
    [],
    [P("Complex", "複數頻譜"), P("Row", "列"), P("Column", "行")],
    [P("Real", "實部"), P("Imag", "虛部")],
    "讀一個頻譜點",
    "一次讀一個點。原點在左上角。常數影像的能量全在 (0, 0)，數值是灰階乘上寬乘高。",
    "get_complex (Complex, 0, 0, Real, Imag)",
)
def get_complex(complex_image, row, column, *, ctx):
    spectrum = _expect_complex(complex_image)
    item_row, item_col = int(round(scalar(row, "Row"))), int(round(scalar(column, "Column")))
    height, width = spectrum.data.shape[:2]
    if item_row < 0 or item_col < 0 or item_row >= height or item_col >= width:
        raise ValueError("座標超出複數影像")
    value = spectrum.data[item_row, item_col]
    return Result([], [float(value.real), float(value.imag)])


_FFT_SHIFT = ["'center'", "'corner'"]


@operator(
    "灰階與濾波",
    [],
    [],
    [P("Complex", "複數頻譜"), P("Mode", "center 把低頻移到中央，corner 移回角落", "'center'", _FFT_SHIFT)],
    [P("ComplexShifted", "移動後的頻譜")],
    "移動頻譜原點",
    "center 把左上角的低頻搬到中央。corner 做相反的搬移。寬和高都是偶數時，兩種搬法一樣，再做一次就回到原點。奇數時，center 和 corner 會差一點。",
    "fftshift (Complex, 'center', ComplexShifted)",
)
def fftshift(complex_image, mode, *, ctx):
    spectrum = _expect_complex(complex_image)
    kind = _spectrum_mode(mode, "Mode")
    if kind == "center":
        shifted = np.fft.fftshift(spectrum.data, axes=(0, 1))
    elif kind == "corner":
        shifted = np.fft.ifftshift(spectrum.data, axes=(0, 1))
    else:
        raise ValueError("Mode 只能是 center 或 corner")
    return Result([], [ComplexImage(shifted)])


@operator(
    "灰階與濾波",
    [],
    [],
    [P("Complex", "複數頻譜"), P("Factor", "倍率")],
    [P("ComplexScaled", "縮放後的頻譜")],
    "縮放頻譜",
    "每個頻率都乘上同一個數。反轉換後，灰階也變成原來的倍率。",
    "scale_fft (Complex, 0.5, ComplexScaled)",
)
def scale_fft(complex_image, factor, *, ctx):
    spectrum = _expect_complex(complex_image)
    return Result([], [ComplexImage(spectrum.data * scalar(factor, "Factor"))])


@operator(
    "灰階與濾波",
    [],
    [],
    [P("Real", "實部"), P("Imag", "虛部"), P("Width", "寬"), P("Height", "高")],
    [P("Complex", "常數複數影像")],
    "建立常數複數影像",
    "整張都是同一個複數。可用來和頻譜相乘。",
    "gen_complex_const (1, 0, 80, 60, Complex)",
)
def gen_complex_const(real, imag, width, height, *, ctx):
    wide, high = whole(width, "Width"), whole(height, "Height")
    if wide < 1 or high < 1:
        raise ValueError("寬和高至少要 1")
    value = complex(scalar(real, "Real"), scalar(imag, "Imag"))
    return Result([], [ComplexImage(np.full((high, wide), value, np.complex128))])


@operator(
    "灰階與濾波",
    [],
    [],
    [P("Complex1", "第一張複數影像"), P("Complex2", "第二張複數影像")],
    [P("ComplexProduct", "逐點相乘")],
    "頻譜相乘",
    "尺寸要相同。在頻率相乘就是在空間做循環卷積。",
    "mult_fft (Complex1, Complex2, ComplexProduct)",
)
def mult_fft(complex1, complex2, *, ctx):
    left, right = _expect_complex(complex1), _expect_complex(complex2)
    if left.data.shape != right.data.shape:
        raise ValueError("兩張複數影像的尺寸要相同")
    return Result([], [ComplexImage(left.data * right.data)])


@operator(
    "灰階與濾波",
    [P("Image1", "第一張影像"), P("Image2", "第二張影像")],
    [],
    [],
    [P("Row", "列位移"), P("Column", "行位移"), P("Score", "相關峰值")],
    "相位相關找位移",
    "回傳的是把第一張對到第二張要加的列和行。位移以左上角為原點，超過半張會捲到另一邊。兩張尺寸要相同，彩色先轉成亮度。完全重合時峰值是 1、位移是 0。",
    "phase_correlation (Image1, Image2, Row, Column, Score)",
)
def phase_correlation(image1, image2, *, ctx):
    first = expect_image(image1, "Image1").gray().astype(np.float64)
    second = expect_image(image2, "Image2").gray().astype(np.float64)
    if first.shape != second.shape:
        raise ValueError("兩張影像的尺寸要相同")
    cross = np.fft.fft2(second) * np.conj(np.fft.fft2(first))
    magnitude = np.abs(cross)
    cross = np.divide(cross, magnitude, out=np.zeros_like(cross), where=magnitude > 1e-8)
    corr = np.fft.ifft2(cross).real
    row, column = np.unravel_index(int(np.argmax(corr)), corr.shape)
    return Result([], [int(row), int(column), float(corr[row, column])])


def _same_complex(left: ComplexImage, right: ComplexImage) -> None:
    if left.data.shape != right.data.shape:
        raise ValueError("兩張複數影像的尺寸要相同")


@operator(
    "灰階與濾波",
    [],
    [],
    [P("Complex", "複數頻譜")],
    [P("ComplexConj", "共軛頻譜")],
    "取頻譜的共軛",
    "虛部變號，實部不變。實數影像的頻譜取共軛再反轉換，會得到左右上下都翻轉的循環結果。",
    "conj_fft (Complex, ComplexConj)",
)
def conj_fft(complex_image, *, ctx):
    spectrum = _expect_complex(complex_image)
    return Result([], [ComplexImage(np.conj(spectrum.data))])


@operator(
    "灰階與濾波",
    [],
    [],
    [P("Complex1", "第一張複數影像"), P("Complex2", "第二張複數影像")],
    [P("ComplexSum", "逐點相加")],
    "頻譜相加",
    "尺寸要相同。常數影像的頻譜加自己，反轉換後灰階變成兩倍。",
    "add_fft (Complex1, Complex2, ComplexSum)",
)
def add_fft(complex1, complex2, *, ctx):
    left, right = _expect_complex(complex1), _expect_complex(complex2)
    _same_complex(left, right)
    return Result([], [ComplexImage(left.data + right.data)])


@operator(
    "灰階與濾波",
    [],
    [],
    [P("Complex1", "被減數"), P("Complex2", "減數")],
    [P("ComplexDiff", "逐點相減")],
    "頻譜相減",
    "尺寸要相同。同一張減自己，反轉換後是全黑。",
    "sub_fft (Complex1, Complex2, ComplexDiff)",
)
def sub_fft(complex1, complex2, *, ctx):
    left, right = _expect_complex(complex1), _expect_complex(complex2)
    _same_complex(left, right)
    return Result([], [ComplexImage(left.data - right.data)])


_FFT_MASK = ["'dc'", "'no_dc'", "'real'", "'imag'"]


@operator(
    "灰階與濾波",
    [],
    [],
    [P("Complex", "複數頻譜"), P("Mode", "dc、no_dc、real 或 imag", "'dc'", _FFT_MASK)],
    [P("ComplexMasked", "遮罩後的頻譜")],
    "遮罩頻譜",
    "dc 只留左上角。no_dc 把左上角清成 0。real 去掉虛部，imag 去掉實部。低頻還沒有 fftshift 時，直流就在 (0, 0)。",
    "mask_fft (Complex, 'dc', ComplexMasked)",
)
def mask_fft(complex_image, mode, *, ctx):
    spectrum = _expect_complex(complex_image)
    kind = _spectrum_mode(mode, "Mode")
    data = spectrum.data
    if kind == "dc":
        masked = np.zeros_like(data)
        masked[0, 0] = data[0, 0]
    elif kind == "no_dc":
        masked = data.copy()
        masked[0, 0] = 0
    elif kind == "real":
        masked = data.real.astype(np.complex128)
    elif kind == "imag":
        masked = (1j * data.imag).astype(np.complex128)
    else:
        raise ValueError("Mode 只能是 dc、no_dc、real 或 imag")
    return Result([], [ComplexImage(masked)])


@operator(
    "灰階與濾波",
    [],
    [],
    [P("Complex", "複數頻譜")],
    [P("Energy", "能量")],
    "計算頻譜能量",
    "能量是所有頻率絕對值平方的平均，等於空間上每個像素平方的總和。常數灰階 g 的能量是 g 的平方乘上寬乘高。",
    "fft_energy (Complex, Energy)",
)
def fft_energy(complex_image, *, ctx):
    spectrum = _expect_complex(complex_image)
    energy = float(np.sum(np.abs(spectrum.data) ** 2) / spectrum.data.size)
    return Result([], [energy])


@operator(
    "灰階與濾波",
    [],
    [],
    [P("Complex1", "被除數"), P("Complex2", "除數")],
    [P("ComplexQuotient", "逐點相除")],
    "頻譜相除",
    "尺寸要相同。除數的絕對值接近 0 時，那個頻率的結果是 0。",
    "div_fft (Complex1, Complex2, ComplexQuotient)",
)
def div_fft(complex1, complex2, *, ctx):
    left, right = _expect_complex(complex1), _expect_complex(complex2)
    _same_complex(left, right)
    magnitude = np.abs(right.data)
    quotient = np.divide(left.data, right.data, out=np.zeros_like(left.data), where=magnitude > 1e-12)
    return Result([], [ComplexImage(quotient)])


_COMPLEX_VIEW = ["'real'", "'imag'", "'abs'", "'log'"]


@operator(
    "灰階與濾波",
    [],
    [P("Image", "看出來的灰階")],
    [P("Complex", "複數頻譜"), P("Mode", "real、imag、abs 或 log", "'abs'", _COMPLEX_VIEW)],
    [],
    "把頻譜畫成灰階",
    "real、imag、abs 直接四捨五入後裁到 0 到 255，很大的直流會飽和成 255。log 用 log(1+絕對值)，再拉到這張裡的最大值。",
    "complex_to_image (Image, Complex, 'log')",
)
def complex_to_image(complex_image, mode, *, ctx):
    spectrum = _expect_complex(complex_image)
    kind = _spectrum_mode(mode, "Mode")
    data = spectrum.data
    if kind == "real":
        plane = data.real
    elif kind == "imag":
        plane = data.imag
    elif kind == "abs":
        plane = np.abs(data)
    elif kind == "log":
        plane = np.log1p(np.abs(data))
        peak = float(plane.max()) if plane.size else 0.0
        plane = np.zeros_like(plane) if peak <= 0 else plane / peak * 255.0
    else:
        raise ValueError("Mode 只能是 real、imag、abs 或 log")
    return Result([Image(np.clip(np.round(plane), 0, 255).astype(np.uint8))], [])


@operator(
    "灰階與濾波",
    [P("Image", "要卷積的影像"), P("Kernel", "核，灰階就是權重")],
    [P("ImageResult", "循環卷積結果")],
    [],
    [],
    "用傅立葉做循環卷積",
    "核的 (0, 0) 放在影像左上角。核比較小時，其餘補 0。權重不會再除以總和，灰階 1 就是乘 1。超出影像的部分從對面捲回來。彩色先轉成亮度。",
    "convol_fft (Image, Kernel, ImageResult)",
)
def convol_fft(image, kernel, *, ctx):
    picture = expect_image(image, "Image")
    weights = expect_image(kernel, "Kernel")
    source = picture.gray().astype(np.float64)
    mask = weights.gray().astype(np.float64)
    if mask.shape[0] > source.shape[0] or mask.shape[1] > source.shape[1]:
        raise ValueError("核不能比影像大")
    embedded = np.zeros_like(source)
    embedded[: mask.shape[0], : mask.shape[1]] = mask
    result = np.fft.ifft2(np.fft.fft2(source) * np.fft.fft2(embedded)).real
    return Result([Image(np.clip(np.round(result), 0, 255).astype(np.uint8))], [])


def _freq_radius(height: int, width: int) -> np.ndarray:
    rows = np.minimum(np.arange(height), height - np.arange(height)).astype(np.float64)[:, None]
    cols = np.minimum(np.arange(width), width - np.arange(width)).astype(np.float64)[None, :]
    return np.hypot(rows, cols)


def _embed_gray(image: Image, template: Image, what: str) -> tuple[np.ndarray, np.ndarray]:
    source = image.gray().astype(np.float64)
    mask = template.gray().astype(np.float64)
    if mask.shape[0] > source.shape[0] or mask.shape[1] > source.shape[1]:
        raise ValueError(f"{what}不能比影像大")
    embedded = np.zeros_like(source)
    embedded[: mask.shape[0], : mask.shape[1]] = mask
    return source, embedded


def _correlation(image: Image, template: Image) -> np.ndarray:
    source, embedded = _embed_gray(image, template, "樣板")
    return np.fft.ifft2(np.fft.fft2(source) * np.conj(np.fft.fft2(embedded))).real


@operator(
    "幾何",
    [P("Image", "輸入影像")],
    [P("ImageShifted", "循環平移後的影像")],
    [P("Row", "列位移，正值往下"), P("Column", "行位移，正值往右")],
    [],
    "循環平移影像",
    "位移會四捨五入成整數像素。超出的部分從對面捲回來。定義域一起移。",
    "shift_image (Image, ImageShifted, 5, 2)",
)
def shift_image(image, row, column, *, ctx):
    picture = expect_image(image, "Image")
    drow, dcol = int(round(scalar(row, "Row"))), int(round(scalar(column, "Column")))
    data = np.roll(np.roll(picture.data, drow, axis=0), dcol, axis=1)
    domain = None if picture.domain is None else np.roll(np.roll(picture.domain, drow, axis=0), dcol, axis=1)
    return Result([Image(data, domain)], [])


_PAD = ["'constant'", "'mirror'", "'cyclic'"]


@operator(
    "幾何",
    [P("Image", "輸入影像")],
    [P("ImagePadded", "補邊後的影像")],
    [
        P("Top", "上方列數"),
        P("Left", "左方行數"),
        P("Bottom", "下方列數"),
        P("Right", "右方行數"),
        P("Gray", "constant 時的灰階", "0"),
        P("Mode", "constant、mirror 或 cyclic", "'constant'", _PAD),
    ],
    [],
    "幫影像補邊",
    "constant 用同一個灰階。mirror 以邊緣為鏡面，不重複邊緣那一點，補邊不能比該邊的影像還長。cyclic 從對面捲過來。補完的定義域是整張新影像，尺寸也會變成之後區域的參考尺寸。",
    "pad_image (Image, ImagePadded, 1, 0, 0, 0, 0, 'constant')",
)
def pad_image(image, top, left, bottom, right, gray, mode, *, ctx):
    picture = expect_image(image, "Image")
    amounts = [whole(top, "Top"), whole(left, "Left"), whole(bottom, "Bottom"), whole(right, "Right")]
    if any(item < 0 for item in amounts):
        raise ValueError("補邊不能是負的")
    kind = _spectrum_mode(mode, "Mode")
    above, before, below, after = amounts
    if kind == "mirror" and (
        above >= picture.height or below >= picture.height or before >= picture.width or after >= picture.width
    ):
        raise ValueError("鏡射補邊不能比影像本身還長")
    pad = ((above, below), (before, after))
    if picture.data.ndim == 3:
        pad = pad + ((0, 0),)
    if kind == "constant":
        value = int(np.clip(round(scalar(gray, "Gray")), 0, 255))
        data = np.pad(picture.data, pad, mode="constant", constant_values=value)
    elif kind == "mirror":
        data = np.pad(picture.data, pad, mode="reflect")
    elif kind == "cyclic":
        data = np.pad(picture.data, pad, mode="wrap")
    else:
        raise ValueError("Mode 只能是 constant、mirror 或 cyclic")
    return Result([Image(data)], [])


@operator(
    "灰階與濾波",
    [],
    [],
    [P("Complex", "複數頻譜"), P("Row", "列位移，正值往下"), P("Column", "行位移，正值往右")],
    [P("ComplexShifted", "平移後的頻譜")],
    "在頻率裡平移",
    "用相位把影像平移，可以是小數。整數時結果和 shift_image 一樣，超出的部分從對面捲回來。",
    "translate_fft (Complex, 5, 2, ComplexShifted)",
)
def translate_fft(complex_image, row, column, *, ctx):
    spectrum = _expect_complex(complex_image)
    drow, dcol = scalar(row, "Row"), scalar(column, "Column")
    height, width = spectrum.data.shape[:2]
    freqs_row = np.fft.fftfreq(height)[:, None]
    freqs_col = np.fft.fftfreq(width)[None, :]
    phase = np.exp(-2j * np.pi * (freqs_row * drow + freqs_col * dcol))
    return Result([], [ComplexImage(spectrum.data * phase)])


_FREQ_PASS = ["'low'", "'high'"]


@operator(
    "灰階與濾波",
    [],
    [],
    [
        P("Complex", "複數頻譜"),
        P("Mode", "low 留下靠近直流的頻率，high 去掉", "'low'", _FREQ_PASS),
        P("Radius", "頻率半徑，單位是頻率格"),
    ],
    [P("ComplexFiltered", "濾波後的頻譜")],
    "用半徑濾頻率",
    "距離從左上角量起，也把對面的低頻算進來。low 留下距離不大於半徑的頻率，high 去掉那些。半徑 0 的 low 只留直流，反轉換後整張都是平均值。",
    "freq_filter (Complex, 'low', 0, ComplexFiltered)",
)
def freq_filter(complex_image, mode, radius, *, ctx):
    spectrum = _expect_complex(complex_image)
    kind = _spectrum_mode(mode, "Mode")
    limit = scalar(radius, "Radius")
    if limit < 0:
        raise ValueError("Radius 不能是負的")
    if kind not in ("low", "high"):
        raise ValueError("Mode 只能是 low 或 high")
    distance = _freq_radius(*spectrum.data.shape[:2])
    keep = distance <= limit if kind == "low" else distance > limit
    return Result([], [ComplexImage(np.where(keep, spectrum.data, 0))])


@operator(
    "灰階與濾波",
    [],
    [],
    [P("Complex", "複數頻譜"), P("Radius", "頻率半徑，含")],
    [P("Energy", "半徑內的能量")],
    "計算一塊頻率的能量",
    "把距離不大於半徑的頻率，絕對值平方加總後除以整張的點數。半徑 0 只有直流。常數影像的能量全在這裡。",
    "freq_energy (Complex, 0, Energy)",
)
def freq_energy(complex_image, radius, *, ctx):
    spectrum = _expect_complex(complex_image)
    limit = scalar(radius, "Radius")
    if limit < 0:
        raise ValueError("Radius 不能是負的")
    distance = _freq_radius(*spectrum.data.shape[:2])
    selected = np.where(distance <= limit, spectrum.data, 0)
    energy = float(np.sum(np.abs(selected) ** 2) / spectrum.data.size)
    return Result([], [energy])


@operator(
    "灰階與濾波",
    [],
    [],
    [P("Complex", "複數頻譜")],
    [P("ComplexNormalized", "單位振幅頻譜")],
    "把每個頻率變成單位長度",
    "每個不是 0 的頻率都除以自己的絕對值，只留相位。本來是 0 的頻率還是 0。",
    "normalize_fft (Complex, ComplexNormalized)",
)
def normalize_fft(complex_image, *, ctx):
    spectrum = _expect_complex(complex_image)
    magnitude = np.abs(spectrum.data)
    normal = np.divide(spectrum.data, magnitude, out=np.zeros_like(spectrum.data), where=magnitude > 1e-12)
    return Result([], [ComplexImage(normal)])


@operator(
    "灰階與濾波",
    [P("Image", "要搜尋的影像"), P("Template", "樣板，灰階就是權重")],
    [],
    [],
    [P("Row", "樣板左上角的列"), P("Column", "樣板左上角的行"), P("Score", "相關分數")],
    "用循環相關找樣板",
    "樣板的 (0, 0) 放在影像左上角，比較小時其餘補 0。回傳分數最高的位置，分數是灰階乘積的和，不會飽和。樣板不能比影像大。彩色先轉成亮度。",
    "locate_template (Image, Template, Row, Column, Score)",
)
def locate_template(image, template, *, ctx):
    score = _correlation(expect_image(image, "Image"), expect_image(template, "Template"))
    row, column = np.unravel_index(int(np.argmax(score)), score.shape)
    return Result([], [int(row), int(column), float(score[row, column])])


@operator(
    "灰階與濾波",
    [P("Image", "要搜尋的影像"), P("Template", "樣板，灰階就是權重")],
    [P("ImageScore", "相關圖")],
    [],
    [],
    "把循環相關畫成灰階",
    "和 locate_template 同一種相關。負分變成 0，超過 255 會飽和。要看真正的分數請用 locate_template。",
    "correlation_image (Image, Template, ImageScore)",
)
def correlation_image(image, template, *, ctx):
    score = _correlation(expect_image(image, "Image"), expect_image(template, "Template"))
    return Result([Image(np.clip(np.round(score), 0, 255).astype(np.uint8))], [])
