"""Local cameras and image folders.

There is no framegrabber SDK. Name is File or OpenCV. The call is shorter
than HALCON's open_framegrabber: Name, Device, ColorSpace, Width, Height.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2

from ..objects import Image, Result
from ..operators import P, operator, text, whole
from ..render import read_cv_image, to_uint8

_NAMES = ["'File'", "'OpenCV'"]
_COLORS = ["'gray'", "'rgb'", "'default'"]
_QUERIES = ["'device'", "'color_space'", "'name'"]
_PARAMS = ["'image_dir'", "'color_space'", "'start_index'"]
_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class AcqHandle:
    kind: str
    device: str
    color: str
    width: int
    height: int
    paths: list = field(default_factory=list)
    cursor: int = 0
    cap: object | None = None
    closed: bool = False

    def summary(self) -> str:
        if self.closed:
            return "採集  已關閉"
        if self.kind == "file":
            return f"採集  檔案 {self.cursor}/{len(self.paths)}"
        return f"採集  相機 {self.device}"


def _alive(handle, name: str) -> AcqHandle:
    if not isinstance(handle, AcqHandle):
        raise TypeError(f"{name} 必須是 open_framegrabber 建立的採集")
    if handle.closed:
        raise RuntimeError("採集已經 close_framegrabber")
    return handle


def _color_name(value, name: str) -> str:
    color = text(value, name).strip().strip("'\"").lower()
    if color in ("default", ""):
        return "gray"
    if color not in ("gray", "rgb"):
        raise ValueError("ColorSpace 只能是 gray、rgb 或 default")
    return color


def _device_path(ctx, device: str) -> Path:
    raw = Path(device).expanduser()
    candidates = [raw] if raw.is_absolute() else [ctx.root / raw, ctx.root / "samples" / raw, ctx.root / "workspace" / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"找不到 {device}")


def _image_paths(folder: Path) -> list[Path]:
    paths = [item for item in folder.iterdir() if item.is_file() and item.suffix.lower() in _SUFFIXES]
    return sorted(paths, key=lambda item: item.name.lower())


def _paint(data: np.ndarray, color: str, width: int, height: int) -> np.ndarray:
    data = to_uint8(data)
    if data.ndim == 3 and data.shape[2] == 4:
        data = cv2.cvtColor(data, cv2.COLOR_BGRA2BGR)
    if color == "gray" and data.ndim == 3:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    elif color == "rgb" and data.ndim == 2:
        data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    src_h, src_w = data.shape[:2]
    if width <= 0 and height <= 0:
        return data
    if width <= 0:
        width = max(1, int(round(src_w * height / src_h)))
    if height <= 0:
        height = max(1, int(round(src_h * width / src_w)))
    return cv2.resize(data, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)


def _note_image(ctx) -> None:
    if ctx is None:
        return
    ctx.image_stamp = int(getattr(ctx, "image_stamp", 0)) + 1
    ctx.threshold_gray = None
    ctx.threshold_domain = None
    ctx.threshold_mask = None
    ctx.threshold_region = None
    ctx.histogram = None
    ctx.preview_region = None


def _load_file(handle: AcqHandle) -> np.ndarray:
    if handle.cursor >= len(handle.paths):
        raise RuntimeError("沒有下一張影像")
    path = handle.paths[handle.cursor]
    handle.cursor += 1
    data = read_cv_image(path)
    if data is None:
        raise RuntimeError(f"無法讀取影像 {path}")
    return _paint(data, handle.color, handle.width, handle.height)


def _load_camera(handle: AcqHandle, flush: bool) -> np.ndarray:
    cap = handle.cap
    if cap is None or not cap.isOpened():
        raise RuntimeError("相機沒有打開")
    frame = None
    pulls = 2 if flush else 1
    for _ in range(pulls):
        ok, grabbed = cap.read()
        if ok and grabbed is not None:
            frame = grabbed
    if frame is None:
        raise RuntimeError("讀不到畫面")
    return _paint(frame, handle.color, handle.width, handle.height)


def _grab(handle: AcqHandle, flush: bool, ctx) -> Result:
    picture = _load_file(handle) if handle.kind == "file" else _load_camera(handle, flush)
    _note_image(ctx)
    return Result([Image(picture)], [])


@operator(
    "相機",
    [],
    [],
    [
        P("Name", "File 讀資料夾或單張圖，OpenCV 開本機相機", "'File'", _NAMES),
        P("Device", "檔案路徑，或相機編號", "'0'"),
        P("ColorSpace", "gray、rgb 或 default", "'gray'", _COLORS),
        P("Width", "0 表示維持原寬", "0"),
        P("Height", "0 表示維持原高", "0"),
    ],
    [P("AcqHandle", "採集")],
    "開啟檔案或本機相機",
    "沒有廠商採集卡。參數比 HALCON 少，順序是名稱、裝置、色彩、寬、高。"
    "File 的 Device 可以是一張圖或一個資料夾，資料夾裡的圖按檔名排序。"
    "OpenCV 的 Device 是相機編號，0 是第一台。Width 或 Height 只有一個大於 0 時會保持比例。",
    "Vizion.open_framegrabber ('File', 'samples/code39.png', AcqHandle)",
)
def open_framegrabber(name, device, color_space, width, height, *, ctx):
    kind = text(name, "Name").strip().strip("'\"").lower()
    if kind not in ("file", "opencv"):
        raise ValueError("Name 這版只接受 File 或 OpenCV")
    device_text = text(device, "Device").strip().strip("'\"")
    color = _color_name(color_space, "ColorSpace")
    frame_w = whole(width, "Width")
    frame_h = whole(height, "Height")
    handle = AcqHandle(kind, device_text, color, frame_w, frame_h)
    if kind == "file":
        path = _device_path(ctx, device_text)
        handle.paths = [path] if path.is_file() else _image_paths(path)
        if not handle.paths:
            raise RuntimeError(f"{path} 裡沒有影像")
    else:
        try:
            index = int(device_text)
        except ValueError as exc:
            raise ValueError("OpenCV 的 Device 必須是相機編號") from exc
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"打不開相機 {index}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if frame_w > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
        if frame_h > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
        handle.cap = cap
        handle.device = str(index)
    return Result([], [handle])


@operator(
    "相機",
    [],
    [P("Image", "這一張影像")],
    [P("AcqHandle", "採集")],
    [],
    "抓一張影像",
    "檔案會依序往下讀，讀完會失敗。相機讀目前畫面。",
    "Vizion.grab_image (Image, AcqHandle)",
)
def grab_image(handle, *, ctx):
    return _grab(_alive(handle, "AcqHandle"), False, ctx)


@operator(
    "相機",
    [],
    [P("Image", "最新的影像")],
    [P("AcqHandle", "採集"), P("MaxDelay", "這版不依這個時間等待", "-1")],
    [],
    "抓最新的影像",
    "相機連讀兩次，留下較新的那張。檔案沒有延遲，和 grab_image 一樣往下讀一張。",
    "Vizion.grab_image_async (Image, AcqHandle, -1)",
)
def grab_image_async(handle, _delay, *, ctx):
    return _grab(_alive(handle, "AcqHandle"), True, ctx)


@operator(
    "相機",
    [],
    [],
    [P("AcqHandle", "採集")],
    [],
    "關閉採集",
    "會放開相機。之後不能再抓圖。",
    "Vizion.close_framegrabber (AcqHandle)",
)
def close_framegrabber(handle, *, ctx):
    handle = _alive(handle, "AcqHandle")
    if handle.cap is not None:
        handle.cap.release()
        handle.cap = None
    handle.closed = True
    return Result([], [])


@operator(
    "相機",
    [],
    [],
    [
        P("Name", "File 或 OpenCV", "'File'", _NAMES),
        P("Query", "要問的項目", "'device'", _QUERIES),
    ],
    [P("Information", "說明")],
    "查詢本機採集能做什麼",
    "這版不掃描每一台相機。device 對 File 是資料夾或檔案，對 OpenCV 是從 0 開始的編號。",
    "Vizion.info_framegrabber ('OpenCV', 'device', Information)",
)
def info_framegrabber(name, query, *, ctx):
    kind = text(name, "Name").strip().strip("'\"").lower()
    item = text(query, "Query").strip().strip("'\"").lower()
    if kind not in ("file", "opencv"):
        raise ValueError("Name 這版只接受 File 或 OpenCV")
    if item == "name":
        return Result([], ["File" if kind == "file" else "OpenCV"])
    if item == "color_space":
        return Result([], [["gray", "rgb"]])
    if item == "device":
        if kind == "file":
            return Result([], ["資料夾或單張影像的路徑"])
        return Result([], ["0"])
    raise ValueError("Query 只能是 device、color_space 或 name")


@operator(
    "相機",
    [],
    [],
    [
        P("AcqHandle", "採集"),
        P("Param", "image_dir、color_space 或 start_index", "'color_space'", _PARAMS),
        P("Value", "新的值"),
    ],
    [],
    "修改採集參數",
    "image_dir 只對檔案有效，會把讀取位置歸零。start_index 從 0 開始。color_space 可以是 gray 或 rgb。",
    "Vizion.set_framegrabber_param (AcqHandle, 'color_space', 'gray')",
)
def set_framegrabber_param(handle, param, value, *, ctx):
    handle = _alive(handle, "AcqHandle")
    name = text(param, "Param").strip().strip("'\"").lower()
    if name == "color_space":
        handle.color = _color_name(value, "Value")
    elif name == "start_index":
        index = whole(value, "Value")
        if index < 0 or (handle.kind == "file" and index > len(handle.paths)):
            raise ValueError("start_index 超出範圍")
        handle.cursor = index
    elif name == "image_dir":
        if handle.kind != "file":
            raise ValueError("image_dir 只對 File 有效")
        path = _device_path(ctx, text(value, "Value").strip().strip("'\""))
        handle.paths = [path] if path.is_file() else _image_paths(path)
        if not handle.paths:
            raise RuntimeError(f"{path} 裡沒有影像")
        handle.device = str(path)
        handle.cursor = 0
    else:
        raise ValueError("Param 只能是 image_dir、color_space 或 start_index")
    return Result([], [])
