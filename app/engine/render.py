from __future__ import annotations

import base64
import re
from pathlib import Path

import cv2
import numpy as np

from .objects import Image, Region, XLD

NAMED_COLORS = {
    "black": "#000000",
    "white": "#ffffff",
    "red": "#ff2a2a",
    "green": "#00e05a",
    "blue": "#2f7dff",
    "yellow": "#ffe14a",
    "cyan": "#2ce6e6",
    "magenta": "#f04ad4",
    "orange": "#ff8a1e",
    "coral": "#ff6a4a",
    "gold": "#e6c14a",
    "pink": "#ff7ab8",
    "gray": "#9aa3b2",
    "grey": "#9aa3b2",
    "spring green": "#00ff7f",
    "forest green": "#228b22",
}


def normalize_color(value: str) -> str:
    key = str(value).strip().lower()
    if key in NAMED_COLORS:
        return NAMED_COLORS[key]
    if re.fullmatch(r"#(?:[0-9a-f]{3}|[0-9a-f]{6})", key):
        return key
    raise ValueError(f"不認識的顏色：{value}")


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    raw = color.lstrip("#")
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    return tuple(int(raw[i : i + 2], 16) for i in (0, 2, 4))


def read_cv_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def write_cv_image(path: Path, image: np.ndarray) -> bool:
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))
    return True


def to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if image.dtype == np.uint16:
        return (image / 257).astype(np.uint8)
    if np.issubdtype(image.dtype, np.floating):
        peak = float(np.max(image)) if image.size else 1.0
        if peak <= 1.0:
            return np.clip(image * 255.0, 0, 255).astype(np.uint8)
        return np.clip(image, 0, 255).astype(np.uint8)
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def png_data_url(array: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", array)
    if not ok:
        raise RuntimeError("影像編碼失敗")
    payload = base64.b64encode(buf.tobytes()).decode("ascii")
    return "data:image/png;base64," + payload


def jpeg_thumb(array: np.ndarray, max_edge: int = 112) -> str:
    height, width = array.shape[:2]
    scale = max_edge / max(height, width, 1)
    view = array
    if scale < 1:
        view = cv2.resize(
            array,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", view, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ok:
        return ""
    payload = base64.b64encode(buf.tobytes()).decode("ascii")
    return "data:image/jpeg;base64," + payload


def apply_domain_dim(image: Image) -> np.ndarray:
    if image.domain is None:
        return image.data
    faded = (image.data.astype(np.uint16) // 3).astype(np.uint8)
    out = image.data.copy()
    outside = image.domain == 0
    out[outside] = faded[outside]
    return out


def contours_of(mask: np.ndarray) -> list[list[list[int]]]:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    found = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = found[0] if len(found) == 2 else found[1]
    lines = []
    for contour in cnts:
        if len(contour) < 2:
            continue
        if len(contour) > 800:
            contour = cv2.approxPolyDP(contour, 0.7, True)
        points = contour.reshape(-1, 2)
        lines.append([[int(x), int(y)] for x, y in points])
    return lines


def region_thumb(region: Region) -> np.ndarray:
    canvas = np.zeros((region.height, region.width), np.uint8)
    for mask in region.objects:
        canvas = np.maximum(canvas, mask)
    return canvas


def xld_thumb(xld: XLD) -> np.ndarray:
    canvas = np.zeros((max(xld.height, 1), max(xld.width, 1)), np.uint8)
    for line in xld.contours:
        if len(line) < 2:
            continue
        pts = np.round(np.array(line, np.float32)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, 255, 1)
    return canvas


def fill_overlay(region: Region, color: str, alpha: int = 110, colors: list[str] | None = None) -> np.ndarray:
    overlay = np.zeros((region.height, region.width, 4), np.uint8)
    for index, mask in enumerate(region.objects):
        chosen = colors[index] if colors and index < len(colors) else color
        red, green, blue = hex_to_rgb(chosen)
        on = mask > 0
        overlay[on, 0] = blue
        overlay[on, 1] = green
        overlay[on, 2] = red
        overlay[on, 3] = alpha
    return overlay


def compose_display(base: Image | None, overlays: list) -> dict | None:
    if base is None and not overlays:
        return None
    width = base.width if base is not None else None
    height = base.height if base is not None else None
    image = png_data_url(apply_domain_dim(base)) if base is not None else None
    shapes = []
    drawn = 0
    truncated = False
    for item in overlays:
        kind = item["kind"]
        if kind == "region":
            region: Region = item["region"]
            width = width or region.width
            height = height or region.height
            colors = item.get("colors")
            if item["draw"] == "fill":
                shapes.append(
                    {
                        "type": "fill",
                        "image": png_data_url(
                            fill_overlay(region, item["color"], alpha=int(item.get("alpha", 110)), colors=colors)
                        ),
                    }
                )
                continue
            for index, mask in enumerate(region.objects):
                if drawn >= 1500:
                    truncated = True
                    break
                color = colors[index] if colors and index < len(colors) else item["color"]
                stored = region.links[index]["points"] if region.links and index < len(region.links) else None
                for points in (stored if stored else contours_of(mask)):
                    if len(points) < 2:
                        continue
                    shapes.append(
                        {
                            "type": "polyline",
                            "color": color,
                            "width": item["width"],
                            "closed": True,
                            "points": points,
                            "object": index,
                            "source": item.get("source"),
                            "kind": "region",
                        }
                    )
                drawn += 1
        elif kind == "xld":
            xld: XLD = item["xld"]
            width = width or xld.width
            height = height or xld.height
            palette = item.get("colors")
            for index, points in enumerate(xld.contours):
                if len(points) < 2:
                    continue
                if drawn >= 4000:
                    truncated = True
                    break
                color = palette[index] if palette and index < len(palette) else item["color"]
                shapes.append(
                    {
                        "type": "polyline",
                        "color": color,
                        "width": item["width"],
                        "closed": False,
                        "points": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in points],
                        "object": index,
                        "source": item.get("source"),
                        "kind": "xld",
                    }
                )
                drawn += 1
        elif kind == "text":
            shapes.append(
                {
                    "type": "text",
                    "text": item["text"],
                    "row": item["row"],
                    "col": item["col"],
                    "color": item["color"],
                }
            )
    if truncated:
        shapes.append(
            {
                "type": "text",
                "text": "物件太多，只畫出前面一部分",
                "row": 18,
                "col": 8,
                "color": "#ffe14a",
            }
        )
    if width is None:
        width, height = 640, 480
    return {"width": width, "height": height, "image": image, "shapes": shapes}
