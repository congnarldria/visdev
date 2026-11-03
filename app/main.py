from __future__ import annotations

import asyncio
import base64
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import cv2
import numpy as np

from app.engine.objects import Image, Region, XLD
from app.engine.operators import (
    GRAY_FEATURES,
    REGION_FEATURES,
    XLD_FEATURES,
    gray_measures,
    halcon_components,
    list_operators,
    merge_within_distance,
    opencv_components,
    preview_mask,
    refine_polyline,
    region_measures,
    subpix_parts,
    subpixel_boundary,
    xld_measures,
)
from app.engine.parser import ParseError
from app.engine.render import contours_of, fill_overlay, normalize_color, png_data_url, read_cv_image
from app.engine.samples_gen import ensure_samples
from app.engine.gauge import distance_pp as gauge_distance_pp
from app.engine.gauge import measure_point as gauge_measure_point
from app.engine.session import Session, _remember_thumb, encode_thumb_job, omit_known_state
from app.engine.shape_model import MATCH_COLORS

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
SESSIONS: dict[str, Session] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_samples(ROOT)
    yield


app = FastAPI(lifespan=lifespan)


class ExecIn(BaseModel):
    id: str | None = None
    source: str = ""
    breakpoints: list[int] = []
    action: str
    trace: bool = False
    regionColor: str | None = None
    line: int | None = None
    project: str | None = None


class BreakpointsIn(BaseModel):
    lines: list[int] = []


class SaveIn(BaseModel):
    name: str
    source: str


class ProjectIn(BaseModel):
    name: str


class RenameProjectIn(BaseModel):
    name: str


class ExportCsIn(BaseModel):
    source: str = ""
    name: str = "ExportedProgram"


class FeatureIn(BaseModel):
    id: str
    name: str
    index: int
    kind: str = "region"
    region: list[str] = []
    xld: list[str] = []
    gray: list[str] = []
    subpixel: bool = True


class PreviewIn(BaseModel):
    id: str
    kind: str = "range"
    min: int = 0
    max: int = 255
    threshold: int = 0
    lightDark: str = "binary"
    color: str = "#ff3b30"
    image: bool = False
    connect: bool = False
    distance: float = 1
    minArea: int = 1


class RegionColorIn(BaseModel):
    id: str
    color: str


class SubpixIn(BaseModel):
    id: str
    threshold: float = 128
    image: bool = False
    color: str = "#ff3b30"


class CalibViewIn(BaseModel):
    index: int | None = None


class GaugePreviewIn(BaseModel):
    id: str
    row1: float
    col1: float
    row2: float
    col2: float
    sigma: float = 1
    threshold: float = 8
    transition: str = "all"
    select: str = "all"


def trim_sessions():
    while len(SESSIONS) > 20:
        _sid, old = next(iter(SESSIONS.items()))
        SESSIONS.pop(_sid, None)
        old._release_handles()


def session_for(body: ExecIn) -> Session:
    current = SESSIONS.get(body.id) if body.id else None
    if current is None or current.source != body.source:
        if current is not None and current.source != body.source:
            current._release_handles()
        current = Session(body.source, ROOT)
        SESSIONS[current.id] = current
        trim_sessions()
    current.breakpoints = {int(line) for line in body.breakpoints}
    current.trace = bool(body.trace)
    if body.project:
        try:
            current.project_name = clean_project_name(body.project)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    else:
        current.project_name = None
    return current


@app.middleware("http")
async def no_cache(request, call_next):
    response = await call_next(request)
    if request.url.path in ("/", "/bin", "/features", "/threshold", "/gfx") or request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/bin")
def bin_tool():
    return FileResponse(STATIC / "bin.html")


@app.get("/features")
def feature_tool():
    return FileResponse(STATIC / "features.html")


@app.get("/threshold")
def threshold_tool():
    return FileResponse(STATIC / "threshold.html")


@app.get("/gfx")
def gfx_tool():
    return FileResponse(STATIC / "gfx.html")


def _session_or_404(sid: str) -> Session:
    current = SESSIONS.get(sid)
    if current is None:
        raise HTTPException(404, "沒有這個執行階段，請先按重設")
    return current


def _snapshot(current: Session) -> dict:
    with current.lock:
        state = current.public_state(defer_display=True)
        # 和這張畫面一起讀。執行緒只在鎖內改 running，避免預覽配上「已停止」。
        state["running"] = bool(getattr(current, "running", False))
        job = state.pop("_display_job", None)
        thumb_jobs = state.pop("_thumb_jobs", None)
    if thumb_jobs:
        for item in thumb_jobs:
            item["thumb"] = encode_thumb_job(item)
            item["target"]["thumb"] = item["thumb"]
    if job is not None:
        payload = current.render_display_job(job)
        state["display"] = payload
    if job is not None or thumb_jobs:
        with current.lock:
            if job is not None:
                current.store_display(job, state["display"])
            if thumb_jobs:
                for item in thumb_jobs:
                    token = item["token"]
                    if token in current._thumb_cache:
                        continue
                    thumb = item["thumb"]
                    _remember_thumb(current._thumb_cache, token, lambda thumb=thumb: thumb)
    return state


@app.get("/api/session/{sid}/display")
def session_display(sid: str, displayRev: int | None = None):
    state = omit_known_state(_snapshot(_session_or_404(sid)), displayRev, None, None)
    return {
        "id": sid,
        "display": state.get("display"),
        "displayStamp": state.get("displayStamp"),
        "displayRev": state.get("displayRev"),
        "displayKept": bool(state.get("displayKept")),
        "running": state["running"],
    }


@app.get("/api/session/{sid}/state")
def session_state(
    sid: str,
    displayRev: int | None = None,
    varRev: int | None = None,
    logRev: int | None = None,
    logAt: int | None = None,
):
    return omit_known_state(_snapshot(_session_or_404(sid)), displayRev, varRev, logRev, logAt)


@app.get("/api/operators")
def operators():
    return list_operators()


@app.get("/api/ocr")
def ocr_engines():
    from app.engine.ocr import catalog

    return [
        {
            "key": item.key,
            "title": item.title,
            "group": item.group,
            "purpose": item.purpose,
            "installed": item.installed,
            "note": item.note,
        }
        for item in catalog().engines
    ]


def _program_title(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return path.stem
    for line in lines:
        text = line.strip()
        if text.startswith("*"):
            title = text.lstrip("*").strip()
            return title or path.stem
        if text and not text.startswith("//"):
            break
    return path.stem


_DATA_SUFFIX = {
    "image": (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
    "plate": (".cpd", ".descr"),
}


@app.get("/api/data-files")
def data_files(kind: str = "image", project: str | None = None):
    suffixes = _DATA_SUFFIX.get(kind)
    if suffixes is None:
        raise HTTPException(400, "不認識的檔案種類")
    found = []
    if project:
        try:
            folder = project_dir(project)
        except ValueError:
            folder = None
        if folder is not None and folder.is_dir():
            for path in sorted(folder.iterdir()):
                if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in suffixes:
                    found.append({"name": path.name, "path": path.name})
    for folder, prefix in ((ROOT / "samples", "samples"), (ROOT / "workspace", "workspace")):
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            if path.suffix.lower() not in suffixes:
                continue
            found.append({"name": path.name, "path": f"{prefix}/{path.name}"})
    return found


@app.get("/api/files")
def list_files():
    from app.engine.samples_gen import SAMPLE_ORDER

    def collect(folder: Path, kind: str, order: tuple[str, ...] = ()):
        if not folder.exists():
            return []
        files = {}
        for pattern in ("*.hdv", "*.viz"):
            files.update({path.name: path for path in folder.glob(pattern)})
        names = [name for name in order if name in files]
        names.extend(sorted(name for name in files if name not in order))
        return [{"name": name, "kind": kind, "title": _program_title(files[name])} for name in names]

    return {
        "samples": collect(ROOT / "samples", "samples", SAMPLE_ORDER),
        "workspace": collect(ROOT / "workspace", "workspace"),
    }


@app.get("/api/files/{kind}/{name}")
def read_program(kind: str, name: str):
    folder = {"samples": ROOT / "samples", "workspace": ROOT / "workspace"}.get(kind)
    if folder is None or Path(name).name != name or not name.lower().endswith((".hdv", ".viz")):
        raise HTTPException(400, "不正確的檔名")
    path = folder / name
    if not path.is_file():
        raise HTTPException(404, "找不到程式")
    return {"name": name, "kind": kind, "source": path.read_text(encoding="utf-8")}


@app.post("/api/export/csharp")
def export_csharp_program(body: ExportCsIn):
    from app.engine.export_cs import ExportError, csharp_class_name, export_csharp

    class_name = csharp_class_name(body.name)
    try:
        code = export_csharp(body.source, class_name)
    except (ExportError, ParseError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"name": f"{class_name}.cs", "source": code}


@app.post("/api/programs")
def save_program(body: SaveIn):
    name = Path(body.name).name.strip() or "program.viz"
    if not name.lower().endswith((".hdv", ".viz")):
        name += ".viz"
    if any(ch in name for ch in '\\/:*?"<>|'):
        raise HTTPException(400, "檔名含有不允許的字元")
    path = ROOT / "workspace" / name
    path.write_text(body.source.replace("\r\n", "\n").replace("\r", "\n"), encoding="utf-8")
    return {"name": name, "path": f"workspace/{name}"}


def clean_project_name(name: str) -> str:
    text = str(name or "").strip()
    if not text or text in (".", "..") or any(ch in text for ch in '\\/:*?"<>|'):
        raise ValueError("專案名稱不正確")
    return text


def clean_program_name(name: str) -> str:
    text = Path(str(name or "")).name.strip() or "program.viz"
    if text in (".", "..") or any(ch in text for ch in '\\/:*?"<>|'):
        raise ValueError("程式名稱不正確")
    if not text.lower().endswith((".hdv", ".viz")):
        text += ".viz"
    return text


def projects_dir() -> Path:
    folder = ROOT / "workspace" / "projects"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def project_dir(name: str) -> Path:
    return projects_dir() / clean_project_name(name)


def list_project_records() -> list[dict]:
    folder = projects_dir()
    names = []
    for path in folder.iterdir():
        if path.is_dir() and (path / "visdev.json").is_file():
            names.append(path.name)
    return [{"name": name} for name in sorted(names)]


def create_project_record(name: str) -> dict:
    folder = project_dir(name)
    if folder.exists():
        raise ValueError("已經有這個專案")
    folder.mkdir(parents=True)
    (folder / "visdev.json").write_text("{}\n", encoding="utf-8")
    return {"name": folder.name, "programs": []}


def project_programs(name: str) -> dict:
    folder = project_dir(name)
    if not folder.is_dir() or not (folder / "visdev.json").is_file():
        raise ValueError("找不到專案")
    files = sorted(path.name for path in folder.iterdir() if path.is_file() and path.suffix.lower() in (".viz", ".hdv"))
    return {"name": folder.name, "programs": [{"name": item} for item in files]}


_IMAGE_SUFFIX = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _workspace_image(raw: str) -> Path | None:
    text = str(raw).strip().replace("\\", "/")
    if not text or text.startswith("samples/"):
        return None
    workspace = (ROOT / "workspace").resolve()
    path = Path(text)
    if path.is_absolute():
        candidate = path
    elif text.lower().startswith("workspace/"):
        candidate = ROOT / text
    else:
        return None
    try:
        resolved = candidate.resolve()
        resolved.relative_to(workspace)
    except (OSError, ValueError):
        return None
    if not resolved.is_file() or resolved.suffix.lower() not in _IMAGE_SUFFIX:
        return None
    return resolved


def _copy_image_into(folder: Path, source: Path) -> str:
    try:
        source.resolve().relative_to(folder.resolve())
        return source.name
    except ValueError:
        pass
    dest = folder / source.name
    if dest.exists():
        try:
            if dest.is_file() and dest.read_bytes() == source.read_bytes():
                return dest.name
        except OSError:
            pass
        number = 2
        while (folder / f"{source.stem}_{number}{source.suffix}").exists():
            number += 1
        dest = folder / f"{source.stem}_{number}{source.suffix}"
    shutil.copy2(source, dest)
    return dest.name


def adopt_workspace_images(folder: Path, source: str) -> tuple[str, list[str]]:
    """Copy workspace images into the project and point the program at the file name."""
    copied: list[str] = []
    assigned: dict[Path, str] = {}

    def replace(match: re.Match) -> str:
        quote, body = match.group(1), match.group(2)
        image = _workspace_image(body)
        if image is None:
            return match.group(0)
        if image not in assigned:
            assigned[image] = _copy_image_into(folder, image)
            if assigned[image] not in copied and not _already_in_folder(folder, image):
                copied.append(assigned[image])
        return f"{quote}{assigned[image]}{quote}"

    rewritten = re.sub(r"""(['"])([^'"\r\n]*)\1""", replace, source)
    return rewritten, copied


def _already_in_folder(folder: Path, image: Path) -> bool:
    try:
        image.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


def write_project_program(project: str, name: str, source: str) -> dict:
    folder = project_dir(project)
    if not folder.is_dir():
        raise ValueError("找不到專案")
    filename = clean_program_name(name)
    text = str(source).replace("\r\n", "\n").replace("\r", "\n")
    text, images = adopt_workspace_images(folder, text)
    path = folder / filename
    path.write_text(text, encoding="utf-8")
    return {
        "project": folder.name,
        "name": filename,
        "path": f"workspace/projects/{folder.name}/{filename}",
        "source": text,
        "images": images,
    }


def rename_project_record(old: str, new: str) -> dict:
    folder = project_dir(old)
    if not folder.is_dir() or not (folder / "visdev.json").is_file():
        raise ValueError("找不到專案")
    target_name = clean_project_name(new)
    if target_name == folder.name:
        return {"name": target_name, "previous": folder.name}
    target = projects_dir() / target_name
    if target.exists():
        raise ValueError("已經有這個專案")
    folder.rename(target)
    return {"name": target.name, "previous": old}


def reveal_project_folder(name: str) -> dict:
    folder = project_dir(name)
    if not folder.is_dir() or not (folder / "visdev.json").is_file():
        raise ValueError("找不到專案")
    target = folder.resolve()
    opened = False
    if sys.platform == "win32":
        os.startfile(target)  # noqa: S606  Windows 檔案總管
        opened = True
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        try:
            subprocess.Popen(
                [opener, str(target)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            opened = True
        except OSError:
            opened = False
    return {"name": folder.name, "path": str(target), "opened": opened}


def read_project_program(project: str, name: str) -> dict:
    folder = project_dir(project)
    filename = clean_program_name(name)
    path = folder / filename
    if not path.is_file():
        raise ValueError("找不到程式")
    return {"project": folder.name, "name": filename, "source": path.read_text(encoding="utf-8")}


@app.get("/api/projects")
def api_list_projects():
    return list_project_records()


@app.post("/api/projects")
def api_create_project(body: ProjectIn):
    try:
        return create_project_record(body.name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/projects/{name}")
def api_project_programs(name: str):
    try:
        return project_programs(name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/projects/{name}/folder")
def api_reveal_project_folder(name: str):
    try:
        return reveal_project_folder(name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/projects/{name}/rename")
def api_rename_project(name: str, body: RenameProjectIn):
    try:
        return rename_project_record(name, body.name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/projects/{name}/programs")
def api_write_project_program(name: str, body: SaveIn):
    try:
        return write_project_program(name, body.name, body.source)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/projects/{name}/programs/{filename}")
def api_read_project_program(name: str, filename: str):
    try:
        return read_project_program(name, filename)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/import/hdevelop")
async def import_hdevelop(file: UploadFile = File(...)):
    from app.engine.hdevelop_file import convert_hdevelop

    raw = await file.read()
    source, skipped = convert_hdevelop(raw)
    stem = Path(file.filename or "program").stem.strip() or "program"
    stem = re.sub(r'[\\/:*?"<>|]', "_", stem)
    return {"name": f"{stem}.viz", "source": source, "skipped": skipped}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), project: str | None = None):
    raw = Path(file.filename or "image.png").name
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", raw)
    if not safe.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        safe += ".png"
    if project:
        try:
            dest_dir = project_dir(project)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not dest_dir.is_dir():
            raise HTTPException(400, "找不到專案")
        stored = safe
    else:
        dest_dir = ROOT / "workspace"
        stored = f"workspace/{safe}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / safe
    if dest.exists():
        dest = dest_dir / f"{dest.stem}_{uuid.uuid4().hex[:6]}{dest.suffix}"
        stored = dest.name if project else f"workspace/{dest.name}"
    dest.write_bytes(await file.read())
    if read_cv_image(dest) is None:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "這不是可讀的影像")
    return {"path": stored}


def _level_from_histogram(histogram: dict | None) -> float | None:
    if not histogram:
        return None
    if histogram.get("kind") == "local":
        return None
    if histogram.get("kind") == "range":
        return float(histogram.get("min", 0))
    if "threshold" in histogram and histogram["threshold"] is not None:
        return float(histogram["threshold"])
    return None


@app.get("/api/features")
def feature_catalog():
    return {"region": REGION_FEATURES, "xld": XLD_FEATURES, "gray": GRAY_FEATURES}


@app.post("/api/features")
def object_features(body: FeatureIn):
    current = SESSIONS.get(body.id)
    if current is None:
        raise HTTPException(404, "請先執行程式")
    with current.lock:
        if body.name == "__preview__":
            value = current.preview_region
            if value is None:
                raise HTTPException(404, "請先在二值化視窗點一個區域")
        elif body.name not in current.vars:
            raise HTTPException(404, "沒有這個物件")
        else:
            value = current.vars[body.name]
        index = int(body.index)
        gray = current.threshold_gray
        level = _level_from_histogram(current.histogram)
        if body.kind == "xld":
            if not isinstance(value, XLD):
                raise HTTPException(400, "這不是可查的輪廓")
            groups = value.groups if value.groups and len(value.groups) == len(value.contours) else None
            if groups is not None:
                chosen = [chain for chain, group in zip(value.contours, groups) if group == index]
                if not chosen:
                    raise HTTPException(400, "這不是可查的輪廓")
                chain = max(chosen, key=len)
            elif index < 0 or index >= len(value.contours):
                raise HTTPException(400, "這不是可查的輪廓")
            else:
                chain = value.contours[index]
            contour = refine_polyline(chain, gray, level) if body.subpixel else [[float(p[0]), float(p[1])] for p in chain]
            measures = xld_measures(contour)
            rounded = {name: round(float(value), 4) for name, value in measures.items()}
            return {
                "kind": "xld",
                "name": body.name,
                "index": index,
                "region": {name: rounded[name] for name in body.region if name in rounded},
                "xld": {name: rounded[name] for name in body.xld if name in rounded},
                "gray": {},
                "grayReady": False,
                "contour": [contour],
            }
        if not isinstance(value, Region) or index < 0 or index >= value.count():
            raise HTTPException(400, "這不是可查的連通物件")
        mask = value.objects[index]
        source_gray, _domain = _session_gray(current)
        if source_gray is not None and source_gray.shape != mask.shape:
            source_gray = None
        if gray is not None and gray.shape != mask.shape:
            gray = None
        edge_gray = gray if gray is not None else source_gray
        links = value.links[index] if value.links and index < len(value.links) else None
        contour = subpixel_boundary(mask, edge_gray if body.subpixel else None, level if body.subpixel else None, links)
        region_values = region_measures(mask)
        gray_values = gray_measures(mask, source_gray)
        xld_source = contour[0] if contour else []
        xld_values = xld_measures(xld_source) if xld_source else {name: 0.0 for name in XLD_FEATURES}
        return {
            "kind": "region",
            "name": body.name,
            "index": index,
            "region": {name: round(float(region_values[name]), 4) for name in body.region if name in region_values},
            "xld": {name: round(float(xld_values[name]), 4) for name in body.xld if name in xld_values},
            "gray": {name: round(float(gray_values[name]), 4) for name in body.gray if name in gray_values},
            "grayReady": source_gray is not None,
            "contour": contour,
        }


@app.post("/api/gauge-preview")
def gauge_preview(body: GaugePreviewIn):
    current = SESSIONS.get(body.id)
    if current is None:
        raise HTTPException(404, "請先執行程式，圖上要有一張影像")
    with current.lock:
        image = current.display.base
        if image is None:
            return {"hits": [], "span": None}
        gray = image.gray().copy()
        domain = None if image.domain is None else image.domain.copy()
    try:
        hits = gauge_measure_point(
            gray, domain, body.row1, body.col1, body.row2, body.col2,
            body.sigma, body.threshold, body.transition, body.select,
        )
    except ValueError:
        return {"hits": [], "span": None}
    payload = [
        {"row": float(hit.row), "column": float(hit.column), "distance": float(hit.distance)}
        for hit in hits
    ]
    span = None
    if len(hits) >= 2:
        span = float(gauge_distance_pp(hits[0].row, hits[0].column, hits[-1].row, hits[-1].column))
    return {"hits": payload, "span": span}


@app.post("/api/threshold-preview")
def threshold_preview(body: PreviewIn):
    current = SESSIONS.get(body.id)
    if current is None or current.threshold_gray is None:
        raise HTTPException(404, "請先執行一次二值化")
    with current.lock:
        gray = current.threshold_gray
        mask = preview_mask(gray, current.threshold_domain, body.kind, body.min, body.max, body.threshold, body.lightDark)
        try:
            color = normalize_color(body.color)
        except ValueError:
            color = "#ff3b30"
        objects = [mask] if int(cv2.countNonZero(mask)) else []
        count = 1 if objects else 0
        colors = None
        if body.connect and objects:
            pieces = opencv_components(mask, 8) if body.kind == "opencv" else halcon_components(mask)
            pieces = merge_within_distance(pieces, max(0.0, float(body.distance)))
            min_area = max(0, int(body.minArea))
            objects = [item for item in pieces if int(cv2.countNonZero(item)) >= min_area]
            count = len(objects)
            colors = [MATCH_COLORS[index % 16] for index in range(count)]
        region = Region(int(gray.shape[1]), int(gray.shape[0]), objects, multi_color=bool(colors))
        current.preview_region = region
        overlay = fill_overlay(region, color, alpha=160, colors=colors)
        rings = []
        for item in objects:
            rings.append([chain for chain in contours_of(item) if len(chain) >= 3])
        payload = {
            "image": png_data_url(overlay),
            "area": int(cv2.countNonZero(mask)),
            "count": count,
            "width": int(gray.shape[1]),
            "height": int(gray.shape[0]),
            "objects": rings,
        }
        if body.image:
            ok, buf = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                raise HTTPException(500, "影像編碼失敗")
            payload["gray"] = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
        return payload


def _session_gray(current: Session):
    if current.threshold_gray is not None:
        return current.threshold_gray, current.threshold_domain
    if current.display.base is not None:
        return current.display.base.gray(), current.display.base.domain
    for value in current.vars.values():
        if isinstance(value, Image):
            return value.gray(), value.domain
    return None, None


@app.post("/api/subpix-preview")
def subpix_preview(body: SubpixIn):
    current = SESSIONS.get(body.id)
    if current is None:
        raise HTTPException(404, "請先執行程式")
    with current.lock:
        gray, domain = _session_gray(current)
        if gray is None:
            raise HTTPException(404, "還沒有影像")
        level = min(255.0, max(0.0, float(body.threshold)))
        mask, parts = subpix_parts(gray, level, domain)
        if len(parts) > 400:
            parts = parts[:400]
        contours = []
        groups = []
        for index, (_component, part_lines) in enumerate(parts):
            for line in part_lines:
                contours.append([[round(float(point[0]), 3), round(float(point[1]), 3)] for point in line])
                groups.append(index)
        counted = gray[domain > 0] if domain is not None else gray.ravel()
        bins = np.bincount(counted.astype(np.uint8).ravel(), minlength=256)[:256]
        try:
            color = normalize_color(body.color)
        except ValueError:
            color = "#ff3b30"
        objects = [component for component, _part_lines in parts]
        colors = [MATCH_COLORS[index % 16] for index in range(len(objects))] if len(objects) > 1 else None
        region = Region(int(gray.shape[1]), int(gray.shape[0]), objects, multi_color=len(objects) > 1)
        current.preview_region = region
        overlay = fill_overlay(region, color, alpha=110, colors=colors)
        region_objects = []
        for component in objects:
            region_objects.append([chain for chain in contours_of(component) if len(chain) >= 3])
        payload = {
            "width": int(gray.shape[1]),
            "height": int(gray.shape[0]),
            "threshold": level,
            "contours": contours,
            "groups": groups,
            "count": len(objects),
            "region": png_data_url(overlay),
            "regionObjects": region_objects,
            "bins": [int(value) for value in bins],
            "total": int(bins.sum()),
            "area": int(cv2.countNonZero(mask)),
        }
        if body.image:
            ok, buf = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                raise HTTPException(500, "影像編碼失敗")
            payload["image"] = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
        return payload


@app.post("/api/region-color")
def set_region_color(body: RegionColorIn):
    current = SESSIONS.get(body.id)
    if current is None:
        raise HTTPException(404, "請先執行程式")
    with current.lock:
        try:
            current.region_color = normalize_color(body.color)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        state = current.public_state()
        return {"color": current.region_color, "display": state["display"]}


def _prepare_action(current: Session, body: ExecIn, resume: bool) -> None:
    if resume:
        current.run(resume=True)
        return
    if body.regionColor:
        try:
            current.region_color = normalize_color(body.regionColor)
        except ValueError:
            current.region_color = "#ff3b30"
    if body.action == "reset":
        current.reset()
    elif body.action == "step":
        current.step()
    elif body.action == "run":
        current.defer_wait = True
        current.run()
    elif body.action == "goto":
        if body.line is None:
            raise HTTPException(400, "沒有指定要移到的行")
        current.set_line(int(body.line))
    else:
        raise HTTPException(400, "未知動作")


def _finish_wait(current: Session) -> None:
    while current.paused == "wait" and not current.cancel:
        left = float(getattr(current, "wait_until", 0.0)) - time.monotonic()
        if left <= 0:
            current.run(resume=True)
            continue
        time.sleep(min(0.05, max(0.0, left)))
    if current.cancel and current.paused == "wait":
        current.paused = "stopped"
        current.log.append("已停止")


@app.post("/api/exec")
async def exec_program(body: ExecIn):
    try:
        current = session_for(body)
    except ParseError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc), "line": exc.line}) from exc
    if body.action not in ("reset", "step", "run", "goto"):
        raise HTTPException(400, "未知動作")
    if body.action == "goto" and body.line is None:
        raise HTTPException(400, "沒有指定要移到的行")
    if body.action == "run" and getattr(current, "running", False):
        raise HTTPException(409, "程式還在跑")

    if body.action != "run":
        def sync():
            with current.lock:
                _prepare_action(current, body, False)
            return _snapshot(current)

        return await asyncio.to_thread(sync)

    def background():
        try:
            _prepare_action(current, body, False)
            _finish_wait(current)
        finally:
            with current.lock:
                current.running = False

    with current.lock:
        current.running = True
    current.runner = threading.Thread(target=background, daemon=True, name="visdev-run")
    current.runner.start()
    return {"id": current.id, "running": True}


@app.post("/api/session/{sid}/breakpoints")
def set_breakpoints(sid: str, body: BreakpointsIn):
    current = _session_or_404(sid)
    with current.lock:
        current.breakpoints = {int(line) for line in body.lines}
        lines = sorted(current.breakpoints)
    return {"ok": True, "lines": lines}


@app.post("/api/stop/{sid}")
def stop(sid: str):
    current = SESSIONS.get(sid)
    if current is None:
        return {"ok": False}
    current.cancel = True
    return {"ok": True}


@app.post("/api/session/{sid}/calib-view")
def set_calib_view(sid: str, body: CalibViewIn):
    current = _session_or_404(sid)
    with current.lock:
        current.show_calib_view(body.index)
        state = current.public_state()
        state["running"] = bool(getattr(current, "running", False))
    return state


@app.get("/api/session/{sid}/object/{name}")
def object_view(sid: str, name: str):
    current = SESSIONS.get(sid)
    if current is None:
        raise HTTPException(404, "沒有這個執行階段，請先按重設")
    with current.lock:
        if name not in current.vars:
            raise HTTPException(404, "沒有這個變數")
        try:
            return current.view_object(name)
        except TypeError as exc:
            raise HTTPException(400, str(exc)) from exc


app.mount("/static", StaticFiles(directory=STATIC), name="static")
