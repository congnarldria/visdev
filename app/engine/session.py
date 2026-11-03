from __future__ import annotations

import math
import re
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from .evalexpr import ExprError, eval_expr, truthy
from .objects import Image, Region, StopSignal, XLD, coerce_iconic
from .operators import REGISTRY, operator_bare
from .parser import (
    Assign, Call, CatchStmt, Comment, Blank, ElseStmt, EndForStmt, EndIfStmt, EndProcedureStmt,
    EndTryStmt, EndWhileStmt, ForStmt, IfStmt, ParseError, ProcedureStmt, TryStmt, WhileStmt, parse_program,
)
from .render import apply_domain_dim, compose_display, copy_display_scene, jpeg_thumb, read_cv_image, region_thumb, xld_thumb
from .ncc_model import NccModel
from .shape_model import MATCH_COLORS, ShapeModel


class Display:
    def __init__(self):
        self.color = "#00e05a"
        self.draw = "margin"
        self.line_width = 2
        self.colored = 0
        self.base: Image | None = None
        self.base_provisional = False
        self.overlays: list = []
        self.calib_panels = None
        self.calib_note = None
        self.calib_back = False

    def clear(self):
        self.base = None
        self.base_provisional = False
        self.overlays = []
        self.calib_panels = None
        self.calib_note = None
        self.calib_back = False

    def drop_provisional(self):
        self.overlays = [item for item in self.overlays if not item.get("provisional")]


def pythonize(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [pythonize(item) for item in value]
    if isinstance(value, tuple):
        return [pythonize(item) for item in value]
    return value


def tidy_number(value: float):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)) and abs(float(value) - round(float(value))) < 1e-8:
        return int(round(float(value)))
    return value


def fmt_num(value) -> str:
    value = pythonize(value)
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if math.isfinite(value) and abs(value - round(value)) < 1e-6:
            return str(int(round(value)))
        text = f"{value:.5f}".rstrip("0").rstrip(".")
        return text or "0"
    if isinstance(value, str):
        return "'" + value.replace("'", "\\'") + "'"
    return str(value)


def fmt_value(value) -> str:
    if isinstance(value, Region):
        return f"{value.count()} 個區域"
    if isinstance(value, XLD):
        return f"{len(value.contours)} 條輪廓"
    if isinstance(value, Image):
        return f"影像 {value.width}×{value.height}"
    value = pythonize(value)
    if isinstance(value, list):
        shown = [fmt_num(item) for item in value[:12]]
        body = ", ".join(shown)
        if len(value) > 12:
            return f"[{body}, …]  ({len(value)})"
        return f"[{body}]"
    return fmt_num(value)


def full_value(value) -> str:
    value = pythonize(value)
    if isinstance(value, list):
        body = ", ".join(fmt_num(item) for item in value[:400])
        if len(value) > 400:
            body += ", …"
        return f"[{body}]"
    return fmt_num(value)


_OS_HANDLES = {"FileHandle", "SerialPort", "SocketLink", "AcqHandle"}


def _release_external(value) -> None:
    if isinstance(value, list):
        for item in value:
            _release_external(item)
        return
    if type(value).__name__ not in _OS_HANDLES:
        return
    release = getattr(value, "release", None)
    if not callable(release):
        return
    try:
        release()
    except Exception:
        return


def _var_role(name: str, kind: str, iconic: bool) -> str:
    low = name.lower()
    if iconic and low.startswith("ho_"):
        return "HObject"
    if (not iconic) and low.startswith("hv_"):
        return "HTuple"
    return kind


_NG_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _ng_filename(name: str) -> str:
    filename = str(name).strip().strip("'\"")
    if not filename or filename in (".", "..") or "/" in filename or "\\" in filename or Path(filename).name != filename:
        raise RuntimeError("不合格圖的檔名不正確")
    if Path(filename).suffix.lower() not in _NG_SUFFIXES:
        raise RuntimeError("不合格圖必須是影像檔")
    return filename


def _csv_cells(row: str) -> list[str]:
    return [cell.strip() for cell in str(row).split(",")]


def _csv_column(header: str, names: tuple[str, ...]) -> int | None:
    cells = [cell.strip().lower() for cell in _csv_cells(header)]
    for name in names:
        if name in cells:
            return cells.index(name)
    return None


def _csv_is_ng(row: str, judge_at: int | None) -> bool:
    if judge_at is None:
        return False
    cells = _csv_cells(row)
    if judge_at >= len(cells):
        return False
    token = cells[judge_at].strip().strip("'\"")
    return token.upper() == "NG" or token == "不合格"


def _csv_open_name(row: str, file_at: int | None, judge_at: int | None) -> str:
    if file_at is None or not _csv_is_ng(row, judge_at):
        return ""
    cells = _csv_cells(row)
    if file_at >= len(cells):
        return ""
    try:
        return _ng_filename(cells[file_at])
    except RuntimeError:
        return ""


class Session:
    def __init__(self, source: str, root: Path):
        self.id = uuid.uuid4().hex[:12]
        self.source = source
        self.root = root
        self.file_dir = None
        program = parse_program(source)
        self.stmts = program.stmts
        self.if_false = program.if_false
        self.else_jump = program.else_jump
        self.for_end = program.for_end
        self.end_for = program.end_for
        self.while_end = program.while_end
        self.end_while = program.end_while
        self.proc_end = program.proc_end
        self.procs = program.procs
        self.try_catch = program.try_catch
        self.catch_end = program.catch_end
        self.lock = threading.Lock()
        self.breakpoints: set[int] = set()
        self.trace = False
        self.ref_shape: tuple[int, int] | None = None
        self.region_color = "#ff3b30"
        self.threshold_mask = None
        self.threshold_region = None
        self.image_stamp = 0
        self.display_stamp = 0
        self.running = False
        self.reset()

    def _release_handles(self) -> None:
        for value in list(getattr(self, "vars", {}).values()):
            _release_external(value)

    def _release_error_frames(self) -> None:
        seen = {id(value) for value in getattr(self, "vars", {}).values()}
        for frame in getattr(self, "error_frames", ()) or ():
            for value in frame.get("vars", {}).values():
                if id(value) in seen:
                    continue
                seen.add(id(value))
                _release_external(value)

    def _remember_failed_proc(self, proc, local: dict) -> None:
        if not self.error:
            return
        frames = getattr(self, "error_frames", None)
        if frames is None:
            self.error_frames = []
            frames = self.error_frames
        frames.append({"procedure": getattr(proc, "name", None) or str(proc), "vars": dict(local)})

    def _set_error(self, exc, line, *, log: bool) -> None:
        number = int(line) if isinstance(line, int) else None
        text = str(exc)
        if number and f"第 {number} 行" not in text:
            text = f"第 {number} 行：{text}"
        self.error = text
        self.error_line = number
        if log:
            self._append_log(text if text.startswith("錯誤") else f"錯誤 {text}")
        self._release_handles()

    def reset(self):
        self._release_error_frames()
        self._release_handles()
        self.pc = 0
        self.vars: dict = {}
        self.done = False
        self.error = None
        self.error_line = None
        self.error_frames = []
        self.calib_board = None
        self.calib_owners = None
        self.paused = None
        self.log: list[str] = []
        self.display = Display()
        self._ng_hold = None
        self._ng_showing = False
        self._ng_file = ""
        self.loops: dict = {}
        self.try_stack: list = []
        self.proc_depth = 0
        self.proc_stack = []
        self.proc_names = []
        self.step_into = False
        self.cancel = False
        self.defer_wait = False
        self.wait_until = 0.0
        self._wait_frame = None
        self.break_skip = None
        self.arrow_line = None
        self.arrow_note = None
        self.line_times: dict[int, dict] = {}
        self.step_count = 0
        self.ref_shape = None
        self.histogram = None
        self.threshold_gray = None
        self.threshold_domain = None
        self.threshold_mask = None
        self.threshold_region = None
        self.preview_region = None
        self._display_cache = None
        self._thumb_cache = {}
        self.prepare()

    def prepare(self):
        while self.pc < len(self.stmts) and isinstance(self.stmts[self.pc], (Blank, Comment)):
            self.pc += 1
        if self.pc >= len(self.stmts):
            self.done = True

    def current_line(self):
        if self.error and self.error_line:
            return self.error_line
        frame = getattr(self, "_wait_frame", None)
        if frame is not None and self.paused in ("breakpoint", "wait"):
            inner = frame.get("pc")
            if isinstance(inner, int) and inner < len(self.stmts):
                return self.stmts[inner].line
        if not self.done and self.pc < len(self.stmts):
            return self.stmts[self.pc].line
        return self.arrow_line

    def _paused_procedure(self):
        if self.proc_stack:
            return self.proc_stack[-1]["proc"].name
        frame = getattr(self, "_wait_frame", None)
        if frame is not None and self.paused in ("breakpoint", "wait"):
            proc = frame.get("proc")
            if proc is not None:
                return proc.name
        frames = getattr(self, "error_frames", None) or []
        if self.error and frames:
            return frames[0].get("procedure")
        return None

    def _scoped_vars(self) -> tuple[dict, set]:
        layers = []
        if self.proc_stack:
            layers = [frame["caller_vars"] for frame in reversed(self.proc_stack)]
        elif self.paused in ("breakpoint", "wait") and getattr(self, "_wait_frame", None) is not None:
            frame = self._wait_frame
            while frame is not None:
                layers.append(frame.get("caller_vars") or {})
                frame = frame.get("caller")
        if not layers:
            frames = getattr(self, "error_frames", None) or []
            if not (self.error and frames):
                return self.vars, set()
            local_names: set = set()
            merged = {}
            for frame in frames:
                for name, value in frame.get("vars", {}).items():
                    local_names.add(name)
                    if name not in merged:
                        merged[name] = value
            for name, value in self.vars.items():
                if name not in merged:
                    merged[name] = value
            return merged, local_names
        local_names = set(self.vars)
        merged = dict(self.vars)
        for layer in layers:
            for name, value in layer.items():
                if name not in merged:
                    merged[name] = value
        return merged, local_names

    def set_line(self, line: int):
        """Move the program arrow. Variables stay; the next step starts at this line.

        A drop inside a procedure body stays there only when that call is already
        active. Otherwise the arrow stops on the procedure header, so the body
        does not run with the caller's variables.
        """
        target = None
        for index, stmt in enumerate(self.stmts):
            if isinstance(stmt, (Blank, Comment)):
                continue
            if stmt.line >= int(line):
                target = index
                break
        self._release_error_frames()
        self.error_frames = []
        self.error = None
        self.error_line = None
        self.paused = None
        self.break_skip = None
        self.arrow_note = None
        if target is None:
            self._release_proc_jump(None)
            self.pc = len(self.stmts)
            self.done = True
            self.arrow_line = None
            return
        adjusted = self._procedure_arrow_index(target)
        self._release_proc_jump(adjusted)
        if adjusted != target:
            self.arrow_note = "程序還沒呼叫，箭頭停在程序標題"
            self._append_log(self.arrow_note)
        self.pc = adjusted
        self.done = False

    def _enclosing_proc(self, index: int):
        found = None
        span = None
        for proc_index, end in self.proc_end.items():
            if proc_index < index < end:
                width = end - proc_index
                if span is None or width < span:
                    found = proc_index
                    span = width
        if found is None:
            return None
        return found, self.proc_end[found]

    def _frame_proc_index(self, frame) -> int | None:
        proc = frame.get("proc")
        end = frame.get("end")
        for proc_index, proc_end in self.proc_end.items():
            if proc_end == end and self.stmts[proc_index] is proc:
                return proc_index
        return None

    def _proc_is_active(self, proc_index: int) -> bool:
        for frame in self.proc_stack:
            if self._frame_proc_index(frame) == proc_index:
                return True
        frame = getattr(self, "_wait_frame", None)
        while frame is not None:
            if self._frame_proc_index(frame) == proc_index:
                return True
            frame = frame.get("caller")
        return False

    def _procedure_arrow_index(self, index: int) -> int:
        found = self._enclosing_proc(index)
        if found is None:
            return index
        proc_index, _end = found
        if self._proc_is_active(proc_index):
            return index
        return proc_index

    def _release_proc_jump(self, index: int | None) -> None:
        while self.proc_stack:
            proc_index = self._frame_proc_index(self.proc_stack[-1])
            end = self.proc_stack[-1]["end"]
            if index is not None and proc_index is not None and proc_index < index < end:
                break
            frame = self.proc_stack.pop()
            self.vars = frame["caller_vars"]
            self.proc_depth = max(0, self.proc_depth - 1)
        if self.proc_stack:
            self._wait_frame = None
            return
        if self._wait_frame is None:
            return
        if index is not None and self._wait_contains(index):
            self._adopt_wait_chain(index)
            return
        self._drop_wait_frames()

    def _wait_contains(self, index: int) -> bool:
        frame = self._wait_frame
        while frame is not None:
            proc_index = self._frame_proc_index(frame)
            end = frame.get("end")
            if proc_index is not None and end is not None and proc_index < index < end:
                return True
            frame = frame.get("caller")
        return False

    def _drop_wait_frames(self) -> None:
        frame = self._wait_frame
        outer = None
        while frame is not None:
            outer = frame.get("caller_vars")
            frame = frame.get("caller")
        if outer is not None:
            self.vars = outer
        self._wait_frame = None
        self.proc_depth = len(self.proc_stack)

    def _adopt_wait_chain(self, index: int) -> None:
        chain = []
        frame = self._wait_frame
        while frame is not None:
            chain.append(frame)
            frame = frame.get("caller")
        chosen = None
        for frame in chain:
            proc_index = self._frame_proc_index(frame)
            end = frame.get("end")
            if proc_index is not None and end is not None and proc_index < index < end:
                chosen = frame
                break
        if chosen is None:
            self._drop_wait_frames()
            return
        selected = []
        frame = chosen
        while frame is not None:
            selected.append(frame)
            frame = frame.get("caller")
        self.proc_stack = [self._stack_frame(item) for item in reversed(selected)]
        self.vars = chosen["local"]
        self.proc_depth = len(self.proc_stack)
        self._wait_frame = None

    def _stack_frame(self, frame) -> dict:
        return {
            "caller_vars": frame["caller_vars"],
            "end": frame["end"],
            "saved_pc": frame["saved_pc"],
            "stmt": frame["stmt"],
            "proc": frame["proc"],
        }

    def resolve_read_path(self, filename: str) -> Path:
        path = Path(str(filename).strip().replace("\\", "/")).expanduser()
        candidates = []
        if path.is_absolute():
            candidates.append(path)
        else:
            project = getattr(self, "project_name", None)
            if project:
                folder = (self.root / "workspace" / "projects" / project).resolve()
                candidate = (folder / path).resolve()
                try:
                    candidate.relative_to(folder)
                except ValueError:
                    candidate = None
                if candidate is not None:
                    candidates.append(candidate)
            candidates.extend([self.root / path, self.root / "samples" / path, self.root / "workspace" / path])
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"找不到影像 {filename}")

    def resolve_write_path(self, filename: str) -> Path:
        path = Path(filename).expanduser()
        if not path.is_absolute():
            path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def canvas_size(self) -> tuple[int, int]:
        if self.ref_shape:
            return self.ref_shape
        if self.display.base is not None:
            return self.display.base.height, self.display.base.width
        return 480, 640

    def _drop_ng_view(self) -> None:
        if getattr(self, "_ng_showing", False) and getattr(self, "_ng_hold", None) is not None:
            self.display = self._ng_hold
            self._touch_display()
        self._ng_hold = None
        self._ng_showing = False
        self._ng_file = ""

    def show_last_ng(self) -> str:
        """Show workspace/line_ng/<LastNg>, or restore the program picture."""
        if getattr(self, "_ng_showing", False):
            self._drop_ng_view()
            return ""
        raw = self.vars.get("LastNg", "")
        name = str(raw).strip().strip("'\"")
        if not name:
            raise RuntimeError("這一輪沒有不合格圖")
        return self.show_ng_file(name)

    def show_ng_file(self, name: str) -> str:
        """Show one image already saved in workspace/line_ng. Counts stay."""
        filename = _ng_filename(name)
        folder = (self.root / "workspace" / "line_ng").resolve()
        path = (folder / filename).resolve()
        try:
            path.relative_to(folder)
        except ValueError as exc:
            raise RuntimeError("不合格圖不能離開不合格圖資料夾") from exc
        if not path.is_file():
            raise RuntimeError(f"找不到不合格圖 {filename}。它可能已被清掉，請再跑一次產線。")
        data = read_cv_image(path)
        if data is None:
            raise RuntimeError(f"無法讀取不合格圖 {filename}")
        if not getattr(self, "_ng_showing", False) or getattr(self, "_ng_hold", None) is None:
            self._ng_hold = self.display
        fresh = Display()
        fresh.line_width = self._ng_hold.line_width
        fresh.color = "#ff2a2a"
        self.display = fresh
        self.display_obj(Image(np.array(data)), name="LastNg")
        self._ng_showing = True
        self._ng_file = filename
        return filename

    def line_csv_preview(self, limit: int = 20, only_ng: bool = False) -> dict:
        """Read the line CSV named by CsvFile. Only files inside workspace.

        only_ng keeps rows whose judge cell is NG. The file on disk is not changed.
        """
        raw = self.vars.get("CsvFile", "")
        text = str(raw).strip().strip("'\"")
        if not text:
            raise RuntimeError("這一輪沒有明細")
        normalized = text.replace("\\", "/")
        if normalized.startswith("/") or ".." in Path(normalized).parts:
            raise RuntimeError("明細路徑不正確")
        path = (self.root / normalized).resolve()
        workspace = (self.root / "workspace").resolve()
        try:
            path.relative_to(workspace)
        except ValueError as exc:
            raise RuntimeError("明細只能讀 workspace 裡的 CSV") from exc
        if path.suffix.lower() != ".csv" or not path.is_file():
            raise RuntimeError(f"找不到明細 {normalized}")
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise RuntimeError("明細是空的")
        header = lines[0]
        body = lines[1:]
        judge_at = _csv_column(header, ("judge", "判定"))
        file_at = _csv_column(header, ("file", "檔名"))
        ng_rows = [row for row in body if _csv_is_ng(row, judge_at)]
        chosen = ng_rows if only_ng else body
        shown = chosen[-limit:] if len(chosen) > limit else list(chosen)
        return {
            "file": normalized,
            "total": len(body),
            "ngTotal": len(ng_rows),
            "header": header,
            "rows": shown,
            "opens": [_csv_open_name(row, file_at, judge_at) for row in shown],
            "truncated": len(chosen) > limit,
            "onlyNg": bool(only_ng),
            "noJudge": judge_at is None,
        }

    def show_calib_view(self, index: int | None) -> None:
        from .chapters.calib_ops import show_saved_calibration

        show_saved_calibration(self, index)

    def display_obj(self, obj, provisional: bool = False, name: str | None = None):
        source = name if name is not None else getattr(self, "pending_display_name", None)
        obj = coerce_iconic(obj)
        if isinstance(obj, list) and obj and all(isinstance(item, Image) for item in obj):
            obj = obj[-1]
        if isinstance(obj, Image):
            self.display.base = obj
            self.display.base_provisional = provisional
            self.display.overlays = []
            self.display.calib_panels = None
            self.display.calib_note = None
            self.display.calib_back = False
            self.ref_shape = (obj.height, obj.width)
            self._touch_display()
            return
        self.display.drop_provisional()
        if isinstance(obj, Region):
            self.ref_shape = self.ref_shape or (obj.height, obj.width)
            colors = self.region_palette(obj)
            color = self.region_color if not obj.multi_color else self.display.color
            self.display.overlays.append(
                {
                    "kind": "region",
                    "region": obj,
                    "color": color,
                    "colors": colors,
                    "draw": self.display.draw,
                    "width": self.display.line_width,
                    "provisional": provisional,
                    "source": source,
                }
            )
            self._touch_display()
            return
        if isinstance(obj, XLD):
            self.ref_shape = self.ref_shape or (obj.height, obj.width)
            groups = getattr(obj, "groups", None) or []
            grouped = len(groups) == len(obj.contours) and len(obj.contours) > 0
            color_count = (max(groups) + 1) if grouped else len(obj.contours)
            colors = None
            if getattr(obj, "multi_color", False) and color_count > 1:
                colors = [MATCH_COLORS[index % 16] for index in range(color_count)]
            elif self.display.colored > 1 and color_count > 1:
                colors = [MATCH_COLORS[index % self.display.colored] for index in range(color_count)]
            self.display.overlays.append(
                {
                    "kind": "xld",
                    "xld": obj,
                    "color": self.display.color,
                    "colors": colors,
                    "width": self.display.line_width,
                    "provisional": provisional,
                    "source": source,
                }
            )
            self._touch_display()
            return
        raise TypeError("dev_display 只能顯示影像、區域或輪廓")

    def _touch_display(self) -> None:
        self.display_stamp = int(getattr(self, "display_stamp", 0)) + 1

    def region_palette(self, region: Region):
        count = region.count()
        if count <= 1:
            return None
        if region.multi_color:
            limit = 16
        elif self.display.colored > 1:
            limit = self.display.colored
        else:
            return None
        return [MATCH_COLORS[index % limit] for index in range(count)]

    def _append_log(self, message: str) -> None:
        log = self.log
        log.append(message)
        extra = len(log) - 2000
        if extra > 0:
            del log[: extra + 500]

    def trace_line(self, stmt, message: str):
        if not self.trace:
            return
        label = self._trace_proc_label()
        if label:
            message = f"{label}: {message}"
        self._append_log(f"{stmt.line:>4}  {message}")

    def _trace_proc_label(self) -> str | None:
        names = [frame["proc"].name for frame in self.proc_stack]
        names.extend(self.proc_names)
        if not names:
            return None
        name = names[-1]
        depth = sum(1 for item in names if item == name)
        return name if depth == 1 else f"{name}×{depth}"

    def _on_breakpoint(self) -> bool:
        if self.pc >= len(self.stmts) or not self.breakpoints:
            return False
        stmt = self.stmts[self.pc]
        covered = set(stmt.lines or ())
        covered.add(stmt.line)
        if covered & self.breakpoints:
            return True
        start = stmt.line
        prev_end = 0
        for earlier in range(self.pc - 1, -1, -1):
            other = self.stmts[earlier]
            if isinstance(other, (Blank, Comment)):
                continue
            prev_end = max(other.lines) if other.lines else other.line
            break
        return any(prev_end < line < start for line in self.breakpoints)

    def _note_time(self, stmt, seconds: float):
        line = max(stmt.lines) if stmt.lines else stmt.line
        slot = self.line_times.setdefault(line, {"total": 0.0, "count": 0})
        slot["total"] += seconds
        slot["count"] += 1

    def step(self):
        if self.error:
            return
        self._drop_ng_view()
        self.arrow_note = None
        self.paused = None
        self.break_skip = None
        self.prepare()
        self._close_finished_procs()
        if self.done:
            return
        self.step_into = True
        try:
            self._execute_current()
            if not self.error:
                self._close_finished_procs()
        except Exception as exc:
            if not self.error:
                self.error = str(exc)
                self.error_line = self.current_line()
        finally:
            self.step_into = False
        if self.error:
            return
        self.step_count += 1
        self.prepare()
        try:
            self._close_finished_procs()
        except Exception as exc:
            self.error = str(exc)
            self.error_line = self.current_line()

    def run(self, max_steps: int = 100000, *, resume: bool = False):
        with self.lock:
            parked = getattr(self, "_wait_frame", None)
            resume_parked = bool(resume or self.paused == "breakpoint")
            if not resume and (self.done or self.error):
                self.reset()
                parked = None
                resume_parked = False
            else:
                self._drop_ng_view()
            self.arrow_note = None
            self.paused = None
            if not resume:
                self.cancel = False
            if parked and resume_parked:
                self._resume_proc()
                if self.paused in ("wait", "breakpoint") or self.error:
                    return
        taken = 0
        while taken < max_steps:
            with self.lock:
                self.prepare()
                if self.done:
                    self._append_log(f"執行完成（{self.step_count} 步）")
                    return
                if self.cancel:
                    self.paused = "stopped"
                    self._append_log("已停止")
                    self._release_handles()
                    return
                if self._on_breakpoint() and self.pc != self.break_skip:
                    self.break_skip = self.pc
                    self.paused = "breakpoint"
                    self._append_log(f"中斷點 第 {self.stmts[self.pc].line} 行")
                    return
                self.break_skip = None
                self._execute_current()
                self._close_finished_procs()
                if self.error:
                    return
                self.step_count += 1
                taken += 1
                stop_now = self.paused in ("stop", "wait", "stopped", "breakpoint")
            if stop_now:
                return
        with self.lock:
            self._set_error(RuntimeError("超過最大步數，可能有無窮迴圈"), self.current_line(), log=True)

    def _execute_current(self):
        index = self.pc
        stmt = self.stmts[index]
        self.arrow_line = stmt.line
        started = time.perf_counter()
        try:
            self._execute(stmt, index)
        except StopSignal:
            self.paused = "stopped" if self.cancel else "stop"
            self._append_log("已停止" if self.cancel else f"Vizion.stop() 於第 {stmt.line} 行")
        except (ExprError, ParseError, Exception) as exc:
            frame = self.try_stack[-1] if self.try_stack and self.try_stack[-1][2] == self.proc_depth else None
            if frame is not None:
                self.try_stack.pop()
                catch_index, var, _depth = frame
                self.vars[var] = str(exc)
                self.pc = catch_index + 1
                self._append_log(f"catch {var} 第 {stmt.line} 行：{exc}")
            elif self.proc_depth:
                line = getattr(exc, "line", None) or stmt.line
                if getattr(exc, "already_reported", False):
                    self.error = str(exc)
                    self.error_line = line
                else:
                    self._set_error(exc, line, log=True)
            else:
                line = getattr(exc, "line", None) or stmt.line
                if getattr(exc, "already_reported", False):
                    self.error = str(exc)
                    self.error_line = line
                else:
                    self._set_error(exc, line, log=True)
        finally:
            self._note_time(stmt, time.perf_counter() - started)

    def _execute(self, stmt, index: int):
        if isinstance(stmt, Assign):
            self.vars[stmt.name] = coerce_iconic(pythonize(eval_expr(stmt.expr, self.vars)))
            self.pc = index + 1
            self.trace_line(stmt, f"{stmt.name} := {fmt_value(self.vars[stmt.name])}")
            return
        if isinstance(stmt, Call):
            if operator_bare(stmt.name) == "stop":
                self.pc = index + 1
                raise StopSignal()
            if stmt.name in self.procs:
                if self.step_into:
                    self._enter_proc(stmt, self.procs[stmt.name])
                    return
                self._call_proc(stmt, self.procs[stmt.name])
                if self.paused in ("wait", "breakpoint"):
                    if self.proc_depth == 0:
                        frame = getattr(self, "_wait_frame", None)
                        if frame is not None:
                            self.vars = frame["local"]
                    return
            else:
                self._call(stmt)
            self.pc = index + 1
            return
        if isinstance(stmt, ProcedureStmt):
            self.pc = self.proc_end[index] + 1
            return
        if isinstance(stmt, EndProcedureStmt):
            self.pc = index + 1
            return
        if isinstance(stmt, TryStmt):
            catch_index = self.try_catch[index]
            var = self.stmts[catch_index].var
            self.try_stack.append((catch_index, var, self.proc_depth))
            self.pc = index + 1
            return
        if isinstance(stmt, CatchStmt):
            if self.try_stack and self.try_stack[-1][0] == index and self.try_stack[-1][2] == self.proc_depth:
                self.try_stack.pop()
            self.pc = self.catch_end[index]
            return
        if isinstance(stmt, EndTryStmt):
            self.pc = index + 1
            return
        if isinstance(stmt, IfStmt):
            cond = eval_expr(stmt.expr, self.vars)
            self.trace_line(stmt, f"if ({stmt.expr}) → {1 if truthy(cond) else 0}")
            self.pc = index + 1 if truthy(cond) else self.if_false[index]
            return
        if isinstance(stmt, ElseStmt):
            self.pc = self.else_jump[index]
            return
        if isinstance(stmt, EndIfStmt):
            self.pc = index + 1
            return
        if isinstance(stmt, ForStmt):
            self._enter_for(stmt, index)
            return
        if isinstance(stmt, EndForStmt):
            self._leave_for(index)
            return
        if isinstance(stmt, WhileStmt):
            cond = eval_expr(stmt.expr, self.vars)
            self.pc = index + 1 if truthy(cond) else self.while_end[index] + 1
            return
        if isinstance(stmt, EndWhileStmt):
            self.pc = self.end_while[index]
            return
        self.pc = index + 1

    def _enter_for(self, stmt: ForStmt, index: int):
        start = float(eval_expr(stmt.start, self.vars))
        end = float(eval_expr(stmt.end, self.vars))
        step = float(eval_expr(stmt.step, self.vars))
        if step == 0:
            raise RuntimeError("for 的 step 不能為 0")
        self.vars[stmt.var] = tidy_number(start)
        self.loops[index] = {"end": end, "step": step, "var": stmt.var}
        self.trace_line(stmt, f"for {stmt.var} := {fmt_num(start)} to {fmt_num(end)} by {fmt_num(step)}")
        if _in_range(start, end, step):
            self.pc = index + 1
        else:
            self.loops.pop(index, None)
            self.pc = self.for_end[index] + 1

    def _leave_for(self, index: int):
        for_index = self.end_for[index]
        frame = self.loops.get(for_index)
        if frame is None:
            self.pc = index + 1
            return
        current = float(self.vars[frame["var"]]) + frame["step"]
        self.vars[frame["var"]] = tidy_number(current)
        if _in_range(current, frame["end"], frame["step"]):
            self.pc = for_index + 1
        else:
            self.loops.pop(for_index, None)
            self.pc = index + 1

    def _enter_proc(self, stmt: Call, proc_index: int) -> None:
        if self.proc_depth >= 32:
            raise RuntimeError("程序遞迴超過 32 層")
        proc = self.stmts[proc_index]
        outer = self.vars
        local: dict = {}
        for name, src in zip(proc.iconic_in, stmt.iconic_in):
            if src not in outer:
                raise RuntimeError(f"變數 {src} 還沒有值")
            local[name] = coerce_iconic(outer[src])
        for name, expr in zip(proc.control_in, stmt.control_in):
            local[name] = pythonize(eval_expr(expr, outer))
        self.proc_stack.append({
            "caller_vars": outer,
            "end": self.proc_end[proc_index],
            "saved_pc": self.pc,
            "stmt": stmt,
            "proc": proc,
        })
        self.proc_depth += 1
        self.vars = local
        self.pc = proc_index + 1

    def _close_finished_procs(self) -> None:
        while self.proc_stack and self.pc >= self.proc_stack[-1]["end"]:
            frame = self.proc_stack.pop()
            local = self.vars
            self.vars = frame["caller_vars"]
            self.proc_depth = max(0, self.proc_depth - 1)
            self._export_proc(frame["stmt"], frame["proc"], local)
            self.pc = frame["saved_pc"] + 1

    def _call_proc(self, stmt: Call, proc_index: int):
        if self.proc_depth >= 32:
            raise RuntimeError("程序遞迴超過 32 層")
        proc = self.stmts[proc_index]
        outer = self.vars
        local = {}
        for name, src in zip(proc.iconic_in, stmt.iconic_in):
            if src not in outer:
                raise RuntimeError(f"變數 {src} 還沒有值")
            local[name] = coerce_iconic(outer[src])
        for name, expr in zip(proc.control_in, stmt.control_in):
            local[name] = pythonize(eval_expr(expr, outer))
        saved_pc = self.pc
        self.proc_depth += 1
        self.proc_names.append(proc.name)
        self.vars = local
        self.pc = proc_index + 1
        end = self.proc_end[proc_index]
        guard = None
        try:
            steps = 0
            while self.pc < end and not self.error and self.paused not in ("stop", "breakpoint", "stopped", "wait"):
                if self._on_breakpoint() and self.pc != self.break_skip:
                    self.break_skip = self.pc
                    self.paused = "breakpoint"
                    break
                self.break_skip = None
                self._execute_current()
                steps += 1
                if steps > 100000:
                    guard = "程序超過 100000 步"
                    break
        finally:
            local = self.vars
            inner_pc = self.pc
            waiting = self.paused == "wait"
            breaking = self.paused == "breakpoint"
            self._remember_failed_proc(proc, local)
            self.vars = outer
            self.pc = saved_pc
            self.proc_depth -= 1
            self.proc_names.pop()
        if waiting or breaking:
            self._park_wait({
                "local": local,
                "caller_vars": outer,
                "pc": inner_pc,
                "end": end,
                "saved_pc": saved_pc,
                "stmt": stmt,
                "proc": proc,
            })
            return
        if guard:
            raise RuntimeError(guard)
        if self.error:
            message = self.error
            line = self.error_line
            self.error = None
            self.error_line = None
            err = RuntimeError(message)
            err.line = line
            err.already_reported = True
            raise err
        self._export_proc(stmt, proc, local)

    def _export_proc(self, stmt: Call, proc, local) -> None:
        for name, src in zip(stmt.iconic_out, proc.iconic_out):
            if src not in local:
                raise RuntimeError(f"程序 {proc.name} 沒有設定輸出 {src}")
            value = coerce_iconic(local[src])
            self.vars[name] = value
            if isinstance(value, Image):
                self.ref_shape = (value.height, value.width)
        for name, src in zip(stmt.control_out, proc.control_out):
            if src not in local:
                raise RuntimeError(f"程序 {proc.name} 沒有設定輸出 {src}")
            self.vars[name] = pythonize(local[src])
        returned = []
        for name, src in zip(stmt.iconic_out, proc.iconic_out):
            if src in local:
                returned.append(f"{name} := {fmt_value(local[src])}")
        for name, src in zip(stmt.control_out, proc.control_out):
            if src in local:
                returned.append(f"{name} := {fmt_value(self.vars[name])}")
        call = stmt.name if not returned else stmt.name + " → " + ", ".join(returned)
        self.trace_line(stmt, call)

    def _park_wait(self, frame) -> None:
        """Remember a procedure paused on wait. A caller stays linked behind the inner one."""
        current = self._wait_frame
        if current is None:
            self._wait_frame = frame
            return
        node = current
        while node.get("caller") is not None:
            node = node["caller"]
        node["caller"] = frame

    def _resume_proc(self) -> None:
        frame = self._wait_frame
        self._wait_frame = None
        self._run_wait_frame(frame, frame["pc"])

    def _run_wait_frame(self, frame, start_pc: int) -> None:
        caller_vars = frame["caller_vars"]
        self.vars = frame["local"]
        self.pc = start_pc
        self.proc_depth += 1
        self.proc_names.append(frame["proc"].name)
        try:
            steps = 0
            while self.pc < frame["end"] and not self.error and self.paused not in ("stop", "breakpoint", "stopped", "wait"):
                if self._on_breakpoint() and self.pc != self.break_skip:
                    self.break_skip = self.pc
                    self.paused = "breakpoint"
                    break
                self.break_skip = None
                self._execute_current()
                steps += 1
                if steps > 100000:
                    self._set_error(RuntimeError("程序超過 100000 步"), self.current_line(), log=True)
                    break
        finally:
            local = self.vars
            inner_pc = self.pc
            waiting = self.paused == "wait"
            breaking = self.paused == "breakpoint"
            self._remember_failed_proc(frame["proc"], local)
            self.vars = caller_vars
            self.pc = frame["saved_pc"]
            self.proc_depth -= 1
            self.proc_names.pop()
        if waiting or breaking:
            frame["local"] = local
            frame["pc"] = inner_pc
            self._park_wait(frame)
            self.vars = local
            return
        if self.error:
            return
        self._export_proc(frame["stmt"], frame["proc"], local)
        parent = frame.get("caller")
        if parent is not None:
            self._run_wait_frame(parent, frame["saved_pc"] + 1)
            return
        self.pc = frame["saved_pc"] + 1

    def _call(self, stmt: Call):
        bare = operator_bare(stmt.name)
        op = REGISTRY[bare]
        iconic = []
        for name in stmt.iconic_in:
            if name not in self.vars:
                raise RuntimeError(f"變數 {name} 還沒有值")
            iconic.append(coerce_iconic(self.vars[name]))
        controls = [pythonize(eval_expr(expr, self.vars)) for expr in stmt.control_in]
        self.pending_display_name = stmt.iconic_in[0] if bare == "dev_display" and stmt.iconic_in else None
        result = op.fn(*iconic, *controls, ctx=self)
        if len(result.iconic) != len(stmt.iconic_out) or len(result.control) != len(stmt.control_out):
            raise RuntimeError(f"{stmt.name} 的輸出數量和呼叫不一致")
        for name, value in zip(stmt.iconic_out, result.iconic):
            value = coerce_iconic(value)
            self.vars[name] = value
            if isinstance(value, Image):
                self.ref_shape = (value.height, value.width)
        for name, value in zip(stmt.control_out, result.control):
            self.vars[name] = pythonize(value)
        if stmt.iconic_out and not bare.startswith("dev_"):
            self.display_obj(self.vars[stmt.iconic_out[-1]], provisional=True, name=stmt.iconic_out[-1])
        if bare in ("threshold", "binary_threshold", "opencv_threshold", "threshold_sub_pix") and self.histogram is not None:
            self.histogram["line"] = stmt.line
            if bare == "threshold":
                numeric = len(stmt.control_in) >= 2 and all(_num_literal(item) for item in stmt.control_in[:2])
                self.histogram["literal"] = numeric and len(stmt.lines) == 1
            elif bare == "opencv_threshold":
                numeric = len(stmt.control_in) >= 1 and _num_literal(stmt.control_in[0])
                self.histogram["literal"] = numeric and len(stmt.lines) == 1
            elif bare == "threshold_sub_pix":
                numeric = len(stmt.control_in) >= 1 and _num_literal(stmt.control_in[0])
                self.histogram["literal"] = numeric and len(stmt.lines) == 1
        outputs = stmt.iconic_out + stmt.control_out
        message = stmt.name if not outputs else stmt.name + " → " + ", ".join(outputs)
        self.trace_line(stmt, message)

    def _window_overlays(self) -> list:
        overlays = list(self.display.overlays)
        mask = self.threshold_mask
        base = self.display.base
        if mask is not None and base is not None and mask.shape[:2] != (base.height, base.width):
            mask = None
        showing_parts = any(
            item.get("kind") == "region"
            and getattr(item.get("region"), "multi_color", False)
            and item["region"].count() > 1
            for item in overlays
        )
        parts = getattr(self, "threshold_region", None)
        subpix_parts = (
            self.histogram is not None
            and self.histogram.get("kind") == "subpix"
            and isinstance(parts, Region)
            and parts.count() > 0
        )
        if subpix_parts:
            colors = [MATCH_COLORS[index % 16] for index in range(parts.count())] if parts.count() > 1 else None
            overlays.insert(
                0,
                {
                    "kind": "region",
                    "region": parts,
                    "color": self.region_color,
                    "colors": colors,
                    "draw": "fill",
                    "width": 1,
                    "alpha": 120,
                    "source": "__preview__",
                },
            )
        elif mask is not None and not showing_parts and int(np.count_nonzero(mask)):
            binary = Region(int(mask.shape[1]), int(mask.shape[0]), [mask])
            overlays.insert(
                0,
                {
                    "kind": "region",
                    "region": binary,
                    "color": self.region_color,
                    "colors": None,
                    "draw": "fill",
                    "width": 1,
                    "alpha": 120,
                },
            )
        return overlays

    def _display_token(self, overlays: list) -> tuple:
        base = self.display.base
        if base is None:
            base_token = None
        else:
            domain = base.domain
            base_token = (
                id(base.data),
                base.data.shape,
                None if domain is None else id(domain),
            )
        parts = []
        for item in overlays:
            kind = item.get("kind")
            if kind == "region":
                region = item["region"]
                parts.append(
                    (
                        "region",
                        tuple(id(mask) for mask in region.objects),
                        region.width,
                        region.height,
                        item.get("color"),
                        item.get("draw"),
                        item.get("width"),
                        item.get("alpha"),
                        tuple(item.get("colors") or ()),
                    )
                )
            elif kind == "xld":
                xld = item["xld"]
                parts.append(
                    (
                        "xld",
                        id(xld),
                        len(xld.contours),
                        item.get("color"),
                        item.get("width"),
                        tuple(item.get("colors") or ()),
                    )
                )
            elif kind == "text":
                parts.append(("text", item.get("text"), item.get("row"), item.get("col"), item.get("color")))
            else:
                parts.append((kind, id(item)))
        panels = getattr(self.display, "calib_panels", None) or ()
        extra = (
            tuple((item.get("index"), item.get("top"), item.get("bottom")) for item in panels),
            getattr(self.display, "calib_note", None),
            bool(getattr(self.display, "calib_back", False)),
        )
        return (base_token, tuple(parts), extra)

    def _revision(self, cache_name: str, token: tuple) -> int:
        cached = getattr(self, cache_name, None)
        if cached is not None and cached[0] == token:
            return cached[1]
        rev = int(cached[1] if cached is not None else 0) + 1
        setattr(self, cache_name, (token, rev))
        return rev

    def _calib_extra(self, base: Image | None, preview: bool) -> dict | None:
        panels = getattr(self.display, "calib_panels", None)
        note = getattr(self.display, "calib_note", None)
        back = bool(getattr(self.display, "calib_back", False))
        if not panels and not note and not back:
            return None
        extra = {}
        if panels:
            shown = [dict(item) for item in panels]
            if preview and base is not None:
                scale = 640 / max(base.width, base.height)
                for item in shown:
                    item["top"] = float(item["top"]) * scale
                    item["bottom"] = float(item["bottom"]) * scale
            extra["calibPanels"] = shown
        if note:
            extra["calibNote"] = note
        if back:
            extra["calibBack"] = True
        return extra

    def _paint_display(self, base, overlays, preview: bool, token: tuple, rev: int) -> dict | None:
        payload = compose_display(base, overlays, preview=preview)
        extra = self._calib_extra(base, preview)
        if payload is not None and extra:
            payload = dict(payload)
            payload.update(extra)
        self._display_cache = (token, payload, rev)
        return payload

    def render_display_job(self, job: dict) -> dict | None:
        payload = compose_display(job["base"], job["overlays"], preview=bool(job["preview"]))
        extra = job.get("extra")
        if payload is not None and extra:
            payload = dict(payload)
            payload.update(extra)
        return payload

    def window_display(self, *, defer: bool = False):
        overlays = self._window_overlays()
        base = self.display.base
        preview = bool(getattr(self, "running", False)) and base is not None and max(base.width, base.height) > 640
        token = (self._display_token(overlays), preview)
        cached = getattr(self, "_display_cache", None)
        if cached is not None and cached[0] == token:
            return cached[1], cached[2], None
        rev = int(getattr(self, "_display_rev", 0)) + 1
        self._display_rev = rev
        if not defer:
            payload = self._paint_display(base, overlays, preview, token, rev)
            return payload, rev, None
        base_copy, overlay_copy = copy_display_scene(base, overlays)
        return None, rev, {
            "token": token,
            "rev": rev,
            "base": base_copy,
            "overlays": overlay_copy,
            "preview": preview,
            "extra": self._calib_extra(base, preview),
        }

    def store_display(self, job: dict, payload: dict | None) -> None:
        if int(getattr(self, "_display_rev", 0)) == int(job["rev"]):
            self._display_cache = (job["token"], payload, job["rev"])

    def public_state(self, *, defer_display: bool = False) -> dict:
        iconic = []
        control = []
        thumb_jobs = []
        variables, local_names = self._scoped_vars()
        for name, value in variables.items():
            scope = None
            if local_names:
                scope = "程序" if name in local_names else "呼叫端"
            if isinstance(value, list) and value and all(isinstance(item, Image) for item in value):
                kind, info, thumb, thumb_job = describe_iconic(value[-1], self._thumb_cache, defer=defer_display)
                entry = {
                    "name": name, "kind": kind, "role": _var_role(name, kind, True),
                    "scope": scope,
                    "info": f"{len(value)} 張  {info}", "thumb": thumb,
                }
                if thumb_job is not None:
                    thumb_job["target"] = entry
                    thumb_jobs.append(thumb_job)
                iconic.append(entry)
            elif isinstance(value, (Image, Region, XLD)):
                kind, info, thumb, thumb_job = describe_iconic(value, self._thumb_cache, defer=defer_display)
                entry = {
                    "name": name, "kind": kind, "role": _var_role(name, kind, True),
                    "scope": scope,
                    "info": info, "thumb": thumb,
                }
                if thumb_job is not None:
                    thumb_job["target"] = entry
                    thumb_jobs.append(thumb_job)
                iconic.append(entry)
            elif isinstance(value, ShapeModel):
                control.append({"name": name, "type": "shape_model", "scope": scope, "text": value.summary(), "full": value.summary()})
            elif isinstance(value, NccModel):
                control.append({"name": name, "type": "ncc_model", "scope": scope, "text": value.summary(), "full": value.summary()})
            elif callable(getattr(value, "summary", None)):
                text = value.summary()
                control.append({"name": name, "type": "model", "scope": scope, "text": text, "full": text})
            else:
                kind = "tuple" if isinstance(value, list) else "string" if isinstance(value, str) else "real" if isinstance(value, float) else "integer"
                role = _var_role(name, kind, False)
                control.append({"name": name, "type": role, "scope": scope, "text": fmt_value(value), "full": full_value(value)})
        histogram = None
        if self.histogram is not None:
            histogram = dict(self.histogram)
            histogram["color"] = self.region_color
        display, display_rev, display_job = self.window_display(defer=defer_display)
        var_rev = self._revision(
            "_var_cache",
            (
                tuple((item["name"], item.get("scope"), item["kind"], item["role"], item["info"], item["thumb"]) for item in iconic),
                tuple((item["name"], item.get("scope"), item["type"], item["text"], item["full"]) for item in control),
            ),
        )
        log_rev = self._revision("_log_cache", tuple(self.log[-300:]))
        state = {
            "id": self.id,
            "line": self.current_line(),
            "procedure": self._paused_procedure(),
            "arrowNote": getattr(self, "arrow_note", None),
            "done": self.done,
            "error": self.error,
            "errorLine": self.error_line,
            "paused": self.paused,
            "log": self.log[-300:],
            "logSize": len(self.log),
            "iconic": iconic,
            "control": control,
            "display": display,
            "displayRev": display_rev,
            "varRev": var_rev,
            "logRev": log_rev,
            "histogram": histogram,
            "steps": self.step_count,
            "imageStamp": int(getattr(self, "image_stamp", 0)),
            "displayStamp": int(getattr(self, "display_stamp", 0)),
            "ngShowing": bool(getattr(self, "_ng_showing", False)),
            "ngFile": str(getattr(self, "_ng_file", "") or ""),
            "lineTimes": [
                {"line": line, "ms": round(slot["total"] * 1000, 3), "count": slot["count"]}
                for line, slot in sorted(self.line_times.items())
            ],
        }
        if display_job is not None:
            state["_display_job"] = display_job
        if thumb_jobs:
            state["_thumb_jobs"] = thumb_jobs
        return state

    def view_object(self, name: str) -> dict:
        value = self.vars[name]
        if isinstance(value, list) and value and all(isinstance(item, Image) for item in value):
            value = value[-1]
        if isinstance(value, Image):
            display = compose_display(value, [])
            kind = "image"
        elif isinstance(value, Region):
            colors = None
            if value.multi_color and value.count() > 1:
                colors = [MATCH_COLORS[index % 16] for index in range(value.count())]
            display = compose_display(
                None,
                [{"kind": "region", "region": value, "color": "#ff3b30", "colors": colors, "draw": "margin", "width": 2}],
            )
            kind = "region"
        elif isinstance(value, XLD):
            groups = getattr(value, "groups", None) or []
            grouped = len(groups) == len(value.contours) and len(value.contours) > 0
            color_count = (max(groups) + 1) if grouped else len(value.contours)
            colors = None
            if value.multi_color and color_count > 1:
                colors = [MATCH_COLORS[index % 16] for index in range(color_count)]
            display = compose_display(
                None,
                [{"kind": "xld", "xld": value, "color": "#ffe14a", "colors": colors, "width": 2}],
            )
            kind = "xld"
        else:
            raise TypeError("這不是圖像變數")
        return {"name": name, "kind": kind, "display": display}


def _omit_when(state: dict, have: int | None, rev_key: str, drop: dict) -> dict:
    if have is None or state.get(rev_key) != have:
        return state
    trimmed = dict(state)
    trimmed.update(drop)
    return trimmed


def omit_known_display(state: dict, have: int | None) -> dict:
    return _omit_when(state, have, "displayRev", {"display": None, "displayKept": True})


def omit_known_state(
    state: dict,
    display_rev: int | None,
    var_rev: int | None,
    log_rev: int | None,
    log_at: int | None = None,
) -> dict:
    state = omit_known_display(state, display_rev)
    state = _omit_when(
        state,
        var_rev,
        "varRev",
        {"iconic": None, "control": None, "iconicKept": True, "controlKept": True},
    )
    if log_rev is not None and state.get("logRev") == log_rev:
        return _omit_when(state, log_rev, "logRev", {"log": None, "logKept": True})
    shown = state.get("log")
    size = state.get("logSize")
    if log_at is None or not isinstance(shown, list) or not isinstance(size, int):
        return state
    if log_at < 0 or log_at > size:
        return state
    start = size - len(shown)
    if log_at < start:
        return state
    trimmed = dict(state)
    trimmed["log"] = shown[log_at - start :]
    trimmed["logFrom"] = log_at
    return trimmed


def _num_literal(text: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", str(text).strip()))


def _in_range(current: float, end: float, step: float) -> bool:
    if step > 0:
        return current <= end + 1e-9
    return current >= end - 1e-9


def _remember_thumb(thumbs, token, build):
    if thumbs is None:
        return build()
    found = thumbs.get(token)
    if found is not None:
        thumbs.pop(token)
        thumbs[token] = found
        return found
    thumb = build()
    thumbs[token] = thumb
    if len(thumbs) > 48:
        while len(thumbs) > 32:
            thumbs.pop(next(iter(thumbs)))
    return thumb


def encode_thumb_job(job: dict) -> str:
    kind = job["kind"]
    if kind == "image":
        return jpeg_thumb(job["array"])
    if kind == "region":
        return jpeg_thumb(region_thumb(job["region"]))
    return jpeg_thumb(xld_thumb(job["xld"]))


def describe_iconic(value, thumbs=None, *, defer: bool = False):
    if isinstance(value, Image):
        channels = "byte" if value.channels == 1 else f"{value.channels} 通道"
        domain = value.domain
        token = ("image", id(value.data), value.data.shape, None if domain is None else id(domain))
        info = f"{value.width}×{value.height}  {channels}"
        if defer and (thumbs is None or token not in thumbs):
            return "image", info, "", {"token": token, "kind": "image", "array": apply_domain_dim(value).copy()}
        thumb = _remember_thumb(thumbs, token, lambda: jpeg_thumb(apply_domain_dim(value)))
        return "image", info, thumb, None
    if isinstance(value, Region):
        info = f"{value.count()} 個物件  {value.width}×{value.height}"
        if value.links:
            outers = holes = 0
            for item in value.links:
                for parent in item.get("parent") or []:
                    if parent < 0:
                        outers += 1
                    else:
                        holes += 1
            info += f"  link 外{outers} 孔{holes}"
        token = ("region", tuple(id(mask) for mask in value.objects), value.width, value.height)
        if defer and (thumbs is None or token not in thumbs):
            masks = [np.array(mask, copy=True) for mask in value.objects]
            copied = Region(value.width, value.height, masks, bool(value.multi_color), None)
            return "region", info, "", {"token": token, "kind": "region", "region": copied}
        return "region", info, _remember_thumb(thumbs, token, lambda: jpeg_thumb(region_thumb(value))), None
    token = ("xld", id(value), len(value.contours), value.width, value.height)
    info = f"{len(value.contours)} 條輪廓"
    if defer and (thumbs is None or token not in thumbs):
        contours = [[[float(point[0]), float(point[1])] for point in line] for line in value.contours]
        copied = XLD(value.width, value.height, contours, bool(value.multi_color), None)
        return "xld", info, "", {"token": token, "kind": "xld", "xld": copied}
    return "xld", info, _remember_thumb(thumbs, token, lambda: jpeg_thumb(xld_thumb(value))), None
