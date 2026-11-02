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
from .render import apply_domain_dim, compose_display, jpeg_thumb, region_thumb, xld_thumb
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

    def clear(self):
        self.base = None
        self.base_provisional = False
        self.overlays = []

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

    def reset(self):
        self._release_handles()
        self.pc = 0
        self.vars: dict = {}
        self.done = False
        self.error = None
        self.error_line = None
        self.paused = None
        self.log: list[str] = []
        self.display = Display()
        self.loops: dict = {}
        self.try_stack: list = []
        self.proc_depth = 0
        self.cancel = False
        self.defer_wait = False
        self.wait_until = 0.0
        self._wait_frame = None
        self.break_skip = None
        self.arrow_line = None
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
        if not self.done and self.pc < len(self.stmts):
            return self.stmts[self.pc].line
        return self.arrow_line

    def set_line(self, line: int):
        """Move the program arrow. Variables stay; the next step starts at this line."""
        target = None
        for index, stmt in enumerate(self.stmts):
            if isinstance(stmt, (Blank, Comment)):
                continue
            if stmt.line >= int(line):
                target = index
                break
        self.error = None
        self.error_line = None
        self.paused = None
        self.break_skip = None
        if target is None:
            self.pc = len(self.stmts)
            self.done = True
            self.arrow_line = None
            return
        self.pc = target
        self.done = False

    def resolve_read_path(self, filename: str) -> Path:
        path = Path(filename).expanduser()
        candidates = [path] if path.is_absolute() else [self.root / path, self.root / "samples" / path, self.root / "workspace" / path]
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

    def display_obj(self, obj, provisional: bool = False, name: str | None = None):
        source = name if name is not None else getattr(self, "pending_display_name", None)
        obj = coerce_iconic(obj)
        if isinstance(obj, list) and obj and all(isinstance(item, Image) for item in obj):
            obj = obj[-1]
        if isinstance(obj, Image):
            self.display.base = obj
            self.display.base_provisional = provisional
            self.display.overlays = []
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

    def trace_line(self, stmt, message: str):
        if self.trace:
            self.log.append(f"{stmt.line:>4}  {message}")

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
        self.paused = None
        self.break_skip = None
        self.prepare()
        if self.done:
            return
        self._execute_current()
        if not self.error:
            self.step_count += 1
        self.prepare()

    def run(self, max_steps: int = 100000, *, resume: bool = False):
        with self.lock:
            if not resume and (self.done or self.error):
                self.reset()
            self.paused = None
            if not resume:
                self.cancel = False
            if resume and getattr(self, "_wait_frame", None):
                self._resume_proc()
                if self.paused == "wait" or self.error:
                    return
        taken = 0
        while taken < max_steps:
            with self.lock:
                self.prepare()
                if self.done:
                    self.log.append(f"執行完成（{self.step_count} 步）")
                    return
                if self.cancel:
                    self.paused = "stopped"
                    self.log.append("已停止")
                    return
                if self._on_breakpoint() and self.pc != self.break_skip:
                    self.break_skip = self.pc
                    self.paused = "breakpoint"
                    self.log.append(f"中斷點 第 {self.stmts[self.pc].line} 行")
                    return
                self.break_skip = None
                self._execute_current()
                if self.error:
                    return
                self.step_count += 1
                taken += 1
                stop_now = self.paused in ("stop", "wait", "stopped")
            if stop_now:
                return
        with self.lock:
            self.error = "超過最大步數，可能有無窮迴圈"
            self.log.append(self.error)

    def _execute_current(self):
        index = self.pc
        stmt = self.stmts[index]
        self.arrow_line = stmt.line
        started = time.perf_counter()
        try:
            self._execute(stmt, index)
        except StopSignal:
            self.paused = "stopped" if self.cancel else "stop"
            self.log.append("已停止" if self.cancel else f"Vizion.stop() 於第 {stmt.line} 行")
        except (ExprError, ParseError, Exception) as exc:
            frame = self.try_stack[-1] if self.try_stack and self.try_stack[-1][2] == self.proc_depth else None
            if frame is not None:
                self.try_stack.pop()
                catch_index, var, _depth = frame
                self.vars[var] = str(exc)
                self.pc = catch_index + 1
                self.log.append(f"catch {var} 第 {stmt.line} 行：{exc}")
            elif self.proc_depth:
                self.error = f"程序第 {stmt.line} 行：{exc}"
                self.error_line = stmt.line
            else:
                self.error = str(exc)
                self.error_line = stmt.line
                self.log.append(f"錯誤 第 {stmt.line} 行：{exc}")
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
                self._call_proc(stmt, self.procs[stmt.name])
                if self.paused == "wait":
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
        self.vars = local
        self.pc = proc_index + 1
        end = self.proc_end[proc_index]
        guard = None
        try:
            steps = 0
            while self.pc < end and not self.error and self.paused not in ("stop", "breakpoint", "stopped", "wait"):
                self._execute_current()
                steps += 1
                if steps > 100000:
                    guard = "程序超過 100000 步"
                    break
        finally:
            local = self.vars
            inner_pc = self.pc
            waiting = self.paused == "wait"
            self.vars = outer
            self.pc = saved_pc
            self.proc_depth -= 1
        if waiting:
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
            self.error = None
            self.error_line = None
            raise RuntimeError(message)
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
        self.trace_line(stmt, stmt.name)

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
        try:
            steps = 0
            while self.pc < frame["end"] and not self.error and self.paused not in ("stop", "breakpoint", "stopped", "wait"):
                self._execute_current()
                steps += 1
                if steps > 100000:
                    self.error = "程序超過 100000 步"
                    break
        finally:
            local = self.vars
            inner_pc = self.pc
            waiting = self.paused == "wait"
            self.vars = caller_vars
            self.pc = frame["saved_pc"]
            self.proc_depth -= 1
        if waiting:
            frame["local"] = local
            frame["pc"] = inner_pc
            self._park_wait(frame)
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
        return (base_token, tuple(parts))

    def _revision(self, cache_name: str, token: tuple) -> int:
        cached = getattr(self, cache_name, None)
        if cached is not None and cached[0] == token:
            return cached[1]
        rev = int(cached[1] if cached is not None else 0) + 1
        setattr(self, cache_name, (token, rev))
        return rev

    def window_display(self) -> tuple[dict | None, int]:
        overlays = self._window_overlays()
        token = self._display_token(overlays)
        cached = getattr(self, "_display_cache", None)
        if cached is not None and cached[0] == token:
            return cached[1], cached[2]
        payload = compose_display(self.display.base, overlays)
        rev = int(getattr(self, "_display_rev", 0)) + 1
        self._display_rev = rev
        self._display_cache = (token, payload, rev)
        return payload, rev

    def public_state(self) -> dict:
        iconic = []
        control = []
        for name, value in self.vars.items():
            if isinstance(value, list) and value and all(isinstance(item, Image) for item in value):
                kind, info, thumb = describe_iconic(value[-1], self._thumb_cache)
                iconic.append({
                    "name": name, "kind": kind, "role": _var_role(name, kind, True),
                    "info": f"{len(value)} 張  {info}", "thumb": thumb,
                })
            elif isinstance(value, (Image, Region, XLD)):
                kind, info, thumb = describe_iconic(value, self._thumb_cache)
                iconic.append({
                    "name": name, "kind": kind, "role": _var_role(name, kind, True),
                    "info": info, "thumb": thumb,
                })
            elif isinstance(value, ShapeModel):
                control.append({"name": name, "type": "shape_model", "text": value.summary(), "full": value.summary()})
            elif isinstance(value, NccModel):
                control.append({"name": name, "type": "ncc_model", "text": value.summary(), "full": value.summary()})
            elif callable(getattr(value, "summary", None)):
                text = value.summary()
                control.append({"name": name, "type": "model", "text": text, "full": text})
            else:
                kind = "tuple" if isinstance(value, list) else "string" if isinstance(value, str) else "real" if isinstance(value, float) else "integer"
                role = _var_role(name, kind, False)
                control.append({"name": name, "type": role, "text": fmt_value(value), "full": full_value(value)})
        histogram = None
        if self.histogram is not None:
            histogram = dict(self.histogram)
            histogram["color"] = self.region_color
        display, display_rev = self.window_display()
        var_rev = self._revision(
            "_var_cache",
            (
                tuple((item["name"], item["kind"], item["role"], item["info"], item["thumb"]) for item in iconic),
                tuple((item["name"], item["type"], item["text"], item["full"]) for item in control),
            ),
        )
        log_rev = self._revision("_log_cache", tuple(self.log[-300:]))
        return {
            "id": self.id,
            "line": self.current_line(),
            "done": self.done,
            "error": self.error,
            "errorLine": self.error_line,
            "paused": self.paused,
            "log": self.log[-300:],
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
            "lineTimes": [
                {"line": line, "ms": round(slot["total"] * 1000, 3), "count": slot["count"]}
                for line, slot in sorted(self.line_times.items())
            ],
        }

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


def omit_known_state(state: dict, display_rev: int | None, var_rev: int | None, log_rev: int | None) -> dict:
    state = omit_known_display(state, display_rev)
    state = _omit_when(
        state,
        var_rev,
        "varRev",
        {"iconic": None, "control": None, "iconicKept": True, "controlKept": True},
    )
    return _omit_when(state, log_rev, "logRev", {"log": None, "logKept": True})


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
        return found
    thumb = build()
    if len(thumbs) > 64:
        thumbs.clear()
    thumbs[token] = thumb
    return thumb


def describe_iconic(value, thumbs=None):
    if isinstance(value, Image):
        channels = "byte" if value.channels == 1 else f"{value.channels} 通道"
        domain = value.domain
        token = ("image", id(value.data), value.data.shape, None if domain is None else id(domain))
        thumb = _remember_thumb(thumbs, token, lambda: jpeg_thumb(apply_domain_dim(value)))
        return "image", f"{value.width}×{value.height}  {channels}", thumb
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
        return "region", info, _remember_thumb(thumbs, token, lambda: jpeg_thumb(region_thumb(value)))
    token = ("xld", id(value), len(value.contours), value.width, value.height)
    return "xld", f"{len(value.contours)} 條輪廓", _remember_thumb(thumbs, token, lambda: jpeg_thumb(xld_thumb(value)))
