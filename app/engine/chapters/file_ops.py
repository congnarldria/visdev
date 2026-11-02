"""File and system operators. Paths are relative to the project, unless set_current_dir changed it."""

from __future__ import annotations

import os
import platform
import shutil
import socket
import time
from datetime import datetime
from pathlib import Path

from ..objects import Result
from ..operators import P, as_list, operator, scalar, text, whole

_IMAGE_NOTE = "相對路徑從專案目錄起算。set_current_dir 只改這一支程式的目前目錄，不改 Windows 的工作目錄。"


class FileHandle:
    def __init__(self, path: Path, mode: str, handle):
        self.path = path
        self.mode = mode
        self.closed = False
        self.sem_type = "file"
        self._file = handle

    def release(self) -> None:
        if self.closed:
            return
        try:
            self._file.close()
        except OSError:
            pass
        self.closed = True

    def summary(self) -> str:
        if self.closed:
            return "檔案  已關閉"
        return f"檔案  {self.mode}"


def _base(ctx) -> Path:
    current = getattr(ctx, "file_dir", None)
    return Path(current) if current else Path(ctx.root)


def _path(ctx, value, label: str) -> Path:
    raw = str(value).strip().strip("'\"")
    if not raw:
        raise ValueError(f"{label} 不能是空的")
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    base = _base(ctx).resolve()
    full = (base / path).resolve()
    try:
        full.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"{label} 不能用 .. 離開目前目錄") from exc
    return full


def _open(handle, label: str) -> FileHandle:
    if not isinstance(handle, FileHandle):
        raise TypeError(f"{label} 必須是 open_file 的句柄")
    if handle.closed or handle._file.closed:
        raise RuntimeError("檔案已經關閉")
    return handle


def _options(value) -> set[str]:
    found = set()
    for item in as_list(value):
        token = str(item).strip().strip("'\"").lower()
        if token:
            found.add(token)
    return found


@operator(
    "檔案",
    [],
    [],
    [P("FileName", "檔案或目錄")],
    [P("FileExists", "存在是 1")],
    "檢查檔案或目錄在不在",
    _IMAGE_NOTE + "檔案和目錄都算存在。",
    "Vizion.file_exists ('workspace/note.txt', FileExists)",
    "新算子。檔案或目錄都算存在，回 1 或 0。",
)
def file_exists(file_name, *, ctx):
    return Result([], [1 if _path(ctx, file_name, "FileName").exists() else 0])


@operator(
    "檔案",
    [],
    [],
    [
        P("Directory", "目錄"),
        P("Options", "files、directories、recursive，可同時給", "'files'", ["'files'", "'directories'", "'recursive'"]),
    ],
    [P("Files", "路徑")],
    "列出目錄裡的檔案",
    "files 只列檔案，directories 只列目錄。兩個都給就都列。recursive 會進子目錄。路徑用斜線，前面加上你寫的目錄。最多 5000 個。",
    "Vizion.list_files ('workspace', 'files', Files)",
    "新算子。files、directories、recursive 可同時給。最多 5000 筆，不跟隨捷徑。",
)
def list_files(directory, options, *, ctx):
    folder = _path(ctx, directory, "Directory")
    if not folder.is_dir():
        raise ValueError(f"找不到目錄 {folder}")
    flags = _options(options)
    unknown = flags - {"files", "directories", "recursive"}
    if unknown:
        raise ValueError("Options 只能是 files、directories 或 recursive")
    want_files = "files" in flags or "directories" not in flags
    want_dirs = "directories" in flags
    prefix = str(directory).strip().strip("'\"").replace("\\", "/").rstrip("/")
    listed = []
    walker = folder.rglob("*") if "recursive" in flags else folder.glob("*")
    for path in walker:
        if path.is_symlink():
            continue
        if path.is_file() and want_files:
            listed.append(path)
        elif path.is_dir() and want_dirs:
            listed.append(path)
        if len(listed) > 5000:
            raise RuntimeError("檔案太多，超過 5000 個")
    names = []
    for path in sorted(listed, key=lambda item: item.as_posix().lower()):
        rel = path.relative_to(folder).as_posix()
        names.append(f"{prefix}/{rel}" if prefix else rel)
    return Result([], [names])


@operator(
    "檔案",
    [],
    [],
    [P("FileName", "要刪的檔案")],
    [],
    "刪除檔案",
    "只刪檔案，不刪目錄。檔案不在會失敗。",
    "Vizion.delete_file ('workspace/note.txt')",
    "新算子。只刪檔案。目錄會拒絕。",
)
def delete_file(file_name, *, ctx):
    path = _path(ctx, file_name, "FileName")
    if path.is_dir():
        raise ValueError("這是目錄，請用 remove_dir")
    if not path.is_file():
        raise ValueError(f"找不到檔案 {path}")
    path.unlink()
    return Result([], [])


@operator(
    "檔案",
    [],
    [],
    [P("Source", "來源檔"), P("Destination", "目的檔")],
    [],
    "複製檔案",
    "目的檔的目錄必須已經存在。",
    "Vizion.copy_file ('workspace/note.txt', 'workspace/note_copy.txt')",
    "新算子。目的地的上一層目錄要先存在。",
)
def copy_file(source, destination, *, ctx):
    src = _path(ctx, source, "Source")
    dest = _path(ctx, destination, "Destination")
    if not src.is_file():
        raise ValueError(f"找不到檔案 {src}")
    if not dest.parent.is_dir():
        raise ValueError(f"找不到目錄 {dest.parent}")
    shutil.copyfile(src, dest)
    return Result([], [])


@operator(
    "檔案",
    [],
    [],
    [
        P("FileName", "檔名"),
        P("FileType", "input、output 或 append", "'input'", ["'input'", "'output'", "'append'"]),
    ],
    [P("FileHandle", "檔案句柄")],
    "打開檔案",
    "input 讀現有檔案。output 清掉重寫。append 接在後面。讀寫共用同一個位元組位置。",
    "Vizion.open_file ('workspace/note.txt', 'output', FileHandle)",
    "新算子。文字和位元組共用同一個讀寫位置。",
)
def open_file(file_name, file_type, *, ctx):
    mode = text(file_type, "FileType").strip().strip("'\"").lower()
    path = _path(ctx, file_name, "FileName")
    flags = {"input": "rb", "output": "wb", "append": "ab"}
    if mode not in flags:
        raise ValueError("FileType 只能是 input、output 或 append")
    if mode == "input" and not path.is_file():
        raise ValueError(f"讀不到檔案 {path}")
    if mode != "input" and not path.parent.is_dir():
        raise ValueError(f"找不到目錄 {path.parent}")
    handle = path.open(flags[mode])
    return Result([], [FileHandle(path, mode, handle)])


@operator(
    "檔案",
    [],
    [],
    [P("FileHandle", "檔案句柄")],
    [],
    "關閉檔案",
    "關掉之後不能再讀寫。句柄還在，但不再有效。",
    "Vizion.close_file (FileHandle)",
    "新算子。關掉之後再讀會失敗。",
)
def close_file(handle, *, ctx):
    opened = _open(handle, "FileHandle")
    opened.release()
    return Result([], [])


@operator(
    "檔案",
    [],
    [],
    [P("FileHandle", "已打開的檔案")],
    [P("OutLine", "一行，不含換行"), P("IsEOF", "讀到結尾是 1")],
    "讀一行",
    "遇到換行就停。檔案結尾時 IsEOF 是 1，OutLine 是剩下的字，可能是空的。",
    "Vizion.fread_line (FileHandle, OutLine, IsEOF)",
    "新算子。讀到換行就停。這一行後面還有字時，IsEOF 是 0。",
)
def fread_line(handle, *, ctx):
    opened = _open(handle, "FileHandle")
    if opened.mode != "input":
        raise RuntimeError("這個檔案不是 input")
    buf = bytearray()
    eof = 0
    while True:
        chunk = opened._file.read(1)
        if not chunk:
            eof = 1
            break
        if chunk == b"\n":
            break
        if chunk != b"\r":
            buf.extend(chunk)
    return Result([], [buf.decode("utf-8", errors="replace"), eof])


@operator(
    "檔案",
    [],
    [],
    [P("FileHandle", "已打開的檔案")],
    [P("OutString", "下一個字"), P("IsEOF", "沒讀到字是 1")],
    "讀一個以空白分開的字",
    "先跳過空白，再讀到下一個空白為止。什麼都沒有時 OutString 是空的，IsEOF 是 1。",
    "Vizion.fread_string (FileHandle, OutString, IsEOF)",
    "新算子。跳過空白，讀到下一個空白為止，空白留在檔案裡。",
)
def fread_string(handle, *, ctx):
    opened = _open(handle, "FileHandle")
    if opened.mode != "input":
        raise RuntimeError("這個檔案不是 input")
    while True:
        chunk = opened._file.read(1)
        if not chunk:
            return Result([], ["", 1])
        if chunk not in b" \t\r\n":
            buf = bytearray(chunk)
            break
    while True:
        pos = opened._file.tell()
        chunk = opened._file.read(1)
        if not chunk:
            break
        if chunk in b" \t\r\n":
            opened._file.seek(pos)
            break
        buf.extend(chunk)
    return Result([], [buf.decode("utf-8", errors="replace"), 0])


@operator(
    "檔案",
    [],
    [],
    [P("FileHandle", "已打開的檔案")],
    [P("Char", "一個字元")],
    "讀一個字元",
    "一次一個位元組。已經到結尾會失敗。",
    "Vizion.fread_char (FileHandle, Char)",
    "新算子。一次一個位元組。讀完會報錯。",
)
def fread_char(handle, *, ctx):
    opened = _open(handle, "FileHandle")
    if opened.mode != "input":
        raise RuntimeError("這個檔案不是 input")
    chunk = opened._file.read(1)
    if not chunk:
        raise RuntimeError("檔案已經讀完")
    return Result([], [chunk.decode("latin-1")])


@operator(
    "檔案",
    [],
    [],
    [P("FileHandle", "已打開的檔案"), P("NumberOfBytes", "要讀的位元組數")],
    [P("Data", "0 到 255 的 tuple"), P("IsEOF", "不夠數是 1")],
    "讀位元組",
    "每個位元組是 0 到 255。讀不夠時 IsEOF 是 1，Data 是實際讀到的。",
    "Vizion.fread_bytes (FileHandle, 3, Data, IsEOF)",
    "新算子。回 0 到 255 的整數。要的比剩下的多時，IsEOF 是 1。",
)
def fread_bytes(handle, count, *, ctx):
    opened = _open(handle, "FileHandle")
    if opened.mode != "input":
        raise RuntimeError("這個檔案不是 input")
    number = whole(count, "NumberOfBytes")
    if number < 1:
        raise ValueError("位元組數至少要 1")
    data = opened._file.read(number)
    return Result([], [[item for item in data], 1 if len(data) < number else 0])


@operator(
    "檔案",
    [],
    [],
    [P("FileHandle", "已打開的檔案"), P("String", "要寫的文字")],
    [],
    "寫出文字",
    "不加換行。換行用 fnew_line。",
    "Vizion.fwrite_string (FileHandle, 'VISDEV')",
    "新算子。寫完就寫進磁碟。",
)
def fwrite_string(handle, value, *, ctx):
    opened = _open(handle, "FileHandle")
    if opened.mode == "input":
        raise RuntimeError("這個檔案不是 output 或 append")
    opened._file.write(text(value, "String").encode("utf-8"))
    opened._file.flush()
    return Result([], [])


@operator(
    "檔案",
    [],
    [],
    [P("FileHandle", "已打開的檔案")],
    [],
    "寫出換行",
    "寫一個換行並送出緩衝。",
    "Vizion.fnew_line (FileHandle)",
    "新算子。寫一個換行。",
)
def fnew_line(handle, *, ctx):
    opened = _open(handle, "FileHandle")
    if opened.mode == "input":
        raise RuntimeError("這個檔案不是 output 或 append")
    opened._file.write(b"\n")
    opened._file.flush()
    return Result([], [])


@operator(
    "檔案",
    [],
    [],
    [P("DirName", "新目錄")],
    [],
    "建立目錄",
    "上一層必須已經存在。目錄已經在會失敗。",
    "Vizion.make_dir ('workspace/empty_dir')",
    "新算子。只建一層。已經有的目錄會報錯。",
)
def make_dir(dir_name, *, ctx):
    path = _path(ctx, dir_name, "DirName")
    if path.exists():
        raise ValueError("目錄已經存在")
    if not path.parent.is_dir():
        raise ValueError(f"找不到目錄 {path.parent}")
    path.mkdir()
    return Result([], [])


@operator(
    "檔案",
    [],
    [],
    [P("DirName", "空目錄")],
    [],
    "刪除空目錄",
    "目錄裡還有東西會失敗。",
    "Vizion.remove_dir ('workspace/empty_dir')",
    "新算子。目錄裡還有檔案時不會刪。",
)
def remove_dir(dir_name, *, ctx):
    path = _path(ctx, dir_name, "DirName")
    if not path.is_dir():
        raise ValueError(f"找不到目錄 {path}")
    try:
        path.rmdir()
    except OSError as exc:
        raise ValueError("目錄不是空的") from exc
    return Result([], [])


@operator(
    "檔案",
    [],
    [],
    [],
    [P("DirName", "目前目錄")],
    "讀出目前目錄",
    _IMAGE_NOTE,
    "Vizion.get_current_dir (DirName)",
    "新算子。還沒改過時，目前目錄就是專案目錄。",
)
def get_current_dir(*, ctx):
    return Result([], [str(_base(ctx))])


@operator(
    "檔案",
    [],
    [],
    [P("DirName", "要改去的目錄")],
    [],
    "改這支程式的目前目錄",
    "只影響後面的相對路徑。不改 Windows 的工作目錄。目錄必須存在。",
    "Vizion.set_current_dir ('workspace')",
    "新算子。只改這一支程式後面的相對路徑，不改 Windows 的工作目錄。",
)
def set_current_dir(dir_name, *, ctx):
    path = _path(ctx, dir_name, "DirName")
    if not path.is_dir():
        raise ValueError(f"找不到目錄 {path}")
    ctx.file_dir = path
    return Result([], [])


@operator(
    "系統",
    [],
    [],
    [],
    [P("Seconds", "從一個固定起點算的秒數")],
    "讀一個單調增加的秒數",
    "兩次相減是經過的秒。數字本身沒有日期意義。",
    "Vizion.count_seconds (Seconds)",
    "新算子。兩次相減才是經過的秒，不是牆上時鐘。",
)
def count_seconds(*, ctx):
    return Result([], [time.perf_counter()])


@operator(
    "系統",
    [],
    [],
    [P("Seconds", "要停的秒數")],
    [],
    "停住這一步",
    "0 是立刻回來。超過 30 秒會失敗，避免整支程式卡住。等待中可以按停止。",
    "Vizion.wait_seconds (0)",
    "新算子。這一支會停住，最長 30 秒。網頁執行時不佔住伺服器。",
)
def wait_seconds(seconds, *, ctx):
    delay = scalar(seconds, "Seconds")
    if delay < 0:
        raise ValueError("秒數不能是負的")
    if delay > 30:
        raise ValueError("等待不能超過 30 秒")
    if delay <= 0:
        return Result([], [])
    if getattr(ctx, "defer_wait", False) and int(getattr(ctx, "proc_depth", 0)) == 0:
        ctx.wait_until = time.monotonic() + delay
        ctx.paused = "wait"
        return Result([], [])
    deadline = time.monotonic() + delay
    while True:
        if getattr(ctx, "cancel", False):
            return Result([], [])
        left = deadline - time.monotonic()
        if left <= 0:
            break
        time.sleep(min(0.05, left))
    return Result([], [])


@operator(
    "系統",
    [],
    [],
    [],
    [
        P("MSecond", "毫秒"),
        P("Second", "秒"),
        P("Minute", "分"),
        P("Hour", "時"),
        P("Day", "日"),
        P("YDay", "一年中的第幾天"),
        P("Month", "月"),
        P("Year", "年"),
    ],
    "讀出現在的時間",
    "用這台電腦的本地時間。月從 1 開始，YDay 也從 1 開始。",
    "Vizion.get_system_time (MSecond, Second, Minute, Hour, Day, YDay, Month, Year)",
    "新算子。八個數依序是毫秒、秒、分、時、日、年內第幾天、月、年。",
)
def get_system_time(*, ctx):
    now = datetime.now()
    return Result([], [
        now.microsecond // 1000,
        now.second,
        now.minute,
        now.hour,
        now.day,
        now.timetuple().tm_yday,
        now.month,
        now.year,
    ])


@operator(
    "系統",
    [],
    [],
    [P("Query", "operating_system、hostname 或 num_proc", "'operating_system'", ["'operating_system'", "'hostname'", "'num_proc'"])],
    [P("Information", "結果")],
    "讀這台電腦的資訊",
    "operating_system 是作業系統名稱。hostname 是電腦名稱。num_proc 是處理器數量。",
    "Vizion.get_system_info ('operating_system', Information)",
    "新算子。可問 operating_system、hostname、num_proc。",
)
def get_system_info(query, *, ctx):
    kind = text(query, "Query").strip().strip("'\"").lower()
    if kind == "operating_system":
        return Result([], [platform.system()])
    if kind == "hostname":
        return Result([], [socket.gethostname()])
    if kind == "num_proc":
        return Result([], [int(os.cpu_count() or 1)])
    raise ValueError("沒有這個系統資訊")
