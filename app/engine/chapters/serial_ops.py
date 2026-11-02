"""Serial ports through the Windows API, plus an in-memory loopback.

COM names use CreateFile, ReadFile, and WriteFile, so Windows 7 and Windows 10
share one path. The port name loopback only stores bytes in this process so a
sample can round-trip without a cable.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

from ..operators import P, operator, text, whole

_GENERIC_READ = 0x80000000
_GENERIC_WRITE = 0x40000000
_OPEN_EXISTING = 3
_INVALID = ctypes.c_void_p(-1).value


class _DCB(ctypes.Structure):
    _fields_ = [
        ("DCBlength", wintypes.DWORD),
        ("BaudRate", wintypes.DWORD),
        ("fBinary", wintypes.DWORD, 1),
        ("fParity", wintypes.DWORD, 1),
        ("fOutxCtsFlow", wintypes.DWORD, 1),
        ("fOutxDsrFlow", wintypes.DWORD, 1),
        ("fDtrControl", wintypes.DWORD, 2),
        ("fDsrSensitivity", wintypes.DWORD, 1),
        ("fTXContinueOnXoff", wintypes.DWORD, 1),
        ("fOutX", wintypes.DWORD, 1),
        ("fInX", wintypes.DWORD, 1),
        ("fErrorChar", wintypes.DWORD, 1),
        ("fNull", wintypes.DWORD, 1),
        ("fRtsControl", wintypes.DWORD, 2),
        ("fAbortOnError", wintypes.DWORD, 1),
        ("fDummy2", wintypes.DWORD, 17),
        ("wReserved", wintypes.WORD),
        ("XonLim", wintypes.WORD),
        ("XoffLim", wintypes.WORD),
        ("ByteSize", wintypes.BYTE),
        ("Parity", wintypes.BYTE),
        ("StopBits", wintypes.BYTE),
        ("XonChar", ctypes.c_char),
        ("XoffChar", ctypes.c_char),
        ("ErrorChar", ctypes.c_char),
        ("EofChar", ctypes.c_char),
        ("EvtChar", ctypes.c_char),
        ("wReserved1", wintypes.WORD),
    ]


class _Timeouts(ctypes.Structure):
    _fields_ = [
        ("ReadIntervalTimeout", wintypes.DWORD),
        ("ReadTotalTimeoutMultiplier", wintypes.DWORD),
        ("ReadTotalTimeoutConstant", wintypes.DWORD),
        ("WriteTotalTimeoutMultiplier", wintypes.DWORD),
        ("WriteTotalTimeoutConstant", wintypes.DWORD),
    ]


if ctypes.sizeof(_DCB) != 28:
    raise RuntimeError("DCB 大小不對")

_kernel = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel.CreateFileW.argtypes = [
    wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
    wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
]
_kernel.CreateFileW.restype = ctypes.c_void_p
_kernel.ReadFile.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p,
]
_kernel.ReadFile.restype = wintypes.BOOL
_kernel.WriteFile.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p,
]
_kernel.WriteFile.restype = wintypes.BOOL
_kernel.CloseHandle.argtypes = [ctypes.c_void_p]
_kernel.CloseHandle.restype = wintypes.BOOL
_kernel.GetCommState.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DCB)]
_kernel.GetCommState.restype = wintypes.BOOL
_kernel.SetCommState.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DCB)]
_kernel.SetCommState.restype = wintypes.BOOL
_kernel.SetCommTimeouts.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Timeouts)]
_kernel.SetCommTimeouts.restype = wintypes.BOOL


class SerialPort:
    def __init__(self, name: str, native: int | None, loopback: bool) -> None:
        self.name = name
        self.native = native
        self.loopback = loopback
        self.buffer = bytearray()
        self.closed = False
        self.baud = 9600
        self.data_bits = 8
        self.flow = "none"
        self.parity = "none"
        self.stop_bits = 1
        self.total_timeout = 200
        self.inter_char = 50

    def summary(self) -> str:
        if self.closed:
            return "序列埠  已關閉"
        return f"序列埠  {self.name}"


def _port_name(value) -> str:
    name = text(value, "PortName").strip().strip("'\"")
    if name.lower().endswith(":") and name.lower().startswith("com"):
        name = name[:-1]
    if not name:
        raise ValueError("PortName 不能是空的")
    return name


def _as_bytes(value) -> bytes:
    if isinstance(value, str):
        try:
            return value.encode("latin-1")
        except UnicodeEncodeError as exc:
            raise ValueError("序列埠只能寫 0 到 255 的位元組") from exc
    if isinstance(value, bool) or not isinstance(value, (int, float, list, tuple)):
        raise TypeError("Data 必須是字串或位元組")
    items = value if isinstance(value, (list, tuple)) else [value]
    out = bytearray()
    for item in items:
        number = int(item)
        if number < 0 or number > 255:
            raise ValueError("序列埠只能寫 0 到 255 的位元組")
        out.append(number)
    return bytes(out)


def _require(port: SerialPort) -> None:
    if not isinstance(port, SerialPort):
        raise TypeError("SerialHandle 必須是 open_serial 打開的序列埠")
    if port.closed:
        raise RuntimeError("序列埠已經關閉")


def _device_path(name: str) -> str:
    if name.startswith("\\\\.\\"):
        return name
    return "\\\\.\\" + name


def _apply(port: SerialPort) -> None:
    if port.native is None:
        return
    timeouts = _Timeouts(port.inter_char, 0, port.total_timeout, 0, port.total_timeout)
    _kernel.SetCommTimeouts(port.native, ctypes.byref(timeouts))
    state = _DCB()
    state.DCBlength = ctypes.sizeof(_DCB)
    if not _kernel.GetCommState(port.native, ctypes.byref(state)):
        raise RuntimeError("這個序列埠套不到這些參數")
    state.BaudRate = port.baud
    state.ByteSize = port.data_bits
    state.Parity = {"none": 0, "odd": 1, "even": 2}[port.parity]
    state.StopBits = 0 if port.stop_bits == 1 else 2
    state.fBinary = 1
    state.fParity = 0 if port.parity == "none" else 1
    state.fOutX = 1 if port.flow == "xon/xoff" else 0
    state.fInX = state.fOutX
    state.fOutxCtsFlow = 1 if port.flow == "rts/cts" else 0
    state.fRtsControl = 2 if port.flow == "rts/cts" else 1
    if not _kernel.SetCommState(port.native, ctypes.byref(state)):
        raise RuntimeError("這個序列埠套不到這些參數")


def _open_windows(name: str) -> SerialPort:
    native = _kernel.CreateFileW(
        _device_path(name),
        _GENERIC_READ | _GENERIC_WRITE,
        0,
        None,
        _OPEN_EXISTING,
        0,
        None,
    )
    if not native or native == _INVALID:
        raise ValueError(f"打不開 {name}")
    port = SerialPort(name, int(native), False)
    _apply(port)
    return port


@operator(
    "通訊",
    [],
    [],
    [P("PortName", "COM 名稱，或 loopback")],
    [P("SerialHandle", "序列埠")],
    "打開序列埠",
    "COM 用 Windows API 打開，9600、8 個資料位元、沒有同位、1 個停止位元。Windows 7 和 Windows 10 同一套。"
    "loopback 不接真的 COM，寫進去的位元組存在這支程式裡，馬上讀得回來。打不開時會說是哪一個埠。",
    "Vizion.open_serial ('loopback', SerialHandle)",
)
def open_serial(port_name, *, ctx):
    from ..objects import Result

    name = _port_name(port_name)
    if name.lower() == "loopback":
        return Result([], [SerialPort("loopback", None, True)])
    return Result([], [_open_windows(name)])


@operator(
    "通訊",
    [],
    [],
    [P("SerialHandle", "序列埠"), P("Data", "要寫出的字串或 0 到 255 的位元組")],
    [],
    "寫出序列埠資料",
    "字串每個字是一個位元組。整數也要在 0 到 255。loopback 先存著，等 read_serial 讀走。",
    "Vizion.write_serial (SerialHandle, 'VISDEV')",
)
def write_serial(handle, data, *, ctx):
    from ..objects import Result

    port = handle if isinstance(handle, SerialPort) else None
    _require(port)
    payload = _as_bytes(data)
    if port.loopback:
        port.buffer.extend(payload)
        return Result([], [])
    written = wintypes.DWORD(0)
    buffer = ctypes.create_string_buffer(payload)
    ok = _kernel.WriteFile(port.native, buffer, len(payload), ctypes.byref(written), None)
    if not ok or written.value != len(payload):
        raise RuntimeError(f"寫不到 {port.name}")
    return Result([], [])


@operator(
    "通訊",
    [],
    [],
    [P("SerialHandle", "序列埠"), P("NumCharacters", "要讀的位元組數")],
    [P("Data", "讀到的字串")],
    "讀回序列埠資料",
    "loopback 讀的是剛剛寫進去的位元組。真的 COM 的等待時間用 set_serial_param，預設 0.2 秒。讀到的每個位元組組成一個字串。",
    "Vizion.read_serial (SerialHandle, 6, Data)",
)
def read_serial(handle, count, *, ctx):
    from ..objects import Result

    port = handle if isinstance(handle, SerialPort) else None
    _require(port)
    if isinstance(count, bool) or not isinstance(count, (int, float)):
        raise TypeError("NumCharacters 必須是數")
    number = int(count)
    if number < 0:
        raise ValueError("NumCharacters 不能是負的")
    if number == 0:
        return Result([], [""])
    if port.loopback:
        if len(port.buffer) < number:
            raise RuntimeError("序列埠還沒有這麼多資料")
        payload = bytes(port.buffer[:number])
        del port.buffer[:number]
        return Result([], [payload.decode("latin-1")])
    buffer = ctypes.create_string_buffer(number)
    got = wintypes.DWORD(0)
    ok = _kernel.ReadFile(port.native, buffer, number, ctypes.byref(got), None)
    if not ok or got.value == 0:
        raise RuntimeError(f"{port.name} 沒有讀到資料")
    return Result([], [buffer.raw[: got.value].decode("latin-1")])


@operator(
    "通訊",
    [],
    [],
    [P("SerialHandle", "序列埠")],
    [],
    "關閉序列埠",
    "關掉 Windows 開的 COM。loopback 也從這裡關掉，關掉後不能再讀寫。",
    "Vizion.close_serial (SerialHandle)",
)
def close_serial(handle, *, ctx):
    from ..objects import Result

    port = handle if isinstance(handle, SerialPort) else None
    _require(port)
    if port.native is not None:
        _kernel.CloseHandle(port.native)
        port.native = None
    port.buffer.clear()
    port.closed = True
    return Result([], [])


_FLOWS = ["'none'", "'xon/xoff'", "'rts/cts'"]
_PARITY = ["'none'", "'even'", "'odd'"]


def _choice(value, name: str, allowed: tuple[str, ...]) -> str:
    picked = text(value, name).strip().strip("'\"").lower()
    if picked not in allowed:
        shown = "、".join(allowed)
        raise ValueError(f"{name} 只能是 {shown}")
    return picked


def _nonneg(value, name: str) -> int:
    number = whole(value, name)
    if number < 0:
        raise ValueError(f"{name} 不能是負的")
    return number


@operator(
    "通訊",
    [],
    [],
    [
        P("SerialHandle", "序列埠"),
        P("BaudRate", "每秒位元數", "9600"),
        P("DataBits", "資料位元，5 到 8", "8"),
        P("FlowControl", "none、xon/xoff 或 rts/cts", "'none'", _FLOWS),
        P("Parity", "none、even 或 odd", "'none'", _PARITY),
        P("StopBits", "1 或 2", "1"),
        P("TotalTimeout", "整段讀取最多等幾毫秒", "200"),
        P("InterCharTimeout", "兩個位元組之間最多等幾毫秒", "50"),
    ],
    [],
    "設定序列埠參數",
    "真的 COM 會寫進 Windows 的序列埠設定。loopback 把同一組數字記下來，給 get_serial_param 讀回。",
    "Vizion.set_serial_param (SerialHandle, 19200, 8, 'none', 'none', 1, 500, 20)",
)
def set_serial_param(handle, baud, bits, flow, parity, stop, total, gap, *, ctx):
    from ..objects import Result

    port = handle if isinstance(handle, SerialPort) else None
    _require(port)
    rate = whole(baud, "BaudRate")
    if rate <= 0:
        raise ValueError("BaudRate 必須大於 0")
    width = whole(bits, "DataBits")
    if width < 5 or width > 8:
        raise ValueError("DataBits 只能是 5 到 8")
    stop_bits = whole(stop, "StopBits")
    if stop_bits not in (1, 2):
        raise ValueError("StopBits 只能是 1 或 2")
    port.baud = rate
    port.data_bits = width
    port.flow = _choice(flow, "FlowControl", ("none", "xon/xoff", "rts/cts"))
    port.parity = _choice(parity, "Parity", ("none", "even", "odd"))
    port.stop_bits = stop_bits
    port.total_timeout = _nonneg(total, "TotalTimeout")
    port.inter_char = _nonneg(gap, "InterCharTimeout")
    _apply(port)
    return Result([], [])


@operator(
    "通訊",
    [],
    [],
    [P("SerialHandle", "序列埠")],
    [
        P("BaudRate", "每秒位元數"),
        P("DataBits", "資料位元"),
        P("FlowControl", "流量控制"),
        P("Parity", "同位"),
        P("StopBits", "停止位元"),
        P("TotalTimeout", "整段讀取的毫秒"),
        P("InterCharTimeout", "位元組間隔的毫秒"),
    ],
    "讀回序列埠參數",
    "打開時是 9600、8 個資料位元、沒有流量控制、沒有同位、1 個停止位元、整段 200 毫秒、間隔 50 毫秒。",
    "Vizion.get_serial_param (SerialHandle, BaudRate, DataBits, FlowControl, Parity, StopBits, TotalTimeout, InterCharTimeout)",
)
def get_serial_param(handle, *, ctx):
    from ..objects import Result

    port = handle if isinstance(handle, SerialPort) else None
    _require(port)
    return Result([], [
        port.baud, port.data_bits, port.flow, port.parity, port.stop_bits, port.total_timeout, port.inter_char,
    ])
