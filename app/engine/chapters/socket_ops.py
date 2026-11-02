"""TCP sockets from the Python standard library.

Host name loopback pairs two local sockets and echoes bytes, so a sample can
round-trip without another program. Any other host is a real connect.
open_socket_accept listens on 127.0.0.1 and takes one client.
"""

from __future__ import annotations

import ctypes
import socket
import threading
from ctypes import wintypes

from ..operators import P, operator, text, whole


_PARAMS = ["'timeout'", "'nodelay'", "'available'", "'peer'"]
_FIONREAD = 0x4004667F


class SocketLink:
    def __init__(self, sock: socket.socket | None, peer: str) -> None:
        self.sock = sock
        self.peer = peer
        self.closed = False
        self.timeout = 2.0
        self.nodelay = 0
        self.listener: socket.socket | None = None
        self.ready = threading.Event()
        if sock is not None:
            self.ready.set()

    def summary(self) -> str:
        if self.closed:
            return "連線  已關閉"
        if self.sock is None:
            return f"連線  {self.peer} 聽"
        return f"連線  {self.peer}"


def _peer(value) -> str:
    name = text(value, "HostName").strip().strip("'\"")
    if not name:
        raise ValueError("HostName 不能是空的")
    return name


def _as_bytes(value) -> bytes:
    if isinstance(value, str):
        try:
            return value.encode("latin-1")
        except UnicodeEncodeError as exc:
            raise ValueError("連線只能寫 0 到 255 的位元組") from exc
    if isinstance(value, bool) or not isinstance(value, (int, float, list, tuple)):
        raise TypeError("Data 必須是字串或位元組")
    items = value if isinstance(value, (list, tuple)) else [value]
    out = bytearray()
    for item in items:
        number = int(item)
        if number < 0 or number > 255:
            raise ValueError("連線只能寫 0 到 255 的位元組")
        out.append(number)
    return bytes(out)


def _require(link) -> SocketLink:
    if not isinstance(link, SocketLink):
        raise TypeError("Socket 必須是 open_socket_connect 或 open_socket_accept 打開的連線")
    if link.closed:
        raise RuntimeError("連線已經關閉")
    return link


def _ready(link) -> SocketLink:
    link = _require(link)
    if link.sock is None:
        if not link.ready.wait(link.timeout):
            raise RuntimeError("沒有連線進來")
        if link.sock is None:
            raise RuntimeError("沒有連線進來")
    return link


def _tune(link: SocketLink) -> None:
    if link.sock is None:
        return
    link.sock.settimeout(link.timeout)
    try:
        link.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, int(link.nodelay))
    except OSError:
        pass


def _pending(sock: socket.socket) -> int:
    count = wintypes.DWORD(0)
    ws = ctypes.WinDLL("ws2_32", use_last_error=True)
    ws.ioctlsocket.argtypes = [ctypes.c_size_t, ctypes.c_long, ctypes.POINTER(wintypes.DWORD)]
    ws.ioctlsocket.restype = ctypes.c_int
    if ws.ioctlsocket(sock.fileno(), _FIONREAD, ctypes.byref(count)) != 0:
        return 0
    return int(count.value)


def _echo(side: socket.socket) -> None:
    try:
        while True:
            data = side.recv(4096)
            if not data:
                break
            side.sendall(data)
    except OSError:
        pass
    finally:
        try:
            side.close()
        except OSError:
            pass


def _loopback() -> socket.socket:
    left, right = socket.socketpair()
    left.settimeout(2)
    right.settimeout(2)
    threading.Thread(target=_echo, args=(right,), daemon=True).start()
    return left


def _connect(host: str, port: int) -> socket.socket:
    try:
        link = socket.create_connection((host, port), timeout=2)
    except OSError as exc:
        raise RuntimeError(f"連不上 {host}:{port}") from exc
    link.settimeout(2)
    return link


@operator(
    "通訊",
    [],
    [],
    [P("HostName", "主機名稱，或 loopback"), P("Port", "埠號。loopback 不使用")],
    [P("Socket", "連線")],
    "連上 TCP",
    "用 Python 內建 socket 連到主機和埠號。loopback 在這支程式裡配對兩端，寫出的位元組會被送回來，不必另外開服務。"
    "連不上時會說出主機和埠號。",
    "Vizion.open_socket_connect ('loopback', 0, Socket)",
)
def open_socket_connect(host_name, port, *, ctx):
    from ..objects import Result

    host = _peer(host_name)
    number = whole(port, "Port")
    if host.lower() == "loopback":
        return Result([], [SocketLink(_loopback(), "loopback")])
    if number < 1 or number > 65535:
        raise ValueError("Port 必須在 1 到 65535")
    return Result([], [SocketLink(_connect(host, number), f"{host}:{number}")])


@operator(
    "通訊",
    [],
    [],
    [P("Socket", "連線"), P("Data", "要送出的字串或 0 到 255 的位元組")],
    [],
    "送出 TCP 資料",
    "字串每個字是一個位元組。整數也要在 0 到 255。",
    "Vizion.send_data (Socket, 'VISDEV')",
)
def send_data(link, data, *, ctx):
    from ..objects import Result

    sock = _ready(link)
    payload = _as_bytes(data)
    try:
        sock.sock.sendall(payload)
    except OSError as exc:
        raise RuntimeError(f"送不到 {sock.peer}") from exc
    return Result([], [])


@operator(
    "通訊",
    [],
    [],
    [P("Socket", "連線"), P("NumCharacters", "要讀的位元組數")],
    [P("Data", "讀到的字串")],
    "讀回 TCP 資料",
    "最多等 2 秒，讀滿要求的位元組。loopback 讀的是剛剛送出的內容。",
    "Vizion.receive_data (Socket, 6, Data)",
)
def receive_data(link, count, *, ctx):
    from ..objects import Result

    sock = _ready(link)
    if isinstance(count, bool) or not isinstance(count, (int, float)):
        raise TypeError("NumCharacters 必須是數")
    number = int(count)
    if number < 0:
        raise ValueError("NumCharacters 不能是負的")
    if number == 0:
        return Result([], [""])
    payload = bytearray()
    try:
        while len(payload) < number:
            chunk = sock.sock.recv(number - len(payload))
            if not chunk:
                break
            payload.extend(chunk)
    except socket.timeout as exc:
        if not payload:
            raise RuntimeError(f"{sock.peer} 沒有讀到資料") from exc
    except OSError as exc:
        raise RuntimeError(f"{sock.peer} 沒有讀到資料") from exc
    if len(payload) < number:
        raise RuntimeError("連線還沒有這麼多資料")
    return Result([], [bytes(payload).decode("latin-1")])


@operator(
    "通訊",
    [],
    [],
    [P("Socket", "連線")],
    [],
    "關閉 TCP 連線",
    "關掉這條 socket。關掉後不能再送或讀。",
    "Vizion.close_socket (Socket)",
)
def close_socket(link, *, ctx):
    from ..objects import Result

    sock = _require(link)
    if sock.listener is not None:
        try:
            sock.listener.close()
        except OSError:
            pass
        sock.listener = None
    if sock.sock is not None:
        try:
            sock.sock.close()
        except OSError:
            pass
    sock.ready.set()
    sock.closed = True
    return Result([], [])


def _accept_one(link: SocketLink, server: socket.socket) -> None:
    try:
        conn, addr = server.accept()
        link.sock = conn
        link.peer = f"{addr[0]}:{addr[1]}"
        _tune(link)
    except OSError:
        pass
    finally:
        link.ready.set()


@operator(
    "通訊",
    [],
    [],
    [P("Port", "要聽的埠號。0 會挑一個空埠")],
    [P("Socket", "聽進來的連線"), P("BoundPort", "實際埠號")],
    "聽一個 TCP 埠",
    "在 127.0.0.1 聽一個連線。埠號 0 由系統挑空埠，BoundPort 是實際數字。"
    "另一端用 open_socket_connect 連上來之後，就可以 send_data 和 receive_data。",
    "Vizion.open_socket_accept (0, Socket, BoundPort)",
)
def open_socket_accept(port, *, ctx):
    from ..objects import Result

    number = whole(port, "Port")
    if number < 0 or number > 65535:
        raise ValueError("Port 必須在 0 到 65535")
    server = socket.socket()
    try:
        server.bind(("127.0.0.1", number))
        server.listen(1)
    except OSError as exc:
        server.close()
        raise RuntimeError(f"聽不了 127.0.0.1:{number}") from exc
    bound = int(server.getsockname()[1])
    link = SocketLink(None, f"127.0.0.1:{bound}")
    link.listener = server
    threading.Thread(target=_accept_one, args=(link, server), daemon=True).start()
    return Result([], [link, bound])


@operator(
    "通訊",
    [],
    [],
    [
        P("Socket", "連線"),
        P("ParamName", "timeout、nodelay、available 或 peer", "'timeout'", _PARAMS),
        P("ParamValue", "新的值"),
    ],
    [],
    "設定 TCP 參數",
    "timeout 是秒，讀資料最多等這麼久。nodelay 是 0 或 1，1 表示送出時不要等小包湊滿。",
    "Vizion.set_socket_param (Socket, 'timeout', 1)",
)
def set_socket_param(link, name, value, *, ctx):
    from ..objects import Result

    sock = _require(link)
    param = text(name, "ParamName").strip().strip("'\"").lower()
    if param == "timeout":
        if isinstance(value, bool):
            raise TypeError("timeout 必須是數")
        try:
            seconds = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("timeout 必須是數") from exc
        if seconds < 0:
            raise ValueError("timeout 不能是負的")
        sock.timeout = seconds
        _tune(sock)
        return Result([], [])
    if param == "nodelay":
        flag = whole(value, "ParamValue")
        if flag not in (0, 1):
            raise ValueError("nodelay 只能是 0 或 1")
        sock.nodelay = flag
        _tune(sock)
        return Result([], [])
    raise ValueError("ParamName 只能是 timeout 或 nodelay")


@operator(
    "通訊",
    [],
    [],
    [
        P("Socket", "連線"),
        P("ParamName", "timeout、nodelay、available 或 peer", "'timeout'", _PARAMS),
    ],
    [P("ParamValue", "現在的值")],
    "讀回 TCP 參數",
    "timeout 和 nodelay 是設定過的值。available 是對面已經送到、還沒讀的位元組數。peer 是對方或正在聽的位址。",
    "Vizion.get_socket_param (Socket, 'timeout', ParamValue)",
)
def get_socket_param(link, name, *, ctx):
    from ..objects import Result

    sock = _require(link)
    param = text(name, "ParamName").strip().strip("'\"").lower()
    if param == "timeout":
        return Result([], [sock.timeout])
    if param == "nodelay":
        return Result([], [sock.nodelay])
    if param == "peer":
        return Result([], [sock.peer])
    if param == "available":
        live = _ready(sock)
        return Result([], [_pending(live.sock)])
    raise ValueError("ParamName 只能是 timeout、nodelay、available 或 peer")
