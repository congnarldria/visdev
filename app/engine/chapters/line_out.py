"""One ASCII line for a production result, on TCP or a serial port.

The line is ``VISDEV,<OK or NG>,<ok count>,<ng count>,<file>\\n``.
Chinese batch words map to OK and NG. The bytes are the same on Windows COM
and on TCP. This does not open the port; the caller already has a handle.
"""

from __future__ import annotations

from ..operators import P, operator, text, whole
from .serial_ops import SerialPort, write_serial
from .socket_ops import SocketLink, send_data

_VERDICTS = {
    "OK": "OK",
    "NG": "NG",
    "合格": "OK",
    "不合格": "NG",
}


def format_line_result(judge, ok_count, ng_count, file_name) -> str:
    raw = text(judge, "Judge").strip().strip("'\"")
    verdict = _VERDICTS.get(raw) or _VERDICTS.get(raw.upper())
    if verdict is None:
        raise ValueError("判定只能是 OK、NG、合格或不合格")
    ok = whole(ok_count, "OkCount")
    ng = whole(ng_count, "NgCount")
    if ok < 0 or ng < 0:
        raise ValueError("合格和不合格張數不能是負的")
    name = text(file_name, "FileName").strip().strip("'\"")
    if any(ch in name for ch in ",\r\n") or not name.isascii():
        raise ValueError("檔名只能用英文、數字和 . _ -，中間不能有逗號")
    return f"VISDEV,{verdict},{ok},{ng},{name}\n"


@operator(
    "通訊",
    [],
    [],
    [
        P("Link", "TCP 連線或序列埠"),
        P("Judge", "OK、NG、合格或不合格", "'NG'"),
        P("OkCount", "目前合格張數", "0"),
        P("NgCount", "目前不合格張數", "0"),
        P("FileName", "這一張的檔名", "''"),
    ],
    [],
    "送出一筆產線結果",
    "寫一行 VISDEV,OK 或 NG,合格張數,不合格張數,檔名。結尾是換行。"
    "合格會寫成 OK，不合格會寫成 NG。TCP 和序列埠用同一行。"
    "Windows 的 COM 仍用原來的開啟方式，這支不算子不自己開埠。"
    "loopback 不接真的線，送出的字可以再讀回來。",
    "Vizion.send_line_result (Link, 'NG', 0, 1, 'ng_01.png')",
)
def send_line_result(link, judge, ok_count, ng_count, file_name, *, ctx):
    from ..objects import Result

    payload = format_line_result(judge, ok_count, ng_count, file_name)
    if isinstance(link, SocketLink):
        send_data(link, payload, ctx=ctx)
    elif isinstance(link, SerialPort):
        write_serial(link, payload, ctx=ctx)
    else:
        raise TypeError("Link 必須是 TCP 連線或序列埠")
    return Result([], [])
