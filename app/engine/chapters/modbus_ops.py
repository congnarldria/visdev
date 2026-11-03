"""Modbus TCP holding registers.

Function 6 writes one register. Function 3 reads one back. Both use the
Python socket library, the same on Windows and Linux. Neither opens a COM port.
"""

from __future__ import annotations

import socket
import struct

from ..operators import P, operator, text, whole


def _register(value, name: str) -> int:
    number = whole(value, name)
    if number < 0 or number > 65535:
        raise ValueError(f"{name} 必須在 0 到 65535")
    return number


def _recv_exact(link: socket.socket, count: int) -> bytes:
    reply = bytearray()
    while len(reply) < count:
        chunk = link.recv(count - len(reply))
        if not chunk:
            break
        reply.extend(chunk)
    return bytes(reply)


def _exchange(host: str, port: int, pdu: bytes) -> bytes:
    packet = struct.pack(">HHHB", 1, 0, len(pdu) + 1, 1) + pdu
    try:
        link = socket.create_connection((host, port), timeout=2)
    except OSError as exc:
        raise RuntimeError(f"連不上 Modbus {host}:{port}") from exc
    try:
        link.settimeout(2)
        link.sendall(packet)
        head = _recv_exact(link, 7)
        if len(head) < 7:
            raise RuntimeError(f"Modbus {host}:{port} 沒有回應")
        length = struct.unpack(">H", head[4:6])[0]
        if length < 2 or length > 16:
            raise RuntimeError(f"Modbus {host}:{port} 沒有回應")
        rest = _recv_exact(link, length - 1)
    except socket.timeout as exc:
        raise RuntimeError(f"Modbus {host}:{port} 沒有回應") from exc
    except OSError as exc:
        raise RuntimeError(f"Modbus {host}:{port} 沒有回應") from exc
    finally:
        link.close()
    if len(head) < 7 or len(rest) < max(0, length - 1):
        raise RuntimeError(f"Modbus {host}:{port} 沒有回應")
    return head + rest


def write_holding_register(host: str, port: int, address: int, value: int) -> None:
    pdu = struct.pack(">BHH", 6, address, value)
    reply = _exchange(host, port, pdu)
    if len(reply) < 12 or reply[7] in (0x86, 0x83):
        raise RuntimeError(f"Modbus {host}:{port} 拒絕寫入位址 {address}")
    if reply[6:12] != struct.pack(">BBHH", 1, 6, address, value):
        raise RuntimeError(f"Modbus {host}:{port} 回應的位址或數值不對")


def _span(address: int, count: int) -> None:
    if count < 1 or count > 16:
        raise ValueError("一次要讀寫 1 到 16 個暫存器")
    if address + count - 1 > 65535:
        raise ValueError("位址超出範圍")


def read_holding_registers(host: str, port: int, address: int, count: int) -> list[int]:
    _span(address, count)
    reply = _exchange(host, port, struct.pack(">BHH", 3, address, count))
    if len(reply) < 9 or reply[7] in (0x83, 0x86):
        raise RuntimeError(f"Modbus {host}:{port} 拒絕讀取位址 {address}")
    if reply[7] != 3 or reply[8] != count * 2 or len(reply) < 9 + count * 2:
        raise RuntimeError(f"Modbus {host}:{port} 回應的位址或數值不對")
    return list(struct.unpack(">" + "H" * count, reply[9:9 + count * 2]))


def write_holding_registers(host: str, port: int, address: int, values: list[int]) -> None:
    count = len(values)
    _span(address, count)
    pdu = struct.pack(">BHHB" + "H" * count, 16, address, count, count * 2, *values)
    reply = _exchange(host, port, pdu)
    if len(reply) < 12 or reply[7] in (0x90, 0x86, 0x83):
        raise RuntimeError(f"Modbus {host}:{port} 拒絕寫入位址 {address}")
    if reply[6:12] != struct.pack(">BBHH", 1, 16, address, count):
        raise RuntimeError(f"Modbus {host}:{port} 回應的位址或數值不對")


def read_holding_register(host: str, port: int, address: int) -> int:
    reply = _exchange(host, port, struct.pack(">BHH", 3, address, 1))
    if len(reply) < 9 or reply[7] in (0x83, 0x86):
        raise RuntimeError(f"Modbus {host}:{port} 拒絕讀取位址 {address}")
    if reply[7] != 3 or reply[8] != 2 or len(reply) < 11:
        raise RuntimeError(f"Modbus {host}:{port} 回應的位址或數值不對")
    return struct.unpack(">H", reply[9:11])[0]


_JUDGE_CODE = {"OK": 1, "NG": 2, "合格": 1, "不合格": 2}


def judge_code(judge) -> int:
    raw = text(judge, "Judge").strip().strip("'\"")
    code = _JUDGE_CODE.get(raw) or _JUDGE_CODE.get(raw.upper())
    if code is None:
        raise ValueError("判定只能是 OK、NG、合格或不合格")
    return code


@operator(
    "通訊",
    [],
    [],
    [
        P("Host", "Modbus TCP 主機", "'127.0.0.1'"),
        P("Port", "埠號，通常是 502", "502"),
        P("Address", "暫存器位址，0 到 65535", "0"),
        P("Value", "要寫的數，0 到 65535", "0"),
    ],
    [],
    "寫一個 Modbus 暫存器",
    "用 Modbus TCP 功能碼 6 寫一個 16 位元暫存器。不開序列埠，Windows 的 COM 仍用原來的方式。"
    "連不上或對方拒絕時會說出主機、埠號和位址。",
    "Vizion.write_modbus_register ('127.0.0.1', 502, 16, 2)",
)
def write_modbus_register(host, port, address, value, *, ctx):
    from ..objects import Result

    name = text(host, "Host").strip().strip("'\"")
    if not name:
        raise ValueError("Host 不能是空的")
    number = whole(port, "Port")
    if number < 1 or number > 65535:
        raise ValueError("Port 必須在 1 到 65535")
    write_holding_register(name, number, _register(address, "Address"), _register(value, "Value"))
    return Result([], [])


@operator(
    "通訊",
    [],
    [],
    [
        P("Host", "Modbus TCP 主機", "'127.0.0.1'"),
        P("Port", "埠號，通常是 502", "502"),
        P("Address", "暫存器位址，0 到 65535", "0"),
    ],
    [P("Value", "讀回的數")],
    "讀一個 Modbus 暫存器",
    "用 Modbus TCP 功能碼 3 讀一個暫存器。讀回的是 0 到 65535。"
    "不開序列埠，Windows 的 COM 仍用原來的方式。連不上或對方拒絕時會說出主機、埠號和位址。",
    "Vizion.read_modbus_register ('127.0.0.1', 502, 16, Register)",
)
def read_modbus_register(host, port, address, *, ctx):
    from ..objects import Result

    name = text(host, "Host").strip().strip("'\"")
    if not name:
        raise ValueError("Host 不能是空的")
    number = whole(port, "Port")
    if number < 1 or number > 65535:
        raise ValueError("Port 必須在 1 到 65535")
    value = read_holding_register(name, number, _register(address, "Address"))
    return Result([], [value])


def _host_port(host, port) -> tuple[str, int]:
    name = text(host, "Host").strip().strip("'\"")
    if not name:
        raise ValueError("Host 不能是空的")
    number = whole(port, "Port")
    if number < 1 or number > 65535:
        raise ValueError("Port 必須在 1 到 65535")
    return name, number


def _value_list(value) -> list[int]:
    items = list(value) if isinstance(value, (list, tuple)) else [value]
    if not items:
        raise ValueError("一次要讀寫 1 到 16 個暫存器")
    return [_register(item, "Values") for item in items]


@operator(
    "通訊",
    [],
    [],
    [
        P("Host", "Modbus TCP 主機", "'127.0.0.1'"),
        P("Port", "埠號，通常是 502", "502"),
        P("Address", "起始位址", "0"),
        P("Count", "個數，1 到 16", "2"),
    ],
    [P("Values", "讀回的數")],
    "讀連續的 Modbus 暫存器",
    "用功能碼 3 從起始位址連著讀，最多 16 個。不開序列埠。",
    "Vizion.read_modbus_registers ('127.0.0.1', 502, 16, 3, Values)",
)
def read_modbus_registers(host, port, address, count, *, ctx):
    from ..objects import Result

    name, number = _host_port(host, port)
    values = read_holding_registers(name, number, _register(address, "Address"), whole(count, "Count"))
    return Result([], [values])


@operator(
    "通訊",
    [],
    [],
    [
        P("Host", "Modbus TCP 主機", "'127.0.0.1'"),
        P("Port", "埠號，通常是 502", "502"),
        P("Address", "起始位址", "0"),
        P("Values", "要寫的數", "[1, 3, 2]"),
    ],
    [],
    "寫連續的 Modbus 暫存器",
    "用功能碼 16 從起始位址連著寫，最多 16 個。不開序列埠，也不改一次寫一個的那支算子。",
    "Vizion.write_modbus_registers ('127.0.0.1', 502, 16, [2, 3, 1])",
)
def write_modbus_registers(host, port, address, values, *, ctx):
    from ..objects import Result

    name, number = _host_port(host, port)
    write_holding_registers(name, number, _register(address, "Address"), _value_list(values))
    return Result([], [])


@operator(
    "通訊",
    [],
    [],
    [
        P("Host", "Modbus TCP 主機", "'127.0.0.1'"),
        P("Port", "埠號，通常是 502", "502"),
        P("Address", "暫存器位址，0 到 65535", "0"),
        P("Judge", "OK、NG、合格或不合格", "'NG'"),
    ],
    [P("Value", "讀回的數")],
    "把判定寫進暫存器並讀回",
    "合格或 OK 寫 1，不合格或 NG 寫 2，然後讀回同一個位址。"
    "讀回的數不一樣就失敗。不開序列埠，也不改產線 OK、NG 那一行的欄位。",
    "Vizion.write_modbus_judge ('127.0.0.1', 502, 16, '不合格', Register)",
)
def write_modbus_judge(host, port, address, judge, *, ctx):
    from ..objects import Result

    name = text(host, "Host").strip().strip("'\"")
    if not name:
        raise ValueError("Host 不能是空的")
    number = whole(port, "Port")
    if number < 1 or number > 65535:
        raise ValueError("Port 必須在 1 到 65535")
    slot = _register(address, "Address")
    code = judge_code(judge)
    write_holding_register(name, number, slot, code)
    read_back = read_holding_register(name, number, slot)
    if read_back != code:
        raise RuntimeError(f"暫存器 {slot} 讀回 {read_back}，和寫入的 {code} 不一樣")
    return Result([], [read_back])
