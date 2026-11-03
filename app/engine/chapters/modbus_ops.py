"""Modbus TCP, one holding-register write.

Function 6 writes a single 16-bit register. It uses the Python socket
library, the same on Windows and Linux. It does not open a COM port.
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


def write_holding_register(host: str, port: int, address: int, value: int) -> None:
    payload = struct.pack(">HHHBBHH", 1, 0, 6, 1, 6, address, value)
    try:
        link = socket.create_connection((host, port), timeout=2)
    except OSError as exc:
        raise RuntimeError(f"連不上 Modbus {host}:{port}") from exc
    try:
        link.settimeout(2)
        link.sendall(payload)
        reply = bytearray()
        while len(reply) < 12:
            chunk = link.recv(12 - len(reply))
            if not chunk:
                break
            reply.extend(chunk)
    except socket.timeout as exc:
        raise RuntimeError(f"Modbus {host}:{port} 沒有回應") from exc
    except OSError as exc:
        raise RuntimeError(f"Modbus {host}:{port} 沒有回應") from exc
    finally:
        link.close()
    if len(reply) < 12 or reply[7] == 0x86:
        raise RuntimeError(f"Modbus {host}:{port} 拒絕寫入位址 {address}")
    if bytes(reply[6:12]) != payload[6:12]:
        raise RuntimeError(f"Modbus {host}:{port} 回應的位址或數值不對")


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
