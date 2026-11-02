"""Compile the native matcher. Windows writes visdev_match.pyd, other systems write visdev_match.so."""

from __future__ import annotations

import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NATIVE = Path(__file__).resolve().parent
INCLUDE = Path(sysconfig.get_path("include"))
SOURCE = NATIVE / "match_core.c"

VCVARS = (
    Path(r"C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat"),
    Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"),
)


def _windows() -> None:
    out = ROOT / "app" / "engine" / "visdev_match.pyd"
    libs = Path(sys.base_prefix) / "libs"
    vcvars = next((path for path in VCVARS if path.is_file()), None)
    if vcvars is None:
        raise SystemExit("找不到 vcvars64.bat，無法編譯 visdev_match.pyd")
    if not (INCLUDE / "Python.h").is_file():
        raise SystemExit(f"找不到 Python.h：{INCLUDE}")
    command = (
        f'call "{vcvars}" && cl /nologo /O2 /Oi /Ot /GL /arch:AVX2 /MD /DNDEBUG /LD '
        f'/I "{INCLUDE}" "{SOURCE}" /link /LIBPATH:"{libs}" python310.lib '
        f'/OUT:"{out}"'
    )
    print(command)
    subprocess.check_call(command, cwd=NATIVE, shell=True)
    if not out.is_file():
        raise SystemExit(f"沒有產生 {out}")
    print(out, out.stat().st_size)


def _unix() -> None:
    out = ROOT / "app" / "engine" / "visdev_match.so"
    if not (INCLUDE / "Python.h").is_file():
        raise SystemExit(f"找不到 Python.h：{INCLUDE}")
    command = [
        "gcc", "-shared", "-fPIC", "-O2", "-mavx2", "-mfma",
        f"-I{INCLUDE}", str(SOURCE), "-o", str(out), "-pthread", "-lm",
    ]
    print(" ".join(command))
    subprocess.check_call(command, cwd=NATIVE)
    if not out.is_file():
        raise SystemExit(f"沒有產生 {out}")
    print(out, out.stat().st_size)


def main() -> None:
    if sys.platform == "win32":
        _windows()
    else:
        _unix()


if __name__ == "__main__":
    main()
