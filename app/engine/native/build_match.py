"""Compile app/engine/visdev_match.pyd with MSVC AVX2."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NATIVE = Path(__file__).resolve().parent
OUT = ROOT / "app" / "engine" / "visdev_match.pyd"
INCLUDE = Path(sys.base_prefix) / "include"
LIBS = Path(sys.base_prefix) / "libs"

VCVARS = (
    Path(r"C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat"),
    Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"),
)


def main() -> None:
    vcvars = next((path for path in VCVARS if path.is_file()), None)
    if vcvars is None:
        raise SystemExit("找不到 vcvars64.bat，無法編譯 visdev_match.pyd")
    if not (INCLUDE / "Python.h").is_file():
        raise SystemExit(f"找不到 Python.h：{INCLUDE}")
    source = NATIVE / "match_core.c"
    command = (
        f'call "{vcvars}" && cl /nologo /O2 /Oi /Ot /GL /arch:AVX2 /MD /DNDEBUG /LD '
        f'/I "{INCLUDE}" "{source}" /link /LIBPATH:"{LIBS}" python310.lib '
        f'/OUT:"{OUT}"'
    )
    print(command)
    subprocess.check_call(command, cwd=NATIVE, shell=True)
    if not OUT.is_file():
        raise SystemExit(f"沒有產生 {OUT}")
    print(OUT, OUT.stat().st_size)


if __name__ == "__main__":
    main()
