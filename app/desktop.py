"""Open VisDev in its own window. Double-click the Desktop shortcut; no browser."""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ICON = ROOT / "app" / "static" / "magpie.ico"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# pythonw has no console. Libraries that write startup logs need a real stream.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(ROOT / "visdev.log", "a", encoding="utf-8")


def free_port() -> int:
    for port in range(8765, 8790):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("找不到可用的連接埠")


def serve(port: int, errors: list[str]) -> None:
    try:
        import uvicorn

        from app.main import app

        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
    except Exception:
        errors.append(traceback.format_exc())


def wait_ready(port: int, errors: list[str]) -> None:
    url = f"http://127.0.0.1:{port}/"
    last_error = "沒有回應"
    for _ in range(100):
        if errors:
            raise RuntimeError(errors[-1])
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
            time.sleep(0.2)
    detail = errors[-1] if errors else last_error
    raise RuntimeError(f"VisDev 沒有啟動：{detail}")


def bind_icon(window) -> None:
    def apply():
        native = getattr(window, "native", None)
        if native is None or not ICON.is_file():
            return

        def assign():
            from System.Drawing import Icon

            native.Icon = Icon(str(ICON))

        try:
            if native.InvokeRequired:
                from System import Action

                native.BeginInvoke(Action(assign))
            else:
                assign()
        except Exception:
            return

    window.events.shown += apply
    if window.events.shown.is_set():
        apply()


class DesktopApi:
    def _open_tool(self, title: str, url: str, width: int, height: int, min_size: tuple[int, int]) -> bool:
        import webview

        for window in list(webview.windows):
            if window.title == title:
                window.show()
                try:
                    window.resize(width, height)
                except Exception:
                    pass
                window.load_url(url)
                return True
        created = webview.create_window(
            title,
            url,
            width=width,
            height=height,
            min_size=min_size,
            background_color="#14171c",
            text_select=True,
            focus=True,
        )
        bind_icon(created)
        return True

    def open_bin_window(self, url: str) -> bool:
        return self._open_tool("次像素二值化", url, 920, 720, (640, 520))

    def open_feature_window(self, url: str) -> bool:
        return self._open_tool("區域特徵", url, 880, 720, (640, 480))

    def open_gfx_window(self, url: str) -> bool:
        return self._open_tool("圖形視窗", url, 960, 720, (640, 480))

    def open_threshold_window(self, url: str) -> bool:
        return self._open_tool("二值化", url, 920, 720, (640, 520))

    def _main_window(self):
        import webview

        for window in webview.windows:
            if str(window.title).startswith("VisDev"):
                return window
        return webview.windows[0] if webview.windows else None

    def _picked_path(self, chosen) -> str:
        if not chosen:
            return ""
        if isinstance(chosen, (list, tuple)):
            return str(chosen[0]) if chosen else ""
        return str(chosen)

    def _portable_path(self, path: str) -> str:
        file = Path(path).expanduser()
        try:
            resolved = file.resolve()
        except OSError:
            resolved = file
        for folder, prefix in ((ROOT / "samples", "samples"), (ROOT / "workspace", "workspace")):
            try:
                rel = resolved.relative_to(folder.resolve())
            except ValueError:
                continue
            if rel.parts:
                return f"{prefix}/{rel.as_posix()}"
        return resolved.as_posix()

    def pick_data_file(self, kind: str = "image") -> dict:
        import webview

        window = self._main_window()
        if window is None:
            return {"cancelled": True}
        filters = {
            "image": ("影像 (*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff)", "All files (*.*)"),
            "plate": ("標定描述檔 (*.cpd;*.descr)", "All files (*.*)"),
        }.get(str(kind), ("All files (*.*)",))
        chosen = window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False, file_types=filters)
        path = self._picked_path(chosen)
        if not path:
            return {"cancelled": True}
        return {"cancelled": False, "path": self._portable_path(path)}

    def open_program_dialog(self) -> dict:
        import webview

        window = self._main_window()
        if window is None:
            return {"cancelled": True}
        chosen = window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=False,
            file_types=("程式 (*.viz;*.hdv;*.hdev;*.hdvp;*.hdpl)", "All files (*.*)"),
        )
        path = self._picked_path(chosen)
        if not path:
            return {"cancelled": True}
        return self._read_program(path)

    def save_program_dialog(self, suggested: str, source: str) -> dict:
        import webview

        window = self._main_window()
        if window is None:
            return {"cancelled": True}
        chosen = window.create_file_dialog(
            webview.SAVE_DIALOG,
            save_filename=suggested or "program.viz",
            file_types=("VisDev (*.viz;*.hdv)", "All files (*.*)"),
        )
        path = self._picked_path(chosen)
        if not path:
            return {"cancelled": True}
        written = self._write_program(path, source)
        return {"cancelled": False, "path": written, "name": Path(written).name}

    def write_program_path(self, path: str, source: str) -> dict:
        written = self._write_program(path, source)
        return {"path": written, "name": Path(written).name}

    def _read_program(self, path: str) -> dict:
        file = Path(path)
        if not file.is_file():
            return {"cancelled": False, "error": "找不到程式"}
        raw = file.read_bytes()
        lower = file.name.lower()
        if lower.endswith((".hdev", ".hdvp", ".hdpl", ".dev", ".dvp")):
            from app.engine.hdevelop_file import convert_hdevelop

            source, skipped = convert_hdevelop(raw)
            return {
                "cancelled": False,
                "source": source,
                "name": f"{file.stem}.viz",
                "path": "",
                "skipped": skipped,
                "converted": True,
            }
        return {
            "cancelled": False,
            "source": raw.decode("utf-8"),
            "name": file.name,
            "path": str(file),
            "skipped": [],
            "converted": False,
        }

    def _write_program(self, path: str, source: str) -> str:
        file = Path(path)
        if file.suffix.lower() not in (".viz", ".hdv"):
            file = file.with_suffix(".viz")
        file.parent.mkdir(parents=True, exist_ok=True)
        text = str(source).replace("\r\n", "\n").replace("\r", "\n")
        file.write_text(text, encoding="utf-8")
        return str(file)


def report_startup_failure(text: str) -> None:
    log = ROOT / "visdev.log"
    try:
        log.write_text(text, encoding="utf-8")
    except OSError:
        pass
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, text[-1500:], "VisDev 沒有啟動", 0x00000010)
    except Exception:
        return


def main() -> None:
    port = free_port()
    errors: list[str] = []
    threading.Thread(target=serve, args=(port, errors), daemon=True, name="visdev-server").start()
    wait_ready(port, errors)
    import webview

    window = webview.create_window(
        "VisDev 1.1",
        f"http://127.0.0.1:{port}/",
        width=1440,
        height=900,
        min_size=(1100, 700),
        maximized=True,
        text_select=True,
        background_color="#14171c",
        js_api=DesktopApi(),
    )
    bind_icon(window)
    webview.start()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        report_startup_failure(traceback.format_exc())
        raise
