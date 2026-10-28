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
        return self._open_tool("次像素二值化", url, 460, 720, (380, 520))

    def open_feature_window(self, url: str) -> bool:
        return self._open_tool("區域特徵", url, 460, 720, (360, 480))

    def open_threshold_window(self, url: str) -> bool:
        return self._open_tool("二值化", url, 920, 720, (640, 520))


def main() -> None:
    port = free_port()
    errors: list[str] = []
    threading.Thread(target=serve, args=(port, errors), daemon=True, name="visdev-server").start()
    wait_ready(port, errors)
    import webview

    window = webview.create_window(
        "VisDev 視覺開發環境",
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
        log = ROOT / "visdev.log"
        log.write_text(traceback.format_exc(), encoding="utf-8")
        raise
