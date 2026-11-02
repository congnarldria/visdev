# Improvement plan

The local snapshot on this branch is the source of truth. The notes below are what is still worth doing after that snapshot.

## Now

1. **Graphics window shows a stale picture.** `app/static/view.js` decides the picture is unchanged from width, height, image length, shape count, and colors. A moved region, a new label, or a different fill with those the same is skipped. The window also never draws text, and a click inside a closed contour does not select it the way the main window does.
2. **The operator registry does not import off Windows.** `serial_ops` checks the Windows `DCB` size and loads `kernel32` while the module is imported. `ctypes` `DWORD` is 8 bytes here, so the check raises and every operator disappears with it. Loopback serial does not need that. `dshow` loads `ole32` at import, and `open_framegrabber` only catches `RuntimeError`, so a missing DirectShow DLL never falls through to OpenCV. `get_socket_param(..., 'available')` asks `ws2_32` for the unread byte count.
3. **Reset leaves OS handles open.** `Session.reset` drops the variable table without closing files, sockets, serial ports, or cameras. A second run keeps the previous file locked.
4. **File operators have no tests.** Read, write, copy, list, and “reset closes the handle” are untested. Socket `available` is untested off Windows.

## Next

5. **Help text and the chapter export list.** `list_files` says it only lists one directory level, but `recursive` works. `chapters/__all__` omits `serial_ops` even though the module is imported.
6. **A short README.** There is no description of how to start the desktop app, how a program is written, or how to run tests.
7. **Run the engine suite** after the import fix, including the new file and socket cases and a small check of the graphics-window fingerprint.

## Later

8. **`file_ops` and `..`.** Relative paths can leave the project directory. Absolute paths should stay allowed; escaping with `..` is the part to decide.
9. **`wait_seconds` blocks the request thread** for up to 30 seconds.
10. **Native matcher.** `visdev_match.pyd` is Windows-only. Other systems use the Python matcher in `shape_model.py` / `ncc_model.py`. Building `match_core.c` for those systems is optional.
11. **Operators marked「這版」.** Calibration, several code symbologies, and some image types still refuse inputs on purpose. Extend them one family at a time, with a sample program each.
12. **Frontend checks.** The editor canvas has no automated test beyond the graphics-window fingerprint.
