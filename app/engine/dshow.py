"""One frame from a USB camera through DirectShow.

OpenCV's video backend on this build cannot capture by index. qedit's sample
grabber is already on Windows 7 and Windows 10, so no extra package is needed.
"""

from __future__ import annotations

import ctypes
import time
import uuid
from ctypes import wintypes

import numpy as np

_CLSCTX_INPROC = 1
_COINIT_APARTMENT = 2


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class _MediaType(ctypes.Structure):
    _fields_ = [
        ("majortype", _GUID),
        ("subtype", _GUID),
        ("fixed", wintypes.BOOL),
        ("temporal", wintypes.BOOL),
        ("sample_size", wintypes.ULONG),
        ("formattype", _GUID),
        ("unk", ctypes.c_void_p),
        ("format_size", wintypes.ULONG),
        ("format_ptr", ctypes.c_void_p),
    ]


class _MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("x", wintypes.LONG),
        ("y", wintypes.LONG),
    ]


def _guid(text: str) -> _GUID:
    value = uuid.UUID(text)
    data = _GUID()
    data.Data1, data.Data2, data.Data3 = value.fields[:3]
    for index, byte in enumerate(value.bytes[8:]):
        data.Data4[index] = byte
    return data


_VIDEO = _guid("73646976-0000-0010-8000-00aa00389b71")
_RGB24 = _guid("e436eb7d-524f-11ce-9f53-0020af0ba770")
_VIDEOINFO = _guid("05589f80-c356-11ce-bf01-00aa0055595a")
_CAPTURE = _guid("fb6c4281-0353-11d1-905f-0000c0cc16ba")
_NULL = _guid("00000000-0000-0000-0000-000000000000")

_ole32 = ctypes.WinDLL("ole32", use_last_error=True)
_user32 = ctypes.WinDLL("user32", use_last_error=True)
try:
    ctypes.WinDLL("qedit")
except OSError:
    pass

_ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
_ole32.CoInitializeEx.restype = ctypes.c_long
_ole32.CoUninitialize.argtypes = []
_ole32.CoUninitialize.restype = None
_ole32.CoCreateInstance.argtypes = [
    ctypes.POINTER(_GUID), ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p),
]
_ole32.CoCreateInstance.restype = ctypes.c_long
_ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
_ole32.CoTaskMemFree.restype = None
_user32.PeekMessageW.argtypes = [ctypes.POINTER(_MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT]
_user32.PeekMessageW.restype = wintypes.BOOL
_user32.TranslateMessage.argtypes = [ctypes.POINTER(_MSG)]
_user32.TranslateMessage.restype = wintypes.BOOL
_user32.DispatchMessageW.argtypes = [ctypes.POINTER(_MSG)]
_user32.DispatchMessageW.restype = ctypes.c_ssize_t


def _method(ptr, index, restype, argtypes):
    table = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    proto = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    return proto(table[index])


def _call(ptr, index, restype, argtypes, *args):
    return _method(ptr, index, restype, argtypes)(ptr, *args)


def _release(ptr) -> None:
    if ptr:
        _call(ptr, 2, ctypes.c_ulong, [])


def _create(clsid: str, iid: str):
    out = ctypes.c_void_p()
    class_id = _guid(clsid)
    face_id = _guid(iid)
    hr = _ole32.CoCreateInstance(ctypes.byref(class_id), None, _CLSCTX_INPROC, ctypes.byref(face_id), ctypes.byref(out))
    if hr >= 0 and out.value:
        return out
    if hr != -2147221164:
        raise OSError(f"建立 DirectShow 元件失敗 {hr}")
    factory_id = _guid("00000001-0000-0000-c000-000000000046")
    for dll in ("qedit.dll", "quartz.dll", "devenum.dll"):
        try:
            library = ctypes.WinDLL(dll)
        except OSError:
            continue
        get_class = library.DllGetClassObject
        get_class.argtypes = [ctypes.POINTER(_GUID), ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
        get_class.restype = ctypes.c_long
        factory = ctypes.c_void_p()
        made = get_class(ctypes.byref(class_id), ctypes.byref(factory_id), ctypes.byref(factory))
        if made < 0 or not factory.value:
            continue
        out = ctypes.c_void_p()
        made = _call(
            factory, 3, ctypes.c_long,
            [ctypes.c_void_p, ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)],
            None, ctypes.byref(face_id), ctypes.byref(out),
        )
        _release(factory)
        if made >= 0 and out.value:
            return out
    raise OSError(f"建立 DirectShow 元件失敗 {clsid}")


def _query(ptr, iid: str):
    out = ctypes.c_void_p()
    hr = _call(ptr, 0, ctypes.c_long, [ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)], ctypes.byref(_guid(iid)), ctypes.byref(out))
    if hr < 0 or not out.value:
        raise OSError(f"介面失敗 {hr}")
    return out


def _device(index: int):
    enumerator = _create("62be8d8a-8c4f-11d0-8d52-00a0c91b0e0c", "29840822-5b84-11d0-bd3b-00a0c911ce86")
    try:
        listing = ctypes.c_void_p()
        hr = _call(
            enumerator, 3, ctypes.c_long,
            [ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p), wintypes.DWORD],
            ctypes.byref(_guid("860bb310-5d01-11d0-bd3b-00a0c911ce86")), ctypes.byref(listing), 0,
        )
        if hr < 0 or not listing.value:
            raise RuntimeError(f"打不開相機 {index}")
        try:
            chosen = None
            for cursor in range(index + 1):
                moniker = ctypes.c_void_p()
                fetched = wintypes.ULONG(0)
                hr = _call(
                    listing, 3, ctypes.c_long,
                    [wintypes.ULONG, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.ULONG)],
                    1, ctypes.byref(moniker), ctypes.byref(fetched),
                )
                if hr < 0 or fetched.value != 1 or not moniker.value:
                    raise RuntimeError(f"打不開相機 {index}")
                if cursor == index:
                    chosen = moniker
                else:
                    _release(moniker)
            source = ctypes.c_void_p()
            hr = _call(
                chosen, 8, ctypes.c_long,
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)],
                None, None, ctypes.byref(_guid("56a86895-0ad4-11ce-b03a-0020af0ba770")), ctypes.byref(source),
            )
            _release(chosen)
            if hr < 0 or not source.value:
                raise RuntimeError(f"打不開相機 {index}")
            return source
        finally:
            _release(listing)
    finally:
        _release(enumerator)


def _media(subtype: _GUID) -> _MediaType:
    kind = _MediaType()
    kind.majortype = _VIDEO
    kind.subtype = subtype
    kind.fixed = True
    kind.formattype = _VIDEOINFO
    return kind


def _pump() -> None:
    message = _MSG()
    while _user32.PeekMessageW(ctypes.byref(message), None, 0, 0, 1):
        _user32.TranslateMessage(ctypes.byref(message))
        _user32.DispatchMessageW(ctypes.byref(message))


def _buffer(grabber) -> bytes | None:
    size = ctypes.c_long(0)
    hr = _call(grabber, 7, ctypes.c_long, [ctypes.POINTER(ctypes.c_long), ctypes.c_void_p], ctypes.byref(size), None)
    if hr < 0 or size.value <= 0:
        return None
    raw = ctypes.create_string_buffer(size.value)
    hr = _call(
        grabber, 7, ctypes.c_long,
        [ctypes.POINTER(ctypes.c_long), ctypes.c_void_p],
        ctypes.byref(size), ctypes.cast(raw, ctypes.c_void_p),
    )
    if hr < 0:
        return None
    return raw.raw[: size.value]


def _shape(grabber):
    kind = _MediaType()
    hr = _call(grabber, 5, ctypes.c_long, [ctypes.POINTER(_MediaType)], ctypes.byref(kind))
    if hr < 0 or not kind.format_ptr or kind.format_size < 64:
        return 0, 0, 0
    header = ctypes.string_at(kind.format_ptr, kind.format_size)
    if kind.format_ptr:
        _ole32.CoTaskMemFree(kind.format_ptr)
    if kind.unk:
        _release(ctypes.c_void_p(kind.unk))
    width, height, _planes, bits = __import__("struct").unpack_from("<llHH", header, 52)
    return int(width), int(abs(height)), int(bits)


def _directshow(index: int) -> np.ndarray:
    if index < 0:
        raise RuntimeError(f"打不開相機 {index}")
    started = _ole32.CoInitializeEx(None, _COINIT_APARTMENT)
    graph = builder = grabber = grabber_base = sink = control = source = None
    try:
        source = _device(index)
        graph = _create("e436ebb3-524f-11ce-9f53-0020af0ba770", "56a868a9-0ad4-11ce-b03a-0020af0ba770")
        builder = _create("bf87b6e1-8c27-11d0-b3f0-00aa003761c5", "93e5a4e0-2d50-11d2-abfa-00a0c9c6e38d")
        grabber = _create("c1f400a0-3f08-11d3-9f0b-006008039e37", "6b652fff-11fe-4fce-92ad-0266b5d7c78f")
        grabber_base = _query(grabber, "56a86895-0ad4-11ce-b03a-0020af0ba770")
        sink = _create("c1f400a4-3f08-11d3-9f0b-006008039e37", "56a86895-0ad4-11ce-b03a-0020af0ba770")
        hr = _call(builder, 3, ctypes.c_long, [ctypes.c_void_p], graph)
        if hr < 0:
            raise RuntimeError(f"打不開相機 {index}")
        for name, pin in (("src", source), ("grab", grabber_base), ("sink", sink)):
            hr = _call(graph, 3, ctypes.c_long, [ctypes.c_void_p, wintypes.LPCWSTR], pin, name)
            if hr < 0:
                raise RuntimeError(f"打不開相機 {index}")
        rendered = -1
        for subtype in (_RGB24, _NULL):
            kind = _media(subtype)
            _call(grabber, 4, ctypes.c_long, [ctypes.POINTER(_MediaType)], ctypes.byref(kind))
            _call(grabber, 6, ctypes.c_long, [wintypes.BOOL], True)
            _call(grabber, 3, ctypes.c_long, [wintypes.BOOL], True)
            rendered = _call(
                builder, 7, ctypes.c_long,
                [ctypes.POINTER(_GUID), ctypes.POINTER(_GUID), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
                ctypes.byref(_CAPTURE), ctypes.byref(_VIDEO), source, grabber_base, sink,
            )
            if rendered >= 0:
                break
        if rendered < 0:
            raise RuntimeError(f"打不開相機 {index}")
        control = _query(graph, "56a868b1-0ad4-11ce-b03a-0020af0ba770")
        hr = _call(control, 7, ctypes.c_long, [])
        if hr < 0:
            raise RuntimeError(f"打不開相機 {index}")
        payload = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and payload is None:
            _pump()
            payload = _buffer(grabber)
            if payload is None:
                time.sleep(0.05)
        _call(control, 9, ctypes.c_long, [])
        if not payload:
            raise RuntimeError(f"打不開相機 {index}")
        width, height, bits = _shape(grabber)
        if width <= 0 or height <= 0 or bits not in (24, 32):
            raise RuntimeError(f"打不開相機 {index}")
        channels = bits // 8
        stride = ((width * channels + 3) // 4) * 4
        if len(payload) < stride * height:
            raise RuntimeError(f"打不開相機 {index}")
        rows = np.frombuffer(payload, np.uint8, count=stride * height).reshape(height, stride)
        frame = rows[:, : width * channels].reshape(height, width, channels)
        if channels == 4:
            frame = frame[:, :, :3]
        return np.ascontiguousarray(frame[::-1, :, ::-1])
    except OSError as exc:
        raise RuntimeError(f"打不開相機 {index}") from exc
    finally:
        for item in (control, sink, grabber_base, grabber, source, builder, graph):
            _release(item)
        if started >= 0:
            _ole32.CoUninitialize()


def _media_foundation(index: int) -> np.ndarray:
    """Win10 often has no registered device enumerator. Media Foundation is the other Windows camera list."""
    if index < 0:
        raise RuntimeError(f"打不開相機 {index}")
    started = _ole32.CoInitializeEx(None, _COINIT_APARTMENT)
    mfplat = ctypes.WinDLL("mfplat")
    mf = ctypes.WinDLL("mf")
    mfplat.MFStartup.argtypes = [ctypes.c_ulong, ctypes.c_ulong]
    mfplat.MFStartup.restype = ctypes.c_long
    mfplat.MFShutdown.argtypes = []
    mfplat.MFShutdown.restype = ctypes.c_long
    mfplat.MFCreateAttributes.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    mfplat.MFCreateAttributes.restype = ctypes.c_long
    mf.MFEnumDeviceSources.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint)]
    mf.MFEnumDeviceSources.restype = ctypes.c_long
    attrs = ctypes.c_void_p()
    listing = ctypes.c_void_p()
    try:
        if mfplat.MFStartup(0x00020070, 0) < 0:
            raise RuntimeError(f"打不開相機 {index}")
        if mfplat.MFCreateAttributes(ctypes.byref(attrs), 1) < 0 or not attrs.value:
            raise RuntimeError(f"打不開相機 {index}")
        hr = _call(
            attrs, 24, ctypes.c_long,
            [ctypes.POINTER(_GUID), ctypes.POINTER(_GUID)],
            ctypes.byref(_guid("c60ac5fe-252a-478f-a0ef-bc8fa5f7cad3")),
            ctypes.byref(_guid("8ac3587a-4ae7-42d8-99e0-0a6013eef90f")),
        )
        if hr < 0:
            raise RuntimeError(f"打不開相機 {index}")
        count = ctypes.c_uint(0)
        hr = mf.MFEnumDeviceSources(attrs, ctypes.byref(listing), ctypes.byref(count))
        if hr < 0 or count.value <= index or not listing.value:
            raise RuntimeError(f"打不開相機 {index}")
        slots = ctypes.cast(listing, ctypes.POINTER(ctypes.c_void_p))
        chosen = ctypes.c_void_p(slots[index])
        source = ctypes.c_void_p()
        hr = _call(
            chosen, 33, ctypes.c_long,
            [ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)],
            ctypes.byref(_guid("279a808d-aec7-40c8-9c6b-a6b492c78a66")), ctypes.byref(source),
        )
        if hr < 0 or not source.value:
            raise RuntimeError(f"打不開相機 {index}")
        reader_lib = ctypes.WinDLL("mfreadwrite")
        reader_lib.MFCreateSourceReaderFromMediaSource.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
        ]
        reader_lib.MFCreateSourceReaderFromMediaSource.restype = ctypes.c_long
        reader = ctypes.c_void_p()
        hr = reader_lib.MFCreateSourceReaderFromMediaSource(source, None, ctypes.byref(reader))
        _release(source)
        if hr < 0 or not reader.value:
            raise RuntimeError(f"打不開相機 {index}")
        try:
            return _read_rgb(reader, index)
        finally:
            _release(reader)
    finally:
        found = int(count.value) if "count" in locals() else 0
        if listing.value and found:
            slots = ctypes.cast(listing, ctypes.POINTER(ctypes.c_void_p))
            for slot in range(found):
                _release(ctypes.c_void_p(slots[slot]))
            _ole32.CoTaskMemFree(listing)
        _release(attrs)
        mfplat.MFShutdown()
        if started >= 0:
            _ole32.CoUninitialize()


def _read_rgb(reader, index: int) -> np.ndarray:
    sample = ctypes.c_void_p()
    flags = ctypes.c_ulong(0)
    for _ in range(8):
        sample = ctypes.c_void_p()
        hr = _call(
            reader, 9, ctypes.c_long,
            [
                ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong),
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
            ],
            0xFFFFFFFC, 0, None, ctypes.byref(flags), None, ctypes.byref(sample),
        )
        if hr >= 0 and sample.value:
            break
    else:
        raise RuntimeError(f"打不開相機 {index}")
    try:
        buffer = ctypes.c_void_p()
        hr = _call(sample, 41, ctypes.c_long, [ctypes.POINTER(ctypes.c_void_p)], ctypes.byref(buffer))
        if hr < 0 or not buffer.value:
            raise RuntimeError(f"打不開相機 {index}")
        try:
            data = ctypes.c_void_p()
            current = ctypes.c_ulong(0)
            hr = _call(
                buffer, 3, ctypes.c_long,
                [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)],
                ctypes.byref(data), None, ctypes.byref(current),
            )
            if hr < 0 or not data.value or current.value < 4:
                raise RuntimeError(f"打不開相機 {index}")
            raw = ctypes.string_at(data.value, current.value)
            _call(buffer, 4, ctypes.c_long, [ctypes.c_void_p], data)
        finally:
            _release(buffer)
    finally:
        _release(sample)
    side = int(current.value ** 0.5)
    if side * side * 3 == current.value:
        frame = np.frombuffer(raw, np.uint8).reshape(side, side, 3)
        return np.ascontiguousarray(frame[:, :, ::-1])
    if side * side * 4 == current.value:
        frame = np.frombuffer(raw, np.uint8).reshape(side, side, 4)
        return np.ascontiguousarray(frame[:, :, 2::-1])
    raise RuntimeError(f"打不開相機 {index}")


def grab_bgr(index: int) -> np.ndarray:
    try:
        return _directshow(index)
    except RuntimeError:
        return _media_foundation(index)


def grab_frame(index: int):
    """BGR image, or a clear open failure."""
    return grab_bgr(index)
