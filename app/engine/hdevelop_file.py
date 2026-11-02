"""Open an HDevelop program (.hdev / .hdvp / .hdpl) as a VisDev program.

Password-protected procedures are stored by HALCON in a form we do not read.
A locked procedure body is an <xb> element, or a protection tag.
Those procedures are skipped. Nothing is decrypted.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

_CALL = re.compile(r"^(\s*)([A-Za-z_][\w.]*)(\s*\(.*)\s*$", re.DOTALL)
_SKIP = {
    "if", "else", "endif", "for", "endfor", "while", "endwhile",
    "try", "catch", "endtry", "procedure", "endprocedure", "endproc", "return",
}
_GROUPS = {
    "io": "iconic_in",
    "ii": "iconic_in",
    "oo": "iconic_out",
    "ic": "control_in",
    "ci": "control_in",
    "oc": "control_out",
    "co": "control_out",
}
_PROTECTED_TAGS = {"protection", "crypted", "encrypted", "password"}
_LOCKED = "* 這份 HALCON 程式是二進位，或整份被密碼鎖住，內容沒有打開。\n"


def convert_hdevelop(raw: bytes) -> tuple[str, list[str]]:
    text = _decode(raw)
    if text is None:
        return _LOCKED, ["整份程式"]
    stripped = text.lstrip("\ufeff \t\r\n")
    if stripped.startswith("<?xml") or stripped.startswith("<hdevelop"):
        return _from_xml(text)
    return _from_lines(text.splitlines(), [], set()), []


def _decode(raw: bytes) -> str | None:
    if not raw:
        return ""
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    if b"\x00" in raw[:400]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _tag(node: ET.Element) -> str:
    return node.tag.rsplit("}", 1)[-1].lower()


def _is_protected(proc: ET.Element) -> bool:
    blob = " ".join(f"{key} {value}" for key, value in proc.attrib.items()).lower()
    if any(mark in blob for mark in ("password", "protected", "crypted", "encrypted", "locked")):
        return True
    for node in proc.iter():
        if _tag(node) == "xb" or _tag(node) in _PROTECTED_TAGS:
            return True
        kind = str(node.attrib.get("type", "")).lower()
        if kind in ("protected", "crypted", "encrypted", "password"):
            return True
        text = (node.text or "").lower()
        if "password protected" in text or "password-protected" in text:
            return True
    return False


def _par_names(group: ET.Element) -> list[str]:
    names = []
    own = group.attrib.get("name")
    if own:
        names.append(own.strip())
    for node in group.iter():
        if node is group:
            continue
        if _tag(node) != "par":
            continue
        name = (node.attrib.get("name") or "").strip()
        if name:
            names.append(name)
    return names


def _interface(proc: ET.Element) -> str:
    slots = {"iconic_in": [], "iconic_out": [], "control_in": [], "control_out": []}
    interface = next((node for node in list(proc) if _tag(node) == "interface"), None)
    if interface is not None:
        for group in list(interface):
            key = _GROUPS.get(_tag(group))
            if key:
                slots[key].extend(_par_names(group))
    name = proc.attrib.get("name") or "procedure"
    return (
        f"procedure {name} ("
        f"{', '.join(slots['iconic_in'])} : "
        f"{', '.join(slots['iconic_out'])} : "
        f"{', '.join(slots['control_in'])} : "
        f"{', '.join(slots['control_out'])})"
    )


def _body_lines(proc: ET.Element) -> list[str]:
    body = next((node for node in list(proc) if _tag(node) == "body"), None)
    if body is None:
        return []
    lines = []
    for node in list(body):
        tag = _tag(node)
        text = "".join(node.itertext()).replace("\r\n", "\n").replace("\r", "\n").strip("\n").rstrip()
        if tag == "c":
            content = text.strip()
            if not content:
                lines.append("")
            elif content.startswith("*") or content.startswith("//"):
                lines.append(content if content.startswith("*") else "* " + content[2:].strip())
            else:
                lines.append("* " + content)
        elif tag == "l":
            lines.append(text)
    return lines


def _from_xml(text: str) -> tuple[str, list[str]]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return _LOCKED, ["整份程式"]
    procedures = [node for node in root.iter() if _tag(node) == "procedure"]
    if not procedures:
        return _LOCKED, ["整份程式"]
    names = set()
    skipped = []
    plain = []
    for proc in procedures:
        name = (proc.attrib.get("name") or "").strip() or "procedure"
        if _is_protected(proc):
            skipped.append(name)
            continue
        names.add(name.lower())
        plain.append(proc)
    source = _from_lines([], plain, names)
    if skipped:
        notes = "\n".join(f"* 程序 {name} 有密碼，沒有打開" for name in skipped)
        source = notes + ("\n" + source if source.strip() else "\n")
    return source, skipped


def _from_lines(lines: list[str], procedures: list[ET.Element], proc_names: set[str]) -> str:
    from .operators import REGISTRY

    ops = {key.lower(): op.name for key, op in REGISTRY.items()}
    main = []
    others = []
    if procedures:
        for proc in procedures:
            name = (proc.attrib.get("name") or "").strip()
            body = _rewrite(proc and _body_lines(proc), ops, proc_names)
            if name.lower() == "main":
                main.extend(body)
            else:
                others.append(_interface(proc))
                others.extend(body)
                others.append("endprocedure")
                others.append("")
    else:
        main.extend(_rewrite(lines, ops, proc_names))
    chunks = [line for line in main]
    if chunks and others:
        chunks.append("")
    chunks.extend(others)
    text = "\n".join(chunks).strip()
    return (text + "\n") if text else "\n"


def _rewrite(lines: list[str], ops: dict[str, str], proc_names: set[str]) -> list[str]:
    written = []
    for line in lines:
        written.extend(_prefix_line(line, ops, proc_names))
    return written


def _slash_comment(text: str) -> tuple[str, str]:
    quote = ""
    index = 0
    while index < len(text) - 1:
        ch = text[index]
        if quote:
            if ch == quote:
                quote = ""
            index += 1
            continue
        if ch in "'\"":
            quote = ch
            index += 1
            continue
        if text[index : index + 2] == "//":
            return text[:index].rstrip(), text[index + 2 :].strip()
        index += 1
    return text, ""


def _prefix_line(line: str, ops: dict[str, str], proc_names: set[str]) -> list[str]:
    code, comment = _slash_comment(line)
    out = []
    if comment:
        out.append("* " + comment)
    matched = _CALL.match(code)
    if matched:
        name = matched.group(2)
        low = name.lower()
        if low not in _SKIP and low not in proc_names and not low.startswith("vizion."):
            public = ops.get(low)
            if public:
                code = f"{matched.group(1)}{public}{matched.group(3)}"
    if code.strip() or not comment:
        out.append(code)
    return out
