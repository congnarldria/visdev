"""把只含讀圖、二值化、連通的程式匯出成 C#。"""

from __future__ import annotations

import re

from .evalexpr import ExprError, eval_expr
from .operators import scalar, whole
from .parser import Assign, Blank, Call, Comment, Program, parse_program

_SUPPORTED = {"read_image", "threshold", "connection", "count_obj", "area_center", "dev_display"}
_KEYWORDS = {
    "abstract", "as", "base", "bool", "break", "byte", "case", "catch", "char", "checked",
    "class", "const", "continue", "decimal", "default", "delegate", "do", "double", "else",
    "enum", "event", "explicit", "extern", "false", "finally", "fixed", "float", "for",
    "foreach", "goto", "if", "implicit", "in", "int", "interface", "internal", "is", "lock",
    "long", "namespace", "new", "null", "object", "operator", "out", "override", "params",
    "private", "protected", "public", "readonly", "ref", "return", "sbyte", "sealed",
    "short", "sizeof", "stackalloc", "static", "string", "struct", "switch", "this",
    "throw", "true", "try", "typeof", "uint", "ulong", "unchecked", "unsafe", "ushort",
    "using", "var", "virtual", "void", "volatile", "while",
}


class ExportError(Exception):
    def __init__(self, message: str, line: int | None = None):
        self.line = line
        if line:
            message = f"第 {line} 行：{message}"
        super().__init__(message)


def csharp_class_name(name: str) -> str:
    stem = name.strip()
    lowered = stem.lower()
    if lowered.endswith(".viz") or lowered.endswith(".hdv"):
        stem = stem.rsplit(".", 1)[0]
    cleaned = re.sub(r"[^A-Za-z0-9_]", "", stem)
    if not cleaned or cleaned[0].isdigit() or cleaned in _KEYWORDS:
        cleaned = "Exported" + cleaned
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", cleaned):
        return "ExportedProgram"
    return cleaned


def export_csharp(source: str, class_name: str = "ExportedProgram") -> str:
    """匯出成一支 Run(ViewShow)。圖形視窗可以把自己的 Show 傳進來。"""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", class_name):
        raise ExportError("類別名稱必須是 C# 識別字")
    program = parse_program(source)
    return _emit(source, program, class_name)


def _bare(name: str) -> str:
    return name.split(".", 1)[1].lower() if "." in name else name.lower()


def _ident(name: str) -> str:
    if name in _KEYWORDS:
        return "@" + name
    return name


def _cs_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("\"", "\\\"").replace("\r", "\\r").replace("\n", "\\n")
    return '"' + escaped + '"'


def _cs_number(value: float) -> str:
    number = float(value)
    if number.is_integer() and abs(number) < 1e15:
        return str(int(number))
    text = format(number, ".17g")
    if "e" not in text and "." not in text:
        text += ".0"
    return text


def _literal(expr: str, variables: dict, line: int):
    try:
        return eval_expr(expr, variables)
    except ExprError as exc:
        raise ExportError(str(exc), line) from exc


def _emit(source: str, program: Program, class_name: str) -> str:
    raw_lines = source.splitlines()
    constants: dict = {}
    images: set[str] = set()
    regions: dict[str, str] = {}
    kinds: dict[str, str] = {}
    body: list[str] = []
    last_image = ""
    last_region = ""
    shown_image = ""
    shown_region = ""
    displayed = False
    measure_index = 0

    def original(stmt) -> str:
        if 0 < stmt.line <= len(raw_lines):
            return raw_lines[stmt.line - 1].strip()
        return ""

    def assign(name: str, expr: str, kind: str, line: int) -> None:
        previous = kinds.get(name)
        if previous is None:
            kinds[name] = kind
            body.append(f"        var {_ident(name)} = {expr};")
            return
        if previous != kind:
            raise ExportError(f"變數 {name} 不能改成另一種結果", line)
        body.append(f"        {_ident(name)} = {expr};")

    def number(value, name: str, line: int) -> int:
        try:
            return whole(value, name)
        except ValueError as exc:
            raise ExportError(str(exc), line) from exc

    for stmt in program.stmts:
        if isinstance(stmt, (Blank, Comment)):
            continue
        if isinstance(stmt, Assign):
            value = _literal(stmt.expr, constants, stmt.line)
            if isinstance(value, (list, tuple)):
                raise ExportError("匯出還沒有做 tuple 指定", stmt.line)
            if not isinstance(value, (int, float, str)) or isinstance(value, bool):
                raise ExportError("匯出只能指定數字或字串", stmt.line)
            constants[stmt.name] = value
            continue
        if not isinstance(stmt, Call):
            raise ExportError("匯出只做讀圖、二值化、連通和顯示", stmt.line)
        bare = _bare(stmt.name)
        if bare not in _SUPPORTED:
            raise ExportError(f"匯出還沒有做 {stmt.name}", stmt.line)
        note = original(stmt)
        if note:
            body.append(f"        // {note}")
        if bare == "read_image":
            filename = _literal(stmt.control_in[0], constants, stmt.line)
            if not isinstance(filename, str):
                raise ExportError("FileName 要是字串", stmt.line)
            name = stmt.iconic_out[0]
            assign(name, f"Operators.ReadImage({_cs_string(filename)})", "image", stmt.line)
            images.add(name)
            regions.pop(name, None)
            last_image = name
            continue
        if bare == "threshold":
            source_name = stmt.iconic_in[0]
            if source_name not in images:
                raise ExportError(f"找不到影像 {source_name}", stmt.line)
            low = number(_literal(stmt.control_in[0], constants, stmt.line), "MinGray", stmt.line)
            high = number(_literal(stmt.control_in[1], constants, stmt.line), "MaxGray", stmt.line)
            name = stmt.iconic_out[0]
            assign(name, f"Operators.Threshold({_ident(source_name)}, {low}, {high})", "region", stmt.line)
            regions[name] = "one"
            images.discard(name)
            last_region = name
            continue
        if bare == "connection":
            source_name = stmt.iconic_in[0]
            if source_name not in regions:
                raise ExportError(f"找不到區域 {source_name}", stmt.line)
            if regions[source_name] != "one":
                raise ExportError("連通的輸入要是還沒拆開的區域", stmt.line)
            try:
                distance = scalar(_literal(stmt.control_in[0], constants, stmt.line), "Distance")
            except ValueError as exc:
                raise ExportError(str(exc), stmt.line) from exc
            if distance < 0:
                raise ExportError("Distance 不能是負數", stmt.line)
            if distance >= 2:
                raise ExportError("匯出還沒有做連通距離 2 以上", stmt.line)
            name = stmt.iconic_out[0]
            if abs(distance - 1.0) < 1e-12:
                call = f"Operators.Connection({_ident(source_name)})"
            else:
                call = f"Operators.Connection({_ident(source_name)}, {_cs_number(distance)})"
            assign(name, call, "regions", stmt.line)
            regions[name] = "many"
            images.discard(name)
            last_region = name
            continue
        if bare == "count_obj":
            source_name = stmt.iconic_in[0]
            if source_name not in regions:
                raise ExportError(f"找不到區域 {source_name}", stmt.line)
            assign(stmt.control_out[0], f"Operators.Count({_ident(source_name)})", "number", stmt.line)
            continue
        if bare == "area_center":
            source_name = stmt.iconic_in[0]
            if source_name not in regions:
                raise ExportError(f"找不到區域 {source_name}", stmt.line)
            measure_index += 1
            temp = f"exportMeasured{measure_index}"
            body.append(f"        var {temp} = Operators.AreaCenterOf({_ident(source_name)});")
            area_name, row_name, column_name = stmt.control_out
            assign(area_name, f"new int[{temp}.Length]", "numbers", stmt.line)
            assign(row_name, f"new double[{temp}.Length]", "numbers", stmt.line)
            assign(column_name, f"new double[{temp}.Length]", "numbers", stmt.line)
            body.append(f"        for (int exportIndex = 0; exportIndex < {temp}.Length; exportIndex++)")
            body.append("        {")
            body.append(f"            {_ident(area_name)}[exportIndex] = {temp}[exportIndex].Area;")
            body.append(f"            {_ident(row_name)}[exportIndex] = {temp}[exportIndex].Row;")
            body.append(f"            {_ident(column_name)}[exportIndex] = {temp}[exportIndex].Column;")
            body.append("        }")
            continue
        target = stmt.iconic_in[0]
        displayed = True
        if target in images:
            shown_image = target
            shown_region = ""
            continue
        if target in regions:
            if not shown_image:
                shown_image = last_image
            shown_region = target
            continue
        raise ExportError(f"找不到要顯示的 {target}", stmt.line)

    if not displayed:
        shown_image = last_image
        shown_region = last_region
    if not shown_image:
        raise ExportError("沒有影像可以顯示")
    if shown_region:
        overlay = f"Operators.Overlays({_ident(shown_region)})"
    else:
        overlay = "System.Array.Empty<Overlay>()"
    image = _ident(shown_image)
    body.append(f"        view({image}.Width, {image}.Height, {image}.Pixels, {image}.Layout, {overlay});")
    joined = "\n".join(body)
    return (
        "using System;\n"
        "using Vizion;\n"
        "\n"
        "// WinForms：using Vizion.Forms; 然後 " + class_name + ".Run(window.Show);\n"
        f"public static class {class_name}\n"
        "{\n"
        "    public static void Run(ViewShow view)\n"
        "    {\n"
        f"{joined}\n"
        "    }\n"
        "}\n"
    )
