"""把讀圖、二值化、連通和常用區域算子匯出成 C#。"""

from __future__ import annotations

import re

from .evalexpr import ExprError, eval_expr
from .operators import scalar, whole
from .parser import Assign, Blank, Call, Comment, Program, parse_program

_SUPPORTED = {
    "read_image", "threshold", "connection", "count_obj", "area_center", "dev_display",
    "select_shape", "select_obj", "concat_obj", "union1", "union2", "intersection", "difference",
    "gen_rectangle1", "smallest_rectangle1", "get_image_size", "get_grayval", "test_region_point",
    "intensity", "dilation_rectangle1", "erosion_rectangle1", "opening_rectangle1", "closing_rectangle1",
    "fill_up", "complement",
}
_SHAPE = {"area", "row", "column", "width", "height", "ratio", "center_row", "center_column", "center_col"}
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


def _cs_double(value: float) -> str:
    text = _cs_number(value)
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text


def _as_strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _as_numbers(value) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


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
        if bare in ("union1", "dilation_rectangle1", "erosion_rectangle1", "opening_rectangle1", "closing_rectangle1", "fill_up", "complement"):
            source_name = stmt.iconic_in[0]
            if source_name not in regions:
                raise ExportError(f"找不到區域 {source_name}", stmt.line)
            name = stmt.iconic_out[0]
            if bare == "union1":
                source = _ident(source_name) if regions[source_name] == "many" else f"new[] {{ {_ident(source_name)} }}"
                call = f"Operators.Union({source})"
            elif bare == "fill_up":
                call = f"Operators.FillUp({_ident(source_name)})"
            elif bare == "complement":
                call = f"Operators.Complement({_ident(source_name)})"
            else:
                wide = number(_literal(stmt.control_in[0], constants, stmt.line), "Width", stmt.line)
                high = number(_literal(stmt.control_in[1], constants, stmt.line), "Height", stmt.line)
                method = {
                    "dilation_rectangle1": "DilateRectangle",
                    "erosion_rectangle1": "ErodeRectangle",
                    "opening_rectangle1": "OpenRectangle",
                    "closing_rectangle1": "CloseRectangle",
                }[bare]
                call = f"Operators.{method}({_ident(source_name)}, {wide}, {high})"
            assign(name, call, "region", stmt.line)
            regions[name] = "one"
            images.discard(name)
            last_region = name
            continue
        if bare in ("union2", "intersection", "difference", "concat_obj"):
            left_name, right_name = stmt.iconic_in
            if left_name not in regions or right_name not in regions:
                missing = left_name if left_name not in regions else right_name
                raise ExportError(f"找不到區域 {missing}", stmt.line)
            name = stmt.iconic_out[0]
            method = {"union2": "Union", "intersection": "Intersection", "difference": "Difference", "concat_obj": "Concat"}[bare]
            assign(name, f"Operators.{method}({_ident(left_name)}, {_ident(right_name)})", "regions" if bare == "concat_obj" else "region", stmt.line)
            regions[name] = "many" if bare == "concat_obj" else "one"
            images.discard(name)
            last_region = name
            continue
        if bare == "select_obj":
            source_name = stmt.iconic_in[0]
            if source_name not in regions:
                raise ExportError(f"找不到區域 {source_name}", stmt.line)
            index = _literal(stmt.control_in[0], constants, stmt.line)
            name = stmt.iconic_out[0]
            if isinstance(index, (list, tuple)):
                slots = [int(round(float(item))) for item in index]
                listed = ", ".join(str(item) for item in slots)
                source = _ident(source_name) if regions[source_name] == "many" else f"new[] {{ {_ident(source_name)} }}"
                assign(name, f"Operators.Select({source}, new[] {{ {listed} }})", "regions", stmt.line)
                regions[name] = "many"
            else:
                slot = int(round(float(index)))
                assign(name, f"Operators.Select({_ident(source_name)}, {slot})", "region", stmt.line)
                regions[name] = "one"
            images.discard(name)
            last_region = name
            continue
        if bare == "select_shape":
            source_name = stmt.iconic_in[0]
            if source_name not in regions:
                raise ExportError(f"找不到區域 {source_name}", stmt.line)
            features = [item.strip().strip("'\"").lower() for item in _as_strings(_literal(stmt.control_in[0], constants, stmt.line))]
            unknown = [item for item in features if item not in _SHAPE]
            if unknown:
                raise ExportError(f"匯出的 select_shape 還沒有做特徵 {unknown[0]}", stmt.line)
            operation = str(_literal(stmt.control_in[1], constants, stmt.line)).strip().strip("'\"").lower()
            if operation not in ("and", "or"):
                raise ExportError("Operation 只能是 and 或 or", stmt.line)
            lows = _as_numbers(_literal(stmt.control_in[2], constants, stmt.line))
            highs = _as_numbers(_literal(stmt.control_in[3], constants, stmt.line))
            if len(lows) == 1 and len(features) > 1:
                lows = lows * len(features)
            if len(highs) == 1 and len(features) > 1:
                highs = highs * len(features)
            if len(lows) != len(features) or len(highs) != len(features):
                raise ExportError("Min 和 Max 的長度要和特徵一樣", stmt.line)
            names = ", ".join(_cs_string(item) for item in features)
            low_text = ", ".join(_cs_double(item) for item in lows)
            high_text = ", ".join(_cs_double(item) for item in highs)
            name = stmt.iconic_out[0]
            assign(
                name,
                f"Operators.SelectShape({_ident(source_name)}, new[] {{ {names} }}, {_cs_string(operation)}, new[] {{ {low_text} }}, new[] {{ {high_text} }})",
                "regions",
                stmt.line,
            )
            regions[name] = "many"
            images.discard(name)
            last_region = name
            continue
        if bare == "gen_rectangle1":
            coords = [number(_literal(expr, constants, stmt.line), label, stmt.line) for expr, label in zip(stmt.control_in, ("Row1", "Column1", "Row2", "Column2"))]
            name = stmt.iconic_out[0]
            if last_image:
                call = f"Operators.Rectangle({_ident(last_image)}, {', '.join(str(item) for item in coords)})"
            else:
                call = f"Operators.Rectangle({', '.join(str(item) for item in coords)})"
            assign(name, call, "region", stmt.line)
            regions[name] = "one"
            images.discard(name)
            last_region = name
            continue
        if bare == "get_image_size":
            source_name = stmt.iconic_in[0]
            if source_name not in images:
                raise ExportError(f"找不到影像 {source_name}", stmt.line)
            width_name, height_name = stmt.control_out
            assign(width_name, f"{_ident(source_name)}.Width", "number", stmt.line)
            assign(height_name, f"{_ident(source_name)}.Height", "number", stmt.line)
            continue
        if bare == "get_grayval":
            source_name = stmt.iconic_in[0]
            if source_name not in images:
                raise ExportError(f"找不到影像 {source_name}", stmt.line)
            row = _literal(stmt.control_in[0], constants, stmt.line)
            column = _literal(stmt.control_in[1], constants, stmt.line)
            if isinstance(row, (list, tuple)) or isinstance(column, (list, tuple)):
                raise ExportError("匯出的 get_grayval 一次讀一個點", stmt.line)
            name = stmt.control_out[0]
            assign(name, f"Operators.GrayAt({_ident(source_name)}, {int(round(float(row)))}, {int(round(float(column)))})", "intarray", stmt.line)
            continue
        if bare == "test_region_point":
            source_name = stmt.iconic_in[0]
            if source_name not in regions:
                raise ExportError(f"找不到區域 {source_name}", stmt.line)
            row = _literal(stmt.control_in[0], constants, stmt.line)
            column = _literal(stmt.control_in[1], constants, stmt.line)
            if isinstance(row, (list, tuple)) or isinstance(column, (list, tuple)):
                raise ExportError("匯出的 test_region_point 一次測一個點", stmt.line)
            assign(stmt.control_out[0], f"Operators.Contains({_ident(source_name)}, {int(round(float(row)))}, {int(round(float(column)))})", "number", stmt.line)
            continue
        if bare == "smallest_rectangle1":
            source_name = stmt.iconic_in[0]
            if source_name not in regions:
                raise ExportError(f"找不到區域 {source_name}", stmt.line)
            measure_index += 1
            temp = f"exportBounds{measure_index}"
            body.append(f"        var {temp} = Operators.Bounds({_ident(source_name)});")
            row1, column1, row2, column2 = stmt.control_out
            for out_name, field in ((row1, "Row1"), (column1, "Column1"), (row2, "Row2"), (column2, "Column2")):
                assign(out_name, f"new int[{temp}.Length]", "intarray", stmt.line)
            body.append(f"        for (int exportIndex = 0; exportIndex < {temp}.Length; exportIndex++)")
            body.append("        {")
            body.append(f"            {_ident(row1)}[exportIndex] = {temp}[exportIndex].Row1;")
            body.append(f"            {_ident(column1)}[exportIndex] = {temp}[exportIndex].Column1;")
            body.append(f"            {_ident(row2)}[exportIndex] = {temp}[exportIndex].Row2;")
            body.append(f"            {_ident(column2)}[exportIndex] = {temp}[exportIndex].Column2;")
            body.append("        }")
            continue
        if bare == "intensity":
            region_name, image_name = stmt.iconic_in
            if region_name not in regions:
                raise ExportError(f"找不到區域 {region_name}", stmt.line)
            if image_name not in images:
                raise ExportError(f"找不到影像 {image_name}", stmt.line)
            measure_index += 1
            temp = f"exportTone{measure_index}"
            body.append(f"        var {temp} = Operators.IntensityOf({_ident(region_name)}, {_ident(image_name)});")
            mean_name, deviation_name = stmt.control_out
            assign(mean_name, f"new double[{temp}.Length]", "numbers", stmt.line)
            assign(deviation_name, f"new double[{temp}.Length]", "numbers", stmt.line)
            body.append(f"        for (int exportIndex = 0; exportIndex < {temp}.Length; exportIndex++)")
            body.append("        {")
            body.append(f"            {_ident(mean_name)}[exportIndex] = {temp}[exportIndex].Mean;")
            body.append(f"            {_ident(deviation_name)}[exportIndex] = {temp}[exportIndex].Deviation;")
            body.append("        }")
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
