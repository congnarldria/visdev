from __future__ import annotations

import re
from dataclasses import dataclass, field

from .operators import REGISTRY


class ParseError(Exception):
    def __init__(self, line: int, message: str):
        self.line = line
        super().__init__(f"第 {line} 行：{message}")


@dataclass
class Stmt:
    line: int
    lines: list[int] = field(default_factory=list)


@dataclass
class Blank(Stmt):
    pass


@dataclass
class Comment(Stmt):
    pass


@dataclass
class Assign(Stmt):
    name: str = ""
    expr: str = ""


@dataclass
class Call(Stmt):
    name: str = ""
    iconic_in: list[str] = field(default_factory=list)
    iconic_out: list[str] = field(default_factory=list)
    control_in: list[str] = field(default_factory=list)
    control_out: list[str] = field(default_factory=list)


@dataclass
class IfStmt(Stmt):
    expr: str = ""


@dataclass
class ElseStmt(Stmt):
    pass


@dataclass
class EndIfStmt(Stmt):
    pass


@dataclass
class ForStmt(Stmt):
    var: str = ""
    start: str = ""
    end: str = ""
    step: str = "1"


@dataclass
class EndForStmt(Stmt):
    pass


@dataclass
class WhileStmt(Stmt):
    expr: str = ""


@dataclass
class EndWhileStmt(Stmt):
    pass


@dataclass
class Program:
    stmts: list
    if_false: dict
    else_jump: dict
    for_end: dict
    end_for: dict
    while_end: dict
    end_while: dict


def split_commas(text: str) -> list[str]:
    raw = text.strip()
    if raw == "":
        return []
    parts: list[str] = []
    buf: list[str] = []
    quote = None
    depth = 0
    for ch in raw:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
            buf.append(ch)
            continue
        if ch in "[(":
            depth += 1
            buf.append(ch)
            continue
        if ch in "])":
            depth -= 1
            buf.append(ch)
            continue
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    parts.append("".join(buf).strip())
    return parts


def is_ident(text: str) -> bool:
    return re.fullmatch(r"[A-Za-z_]\w*", text) is not None


def split_keyword(text: str, keyword: str):
    key = keyword.lower()
    quote = None
    depth = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            i += 1
            continue
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if depth == 0 and text[i : i + len(key)].lower() == key:
            before = text[i - 1] if i else " "
            after = text[i + len(key)] if i + len(key) < len(text) else " "
            if before.isalnum() or before == "_" or after.isalnum() or after == "_":
                i += 1
                continue
            return text[:i], text[i + len(key) :]
        i += 1
    return None


def _logical_lines(source: str):
    raw_lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    index = 0
    while index < len(raw_lines):
        text = raw_lines[index]
        lines = [index + 1]
        while text.rstrip().endswith("\\"):
            text = text.rstrip()[:-1] + " "
            index += 1
            if index >= len(raw_lines):
                break
            text += raw_lines[index]
            lines.append(index + 1)
        yield lines[0], text, lines
        index += 1


def _parse_for(text: str, line: int, lines: list[int]) -> ForStmt:
    matched = re.match(r"for\s+([A-Za-z_]\w*)\s*:=\s*", text, flags=re.IGNORECASE)
    if not matched:
        raise ParseError(line, "for 要寫成 for Index := 1 to 10 by 1")
    var = matched.group(1)
    rest = text[matched.end() :]
    parts = split_keyword(rest, "to")
    if parts is None:
        raise ParseError(line, "for 缺少 to")
    start, after = parts
    stepped = split_keyword(after, "by")
    if stepped is None:
        end, step = after, "1"
    else:
        end, step = stepped
    if start.strip() == "" or end.strip() == "" or step.strip() == "":
        raise ParseError(line, "for 的範圍不完整")
    return ForStmt(line, lines, var, start.strip(), end.strip(), step.strip())


def _parse_call(name: str, args: str, line: int, lines: list[int]) -> Call:
    op = REGISTRY.get(name.lower())
    if op is None:
        raise ParseError(line, f"未知算子 {name}")
    tokens = split_commas(args)
    n_in = len(op.iconic_in)
    n_out = len(op.iconic_out)
    n_ctrl = len(op.control_in)
    n_cout = len(op.control_out)
    optional = 0
    for param in reversed(op.control_in):
        if param.default is None:
            break
        optional += 1
    minimum = n_in + n_out + (n_ctrl - optional) + n_cout
    maximum = n_in + n_out + n_ctrl + n_cout
    if not minimum <= len(tokens) <= maximum:
        span = f"{minimum} 到 {maximum}" if minimum != maximum else str(minimum)
        raise ParseError(line, f"{op.name} 需要 {span} 個參數，目前是 {len(tokens)} 個。範本：{op.example}")
    iconic_in = tokens[:n_in]
    iconic_out = tokens[n_in : n_in + n_out]
    tail = tokens[n_in + n_out :]
    if n_cout:
        control_out = tail[-n_cout:]
        control_in = tail[:-n_cout]
    else:
        control_out = []
        control_in = tail
    while len(control_in) < n_ctrl:
        control_in.append(op.control_in[len(control_in)].default)
    for token in iconic_in + iconic_out + control_out:
        if not is_ident(token):
            raise ParseError(line, f"輸出與圖像輸入必須是變數名稱，得到 {token}")
    return Call(line, lines, op.name, iconic_in, iconic_out, control_in, control_out)


def parse_program(source: str) -> Program:
    stmts = []
    for line, text, lines in _logical_lines(source):
        raw = text.strip()
        if raw.endswith(";"):
            raw = raw[:-1].strip()
        if raw == "":
            stmts.append(Blank(line, lines))
            continue
        if raw.startswith("*") or raw.startswith("//"):
            stmts.append(Comment(line, lines))
            continue
        low = raw.lower()
        if low == "endif":
            stmts.append(EndIfStmt(line, lines))
            continue
        if low == "else":
            stmts.append(ElseStmt(line, lines))
            continue
        if low == "endfor":
            stmts.append(EndForStmt(line, lines))
            continue
        if low == "endwhile":
            stmts.append(EndWhileStmt(line, lines))
            continue
        if low.startswith("if ") or low.startswith("if("):
            matched = re.fullmatch(r"if\s*\((.*)\)\s*", raw, flags=re.IGNORECASE | re.DOTALL)
            if not matched:
                raise ParseError(line, "if 要寫成 if (條件)")
            stmts.append(IfStmt(line, lines, matched.group(1).strip()))
            continue
        if low.startswith("while ") or low.startswith("while("):
            matched = re.fullmatch(r"while\s*\((.*)\)\s*", raw, flags=re.IGNORECASE | re.DOTALL)
            if not matched:
                raise ParseError(line, "while 要寫成 while (條件)")
            stmts.append(WhileStmt(line, lines, matched.group(1).strip()))
            continue
        if low.startswith("for ") or low.startswith("for\t"):
            stmts.append(_parse_for(raw, line, lines))
            continue
        matched = re.fullmatch(r"([A-Za-z_]\w*)\s*:=\s*(.+)", raw, flags=re.DOTALL)
        if matched:
            stmts.append(Assign(line, lines, matched.group(1), matched.group(2).strip()))
            continue
        matched = re.fullmatch(r"([A-Za-z_]\w*)\s*\((.*)\)\s*", raw, flags=re.DOTALL)
        if matched:
            stmts.append(_parse_call(matched.group(1), matched.group(2), line, lines))
            continue
        raise ParseError(line, "看不懂這一行。註解用 *，指定用 :=，呼叫用 operator (...)")
    return _link(stmts)


def _link(stmts: list) -> Program:
    stack: list[int] = []
    if_false: dict[int, int] = {}
    else_jump: dict[int, int] = {}
    for_end: dict[int, int] = {}
    end_for: dict[int, int] = {}
    while_end: dict[int, int] = {}
    end_while: dict[int, int] = {}
    for index, stmt in enumerate(stmts):
        if isinstance(stmt, (IfStmt, ForStmt, WhileStmt)):
            stack.append(index)
            continue
        if isinstance(stmt, ElseStmt):
            if not stack or not isinstance(stmts[stack[-1]], IfStmt):
                raise ParseError(stmt.line, "else 沒有對應的 if")
            if_false[stack[-1]] = index + 1
            stack.append(index)
            continue
        if isinstance(stmt, EndIfStmt):
            if stack and isinstance(stmts[stack[-1]], ElseStmt):
                else_index = stack.pop()
                else_jump[else_index] = index + 1
            if not stack or not isinstance(stmts[stack[-1]], IfStmt):
                raise ParseError(stmt.line, "endif 沒有對應的 if")
            if_index = stack.pop()
            if_false.setdefault(if_index, index + 1)
            continue
        if isinstance(stmt, EndForStmt):
            if not stack or not isinstance(stmts[stack[-1]], ForStmt):
                raise ParseError(stmt.line, "endfor 沒有對應的 for")
            for_index = stack.pop()
            for_end[for_index] = index
            end_for[index] = for_index
            continue
        if isinstance(stmt, EndWhileStmt):
            if not stack or not isinstance(stmts[stack[-1]], WhileStmt):
                raise ParseError(stmt.line, "endwhile 沒有對應的 while")
            while_index = stack.pop()
            while_end[while_index] = index
            end_while[index] = while_index
            continue
    if stack:
        raise ParseError(stmts[stack[-1]].line, "這個區塊沒有結束")
    return Program(stmts, if_false, else_jump, for_end, end_for, while_end, end_while)
