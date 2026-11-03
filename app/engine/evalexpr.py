from __future__ import annotations

import ast
import math
import operator

from .objects import Image, Region, XLD


class ExprError(Exception):
    pass


_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CMP = {
    ast.Gt: operator.gt,
    ast.Lt: operator.lt,
    ast.GtE: operator.ge,
    ast.LtE: operator.le,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


def is_seq(value) -> bool:
    return isinstance(value, (list, tuple))


def to_list(value) -> list:
    if is_seq(value):
        return list(value)
    return [value]


def truthy(value) -> bool:
    if is_seq(value):
        return len(value) > 0 and all(truthy(item) for item in value)
    if isinstance(value, str):
        return value != ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return value is not None


def rewrite(expr: str) -> str:
    chars: list[str] = []
    i = 0
    n = len(expr)
    while i < n:
        ch = expr[i]
        if ch in "'\"":
            j = i + 1
            while j < n and expr[j] != ch:
                j += 1
            piece = expr[i : j + 1]
            if j < n and len(piece) >= 2:
                piece = piece[0] + piece[1:-1].replace("\\", "\\\\") + piece[-1]
            chars.append(piece)
            i = j + 1
            continue
        if ch == "|":
            j = i + 1
            while j < n and (expr[j].isalnum() or expr[j] == "_"):
                j += 1
            if j < n and expr[j] == "|" and j > i + 1:
                chars.append(f"__len__({expr[i + 1 : j]})")
                i = j + 1
                continue
        if ch == "<" and i + 1 < n and expr[i + 1] == ">":
            chars.append("!=")
            i += 2
            continue
        if ch == "#" :
            chars.append("!=")
            i += 1
            continue
        if ch == "=" and (i == 0 or expr[i - 1] not in "<>!=") and (i + 1 >= n or expr[i + 1] != "="):
            chars.append("==")
            i += 1
            continue
        chars.append(ch)
        i += 1
    return "".join(chars)


def _num(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    raise ExprError(f"需要數值，得到 {value!r}")


def apply_bin(fn, left, right):
    if not is_seq(left) and not is_seq(right):
        return fn(_num(left), _num(right))
    seq_l = list(left) if is_seq(left) else None
    seq_r = list(right) if is_seq(right) else None
    if seq_l is None:
        seq_l = [left] * len(seq_r)
    if seq_r is None:
        seq_r = [right] * len(seq_l)
    if len(seq_l) != len(seq_r):
        raise ExprError(f"tuple 長度不同（{len(seq_l)} 與 {len(seq_r)}）")
    return [fn(_num(a), _num(b)) for a, b in zip(seq_l, seq_r)]


def _cmp_one(fn, left, right):
    if isinstance(left, str) or isinstance(right, str):
        return 1 if fn(left, right) else 0
    return 1 if fn(_num(left), _num(right)) else 0


def apply_cmp(fn, left, right):
    if not is_seq(left) and not is_seq(right):
        return _cmp_one(fn, left, right)
    seq_l = list(left) if is_seq(left) else [left]
    seq_r = list(right) if is_seq(right) else [right]
    if len(seq_l) == 1 and len(seq_r) > 1:
        seq_l = seq_l * len(seq_r)
    if len(seq_r) == 1 and len(seq_l) > 1:
        seq_r = seq_r * len(seq_l)
    if len(seq_l) != len(seq_r):
        raise ExprError("tuple 長度不同")
    return [_cmp_one(fn, a, b) for a, b in zip(seq_l, seq_r)]


def eval_expr(expr: str, variables: dict):
    source = rewrite(expr.strip())
    if source == "":
        raise ExprError("空運算式")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"運算式語法錯誤：{expr}") from exc
    return _eval(tree.body, variables)


def _eval(node, variables):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in ("true", "True"):
            return 1
        if node.id in ("false", "False"):
            return 0
        if node.id not in variables:
            raise ExprError(f"未知變數 {node.id}")
        return variables[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _eval(node.operand, variables)
        if is_seq(value):
            return [-_num(item) for item in value]
        return -_num(value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _eval(node.operand, variables)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return 0 if truthy(_eval(node.operand, variables)) else 1
    if isinstance(node, ast.BinOp):
        left = _eval(node.left, variables)
        right = _eval(node.right, variables)
        if isinstance(node.op, ast.Add) and (isinstance(left, str) or isinstance(right, str)):
            if is_seq(left) or is_seq(right):
                raise ExprError("字串不能和 tuple 相加")
            return str(left) + str(right)
        fn = _BIN.get(type(node.op))
        if fn is None:
            raise ExprError("不支援的運算")
        return apply_bin(fn, left, right)
    if isinstance(node, ast.BoolOp):
        values = [_eval(item, variables) for item in node.values]
        if isinstance(node.op, ast.And):
            return 1 if all(truthy(item) for item in values) else 0
        if isinstance(node.op, ast.Or):
            return 1 if any(truthy(item) for item in values) else 0
        raise ExprError("不支援的邏輯運算")
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise ExprError("暫不支援連續比較")
        fn = _CMP.get(type(node.ops[0]))
        if fn is None:
            raise ExprError("不支援的比較")
        return apply_cmp(fn, _eval(node.left, variables), _eval(node.comparators[0], variables))
    if isinstance(node, ast.List):
        return [_eval(item, variables) for item in node.elts]
    if isinstance(node, ast.Subscript):
        value = _eval(node.value, variables)
        index = _eval(node.slice, variables)
        name = node.value.id if isinstance(node.value, ast.Name) else ""
        return _subscript(value, index, name)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        args = [_eval(arg, variables) for arg in node.args]
        return _call(node.func.id, args)
    raise ExprError("不支援的運算式")


def _index(seq: list, index, origin: int):
    pos = int(_num(index))
    if origin == 0:
        if not seq or pos < 0 or pos >= len(seq):
            if not seq:
                raise ExprError(f"索引 {pos} 超出範圍，tuple 是空的")
            raise ExprError(f"索引 {pos} 超出範圍 0..{len(seq) - 1}")
        return seq[pos]
    if pos < 1 or pos > len(seq):
        raise ExprError(f"索引 {pos} 超出範圍 1..{len(seq)}")
    return seq[pos - 1]


def _slot(index, count: int, origin: int) -> list[int]:
    items = list(index) if is_seq(index) else [index]
    slots = []
    for item in items:
        pos = int(_num(item))
        if origin == 0:
            if count == 0 or pos < 0 or pos >= count:
                if count == 0:
                    raise ExprError(f"索引 {pos} 超出範圍，tuple 是空的")
                raise ExprError(f"索引 {pos} 超出範圍 0..{count - 1}")
            slots.append(pos)
            continue
        if pos < 1 or pos > count:
            raise ExprError(f"索引 {pos} 超出範圍 1..{count}")
        slots.append(pos - 1)
    return slots


def _subscript(value, index, name: str):
    # HObject (region, contour, image tuple) is 1-based, including C# ho_ variables.
    # HTuple named hv_ is 0-based, matching HALCON C#. Other control tuples stay 1-based.
    if isinstance(value, Region):
        slots = _slot(index, value.count(), 1)
        links = None
        if value.links is not None and len(value.links) == value.count():
            links = [value.links[slot] for slot in slots]
        masks = [value.objects[slot] for slot in slots]
        return Region(value.width, value.height, masks, multi_color=len(masks) > 1, links=links)
    if isinstance(value, XLD):
        slots = _slot(index, len(value.contours), 1)
        groups = None
        if value.groups is not None and len(value.groups) == len(value.contours):
            groups = [value.groups[slot] for slot in slots]
        contours = [value.contours[slot] for slot in slots]
        return XLD(value.width, value.height, contours, multi_color=len(contours) > 1, groups=groups)
    if isinstance(value, str):
        raise ExprError("字串不能用索引")
    if isinstance(value, list) and value and all(isinstance(item, Image) for item in value):
        if is_seq(index):
            return [_index(value, item, 1) for item in index]
        return _index(value, index, 1)
    origin = 0 if name.lower().startswith("hv_") else 1
    seq = to_list(value)
    if is_seq(index):
        return [_index(seq, item, origin) for item in index]
    return _index(seq, index, origin)


def _flat_nums(args: list) -> list:
    nums = []
    for arg in args:
        if is_seq(arg):
            nums.extend(_num(item) for item in arg)
        else:
            nums.append(_num(arg))
    if not nums:
        raise ExprError("空 tuple")
    return nums


def _call(name: str, args: list):
    if name == "__len__":
        if len(args) != 1:
            raise ExprError("長度語法錯誤")
        return len(args[0]) if is_seq(args[0]) else 1
    if name == "min":
        return min(_flat_nums(args))
    if name == "max":
        return max(_flat_nums(args))
    if name == "sum":
        return sum(_flat_nums(args))
    if name == "mean":
        nums = _flat_nums(args)
        return sum(nums) / len(nums)
    if name == "abs":
        value = args[0]
        if is_seq(value):
            return [abs(_num(item)) for item in value]
        return abs(_num(value))
    if name == "rad":
        value = args[0]
        if is_seq(value):
            return [math.radians(_num(item)) for item in value]
        return math.radians(_num(value))
    if name == "deg":
        value = args[0]
        if is_seq(value):
            return [math.degrees(_num(item)) for item in value]
        return math.degrees(_num(value))
    if name == "sqrt":
        value = args[0]
        if is_seq(value):
            return [math.sqrt(_num(item)) for item in value]
        return math.sqrt(_num(value))
    if name == "round":
        if len(args) not in (1, 2):
            raise ExprError("round 需要 1 或 2 個參數")
        digits = 0 if len(args) == 1 else int(_num(args[1]))

        def conv(item):
            number = round(_num(item), digits)
            return int(number) if digits <= 0 else number

        value = args[0]
        if is_seq(value):
            return [conv(item) for item in value]
        return conv(value)
    if name == "int":
        value = args[0]

        def conv_int(item):
            return int(_num(item))

        if is_seq(value):
            return [conv_int(item) for item in value]
        return conv_int(value)
    raise ExprError(f"未知函式 {name}。可用 min、max、sum、mean、abs、sqrt、round、rad、deg")
