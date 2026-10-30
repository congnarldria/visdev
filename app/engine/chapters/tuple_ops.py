"""HALCON-style tuple operators.

Indices follow HALCON and start at 0. The expression form T[1] stays 1-based.
Handle, semaphore, environment, and vector operators are not registered:
there is no HALCON handle type in this runtime.
"""

from __future__ import annotations

import math
import random
import re

from ..objects import Result
from ..operators import P, operator

IMPLEMENTED: set[str] = set()

EXISTING = {"tuple_length", "tuple_min", "tuple_max", "tuple_sum", "tuple_mean"}

SKIPPED = {
    "tuple_environment",
    "tuple_is_handle",
    "tuple_is_handle_elem",
    "tuple_is_serializable",
    "tuple_is_serializable_elem",
    "tuple_is_valid_handle",
    "tuple_sem_type",
    "tuple_sem_type_elem",
    "tuple_str_bit_select",
    "tuple_to_vector_1d",
}

HALCON_TUPLE = {
    "tuple_abs", "tuple_acos", "tuple_acosh", "tuple_add", "tuple_and", "tuple_asin", "tuple_asinh",
    "tuple_at", "tuple_atan", "tuple_atan2", "tuple_atanh", "tuple_band", "tuple_bnot", "tuple_bor",
    "tuple_bxor", "tuple_cbrt", "tuple_ceil", "tuple_chr", "tuple_chrt", "tuple_concat", "tuple_constant",
    "tuple_cos", "tuple_cosh", "tuple_cumul", "tuple_deg", "tuple_deviation", "tuple_difference",
    "tuple_div", "tuple_environment", "tuple_equal", "tuple_equal_elem", "tuple_erf", "tuple_erfc",
    "tuple_exp", "tuple_exp10", "tuple_exp2", "tuple_fabs", "tuple_find", "tuple_find_first",
    "tuple_find_last", "tuple_first_n", "tuple_floor", "tuple_fmod", "tuple_gen_const", "tuple_gen_sequence",
    "tuple_greater", "tuple_greater_elem", "tuple_greater_equal", "tuple_greater_equal_elem",
    "tuple_histo_range", "tuple_hypot", "tuple_insert", "tuple_int", "tuple_intersection", "tuple_inverse",
    "tuple_is_handle", "tuple_is_handle_elem", "tuple_is_int", "tuple_is_int_elem", "tuple_is_mixed",
    "tuple_is_nan_elem", "tuple_is_number", "tuple_is_real", "tuple_is_real_elem", "tuple_is_serializable",
    "tuple_is_serializable_elem", "tuple_is_string", "tuple_is_string_elem", "tuple_is_valid_handle",
    "tuple_join", "tuple_last_n", "tuple_ldexp", "tuple_length", "tuple_less", "tuple_less_elem",
    "tuple_less_equal", "tuple_less_equal_elem", "tuple_lgamma", "tuple_log", "tuple_log10", "tuple_log2",
    "tuple_lsh", "tuple_max", "tuple_max2", "tuple_mean", "tuple_median", "tuple_min", "tuple_min2",
    "tuple_mod", "tuple_mult", "tuple_neg", "tuple_not", "tuple_not_equal", "tuple_not_equal_elem",
    "tuple_number", "tuple_or", "tuple_ord", "tuple_ords", "tuple_pow", "tuple_rad", "tuple_rand",
    "tuple_real", "tuple_regexp_match", "tuple_regexp_replace", "tuple_regexp_select", "tuple_regexp_test",
    "tuple_remove", "tuple_repeat", "tuple_repeat_elem", "tuple_replace", "tuple_round", "tuple_rsh",
    "tuple_select", "tuple_select_mask", "tuple_select_range", "tuple_select_rank", "tuple_sem_type",
    "tuple_sem_type_elem", "tuple_sgn", "tuple_sin", "tuple_sinh", "tuple_sort", "tuple_sort_index",
    "tuple_split", "tuple_sqrt", "tuple_str_bit_select", "tuple_str_distance", "tuple_str_first_n",
    "tuple_str_last_n", "tuple_str_replace", "tuple_strchr", "tuple_string", "tuple_strlen",
    "tuple_strrchr", "tuple_strrstr", "tuple_strstr", "tuple_sub", "tuple_substr", "tuple_sum",
    "tuple_symmdiff", "tuple_tan", "tuple_tanh", "tuple_tgamma", "tuple_to_vector_1d", "tuple_type",
    "tuple_type_elem", "tuple_union", "tuple_uniq", "tuple_xor",
}


def _items(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _listed(value) -> bool:
    return isinstance(value, (list, tuple))


def _tidy(value):
    if isinstance(value, float) and math.isfinite(value) and abs(value - round(value)) < 1e-9:
        return int(round(value))
    return value


def _num(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"不是有限數值：{value}")
        return value
    raise ValueError(f"需要數值，得到 {value!r}")


def _as_int(value) -> int:
    number = _num(value)
    if isinstance(number, float) and abs(number - round(number)) > 1e-9:
        raise ValueError(f"需要整數，得到 {value}")
    return int(round(number))


def _finish(values: list, force_tuple: bool):
    if force_tuple or len(values) != 1:
        return Result([], [values])
    return Result([], [values[0]])


def _bind(name, control_in, control_out, short, detail, example, fn):
    if name in IMPLEMENTED:
        raise RuntimeError(f"重複註冊 {name}")
    fn.__name__ = name
    operator("Tuple", [], [], control_in, control_out, short, detail, example)(fn)
    IMPLEMENTED.add(name)


def _unary(fn, *, keep_float=False):
    def run(value, *, ctx, _fn=fn, _keep=keep_float):
        src = _items(value)
        if not src:
            raise ValueError("空 tuple")
        out = []
        for item in src:
            try:
                result = _fn(_num(item))
            except ValueError as exc:
                raise ValueError(f"無法計算 {item}") from exc
            out.append(result if _keep else _tidy(result))
        return _finish(out, _listed(value))

    return run


def _binary(fn):
    def run(left, right, *, ctx, _fn=fn):
        first, second = _align(left, right)
        if not first:
            return Result([], [[]])
        return _finish([_tidy(_fn(a, b)) for a, b in zip(first, second)], _listed(left) or _listed(right))

    return run


def _align(left, right):
    first, second = _items(left), _items(right)
    if len(first) == 1 and len(second) > 1:
        first = first * len(second)
    elif len(second) == 1 and len(first) > 1:
        second = second * len(first)
    elif len(first) != len(second):
        raise ValueError(f"tuple 長度不同（{len(first)} 與 {len(second)}）")
    return first, second


def _same(left, right) -> bool:
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    return _num(left) == _num(right)


def _order(left, right):
    if isinstance(left, str) or isinstance(right, str):
        if not isinstance(left, str) or not isinstance(right, str):
            return None
        if left < right:
            return -1
        if left > right:
            return 1
        return 0
    a, b = _num(left), _num(right)
    return (a > b) - (a < b)


def _cmp_all(sign):
    def run(left, right, *, ctx):
        first, second = _items(left), _items(right)
        if len(first) != len(second):
            return Result([], [0])
        ok = True
        for a, b in zip(first, second):
            order = _order(a, b)
            if order is None or not sign(order):
                ok = False
                break
        return Result([], [1 if ok else 0])

    return run


def _cmp_elem(sign):
    def run(left, right, *, ctx):
        first, second = _align(left, right)
        out = []
        for a, b in zip(first, second):
            order = _order(a, b)
            out.append(1 if order is not None and sign(order) else 0)
        return _finish(out, _listed(left) or _listed(right))

    return run


def _index(length: int, raw, name="Index") -> int:
    if length <= 0:
        raise ValueError("空 tuple")
    pos = _as_int(raw)
    if pos < 0:
        pos += length
    if pos < 0 or pos >= length:
        raise ValueError(f"{name} 超出範圍。長度是 {length}，索引從 0 開始，負數從尾巴數")
    return pos


def _kind(value) -> str:
    if isinstance(value, bool):
        return "integer"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "real"
    if isinstance(value, str):
        return "string"
    raise ValueError(f"不能放進 tuple：{value!r}")


def _round_away(value):
    number = _num(value)
    if number >= 0:
        return int(math.floor(number + 0.5))
    return int(math.ceil(number - 0.5))


def _mod(left, right):
    divisor = _num(right)
    if divisor == 0:
        raise ValueError("除數是 0")
    dividend = _num(left)
    return dividend - divisor * math.trunc(dividend / divisor)


def _div(left, right):
    divisor = _num(right)
    if divisor == 0:
        raise ValueError("除數是 0")
    return _num(left) / divisor


def _fmod(left, right):
    divisor = _num(right)
    if divisor == 0:
        raise ValueError("除數是 0")
    return math.fmod(_num(left), divisor)


def _bit(fn):
    def run(left, right, _fn=fn):
        return _fn(_as_int(left), _as_int(right))

    return run


def _sequence(start, end, step):
    start, end, step = _num(start), _num(end), _num(step)
    if step == 0:
        raise ValueError("Step 不能是 0")
    if (end - start) / step < -1e-9:
        return []
    count = int(math.floor((end - start) / step + 1e-9)) + 1
    return [_tidy(start + index * step) for index in range(max(count, 0))]


def _format_one(value, fmt: str) -> str:
    spec = str(fmt).strip()
    if spec in ("", "s"):
        return str(value)
    if spec in ("d", "i"):
        return str(_as_int(value))
    if spec.startswith("%"):
        payload = _num(value) if spec[-1] in "diouxXeEfFgG" else value
        return spec % payload
    if re.fullmatch(r"[0-9]*d", spec) or re.fullmatch(r"[0-9]*\.?[0-9]*[feg]", spec) or re.fullmatch(r"\.[0-9]+[feg]", spec):
        return format(_num(value), spec)
    raise ValueError(f"不支援的格式 {fmt}。可用 d、.2f、g 或 % 開頭的格式")


def _slice_text(text: str, start, end) -> str:
    size = len(text)
    left = _as_int(start)
    right = _as_int(end)
    if left < 0:
        left += size
    if right < 0:
        right += size
    if left < 0 or right < left or left >= size:
        return ""
    return text[left : min(size, right + 1)]


def _find_text(text: str, needle: str, reverse: bool) -> int:
    if needle == "":
        return -1
    return text.rfind(needle) if reverse else text.find(needle)


def _distance(left: str, right: str) -> int:
    if left == right:
        return 0
    prev = list(range(len(right) + 1))
    for i, ch in enumerate(left, 1):
        cur = [i]
        for j, other in enumerate(right, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ch != other)))
        prev = cur
    return prev[-1]


_ONE = [P("Tuple", "輸入 tuple")]
_TWO = [P("T1", "第一個 tuple"), P("T2", "第二個 tuple。長度 1 時會套用到另一邊的每個元素")]
_ELEM = "兩邊長度相同就逐一計算。其中一邊只有一個值時，套用到另一邊的每個元素。單一數值回傳單一數值。"
_ALL = "回傳一個 0 或 1。長度不同就是 0。長度相同時，每一對都成立才是 1。數字比大小，字串比字典順序。"


def _reg_unary(name, out_name, short, fn, example, *, keep_float=False):
    _bind(
        name,
        _ONE,
        [P(out_name, short)],
        short,
        f"{short}。對每個元素各算一次。空 tuple 會報錯。{_ELEM}",
        example,
        _unary(fn, keep_float=keep_float),
    )


def _reg_binary(name, out_name, short, fn, example, detail=None):
    _bind(
        name,
        _TWO,
        [P(out_name, short)],
        short,
        detail or f"{short}。{_ELEM}",
        example,
        _binary(fn),
    )


def _register_math():
    unary = [
        ("tuple_abs", "Abs", "絕對值", abs, "tuple_abs ([-2, 3], Abs)", False),
        ("tuple_fabs", "Fabs", "浮點絕對值", abs, "tuple_fabs (-2.5, Abs)", True),
        ("tuple_neg", "Neg", "變號", lambda v: -v, "tuple_neg ([1, -2], Neg)", False),
        ("tuple_sqrt", "Sqrt", "平方根", math.sqrt, "tuple_sqrt ([4, 9], Root)", False),
        ("tuple_cbrt", "Cbrt", "立方根", lambda v: math.copysign(abs(v) ** (1 / 3), v), "tuple_cbrt ([8, -8], Root)", False),
        ("tuple_sin", "Sin", "正弦，弳度", math.sin, "tuple_sin (0, Sin)", True),
        ("tuple_cos", "Cos", "餘弦，弳度", math.cos, "tuple_cos (0, Cos)", True),
        ("tuple_tan", "Tan", "正切，弳度", math.tan, "tuple_tan (0, Tan)", True),
        ("tuple_asin", "Asin", "反正弦，回傳弳度", math.asin, "tuple_asin (0, Angle)", True),
        ("tuple_acos", "Acos", "反餘弦，回傳弳度", math.acos, "tuple_acos (1, Angle)", True),
        ("tuple_atan", "Atan", "反正切，回傳弳度", math.atan, "tuple_atan (0, Angle)", True),
        ("tuple_sinh", "Sinh", "雙曲正弦", math.sinh, "tuple_sinh (0, Value)", True),
        ("tuple_cosh", "Cosh", "雙曲餘弦", math.cosh, "tuple_cosh (0, Value)", True),
        ("tuple_tanh", "Tanh", "雙曲正切", math.tanh, "tuple_tanh (0, Value)", True),
        ("tuple_asinh", "Asinh", "反雙曲正弦", math.asinh, "tuple_asinh (0, Value)", True),
        ("tuple_acosh", "Acosh", "反雙曲餘弦", math.acosh, "tuple_acosh (1, Value)", True),
        ("tuple_atanh", "Atanh", "反雙曲正切", math.atanh, "tuple_atanh (0, Value)", True),
        ("tuple_exp", "Exp", "e 的次方", math.exp, "tuple_exp (0, Value)", True),
        ("tuple_exp2", "Exp2", "2 的次方", lambda v: 2 ** v, "tuple_exp2 ([1, 3], Value)", False),
        ("tuple_exp10", "Exp10", "10 的次方", lambda v: 10 ** v, "tuple_exp10 (2, Value)", False),
        ("tuple_log", "Log", "自然對數", math.log, "tuple_log (1, Value)", True),
        ("tuple_log10", "Log10", "常用對數", math.log10, "tuple_log10 (100, Value)", False),
        ("tuple_log2", "Log2", "以 2 為底的對數", math.log2, "tuple_log2 (8, Value)", False),
        ("tuple_floor", "Floor", "向下取整", math.floor, "tuple_floor ([1.2, -1.2], Value)", False),
        ("tuple_ceil", "Ceil", "向上取整", math.ceil, "tuple_ceil ([1.2, -1.2], Value)", False),
        ("tuple_deg", "Deg", "弳度轉角度", math.degrees, "tuple_deg (rad(180), Deg)", False),
        ("tuple_rad", "Rad", "角度轉弳度", math.radians, "tuple_rad (180, Rad)", False),
        ("tuple_erf", "Erf", "誤差函數", math.erf, "tuple_erf (0, Value)", True),
        ("tuple_erfc", "Erfc", "互補誤差函數", math.erfc, "tuple_erfc (0, Value)", True),
        ("tuple_lgamma", "Lgamma", "gamma 絕對值的自然對數", math.lgamma, "tuple_lgamma (1, Value)", True),
        ("tuple_tgamma", "Tgamma", "gamma 函數", math.gamma, "tuple_tgamma (1, Value)", True),
        ("tuple_real", "Real", "轉成浮點數", float, "tuple_real ([1, 2], Real)", True),
    ]
    for name, out_name, short, fn, example, keep in unary:
        _reg_unary(name, out_name, short, fn, example, keep_float=keep)

    _bind("tuple_int", _ONE, [P("Int", "去掉小數的整數")], "去掉小數，朝 0 截斷", "1.9 變成 1，-1.9 變成 -1。", "tuple_int ([1.9, -1.9], Int)", _unary(lambda v: int(math.trunc(v))))
    _bind("tuple_round", _ONE, [P("Round", "四捨五入")], "四捨五入，.5 遠離 0", "1.5 變成 2，-1.5 變成 -2。", "tuple_round ([1.5, -1.5], Round)", _unary(_round_away))
    _bind("tuple_sgn", _ONE, [P("Sgn", "符號")], "符號：負數 -1、0 是 0、正數 1", "對每個元素各算一次。", "tuple_sgn ([-2, 0, 3], Sgn)", _unary(lambda v: (v > 0) - (v < 0)))
    _bind("tuple_not", _ONE, [P("Not", "邏輯否定")], "邏輯否定", "0 變成 1，其他數值變成 0。", "tuple_not ([0, 2], Not)", _unary(lambda v: 0 if v != 0 else 1))
    _bind("tuple_bnot", _ONE, [P("BNot", "位元否定")], "整數位元否定", "用二補數的 NOT。非整數會報錯。", "tuple_bnot (0, Bits)", _unary(lambda v: ~_as_int(v)))

    binary = [
        ("tuple_add", "Sum", "相加", lambda a, b: _num(a) + _num(b), "tuple_add ([1, 2], 10, Sum)"),
        ("tuple_sub", "Diff", "相減", lambda a, b: _num(a) - _num(b), "tuple_sub ([5, 4], 1, Diff)"),
        ("tuple_mult", "Prod", "相乘", lambda a, b: _num(a) * _num(b), "tuple_mult ([2, 3], 4, Prod)"),
        ("tuple_div", "Quot", "相除", _div, "tuple_div ([4, 9], 2, Quot)"),
        ("tuple_mod", "Mod", "餘數，符號跟被除數", _mod, "tuple_mod ([5, -5], 2, Mod)"),
        ("tuple_fmod", "Fmod", "浮點餘數", _fmod, "tuple_fmod (5.5, 2, Mod)"),
        ("tuple_pow", "Pow", "次方", lambda a, b: _num(a) ** _num(b), "tuple_pow (2, [3, 4], Pow)"),
        ("tuple_atan2", "ATan2", "兩個引數的反正切，Y 在前", lambda y, x: math.atan2(_num(y), _num(x)), "tuple_atan2 (0, 1, Angle)"),
        ("tuple_hypot", "Hypot", "直角三角形斜邊", lambda a, b: math.hypot(_num(a), _num(b)), "tuple_hypot (3, 4, Length)"),
        ("tuple_ldexp", "Ldexp", "乘上 2 的整數次方", lambda a, b: math.ldexp(_num(a), _as_int(b)), "tuple_ldexp (1.5, 2, Value)"),
        ("tuple_min2", "Min2", "兩個 tuple 逐一取較小值", lambda a, b: min(_num(a), _num(b)), "tuple_min2 ([1, 8], [3, 2], Min2)"),
        ("tuple_max2", "Max2", "兩個 tuple 逐一取較大值", lambda a, b: max(_num(a), _num(b)), "tuple_max2 ([1, 8], [3, 2], Max2)"),
        ("tuple_and", "And", "邏輯且", lambda a, b: 1 if _num(a) != 0 and _num(b) != 0 else 0, "tuple_and ([1, 0], [1, 1], And)"),
        ("tuple_or", "Or", "邏輯或", lambda a, b: 1 if _num(a) != 0 or _num(b) != 0 else 0, "tuple_or ([1, 0], [0, 0], Or)"),
        ("tuple_xor", "Xor", "邏輯互斥或", lambda a, b: 1 if (_num(a) != 0) != (_num(b) != 0) else 0, "tuple_xor ([1, 0], [1, 1], Xor)"),
        ("tuple_band", "BAnd", "位元且", _bit(lambda a, b: a & b), "tuple_band (6, 3, Bits)"),
        ("tuple_bor", "BOr", "位元或", _bit(lambda a, b: a | b), "tuple_bor (6, 3, Bits)"),
        ("tuple_bxor", "BXor", "位元互斥或", _bit(lambda a, b: a ^ b), "tuple_bxor (6, 3, Bits)"),
        ("tuple_lsh", "Lsh", "位元左移", _bit(lambda a, b: a << b), "tuple_lsh (1, 3, Bits)"),
        ("tuple_rsh", "Rsh", "位元右移", _bit(lambda a, b: a >> b), "tuple_rsh (16, 2, Bits)"),
    ]
    for name, out_name, short, fn, example in binary:
        detail = f"{short}。{_ELEM}"
        if name == "tuple_atan2":
            detail = "Y 在前、X 在後，回傳弳度。" + _ELEM
        if name in ("tuple_band", "tuple_bor", "tuple_bxor", "tuple_lsh", "tuple_rsh"):
            detail = f"{short}。兩邊都必須是整數。" + _ELEM
        if name == "tuple_div":
            detail = "真除法，除數 0 會報錯。" + _ELEM
        _reg_binary(name, out_name, short, fn, example, detail)

    for name, short, sign in (
        ("tuple_less", "整個 tuple 是否較小", lambda order: order < 0),
        ("tuple_less_equal", "整個 tuple 是否較小或相等", lambda order: order <= 0),
        ("tuple_greater", "整個 tuple 是否較大", lambda order: order > 0),
        ("tuple_greater_equal", "整個 tuple 是否較大或相等", lambda order: order >= 0),
        ("tuple_equal", "兩個 tuple 是否完全相同", lambda order: order == 0),
        ("tuple_not_equal", "兩個 tuple 是否不完全相同", lambda order: order != 0),
    ):
        _bind(name, _TWO, [P("Result", "0 或 1")], short, _ALL, f"{name} ([1, 2], [1, 2], Result)", _cmp_all(sign))
    for name, short, sign in (
        ("tuple_less_elem", "逐一比較是否較小", lambda order: order < 0),
        ("tuple_less_equal_elem", "逐一比較是否較小或相等", lambda order: order <= 0),
        ("tuple_greater_elem", "逐一比較是否較大", lambda order: order > 0),
        ("tuple_greater_equal_elem", "逐一比較是否較大或相等", lambda order: order >= 0),
        ("tuple_equal_elem", "逐一比較是否相等", lambda order: order == 0),
        ("tuple_not_equal_elem", "逐一比較是否不相等", lambda order: order != 0),
    ):
        _bind(name, _TWO, [P("Result", "每個元素 0 或 1")], short, f"每個位置回傳 0 或 1。{_ELEM}", f"{name} ([1, 3], 2, Result)", _cmp_elem(sign))


def _register_select():
    def tuple_select(value, index, *, ctx):
        src = _items(value)
        if not src:
            raise ValueError("空 tuple")
        picked = [src[_index(len(src), item)] for item in _items(index)]
        return _finish(picked, _listed(index))

    def tuple_remove(value, index, *, ctx):
        src = _items(value)
        gone = {_index(len(src), item) for item in _items(index)}
        return Result([], [[item for pos, item in enumerate(src) if pos not in gone]])

    def tuple_find(value, to_find, *, ctx):
        src = _items(value)
        needles = _items(to_find)
        found = [pos for pos, item in enumerate(src) if any(_same(item, needle) for needle in needles)]
        if not found:
            return Result([], [-1])
        return _finish(found, len(found) != 1)

    def tuple_find_edge(value, to_find, *, ctx, reverse):
        src = _items(value)
        order = range(len(src) - 1, -1, -1) if reverse else range(len(src))
        for pos in order:
            if _same(src[pos], to_find if not _listed(to_find) else _items(to_find)[0]):
                return Result([], [pos])
        return Result([], [-1])

    def tuple_first_n(value, index, *, ctx):
        src = _items(value)
        pos = _index(len(src), index)
        return Result([], [src[: pos + 1]])

    def tuple_last_n(value, index, *, ctx):
        src = _items(value)
        pos = _index(len(src), index)
        return Result([], [src[pos:]])

    def tuple_select_range(value, left, right, *, ctx):
        src = _items(value)
        if not src:
            raise ValueError("空 tuple")
        start = _index(len(src), left, "Left")
        end = _index(len(src), right, "Right")
        if end < start:
            return Result([], [[]])
        return Result([], [src[start : end + 1]])

    def tuple_select_mask(value, mask, *, ctx):
        src, flags = _align(value, mask)
        picked = [item for item, flag in zip(src, flags) if _num(flag) != 0]
        return Result([], [picked])

    def tuple_select_rank(value, rank, *, ctx):
        src = _items(value)
        if not src:
            raise ValueError("空 tuple")
        if any(isinstance(item, str) for item in src):
            ordered = sorted(src)
        else:
            ordered = sorted(src, key=_num)
        return Result([], [ordered[_index(len(ordered), rank, "RankIndex")]])

    def tuple_insert(value, insert, index, *, ctx):
        src = _items(value)
        pos = _as_int(index)
        if pos < 0:
            pos += len(src)
        if pos < 0 or pos > len(src):
            raise ValueError("Index 超出範圍。可以等於長度，表示加在尾巴")
        block = _items(insert)
        return Result([], [src[:pos] + block + src[pos:]])

    def tuple_replace(value, new_value, index, *, ctx):
        src = _items(value)
        positions = [_index(len(src), item) for item in _items(index)]
        block = _items(new_value)
        if len(block) == 1 and len(positions) > 1:
            block = block * len(positions)
        if len(block) != len(positions):
            raise ValueError("Value 和 Index 的長度不同")
        out = list(src)
        for pos, item in zip(positions, block):
            out[pos] = item
        return Result([], [out])

    def tuple_inverse(value, *, ctx):
        return Result([], [list(reversed(_items(value)))])

    def tuple_sort(value, *, ctx):
        src = _items(value)
        if any(isinstance(item, str) for item in src):
            if not all(isinstance(item, str) for item in src):
                raise ValueError("字串和數值不能排在同一個 tuple")
            return Result([], [sorted(src)])
        return Result([], [sorted(src, key=_num)])

    def tuple_sort_index(value, *, ctx):
        src = _items(value)
        if any(isinstance(item, str) for item in src):
            order = sorted(range(len(src)), key=lambda pos: src[pos])
        else:
            order = sorted(range(len(src)), key=lambda pos: _num(src[pos]))
        return Result([], [order])

    def tuple_uniq(value, *, ctx):
        out = []
        for item in _items(value):
            if not any(_same(item, old) for old in out):
                out.append(item)
        return Result([], [out])

    def _set_op(left, right, mode):
        first, second = _items(left), _items(right)
        if mode == "union":
            return tuple_uniq(first + second, ctx=None).control[0]
        if mode == "intersection":
            return [item for item in first if any(_same(item, other) for other in second)]
        if mode == "difference":
            return [item for item in first if not any(_same(item, other) for other in second)]
        extra = [item for item in second if not any(_same(item, other) for other in first)]
        return [item for item in first if not any(_same(item, other) for other in second)] + extra

    def tuple_concat(left, right, *, ctx):
        return Result([], [_items(left) + _items(right)])

    def tuple_repeat(value, num, *, ctx):
        count = _as_int(num)
        if count < 0:
            raise ValueError("Num 不能是負的")
        return Result([], [_items(value) * count])

    def tuple_repeat_elem(value, num, *, ctx):
        count = _as_int(num)
        if count < 0:
            raise ValueError("Num 不能是負的")
        return Result([], [[item for item in _items(value) for _ in range(count)]])

    def tuple_gen_const(length, const, *, ctx):
        count = _as_int(length)
        if count < 0:
            raise ValueError("Length 不能是負的")
        return Result([], [[const] * count])

    def tuple_gen_sequence(start, end, step, *, ctx):
        return Result([], [_sequence(start, end, step)])

    def tuple_cumul(value, *, ctx):
        total = 0
        out = []
        for item in _items(value):
            total += _num(item)
            out.append(_tidy(total))
        return Result([], [out])

    def tuple_median(value, *, ctx):
        src = sorted(_num(item) for item in _items(value))
        if not src:
            raise ValueError("空 tuple")
        mid = len(src) // 2
        if len(src) % 2:
            return Result([], [_tidy(src[mid])])
        return Result([], [_tidy((src[mid - 1] + src[mid]) / 2)])

    def tuple_deviation(value, *, ctx):
        src = [_num(item) for item in _items(value)]
        if not src:
            raise ValueError("空 tuple")
        mean = sum(src) / len(src)
        var = sum((item - mean) ** 2 for item in src) / len(src)
        return Result([], [_tidy(math.sqrt(var))])

    def tuple_histo_range(value, low, high, num_bins, *, ctx):
        bins = _as_int(num_bins)
        if bins < 1:
            raise ValueError("NumBins 至少要 1")
        start, stop = _num(low), _num(high)
        if stop <= start:
            raise ValueError("Max 必須大於 Min")
        width = (stop - start) / bins
        histo = [0] * bins
        for item in _items(value):
            number = _num(item)
            if number < start or number > stop:
                continue
            slot = bins - 1 if number == stop else int((number - start) / width)
            histo[slot] += 1
        return Result([], [histo, _tidy(width)])

    def tuple_rand(length, *, ctx):
        count = _as_int(length)
        if count < 0:
            raise ValueError("Length 不能是負的")
        return Result([], [[random.random() for _ in range(count)]])

    select_note = "索引從 0 開始，負數從尾巴數。程式裡的 T[1] 仍從 1 開始，兩套不要混用。"
    specs = [
        ("tuple_select", [P("Tuple", "輸入 tuple"), P("Index", "索引，可以是好幾個")], [P("Selected", "選到的元素")], "依索引取出元素", select_note, "tuple_select ([10, 20, 30], 0, Selected)", tuple_select),
        ("tuple_at", [P("Tuple", "輸入 tuple"), P("Index", "索引")], [P("Selected", "選到的元素")], "依索引取出元素", "和 tuple_select 相同。" + select_note, "tuple_at ([10, 20], 1, Selected)", tuple_select),
        ("tuple_remove", [P("Tuple", "輸入 tuple"), P("Index", "要刪掉的索引")], [P("Reduced", "刪掉之後的 tuple")], "刪掉指定索引", select_note, "tuple_remove ([10, 20, 30], 1, Reduced)", tuple_remove),
        ("tuple_find", [P("Tuple", "輸入 tuple"), P("ToFind", "要找的值")], [P("Indices", "索引，找不到是 -1")], "找出所有相同元素的索引", "索引從 0 開始。一個都沒有時回傳 -1。", "tuple_find ([4, 8, 4], 4, Indices)", tuple_find),
        ("tuple_find_first", [P("Tuple", "輸入 tuple"), P("ToFind", "要找的值")], [P("Index", "第一個索引，找不到是 -1")], "找出第一個相同元素", "索引從 0 開始。", "tuple_find_first ([4, 8, 4], 4, Index)", lambda value, to_find, *, ctx: tuple_find_edge(value, to_find, ctx=ctx, reverse=False)),
        ("tuple_find_last", [P("Tuple", "輸入 tuple"), P("ToFind", "要找的值")], [P("Index", "最後一個索引，找不到是 -1")], "找出最後一個相同元素", "索引從 0 開始。", "tuple_find_last ([4, 8, 4], 4, Index)", lambda value, to_find, *, ctx: tuple_find_edge(value, to_find, ctx=ctx, reverse=True)),
        ("tuple_first_n", [P("Tuple", "輸入 tuple"), P("Index", "結束索引，含自己")], [P("Selected", "從開頭到這裡")], "取開頭到指定索引", select_note, "tuple_first_n ([10, 20, 30], 1, Selected)", tuple_first_n),
        ("tuple_last_n", [P("Tuple", "輸入 tuple"), P("Index", "開始索引，含自己")], [P("Selected", "從這裡到結尾")], "取指定索引到結尾", select_note, "tuple_last_n ([10, 20, 30], 1, Selected)", tuple_last_n),
        ("tuple_select_range", [P("Tuple", "輸入 tuple"), P("Left", "開始索引"), P("Right", "結束索引")], [P("Selected", "這一段")], "取一段連續元素", "左右都包含。索引從 0 開始。Right 比 Left 小就得到空 tuple。", "tuple_select_range ([10, 20, 30, 40], 1, 2, Selected)", tuple_select_range),
        ("tuple_select_mask", [P("Tuple", "輸入 tuple"), P("Mask", "0 丟掉，非 0 留下")], [P("Selected", "留下的元素")], "用遮罩挑選元素", "Mask 和 Tuple 長度要相同，或其中一邊長度是 1。", "tuple_select_mask ([10, 20, 30], [1, 0, 1], Selected)", tuple_select_mask),
        ("tuple_select_rank", [P("Tuple", "輸入 tuple"), P("RankIndex", "排序後的名次，從 0")], [P("Selected", "那個名次的值")], "取排序後的第幾名", "0 是最小，最後一個名次是最大。", "tuple_select_rank ([8, 1, 5], 0, Selected)", tuple_select_rank),
        ("tuple_insert", [P("Tuple", "原 tuple"), P("Value", "要插入的值"), P("Index", "插入位置")], [P("Inserted", "插入後的 tuple")], "在指定位置插入", "原本那個位置的元素往後移。Index 等於長度就是加在尾巴。索引從 0 開始。", "tuple_insert ([10, 30], 20, 1, Inserted)", tuple_insert),
        ("tuple_replace", [P("Tuple", "原 tuple"), P("Value", "新值"), P("Index", "要換掉的索引")], [P("Replaced", "換過的 tuple")], "換掉指定位置的值", select_note, "tuple_replace ([10, 20, 30], 99, 1, Replaced)", tuple_replace),
        ("tuple_inverse", _ONE, [P("Inverted", "倒過來的 tuple")], "順序倒過來", "第一個變成最後一個。", "tuple_inverse ([1, 2, 3], Inverted)", tuple_inverse),
        ("tuple_sort", _ONE, [P("Sorted", "由小到大")], "由小到大排序", "數值和字串要分開排，不能混在同一個 tuple。", "tuple_sort ([3, 1, 2], Sorted)", tuple_sort),
        ("tuple_sort_index", _ONE, [P("Indices", "排序後的原索引")], "排序後的原索引", "索引從 0 開始。數值由小到大，字串由字典順序。", "tuple_sort_index ([3, 1, 2], Indices)", tuple_sort_index),
        ("tuple_uniq", _ONE, [P("Uniq", "去掉重複後")], "去掉重複並保留第一次出現的順序", "1 和 1.0 視為同一個值。", "tuple_uniq ([1, 2, 1, 3], Uniq)", tuple_uniq),
        ("tuple_concat", _TWO, [P("Concat", "接起來")], "前後相接", "第二個 tuple 接在第一個後面。", "tuple_concat ([1, 2], [3], Concat)", tuple_concat),
        ("tuple_repeat", [P("Tuple", "輸入 tuple"), P("Num", "重複次數")], [P("Repeated", "重複後")], "整個 tuple 重複數次", "Num 是 0 就得到空 tuple。", "tuple_repeat ([1, 2], 2, Repeated)", tuple_repeat),
        ("tuple_repeat_elem", [P("Tuple", "輸入 tuple"), P("Num", "每個元素重複次數")], [P("Repeated", "重複後")], "每個元素各重複數次", "Num 是 0 就得到空 tuple。", "tuple_repeat_elem ([1, 2], 2, Repeated)", tuple_repeat_elem),
        ("tuple_gen_const", [P("Length", "長度"), P("Const", "填進去的值")], [P("Newtuple", "新 tuple")], "產生長度固定、內容相同的 tuple", "Length 是 0 就得到空 tuple。", "tuple_gen_const (3, 7, Newtuple)", tuple_gen_const),
        ("tuple_constant", [P("Length", "長度"), P("Const", "填進去的值")], [P("Newtuple", "新 tuple")], "產生長度固定、內容相同的 tuple", "和 tuple_gen_const 相同。", "tuple_constant (3, 7, Newtuple)", tuple_gen_const),
        ("tuple_gen_sequence", [P("Start", "起點"), P("End", "終點，包含"), P("Step", "間隔")], [P("Sequence", "數列")], "產生等差數列", "終點包含在內。Step 的方向和起終點相反時得到空 tuple。", "tuple_gen_sequence (1, 5, 2, Sequence)", tuple_gen_sequence),
        ("tuple_cumul", _ONE, [P("Cumul", "累加")], "由左到右累加", "每個位置是從頭加到那裡的和。", "tuple_cumul ([1, 2, 3], Cumul)", tuple_cumul),
        ("tuple_median", _ONE, [P("Median", "中位數")], "中位數", "偶數個時取中間兩個的平均。", "tuple_median ([1, 9, 3], Median)", tuple_median),
        ("tuple_deviation", _ONE, [P("Deviation", "母體標準差")], "母體標準差", "先算平均，再把平方差的平均開根號。除以個數，不是除以個數減 1。", "tuple_deviation ([1, 2, 3], Deviation)", tuple_deviation),
        ("tuple_histo_range", [P("Tuple", "數值"), P("Min", "下界，含"), P("Max", "上界，含"), P("NumBins", "箱子數")], [P("Histo", "每個箱子的個數"), P("BinSize", "箱子寬度")], "把數值分箱計數", "小於 Min 或大於 Max 的值不計。剛好等於 Max 的值放進最後一箱。", "tuple_histo_range ([1, 2, 9], 0, 10, 2, Histo, BinSize)", tuple_histo_range),
        ("tuple_rand", [P("Length", "要幾個亂數")], [P("Rand", "0 到 1 之間的亂數")], "產生 0 到 1 的亂數", "每次執行的結果不同，不包含 1。", "tuple_rand (4, Rand)", tuple_rand),
    ]
    for name, cin, cout, short, detail, example, fn in specs:
        _bind(name, cin, cout, short, detail, example, fn)
    for name, mode, short in (
        ("tuple_union", "union", "聯集，保留先出現的順序"),
        ("tuple_intersection", "intersection", "交集，順序跟第一個 tuple"),
        ("tuple_difference", "difference", "差集，留下第一個有而第二個沒有的"),
        ("tuple_symmdiff", "symmdiff", "對稱差，只留一邊有的"),
    ):
        _bind(
            name,
            _TWO,
            [P("Result", short)],
            short,
            "1 和 1.0 視為同一個值。結果保留第一次出現的順序。",
            f"{name} ([1, 2, 3], [2, 4], Result)",
            lambda left, right, *, ctx, _mode=mode: Result([], [_set_op(left, right, _mode)]),
        )


def _register_text():
    def tuple_string(value, fmt, *, ctx):
        out = [_format_one(item, fmt) for item in _items(value)]
        return _finish(out, _listed(value))

    def tuple_number(value, *, ctx):
        out = []
        for item in _items(value):
            text = str(item).strip()
            try:
                number = float(text)
            except ValueError as exc:
                raise ValueError(f"不是數值：{item}") from exc
            out.append(_tidy(number))
        return _finish(out, _listed(value))

    def tuple_split(value, separator, *, ctx):
        parts = []
        for item in _items(value):
            parts.extend(str(item).split(str(separator)))
        return Result([], [parts])

    def tuple_join(value, separator, *, ctx):
        return Result([], [str(separator).join(str(item) for item in _items(value))])

    def tuple_strlen(value, *, ctx):
        out = [len(str(item)) for item in _items(value)]
        return _finish(out, _listed(value))

    def tuple_substr(value, position1, position2, *, ctx):
        texts, starts = _align(value, position1)
        _, ends = _align(value, position2)
        if len(starts) != len(ends):
            starts, ends = _align(position1, position2)
            texts = _items(value)
            if len(texts) == 1:
                texts = texts * len(starts)
        out = [_slice_text(str(text), start, end) for text, start, end in zip(texts, starts, ends)]
        return _finish(out, _listed(value))

    def tuple_str_first_n(value, position, *, ctx):
        texts, positions = _align(value, position)
        out = [_slice_text(str(text), 0, pos) for text, pos in zip(texts, positions)]
        return _finish(out, _listed(value))

    def tuple_str_last_n(value, position, *, ctx):
        texts, positions = _align(value, position)
        out = []
        for text, pos in zip(texts, positions):
            raw = str(text)
            start = _as_int(pos)
            if start < 0:
                start += len(raw)
            out.append("" if start >= len(raw) else raw[max(0, start) :])
        return _finish(out, _listed(value))

    def tuple_chr(value, *, ctx):
        out = [chr(_as_int(item)) for item in _items(value)]
        return _finish(out, _listed(value))

    def tuple_chrt(value, *, ctx):
        return Result([], ["".join(chr(_as_int(item)) for item in _items(value))])

    def tuple_ord(value, *, ctx):
        out = []
        for item in _items(value):
            text = str(item)
            if text == "":
                raise ValueError("空字串沒有字元碼")
            out.append(ord(text[0]))
        return _finish(out, _listed(value))

    def tuple_ords(value, *, ctx):
        codes = [ord(ch) for item in _items(value) for ch in str(item)]
        return Result([], [codes])

    def _str_index(value, needle, reverse, *, ctx):
        texts, needles = _align(value, needle)
        out = [_find_text(str(text), str(part), reverse) for text, part in zip(texts, needles)]
        return _finish(out, _listed(value) or _listed(needle))

    def tuple_str_replace(value, pattern, replace, *, ctx):
        texts, patterns = _align(value, pattern)
        _, replacements = _align(value, replace)
        if len(patterns) != len(replacements):
            patterns, replacements = _align(pattern, replace)
            texts = _items(value) * len(patterns) if len(_items(value)) == 1 else _items(value)
        out = [str(text).replace(str(old), str(new)) for text, old, new in zip(texts, patterns, replacements)]
        return _finish(out, _listed(value))

    def tuple_str_distance(left, right, *, ctx):
        first, second = _align(left, right)
        out = [_distance(str(a), str(b)) for a, b in zip(first, second)]
        return _finish(out, _listed(left) or _listed(right))

    def _regexp(value, expression, *, ctx, mode, replace=""):
        pattern = re.compile(str(expression))
        texts = [str(item) for item in _items(value)]
        if mode == "test":
            out = [1 if pattern.search(text) else 0 for text in texts]
            return _finish(out, _listed(value))
        if mode == "match":
            out = [(pattern.search(text).group(0) if pattern.search(text) else "") for text in texts]
            return _finish(out, _listed(value))
        if mode == "replace":
            out = [pattern.sub(str(replace), text) for text in texts]
            return _finish(out, _listed(value))
        return Result([], [[text for text in texts if pattern.search(text)]])

    def tuple_is(value, kind, *, ctx):
        src = _items(value)
        if kind == "mixed":
            kinds = {_kind(item) for item in src}
            return Result([], [1 if len(kinds) > 1 else 0])
        if kind == "number":
            return Result([], [1 if src and all(_kind(item) in ("integer", "real") for item in src) else 0])
        return Result([], [1 if src and all(_kind(item) == kind for item in src) else 0])

    def tuple_is_elem(value, kind, *, ctx):
        out = []
        for item in _items(value):
            if kind == "nan":
                out.append(1 if isinstance(item, float) and math.isnan(item) else 0)
            elif kind == "number":
                out.append(1 if _kind(item) in ("integer", "real") else 0)
            else:
                out.append(1 if _kind(item) == kind else 0)
        return _finish(out, True)

    def tuple_type(value, *, ctx):
        src = _items(value)
        if not src:
            return Result([], ["empty"])
        kinds = {_kind(item) for item in src}
        if len(kinds) > 1:
            return Result([], ["mixed"])
        return Result([], [kinds.pop()])

    def tuple_type_elem(value, *, ctx):
        return Result([], [[_kind(item) for item in _items(value)]])

    text_in = [P("String", "字串或字串 tuple")]
    specs = [
        ("tuple_string", [P("Tuple", "要格式化的值"), P("Format", "d、.2f、g，或 % 開頭的格式")], [P("String", "格式化後的字串")], "依格式把值轉成字串", "d 是整數，.2f 是兩位小數。tuple 的每個元素各轉一次。", "tuple_string ([1.5, 2], '.1f', String)", tuple_string),
        ("tuple_number", text_in, [P("Number", "數值")], "把字串轉成數值", "轉不出來會報錯。整數會收成整數。", "tuple_number (['12', '1.5'], Number)", tuple_number),
        ("tuple_split", [P("String", "要切開的字串"), P("Separator", "分隔字串")], [P("Substrings", "切出來的片段")], "用分隔字串切開", "不是正規表達式。字串 tuple 會把所有片段接成一個結果。", "tuple_split ('a,b,c', ',', Substrings)", tuple_split),
        ("tuple_join", [P("Tuple", "要接起來的值"), P("Separator", "中間的分隔字串")], [P("String", "接好的字串")], "用分隔字串接起來", "每個元素先轉成字串。", "tuple_join ([1, 2, 3], '-', String)", tuple_join),
        ("tuple_strlen", text_in, [P("Length", "字元數")], "字串長度", "字串 tuple 會各算一次。", "tuple_strlen ('HALCON', Length)", tuple_strlen),
        ("tuple_substr", [P("String", "字串"), P("Position1", "開始，從 0，含"), P("Position2", "結束，從 0，含")], [P("Substring", "子字串")], "取子字串", "索引從 0 開始，負數從尾巴數。結束比開始小就得到空字串。", "tuple_substr ('HALCON', 0, 2, Substring)", tuple_substr),
        ("tuple_str_first_n", [P("String", "字串"), P("Position", "結束位置，含")], [P("Substring", "開頭到這裡")], "取字串開頭", "索引從 0 開始。", "tuple_str_first_n ('HALCON', 2, Substring)", tuple_str_first_n),
        ("tuple_str_last_n", [P("String", "字串"), P("Position", "開始位置，含")], [P("Substring", "這裡到結尾")], "取字串尾巴", "索引從 0 開始。", "tuple_str_last_n ('HALCON', 3, Substring)", tuple_str_last_n),
        ("tuple_chr", [P("Int", "字元碼")], [P("String", "一個字元")], "字元碼轉成一個字元", "tuple 的每個碼各轉一個字元。", "tuple_chr (65, Char)", tuple_chr),
        ("tuple_chrt", [P("Int", "字元碼 tuple")], [P("String", "接成一個字串")], "字元碼 tuple 接成一個字串", "65 和 66 會得到 AB。", "tuple_chrt ([65, 66], String)", tuple_chrt),
        ("tuple_ord", text_in, [P("Ord", "第一個字元的碼")], "取第一個字元的碼", "空字串會報錯。字串 tuple 各取第一個字元。", "tuple_ord ('A', Ord)", tuple_ord),
        ("tuple_ords", text_in, [P("Ords", "每個字元的碼")], "取出每個字元的碼", "多個字串會把字元碼接成一個 tuple。", "tuple_ords ('AB', Ords)", tuple_ords),
        ("tuple_strchr", [P("String", "字串"), P("Char", "要找的字元")], [P("Position", "位置，找不到是 -1")], "找字元第一次出現的位置", "索引從 0 開始。", "tuple_strchr ('HALCON', 'C', Position)", lambda value, needle, *, ctx: _str_index(value, needle, False, ctx=ctx)),
        ("tuple_strrchr", [P("String", "字串"), P("Char", "要找的字元")], [P("Position", "位置，找不到是 -1")], "找字元最後一次出現的位置", "索引從 0 開始。", "tuple_strrchr ('HALCON', 'A', Position)", lambda value, needle, *, ctx: _str_index(value, needle, True, ctx=ctx)),
        ("tuple_strstr", [P("String", "字串"), P("Search", "要找的片段")], [P("Position", "位置，找不到是 -1")], "找片段第一次出現的位置", "索引從 0 開始。", "tuple_strstr ('HALCON', 'LC', Position)", lambda value, needle, *, ctx: _str_index(value, needle, False, ctx=ctx)),
        ("tuple_strrstr", [P("String", "字串"), P("Search", "要找的片段")], [P("Position", "位置，找不到是 -1")], "找片段最後一次出現的位置", "索引從 0 開始。", "tuple_strrstr ('aaa', 'a', Position)", lambda value, needle, *, ctx: _str_index(value, needle, True, ctx=ctx)),
        ("tuple_str_replace", [P("String", "字串"), P("Search", "要換掉的文字"), P("Replace", "換成")], [P("Result", "換過的字串")], "替換文字", "換掉所有出現的地方。這不是正規表達式。", "tuple_str_replace ('a-b-a', '-', '/', Result)", tuple_str_replace),
        ("tuple_str_distance", [P("String1", "第一個字串"), P("String2", "第二個字串")], [P("Distance", "編輯距離")], "字串編輯距離", "插入、刪除、替換各算 1。長度 1 的那一邊會套用到每個字串。", "tuple_str_distance ('kitten', 'sitting', Distance)", tuple_str_distance),
        ("tuple_regexp_test", [P("String", "字串"), P("Expression", "正規表達式")], [P("NumMatches", "有對到是 1")], "正規表達式是否對到", "用 Python 的正規表達式。有對到就是 1。", "tuple_regexp_test ('VisDev', 'Vis', NumMatches)", lambda value, expression, *, ctx: _regexp(value, expression, ctx=ctx, mode="test")),
        ("tuple_regexp_match", [P("String", "字串"), P("Expression", "正規表達式")], [P("Match", "對到的文字")], "取出正規表達式對到的文字", "沒對到就是空字串。只取第一段。", "tuple_regexp_match ('ab12', '[0-9]+', Match)", lambda value, expression, *, ctx: _regexp(value, expression, ctx=ctx, mode="match")),
        ("tuple_regexp_replace", [P("String", "字串"), P("Expression", "正規表達式"), P("Replace", "換成")], [P("Result", "換過的字串")], "用正規表達式替換", "換成的文字可以用 \\1 表示第一個括號。", "tuple_regexp_replace ('ab12', '[0-9]+', '#', Result)", lambda value, expression, replace, *, ctx: _regexp(value, expression, ctx=ctx, mode="replace", replace=replace)),
        ("tuple_regexp_select", [P("String", "字串 tuple"), P("Expression", "正規表達式")], [P("Selection", "有對到的字串")], "留下對到正規表達式的字串", "沒有對到就得到空 tuple。", "tuple_regexp_select (['a', 'b2'], '[0-9]', Selection)", lambda value, expression, *, ctx: _regexp(value, expression, ctx=ctx, mode="select")),
        ("tuple_is_int", _ONE, [P("IsInt", "全部是整數才是 1")], "是不是全部都是整數", "有一個浮點數或字串就是 0。空 tuple 是 0。", "tuple_is_int ([1, 2], IsInt)", lambda value, *, ctx: tuple_is(value, "integer", ctx=ctx)),
        ("tuple_is_int_elem", _ONE, [P("IsInt", "每個元素 0 或 1")], "每個元素是不是整數", "浮點數即使值是整數也算實數。", "tuple_is_int_elem ([1, 1.5], IsInt)", lambda value, *, ctx: tuple_is_elem(value, "integer", ctx=ctx)),
        ("tuple_is_real", _ONE, [P("IsReal", "全部是浮點數才是 1")], "是不是全部都是浮點數", "整數不算浮點數。", "tuple_is_real (1.5, IsReal)", lambda value, *, ctx: tuple_is(value, "real", ctx=ctx)),
        ("tuple_is_real_elem", _ONE, [P("IsReal", "每個元素 0 或 1")], "每個元素是不是浮點數", "整數的位置是 0。", "tuple_is_real_elem ([1, 1.5], IsReal)", lambda value, *, ctx: tuple_is_elem(value, "real", ctx=ctx)),
        ("tuple_is_string", _ONE, [P("IsString", "全部是字串才是 1")], "是不是全部都是字串", "有一個數值就是 0。", "tuple_is_string ('abc', IsString)", lambda value, *, ctx: tuple_is(value, "string", ctx=ctx)),
        ("tuple_is_string_elem", _ONE, [P("IsString", "每個元素 0 或 1")], "每個元素是不是字串", "數值的位置是 0。", "tuple_is_string_elem (['a', 1], IsString)", lambda value, *, ctx: tuple_is_elem(value, "string", ctx=ctx)),
        ("tuple_is_number", _ONE, [P("IsNumber", "全部是數值才是 1")], "是不是全部都是數值", "整數和浮點數都算。有字串就是 0。", "tuple_is_number ([1, 1.5], IsNumber)", lambda value, *, ctx: tuple_is(value, "number", ctx=ctx)),
        ("tuple_is_mixed", _ONE, [P("IsMixed", "有兩種以上型別才是 1")], "裡面是不是混了不同型別", "整數和浮點數算不同型別。", "tuple_is_mixed ([1, 'a'], IsMixed)", lambda value, *, ctx: tuple_is(value, "mixed", ctx=ctx)),
        ("tuple_is_nan_elem", _ONE, [P("IsNaN", "每個元素 0 或 1")], "每個元素是不是 NaN", "只有浮點 NaN 是 1。", "tuple_is_nan_elem ([1, 2], IsNaN)", lambda value, *, ctx: tuple_is_elem(value, "nan", ctx=ctx)),
        ("tuple_type", _ONE, [P("Type", "integer、real、string、mixed 或 empty")], "tuple 的型別", "全部同一種才回傳那種名字。空 tuple 是 empty。", "tuple_type ([1, 2], Type)", tuple_type),
        ("tuple_type_elem", _ONE, [P("Type", "每個元素的型別")], "每個元素的型別", "名字是 integer、real 或 string。", "tuple_type_elem ([1, 'a'], Type)", tuple_type_elem),
    ]
    for name, cin, cout, short, detail, example, fn in specs:
        _bind(name, cin, cout, short, detail, example, fn)


_register_math()
_register_select()
_register_text()

_missing = HALCON_TUPLE - EXISTING - SKIPPED - IMPLEMENTED
_extra = IMPLEMENTED - HALCON_TUPLE
if _missing or _extra:
    raise RuntimeError(f"tuple 章節和 HALCON 名單不一致 missing={sorted(_missing)} extra={sorted(_extra)}")
