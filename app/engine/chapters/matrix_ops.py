"""Matrix handles backed by NumPy.

Indices start at 0, the same as HALCON matrices. Values are stored row by row.
A matrix is a handle: set_* operators change that handle, and clear_matrix
retires it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..objects import Result
from ..operators import P, as_list, operator, scalar, whole

_MULT = ["'AB'", "'ATB'", "'ABT'", "'ATBT'"]
_MATRIX_TYPE = ["'general'", "'symmetric'", "'positive_definite'"]
_SVD_TYPE = ["'full'", "'reduced'"]
_QR_TYPE = ["'full'", "'reduced'"]
_NORM = ["'frobenius'", "'1'", "'2'", "'infinity'"]
_REDUCE = ["'rows'", "'columns'", "'all'"]


@dataclass
class Matrix:
    data: np.ndarray
    closed: bool = False

    def summary(self) -> str:
        if self.closed:
            return "矩陣  已關閉"
        rows, cols = self.data.shape
        return f"矩陣  {rows}×{cols}"


def _tidy(value: float):
    if isinstance(value, (int, np.integer)):
        return int(value)
    number = float(value)
    if math.isfinite(number) and abs(number - round(number)) < 1e-8:
        return int(round(number))
    return number


def _token(value, name: str) -> str:
    return str(value).strip().strip("'\"").lower()


def _expect(value) -> Matrix:
    if not isinstance(value, Matrix):
        raise TypeError("MatrixID 必須是 create_matrix 建立的矩陣")
    if value.closed:
        raise RuntimeError("矩陣已經 clear_matrix")
    return value


def _wrap(array) -> Matrix:
    data = np.asarray(array, np.float64)
    if data.ndim != 2:
        data = data.reshape(-1, 1)
    return Matrix(data)


def _numbers(value, name: str) -> list[float]:
    try:
        return [float(item) for item in as_list(value)]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 需要數值") from exc


def _same_shape(left: Matrix, right: Matrix, what: str) -> None:
    if left.data.shape != right.data.shape:
        raise ValueError(f"{what}：兩個矩陣必須一樣大，現在是 {left.data.shape} 和 {right.data.shape}")


def _reduce(data: np.ndarray, kind: str, fn) -> np.ndarray:
    if kind == "rows":
        return fn(data, axis=1, keepdims=True)
    if kind == "columns":
        return fn(data, axis=0, keepdims=True)
    if kind == "all":
        return np.array([[fn(data)]], np.float64)
    raise ValueError("型別只能是 rows、columns 或 all")


@operator(
    "矩陣",
    [],
    [],
    [
        P("Rows", "列數"),
        P("Columns", "行數"),
        P("Value", "一個數字會填滿。多個數字依列排，長度必須是列數乘行數"),
    ],
    [P("MatrixID", "矩陣")],
    "建立矩陣",
    "索引從 0 開始。數值按列排：第一列從左到右，再排下一列。",
    "Vizion.create_matrix (2, 2, [1, 2, 3, 4], MatrixID)",
)
def create_matrix(rows, columns, value, *, ctx):
    height = whole(rows, "Rows")
    width = whole(columns, "Columns")
    if height < 1 or width < 1:
        raise ValueError("Rows 和 Columns 至少要 1")
    nums = _numbers(value, "Value")
    if len(nums) == 1:
        data = np.full((height, width), nums[0], np.float64)
    elif len(nums) == height * width:
        data = np.array(nums, np.float64).reshape(height, width)
    else:
        raise ValueError(f"Value 要 1 個或 {height * width} 個數字，現在是 {len(nums)} 個")
    return Result([], [_wrap(data)])


@operator(
    "矩陣",
    [],
    [],
    [P("MatrixID", "要關閉的矩陣")],
    [],
    "關閉矩陣",
    "關閉後不能再讀寫。變數還在。",
    "Vizion.clear_matrix (MatrixID)",
)
def clear_matrix(matrix, *, ctx):
    handle = _expect(matrix)
    handle.closed = True
    handle.data = np.zeros((0, 0), np.float64)
    return Result([], [])


@operator(
    "矩陣",
    [],
    [],
    [P("MatrixID", "矩陣")],
    [P("Rows", "列數"), P("Columns", "行數")],
    "讀取矩陣大小",
    "回傳列數和行數。",
    "Vizion.get_size_matrix (MatrixID, Rows, Columns)",
)
def get_size_matrix(matrix, *, ctx):
    data = _expect(matrix).data
    return Result([], [int(data.shape[0]), int(data.shape[1])])


@operator(
    "矩陣",
    [],
    [],
    [P("MatrixID", "矩陣")],
    [P("Values", "依列排開的數字")],
    "把矩陣讀成 tuple",
    "順序和 create_matrix 一樣，先左後右、先上後下。",
    "Vizion.get_full_matrix (MatrixID, Values)",
)
def get_full_matrix(matrix, *, ctx):
    data = _expect(matrix).data
    return Result([], [[_tidy(item) for item in data.reshape(-1)]])


@operator(
    "矩陣",
    [],
    [],
    [P("MatrixID", "矩陣"), P("Values", "依列排開的數字")],
    [],
    "整批寫入矩陣",
    "個數必須等於列數乘行數。寫進原來的矩陣。",
    "Vizion.set_full_matrix (MatrixID, [1, 0, 0, 1])",
)
def set_full_matrix(matrix, values, *, ctx):
    handle = _expect(matrix)
    nums = _numbers(values, "Values")
    needed = int(handle.data.size)
    if len(nums) != needed:
        raise ValueError(f"Values 需要 {needed} 個數字，現在是 {len(nums)} 個")
    handle.data = np.array(nums, np.float64).reshape(handle.data.shape)
    return Result([], [])


@operator(
    "矩陣",
    [],
    [],
    [P("MatrixID", "矩陣"), P("Row", "列，從 0 開始"), P("Column", "行，從 0 開始")],
    [P("Value", "元素")],
    "讀一個元素",
    "Row 和 Column 可以是等長 tuple，一次讀多個。",
    "Vizion.get_value_matrix (MatrixID, 0, 1, Value)",
)
def get_value_matrix(matrix, row, column, *, ctx):
    data = _expect(matrix).data
    rows = as_list(row)
    cols = as_list(column)
    if len(rows) == 1 and len(cols) > 1:
        rows = rows * len(cols)
    if len(cols) == 1 and len(rows) > 1:
        cols = cols * len(rows)
    if len(rows) != len(cols):
        raise ValueError("Row 和 Column 的長度要相同")
    found = []
    for item_row, item_col in zip(rows, cols):
        rr = int(round(float(item_row)))
        cc = int(round(float(item_col)))
        if rr < 0 or cc < 0 or rr >= data.shape[0] or cc >= data.shape[1]:
            raise IndexError(f"元素 ({rr}, {cc}) 超出 {data.shape[0]}×{data.shape[1]}")
        found.append(_tidy(data[rr, cc]))
    if len(found) == 1 and not isinstance(row, (list, tuple)) and not isinstance(column, (list, tuple)):
        return Result([], [found[0]])
    return Result([], [found])


@operator(
    "矩陣",
    [],
    [],
    [
        P("MatrixID", "矩陣"),
        P("Row", "列，從 0 開始"),
        P("Column", "行，從 0 開始"),
        P("Value", "要寫入的數字"),
    ],
    [],
    "寫一個元素",
    "多個位置時 Value 也要等長。寫進原來的矩陣。",
    "Vizion.set_value_matrix (MatrixID, 0, 0, 1)",
)
def set_value_matrix(matrix, row, column, value, *, ctx):
    handle = _expect(matrix)
    rows = as_list(row)
    cols = as_list(column)
    values = _numbers(value, "Value")
    if len(rows) == 1 and len(cols) > 1:
        rows = rows * len(cols)
    if len(values) == 1 and max(len(rows), len(cols)) > 1:
        values = values * max(len(rows), len(cols))
    if not (len(rows) == len(cols) == len(values)):
        raise ValueError("Row、Column、Value 的長度要相同")
    for item_row, item_col, item in zip(rows, cols, values):
        rr = int(round(float(item_row)))
        cc = int(round(float(item_col)))
        if rr < 0 or cc < 0 or rr >= handle.data.shape[0] or cc >= handle.data.shape[1]:
            raise IndexError(f"元素 ({rr}, {cc}) 超出範圍")
        handle.data[rr, cc] = item
    return Result([], [])


@operator(
    "矩陣",
    [],
    [],
    [
        P("MatrixID", "矩陣"),
        P("Row", "子矩陣的左上列，從 0 開始"),
        P("Column", "子矩陣的左上行，從 0 開始"),
        P("Rows", "子矩陣列數"),
        P("Columns", "子矩陣行數"),
    ],
    [P("MatrixSubID", "子矩陣")],
    "取出子矩陣",
    "含左上角，往右往下數 Rows 列、Columns 行。",
    "Vizion.get_sub_matrix (MatrixID, 0, 0, 2, 2, MatrixSubID)",
)
def get_sub_matrix(matrix, row, column, rows, columns, *, ctx):
    data = _expect(matrix).data
    rr = whole(row, "Row")
    cc = whole(column, "Column")
    height = whole(rows, "Rows")
    width = whole(columns, "Columns")
    if height < 1 or width < 1 or rr < 0 or cc < 0 or rr + height > data.shape[0] or cc + width > data.shape[1]:
        raise ValueError("子矩陣超出範圍")
    return Result([], [_wrap(data[rr : rr + height, cc : cc + width].copy())])


@operator(
    "矩陣",
    [],
    [],
    [
        P("MatrixID", "要寫入的矩陣"),
        P("MatrixSubID", "來源矩陣"),
        P("Row", "寫入位置的列"),
        P("Column", "寫入位置的行"),
    ],
    [],
    "寫入子矩陣",
    "來源矩陣必須放得進目標矩陣。",
    "Vizion.set_sub_matrix (MatrixID, MatrixSubID, 0, 0)",
)
def set_sub_matrix(matrix, sub, row, column, *, ctx):
    handle = _expect(matrix)
    block = _expect(sub).data
    rr = whole(row, "Row")
    cc = whole(column, "Column")
    if rr < 0 or cc < 0 or rr + block.shape[0] > handle.data.shape[0] or cc + block.shape[1] > handle.data.shape[1]:
        raise ValueError("子矩陣超出範圍")
    handle.data[rr : rr + block.shape[0], cc : cc + block.shape[1]] = block
    return Result([], [])


@operator(
    "矩陣",
    [],
    [],
    [P("MatrixID", "矩陣"), P("Diagonal", "0 是主對角線，正數往右", "0")],
    [P("VectorID", "對角元素，排成一行")],
    "讀取對角線",
    "正的 Diagonal 是主對角線右上方。負的往左下方。",
    "Vizion.get_diagonal_matrix (MatrixID, 0, VectorID)",
)
def get_diagonal_matrix(matrix, diagonal, *, ctx):
    data = _expect(matrix).data
    offset = whole(diagonal, "Diagonal")
    diag = np.diag(data, offset)
    if diag.size == 0:
        raise ValueError("這條對角線是空的")
    return Result([], [_wrap(diag.reshape(-1, 1))])


@operator(
    "矩陣",
    [],
    [],
    [
        P("MatrixID", "矩陣"),
        P("Vector", "新的對角元素"),
        P("Diagonal", "0 是主對角線，正數往右", "0"),
    ],
    [],
    "寫入對角線",
    "元素個數必須和那條對角線一樣長。",
    "Vizion.set_diagonal_matrix (MatrixID, [1, 1], 0)",
)
def set_diagonal_matrix(matrix, vector, diagonal, *, ctx):
    handle = _expect(matrix)
    offset = whole(diagonal, "Diagonal")
    length = np.diag(handle.data, offset).size
    nums = _numbers(vector, "Vector")
    if len(nums) != length:
        raise ValueError(f"這條對角線需要 {length} 個數字")
    index = np.arange(length)
    rows = index + max(-offset, 0)
    cols = index + max(offset, 0)
    handle.data[rows, cols] = nums
    return Result([], [])


@operator(
    "矩陣",
    [],
    [],
    [P("MatrixID", "矩陣"), P("Rows", "向下重複幾次"), P("Columns", "向右重複幾次")],
    [P("MatrixRepeatedID", "重複後的矩陣")],
    "重複排矩陣",
    "Rows 和 Columns 是重複次數，不是最後的大小。",
    "Vizion.repeat_matrix (MatrixID, 2, 1, MatrixRepeatedID)",
)
def repeat_matrix(matrix, rows, columns, *, ctx):
    data = _expect(matrix).data
    down = whole(rows, "Rows")
    across = whole(columns, "Columns")
    if down < 1 or across < 1:
        raise ValueError("重複次數至少要 1")
    return Result([], [_wrap(np.tile(data, (down, across)))])


def _binary(name, short, detail, example, fn):
    def impl(left, right, *, ctx, _fn=fn):
        a = _expect(left)
        b = _expect(right)
        _same_shape(a, b, name)
        return Result([], [_wrap(_fn(a.data, b.data))])

    impl.__name__ = name
    return operator(
        "矩陣",
        [],
        [],
        [P("MatrixAID", "第一個矩陣"), P("MatrixBID", "第二個矩陣")],
        [P("MatrixResultID", "結果矩陣")],
        short,
        detail,
        example,
    )(impl)


_binary("add_matrix", "矩陣相加", "兩個矩陣必須一樣大。", "Vizion.add_matrix (A, B, Sum)", lambda a, b: a + b)
_binary("sub_matrix", "矩陣相減", "結果是 A - B。", "Vizion.sub_matrix (A, B, Difference)", lambda a, b: a - b)
_binary("mult_element_matrix", "元素相乘", "對應位置相乘，不是矩陣乘法。", "Vizion.mult_element_matrix (A, B, Product)", lambda a, b: a * b)


@operator(
    "矩陣",
    [],
    [],
    [P("MatrixAID", "被除數"), P("MatrixBID", "除數")],
    [P("MatrixDivID", "元素相除的結果")],
    "元素相除",
    "除數裡有 0 會失敗。",
    "Vizion.div_element_matrix (A, B, Quotient)",
)
def div_element_matrix(left, right, *, ctx):
    a = _expect(left)
    b = _expect(right)
    _same_shape(a, b, "div_element_matrix")
    if np.any(b.data == 0):
        raise ZeroDivisionError("除數矩陣裡有 0")
    return Result([], [_wrap(a.data / b.data)])


@operator(
    "矩陣",
    [],
    [],
    [P("MatrixID", "矩陣"), P("Factor", "倍率")],
    [P("MatrixScaledID", "縮放後的矩陣")],
    "矩陣乘上常數",
    "每個元素都乘 Factor。",
    "Vizion.scale_matrix (MatrixID, 2, MatrixScaledID)",
)
def scale_matrix(matrix, factor, *, ctx):
    return Result([], [_wrap(_expect(matrix).data * scalar(factor, "Factor"))])


@operator(
    "矩陣",
    [],
    [],
    [
        P("MatrixAID", "左邊的矩陣"),
        P("MatrixBID", "右邊的矩陣"),
        P("MultType", "AB、ATB、ABT 或 ATBT", "'AB'", _MULT),
    ],
    [P("MatrixMultID", "乘積")],
    "矩陣乘法",
    "AB 是 A×B。T 表示先轉置。A 的行數必須等於 B 的列數；轉置後也要對得起來。",
    "Vizion.mult_matrix (A, B, 'AB', Product)",
)
def mult_matrix(left, right, mult_type, *, ctx):
    a = _expect(left).data
    b = _expect(right).data
    kind = _token(mult_type, "MultType").upper()
    if kind == "ATB":
        a = a.T
    elif kind == "ABT":
        b = b.T
    elif kind == "ATBT":
        a, b = a.T, b.T
    elif kind != "AB":
        raise ValueError("MultType 只能是 AB、ATB、ABT 或 ATBT")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"矩陣乘法的內側尺寸不同：{a.shape} 和 {b.shape}")
    return Result([], [_wrap(a @ b)])


@operator(
    "矩陣",
    [],
    [],
    [P("MatrixID", "矩陣")],
    [P("MatrixTransposedID", "轉置")],
    "轉置",
    "列和行對調。",
    "Vizion.transpose_matrix (MatrixID, MatrixTransposedID)",
)
def transpose_matrix(matrix, *, ctx):
    return Result([], [_wrap(_expect(matrix).data.T.copy())])


@operator(
    "矩陣",
    [],
    [],
    [
        P("MatrixID", "方陣"),
        P("MatrixType", "general。symmetric 和 positive_definite 這版用同一套算法", "'general'", _MATRIX_TYPE),
        P("Epsilon", "大於 0 時改用偽逆，太小的奇異值會被丟掉", "0"),
    ],
    [P("MatrixInvertedID", "逆矩陣")],
    "矩陣求逆",
    "必須是方陣。Epsilon 為 0 時，奇異矩陣會失敗。大於 0 時用偽逆。",
    "Vizion.invert_matrix (MatrixID, 'general', 0, MatrixInvertedID)",
)
def invert_matrix(matrix, matrix_type, epsilon, *, ctx):
    data = _expect(matrix).data
    if data.shape[0] != data.shape[1]:
        raise ValueError("只有方陣可以求逆")
    kind = _token(matrix_type, "MatrixType")
    if kind not in ("general", "symmetric", "positive_definite"):
        raise ValueError("MatrixType 只能是 general、symmetric 或 positive_definite")
    tol = scalar(epsilon, "Epsilon")
    try:
        if tol > 0:
            inverse = np.linalg.pinv(data, rcond=tol)
        else:
            inverse = np.linalg.inv(data)
    except np.linalg.LinAlgError as exc:
        raise ValueError("這個矩陣不能求逆") from exc
    return Result([], [_wrap(inverse)])


@operator(
    "矩陣",
    [],
    [],
    [P("MatrixID", "方陣")],
    [P("Value", "行列式")],
    "行列式",
    "必須是方陣。",
    "Vizion.determinant_matrix (MatrixID, Value)",
)
def determinant_matrix(matrix, *, ctx):
    data = _expect(matrix).data
    if data.shape[0] != data.shape[1]:
        raise ValueError("只有方陣有行列式")
    return Result([], [_tidy(float(np.linalg.det(data)))])


@operator(
    "矩陣",
    [],
    [],
    [
        P("MatrixLID", "左邊的係數"),
        P("MatrixType", "general。這版不換成別的分解", "'general'", _MATRIX_TYPE),
        P("Epsilon", "大於 0 時用最小平方", "0"),
        P("MatrixRID", "右邊的常數"),
    ],
    [P("MatrixXID", "解")],
    "解線性方程",
    "解 L × X = R。L 是方陣而且 Epsilon 為 0 時用直接法。否則用最小平方。",
    "Vizion.solve_matrix (L, 'general', 0, R, X)",
)
def solve_matrix(left, matrix_type, epsilon, right, *, ctx):
    coeff = _expect(left).data
    target = _expect(right).data
    if coeff.shape[0] != target.shape[0]:
        raise ValueError("L 的列數必須和 R 的列數相同")
    kind = _token(matrix_type, "MatrixType")
    if kind not in ("general", "symmetric", "positive_definite"):
        raise ValueError("MatrixType 只能是 general、symmetric 或 positive_definite")
    tol = scalar(epsilon, "Epsilon")
    try:
        if tol <= 0 and coeff.shape[0] == coeff.shape[1]:
            solved = np.linalg.solve(coeff, target)
        else:
            solved = np.linalg.lstsq(coeff, target, rcond=None if tol <= 0 else tol)[0]
    except np.linalg.LinAlgError as exc:
        raise ValueError("這個方程沒有唯一解") from exc
    return Result([], [_wrap(solved)])


@operator(
    "矩陣",
    [],
    [],
    [P("MatrixID", "矩陣"), P("MatrixType", "full 或 reduced", "'reduced'", _SVD_TYPE)],
    [P("MatrixUID", "左奇異向量"), P("MatrixSID", "奇異值對角矩陣"), P("MatrixVID", "右奇異向量的轉置")],
    "奇異值分解",
    "A = U × S × V。S 是對角矩陣。reduced 用經濟分解，full 把 U 和 V 補成方陣。",
    "Vizion.svd_matrix (MatrixID, 'reduced', U, S, V)",
)
def svd_matrix(matrix, matrix_type, *, ctx):
    data = _expect(matrix).data
    kind = _token(matrix_type, "MatrixType")
    if kind not in ("full", "reduced"):
        raise ValueError("MatrixType 只能是 full 或 reduced")
    u, singular, vt = np.linalg.svd(data, full_matrices=kind == "full")
    shape = (u.shape[1], vt.shape[0])
    diagonal = np.zeros(shape, np.float64)
    np.fill_diagonal(diagonal, singular)
    return Result([], [_wrap(u), _wrap(diagonal), _wrap(vt)])


@operator(
    "矩陣",
    [],
    [],
    [P("MatrixID", "矩陣"), P("NormType", "frobenius、1、2 或 infinity", "'frobenius'", _NORM)],
    [P("Value", "範數")],
    "矩陣範數",
    "frobenius 是元素平方和的平方根。1 是最大的行絕對值和。2 是最大奇異值。infinity 是最大的列絕對值和。",
    "Vizion.norm_matrix (MatrixID, 'frobenius', Value)",
)
def norm_matrix(matrix, norm_type, *, ctx):
    data = _expect(matrix).data
    kind = _token(norm_type, "NormType")
    order = {"frobenius": "fro", "1": 1, "2": 2, "infinity": np.inf}.get(kind)
    if order is None:
        raise ValueError("NormType 只能是 frobenius、1、2 或 infinity")
    return Result([], [_tidy(float(np.linalg.norm(data, order)))])


def _reduce_op(name, short, detail, example, fn):
    def impl(matrix, mean_type, *, ctx, _fn=fn):
        data = _expect(matrix).data
        kind = _token(mean_type, "MeanType")
        return Result([], [_wrap(_reduce(data, kind, _fn))])

    impl.__name__ = name
    return operator(
        "矩陣",
        [],
        [],
        [P("MatrixID", "矩陣"), P("MeanType", "rows、columns 或 all", "'all'", _REDUCE)],
        [P("MatrixResultID", "結果")],
        short,
        detail,
        example,
    )(impl)


_reduce_op("mean_matrix", "矩陣平均", "rows 得到一行一個平均值的直條。columns 得到橫條。all 得到 1×1。", "Vizion.mean_matrix (MatrixID, 'all', Mean)", np.mean)
_reduce_op("min_matrix", "矩陣最小值", "rows、columns、all 的排法和 mean_matrix 相同。", "Vizion.min_matrix (MatrixID, 'all', Minimum)", np.min)
_reduce_op("max_matrix", "矩陣最大值", "rows、columns、all 的排法和 mean_matrix 相同。", "Vizion.max_matrix (MatrixID, 'all', Maximum)", np.max)


@operator(
    "矩陣",
    [],
    [],
    [P("MatrixID", "方陣"), P("MatrixType", "full 或 reduced", "'reduced'", _QR_TYPE)],
    [P("MatrixQID", "正交矩陣"), P("MatrixRID", "上三角矩陣")],
    "QR 分解",
    "A = Q × R。reduced 讓 R 是方陣。這版沒有再做別種正交分解。",
    "Vizion.orthogonal_decompose_matrix (MatrixID, 'reduced', Q, R)",
)
def orthogonal_decompose_matrix(matrix, matrix_type, *, ctx):
    data = _expect(matrix).data
    kind = _token(matrix_type, "MatrixType")
    if kind not in ("full", "reduced"):
        raise ValueError("MatrixType 只能是 full 或 reduced")
    q, r = np.linalg.qr(data, mode="reduced" if kind == "reduced" else "complete")
    return Result([], [_wrap(q), _wrap(r)])


@operator(
    "矩陣",
    [],
    [],
    [P("MatrixID", "對稱方陣")],
    [P("EigenvaluesID", "特徵值，由大到小排成一行"), P("EigenvectorsID", "特徵向量，每一行一個")],
    "對稱矩陣的特徵值",
    "矩陣必須對稱。特徵向量的每一行對應同一個位置的特徵值。",
    "Vizion.eigenvalues_symmetric_matrix (MatrixID, Eigenvalues, Eigenvectors)",
)
def eigenvalues_symmetric_matrix(matrix, *, ctx):
    data = _expect(matrix).data
    if data.shape[0] != data.shape[1] or not np.allclose(data, data.T):
        raise ValueError("這個算子只接受對稱方陣")
    values, vectors = np.linalg.eigh(data)
    order = np.argsort(values)[::-1]
    return Result([], [_wrap(values[order].reshape(-1, 1)), _wrap(vectors[:, order])])


@operator(
    "矩陣",
    [],
    [],
    [P("MatrixID", "方陣"), P("Power", "整數次方")],
    [P("MatrixPowID", "次方")],
    "矩陣的整數次方",
    "0 次方是單位矩陣。負次方先求逆。元素各自的次方用 pow_scalar_element_matrix。",
    "Vizion.pow_matrix (MatrixID, 2, MatrixPowID)",
)
def pow_matrix(matrix, power, *, ctx):
    data = _expect(matrix).data
    exponent = scalar(power, "Power")
    if abs(exponent - round(exponent)) > 1e-8:
        raise ValueError("pow_matrix 的次方必須是整數")
    if data.shape[0] != data.shape[1]:
        raise ValueError("只有方陣可以做矩陣次方")
    try:
        result = np.linalg.matrix_power(data, int(round(exponent)))
    except np.linalg.LinAlgError as exc:
        raise ValueError("負次方需要可逆的矩陣") from exc
    return Result([], [_wrap(result)])


def _unary(name, short, detail, example, fn):
    def impl(matrix, *, ctx, _fn=fn):
        data = _expect(matrix).data
        try:
            result = _fn(data)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} 的結果不是有限數")
        return Result([], [_wrap(result)])

    impl.__name__ = name
    return operator("矩陣", [], [], [P("MatrixID", "矩陣")], [P("MatrixResultID", "結果")], short, detail, example)(impl)


def _sqrt(data):
    if np.any(data < 0):
        raise ValueError("sqrt_matrix 不接受負數")
    return np.sqrt(data)


def _log(data):
    if np.any(data <= 0):
        raise ValueError("log_matrix 只接受正數")
    return np.log(data)


def _domain(low, high, message):
    def apply(data, _low=low, _high=high, _message=message):
        if np.any((data < _low) | (data > _high)):
            raise ValueError(_message)
        return data

    return apply


_unary("abs_matrix", "元素絕對值", "每個元素取絕對值。", "Vizion.abs_matrix (MatrixID, MatrixAbsID)", np.abs)
_unary("sqrt_matrix", "元素平方根", "有負數會失敗。", "Vizion.sqrt_matrix (MatrixID, MatrixSqrtID)", _sqrt)
_unary("exp_matrix", "元素指數", "每個元素做 e 的次方。", "Vizion.exp_matrix (MatrixID, MatrixExpID)", np.exp)
_unary("log_matrix", "元素自然對數", "只接受正數。", "Vizion.log_matrix (MatrixID, MatrixLogID)", _log)
_unary("sin_matrix", "元素正弦", "角度是弳度。", "Vizion.sin_matrix (MatrixID, MatrixSinID)", np.sin)
_unary("cos_matrix", "元素餘弦", "角度是弳度。", "Vizion.cos_matrix (MatrixID, MatrixCosID)", np.cos)
_unary("tan_matrix", "元素正切", "角度是弳度。", "Vizion.tan_matrix (MatrixID, MatrixTanID)", np.tan)
_unary("asin_matrix", "元素反正弦", "元素必須在 -1 到 1。", "Vizion.asin_matrix (MatrixID, MatrixAsinID)", lambda data: np.arcsin(_domain(-1, 1, "asin_matrix 的元素必須在 -1 到 1")(data)))
_unary("acos_matrix", "元素反餘弦", "元素必須在 -1 到 1。", "Vizion.acos_matrix (MatrixID, MatrixAcosID)", lambda data: np.arccos(_domain(-1, 1, "acos_matrix 的元素必須在 -1 到 1")(data)))
_unary("atan_matrix", "元素反正切", "結果是弳度。", "Vizion.atan_matrix (MatrixID, MatrixAtanID)", np.arctan)
_unary("sinh_matrix", "元素雙曲正弦", "每個元素做 sinh。", "Vizion.sinh_matrix (MatrixID, MatrixSinhID)", np.sinh)
_unary("cosh_matrix", "元素雙曲餘弦", "每個元素做 cosh。", "Vizion.cosh_matrix (MatrixID, MatrixCoshID)", np.cosh)
_unary("tanh_matrix", "元素雙曲正切", "每個元素做 tanh。", "Vizion.tanh_matrix (MatrixID, MatrixTanhID)", np.tanh)


@operator(
    "矩陣",
    [],
    [],
    [P("MatrixYID", "y"), P("MatrixXID", "x")],
    [P("MatrixAtan2ID", "atan2(y, x)，弳度")],
    "元素的雙參數反正切",
    "兩個矩陣必須一樣大。",
    "Vizion.atan2_matrix (Y, X, Angle)",
)
def atan2_matrix(y_matrix, x_matrix, *, ctx):
    y_data = _expect(y_matrix)
    x_data = _expect(x_matrix)
    _same_shape(y_data, x_data, "atan2_matrix")
    return Result([], [_wrap(np.arctan2(y_data.data, x_data.data))])


@operator(
    "矩陣",
    [],
    [],
    [P("MatrixID", "底數"), P("Power", "次方")],
    [P("MatrixPowID", "每個元素的次方")],
    "每個元素乘上同一個次方",
    "負數的底配上非整數次方會失敗。",
    "Vizion.pow_scalar_element_matrix (MatrixID, 2, MatrixPowID)",
)
def pow_scalar_element_matrix(matrix, power, *, ctx):
    data = _expect(matrix).data
    result = np.power(data, scalar(power, "Power"))
    if not np.all(np.isfinite(result)):
        raise ValueError("pow_scalar_element_matrix 的結果不是有限數")
    return Result([], [_wrap(result)])


@operator(
    "矩陣",
    [],
    [],
    [P("MatrixBaseID", "底數"), P("MatrixExpID", "次方")],
    [P("MatrixPowID", "元素次方")],
    "元素各自的次方",
    "兩個矩陣必須一樣大。",
    "Vizion.pow_element_matrix (Base, Exponent, Power)",
)
def pow_element_matrix(base, exponent, *, ctx):
    left = _expect(base)
    right = _expect(exponent)
    _same_shape(left, right, "pow_element_matrix")
    result = np.power(left.data, right.data)
    if not np.all(np.isfinite(result)):
        raise ValueError("pow_element_matrix 的結果不是有限數")
    return Result([], [_wrap(result)])
