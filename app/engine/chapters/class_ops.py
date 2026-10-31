"""In-memory k nearest neighbors.

Samples live in the handle. This does not read or write HALCON classifier files,
and it is not a GMM, SVM, or MLP.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..objects import Result
from ..operators import P, as_list, operator, whole


@dataclass
class ClassKnn:
    k: int
    samples: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    closed: bool = False

    def summary(self) -> str:
        if self.closed:
            return "kNN  已關閉"
        return f"kNN  k={self.k}  {len(self.labels)} 筆"


def _alive(model, name: str) -> ClassKnn:
    if not isinstance(model, ClassKnn):
        raise TypeError(f"{name} 必須是 create_class_knn 建立的分類器")
    if model.closed:
        raise RuntimeError("分類器已經 clear_class_knn")
    return model


def _features(value, name: str) -> np.ndarray:
    try:
        nums = [float(item) for item in as_list(value)]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必須是數值 tuple") from exc
    if not nums:
        raise ValueError(f"{name} 不能是空的")
    return np.asarray(nums, np.float64)


@operator(
    "分類",
    [],
    [],
    [P("K", "鄰居數", "1")],
    [P("ClassKnn", "分類器")],
    "建立 k 近鄰分類器",
    "樣本存在記憶體裡。不讀 HALCON 的分類檔，也沒有 GMM、SVM 或 MLP。",
    "create_class_knn (1, ClassKnn)",
)
def create_class_knn(k, *, ctx):
    neighbors = whole(k, "K")
    if neighbors < 1:
        raise ValueError("K 至少是 1")
    return Result([], [ClassKnn(neighbors)])


@operator(
    "分類",
    [],
    [],
    [P("ClassKnn", "分類器"), P("Features", "特徵"), P("ClassID", "類別")],
    [],
    "加入一筆訓練樣本",
    "特徵是一個數值 tuple。每一筆的長度要相同。類別可以是數字或字串。",
    "add_sample_class_knn (ClassKnn, [0, 0], 1)",
)
def add_sample_class_knn(model, features, class_id, *, ctx):
    knn = _alive(model, "ClassKnn")
    vector = _features(features, "Features")
    if knn.samples and len(vector) != len(knn.samples[0]):
        raise ValueError("特徵長度和已經加入的樣本不同")
    knn.samples.append(vector)
    knn.labels.append(class_id)
    return Result([], [])


@operator(
    "分類",
    [],
    [],
    [P("ClassKnn", "分類器"), P("Features", "要分類的特徵")],
    [P("ClassID", "類別"), P("Confidence", "信心")],
    "用 k 近鄰分類",
    "鄰居不足 K 時用全部樣本。信心是距離倒數加權後，勝出類別佔的比例。距離 0 的鄰居權重最大。",
    "classify_class_knn (ClassKnn, [0.2, 0], ClassID, Confidence)",
)
def classify_class_knn(model, features, *, ctx):
    knn = _alive(model, "ClassKnn")
    if not knn.samples:
        raise RuntimeError("還沒有訓練樣本")
    vector = _features(features, "Features")
    if len(vector) != len(knn.samples[0]):
        raise ValueError("特徵長度和訓練樣本不同")
    stack = np.vstack(knn.samples)
    distance = np.sqrt(((stack - vector) ** 2).sum(axis=1))
    used = min(knn.k, len(distance))
    chosen = np.argpartition(distance, used - 1)[:used]
    weights: dict = {}
    for index in chosen:
        label = knn.labels[int(index)]
        weights[label] = weights.get(label, 0.0) + 1.0 / (float(distance[int(index)]) + 1e-6)
    best = max(weights, key=weights.get)
    total = sum(weights.values())
    return Result([], [best, float(weights[best] / total)])


@operator(
    "分類",
    [],
    [],
    [P("ClassKnn", "分類器")],
    [],
    "清掉訓練樣本",
    "K 保留，樣本全部移除。",
    "clear_samples_class_knn (ClassKnn)",
)
def clear_samples_class_knn(model, *, ctx):
    knn = _alive(model, "ClassKnn")
    knn.samples.clear()
    knn.labels.clear()
    return Result([], [])


@operator(
    "分類",
    [],
    [],
    [P("ClassKnn", "分類器")],
    [],
    "關閉分類器",
    "關閉後不能再加入樣本或分類。",
    "clear_class_knn (ClassKnn)",
)
def clear_class_knn(model, *, ctx):
    knn = _alive(model, "ClassKnn")
    knn.samples.clear()
    knn.labels.clear()
    knn.closed = True
    return Result([], [])
