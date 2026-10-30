from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class StopSignal(Exception):
    """Raised by stop() so the runner pauses without treating it as a failure."""


@dataclass
class Image:
    data: np.ndarray
    domain: np.ndarray | None = None

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def channels(self) -> int:
        return 1 if self.data.ndim == 2 else int(self.data.shape[2])

    def gray(self) -> np.ndarray:
        if self.data.ndim == 2:
            return self.data
        import cv2

        return cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)

    def copy(self) -> "Image":
        domain = None if self.domain is None else self.domain.copy()
        return Image(self.data.copy(), domain)


@dataclass
class Region:
    width: int
    height: int
    objects: list = field(default_factory=list)
    multi_color: bool = False
    links: list | None = None

    def union_mask(self) -> np.ndarray:
        acc = np.zeros((self.height, self.width), np.uint8)
        for mask in self.objects:
            acc = np.maximum(acc, mask)
        return acc

    def count(self) -> int:
        return len(self.objects)


@dataclass
class XLD:
    width: int
    height: int
    contours: list = field(default_factory=list)
    multi_color: bool = False
    groups: list | None = None


@dataclass
class Result:
    iconic: list
    control: list
