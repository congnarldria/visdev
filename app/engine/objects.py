from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class StopSignal(Exception):
    """Raised by Vizion.stop() so the runner pauses without treating it as a failure."""


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


def coerce_iconic(value):
    """A list of regions or contours is one iconic object tuple, like HALCON HObject."""
    if not isinstance(value, list) or not value:
        return value
    if all(isinstance(item, Region) for item in value):
        width, height = value[0].width, value[0].height
        if any(item.width != width or item.height != height for item in value):
            raise ValueError("region tuple 裡每個區域的尺寸要相同")
        objects = []
        links = []
        keep_links = True
        for item in value:
            objects.extend(item.objects)
            if item.links is None or len(item.links) != item.count():
                keep_links = False
            elif keep_links:
                links.extend(item.links)
        return Region(width, height, objects, multi_color=len(objects) > 1, links=links if keep_links else None)
    if all(isinstance(item, XLD) for item in value):
        contours = []
        groups = []
        keep_groups = True
        for item in value:
            contours.extend(item.contours)
            if item.groups is None or len(item.groups) != len(item.contours):
                keep_groups = False
            elif keep_groups:
                groups.extend(item.groups)
        width = max(item.width for item in value)
        height = max(item.height for item in value)
        return XLD(width, height, contours, multi_color=len(contours) > 1, groups=groups if keep_groups else None)
    return value
