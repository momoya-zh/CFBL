# -*- coding: utf-8 -*-
"""传统图像预处理与插值模块。"""

from src.processing.interpolation import (
    INTERPOLATION_METHODS,
    SUPPORTED_SCALES,
    upscale_image,
)

__all__ = [
    "INTERPOLATION_METHODS",
    "SUPPORTED_SCALES",
    "upscale_image",
]
