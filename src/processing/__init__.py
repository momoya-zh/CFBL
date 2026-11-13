# -*- coding: utf-8 -*-
"""传统图像预处理与插值模块。"""

from src.processing.interpolation import (
    INTERPOLATION_METHODS,
    SUPPORTED_SCALES,
    upscale_image,
)
from src.processing.resample import bicubic_upsample_pillow

__all__ = [
    "INTERPOLATION_METHODS",
    "SUPPORTED_SCALES",
    "bicubic_upsample_pillow",
    "upscale_image",
]
