"""Comparable measurements for the batch ranker; legacy diagnostics stay separate."""

import cv2
import numpy as np
from PIL import Image

from ..config import QUALITY_LONG_SIDE


def normalized_sharpness(image):
    """Measure denoised detail at a fixed scale, with local contrast normalization.

    This remains a heuristic: texture and strong noise can still affect it.
    Compare only similar photos using the same region (face or scene).
    """
    scale = QUALITY_LONG_SIDE / max(image.size)
    size = tuple(max(1, round(v * scale)) for v in image.size)
    gray = np.asarray(image.convert('L').resize(size, Image.Resampling.LANCZOS))
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0.7)
    contrast = cv2.GaussianBlur(smooth * smooth, (0, 0), 3) - cv2.GaussianBlur(smooth, (0, 0), 3) ** 2
    laplacian = cv2.Laplacian(smooth, cv2.CV_32F)
    # A contrast floor stops almost-flat images from amplifying tiny variations.
    energy = laplacian * laplacian / np.maximum(contrast, 25.0)
    return float(np.mean(np.minimum(energy, 20.0)))


def exposure_quality(image):
    """Penalize actual channel clipping and extreme midtones, not dark backgrounds."""
    sample = image.convert('RGB').copy()
    sample.thumbnail((QUALITY_LONG_SIDE, QUALITY_LONG_SIDE))
    rgb = np.asarray(sample, dtype=np.float32)
    luminance = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    shadows = float(np.mean(np.all(rgb <= 5, axis=2)))
    highlights = float(np.mean(np.any(rgb >= 250, axis=2)))
    median = float(np.median(luminance))
    midpoint_penalty = max(0.0, 40 - median) / 40 * 35 + max(0.0, median - 215) / 40 * 35
    score = max(0.0, 100 - 80 * shadows - 100 * highlights - midpoint_penalty)
    return {
        'score': score,
        'shadow_clipping': shadows,
        'highlight_clipping': highlights,
        'median_brightness': median,
    }


def crop_box(image, box):
    """Clip a detection rectangle to the image; never add black padding."""
    left = max(0, int(box['x']))
    top = max(0, int(box['y']))
    right = min(image.width, int(box['x'] + box['width']))
    bottom = min(image.height, int(box['y'] + box['height']))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom))
