"""Conservative visual grouping for near-duplicate photos."""

import cv2
import numpy as np
from PIL import Image

from .config import GROUP_COLOR_SIMILARITY, GROUP_HASH_DISTANCE


def describe_image(image):
    gray = np.asarray(image.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    low = cv2.dct(gray)[:8, :8].flatten()[1:]
    bits = low > np.median(low)
    signature = sum(int(bit) << i for i, bit in enumerate(bits))
    rgb = np.asarray(image.resize((64, 64), Image.Resampling.BILINEAR).convert('RGB'))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Ignore brightness so an underexposed burst frame can still join its peers.
    histogram = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
    histogram /= max(float(np.linalg.norm(histogram)), 1e-9)
    return {'hash': signature, 'colors': histogram.tolist(), 'aspect': image.width / image.height}


def similar(a, b, hash_distance=GROUP_HASH_DISTANCE):
    if abs(np.log(a['aspect'] / b['aspect'])) > 0.12:
        return False
    distance = (a['hash'] ^ b['hash']).bit_count()
    color_similarity = float(np.dot(a['colors'], b['colors']))
    return distance <= hash_distance and color_similarity >= GROUP_COLOR_SIMILARITY


def group_photos(photos, hash_distance=GROUP_HASH_DISTANCE):
    """Require similarity to every member, preventing A-B-C chaining.

    Deterministic greedy complete-link grouping; O(n²), suitable for a prototype.
    It groups by appearance, not identity, EXIF time, or photographic intent.
    """
    groups = []
    for photo in sorted(photos, key=lambda p: p['path']):
        for group in groups:
            if all(similar(photo['descriptor'], other['descriptor'], hash_distance) for other in group):
                group.append(photo)
                break
        else:
            groups.append([photo])
    return groups
