"""Global, face, and portrait exposure measurements."""

import numpy as np

from ..config import (
    BRIGHT_LIMIT,
    DARK_LIMIT,
    EXPOSURE_DIFFERENCE_MEDIUM,
    EXPOSURE_DIFFERENCE_SMALL,
    EXPOSURE_DIFFERENCE_STRONG,
    MEDIUM_EXPOSURE_PENALTY,
    SMALL_EXPOSURE_PENALTY,
    STRONG_EXPOSURE_PENALTY,
)


def analyze_exposure(image):
    """Average brightness, dark %, bright %, and a 0-100 exposure score."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    pixels = np.array(image, dtype="float32")
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]

    # Perceptual brightness weighting (ITU-R BT.601).
    brightness = 0.299 * red + 0.587 * green + 0.114 * blue

    average_brightness = brightness.mean()
    total_pixels = brightness.size

    dark_pixels = np.count_nonzero(brightness < DARK_LIMIT)
    bright_pixels = np.count_nonzero(brightness > BRIGHT_LIMIT)

    dark_percentage = dark_pixels / total_pixels * 100
    bright_percentage = bright_pixels / total_pixels * 100

    exposure_score = 100 - dark_percentage - bright_percentage
    exposure_score = max(0, min(100, exposure_score))

    return average_brightness, dark_percentage, bright_percentage, exposure_score


def exposure_in_box(image, box):
    """analyze_exposure() restricted to one rectangle, or None if it's empty."""
    if box is None:
        return None

    left = max(0, box["x"])
    top = max(0, box["y"])
    right = min(image.width, box["x"] + box["width"])
    bottom = min(image.height, box["y"] + box["height"])

    if right <= left or bottom <= top:
        return None

    crop = image.crop((left, top, right, bottom))
    average_brightness, dark_percentage, bright_percentage, exposure_score = analyze_exposure(crop)

    return {
        "average_brightness": average_brightness,
        "dark_percentage": dark_percentage,
        "bright_percentage": bright_percentage,
        "exposure_score": exposure_score,
    }


def analyze_face_exposure(image, face_box):
    """Exposure inside the face box - skin brightness is a fairer signal than clothing."""
    return exposure_in_box(image, face_box)


def exposure_imbalance_penalty(brightness_difference):
    """Penalty from the size of the face-vs-scene brightness gap (direction doesn't matter)."""
    gap = abs(brightness_difference)

    if gap < EXPOSURE_DIFFERENCE_SMALL:
        return 0
    if gap < EXPOSURE_DIFFERENCE_MEDIUM:
        return SMALL_EXPOSURE_PENALTY
    if gap < EXPOSURE_DIFFERENCE_STRONG:
        return MEDIUM_EXPOSURE_PENALTY

    return STRONG_EXPOSURE_PENALTY


def analyze_portrait_exposure(image, faces):
    """Face exposure (penalised for scene mismatch) when a face exists, else global."""
    global_brightness, _, _, global_exposure_score = analyze_exposure(image)

    def global_result():
        return {
            "method": "global",
            "global_exposure_score": global_exposure_score,
            "global_brightness": global_brightness,
            "face_exposure_score": None,
            "face_brightness": None,
            "brightness_difference": None,
            "imbalance_penalty": None,
            "final_exposure_score": global_exposure_score,
        }

    if len(faces) == 0:
        return global_result()

    selected_face = max(faces, key=lambda face: face["width"] * face["height"])
    face_exposure = analyze_face_exposure(image, selected_face)

    if face_exposure is None:
        return global_result()

    face_brightness = face_exposure["average_brightness"]
    brightness_difference = face_brightness - global_brightness
    imbalance_penalty = exposure_imbalance_penalty(brightness_difference)

    final_exposure_score = face_exposure["exposure_score"] - imbalance_penalty
    final_exposure_score = max(0, min(100, final_exposure_score))

    return {
        "method": "face",
        "global_exposure_score": global_exposure_score,
        "global_brightness": global_brightness,
        "face_exposure_score": face_exposure["exposure_score"],
        "face_brightness": face_brightness,
        "brightness_difference": brightness_difference,
        "imbalance_penalty": imbalance_penalty,
        "final_exposure_score": final_exposure_score,
    }


def analyze_person_exposure(image, primary_person):
    """LEGACY - superseded by analyze_face_exposure(). Dark clothing and background
    inside the person box make this unreliable; kept only for comparison."""
    return exposure_in_box(image, primary_person)
