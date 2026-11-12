"""Global, face, and eye sharpness with the existing fallback chain."""

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

from ..config import (
    FACE_LANDMARK_MODEL_PATH,
    FACE_MODEL_PATH,
    LEFT_EYE_CORNERS,
    MIN_EYE_CROP_PIXELS,
    MIN_FACE_WIDTH_FOR_EYES,
    RIGHT_EYE_CORNERS,
)

from .detection import detect_faces_with_tiling


def analyze_sharpness(image):
    """Laplacian variance and a 0-100 score."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    pixels = np.array(image)
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness_value = edges.var()
    sharpness_score = min(sharpness_value / 10, 100)

    return sharpness_value, sharpness_score


def sharpness_in_box(image, box):
    """Laplacian variance inside one rectangle - a face, an eye, anything."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    left = box["x"]
    top = box["y"]
    right = box["x"] + box["width"]
    bottom = box["y"] + box["height"]

    crop = image.crop((left, top, right, bottom))
    pixels = np.array(crop)
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_64F)

    return edges.var()


def analyze_face_sharpness(image, face_box):
    return sharpness_in_box(image, face_box)


def find_eye_boxes(image, face_box):
    """Left/right eye boxes from face landmarks, or (None, None) if none were found."""
    padding = int(face_box["width"] * 0.4)
    crop_left = max(0, face_box["x"] - padding)
    crop_top = max(0, face_box["y"] - padding)
    crop_right = min(image.width, face_box["x"] + face_box["width"] + padding)
    crop_bottom = min(image.height, face_box["y"] + face_box["height"] + padding)

    face_area = image.crop((crop_left, crop_top, crop_right, crop_bottom))

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_LANDMARK_MODEL_PATH)),
        num_faces=1,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(face_area))
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None, None

    landmarks = result.face_landmarks[0]
    eye_boxes = []

    for outer_index, inner_index in (LEFT_EYE_CORNERS, RIGHT_EYE_CORNERS):
        outer = landmarks[outer_index]
        inner = landmarks[inner_index]

        outer_x = crop_left + outer.x * face_area.width
        outer_y = crop_top + outer.y * face_area.height
        inner_x = crop_left + inner.x * face_area.width
        inner_y = crop_top + inner.y * face_area.height

        center_x = (outer_x + inner_x) / 2
        center_y = (outer_y + inner_y) / 2
        eye_width = ((inner_x - outer_x) ** 2 + (inner_y - outer_y) ** 2) ** 0.5

        half_width = eye_width * 0.8
        half_height = eye_width * 0.5

        x = int(center_x - half_width)
        y = int(center_y - half_height)
        width = int(half_width * 2)
        height = int(half_height * 2)

        if x < 0:
            width = width + x
            x = 0
        if y < 0:
            height = height + y
            y = 0
        if x + width > image.width:
            width = image.width - x
        if y + height > image.height:
            height = image.height - y

        eye_boxes.append({"x": x, "y": y, "width": width, "height": height})

    return eye_boxes[0], eye_boxes[1]


def measure_eye_sharpness(image, face_box):
    """Eye-region sharpness plus a reliability verdict.

    Landmarks being returned is not proof the eyes were visible - the model
    places points even behind sunglasses - so size is checked before trusting it.
    """
    answer = {
        "landmarks_found": False,
        "reliable": False,
        "reason": "",
        "left_eye_sharpness": None,
        "right_eye_sharpness": None,
        "average_eye_sharpness": None,
    }

    if not FACE_LANDMARK_MODEL_PATH.exists():
        answer["reason"] = "landmark model not downloaded"
        return answer

    left_eye_box, right_eye_box = find_eye_boxes(image, face_box)

    if left_eye_box is None:
        answer["reason"] = "no landmarks returned"
        return answer

    answer["landmarks_found"] = True

    if face_box["width"] < MIN_FACE_WIDTH_FOR_EYES:
        answer["reason"] = f"face only {face_box['width']}px wide, need {MIN_FACE_WIDTH_FOR_EYES}"
        return answer

    for eye_box in (left_eye_box, right_eye_box):
        if eye_box["width"] < MIN_EYE_CROP_PIXELS or eye_box["height"] < MIN_EYE_CROP_PIXELS:
            answer["reason"] = (
                f"eye region only {eye_box['width']}x{eye_box['height']}px, need {MIN_EYE_CROP_PIXELS}"
            )
            return answer

    left_sharpness = sharpness_in_box(image, left_eye_box)
    right_sharpness = sharpness_in_box(image, right_eye_box)

    answer["left_eye_sharpness"] = left_sharpness
    answer["right_eye_sharpness"] = right_sharpness
    answer["average_eye_sharpness"] = (left_sharpness + right_sharpness) / 2
    answer["reliable"] = True
    answer["reason"] = "ok"

    return answer


def portrait_result(sharpness_value, method, faces_detected, eyes=None):
    if eyes is None:
        eyes = {
            "landmarks_found": False,
            "reliable": False,
            "reason": "no face to look at",
            "left_eye_sharpness": None,
            "right_eye_sharpness": None,
        }

    return {
        "sharpness_value": sharpness_value,
        "method": method,
        "faces_detected": faces_detected,
        "left_eye_sharpness": eyes["left_eye_sharpness"],
        "right_eye_sharpness": eyes["right_eye_sharpness"],
        "eye_reliable": eyes["reliable"],
        "landmarks_found": eyes["landmarks_found"],
        "eye_reason": eyes["reason"],
    }


def analyze_portrait_sharpness(image, show_details=True, faces=None):
    """Fallback chain: eye sharpness -> face sharpness -> global sharpness."""
    global_sharpness, _ = analyze_sharpness(image)

    if not FACE_MODEL_PATH.exists():
        return portrait_result(global_sharpness, "global", 0)

    if faces is None:
        faces = detect_faces_with_tiling(image, show_details)

    if len(faces) == 0:
        return portrait_result(global_sharpness, "global", 0)

    selected_face = max(faces, key=lambda face: face["width"] * face["height"])

    # Computed before the eye attempt, so the fallback is ready if that fails.
    face_sharpness = analyze_face_sharpness(image, selected_face)
    eyes = measure_eye_sharpness(image, selected_face)

    if eyes["reliable"]:
        return portrait_result(eyes["average_eye_sharpness"], "eye", len(faces), eyes)

    return portrait_result(face_sharpness, "face", len(faces), eyes)
