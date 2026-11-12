"""Face and person detection, bounding boxes, and primary-subject selection."""

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions
from ultralytics import YOLO

from ..config import (
    CONFIDENCE_WEIGHT,
    DUPLICATE_OVERLAP_LIMIT,
    EDGE_PENALTY_WEIGHT,
    EDGE_TOUCH_TOLERANCE,
    FACE_BONUS,
    FACE_INSIDE_THRESHOLD,
    FACE_MODEL_PATH,
    PERSON_CLASS_ID,
    PERSON_CONFIDENCE_THRESHOLD,
    PERSON_CROP_EXPANSION,
    PERSON_DETECTION_SIZE,
    PERSON_MODEL_PATH,
    SIZE_WEIGHT,
    TILE_GRID,
    TILE_SIZE_FRACTION,
)

loaded_person_model = None


def create_face_detector():
    """Build a MediaPipe face detector - reuse one instance across many calls."""
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_MODEL_PATH))
    )
    return FaceDetector.create_from_options(options)


def read_face_boxes(detector, image):
    """Run a detector on one image. Returns boxes with x, y, width, height, confidence."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    pixels = np.array(image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=pixels)
    result = detector.detect(mp_image)

    faces = []

    for detection in result.detections:
        box = detection.bounding_box
        x = box.origin_x
        y = box.origin_y
        width = box.width
        height = box.height

        # Clip to the image, in case the box pokes outside it.
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

        confidence = float(detection.categories[0].score)

        if width > 0 and height > 0:
            faces.append({"x": x, "y": y, "width": width, "height": height, "confidence": confidence})

    return faces


def detect_faces(image):
    with create_face_detector() as detector:
        return read_face_boxes(detector, image)


def box_overlap(box_a, box_b):
    """IoU: shared area / combined area."""
    left = max(box_a["x"], box_b["x"])
    top = max(box_a["y"], box_b["y"])
    right = min(box_a["x"] + box_a["width"], box_b["x"] + box_b["width"])
    bottom = min(box_a["y"] + box_a["height"], box_b["y"] + box_b["height"])

    if right <= left or bottom <= top:
        return 0.0

    shared_area = (right - left) * (bottom - top)
    area_a = box_a["width"] * box_a["height"]
    area_b = box_b["width"] * box_b["height"]
    combined_area = area_a + area_b - shared_area

    return shared_area / combined_area


def remove_duplicate_faces(faces):
    """Keep the largest box when two faces overlap heavily."""
    faces_by_size = sorted(faces, key=lambda face: face["width"] * face["height"], reverse=True)
    kept = []

    for face in faces_by_size:
        if not any(box_overlap(face, k) > DUPLICATE_OVERLAP_LIMIT for k in kept):
            kept.append(face)

    return kept


def tile_start_positions(image_length, tile_length):
    """Evenly spaced tile start positions, first at 0 and last flush with the far edge."""
    last_start = image_length - tile_length
    return [int(i * last_start / (TILE_GRID - 1)) for i in range(TILE_GRID)]


def detect_faces_in_tiles(image, show_details=True):
    tile_width = int(image.width * TILE_SIZE_FRACTION)
    tile_height = int(image.height * TILE_SIZE_FRACTION)

    x_positions = tile_start_positions(image.width, tile_width)
    y_positions = tile_start_positions(image.height, tile_height)

    faces = []
    tile_number = 0

    with create_face_detector() as detector:
        for tile_y in y_positions:
            for tile_x in x_positions:
                tile_number += 1
                tile = image.crop((tile_x, tile_y, tile_x + tile_width, tile_y + tile_height))
                tile_faces = read_face_boxes(detector, tile)

                if show_details:
                    print(f"Tile {tile_number} faces:", len(tile_faces))

                for face in tile_faces:
                    faces.append(
                        {
                            "x": face["x"] + tile_x,
                            "y": face["y"] + tile_y,
                            "width": face["width"],
                            "height": face["height"],
                            "confidence": face["confidence"],
                        }
                    )

    return faces


def detect_faces_with_tiling(image, show_details=True):
    """Full-frame detection, falling back to the 3x3 tile sweep."""
    if show_details:
        print()
        print("--- Face Detection ---")
        print()

    faces = detect_faces(image)

    if show_details:
        print("Full-frame faces detected:", len(faces))

    if len(faces) > 0:
        return faces

    if show_details:
        print(f"Trying {TILE_GRID}x{TILE_GRID} tiled detection...")
        print()

    faces = remove_duplicate_faces(detect_faces_in_tiles(image, show_details))

    if show_details:
        print()
        print("Faces after duplicate removal:", len(faces))

    return faces


def load_person_model():
    global loaded_person_model

    if loaded_person_model is None:
        loaded_person_model = YOLO(str(PERSON_MODEL_PATH))

    return loaded_person_model


def detect_people(image):
    """Person boxes with x, y, width, height, confidence, biggest first."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    model = load_person_model()
    results = model.predict(
        np.array(image),
        conf=PERSON_CONFIDENCE_THRESHOLD,
        imgsz=PERSON_DETECTION_SIZE,
        verbose=False,
    )

    detections = results[0].boxes
    people = []

    for detection in detections:
        if int(detection.cls[0]) != PERSON_CLASS_ID:
            continue

        left, top, right, bottom = detection.xyxy[0].tolist()

        people.append(
            {
                "x": int(left),
                "y": int(top),
                "width": int(right - left),
                "height": int(bottom - top),
                "confidence": float(detection.conf[0]),
            }
        )

    people.sort(key=lambda person: person["width"] * person["height"], reverse=True)

    return people


def fraction_inside(inner_box, outer_box):
    """Fraction of inner_box covered by outer_box (not IoU - divides by inner's area only)."""
    left = max(inner_box["x"], outer_box["x"])
    top = max(inner_box["y"], outer_box["y"])
    right = min(inner_box["x"] + inner_box["width"], outer_box["x"] + outer_box["width"])
    bottom = min(inner_box["y"] + inner_box["height"], outer_box["y"] + outer_box["height"])

    if right <= left or bottom <= top:
        return 0.0

    shared_area = (right - left) * (bottom - top)
    inner_area = inner_box["width"] * inner_box["height"]

    if inner_area == 0:
        return 0.0

    return shared_area / inner_area


def count_touching_edges(image, box):
    """How many of the photo's four borders this box touches."""
    side_tolerance = image.width * EDGE_TOUCH_TOLERANCE
    top_tolerance = image.height * EDGE_TOUCH_TOLERANCE

    touching = 0
    if box["x"] <= side_tolerance:
        touching += 1
    if box["y"] <= top_tolerance:
        touching += 1
    if box["x"] + box["width"] >= image.width - side_tolerance:
        touching += 1
    if box["y"] + box["height"] >= image.height - top_tolerance:
        touching += 1

    return touching


def score_person(image, person, faces):
    """Subject-likelihood score: size + confidence + face bonus - edge penalty."""
    image_area = image.width * image.height
    person_area = person["width"] * person["height"]

    area_ratio = person_area / image_area
    size_component = SIZE_WEIGHT * area_ratio

    confidence_component = CONFIDENCE_WEIGHT * person["confidence"]

    contains_face = any(fraction_inside(face, person) >= FACE_INSIDE_THRESHOLD for face in faces)
    face_component = FACE_BONUS if contains_face else 0.0

    touching = count_touching_edges(image, person)
    edge_penalty = EDGE_PENALTY_WEIGHT * touching / 4

    subject_score = size_component + confidence_component + face_component - edge_penalty

    return {
        "person": person,
        "area_ratio": area_ratio,
        "confidence": person["confidence"],
        "contains_face": contains_face,
        "edge_penalty": edge_penalty,
        "subject_score": subject_score,
    }


def select_primary_person(image, people, faces, show_details=True):
    """Highest-scoring person box, or None if no people were detected."""
    if len(people) == 0:
        return None

    scored = [score_person(image, person, faces) for person in people]

    if show_details:
        print()
        print("--- Primary Person Candidates ---")
        for number, entry in enumerate(scored, start=1):
            print()
            print(f"Person {number}")
            print(f"Area ratio: {entry['area_ratio']:.4f}")
            print(f"Confidence: {entry['confidence']:.3f}")
            print("Contains detected face:", "yes" if entry["contains_face"] else "no")
            print(f"Edge penalty: {entry['edge_penalty']:.3f}")
            print(f"Subject score: {entry['subject_score']:.3f}")

    best = max(scored, key=lambda entry: entry["subject_score"])

    if show_details:
        print()
        print(f"Selected primary person: Person {scored.index(best) + 1}")

    selected = dict(best["person"])
    selected["subject_score"] = best["subject_score"]

    return selected


def expanded_person_box(image, person):
    """Person box grown by PERSON_CROP_EXPANSION on each side, clipped to the photo."""
    grow_x = person["width"] * PERSON_CROP_EXPANSION
    grow_y = person["height"] * PERSON_CROP_EXPANSION

    left = max(0, int(person["x"] - grow_x))
    top = max(0, int(person["y"] - grow_y))
    right = min(image.width, int(person["x"] + person["width"] + grow_x))
    bottom = min(image.height, int(person["y"] + person["height"] + grow_y))

    return left, top, right, bottom


def detect_face_with_person_guidance(image, people=None, primary_person=None):
    """Face search order: inside the primary person's crop, then full frame, then 3x3 tiles.

    Each returned box carries "found_by" naming which step found it.
    """
    if people is None:
        people = detect_people(image) if PERSON_MODEL_PATH.exists() else []

    if len(people) > 0:
        if primary_person is None:
            primary_person = select_primary_person(image, people, [], show_details=False)

        if primary_person is not None:
            left, top, right, bottom = expanded_person_box(image, primary_person)
            person_crop = image.crop((left, top, right, bottom))
            crop_faces = detect_faces(person_crop)

            if len(crop_faces) > 0:
                faces = []
                for face in crop_faces:
                    faces.append(
                        {
                            "x": face["x"] + left,
                            "y": face["y"] + top,
                            "width": face["width"],
                            "height": face["height"],
                            "confidence": face["confidence"],
                            "found_by": "person-guided",
                        }
                    )
                return faces

    faces = detect_faces(image)

    if len(faces) > 0:
        for face in faces:
            face["found_by"] = "full-frame"
        return faces

    faces = remove_duplicate_faces(detect_faces_in_tiles(image, show_details=False))

    for face in faces:
        face["found_by"] = "tiled"

    return faces
