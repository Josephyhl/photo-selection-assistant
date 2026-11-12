"""Crop the face dataset into square 224x224 face crops.

    python3 src/prepare_face_dataset.py                  full-frame + 3x3 tiles
    python3 src/prepare_face_dataset.py --person-guided   person crop first

The two paths write to different folders so they can be compared.
Originals are read-only.
"""

import csv
import sys
from pathlib import Path

from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.detection import (
    detect_face_with_person_guidance,
    detect_faces,
    detect_faces_in_tiles,
    detect_people,
    remove_duplicate_faces,
)
from src.config import PROJECT_FOLDER

SOURCE_FOLDER = PROJECT_FOLDER / "data" / "face-dataset"

USE_PERSON_GUIDANCE = "--person-guided" in sys.argv

if USE_PERSON_GUIDANCE:
    OUTPUT_FOLDER = PROJECT_FOLDER / "data" / "face-dataset-processed-person-guided"
else:
    OUTPUT_FOLDER = PROJECT_FOLDER / "data" / "face-dataset-processed"

CLASS_FOLDERS = ["Good Facial", "Bad Facial"]

PHOTO_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# Added to each side of the face box, so a crop 1.5x the box keeps forehead,
# hair and chin.
FACE_CROP_MARGIN = 0.25

OUTPUT_SIZE = 224

# MediaPipe never reports below ~0.50, so 0.50 keeps everything. Raising this
# cleans the data but shrinks it fast (0.60 -> 56 crops, 0.70 -> 37 crops), and
# confidence alone cannot separate faces from false positives: a real face here
# scored 0.684, a graduation cap 0.692. Check manifest.csv by eye too.
MIN_FACE_CONFIDENCE = 0.50


def square_crop_box(image, face_box):
    """Square crop around a face with margin, always inside the image."""
    center_x = face_box["x"] + face_box["width"] / 2
    center_y = face_box["y"] + face_box["height"] / 2

    face_size = max(face_box["width"], face_box["height"])
    crop_size = face_size * (1 + 2 * FACE_CROP_MARGIN)
    crop_size = int(min(crop_size, image.width, image.height))

    # Slide back inside the photo rather than trimming, which would break the
    # square.
    left = max(0, min(int(center_x - crop_size / 2), image.width - crop_size))
    top = max(0, min(int(center_y - crop_size / 2), image.height - crop_size))

    return left, top, left + crop_size, top + crop_size


def process_one_photo(photo_path, output_path):
    image = Image.open(photo_path)

    if image.mode != "RGB":
        image = image.convert("RGB")

    if USE_PERSON_GUIDANCE:
        faces = detect_face_with_person_guidance(image)
        source = faces[0]["found_by"] if len(faces) > 0 else "none"
    else:
        # Done in two steps rather than via detect_faces_with_tiling so we can
        # record which path found the face.
        faces = detect_faces(image)
        source = "full-frame"

        if len(faces) == 0:
            faces = remove_duplicate_faces(
                detect_faces_in_tiles(image, show_details=False)
            )
            source = "tiled"

    if len(faces) == 0:
        image.close()
        return {"saved": False, "reason": "no face detected", "source": "none",
                "confidence": None, "face_size": None}

    largest_face = max(faces, key=lambda face: face["width"] * face["height"])

    if largest_face["confidence"] < MIN_FACE_CONFIDENCE:
        image.close()
        return {"saved": False, "reason": "confidence too low", "source": source,
                "confidence": largest_face["confidence"],
                "face_size": largest_face["width"]}

    face_crop = image.crop(square_crop_box(image, largest_face))
    face_crop = face_crop.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)
    face_crop.save(output_path, quality=95)

    image.close()
    return {"saved": True, "reason": "ok", "source": source,
            "confidence": largest_face["confidence"],
            "face_size": largest_face["width"]}


def process_class_folder(class_name):
    source = SOURCE_FOLDER / class_name
    destination = OUTPUT_FOLDER / class_name
    destination.mkdir(parents=True, exist_ok=True)

    photos = sorted(
        item
        for item in source.iterdir()
        if item.suffix.lower() in PHOTO_EXTENSIONS
    )

    processed = 0
    failed = []
    manifest_rows = []

    for photo_path in photos:
        output_path = destination / (photo_path.stem + ".jpg")
        report = process_one_photo(photo_path, output_path)

        manifest_rows.append(
            {
                "class": class_name,
                "filename": photo_path.name,
                "saved": report["saved"],
                "reason": report["reason"],
                "found_by": report["source"],
                "confidence": (
                    f"{report['confidence']:.3f}"
                    if report["confidence"] is not None
                    else ""
                ),
                "face_box_px": report["face_size"] or "",
            }
        )

        if report["saved"]:
            processed += 1
        else:
            failed.append((photo_path.name, report["reason"]))

    return len(photos), processed, failed, manifest_rows


def check_all_crops_are_square():
    wrong_size = []
    total = 0

    for class_name in CLASS_FOLDERS:
        folder = OUTPUT_FOLDER / class_name

        if not folder.exists():
            continue

        for crop_path in sorted(folder.iterdir()):
            if crop_path.suffix.lower() not in PHOTO_EXTENSIONS:
                continue

            total += 1
            crop = Image.open(crop_path)

            if crop.size != (OUTPUT_SIZE, OUTPUT_SIZE):
                wrong_size.append((crop_path.name, crop.size))

            crop.close()

    return total, wrong_size


def main():
    if not SOURCE_FOLDER.exists():
        print("Could not find the dataset at:", SOURCE_FOLDER)
        return

    print("Preparing face dataset...")
    print("Detection path:", "person-guided (new)" if USE_PERSON_GUIDANCE else "full-frame + tiled (old)")
    print("Reading from:", SOURCE_FOLDER)
    print("Writing to:  ", OUTPUT_FOLDER)
    print()

    results = {}
    all_manifest_rows = []

    for class_name in CLASS_FOLDERS:
        found, processed, failed, manifest_rows = process_class_folder(class_name)
        results[class_name] = (found, processed, failed)
        all_manifest_rows.extend(manifest_rows)

        print(f"{class_name} images found: {found}")
        print(f"{class_name} successfully processed: {processed}")
        print(f"{class_name} failed: {len(failed)}")

        for name, reason in failed:
            print(f"    skipped ({reason}): {name}")

        print()

    manifest_path = OUTPUT_FOLDER / "manifest.csv"
    with open(manifest_path, "w", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "class",
                "filename",
                "saved",
                "reason",
                "found_by",
                "confidence",
                "face_box_px",
            ],
        )
        writer.writeheader()
        writer.writerows(all_manifest_rows)

    total_processed = sum(results[name][1] for name in CLASS_FOLDERS)
    total_failed = sum(len(results[name][2]) for name in CLASS_FOLDERS)

    print("Total processed:", total_processed)
    print("Total failed:", total_failed)
    print()
    print("Wrote:", manifest_path)
    print()

    total_crops, wrong_size = check_all_crops_are_square()
    print(f"Checked {total_crops} saved crops.")

    if len(wrong_size) == 0:
        print(f"All crops are exactly {OUTPUT_SIZE}x{OUTPUT_SIZE}.")
    else:
        print("These crops are the wrong size:")
        for name, size in wrong_size:
            print(f"    {name}: {size}")


if __name__ == "__main__":
    main()
