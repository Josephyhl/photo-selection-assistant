"""Convert the closed-eye dataset to square 224x224 RGB.

data/closed_eye_dataset/{OpenFace,ClosedFace}
  -> data/closed_eye_dataset_processed/{OpenFace,ClosedFace}

Originals are read-only. These images are already face crops, so no detection
is run.
"""

from pathlib import Path

from PIL import Image

PROJECT_FOLDER = Path(__file__).parent.parent
SOURCE_FOLDER = PROJECT_FOLDER / "data" / "closed_eye_dataset"
OUTPUT_FOLDER = PROJECT_FOLDER / "data" / "closed_eye_dataset_processed"

CLASS_FOLDERS = ["OpenFace", "ClosedFace"]

# The dataset also contains Thumbs.db, which must be skipped.
PHOTO_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

OUTPUT_SIZE = 224


def make_square_by_padding(image):
    """Pad to square. Cropping could slice off an eye, which is all that matters here."""
    if image.width == image.height:
        return image, False

    side = max(image.width, image.height)

    # Fill with the average border colour so the strips blend in.
    small = image.resize((3, 3), Image.BOX)
    edge_pixels = [
        small.getpixel((x, y))
        for x in range(3)
        for y in range(3)
        if not (x == 1 and y == 1)
    ]
    fill = tuple(sum(channel) // len(edge_pixels) for channel in zip(*edge_pixels))

    padded = Image.new("RGB", (side, side), fill)
    padded.paste(image, ((side - image.width) // 2, (side - image.height) // 2))

    return padded, True


def process_one_image(source_path, output_path):
    try:
        image = Image.open(source_path)
    except Exception as error:
        return {"saved": False, "reason": f"could not open ({error})",
                "original_size": None, "padded": False}

    original_size = image.size

    if image.mode != "RGB":
        image = image.convert("RGB")

    try:
        square_image, was_padded = make_square_by_padding(image)
        final_image = square_image.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)
        final_image.save(output_path, quality=95)
    except Exception as error:
        image.close()
        return {"saved": False, "reason": f"could not process ({error})",
                "original_size": original_size, "padded": False}

    image.close()
    return {"saved": True, "reason": "ok", "original_size": original_size,
            "padded": was_padded}


def process_class_folder(class_name):
    source = SOURCE_FOLDER / class_name
    destination = OUTPUT_FOLDER / class_name
    destination.mkdir(parents=True, exist_ok=True)

    images = sorted(
        item for item in source.iterdir()
        if item.suffix.lower() in PHOTO_EXTENSIONS
    )
    skipped_non_images = sorted(
        item.name for item in source.iterdir()
        if item.suffix.lower() not in PHOTO_EXTENSIONS
    )

    processed = 0
    failures = []
    padded_count = 0
    sizes_seen = {}

    for source_path in images:
        output_path = destination / (source_path.stem + ".jpg")
        report = process_one_image(source_path, output_path)

        if report["original_size"] is not None:
            sizes_seen[report["original_size"]] = (
                sizes_seen.get(report["original_size"], 0) + 1
            )

        if report["saved"]:
            processed += 1
            if report["padded"]:
                padded_count += 1
        else:
            failures.append((source_path.name, report["reason"]))

    return {
        "found": len(images),
        "processed": processed,
        "failures": failures,
        "padded": padded_count,
        "sizes_seen": sizes_seen,
        "skipped_non_images": skipped_non_images,
    }


def check_all_outputs():
    total = 0
    wrong_size = []
    wrong_mode = []

    for class_name in CLASS_FOLDERS:
        folder = OUTPUT_FOLDER / class_name
        if not folder.exists():
            continue

        for path in sorted(folder.glob("*.jpg")):
            total += 1
            image = Image.open(path)

            if image.size != (OUTPUT_SIZE, OUTPUT_SIZE):
                wrong_size.append((path.name, image.size))
            if image.mode != "RGB":
                wrong_mode.append((path.name, image.mode))

            image.close()

    return total, wrong_size, wrong_mode


def main():
    if not SOURCE_FOLDER.exists():
        print("Could not find the dataset at:", SOURCE_FOLDER)
        return

    print("Preparing closed-eye dataset...")
    print("Reading from:", SOURCE_FOLDER)
    print("Writing to:  ", OUTPUT_FOLDER)
    print()

    results = {}

    for class_name in CLASS_FOLDERS:
        result = process_class_folder(class_name)
        results[class_name] = result

        print(f"{class_name} images found: {result['found']}")
        print(f"{class_name} processed successfully: {result['processed']}")
        print(f"{class_name} failures: {len(result['failures'])}")

        for name, reason in result["failures"]:
            print(f"    failed: {name} - {reason}")

        if result["skipped_non_images"]:
            print(f"    skipped non-image files: {', '.join(result['skipped_non_images'])}")

        print(f"    original sizes: "
              f"{', '.join(f'{w}x{h} ({n})' for (w, h), n in sorted(result['sizes_seen'].items(), key=lambda kv: -kv[1])[:4])}")
        print(f"    needed padding to become square: {result['padded']}")
        print()

    total_processed = sum(results[name]["processed"] for name in CLASS_FOLDERS)
    total_failed = sum(len(results[name]["failures"]) for name in CLASS_FOLDERS)

    print("Total processed:", total_processed)
    print("Total failed:", total_failed)
    print()

    total, wrong_size, wrong_mode = check_all_outputs()
    print(f"Checked {total} saved images.")

    if not wrong_size and not wrong_mode:
        print(f"All images are exactly {OUTPUT_SIZE}x{OUTPUT_SIZE} and RGB.")
    else:
        for name, size in wrong_size:
            print(f"    wrong size: {name} {size}")
        for name, mode in wrong_mode:
            print(f"    wrong mode: {name} {mode}")


if __name__ == "__main__":
    main()
