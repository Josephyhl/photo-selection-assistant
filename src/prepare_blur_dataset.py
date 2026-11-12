"""Map raw blur/focus datasets into data/blur_dataset_processed/ + manifest.csv.

SOURCE_DATASETS is empty, so the script exits with a message instead of
inventing data. Add an entry and re-run when a dataset arrives.

Two kinds of transform, kept apart on purpose:
  (A) label-preserving augmentation -> train_blur_classifier.py, on the fly,
      never changes the correct answer.
  (B) synthetic blur generation -> this file, deliberately changes the label,
      written to disk once and recorded in the manifest so it stays inspectable.
"""

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

PROJECT_FOLDER = Path(__file__).parent.parent
OUTPUT_FOLDER = PROJECT_FOLDER / "data" / "blur_dataset_processed"
MANIFEST_PATH = OUTPUT_FOLDER / "manifest.csv"

# ImageFolder sorts alphabetically: Blurry = 0, Sharp = 1. blur_classifier.py
# hard-codes the same order, so renaming these means changing that file too.
CLASS_FOLDERS = ["Blurry", "Sharp"]

PHOTO_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

OUTPUT_SIZE = 224

MANIFEST_FIELDS = [
    "source_dataset",
    "source_file",
    "group_id",
    "label",
    "processed_file",
    "original_width",
    "original_height",
    "processed_width",
    "processed_height",
    "synthetic",
    "synthetic_transform",
]

# One entry per raw dataset; several can coexist and are merged into the same
# folders. "source_dataset" in the manifest keeps track of where each image
# came from, so a class/source confound can be spotted before training.
#
#     {
#         "name": "kaggle_blur_dataset",
#         "root": PROJECT_FOLDER / "data" / "blur_raw" / "kaggle",
#         "class_map": {"sharp": "Sharp", "defocused_blurred": "Blurry",
#                       "motion_blurred": "Blurry"},
#         "group_from": "stem",       # "stem" | "parent" | regex
#     }
SOURCE_DATASETS = []

# Leave off until the real Blurry images have been inspected: a distinctive
# synthetic signature is another shortcut waiting to be learned.
SYNTHETIC_BLUR_ENABLED = False

SYNTHETIC_BLUR_TRANSFORMS = ["gaussian", "motion", "defocus", "downsample"]


def synthetic_gaussian_blur(image, strength=3.0):
    return image.filter(ImageFilter.GaussianBlur(strength))


def synthetic_motion_blur(image, length=11, horizontal=True):
    """Directional smear, by averaging shifted copies.

    Pillow's Kernel filter only accepts 3x3 and 5x5, too small for a
    convincing smear. Edges are padded so nothing wraps around.
    """
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    half = length // 2

    if horizontal:
        padded = np.pad(array, ((0, 0), (half, half), (0, 0)), mode="edge")
        shifted = [padded[:, i:i + array.shape[1], :] for i in range(length)]
    else:
        padded = np.pad(array, ((half, half), (0, 0), (0, 0)), mode="edge")
        shifted = [padded[i:i + array.shape[0], :, :] for i in range(length)]

    return Image.fromarray(np.mean(shifted, axis=0).astype("uint8"))


def synthetic_defocus_blur(image, radius=4):
    """Rounder than Gaussian, closer to a real focus miss."""
    return image.filter(ImageFilter.BoxBlur(radius))


def synthetic_downsample_degradation(image, factor=4):
    """Shrink then enlarge. This is the degradation that fooled the closed-eye model."""
    small = image.resize(
        (max(1, image.width // factor), max(1, image.height // factor)),
        Image.BILINEAR,
    )
    return small.resize((image.width, image.height), Image.BICUBIC)


SYNTHETIC_TRANSFORMS = {
    "gaussian": synthetic_gaussian_blur,
    "motion": synthetic_motion_blur,
    "defocus": synthetic_defocus_blur,
    "downsample": synthetic_downsample_degradation,
}


def make_square_by_padding(image):
    """Pad to square. Stretching would change how blurred the image looks."""
    if image.width == image.height:
        return image, False

    side = max(image.width, image.height)

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


def group_id_for(source_path, source_root, rule):
    """Which original photo a file belongs to; group members never split.

    Covers sibling crops, a Sharp original and its synthetic twins, and bursts.

    rule: "stem" | "parent" | a regex whose first group is the identifier.
    """
    if rule == "stem":
        return source_path.stem

    if rule == "parent":
        return source_path.parent.name

    import re

    match = re.match(rule, source_path.name)
    if match and match.groups():
        return match.group(1)

    return source_path.stem


def process_one_image(image, output_path):
    if image.mode != "RGB":
        image = image.convert("RGB")

    square_image, _ = make_square_by_padding(image)
    final_image = square_image.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_image.save(output_path, quality=95)

    return final_image.size


def collect_source_files(dataset):
    root = Path(dataset["root"])
    found = []

    for raw_folder, label in dataset["class_map"].items():
        folder = root / raw_folder

        if not folder.exists():
            print(f"    warning: {folder} does not exist, skipping")
            continue

        if label not in CLASS_FOLDERS:
            print(f"    warning: '{label}' is not one of {CLASS_FOLDERS}, skipping")
            continue

        # rglob so nested raw layouts work too.
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in PHOTO_EXTENSIONS:
                found.append((path, label))

    return found


def process_dataset(dataset):
    name = dataset["name"]
    root = Path(dataset["root"])
    group_rule = dataset.get("group_from", "stem")

    print(f"  source dataset: {name}")
    print(f"    root: {root}")

    files = collect_source_files(dataset)
    print(f"    images found: {len(files)}")

    rows = []
    failures = []

    for source_path, label in files:
        try:
            image = Image.open(source_path)
            original_size = image.size
        except Exception as error:
            failures.append((source_path.name, f"could not open ({error})"))
            continue

        group_id = group_id_for(source_path, root, group_rule)

        # Dataset-name prefix so two sources cannot collide on a filename.
        processed_name = f"{name}__{source_path.stem}.jpg"
        processed_path = OUTPUT_FOLDER / label / processed_name

        try:
            processed_size = process_one_image(image, processed_path)
        except Exception as error:
            image.close()
            failures.append((source_path.name, f"could not process ({error})"))
            continue

        rows.append(
            {
                "source_dataset": name,
                "source_file": str(source_path.relative_to(root)),
                "group_id": f"{name}:{group_id}",
                "label": label,
                "processed_file": f"{label}/{processed_name}",
                "original_width": original_size[0],
                "original_height": original_size[1],
                "processed_width": processed_size[0],
                "processed_height": processed_size[1],
                "synthetic": "no",
                "synthetic_transform": "",
            }
        )

        if SYNTHETIC_BLUR_ENABLED and label == "Sharp":
            for transform_name in SYNTHETIC_BLUR_TRANSFORMS:
                transform = SYNTHETIC_TRANSFORMS[transform_name]

                blurred = transform(image.convert("RGB"))
                synthetic_name = f"{name}__{source_path.stem}__{transform_name}.jpg"
                synthetic_path = OUTPUT_FOLDER / "Blurry" / synthetic_name
                synthetic_size = process_one_image(blurred, synthetic_path)

                rows.append(
                    {
                        "source_dataset": name,
                        "source_file": str(source_path.relative_to(root)),
                        # Same group_id as the sharp original, so the two
                        # cannot land on opposite sides of the split.
                        "group_id": f"{name}:{group_id}",
                        "label": "Blurry",
                        "processed_file": f"Blurry/{synthetic_name}",
                        "original_width": original_size[0],
                        "original_height": original_size[1],
                        "processed_width": synthetic_size[0],
                        "processed_height": synthetic_size[1],
                        "synthetic": "yes",
                        "synthetic_transform": transform_name,
                    }
                )

        image.close()

    print(f"    processed: {len(rows)}   failures: {len(failures)}")
    for filename, reason in failures[:10]:
        print(f"      failed: {filename} - {reason}")

    return rows


def write_manifest(rows):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_PATH, "w", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def report_class_source_matrix(rows):
    """Warn if a class comes from only one source - the closed-eye failure mode."""
    counts = {}
    for row in rows:
        key = (row["source_dataset"], row["label"])
        counts[key] = counts.get(key, 0) + 1

    datasets = sorted({row["source_dataset"] for row in rows})

    print()
    print("--- class / source matrix ---")
    print(f"{'source':>28s}" + "".join(f"{label:>10s}" for label in CLASS_FOLDERS))
    for dataset in datasets:
        line = f"{dataset:>28s}"
        for label in CLASS_FOLDERS:
            line += f"{counts.get((dataset, label), 0):>10d}"
        print(line)

    print()
    for label in CLASS_FOLDERS:
        contributing = [d for d in datasets if counts.get((d, label), 0) > 0]
        if len(contributing) == 1 and len(datasets) > 1:
            print(f"  WARNING: every '{label}' image comes from '{contributing[0]}'.")
            print("           A model can separate the classes by source style")
            print("           instead of by blur. Mix sources before training.")


def check_all_outputs():
    total = 0
    problems = []

    for label in CLASS_FOLDERS:
        folder = OUTPUT_FOLDER / label
        if not folder.exists():
            continue

        for path in sorted(folder.glob("*.jpg")):
            total += 1
            image = Image.open(path)
            if image.size != (OUTPUT_SIZE, OUTPUT_SIZE):
                problems.append((path.name, f"size {image.size}"))
            if image.mode != "RGB":
                problems.append((path.name, f"mode {image.mode}"))
            image.close()

    return total, problems


def main():
    print("Preparing blur / focus dataset...")
    print("Output:", OUTPUT_FOLDER)
    print()

    if not SOURCE_DATASETS:
        print("No source datasets configured yet.")
        print()
        print("Nothing has been created, and no placeholder data has been invented.")
        print("When the raw dataset arrives, add an entry to SOURCE_DATASETS at the")
        print("top of this file - there is a worked example in the comment there -")
        print("and run this script again.")
        print()
        print("Reminders for when that happens:")
        print("  * check the class / source matrix printed at the end;")
        print("    one source feeding only one class is a shortcut waiting to happen")
        print("  * leave SYNTHETIC_BLUR_ENABLED = False until the real Blurry images")
        print("    have been looked at")
        print("  * pick group_from so that crops, bursts and synthetic twins of one")
        print("    photo share a group_id")
        return 0

    for label in CLASS_FOLDERS:
        (OUTPUT_FOLDER / label).mkdir(parents=True, exist_ok=True)

    all_rows = []
    for dataset in SOURCE_DATASETS:
        all_rows.extend(process_dataset(dataset))
        print()

    if not all_rows:
        print("No images were processed.")
        return 1

    write_manifest(all_rows)

    print("--- totals ---")
    for label in CLASS_FOLDERS:
        n = sum(1 for r in all_rows if r["label"] == label)
        synthetic = sum(1 for r in all_rows if r["label"] == label and r["synthetic"] == "yes")
        print(f"  {label:>8s}: {n:5d}  (synthetic: {synthetic})")
    print(f"  {'groups':>8s}: {len({r['group_id'] for r in all_rows}):5d}")

    report_class_source_matrix(all_rows)

    total, problems = check_all_outputs()
    print()
    print(f"Checked {total} saved images.")
    if problems:
        for name, problem in problems[:10]:
            print(f"    {name}: {problem}")
    else:
        print(f"All images are exactly {OUTPUT_SIZE}x{OUTPUT_SIZE} and RGB.")

    print()
    print("Wrote:", MANIFEST_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
