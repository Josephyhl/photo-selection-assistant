"""Blur / focus-quality classifier (Sharp vs Blurry).

No dataset yet, so running this prints what it needs and exits.

    python3 src/train_blur_classifier.py                  # public validation
    python3 src/train_blur_classifier.py --real-eval DIR  # + real-photo test

The closed-eye model hit 97.7% on its own validation set and then mislabelled
~40% of real portraits, because its classes came from different sources and it
learned image detail as a proxy for the label. That risk is worse here, since
detail IS the signal for blur. Hence: group-aware splitting, augmentation that
scrambles resolution / JPEG quality / sharpness, a class-source report before
training, False Blurry Rate as the headline metric, and a separate real-photo
evaluation.
"""

import argparse
import csv
import random
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Shared with inference so class order and normalisation cannot disagree.
from blur_classifier import (  # noqa: E402
    BLURRY_INDEX,
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    SHARP_INDEX,
    make_square_by_padding,
)

PROJECT_FOLDER = Path(__file__).parent.parent
DATA_FOLDER = PROJECT_FOLDER / "data" / "blur_dataset_processed"
MANIFEST_PATH = DATA_FOLDER / "manifest.csv"
MODEL_PATH = PROJECT_FOLDER / "models" / "blur_resnet18.pth"
RESULTS_FOLDER = PROJECT_FOLDER / "results" / "ml" / "blur"
GRADCAM_FOLDER = RESULTS_FOLDER / "gradcam"

SEED = 42

BATCH_SIZE = 32
EPOCHS = 18
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
VALIDATION_SHARE = 0.20
UNFREEZE_LAYER4_AT_EPOCH = 6

GRADCAM_EXAMPLES = 4


# (A) LABEL-PRESERVING AUGMENTATION - train split only.
#
# All of these must leave the correct answer unchanged. The blur and
# resolution ones are deliberately weak: they exist to stop the model using
# absolute image quality as a source fingerprint, not to turn Sharp into
# Blurry. Label-changing blur lives in prepare_blur_dataset.py.
class RandomJPEGQuality:
    """Randomise compression artefacts, an easy source fingerprint.

    Kept above 55 so no real detail is destroyed.
    """

    def __init__(self, minimum=55, maximum=95, probability=0.5):
        self.minimum = minimum
        self.maximum = maximum
        self.probability = probability

    def __call__(self, image):
        if random.random() > self.probability:
            return image

        quality = random.randint(self.minimum, self.maximum)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class RandomResolutionJitter:
    """Shrink then re-enlarge. The cue that broke the closed-eye model.

    Capped at 1.6x so a sharp image stays recognisably sharp; 4x would be a
    label change, not augmentation.
    """

    def __init__(self, minimum_factor=1.0, maximum_factor=1.6, probability=0.5):
        self.minimum_factor = minimum_factor
        self.maximum_factor = maximum_factor
        self.probability = probability

    def __call__(self, image):
        if random.random() > self.probability:
            return image

        factor = random.uniform(self.minimum_factor, self.maximum_factor)
        small = image.resize(
            (max(1, int(image.width / factor)), max(1, int(image.height / factor))),
            Image.BILINEAR,
        )
        return small.resize((image.width, image.height), Image.BICUBIC)


class RandomMildBlur:
    """Radius under 1 pixel. Anything stronger would change the label."""

    def __init__(self, maximum_radius=0.8, probability=0.3):
        self.maximum_radius = maximum_radius
        self.probability = probability

    def __call__(self, image):
        if random.random() > self.probability:
            return image

        radius = random.uniform(0.1, self.maximum_radius)
        return image.filter(ImageFilter.GaussianBlur(radius))


class RandomMildSharpen:
    """Paired with RandomMildBlur to widen global sharpness in both classes."""

    def __init__(self, maximum_percent=60, probability=0.3):
        self.maximum_percent = maximum_percent
        self.probability = probability

    def __call__(self, image):
        if random.random() > self.probability:
            return image

        percent = random.randint(20, self.maximum_percent)
        return image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=percent))


class SquarePad:
    """Matches inference preprocessing."""

    def __call__(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        return make_square_by_padding(image)


def build_transforms():
    """Augmented for training, deterministic for validation."""
    train_transform = transforms.Compose(
        [
            SquarePad(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.RandomResizedCrop(
                INPUT_SIZE, scale=(0.85, 1.0), ratio=(0.95, 1.05)
            ),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02
            ),
            # Anti-shortcut group: scrambles source fingerprints.
            RandomJPEGQuality(),
            RandomResolutionJitter(),
            RandomMildBlur(),
            RandomMildSharpen(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    validation_transform = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return train_transform, validation_transform


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(SEED)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_manifest():
    """Read manifest.csv into {processed_file: row}. Returns {} if absent."""
    if not MANIFEST_PATH.exists():
        return {}

    with open(MANIFEST_PATH) as manifest_file:
        return {row["processed_file"]: row for row in csv.DictReader(manifest_file)}


def group_for_sample(dataset, index, manifest):
    """Manifest group_id if present, else the filename.

    group_id is authoritative: it ties a Sharp original to its synthetic twins
    and sibling crops.
    """
    path = Path(dataset.samples[index][0])
    key = f"{path.parent.name}/{path.name}"

    row = manifest.get(key)
    if row and row.get("group_id"):
        return row["group_id"]

    return "file:" + path.stem


def split_by_group(dataset, manifest):
    """80/20 by group, never by image. Per class, since group sizes vary."""
    groups_per_class = {}
    for index in range(len(dataset)):
        label = dataset.samples[index][1]
        group = group_for_sample(dataset, index, manifest)
        groups_per_class.setdefault(label, {}).setdefault(group, []).append(index)

    train_indices = []
    validation_indices = []
    group_total = 0
    leaked = []

    for label in sorted(groups_per_class.keys()):
        groups = groups_per_class[label]
        group_total += len(groups)

        names = sorted(groups.keys())
        random.Random(SEED + label).shuffle(names)

        target = int(sum(len(v) for v in groups.values()) * VALIDATION_SHARE)

        taken = 0
        for name in names:
            if taken < target:
                validation_indices.extend(groups[name])
                taken += len(groups[name])
            else:
                train_indices.extend(groups[name])

    # A group holding both labels gets split by the per-class loop above.
    # Force those wholly into training.
    group_of = {}
    for index in range(len(dataset)):
        group_of[index] = group_for_sample(dataset, index, manifest)

    validation_groups = {group_of[i] for i in validation_indices}
    training_groups = {group_of[i] for i in train_indices}
    overlap = validation_groups & training_groups

    if overlap:
        moved = [i for i in validation_indices if group_of[i] in overlap]
        validation_indices = [i for i in validation_indices if group_of[i] not in overlap]
        train_indices.extend(moved)
        leaked = sorted(overlap)

    return sorted(train_indices), sorted(validation_indices), group_total, leaked


def count_classes(dataset, indices):
    counts = Counter(dataset.samples[i][1] for i in indices)
    return {CLASS_NAMES[label]: counts.get(label, 0) for label in range(len(CLASS_NAMES))}


def report_class_source_matrix(dataset, manifest):
    """Warn if a class comes from only one source."""
    if not manifest:
        print("  (no manifest, so source composition cannot be checked)")
        return

    counts = Counter()
    for index in range(len(dataset)):
        path = Path(dataset.samples[index][0])
        row = manifest.get(f"{path.parent.name}/{path.name}")
        source = row["source_dataset"] if row else "unknown"
        counts[(source, CLASS_NAMES[dataset.samples[index][1]])] += 1

    sources = sorted({s for s, _ in counts})
    print(f"  {'source':>28s}" + "".join(f"{n:>10s}" for n in CLASS_NAMES))
    for source in sources:
        line = f"  {source:>28s}"
        for name in CLASS_NAMES:
            line += f"{counts.get((source, name), 0):>10d}"
        print(line)

    for name in CLASS_NAMES:
        contributing = [s for s in sources if counts.get((s, name), 0) > 0]
        if len(contributing) == 1 and len(sources) > 1:
            print(f"  WARNING: every '{name}' image comes from '{contributing[0]}'.")
            print("           The model may separate classes by source, not by blur.")


def report_low_level_statistics(dataset, indices, sample_size=200):
    """Per-class Laplacian variance.

    Blurry should sit lower - that is the real signal. A gap so large that one
    threshold would suffice means the dataset is too easy to transfer.
    """
    by_class = {}
    for index in indices:
        by_class.setdefault(dataset.samples[index][1], []).append(index)

    print(f"  {'class':>10s} {'n':>5s} {'mean lapvar':>12s} {'median':>10s}")
    for label in sorted(by_class.keys()):
        chosen = random.Random(SEED).sample(
            by_class[label], min(sample_size, len(by_class[label]))
        )
        values = []
        for index in chosen:
            grey = Image.open(dataset.samples[index][0]).convert("L")
            array = np.asarray(grey, dtype=np.float32)
            laplacian = (
                array[:-2, 1:-1] + array[2:, 1:-1]
                + array[1:-1, :-2] + array[1:-1, 2:]
                - 4 * array[1:-1, 1:-1]
            )
            values.append(laplacian.var())
        print(f"  {CLASS_NAMES[label]:>10s} {len(chosen):5d} "
              f"{np.mean(values):12.1f} {np.median(values):10.1f}")


def build_model(device):
    """Pretrained ResNet18, backbone frozen, new 2-class head."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    return model.to(device)


def describe_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    names = sorted({n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad})
    return trainable, frozen, names


# ==========================================================================
# Train / evaluate
# ==========================================================================
def run_one_epoch(model, loader, criterion, optimizer, device, training):
    model.train() if training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def collect_predictions(model, loader, device):
    model.eval()
    predictions = []
    truths = []
    blurry_probabilities = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            probabilities = torch.softmax(outputs, dim=1).cpu()
            predictions.extend(probabilities.argmax(dim=1).tolist())
            blurry_probabilities.extend(probabilities[:, BLURRY_INDEX].tolist())
            truths.extend(labels.tolist())

    return truths, predictions, blurry_probabilities


def confusion_matrix(truths, predictions):
    size = len(CLASS_NAMES)
    matrix = [[0] * size for _ in range(size)]
    for truth, prediction in zip(truths, predictions):
        matrix[truth][prediction] += 1
    return matrix


def precision_recall_f1(matrix, class_index):
    true_positive = matrix[class_index][class_index]
    false_positive = sum(matrix[r][class_index] for r in range(len(matrix))) - true_positive
    false_negative = sum(matrix[class_index]) - true_positive

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def report_evaluation(truths, predictions, title):
    matrix = confusion_matrix(truths, predictions)
    correct = sum(matrix[i][i] for i in range(len(CLASS_NAMES)))

    print(f"--- {title} ---")
    print()
    print(f"Samples: {len(truths)}")
    if not truths:
        print("  nothing to evaluate")
        return None
    print(f"Accuracy: {correct / len(truths):.4f} ({correct}/{len(truths)})")
    print()

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(" " * 12 + "".join(f"{name:>12s}" for name in CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:>11s} " + "".join(f"{v:>12d}" for v in matrix[i]))
    print()

    print(f"{'class':>11s} {'precision':>10s} {'recall':>8s} {'f1':>8s} {'support':>8s}")
    for i, name in enumerate(CLASS_NAMES):
        precision, recall, f1 = precision_recall_f1(matrix, i)
        print(f"{name:>11s} {precision:10.4f} {recall:8.4f} {f1:8.4f} {sum(matrix[i]):8d}")
    print()

    # Rejecting a good photo is the expensive mistake in culling.
    sharp_support = sum(matrix[SHARP_INDEX])
    blurry_support = sum(matrix[BLURRY_INDEX])

    false_blurry = matrix[SHARP_INDEX][BLURRY_INDEX]
    false_sharp = matrix[BLURRY_INDEX][SHARP_INDEX]

    false_blurry_rate = false_blurry / sharp_support if sharp_support else 0.0
    false_sharp_rate = false_sharp / blurry_support if blurry_support else 0.0

    print("KEY PRODUCTION METRICS")
    print(f"  False Blurry Rate  (Sharp -> Blurry): {false_blurry_rate:.4f}"
          f"  ({false_blurry}/{sharp_support})   <-- most important")
    print(f"  False Sharp Rate  (Blurry -> Sharp):  {false_sharp_rate:.4f}"
          f"  ({false_sharp}/{blurry_support})")
    print()
    print("  Rejecting a good photo costs the user a keeper, so the first")
    print("  number is the one to drive down - even at the cost of the second.")
    print()

    return {
        "accuracy": correct / len(truths),
        "false_blurry_rate": false_blurry_rate,
        "false_sharp_rate": false_sharp_rate,
        "confusion_matrix": matrix,
    }


def save_curve(train_values, validation_values, title, y_label, filename):
    epochs = range(1, len(train_values) + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_values, marker="o", label="Training")
    plt.plot(epochs, validation_values, marker="s", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS_FOLDER / filename, dpi=130)
    plt.close()


# Grad-CAM shows WHERE the model looks, not WHAT it reads there. The
# closed-eye model attended to the eyes and still used sharpness as its cue,
# so treat these as one diagnostic, not proof.
def grad_cam(model, image_tensor, class_index, device):
    """Return a 224x224 heat map, 0 to 1."""
    model.eval()
    stored = {}

    def keep_activations(module, inputs, output):
        output.retain_grad()
        stored["activations"] = output

    handle = model.layer4.register_forward_hook(keep_activations)

    single = image_tensor.unsqueeze(0).to(device).requires_grad_(True)
    outputs = model(single)
    model.zero_grad()
    outputs[0, class_index].backward()
    handle.remove()

    activations = stored["activations"][0]
    gradients = stored["activations"].grad[0]

    weights = gradients.mean(dim=(1, 2))
    heat = torch.relu((weights[:, None, None] * activations).sum(dim=0))

    heat = torch.nn.functional.interpolate(
        heat[None, None], size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False
    )[0, 0]

    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / heat.max()

    return heat.detach().cpu().numpy()


def border_share(heat, margin=0.15):
    """Heat in the outer ring. High values point at borders or watermarks."""
    total = heat.sum()
    if total <= 0:
        return 0.0
    k = int(margin * heat.shape[0])
    return float((total - heat[k:-k, k:-k].sum()) / total)


def denormalize(image_tensor):
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    return (image_tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def save_gradcam_picture(image_tensor, heat, title, filename):
    picture = denormalize(image_tensor)

    figure, axes = plt.subplots(1, 2, figsize=(6.4, 3.4))
    axes[0].imshow(picture)
    axes[0].set_title("input", fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(picture)
    axes[1].imshow(heat, cmap="jet", alpha=0.45)
    axes[1].set_title("Grad-CAM", fontsize=9)
    axes[1].axis("off")
    figure.suptitle(title, fontsize=9)
    figure.tight_layout()

    GRADCAM_FOLDER.mkdir(parents=True, exist_ok=True)
    figure.savefig(GRADCAM_FOLDER / filename, dpi=120)
    plt.close(figure)


def run_gradcam_survey(model, dataset, subset_indices, truths, predictions, device):
    """Save example heat maps and summarise where attention falls."""
    subset = Subset(dataset, subset_indices)

    correct_by_class = {name: [] for name in CLASS_NAMES}
    wrong = []
    for position in range(len(subset)):
        if truths[position] == predictions[position]:
            correct_by_class[CLASS_NAMES[truths[position]]].append(position)
        else:
            wrong.append(position)

    chooser = random.Random(SEED)
    saved = 0

    for name in CLASS_NAMES:
        picks = chooser.sample(
            correct_by_class[name], min(GRADCAM_EXAMPLES, len(correct_by_class[name]))
        )
        for rank, position in enumerate(picks, start=1):
            image_tensor, label = subset[position]
            heat = grad_cam(model, image_tensor, label, device)
            save_gradcam_picture(
                image_tensor, heat,
                f"correct {name} - border share {border_share(heat):.2f}",
                f"correct_{name}_{rank}.png",
            )
            saved += 1

    for rank, position in enumerate(
        chooser.sample(wrong, min(GRADCAM_EXAMPLES, len(wrong))), start=1
    ):
        image_tensor, label = subset[position]
        heat = grad_cam(model, image_tensor, predictions[position], device)
        save_gradcam_picture(
            image_tensor, heat,
            f"WRONG: actual {CLASS_NAMES[label]}, predicted "
            f"{CLASS_NAMES[predictions[position]]} - border share {border_share(heat):.2f}",
            f"wrong_{rank}_{CLASS_NAMES[label]}_as_{CLASS_NAMES[predictions[position]]}.png",
        )
        saved += 1

    print(f"Saved {saved} Grad-CAM pictures to: {GRADCAM_FOLDER}")

    print()
    print("Border share of Grad-CAM heat (outer 15% ring; ~0.51 = no preference,")
    print("well below that means the model is reading picture content):")
    for name in CLASS_NAMES:
        sample = chooser.sample(
            correct_by_class[name], min(50, len(correct_by_class[name]))
        )
        shares = []
        for position in sample:
            image_tensor, label = subset[position]
            shares.append(border_share(grad_cam(model, image_tensor, label, device)))
        if shares:
            print(f"  {name:>10s}: mean {np.mean(shares):.3f}  median "
                  f"{np.median(shares):.3f}  (n={len(shares)})")


# Kept separate from public validation: the closed-eye model scored 97.7% on
# its own validation set and still failed on real photos.
def evaluate_on_real_photos(model, folder, device):
    """Score <folder>/Sharp and <folder>/Blurry - hand-labelled own crops."""
    folder = Path(folder)

    if not folder.exists():
        print(f"Real-photo folder not found: {folder}")
        return None

    missing = [name for name in ("Sharp", "Blurry") if not (folder / name).exists()]
    if missing:
        print(f"Real-photo folder must contain {missing} subfolders.")
        return None

    _, validation_transform = build_transforms()
    dataset = ImageFolder(folder, transform=validation_transform)

    if len(dataset) == 0:
        print("Real-photo folder contains no images.")
        return None

    # ImageFolder gives Blurry=0 / Sharp=1, matching training.
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    truths, predictions, _ = collect_predictions(model, loader, device)

    return report_evaluation(truths, predictions, "REAL PHOTOS (own domain)")


def dataset_ready():
    if not DATA_FOLDER.exists():
        return False, f"{DATA_FOLDER} does not exist"

    for name in ("Sharp", "Blurry"):
        folder = DATA_FOLDER / name
        if not folder.exists():
            return False, f"{folder} does not exist"
        if not any(folder.glob("*.jpg")):
            return False, f"{folder} has no images"

    return True, "ready"


def main():
    parser = argparse.ArgumentParser(description="Train the Sharp/Blurry classifier.")
    parser.add_argument(
        "--real-eval", metavar="DIR", default=None,
        help="folder with Sharp/ and Blurry/ subfolders of your own photos, "
             "evaluated separately from public validation",
    )
    arguments = parser.parse_args()

    set_seeds()
    device = pick_device()

    print("Blur / focus classifier - training")
    print("Using device:", device.type)
    print("Seed:", SEED)
    print()

    ready, reason = dataset_ready()
    if not ready:
        print("No dataset to train on:", reason)
        print()
        print("Nothing has been trained and no placeholder data was created.")
        print()
        print("To get to a first run:")
        print("  1. put the raw dataset somewhere under data/")
        print("  2. add an entry to SOURCE_DATASETS in src/prepare_blur_dataset.py")
        print("  3. python3 src/prepare_blur_dataset.py")
        print("  4. python3 src/train_blur_classifier.py")
        print()
        print("Then, before believing any accuracy number:")
        print("  5. hand-label a few dozen crops from your own photos into")
        print("     Sharp/ and Blurry/ folders, and run")
        print("     python3 src/train_blur_classifier.py --real-eval <that folder>")
        return 0

    train_transform, validation_transform = build_transforms()

    train_dataset_full = ImageFolder(DATA_FOLDER, transform=train_transform)
    validation_dataset_full = ImageFolder(DATA_FOLDER, transform=validation_transform)

    # Guard against a silent class-order swap between the two modules.
    folder_classes = [name.lower() for name in train_dataset_full.classes]
    if folder_classes != CLASS_NAMES:
        print(f"Class order mismatch: folders give {folder_classes}, "
              f"blur_classifier.py expects {CLASS_NAMES}.")
        print("Fix CLASS_NAMES / BLURRY_INDEX / SHARP_INDEX before training.")
        return 1

    manifest = load_manifest()

    print("Class mapping:", train_dataset_full.class_to_idx)
    print("Total images:", len(train_dataset_full))
    print("Manifest rows:", len(manifest) if manifest else "none found")
    print()

    print("--- source composition check (before training) ---")
    report_class_source_matrix(train_dataset_full, manifest)
    print()

    train_indices, validation_indices, group_total, leaked = split_by_group(
        train_dataset_full, manifest
    )

    print("--- grouped split ---")
    print("Groups:", group_total)
    print("Training images:", len(train_indices))
    print("Validation images:", len(validation_indices))
    print("  training class counts:  ", count_classes(train_dataset_full, train_indices))
    print("  validation class counts:", count_classes(train_dataset_full, validation_indices))
    if leaked:
        print(f"  moved {len(leaked)} mixed-label group(s) wholly into training "
              f"to stop a sharp original and its synthetic twin being split")
    print()

    print("--- per-class low-level statistics ---")
    report_low_level_statistics(train_dataset_full, train_indices)
    print("  (Blurry SHOULD sit lower - that is the real signal. A gigantic gap")
    print("   means the dataset is too easy and will not transfer.)")
    print()

    train_loader = DataLoader(
        Subset(train_dataset_full, train_indices), batch_size=BATCH_SIZE, shuffle=True
    )
    validation_loader = DataLoader(
        Subset(validation_dataset_full, validation_indices),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )

    trainable, frozen, names = describe_trainable(model)
    print("--- transfer learning setup ---")
    print(f"Trainable: {trainable:,} parameters in {names}")
    print(f"Frozen:    {frozen:,} parameters")
    print(f"layer4 joins in at epoch {UNFREEZE_LAYER4_AT_EPOCH}.")
    print()

    train_losses, train_accuracies = [], []
    validation_losses, validation_accuracies = [], []
    best_false_blurry_rate = None
    best_epoch = 0

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        if epoch == UNFREEZE_LAYER4_AT_EPOCH:
            for parameter in model.layer4.parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            )
            trainable, frozen, names = describe_trainable(model)
            print(f">>> Unfroze layer4. Trainable now {trainable:,} in {names}.")
            print()

        train_loss, train_accuracy = run_one_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        validation_loss, validation_accuracy = run_one_epoch(
            model, validation_loader, criterion, optimizer, device, training=False
        )

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

        truths, predictions, _ = collect_predictions(model, validation_loader, device)
        matrix = confusion_matrix(truths, predictions)
        sharp_support = sum(matrix[SHARP_INDEX])
        false_blurry_rate = (
            matrix[SHARP_INDEX][BLURRY_INDEX] / sharp_support if sharp_support else 1.0
        )

        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc: {train_accuracy:.4f}")
        print(f"Val Loss: {validation_loss:.4f}")
        print(f"Val Acc: {validation_accuracy:.4f}")
        print(f"False Blurry Rate (Sharp->Blurry): {false_blurry_rate:.4f}")

        # Checkpoint on the production metric; accuracy only breaks ties.
        score = (-false_blurry_rate, validation_accuracy)
        if best_false_blurry_rate is None or score > best_false_blurry_rate:
            best_false_blurry_rate = score
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_PATH)
            print("  (best so far - saved)")

        print()

    print("=" * 62)
    print(f"Best checkpoint: epoch {best_epoch}")
    print(f"  False Blurry Rate {-best_false_blurry_rate[0]:.4f}, "
          f"accuracy {best_false_blurry_rate[1]:.4f}")
    print("Saved to:", MODEL_PATH)
    print()

    model.load_state_dict(torch.load(MODEL_PATH))
    truths, predictions, _ = collect_predictions(model, validation_loader, device)
    report_evaluation(truths, predictions, "PUBLIC VALIDATION SET")

    save_curve(train_losses, validation_losses,
               "Blur: Training vs Validation Loss", "Loss", "loss_curve.png")
    save_curve(train_accuracies, validation_accuracies,
               "Blur: Training vs Validation Accuracy", "Accuracy", "accuracy_curve.png")
    print("Saved learning curves to:", RESULTS_FOLDER)
    print()

    print("--- Grad-CAM diagnostic ---")
    print()
    run_gradcam_survey(model, validation_dataset_full, validation_indices,
                       truths, predictions, device)
    print()
    print("Reminder: Grad-CAM shows WHERE the model looks, not WHAT it reads")
    print("there. It is one diagnostic, not proof of correct feature use.")
    print()

    if arguments.real_eval:
        print("=" * 62)
        evaluate_on_real_photos(model, arguments.real_eval, device)
    else:
        print("=" * 62)
        print("NO REAL-PHOTO EVALUATION WAS RUN.")
        print("The public validation number above is NOT evidence that this model")
        print("works on your photos. Hand-label crops from your own archive into")
        print("Sharp/ and Blurry/ folders and re-run with --real-eval <folder>.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
