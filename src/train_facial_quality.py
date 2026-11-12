"""ResNet18: Good Facial vs Bad Facial.

Prototype only. ~74 usable images, so this checks the pipeline works end to
end, not that the model is accurate.
"""

import random
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

PROJECT_FOLDER = Path(__file__).parent.parent
DATA_FOLDER = PROJECT_FOLDER / "data" / "face-dataset-processed-person-guided"
MODEL_FOLDER = PROJECT_FOLDER / "models"
RESULTS_FOLDER = PROJECT_FOLDER / "results" / "ml"
MODEL_PATH = MODEL_FOLDER / "facial_quality_resnet18.pth"

SEED = 42

# Face-detector false positives - these crops contain no face.
EXCLUDED_FILES = {
    # Good Facial
    "B0011972.jpg",  # university sign
    "B0011952.jpg",  # shopfront
    "B0010727.jpg",  # distant figure in a doorway
    "B0011733.jpg",  # backpack
    "B0010620.jpg",  # carved stone ornament
    "B0010745.jpg",  # graduation cap
    # Bad Facial
    "B0011954.jpg",  # dark interior
    "B0011872.jpg",  # distant footballer
    "B0011978.jpg",  # distant graduate
    "B0011950.jpg",  # window glass
}

# Filenames are camera counters (B0010593.jpg). Files within this many counts
# are treated as one burst and never split across train/validation.
GROUP_GAP = 5

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4
VALIDATION_SHARE = 0.20
UNFREEZE_LAYER4_AT_EPOCH = 8

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.mps.manual_seed(SEED) if torch.backends.mps.is_available() else None


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def keep_this_file(path):
    """ImageFolder include/exclude callback."""
    file_path = Path(path)

    if file_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        return False

    return file_path.name not in EXCLUDED_FILES


def build_transforms():
    # Light only - stronger augmentation could alter the facial expression,
    # which is the thing being classified.
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return train_transform, validation_transform


def group_name_for(filename):
    stem = Path(filename).stem

    # Anything not named like B0012345 gets its own group.
    if not stem.startswith("B00") or not stem[3:].isdigit():
        return stem

    return int(stem[3:])


def build_groups(dataset):
    """Chain nearby photo numbers into bursts. One label per image."""
    numbers = []
    for index in range(len(dataset)):
        path = dataset.samples[index][0]
        numbers.append((group_name_for(path), index))

    numeric = sorted((n, i) for n, i in numbers if isinstance(n, int))

    labels = [None] * len(dataset)
    group_id = 0
    previous_number = None

    for number, index in numeric:
        if previous_number is None or number - previous_number > GROUP_GAP:
            group_id += 1
        labels[index] = f"burst{group_id}"
        previous_number = number

    for name, index in numbers:
        if labels[index] is None:
            labels[index] = f"single:{name}"

    return labels


def split_by_group(dataset, group_labels):
    """Assign whole bursts to validation, never half a burst."""
    images_in_group = {}
    for index, label in enumerate(group_labels):
        images_in_group.setdefault(label, []).append(index)

    group_names = sorted(images_in_group.keys())

    shuffler = random.Random(SEED)
    shuffler.shuffle(group_names)

    target_validation_size = int(len(dataset) * VALIDATION_SHARE)

    validation_indices = []
    training_indices = []

    for name in group_names:
        if len(validation_indices) < target_validation_size:
            validation_indices.extend(images_in_group[name])
        else:
            training_indices.extend(images_in_group[name])

    return sorted(training_indices), sorted(validation_indices), len(group_names)


def count_classes(dataset, indices, class_names):
    counts = {name: 0 for name in class_names}
    for index in indices:
        label = dataset.samples[index][1]
        counts[class_names[label]] += 1
    return counts


def build_model(device):
    """Pretrained ResNet18, backbone frozen, new 2-class head."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 2)

    return model.to(device)


def trainable_parameter_report(model):
    trainable = 0
    frozen = 0
    trainable_names = []

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable += parameter.numel()
            trainable_names.append(name)
        else:
            frozen += parameter.numel()

    return trainable, frozen, trainable_names


def run_one_epoch(model, loader, criterion, optimizer, device, training):
    if training:
        model.train()
    else:
        model.eval()

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
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            all_predictions.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())

    return all_labels, all_predictions


def confusion_matrix(true_labels, predicted_labels, number_of_classes):
    """Rows = actual, columns = predicted."""
    matrix = [[0] * number_of_classes for _ in range(number_of_classes)]
    for true_label, predicted in zip(true_labels, predicted_labels):
        matrix[true_label][predicted] += 1
    return matrix


def precision_recall_f1(matrix, class_index):
    true_positive = matrix[class_index][class_index]
    false_positive = sum(matrix[r][class_index] for r in range(len(matrix))) - true_positive
    false_negative = sum(matrix[class_index]) - true_positive

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return precision, recall, f1


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
    plt.savefig(RESULTS_FOLDER / filename, dpi=130)
    plt.close()


def main():
    set_seeds()
    device = pick_device()

    print("Using device:", device.type)
    print("Seed:", SEED)
    print()

    if not DATA_FOLDER.exists():
        print("Could not find the dataset at:", DATA_FOLDER)
        return

    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    train_transform, validation_transform = build_transforms()

    # Two views of the same folder so each split gets its own transform.
    train_dataset_full = ImageFolder(
        DATA_FOLDER, transform=train_transform, is_valid_file=keep_this_file
    )
    validation_dataset_full = ImageFolder(
        DATA_FOLDER, transform=validation_transform, is_valid_file=keep_this_file
    )

    class_names = train_dataset_full.classes
    print("Class mapping:", train_dataset_full.class_to_idx)
    print("Excluded false-positive crops:", len(EXCLUDED_FILES))
    print("Total usable images:", len(train_dataset_full))
    print()

    group_labels = build_groups(train_dataset_full)
    train_indices, validation_indices, number_of_groups = split_by_group(
        train_dataset_full, group_labels
    )

    print("Burst groups found:", number_of_groups)
    print("Training images:", len(train_indices))
    print("Validation images:", len(validation_indices))
    print("  training class counts:", count_classes(train_dataset_full, train_indices, class_names))
    print("  validation class counts:", count_classes(train_dataset_full, validation_indices, class_names))
    print()

    train_loader = DataLoader(
        Subset(train_dataset_full, train_indices), batch_size=BATCH_SIZE, shuffle=True
    )
    validation_loader = DataLoader(
        Subset(validation_dataset_full, validation_indices), batch_size=BATCH_SIZE, shuffle=False
    )

    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE
    )

    trainable, frozen, names = trainable_parameter_report(model)
    print("--- Transfer learning setup ---")
    print(f"Trainable parameters: {trainable:,}  ({', '.join(names)})")
    print(f"Frozen parameters:    {frozen:,}")
    print(f"Everything except the new final layer is frozen for the first "
          f"{UNFREEZE_LAYER4_AT_EPOCH} epochs.")
    print()

    train_losses, train_accuracies = [], []
    validation_losses, validation_accuracies = [], []

    best_validation_accuracy = -1.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        if epoch == UNFREEZE_LAYER4_AT_EPOCH:
            for parameter in model.layer4.parameters():
                parameter.requires_grad = True

            # Rebuild the optimizer so it picks up the new parameters.
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE
            )

            trainable, frozen, _ = trainable_parameter_report(model)
            print(f">>> Unfroze layer4 at epoch {epoch}. "
                  f"Trainable now {trainable:,}, frozen {frozen:,}.")
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

        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc: {train_accuracy:.4f}")
        print(f"Val Loss: {validation_loss:.4f}")
        print(f"Val Acc: {validation_accuracy:.4f}")

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_PATH)
            print("  (best so far - saved)")

        print()

    print("=" * 60)
    print(f"Best validation accuracy: {best_validation_accuracy:.4f} at epoch {best_epoch}")
    print("Saved to:", MODEL_PATH)
    print()

    # Evaluate the best checkpoint, not the last epoch.
    model.load_state_dict(torch.load(MODEL_PATH))
    true_labels, predicted_labels = collect_predictions(model, validation_loader, device)

    matrix = confusion_matrix(true_labels, predicted_labels, len(class_names))
    correct = sum(matrix[i][i] for i in range(len(class_names)))

    print("--- Evaluation of best model on validation set ---")
    print()
    print(f"Validation images: {len(true_labels)}")
    print(f"Validation accuracy: {correct / len(true_labels):.4f} "
          f"({correct}/{len(true_labels)})")
    print()

    print("Confusion matrix (rows = actual, columns = predicted):")
    header = " " * 16 + "".join(f"{name[:10]:>12s}" for name in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = "".join(f"{value:>12d}" for value in matrix[i])
        print(f"{name[:14]:>14s}  {row}")
    print()

    print(f"{'class':>14s} {'precision':>10s} {'recall':>8s} {'f1':>8s} {'support':>8s}")
    for i, name in enumerate(class_names):
        precision, recall, f1 = precision_recall_f1(matrix, i)
        support = sum(matrix[i])
        print(f"{name:>14s} {precision:10.3f} {recall:8.3f} {f1:8.3f} {support:8d}")
    print()

    save_curve(train_losses, validation_losses, "Training vs Validation Loss",
               "Loss", "loss_curve.png")
    save_curve(train_accuracies, validation_accuracies, "Training vs Validation Accuracy",
               "Accuracy", "accuracy_curve.png")
    print("Saved charts to:", RESULTS_FOLDER)
    print()

    print("--- Overfitting check ---")
    final_train_accuracy = train_accuracies[-1]
    final_validation_accuracy = validation_accuracies[-1]
    accuracy_gap = final_train_accuracy - final_validation_accuracy

    best_validation_loss = min(validation_losses)
    final_validation_loss = validation_losses[-1]

    print(f"Final training accuracy:   {final_train_accuracy:.4f}")
    print(f"Final validation accuracy: {final_validation_accuracy:.4f}")
    print(f"Gap: {accuracy_gap:+.4f}")
    print(f"Lowest validation loss: {best_validation_loss:.4f} "
          f"(epoch {validation_losses.index(best_validation_loss) + 1})")
    print(f"Final validation loss:  {final_validation_loss:.4f}")

    warnings = []
    if final_train_accuracy > 0.95:
        warnings.append("training accuracy is near perfect")
    if accuracy_gap > 0.15:
        warnings.append("training accuracy is far above validation accuracy")
    if final_validation_loss > best_validation_loss * 1.15:
        warnings.append("validation loss rose after its lowest point")

    if warnings:
        print("Warning signs of overfitting:")
        for warning in warnings:
            print("  -", warning)
    else:
        print("No strong overfitting signs in these numbers.")

    print()
    print(f"Note: the validation set holds only {len(true_labels)} images, so a "
          f"single image is worth {100 / len(true_labels):.1f} percentage points.")


if __name__ == "__main__":
    main()
