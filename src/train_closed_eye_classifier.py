"""ResNet18: open eyes vs closed eyes.

Reads:   data/closed_eye_dataset_processed/{OpenFace, ClosedFace}
Writes:  models/closed_eye_resnet18.pth
         results/ml/closed_eye/{loss,accuracy}_curve.png, gradcam/
"""

import random
import re
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
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_FOLDER = Path(__file__).parent.parent
DATA_FOLDER = PROJECT_FOLDER / "data" / "closed_eye_dataset_processed"
MODEL_PATH = PROJECT_FOLDER / "models" / "closed_eye_resnet18.pth"
RESULTS_FOLDER = PROJECT_FOLDER / "results" / "ml" / "closed_eye"
GRADCAM_FOLDER = RESULTS_FOLDER / "gradcam"

SEED = 42

BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
VALIDATION_SHARE = 0.20

UNFREEZE_LAYER4_AT_EPOCH = 5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Roughly where the eyes sit in these crops. Diagnostic only, never training.
EYE_BAND_TOP = 0.20
EYE_BAND_BOTTOM = 0.55

GRADCAM_EXAMPLES = 4


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


def build_transforms():
    """Augmented for training, deterministic for validation."""
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.90, 1.0), ratio=(0.97, 1.03)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
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


def source_group_for(path):
    """Which original photo this face came from.

    ClosedFace names (closed_eye_1362.jpg_face_2.jpg) can share a source photo;
    those must not be split across train/validation. OpenFace is one per person.
    """
    name = Path(path).name

    match = re.match(r"(closed_eye_\d+)", name)
    if match:
        return "src:" + match.group(1)

    return "one:" + Path(name).stem


def split_by_group(dataset):
    """80/20 by source photo, done per class so both keep the proportion."""
    groups_per_class = {}
    for index in range(len(dataset)):
        path, label = dataset.samples[index]
        key = source_group_for(path)
        groups_per_class.setdefault(label, {}).setdefault(key, []).append(index)

    train_indices = []
    validation_indices = []
    group_total = 0

    for label in sorted(groups_per_class.keys()):
        groups = groups_per_class[label]
        group_total += len(groups)

        names = sorted(groups.keys())
        random.Random(SEED + label).shuffle(names)

        images_in_class = sum(len(v) for v in groups.values())
        target = int(images_in_class * VALIDATION_SHARE)

        taken = 0
        for name in names:
            if taken < target:
                validation_indices.extend(groups[name])
                taken += len(groups[name])
            else:
                train_indices.extend(groups[name])

    return sorted(train_indices), sorted(validation_indices), group_total


def count_classes(dataset, indices, class_names):
    counts = {name: 0 for name in class_names}
    for index in indices:
        counts[class_names[dataset.samples[index][1]]] += 1
    return counts


def build_model(device):
    """Pretrained ResNet18, backbone frozen, new 2-class head."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 2)

    return model.to(device)


def describe_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    names = sorted({n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad})
    return trainable, frozen, names


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

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())
            truths.extend(labels.tolist())

    return truths, predictions


def confusion_matrix(truths, predictions, number_of_classes):
    matrix = [[0] * number_of_classes for _ in range(number_of_classes)]
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


# Grad-CAM: weight the last conv block's feature maps by how much each
# influenced the chosen class score, then sum into a heat map.
#
# Caveat: this shows WHERE the model looks, not WHAT it reads there.
def grad_cam(model, image_tensor, class_index, device):
    """Return a 224x224 heat map, 0 to 1."""
    model.eval()

    stored = {}

    def keep_activations(module, inputs, output):
        output.retain_grad()
        stored["activations"] = output

    handle = model.layer4.register_forward_hook(keep_activations)

    # The input needs gradients so a graph exists even with frozen weights.
    single = image_tensor.unsqueeze(0).to(device).requires_grad_(True)

    outputs = model(single)
    model.zero_grad()
    outputs[0, class_index].backward()

    handle.remove()

    activations = stored["activations"][0]          # 512 maps of 7x7
    gradients = stored["activations"].grad[0]

    weights = gradients.mean(dim=(1, 2))
    heat = torch.relu((weights[:, None, None] * activations).sum(dim=0))

    heat = torch.nn.functional.interpolate(
        heat[None, None], size=(224, 224), mode="bilinear", align_corners=False
    )[0, 0]

    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / heat.max()

    return heat.detach().cpu().numpy()


def eye_band_share(heat):
    """Fraction of heat inside the eye band. The band covers 35%, so 0.35 = chance."""
    total = heat.sum()
    if total <= 0:
        return 0.0

    top = int(EYE_BAND_TOP * heat.shape[0])
    bottom = int(EYE_BAND_BOTTOM * heat.shape[0])

    return float(heat[top:bottom, :].sum() / total)


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

    # Mark the eye band.
    for axis in axes:
        axis.axhline(EYE_BAND_TOP * 224, color="white", linewidth=0.7, linestyle="--", alpha=0.8)
        axis.axhline(EYE_BAND_BOTTOM * 224, color="white", linewidth=0.7, linestyle="--", alpha=0.8)

    figure.suptitle(title, fontsize=9)
    figure.tight_layout()
    figure.savefig(GRADCAM_FOLDER / filename, dpi=120)
    plt.close(figure)


def main():
    set_seeds()
    device = pick_device()

    print("Using device:", device.type)
    print("Seed:", SEED)
    print()

    if not DATA_FOLDER.exists():
        print("Could not find the dataset at:", DATA_FOLDER)
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRADCAM_FOLDER.mkdir(parents=True, exist_ok=True)

    train_transform, validation_transform = build_transforms()

    train_dataset_full = ImageFolder(DATA_FOLDER, transform=train_transform)
    validation_dataset_full = ImageFolder(DATA_FOLDER, transform=validation_transform)

    class_names = train_dataset_full.classes
    print("Class mapping:", train_dataset_full.class_to_idx)
    print("Total images:", len(train_dataset_full))
    print()

    train_indices, validation_indices, group_total = split_by_group(train_dataset_full)

    print("--- Split (grouped by source photo, so related faces stay together) ---")
    print("Source groups:", group_total)
    print("Training images:", len(train_indices))
    print("Validation images:", len(validation_indices))
    print("  training class counts:  ", count_classes(train_dataset_full, train_indices, class_names))
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

    trainable, frozen, names = describe_trainable(model)
    print("--- Transfer learning setup ---")
    print(f"Trainable: {trainable:,} parameters in {names}")
    print(f"Frozen:    {frozen:,} parameters")
    print(f"layer4 joins in at epoch {UNFREEZE_LAYER4_AT_EPOCH}.")
    print()

    train_losses, train_accuracies = [], []
    validation_losses, validation_accuracies = [], []
    best_validation_accuracy = -1.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        if epoch == UNFREEZE_LAYER4_AT_EPOCH:
            for parameter in model.layer4.parameters():
                parameter.requires_grad = True

            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE
            )

            trainable, frozen, names = describe_trainable(model)
            print(f">>> Unfroze layer4. Trainable now {trainable:,} in {names}; frozen {frozen:,}.")
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

    print("=" * 62)
    print(f"Best validation accuracy: {best_validation_accuracy:.4f} at epoch {best_epoch}")
    print("Saved to:", MODEL_PATH)
    print()

    # Evaluate the best checkpoint, not the last epoch.
    model.load_state_dict(torch.load(MODEL_PATH))
    truths, predictions = collect_predictions(model, validation_loader, device)

    matrix = confusion_matrix(truths, predictions, len(class_names))
    correct = sum(matrix[i][i] for i in range(len(class_names)))

    print("--- Evaluation of best model ---")
    print()
    print(f"Validation samples: {len(truths)}")
    print(f"Validation accuracy: {correct / len(truths):.4f} ({correct}/{len(truths)})")
    print()
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(" " * 14 + "".join(f"{name:>14s}" for name in class_names))
    for i, name in enumerate(class_names):
        print(f"{name:>13s} " + "".join(f"{value:>14d}" for value in matrix[i]))
    print()
    print(f"{'class':>13s} {'precision':>10s} {'recall':>8s} {'f1':>8s} {'support':>8s}")
    for i, name in enumerate(class_names):
        precision, recall, f1 = precision_recall_f1(matrix, i)
        print(f"{name:>13s} {precision:10.4f} {recall:8.4f} {f1:8.4f} {sum(matrix[i]):8d}")
    print()

    save_curve(train_losses, validation_losses, "Closed-Eye: Training vs Validation Loss",
               "Loss", "loss_curve.png")
    save_curve(train_accuracies, validation_accuracies,
               "Closed-Eye: Training vs Validation Accuracy", "Accuracy", "accuracy_curve.png")
    print("Saved learning curves to:", RESULTS_FOLDER)
    print()

    print("--- Grad-CAM sanity check ---")
    print()

    validation_subset = Subset(validation_dataset_full, validation_indices)

    correct_by_class = {name: [] for name in class_names}
    wrong = []

    for position in range(len(validation_subset)):
        truth = truths[position]
        prediction = predictions[position]
        if truth == prediction:
            correct_by_class[class_names[truth]].append(position)
        else:
            wrong.append(position)

    chooser = random.Random(SEED)
    saved = 0
    band_shares = {name: [] for name in class_names}

    for name in class_names:
        picks = chooser.sample(
            correct_by_class[name], min(GRADCAM_EXAMPLES, len(correct_by_class[name]))
        )
        for rank, position in enumerate(picks, start=1):
            image_tensor, label = validation_subset[position]
            heat = grad_cam(model, image_tensor, label, device)
            share = eye_band_share(heat)
            band_shares[name].append(share)
            save_gradcam_picture(
                image_tensor, heat,
                f"correct {name} - eye-band share {share:.2f}",
                f"correct_{name}_{rank}.png",
            )
            saved += 1

    for rank, position in enumerate(chooser.sample(wrong, min(GRADCAM_EXAMPLES, len(wrong))), start=1):
        image_tensor, label = validation_subset[position]
        predicted = predictions[position]
        # Explain the wrong answer, so use the predicted class.
        heat = grad_cam(model, image_tensor, predicted, device)
        share = eye_band_share(heat)
        save_gradcam_picture(
            image_tensor, heat,
            f"WRONG: actual {class_names[label]}, predicted {class_names[predicted]}"
            f" - eye-band share {share:.2f}",
            f"wrong_{rank}_{class_names[label]}_as_{class_names[predicted]}.png",
        )
        saved += 1

    print(f"Saved {saved} Grad-CAM pictures to: {GRADCAM_FOLDER}")
    print(f"Misclassified validation images available: {len(wrong)}")
    print()

    print("Where is the model looking? (share of Grad-CAM heat inside the eye band)")
    band_height = EYE_BAND_BOTTOM - EYE_BAND_TOP
    print(f"  the band covers {band_height:.0%} of the image, so {band_height:.2f} = no preference")

    survey = {}
    for name in class_names:
        sample = chooser.sample(correct_by_class[name], min(60, len(correct_by_class[name])))
        shares = []
        for position in sample:
            image_tensor, label = validation_subset[position]
            shares.append(eye_band_share(grad_cam(model, image_tensor, label, device)))
        survey[name] = shares
        print(f"  {name:>12s}: mean {np.mean(shares):.3f}  median {np.median(shares):.3f}  "
              f"(n={len(shares)}, above chance in {sum(1 for s in shares if s > band_height)}/{len(shares)})")
    print()

    print("--- Overfitting check ---")
    gap = train_accuracies[-1] - validation_accuracies[-1]
    lowest_validation_loss = min(validation_losses)

    print(f"Final training accuracy:   {train_accuracies[-1]:.4f}")
    print(f"Final validation accuracy: {validation_accuracies[-1]:.4f}")
    print(f"Gap: {gap:+.4f}")
    print(f"Lowest validation loss: {lowest_validation_loss:.4f} "
          f"(epoch {validation_losses.index(lowest_validation_loss) + 1})")
    print(f"Final validation loss:  {validation_losses[-1]:.4f}")

    warnings = []
    if train_accuracies[-1] > 0.99:
        warnings.append("training accuracy is essentially perfect")
    if gap > 0.10:
        warnings.append("training accuracy is well above validation accuracy")
    if validation_losses[-1] > lowest_validation_loss * 1.15:
        warnings.append("validation loss rose after its lowest point")

    if warnings:
        print("Warning signs:")
        for warning in warnings:
            print("  -", warning)
    else:
        print("No strong overfitting signs in these numbers.")


if __name__ == "__main__":
    main()
