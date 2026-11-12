"""Sharp vs Blurry inference on a single crop. No detection here.

The pipeline picks the crop; this only judges it. Intentional background bokeh
is not "blurry", so send a subject crop rather than the whole frame when a
subject exists.

Untrained: predict_blur() returns None and model_available() returns False.
"""

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

MODEL_PATH = Path(__file__).parent.parent / "models" / "blur_resnet18.pth"

# Must match the training folder names (ImageFolder sorts alphabetically).
# A silent swap here inverts every prediction.
CLASS_NAMES = ["blurry", "sharp"]
BLURRY_INDEX = 0
SHARP_INDEX = 1

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INPUT_SIZE = 224

loaded_model = None
loaded_device = None

preprocess = transforms.Compose(
    [
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model():
    """Shared with the training script so the architecture is defined once."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def model_available():
    return MODEL_PATH.exists()


def load_model():
    """Load once, then reuse."""
    global loaded_model, loaded_device

    if loaded_model is not None:
        return loaded_model, loaded_device

    if not model_available():
        return None, None

    device = pick_device()

    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model.to(device)

    loaded_model = model
    loaded_device = device

    return loaded_model, loaded_device


def make_square_by_padding(image):
    """Pad to square. Stretching would change how blurred the crop looks."""
    if image.width == image.height:
        return image

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

    return padded


def prepare_crop(crop):
    if crop.mode != "RGB":
        crop = crop.convert("RGB")

    return make_square_by_padding(crop)


def predict_blur(crop):
    """Return probabilities and prediction, or None if the model or crop is missing."""
    if crop is None or crop.width < 1 or crop.height < 1:
        return None

    model, device = load_model()

    if model is None:
        return None

    square_crop = prepare_crop(crop)
    batch = preprocess(square_crop).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(batch), dim=1)[0].cpu()

    blurry_probability = float(probabilities[BLURRY_INDEX])
    sharp_probability = float(probabilities[SHARP_INDEX])

    if sharp_probability >= blurry_probability:
        prediction = CLASS_NAMES[SHARP_INDEX]
        confidence = sharp_probability
    else:
        prediction = CLASS_NAMES[BLURRY_INDEX]
        confidence = blurry_probability

    return {
        "sharp_probability": sharp_probability,
        "blurry_probability": blurry_probability,
        "prediction": prediction,
        "confidence": confidence,
        # For checking inference crops against training crops (domain shift).
        "crop_size": (crop.width, crop.height),
        "square_size": (square_crop.width, square_crop.height),
    }
