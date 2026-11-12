"""Closed-eye inference on an already-detected face crop. No detection here."""

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

MODEL_PATH = Path(__file__).parent.parent / "models" / "closed_eye_resnet18.pth"

# Must match training: ImageFolder sorted ClosedFace before OpenFace.
CLASS_NAMES = ["closed", "open"]
CLOSED_INDEX = 0
OPEN_INDEX = 1

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


def load_model():
    """Load once, then reuse."""
    global loaded_model, loaded_device

    if loaded_model is not None:
        return loaded_model, loaded_device

    if not MODEL_PATH.exists():
        return None, None

    device = pick_device()

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model.to(device)

    loaded_model = model
    loaded_device = device

    return loaded_model, loaded_device


def make_square_by_padding(image):
    """Pad to square. Stretching would distort the face, cropping could cut an eye."""
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


def prepare_face_crop(face_crop):
    if face_crop.mode != "RGB":
        face_crop = face_crop.convert("RGB")

    return make_square_by_padding(face_crop)


def predict_eye_state(face_crop):
    """Return probabilities and prediction, or None if the model or crop is missing."""
    if face_crop is None or face_crop.width < 1 or face_crop.height < 1:
        return None

    model, device = load_model()

    if model is None:
        return None

    square_crop = prepare_face_crop(face_crop)
    batch = preprocess(square_crop).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(batch), dim=1)[0].cpu()

    closed_probability = float(probabilities[CLOSED_INDEX])
    open_probability = float(probabilities[OPEN_INDEX])

    if closed_probability >= open_probability:
        prediction = CLASS_NAMES[CLOSED_INDEX]
        confidence = closed_probability
    else:
        prediction = CLASS_NAMES[OPEN_INDEX]
        confidence = open_probability

    return {
        "closed_probability": closed_probability,
        "open_probability": open_probability,
        "prediction": prediction,
        "confidence": confidence,
        "crop_size": (face_crop.width, face_crop.height),
        "square_size": (square_crop.width, square_crop.height),
    }
