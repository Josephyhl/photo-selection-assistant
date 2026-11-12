"""Face-crop preparation for the closed-eye classifier."""

from ..closed_eye_classifier import predict_eye_state


def analyze_eye_state(image, faces, save_crop_to=None):
    """Closed-eye classifier on the largest detected face, or None if there is none."""
    if len(faces) == 0:
        return None

    selected_face = max(faces, key=lambda face: face["width"] * face["height"])

    left = max(0, selected_face["x"])
    top = max(0, selected_face["y"])
    right = min(image.width, selected_face["x"] + selected_face["width"])
    bottom = min(image.height, selected_face["y"] + selected_face["height"])

    if right <= left or bottom <= top:
        return None

    face_crop = image.crop((left, top, right, bottom))

    if save_crop_to is not None:
        save_crop_to.parent.mkdir(parents=True, exist_ok=True)
        face_crop.save(save_crop_to, quality=95)

    return predict_eye_state(face_crop)
