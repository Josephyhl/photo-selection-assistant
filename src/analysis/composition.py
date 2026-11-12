"""Subject positioning and composition scoring."""

from ..config import (
    MODERATE_EDGE_PENALTY,
    MODERATE_EDGE_ZONE,
    STRONG_EDGE_PENALTY,
    STRONG_EDGE_ZONE,
    THIRDS_BONUS,
    THIRDS_BONUS_DISTANCE,
    VERTICAL_THIRDS,
)


def analyze_subject_composition(image, primary_person):
    """Subject position/margins/thirds-proximity and a 0-100 composition score."""
    if primary_person is None:
        return None

    center_x = primary_person["x"] + primary_person["width"] / 2
    center_y = primary_person["y"] + primary_person["height"] / 2
    center_x_ratio = center_x / image.width
    center_y_ratio = center_y / image.height

    left_margin_ratio = primary_person["x"] / image.width
    right_margin_ratio = (image.width - (primary_person["x"] + primary_person["width"])) / image.width
    top_margin_ratio = primary_person["y"] / image.height
    bottom_margin_ratio = (image.height - (primary_person["y"] + primary_person["height"])) / image.height

    # Horizontal-only: a standing subject's feet reaching the bottom edge is normal.
    distance_to_side = min(center_x_ratio, 1 - center_x_ratio)

    if distance_to_side < STRONG_EDGE_ZONE:
        edge_penalty = STRONG_EDGE_PENALTY
    elif distance_to_side < MODERATE_EDGE_ZONE:
        edge_penalty = MODERATE_EDGE_PENALTY
    else:
        edge_penalty = 0

    distance_to_third = min(abs(center_x_ratio - third) for third in VERTICAL_THIRDS)
    thirds_bonus = THIRDS_BONUS if distance_to_third <= THIRDS_BONUS_DISTANCE else 0

    subject_area_ratio = (primary_person["width"] * primary_person["height"]) / (image.width * image.height)

    composition_score = 100 - THIRDS_BONUS - edge_penalty + thirds_bonus
    composition_score = max(0, min(100, composition_score))

    return {
        "center_x_ratio": center_x_ratio,
        "center_y_ratio": center_y_ratio,
        "left_margin_ratio": left_margin_ratio,
        "right_margin_ratio": right_margin_ratio,
        "top_margin_ratio": top_margin_ratio,
        "bottom_margin_ratio": bottom_margin_ratio,
        "subject_area_ratio": subject_area_ratio,
        "distance_to_third": distance_to_third,
        "edge_penalty": edge_penalty,
        "thirds_bonus": thirds_bonus,
        "composition_score": composition_score,
    }
