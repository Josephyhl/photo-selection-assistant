"""Project paths and thresholds shared by the analysis pipeline."""

from pathlib import Path

PROJECT_FOLDER = Path(__file__).parent.parent
PHOTO_FOLDER = PROJECT_FOLDER / "data" / "sample_photos"

# Brightness runs 0 (black) to 255 (white). Below DARK_LIMIT = crushed shadow,
# above BRIGHT_LIMIT = blown highlight.
DARK_LIMIT = 25
BRIGHT_LIMIT = 230

# Downloaded models - see README for the download commands.
FACE_MODEL_PATH = PROJECT_FOLDER / "models" / "blaze_face_short_range.tflite"
FACE_LANDMARK_MODEL_PATH = PROJECT_FOLDER / "models" / "face_landmarker.task"

# Landmark indices for the inner/outer corner of each eye, as the viewer sees it.
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)

# V1 eye-sharpness reliability limits.
MIN_FACE_WIDTH_FOR_EYES = 200
MIN_EYE_CROP_PIXELS = 24

# YOLO person detection. "n" = nano, the smallest/fastest variant.
PERSON_MODEL_PATH = PROJECT_FOLDER / "models" / "yolo11n.pt"
PERSON_CLASS_ID = 0  # COCO class 0 = person
PERSON_CONFIDENCE_THRESHOLD = 0.4

# 640 is YOLO's default input size; a distant person in a landscape disappears
# at that size, so we use 1280 to keep small people visible.
PERSON_DETECTION_SIZE = 1280

# Grown around a person box before searching it for a face, since YOLO boxes
# can clip the top of the head.
PERSON_CROP_EXPANSION = 0.15

# Primary-subject scoring: subject_score = size + confidence + face_bonus - edge_penalty
# Plain round numbers, chosen by hand rather than tuned.
SIZE_WEIGHT = 2.0
CONFIDENCE_WEIGHT = 1.0
FACE_BONUS = 1.0
EDGE_PENALTY_WEIGHT = 0.5

# A face counts as "inside" a person when this much of the face box overlaps it.
FACE_INSIDE_THRESHOLD = 0.6

# A box edge within this fraction of the photo's size counts as touching the border.
EDGE_TOUCH_TOLERANCE = 0.01

# Portrait exposure: how far face brightness may drift from scene brightness
# before it's treated as a lighting problem.
EXPOSURE_DIFFERENCE_SMALL = 20
EXPOSURE_DIFFERENCE_MEDIUM = 40
EXPOSURE_DIFFERENCE_STRONG = 70

SMALL_EXPOSURE_PENALTY = 5
MEDIUM_EXPOSURE_PENALTY = 10
STRONG_EXPOSURE_PENALTY = 20

# Subject composition (V1). Horizontal only - vertical is skipped because a
# standing person's feet naturally reach the bottom edge.
STRONG_EDGE_ZONE = 0.10
MODERATE_EDGE_ZONE = 0.20
STRONG_EDGE_PENALTY = 30
MODERATE_EDGE_PENALTY = 10

VERTICAL_THIRDS = (1 / 3, 2 / 3)
THIRDS_BONUS_DISTANCE = 0.08
THIRDS_BONUS = 5

# Tiled face detection fallback. 3x3 grid, 40% tile size: three 40% tiles sum
# to 120% of each side, giving ~10% overlap per seam and a tile centred on
# the photo. Tiles must sum to over 100% or gaps appear between them.
TILE_GRID = 3
TILE_SIZE_FRACTION = 0.40

# Two boxes overlapping more than this are treated as one face.
DUPLICATE_OVERLAP_LIMIT = 0.3

# Batch ranking heuristics; tune these against hand-ranked real photo groups.
QUALITY_LONG_SIDE = 512
DETECTION_LONG_SIDE = 1600
MIN_RANK_FACE_SIZE = 64
GROUP_HASH_DISTANCE = 10  # out of 63 perceptual hash bits
GROUP_COLOR_SIMILARITY = 0.90
RANK_WEIGHTS = {'sharpness': 0.65, 'exposure': 0.30, 'composition': 0.05}
REVIEW_SCORE_MARGIN = 3.0
