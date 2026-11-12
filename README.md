# AI Photo Culling & Best-Shot Selection System

A tool for sorting through large batches of photos and picking out the keepers.

## What it will do

- **Analyze batches of photos.** Point it at a folder of images and it will look
  at every photo in that batch.
- **Group similar photos.** Photos taken seconds apart — the same pose, the same
  scene — get clustered together so you can compare them side by side instead of
  scrolling through hundreds of near-duplicates.
- **Rank photos and select the best shots.** Within each group it will score the
  photos and recommend the single best image to carry forward into editing.

## Project status

The command-line prototype recursively processes a folder, groups visually
similar images, ranks each group, and recommends one photo per group. It exports
an HTML review report, CSV rankings, and JSON measurements. Originals are never
changed. Closed-eye predictions are excluded from this workflow.

Ranking is a heuristic, not a validated model of photographic quality. Scores
are comparable **within a group only**. Validate the recommendations against
your own hand-ranked photo groups before relying on them.

## Code layout

```text
src/
  main.py                     command-line entry point
  workflow.py                 analysis sequence and console output
  grouping.py                 perceptual hash and color similarity groups
  ranking.py                  group ranking and recommendation
  reporting.py                HTML, CSV, and JSON export
  detection_session.py        isolated detector process and crash fallback
  config.py                   shared paths and analysis thresholds
  analysis/
    detection.py              people, faces, boxes, primary subject
    exposure.py               scene and portrait exposure
    sharpness.py              scene, face, and eye sharpness
    composition.py            subject composition scoring
    eye_state.py              face crop for closed-eye inference
    quality.py                normalized detail and clipping measurements
  closed_eye_classifier.py    trained closed-eye model inference
  blur_classifier.py          blur model inference
  prepare_*_dataset.py        dataset preparation entry points
  train_*.py                  training and evaluation entry points
```

Analysis modules can be imported independently, for example:

```python
from src.analysis.exposure import analyze_exposure
from src.analysis.detection import detect_faces
```

Dataset preparation imports detection directly rather than importing the
command-line entry point. Existing preparation and training commands retain
their original paths.

## Setup

Install the libraries:

```bash
pip3 install -r requirements.txt
```

Download the two MediaPipe models (a one-time step — they are not code, so they
are not stored in this repo):

```bash
mkdir -p models

# finds faces
curl -L -o models/blaze_face_short_range.tflite \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite

# finds the points of a face (eyes, nose, mouth)
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# finds whole people (bodies, not faces)
curl -L -o models/yolo11n.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt
```

The models serve these roles (batch ranking uses the first and third):

| Model | Finds |
|---|---|
| `blaze_face_short_range.tflite` | face rectangles |
| `face_landmarker.task` | points on a face (eyes, nose, mouth) |
| `yolo11n.pt` | whole people |

## Running it

```bash
python3 src/main.py
```

From the project root, package execution also works:

```bash
python3 -m src.main
```

Choose an input folder and report destination:

```bash
python3 src/main.py data/sample_photos --output results/culling
```

Use scene measurements without loading detection models:

```bash
python3 src/main.py data/sample_photos --no-detection
```

Open `results/culling/report.html` to review each group side by side. JSON includes
component measurements, detection failures, and skipped files. CSV includes
each photo's group, rank, recommendation, and scores. Supported inputs are JPEG,
PNG, TIFF, and WebP; EXIF orientation is applied. Damaged files are recorded and
the batch continues. Re-running excludes the output directory from input.
Detection runs in a reusable separate process. If a native model crashes or
times out, the batch continues with scene measurements and records the fallback
in each affected photo's notes.

## Grouping and scoring

- Grouping requires similar perceptual hashes, hue/saturation histograms, and
  aspect ratios. Every member must match every other member; similarity chains
  cannot merge unrelated endpoints. `--hash-distance` controls permissiveness
  (default 10, lower is stricter). It does not use timestamps or recognize people.
- A primary person is selected once per photo, and only a face inside that
  person's box and in its upper 45% is used for quality measurements. Without
  people, the largest detected face is used. This does not evaluate every person
  in a group portrait.
- Detail is measured after resizing the measurement region to a 512-pixel long
  side, mild noise smoothing, and local contrast normalization. If every member
  has a sufficiently large face and their relative face sizes agree within 1.5×,
  the whole group uses face measurements;
  otherwise it uses person crops if all members have usable person boxes, then
  scene measurements as the final fallback. Detail is scaled relative
  to the best measured detail in that group.
- Exposure penalizes clipped shadows, clipped color channels, and extreme
  median brightness. It does not penalize face-to-background brightness gaps.
  Deliberate silhouettes, saturated colors, texture, and strong noise can still
  confuse these heuristics.
- Weights are 65% detail, 30% exposure, and 5% composition. Composition is omitted
  and other weights renormalized if any member lacks a person box. The thirds
  bonus has headroom to affect the final score. These settings live in
  `src/config.py`.
- A difference below 3 points between the first two candidates marks the group
  for manual review. Multi-photo groups using person or scene measurements also
  require review because background or clothing detail can dominate focus.
  A winner is a suggestion; the tool neither deletes photos
  nor automatically exports originals.

## Checks

```bash
python3 -m unittest discover -s tests -v
```
