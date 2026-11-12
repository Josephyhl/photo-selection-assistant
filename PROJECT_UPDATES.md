# Project Updates — AI Photo Culling & Best-Shot Selection System

Chronological engineering / ML development record.

**Rules for this file**

- Append new entries at the bottom. Never rewrite or delete an old entry.
- If an earlier conclusion turns out to be wrong, say so in a *new* entry and
  reference the old one. The record of the mistake is the valuable part.
- Every entry answers: what changed, why, what was tested, what happened,
  what we learned, what we decided, what comes next.

---

# Part 1 — Backfilled History

> Reconstructed on 2026-09-07 from work completed 2026-09-05 → 2026-09-07.
> Grouped by module rather than by exact timestamp, since these entries were
> written after the fact. Entries from Part 2 onward are written as the work
> happens.

---

## v0.1 — Project skeleton

**Module:** project structure

**Implemented:** `data/`, `src/`, `results/` layout; `requirements.txt`;
`.gitignore` with `.gitkeep` placeholders so empty folders survive Git;
`main.py` printing a startup message.

**Problem being solved:** a clean, beginner-legible foundation with inputs,
code and outputs strictly separated, so image data never mixes with source.

**Decision:** keep all analysis in `main.py` until it grows too large; keep ML
training in separate modules from the start.

---

## v0.2 — Image loading (Pillow)

**Implemented:** read the first JPEG from `data/sample_photos/`, report
filename, width, height, colour mode.

**Learned:** `Image.open()` is lazy — it reads only the header until pixel data
is requested, so metadata on a 51 MP file is effectively free. Paths are built
from `Path(__file__).parent.parent` so the program runs from any directory.

---

## v0.3 — Global exposure baseline (heuristic)

**Module:** `analyze_exposure()`

**Implemented:** per-pixel brightness via the perceptual weighting
`0.299R + 0.587G + 0.114B`; average brightness; percentage of pixels below 25
("crushed") and above 230 ("blown"); score `100 − dark% − bright%`, clamped.

**Why:** an average alone cannot distinguish a flat mid-grey frame from a
half-black/half-white one — both average to 128. Measuring the extremes
separately is what makes the score informative.

**Results on samples:** a Yosemite landscape scored 91.6 (8.2% dark, 0.3%
bright) — plausible and consistent with the image.

**Limitations found:**
- Thresholds 25/230 and the subtraction formula are hand-chosen, not derived.
- Every pixel weighs the same: a blown forehead counts no more than a blown
  streetlight.
- No notion of intent — a deliberate silhouette scores as a failure.
- Cannot distinguish "underexposed" from "legitimately dark-coloured".

**Decision:** keep as the global baseline and the always-available fallback.

---

## v0.4 — Global sharpness baseline (Laplacian variance)

**Module:** `analyze_sharpness()`

**Implemented:** greyscale → `cv2.Laplacian(gray, cv2.CV_64F).var()`.
V1 score `min(variance / 10, 100)`.

**Why Laplacian:** blur spreads an abrupt brightness change across many
pixels. The Laplacian responds to the *second* derivative, so it ignores flat
areas *and* smooth gradients, firing only where brightness turns a corner.
Variance (not mean) is used because positive and negative edge responses
cancel to ~0 for any image.

**Results:** the landscape scored variance 227.1 → 22.7/100.

**Problems discovered:**
- The `/10` scale is wrong for real photos: a genuinely sharp 51 MP image
  scored 22.7/100.
- Noise reads as sharpness; texture (gravel, foliage, brick) inflates scores.
- **The key flaw:** a whole-image number is misleading for portraits. A
  well-shot portrait with deliberate background bokeh scores *low*; a photo
  focused past the subject onto a textured wall scores *high*.

**Learned:** Laplacian variance is only meaningful when comparing *similar*
images. Fortunately, that is exactly what culling does (comparing burst
frames) — but it motivated subject-aware sharpness.

**Next step at the time:** measure sharpness on the subject, not the frame.

---

## v0.5 — Face and subject detection

**Modules:** `detect_faces()`, `detect_faces_in_tiles()`,
`detect_faces_with_tiling()`, `detect_people()`,
`detect_face_with_person_guidance()`

### Environment problems solved first

- MediaPipe 1.0.1 **crashes hard** on Apple Silicon macOS when the face
  detector initialises (`DrishtiMetalHelper` / `Check failed: service_`), and
  it is not a catchable Python exception. Pinned `mediapipe<1.0` (0.10.35).
- The legacy `mp.solutions.face_detection` API is gone in both 1.0 and
  0.10.35 on Python 3.13. Migrated to the Tasks API
  (`FaceDetector` + `blaze_face_short_range.tflite`), which requires a
  downloaded model file (`models/`, documented in the README).

### The core detection problem

Full-frame MediaPipe found **zero faces** in environmental portraits that
clearly contained people. Root cause: BlazeFace shrinks its input to ~128×128,
so a face at 8% of frame width survives as ~10 px — too little structure.
Confirmed by cropping: a face undetectable at 9905×7429 was found reliably in
a 6000×4634 crop of the same photo.

### Fix 1 — tiled detection

2×2 grid at 60% tile size. Recovered one previously-impossible photo.
Then found a **structural flaw**: with a 2×2 grid, tiles must exceed 50% of
each side or a gap opens down the centre of every photo — and at 60% the tiles
were too coarse for one sample. No 2×2 setting worked for it.

### Fix 2 — 3×3 grid

`TILE_GRID = 3`, `TILE_SIZE_FRACTION = 0.40`, with a general position formula.
A centre tile now exists, and smaller tiles magnify more (3 × 0.40 = 1.20, so
~10% overlap per seam). Both viable portraits detected. Also reused one
detector across all nine tiles: **5.5× faster** (0.26 s vs 1.43 s).

### Fix 3 — YOLO person detection

Added `ultralytics` + `yolo11n.pt` (5.4 MB). Set
`PERSON_DETECTION_SIZE = 1280` rather than the default 640, because a distant
hiker scored 0.17 confidence at 640 and **0.82** at 1280.

Detected a back-facing subject occupying **3.0% of frame width** that no
amount of face detection could ever find — there was no face to find.

Cost: at 1280, YOLO also detects **painted figures in artwork** (0.41–0.46
confidence) — real false positives, though smaller and less confident than the
true subject.

### Fix 4 — person-guided face detection

`full image → YOLO person → primary person → crop → MediaPipe face`,
falling back to full-frame then 3×3 tiles.

**Measured on the facial dataset (98 images):**

| Path | Crops | Valid | Precision |
|---|---|---|---|
| person-guided | 49 | 49 | **100%** |
| tiled / full-frame fallback | 35 | 25 | 71% |

Overall valid crops rose from 59/82 (72%) to **74/84 (88%)**, with
**zero regressions** — the new failure list is a strict subset of the old.
Three former false positives (a university sign, a mural, a stone baluster)
converted into correct faces with large confidence gains.

**Decision:** keep YOLO and MediaPipe as complementary, not competing.
YOLO finds the body (works when facing away); MediaPipe finds the face and
landmarks (needed for eyes, exposure, expression). Neither replaces the other.

---

## v0.6 — Portrait sharpness with a fallback hierarchy

**Module:** `analyze_portrait_sharpness()`, `find_eye_boxes()`,
`measure_eye_sharpness()`

**Implemented:** a chain from most to least specific:

```
eye sharpness (when reliable)
  → face sharpness
    → global sharpness
```

Eye regions are derived from Face Landmarker points (outer/inner corners,
indices 33/133 and 362/263), scaled *from the eye width* so the box
self-scales with face size.

**Key finding:** the landmarker returned **all 478 landmarks for a face
wearing opaque sunglasses**. It fits a mesh template, so "landmarks returned"
is no evidence that eyes were seen. Reliability must be checked separately.

**V1 reliability check:** face width ≥ 200 px and eye crop ≥ 24 px.

**Honest limitation recorded at the time:** that check catches only "too few
pixels". Stress-tested by upscaling the sunglasses photo 4× so it passed the
size gate — the system then reported "reliable" and measured a dark lens and a
patch of cheek. The sunglasses photo was protected by luck of framing, not by
design.

**Decision:** compute the fallback value *before* attempting the specific one,
so failure is a `return` with a value already in hand.

---

## v0.7 — Exposure improvements (V2)

**Module:** `analyze_portrait_exposure()`, `analyze_face_exposure()`

**Problem:** person-box exposure was tested first and proved misleading.
A subject in a black jacket scored **47.0/100** while the photo's global
exposure was 82.1 — because 52.9% of the person box was below brightness 25.
The box was measuring *clothing*, not lighting.

**Fix:** measure the **face** instead. Skin brightness is reasonably
consistent across people; clothing carries no lighting information.

**Results (same photos, same untouched `analyze_exposure()`):**

| Photo | person-box | **face** |
|---|---|---|
| gallery portrait | 47.0 | **84.1** |
| backlit cliff portrait | 59.6 | **78.7** |

Both faces were in fact well-lit (+9.7 and +20.5 brightness above the scene).
The entire earlier deficit was clothing colour.

**Also added:** face-vs-global brightness difference with a tiered imbalance
penalty (<20 → 0, 20–40 → 5, 40–70 → 10, >70 → 20), using `abs()` so
too-dark and too-bright faces are treated alike. Verified symmetric.

**Decision:**
- face exposure when a face exists, global exposure otherwise
- **never** person-box exposure as the primary signal — for a back-facing
  subject the box contains even less skin
- person exposure retained as clearly-marked legacy/experimental code

**Learned:** choosing the right *region* mattered far more than refining the
formula. The same function went from 47.0 to 84.1 purely by being pointed
somewhere better.

---

## v0.8 — Subject-aware composition (V1)

**Module:** `analyze_subject_composition()`

**Implemented:** normalised subject centre and margins; a horizontal
extreme-edge penalty (centre inside 10% of a side → −30, 10–20% → −10); a
small rule-of-thirds bonus (+5 within 0.08 of a thirds line); subject area
ratio recorded but unscored.

**Results:** a subject at `center_x = 0.056` (4% margin one side, 93% the
other) correctly scored 70/100. A full-body subject whose feet reach the frame
bottom (`bottom margin 0.001`) correctly received **no** penalty.

**Deliberate design choices:**
- **No bottom-edge penalty.** Standing people touch the ground; that is
  gravity, not bad composition. An earlier person-scorer that counted *any*
  touched border was wrong about this.
- **Centred composition is not penalised.** Symmetry, formal portraits and
  eye-contact shots are all legitimately centred.

**Bug found and documented:** the thirds bonus can **never** affect the score.
Any position close enough to a thirds line to earn +5 is automatically far
enough from the edges to have zero penalty, so `100 − 0 + 5` clamps back to
100. Proven by sweeping centre positions. Left as specified; the fix is to
start from a base below 100.

**Decision:** focus and exposure are primary signals; composition is a small
bonus only, and must not reject otherwise-good photos.

---

## v0.9 — Facial-quality dataset preparation

**Module:** `src/prepare_face_dataset.py`

**Implemented:** detect the largest face, expand by 25% margin, square-crop
without stretching (slide inward at edges rather than trimming), resize to
224×224, write a manifest with confidence and detection path.

**Results:** 82/98 processed; all verified 224×224; 78 perfectly centred, 4
slid inward, 0 shrunk.

**Problems discovered by visual QC of every crop:**
- **~23 of 82 crops (28%) contained no face at all** — a graduation cap, a
  backpack, signage, a stone ornament, a mural. MediaPipe false positives,
  accepted silently.
- All false positives scored below 0.70 confidence, but ~22 *genuine* faces
  also scored below 0.70 — so confidence alone cannot separate them.
- **Detection failures were class-skewed:** 4 in Good Facial vs **12** in Bad
  Facial. Bad photos are bad because of blur, closed eyes and extreme angles —
  the same things that defeat a detector. So the hardest negatives were being
  silently dropped, biasing the class.

**Fix:** person-guided detection (v0.5) cut false positives from 23 to 10, and
its own 49 crops were 100% valid.

**Learned:** manual QC of training data is not optional. Automated cropping
produced a 28% label-noise rate that no metric would have surfaced.

---

## v0.10 — Closed-eye classifier V1 (CEW)

**Modules:** `src/prepare_closed_eye_dataset.py`,
`src/train_closed_eye_classifier.py`

**Dataset:** CEW (Closed Eyes In The Wild) — 1231 `OpenFace` + 1192
`ClosedFace`, all 100×100 RGB. Processed to 224×224 (already square, so
resize only). 2423/2423 processed, zero failures.

**Dataset inspection findings recorded at the time:**
- `OpenFace` filenames are LFW-style person names; `ClosedFace` are
  web-scraped `closed_eye_NNNN.jpg_face_N`. **The two classes come from
  different sources.**
- Stock-photo watermarks (iStockphoto, BIGSTOCK, depositphotos) appear only in
  `ClosedFace`.
- Demographics differ sharply (LFW skews older/male; ClosedFace skews
  younger/female, more makeup).
- Global brightness/saturation/edge-energy were nearly identical between
  classes, so no *global* shortcut was evident.

**Training:** ImageNet ResNet18, `fc` only for 4 epochs (1,026 params), then
`layer4` unfrozen (8.4 M params). AdamW-style setup: Adam, lr 1e-4, batch 32,
15 epochs. **Group-aware split** by source photo (2365 groups from 2423
images, so the 58 multi-face siblings stayed together).

**Results:**
- Best validation accuracy **97.73%** (473/484) at epoch 11
- Balanced confusion matrix: F1 0.9765 (Closed) / 0.9780 (Open)
- Train/val gap only **+0.0196**; validation loss plateaued, did not rise
- Grad-CAM: eye-band heat share **0.523** vs 0.35 chance (above chance in
  143/150); border-ring share **0.265** vs 0.51 chance — i.e. the model
  *avoided* the borders where watermarks live

**Conclusion drawn at the time:** the model had learned eye closure rather
than a spurious cue. **This conclusion was wrong — see v0.11.**

---

## v0.11 — Closed-eye real-world failure (important ML lesson)

**Module:** `src/closed_eye_classifier.py` (inference integration)

**What was tested:** the trained model wired into the production flow
(`full image → YOLO → primary person → person-guided MediaPipe → face crop →
ResNet18`) and run on real project photos.

### Results

| Test set | Predicted CLOSED |
|---|---|
| CEW validation (in-distribution) | 2.3% error, 97.7% accuracy |
| 108 real project photos (89 faces found) | **62% CLOSED** |
| 49 project face crops (real, open-eyed) | **78% CLOSED** |
| Curated *Good Facial*, fresh production flow (36 crops) | **38.9% CLOSED** |
| — of which clean person-guided subset (25 crops) | **44% CLOSED** |

Manual review of the 14 CLOSED predictions on the curated set: **13 were clear
false positives** with visible irises; 1 was unassessable because the crop was
cut at the eyebrow (a detection-framing issue, not a classifier one). No false
negatives; no image in the set actually had closed eyes.

Zoomed inspection confirmed the errors were not squinting — pupils and irises
were fully visible at closed probabilities of 0.836 to 0.998.

### Root-cause investigation

The inference module itself was verified correct first: **80/80** on processed
CEW, **39/40** on raw originals. No index inversion, no preprocessing bug.

Hypotheses tested and **ruled out**:
- **Crop framing** — sweeping margin from −20% to +50% left closed probability
  at 0.77–0.999. Not framing.
- **Demographics / eye shape** — CEW's own East Asian subjects classified
  **40/40 correctly**. Not demographics.
- **In-distribution fragility** — blurring or sharpening CEW images barely
  flipped them (105–119/120 held).

Hypothesis **confirmed** — an image-detail / sharpness shortcut:

CEW carries a detail confound. At native 100×100, `ClosedFace` has nearly
**double** the Laplacian variance of `OpenFace` (mean 443.5 vs 240.8; median
343.2 vs 189.6). Plausible cause: LFW faces are cropped from distant press
photos and therefore soft, while ClosedFace stock images are close-ups and
crisper. **In training, more detail correlated with closed eyes.**

Detail level *as the model sees it* (after resize to 224):

| Image set | mean Laplacian variance |
|---|---|
| CEW OpenFace (training) | 14 |
| CEW ClosedFace (training) | 23 |
| Pipeline inference crops | **94** |
| Project face crops | **587** |

Real crops carry **4× to 40×** more fine detail than anything in training.
The model extrapolated "more detail → closed" far beyond its training range.

Decisive experiment — degrade the real crops without touching eyelid geometry:

| Project face crops (real, open eyes) | CLOSED | mean p_closed |
|---|---|---|
| as-is (sharp) | 38/49 (78%) | 0.750 |
| downsampled to 100×100 and back | 29/49 (59%) | 0.631 |
| **Gaussian blur 2.0** | **7/49 (14%)** | **0.202** |

**Blur alone took the error from 78% to 14%. Nothing about the eyes changed.**

### Correction to v0.10

The v0.10 conclusion was too strong. Grad-CAM showed **where** the model looks;
it cannot show **what** it reads there. The texture cue lives *inside the eye
region*, so a model reading sharpness produces exactly the eye-centred
heatmaps that were observed. The eye-band enrichment and border avoidance were
real and did rule out watermarks — they could not rule out this.

The 97.7% was also real and honestly measured, with correct grouped
splitting. It simply measured performance on CEW's distribution.

### Lessons

1. **High in-distribution validation accuracy does not guarantee real-world
   performance.** Real-domain testing is a first-class evaluation step, not a
   nice-to-have.
2. **Grad-CAM localises attention, not causation.** Use it as one diagnostic
   among several, never as proof.
3. **Dataset source bias survives correct grouped splitting.** Grouping stops
   *leakage*; it does nothing about a *confound* where class correlates with
   source.
4. **If one source supplies one class, the model can classify by source.**
   Check per-class low-level statistics *before* training.
5. **Ablation by input degradation is a powerful diagnostic.** Changing one
   image property while holding the label fixed exposed the shortcut in a
   single experiment.

### Decision

- Keep the closed-eye ResNet18 as a baseline/experiment.
- **Do not apply its output to culling or ranking.** At 40–44% false CLOSED on
  curated keepers it would reject roughly 4 in 10 good photos.
- Candidates for later: MediaPipe Face Landmarker blink blendshapes
  (`eyeBlinkLeft` / `eyeBlinkRight`), trained on far more diverse data; or
  fine-tuning on hand-labelled crops from this archive.

---

# Part 2 — Ongoing Log

---

## v0.12 — Blur / focus classifier: infrastructure only

**Date:** 2026-09-07
**Module:** `src/blur_classifier.py`, `src/train_blur_classifier.py`,
`src/prepare_blur_dataset.py`

### 1. What changed

Created the code infrastructure for a new ML component: a **general
crop-level Sharp vs Blurry classifier**. No training was run; no dataset
exists yet.

- `blur_classifier.py` — inference only. `predict_blur(crop)` returns
  `{sharp_probability, blurry_probability, prediction, confidence, crop_size,
  square_size}`, or `None` when the model is untrained. `model_available()`
  lets callers skip cleanly. Contains **no** YOLO or MediaPipe.
- `prepare_blur_dataset.py` — maps one or many raw datasets into
  `data/blur_dataset_processed/{Sharp,Blurry}/` plus `manifest.csv`.
  `SOURCE_DATASETS = []`, so it exits with instructions rather than inventing
  data.
- `train_blur_classifier.py` — ResNet18 transfer learning, grouped split,
  anti-shortcut augmentation, full metrics, Grad-CAM, and a separate
  real-photo evaluation path. Exits cleanly when no dataset is present.

Nothing in the closed-eye code, ranking, exposure logic or Laplacian sharpness
was modified.

### 2. Why

The existing sharpness signal is Laplacian variance, which v0.4 showed is
only meaningful when comparing similar images, is fooled by noise and texture,
and has no absolute scale. A learned classifier could judge focus quality on a
single crop.

Deliberately **not portrait-specific**: the same model should serve face
crops, person crops and whole landscapes. The production pipeline decides
*which* region to send:

```
person detected
├── face available → face/person crop → blur classifier
└── no face        → person crop      → blur classifier
no person detected → whole image      → blur classifier
```

Keeping the classifier detector-agnostic means the crop-selection policy can
change without retraining.

### 3. What we tested

Lightweight syntax/import checks only, as instructed:

- `py_compile` clean on all three modules.
- `predict_blur(None)` → `None`; `predict_blur(crop)` with no trained model →
  `None`; `model_available()` → `False`.
- Square-padding: 90×140 → 140×140 RGB; 30×90 → 90×90.
- All four synthetic blur transforms produce valid same-size RGB output.
- `group_id_for` handles all three rules (`stem`, `parent`, regex).
- Train transform produces `(3, 224, 224)` across repeated random draws; val
  transform likewise, from arbitrary input aspect ratios.
- Augmentation classes each run standalone.
- Shared constants are the *same objects* across modules (`CLASS_NAMES`,
  `INPUT_SIZE` imported from `blur_classifier`), so they cannot drift.
- Both entry points exit 0 with actionable guidance and create nothing.

### 4. What happened

One real bug found and fixed: `synthetic_motion_blur` used
`ImageFilter.Kernel` with an 11×1 kernel, which Pillow rejects
(`ValueError: bad kernel size` — only 3×3 and 5×5 are supported). Rewrote it
as a shift-and-average over an edge-padded array, which works at any length
and direction and avoids wrap-around artefacts.

### 5. What we learned (carried forward from v0.11)

This model's shortcut risk is **worse** than the closed-eye model's, because
for a blur classifier *image detail is the actual signal*. The line between
"reads detail to judge focus" and "reads detail to guess which dataset this
came from" is thin. Countermeasures built in:

**Metadata preservation.** Manifest fields: `source_dataset`, `source_file`,
`group_id`, `label`, `processed_file`, `original_width/height`,
`processed_width/height`, `synthetic`, `synthetic_transform`.

**Pre-training source audit.** Both the prep and training scripts print a
class/source matrix and warn explicitly when one class comes from a single
source — the exact condition that broke the closed-eye model.

**Per-class low-level statistics.** Training prints per-class Laplacian
variance before the first epoch. Note the nuance: Blurry *should* score
lower — that is the real signal. The warning sign is a gap so extreme that a
single threshold would suffice, meaning the dataset is too easy to transfer.

**Two strictly separated transform categories.**

| | (A) label-preserving augmentation | (B) synthetic blur generation |
|---|---|---|
| Lives in | `train_blur_classifier.py` | `prepare_blur_dataset.py` |
| When | on the fly, train split only | once, up front, written to disk |
| Effect on label | must **never** change it | **deliberately** changes it |
| Strength | mild (blur radius ≤ 0.8 px, resolution factor ≤ 1.6×) | strong (blur 3.0, 4× downsample) |
| Recorded | no | yes — `synthetic=yes` + transform name |

(A) includes flip, ±8° rotation, mild colour jitter, **random JPEG quality
(55–95)**, **random resolution jitter (1.0–1.6×)**, **mild blur (≤0.8 px)** and
**mild sharpen** — the last four exist specifically to make resolution, JPEG
artefacts and global sharpness useless as source fingerprints. Blur and
sharpen are paired on purpose, to widen the global-sharpness distribution
within *both* classes.

(B) is `SYNTHETIC_BLUR_ENABLED = False` until the real Blurry images have been
inspected. A distinctive synthetic signature is just another shortcut waiting
to be learned.

**Group-aware split, extended.** Synthetic blurry samples inherit the
`group_id` of their sharp original, so a Sharp original and its blurred twin
can never land on opposite sides. The splitter also detects groups holding
*both* labels and forces them wholly into training.

**Production metric front and centre.** `False Blurry Rate (Sharp → Blurry)`
is printed every epoch, reported separately from
`False Sharp Rate (Blurry → Sharp)`, and is the **checkpoint selection
criterion** (accuracy only breaks ties). Rationale: in culling, wrongly
rejecting a good photo costs the user a keeper they never see again;
wrongly keeping a blurry one costs a few seconds of review.

**Real-domain evaluation is structural.** `--real-eval DIR` scores a
hand-labelled folder of the user's own crops. When it is omitted, the script
prints an explicit warning that the public number is *not* evidence the model
works. That warning is not skippable.

**Grad-CAM demoted.** Support is built (heat maps plus a border-share
statistic), with a comment recording that Grad-CAM localises attention rather
than establishing which feature is used.

### 6. Decision

- Binary Sharp/Blurry for V1 — no third "slightly soft" class yet.
- ResNet18 transfer learning, backbone frozen, `layer4` unfrozen at epoch 6.
- AdamW, lr 1e-4, weight decay 1e-4, batch 32, 18 epochs, seed 42.
- Do not touch the Laplacian sharpness path in `main.py` until this model has
  passed a real-domain test.
- Synthetic blur stays off until the real dataset is inspected.

### 7. Waiting on / next step

Blocked on the raw dataset. When it arrives:

1. Add an entry to `SOURCE_DATASETS` (worked example is in the file).
2. Run `prepare_blur_dataset.py`; **read the class/source matrix** before
   anything else.
3. Inspect real Blurry images, then decide on synthetic blur.
4. Choose `group_from` so crops, bursts and synthetic twins share a group.
5. Train; watch False Blurry Rate rather than accuracy.
6. Hand-label a few dozen crops from this archive and run `--real-eval`
   **before** drawing any conclusion.

Deferred: three-class severity, replacing Laplacian sharpness in the ranking,
awkward-expression model, exposure/sharpness fusion, final ranking, UI.
