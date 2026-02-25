# EAS 510 — Digital Forensics Expert System

A rule-based expert system that matches modified images back to their originals using image processing techniques. Built in Python with OpenCV and Pillow.

---

## Project Structure

```
project/
├── originals/                  # Original reference images
├── modified_images/            # Easy modified images (Phase 1)
├── hard/                       # Hard combined-transformation images (Phase 2)
├── random/                     # Unrelated images (false positive test)
├── rules.py                    # V1: Rules 1–3
├── rules_v2.py                 # V2: Rules 1–4 (adds Rule 4 edge detection)
├── forensics_detective.py      # Main detective class (supports V1 and V2)
├── test_system.py              # Test runner script
├── results_v1.txt              # V1 output: modified_images + random
├── results_v1_hard.txt         # V1 output: hard folder only
└── results_v2.txt              # V2 output: modified + hard + random
```

---

## Setup Instructions

### Requirements

```
Python 3.8+
opencv-python
Pillow
numpy
```

### Install Dependencies

```bash
pip install opencv-python Pillow numpy
```

### Folder Setup

Make sure these folders exist in the same directory as the scripts:

```
originals/        — contains original_00.jpg ... original_09.jpg
modified_images/  — contains modified_*_*.jpg/png
hard/             — contains hard case images
random/           — contains random_*.jpg and random_noise_*.jpg
```

### Running the System

Open `test_system.py` and set exactly **one** flag to `True`:

```python
phase_1    = True   # V1: modified_images/ + random/   → results_v1.txt
phase_hard = True   # V1: hard/ only                   → results_v1_hard.txt
phase_2    = True   # V2: modified/ + hard/ + random/  → results_v2.txt
```

Then run:

```bash
python test_system.py
```

---

## Rule Explanations

### Rule 1 — Metadata Analysis `rules.py`

Compares file size and image dimensions between the input and each target.

- **File size ratio**  `min(size_a, size_b) / max(size_a, size_b)`. A compressed image will have a smaller file size, so a low ratio means significant compression. Same-size files score close to 15.
- **Dimension ratio** : Compares width and height separately using the same min/max ratio, then averages. A cropped image will have different dimensions, lowering this score.
- **Mode gate**: If one image is grayscale and the other is color (different PIL mode groups), the rule returns 0 immediately — they cannot be the same image.

**Why it works:** Metadata is fast to compute and provides strong signal for non-cropped modifications like brightness, compression, and format changes. It scores low for crops (different dimensions) but those cases are covered by Rules 2 and 3.

---

### Rule 2 — Color Histogram Analysis`rules.py`

Compares the color distribution of both images using HSV histograms.

- Both images are resized to 256×256 pixels to remove spatial information.
- Converted to **HSV color space** and histograms computed on **H (hue) and S (saturation) channels only**, deliberately excluding V (brightness/value). This means brightness changes have near-zero effect on the score.
- `cv2.compareHist(HISTCMP_CORREL)` returns correlation in [-1, 1], clamped to [0, 1].
- **Sub-region comparison for crops**: The target is also divided into 4 quadrants and each quadrant's histogram is compared against the input. The best quadrant score is taken. A 25% crop came from one region of the image, so one quadrant should match closely even if the full image doesn't.

**Why it works:** Color distribution is a strong fingerprint that survives brightness changes, format conversion, and mild compression. The quadrant comparison extends coverage to cropped images without increasing false positives, since random images will not match any quadrant.

---

### Rule 3 — Template Matching `rules.py`

Uses `cv2.matchTemplate()` to detect whether the input image visually appears inside the target image.

- **Size-aware approach**: Computes `area_ratio = input_area / target_area`. For a 25% crop, this is 0.25, so `linear_ratio = sqrt(0.25) = 0.5`. The search image (target) is set to 500×500 and the template (input) is set to 250×250, so the template occupies the correct proportional space in the search image and has room to slide around to find the best position.
- **Attempt 2 for same-size images**: Input resized to 200×200 as template, target to 240×240 as search (20% larger). Handles brightness, compression, and format changes where both images are the same size.
- Takes the best score from both attempts. `TM_CCOEFF_NORMED` returns a normalized correlation in [-1, 1].

**Why it works:** Template matching directly detects whether one image's visual content appears in another, regardless of brightness or compression. The size-aware ratio ensures the template has enough room to move within the search image, which is critical for crop cases where the old fixed-size approach would fail.

---

### Rule 4 — Canny Edge Detection`rules_v2.py` — V2 only

Compares edge maps extracted from both images using `cv2.Canny()`.

**Why it was added:** V1 failed on `resize_scale75` and `crop+resized` images because Rule 3's size-aware ratio calculated incorrectly when the image was resized to a different resolution — the template ended up too small. Edges are scale-independent: the same structural boundaries appear in the same relative positions regardless of whether the image is 800×600 or 600×450.

- Both images are loaded as grayscale.
- Gaussian blur (5×5) is applied to reduce compression noise before edge detection.
- `cv2.Canny()` with **fixed thresholds (50, 150)** extracts edges. Fixed thresholds were chosen over adaptive (median-based) because very dark or heavily compressed images can have a near-zero median, causing adaptive thresholds to produce empty edge maps.
- Edge maps use the same **size-aware template matching** as Rule 3: `area_ratio` → `linear_ratio` → proportional template and search sizes. This handles both resized and cropped cases.
- Attempt 2 handles same-size modifications with fixed 200×200 template in 240×240 search.

**Why it works:** Edges represent structural boundaries — outlines of objects, corners, texture edges — that survive resizing and moderate compression. Comparing edge maps rather than raw pixels removes sensitivity to brightness and color changes entirely.

---

## V1 → V2 Reflection

### V1 Performance

| Folder | Matched | Rejected | Accuracy |
|---|---|---|---|
| `modified_images/` | 60/60 | 0 | 100% |
| `random/` | 0 FP | 15/15 | 0% false positives |
| `hard/` | 52/60 | 8 | 87% |

### V1 Failure Pattern

7 of 8 hard case failures in V1 shared the same pattern: images with **resize transformations** combined with compression:

```
original_01__resize_scale75__compress__q30   → 0/100 REJECTED
original_02__crop_keep60__resized__q60       → 0/100 REJECTED
original_03__resize_scale75__compress__q45   → 0/100 REJECTED
original_03__crop_keep60__resized__q60       → 0/100 REJECTED
original_04__crop_keep50__resized__q45       → 0/100 REJECTED
original_08__crop_keep60__resized__q45       → 0/100 REJECTED
```

**Why V1 failed:** Rule 3 computes `area_ratio` from raw pixel dimensions. When an image is resized (e.g. 75% scale), the pixel dimensions change but the visual content is identical. The ratio calculation then produces a template that is too small to find a meaningful match, so `matchTemplate` returns near-zero scores. Rule 1 also fails because dimensions differ. With both Rules 1 and 3 near zero, total score fell below the threshold even when Rule 2 was strong.

Rule 4's edge-based matching is **scale-independent** because both images are processed through Canny before comparison, and the same size-aware sliding window approach is used. This directly addresses the failure pattern.

### V2 Performance

| Folder | Matched | Rejected | Accuracy |
|---|---|---|---|
| `modified_images/` | 60/60 | 0 | 100% |
| `hard/` | 60/60 | 0 | 100% |
| `random/` | 1 FP | 14/15 | 93% correct |

V2 resolved all 8 hard case failures. One random image produced a false positive — this is a borderline edge case where the random image happened to share enough structural edge patterns with one original to pass the combined threshold. The overall false positive rate remains very low at 7%.

### Key Design Decisions in V2

**Scoring:** V2 uses a raw total out of 120 (30+30+40+20) rather than normalizing to 100, so that rule scores always add up to the displayed total. The match threshold was adjusted to 62.

**Bonus system:** Two confidence bonuses are applied after scoring:
- If Rule 3 score ≥ 3: total += 12 (visual match confirmed)
- If Rule 1 ≥ 10 and Rule 2 ≥ 8: total += 10 (metadata and color both agree)

These bonuses handle edge cases where combined transformations weaken individual rules below their thresholds while still providing genuine evidence across multiple signals.
