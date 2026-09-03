# Perception Plan

How the robot will detect the ball, walls, allies, opponents, the goal, and
the forbidden zone — and why each target uses a different method.

## Detection targets and methods

| Target | Method | Why |
| --- | --- | --- |
| **Ball** | YOLO (custom-trained) or MediaPipe Object Detector (pretrained), not color alone | Rules say the ball *should* be orange/yellow, but that's not guaranteed on the day — a scuffed ball, different lighting, or a swapped ball would break a pure color detector with no fallback. A trained/pretrained detector recognizes the ball's *shape*, not just its color. |
| **Walls** | Black color mask + ultrasonic backup | Walls are painted black — high contrast, cheap to detect. Ultrasonic catches what the camera misses (blind spots, motion blur) right before contact. |
| **Ally** | HSV color blob, our side marker (red) | Fixed, known color — color detection is fast and reliable for this. |
| **Opponent** | HSV color blob, the other side's marker color (blue or yellow — set per match, not hardcoded) | Same as ally, just the other color. Needs a config value updated each match since side colors rotate. |
| **Goal** | Gap in the black wall line, or a distinct goal color if confirmed on-site | Not fully specified by the rulebook — confirm the actual goal appearance before committing to a method. |
| **Forbidden zone (near goal)** | Black tape line detection (color mask) + tracked position cross-check | Same toolkit as walls. `guidance/Navigator`'s dead-reckoning position is a cheap secondary check, but vision is authoritative since it self-corrects and doesn't drift. |

## Why YOLO or MediaPipe for the ball specifically

Color detection (`cv2.inRange` on HSV) is the right tool when a target's
color is fixed and known — that's true for the wall, the tape, and the
team markers, so use it there. It's the *wrong* tool for the ball if its
color can't be guaranteed, because there's no fallback when the color
assumption breaks — the detector just doesn't see the ball at all.

Two options, both already partly set up in this repo:

- **YOLO (`perception/train.py`, `perception/test.py`)** — train on your own
  photos of the actual ball/field. Most accurate if you can get real
  training images before the match, since it learns your exact ball, but
  needs that data collection + training step done first.
- **MediaPipe Object Detector (EfficientDet-Lite, pretrained)** — COCO's
  pretrained classes already include "sports ball," so this can work with
  **zero training** as a fallback or a quick first pass. Less accurate for
  an unusual-looking ball and can't tell teams apart (it's not built for
  that), but it's there immediately if training data isn't ready in time.

Practical combo: run color detection as a **fast first guess** (near-zero
cost) — if it finds a plausible orange/yellow blob, trust it and skip the
heavier model that frame. If color detection finds nothing (wrong color,
bad lighting), fall back to YOLO/MediaPipe on that frame only. This keeps
the common case cheap and only pays the heavier cost when color fails.

## Staying lightweight and fast on the Pi

- **Small input size**: already using 320×320 inference / 320×240 capture
  in `perception/test.py` — keep it there. Bigger frames cost roughly
  quadratically more compute for very little accuracy gain at this
  distance/task.
- **Smallest model variant**: `yolo11n` (nano), already selected in
  `perception/train.py`. Don't upgrade to a bigger variant unless accuracy
  genuinely requires it — the Pi's CPU is the bottleneck, not the model's
  capacity.
- **Frame skipping**: already implemented (`SKIP_FACTOR`) — only run the
  heavy detector every 3rd frame and reuse the last result in between.
  Works because the ball/robots don't teleport between frames at this
  camera rate.
- **ROI crop before anything else**: crop out regions that can't contain
  anything useful (ceiling, your own bumper) before color masking or
  inference — every pixel dropped early is compute you never spend.
- **Color detection as a gate, not just a target**: use it to decide
  *whether* to run the expensive model at all (see combo above), and to
  narrow the search region for it (crop to the color mask's bounding area
  before running YOLO/MediaPipe on just that patch, not the whole frame).
- **Convert color spaces once per frame**: one BGR→HSV conversion, reused
  for every color mask (ball/ally/opponent/wall/tape) — don't re-convert
  per target.
- **Avoid per-pixel Python loops**: everything above uses OpenCV's
  vectorized C++ ops (`inRange`, `bitwise_and`, `findContours`) — a
  hand-written pixel loop in Python would be orders of magnitude slower.
- **Quantized/optimized model export** (if accuracy allows): export the
  trained YOLO model to a Pi-optimized format (e.g. NCNN or TFLite int8)
  instead of running the raw PyTorch `.pt` weights — meaningfully faster
  inference on Pi CPU with a small accuracy tradeoff. Worth doing once the
  model is trained and working, not before.
