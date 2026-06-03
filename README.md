# BlueROV2 LED-Based Tracking with OpenCV

Python/OpenCV pipeline and Unity LED control for LED-based visual tracking of a BlueROV2 model. The system detects LED blobs from Unity PNG sequences (or live video later), decodes temporal blink patterns, identifies the visible face, estimates relative distance from same-face LED spacing, and produces controller-ready JSON observation packets for a Linux-side control module.

---

## Project Goal

The BlueROV2 model has 8 LEDs in total:

* 2 LEDs on the front face
* 2 LEDs on the back face
* 2 LEDs on the left face
* 2 LEDs on the right face

Each face emits a unique binary pattern. The two LEDs on the same face use the same pattern and the same phase.

This allows the vision system to:

1. detect candidate LED blobs,
2. track blobs over time and match same-face pairs,
3. decode the temporal blink pattern,
4. verify the visible face (`FRONT`, `BACK`, `LEFT`, `RIGHT`),
5. compute the midpoint of the LED pair,
6. calculate image-center alignment error,
7. estimate distance using the pixel distance between the two LEDs,
8. generate a controller-ready observation packet.

Color is used only as an initial candidate detection cue. Final decisions rely on temporal pattern matching, two-LED geometric consistency, and temporal stability.

---

## Current Development Stage

Controlled tests focus on the **back face** (primary follower scenario: observing the leader’s rear). The matcher decodes all four face patterns from `unity/RovLeds.cs` but validation datasets are back-only.

Back-face Unity setup:

```text
Active face: back only
LED color: green
Pattern: 11001100
FPS: 60
Bit duration: 0.1 s
Frames per bit: 6
```

Default pair selection: **spatio-temporal matcher** (centroid tracking + per-track ON/OFF buffers + correlation + pattern decode + geometry). Legacy **largest-two-blobs** mode remains for parity (`--matcher legacy_largest2`).

---

## Pipeline Overview

```text
PNG frame sequence
        ↓
HSV LED candidate extraction (vision_core)
        ↓
Centroid tracking + spatio-temporal pair matching
        ↓
Frame-level ON/OFF bit + fused pair bit
        ↓
Face pattern decode (BACK / FRONT / LEFT / RIGHT)
        ↓
Pixel-distance range estimation (calibrated model)
        ↓
Midpoint / image error / camera ray
        ↓
Controller-ready JSON observation packet
        ↓
Optional UDP send (send-udp / recv-udp)
```

---

## Repository Structure

```text
.
├── README.md
├── Rules.md
├── requirements.txt
├── main.py                 # CLI entry point
├── run_tests.py            # Offline validation suite (6 datasets)
│
├── bluerov_led/            # Core package
│   ├── config.py           # VisionConfig, face patterns, paths
│   ├── pipeline.py         # BackFacePipeline orchestration
│   ├── vision_core.py      # HSV mask + blob candidates
│   ├── centroid_tracker.py
│   ├── spatio_temporal_matcher.py
│   ├── face_decoder.py
│   ├── temporal_decoder.py
│   ├── pair_selector.py    # legacy largest2
│   ├── geometry.py
│   ├── bit_extractor.py
│   ├── distance_model.py
│   ├── packet_builder.py
│   ├── dataset_io.py
│   ├── validation.py
│   ├── udp_transport.py
│   └── preview.py
│
├── tools/
│   └── hsv_tuner.py        # Interactive HSV tuning (tune-hsv)
│
├── unity/
│   └── RovLeds.cs
│
├── docs/
│   ├── progress_log.md
│   ├── calibration_log.md
│   ├── dataset_notes.md
│   ├── next_steps.md
│   └── phase4_tuning_log.md
│
├── datasets/               # Local PNG sequences (gitignored)
└── outputs/                # CSV, JSON, validation reports (gitignored)
```

The old numbered `scripts/` folder has been removed; use `main.py` subcommands instead.

---

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## CLI Usage (`main.py`)

Global options: `--datasets-dir`, `--outputs-dir`.

| Command | Purpose |
| -------- | -------- |
| `preview` | Play PNG sequence |
| `tune-hsv` | Interactive HSV sliders |
| `extract` | Per-frame CSV (`back_pair_results.csv`) |
| `decode` | Temporal pattern summary JSON |
| `filter` | IQR-filtered distance CSV |
| `calibrate` | Fit distance model from built-in points |
| `packet` | Observation JSON for one frame |
| `run` | Full pipeline (extract → packet) |
| `send-udp` | Send packet JSON over UDP |
| `recv-udp` | Listen for UDP packets |

Matcher flag (extract / run): `--matcher spatio_temporal` (default) or `legacy_largest2`.

### Quick start (single dataset)

```powershell
python main.py run --dataset BackOnly_Test_04
```

### Step-by-step (same as former scripts 01–08)

**Preview**

```powershell
python main.py preview --dataset BackOnly_Test_01
```

**Tune HSV**

```powershell
python main.py tune-hsv --dataset BackOnly_Test_02 --frame-index 80
```

**Extract** → `outputs/<DATASET>/back_pair_results.csv`

```powershell
python main.py extract --dataset BackOnly_Test_04
python main.py extract --dataset BackOnly_Test_04 --preview
```

CSV fields include: frame, `candidate_count`, `bit`, `pair_found`, LED coordinates, `pixel_distance`, midpoint, normalized error, camera ray, `face_id`, `pattern`, `pattern_accuracy`, track IDs, matcher scores, `matcher_mode`.

**Decode pattern**

```powershell
python main.py decode --dataset BackOnly_Test_04
```

**Filter distances** (reliable frames: `bit == 1`, `pair_found == 1`, valid `pixel_distance`)

```powershell
python main.py filter --dataset BackOnly_Test_04
```

**Calibrate distance model**

```powershell
python main.py calibrate
```

Fitted model:

```text
estimated_distance = 168.628584 / pixel_distance + 0.609526
```

**Observation packet**

```powershell
python main.py packet --dataset BackOnly_Test_04
python main.py packet --dataset BackOnly_Test_04 --frame 120
```

Example packet (frame 120, Test_04):

```json
{
    "dataset": "BackOnly_Test_04",
    "requested_frame": 120,
    "selected_frame_delta": 0,
    "frame": 120,
    "valid": true,
    "face_id": "BACK",
    "pattern": "11001100",
    "pattern_accuracy": 1.0,
    "bit_error_count": 0,
    "bit_error_rate": 0.0,
    "pair_found": true,
    "candidate_count": 2,
    "led1_px": [853.0, 555.0],
    "led2_px": [927.0, 556.0],
    "midpoint_px": [890.0, 555.5],
    "error_norm": [-0.0729, -0.0287],
    "ray_cam": [-0.0746, -0.0165, 0.9971],
    "pixel_distance": 74.0068,
    "estimated_distance": 2.8881,
    "distance_confidence": 1.0,
    "image_size": [1920, 1080]
}
```

### UDP (local test)

```powershell
# Terminal 1
python main.py recv-udp --port 5005

# Terminal 2
python main.py send-udp --dataset BackOnly_Test_04 --frame 120 --ip 127.0.0.1 --port 5005
```

---

## Validation Suite (`run_tests.py`)

Runs extract + metrics on offline datasets. Reports under `outputs/validation/`.

```powershell
python run_tests.py
python run_tests.py --dataset BackOnly_Test_04
python run_tests.py --force-reextract
python run_tests.py --matcher legacy_largest2 --dataset BackOnly_Test_04
```

Metrics (after 48-frame warmup): pair recall on ON frames, face ID accuracy, temporal decode accuracy, distance MAE vs calibration ground truth. Per-dataset thresholds can be overridden in `bluerov_led/config.py` (`DEFAULT_CALIBRATION_POINTS`).

Latest full-suite result (spatio-temporal matcher):

| Dataset | Pair recall | Face acc. | Temporal | Dist. MAE |
| -------- | ----------- | --------- | -------- | --------- |
| BackOnly_Test_01 | 0.66 | 1.00 | 1.00 | 0.20 |
| BackOnly_Test_02 | 0.81 | 1.00 | 1.00 | 0.01 |
| BackOnly_Test_03 | 0.86 | 1.00 | 1.00 | 0.15 |
| BackOnly_Test_04 | 0.84 | 1.00 | 1.00 | 0.12 |
| BackOnly_Test_05 | 0.84 | 1.00 | 1.00 | 0.30 |
| BackOnly_Test_06 | 0.68 | 1.00 | 1.00 | 0.02 |

**6/6 PASS** — see `docs/phase4_tuning_log.md` for far-range matcher notes (Test_06 improved from ~0.22 pair recall).

---

## Dataset Policy

PNG sequences are not committed (large files). Store archives externally; unpack locally under:

```text
datasets/
├── BackOnly_Test_01/
├── ...
└── BackOnly_Test_06/
```

Artifacts:

```text
outputs/
├── BackOnly_Test_04/
│   ├── back_pair_results.csv
│   ├── back_pair_distance_filtered.csv
│   ├── back_pattern_decode_summary.json
│   └── observation_packet_frame_120.json
├── calibration/
│   └── distance_model_summary.json
└── validation/
    ├── validation_summary.json
    └── validation_results.csv
```

`datasets/` and `outputs/` are gitignored (except `.gitkeep` where present).

---

## Calibration Reference

| Test | Approx. distance | Pattern accuracy | Median pixel dist. | Notes |
| ---- | ----------------: | ----------------: | -------------------: | ----- |
| BackOnly_Test_01 | 1.47 | 1.00 | ~168 px | Static reference |
| BackOnly_Test_02 | 2.00 | 1.00 | ~118 px | Stable |
| BackOnly_Test_03 | 2.50 | 1.00 | ~82 px | |
| BackOnly_Test_04 | 3.00 | 1.00 | ~73 px | Fixed-axis |
| BackOnly_Test_05 | 4.00 | 1.00 | ~53 px | Fixed-axis |
| BackOnly_Test_06 | 5.00 | 0.99 | ~35 px | Far-range; noisier spacing |

Expected trend: larger camera–target distance → smaller LED pixel spacing.

---

## Design Decisions

### Same-face LEDs share pattern and phase

Enables correlation-based pairing, stable midpoint, and distance from pixel spacing.

### Color is candidate-only

HSV thresholds live in `VisionConfig` (`bluerov_led/config.py`). Tune with `tune-hsv`.

### Spatio-temporal matching (default)

Tracks LED blobs across frames, fuses per-track ON/OFF signals, scores pairs by correlation + decoded face pattern + geometry. Far-range scenes (`max blob area < 90`) use relaxed decode and largest-two fallback; mid/near range stays strict to avoid false face labels.

### Multi-face ready

`FACE_PATTERNS` in config matches Unity. Back-only datasets still expect `BACK`; diagonal multi-face work is future validation.

### Distance model scope

Pixel-distance ranging works best for near-frontal face views. Use `distance_confidence` and `valid` for controller gating.

---

## Controller Observation Fields

| Field | Meaning |
| ----- | ------- |
| `valid` | Use this observation |
| `face_id` | Detected face |
| `pattern_accuracy` | Global repeated-pattern reliability |
| `pixel_distance` | LED spacing (px) |
| `estimated_distance` | Calibrated range estimate |
| `error_norm` | Normalized image-center error |
| `ray_cam` | Camera-frame ray to midpoint |

Prototype mapping: `error_norm[0]` → yaw, `error_norm[1]` → vertical/depth cue, `estimated_distance` → range, `valid` + confidence → gating.

---

## Next Steps

1. Live camera / video input (offline PNG path is validated).
2. Windows → Linux UDP on hardware network.
3. Compact binary packet if JSON latency is too high.
4. Multi-face diagonal datasets and per-face observation selection.
5. Stronger scoring when many false-positive blobs remain (`candidate_count > 3`).
