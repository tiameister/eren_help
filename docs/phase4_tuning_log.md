# Phase 4 Tuning Log

## Root cause (Test_06 low pair recall)

Analysis showed many ON frames had `candidate_count == 2` but `pair_found == 0`.
Extra stale tracks caused wrong track-pair combinations during combinatorial matching.

## Changes applied

1. **Prefer currently matched tracks** (`prefer_matched_tracks=True`) for pairing.
2. **Shorter buffer requirement** when exactly two candidates are visible.
3. **High-correlation face fallback** reuses last good decode when buffer is still filling.
4. **Detection tuning**: `min_area=15`, `on_area_threshold=28`, `min_pixel_distance_px=28`.
5. **Matcher relaxation**: `min_pair_correlation=0.88`, `min_pair_score=0.75`, geometry CV 0.10.

## Round 2 (Test_06 2-candidate gaps)

- Priority pairing: top-2 area candidates mapped to nearest tracks (evaluated first).
- Relaxed path: 24-frame decode window, 4 decoded bits, correlation fallback at 0.82.
- Far-range geometry: `min_pixel_distance_px` drops to 22 when LED area < 100.

## Round 3 (far-range gating + fallback)

- **Far-range scene** (`max candidate area < 90`): top-2-only combinatorics, relaxed decode, largest-two geometry fallback.
- Relaxed matching disabled on mid/near range to avoid FRONT mislabels (Test_03).
- Validation uses IQR-filtered median pixel distance for distance MAE.

## Final suite (`python run_tests.py --force-reextract`)

| Dataset | Pair recall | Dist MAE |
|---------|-------------|----------|
| Test_01 | 0.664 | 0.198 |
| Test_02 | 0.808 | 0.010 |
| Test_03 | 0.862 | 0.148 |
| Test_04 | 0.841 | 0.117 |
| Test_05 | 0.841 | 0.304 |
| Test_06 | **0.681** (was 0.224) | **0.023** |

**6/6 PASS**

## Validation command

```powershell
python run_tests.py --force-reextract
```

Reports: `outputs/validation/validation_summary.json`
