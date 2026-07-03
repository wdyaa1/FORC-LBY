# Stage 2 Preprocessing Smoke Test

- Synthetic labelled files: 6
- Grid size: 128 x 128
- Normalisation: zscore

## Processed Files

- `simulated_smoke/synthetic_forc_001.csv`: shape=(128, 128), min=-1.615, max=3.227, figure=`reports/figures/stage2_smoke/synthetic_forc_001_preprocessed.png`
- `simulated_smoke/synthetic_forc_002.csv`: shape=(128, 128), min=-1.617, max=3.144, figure=`reports/figures/stage2_smoke/synthetic_forc_002_preprocessed.png`
- `simulated_smoke/synthetic_forc_003.csv`: shape=(128, 128), min=-1.523, max=3.125, figure=`reports/figures/stage2_smoke/synthetic_forc_003_preprocessed.png`
- `simulated_smoke/synthetic_forc_004.csv`: shape=(128, 128), min=-1.469, max=3.122, figure=`reports/figures/stage2_smoke/synthetic_forc_004_preprocessed.png`
- `simulated_smoke/synthetic_forc_005.csv`: shape=(128, 128), min=-1.450, max=3.171, figure=`reports/figures/stage2_smoke/synthetic_forc_005_preprocessed.png`
- `simulated_smoke/synthetic_forc_006.csv`: shape=(128, 128), min=-1.403, max=3.236, figure=`reports/figures/stage2_smoke/synthetic_forc_006_preprocessed.png`

## Result

The preprocessing path can read CSV FORC-like data, interpolate it to the configured grid, normalise it, and save inspection figures.