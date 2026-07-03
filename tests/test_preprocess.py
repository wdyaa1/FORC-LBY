import numpy as np

from src.forc_baseline.preprocess import normalise_grid


def test_zscore_normalisation_has_reasonable_scale():
    grid = np.arange(16, dtype=np.float32).reshape(4, 4)
    out = normalise_grid(grid, method="zscore", clip_percentile=None)

    assert abs(float(out.mean())) < 1e-6
    assert 0.9 < float(out.std()) < 1.1
