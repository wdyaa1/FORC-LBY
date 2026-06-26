from __future__ import annotations

import numpy as np


def augment_grid(
    grid: np.ndarray,
    noise_std: float = 0.03,
    intensity_scale_min: float = 0.85,
    intensity_scale_max: float = 1.15,
) -> np.ndarray:
    augmented = grid.astype(np.float32).copy()
    scale = np.random.uniform(intensity_scale_min, intensity_scale_max)
    augmented *= scale

    if noise_std > 0:
        augmented += np.random.normal(0.0, noise_std, size=augmented.shape).astype(np.float32)

    return augmented
