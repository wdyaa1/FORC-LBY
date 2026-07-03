from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def read_forc_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.replace(",", " ").split()
            try:
                values = [float(part) for part in parts]
            except ValueError:
                continue
            if len(values) >= 3:
                rows.append(values[:3])

    if not rows:
        raise ValueError(f"No numeric FORC rows found in {path}")

    return pd.DataFrame(rows, columns=["x", "y", "rho"])


def table_to_grid(
    table: pd.DataFrame,
    height: int,
    width: int,
    fill_value: float = 0.0,
) -> np.ndarray:
    columns = {name.lower(): name for name in table.columns}
    x_col = columns.get("x") or columns.get("hc") or columns.get("field")
    y_col = columns.get("y") or columns.get("hu") or columns.get("bias")
    rho_col = columns.get("rho") or columns.get("forc") or columns.get("density")

    if not all([x_col, y_col, rho_col]):
        raise ValueError(
            "FORC table needs x/y/rho-like columns. "
            f"Available columns: {list(table.columns)}"
        )

    x = table[x_col].to_numpy(dtype=float)
    y = table[y_col].to_numpy(dtype=float)
    rho = table[rho_col].to_numpy(dtype=float)

    grid_x = np.linspace(np.nanmin(x), np.nanmax(x), width)
    grid_y = np.linspace(np.nanmin(y), np.nanmax(y), height)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    grid = griddata((x, y), rho, (mesh_x, mesh_y), method="linear")
    grid = np.where(np.isfinite(grid), grid, fill_value)
    return grid.astype(np.float32)


def normalise_grid(
    grid: np.ndarray,
    method: str = "zscore",
    clip_percentile: float | None = 99.5,
) -> np.ndarray:
    values = grid.astype(np.float32)

    if clip_percentile is not None:
        limit = np.nanpercentile(np.abs(values), clip_percentile)
        if limit > 0:
            values = np.clip(values, -limit, limit)

    if method == "zscore":
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))
        return (values - mean) / max(std, 1e-6)

    if method == "minmax":
        low = float(np.nanmin(values))
        high = float(np.nanmax(values))
        return (values - low) / max(high - low, 1e-6)

    if method == "none":
        return values

    raise ValueError(f"Unknown normalisation method: {method}")


def load_preprocessed_grid(path: str | Path, config: dict) -> np.ndarray:
    table = read_forc_table(path)
    grid_cfg = config["data"]["grid_size"]
    grid = table_to_grid(
        table,
        height=int(grid_cfg["height"]),
        width=int(grid_cfg["width"]),
        fill_value=float(config["preprocess"]["fill_value"]),
    )
    return normalise_grid(
        grid,
        method=config["preprocess"]["normalisation"],
        clip_percentile=float(config["preprocess"]["clip_percentile"]),
    )
