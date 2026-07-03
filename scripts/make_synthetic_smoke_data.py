from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def make_forc_like_points(seed: int, grain_size: float, n: int = 55) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hc = np.linspace(-1.0, 1.0, n)
    hu = np.linspace(-0.7, 0.7, n)
    mesh_hc, mesh_hu = np.meshgrid(hc, hu)

    width = 0.10 + 0.30 * grain_size
    interaction = 0.08 + 0.18 * (1.0 - grain_size)
    positive = np.exp(-((mesh_hc - 0.25) ** 2 / width + mesh_hu**2 / interaction))
    negative = 0.55 * np.exp(-((mesh_hc + 0.35) ** 2 / (width * 1.4) + (mesh_hu + 0.12) ** 2 / 0.10))
    ridge = 0.25 * np.exp(-(mesh_hu - 0.25 * mesh_hc) ** 2 / 0.025)
    noise = rng.normal(0.0, 0.015, size=mesh_hc.shape)
    rho = positive - negative + ridge + noise

    return np.column_stack([mesh_hc.ravel(), mesh_hu.ravel(), rho.ravel()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    simulated_dir = root / "data" / "raw" / "simulated_smoke"
    simulated_dir.mkdir(parents=True, exist_ok=True)

    labels_path = root / "data" / "raw" / "simulated_smoke_labels.csv"
    with labels_path.open("w", encoding="utf-8", newline="") as labels_file:
        label_writer = csv.writer(labels_file)
        label_writer.writerow(["filename", "grain_size"])

        for index in range(args.count):
            grain_size = 0.15 + index * (0.75 / max(args.count - 1, 1))
            filename = f"synthetic_forc_{index + 1:03d}.csv"
            path = simulated_dir / filename
            points = make_forc_like_points(seed=100 + index, grain_size=grain_size)

            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["hc", "hu", "rho"])
                writer.writerows(points)

            label_writer.writerow([f"simulated_smoke/{filename}", f"{grain_size:.4f}"])

    print(f"Wrote {args.count} synthetic files to {simulated_dir}")
    print(f"Wrote labels to {labels_path}")


if __name__ == "__main__":
    main()
