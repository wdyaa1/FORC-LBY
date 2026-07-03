from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))

    from src.forc_baseline.preprocess import load_preprocessed_grid

    with (root / args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    labels_path = root / "data" / "raw" / "simulated_smoke_labels.csv"
    labels = pd.read_csv(labels_path)

    smoke_root = root / "data" / "raw"
    figure_dir = root / "reports" / "figures" / "stage2_smoke"
    figure_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Stage 2 Preprocessing Smoke Test",
        "",
        f"- Synthetic labelled files: {len(labels)}",
        f"- Grid size: {config['data']['grid_size']['height']} x {config['data']['grid_size']['width']}",
        f"- Normalisation: {config['preprocess']['normalisation']}",
        "",
        "## Processed Files",
        "",
    ]

    for _, row in labels.iterrows():
        rel_path = row["filename"]
        grid = load_preprocessed_grid(smoke_root / rel_path, config)
        figure_path = figure_dir / (Path(rel_path).stem + "_preprocessed.png")

        plt.figure(figsize=(4.5, 3.8))
        plt.imshow(grid, origin="lower", cmap="RdBu_r", aspect="auto")
        plt.colorbar(label="normalised FORC density")
        plt.title(Path(rel_path).name)
        plt.tight_layout()
        plt.savefig(figure_path, dpi=160)
        plt.close()

        lines.append(
            f"- `{rel_path}`: shape={grid.shape}, "
            f"min={grid.min():.3f}, max={grid.max():.3f}, "
            f"figure=`{figure_path.relative_to(root).as_posix()}`"
        )

    lines.extend(
        [
            "",
            "## Result",
            "",
            "The preprocessing path can read CSV FORC-like data, interpolate it to the configured grid, normalise it, and save inspection figures.",
        ]
    )

    report_path = root / "reports" / "stage2_preprocessing_smoke_test.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {report_path}")
    print(f"Wrote figures to {figure_dir}")


if __name__ == "__main__":
    main()
