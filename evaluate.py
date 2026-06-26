from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ForcDataset, load_label_splits
from .model import build_model
from .utils import ensure_dir, load_config, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config["training"]["device"])
    run_dir = ensure_dir(config["outputs"]["run_dir"])

    _, _, test_df = load_label_splits(config)
    test_ds = ForcDataset(
        test_df,
        Path(config["data"]["simulated_dir"]),
        list(config["data"]["target_columns"]),
        config,
        augment=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
    )

    model = build_model(config).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model.eval()

    predictions = []
    targets = []
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x.to(device)).cpu().numpy()
            predictions.append(pred)
            targets.append(y.numpy())

    y_pred = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(targets, axis=0)
    mae = np.mean(np.abs(y_pred - y_true), axis=0)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))

    target_names = list(config["data"]["target_columns"])
    lines = ["# Baseline Test Metrics", ""]
    for name, target_mae, target_rmse in zip(target_names, mae, rmse):
        lines.append(f"- {name}: MAE={target_mae:.6f}, RMSE={target_rmse:.6f}")

    (run_dir / "test_metrics.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
