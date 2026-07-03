from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .augment import augment_grid
from .preprocess import load_preprocessed_grid


class ForcDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        root_dir: str | Path,
        target_columns: list[str],
        config: dict,
        augment: bool = False,
    ) -> None:
        self.records = records.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.target_columns = target_columns
        self.config = config
        self.augment = augment
        self.file_column = config["data"]["file_column"]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.records.iloc[index]
        path = self.root_dir / str(row[self.file_column])
        grid = load_preprocessed_grid(path, self.config)

        if self.augment and self.config["augmentation"]["enabled"]:
            aug_cfg = self.config["augmentation"]
            grid = augment_grid(
                grid,
                noise_std=float(aug_cfg["noise_std"]),
                intensity_scale_min=float(aug_cfg["intensity_scale_min"]),
                intensity_scale_max=float(aug_cfg["intensity_scale_max"]),
            )

        x = torch.from_numpy(grid[None, :, :].astype(np.float32))
        y = torch.tensor(row[self.target_columns].to_numpy(dtype=np.float32))
        return x, y


def load_label_splits(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(config["data"]["labels_csv"])
    seed = int(config["project"]["seed"])
    test_size = float(config["data"]["test_size"])
    val_size = float(config["data"]["val_size"])

    train_val, test = train_test_split(labels, test_size=test_size, random_state=seed)
    adjusted_val_size = val_size / max(1.0 - test_size, 1e-6)
    train, val = train_test_split(train_val, test_size=adjusted_val_size, random_state=seed)
    return train, val, test
