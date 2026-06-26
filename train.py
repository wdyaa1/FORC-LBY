from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import ForcDataset, load_label_splits
from .model import build_model
from .utils import ensure_dir, load_config, resolve_device, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0

    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(is_train):
            pred = model(x)
            loss = criterion(pred, y)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_size = x.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]))
    device = resolve_device(config["training"]["device"])
    run_dir = ensure_dir(config["outputs"]["run_dir"])

    train_df, val_df, _ = load_label_splits(config)
    root = Path(config["data"]["simulated_dir"])
    targets = list(config["data"]["target_columns"])

    train_ds = ForcDataset(train_df, root, targets, config, augment=True)
    val_ds = ForcDataset(val_df, root, targets, config, augment=False)

    loader_cfg = config["training"]
    train_loader = DataLoader(
        train_ds,
        batch_size=int(loader_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(loader_cfg["num_workers"]),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(loader_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(loader_cfg["num_workers"]),
    )

    model = build_model(config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(loader_cfg["learning_rate"]),
        weight_decay=float(loader_cfg["weight_decay"]),
    )

    best_val = float("inf")
    history: list[str] = []
    for epoch in range(1, int(loader_cfg["epochs"]) + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        line = f"epoch={epoch}, train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
        print(line)
        history.append(line)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    (run_dir / "training_log.txt").write_text("\n".join(history), encoding="utf-8")


if __name__ == "__main__":
    main()
