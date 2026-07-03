from __future__ import annotations

import argparse
import csv
from pathlib import Path


SUPPORTED_SUFFIXES = {".csv", ".frc"}


def discover_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def sniff_columns(path: Path) -> list[str]:
    if path.suffix.lower() != ".csv":
        return []

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            dialect = csv.excel
            has_header = True

        reader = csv.reader(handle, dialect)
        first_row = next(reader, [])
        if has_header:
            return [item.strip() for item in first_row if item.strip()]
        return []


def write_label_template(simulated_files: list[Path], simulated_root: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "grain_size"])
        for path in simulated_files:
            writer.writerow([path.relative_to(simulated_root).as_posix(), ""])


def write_report(
    simulated_files: list[Path],
    experimental_files: list[Path],
    simulated_root: Path,
    experimental_root: Path,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Stage 1 Data Inspection",
        "",
        "## Summary",
        "",
        f"- Simulated files found: {len(simulated_files)}",
        f"- Experimental files found: {len(experimental_files)}",
        "",
        "## Simulated Files",
        "",
    ]

    if simulated_files:
        for path in simulated_files:
            rel = path.relative_to(simulated_root).as_posix()
            columns = sniff_columns(path)
            column_text = f" columns={columns}" if columns else ""
            lines.append(f"- `{rel}` ({path.suffix.lower()}){column_text}")
    else:
        lines.append("- No `.csv` or `.frc` files found yet.")

    lines.extend(["", "## Experimental Files", ""])

    if experimental_files:
        for path in experimental_files:
            rel = path.relative_to(experimental_root).as_posix()
            columns = sniff_columns(path)
            column_text = f" columns={columns}" if columns else ""
            lines.append(f"- `{rel}` ({path.suffix.lower()}){column_text}")
    else:
        lines.append("- No `.csv` or `.frc` files found yet.")

    lines.extend(
        [
            "",
            "## Next Checks",
            "",
            "- Confirm which columns represent the FORC axes and density values.",
            "- Fill `data/raw/simulated_labels_template.csv` and save a final copy as `data/raw/simulated_labels.csv`.",
            "- Plot a few simulated and experimental diagrams before training.",
            "- Decide whether the first target is regression or classification.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    simulated_root = root / "data" / "raw" / "simulated"
    experimental_root = root / "data" / "raw" / "experimental"
    simulated_files = discover_files(simulated_root)
    experimental_files = discover_files(experimental_root)

    write_label_template(
        simulated_files,
        simulated_root,
        root / "data" / "raw" / "simulated_labels_template.csv",
    )
    write_report(
        simulated_files,
        experimental_files,
        simulated_root,
        experimental_root,
        root / "reports" / "stage1_data_inspection.md",
    )

    print(f"Simulated files: {len(simulated_files)}")
    print(f"Experimental files: {len(experimental_files)}")
    print("Wrote data/raw/simulated_labels_template.csv")
    print("Wrote reports/stage1_data_inspection.md")


if __name__ == "__main__":
    main()
