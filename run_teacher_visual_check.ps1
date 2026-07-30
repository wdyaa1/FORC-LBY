$ErrorActionPreference = "Stop"

$Python = "D:\anaconda\python.exe"
if (!(Test-Path -LiteralPath $Python)) { $Python = "python" }

& $Python scripts/visualize_teacher_dataset.py `
  --labels data/processed/teacher_forc_labels.csv `
  --diagram-dir data/processed/teacher_forc_diagrams `
  --figure-dir reports/figures/teacher_dataset `
  --report-out reports/teacher_dataset_visual_check.md `
  --max-samples 10
