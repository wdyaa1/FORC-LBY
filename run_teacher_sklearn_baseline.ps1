$ErrorActionPreference = "Stop"

$Python = "D:\anaconda\python.exe"
if (!(Test-Path -LiteralPath $Python)) { $Python = "python" }

& $Python scripts/train_teacher_sklearn_baseline.py `
  --labels data/processed/teacher_forc_labels.csv `
  --diagram-dir data/processed/teacher_forc_diagrams `
  --output-dir reports/runs/teacher_sklearn_baseline `
  --feature-size 32
