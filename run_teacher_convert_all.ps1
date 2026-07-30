$ErrorActionPreference = "Stop"

$Python = "D:\anaconda\python.exe"
if (!(Test-Path -LiteralPath $Python)) { $Python = "python" }

& $Python scripts/convert_teacher_all_csv.py `
  --input data/all.csv `
  --output-dir data/processed/teacher_forc_diagrams `
  --labels-out data/processed/teacher_forc_labels.csv `
  --report-out reports/teacher_all_csv_conversion.md `
  --max-samples 0
