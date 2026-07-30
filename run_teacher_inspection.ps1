$ErrorActionPreference = "Stop"

$Python = "D:\anaconda\python.exe"
if (!(Test-Path -LiteralPath $Python)) { $Python = "python" }

& $Python scripts/inspect_teacher_all_csv.py --input data/all.csv --report-out reports/teacher_all_csv_inspection.md
