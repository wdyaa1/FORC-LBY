$ErrorActionPreference = "Stop"

$Python = "D:\anaconda\python.exe"
if (!(Test-Path -LiteralPath $Python)) { $Python = "python" }

& $Python scripts/make_synthetic_smoke_data.py --root .
& $Python scripts/preprocess_smoke_test.py --root .
