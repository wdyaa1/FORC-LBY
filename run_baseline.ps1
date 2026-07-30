$ErrorActionPreference = "Stop"

$Python = "D:\anaconda\python.exe"
if (!(Test-Path -LiteralPath $Python)) { $Python = "python" }

& $Python -m src.forc_baseline.train --config configs/baseline.yaml
& $Python -m src.forc_baseline.evaluate --config configs/baseline.yaml
