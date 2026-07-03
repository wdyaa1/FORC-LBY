$ErrorActionPreference = "Stop"

python -m src.forc_baseline.train --config configs/baseline.yaml
python -m src.forc_baseline.evaluate --config configs/baseline.yaml
