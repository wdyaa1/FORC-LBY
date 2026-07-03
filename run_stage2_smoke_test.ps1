$ErrorActionPreference = "Stop"

python scripts/make_synthetic_smoke_data.py --root .
python scripts/preprocess_smoke_test.py --root .
