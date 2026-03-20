#!/usr/bin/env bash
set -euo pipefail

echo "Downloading transactions-fraud-datasets with KaggleHub..."
python - <<'PY'
from pathlib import Path
import kagglehub

dataset_id = "computingvictor/transactions-fraud-datasets"
path = Path(kagglehub.dataset_download(dataset_id)).resolve()
print(f"Dataset ready at: {path}")
PY
