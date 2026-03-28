#!/usr/bin/env bash
# Run the inference API with the same Python that has torch/ultralytics installed.
# Avoids accidentally using a global `uvicorn` from PATH.
set -euo pipefail
cd "$(dirname "$0")"
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python -m uvicorn inference_server:app --host 0.0.0.0 --port 8000 "$@"
else
  exec python3 -m uvicorn inference_server:app --host 0.0.0.0 --port 8000 "$@"
fi
