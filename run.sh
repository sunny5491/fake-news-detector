#!/usr/bin/env bash
# Reproduce everything from scratch (about 30 minutes on an M-series Mac).
set -euo pipefail
cd "$(dirname "$0")"
pip3 install -r requirements.txt
python3 prepare_data.py          # download + clean dataset
python3 train_baseline.py        # TF-IDF baseline + leakage experiment
python3 train_transformer.py     # fine-tune DistilBERT (MPS/CPU)
python3 eval_report.py           # charts, error analysis, stress test
python3 -m pytest tests -q       # smoke tests
python3 app/app.py               # launch the web app
