#!/usr/bin/env bash
# One-shot battlefield v3 build: pools -> selftest -> listening sheet -> test / val / train -> report.
# Usage: scripts/build_battlefield.sh [python]   (default .venv/Scripts/python.exe on Windows, .venv/bin/python elsewhere)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${1:-}
if [ -z "$PY" ]; then
  if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi
fi
[ -f data/pools.parquet ] || "$PY" -m ancdata.cli pools --workers 4
[ -f data/snippets/lombard6s/meta.parquet ] || "$PY" -m ancdata.cli snippets --seconds 6
"$PY" -m ancdata.cli battlefield-selftest --n 100
"$PY" -m ancdata.cli battlefield-listen --split test --n 10
"$PY" -m ancdata.cli battlefield --split test --no-report
"$PY" -m ancdata.cli battlefield --split val --no-report
"$PY" -m ancdata.cli battlefield --split train --no-report
"$PY" -m ancdata.cli battlefield-report
