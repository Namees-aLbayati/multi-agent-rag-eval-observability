#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -x .venv/bin/python ]]; then
  python_bin=.venv/bin/python
elif [[ -x ../.venv/bin/python ]]; then
  python_bin=../.venv/bin/python
else
  echo 'Create an environment first: python3 -m venv .venv'
  echo 'Then install: .venv/bin/python -m pip install -r requirements.txt'
  exit 1
fi
exec "$python_bin" -m uvicorn backend.main:app --host 127.0.0.1 --port "${PORT:-8000}"
