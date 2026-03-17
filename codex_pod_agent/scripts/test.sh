#!/usr/bin/env bash
set -euo pipefail

if command -v pytest >/dev/null 2>&1; then
  pytest -q
else
  echo "No tests configured yet."
fi

