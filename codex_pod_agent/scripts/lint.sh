#!/usr/bin/env bash
set -euo pipefail

if command -v ruff >/dev/null 2>&1; then ruff check src/python || true; fi
if command -v black >/dev/null 2>&1; then black --check src/python || true; fi
if command -v shfmt >/dev/null 2>&1; then shfmt -d scripts || true; fi

echo "Lint completed (best-effort)."

