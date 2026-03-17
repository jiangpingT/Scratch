#!/usr/bin/env bash
set -euo pipefail

if command -v npm >/dev/null 2>&1 && [ -f package.json ]; then
  if npm run | grep -q "dev"; then npm run dev; else npm start; fi
else
  python -m podagent.runner --once
fi

