#!/usr/bin/env bash
set -euo pipefail

echo "Building podcast feed and latest episode..."
python -m podagent.runner --once
python -m podagent.feed.generate_feed

