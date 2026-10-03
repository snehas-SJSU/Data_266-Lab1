#!/usr/bin/env bash
# Smoke test: Sneha's Part 1 GPT (smoke=true: TinyStories-valid, 256 train / 64 val, 2 epochs).
# Runs in a temporary copy of the repo, so committed results, logs, and manifests stay untouched.
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mktemp -d)"

rsync -a \
  --exclude .git --exclude .venv \
  --exclude 'task1_llm/data/*.txt' \
  --exclude task2_sentiment --exclude task3_gan \
  "$REPO/" "$WORK/"

python - "$WORK/task1_llm/sneha_singh/src/config.json" <<'EOF'
import json, sys
path = sys.argv[1]
cfg = json.load(open(path))
cfg["smoke"] = True
json.dump(cfg, open(path, "w"), indent=2)
EOF

jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=1800 \
  "$WORK/task1_llm/sneha_singh/src/part1_llm.ipynb" \
  --output smoke_run.ipynb

echo "Smoke test passed. Outputs: $WORK/task1_llm/sneha_singh"
