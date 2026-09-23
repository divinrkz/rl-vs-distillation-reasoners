#!/usr/bin/env bash
set -euo pipefail

# Persist HuggingFace downloads to Studio storage so a GPU restart doesn't
# re-download weights (keeps the GPU meter off during downloads).
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
echo "HF_HOME=$HF_HOME"

pip install -r requirements.txt

# Upstream reference repos, pinned to the commits used here.
mkdir -p externals
if [ ! -d externals/thought-anchors ]; then
  git clone https://github.com/divinrkz/thought-anchors.git externals/thought-anchors
  git -C externals/thought-anchors checkout b53ed8c75d3f6112f68adfaec9a13d4d708c442e
fi
if [ ! -d externals/steering-thinking-llms ]; then
  git clone https://github.com/divinrkz/steering-thinking-llms externals/steering-thinking-llms
  git -C externals/steering-thinking-llms checkout 93259bc3410c99293351df41141cd16b4110422a
fi

echo "Bootstrap done. Run: python scripts/01_smoke.py configs/smoke.yaml"
