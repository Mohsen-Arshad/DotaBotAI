$ErrorActionPreference = "Stop"
python -m dotabotai benchmark `
  --train-episodes 10000 `
  --eval-episodes 100 `
  --train-seeds 11 23 37 `
  --eval-seed-base 100000 `
  --output-dir artifacts/reference_run `
  --require-learning
