# DotaBotAI

**A reproducible reinforcement-learning research scaffold for Dota-style lane control.** The project separates a small synthetic environment, a Q-learning agent, honest baselines, held-out evaluation, and optional Laya strategic advice.

> **Important scope boundary:** this repository does not yet control the live Dota 2 client. The current benchmark uses `synthetic-lane-v1`, a deliberately simplified task used to validate the RL loop and its measurement infrastructure. Do not interpret synthetic results as Dota 2 skill.

## Features

- Tabular Q-learning with seeded exploration and versioned, validated checkpoints.
- Random and fixed-heuristic baselines; the heuristic is allowed to beat the learned agent.
- Multiple independent training seeds and a separate held-out evaluation seed range.
- Episode-level CSV datasets, optional compressed transition logs, benchmark JSON, Markdown report and SVG learning curve.
- Paired reward differences with bootstrap confidence intervals.
- Tests and GitHub Actions CI on Python 3.10–3.13.
- Optional Laya integration for high-level advice; Laya is not the RL algorithm.
- A documented, gated roadmap for a future DotaService adapter.

## Quick start (Windows / PowerShell)

Python 3.10 or newer is required. Core training uses only the Python standard library.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m dotabotai --help
```

Train a single agent and save per-episode data:

```powershell
python -m dotabotai train --episodes 10000 --seed 11 --output-dir runs/train-11
```

Evaluate a saved checkpoint without changing the model:

```powershell
python -m dotabotai evaluate --model runs/train-11/qtable.json --episodes 100 --seed 100000 --output-dir runs/eval-11
```

Run the full reproducible benchmark (three independently trained policies):

```powershell
python -m dotabotai benchmark --train-episodes 10000 --eval-episodes 100 --train-seeds 11 23 37 --eval-seed-base 100000 --output-dir artifacts/my-run --require-learning
```

To additionally save every transition (state, action, reward, next state), opt in with `--record-transitions`. It adds disk I/O, so it is disabled by default; episode-level and evaluation datasets are always written.

```powershell
python -m dotabotai benchmark --train-episodes 10000 --eval-episodes 100 --train-seeds 11 23 37 --output-dir artifacts/my-run-full --record-transitions
```

For a faster wiring check, use smaller settings first:

```powershell
python -m dotabotai benchmark --train-episodes 200 --eval-episodes 10 --train-seeds 11 23 --output-dir runs/smoke
```

## What the benchmark writes

| Artifact | Purpose |
|---|---|
| `training_metrics.csv` | One row per training episode, including reward, last hits, gold, deaths, epsilon and Q-table size. |
| `evaluation_episodes.csv` | Raw evaluation outcomes for learned policies, random baseline and fixed heuristic. |
| `transitions.csv.gz` (optional) | Compressed per-step training transitions: state, action, reward, next state, terminal flag and event fields. |
| `learning_curve.csv` / `learning_curve.svg` | Rolling reward and last-hit trend in 100-episode windows. |
| `benchmark_summary.json` | Mean/std metrics, bootstrap intervals, paired comparisons and explicit acceptance checks. |
| `REPORT.md` | Human-readable report generated from the same results. |
| `models/` | Separate checkpoint for each training seed. |

Evaluation uses separate initial seeds from training, disables epsilon exploration and does not apply learning updates. The benchmark reports both learned-vs-random and learned-vs-heuristic results. The promotion check requires the lower bound of the paired 95% bootstrap CI for learned minus random reward to be above zero. A failure is reported, not hidden.

## Optional Laya advisor

Install the optional dependency only after the core project works:

```powershell
python -m pip install -e ".[laya]"
python -m dotabotai laya-check --device cpu
```

The first actual prediction may download a model checkpoint and therefore needs network access. Laya is used only for a high-level recommendation in a synthetic state. It does not automatically train the Q-learning model or execute Dota commands.

## Tests

```powershell
python -m pytest
```

Core test coverage includes deterministic seeded resets, action validation, checkpoint round-trips, evaluation without Q-table mutation and benchmark artifact generation. The benchmark itself is the empirical learning check; a green unit-test run alone is not proof that a policy is strong.

## Architecture

```text
SyntheticLaneEnv -> QLearningAgent -> saved checkpoint
        |                  |
        +-> transition CSV  +-> held-out evaluation
                                  |
                          random / heuristic baselines
                                  |
                    paired deltas + bootstrap CI
```

See [`docs/architecture.md`](docs/architecture.md) for experiment guarantees and [`docs/real_dota_integration.md`](docs/real_dota_integration.md) for the integration gates.

## Roadmap

1. Validate the learning and evaluation pipeline on the synthetic task.
2. Reproduce the historical DotaService setup in an isolated Linux/WSL environment.
3. Build an observation/action adapter and verify it in shadow mode.
4. Enable controlled actions only in a private/custom lobby.
5. Rebuild the state encoder and reward model for the actual game, then evaluate learned policies against fixed baselines.
6. Keep Laya only if measured decision quality offsets its inference latency.

The existing `TimZaman/dotaclient` and `TimZaman/dotaservice` projects are historical dependencies, not vendored code here. Their versions and setup need validation against the user's installed Dota 2 build before an integration claim can be made.

## License and attribution

Original code in this repository is released under the MIT License. The referenced projects retain their own licenses and terms; no code from them is copied into this repository. Dota 2 is a Valve product; this project is an independent research prototype and is not affiliated with Valve.
