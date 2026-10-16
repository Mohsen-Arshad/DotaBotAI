# Synthetic-lane benchmark report

> **Scope:** This benchmark validates the RL pipeline in a small synthetic task. It does not demonstrate playing Dota 2.

## Experimental setup

- Environment: `synthetic-lane-v1`
- Training episodes per seed: 10000
- Training seeds: [11, 23, 37]
- Evaluation episodes per policy: 100
- Evaluation seeds: 100000–100099
- Baselines: random policy and fixed heuristic policy
- Evaluation does not update the Q table; seeds are held out from training.

## Evaluation results

| Policy | Episodes | Mean reward | 95% bootstrap CI (mean reward) | Mean last hits | Mean gold | Survival rate |
|---|---:|---:|---:|---:|---:|---:|
| learned | 300 | +19.641 | [+19.278, +20.022] | 11.90 | 416.4 | 100.0% |
| random | 100 | -4.452 | [-5.003, -3.854] | 4.98 | 174.3 | 100.0% |
| heuristic | 100 | +20.090 | [+19.521, +20.676] | 11.90 | 416.5 | 100.0% |

## Paired comparisons on held-out initial seeds

| Comparison | Mean reward difference | 95% paired bootstrap CI | Positive-seed fraction |
|---|---:|---:|---:|
| learned − random | +24.093 | [+23.413, +24.766] | 100.0% |
| learned − heuristic | -0.449 | [-0.862, -0.042] | 34.0% |

## Decision

- Learned-vs-random acceptance check: **PASS**.
- Learned-vs-heuristic check: **NOT PASSED**. This result is deliberately reported even when the heuristic wins.
- Confidence intervals quantify uncertainty across held-out initial seeds and learned-policy seeds; they do not establish transfer to Dota 2.

## Files

- `training_metrics.csv`: per-episode reward, gold, last hits, survival, epsilon and Q-table size.
- `learning_curve.csv` / `learning_curve.svg`: 100-episode windows for visual inspection.
- `evaluation_episodes.csv`: raw held-out evaluation results per policy and seed.
- `transitions.csv.gz`: not recorded in this run (pass `--record-transitions` to enable it).
- `benchmark_summary.json`: machine-readable metrics and acceptance checks.
- `models/`: Q-table checkpoints, one per training seed.
