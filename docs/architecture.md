# Architecture and experimental guarantees

## Scope

The current environment is `synthetic-lane-v1`, a deliberately small stochastic task. It is not a Dota 2 emulator and has no live-game integration. This allows us to verify that the control loop, learning updates, checkpoint format, datasets and evaluation tooling behave correctly without confusing simulator bugs with game-bridge bugs.

## Components

- `SyntheticLaneEnv`: discrete lane state and action/reward transition; seedable on every reset.
- `QLearningAgent`: tabular Q-learning with a versioned state encoder, epsilon-greedy exploration, a low-health action shield, finite-value validation and atomic checkpoint writes.
- `RandomPolicy`: stochastic lower-bound baseline.
- `HeuristicPolicy`: fixed human-authored reference baseline. It is not trained and can legitimately outperform learned Q-learning.
- `run_benchmark`: several independent training seeds, a held-out evaluation seed range, raw episode-level CSV, compressed transition dataset, paired differences and bootstrap confidence intervals.
- `LayaAdvisor`: optional high-level typed-choice advisory call. It is not the learning algorithm and does not execute actions.

## What is measured

Each training episode records reward, last hits, gold, deaths, episode length, survival, epsilon, Q-state count and update count. Each transition dataset row includes a serialized observation, action name, reward, next observation and terminal flag. Evaluation is run with exploration disabled and the Q table must remain unchanged.

The main promotion check is a paired bootstrap confidence interval for learned policy reward minus the random baseline over held-out initial seeds. The heuristic comparison is always reported, even if learning fails to beat it. Confidence intervals do not imply transfer to the actual game.

## Reproducibility

Training and evaluation use distinct seed ranges. The benchmark records train seeds, held-out evaluation seeds, hyperparameters, model checkpoints and raw results. Bootstrap seeds are fixed. Training can still vary across Python versions if a future code change modifies random call order; use the checked-in code version and seed manifest when comparing runs.

## Safety and deployment

There is no code in this repository that sends inputs to the live Dota 2 client. A real-game adapter must be implemented and separately validated in a local/private environment. It must start in shadow mode (propose/log actions only), validate observation schemas and action legality, then enable execution only by explicit opt-in.
