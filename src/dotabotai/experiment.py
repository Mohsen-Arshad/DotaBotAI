"""Reproducible training/evaluation, episode-level datasets and paired bootstrap CI."""
from __future__ import annotations

import csv
import gzip
import io
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from .agent import QLearningAgent
from .env import ACTIONS, ENVIRONMENT_ID, SyntheticLaneEnv
from .policies import HeuristicPolicy, RandomPolicy

TRAIN_FIELDS = [
    "train_seed", "episode", "episode_seed", "reward", "last_hits", "gold", "deaths",
    "steps", "survived", "epsilon", "q_states", "updates",
]
EVAL_FIELDS = [
    "policy", "train_seed", "eval_seed", "reward", "last_hits", "gold", "deaths",
    "steps", "survived", "epsilon", "q_states",
]
TRANSITION_FIELDS = [
    "train_seed", "episode", "step", "episode_seed", "state_json", "action", "reward",
    "next_state_json", "terminated", "last_hit", "death", "creep_missed",
]

def run_episode(
    env: SyntheticLaneEnv,
    policy: Any,
    seed: int,
    train: bool = False,
    train_seed: int | None = None,
    episode_index: int | None = None,
    transition_writer: csv.DictWriter | None = None,
) -> dict[str, Any]:
    obs = env.reset(seed=seed)
    eval_tie_rng = (
        random.Random((int(getattr(policy, "seed", 0)) * 1_000_003) ^ int(seed))
        if isinstance(policy, QLearningAgent) and not train
        else None
    )
    total_reward = 0.0
    last_hits = 0
    steps = 0
    while True:
        action = (
            policy.choose_action(obs, explore=True)
            if isinstance(policy, QLearningAgent) and train
            else policy.choose_action(obs, explore=False, decision_seed=seed, tie_rng=eval_tie_rng)
            if isinstance(policy, QLearningAgent)
            else policy.choose_action(obs)
        )
        next_obs, reward, done, info = env.step(action)
        if train and isinstance(policy, QLearningAgent):
            policy.learn(obs, action, reward, next_obs, done)
        if transition_writer is not None:
            transition_writer.writerow({
                "train_seed": train_seed,
                "episode": episode_index,
                "step": steps,
                "episode_seed": seed,
                "state_json": json.dumps(obs, sort_keys=True, separators=(",", ":")),
                "action": ACTIONS[action],
                "reward": f"{reward:.6f}",
                "next_state_json": json.dumps(next_obs, sort_keys=True, separators=(",", ":")),
                "terminated": int(done),
                "last_hit": int(info["last_hit"]),
                "death": int(info["death"]),
                "creep_missed": int(info["creep_missed"]),
            })
        total_reward += reward
        last_hits += int(info["last_hit"])
        steps += 1
        obs = next_obs
        if done:
            break
    return {
        "reward": round(total_reward, 6),
        "last_hits": last_hits,
        "gold": obs["gold"],
        "deaths": obs["deaths"],
        "steps": steps,
        "survived": int(obs["hp"] > 0),
        "epsilon": getattr(policy, "epsilon", 0.0),
        "q_states": len(getattr(policy, "q", {})),
        "updates": getattr(policy, "total_updates", 0),
    }


def _bootstrap_ci(values: list[float], seed: int = 12345, samples: int = 2000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(samples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return [means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]]


def _policy_stats(rows: list[dict[str, Any]], clustered: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"n": len(rows)}
    metric_seed = {"reward": 101, "last_hits": 202, "gold": 303}
    for metric in ("reward", "last_hits", "gold", "deaths", "steps", "survived"):
        values = [float(row[metric]) for row in rows]
        mean = statistics.fmean(values) if values else 0.0
        result[f"mean_{metric}"] = round(mean, 6)
        result[f"std_{metric}"] = round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0
        if metric in metric_seed:
            if clustered:
                clusters: dict[int, list[float]] = defaultdict(list)
                for row in rows:
                    clusters[int(row["eval_seed"])].append(float(row[metric]))
                cluster_values = [statistics.fmean(group) for _, group in sorted(clusters.items())]
                ci = _bootstrap_ci(cluster_values, seed=metric_seed[metric])
            else:
                ci = _bootstrap_ci(values, seed=metric_seed[metric])
            result[f"ci95_mean_{metric}"] = [round(x, 6) for x in ci]
    return result


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(
    output_dir: str | Path,
    train_episodes: int = 10000,
    eval_episodes: int = 100,
    train_seeds: list[int] | None = None,
    eval_seed_base: int = 100_000,
    record_transitions: bool = False,
) -> dict[str, Any]:
    if train_episodes < 1 or eval_episodes < 1:
        raise ValueError("train_episodes and eval_episodes must both be >= 1")
    train_seeds = train_seeds or [11, 23, 37]
    if len(set(train_seeds)) != len(train_seeds):
        raise ValueError("train_seeds must be unique")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    models_dir = out / "models"
    models_dir.mkdir(exist_ok=True)
    started = time.perf_counter()
    training_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    transition_path = out / "transitions.csv.gz"
    transition_raw = transition_path.open("wb") if record_transitions else None
    gzip_file = gzip.GzipFile(filename="", mode="wb", fileobj=transition_raw, mtime=0) if transition_raw else None
    transition_text = io.TextIOWrapper(gzip_file, encoding="utf-8", newline="") if gzip_file else None
    transition_writer = csv.DictWriter(transition_text, fieldnames=TRANSITION_FIELDS) if transition_text else None
    if transition_writer:
        transition_writer.writeheader()

    agents: dict[int, QLearningAgent] = {}
    try:
        for train_seed in train_seeds:
            agent = QLearningAgent(seed=train_seed)
            env = SyntheticLaneEnv(seed=train_seed)
            # Persistable seed stream: a single-agent run can reconstruct this
            # position from the checkpoint's cumulative episode count.
            episode_rng = random.Random(train_seed * 1_000_003 + 71)
            report_every = max(1, train_episodes // 10)
            window_rewards: list[float] = []
            for episode in range(1, train_episodes + 1):
                episode_seed = episode_rng.randrange(1, 2**31 - 1)
                metrics = run_episode(
                    env,
                    agent,
                    seed=episode_seed,
                    train=True,
                    train_seed=train_seed,
                    episode_index=episode,
                    transition_writer=transition_writer,
                )
                agent.finish_episode()
                row = {
                    "train_seed": train_seed,
                    "episode": episode,
                    "episode_seed": episode_seed,
                    **metrics,
                    "epsilon": round(agent.epsilon, 8),
                    "q_states": len(agent.q),
                    "updates": agent.total_updates,
                }
                training_rows.append(row)
                window_rewards.append(float(metrics["reward"]))
                if episode % report_every == 0 or episode == train_episodes:
                    print(
                        f"train_seed={train_seed} episode={episode}/{train_episodes} "
                        f"rolling_reward={statistics.fmean(window_rewards):+.3f} "
                        f"epsilon={agent.epsilon:.3f} q_states={len(agent.q)}",
                        flush=True,
                    )
                    window_rewards.clear()
            agent.save(models_dir / f"qtable_seed_{train_seed}.json")
            agents[train_seed] = agent
    finally:
        if transition_text is not None:
            transition_text.flush()
            transition_text.close()
        if transition_raw is not None:
            transition_raw.close()

    eval_seeds = [eval_seed_base + index for index in range(eval_episodes)]
    for eval_seed in eval_seeds:
        env = SyntheticLaneEnv(seed=eval_seed)
        for train_seed, agent in agents.items():
            row = run_episode(env, agent, eval_seed)
            eval_rows.append({"policy": "learned", "train_seed": train_seed, "eval_seed": eval_seed, **row})
        row = run_episode(env, RandomPolicy(seed=eval_seed + 17), eval_seed)
        eval_rows.append({"policy": "random", "train_seed": "", "eval_seed": eval_seed, **row})
        row = run_episode(env, HeuristicPolicy(), eval_seed)
        eval_rows.append({"policy": "heuristic", "train_seed": "", "eval_seed": eval_seed, **row})

    _write_csv(out / "training_metrics.csv", TRAIN_FIELDS, training_rows)
    _write_csv(out / "evaluation_episodes.csv", EVAL_FIELDS, eval_rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        grouped[str(row["policy"])].append(row)
    stats = {name: _policy_stats(rows, clustered=(name == "learned")) for name, rows in grouped.items()}
    trained_by_eval_seed: dict[int, list[float]] = defaultdict(list)
    baselines_by_name: dict[str, dict[int, float]] = {"random": {}, "heuristic": {}}
    for row in eval_rows:
        if row["policy"] == "learned":
            trained_by_eval_seed[int(row["eval_seed"])].append(float(row["reward"]))
        elif row["policy"] in baselines_by_name:
            baselines_by_name[str(row["policy"])][int(row["eval_seed"])] = float(row["reward"])
    comparisons: dict[str, Any] = {}
    for baseline_name, baseline_values in baselines_by_name.items():
        paired_diffs = []
        for eval_seed in eval_seeds:
            model_values = trained_by_eval_seed[eval_seed]
            paired_diffs.append(statistics.fmean(model_values) - baseline_values[eval_seed])
        ci = _bootstrap_ci(paired_diffs, seed=8181 + len(baseline_name))
        comparisons[f"learned_vs_{baseline_name}"] = {
            "paired_eval_seed_count": len(paired_diffs),
            "mean_reward_difference": round(statistics.fmean(paired_diffs), 6),
            "bootstrap_ci95_reward_difference": [round(value, 6) for value in ci],
            "fraction_eval_seeds_with_positive_difference": round(
                sum(value > 0 for value in paired_diffs) / len(paired_diffs), 6
            ),
        }

    training_by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in training_rows:
        training_by_seed[int(row["train_seed"])].append(row)
    window_size = 100
    learning_curve: list[dict[str, Any]] = []
    for train_seed, rows in training_by_seed.items():
        for start in range(0, len(rows), window_size):
            chunk = rows[start:start + window_size]
            learning_curve.append({
                "train_seed": train_seed,
                "episode_start": int(chunk[0]["episode"]),
                "episode_end": int(chunk[-1]["episode"]),
                "mean_reward": round(statistics.fmean(float(row["reward"]) for row in chunk), 6),
                "mean_last_hits": round(statistics.fmean(float(row["last_hits"]) for row in chunk), 6),
            })
    _write_csv(out / "learning_curve.csv", ["train_seed", "episode_start", "episode_end", "mean_reward", "mean_last_hits"], learning_curve)

    random_ci = comparisons["learned_vs_random"]["bootstrap_ci95_reward_difference"]
    pass_random = random_ci[0] > 0
    summary: dict[str, Any] = {
        "schema_version": 1,
        "environment_id": ENVIRONMENT_ID,
        "warning": "Results describe only a synthetic lane-control environment, not live Dota 2.",
        "training": {
            "episodes_per_seed": train_episodes,
            "transition_dataset_recorded": bool(record_transitions),
            "train_seeds": train_seeds,
            "updates_total": sum(agent.total_updates for agent in agents.values()),
            "model_paths": [f"models/qtable_seed_{seed}.json" for seed in train_seeds],
        },
        "evaluation": {
            "episodes_per_policy": eval_episodes,
            "eval_seed_start": eval_seed_base,
            "eval_seed_end": eval_seeds[-1],
            "policies": stats,
            "paired_comparisons": comparisons,
        },
        "acceptance_checks": {
            "learned_beats_random_95pct_paired_bootstrap_ci": pass_random,
            "learned_beats_heuristic_95pct_paired_bootstrap_ci": comparisons["learned_vs_heuristic"]["bootstrap_ci95_reward_difference"][0] > 0,
            "rules": "Promotion criterion here is learned-vs-random CI lower bound > 0; the heuristic result is reported, not hidden.",
        },
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    (out / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(out / "REPORT.md", summary)
    _write_svg_learning_curve(out / "learning_curve.svg", learning_curve)
    print(f"\nArtifacts written to: {out.resolve()}")
    print(f"Learned mean reward: {stats['learned']['mean_reward']:+.3f}")
    print(f"Random mean reward:  {stats['random']['mean_reward']:+.3f}")
    print(f"Heuristic mean reward: {stats['heuristic']['mean_reward']:+.3f}")
    print(f"Learned vs random paired 95% CI: {comparisons['learned_vs_random']['bootstrap_ci95_reward_difference']}")
    print(f"Acceptance vs random: {'PASS' if pass_random else 'FAIL'}")
    return summary


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    evaluation = summary["evaluation"]
    stats = evaluation["policies"]
    comparisons = evaluation["paired_comparisons"]
    lines = [
        "# Synthetic-lane benchmark report",
        "",
        "> **Scope:** This benchmark validates the RL pipeline in a small synthetic task. It does not demonstrate playing Dota 2.",
        "",
        "## Experimental setup",
        "",
        f"- Environment: `{summary['environment_id']}`",
        f"- Training episodes per seed: {summary['training']['episodes_per_seed']}",
        f"- Training seeds: {summary['training']['train_seeds']}",
        f"- Evaluation episodes per policy: {evaluation['episodes_per_policy']}",
        f"- Evaluation seeds: {evaluation['eval_seed_start']}–{evaluation['eval_seed_end']}",
        "- Baselines: random policy and fixed heuristic policy",
        "- Evaluation does not update the Q table; seeds are held out from training.",
        "",
        "## Evaluation results",
        "",
        "| Policy | Episodes | Mean reward | 95% bootstrap CI (mean reward) | Mean last hits | Mean gold | Survival rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("learned", "random", "heuristic"):
        item = stats[name]
        ci = item.get("ci95_mean_reward", [0, 0])
        lines.append(
            f"| {name} | {item['n']} | {item['mean_reward']:+.3f} | "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | {item['mean_last_hits']:.2f} | "
            f"{item['mean_gold']:.1f} | {100 * item['mean_survived']:.1f}% |"
        )
    lines += ["", "## Paired comparisons on held-out initial seeds", "", "| Comparison | Mean reward difference | 95% paired bootstrap CI | Positive-seed fraction |", "|---|---:|---:|---:|"]
    for baseline in ("random", "heuristic"):
        item = comparisons[f"learned_vs_{baseline}"]
        ci = item["bootstrap_ci95_reward_difference"]
        lines.append(
            f"| learned − {baseline} | {item['mean_reward_difference']:+.3f} | "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | {100 * item['fraction_eval_seeds_with_positive_difference']:.1f}% |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- Learned-vs-random acceptance check: **{'PASS' if summary['acceptance_checks']['learned_beats_random_95pct_paired_bootstrap_ci'] else 'FAIL'}**.",
        f"- Learned-vs-heuristic check: **{'PASS' if summary['acceptance_checks']['learned_beats_heuristic_95pct_paired_bootstrap_ci'] else 'NOT PASSED'}**. This result is deliberately reported even when the heuristic wins.",
        "- Confidence intervals quantify uncertainty across held-out initial seeds and learned-policy seeds; they do not establish transfer to Dota 2.",
        "",
        "## Files",
        "",
        "- `training_metrics.csv`: per-episode reward, gold, last hits, survival, epsilon and Q-table size.",
        "- `learning_curve.csv` / `learning_curve.svg`: 100-episode windows for visual inspection.",
        "- `evaluation_episodes.csv`: raw held-out evaluation results per policy and seed.",
        *(["- `transitions.csv.gz`: recorded training transitions for independent analysis."] if summary["training"]["transition_dataset_recorded"] else ["- `transitions.csv.gz`: not recorded in this run (pass `--record-transitions` to enable it)."]),
        "- `benchmark_summary.json`: machine-readable metrics and acceptance checks.",
        "- `models/`: Q-table checkpoints, one per training seed.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_svg_learning_curve(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height, margin = 960, 520, 72
    if not rows:
        return
    values = [float(row["mean_reward"]) for row in rows]
    x_values = [int(row["episode_end"]) for row in rows]
    min_y, max_y = min(values), max(values)
    if math.isclose(min_y, max_y):
        max_y = min_y + 1.0
    def x_pos(x: int) -> float:
        xmax = max(x_values) or 1
        return margin + (width - 2 * margin) * x / xmax
    def y_pos(y: float) -> float:
        return height - margin - (height - 2 * margin) * (y - min_y) / (max_y - min_y)
    train_seeds = sorted({int(row["train_seed"]) for row in rows})
    # Browser-default palette; distinct series are additionally identified in the legend.
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="480" y="34" text-anchor="middle" font-size="22" font-family="sans-serif">Training reward by 100-episode window</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333"/>',
        f'<text x="24" y="260" transform="rotate(-90 24 260)" text-anchor="middle" font-size="14" font-family="sans-serif">Mean episode reward</text>',
        f'<text x="480" y="{height-18}" text-anchor="middle" font-size="14" font-family="sans-serif">Episode</text>',
    ]
    for idx, seed in enumerate(train_seeds):
        series = sorted([row for row in rows if int(row["train_seed"]) == seed], key=lambda r: int(r["episode_end"]))
        points = " ".join(f"{x_pos(int(row['episode_end'])):.2f},{y_pos(float(row['mean_reward'])):.2f}" for row in series)
        color = colors[idx % len(colors)]
        items.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        legend_y = 60 + idx * 22
        items.append(f'<line x1="{width-210}" y1="{legend_y}" x2="{width-180}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        items.append(f'<text x="{width-172}" y="{legend_y+5}" font-size="12" font-family="sans-serif">train seed {seed}</text>')
    for idx in range(5):
        tick_y = min_y + (max_y - min_y) * idx / 4
        py = y_pos(tick_y)
        items.append(f'<line x1="{margin-5}" y1="{py:.2f}" x2="{width-margin}" y2="{py:.2f}" stroke="#ddd"/>')
        items.append(f'<text x="{margin-10}" y="{py+4:.2f}" text-anchor="end" font-size="12" font-family="sans-serif">{tick_y:.1f}</text>')
    items.append('</svg>')
    path.write_text("\n".join(items), encoding="utf-8")
