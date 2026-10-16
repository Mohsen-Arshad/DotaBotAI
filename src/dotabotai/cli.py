"""Command-line interface for training, evaluation and reproducible benchmarks."""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any

from .agent import QLearningAgent
from .env import SyntheticLaneEnv
from .experiment import EVAL_FIELDS, TRAIN_FIELDS, run_benchmark, run_episode
from .laya_advisor import LayaAdvisor


def _train(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "qtable.json"
    agent = QLearningAgent(seed=args.seed)
    if args.resume:
        if not model_path.exists():
            raise SystemExit(f"Cannot resume: checkpoint does not exist: {model_path}")
        agent.load(model_path)
        print(f"Resuming {model_path}; previously trained episodes={agent.episodes_trained}")
    if agent.seed != args.seed:
        raise SystemExit(f"Checkpoint seed is {agent.seed}, but --seed is {args.seed}. Resume with the original seed.")
    env = SyntheticLaneEnv(seed=args.seed)
    seed_rng = random.Random(args.seed * 1_000_003 + 71)
    for _ in range(agent.episodes_trained):
        seed_rng.randrange(1, 2**31 - 1)
    rows: list[dict[str, Any]] = []
    report_every = max(1, args.episodes // 10)
    window: list[float] = []
    for idx in range(1, args.episodes + 1):
        global_episode = agent.episodes_trained + 1
        episode_seed = seed_rng.randrange(1, 2**31 - 1)
        metrics = run_episode(env, agent, episode_seed, train=True)
        agent.finish_episode()
        row = {
            "train_seed": args.seed,
            "episode": agent.episodes_trained,
            "episode_seed": episode_seed,
            **metrics,
            "epsilon": round(agent.epsilon, 8),
            "q_states": len(agent.q),
            "updates": agent.total_updates,
        }
        rows.append(row)
        window.append(float(metrics["reward"]))
        if idx % report_every == 0 or idx == args.episodes:
            print(
                f"episode={idx}/{args.episodes} rolling_reward={statistics.fmean(window):+.3f} "
                f"epsilon={agent.epsilon:.3f} q_states={len(agent.q)}",
                flush=True,
            )
            window.clear()
    agent.save(model_path)
    with (out / "training_metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TRAIN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved checkpoint: {model_path}")
    print(f"Saved episode dataset: {out / 'training_metrics.csv'}")
    print("Scope: training took place in the synthetic environment, not in Dota 2.")


def _evaluate(args: argparse.Namespace) -> None:
    checkpoint = Path(args.model)
    if not checkpoint.exists():
        raise SystemExit(f"Model not found: {checkpoint}. Train a model first.")
    agent = QLearningAgent(seed=args.seed, epsilon=0.0)
    agent.load(checkpoint)
    # Evaluation must never update the Q table or use epsilon exploration.
    q_before = json.dumps(agent.q, sort_keys=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index in range(args.episodes):
        eval_seed = args.seed + index
        metrics = run_episode(SyntheticLaneEnv(seed=eval_seed), agent, eval_seed, train=False)
        rows.append({"policy": "learned", "train_seed": agent.seed, "eval_seed": eval_seed, **metrics})
    if json.dumps(agent.q, sort_keys=True) != q_before:
        raise RuntimeError("Evaluation changed the Q table; this is a bug.")
    with (out / "evaluation_episodes.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=EVAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"episodes={len(rows)} mean_reward={statistics.fmean(r['reward'] for r in rows):+.3f} "
          f"mean_last_hits={statistics.fmean(r['last_hits'] for r in rows):.2f} "
          f"survival_rate={statistics.fmean(r['survived'] for r in rows):.1%}")
    print(f"Saved evaluation dataset: {out / 'evaluation_episodes.csv'}")



def _benchmark(args: argparse.Namespace) -> None:
    summary = run_benchmark(
        output_dir=args.output_dir,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        train_seeds=args.train_seeds,
        eval_seed_base=args.eval_seed_base,
        record_transitions=args.record_transitions,
    )
    if args.require_learning and not summary["acceptance_checks"]["learned_beats_random_95pct_paired_bootstrap_ci"]:
        raise SystemExit("Learning acceptance check failed: the paired 95% CI does not clear the random baseline.")

def _laya_check(args: argparse.Namespace) -> None:
    obs = {
        "hp": args.hp,
        "enemy_distance": args.enemy_distance,
        "creep_hp": args.creep_hp,
        "enemy_hp": args.enemy_hp,
        "gold": args.gold,
        "last_hits": args.last_hits,
    }
    result = LayaAdvisor(device=args.device).advise(obs)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Note: this is a macro-advice test on synthetic state, not a live Dota action.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dotabotai",
        description="Reproducible RL experiments for a synthetic Dota-style lane task.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train one Q-learning agent and save episode-level data")
    p_train.add_argument("--episodes", type=int, default=10_000)
    p_train.add_argument("--seed", type=int, default=11)
    p_train.add_argument("--output-dir", default="runs/train")
    p_train.add_argument("--resume", action="store_true")
    p_train.set_defaults(func=_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate a saved checkpoint without learning updates")
    p_eval.add_argument("--model", default="runs/train/qtable.json")
    p_eval.add_argument("--episodes", type=int, default=100)
    p_eval.add_argument("--seed", type=int, default=100_000)
    p_eval.add_argument("--output-dir", default="runs/evaluation")
    p_eval.set_defaults(func=_evaluate)

    p_bench = sub.add_parser("benchmark", help="Train on multiple seeds and compare against baselines")
    p_bench.add_argument("--train-episodes", type=int, default=10_000)
    p_bench.add_argument("--eval-episodes", type=int, default=100)
    p_bench.add_argument("--train-seeds", type=int, nargs="+", default=[11, 23, 37])
    p_bench.add_argument("--eval-seed-base", type=int, default=100_000)
    p_bench.add_argument("--output-dir", default="runs/benchmark")
    p_bench.add_argument("--record-transitions", action="store_true", help="Save every training transition to a compressed CSV; adds disk I/O")
    p_bench.add_argument("--require-learning", action="store_true", help="Return an error unless learned policy clears the random baseline confidence interval")
    p_bench.set_defaults(func=_benchmark)

    p_laya = sub.add_parser("laya-check", help="Ask optional Laya for a high-level synthetic-state plan")
    p_laya.add_argument("--device", choices=["cpu", "cuda", "mps", "xpu"], default="cpu")
    p_laya.add_argument("--hp", type=int, default=35)
    p_laya.add_argument("--enemy-distance", type=int, default=320)
    p_laya.add_argument("--creep-hp", type=int, default=22)
    p_laya.add_argument("--enemy-hp", type=int, default=80)
    p_laya.add_argument("--gold", type=int, default=0)
    p_laya.add_argument("--last-hits", type=int, default=0)
    p_laya.set_defaults(func=_laya_check)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    for name in ("episodes", "train_episodes", "eval_episodes"):
        value = getattr(args, name, 1)
        if value < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    args.func(args)


if __name__ == "__main__":
    main()
