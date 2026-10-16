import csv
import gzip
import json

from dotabotai.experiment import run_benchmark


def test_tiny_benchmark_writes_auditable_artifacts(tmp_path) -> None:
    summary = run_benchmark(
        output_dir=tmp_path,
        train_episodes=25,
        eval_episodes=4,
        train_seeds=[4, 5],
        eval_seed_base=9000,
        record_transitions=True,
    )
    required = [
        "training_metrics.csv",
        "evaluation_episodes.csv",
        "learning_curve.csv",
        "learning_curve.svg",
        "benchmark_summary.json",
        "REPORT.md",
        "transitions.csv.gz",
        "models/qtable_seed_4.json",
        "models/qtable_seed_5.json",
    ]
    for relative in required:
        assert (tmp_path / relative).exists(), relative

    with (tmp_path / "training_metrics.csv").open(encoding="utf-8-sig", newline="") as file:
        train_rows = list(csv.DictReader(file))
    assert len(train_rows) == 50
    assert all("epsilon" in row and "reward" in row for row in train_rows)

    with (tmp_path / "evaluation_episodes.csv").open(encoding="utf-8-sig", newline="") as file:
        eval_rows = list(csv.DictReader(file))
    assert len(eval_rows) == 4 * (2 + 2)
    assert {row["policy"] for row in eval_rows} == {"learned", "random", "heuristic"}

    with gzip.open(tmp_path / "transitions.csv.gz", "rt", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        assert "state_json" in reader.fieldnames and "next_state_json" in reader.fieldnames
        first_row = next(reader)
    assert first_row["action"] in {"wait", "last_hit", "retreat", "harass"}

    on_disk = json.loads((tmp_path / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert on_disk["environment_id"] == summary["environment_id"]
    assert "paired_comparisons" in on_disk["evaluation"]
