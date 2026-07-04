import json

from esa_adb.experiments.tracking import load_experiments, log_experiment, summarize_experiments


def test_log_experiment_appends_a_json_line(tmp_path):
    log_path = tmp_path / "results.jsonl"
    run_id = log_experiment(
        model_name="PCAReconstructor",
        mission=2,
        config={"n_components": 0.95},
        metrics={"precision": 0.24, "recall": 0.66, "f_beta": 0.28},
        log_path=log_path,
    )

    assert log_path.exists()
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["run_id"] == run_id
    assert record["model_name"] == "PCAReconstructor"
    assert record["config"]["n_components"] == 0.95
    assert record["metrics"]["f_beta"] == 0.28


def test_multiple_runs_append_without_overwriting(tmp_path):
    log_path = tmp_path / "results.jsonl"
    log_experiment("ModelA", 2, {}, {"f_beta": 0.1}, log_path=log_path)
    log_experiment("ModelB", 2, {}, {"f_beta": 0.2}, log_path=log_path)

    records = load_experiments(log_path)
    assert len(records) == 2
    assert {r["model_name"] for r in records} == {"ModelA", "ModelB"}


def test_load_experiments_missing_file_returns_empty_list(tmp_path):
    assert load_experiments(tmp_path / "does_not_exist.jsonl") == []


def test_summarize_experiments_flattens_nested_metrics(tmp_path):
    log_path = tmp_path / "results.jsonl"
    log_experiment(
        "WindowedIsolationForest",
        2,
        {"window_length": 10},
        {"window": {"precision": 0.3, "recall": 0.4, "f_beta": 0.32}, "event_wise": {"f_beta": 0.5}},
        log_path=log_path,
    )

    summary = summarize_experiments(log_path=log_path)

    assert summary.height == 1
    row = summary.row(0, named=True)
    assert row["model_name"] == "WindowedIsolationForest"
    assert row["window_precision"] == 0.3
    assert row["window_f_beta"] == 0.32
    assert row["event_wise_f_beta"] == 0.5


def test_summarize_experiments_empty(tmp_path):
    summary = summarize_experiments(log_path=tmp_path / "nothing.jsonl")
    assert summary.height == 0
