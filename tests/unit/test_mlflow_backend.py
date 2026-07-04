from esa_adb.experiments.mlflow_backend import _flatten_for_metrics, _flatten_for_params, log_to_mlflow


def test_flatten_for_params_stringifies_and_flattens_one_level():
    flat = _flatten_for_params({"a": 1, "b": {"c": 2.5, "d": "x"}}, prefix="config")
    assert flat == {"config_a": "1", "config_b_c": "2.5", "config_b_d": "x"}


def test_flatten_for_metrics_keeps_only_numeric_leaves():
    flat = _flatten_for_metrics({"window": {"f_beta": 0.5, "label": "ok"}, "n": 3, "flag": True})
    assert flat == {"window_f_beta": 0.5, "n": 3.0, "flag": 1.0}


def test_log_to_mlflow_is_a_no_op_without_mlflow_installed(monkeypatch, capsys):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("simulated: mlflow not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    log_to_mlflow("PCAReconstructor", mission=2, config={"n_components": 0.95}, metrics={"f_beta": 0.5})

    assert "mlflow not installed" in capsys.readouterr().out
