import json

import joblib

import train


def test_quick_training_run_writes_a_loadable_model(tiny_csv, tiny_tranco, tmp_path):
    out = tmp_path / "out"
    train.main(["--data", str(tiny_csv), "--tranco-file", str(tiny_tranco),
                "--n-popular", "100", "--n-tail", "100", "--n-typo", "100",
                "--tfidf-features", "5000", "--quick", "--out-dir", str(out)])
    bundle = joblib.load(out / "url_model.joblib")
    assert set(bundle) == {"tfidf", "svm", "gbm", "rank", "thresh"}
    assert len(bundle["rank"]) == 100

    metrics = json.loads((out / "metrics.json").read_text())
    names = [r["name"] for r in metrics["all_test_rows"]]
    assert names == ["Stacked SVM + LightGBM"]
    assert metrics["original_rows_only"]
    assert metrics["config"]["malicious_label"] == "0"
