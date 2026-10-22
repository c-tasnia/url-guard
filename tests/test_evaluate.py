import numpy as np
import pytest

from urlguard.evaluate import best_threshold, report, to_markdown


def test_report_on_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    r = report("perfect", y, y, y.astype(float), verbose=False)
    assert r["F1"] == 1.0 and r["AUC"] == 1.0 and r["n"] == 4


def test_report_mask_restricts_rows():
    y = np.array([0, 1, 0, 1, 0, 1])
    pred = np.array([0, 1, 1, 0, 0, 1])          # wrong on rows 2 and 3
    mask = np.array([True, True, False, False, True, True])
    r = report("masked", y, pred, pred.astype(float), mask=mask, verbose=False)
    assert r["F1"] == 1.0 and r["n"] == 4


def test_best_threshold_finds_the_gap():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr, f1 = best_threshold(y, p)
    assert 0.3 < thr <= 0.7 and f1 == pytest.approx(1.0, abs=1e-6)


def test_markdown_table_is_sorted_by_f1():
    rows = [{"name": "b", "F1": 0.5, "precision": 0.5, "recall": 0.5, "AUC": 0.6, "n": 10},
            {"name": "a", "F1": 0.9, "precision": 0.9, "recall": 0.9, "AUC": 0.95, "n": 10}]
    lines = to_markdown(rows).splitlines()
    assert lines[2].startswith("| a ") and lines[3].startswith("| b ")
