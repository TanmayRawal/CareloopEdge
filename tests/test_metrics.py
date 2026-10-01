import numpy as np
import pytest

from neuroedge import metrics


def test_classification_perfect():
    probs = np.eye(4)[[0, 1, 2, 3, 0, 1]]
    m = metrics.classification_metrics(np.array([0, 1, 2, 3, 0, 1]), probs, ["a", "b", "c", "d"])
    assert m["accuracy"] == 1 and m["macro_f1"] == 1 and m["macro_auroc"] == 1


def test_ece_calibrated_vs_overconfident():
    rng = np.random.default_rng(0)
    n = 20000
    conf = rng.uniform(0.5, 1.0, n)
    correct = rng.random(n) < conf                       # calibrated by construction
    labels = np.where(correct, 0, 1)
    probs = np.stack([conf, 1 - conf], axis=1)
    assert metrics.calibration_error(probs, labels)["ece"] < 0.02
    over = np.stack([np.minimum(conf + 0.3, 0.999), 1 - np.minimum(conf + 0.3, 0.999)], axis=1)
    assert metrics.calibration_error(over, labels)["ece"] > 0.1


def test_mcnemar():
    a = np.array([1, 1, 0, 1], bool)
    assert metrics.mcnemar_exact(a, a)["p_value"] == 1.0
    b = np.zeros(40, bool)
    assert metrics.mcnemar_exact(np.ones(40, bool), b)["p_value"] < 1e-6


def test_bootstrap_contains_point_estimate():
    x = np.random.default_rng(1).normal(size=500)
    est, lo, hi = metrics.bootstrap_ci(lambda idx: x[idx].mean(), len(x), n_boot=500)
    assert lo <= est <= hi


def test_xai_localization_metrics():
    mask = np.zeros((64, 64), np.uint8)
    mask[10:20, 10:20] = 1
    cam = np.zeros((64, 64))
    cam[12:18, 12:18] = 1.0
    assert metrics.pointing_game(cam, mask)
    assert metrics.energy_pointing(cam, mask) == pytest.approx(1.0)
    far = np.zeros((64, 64))
    far[50, 50] = 1.0
    assert not metrics.pointing_game(far, mask, tolerance_px=5)


def test_map_similarity_identity():
    rng = np.random.default_rng(2)
    a = rng.random((14, 14))
    s = metrics.map_similarity(a, a.copy())
    assert s["pearson"] == pytest.approx(1.0) and s["ssim"] == pytest.approx(1.0, abs=1e-6)
