import numpy as np
import pytest

from neuroedge.uncertainty import (ClassConditionalConformal, MahalanobisOOD, TemperatureScaler,
                                   energy_score, softmax)


def _sample_labels(rng, probs):
    cum = probs.cumsum(axis=1)
    return (rng.random((len(probs), 1)) > cum).sum(axis=1)


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    z = rng.normal(scale=1.5, size=(20000, 4))
    labels = _sample_labels(rng, softmax(z))       # labels drawn from the TRUE calibrated probs
    scaler = TemperatureScaler().fit(3.0 * z, labels)  # model is 3x overconfident
    assert scaler.temperature == pytest.approx(3.0, rel=0.08)


def test_class_conditional_conformal_coverage():
    rng = np.random.default_rng(1)
    alpha = 0.05
    probs = rng.dirichlet(np.full(4, 0.7), size=40000)
    labels = _sample_labels(rng, probs)
    cal, test = slice(0, 8000), slice(8000, None)
    cp = ClassConditionalConformal(alpha).fit(probs[cal], labels[cal])
    report = cp.coverage_report(cp.predict_sets(probs[test]), labels[test])
    for cov in report["per_class_coverage"].values():
        assert cov >= 1 - alpha - 0.01
    restored = ClassConditionalConformal.from_dict(cp.to_dict())
    np.testing.assert_array_equal(restored.predict_sets(probs[:50]), cp.predict_sets(probs[:50]))


def test_conformal_small_class_includes_always():
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]])
    cp = ClassConditionalConformal(0.05).fit(probs, np.array([0, 0, 1]))
    assert np.isinf(cp.thresholds).all()  # too few samples for a 95% guarantee
    assert cp.predict_sets(probs).all()


def _gauss_layers(rng, n, shift=0.0):
    labels = rng.integers(0, 3, n)
    l1 = rng.normal(size=(n, 8)) + labels[:, None] * 2.0 + shift
    l2 = rng.normal(size=(n, 16)) + labels[:, None] * 1.0 + shift
    return [l1, l2], labels


def test_mahalanobis_detects_shift(tmp_path):
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(2)
    train, y = _gauss_layers(rng, 3000)
    calib, _ = _gauss_layers(rng, 2000)
    test_id, _ = _gauss_layers(rng, 2000)
    test_ood, _ = _gauss_layers(rng, 2000, shift=4.0)
    det = MahalanobisOOD().fit(train, y, calib, tpr=0.95)
    s_id, s_ood = det.score(test_id), det.score(test_ood)
    assert roc_auc_score(np.r_[np.zeros(2000), np.ones(2000)], np.r_[s_id, s_ood]) > 0.95
    assert abs(det.is_ood(test_id).mean() - 0.05) < 0.02   # ~5% ID false-positive rate
    det.save(tmp_path / "ood.npz")
    det2 = MahalanobisOOD.load(tmp_path / "ood.npz")
    np.testing.assert_allclose(det2.score(test_ood[:]), s_ood, rtol=1e-4)


def test_energy_higher_for_flat_logits():
    confident = np.array([[10.0, 0.0, 0.0]])
    flat = np.array([[0.1, 0.0, 0.05]])
    assert energy_score(flat)[0] > energy_score(confident)[0]
