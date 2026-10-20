"""Patient-level aggregation rules and 3D dataset discovery (manifests/volumes.csv)."""

import numpy as np
import pytest

from neuroedge import aggregate

CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
NT = CLASSES.index("no_tumor")


def study(n_clear=27, tumour_slices=None):
    """Most slices look normal; a few show a meningioma (as in a real study)."""
    p = np.tile([0.02, 0.03, 0.93, 0.02], (n_clear, 1))
    t = np.array(tumour_slices if tumour_slices is not None else [[0.1, 0.8, 0.05, 0.05]] * 3)
    return np.vstack([p, t])


def test_mean_misses_a_small_tumour_that_topk_finds():
    p = study()
    assert int(np.argmax(aggregate.aggregate(p, "mean", NT))) == NT                       # diluted by 27 normal slices
    for rule in ("topk", "max"):
        out = aggregate.aggregate(p, rule, NT, k=3)
        assert CLASSES[int(np.argmax(out))] == "meningioma" and out.sum() == pytest.approx(1.0)


def test_normal_study_stays_normal_for_every_rule():
    p = np.tile([0.03, 0.03, 0.91, 0.03], (30, 1))
    for rule in aggregate.RULES:
        assert int(np.argmax(aggregate.aggregate(p, rule, NT, lesion_area=np.zeros(30)))) == NT


def test_lesion_rule_follows_slices_with_a_lesion():
    p = study()
    area = np.r_[np.zeros(27), [400, 600, 500]]
    assert CLASSES[int(np.argmax(aggregate.aggregate(p, "lesion", NT, lesion_area=area)))] == "meningioma"
    with pytest.raises(ValueError):
        aggregate.aggregate(p, "lesion", NT)


def test_rejected_slices_are_ignored_unless_all_are_rejected():
    p = study()
    ok = np.r_[np.ones(27, bool), [False, False, False]]
    assert int(np.argmax(aggregate.aggregate(p, "topk", NT, ok=ok))) == NT                 # tumour slices unusable
    assert CLASSES[int(np.argmax(aggregate.aggregate(p, "topk", NT, ok=np.zeros(30, bool))))] == "meningioma"


def test_key_slices_and_errors():
    p = study()
    assert sorted(aggregate.key_slices(p, NT, 3)) == [27, 28, 29]
    with pytest.raises(ValueError):
        aggregate.aggregate(np.zeros((0, 4)), "mean", NT)
    with pytest.raises(ValueError):
        aggregate.aggregate(p, "median", NT)


def test_volume_manifest_discovers_each_layout(tmp_path):
    from training.data import build_volumes as bv
    raw = tmp_path / "raw"
    for ds, sid in (("brats_gli", "BraTS-GLI-00001-000"), ("brats_men", "BraTS-MEN-00007-000")):
        d = raw / ds / sid
        d.mkdir(parents=True)
        (d / f"{sid}-t1c.nii.gz").write_bytes(b"x")
        (d / f"{sid}-seg.nii.gz").write_bytes(b"x")
    (raw / "ixi" / "IXI-T1").mkdir(parents=True)
    (raw / "ixi" / "IXI-T1" / "IXI002-Guys-0828-T1.nii.gz").write_bytes(b"x")
    (raw / "pituitary_nhnn").mkdir(parents=True)
    for name in ("12_T1.nii.gz", "12_tumor_f.nii.gz", "12_T2_f.nii.gz", "12_carotids_f.nii.gz"):
        (raw / "pituitary_nhnn" / name).write_bytes(b"x")
    rows = {r["dataset"]: r for r in bv.build(raw)}
    assert rows["brats_gli"]["label"] == "glioma" and rows["brats_gli"]["seg_path"].endswith("-seg.nii.gz")
    assert rows["brats_men"]["label"] == "meningioma" and rows["brats_men"]["role"] == "dev"
    assert rows["ixi"] == {**rows["ixi"], "label": "no_tumor", "site": "Guys", "role": "dev_normal", "seg_path": ""}
    pit = rows["pituitary_nhnn"]
    assert pit["label"] == "pituitary" and pit["seg_path"].endswith("12_tumor_f.nii.gz")
    assert pit["t1c_path"].endswith("12_T1.nii.gz")                              # T2 and carotid masks ignored
