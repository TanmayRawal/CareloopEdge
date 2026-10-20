"""Same-patient leakage check for 3D datasets: a planted duplicate (mirrored, intensity-jittered, stored in another
dataset) must be flagged, and a different patient must not be."""

import numpy as np
import pandas as pd
import pytest

nib = pytest.importorskip("nibabel")

from training.data import check_overlap  # noqa: E402


def head(seed, shape=(48, 56, 44)):
    """Ellipsoid head with random internal structure (a different 'patient' per seed)."""
    rng = np.random.default_rng(seed)
    x, y, z = np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij")
    vol = ((x / 0.85) ** 2 + (y / 0.9) ** 2 + (z / 0.85) ** 2 <= 1).astype(np.float32) * 200
    for _ in range(12):                              # blobs = individual anatomy
        c, r = rng.uniform(-0.5, 0.5, 3), rng.uniform(0.08, 0.2)
        vol[((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) < r ** 2] = rng.uniform(300, 900)
    return vol


def save(vol, path):
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(path))
    return str(path)


def test_planted_duplicate_is_flagged_and_other_patients_are_not(tmp_path):
    a = head(1)
    dup = a[::-1] * 1.3 + np.random.default_rng(9).normal(0, 4, a.shape).astype(np.float32)   # mirrored + jitter
    rows = pd.DataFrame([
        dict(study_id="A", dataset="brats_men", t1c_path=save(a, tmp_path / "a.nii.gz")),
        dict(study_id="A-copy", dataset="tcia_meningioma", t1c_path=save(dup, tmp_path / "dup.nii.gz")),
        dict(study_id="B", dataset="tcia_meningioma", t1c_path=save(head(2), tmp_path / "b.nii.gz")),
    ])
    cross, vs_train = check_overlap.check(rows, t_dup=0.98, t_patient=0.94, train_paths=[])
    pairs = {(tuple(p["a"]), tuple(p["b"])) for p in cross["pairs"]}
    assert (("brats_men", "A"), ("tcia_meningioma", "A-copy")) in pairs
    assert not any("B" in (p["a"][1], p["b"][1]) for p in cross["pairs"])
    assert vs_train == {}


def test_study_matching_is_against_2d_training_slices(tmp_path):
    import cv2
    from neuroedge import volume
    a = head(3)
    path = save(a, tmp_path / "a.nii.gz")
    # a "2D training image" that is one slice of this patient, saved as JPG like BRISC
    sl = volume.select_slices(volume.load_volume(path), per_plane=6)[2].raw.pixels
    cv2.imwrite(str(tmp_path / "slice.jpg"), (sl * 255).astype(np.uint8))
    rows = pd.DataFrame([dict(study_id="P", dataset="brats_gli", t1c_path=path)])
    _, vs_train = check_overlap.check(rows, t_dup=0.98, t_patient=0.94, train_paths=[str(tmp_path / "slice.jpg")])
    assert "brats_gli/P" in vs_train and vs_train["brats_gli/P"]["max_ncc"] >= 0.98
