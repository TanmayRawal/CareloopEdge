"""Whole-series analysis: Engine.analyze_series aggregates slice results into one study answer (key slices,
overlays per slice), and the portal accepts a series (several DICOM files, .zip, NIfTI), stores its slices, shows the
doctor the slice scrubber and still never gives the patient AI overlays."""

import io
import zipfile

import numpy as np
import pytest

pytest.importorskip("flask")
nib = pytest.importorskip("nibabel")

from edge.engine import Engine  # noqa: E402
from edge.portal import services  # noqa: E402
from neuroedge.uncertainty import ClassConditionalConformal  # noqa: E402
from test_portal import PW, StubEngine, csrf, env, login, register  # noqa: E402,F401

CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]


def nifti_bytes(tmp_path, tumour=True):
    x, y, z = np.meshgrid(*[np.linspace(-1, 1, n) for n in (40, 48, 36)], indexing="ij")
    head = ((x / 0.85) ** 2 + (y / 0.9) ** 2 + (z / 0.85) ** 2 <= 1).astype(np.float32) * 300
    if tumour:
        head[(x > 0.2) & (x < 0.5) & (y > 0.2) & (y < 0.5) & (z > -0.2) & (z < 0.2)] = 1000
    p = tmp_path / "study.nii.gz"
    nib.save(nib.Nifti1Image(head, np.eye(4)), str(p))
    return p.read_bytes()


class SliceEngine(Engine):
    """Engine without a model: a slice 'shows a tumour' when it contains the bright marker."""

    def __init__(self, reject_all=False):
        self.meta, self.class_names, self.reject_all = {"series": {"per_plane": 6}}, CLASSES, reject_all
        self.conformal = ClassConditionalConformal(alpha=0.05)
        self.conformal.thresholds = np.full(4, 0.5)
        self.calls = 0

    def analyze(self, source, filename=None, with_overlays=True):
        self.calls += 1
        px = source.pixels
        bright = float((px > 0.9).sum() / max(1, (px > 0.1).sum()))     # marker: a small very bright part of the head
        bright = bright if bright < 0.5 else 0.0
        p = np.array([0.02, 0.9, 0.06, 0.02]) if bright > 0.002 else np.array([0.01, 0.02, 0.96, 0.01])
        res = {"status": "rejected" if self.reject_all else "ok",
               "reject_reasons": ["out_of_distribution"] if self.reject_all else [],
               "probabilities": dict(zip(CLASSES, p)), "lesion": {"present": bright > 0.002, "area_px_model": 50},
               "flags": [], "_state": {"pre": type("P", (), {"original": (source.pixels * 255).astype(np.uint8)})}}
        if with_overlays:
            res["overlays"] = {k: b"PNG" for k in ("original", "cam", "lesion", "cam_lesion")}
        return res


def test_series_aggregates_slices_into_one_answer(tmp_path):
    eng = SliceEngine()
    res = eng.analyze_series(nifti_bytes(tmp_path), "study.nii.gz")
    sr = res["series"]
    assert res["status"] == "ok" and res["prediction"] == "meningioma" and res["prediction_set"] == ["meningioma"]
    assert sr["n_slices"] == len(sr["slices"]) and {s["plane"] for s in sr["slices"]} == {"axial", "coronal", "sagittal"}
    assert len(sr["key_slices"]) == 3 and all(sr["slices"][k]["tumour_prob"] > 0.5 for k in sr["key_slices"])
    ov = res["overlays"]
    assert all(f"s{i}_original" in ov for i in range(sr["n_slices"]))
    assert all(f"s{i}_{k}" in ov for i in range(sr["n_slices"]) for k in ("cam", "lesion", "cam_lesion"))
    assert {"original", "cam"} <= set(ov) and sr["overlays"] == "all"
    assert eng.calls == sr["n_slices"]                     # each slice is run through the model once


def test_series_without_tumour_and_mostly_rejected(tmp_path):
    assert SliceEngine().analyze_series(nifti_bytes(tmp_path, tumour=False), "s.nii.gz")["prediction"] == "no_tumor"
    res = SliceEngine(reject_all=True).analyze_series(nifti_bytes(tmp_path), "s.nii.gz", with_overlays=False)
    assert res["status"] == "rejected" and res["prediction"] is None and "out_of_distribution" in res["reject_reasons"]


def test_combine_files_packs_a_dicom_series():
    assert services.combine_files([("a.png", b"x")]) == ("a.png", b"x")
    name, data = services.combine_files([("IM1", b"d1"), ("IM2", b"d2")])
    assert services.is_series_name(name)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert [z.read(n) for n in z.namelist()] == [b"d1", b"d2"] and "IM1" not in z.namelist()
    with pytest.raises(services.UploadError):
        services.combine_files([("a.zip", b"x"), ("b.dcm", b"y")])


class SeriesStub(StubEngine):
    def analyze_series(self, data, filename=None):
        assert services.is_series_name(filename)
        res = self.analyze(b"ok")
        res["series"] = {"n_slices": 4, "rule": "topk", "k": 5, "key_slices": [2], "slices_rejected": 0,
                         "slices": [{"plane": "axial", "index": i, "tumour_prob": 0.9 if i == 2 else 0.1,
                                     "prediction": "meningioma", "ok": True} for i in range(4)]}
        res["overlays"] |= {f"s{i}_original": b"PNG-s" for i in range(4)} | {"s2_cam": b"PNG-c", "s2_lesion": b"L",
                                                                              "s2_cam_lesion": b"CL"}
        return res


def test_portal_series_upload_scrubber_and_patient_privacy(env):
    env["app"].extensions["portal"]["ctx"].engine = SeriesStub()
    p, d = env["app"].test_client(), env["app"].test_client()
    register(p)
    files = [(io.BytesIO(b"DICM" + bytes([i])), f"IM{i}.dcm") for i in range(4)]
    r = p.post("/patient/upload", data={"csrf": csrf(p, "/patient/upload"), "scan": files, "consent": "yes"},
               content_type="multipart/form-data")
    assert r.status_code == 302
    cid = env["db"].q("SELECT id FROM cases")[0]["id"]
    assert env["store"].exists(cid, "s3_original_1") and env["store"].exists(cid, "s2_cam_1")
    assert p.get(f"/case/{cid}/img/s1_original_1.png").status_code == 200
    assert p.get(f"/case/{cid}/img/s2_cam_1.png").status_code == 404           # no AI overlays for patients
    assert p.get(f"/case/{cid}/img/s2_bogus_1.png").status_code == 404
    login(d, "doc@clinic.test")
    page = d.get(f"/doctor/case/{cid}").get_data(as_text=True)
    assert 'id="series"' in page and "Whole study: 4 slices" in page
    assert d.get(f"/case/{cid}/img/s2_cam_1.png").data == b"PNG-c"
