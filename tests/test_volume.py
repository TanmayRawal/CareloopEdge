"""3D volume reader: orientation (a marker placed anterior-right-superior must show up where a radiologist expects
it in every plane), NIfTI vs DICOM-series agreement, shuffled slice order, anisotropic voxels, masks, slice
selection, and that the 2D preprocessing contract is unchanged."""

import io
import zipfile

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pydicom = pytest.importorskip("pydicom")

from neuroedge import volume  # noqa: E402
from neuroedge.config import PreprocessConfig  # noqa: E402
from neuroedge.preprocess import preprocess  # noqa: E402

SHAPE = (40, 48, 36)          # RAS+ (X, Y, Z)


def phantom():
    """Ellipsoid head with a bright cube near patient right (+X), anterior (+Y) and superior (+Z)."""
    x, y, z = np.meshgrid(*[np.linspace(-1, 1, n) for n in SHAPE], indexing="ij")
    head = ((x / 0.85) ** 2 + (y / 0.9) ** 2 + (z / 0.85) ** 2 <= 1).astype(np.float32) * 300
    marker = (x > 0.3) & (x < 0.6) & (y > 0.3) & (y < 0.6) & (z > 0.3) & (z < 0.6)
    head[marker] = 1000
    return head, marker.astype(np.uint8)


def to_nifti(arr, affine, path):
    nib.save(nib.Nifti1Image(arr, affine), str(path))


def marker_quadrant(img):
    """(top?, left?) of the brightest region in a display image."""
    r, c = np.unravel_index(np.argmax(img), img.shape)
    return r < img.shape[0] / 2, c < img.shape[1] / 2


def check_display(vol):
    k = {"axial": int(0.75 * vol.data.shape[2]), "coronal": int(0.75 * vol.data.shape[1]),
         "sagittal": int(0.75 * vol.data.shape[0])}
    # axial: anterior at top, patient right on image left
    assert marker_quadrant(volume._plane_image(vol.data, "axial", k["axial"])) == (True, True)
    # coronal: superior at top, patient right on image left
    assert marker_quadrant(volume._plane_image(vol.data, "coronal", k["coronal"])) == (True, True)
    # sagittal: superior at top, anterior on image left
    assert marker_quadrant(volume._plane_image(vol.data, "sagittal", k["sagittal"])) == (True, True)


def test_nifti_canonical_orientation_from_flipped_affine(tmp_path):
    head, _ = phantom()
    # store the data in LPS order (X and Y flipped) with an affine that says so
    lps = head[::-1, ::-1, :].copy()
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine[:3, 3] = [SHAPE[0] - 1, SHAPE[1] - 1, 0]
    to_nifti(lps, affine, tmp_path / "v.nii.gz")
    vol = volume.load_volume(tmp_path / "v.nii.gz")
    assert vol.data.shape == SHAPE and 0.0 <= vol.data.min() and vol.data.max() <= 1.0
    check_display(vol)


def test_nifti_permuted_axes_and_spacing(tmp_path):
    head, _ = phantom()
    # stored as (Z, X, Y) with 3 mm slices along stored axis 0 (= S)
    stored = np.transpose(head, (2, 0, 1)).copy()
    affine = np.zeros((4, 4))
    affine[2, 0], affine[0, 1], affine[1, 2], affine[3, 3] = 3.0, 1.0, 1.0, 1.0
    to_nifti(stored, affine, tmp_path / "p.nii")
    vol = volume.load_volume(tmp_path / "p.nii")
    assert vol.data.shape == SHAPE
    assert vol.spacing_mm == pytest.approx((1.0, 1.0, 3.0))
    check_display(vol)
    # coronal slices are resampled to square pixels: 36 slices x 3 mm -> 108 rows at 1 mm
    sl = [s for s in volume.select_slices(vol, per_plane=3) if s.plane == "coronal"][0]
    assert sl.raw.pixels.shape == (108, 40) and sl.raw.pixel_spacing_mm == (1.0, 1.0)


def dicom_series(arr_ras, shuffle=True):
    """Axial DICOM series (one file per z) in LPS patient coordinates from a RAS+ array."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    series = generate_uid()
    lps = arr_ras[::-1, ::-1, :]                       # X: R->L, Y: A->P
    files = []
    for k in range(lps.shape[2]):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        ds.SeriesInstanceUID, ds.Modality = series, "MR"
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]      # rows along +L (x), columns along +P (y)
        ds.ImagePositionPatient = [0, 0, float(k) * 2.0]      # 2 mm slices
        ds.PixelSpacing = [1.0, 1.0]
        img = lps[:, :, k].T.astype(np.uint16)                # (rows=y, cols=x)
        ds.Rows, ds.Columns = img.shape
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.PatientName = "SHOULD-NOT-SURVIVE"
        ds.PixelData = img.tobytes()
        buf = io.BytesIO()
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
        ds.save_as(buf, enforce_file_format=True)
        files.append(buf.getvalue())
    if shuffle:
        rng = np.random.default_rng(0)
        files = [files[i] for i in rng.permutation(len(files))]
    return files


def test_dicom_series_matches_nifti_and_hides_identity(tmp_path):
    head, _ = phantom()
    vol = volume.load_volume(dicom_series(head))
    assert vol.data.shape == SHAPE and vol.spacing_mm == pytest.approx((1.0, 1.0, 2.0))
    check_display(vol)
    assert "SHOULD-NOT-SURVIVE" not in repr(vol.meta)
    # the same series inside a zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i, f in enumerate(dicom_series(head)):
            z.writestr(f"series/IM{i:04d}.dcm", f)
    vz = volume.load_volume(buf.getvalue(), "study.zip")
    assert np.allclose(vz.data, vol.data)


def test_dicom_upload_mixing_series_or_duplicates_is_refused():
    """Every slice shown to the doctor must come from the same scan: two series (e.g. two patients) uploaded
    together, or the same slice twice, are refused instead of being silently reduced to one series."""
    head, _ = phantom()
    a, b = dicom_series(head), dicom_series(head * 0.5)          # each call is a new series
    with pytest.raises(ValueError, match="different series"):
        volume.load_volume(a + b[:5])
    with pytest.raises(ValueError, match="duplicated"):
        volume.load_volume(a + a[:2])


def test_mask_aligned_with_volume_and_slices(tmp_path):
    head, marker = phantom()
    to_nifti(head, np.eye(4), tmp_path / "t1c.nii.gz")
    to_nifti(marker, np.eye(4), tmp_path / "seg.nii.gz")
    vol = volume.load_volume(tmp_path / "t1c.nii.gz")
    mask = volume.load_mask(tmp_path / "seg.nii.gz", like=vol)
    for s in volume.select_slices(vol, per_plane=8, mask=mask):
        if s.mask.any():          # wherever the mask is, the marker (brightest voxels) is
            assert s.raw.pixels[s.mask > 0].mean() > 0.9


def test_slice_selection_covers_three_planes_and_skips_empty_edges():
    head, _ = phantom()
    vol = volume.Volume(volume._scale_unit(head), (1.0, 1.0, 1.0), "nifti")
    slices = volume.select_slices(vol, per_plane=10)
    assert {s.plane for s in slices} == set(volume.PLANES)
    for s in slices:
        assert (s.raw.pixels > 0.1).mean() > 0.05, (s.plane, s.index)    # no blank edge slices


def test_slices_go_through_the_unchanged_2d_preprocessing():
    head, _ = phantom()
    vol = volume.Volume(volume._scale_unit(head), (1.0, 1.0, 1.0), "nifti")
    cfg = PreprocessConfig()
    before = cfg.fingerprint()
    res = preprocess(volume.select_slices(vol, per_plane=2)[0].raw, cfg)
    assert res.tensor.shape == (3, cfg.image_size, cfg.image_size)
    assert cfg.fingerprint() == before


def test_rejects_non_volumes():
    with pytest.raises(ValueError):
        volume.load_volume(b"not an image at all", "x.bin")
    with pytest.raises(ValueError):
        volume.load_volume([b"x", b"y"])


def test_slice_export_labels_from_masks(tmp_path):
    """Volume slices for training: slices through the tumour get the patient's tumour type, slices without tumour
    get no_tumor, slivers are skipped; masks and images are saved side by side."""
    import cv2
    import pandas as pd
    from training.data.export_volume_slices import MIN_TUMOUR_PX, export
    head, marker = phantom()
    to_nifti(head, np.eye(4), tmp_path / "t1c.nii.gz")
    to_nifti(marker, np.eye(4), tmp_path / "seg.nii.gz")
    rows = pd.DataFrame([dict(study_id="S1", dataset="brats_men", label="meningioma", role="dev", split="train",
                              t1c_path=str(tmp_path / "t1c.nii.gz"), seg_path=str(tmp_path / "seg.nii.gz"))])
    recs = export(rows, tmp_path / "out", per_plane=6, tumour_per_plane=4)
    labels = {r["label"] for r in recs}
    assert labels == {"meningioma", "no_tumor"} and all(r["split"] == "train" for r in recs)
    for r in recs:
        m = cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE)
        area = int((m > 0).sum())
        assert (area >= MIN_TUMOUR_PX) == (r["label"] == "meningioma") and (area == 0 or area >= MIN_TUMOUR_PX)
        assert cv2.imread(r["path"], cv2.IMREAD_GRAYSCALE).shape == m.shape
