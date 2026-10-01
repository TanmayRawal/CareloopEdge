import io

import cv2
import numpy as np
import pytest

from neuroedge.config import PreprocessConfig, load_config
from neuroedge.preprocess import inverse_map, preprocess, read_image, window_ct


def _synthetic_scan(h=300, w=260):
    img = np.zeros((h, w), np.uint8)
    cv2.ellipse(img, (130, 160), (90, 110), 0, 0, 360, 170, -1)       # "head"
    cv2.circle(img, (150, 130), 20, 250, -1)                           # "lesion"
    lesion = np.zeros_like(img)
    cv2.circle(lesion, (150, 130), 20, 1, -1)
    return img, lesion


def _png_bytes(img):
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_preprocess_shapes_and_determinism():
    img, lesion = _synthetic_scan()
    cfg = PreprocessConfig()
    a = preprocess(_png_bytes(img), cfg, filename="x.png", mask=lesion)
    b = preprocess(_png_bytes(img), cfg, filename="x.png", mask=lesion)
    assert a.tensor.shape == (3, 224, 224) and a.tensor.dtype == np.float32
    np.testing.assert_array_equal(a.tensor, b.tensor)
    x0, y0, x1, y1 = a.geometry.crop_box
    assert x0 <= 40 and x1 >= 220 and y0 <= 50 and y1 >= 270    # crop contains the head
    assert (x1 - x0) < 260 or (y1 - y0) < 300                   # and actually crops something
    assert a.mask is not None and a.mask.sum() > 0
    assert a.quality_flags == [] or "blurry_or_flat" in a.quality_flags


def test_inverse_map_recovers_mask():
    img, lesion = _synthetic_scan()
    r = preprocess(_png_bytes(img), PreprocessConfig(), mask=lesion)
    back = inverse_map(r.mask.astype(np.float32), r.geometry, cv2.INTER_LINEAR) > 0.5
    inter = (back & (lesion > 0)).sum()
    union = (back | (lesion > 0)).sum()
    assert inter / union > 0.9


def test_ct_window():
    hu = np.array([-100.0, 0.0, 40.0, 80.0, 1000.0])
    np.testing.assert_allclose(window_ct(hu, 40, 80), [0, 0, 0.5, 1, 1])


def test_dicom_ct_reads_hu_and_drops_phi():
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientName, ds.PatientID, ds.Modality = "Doe^Jane", "12345", "CT"
    ds.Rows, ds.Columns = 64, 64
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
    ds.PixelRepresentation, ds.SamplesPerPixel = 0, 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
    ds.PixelSpacing = [0.5, 0.5]
    ds.PixelData = np.full((64, 64), 1064, np.uint16).tobytes()   # 1064 - 1024 = 40 HU
    buf = io.BytesIO()
    try:
        ds.save_as(buf, enforce_file_format=True)    # pydicom >= 3
    except TypeError:
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.save_as(buf, write_like_original=False)    # pydicom 2.x
    raw = read_image(buf.getvalue(), "slice.dcm")
    assert raw.is_hu and raw.modality == "CT"
    np.testing.assert_allclose(raw.pixels, 40.0)
    assert raw.pixel_spacing_mm == (0.5, 0.5)
    assert "PatientName" not in raw.meta and "PatientID" not in raw.meta


def test_color_photo_flagged():
    rng = np.random.default_rng(0)
    photo = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    r = preprocess(_png_bytes(photo), PreprocessConfig(), filename="cat.png")
    assert "color_image" in r.quality_flags


def test_config_roundtrip_and_fingerprint(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: mri\npreprocess:\n  image_size: 256\n  clahe: true\nmodel:\n  output_stride: 32\n")
    cfg = load_config(p)
    assert cfg.preprocess.image_size == 256 and cfg.model.output_stride == 32 and cfg.num_classes == 4
    assert cfg.preprocess.fingerprint() != PreprocessConfig().fingerprint()
    assert PreprocessConfig().fingerprint() == PreprocessConfig().fingerprint()
    p.write_text("name: mri\nbogus_key: 1\n")
    with pytest.raises(ValueError):
        load_config(p)
