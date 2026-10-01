import cv2
import numpy as np
import pytest

pytest.importorskip("torch")
from training.dataset import augment  # noqa: E402


def test_augment_keeps_image_and_mask_aligned():
    mask = np.zeros((224, 224), np.uint8)
    cv2.circle(mask, (80, 120), 30, 1, -1)
    cv2.rectangle(mask, (140, 40), (190, 90), 1, -1)
    img = (0.05 + 0.85 * mask).astype(np.float32)          # image bright exactly where mask = 1
    rng = np.random.default_rng(0)
    for _ in range(50):
        a_img, a_mask = augment(img, mask, rng)
        assert a_img.shape == img.shape and a_mask.shape == mask.shape
        assert a_img.min() >= 0 and a_img.max() <= 1
        assert set(np.unique(a_mask)) <= {0, 1}             # nearest-neighbour: mask stays binary
        bright = a_img > 0.4
        inter, union = (bright & (a_mask > 0)).sum(), (bright | (a_mask > 0)).sum()
        assert inter / union > 0.9
