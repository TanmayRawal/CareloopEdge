import cv2
import numpy as np

from training.data.dedupe import best_similarity, group_ids, md5_all, similar_pairs, thumbnails


def _scan(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((256, 256), np.uint8)
    cv2.ellipse(img, (128, 128), (90, 110), 0, 0, 360, 150, -1)
    for _ in range(6):  # asymmetric, seed-dependent internal structure
        c = tuple(int(v) for v in rng.integers(70, 190, 2))
        cv2.circle(img, c, int(rng.integers(8, 30)), int(rng.integers(40, 255)), -1)
    return cv2.GaussianBlur(img, (5, 5), 0)


def test_copies_and_mirrors_grouped_distinct_kept_apart(tmp_path):
    a, b = _scan(1), _scan(2)
    specs = [
        ("a.png", a, []),
        ("a_resized.jpg", cv2.resize(a, (300, 300)), [cv2.IMWRITE_JPEG_QUALITY, 80]),
        ("a_mirror.png", a[:, ::-1].copy(), []),          # augmentation-style leak
        ("a_exact.png", a, []),
        ("b.png", b, []),
    ]
    paths = []
    for name, img, params in specs:
        p = tmp_path / name
        cv2.imwrite(str(p), img, params)
        paths.append(str(p))
    t = thumbnails(paths, workers=2)
    sim = best_similarity(t, t)
    assert sim[0, 1] > 0.98 and sim[0, 2] > 0.98       # re-encode and mirror both detected
    # A different image with the same head silhouette still correlates highly (~0.94), which is
    # why the same-patient threshold is calibrated on real patient IDs; it must stay below t_dup.
    assert sim[0, 4] < 0.98
    pairs, _ = similar_pairs(t, 0.98)
    ids = group_ids(len(paths), pairs, md5_all(paths, workers=2))
    assert ids[0] == ids[1] == ids[2] == ids[3]
    assert ids[4] != ids[0]
