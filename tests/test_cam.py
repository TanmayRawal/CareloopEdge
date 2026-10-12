"""The CAM closed forms in neuroedge.cam must match the definitions exactly."""

import numpy as np
import pytest

from neuroedge import cam


def _random_head(seed=0, k=32, h=14, w=14, c=4):
    rng = np.random.default_rng(seed)
    feats = np.maximum(rng.normal(size=(k, h, w)), 0)  # post-ReLU-like activations
    weight, bias = rng.normal(size=(c, k)), rng.normal(size=c)
    logits = weight @ feats.mean(axis=(1, 2)) + bias
    return feats, weight, bias, logits


def test_contribution_maps_sum_to_logits():
    feats, w, b, logits = _random_head()
    assert cam.decomposition_error(feats, w, b, logits) < 1e-9
    for c in range(4):
        assert cam.exact_cam(feats, w, c).sum() + b[c] == pytest.approx(logits[c], abs=1e-9)


def test_finer_cam_decomposes_margin():
    feats, w, b, logits = _random_head(seed=1)
    t = int(np.argmax(logits))
    r = cam.most_similar_class(logits, t)
    assert r != t
    for gamma in (0.0, 0.5, 1.0):
        m = cam.finer_cam(feats, w, t, r, gamma)
        expected = (logits[t] - b[t]) - gamma * (logits[r] - b[r])
        assert m.sum() == pytest.approx(expected, abs=1e-9)


def test_display_scaling_is_linear():
    x = np.array([[-2.0, 0.0], [1.0, 4.0]])
    np.testing.assert_allclose(cam.scale_for_display(x), [[0, 0], [0.25, 1.0]])
    np.testing.assert_allclose(cam.scale_for_display(x, signed=True), [[-0.5, 0], [0.25, 1.0]])
    assert cam.scale_for_display(np.zeros((3, 3))).max() == 0


def test_evidence_inside_fraction():
    m = np.zeros((4, 4))
    m[0, 0], m[3, 3] = 3.0, 1.0
    mask = np.zeros((4, 4), np.uint8)
    mask[0, 0] = 1
    assert cam.evidence_inside_fraction(m, mask) == pytest.approx(0.75)


@pytest.fixture(scope="module")
def net():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from neuroedge.config import ModelConfig
    from neuroedge.models import MultiTaskNet

    torch.manual_seed(0)
    model = MultiTaskNet(4, ModelConfig(pretrained=False)).eval()
    return model


def test_model_outputs_and_stride16(net):
    import torch

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = net(x)
    assert out["logits"].shape == (2, 4)
    assert out["features"].shape[-2:] == (14, 14)          # output_stride = 16
    assert out["seg_logits"].shape == (2, 1, 224, 224)
    assert [e.shape[1] for e in out["embeddings"]] == [256, 512, 1024, 2048]


def _np_head(net, x):
    import torch

    with torch.no_grad():
        out = net(x)
    w, b = net.head_weights()
    return out["features"][0].double().numpy(), w, b, out["logits"][0].double().numpy()


def test_exact_cam_equals_autograd_gradcam_and_hirescam(net):
    import torch

    x = torch.randn(1, 3, 224, 224)
    feats, w, b, logits = _np_head(net, x)
    assert cam.decomposition_error(feats, w, b, logits) < 1e-4
    c = int(np.argmax(logits))
    gradcam, hires = cam.autograd_cams(net, x, lambda y: y[:, c])
    exact = cam.exact_cam(feats, w, c)
    scale = np.abs(exact).max()
    np.testing.assert_allclose(hires[0], exact, atol=1e-5 * scale)
    np.testing.assert_allclose(gradcam[0], np.maximum(exact, 0), atol=1e-5 * scale)


def test_finer_cam_equals_autograd_of_logit_difference(net):
    import torch

    x = torch.randn(1, 3, 224, 224)
    feats, w, b, logits = _np_head(net, x)
    t = int(np.argmax(logits))
    r = cam.most_similar_class(logits, t)
    gamma = 0.7
    _, hires = cam.autograd_cams(net, x, lambda y: y[:, t] - gamma * y[:, r])
    exact = cam.finer_cam(feats, w, t, r, gamma)
    np.testing.assert_allclose(hires[0], exact, atol=1e-5 * np.abs(exact).max())


# ------------------------------------------------------------------ v1 lesion-attention pooling
def test_weighted_pooling_decomposition_numpy():
    rng = np.random.default_rng(3)
    feats = np.maximum(rng.normal(size=(16, 14, 14)), 0)
    w, b = rng.normal(size=(4, 16)), rng.normal(size=4)
    att = rng.random((14, 14))
    att /= att.sum()
    logits = w @ np.einsum("khw,hw->k", feats, att) + b
    assert cam.decomposition_error(feats, w, b, logits, att) < 1e-9
    assert cam.decomposition_error(feats, w, b, logits) > 1e-3          # GAP formula is wrong for this head


@pytest.fixture(scope="module")
def lesion_net():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from neuroedge.config import ModelConfig
    from neuroedge.models import MultiTaskNet

    torch.manual_seed(1)
    return MultiTaskNet(4, ModelConfig(pretrained=False, pooling="lesion", attn_eps=0.02, seg_aux_weight=0.4)).eval()


def test_lesion_head_exact_cam_and_hirescam(lesion_net):
    import torch

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = lesion_net(x)
    att = out["attention"][0].double().numpy()
    assert att.shape == (14, 14) and abs(att.sum() - 1) < 1e-5
    feats, (w, b) = out["features"][0].double().numpy(), lesion_net.head_weights()
    logits = out["logits"][0].double().numpy()
    assert cam.decomposition_error(feats, w, b, logits, att) < 1e-4
    c = int(np.argmax(logits))
    _, hires = cam.autograd_cams(lesion_net, x, lambda y: y[:, c])
    exact = cam.exact_cam(feats, w, c, att)
    np.testing.assert_allclose(hires[0], exact, atol=1e-5 * np.abs(exact).max())
    assert "seg_aux_logits" in out and out["seg_aux_logits"].shape == (1, 1, 224, 224)


def test_lesion_attention_uniform_without_lesion_and_focused_with_lesion():
    import torch

    from neuroedge.models import lesion_attention

    none = lesion_attention(torch.full((1, 1, 224, 224), -30.0), 16, 0.02)
    np.testing.assert_allclose(none.numpy(), 1 / 196, rtol=1e-4)        # no lesion -> exactly GAP
    seg = torch.full((1, 1, 224, 224), -30.0)
    seg[..., :32, :32] = 30.0                                          # tumour = 2 x 2 cells of 196
    a = lesion_attention(seg, 16, 0.02)[0]
    # exact: tumour cells weigh (1 + eps) each, the other 192 cells eps each
    assert float(a[:2, :2].sum()) == pytest.approx(4 * 1.02 / (4 * 1.02 + 192 * 0.02), rel=1e-4)
    assert abs(float(a.sum()) - 1) < 1e-6


def test_classification_loss_does_not_reach_segmentation_head(lesion_net):
    import torch

    lesion_net.train()
    lesion_net.zero_grad()
    out = lesion_net(torch.randn(2, 3, 224, 224))
    torch.nn.functional.cross_entropy(out["logits"], torch.tensor([0, 1])).backward()
    seg_grads = [p.grad for p in lesion_net.seg_head.parameters()]
    assert all(g is None or float(g.abs().max()) == 0.0 for g in seg_grads)
    assert float(lesion_net.classifier.weight.grad.abs().max()) > 0
    lesion_net.eval()
