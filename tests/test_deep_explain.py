"""The Pi's dependency-free LayerCAM must equal pytorch-grad-cam's (the implementation that was evaluated)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gradcam = pytest.importorskip("pytorch_grad_cam")

from edge.deep_explain import layercam  # noqa: E402
from neuroedge.config import ModelConfig  # noqa: E402
from neuroedge.models import MultiTaskNet  # noqa: E402


class _Logits(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)["logits"]


@pytest.mark.parametrize("target", [0, 2])
def test_layercam_matches_library(target):
    from pytorch_grad_cam import LayerCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    torch.manual_seed(0)
    net = MultiTaskNet(4, ModelConfig(pretrained=False)).eval()
    x = torch.randn(1, 3, 224, 224)
    layers = [net.encoder.layer3, net.encoder.layer4]
    ours = layercam(net, x, target, layers)
    with LayerCAM(model=_Logits(net).eval(), target_layers=layers) as cam:
        ref = cam(input_tensor=x, targets=[ClassifierOutputTarget(target)])[0]
    assert ours.shape == ref.shape == (224, 224)
    np.testing.assert_allclose(ours, ref, atol=1e-5)
