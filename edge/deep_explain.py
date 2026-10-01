"""On-demand high-resolution explanation ("Deep Explain") for the Pi: LayerCAM over layer3 + layer4.

Chosen from the measured comparison in training/xai_eval.py: LayerCAM had the best pointing game
(0.76 vs 0.53 for Grad-CAM) at lower ROAD faithfulness, so it is offered as an optional detail view,
never as the default. It needs real back-propagation through layer4, which the INT8 ONNX graph cannot
provide, so it runs the FP32 PyTorch weights shipped (signed + encrypted) in the bundle.

LayerCAM is implemented here with plain hooks rather than imported from pytorch-grad-cam, so the Pi needs
only torch + timm (grad-cam drags in matplotlib, torchvision transforms and full opencv). The arithmetic
reproduces pytorch-grad-cam 1.5 exactly (tests/test_deep_explain.py checks this), so the map shown on the
Pi is the one that was evaluated:
  per layer l   M_l = minmax( ReLU( sum_k ReLU(dy_c/dA_k) * A_k ) ), bilinearly resized to the input
  fused         M   = minmax( mean_l M_l )
"""

from __future__ import annotations

import io
import time

import cv2
import numpy as np


def _minmax(m: np.ndarray) -> np.ndarray:
    m = m - m.min()
    return m / (1e-7 + m.max())


def layercam(net, x, target: int, layers) -> np.ndarray:
    """LayerCAM (Jiang et al., IEEE TIP 2021) of logit `target` fused over `layers`. x: (1, C, H, W) tensor."""
    import torch

    acts: list = []
    handles = [layer.register_forward_hook(lambda _m, _i, out: acts.append(out)) for layer in layers]
    try:
        logits = net(x)["logits"]
        grads = torch.autograd.grad(logits[0, target], acts)
    finally:
        for h in handles:
            h.remove()
    h, w = x.shape[-2:]
    per_layer = []
    for a, g in zip(acts, grads):
        a, g = a[0].detach().cpu().numpy(), g[0].detach().cpu().numpy()
        cam = np.maximum((np.maximum(g, 0) * a).sum(axis=0), 0)
        per_layer.append(cv2.resize(np.float32(_minmax(cam)), (w, h)))
    return np.float32(_minmax(np.mean(np.maximum(np.stack(per_layer), 0), axis=0)))


class DeepExplainer:
    def __init__(self, weights: bytes, meta: dict, threads: int = 4):
        import torch

        from neuroedge.config import ModelConfig
        from neuroedge.models import MultiTaskNet

        torch.set_num_threads(threads)
        state = torch.load(io.BytesIO(weights), map_location="cpu", weights_only=True)
        net = MultiTaskNet(len(meta["class_names"]), ModelConfig(**{**meta["model"], "pretrained": False}))
        net.load_state_dict(state)
        self.net = net.eval()
        self.class_names = meta["class_names"]

    def explain(self, tensor: np.ndarray, target: int | None = None) -> dict:
        import torch

        t0 = time.perf_counter()
        x = torch.from_numpy(tensor[None])
        if target is None:
            with torch.no_grad():
                target = int(self.net(x)["logits"].argmax(1))
        m = layercam(self.net, x, target, [self.net.encoder.layer3, self.net.encoder.layer4])
        return {"method": "LayerCAM (layer3+layer4, FP32)", "explained_class": self.class_names[target],
                "map": m, "time_ms": round((time.perf_counter() - t0) * 1000, 1)}
