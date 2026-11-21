"""Local ONNX-free conversion route: PyTorch checkpoints -> .tflite via
Google's ai-edge-torch converter. Weights are taken from the studies' own
TorchScript encoder checkpoints into the studies' own SemanticEncoder
architecture class; equivalence of the rebuilt module to the checkpoint
is verified numerically before conversion. No AI Hub artifacts are used.
"""
import os
import sys

import numpy as np
import torch

BASE = "/home/hatch/workspace/edge-apps-prep"
HERE = os.path.join(BASE, "litert_check")
sys.path.insert(0, os.path.join(BASE, "semantic_encoder"))
from semantic_encoder import SemanticEncoder  # noqa: E402

import litert_torch  # noqa: E402  (Google's converter; ai-edge-torch is its deprecated former name)

APPS = {
    "india": f"{BASE}/semantic_encoder/india/semantic_encoder_india_trained.pt",
    "outage": f"{BASE}/outage_sim/semantic_encoder_outage_trained.pt",
    "metering": f"{BASE}/metering_sim/semantic_encoder_metering_trained.pt",
    "parking": f"{BASE}/parking_sim/semantic_encoder_parking_trained.pt",
}

for app, pt_path in APPS.items():
    traced = torch.jit.load(pt_path).eval()
    m = SemanticEncoder().eval()
    m.load_state_dict(traced.state_dict())  # strict: raises on any mismatch
    x = torch.randn(4, 64)
    with torch.no_grad():
        d = (m(x) - traced(x)).abs().max().item()
    assert d == 0.0, f"{app}: rebuilt module differs from checkpoint ({d})"
    out = os.path.join(HERE, f"semantic_encoder_{app}_local.tflite")
    edge = litert_torch.convert(m, (torch.zeros(1, 64),))
    edge.export(out)
    print(f"{app}: converted -> {out} ({os.path.getsize(out)} bytes), "
          f"rebuilt-vs-checkpoint max diff {d}", flush=True)
print("ALL CONVERTED")
