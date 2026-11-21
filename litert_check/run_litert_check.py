"""LiteRT compatibility check for the four trained semantic encoders.

For each application: take real test windows from the study's own saved
dataset, run the PyTorch (TorchScript) encoder and the locally converted
.tflite encoder (Google LiteRT interpreter, ai-edge-litert) on identical
windows, compare latents, then push both latents through the study's own
eval path -- int8 quantisation round(z*127)/127, the trained gateway
decoder, sigmoid, and the study's validation-chosen decision threshold --
and compare the resulting event decisions.
"""
import json
import os
import sys

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
HERE = os.path.join(BASE, "litert_check")

from ai_edge_litert.interpreter import Interpreter  # noqa: E402

APPS = {
    "india": {
        "enc_pt": f"{BASE}/flood_india/india/semantic_encoder_india_trained.pt",
        "dec_pt": f"{BASE}/flood_india/india/gateway_decoder_india_trained.pt",
        "tflite": f"{HERE}/semantic_encoder_india_local.tflite",
        "npz": f"{BASE}/flood_india/india/dataset_india.npz",
        "threshold": 0.35,
    },
    "outage": {
        "enc_pt": f"{BASE}/outage/semantic_encoder_outage_trained.pt",
        "dec_pt": f"{BASE}/outage/gateway_decoder_outage_trained.pt",
        "tflite": f"{HERE}/semantic_encoder_outage_local.tflite",
        "npz": f"{BASE}/outage/dataset_outage.npz",
        "threshold": 0.95,
    },
    "metering": {
        "enc_pt": f"{BASE}/metering/semantic_encoder_metering_trained.pt",
        "dec_pt": f"{BASE}/metering/gateway_decoder_metering_trained.pt",
        "tflite": f"{HERE}/semantic_encoder_metering_local.tflite",
        "npz": f"{BASE}/metering/dataset_metering.npz",
        "threshold": 0.70,
    },
    "parking": {
        "enc_pt": f"{BASE}/parking/semantic_encoder_parking_trained.pt",
        "dec_pt": f"{BASE}/parking/gateway_decoder_parking_trained.pt",
        "tflite": f"{HERE}/semantic_encoder_parking_local.tflite",
        "npz": f"{BASE}/parking/dataset_parking.npz",
        "threshold": 0.75,
    },
}

N_FIRST, N_POS = 16, 16


def load_test_windows(app, cfg):
    d = np.load(cfg["npz"])
    if "Xte" in d:
        X, y = d["Xte"], d["yte"]
    else:  # india: single array, test = window end year >= 2024
        m = d["end_year"] >= 2024
        X, y = d["X"][m], d["y"][m]
    idx = list(range(min(N_FIRST, len(X))))
    pos = [int(i) for i in np.where(y == 1)[0] if int(i) not in idx][:N_POS]
    idx += pos
    return X[idx].astype(np.float32), y[idx].astype(int), int((y == 1).sum()), len(X)


def quantise(z):
    return np.clip(np.round(z * 127.0) / 127.0, -1.0, 1.0)


def main():
    results = {}
    for app, cfg in APPS.items():
        X, y, n_pos_total, n_test_total = load_test_windows(app, cfg)
        # ---- PyTorch reference (the exact artifacts the studies exported)
        enc = torch.jit.load(cfg["enc_pt"]).eval()
        dec = torch.jit.load(cfg["dec_pt"]).eval()
        with torch.no_grad():
            z_pt = enc(torch.from_numpy(X)).numpy()
            logit_pt = dec(torch.from_numpy(quantise(z_pt).astype(np.float32)))
            logit_pt = logit_pt.numpy().reshape(-1)
        prob_pt = 1.0 / (1.0 + np.exp(-logit_pt))
        dec_pt = (prob_pt >= cfg["threshold"]).astype(int)

        # ---- LiteRT interpreter on the locally converted .tflite
        itp = Interpreter(model_path=cfg["tflite"])
        itp.allocate_tensors()
        inp = itp.get_input_details()[0]
        out = itp.get_output_details()[0]
        z_lite = np.empty((len(X), 8), dtype=np.float32)
        for i in range(len(X)):
            itp.set_tensor(inp["index"], X[i:i + 1])
            itp.invoke()
            z_lite[i] = itp.get_tensor(out["index"])[0]
        with torch.no_grad():
            logit_li = dec(torch.from_numpy(quantise(z_lite).astype(np.float32)))
            logit_li = logit_li.numpy().reshape(-1)
        prob_li = 1.0 / (1.0 + np.exp(-logit_li))
        dec_li = (prob_li >= cfg["threshold"]).astype(int)

        dz = np.abs(z_pt - z_lite)
        codes_pt = np.clip(np.round(z_pt * 127.0), -127, 127)
        codes_li = np.clip(np.round(z_lite * 127.0), -127, 127)
        r = {
            "n_windows": int(len(X)),
            "n_positive_windows_in_sample": int(y.sum()),
            "test_set_size": int(n_test_total),
            "test_set_positives": int(n_pos_total),
            "threshold": cfg["threshold"],
            "latent_max_abs_diff": float(dz.max()),
            "latent_mean_abs_diff": float(dz.mean()),
            "int8_code_exact_match_frac": float((codes_pt == codes_li).mean()),
            "int8_code_max_abs_diff_in_levels": int(np.abs(codes_pt - codes_li).max()),
            "prob_max_abs_diff": float(np.abs(prob_pt - prob_li).max()),
            "decision_agreement": int((dec_pt == dec_li).sum()),
            "decision_total": int(len(X)),
            "pytorch_decisions_positive": int(dec_pt.sum()),
            "litert_decisions_positive": int(dec_li.sum()),
        }
        results[app] = r
        print(app, json.dumps(r, indent=2), flush=True)
    with open(os.path.join(HERE, "litert_check_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
