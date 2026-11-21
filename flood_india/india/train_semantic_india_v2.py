"""Train semantic encoder + gateway decoder v2 on the SAME India windows
as v1, with INDOFLOODS catchment attributes appended as static per-station
inputs at the GATEWAY DECODER.

What changes vs train_semantic_india.py (v1) — and only this:
  * The decoder's first layer takes [latent(8), station_attributes(6)]
    instead of latent(8) alone. The six attributes per station come from
    raw/metadata_indofloods.csv via catchment_attributes_v2.json
    (log catchment area, area-estimate variation, danger level, lat, lon,
    reliability flag — see india_v2_analysis.py).
  * The encoder is UNCHANGED (64->48->24->8, same (1,64)->(1,8) contract),
    so the over-the-air payload is still the 8-byte int8 latent; the
    attributes are station metadata known at the gateway, not transmitted.

Everything else is v1's protocol verbatim: windows/splits loaded from
dataset_india.npz (built by v1 — identical windows guaranteed), seed
20261118, joint training, int8 quantisation in the loop via STE, Gaussian
channel noise sigma=0.03 in training, loss = pos-weighted BCE + 0.05*MSE,
Adam lr 1e-3 wd 1e-4, batch 256, max 40 epochs, early stop patience 8 on
validation F1, decision threshold selected on validation over the same
0.05..0.95 grid. All outputs carry "v2" in their names; no v1 file is
touched.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)                       # semantic_encoder/
sys.path.insert(0, BASE)

from semantic_encoder import SemanticEncoder, INPUT_DIM, LATENT_DIM  # noqa: E402

NOISE_SIGMA = 0.03
RECON_WEIGHT = 0.05
STATION_ORDER = ["Jainagar", "Jhanjharpur", "Benibad",
                 "Kallooppara", "Akhuapada", "Kunavaram"]


class GatewayDecoderV2(nn.Module):
    """v1 decoder body, first layer widened to latent + static attributes."""

    def __init__(self, attr_dim, latent_dim=LATENT_DIM, input_dim=INPUT_DIM):
        super().__init__()
        self.attr_dim = attr_dim
        self.body = nn.Sequential(
            nn.Linear(latent_dim + attr_dim, 24), nn.ReLU(),
            nn.Linear(24, 48), nn.ReLU(),
        )
        self.task_head = nn.Linear(48, 1)
        self.recon_head = nn.Linear(48, input_dim)

    def forward(self, z, attr):
        h = self.body(torch.cat([z, attr], dim=-1))
        return self.task_head(h).squeeze(-1), self.recon_head(h)


def quantise_ste(z):
    q = torch.clamp(torch.round(z * 127.0) / 127.0, -1.0, 1.0)
    return z + (q - z).detach()


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    far = fp / (fp + tn) if fp + tn else 0.0
    return {"accuracy": round(float(acc), 4), "precision": round(float(prec), 4),
            "recall": round(float(rec), 4), "f1": round(float(f1), 4),
            "false_alarm_rate": round(float(far), 4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": int(len(y_true)),
            "positives": int(y_true.sum())}


def main():
    t0 = time.time()
    torch.manual_seed(20261118)
    np.random.seed(20261118)

    d = np.load(os.path.join(HERE, "dataset_india.npz"))
    X, y, r, sidx, end_year = d["X"], d["y"], d["rule"], d["station"], d["end_year"]
    with open(os.path.join(HERE, "catchment_attributes_v2.json")) as f:
        attr_doc = json.load(f)
    ATTR = np.stack([attr_doc["vectors"][n] for n in STATION_ORDER]).astype(np.float32)
    attr_dim = ATTR.shape[1]
    A = torch.from_numpy(ATTR[sidx])          # per-window attribute rows
    print(f"windows {len(X)}; attribute dim {attr_dim}; "
          f"definition {attr_doc['attribute_vector_definition']}", flush=True)

    splits = {"train": end_year <= 2022, "val": end_year == 2023,
              "test": end_year >= 2024}
    for k, m in splits.items():
        print(f"{k}: {int(m.sum())} windows, {int(y[m].sum())} positives", flush=True)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    Xtr_t = torch.from_numpy(X[tr]); ytr_t = torch.from_numpy(y[tr]).float()
    Atr_t = A[tr]
    Xva_t = torch.from_numpy(X[va]); Ava_t = A[va]
    Xte_t = torch.from_numpy(X[te]); Ate_t = A[te]
    pos = float(y[tr].sum()); neg = float(tr.sum() - pos)
    pos_weight = torch.tensor(neg / max(1.0, pos))

    enc = SemanticEncoder()
    dec = GatewayDecoderV2(attr_dim)
    params = list(enc.parameters()) + list(dec.parameters())
    print(f"encoder params {sum(p.numel() for p in enc.parameters())}, "
          f"decoder v2 params {sum(p.numel() for p in dec.parameters())}", flush=True)
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse = nn.MSELoss()

    def predict(X_t, A_t, noisy=False):
        enc.eval(); dec.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(X_t), 4096):
                xb, ab = X_t[i:i + 4096], A_t[i:i + 4096]
                z = enc(xb)
                zq = torch.clamp(torch.round(z * 127.0) / 127.0, -1, 1)
                if noisy:
                    zq = zq + torch.randn_like(zq) * NOISE_SIGMA
                logit, _ = dec(zq, ab)
                probs.append(torch.sigmoid(logit).numpy())
        return np.concatenate(probs)

    best_f1, best_state, best_thr, bad = -1.0, None, 0.5, 0
    n = len(Xtr_t)
    for epoch in range(40):
        enc.train(); dec.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            xb, yb, ab = Xtr_t[idx], ytr_t[idx], Atr_t[idx]
            z = enc(xb)
            zq = quantise_ste(z) + torch.randn_like(z) * NOISE_SIGMA
            logit, recon = dec(zq, ab)
            loss = bce(logit, yb) + RECON_WEIGHT * mse(recon, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        pv = predict(Xva_t, Ava_t)
        thrs = np.linspace(0.05, 0.95, 19)
        f1s = [metrics(y[va], (pv >= t).astype(int))["f1"] for t in thrs]
        k = int(np.argmax(f1s))
        print(f"epoch {epoch}: loss {tot/n:.4f} val_f1 {f1s[k]:.4f} @thr {thrs[k]:.2f}",
              flush=True)
        if f1s[k] > best_f1:
            best_f1, best_thr, bad = f1s[k], float(thrs[k]), 0
            best_state = ({k2: v.clone() for k2, v in enc.state_dict().items()},
                          {k2: v.clone() for k2, v in dec.state_dict().items()})
        else:
            bad += 1
            if bad >= 8:
                print("early stop", flush=True)
                break

    enc.load_state_dict(best_state[0]); dec.load_state_dict(best_state[1])
    print(f"best val F1 {best_f1:.4f} @ threshold {best_thr}", flush=True)

    p_clean = predict(Xte_t, Ate_t, noisy=False)
    p_noisy = predict(Xte_t, Ate_t, noisy=True)
    yte, rte, ste = y[te], r[te], sidx[te]
    res = {"overall": {
        "nn_pair_v2": metrics(yte, (p_clean >= best_thr).astype(int)),
        "nn_pair_v2_noisy_latent": metrics(yte, (p_noisy >= best_thr).astype(int)),
        "rule_based": metrics(yte, rte)}}
    per_st = {}
    for si, name in enumerate(STATION_ORDER):
        m = ste == si
        if m.sum() == 0:
            continue
        per_st[name] = {
            "nn_pair_v2": metrics(yte[m], (p_clean[m] >= best_thr).astype(int)),
            "rule_based": metrics(yte[m], rte[m])}
    res["per_station"] = per_st
    enc.eval(); dec.eval()
    with torch.no_grad():
        zq = torch.clamp(torch.round(enc(Xte_t) * 127.0) / 127.0, -1, 1)
        _, recon = dec(zq, Ate_t)
        recon_mse = float(mse(recon, Xte_t))
    print(json.dumps(res, indent=2), flush=True)
    print(f"recon MSE (test): {recon_mse:.5f}", flush=True)

    # ---- v1 comparison (read from v1's own report; v1 files untouched) ----
    with open(os.path.join(HERE, "training_report_india.json")) as f:
        v1 = json.load(f)
    v1_overall = v1["test_metrics"]["overall"]
    v1_per = {k2: v2_["nn_pair"] for k2, v2_ in v1["test_metrics"]["per_station"].items()}
    comparison = {
        "overall": {
            "v2_nn": res["overall"]["nn_pair_v2"],
            "v1_nn": v1_overall["nn_pair"],
            "rule": v1_overall["rule_based"]},
        "per_station_f1_nn": {
            name: {"v2": per_st[name]["nn_pair_v2"]["f1"],
                   "v1": (v1_per[name]["f1"] if name in v1_per else None),
                   "v2_precision": per_st[name]["nn_pair_v2"]["precision"],
                   "v1_precision": (v1_per[name]["precision"] if name in v1_per else None),
                   "v2_recall": per_st[name]["nn_pair_v2"]["recall"],
                   "v1_recall": (v1_per[name]["recall"] if name in v1_per else None)}
            for name in per_st},
    }
    print(json.dumps(comparison, indent=2), flush=True)

    # ---- exports (v2 names only) ----
    enc.eval(); dec.eval()
    example = torch.zeros(1, INPUT_DIM)
    with torch.no_grad():
        _ = enc(example)
    traced_enc = torch.jit.trace(enc, example)
    enc_path = os.path.join(HERE, "semantic_encoder_india_v2_trained.pt")
    traced_enc.save(enc_path)

    class DecoderTaskOnlyV2(nn.Module):
        def __init__(self, d):
            super().__init__(); self.d = d
        def forward(self, z_attr):            # (batch, latent+attr)
            z, a = z_attr[:, :LATENT_DIM], z_attr[:, LATENT_DIM:]
            logit, _ = self.d(z, a)
            return logit.unsqueeze(-1)

    traced_dec = torch.jit.trace(
        DecoderTaskOnlyV2(dec), torch.zeros(1, LATENT_DIM + attr_dim))
    dec_path = os.path.join(HERE, "gateway_decoder_india_v2_trained.pt")
    traced_dec.save(dec_path)

    onnx_path = os.path.join(HERE, "semantic_encoder_india_v2_trained.onnx")
    torch.onnx.export(enc, example, onnx_path, opset_version=14,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
    import onnx
    onnx.checker.check_model(onnx.load(onnx_path))

    report = {
        "change_vs_v1": ("Gateway decoder conditioned on 6 static per-station "
                         "catchment attributes from INDOFLOODS metadata "
                         "(appended to the 8-dim latent at the decoder). "
                         "Encoder, windows, splits, seed and training "
                         "protocol identical to v1; windows loaded from v1's "
                         "dataset_india.npz."),
        "attributes": {"definition": attr_doc["attribute_vector_definition"],
                       "vectors_by_station": attr_doc["vectors"],
                       "source": attr_doc["source"]},
        "dataset_sizes": {k: {"n": int(m.sum()), "positives": int(y[m].sum())}
                          for k, m in splits.items()},
        "model": {"encoder": "SemanticEncoder MLP 64->48->24->8 tanh (unchanged)",
                  "decoder_v2": f"GatewayDecoderV2 ({LATENT_DIM}+{attr_dim})->24->48, "
                                "task head 48->1, recon head 48->64",
                  "decision_threshold_from_val": best_thr,
                  "best_val_f1": round(float(best_f1), 4)},
        "test_metrics_v2": res,
        "comparison_with_v1": comparison,
        "recon_mse_test": round(recon_mse, 5),
        "payload": ("unchanged: 8 int8 latent = 8 bytes over the air; the 6 "
                    "attributes are station metadata applied at the gateway"),
        "artifacts": {"encoder_traced": enc_path, "encoder_onnx": onnx_path,
                      "decoder_traced": dec_path},
        "train_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(HERE, "training_report_india_v2.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
