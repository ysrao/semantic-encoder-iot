"""Train the semantic encoder + gateway decoder on the semantic-mIoT-NTN
study traces.

DATA ROUTE (state in report): the study repo's OWN generators are used
as-is — `ntn_eval.traces.gen_agri_trace` / `gen_env_trace` with the repo's
`configs/agri.yaml` / `configs/env.yaml`. No reimplemented generator.
(The only environment change was installing a numpy-2-compatible pandas
into ~/v2env; the study code itself is untouched.)

SPLITS (by seed, following the study's own discipline — tuning seeds
101-105, held-out evaluation seeds 201-205, never random interleaving):
  train: 101-105 + 301-308      val: 401-402      test: 201-205

WINDOWS: causal, length 16, stride 8, ending at index i. Input channels
follow semantic_encoder.py's contract ([primary, threshold, rain, temp]
per timestep, row-major):
  agri: primary = sm_obs / field_capacity(0.38), thr = sm_thr / 0.38,
        rain = clip(rain_mm_h / 20, 0, 3), temp = temp_obs / 50
  env:  primary = level_obs / danger(4.5), thr = 1.0,
        rain = clip(rain_obs / 20, 0, 3), temp = temp_C / 50

LABEL: ground-truth task event at the window's final timestep
(`irrig_need` for agri, `flood` for env) — the study scores all task
outcomes against ground truth (traces.py docstring).

RULE BASELINE on identical windows: prediction at the same final timestep
= membership in a stage3.rule_below / rule_above episode computed on the
full trace with the study's task parameters (agri persist=2, hyst=0.02,
smooth=8; env persist=2, hyst=0.3, smooth=6). The rule is causal (causal
smoothing), so this is a fair same-information baseline.

TRAINING: encoder (SemanticEncoder, unchanged architecture) + gateway
decoder trained JOINTLY end-to-end. Latent is int8-quantised IN THE LOOP
with a straight-through estimator (forward: round to 1/127 grid; backward:
identity), plus Gaussian channel noise sigma=0.03 on the quantised latent
during training. Loss = class-weighted BCE (task) + 0.05 * MSE (optional
reconstruction head). Decision threshold selected on validation F1.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
SRC = os.path.join(BASE, "companion_traces", "src")      # the companion study's ntn_eval package
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)

from ntn_eval.config import load as load_cfg          # noqa: E402
from ntn_eval.traces import gen_agri_trace, gen_env_trace  # noqa: E402
from ntn_eval import stage3                            # noqa: E402
from semantic_encoder import SemanticEncoder, INPUT_DIM, LATENT_DIM  # noqa: E402

WINDOW = 16
STRIDE = 8
TRAIN_SEEDS = [101, 102, 103, 104, 105] + list(range(301, 309))
VAL_SEEDS = [401, 402]
TEST_SEEDS = [201, 202, 203, 204, 205]
NOISE_SIGMA = 0.03
RECON_WEIGHT = 0.05


# ---------------------------------------------------------------- data

def _windows(primary, thr, rain, temp, gt, rule_pred):
    """Causal windows ending at i; label/rule = value at final timestep."""
    n = len(primary)
    feats = np.stack([primary, thr, rain, temp], axis=-1)  # (n,4)
    X, y, r, ends = [], [], [], []
    for i in range(WINDOW - 1, n, STRIDE):
        X.append(feats[i - WINDOW + 1:i + 1].reshape(-1))
        y.append(int(gt[i]))
        r.append(int(rule_pred[i]))
        ends.append(i)
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64),
            np.asarray(r, dtype=np.int64), np.asarray(ends))


def _episode_membership(n, episodes):
    m = np.zeros(n, dtype=bool)
    for s, e in episodes:
        m[s:e + 1] = True
    return m


def build_split(seeds, agri_cfg, env_cfg):
    Xa, ya, ra, ta = [], [], [], []   # ta: 0=agri, 1=env
    counts = {}
    for seed in seeds:
        # --- agri ---
        df, _ev = gen_agri_trace(seed, agri_cfg)
        fc = agri_cfg["trace"]["field_capacity"]
        t = df["t_h"].values
        eps = stage3.rule_below(t, df["sm_obs"].values, df["sm_thr"].values,
                                agri_cfg["task"]["persist_confirm"],
                                agri_cfg["task"]["hysteresis"],
                                agri_cfg["task"]["smooth"])
        rule = _episode_membership(len(df), eps)
        X, y, r, _ = _windows(df["sm_obs"].values / fc,
                              df["sm_thr"].values / fc,
                              np.clip(df["rain_mm_h"].values / 20.0, 0, 3),
                              df["temp_obs"].values / 50.0,
                              df["irrig_need"].values, rule)
        Xa.append(X); ya.append(y); ra.append(r); ta.append(np.zeros(len(y)))
        counts[f"agri_{seed}"] = {"windows": int(len(y)), "pos": int(y.sum())}
        # --- env ---
        df2, _ev2, _b = gen_env_trace(seed, env_cfg)
        danger = env_cfg["trace"]["danger_level_m"]
        t2 = df2["t_h"].values
        eps2 = stage3.rule_above(t2, df2["level_obs"].values, danger,
                                 env_cfg["task"]["persist_confirm"],
                                 env_cfg["task"]["hysteresis"],
                                 env_cfg["task"]["smooth"])
        rule2 = _episode_membership(len(df2), eps2)
        X2, y2, r2, _ = _windows(df2["level_obs"].values / danger,
                                 np.ones(len(df2)),
                                 np.clip(df2["rain_obs"].values / 20.0, 0, 3),
                                 df2["temp_C"].values / 50.0,
                                 df2["flood"].values, rule2)
        Xa.append(X2); ya.append(y2); ra.append(r2); ta.append(np.ones(len(y2)))
        counts[f"env_{seed}"] = {"windows": int(len(y2)), "pos": int(y2.sum())}
    return (np.concatenate(Xa), np.concatenate(ya), np.concatenate(ra),
            np.concatenate(ta), counts)


# ---------------------------------------------------------------- model

class GatewayDecoder(nn.Module):
    """Gateway-side mirror decoder: 8 -> 24 -> 48 -> task head (+ recon)."""

    def __init__(self, latent_dim: int = LATENT_DIM, input_dim: int = INPUT_DIM):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(latent_dim, 24), nn.ReLU(),
            nn.Linear(24, 48), nn.ReLU(),
        )
        self.task_head = nn.Linear(48, 1)
        self.recon_head = nn.Linear(48, input_dim)

    def forward(self, z):
        h = self.body(z)
        return self.task_head(h).squeeze(-1), self.recon_head(h)


def quantise_ste(z):
    """int8 quantisation on the tanh range with straight-through gradient."""
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
    return {"accuracy": round(float(acc), 4), "precision": round(float(prec), 4),
            "recall": round(float(rec), 4), "f1": round(float(f1), 4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": int(len(y_true)),
            "positives": int(y_true.sum())}


# ---------------------------------------------------------------- train

def main():
    t0 = time.time()
    torch.manual_seed(20261118)
    np.random.seed(20261118)
    agri_cfg = load_cfg("agri")
    env_cfg = load_cfg("env")

    print("building datasets from study generators...", flush=True)
    Xtr, ytr, _rtr, ttr, ctr = build_split(TRAIN_SEEDS, agri_cfg, env_cfg)
    Xva, yva, rva, tva, cva = build_split(VAL_SEEDS, agri_cfg, env_cfg)
    Xte, yte, rte, tte, cte = build_split(TEST_SEEDS, agri_cfg, env_cfg)
    # saved for extras/robustness.py (identical windows)
    np.savez_compressed(os.path.join(HERE, "dataset_flood_syn.npz"), Xtr=Xtr, ytr=ytr,
                        Xva=Xva, yva=yva, Xte=Xte, yte=yte, rte=rte)
    print(f"train {len(ytr)} (pos {ytr.sum()}), val {len(yva)} (pos {yva.sum()}), "
          f"test {len(yte)} (pos {yte.sum()})", flush=True)

    Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr).float()
    Xva_t = torch.from_numpy(Xva)
    Xte_t = torch.from_numpy(Xte)
    pos = float(ytr.sum()); neg = float(len(ytr) - pos)
    pos_weight = torch.tensor(neg / max(1.0, pos))

    enc = SemanticEncoder()
    dec = GatewayDecoder()
    params = list(enc.parameters()) + list(dec.parameters())
    print(f"encoder params {sum(p.numel() for p in enc.parameters())}, "
          f"decoder params {sum(p.numel() for p in dec.parameters())}", flush=True)
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse = nn.MSELoss()

    def predict(X_t, noisy=False):
        enc.eval(); dec.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(X_t), 4096):
                xb = X_t[i:i + 4096]
                z = enc(xb)
                zq = torch.clamp(torch.round(z * 127.0) / 127.0, -1, 1)
                if noisy:
                    zq = zq + torch.randn_like(zq) * NOISE_SIGMA
                logit, _ = dec(zq)
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
            xb, yb = Xtr_t[idx], ytr_t[idx]
            z = enc(xb)
            zq = quantise_ste(z) + torch.randn_like(z) * NOISE_SIGMA
            logit, recon = dec(zq)
            loss = bce(logit, yb) + RECON_WEIGHT * mse(recon, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        pv = predict(Xva_t)
        # threshold sweep on val
        thrs = np.linspace(0.05, 0.95, 19)
        f1s = [metrics(yva, (pv >= t).astype(int))["f1"] for t in thrs]
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

    # ---- test evaluation: NN pair vs rule, identical windows ----
    p_clean = predict(Xte_t, noisy=False)
    p_noisy = predict(Xte_t, noisy=True)
    res = {}
    for name, tt in [("combined", None), ("agri", 0), ("env", 1)]:
        m = np.ones(len(yte), bool) if tt is None else (tte == tt)
        res[name] = {
            "nn_pair": metrics(yte[m], (p_clean[m] >= best_thr).astype(int)),
            "nn_pair_noisy_latent": metrics(yte[m], (p_noisy[m] >= best_thr).astype(int)),
            "rule_based": metrics(yte[m], rte[m]),
        }
    # reconstruction quality of the trained pair (secondary)
    enc.eval(); dec.eval()
    with torch.no_grad():
        zq = torch.clamp(torch.round(enc(Xte_t) * 127.0) / 127.0, -1, 1)
        _, recon = dec(zq)
        recon_mse = float(mse(recon, Xte_t))
    for name, r in res.items():
        print(name, json.dumps(r), flush=True)
    print(f"recon MSE (test, normalised inputs): {recon_mse:.5f}", flush=True)

    # ---- exports ----
    enc.eval(); dec.eval()
    example = torch.zeros(1, INPUT_DIM)
    with torch.no_grad():
        _ = enc(example)  # warm
    traced_enc = torch.jit.trace(enc, example)
    enc_path = os.path.join(HERE, "semantic_encoder_trained.pt")
    traced_enc.save(enc_path)
    dec_example = torch.zeros(1, LATENT_DIM)

    class DecoderTaskOnly(nn.Module):
        def __init__(self, d):
            super().__init__(); self.d = d
        def forward(self, z):
            logit, _ = self.d(z)
            return logit.unsqueeze(-1)  # (batch,1) task logit

    traced_dec = torch.jit.trace(DecoderTaskOnly(dec), dec_example)
    dec_path = os.path.join(HERE, "gateway_decoder_trained.pt")
    traced_dec.save(dec_path)

    onnx_path = os.path.join(HERE, "semantic_encoder_trained.onnx")
    torch.onnx.export(enc, example, onnx_path, opset_version=14,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
    import onnx
    onnx.checker.check_model(onnx.load(onnx_path))

    report = {
        "data_route": (
            "Study repo's own trace generators used as-is "
            "(ntn_eval.traces.gen_agri_trace / gen_env_trace, repo configs). "
            "Cloned from https://github.com/ysrao/semantic-miot-ntn into "
            "companion_traces/. Only environment change: pandas 2.2.3 "
            "installed into ~/v2env (shipped pandas was binary-incompatible "
            "with venv numpy 2.5.3); study code untouched."),
        "data_nature": "Seeded SYNTHETIC simulation traces from the study — not field data.",
        "splits_by_seed": {"train": TRAIN_SEEDS, "val": VAL_SEEDS, "test": TEST_SEEDS},
        "windows": {"length": WINDOW, "stride": STRIDE,
                    "label": "ground-truth task event at final timestep (irrig_need / flood)",
                    "normalisation": {
                        "agri": "primary=sm_obs/0.38, thr=sm_thr/0.38, rain=clip(rain_mm_h/20,0,3), temp=temp_obs/50",
                        "env": "primary=level_obs/4.5, thr=1.0, rain=clip(rain_obs/20,0,3), temp=temp_C/50"}},
        "dataset_sizes": {"train": {"n": int(len(ytr)), "positives": int(ytr.sum())},
                          "val": {"n": int(len(yva)), "positives": int(yva.sum())},
                          "test": {"n": int(len(yte)), "positives": int(yte.sum())},
                          "per_seed_counts_train": ctr, "per_seed_counts_test": cte},
        "model": {"encoder": "SemanticEncoder MLP 64->48->24->8 tanh (4,496 params, architecture unchanged)",
                  "decoder": "GatewayDecoder 8->24->48, task head 48->1, recon head 48->64",
                  "training": "joint end-to-end; int8 quantisation IN THE LOOP via straight-through estimator (forward round to 1/127 grid, backward identity); Gaussian channel noise sigma=0.03 on quantised latent during training; loss = pos-weighted BCE + 0.05*MSE(recon); Adam lr 1e-3, batch 256, early stop on val F1 (patience 8, max 40 epochs)",
                  "decision_threshold_from_val": best_thr,
                  "best_val_f1": round(float(best_f1), 4)},
        "test_metrics": res,
        "recon_mse_test": round(recon_mse, 5),
        "payload_comparison": {
            "raw_window": "64 float32 values = 256 bytes (study raw report payload: 200 B)",
            "nn_latent": "8 int8 values = 8 bytes",
            "study_semantic_report": "20 bytes (task output + confidence + header)",
            "ratio_vs_raw_window_bytes": 32.0},
        "artifacts": {"encoder_traced": enc_path, "encoder_onnx": onnx_path,
                      "decoder_traced": dec_path},
        "train_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(HERE, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"test_metrics": res, "thr": best_thr}, indent=2), flush=True)
    print(f"DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
