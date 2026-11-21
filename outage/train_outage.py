"""Second simulated application for the task-oriented semantic
encoder/decoder paper: POWER OUTAGE DETECTION from a voltage sensor.

Mirrors the flood-warning build (../semantic_encoder/train_semantic.py)
in architecture, training recipe, rule baseline and scoring discipline:

  * SAME encoder (SemanticEncoder MLP 64 -> 48 -> 24 -> 8 tanh latent,
    unchanged) and SAME gateway decoder design, trained jointly.
  * SAME recipe: int8 quantisation in the loop (straight-through
    estimator), Gaussian latent noise sigma=0.03 in training,
    pos-weighted BCE + 0.05 * MSE(recon), Adam lr 1e-3, batch 256,
    early stop on validation F1 (patience 8, max 40 epochs), decision
    threshold swept on validation.
  * SAME rule-baseline family: the study's own stage3.rule_below
    (threshold + persistence + hysteresis + causal smoothing), imported
    as-is from ../semantic_traces_src — causal, same information.
  * Splits by trace seed, never random interleaving.

DATA: fully SYNTHETIC feeder traces generated here (no field data).
  1 Hz RMS sampling, 3,600 samples (1 h) per trace. Channels per
  timestep (the encoder's 16x4 = 64-float window contract):
    ch0 RMS voltage, per-unit (nominal 1.0)
    ch1 outage threshold, constant 0.20 pu
    ch2 load current, per-unit
    ch3 grid frequency normalised as (f - 49.5), ~0.5 nominal
  Trace construction: slow voltage drift (+/-0.035 pu), load wander,
  frequency wander, measurement noise; injected events —
    * sags: residual 0.45-0.88 pu, 0.3-25 s (NOT outages)
    * occasional deep sags: residual 0.25-0.42 pu (NOT outages)
    * momentary dips: residual 0.0-0.15 pu, 1-2 s (NOT outages —
      transients; the duration requirement is what rejects them)
    * outages: residual 0.0-0.08 pu, 30-600 s, with a restoration ramp
      and sometimes a brief post-restoration inrush sag (NOT an outage)
  GROUND TRUTH: outage state = 1 exactly on the injected outage
  intervals (generative truth, not a threshold on the observed series).
  Window label = ground-truth state at the window's final timestep
  (the flood simulation run's convention).

TASK (paper framing): the device transmits either the raw window
(256 B), a hand-crafted rule report (20 B, study budget: event type,
onset, minimum RMS, duration, flags) at each rule episode start, or
the learned 8-byte int8 latent per window; the gateway decides
"outage active" from what it receives.

One principled run. Hyperparameters are the flood recipe verbatim —
nothing is tuned against the test set or against the rule.
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
SRC = os.path.join(BASE, "semantic_traces_src", "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(BASE, "semantic_encoder"))

from ntn_eval import stage3                              # noqa: E402
from semantic_encoder import SemanticEncoder, INPUT_DIM, LATENT_DIM  # noqa: E402

WINDOW = 16
STRIDE = 8
TRACE_LEN = 3600          # 1 h at 1 Hz
OUTAGE_THR = 0.20         # pu — interruption boundary used by the rule
RULE_PERSIST = 3          # samples below threshold before the rule fires
RULE_HYST = 0.05          # pu
RULE_SMOOTH = 4           # causal moving-average length
NOISE_SIGMA = 0.03
RECON_WEIGHT = 0.05

TRAIN_SEEDS = [101, 102, 103, 104, 105] + list(range(501, 526))   # 30 traces
VAL_SEEDS = list(range(601, 609))                                  # 8 traces
TEST_SEEDS = [201, 202, 203, 204, 205] + list(range(701, 716))    # 20 traces


# ---------------------------------------------------------------- generator

def gen_trace(seed):
    """One synthetic 1-hour feeder trace. Returns dict of channels,
    the ground-truth outage state, and the injected event log."""
    rng = np.random.default_rng(seed)
    n = TRACE_LEN
    t = np.arange(n, dtype=float)

    # slow drift: two sinusoids + clipped random walk
    ph1, ph2 = rng.uniform(0, 2 * np.pi, 2)
    drift = (0.018 * np.sin(2 * np.pi * t / rng.uniform(1400, 2200) + ph1)
             + 0.010 * np.sin(2 * np.pi * t / rng.uniform(500, 900) + ph2))
    walk = np.cumsum(rng.normal(0, 0.0006, n))
    walk = np.clip(walk, -0.012, 0.012)
    v_clean = 1.0 + drift + walk

    # load current (pu): slow wander 0.35-0.9
    load = 0.62 + 0.18 * np.sin(2 * np.pi * t / rng.uniform(1800, 3000)
                                 + rng.uniform(0, 2 * np.pi))
    load += np.cumsum(rng.normal(0, 0.0008, n))
    load = np.clip(load, 0.3, 0.95)

    # frequency (Hz): 50 +/- ~0.12 slow wander
    freq = 50.0 + 0.09 * np.sin(2 * np.pi * t / rng.uniform(900, 1600)
                                + rng.uniform(0, 2 * np.pi))
    freq += np.cumsum(rng.normal(0, 0.0012, n))
    freq = np.clip(freq, 49.7, 50.3)

    gt = np.zeros(n, dtype=np.int64)
    mult = np.ones(n)          # voltage multiplier from events
    events = []

    def apply_dip(start, dur, residual, kind):
        """Multiply voltage by a dip profile: 1-sample fall, flat
        bottom, linear recovery over 1-3 samples."""
        s = int(start); d = max(1, int(round(dur)))
        e = min(n, s + d)
        if s >= n:
            return
        prof = np.full(e - s, residual)
        rec = min(3, max(1, (e - s) // 4))
        if e - s > rec:
            prof[-rec:] = np.linspace(residual, 1.0, rec + 1)[1:]
        mult[s:e] = np.minimum(mult[s:e], prof)
        events.append({"kind": kind, "start": s, "end": e - 1,
                       "residual": round(float(residual), 3)})

    # sags (not outages)
    for _ in range(rng.poisson(2.5)):
        apply_dip(rng.integers(30, n - 60),
                  float(np.exp(rng.uniform(np.log(0.3), np.log(25)))),
                  rng.uniform(0.45, 0.88), "sag")
    # occasional deep sag (not an outage)
    if rng.random() < 0.35:
        apply_dip(rng.integers(30, n - 60),
                  float(np.exp(rng.uniform(np.log(1), np.log(30)))),
                  rng.uniform(0.25, 0.42), "deep_sag")
    # momentary dips to near zero, 1-2 s (not outages)
    for _ in range(rng.poisson(0.8)):
        apply_dip(rng.integers(30, n - 60), rng.integers(1, 3),
                  rng.uniform(0.0, 0.15), "momentary_dip")
    # outages (ground truth)
    n_out = rng.choice([0, 1, 2], p=[0.25, 0.45, 0.30])
    placed = []
    for _ in range(n_out):
        dur = int(rng.uniform(30, 600))
        for _try in range(20):
            s = int(rng.integers(60, n - dur - 30))
            if all(s > e + 60 or s + dur < b - 60 for b, e in placed):
                break
        else:
            continue
        placed.append((s, s + dur))
        residual = rng.uniform(0.0, 0.08)
        apply_dip(s, dur, residual, "outage")
        gt[s:s + dur] = 1
        # post-restoration inrush sag (not an outage)
        if rng.random() < 0.5:
            apply_dip(s + dur + rng.integers(2, 8),
                      rng.uniform(1, 3), rng.uniform(0.65, 0.8),
                      "inrush_sag")

    v_event = v_clean * mult
    # current: constant-power-ish under sags, collapses in outages
    i_clean = load * np.clip(0.55 + 0.45 * v_event, 0.0, 1.1)
    i_clean = np.where(gt == 1, load * rng.uniform(0.0, 0.03), i_clean)
    # frequency: small dip at outage spans (loss of local generation)
    f_event = freq - 0.06 * gt

    v_obs = v_event + rng.normal(0, 0.006, n)
    i_obs = np.clip(i_clean + rng.normal(0, 0.01, n), 0, None)
    f_obs = f_event + rng.normal(0, 0.008, n)
    return {"v": v_obs, "i": i_obs, "f": f_obs, "gt": gt, "events": events}


# ---------------------------------------------------------------- data

def _episode_membership(n, episodes):
    m = np.zeros(n, dtype=bool)
    for s, e in episodes:
        m[s:e + 1] = True
    return m


def build_split(seeds):
    X, y, r, ends_all, trace_id = [], [], [], [], []
    ev_counts = {}
    for k, seed in enumerate(seeds):
        tr = gen_trace(seed)
        for ev in tr["events"]:
            ev_counts[ev["kind"]] = ev_counts.get(ev["kind"], 0) + 1
        t = np.arange(TRACE_LEN, dtype=float)
        eps = stage3.rule_below(t, tr["v"], np.full(TRACE_LEN, OUTAGE_THR),
                                RULE_PERSIST, RULE_HYST, RULE_SMOOTH)
        rule = _episode_membership(TRACE_LEN, eps)
        feats = np.stack([tr["v"], np.full(TRACE_LEN, OUTAGE_THR),
                          tr["i"], tr["f"] - 49.5], axis=-1)  # (n,4)
        for i in range(WINDOW - 1, TRACE_LEN, STRIDE):
            X.append(feats[i - WINDOW + 1:i + 1].reshape(-1))
            y.append(int(tr["gt"][i]))
            r.append(int(rule[i]))
            ends_all.append(i)
            trace_id.append(k)
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64),
            np.asarray(r, dtype=np.int64), np.asarray(ends_all),
            np.asarray(trace_id), ev_counts)


# ---------------------------------------------------------------- model

class GatewayDecoder(nn.Module):
    """Gateway-side mirror decoder: 8 -> 24 -> 48 -> task head (+ recon).
    Identical to the flood build's decoder."""

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


def episode_scores(seeds, nn_pred, rule_pred, ends, trace_id):
    """Episode-level scoring on the test traces: a ground-truth outage
    episode is DETECTED if the system flags >= 1 window ending inside
    it; a false-alarm episode is a contiguous run of positive window
    decisions (per trace, in window order) overlapping no GT episode.
    Also median detection delay (onset -> first flagged window end)."""
    out = {}
    for name, pred in [("nn_pair", nn_pred), ("rule_based", rule_pred)]:
        detected, delays, fa = 0, [], 0
        n_eps = 0
        for k, seed in enumerate(seeds):
            tr = gen_trace(seed)
            eps = [(ev["start"], ev["end"]) for ev in tr["events"]
                   if ev["kind"] == "outage"]
            n_eps += len(eps)
            m = trace_id == k
            e_k, p_k = ends[m], pred[m]
            order = np.argsort(e_k)
            e_k, p_k = e_k[order], p_k[order]
            covered = np.zeros(len(e_k), dtype=bool)
            for s, e in eps:
                hit = np.where((e_k >= s) & (e_k <= e) & (p_k == 1))[0]
                if len(hit):
                    detected += 1
                    delays.append(int(e_k[hit[0]] - s))
                    covered |= (e_k >= s) & (e_k <= e)
            # false-alarm runs: positive windows outside every GT episode
            in_ep = np.zeros(len(e_k), dtype=bool)
            for s, e in eps:
                in_ep |= (e_k >= s) & (e_k <= e)
            run = False
            for j in range(len(e_k)):
                if p_k[j] == 1 and not in_ep[j]:
                    if not run:
                        fa += 1
                        run = True
                else:
                    run = False
        out[name] = {"gt_episodes": int(n_eps), "detected": int(detected),
                     "missed": int(n_eps - detected),
                     "false_alarm_episodes": int(fa),
                     "median_detection_delay_s": (float(np.median(delays))
                                                  if delays else None)}
    return out


# ---------------------------------------------------------------- train

def main():
    t0 = time.time()
    torch.manual_seed(20261121)
    np.random.seed(20261121)

    print("generating synthetic feeder traces...", flush=True)
    Xtr, ytr, _rtr, _etr, _ttr, ctr = build_split(TRAIN_SEEDS)
    Xva, yva, _rva, _eva, _tva, cva = build_split(VAL_SEEDS)
    Xte, yte, rte, ete, tte, cte = build_split(TEST_SEEDS)
    print(f"train {len(ytr)} (pos {ytr.sum()}), val {len(yva)} (pos {yva.sum()}), "
          f"test {len(yte)} (pos {yte.sum()})", flush=True)
    print(f"train events {ctr}", flush=True)
    print(f"test events {cte}", flush=True)

    Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr).float()
    Xva_t = torch.from_numpy(Xva)
    Xte_t = torch.from_numpy(Xte)
    pos = float(ytr.sum()); neg = float(len(ytr) - pos)
    pos_weight = torch.tensor(neg / max(1.0, pos))

    enc = SemanticEncoder()
    dec = GatewayDecoder()
    enc_params = sum(p.numel() for p in enc.parameters())
    dec_params = sum(p.numel() for p in dec.parameters())
    print(f"encoder params {enc_params}, decoder params {dec_params}", flush=True)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                           lr=1e-3, weight_decay=1e-4)
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
    nn_pred = (p_clean >= best_thr).astype(int)
    res = {
        "nn_pair": metrics(yte, nn_pred),
        "nn_pair_noisy_latent": metrics(yte, (p_noisy >= best_thr).astype(int)),
        "rule_based": metrics(yte, rte),
    }
    eps_res = episode_scores(TEST_SEEDS, nn_pred, rte.astype(int), ete, tte)
    enc.eval(); dec.eval()
    with torch.no_grad():
        zq = torch.clamp(torch.round(enc(Xte_t) * 127.0) / 127.0, -1, 1)
        _, recon = dec(zq)
        recon_mse = float(mse(recon, Xte_t))
    print(json.dumps(res, indent=2), flush=True)
    print(json.dumps(eps_res, indent=2), flush=True)
    print(f"recon MSE (test): {recon_mse:.5f}", flush=True)

    # ---- encoder size + CPU inference cost (measured on this VM) ----
    sd = enc.state_dict()
    fp32_bytes = sum(v.numel() * v.element_size() for v in sd.values())
    int8_bytes = sum(v.numel() for v in sd.values())
    enc.eval()
    x1 = torch.zeros(1, INPUT_DIM)
    with torch.no_grad():
        for _ in range(50):
            enc(x1)
        t1 = time.perf_counter()
        for _ in range(1000):
            enc(x1)
        single_ms = (time.perf_counter() - t1) / 1000 * 1000
        xb = torch.zeros(1024, INPUT_DIM)
        for _ in range(10):
            enc(xb)
        t1 = time.perf_counter()
        for _ in range(50):
            enc(xb)
        batch_us = (time.perf_counter() - t1) / 50 / 1024 * 1e6
    print(f"encoder fp32 weights {fp32_bytes} B, int8 weights {int8_bytes} B, "
          f"CPU single-window {single_ms:.3f} ms, batched {batch_us:.2f} us/window",
          flush=True)

    # ---- exports ----
    example = torch.zeros(1, INPUT_DIM)
    with torch.no_grad():
        _ = enc(example)
    traced_enc = torch.jit.trace(enc, example)
    enc_path = os.path.join(HERE, "semantic_encoder_outage_trained.pt")
    traced_enc.save(enc_path)

    class DecoderTaskOnly(nn.Module):
        def __init__(self, d):
            super().__init__(); self.d = d
        def forward(self, z):
            logit, _ = self.d(z)
            return logit.unsqueeze(-1)

    traced_dec = torch.jit.trace(DecoderTaskOnly(dec), torch.zeros(1, LATENT_DIM))
    dec_path = os.path.join(HERE, "gateway_decoder_outage_trained.pt")
    traced_dec.save(dec_path)

    onnx_path = os.path.join(HERE, "semantic_encoder_outage_trained.onnx")
    torch.onnx.export(enc, example, onnx_path, opset_version=14,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
    import onnx
    onnx.checker.check_model(onnx.load(onnx_path))

    np.savez_compressed(os.path.join(HERE, "dataset_outage.npz"),
                        Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva,
                        Xte=Xte, yte=yte, rte=rte)

    sizes = {p: os.path.getsize(p) for p in (enc_path, dec_path, onnx_path)}
    print(f"artifact sizes {sizes}", flush=True)

    report = {
        "application": "Power outage detection from a feeder voltage sensor "
                       "(SECOND simulated application; flood warning was the first)",
        "data_nature": "Fully SYNTHETIC traces generated by gen_trace() in this "
                       "script (seeded). No field data.",
        "generator": {
            "sampling": "1 Hz RMS, 3,600 samples (1 h) per trace",
            "channels_per_timestep": ["voltage_pu", "threshold_const_0.20",
                                      "current_pu", "freq_norm_(f-49.5)"],
            "background": "voltage drift +/-0.035 pu (slow sinusoids + clipped "
                          "random walk), load wander 0.3-0.95 pu, frequency "
                          "wander 49.7-50.3 Hz, noise sigma V 0.006 / I 0.01 / f 0.008",
            "events": {
                "sag": "residual 0.45-0.88 pu, 0.3-25 s, Poisson(2.5)/trace — NOT outages",
                "deep_sag": "residual 0.25-0.42 pu, 1-30 s, 35% of traces — NOT outages",
                "momentary_dip": "residual 0.0-0.15 pu, 1-2 s, Poisson(0.8)/trace — NOT outages",
                "outage": "residual 0.0-0.08 pu, 30-600 s, 0/1/2 per trace "
                          "(p 0.25/0.45/0.30); 50% followed by a 1-3 s inrush "
                          "sag to 0.65-0.8 pu at restoration (NOT an outage)"},
            "ground_truth": "outage state = 1 exactly on injected outage "
                            "intervals; window label = state at final timestep",
            "train_event_counts": ctr, "val_event_counts": cva,
            "test_event_counts": cte},
        "splits_by_seed": {"train": TRAIN_SEEDS, "val": VAL_SEEDS,
                           "test": TEST_SEEDS},
        "windows": {"length": WINDOW, "stride": STRIDE,
                    "sizes": {"train": {"n": int(len(ytr)), "positives": int(ytr.sum())},
                              "val": {"n": int(len(yva)), "positives": int(yva.sum())},
                              "test": {"n": int(len(yte)), "positives": int(yte.sum())}}},
        "model": {"encoder": "SemanticEncoder MLP 64->48->24->8 tanh "
                             "(architecture unchanged from the flood build)",
                  "encoder_params": enc_params, "decoder_params": dec_params,
                  "encoder_fp32_weight_bytes": fp32_bytes,
                  "encoder_int8_weight_bytes": int8_bytes,
                  "training": "joint end-to-end; int8 quantisation in the loop "
                              "(STE); latent noise sigma=0.03 in training; "
                              "pos-weighted BCE + 0.05*MSE(recon); Adam lr 1e-3, "
                              "batch 256, early stop on val F1 (patience 8, "
                              "max 40 epochs) — flood recipe verbatim",
                  "decision_threshold_from_val": best_thr,
                  "best_val_f1": round(float(best_f1), 4)},
        "rule_baseline": {"rule": "study stage3.rule_below imported as-is",
                          "params": {"threshold_pu": OUTAGE_THR,
                                     "persist_samples": RULE_PERSIST,
                                     "hysteresis_pu": RULE_HYST,
                                     "causal_smooth": RULE_SMOOTH},
                          "report_bytes": 20,
                          "report_layout": "event type 1 B, onset timestamp "
                                           "4 B, min RMS 2 B, duration 2 B, "
                                           "flags/confidence + header/CRC 11 B "
                                           "(study's 20 B semantic-report budget)"},
        "test_metrics_window_level": res,
        "test_metrics_episode_level": eps_res,
        "recon_mse_test": round(recon_mse, 5),
        "payload_comparison": {
            "raw_window": "64 float32 values = 256 bytes",
            "rule_report": "20 bytes per episode start",
            "nn_latent": "8 int8 values = 8 bytes per window",
            "ratio_latent_vs_raw_window": 32.0},
        "encoder_cpu_inference_this_vm": {
            "single_window_ms": round(single_ms, 3),
            "batched_us_per_window": round(batch_us, 2)},
        "artifacts": {"encoder_traced": enc_path, "encoder_onnx": onnx_path,
                      "decoder_traced": dec_path, "artifact_sizes": sizes},
        "train_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(HERE, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
