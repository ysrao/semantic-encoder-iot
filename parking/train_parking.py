"""Fourth simulated application for the task-oriented semantic
encoder/decoder paper: PARKING OCCUPANCY from a magnetometer.

Mirrors the flood-warning build (../semantic_encoder/train_semantic.py)
and the outage build (../outage_sim/train_outage.py) in architecture,
training recipe, rule baseline and scoring discipline:

  * SAME encoder (SemanticEncoder MLP 64 -> 48 -> 24 -> 8 tanh latent,
    unchanged) and SAME gateway decoder design, trained jointly.
  * SAME recipe: int8 quantisation in the loop (straight-through
    estimator), Gaussian latent noise sigma=0.03 in training,
    pos-weighted BCE + 0.05 * MSE(recon), Adam lr 1e-3, batch 256,
    early stop on validation F1 (patience 8, max 40 epochs), decision
    threshold swept on validation.
  * SAME rule-baseline family: the study's own stage3.rule_above
    (threshold + persistence + hysteresis + causal smoothing), imported
    as-is from ../semantic_traces_src — causal, same information.
  * Splits by trace seed, never random interleaving.

DATA: fully SYNTHETIC magnetometer traces generated here (no field
data). 1 Hz sampling, 21,600 samples (6 h) per trace, one parking bay.
  Channels per timestep (the encoder's 16x4 = 64-float window contract):
    ch0..2  field deviation vector dBx, dBy, dBz (uT) against the
            device's own slowly-tracked empty-bay baseline
    ch3     deviation magnitude |dB| (uT)
  Magnetometer parking sensors report against a tracked baseline
  (install-time calibration + slow gated adaptation); the deviation
  vector is the quantity such a device actually works with, and both
  the NN and the rule consume the identical deviation series.
  Baseline here: causal EMA, tau = 1800 s, updated only while the
  running deviation is < 1.0 uT (confident-empty gating, standard for
  this sensor class), so a parked vehicle does not drag the baseline.

  Trace construction: background field 42-58 uT (per-trace direction),
  slow drift (+/-0.4 uT sinusoids + clipped walk per axis), per-axis
  measurement noise sigma 0.15 uT; injected events —
    * own-bay occupancy: step change of the field vector, magnitude
      log-uniform 3-14 uT, direction in a cone around a canonical
      own-bay axis (defined relative to the background-field frame),
      duration log-uniform 10-240 min — GROUND TRUTH
    * neighbouring-bay parking: sustained SMALLER/offset distortion,
      1.5-5 uT, direction around a different canonical axis, duration
      10-180 min, Poisson(1.2)/trace — NOT occupancy of this bay
    * passing vehicle (adjacent lane): brief raised-cosine distortion,
      3-12 uT, own-bay-like direction, 2-8 s — NOT occupancy
    * transient magnetic noise: 1-3 s single-axis spikes 2-6 uT,
      Poisson(6)/trace — NOT occupancy
  GROUND TRUTH: occupancy state = 1 exactly on the injected own-bay
  intervals (generative truth). Window label = state at the window's
  final timestep (the flood/outage convention).

WINDOWS / REPORTING CADENCE: the device evaluates one 16-sample (16 s)
window per minute — window ends every 60 s. Bytes/day accounting uses
exactly this cadence: 1,440 windows/day.

TASK (paper framing): the device transmits either the raw window
(256 B), a hand-crafted rule report (20 B, study budget: event type,
onset, min/max deviation, duration, flags) at each rule episode edge,
or the learned 8-byte int8 latent per evaluated window; the gateway
decides "bay occupied" from what it receives.

One principled run. Hyperparameters are the flood recipe verbatim —
nothing is tuned against the test set or against the rule.
"""
import json
import math
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
CADENCE = 60              # one evaluated window per minute
TRACE_LEN = 21600         # 6 h at 1 Hz
RULE_THR = 2.0            # uT deviation-magnitude threshold for the rule
RULE_PERSIST = 15         # samples above threshold before the rule fires
RULE_HYST = 0.5           # uT
RULE_SMOOTH = 4           # causal moving-average length
BASE_TAU = 1800.0         # s, baseline EMA time constant
BASE_GATE = 1.0           # uT, baseline adapts only below this deviation
FEAT_SCALE = 0.1          # uT -> encoder input scale (10 uT = 1.0)
NOISE_SIGMA = 0.03
RECON_WEIGHT = 0.05

TRAIN_SEEDS = [111, 112, 113, 114, 115] + list(range(531, 556))   # 30 traces
VAL_SEEDS = list(range(611, 619))                                  # 8 traces
TEST_SEEDS = [211, 212, 213, 214, 215] + list(range(721, 736))    # 20 traces


# ---------------------------------------------------------------- generator

def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def gen_trace(seed):
    """One synthetic 6-hour magnetometer trace for a single bay.
    Returns the deviation channels, the ground-truth occupancy state,
    and the injected event log."""
    rng = np.random.default_rng(seed)
    n = TRACE_LEN
    t = np.arange(n, dtype=float)

    # background field: per-trace magnitude + direction, slow drift
    bmag = rng.uniform(42, 58)
    bdir = _unit(rng.normal(0, 1, 3))
    drift = np.zeros((n, 3))
    for ax in range(3):
        drift[:, ax] = (0.28 * np.sin(2 * np.pi * t / rng.uniform(3600, 7200)
                                      + rng.uniform(0, 2 * np.pi))
                        + 0.18 * np.sin(2 * np.pi * t / rng.uniform(1200, 3000)
                                        + rng.uniform(0, 2 * np.pi)))
        walk = np.cumsum(rng.normal(0, 0.004, n))
        drift[:, ax] += np.clip(walk, -0.8, 0.8)
    field = bmag * bdir[None, :] + drift

    # canonical signature axes in the background-field frame
    e1 = bdir
    arb = np.array([1.0, 0.0, 0.0]) if abs(e1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e2 = _unit(np.cross(e1, arb))
    e3 = np.cross(e1, e2)
    own_axis = _unit(0.30 * e1 + 0.90 * e2 + 0.30 * e3)
    neigh_axis = _unit(-0.25 * e1 + 0.30 * e2 + 0.90 * e3)

    def sig_dir(axis, jitter):
        return _unit(axis + jitter * rng.normal(0, 1, 3))

    gt = np.zeros(n, dtype=np.int64)
    events = []

    def add_step(start, dur, mag, direction, kind):
        s = int(start); d = int(round(dur)); e = min(n, s + d)
        if s >= n or e - s < 4:
            return None
        tt = np.arange(s, e, dtype=float)
        prof = np.minimum(np.clip((tt - s) / 2.0, 0, 1),
                          np.clip((e - tt) / 2.0, 0, 1))
        field[s:e] += (mag * direction)[None, :] * prof[:, None]
        events.append({"kind": kind, "start": s, "end": e - 1,
                       "mag": round(float(mag), 2)})
        return (s, e)

    # own-bay occupancy episodes (ground truth)
    cursor = int(rng.uniform(15 * 60, 75 * 60))
    while cursor < n - 600:
        dur = float(np.exp(rng.uniform(np.log(600), np.log(14400))))
        placed = add_step(cursor, dur, float(np.exp(rng.uniform(np.log(3),
                                                                np.log(14)))),
                          sig_dir(own_axis, 0.12), "occupancy")
        if placed is None:
            break
        gt[placed[0]:placed[1]] = 1
        cursor = placed[1] + int(rng.uniform(20 * 60, 90 * 60))

    # neighbouring-bay parking (not this bay's occupancy)
    for _ in range(rng.poisson(1.2)):
        add_step(rng.integers(60, n - 700),
                 float(np.exp(rng.uniform(np.log(600), np.log(10800)))),
                 rng.uniform(1.5, 5.0), sig_dir(neigh_axis, 0.15), "neighbour")

    # passing vehicles in the adjacent lane (not occupancy)
    for _ in range(rng.poisson(3.0)):
        s = int(rng.integers(30, n - 30)); d = float(rng.uniform(2, 8))
        e = min(n, int(s + d))
        tt = np.arange(s, e, dtype=float)
        prof = 0.5 - 0.5 * np.cos(2 * np.pi * (tt - s) / max(1.0, d))
        mag = rng.uniform(3, 12)
        field[s:e] += (mag * sig_dir(own_axis, 0.20))[None, :] * prof[:, None]
        events.append({"kind": "passing", "start": s, "end": e - 1,
                       "mag": round(float(mag), 2)})

    # transient magnetic noise spikes (not occupancy)
    for _ in range(rng.poisson(6.0)):
        s = int(rng.integers(30, n - 30)); d = int(rng.integers(1, 4))
        ax = int(rng.integers(0, 3))
        field[s:s + d, ax] += rng.choice([-1, 1]) * rng.uniform(2, 6)
        events.append({"kind": "transient", "start": s, "end": s + d - 1,
                       "mag": None})

    field += rng.normal(0, 0.15, (n, 3))

    # device baseline: causal gated EMA over the observed field
    alpha = 1.0 - math.exp(-1.0 / BASE_TAU)
    ema = field[0].copy()
    dev = np.empty((n, 3))
    for i in range(n):
        d = field[i] - ema
        dev[i] = d
        if math.sqrt(float(d @ d)) < BASE_GATE:
            ema = ema + alpha * d
    dev_mag = np.sqrt((dev ** 2).sum(axis=1))
    return {"dev": dev, "dev_mag": dev_mag, "gt": gt, "events": events}


# ---------------------------------------------------------------- data

def _episode_membership(n, episodes):
    m = np.zeros(n, dtype=bool)
    for s, e in episodes:
        m[s:e + 1] = True
    return m


def build_split(seeds):
    X, y, r, ends_all, trace_id = [], [], [], [], []
    ev_counts = {}
    ends_idx = np.arange(CADENCE - 1, TRACE_LEN, CADENCE)
    for k, seed in enumerate(seeds):
        tr = gen_trace(seed)
        for ev in tr["events"]:
            ev_counts[ev["kind"]] = ev_counts.get(ev["kind"], 0) + 1
        t = np.arange(TRACE_LEN, dtype=float)
        eps = stage3.rule_above(t, tr["dev_mag"], np.full(TRACE_LEN, RULE_THR),
                                RULE_PERSIST, RULE_HYST, RULE_SMOOTH)
        rule = _episode_membership(TRACE_LEN, eps)
        feats = np.concatenate([tr["dev"], tr["dev_mag"][:, None]],
                               axis=-1) * FEAT_SCALE      # (n, 4)
        for i in ends_idx:
            X.append(feats[i - WINDOW + 1:i + 1].reshape(-1))
            y.append(int(tr["gt"][i]))
            r.append(int(rule[i]))
            ends_all.append(int(i))
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
    """Episode-level scoring on the test traces: a ground-truth occupancy
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
                   if ev["kind"] == "occupancy"]
            n_eps += len(eps)
            m = trace_id == k
            e_k, p_k = ends[m], pred[m]
            order = np.argsort(e_k)
            e_k, p_k = e_k[order], p_k[order]
            for s, e in eps:
                hit = np.where((e_k >= s) & (e_k <= e) & (p_k == 1))[0]
                if len(hit):
                    detected += 1
                    delays.append(int(e_k[hit[0]] - s))
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
                     "arrivals_departures_total": int(2 * n_eps),
                     "false_alarm_episodes": int(fa),
                     "median_detection_delay_s": (float(np.median(delays))
                                                  if delays else None)}
    return out


# ---------------------------------------------------------------- train

def main():
    t0 = time.time()
    torch.manual_seed(20261121)
    np.random.seed(20261121)

    print("generating synthetic magnetometer traces...", flush=True)
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

    # ---- bytes/day from the simulated cadence + measured turnover ----
    test_episodes = cte.get("occupancy", 0)
    bay_days_test = len(TEST_SEEDS) * (TRACE_LEN / 86400.0)   # 6 h traces
    arrivals_per_day = test_episodes / bay_days_test
    edges_per_day = 2.0 * arrivals_per_day
    windows_per_day = 86400 // CADENCE
    bytes_day = {
        "cadence": "one 16-sample window evaluated per minute "
                   "(1,440 windows/day); raw and learned transmit per "
                   "evaluated window; rule transmits one 20 B report per "
                   "episode edge (arrival and departure)",
        "arrivals_per_day_measured_on_test_traces": round(arrivals_per_day, 2),
        "raw_window_bytes_per_day": int(windows_per_day * 256),
        "nn_latent_bytes_per_day": int(windows_per_day * 8),
        "rule_report_bytes_per_day": round(edges_per_day * 20, 1),
        "ratio_latent_vs_raw": 32.0,
    }
    print(json.dumps(bytes_day, indent=2), flush=True)

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
    enc_path = os.path.join(HERE, "semantic_encoder_parking_trained.pt")
    traced_enc.save(enc_path)

    class DecoderTaskOnly(nn.Module):
        def __init__(self, d):
            super().__init__(); self.d = d
        def forward(self, z):
            logit, _ = self.d(z)
            return logit.unsqueeze(-1)

    traced_dec = torch.jit.trace(DecoderTaskOnly(dec), torch.zeros(1, LATENT_DIM))
    dec_path = os.path.join(HERE, "gateway_decoder_parking_trained.pt")
    traced_dec.save(dec_path)

    onnx_path = os.path.join(HERE, "semantic_encoder_parking_trained.onnx")
    torch.onnx.export(enc, example, onnx_path, opset_version=14,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
    import onnx
    onnx.checker.check_model(onnx.load(onnx_path))

    np.savez_compressed(os.path.join(HERE, "dataset_parking.npz"),
                        Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva,
                        Xte=Xte, yte=yte, rte=rte)

    sizes = {p: os.path.getsize(p) for p in (enc_path, dec_path, onnx_path)}
    print(f"artifact sizes {sizes}", flush=True)

    report = {
        "application": "Parking occupancy detection from a bay magnetometer "
                       "(FOURTH simulated application; flood warning first, "
                       "outage detection second of the simulations)",
        "data_nature": "Fully SYNTHETIC traces generated by gen_trace() in "
                       "this script (seeded). No field data.",
        "generator": {
            "sampling": "1 Hz, 21,600 samples (6 h) per trace, single bay",
            "channels_per_timestep": ["deviation_dBx_uT", "deviation_dBy_uT",
                                      "deviation_dBz_uT", "deviation_mag_uT"],
            "channel_note": "deviation of the measured field vector from "
                            "the device's causal gated-EMA empty-bay "
                            "baseline (tau 1800 s, adapts only while "
                            "|dev| < 1.0 uT); features scaled x0.1 (10 uT "
                            "= 1.0) for the encoder; the rule consumes "
                            "the same deviation magnitude unscaled",
            "background": "field magnitude 42-58 uT, per-trace direction, "
                          "slow drift +/-0.4 uT (sinusoids + clipped walk), "
                          "noise sigma 0.15 uT/axis",
            "events": {
                "occupancy": "step 3-14 uT (log-uniform), own-bay axis cone, "
                             "10-240 min (log-uniform), gaps 20-90 min — "
                             "GROUND TRUTH",
                "neighbour": "sustained 1.5-5 uT, different axis, 10-180 min, "
                             "Poisson(1.2)/trace — NOT occupancy",
                "passing": "raised-cosine 3-12 uT, 2-8 s, own-bay-like "
                           "direction, Poisson(3)/trace — NOT occupancy",
                "transient": "single-axis spike 2-6 uT, 1-3 s, "
                             "Poisson(6)/trace — NOT occupancy"},
            "ground_truth": "occupancy state = 1 exactly on injected "
                            "own-bay intervals; window label = state at "
                            "final timestep",
            "train_event_counts": ctr, "val_event_counts": cva,
            "test_event_counts": cte},
        "splits_by_seed": {"train": TRAIN_SEEDS, "val": VAL_SEEDS,
                           "test": TEST_SEEDS},
        "windows": {"length": WINDOW, "cadence_s": CADENCE,
                    "sizes": {"train": {"n": int(len(ytr)), "positives": int(ytr.sum())},
                              "val": {"n": int(len(yva)), "positives": int(yva.sum())},
                              "test": {"n": int(len(yte)), "positives": int(yte.sum())}}},
        "model": {"encoder": "SemanticEncoder MLP 64->48->24->8 tanh "
                             "(architecture unchanged from the flood build)",
                  "encoder_params": enc_params, "decoder_params": dec_params,
                  "encoder_fp32_weight_bytes": fp32_bytes,
                  "encoder_int8_weight_bytes": int8_bytes,
                  "training": "joint end-to-end; int8 quantisation in the "
                              "loop (STE); latent noise sigma=0.03 in "
                              "training; pos-weighted BCE + 0.05*MSE(recon); "
                              "Adam lr 1e-3, batch 256, early stop on val F1 "
                              "(patience 8, max 40 epochs) — flood recipe "
                              "verbatim",
                  "decision_threshold_from_val": best_thr,
                  "best_val_f1": round(float(best_f1), 4)},
        "rule_baseline": {"rule": "study stage3.rule_above imported as-is, "
                                  "on the deviation magnitude series",
                          "params": {"threshold_uT": RULE_THR,
                                     "persist_samples": RULE_PERSIST,
                                     "hysteresis_uT": RULE_HYST,
                                     "causal_smooth": RULE_SMOOTH},
                          "report_bytes": 20,
                          "report_layout": "event type 1 B, onset timestamp "
                                           "4 B, min/max deviation 4 B, "
                                           "duration 2 B, flags/confidence + "
                                           "header/CRC 9 B (study's 20 B "
                                           "semantic-report budget)"},
        "test_metrics_window_level": res,
        "test_metrics_episode_level": eps_res,
        "recon_mse_test": round(recon_mse, 5),
        "payload_comparison": {
            "raw_window": "64 float32 values = 256 bytes",
            "rule_report": "20 bytes per episode edge",
            "nn_latent": "8 int8 values = 8 bytes per evaluated window",
            "ratio_latent_vs_raw_window": 32.0},
        "bytes_per_day_simulated": bytes_day,
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
