#!/usr/bin/env python3
"""Channel-SNR degradation sweep: learned pair vs hand-crafted encoder.

Model (one channel model for both methods): each method's gateway decision
is computed from values that crossed the link. AWGN is added to those
values, with SNR defined per message stream as
    SNR_dB = 10*log10( mean(v^2) / sigma^2 )
over the case's test set.
- Learned pair: v = the 8 int8-quantised latent values (the 8-byte message);
  noise added after quantisation, exactly as in training (which used
  sigma = 0.03); the gateway decoder decides from the noisy latent.
- Hand-crafted encoders: v = the report fields their rules threshold -
  soil: (moisture at window end, 4-day fall, 4-day rain); dust: AOD500 at
  the final bin; storm: (distance to land, closing, wind at window end);
  flood: the reported level series, from which the gateway latches
  exceedance (companion rule: causal smoothing 6, persist 2, hysteresis
  0.3 m, threshold = gazetted danger level, a noiseless gateway constant,
  like soil's site threshold).
Thresholds are the published ones (pair: validation-chosen; rules: the
training-chosen constants). Clean-end scores were verified in prep_sweep.py
to reproduce the published F1s exactly before this sweep was run.

Output: snr_results.json + snr_sweep.png (this folder).
"""
import sys, os, json
import numpy as np

RT = '/home/hatch/workspace/edge-apps-prep/real_tasks'
HERE = os.path.join(RT, 'snr_sweep')
sys.path.insert(0, '/home/hatch/workspace/edge-apps-prep/semantic_encoder')
sys.path.insert(0, '/home/hatch/workspace/edge-apps-prep/semantic_traces_src/src')
import torch
from semantic_encoder import SemanticEncoder
from train_semantic import GatewayDecoder, quantise_ste, metrics
from ntn_eval.stage3 import _smooth_causal

SNRS = [30, 25, 20, 15, 10, 5, 0]
TRIALS = 25

def f1(y, pred):
    return metrics(y, pred)['f1']

def sigma_for(values, snr_db):
    return float(np.sqrt(np.mean(np.asarray(values, float) ** 2) / 10 ** (snr_db / 10.0)))

# ---------------- learned pair (all cases) ----------------
def pair_latents(case):
    z = np.load(f'{RT}/{case}/test_windows.npz')
    X, y, thr = z['X'], z['y'], float(z['thr'])
    w = torch.load(f'{RT}/{case}/pair_weights.pt', weights_only=True)
    enc, dec = SemanticEncoder(), GatewayDecoder()
    enc.load_state_dict(w['enc']); dec.load_state_dict(w['dec'])
    enc.eval(); dec.eval()
    with torch.no_grad():
        zq = quantise_ste(enc(torch.tensor(X))).numpy()
        p_clean = torch.sigmoid(dec(torch.tensor(zq))[0]).numpy()
    return y, thr, zq, dec, f1(y, (p_clean >= thr).astype(int))

def pair_sweep(case, rng):
    y, thr, zq, dec, clean = pair_latents(case)
    train_snr = 10 * np.log10(np.mean(zq ** 2) / 0.03 ** 2)
    out = {}
    for s in SNRS:
        sg = sigma_for(zq, s)
        f1s = []
        for _ in range(TRIALS):
            zn = zq + sg * rng.standard_normal(zq.shape)
            with torch.no_grad():
                p = torch.sigmoid(dec(torch.tensor(zn, dtype=torch.float32))[0]).numpy()
            f1s.append(f1(y, (p >= thr).astype(int)))
        out[s] = (float(np.mean(f1s)), float(np.std(f1s)))
    return clean, float(train_snr), out

# ---------------- hand-crafted rules ----------------
def soil_sweep(rng):
    d = np.load(f'{HERE}/soil_fields.npz')
    sm, th, fall, rain, y = d['sm_end'], d['theta'], d['fall'], d['rain4'], d['y']
    clean = f1(y, ((sm < th + 0.01) & (fall > 0) & (rain < 2.0)).astype(int))
    out = {}
    for s in SNRS:
        f1s = []
        for _ in range(TRIALS):
            sm_n = sm + sigma_for(sm, s) * rng.standard_normal(sm.shape)
            fa_n = fall + sigma_for(fall, s) * rng.standard_normal(fall.shape)
            ra_n = rain + sigma_for(rain, s) * rng.standard_normal(rain.shape)
            f1s.append(f1(y, ((sm_n < th + 0.01) & (fa_n > 0) & (ra_n < 2.0)).astype(int)))
        out[s] = (float(np.mean(f1s)), float(np.std(f1s)))
    return clean, out

def dust_sweep(rng):
    z = np.load(f'{RT}/dust/test_windows.npz'); W = z['X'].reshape(-1, 16, 4)
    aod, y = 2.0 * W[:, 15, 0], z['y']
    clean = f1(y, (aod >= 0.6).astype(int))
    out = {}
    for s in SNRS:
        f1s = []
        for _ in range(TRIALS):
            a = aod + sigma_for(aod, s) * rng.standard_normal(aod.shape)
            f1s.append(f1(y, (a >= 0.6).astype(int)))
        out[s] = (float(np.mean(f1s)), float(np.std(f1s)))
    return clean, out

def storm_sweep(rng):
    z = np.load(f'{RT}/storm/test_windows.npz'); W = z['X'].reshape(-1, 16, 4)
    d2l, wind = W[:, 15, 2] * 1000.0, W[:, 15, 0] * 150.0
    closing, y = (W[:, 0, 2] - W[:, 15, 2]) * 1000.0, z['y']
    clean = f1(y, ((d2l < 300) & (closing > 0) & (wind >= 34)).astype(int))
    out = {}
    for s in SNRS:
        f1s = []
        for _ in range(TRIALS):
            d_n = d2l + sigma_for(d2l, s) * rng.standard_normal(d2l.shape)
            c_n = closing + sigma_for(closing, s) * rng.standard_normal(closing.shape)
            w_n = wind + sigma_for(wind, s) * rng.standard_normal(wind.shape)
            f1s.append(f1(y, ((d_n < 300) & (c_n > 0) & (w_n >= 34)).astype(int)))
        out[s] = (float(np.mean(f1s)), float(np.std(f1s)))
    return clean, out

# flood: vectorised twin of stage3.rule_above + tsi.rule_membership
def rule_above_vec(obs, thr, persist=2, hyst=0.3, smooth=6):
    obs = _smooth_causal(np.asarray(obs, float), smooth)
    n = len(obs)
    mem = np.zeros(n, bool)
    if n == 0:
        return mem
    above = obs > thr
    idx = np.arange(n)
    last_below = np.maximum.accumulate(np.where(~above, idx, -1))
    run = np.where(above, idx - last_below, 0)
    fire = run >= persist
    reset = obs < thr - hyst
    pos = 0
    while True:
        cand = np.where(fire[pos:])[0]
        if len(cand) == 0:
            break
        start = pos + int(cand[0])
        ends = np.where(reset[start + 1:])[0]
        if len(ends) == 0:
            mem[start:] = True
            break
        end = start + 1 + int(ends[0])
        mem[start:end + 1] = True
        pos = end + 1
    return mem

def membership_vec(level, valid, danger):
    mem = np.zeros(len(level), bool)
    i, n = 0, len(level)
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < n and valid[j]:
            j += 1
        if j - i >= 2:
            mem[i:j] = rule_above_vec(level[i:j], danger)
        i = j
    return mem

def flood_sweep(rng):
    d = np.load(f'{HERE}/flood_series.npz')
    dangers, ws, we = d['dangers'], d['win_station'], d['win_end']
    z = np.load(f'{RT}/flood/test_windows.npz'); y = z['y']
    series = [(d[f'level_{i}'].astype(float), d[f'valid_{i}']) for i in range(6)]
    # gate: vectorised rule must reproduce the published clean F1 exactly
    pred_clean = np.zeros(len(y), dtype=int)
    for si, (level, valid) in enumerate(series):
        mem = membership_vec(level, valid, dangers[si])
        sel = ws == si
        pred_clean[sel] = mem[we[sel]].astype(int)
    clean = f1(y, pred_clean)
    assert clean == 0.9111, f'flood vec-rule clean F1 {clean} != 0.9111'
    out = {}
    for s in SNRS:
        f1s = []
        for _ in range(TRIALS):
            preds = np.zeros(len(y), dtype=int)
            for si, (level, valid) in enumerate(series):
                lv = level.copy()
                # message = margin above the danger level (the datum-free
                # quantity the rule thresholds; the gauge altitude is not
                # part of a task report). Noise sigma from margin power.
                margin = level[valid] - dangers[si]
                lv[valid] = level[valid] + sigma_for(margin, s) * rng.standard_normal(int(valid.sum()))
                mem = membership_vec(lv, valid, dangers[si])
                sel = ws == si
                preds[sel] = mem[we[sel]].astype(int)
            f1s.append(f1(y, preds))
        out[s] = (float(np.mean(f1s)), float(np.std(f1s)))
    return clean, out

# ---------------- run ----------------
results = {}
for ci, case in enumerate(('soil', 'dust', 'storm', 'flood')):
    rng = np.random.default_rng(20261122 + ci)
    p_clean, train_snr, p_out = pair_sweep(case, rng)
    rng = np.random.default_rng(20261122 + 100 + ci)
    r_clean, r_out = {'soil': soil_sweep, 'dust': dust_sweep,
                      'storm': storm_sweep, 'flood': flood_sweep}[case](rng)
    results[case] = dict(pair_clean=p_clean, rule_clean=r_clean,
                         pair_train_snr=train_snr,
                         pair={str(s): p_out[s] for s in SNRS},
                         rule={str(s): r_out[s] for s in SNRS})
    print(f"{case}: clean pair {p_clean:.4f} / rule {r_clean:.4f} | pair training-noise ~ {train_snr:.1f} dB")
    for s in SNRS:
        print(f"   SNR {s:>2} dB: pair {p_out[s][0]:.3f} ± {p_out[s][1]:.3f} | rule {r_out[s][0]:.3f} ± {r_out[s][1]:.3f}")

json.dump(results, open(f'{HERE}/snr_results.json', 'w'), indent=1)

# ---------------- figure ----------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
titles = {'soil': 'Soil dry-down', 'dust': 'Desert dust', 'storm': 'Cyclone landfall', 'flood': 'Flood warning'}
for ax, case in zip(axes.flat, ('soil', 'dust', 'storm', 'flood')):
    r = results[case]
    xs = np.array(SNRS, float)
    pm = np.array([r['pair'][str(s)][0] for s in SNRS]); ps = np.array([r['pair'][str(s)][1] for s in SNRS])
    rm = np.array([r['rule'][str(s)][0] for s in SNRS]); rs = np.array([r['rule'][str(s)][1] for s in SNRS])
    ax.plot(xs, pm, 'o-', color='#1f6fb2', lw=1.4, ms=4, label='Learned pair')
    ax.fill_between(xs, pm - ps, pm + ps, color='#1f6fb2', alpha=0.18)
    ax.plot(xs, rm, 's--', color='#c0532b', lw=1.4, ms=4, label='Hand-crafted encoder')
    ax.fill_between(xs, rm - rs, rm + rs, color='#c0532b', alpha=0.18)
    ax.axhline(r['pair_clean'], color='#1f6fb2', lw=0.7, ls=':')
    ax.axhline(r['rule_clean'], color='#c0532b', lw=0.7, ls=':')
    ax.axvline(r['pair_train_snr'], color='grey', lw=0.8, ls='-.')
    ax.set_title(titles[case], fontsize=10)
    ax.set_xlabel('Channel SNR (dB)', fontsize=9)
    ax.set_ylabel('Test F1', fontsize=9)
    ax.set_xlim(32, -2); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)
axes[0, 0].legend(fontsize=8, loc='upper right')
fig.tight_layout()
fig.savefig(f'{HERE}/snr_sweep.png', dpi=200)
fig.savefig(f'{HERE}/snr_sweep.pdf')
print('saved snr_results.json + snr_sweep.png/.pdf')
