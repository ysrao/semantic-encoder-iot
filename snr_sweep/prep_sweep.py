#!/usr/bin/env python3
"""Prep + clean-end verification for the SNR sweep.

Gate: before any noise is added, BOTH methods must reproduce the published
test scores exactly from the artefacts the sweep will use.
- pair: pair_weights.pt + test_windows.npz must reproduce results.json F1.
- dust/storm rules: fields read back out of the saved X must reproduce F1.
- soil rule: fields recovered by re-running build_soil.py's data build
  (Open-Meteo re-pull); X must match the saved test X exactly.
- flood rule: station series rebuilt with the india module's own
  load_station; rule_membership at each test window's end_idx must equal
  the saved 'rule' column.
Outputs (this folder): soil_fields.npz, flood_series.npz
"""
import sys, os, json
import numpy as np

RT = '/home/hatch/workspace/edge-apps-prep/real_tasks'
OUT = os.path.join(RT, 'snr_sweep')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, '/home/hatch/workspace/edge-apps-prep/semantic_encoder')
import torch
from semantic_encoder import SemanticEncoder
from train_semantic import GatewayDecoder, quantise_ste, metrics

def f1_counts(y, pred):
    m = metrics(y, pred)
    return m['f1'], m['tp'], m['fp'], m['fn']

ok = True

# ---------- pair clean reproduction, all cases ----------
print('== pair clean reproduction ==')
for case in ('soil', 'dust', 'storm', 'flood'):
    z = np.load(f'{RT}/{case}/test_windows.npz')
    X, y, p_saved, thr = z['X'], z['y'], z['p'], float(z['thr'])
    w = torch.load(f'{RT}/{case}/pair_weights.pt', weights_only=True)
    enc, dec = SemanticEncoder(), GatewayDecoder()
    enc.load_state_dict(w['enc']); dec.load_state_dict(w['dec'])
    enc.eval(); dec.eval()
    with torch.no_grad():
        p = torch.sigmoid(dec(quantise_ste(enc(torch.tensor(X))))[0]).numpy()
    res = json.load(open(f'{RT}/{case}/results.json'))
    f1, tp, fp, fn = f1_counts(y, (p >= thr).astype(int))
    pub = res['pair_test']
    maxdiff = float(np.abs(p - p_saved).max())
    match = (maxdiff < 1e-5 and f1 == pub['f1']
             and tp == pub['tp'] and fp == pub['fp'] and fn == pub['fn'])
    ok &= match
    print(f"{case}: max|p - saved p| = {maxdiff:.2e} | "
          f"F1 {f1} vs published {pub['f1']} | counts {tp}/{fp}/{fn} vs "
          f"{pub['tp']}/{pub['fp']}/{pub['fn']} -> {'OK' if match else 'MISMATCH'}")

# ---------- dust + storm rules from X ----------
print('== dust/storm rules from saved X ==')
z = np.load(f'{RT}/dust/test_windows.npz'); W = z['X'].reshape(-1, 16, 4)
pred = (2.0 * W[:, 15, 0] >= 0.6).astype(int)
f1, tp, fp, fn = f1_counts(z['y'], pred)
match = (f1, tp, fp, fn) == (0.4333, 26, 68, 0); ok &= match
print(f'dust rule from X: F1 {f1} counts {tp}/{fp}/{fn} vs 0.4333 26/68/0 -> {"OK" if match else "MISMATCH"}')

z = np.load(f'{RT}/storm/test_windows.npz'); W = z['X'].reshape(-1, 16, 4)
d2l = W[:, 15, 2] * 1000.0; wind = W[:, 15, 0] * 150.0
closing = (W[:, 0, 2] - W[:, 15, 2]) * 1000.0
pred = ((d2l < 300) & (closing > 0) & (wind >= 34)).astype(int)
f1, tp, fp, fn = f1_counts(z['y'], pred)
match = (f1, tp, fp, fn) == (0.6494, 88, 79, 16); ok &= match
print(f'storm rule from X: F1 {f1} counts {tp}/{fp}/{fn} vs 0.6494 88/79/16 -> {"OK" if match else "MISMATCH"}')

# ---------- soil fields recovery (re-pull) ----------
print('== soil fields recovery ==')
import urllib.request, pandas as pd
SITES = {'Indore': (22.72, 75.86), 'Ludhiana': (30.90, 75.86), 'Nagpur': (21.15, 79.09)}
def pull(lat, lon):
    url = ("https://archive-api.open-meteo.com/v1/archive?latitude=%s&longitude=%s"
           "&start_date=2019-01-01&end_date=2024-12-31&hourly=soil_moisture_0_to_7cm,"
           "soil_moisture_7_to_28cm,soil_moisture_28_to_100cm,precipitation"
           "&timezone=Asia%%2FKolkata" % (lat, lon))
    with urllib.request.urlopen(url, timeout=120) as r:
        d = json.load(r)
    h = d['hourly']
    df = pd.DataFrame({'ts': pd.to_datetime(h['time']),
                       'sm1': h['soil_moisture_0_to_7cm'], 'sm2': h['soil_moisture_7_to_28cm'],
                       'sm3': h['soil_moisture_28_to_100cm'], 'rain': h['precipitation']})
    day = df.set_index('ts').resample('1D').agg({'sm1': 'mean', 'sm2': 'mean', 'sm3': 'mean', 'rain': 'sum'})
    return day.dropna()
rows = []
for name, (lat, lon) in SITES.items():
    day = pull(lat, lon)
    theta = float(day.loc[day.index.year <= 2022, 'sm1'].quantile(0.20))
    sm1 = day['sm1'].values
    ch = np.stack([sm1, day['sm2'].values, day['sm3'].values,
                   np.clip(day['rain'].values / 20.0, 0, 1.5)], axis=1)
    for end in range(15, len(day) - 2, 2):
        if sm1[end] < theta:
            continue
        label = 1 if min(sm1[end + 1], sm1[end + 2]) < theta else 0
        rows.append(dict(year=int(day.index[end].year),
                         X=ch[end-15:end+1].astype(np.float32), y=label,
                         sm_end=float(sm1[end]), theta=theta,
                         fall=float(sm1[end-3] - sm1[end]),
                         rain4=float(day['rain'].values[end-3:end+1].sum())))
te = [r for r in rows if r['year'] == 2024]
Xte = np.stack([r['X'] for r in te]).reshape(len(te), -1)
yte = np.array([r['y'] for r in te])
z = np.load(f'{RT}/soil/test_windows.npz')
xmatch = Xte.shape == z['X'].shape and np.allclose(Xte, z['X'], atol=1e-6) and (yte == z['y']).all()
pred = np.array([1 if (r['sm_end'] < r['theta'] + 0.01 and r['fall'] > 0 and r['rain4'] < 2.0) else 0 for r in te])
f1, tp, fp, fn = f1_counts(yte, pred)
match = xmatch and (f1, tp, fp, fn) == (0.3846, 10, 23, 9); ok &= match
print(f'soil: recovered {len(te)} test rows, X/y match saved: {xmatch} | rule F1 {f1} counts {tp}/{fp}/{fn} '
      f'vs 0.3846 10/23/9 -> {"OK" if match else "MISMATCH"}')
np.savez(f'{OUT}/soil_fields.npz',
         sm_end=np.array([r['sm_end'] for r in te]), theta=np.array([r['theta'] for r in te]),
         fall=np.array([r['fall'] for r in te]), rain4=np.array([r['rain4'] for r in te]), y=yte)

# ---------- flood series rebuild ----------
print('== flood series rebuild ==')
sys.path.insert(0, '/home/hatch/workspace/edge-apps-prep/semantic_encoder/india')
import train_semantic_india as tsi
d = np.load('/home/hatch/workspace/edge-apps-prep/semantic_encoder/india/dataset_india.npz')
Xd, yd, ruled, yrd, std = d['X'], d['y'], d['rule'], d['end_year'], d['station']
test_mask = yrd == 2024
zf = np.load(f'{RT}/flood/test_windows.npz')
print('flood test X matches dataset test rows:', np.array_equal(Xd[test_mask], zf['X']),
      '| y matches:', np.array_equal(yd[test_mask], zf['y']))
levels, valids, dangers, win_station, win_end = [], [], [], [], []
rebuilt_rule = np.zeros(len(Xd), dtype=int)
for si, (name, fname, river, state, lat, lon, danger) in enumerate(tsi.STATIONS):
    grid, level, valid, rain, temp, F, rule, info = tsi.load_station(name, fname, danger)
    levels.append(level); valids.append(valid); dangers.append(float(danger))
    rebuilt_rule[(std == si)] = 0  # placeholder, filled below per window
    mem = tsi.rule_membership(level, valid, danger)
    idx = np.where(std == si)[0]
    # end_idx in dataset order for this station: rebuild from build_all loop
    ends = []
    for i in range(tsi.WINDOW - 1, len(grid) - tsi.HORIZON, tsi.STRIDE):
        if not valid[i - tsi.WINDOW + 1:i + tsi.HORIZON + 1].all():
            continue
        ends.append(i)
    assert len(ends) == len(idx), (name, len(ends), len(idx))
    for j, i in zip(idx, ends):
        rebuilt_rule[j] = int(mem[i])
        if yrd[j] == 2024:
            win_station.append(si); win_end.append(i)
agree = (rebuilt_rule == ruled).mean()
f1, tp, fp, fn = f1_counts(yd[test_mask], rebuilt_rule[test_mask])
match = agree == 1.0 and (f1, tp, fp, fn) == (0.9111, 169, 8, 25); ok &= match
print(f'flood: rebuilt rule agrees with saved column on {agree*100:.2f}% of all windows | '
      f'test F1 {f1} counts {tp}/{fp}/{fn} vs 0.9111 169/8/25 -> {"OK" if match else "MISMATCH"}')
np.savez(f'{OUT}/flood_series.npz',
         **{f'level_{i}': levels[i] for i in range(6)},
         **{f'valid_{i}': valids[i] for i in range(6)},
         dangers=np.array(dangers), win_station=np.array(win_station), win_end=np.array(win_end))

print('ALL CLEAN CHECKS:', 'PASS' if ok else 'FAIL')
