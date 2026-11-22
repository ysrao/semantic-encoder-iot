"""Monsoon onset case — REAL labels (IMD declared onset over Kerala, 2009-2025,
from IMD's own press-release annex) + ERA5 predictors via Open-Meteo.

Task: will the southwest monsoon onset over Kerala be declared within 5 days?
Window: last 16 days x 4 channels, Kerala sector (mean of 4 ERA5 points:
Thiruvananthapuram, Kochi, Kozhikode, Palakkad):
  [daily rain/25, mean RH/100, (1015-MSLP)/20, mean wind/12]
Window ends run Apr 28 - Jul 2 each year; label 1 if onset is 1-5 days ahead.
Hand-crafted rule (IMD-flavoured): fire if 4-day mean rain >= R*, 4-day mean
RH >= 75%, and pressure fell across the window; R* tuned on TRAIN years only.
Splits by year: train 2009-2018, val 2019-2021, test 2022-2025.
"""
import sys, json, urllib.request
sys.path.insert(0, '/home/hatch/workspace/edge-apps-prep/semantic_encoder')
import numpy as np, pandas as pd, torch
from semantic_encoder import SemanticEncoder
from train_semantic import GatewayDecoder, quantise_ste, metrics

torch.manual_seed(7); np.random.seed(7)
BASE = '/home/hatch/workspace/edge-apps-prep/real_tasks/monsoon'
ONSET = {2009: '05-23', 2010: '05-31', 2011: '05-29', 2012: '06-05', 2013: '06-01',
         2014: '06-06', 2015: '06-05', 2016: '06-08', 2017: '05-30', 2018: '05-29',
         2019: '06-08', 2020: '06-01', 2021: '06-03', 2022: '05-29', 2023: '06-08',
         2024: '05-30', 2025: '05-24'}
POINTS = [(8.48, 76.95), (9.97, 76.28), (11.26, 75.78), (10.77, 76.65)]

def pull(lat, lon, year):
    url = ("https://archive-api.open-meteo.com/v1/archive?latitude=%s&longitude=%s"
           "&start_date=%d-04-01&end_date=%d-07-15&hourly=precipitation,"
           "relative_humidity_2m,pressure_msl,wind_speed_10m&timezone=Asia%%2FKolkata"
           % (lat, lon, year, year))
    with urllib.request.urlopen(url, timeout=120) as r:
        d = json.load(r)
    h = d['hourly']
    df = pd.DataFrame({'ts': pd.to_datetime(h['time']), 'rain': h['precipitation'],
                       'rh': h['relative_humidity_2m'], 'p': h['pressure_msl'],
                       'w': h['wind_speed_10m']})
    return df.set_index('ts').resample('1D').agg({'rain': 'sum', 'rh': 'mean', 'p': 'mean', 'w': 'mean'})

rows = []
for year, md in ONSET.items():
    onset = pd.Timestamp(f'{year}-{md}')
    days = None
    for lat, lon in POINTS:
        d = pull(lat, lon, year)
        days = d if days is None else days + d
    days = days / len(POINTS)
    ch = np.stack([np.clip(days['rain'].values / 25.0, 0, 1.5),
                   days['rh'].values / 100.0,
                   np.clip((1015 - days['p'].values) / 20.0, 0, 1.5),
                   np.clip(days['w'].values / 12.0, 0, 1.5)], axis=1)
    idx = days.index
    n0 = 0
    for end in range(15, len(days)):
        d_end = idx[end]
        if not (pd.Timestamp(f'{year}-04-28') <= d_end <= pd.Timestamp(f'{year}-07-02')):
            continue
        ahead = (onset - d_end).days
        label = 1 if 1 <= ahead <= 5 else 0
        rows.append(dict(year=year, X=ch[end-15:end+1].astype(np.float32), y=label,
                         rain4=float(days['rain'].values[end-3:end+1].mean()),
                         rh4=float(days['rh'].values[end-3:end+1].mean()),
                         pfall=float(days['p'].values[end-15:end-11].mean() - days['p'].values[end-3:end+1].mean())))
        n0 += 1
    print(f'{year}: onset {md}, windows {n0}')

def split(lo, hi):
    s = [r for r in rows if lo <= r['year'] <= hi]
    return s, np.stack([r['X'] for r in s]).reshape(len(s), -1), np.array([r['y'] for r in s])

tr_r, Xtr, ytr = split(2009, 2018)
va_r, Xva, yva = split(2019, 2021)
te_r, Xte, yte = split(2022, 2025)
print('windows train/val/test:', len(ytr), len(yva), len(yte),
      '| positive rates: %.3f %.3f %.3f' % (ytr.mean(), yva.mean(), yte.mean()))

def rule_pred(rs, R):
    return np.array([1 if (r['rain4'] >= R and r['rh4'] >= 75 and r['pfall'] > 0) else 0 for r in rs])
best = None
for R in (3, 5, 8, 12):
    m = metrics(ytr, rule_pred(tr_r, R))
    if best is None or m['f1'] > best[1]['f1']:
        best = (R, m)
R_star = best[0]
rule_te = metrics(yte, rule_pred(te_r, R_star))
print('rule: rain threshold', R_star, 'mm (tuned on train) | test:', rule_te)

enc, dec = SemanticEncoder(), GatewayDecoder()
opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr, dtype=torch.float32)
pos_w = torch.tensor((len(ytr) - ytr.sum()) / max(1, ytr.sum()))
bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
for ep in range(40):
    perm = torch.randperm(len(ytr))
    for i in range(0, len(ytr), 256):
        idx = perm[i:i+256]
        xb, yb = Xtr_t[idx], ytr_t[idx]
        z = quantise_ste(enc(xb))
        zn = z + 0.03 * torch.randn_like(z)
        logit, recon = dec(zn)
        loss = bce(logit.squeeze(-1), yb) + 0.05 * torch.mean((recon - xb) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()

def probs(X):
    with torch.no_grad():
        return torch.sigmoid(dec(quantise_ste(enc(torch.tensor(X))))[0]).squeeze(-1).numpy()
pva, pte = probs(Xva), probs(Xte)
ths = np.linspace(0.05, 0.95, 19)
thr = max(ths, key=lambda t: metrics(yva, (pva >= t).astype(int))['f1'])
pair_te = metrics(yte, (pte >= thr).astype(int))
print('pair: threshold %.2f (on val) | test:' % thr, pair_te)

# year-level: in each test year, was onset flagged at least once in the 5 days ahead?
def year_level(rs, pred):
    by = {}
    for r, p in zip(rs, pred):
        by.setdefault(r['year'], []).append((r['y'], p))
    return {y: int(any(yv == 1 and pv == 1 for yv, pv in v)) for y, v in sorted(by.items())}
print('test years flagged in time — rule:', year_level(te_r, rule_pred(te_r, R_star)),
      '| pair:', year_level(te_r, (pte >= thr).astype(int)))

torch.save({'enc': enc.state_dict(), 'dec': dec.state_dict()}, f'{BASE}/pair_weights.pt')
np.savez(f'{BASE}/test_windows.npz', X=Xte, y=yte, p=pte, thr=thr)
json.dump(dict(rule_R=float(R_star), rule_test=rule_te, pair_thr=float(thr), pair_test=pair_te,
               counts=dict(train=int(len(ytr)), val=int(len(yva)), test=int(len(yte)))),
          open(f'{BASE}/results.json', 'w'), indent=1)
print('saved results.json')
