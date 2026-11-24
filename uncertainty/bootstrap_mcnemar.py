#!/usr/bin/env python3
"""Paired bootstrap CIs on F1 difference + McNemar, all four v3 cases.
Decisions recovered exactly as prep_sweep.py verified them (clean F1s are
asserted against the published values before any resampling)."""
import json
import numpy as np
from scipy.stats import binomtest

RT = '/home/hatch/workspace/edge-apps-prep/real_tasks'
rng = np.random.default_rng(20261124)
B = 10000

def counts(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f

def load(case):
    z = np.load(f'{RT}/{case}/test_windows.npz')
    y, pair = z['y'], (z['p'] >= float(z['thr'])).astype(int)
    if case == 'dust':
        W = z['X'].reshape(-1, 16, 4)
        rule = (2.0 * W[:, 15, 0] >= 0.6).astype(int)
    elif case == 'storm':
        W = z['X'].reshape(-1, 16, 4)
        d2l = W[:, 15, 2] * 1000.0; wind = W[:, 15, 0] * 150.0
        closing = (W[:, 0, 2] - W[:, 15, 2]) * 1000.0
        rule = ((d2l < 300) & (closing > 0) & (wind >= 34)).astype(int)
    elif case == 'soil':
        f = np.load(f'{RT}/snr_sweep/soil_fields.npz')
        assert (f['y'] == y).all()
        rule = ((f['sm_end'] < f['theta'] + 0.01) & (f['fall'] > 0) & (f['rain4'] < 2.0)).astype(int)
    elif case == 'flood':
        d = np.load('/home/hatch/workspace/edge-apps-prep/semantic_encoder/india/dataset_india.npz')
        m = d['end_year'] == 2024
        assert np.array_equal(d['X'][m], z['X']) and np.array_equal(d['y'][m], y)
        rule = d['rule'][m].astype(int)
    return y, pair, rule

PUB = {'soil': (0.2393, 0.3846), 'dust': (0.6866, 0.4333),
       'storm': (0.6216, 0.6494), 'flood': (0.8789, 0.9111)}

out = {}
for case in ('soil', 'dust', 'storm', 'flood'):
    y, pair, rule = load(case)
    res = {'n': int(len(y)), 'pos': int(y.sum())}
    f1s = {}
    for name, pred in (('pair', pair), ('rule', rule)):
        tp, fp, fn, tn = counts(y, pred)
        p, r, f1 = prf(tp, fp, fn)
        res[name] = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'P': round(p, 4),
                     'R': round(r, 4), 'F1': round(f1, 4), 'FAR': round(fp / (fp + tn), 4)}
        f1s[name] = f1
    assert abs(f1s['pair'] - PUB[case][0]) < 5e-4 and abs(f1s['rule'] - PUB[case][1]) < 5e-4, (case, f1s)
    res['dF1'] = round(f1s['pair'] - f1s['rule'], 4)
    # paired bootstrap over windows
    n = len(y)
    diffs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        d = []
        for pred in (pair, rule):
            tp, fp, fn, tn = counts(yb, pred[idx])
            d.append(prf(tp, fp, fn)[2])
        diffs[b] = d[0] - d[1]
    res['boot_dF1_mean'] = round(float(diffs.mean()), 4)
    res['boot_dF1_ci95'] = [round(float(np.percentile(diffs, 2.5)), 4),
                            round(float(np.percentile(diffs, 97.5)), 4)]
    res['boot_frac_pair_better'] = round(float((diffs > 0).mean()), 4)
    # McNemar exact on correctness discordance
    pc, rc = (pair == y), (rule == y)
    b01 = int((pc & ~rc).sum()); b10 = int((~pc & rc).sum())
    res['mcnemar'] = {'pair_only_right': b01, 'rule_only_right': b10,
                      'p_exact': round(float(binomtest(min(b01, b10), b01 + b10, 0.5).pvalue), 4) if b01 + b10 else 1.0}
    out[case] = res
    print(case, json.dumps(res))

json.dump(out, open(f'{RT}/uncertainty/uncertainty_results.json', 'w'), indent=1)
print('saved uncertainty_results.json')
