import json, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd
import core as hd
ROOT = Path(__file__).resolve().parent; OUT = ROOT / 'results'; OUT.mkdir(exist_ok=True); FIG = ROOT / 'figures'; FIG.mkdir(exist_ok=True)
T = hd.CFG['agent_types']
def cond(a_type, b_type, beta=1.5, **kw): return dict(A=dict(T[a_type]), B=dict(T[b_type]), beta=beta, **kw)
def run_condition(name, worlds, condition, save=True):
    t = time.time(); d = hd.dialogue(worlds, condition); df = hd.metrics(d)
    if save: df.to_csv(OUT / f'{name}_worlds.csv', index=False)
    return d, df, round(time.time() - t, 2)
def summary_row(name, df, split='all'):
    row = {'run': name, 'split': split, 'n': len(df)}
    for col in df.columns:
        if col in ('world', 'truth'): continue
        v = df[col].dropna(); row[col] = v.mean() if len(v) else np.nan; row[col + '_se'] = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    return row
def logloss_signal(res, d):
    got = (d['ev'][:, 1:] != 2).astype(float); p = np.clip(res['next_sig'], 1e-12, 1 - 1e-12)
    return -(got * np.log(p) + (1 - got) * np.log(1 - p))          # (n, R-1) per-target losses
def logloss_act(res, d, acts_menu):
    idx = np.array([acts_menu.index(a) for a in range(6)]) if False else None
    a = d['acts'][:, 1:, 0].astype(int); amap = np.full(6, -1); amap[[hd.ACT_NAMES.index(x) for x in acts_menu]] = np.arange(len(acts_menu))
    p = np.clip(res['next_act'][np.arange(a.shape[0])[:, None], np.arange(a.shape[1])[None, :], amap[a]], 1e-12, 1)
    return -np.log(p)
