"""Shared helpers for experiment scripts E1–E5. PYTHONPATH=. python E1_phase_map.py"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # package root on path
import json, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd
import core as rd

ROOT = Path(__file__).resolve().parent; OUT = ROOT / 'results'; OUT.mkdir(exist_ok=True)
T = rd.CFG['agent_types']


def cond(a_type, b_type, beta=1.5, **kw):
    return dict(A=dict(T[a_type], beta=beta), B=dict(T[b_type], beta=beta), **kw)


def run_condition(name, worlds, condition, save=True):
    t = time.time(); d = rd.dialogue(worlds, condition); df, tr = rd.metrics(d)
    if save:
        df.to_csv(OUT / f'{name}_worlds.csv', index=False)
        np.savez_compressed(OUT / f'{name}.npz', **{k: v for k, v in d.items() if isinstance(v, np.ndarray)})
    return d, df, tr, round(time.time() - t, 2)


def summary_row(name, df, split='all'):
    row = {'run': name, 'split': split, 'n': len(df)}
    for col in df.columns:
        if col in ('world', 'truth'): continue
        v = df[col].dropna(); row[col] = v.mean() if len(v) else np.nan
        row[col + '_se'] = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    return row


def write_manifest(tag, extra=None):
    files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in sorted(OUT.glob(f'{tag}*'))}
    (OUT / f'{tag}_manifest.json').write_text(json.dumps({'files': files, 'numpy': np.__version__, 'extra': extra or {}}, indent=1))
