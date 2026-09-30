"""Re-score the archived July 2026 classifier outputs on the 169 Tier A cases (Part I of the paper).

Inputs (data/archived_predictions/):
  engine_md3_combined_2026-07.json   – typed engine, three stored runs per case
  direct_prompt_m3_2026-07-01.json   – direct-prompt baseline (same model, same cases), three runs per case
Both files use the `per_run_outcomes` schema written by the engine harness. No new model calls are made.

Scoring rule (identical for both systems): per case, label-majority over the three runs; a three-way tie is
broken by run 0. Strict accuracy = exact match of the primary output with the coordinator-locked gold label.
Macro-F1 averages over the ten gold categories (eight configurations, SEAL, SHIFT); "insufficient context"
is a possible prediction of the engine but never a gold label.

Outputs (results/):
  part1_scores.csv               – strict accuracy (all / non-SHIFT), macro-F1, SHIFT recall for both systems
  part1_confusion_<system>.csv   – 11x11 confusion counts (gold rows, predicted columns)
  part1_confusion_<system>_rownorm.csv – row-normalised confusion (used as the annotation-noise model in
                                   code/reciprocal_dyad_v07 and the LOCK/SEEK rows in code/hybrid_dyad_v08)
Usage: python code/part1_benchmark/rescore_archive.py
"""
import json, collections
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'archived_predictions'; OUT = ROOT / 'results'; OUT.mkdir(exist_ok=True)
LABELS = ['BUILD', 'SEEK', 'UNSEAL', 'LOCK', 'DRAIN', 'FLOOD', 'EDGE', 'VOID', 'SEAL', 'SHIFT', 'INSUFF']
GOLD = LABELS[:10]


def short(label):
    """'regime.LOCK' -> 'LOCK'; 'function.SEAL' -> 'SEAL'; 'event.SHIFT' -> 'SHIFT'; abstentions -> 'INSUFF'."""
    l = label.split('.')[-1]
    return 'INSUFF' if 'insufficient' in l.lower() else l


def majority_vote(data):
    """{case_id: (gold, majority prediction, [run predictions])} for the first primary segment of each case."""
    runs = data['per_run_outcomes']; out = {}
    for cid in [c['case_id'] for c in runs[0]]:
        preds, gold = [], None
        for r in runs:
            c = next(x for x in r if x['case_id'] == cid); rows = c.get('primary_segment_results') or []
            if rows: preds.append(short(rows[0]['predicted'])); gold = short(rows[0]['expected'])
        counts = collections.Counter(preds).most_common()
        win = counts[0][0] if (counts[0][1] > 1 or len(counts) == 1) else preds[0]   # three-way tie -> run 0
        out[cid] = (gold, win, preds)
    return out


def confusion(votes):
    M = pd.DataFrame(0, index=LABELS, columns=LABELS)
    for g, p, _ in votes.values(): M.loc[g, p] += 1
    return M


def macro_f1(M):
    f1 = []
    for g in GOLD:
        tp = M.loc[g, g]; fp = M[g].sum() - tp; fn = M.loc[g].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.; r = tp / (tp + fn) if tp + fn else 0.
        f1.append(2 * p * r / (p + r) if p + r else 0.)
    return float(np.mean(f1))


def score(name, path):
    votes = majority_vote(json.loads(path.read_text())); M = confusion(votes)
    n = len(votes); correct = sum(g == p for g, p, _ in votes.values())
    non_shift = [(g, p) for g, p, _ in votes.values() if g != 'SHIFT']
    row = {'system': name, 'n_cases': n, 'strict_correct': correct, 'strict_acc': correct / n,
           'non_shift_correct': sum(g == p for g, p in non_shift), 'non_shift_n': len(non_shift),
           'non_shift_acc': sum(g == p for g, p in non_shift) / len(non_shift),
           'macro_f1': macro_f1(M), 'shift_recall': M.loc['SHIFT', 'SHIFT'] / M.loc['SHIFT'].sum(),
           'abstentions': int(M['INSUFF'].sum())}
    M.to_csv(OUT / f'part1_confusion_{name}.csv')
    (M.T / M.sum(1)).T.fillna(0).to_csv(OUT / f'part1_confusion_{name}_rownorm.csv')
    return row, votes


if __name__ == '__main__':
    rows = []
    rows.append(score('engine', DATA / 'engine_md3_combined_2026-07.json')[0])
    rows.append(score('direct_prompt', DATA / 'direct_prompt_m3_2026-07-01.json')[0])
    df = pd.DataFrame(rows); df.to_csv(OUT / 'part1_scores.csv', index=False)
    print(df.round(3).to_string(index=False))
