"""Paired contrasts with 95% Monte Carlo intervals (mean +/- 1.96 SE over dialogues) for the 480-dialogue extension of the
observer pilot (results/ext_observer_worlds_480.csv, produced by ext_observer.py). Output: results/ext_observer_contrasts_480.csv.
Metrics are kept separate: argmax accuracy (correct_A/B), posterior on the true profile, arrival log-loss, next-act log-loss."""
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parent / 'results'
ext = pd.read_csv(R / 'ext_observer_worlds_480.csv'); piv = ext.pivot(index='world', columns='mode'); gen = ext.groupby('world').generator.first()
rows = []
for a, b in (('labels_evidence', 'actions_evidence'), ('forms_evidence', 'actions_evidence')):
    for metric in ['correct_A', 'posterior_true_A', 'correct_B', 'posterior_true_B', 'arrival_logloss', 'act_logloss']:
        for g in ['all'] + sorted(gen.unique()):
            m = np.ones(len(piv), bool) if g == 'all' else (gen.reindex(piv.index) == g).to_numpy()
            dlt = (piv[(metric, a)] - piv[(metric, b)]).to_numpy()[m]; se = dlt.std(ddof=1) / np.sqrt(len(dlt))
            rows.append(dict(contrast=f'{a} minus {b}', metric=metric, generator=g, n=len(dlt), mean=dlt.mean(), lo=dlt.mean() - 1.96 * se, hi=dlt.mean() + 1.96 * se))
out = pd.DataFrame(rows); out.to_csv(R / 'ext_observer_contrasts_480.csv', index=False)
print(out[(out.generator == 'all')].round(4).to_string(index=False))
