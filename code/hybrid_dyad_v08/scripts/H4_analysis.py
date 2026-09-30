"""H4 — aggregate results/H4_grid_raw.csv and H4_labels_<k>.npz -> results/H4_sensitivity.csv, H4_label_gain_corners.csv,
H4_sign_summary.json, figures/H4_sensitivity.png. Run from hybrid_dyad/ with PYTHONPATH=."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # package root on path
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import run_utils as ru
from H4_sensitivity import GRID, DEFAULT, CORNERS, CONTRASTS
from figure_helpers import apply_figure_style, panel_letter

RNG = np.random.default_rng(20261014); BOOT = 1000
raw = pd.read_csv(ru.OUT / 'H4_grid_raw.csv')
raw['is_default'] = (raw.tau == DEFAULT['tau']) & (raw.restriction_cost == DEFAULT['delta']) & (raw.lam_floor == DEFAULT['floor']) & (raw.N == DEFAULT['N'])
raw['sign'] = np.sign(raw.delta).astype(int); raw['ci_excludes_zero'] = (raw.mc_lo > 0) | (raw.mc_hi < 0)
ref = raw[raw.is_default].set_index(['contrast', 'metric', 'split'])['sign']
raw['sign_default'] = [ref.loc[(c, m, s)] for c, m, s in zip(raw.contrast, raw.metric, raw.split)]
raw['sign_flip'] = raw.sign != raw.sign_default
raw.to_csv(ru.OUT / 'H4_sensitivity.csv', index=False)

summ = {}
for (c, m, s), g in raw.groupby(['contrast', 'metric', 'split']):
    flips = g[g.sign_flip]
    summ[f'{c} | {m} | {s}'] = dict(sign_default=int(g.sign_default.iloc[0]), n_settings=len(g), n_sign_flips=int(len(flips)),
                                     n_ci_excludes_zero_same_sign=int((g.ci_excludes_zero & ~g.sign_flip).sum()),
                                     flip_settings=[dict(tau=r.tau, delta=r.restriction_cost, lam_floor=r.lam_floor, N=int(r.N), delta_value=round(r.delta, 4)) for r in flips.itertuples()],
                                     delta_range=[float(g.delta.min()), float(g.delta.max())], delta_default=float(g[g.is_default].delta.iloc[0]))

# label gain at the 6 one-at-a-time extremes (N=2, 500 worlds, eta=0.25)
rows = []
for k, s in enumerate(CORNERS):
    r = np.load(ru.OUT / f'H4_labels_{k}.npz', allow_pickle=True)
    for g in ('honest', 'attenuated', 'costly', 'access'):
        la = r[f'{g}/loss_acts'].mean(1); ll = r[f'{g}/loss_labels'].mean(1); d = la - ll; m = len(d)
        bs = d[RNG.integers(0, m, (BOOT, m))].mean(1)
        fa = r[f'{g}/fam_acts']; fl = r[f'{g}/fam_labels']; fi = {'attenuated': 0, 'costly': 1, 'access': 2}.get(g)
        rows.append(dict(setting=k, tau=s['tau'], restriction_cost=s['delta'], lam_floor=s['floor'], N=s['N'], generator=g, n=m, eta=0.25,
                         loss_sig_acts=la.mean(), loss_sig_labels=ll.mean(), gain=d.mean(), gain_lo=np.quantile(bs, .025), gain_hi=np.quantile(bs, .975),
                         correct_family_acts=(fa.argmax(1) == fi).mean() if fi is not None else np.nan,
                         correct_family_labels=(fl.argmax(1) == fi).mean() if fi is not None else np.nan))
lab = pd.DataFrame(rows); lab.to_csv(ru.OUT / 'H4_label_gain_corners.csv', index=False)
summ['label_gain_corners'] = dict(min_gain=float(lab.gain.min()), max_gain=float(lab.gain.max()), all_ci_positive=bool((lab.gain_lo > 0).all()),
                                  n_rows=len(lab), family_recovery_improved_all=bool((lab.dropna().correct_family_labels >= lab.dropna().correct_family_acts).all()))
json.dump(summ, open(ru.OUT / 'H4_sign_summary.json', 'w'), indent=1)

# figure: spread of each contrast over the 54 settings (all worlds and A_own0==1), default marked
apply_figure_style()
TITLES = {('access-honest', 'evidence_count'): 'access − honest\nshared signals per dialogue', ('access-honest', 'A_p_truth'): 'access − honest\nfinal P(truth) of A',
          ('costly-attenuated', 'A_dbelief'): 'costly − attenuated\n|Δ private belief| of A', ('access-costly', 'restrict_A'): 'access − costly\nrestriction rate of A'}
fig, axes = plt.subplots(1, 4, figsize=(8.8, 3.0))
NCOL = {2: '#4c72b0', 4: '#dd8452'}
for ax, ((c, m), title), letter in zip(axes, TITLES.items(), 'abcd'):
    for si, split in enumerate(('all', 'A_own0==1')):
        g = raw[(raw.contrast == c) & (raw.metric == m) & (raw.split == split)]
        for N in (2, 4):
            gg = g[g.N == N]; jit = RNG.uniform(-0.12, 0.12, len(gg))
            ax.scatter(np.full(len(gg), si) + jit + (0.16 if N == 4 else -0.16), gg.delta, s=10, color=NCOL[N], alpha=0.75, edgecolor='none', label=f'N={N}' if (si == 0 and letter == 'a') else None)
        d0 = g[g.is_default]; ax.scatter([si - 0.16], d0.delta, s=40, facecolor='none', edgecolor='black', linewidth=0.9, zorder=4, label='default setting' if (si == 0 and letter == 'a') else None)
    ax.axhline(0, color='#999999', lw=0.8); ax.set_xticks([0, 1]); ax.set_xticklabels(['all worlds', 'A starts on\nown side']); ax.set_title(title, loc='left', fontsize=7.5)
    ax.margins(x=0.25, y=0.15); panel_letter(ax, letter)
axes[0].set_ylabel('paired contrast (mean over worlds)')
axes[0].legend(loc='best', frameon=False, fontsize=6)
fig.suptitle('Key contrasts keep their sign across 54 settings of τ × δ × λ_floor × N (1000 worlds each, partner honest)', x=0.01, ha='left', fontsize=8)
fig.tight_layout(); fig.savefig(ru.FIG / 'H4_sensitivity.png', dpi=200); plt.close(fig)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'flip_settings'} if isinstance(v, dict) else v for k, v in summ.items()}, indent=1))
