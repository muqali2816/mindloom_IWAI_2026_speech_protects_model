"""H2 — aggregate results/H2_raw_<gen>.npz into results/H2_recovery.csv, H2_prediction_gains.csv, H2_controls.csv, H2_P3_check.json
and figures/H2_*.png. Run from hybrid_dyad/ with PYTHONPATH=. after H2_recovery.py for all generators."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # package root on path
import json
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
import core as hd, observer as ob, run_utils as ru
from H2_recovery import GENERATORS, CHANNELS

BOOT = 1000; RNG = np.random.default_rng(20261012)
HYPS, FAM = ob.hypotheses(); PRIOR_MEAN = np.array(HYPS).mean(0)            # mean over unique hypotheses (omega, c, gamma)
TRUE_FAMILY = {'attenuated': 'attenuation', 'costly': 'cost', 'access': 'access', 'access_x_access': 'access', 'costly_x_costly': 'cost'}
FAMS = ['attenuation', 'cost', 'access']
raw = {g: np.load(ru.OUT / f'H2_raw_{g}.npz', allow_pickle=True) for g in GENERATORS}


def boot_mean_diff(a, b, ratio_den=None):
    """a, b: per-world mean losses (m,) for acts and channel. Returns gain, lo, hi (and ratio CIs if ratio_den given)."""
    m = len(a); idx = RNG.integers(0, m, (BOOT, m)); g = a - b
    bs = g[idx].mean(1); out = dict(gain=g.mean(), gain_lo=np.quantile(bs, .025), gain_hi=np.quantile(bs, .975))
    if ratio_den is not None:
        go = a - ratio_den; bso = go[idx].mean(1); r = bs / bso
        out.update(ratio=g.mean() / go.mean(), ratio_lo=np.quantile(r, .025), ratio_hi=np.quantile(r, .975))
    return out


rec_rows, gain_rows, ctrl_rows = [], [], []
for g, r in raw.items():
    true = r['true_hyp']; own = r['A_own0'] == 1
    for ch, (channel, eta, m) in CHANNELS.items():
        F = r[f'{ch}/family_post']; means = {k: r[f'{ch}/{k}_mean'] for k in ('omega', 'c', 'gamma')}
        for split, mask in (('all', np.ones(m, bool)), ('A_own0==1', own[:m])):
            row = dict(generator=g, channel=ch, eta=eta, n=int(mask.sum()), split=split, true_omega=true[0], true_c=true[1], true_gamma=true[2])
            for k, i in zip(('omega', 'c', 'gamma'), range(3)):
                row[f'{k}_MAE'] = np.abs(means[k][mask] - true[i]).mean(); row[f'{k}_MAE_prior'] = abs(PRIOR_MEAN[i] - true[i])
            for fi, f in enumerate(FAMS): row[f'post_{f}'] = F[mask, fi].mean(); row[f'argmax_{f}'] = (F[mask].argmax(1) == fi).mean()
            tf = TRUE_FAMILY.get(g)
            if tf is not None:
                fi = FAMS.index(tf); row['true_family'] = tf; row['correct_family'] = (F[mask].argmax(1) == fi).mean(); row['posterior_on_family'] = F[mask, fi].mean()
            else:
                row['true_family'] = 'undefined'; row['correct_family'] = np.nan; row['posterior_on_family'] = np.nan
            rec_rows.append(row)
    # prediction gains relative to 'acts' on the same worlds
    la_full = r['acts/loss_sig'].mean(1); aa_full = r['acts/loss_act'].mean(1); lo_full = r['oracle/loss_sig'].mean(1); ao_full = r['oracle/loss_act'].mean(1)
    for ch, (channel, eta, m) in CHANNELS.items():
        if ch == 'acts': continue
        ls = r[f'{ch}/loss_sig'].mean(1); la_ = r[f'{ch}/loss_act'].mean(1)
        for split, mask in (('all', np.ones(m, bool)), ('A_own0==1', own[:m])):
            row = dict(generator=g, channel=ch, eta=eta, n=int(mask.sum()), split=split,
                       loss_sig_acts=la_full[:m][mask].mean(), loss_sig_channel=ls[mask].mean(), loss_act_acts=aa_full[:m][mask].mean(), loss_act_channel=la_[mask].mean())
            s = boot_mean_diff(la_full[:m][mask], ls[mask], lo_full[:m][mask]); row.update({f'sig_{k}': v for k, v in s.items()})
            a = boot_mean_diff(aa_full[:m][mask], la_[mask], ao_full[:m][mask]); row.update({f'act_{k}': v for k, v in a.items()})
            gain_rows.append(row)
    # controls
    ctrl_rows.append(dict(generator=g, control='null == acts', n=2000, max_abs_diff_family_post=np.abs(r['null/family_post'] - r['acts/family_post']).max(),
                          max_abs_diff_next_sig=np.abs(r['null/next_sig'] - r['acts/next_sig']).max(),
                          gain_sig=(la_full - r['null/loss_sig'].mean(1)).mean(), passed=bool(np.abs(r['null/family_post'] - r['acts/family_post']).max() <= 1e-9 and np.abs(r['null/next_sig'] - r['acts/next_sig']).max() <= 1e-9)))
    ctrl_rows.append(dict(generator=g, control='labels eta=1 == acts', n=1000, max_abs_diff_family_post=np.abs(r['labels_eta1/family_post'] - r['acts/family_post'][:1000]).max(),
                          max_abs_diff_next_sig=np.abs(r['labels_eta1/next_sig'] - r['acts/next_sig'][:1000]).max(),
                          gain_sig=(la_full[:1000] - r['labels_eta1/loss_sig'].mean(1)).mean(), passed=bool(np.abs(r['labels_eta1/next_sig'] - r['acts/next_sig'][:1000]).max() <= 1e-9)))
    ctrl_rows.append(dict(generator=g, control='oracle >= labels (next-signal gain, all)', n=2000, max_abs_diff_family_post=np.nan, max_abs_diff_next_sig=np.nan,
                          gain_sig=(la_full - lo_full).mean() - (la_full - r['labels_eta0.25/loss_sig'].mean(1)).mean(),
                          passed=bool((la_full - lo_full).mean() >= (la_full - r['labels_eta0.25/loss_sig'].mean(1)).mean())))
rec = pd.DataFrame(rec_rows); gains = pd.DataFrame(gain_rows); ctrl = pd.DataFrame(ctrl_rows)
for df_ in (rec, gains): df_['channel'] = df_['channel'].replace({'null': 'null_labels'})   # 'null' would be parsed as NaN by CSV readers
rec.to_csv(ru.OUT / 'H2_recovery.csv', index=False); gains.to_csv(ru.OUT / 'H2_prediction_gains.csv', index=False); ctrl.to_csv(ru.OUT / 'H2_controls.csv', index=False)
timing = {g: json.loads(str(raw[g]['timing'])) for g in raw}; json.dump(timing, open(ru.OUT / 'H2_timing.json', 'w'), indent=1)

# ---- P3 check
ga = gains[(gains.split == 'all')].set_index(['generator', 'channel']); ra = rec[rec.split == 'all'].set_index(['generator', 'channel'])
p3 = {}
gens_def = list(TRUE_FAMILY)
p3['labels_improve_next_signal_every_generator'] = {g: bool(ga.loc[(g, 'labels_eta0.25'), 'sig_gain_lo'] > 0) for g in GENERATORS}
p3['labels_improve_family_recovery_every_defined_generator'] = {g: float(ra.loc[(g, 'labels_eta0.25'), 'correct_family'] - ra.loc[(g, 'acts'), 'correct_family']) for g in gens_def}
p3['null_zero_gain'] = bool(ctrl[ctrl.control == 'null == acts'].passed.all())
p3['oracle_bounds_labels'] = {g: bool(ga.loc[(g, 'oracle'), 'sig_gain'] >= ga.loc[(g, 'labels_eta0.25'), 'sig_gain']) for g in GENERATORS}
p3['ratio_eta0.25'] = {g: float(ga.loc[(g, 'labels_eta0.25'), 'sig_ratio']) for g in GENERATORS}
p3['ratio_in_0.25_0.5'] = {g: bool(0.25 <= v <= 0.5) for g, v in p3['ratio_eta0.25'].items()}
p3['ratio_decreasing_with_eta'] = {g: bool(ga.loc[(g, 'labels_eta0'), 'sig_ratio'] >= ga.loc[(g, 'labels_eta0.25'), 'sig_ratio'] >= ga.loc[(g, 'labels_eta0.5'), 'sig_ratio']) for g in GENERATORS}
dfam = {g: float(ra.loc[(g, 'labels_eta0.25'), 'correct_family'] - ra.loc[(g, 'acts'), 'correct_family']) for g in ('attenuated', 'costly', 'access')}
p3['family_gain_by_generator'] = dfam; p3['access_gains_most'] = bool(max(dfam, key=dfam.get) == 'access'); p3['attenuation_gains_least'] = bool(min(dfam, key=dfam.get) == 'attenuated')
json.dump(p3, open(ru.OUT / 'H2_P3_check.json', 'w'), indent=1)

# ---- figures
from figure_helpers import apply_figure_style, panel_letter
apply_figure_style()
CH_ORDER = ['acts', 'labels_eta0', 'labels_eta0.25', 'labels_eta0.5', 'oracle']
CH_LABEL = {'acts': 'acts + signals', 'labels_eta0': 'labels η=0', 'labels_eta0.25': 'labels η=0.25', 'labels_eta0.5': 'labels η=0.5', 'oracle': 'oracle forms', 'null': 'null labels'}
GEN_LABEL = {'attenuated': 'attenuated\n(ω=0.2)', 'costly': 'costly\n(c=1.6)', 'access': 'access-protecting\n(γ=2)', 'access_x_access': 'access ×\naccess', 'costly_x_costly': 'costly ×\ncostly'}
COL = {'acts': '#7f7f7f', 'labels_eta0': '#9ecae1', 'labels_eta0.25': '#3182bd', 'labels_eta0.5': '#08519c', 'oracle': '#d95f02', 'null': '#bdbdbd'}
MK = {'acts': 'o', 'labels_eta0': '^', 'labels_eta0.25': '^', 'labels_eta0.5': '^', 'oracle': 's'}

# (a) correct family by channel x generator
fig, ax = plt.subplots(figsize=(6.4, 3.2))
gens = list(TRUE_FAMILY); x = np.arange(len(gens)); off = np.linspace(-0.3, 0.3, len(CH_ORDER))
for k, ch in enumerate(CH_ORDER):
    y = [ra.loc[(g, ch), 'correct_family'] for g in gens]
    ax.scatter(x + off[k], y, color=COL[ch], marker=MK[ch], s=28, zorder=3, label=CH_LABEL[ch], edgecolor='white', linewidth=0.4)
ax.axhline(1 / 3, color='#bbbbbb', lw=0.8, ls=':'); ax.text(len(gens) - 0.6, 1 / 3 + 0.015, 'chance (3 families)', ha='right', va='bottom', fontsize=6, color='#777777')
ax.set_xticks(x); ax.set_xticklabels([GEN_LABEL[g] for g in gens]); ax.set_ylabel('P(argmax family = true family)'); ax.set_ylim(0.25, 1.02); ax.margins(x=0.06)
ax.set_title('Form labels raise family recovery over acts + signals; oracle forms bound the gain', loc='left')
ax.legend(loc='upper left', ncol=5, frameon=False, fontsize=6.5, handletextpad=0.2, columnspacing=0.8)
fig.tight_layout(); fig.savefig(ru.FIG / 'H2_family_recovery.png', dpi=200); plt.close(fig)

# (b) next-signal log-loss gain with bootstrap CIs
fig, ax = plt.subplots(figsize=(6.4, 3.2))
gens_b = list(GENERATORS); x = np.arange(len(gens_b)); chs = ['labels_eta0', 'labels_eta0.25', 'labels_eta0.5', 'oracle']; off = np.linspace(-0.27, 0.27, len(chs))
for k, ch in enumerate(chs):
    sub = ga.loc[[(g, ch) for g in gens_b]]
    ax.errorbar(x + off[k], sub.sig_gain, yerr=[sub.sig_gain - sub.sig_gain_lo, sub.sig_gain_hi - sub.sig_gain], fmt=MK[ch], color=COL[ch], ms=4, capsize=1.5, lw=0.8, label=CH_LABEL[ch], zorder=3)
ax.axhline(0, color='#bbbbbb', lw=0.8)
GEN_B = {'honest': 'honest', 'attenuated': 'attenuated', 'costly': 'costly', 'access': 'access-\nprotecting', 'mixed': 'mixed', 'access_x_access': 'access ×\naccess', 'costly_x_costly': 'costly ×\ncostly'}
ax.set_xticks(x); ax.set_xticklabels([GEN_B[g] for g in gens_b]); ax.set_ylabel('Next-signal log-loss gain\nover acts + signals (nat / outcome)'); ax.margins(x=0.05, y=0.1)
ax.set_title('Labels recover part of the oracle-form gain in next-signal prediction (95% bootstrap CI over worlds)', loc='left')
ax.legend(loc='upper left', ncol=4, frameon=False, fontsize=6.5, handletextpad=0.2, columnspacing=0.8)
fig.tight_layout(); fig.savefig(ru.FIG / 'H2_prediction_gain.png', dpi=200); plt.close(fig)

# (c) histograms of family_post for the access generator by channel
r = raw['access']; chs_c = ['acts', 'labels_eta0.25', 'oracle', 'null']
fig, axes = plt.subplots(1, len(chs_c), figsize=(7.2, 2.4), sharey=True)
bins = np.linspace(0, 1, 21)
for ax, ch in zip(axes, chs_c):
    F = r[f'{ch}/family_post']; ax.hist(F[:, 2], bins=bins, color=COL[ch], edgecolor='white', linewidth=0.3)
    ax.set_title(f'{CH_LABEL[ch]}  (n={len(F)})', loc='left'); ax.set_xlabel('posterior on access family'); ax.margins(x=0.02)
    ax.text(0.03, 0.95, f'mean {F[:, 2].mean():.2f}\nargmax correct {(F.argmax(1) == 2).mean():.2f}', transform=ax.transAxes, va='top', fontsize=6.5)
axes[0].set_ylabel('worlds')
fig.suptitle('Access-protecting generator (γ=2, partner honest): observer posterior on the access family by channel', x=0.01, ha='left', fontsize=8)
fig.tight_layout(); fig.savefig(ru.FIG / 'H2_access_posterior_hist.png', dpi=200); plt.close(fig)
print(json.dumps(p3, indent=1))
