"""H3 figures from results/H3_*.csv -> figures/H3_*.png (200 dpi, English labels)."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
import fig_style as fs

ROOT = Path(__file__).resolve().parent; RES = ROOT / 'results'; FIG = ROOT / 'figures'; FIG.mkdir(exist_ok=True)
fs.style(); TYPES = fs.TYPES; COL = fs.COL

# ------------------------------------------------------------------ H3_interventions.png
con = pd.read_csv(RES / 'H3_interventions_contrasts.csv'); c = con[con.split == 'A_own0']
IVS = [('reassure-placebo', "partner's REASSURE\n(net of placebo)"), ('remove_c-base', 'remove c of A'), ('remove_gamma-base', 'remove γ of A'),
       ('force_open_B-base', 'force partner open'), ('external_obs-base', 'shared strong evidence\n(ρ=0.95)'), ('bypass-base', 'bypass restriction\n(λ→0.9, confounded)')]
mets = [('P_concede_A_late', 'Δ P(A concedes in rounds 8–12)'), ('evidence_count_late', 'Δ shared signals, rounds 8–12'), ('A_p_truth', 'Δ P(true version) of A, final'),
        ('restrict_A_late', 'Δ restriction rate of A, rounds 8–12')]
fig, axes = plt.subplots(1, 4, figsize=(7.4, 3.2), sharey=True)
for ax, (m, xl) in zip(axes, mets):
    for k, t in enumerate(TYPES):
        s = c[(c.generator == t) & (c.metric == m)].set_index('contrast').loc[[i for i, _ in IVS]]
        y = np.arange(len(IVS)) + (k - 1.5) * 0.19
        ax.errorbar(s.delta, y, xerr=[s.delta - s.mc_lo, s.mc_hi - s.delta], fmt='o', color=COL[t], ms=3.2, lw=0.9, capsize=1.2, label=fs.TYPE_SHORT[t])
    ax.axvline(0, color='0.5', lw=0.8); ax.set_xlabel(xl, fontsize=7); ax.margins(x=0.1, y=0.05)
axes[0].set_yticks(range(len(IVS))); axes[0].set_yticklabels([l for _, l in IVS]); axes[0].invert_yaxis()
axes[1].set_xscale('symlog', linthresh=0.5); axes[1].set_xticks([0, 0.2, 0.4, 1, 2, 4]); axes[1].set_xticklabels(['0', '.2', '.4', '1', '2', '4'])
axes[3].legend(loc='upper left', bbox_to_anchor=(0.02, 0.62), fontsize=6, handlelength=1)
for ax, L in zip(axes, 'abcd'): fs.panel_letter(ax, L, dx=-0.04)
fig.suptitle('Interventions from round 8 (partner honest): paired effect vs. no intervention; worlds where A starts on own side (n=1381); 95% MC interval', x=0.01, ha='left', fontsize=7)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / 'H3_interventions.png'); print('iv overlaps', fs.check_overlaps(fig)); plt.close(fig)

# ------------------------------------------------------------------ H3_dialogues_per_group.png  (n for 80% power, heatmap)
n80 = pd.read_csv(RES / 'H3_power_n80.csv')
CH = [('nonc_A', 'late non-concession'), ('restrict_A', 'restriction rate'), ('evidence_count', 'shared signals'), ('A_dbelief', '|Δ private belief|'),
      ('A_p_truth', 'P(true) final'), ('concede_after_remove_c', 'concession after c removed'), ('concede_after_reassure', "concession after partner's REASSURE"),
      ('evidence_after_force_open_B', 'signals after partner forced open'), ('p_truth_after_external_obs', 'P(true) after strong shared evidence'),
      ('B_pC_high', 'listener: P(c_A = 1.6)'), ('B_pW_low', 'listener: P(ω_A = 0.2)'), ('B_pC_true', 'listener: P(true c_A)'), ('B_pW_true', 'listener: P(true ω_A)')]
PAIRS = ['attenuated vs costly', 'attenuated vs access', 'costly vs access']
NG = [8, 12, 17, 25, 35, 50, 70, 100, 140, 200, 400]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), sharey=True)
cmap = plt.get_cmap('viridis_r', len(NG)); cmap.set_bad('0.92')
for ax, split, L in zip(axes, ('A_own0', 'all'), 'ab'):
    M = n80[n80.split == split].pivot(index='channel', columns='pair', values='n_80').loc[[k for k, _ in CH], PAIRS]
    rank = M.apply(lambda col: col.map({n: i for i, n in enumerate(NG)}))
    ax.imshow(np.ma.masked_invalid(rank.to_numpy(float)), cmap=cmap, vmin=-0.5, vmax=len(NG) - 0.5, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iloc[i, j]; txt = '>400' if np.isnan(v) else f'{int(v)}'
            ax.text(j, i, txt, ha='center', va='center', fontsize=6.5, color='white' if (not np.isnan(v) and rank.iloc[i, j] <= 5) else 'black')
    ax.set_xticks(range(3)); ax.set_xticklabels([p.replace(' vs ', '\nvs ') for p in PAIRS]); ax.set_yticks(range(len(CH)))
    ax.set_yticklabels([l for _, l in CH]); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title('A starts on own side' if split == 'A_own0' else 'all worlds'); fs.panel_letter(ax, L, dx=-0.04)
fig.suptitle('Dialogues per group needed to separate two mechanisms (Welch test, α=0.05, power ≥ 0.8, 400 resamples); dark = fewer', x=0.01, ha='left', fontsize=7)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / 'H3_dialogues_per_group.png'); print('n80 overlaps', fs.check_overlaps(fig)); plt.close(fig)

# ------------------------------------------------------------------ H3_power_curves.png
pw = pd.read_csv(RES / 'H3_power.csv'); pw = pw[pw.split == 'A_own0']
SEL = [('restrict_A', 'restriction rate', '#CC79A7', '-'), ('A_dbelief', '|Δ private belief|', '#0072B2', '-'), ('B_pC_high', 'listener P(c_A=1.6)', '#E69F00', '-'),
       ('nonc_A', 'late non-concession', '#4d4d4d', '-'), ('evidence_count', 'shared signals', '#009E73', '--'), ('concede_after_remove_c', 'concession after c removed', '#D55E00', '--'),
       ('p_truth_after_external_obs', 'P(true) after strong evidence', '#56B4E9', '--')]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
for ax, pr, L in zip(axes, PAIRS, 'abc'):
    for ch, lab, colr, ls in SEL:
        s = pw[(pw.pair == pr) & (pw.channel == ch)].sort_values('n_per_group')
        ax.plot(s.n_per_group, s.power, ls, color=colr, lw=1.2, marker='o', ms=2.5, label=lab)
    ax.axhline(0.8, color='0.6', lw=0.7, ls=':'); ax.set_xscale('log'); ax.set_xticks([8, 25, 70, 200, 400]); ax.set_xticklabels(['8', '25', '70', '200', '400'])
    ax.set_title(pr); ax.set_xlabel('dialogues per group'); ax.margins(y=0.06); fs.panel_letter(ax, L, dx=-0.04)
axes[0].set_ylabel('power (Welch, α=0.05)'); axes[0].set_ylim(-0.03, 1.08); fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=4, fontsize=6, handlelength=1.6, bbox_to_anchor=(0.5, 0))
fig.suptitle('Statistical power to separate mechanisms by channel (worlds where A starts on own side; 400 resamples)', x=0.01, ha='left', fontsize=7)
fig.tight_layout(rect=(0, 0.1, 1, 0.94)); fig.savefig(FIG / 'H3_power_curves.png'); print('power overlaps', fs.check_overlaps(fig)); plt.close(fig)
