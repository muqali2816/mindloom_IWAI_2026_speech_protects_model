"""E3 figures from results/E3_summary.csv and results/E3_power.csv. PYTHONPATH=. python E3_figures.py
Requires the figure-style kernel helpers (apply_figure_style, panel_letter) to be importable or injected via exec."""
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent; FIG = ROOT / 'figures'; FIG.mkdir(exist_ok=True); RES = ROOT / 'results'
summ = pd.read_csv(RES / 'E3_summary.csv'); pw = pd.read_csv(RES / 'E3_power.csv')

PAIR_LAB = {'costlyxhonest': 'costly A × honest B', 'costlyxcostly': 'costly A × costly B',
            'attenuatedxhonest': 'attenuated A × honest B', 'attenuatedxattenuated': 'attenuated A × attenuated B'}
PAIR_COL = {'costlyxhonest': '#b2182b', 'costlyxcostly': '#ef8a62', 'attenuatedxhonest': '#2166ac', 'attenuatedxattenuated': '#67a9cf'}
COND = ['base', 'force_silence', 'reassure', 'placebo', 'remove_c_A', 'remove_c_AB', 'external']
COND_LAB = {'base': 'none', 'force_silence': 'B silent\n(control)', 'reassure': 'B reassures\n(cost lifted)',
            'placebo': 'B reassures,\nplacebo', 'remove_c_A': 'cost off\nfor A', 'remove_c_AB': 'cost off\nfor both', 'external': 'shared\nevidence'}
SPLIT_LAB = {'all': 'all worlds (n = 2000)', 'A_own0': 'A starts believing its own version (n = 1225)'}


def fig_interventions(out):
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.8), gridspec_kw={'width_ratios': [2.4, 1], 'hspace': 0.75, 'wspace': 0.28})
    pairs = list(PAIR_LAB)
    for ri, split in enumerate(['all', 'A_own0']):
        s = summ[summ.split == split]
        ax = axes[ri, 0]
        for pi, pair in enumerate(pairs):
            sp = s[s.pair == pair].set_index('run').reindex(COND)
            x = np.arange(len(COND)) + (pi - 1.5) * 0.17
            ax.errorbar(x, sp['conc_A_8_12'], yerr=1.96 * sp['conc_A_8_12_se'], fmt='o', ms=3.5, color=PAIR_COL[pair],
                        ecolor=PAIR_COL[pair], elinewidth=0.8, capsize=1.5, label=PAIR_LAB[pair])
        ax.set_xticks(np.arange(len(COND))); ax.set_xticklabels([COND_LAB[c] for c in COND])
        ax.set_ylabel('P(A concedes at least once,\nrounds 8–12)')
        ax.set_ylim(-0.02, 0.62 if split == 'all' else 0.26); ax.margins(x=0.04)
        ax.set_title('Cost interventions move only the costly agent;\nshared evidence moves mainly the attenuated one', loc='left')
        ax.set_xlabel('intervention at round 8 (B\'s move or shared)')
        ax.text(0, 1.26, SPLIT_LAB[split], transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')
        if ri == 0:
            ax.legend(frameon=False, loc='upper left', ncol=2, columnspacing=1.0, handletextpad=0.4)
        ax = axes[ri, 1]
        for pi, pair in enumerate(pairs):
            sp = s[s.pair == pair].set_index('run')
            y = [sp.loc['base', 'A_p_truth'], sp.loc['external', 'A_p_truth']]
            e = [1.96 * sp.loc['base', 'A_p_truth_se'], 1.96 * sp.loc['external', 'A_p_truth_se']]
            ax.errorbar([0, 1], y, yerr=e, fmt='o-', ms=3.5, lw=1, color=PAIR_COL[pair], ecolor=PAIR_COL[pair], elinewidth=0.8, capsize=1.5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['without', 'with shared\nevidence']); ax.set_xlim(-0.35, 1.35)
        ax.set_ylabel('A truth-weighted belief, final'); ax.set_ylim(0.76, 0.99)
        ax.set_title('Shared evidence narrows the gap', loc='left')
    for ax, L in zip(axes.ravel(), 'abcd'):
        panel_letter(ax, L)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    return fig


CH = {'tail non-concession A (base)': ('late non-concession, no intervention', '#7f7f7f'),
      'A pressure rate (base)': ('pressure rate, no intervention', '#bdbdbd'),
      'A private belief change |dq| (base)': ('private belief change, no intervention', '#1b7837'),
      'A truth-weighted belief (base)': ('truth-weighted belief, no intervention', '#a6dba0'),
      'A concession in 8-12 after cost removal (remove_c_A)': ('concession after cost removal', '#e08214'),
      'A concession in 9-12 after partner reassure (reassure)': ('concession after partner reassures', '#fdb863'),
      'A truth-weighted belief after external evidence (external)': ('truth-weighted belief after shared evidence', '#8073ac'),
      'listener: P_B(c_A = 1.6) (base)': ('listener B: posterior P(c_A = 1.6)', '#b2182b')}


def fig_power(out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True, gridspec_kw={'wspace': 0.08})
    for ax, split in zip(axes, ['all', 'A_own0']):
        s = pw[pw.split == split]
        for ch, (lab, col) in CH.items():
            d = s[s.channel == ch].sort_values('n_per_group')
            ax.plot(d['n_per_group'], d['power'], '-o', ms=2.5, lw=1.2, color=col, label=lab)
        ax.axhline(0.8, color='0.3', lw=0.6, ls=':')
        ax.set_xscale('log'); ax.set_xticks([8, 17, 35, 70, 140, 400]); ax.set_xticklabels(['8', '17', '35', '70', '140', '400'])
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_xlabel('simulated dialogues per group'); ax.set_ylim(-0.03, 1.05); ax.margins(x=0.06)
        ax.set_title(SPLIT_LAB[split], loc='left')
    axes[0].set_ylabel('power to separate attenuated vs costly A\n(Welch t, α = 0.05, 400 resamples)')
    axes[0].text(400, 0.82, 'power 0.8', ha='right', va='bottom', fontsize=6, color='0.3')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=6.5, handlelength=1.8, columnspacing=1.5)
    for ax, L in zip(axes, 'ab'):
        panel_letter(ax, L)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    return fig


if __name__ == '__main__':
    try:
        apply_figure_style
    except NameError:
        def apply_figure_style(**k): pass
        def panel_letter(ax, L, **k): ax.text(-0.12, 1.06, L, transform=ax.transAxes, fontweight='bold', fontsize=10)
    apply_figure_style()
    fig_interventions(FIG / 'E3_interventions.png'); fig_power(FIG / 'E3_power.png')
