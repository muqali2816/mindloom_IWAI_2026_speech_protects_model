"""E5 figure: costly - attenuated contrasts (partner honest) across the 36 main settings + 2 rho settings.
PYTHONPATH=. python E5_figures.py (figure-style helpers injected via exec or stubbed)."""
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent; FIG = ROOT / 'figures'; FIG.mkdir(exist_ok=True); RES = ROOT / 'results'
con = pd.read_csv(RES / 'E5_contrasts.csv')
ca = con[con.contrast == 'costlyxhonest-attenuatedxhonest']
K_COL = {1.0: '#762a83', 2.0: '#1b7837'}; R_MK = {12: 'o', 24: 's'}; LAM_OFF = {0.6: -0.22, 0.7: 0.0, 0.85: 0.22}
TAU_X = {0.1: 0, 0.2: 1, 0.4: 2}
SPLIT_LAB = {'all': 'all worlds (n = 2000)', 'A_own0': 'A starts believing its own version'}
MET = [('A_dbelief', 'private belief change |Δq|,\ncostly − attenuated'), ('nonc_A', 'late non-concession,\ncostly − attenuated'),
       ('B_pC_true', "listener B's mass on A's true cost,\ncostly − attenuated")]


def fig_sensitivity(out):
    fig, axes = plt.subplots(len(MET), 2, figsize=(7.6, 7.2), sharex=True, gridspec_kw={'hspace': 0.25, 'wspace': 0.25})
    for ci, split in enumerate(['all', 'A_own0']):
        for ri, (met, ylab) in enumerate(MET):
            ax = axes[ri, ci]
            d = ca[(ca.metric == met) & (ca.split == split)]
            for _, r in d[d.grid == 'main'].iterrows():
                x = TAU_X[r.tau] + LAM_OFF[r.lam] + (0.05 if r.rounds == 24 else -0.05)
                ax.errorbar(x, r.delta, yerr=[[r.delta - r.mc_lo], [r.mc_hi - r.delta]], fmt=R_MK[int(r.rounds)], ms=3.2,
                            color=K_COL[r.k], mfc=K_COL[r.k] if r.rounds == 12 else 'white', mew=0.9, elinewidth=0.6, capsize=0, lw=0)
            for _, r in d[d.grid == 'extra_rho'].iterrows():
                x = 3 if r.rho < 0.75 else 3.6
                ax.errorbar(x, r.delta, yerr=[[r.delta - r.mc_lo], [r.mc_hi - r.delta]], fmt='D', ms=3.2, color='0.35', elinewidth=0.6, capsize=0, lw=0)
            ax.axhline(0, color='0.2', lw=0.6)
            ax.axvline(2.6, color='0.8', lw=0.6, ls=':')
            if ci == 0: ax.set_ylabel(ylab)
            if ri == 0: ax.set_title(SPLIT_LAB[split], loc='left')
            ax.set_xticks([0, 1, 2, 3, 3.6]); ax.set_xticklabels(['τ = 0.1', 'τ = 0.2', 'τ = 0.4', 'ρ\n0.65', 'ρ\n0.85'])
            ax.margins(y=0.08); ax.set_xlim(-0.55, 3.95)
            if ri == len(MET) - 1: ax.set_xlabel('choice temperature τ' if ci == 0 else 'ρ varied at default τ, k, λ, 12 rounds')
    h = [mpl.lines.Line2D([], [], color=K_COL[1.0], marker='o', lw=0, ms=3.5, label='k = 1'),
         mpl.lines.Line2D([], [], color=K_COL[2.0], marker='o', lw=0, ms=3.5, label='k = 2'),
         mpl.lines.Line2D([], [], color='0.3', marker='o', lw=0, ms=3.5, label='12 rounds'),
         mpl.lines.Line2D([], [], color='0.3', marker='s', mfc='white', lw=0, ms=3.5, label='24 rounds')]
    fig.text(0.5, 0.005, 'within each τ: λ = 0.6, 0.7, 0.85 left→right; filled circles 12 rounds, open squares 24 rounds', ha='center', fontsize=6.5, color='0.3')
    axes[0, 1].legend(handles=h, frameon=False, loc='lower right', ncol=2, fontsize=6.5, handletextpad=0.3, columnspacing=1.0)
    for ax, L in zip(axes.ravel(), 'abcdef'):
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
    fig_sensitivity(FIG / 'E5_sensitivity.png')
