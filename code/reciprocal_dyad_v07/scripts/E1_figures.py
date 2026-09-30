"""E1 figures from results/E1_*.csv. Run inside the analysis kernel (figure-style helpers) or standalone:
   exec(open('reciprocal_dyad/E1_figures.py').read())   |   PYTHONPATH=. python E1_figures.py
Writes figures/E1_phase_map_beta1.5.png, figures/E1_phase_map_beta0.0.png, figures/E1_trajectories.png, figures/E1_recovery.png
"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent if '__file__' in globals() else Path('reciprocal_dyad').resolve()
RES = ROOT / 'results'; FIG = ROOT / 'figures'; FIG.mkdir(exist_ok=True)
try:
    apply_figure_style(sizes=(8, 7, 6))          # figure-style kernel helper
except NameError:
    mpl.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
                         'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
    def panel_letter(ax, letter, dx=-0.18, dy=1.02, **kw):
        ax.text(dx, dy, letter, transform=ax.transAxes, fontweight='bold', fontsize=10, ha='left', va='bottom')

TYPES = ['honest', 'attenuated', 'costly', 'mixed']
RU = {'honest': 'честный', 'attenuated': 'ослабл.', 'costly': 'ценовой', 'mixed': 'смешан.'}
TYPES_NOTE = 'честный: ω=1, c=0; ослабленный: ω=0.2, c=0; ценовой: ω=1, c=1.6; смешанный: ω=0.6, c=0.8'
RU1 = {'honest': 'честный', 'attenuated': 'ослабленный', 'costly': 'ценовой', 'mixed': 'смешанный'}
COL = {'honest': '#4d4d4d', 'costly': '#b2182b', 'attenuated': '#2166ac', 'mixed': '#7b3294', 'costlyxattenuated': '#e08214'}
mat = pd.read_csv(RES / 'E1_pairs_matrix.csv'); traj = pd.read_csv(RES / 'E1_trajectories.csv')


def heat(ax, s, col, cmap, vmin, vmax, title, cbar_label, fmt='{:.2f}', ylab=True):
    M = s.pivot(index='A_type', columns='B_type', values=col).loc[TYPES, TYPES].to_numpy()
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels([RU[t] for t in TYPES]); ax.set_yticklabels([RU[t] for t in TYPES])
    ax.set_xlabel('тип агента B (за версию 0)'); ax.set_ylabel('тип агента A (за версию 1)' if ylab else '')
    for i in range(4):
        for j in range(4):
            v = M[i, j]; rgba = im.cmap(im.norm(v)); lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, fmt.format(v), ha='center', va='center', color='white' if lum < 0.5 else 'black', fontsize=7)
    ax.set_title(title, loc='left'); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label(cbar_label, fontsize=6); cb.outline.set_visible(False)
    return M


def phase_map(beta, split='both_own'):
    s = mat[(mat.beta == beta) & (mat.split == split)]; n = int(s.n.iloc[0])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    heat(axes[0], s, 'nonc_mutual', 'Reds', 0, 1, 'Взаимное непризнание в хвосте (раунды 9–12)', 'доля раундов без уступки обоих')
    heat(axes[1], s, 'hidden_shift_mean', 'Purples', 0, 0.3, 'Скрытый сдвиг убеждения, позиция не менялась', '(A+B)/2 · |Δq| при pos(12) = pos(0)', ylab=False)
    heat(axes[2], s, 'belief_gap_final', 'Blues', 0, 0.65, 'Итоговый разрыв убеждений', '|q_A(12) − q_B(12)|', ylab=False)
    for ax, L in zip(axes, 'abc'): panel_letter(ax, L, dx=-0.22 if L == 'a' else -0.1)
    fig.text(0.01, -0.02, TYPES_NOTE, fontsize=6, ha='left', va='top')
    fig.suptitle(f'E1: фазовая карта пар механизмов, β = {beta}, миры, где оба стартуют со своей версии (n = {n} из 2000)' if split == 'both_own'
                 else f'E1: фазовая карта пар механизмов, β = {beta}, все миры (n = {n})', x=0.0, ha='left', fontsize=8)
    out = FIG / f'E1_phase_map_beta{beta}{"" if split == "both_own" else "_all"}.png'; fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig); return out


def trajectories(beta=1.5, split='both_own'):
    pairs = [('honest', 'honest'), ('costly', 'costly'), ('attenuated', 'attenuated'), ('costly', 'attenuated')]
    names = {('honest', 'honest'): 'честный × честный', ('costly', 'costly'): 'ценовой × ценовой', ('attenuated', 'attenuated'): 'ослабленный × ослабленный', ('costly', 'attenuated'): 'ценовой × ослабленный'}
    cols = {('honest', 'honest'): COL['honest'], ('costly', 'costly'): COL['costly'], ('attenuated', 'attenuated'): COL['attenuated'], ('costly', 'attenuated'): COL['costlyxattenuated']}
    t = traj[(traj.beta == beta) & (traj.split == split)]; n = int(mat[(mat.beta == beta) & (mat.split == split)].n.iloc[0])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharex=False, constrained_layout=True)
    for (a, b) in pairs:
        x = t[(t.A_type == a) & (t.B_type == b)]
        for ax, met in zip(axes, ('A_p_truth_t', 'B_p_truth_t', 'nonc_mutual_t')):
            y = x[x.metric == met].sort_values('round')
            ax.plot(y['round'], y['value'], color=cols[(a, b)], lw=1.8 if (a, b) != ('honest', 'honest') else 1.2, marker='o', ms=2.5, label=names[(a, b)])
    axes[0].set_title('Агент A: вероятность истинной версии', loc='left'); axes[1].set_title('Агент B: вероятность истинной версии', loc='left')
    axes[2].set_title('Доля миров, где в раунде никто не уступил', loc='left')
    axes[0].set_ylabel('P(истинная версия)'); axes[2].set_ylabel('доля миров')
    for ax in axes[:2]: ax.set_xlabel('раунд (0 = до диалога)'); ax.set_ylim(0.5, 0.8); ax.set_xticks(range(0, 13, 2))
    axes[2].set_xlabel('раунд'); axes[2].set_ylim(0, 1.02); axes[2].set_xticks(range(1, 13, 1))
    axes[1].sharey(axes[0]); axes[1].tick_params(labelleft=False)
    axes[2].legend(loc='center left', bbox_to_anchor=(0.02, 0.35), frameon=False, handlelength=1.5)
    for ax, L in zip(axes, 'abc'): panel_letter(ax, L, dx=-0.16); ax.margins(x=0.04)
    fig.suptitle(f'E1: траектории по раундам, β = {beta}, миры, где оба стартуют со своей версии (n = {n})', x=0.01, ha='left', fontsize=8)
    fig.text(0.01, -0.02, TYPES_NOTE, fontsize=6, ha='left', va='top')
    out = FIG / 'E1_trajectories.png'; fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig); return out


def recovery(beta=1.5, split='both_own'):
    s = mat[(mat.beta == beta) & (mat.split == split)]; n = int(s.n.iloc[0])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    heat(axes[0], s, 'A_transp_err', 'Greys', 0, 0.25, 'Ошибка A в оценке убеждения B', '|E_A[q_B] − q_B|, раунд 12')
    heat(axes[1], s, 'A_pW_true', 'RdBu', 0, 2 / 3, 'Масса A на истинном ω партнёра', 'P_A(ω_B = истинное); белый = априор 1/3', ylab=False)
    heat(axes[2], s, 'A_pC_true', 'RdBu', 0, 2 / 3, 'Масса A на истинном c партнёра', 'P_A(c_B = истинное); белый = априор 1/3', ylab=False)
    for ax, L in zip(axes, 'abc'): panel_letter(ax, L, dx=-0.22 if L == 'a' else -0.1)
    fig.text(0.01, -0.02, TYPES_NOTE, fontsize=6, ha='left', va='top')
    fig.suptitle(f'E1/P5: прозрачность и восстановление механизма партнёра агентом A, β = {beta}, оба стартуют со своей версии (n = {n})', x=0.01, ha='left', fontsize=8)
    out = FIG / 'E1_recovery.png'; fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig); return out


outs = [phase_map(1.5), phase_map(0.0), phase_map(1.5, 'all'), trajectories(), recovery()]
print([o.name for o in outs])
