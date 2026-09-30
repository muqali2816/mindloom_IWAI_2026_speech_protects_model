"""E4 figures from results/E4_*.csv (run inside the analysis kernel with figure-style helpers, or standalone).
Writes figures/E4_persistence.png, E4_transitions.png, E4_hysteresis.png, E4_forecast.png
"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent if '__file__' in globals() else Path('reciprocal_dyad').resolve()
RES = ROOT / 'results'; FIG = ROOT / 'figures'; FIG.mkdir(exist_ok=True)
try:
    apply_figure_style(sizes=(8, 7, 6))
except NameError:
    mpl.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
                         'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
    def panel_letter(ax, letter, dx=-0.18, dy=1.02, **kw):
        ax.text(dx, dy, letter, transform=ax.transAxes, fontweight='bold', fontsize=10, ha='left', va='bottom')

TYPES = ['honest', 'attenuated', 'costly', 'mixed']
RU = {'honest': 'честный', 'attenuated': 'ослабл.', 'costly': 'ценовой', 'mixed': 'смешан.'}
TYPES_NOTE = 'честный: ω=1, c=0; ослабленный: ω=0.2, c=0; ценовой: ω=1, c=1.6; смешанный: ω=0.6, c=0.8. R = 24 раунда, seed E4, 2000 миров'
PAIRS = [('honest', 'honest'), ('costly', 'costly'), ('attenuated', 'attenuated'), ('mixed', 'mixed')]
PNAME = {('honest', 'honest'): 'честный × честный', ('costly', 'costly'): 'ценовой × ценовой', ('attenuated', 'attenuated'): 'ослабленный × ослабленный', ('mixed', 'mixed'): 'смешанный × смешанный'}
PCOL = {('honest', 'honest'): '#4d4d4d', ('costly', 'costly'): '#b2182b', ('attenuated', 'attenuated'): '#2166ac', ('mixed', 'mixed'): '#7b3294'}
STATE_RU = ['оба не уступают', 'уступил один', 'уступили оба']; STATE_COL = ['#b2182b', '#fdb863', '#5e3c99']
pers = pd.read_csv(RES / 'E4_persistence.csv'); rl = pd.read_csv(RES / 'E4_runlengths.csv'); tr = pd.read_csv(RES / 'E4_transitions.csv')
occ = pd.read_csv(RES / 'E4_occupancy.csv'); hy = pd.read_csv(RES / 'E4_hysteresis.csv'); fc = pd.read_csv(RES / 'E4_forecast.csv')
BETA = 1.5; SP = 'both_own'


def heat(ax, s, col, cmap, vmin, vmax, title, cbar_label, ylab=True, fmt='{:.2f}'):
    M = s.pivot(index='A_type', columns='B_type', values=col).loc[TYPES, TYPES].to_numpy()
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels([RU[t] for t in TYPES]); ax.set_yticklabels([RU[t] for t in TYPES])
    ax.set_xlabel('тип агента B (за версию 0)'); ax.set_ylabel('тип агента A (за версию 1)' if ylab else '')
    for i in range(4):
        for j in range(4):
            v = M[i, j]; rgba = im.cmap(im.norm(v)); lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, fmt.format(v), ha='center', va='center', color='white' if lum < 0.5 else 'black', fontsize=7)
    ax.set_title(title, loc='left'); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label(cbar_label, fontsize=6); cb.outline.set_visible(False)


def persistence():
    s = pers[(pers.beta == BETA) & (pers.split == SP)]; n = int(s.n.iloc[0])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    heat(axes[0], s, 'P_stay_MN_tail8', 'Reds', 0.5, 1, 'Персистентность взаимного непризнания, раунды 17–24', 'P(оба не уступают в r+1 | оба не уступают в r)')
    heat(axes[1], s, 'share_maxrun_ge6', 'Reds', 0, 1, 'Доля миров с серией непризнания ≥ 6 раундов', 'доля миров', ylab=False)
    ax = axes[2]
    for p in PAIRS:
        x = rl[(rl.beta == BETA) & (rl.split == SP) & (rl.A_type == p[0]) & (rl.B_type == p[1])].sort_values('run_length')
        cdf = np.cumsum(x.n_worlds_maxrun.to_numpy()) / x.n_worlds_maxrun.sum()
        ax.step(x.run_length, cdf, where='post', color=PCOL[p], lw=1.6, label=PNAME[p])
    ax.axvline(6, color='0.6', lw=0.8, ls=':'); ax.text(6.3, 0.05, 'порог 6', fontsize=6, color='0.4')
    ax.set_xlabel('самая длинная серия «оба не уступают», раундов'); ax.set_ylabel('доля миров (накопленная)')
    ax.set_title('Распределение длины самой длинной серии', loc='left'); ax.set_xlim(-0.5, 24.5); ax.set_ylim(0, 1.02); ax.margins(x=0.03)
    ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.68), frameon=False, handlelength=1.5, fontsize=6)
    for a, L in zip(axes, 'abc'): panel_letter(a, L, dx=-0.22 if L == 'a' else -0.1)
    fig.suptitle(f'E4 (i): персистентность состояния «оба не уступают», β = {BETA}, оба стартуют со своей версии (n = {n})', x=0.0, ha='left', fontsize=8)
    fig.text(0.01, -0.02, TYPES_NOTE, fontsize=6, ha='left', va='top')
    out = FIG / 'E4_persistence.png'; fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig); return out


def transitions():
    n = int(pers[(pers.beta == BETA) & (pers.split == SP)].n.iloc[0])
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2), constrained_layout=True, gridspec_kw={'height_ratios': [1.1, 1]})
    for k, p in enumerate(PAIRS):
        ax = axes[0, k]; o = occ[(occ.beta == BETA) & (occ.split == SP) & (occ.A_type == p[0]) & (occ.B_type == p[1])]
        Y = np.stack([o[o.state == i].sort_values('round').share.to_numpy() for i in range(3)]); r = np.arange(1, Y.shape[1] + 1)
        ax.stackplot(r, Y, colors=STATE_COL, labels=STATE_RU, alpha=0.9, lw=0)
        ax.set_title(PNAME[p], loc='left'); ax.set_xlim(1, 24); ax.set_ylim(0, 1); ax.set_xticks([1, 6, 12, 18, 24])
        if k == 0: ax.set_ylabel('доля миров в состоянии'); ax.legend(loc='lower left', frameon=False, fontsize=6)
        else: ax.tick_params(labelleft=False)
        ax.set_xlabel('раунд')
        ax = axes[1, k]; t = tr[(tr.beta == BETA) & (tr.split == SP) & (tr.A_type == p[0]) & (tr.B_type == p[1]) & (tr.round_from == 0)]
        M = t.pivot(index='from', columns='to', values='prob').loc[[0, 1, 2], [0, 1, 2]].to_numpy(); C = t.pivot(index='from', columns='to', values='count').loc[[0, 1, 2], [0, 1, 2]].to_numpy()
        im = ax.imshow(M, cmap='Greys', vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                if np.isnan(M[i, j]): ax.text(j, i, 'н/д', ha='center', va='center', fontsize=6, color='0.5'); continue
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=7, color='white' if M[i, j] > 0.5 else 'black')
        ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(['оба\nне уступ.', 'один', 'оба\nуступ.']); ax.set_yticklabels(['оба не уступ.', 'один', 'оба уступ.'] if k == 0 else [])
        ax.set_xlabel('состояние в r+1'); ax.tick_params(length=0)
        if k == 0: ax.set_ylabel('состояние в r')
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title('переходы, все раунды', loc='left')
    panel_letter(axes[0, 0], 'a', dx=-0.28); panel_letter(axes[1, 0], 'b', dx=-0.28)
    fig.suptitle(f'E4 (ii): занятость состояний по раундам и матрицы переходов, β = {BETA}, оба стартуют со своей версии (n = {n})', x=0.0, ha='left', fontsize=8)
    fig.text(0.01, -0.02, TYPES_NOTE, fontsize=6, ha='left', va='top')
    out = FIG / 'E4_transitions.png'; fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig); return out


def hysteresis():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
    mets = [('any_concede', 'Доля миров, где в раунде кто-то уступил', 'доля миров'), ('U', 'Публичные позиции расходятся', 'доля миров'), ('belief_gap', 'Разрыв убеждений |q_A − q_B|', '|q_A − q_B|')]
    pairs = [('costly', 'costly'), ('mixed', 'mixed')]; n = None
    for ax, (met, title, yl) in zip(axes, mets):
        ax.axvspan(7.5, 10.5, color='0.9', lw=0, zorder=0)
        ax.text(9, 0.98 if met == 'any_concede' else 0.02, 'c = 0\n(раунды 8–10)', ha='center', va='top' if met == 'any_concede' else 'bottom', fontsize=6, color='0.35', transform=ax.get_xaxis_transform())
        for p in pairs:
            x = hy[(hy.beta == BETA) & (hy.split == SP) & (hy.metric == met) & (hy.A_type == p[0]) & (hy.B_type == p[1])].sort_values('round'); n = int(x.n.iloc[0])
            ax.plot(x['round'], x.control, color=PCOL[p], lw=1.2, ls='--', label=f'{PNAME[p]}: без вмешательства')
            ax.plot(x['round'], x.intervention, color=PCOL[p], lw=1.8, marker='o', ms=2.2, label=f'{PNAME[p]}: c снят на 8–10, возвращён с 11')
        ax.set_title(title, loc='left'); ax.set_xlabel('раунд'); ax.set_ylabel(yl); ax.set_xticks([1, 4, 8, 11, 14, 18, 24]); ax.set_ylim(0, 1.02 if met != 'belief_gap' else 0.7); ax.margins(x=0.03)
    axes[2].legend(loc='upper right', frameon=False, fontsize=6, handlelength=1.8)
    for a, L in zip(axes, 'abc'): panel_letter(a, L, dx=-0.16)
    fig.suptitle(f'E4 (iii): гистерезис при снятии и возврате цены уступки обоим агентам, β = {BETA}, оба стартуют со своей версии (n = {n})', x=0.0, ha='left', fontsize=8)
    fig.text(0.01, -0.02, TYPES_NOTE, fontsize=6, ha='left', va='top')
    out = FIG / 'E4_hysteresis.png'; fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig); return out


def forecast():
    g = fc[fc.group.str.endswith('beta1.5')].copy(); g['pair'] = g.group.str.replace('_beta1.5', '')
    order = [f'{a}x{b}' for a in TYPES for b in TYPES] + ['pooled16']
    lab = {f'{a}x{b}': f'{RU[a]} × {RU[b]}' for a in TYPES for b in TYPES}; lab['pooled16'] = 'все 16 пар вместе'
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True, sharey=True)
    y = np.arange(len(order))
    for ax, (met, xl, ttl) in zip(axes, [('dll', 'Δ средней лог-правдоподобия на прогноз (относительно модели b)', 'Прогноз акта A: прирост лог-правдоподобия'), ('dacc', 'Δ точности (относительно модели b)', 'Прогноз акта A: прирост точности')]):
        for k, col, off, name in (('c', '#e08214', -0.18, 'c: режим + последний акт A'), ('d', '#2166ac', 0.18, 'd: режим + последние акты A и B')):
            s = g[g.model == k].set_index('pair').loc[order]
            ax.errorbar(s[f'{met}_vs_b'], y + off, xerr=[s[f'{met}_vs_b'] - s[f'{met}_vs_b_lo'], s[f'{met}_vs_b_hi'] - s[f'{met}_vs_b']], fmt='o', ms=3.5, color=col, ecolor=col, elinewidth=1, capsize=0, label=name)
        ax.axvline(0, color='0.5', lw=0.8); ax.set_xlabel(xl); ax.set_title(ttl, loc='left'); ax.axhspan(len(order) - 1.5, len(order) - 0.5, color='0.93', lw=0)
    axes[0].set_yticks(y); axes[0].set_yticklabels([lab[o] for o in order]); axes[0].legend(loc='lower left', frameon=False, fontsize=6)
    axes[0].text(0.99, 0.02, 'правее 0 = лучше', transform=axes[0].transAxes, ha='right', fontsize=6, color='0.4')
    for a, L in zip(axes, 'ab'): panel_letter(a, L, dx=-0.3 if L == 'a' else -0.06)
    fig.suptitle('E4 (iv): удержанные 50 % миров (1000), раунды 4–24, β = 1.5; модель b = последние акты A и B; 95 % бутстрэп по мирам', x=0.0, ha='left', fontsize=8)
    fig.text(0.01, -0.02, 'режим = (серия «оба не уступают» ≥ 3 раунда, публичные позиции расходятся); a = последний акт A. ' + TYPES_NOTE, fontsize=6, ha='left', va='top')
    out = FIG / 'E4_forecast.png'; fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig); return out


outs = [persistence(), transitions(), hysteresis(), forecast()]
print([o.name for o in outs])
