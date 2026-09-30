"""E2 figures. Run from the python kernel with figure-style helpers loaded:
   exec(open('reciprocal_dyad/E2_figures.py').read())   (cwd = workspace root)
Reads results/E2a_recovery.csv, E2b_labels.csv, E2a_channel_contrasts.csv; writes figures/E2_*.png (200 dpi)."""
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from pathlib import Path

RD = Path('reciprocal_dyad'); RES = RD / 'results'; FIG = RD / 'figures'; FIG.mkdir(exist_ok=True)
try:
    apply_figure_style(sizes=(9, 8, 7))
except NameError:
    pass
dfa = pd.read_csv(RES / 'E2a_recovery.csv'); dfb = pd.read_csv(RES / 'E2b_labels.csv'); dfs = pd.read_csv(RES / 'E2a_sensitivity.csv')
GEN_RU = {'honestxhonest': 'честный (A) × честный (B)', 'attenuatedxhonest': 'ослабленный × честный', 'costlyxhonest': 'затратный × честный',
          'mixedxhonest': 'смешанный × честный', 'costlyxcostly': 'затратный × затратный', 'attenuatedxattenuated': 'ослабл. × ослабл.', 'mixedxmixed': 'смеш. × смеш.'}
NOISE_RU = {'none': 'без шума', 'engine': 'шум движка (129/169)', 'baseline': 'шум прямого промпта (145/169)'}
C_PRIV = '#1b4f72'; C_ACTS = '#e67e22'; C_NOISE = {'none': '#1b4f72', 'engine': '#c0392b', 'baseline': '#7f8c8d'}

# ---------------------------------------------------------------- Figure 1: labels gain (E2b)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
for ax, a_type, title in zip(axes, ('attenuated', 'costly'), ('Ослабленный агент A (ω=0.2, c=0)', 'Затратный агент A (ω=1, c=1.6)')):
    sub = dfb[dfb.A_type == a_type]
    acts_cf = sub.acts_correct_family.iloc[0]; priv_cf = sub.acts_private_correct_family.iloc[0]
    ax.axhline(priv_cf, color=C_PRIV, ls='--', lw=1); ax.axhline(acts_cf, color=C_ACTS, ls=':', lw=1.2)
    for nname in ('none', 'baseline', 'engine'):
        s = sub[sub.noise == nname].sort_values('g')
        se = np.sqrt(s.correct_family * (1 - s.correct_family) / s.n_worlds)
        ax.errorbar(s.g, s.correct_family, yerr=1.96 * se, color=C_NOISE[nname], marker='o', ms=4, lw=1.4, capsize=2, label=NOISE_RU[nname])
    ax.set_title(title, loc='left'); ax.set_xlabel('сила модуляции меток g'); ax.set_xticks([0, 0.25, 0.5, 1.0])
    ax.margins(x=0.08); ax.set_ylim(0.4, 1.02)
    ax.text(1.02, priv_cf, 'акты + приватные', color=C_PRIV, va='bottom', ha='right', fontsize=7)
    ax.text(1.02, acts_cf, 'только акты', color=C_ACTS, va='top', ha='right', fontsize=7)
axes[0].set_ylabel('доля миров с верным семейством')
axes[1].legend(loc='lower right', frameon=False, title='метки z, аннотационный шум', fontsize=7, title_fontsize=7)
fig.tight_layout(); fig.savefig(FIG / 'E2_labels_gain.png', dpi=200)

# ---------------------------------------------------------------- Figure 2: acts recovery (E2a)
gens = [g for g in GEN_RU if g in set(dfa.generator)]
fig2, axes2 = plt.subplots(1, 3, figsize=(9.6, 3.7), gridspec_kw={'width_ratios': [1.3, 1, 1], 'wspace': 0.12})
ax = axes2[0]; y = np.arange(len(gens))
for k, (ch, col, mk) in enumerate((('acts+private', C_PRIV, 'o'), ('acts', C_ACTS, 's'))):
    s = dfa.set_index(['generator', 'channel']).loc[[(g, ch) for g in gens]]
    v = s.posterior_on_family.to_numpy(); cf = s.correct_family.to_numpy(); pa = s.p_attenuation_mean.to_numpy()
    off = -0.15 + 0.3 * k
    ax.scatter(cf, y + off, color=col, marker=mk, s=28, label=f'{"акты + приватные" if ch != "acts" else "только акты"}: доля верных', zorder=3)
    ax.scatter(v, y + off, facecolor='none', edgecolor=col, marker=mk, s=28, label='… средний постериор', zorder=3)
    und = np.isnan(cf)
    ax.scatter(pa[und], y[und] + off, color=col, marker='x', s=26, zorder=3)
ax.axvline(0.5, color='0.6', lw=0.8, ls=':'); ax.set_yticks(y); ax.set_yticklabels([GEN_RU[g] for g in gens]); ax.invert_yaxis()
ax.set_xlim(0, 1.05); ax.set_xlabel('верное семейство: доля миров / постериор\n(× = P(ослабление) при неопределённом семействе)'); ax.set_title('Семейство механизма A по актам диады', loc='left')
ax.legend(frameon=False, fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.36), ncol=2)
for ax, par, lab in ((axes2[1], 'omega', 'ω'), (axes2[2], 'c', 'c')):
    for k, (ch, col, mk) in enumerate((('acts+private', C_PRIV, 'o'), ('acts', C_ACTS, 's'))):
        s = dfa.set_index(['generator', 'channel']).loc[[(g, ch) for g in gens]]
        ax.scatter(s[f'{par}_MAE'], y - 0.15 + 0.3 * k, color=col, marker=mk, s=28, zorder=3)
    s0 = dfa.set_index(['generator', 'channel']).loc[[(g, 'acts') for g in gens]]
    ax.scatter(s0[f'{par}_MAE_prior'], y, marker='|', color='k', s=80, zorder=2, label=f'MAE при априорном среднем {lab}')
    ax.set_yticks(y); ax.set_yticklabels([]); ax.invert_yaxis(); ax.set_xlabel(f'MAE оценки {lab} (ниже = лучше)'); ax.set_xlim(-0.04, 0.9); ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_title(f'Ошибка параметра {lab}', loc='left'); ax.legend(frameon=False, fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.26))
fig2.subplots_adjust(left=0.17, right=0.99, top=0.9, bottom=0.36, wspace=0.12); fig2.savefig(FIG / 'E2_acts_recovery.png', dpi=200)

# ---------------------------------------------------------------- Figure 3: p_attenuation distributions for mixed / honest (family undefined)
fig3, axes3 = plt.subplots(1, 3, figsize=(8.4, 2.7), sharey=True)
import numpy as _np
for ax, gen in zip(axes3, ('honestxhonest', 'mixedxhonest', 'mixedxmixed')):
    for ch, col in (('acts+private', C_PRIV), ('acts', C_ACTS)):
        z = _np.load(RES / 'E2_cache' / f'E2a_{gen}_{ch}.npz')['p_attenuation']
        ax.hist(z, bins=np.linspace(0, 1, 21), histtype='step', color=col, lw=1.4, label='акты + приватные' if ch != 'acts' else 'только акты')
    ax.axvline(0.5, color='0.6', lw=0.8, ls=':'); ax.set_title(GEN_RU[gen], loc='left'); ax.set_xlabel('P(семейство «ослабление» | данные)')
axes3[0].set_ylabel('число миров'); axes3[1].legend(frameon=False, fontsize=7, loc='upper right')
fig3.tight_layout(); fig3.savefig(FIG / 'E2_family_posterior_undefined.png', dpi=200)
print('figures written')
