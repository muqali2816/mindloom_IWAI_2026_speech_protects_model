"""H1 figures from results/H1_*.csv -> figures/H1_*.png (200 dpi, English labels)."""
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import fig_style as fs

ROOT = Path(__file__).resolve().parent; RES = ROOT / 'results'; FIG = ROOT / 'figures'; FIG.mkdir(exist_ok=True)
fs.style(); TYPES = fs.TYPES; COL = fs.COL

# ------------------------------------------------------------------ H1_calibration.png
cal = pd.read_csv(RES / 'H1_calibration.csv'); ch = json.load(open(RES / 'H1_calibration_choice.json'))
_S = pd.read_csv(RES / 'H1_mechanisms_summary.csv'); hon = _S[(_S.set == 'typical') & (_S.run == 'honest') & (_S.split == 'A_own0')].iloc[0].rename({'nonc_A': 'nonc_A_own0'})
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
for ax, (mech, xl) in zip(axes, [('attenuated', 'ω (evidence sensitivity)'), ('costly', 'c (concession cost)'), ('access', 'γ (contradiction aversion)')]):
    s = cal[cal.mechanism == mech]
    ax.errorbar(s.value, s.nonc_A_own0, yerr=1.96 * s.nonc_A_own0_se, fmt='o-', color=COL[mech], ms=4, lw=1.2, capsize=2)
    ax.axhline(ch['target'], color='0.3', lw=0.8, ls='--'); ax.axhline(hon.nonc_A_own0, color=COL['honest'], lw=0.8, ls=':')
    typ = s[s.typical == 1].iloc[0]; ax.plot(typ.value, typ.nonc_A_own0, 'o', mfc='none', mec='k', ms=9, mew=1)
    if mech == 'access': ax.set_xscale('log', base=2); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(['1', '2', '4', '8'])
    ax.set_xlabel(xl); ax.set_title(fs.TYPE_SHORT[mech] + ' A, honest B'); ax.margins(x=0.15, y=0.1)
axes[0].set_ylabel('late non-concession of A\n(rounds 9–12, A starts on own side)')
axes[2].text(1.0, ch['target'] - 0.035, 'target = median of typical (%.3f)' % ch['target'], fontsize=6, color='0.3')
axes[2].text(1.0, hon.nonc_A_own0 - 0.03, 'honest A (%.2f)' % hon.nonc_A_own0, fontsize=6, color=COL['honest'])
axes[0].set_ylim(0.72, 1.02)
for ax, L in zip(axes, 'abc'): fs.panel_letter(ax, L, dx=-0.04)
fig.tight_layout(); fig.savefig(FIG / 'H1_calibration.png'); print('cal overlaps', fs.check_overlaps(fig)); plt.close(fig)

# ------------------------------------------------------------------ H1_channels.png (central result)
S = pd.read_csv(RES / 'H1_mechanisms_summary.csv'); S = S[S.set == 'typical']
panels = [('nonc_A', 'late non-concession of A\n(rounds 9–12)', 'public outcome'), ('evidence_count', 'shared signals per dialogue', 'access channel'),
          ('restrict_A', 'restriction rate of A\n(share of moves with f=0)', 'form channel'), ('A_dbelief', '|Δ private belief| of A', 'sensitivity channel'),
          ('A_hidden_shift', 'hidden shift of A\n(|Δ belief| while position kept)', 'cost channel'), ('A_ASK', 'ASK rate of A', 'act channel')]
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
for ax, (m, yl, ttl), L in zip(axes.ravel(), panels, 'abcdef'):
    for k, split in enumerate(('A_own0', 'all')):
        s = S[S.split == split].set_index('run').loc[TYPES]
        xs = np.arange(4) + (0 if split == 'A_own0' else 0.28)
        for x, t in zip(xs, TYPES):
            ax.errorbar(x, s.loc[t, m], yerr=1.96 * s.loc[t, m + '_se'], fmt='o' if split == 'A_own0' else 's', color=COL[t], ms=5 if split == 'A_own0' else 3.5,
                        mfc=COL[t] if split == 'A_own0' else 'white', lw=1, capsize=2)
    ax.set_xticks(np.arange(4) + 0.14); ax.set_xticklabels([fs.TYPE_SHORT[t] for t in TYPES], rotation=20, ha='right')
    ax.set_ylabel(yl); ax.set_title(ttl); ax.margins(x=0.12, y=0.15); fs.panel_letter(ax, L, dx=-0.04)
axes[0, 0].plot([], [], 'o', color='0.3', label='A starts on own side (n=1381)'); axes[0, 0].plot([], [], 's', mfc='white', color='0.3', ms=3.5, label='all worlds (n=2000)')
fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=6, handlelength=1)
fig.suptitle('Speaker A with an honest partner: the three mechanisms separate on different channels', x=0.01, ha='left', fontsize=8)
fig.tight_layout(rect=(0, 0.04, 1, 0.96)); fig.savefig(FIG / 'H1_channels.png'); print('channels overlaps', fs.check_overlaps(fig)); plt.close(fig)

# ------------------------------------------------------------------ H1_phase_map.png
Ph = pd.read_csv(RES / 'H1_phase_map.csv'); Ph = Ph[Ph.beta == 1.5]
mets = [('nonc_mutual', 'mutual late non-concession', 'Greys', None), ('evidence_count', 'shared signals per dialogue', 'Blues', None), ('belief_gap_final', 'final belief gap |q_A − q_B|', 'Purples', None)]
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
for r, split in enumerate(('all', 'both_own')):
    for c, (m, ttl, cmap, _) in enumerate(mets):
        ax = axes[r, c]; p = Ph[Ph.split == split].pivot(index='A', columns='B', values=m).loc[TYPES, TYPES].to_numpy()
        vmin, vmax = Ph[m].min(), Ph[m].max()
        im = ax.imshow(p, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        for i in range(4):
            for j in range(4):
                v = p[i, j]; ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6, color='white' if (v - vmin) / (vmax - vmin + 1e-9) > 0.6 else 'black')
        ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels([fs.TYPE_SHORT[t] for t in TYPES], rotation=30, ha='right'); ax.set_yticklabels([fs.TYPE_SHORT[t] for t in TYPES])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(length=0)
        if r == 0: ax.set_title(ttl)
        if c == 0: ax.set_ylabel(('all worlds' if split == 'all' else 'both start on own side') + f' (n={int(Ph[Ph.split == split].n.iloc[0])})' + '\nspeaker A type')
        if r == 1: ax.set_xlabel('partner B type')
        fs.panel_letter(ax, 'abcdef'[r * 3 + c], dx=-0.04)
fig.suptitle('Phase map of agent types (protocol-typical parameters, β=1.5)', x=0.01, ha='left', fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(FIG / 'H1_phase_map.png'); print('phase overlaps', fs.check_overlaps(fig)); plt.close(fig)

# ------------------------------------------------------------------ H1_trajectories.png
Tr = pd.read_csv(RES / 'H1_trajectories.csv')
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
for t in TYPES:
    s = Tr[(Tr.run == t) & (Tr.split == 'A_own0')].sort_values('round')
    axes[0].plot(s['round'], s.A_p_truth, '-o', color=COL[t], ms=2.5, lw=1.2, label=fs.TYPE_SHORT[t])
    axes[0].fill_between(s['round'], s.A_p_truth - 1.96 * s.A_p_truth_se, s.A_p_truth + 1.96 * s.A_p_truth_se, color=COL[t], alpha=0.15, lw=0)
    s1 = s[s['round'] >= 1]
    axes[1].plot(s1['round'], s1.cum_signals, '-o', color=COL[t], ms=2.5, lw=1.2)
    axes[2].plot(s1['round'], s1.A_conceded_by, '-o', color=COL[t], ms=2.5, lw=1.2)
axes[0].set_ylabel('P(true version) of A'); axes[0].set_xlabel('round (0 = start)'); axes[0].set_title('private belief in the truth')
axes[1].set_ylabel('cumulative shared signals'); axes[1].set_xlabel('round'); axes[1].set_title('signals accumulated')
axes[2].set_ylabel('P(A has conceded by round)'); axes[2].set_xlabel('round'); axes[2].set_title('public concession')
axes[0].legend(loc='lower right', fontsize=6, handlelength=1.5)
for ax, L in zip(axes, 'abc'): fs.panel_letter(ax, L, dx=-0.04); ax.margins(x=0.04, y=0.08); ax.set_xticks(range(0, 13, 2))
axes[1].set_ylim(0, 3.0); axes[2].set_ylim(-0.01, 0.36)
fig.suptitle('Speaker A with an honest partner, worlds where A starts on own side (n=1381); band = 95% CI', x=0.01, ha='left', fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(FIG / 'H1_trajectories.png'); print('traj overlaps', fs.check_overlaps(fig)); plt.close(fig)

# ------------------------------------------------------------------ H1_beta0.png
BC = pd.read_csv(RES / 'H1_beta0_contrasts.csv'); bc = BC[(BC.set == 'beta0_vs_beta1.5') & (BC.split == 'A_own0')]
mets = ['evidence_count', 'restrict_A', 'A_ASK', 'A_p_truth', 'A_dbelief', 'A_hidden_shift', 'nonc_A', 'belief_gap_final']
mlab = ['shared signals', 'restriction rate A', 'ASK rate A', 'P(true) A', '|Δ belief| A', 'hidden shift A', 'non-concession A', 'belief gap']
fig, ax = plt.subplots(figsize=(4.6, 2.8))
for k, t in enumerate(TYPES):
    s = bc[bc.contrast.str.startswith(t)].set_index('metric').loc[mets]
    y = np.arange(len(mets)) + (k - 1.5) * 0.18
    ax.errorbar(s.delta, y, xerr=[s.delta - s.mc_lo, s.mc_hi - s.delta], fmt='o', color=COL[t], ms=3.5, lw=1, capsize=1.5, label=fs.TYPE_SHORT[t])
ax.axvline(0, color='0.5', lw=0.8); ax.set_yticks(range(len(mets))); ax.set_yticklabels(mlab); ax.invert_yaxis()
ax.set_xlabel('paired difference β=0 − β=1.5 (same worlds, 95% MC interval)'); ax.set_title('Removing the information-gain term changes little')
ax.legend(loc='center left', fontsize=6, handlelength=1); ax.margins(y=0.06)
fig.tight_layout(); fig.savefig(FIG / 'H1_beta0.png'); print('beta0 overlaps', fs.check_overlaps(fig)); plt.close(fig)
