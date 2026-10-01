"""Fig. 2 (channels) redrawn at its final printed size: 122 mm wide (LNCS \\textwidth), 6-7 pt lettering.
Same data as figures/H1_channels_reference.png: code/hybrid_dyad_v08/results/H1_mechanisms_summary.csv (set == 'typical').
Usage: python make_fig2_channels.py <path to H1_mechanisms_summary.csv> [outdir]
Include with \\includegraphics[width=\\textwidth]{H1_channels_print.pdf} (do not rescale: lettering is sized for 4.80 in).
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

src = Path(sys.argv[1]); out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('.')
d = pd.read_csv(src); d = d[d['set'] == 'typical']
RUNS = ['honest', 'attenuated', 'costly', 'access']   # 'honest' is the CSV key of the reference agent
XLAB = ['ref.', 'atten.', 'cost', 'access']           # ref. = reference agent (1, 0, 0); spelt out in the caption
COL = {'honest': '#4d4d4d', 'attenuated': '#0072B2', 'costly': '#E69F00', 'access': '#CC79A7'}   # Okabe-Ito
PANELS = [('nonc_A', 'late non-concession'), ('evidence_count', 'shared signals'), ('restrict_A', 'restriction rate'),
          ('A_dbelief', '|Δ private belief|'), ('A_hidden_shift', 'hidden shift'), ('A_ASK', 'ASK rate')]
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5,
                     'ytick.major.width': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 2, 'xtick.major.pad': 1.5,
                     'ytick.major.pad': 1.5, 'pdf.fonttype': 42, 'legend.frameon': False})
fig, axes = plt.subplots(2, 3, figsize=(4.80, 2.12))
for ax, (m, title), letter in zip(axes.ravel(), PANELS, 'abcdef'):
    for i, r in enumerate(RUNS):
        for split, dx, kw in (('A_own0', -0.14, dict(fmt='o', ms=3.2, mfc=COL[r])), ('all', 0.14, dict(fmt='s', ms=2.6, mfc='white'))):
            row = d[(d.run == r) & (d.split == split)].iloc[0]
            ax.errorbar(i + dx, row[m], yerr=1.96 * row[m + '_se'], color=COL[r], mec=COL[r], mew=0.7, lw=0.7, capsize=1.2, capthick=0.6, **kw)
    ax.set_xticks(range(4)); ax.set_xticklabels(XLAB); ax.set_xlim(-0.5, 3.5); ax.margins(y=0.14)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    if m == 'nonc_A':
        ax.set_yticks([0.6, 0.8, 1.0]); ax.set_ylim(0.52, 1.04)
    ax.set_title(title, loc='left', pad=3.5, x=0.11)
    ax.text(0.0, 1.045, letter, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='left')
h = [plt.Line2D([], [], marker='o', ls='', ms=3.2, color='#4d4d4d', label='A starts on own side (n = 1 381)'),
     plt.Line2D([], [], marker='s', ls='', ms=2.6, mfc='white', color='#4d4d4d', mew=0.7, label='all worlds (n = 2 000)')]
fig.legend(handles=h, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, columnspacing=1.5)
fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=0.9, w_pad=0.9)
# geometric overlap check between all text objects
fig.canvas.draw(); rn = fig.canvas.get_renderer()
tx = [(t.get_text(), t.get_window_extent(rn)) for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip() and t.get_visible()]
ov = [(a, b) for i, (a, ba) in enumerate(tx) for b, bb in tx[i + 1:] if ba.overlaps(bb)]
print('text overlaps:', ov)
fig.savefig(out / 'H1_channels_print.pdf'); fig.savefig(out / 'H1_channels_print.png', dpi=400)
print('saved', fig.get_size_inches())
