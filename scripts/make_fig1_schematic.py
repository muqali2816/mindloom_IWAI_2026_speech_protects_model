"""Fig. 1 (model schematic) drawn at its final printed size: 122 mm wide (LNCS \\textwidth), 6-7 pt lettering, vector PDF.
Content as figures/H0_model_schematic.png minus the act list (acts are listed in the text) and the bottom legend
(duplicated by the caption and eq. (6)). Subscripts via mathtext. No data are read.
Usage: python make_fig1_schematic.py [outdir]
Include with \\includegraphics[width=\\textwidth]{H0_model_schematic_print.pdf} (do not rescale).
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
plt.rcParams.update({'font.size': 7, 'pdf.fonttype': 42, 'mathtext.fontset': 'dejavusans'})
BLUE, ORANGE, AMBER = '#3c6e80', '#b5541f', '#b07a2a'

fig, ax = plt.subplots(figsize=(4.80, 2.2))
ax.set_xlim(0, 12); ax.set_ylim(0.35, 6.45); ax.axis('off')


def box(x, y, w, h, text, fc='#eaf2f6', ec=BLUE, fs=6.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color='#1a1a1a', linespacing=1.3)


def arrow(x0, y0, x1, y1, color=BLUE, ls='-', lw=0.8, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=7, color=color, lw=lw, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))


# agents and the move
ax.set_xlim(0, 12.2); ax.set_ylim(0.3, 6.45)
box(0.1, 4.75, 3.2, 1.5, 'Agent A\nbelief $q_A$\nmechanism $(\\omega_A, c_A, \\gamma_A)$\nposterior over B: $s_B, c_B, \\omega_B$')
box(8.9, 4.75, 3.2, 1.5, 'Agent B\nbelief $q_B$\nmechanism $(\\omega_B, c_B, \\gamma_B)$\nposterior over A: $s_A, c_A, \\omega_A$')
box(3.7, 4.95, 4.6, 1.1, 'each round: act $u$ $\\times$ form $f$\n$f=1$: invite verification, $f=0$: restrict it')
arrow(3.3, 5.5, 3.7, 5.5); arrow(8.9, 5.5, 8.3, 5.5)
arrow(3.95, 4.95, 3.1, 4.75, color=ORANGE); arrow(8.05, 4.95, 8.9, 4.75, color=ORANGE)
ax.text(2.5, 4.5, "B's act is evidence\nfor A (exact update,\nthen $\\times\\omega_A$)", fontsize=6, color=ORANGE, ha='center', va='top', linespacing=1.25)
ax.text(9.55, 4.5, "A's act is evidence\nfor B (exact update,\nthen $\\times\\omega_B$)", fontsize=6, color=ORANGE, ha='center', va='top', linespacing=1.25)
# channel and signal
box(3.7, 2.95, 4.6, 1.1, 'joint verification channel\n$\\lambda = 0.05 + (\\lambda_{\\mathrm{base}} - 0.05)\\, f_A f_B$\n$\\lambda_{\\mathrm{base}} = 0.90$ after ASK, else 0.35')
arrow(6.0, 4.95, 6.0, 4.05)
box(3.7, 1.05, 4.6, 1.0, 'fresh shared signal $e$, reliability 0.75\n$\\mathrm{logit}\\, q_i \\leftarrow \\mathrm{logit}\\, q_i + \\omega_i\\,(2e-1)\\ln 3$')
arrow(6.0, 2.95, 6.0, 2.05)
arrow(3.7, 1.95, 0.3, 4.75); arrow(8.3, 1.95, 11.9, 4.75)
ax.text(1.0, 2.3, 'shared signal\nupdates both', fontsize=6, color=BLUE, ha='center', linespacing=1.25)
# observer
box(8.9, 0.35, 3.2, 1.8, 'External observer\nsees acts + signal record;\nforms only through\nnoisy labels (archived\nLOCK / SEEK rows)', fc='#fdf1e2', ec=AMBER, fs=6)
arrow(7.5, 4.95, 9.4, 2.15, color=AMBER, ls='--', rad=-0.15)

fig.subplots_adjust(0, 0, 1, 1)
# overlap check between all text objects
fig.canvas.draw(); rn = fig.canvas.get_renderer()
tx = [(t.get_text(), t.get_window_extent(rn)) for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip() and t.get_visible()]
ov = [(a[:25], b[:25]) for i, (a, ba) in enumerate(tx) for b, bb in tx[i + 1:] if ba.overlaps(bb)]
print('text overlaps:', ov)
fig.savefig(out / 'H0_model_schematic_print.pdf'); fig.savefig(out / 'H0_model_schematic_print.png', dpi=400)
print('saved', fig.get_size_inches())
