"""Publication-style rcParams and small helpers shared by the H*_figures.py scripts (English labels, IWAI poster)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

TYPES = ['honest', 'attenuated', 'costly', 'access']
TYPE_LABEL = {'honest': 'honest (ω=1, c=0, γ=0)', 'attenuated': 'attenuated (ω=0.2)', 'costly': 'concession-costly (c=1.6)', 'access': 'access-protecting (γ=2)'}
TYPE_SHORT = {'honest': 'honest', 'attenuated': 'attenuated', 'costly': 'costly', 'access': 'access-prot.'}
COL = {'honest': '#4d4d4d', 'attenuated': '#0072B2', 'costly': '#E69F00', 'access': '#CC79A7'}   # Okabe-Ito, CVD-safe
DPI = 200


def style(sizes=(8, 7, 6)):
    base, sec, tick = sizes
    plt.rcParams.update({'font.size': base, 'axes.titlesize': base, 'axes.labelsize': base, 'legend.fontsize': sec, 'xtick.labelsize': tick, 'ytick.labelsize': tick,
                         'axes.spines.top': False, 'axes.spines.right': False, 'xtick.direction': 'out', 'ytick.direction': 'out', 'legend.frameon': False,
                         'axes.titlelocation': 'left', 'axes.titleweight': 'normal', 'savefig.dpi': DPI, 'pdf.fonttype': 42, 'figure.dpi': 100, 'axes.grid': False})


def panel_letter(ax, letter, dx=-0.12, dy=1.04, fontsize=10):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=fontsize, fontweight='bold', va='bottom', ha='right')


def check_overlaps(fig):
    """Geometric bbox check (figure-style §9.1): returns list of overlapping text pairs / text-spine pairs."""
    import matplotlib as mpl
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
    spines = [(s, s.get_window_extent(r)) for ax in fig.axes for s in ax.spines.values() if s.get_visible()]
    tl = {ax: set(ax.get_xticklabels(which='both') + ax.get_yticklabels(which='both')) for ax in fig.axes}
    ov = [(a.get_text(), b.get_text()) for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:] if ba.overlaps(bb)]
    ov += [(t.get_text(), 'spine') for t, bt in texts for s, bs in spines if bt.overlaps(bs) and t not in tl[s.axes]]
    return ov


def dot_ci(ax, xs, means, ses, colors, labels=None, lw=1.2, ms=5):
    for x, m, se, c in zip(xs, means, ses, colors):
        ax.errorbar(x, m, yerr=1.96 * se, fmt='o', color=c, ms=ms, lw=lw, capsize=2)
    ax.set_xticks(list(xs))
    if labels is not None: ax.set_xticklabels(labels)
    ax.margins(x=0.15, y=0.12)
