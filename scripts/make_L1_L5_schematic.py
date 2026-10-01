import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import textwrap

blue = '#e3eef5'; edge = '#2f6f8f'; orange = '#f7ead6'; oedge = '#b9762a'; grey = '#666666'

titles = ['L1 evidence atoms', 'L2 evidence features', 'L3 bridges', 'L4 configuration', 'L5 typed output']
bodies_v = ['surface items recorded with their span: named feeling,\nquestion form, absolute expression, explicit return time …',
            'atoms aggregated into features: minimal response,\nrequired response, reply constraint, closure …',
            'label-specific evidence requirements checked;\na required item missing in L1 blocks the label',
            'at most one of eight configurations; SEAL (function)\nmay co-occur; SHIFT (event) needs before/after evidence',
            'label with evidence trail (L1 spans → L2 → L3),\nor abstention "insufficient context"']

trace_clean = ['"I know what I sent": assertion. "If you trusted me, you would stop questioning this": conditional linking trust to cessation of questioning.',
               'Reply constraint: present. Request for the other to regulate: absent. Closure of the topic: absent.',
               'LOCK bridge (reply constraint in L1): met. SEAL bridge (closure): not met. DRAIN bridge (other as regulator): not met.',
               'Configuration = LOCK; SEAL not assigned; SHIFT: no before/after configuration evidence.',
               'LOCK with trail to the L1 spans; the engine would abstain if a required evidence item were absent at L3.']

W, Hh = 6.3, 9.6
fig, ax = plt.subplots(figsize=(W, Hh)); ax.set_xlim(0, W); ax.set_ylim(0, Hh); ax.axis('off')
ax.text(W/2, Hh-0.08, 'Mindloom typed engine: inference path\nfrom surface evidence to typed output', ha='center', va='top', fontsize=11, color='#1f3d4d', fontweight='bold', linespacing=1.15)
lx, rx, bw = 0.1, 3.25, 2.95
ax.text(lx, Hh-0.75, 'engine layer', fontsize=9.5, color=grey, va='top')
ax.text(rx, Hh-0.75, 'illustrative trace (constructed line,\nnot an engine output): "I know what I sent.\nIf you trusted me, you would stop\nquestioning this."', fontsize=8.6, color=grey, va='top', linespacing=1.15)
bw_txt = [textwrap.fill(b.replace('\n', ' '), 34) for b in bodies_v]; tw_txt = [textwrap.fill(t, 34) for t in trace_clean]
top = Hh-1.6; hh = 1.42; gap = 0.2
for k in range(5):
    y = top - k*(hh+gap) - hh
    ax.add_patch(FancyBboxPatch((lx, y), bw, hh, boxstyle='round,pad=0.02,rounding_size=0.1', fc=blue, ec=edge, lw=1.4))
    ax.text(lx + 0.13, y + hh - 0.12, titles[k], ha='left', va='top', fontsize=10.5, fontweight='bold', color='#1f3d4d')
    ax.text(lx + 0.13, y + hh - 0.46, bw_txt[k], ha='left', va='top', fontsize=8.6, color='#222222', linespacing=1.22)
    ax.add_patch(FancyBboxPatch((rx, y), bw, hh, boxstyle='round,pad=0.02,rounding_size=0.1', fc=orange, ec=oedge, lw=1.1))
    ax.text(rx + 0.13, y + hh - 0.14, tw_txt[k], ha='left', va='top', fontsize=8.6, color='#222222', linespacing=1.25)
    if k < 4:
        for x0, c in ((lx + bw/2, edge), (rx + bw/2, oedge)):
            ax.add_patch(FancyArrowPatch((x0, y - 0.01), (x0, y - gap + 0.01), arrowstyle='-|>', mutation_scale=13, color=c, lw=1.3))
fig.savefig('L1_L5_schematic.png', dpi=220, bbox_inches='tight')