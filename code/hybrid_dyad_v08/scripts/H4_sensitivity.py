"""H4 — sign sensitivity of the key contrasts (partner honest) over tau x delta x lambda_floor x N = 54 settings.
Run from hybrid_dyad/ with PYTHONPATH=.   usage: python H4_sensitivity.py grid | labels <k>
grid   : 1000 worlds (seed H4) per setting, 4 generators, paired contrasts (all worlds and A_own0==1 slice) -> results/H4_grid_raw.csv
labels : setting k of CORNERS (one-at-a-time extremes at N=2), 500 worlds, acts vs labels eta=0.25 next-signal gain -> results/H4_labels_<k>.npz"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # package root on path
import sys, itertools, json, time
import numpy as np, pandas as pd
import core as hd, observer as ob, run_utils as ru

T = hd.CFG['agent_types']; SEEDS = hd.CFG['seeds']
TAU = [0.1, 0.2, 0.4]; DELTA = [0.06, 0.12, 0.24]; FLOOR = [0.02, 0.05, 0.10]; NN = [2, 4]
DEFAULT = dict(tau=0.2, delta=0.12, floor=0.05, N=2)
GRID = [dict(tau=t, delta=d, floor=f, N=n) for t, d, f, n in itertools.product(TAU, DELTA, FLOOR, NN)]
CORNERS = [dict(DEFAULT, tau=0.1), dict(DEFAULT, tau=0.4), dict(DEFAULT, delta=0.06), dict(DEFAULT, delta=0.24),
           dict(DEFAULT, floor=0.02), dict(DEFAULT, floor=0.10)]
CONTRASTS = [('access', 'honest', 'evidence_count'), ('access', 'honest', 'A_p_truth'),
             ('costly', 'attenuated', 'A_dbelief'), ('access', 'costly', 'restrict_A')]
KEYS = {'tau': 'choice_temperature', 'delta': 'restriction_cost', 'floor': 'lam_floor'}


class setting:
    def __init__(self, s): self.s = s
    def __enter__(self):
        self.saved = {v: hd.P[v] for v in KEYS.values()}
        for k, v in KEYS.items(): hd.P[v] = self.s[k]
    def __exit__(self, *a):
        for v, x in self.saved.items(): hd.P[v] = x


def run_grid(n=1000):
    rows = []
    for k, s in enumerate(GRID):
        with setting(s):
            w = hd.make_worlds(n, SEEDS['H4'], N=s['N']); dfs = {}
            for g in ('honest', 'attenuated', 'costly', 'access'):
                dfs[g] = hd.metrics(hd.dialogue(w, ru.cond(g, 'honest')))
        own = dfs['honest']['A_own0'] == 1
        for a, b, metric in CONTRASTS:
            for split, mask in (('all', np.ones(n, bool)), ('A_own0==1', own.to_numpy())):
                delta = (dfs[a][metric] - dfs[b][metric])[mask].to_numpy(); se = delta.std(ddof=1) / np.sqrt(len(delta))
                rows.append(dict(setting=k, tau=s['tau'], restriction_cost=s['delta'], lam_floor=s['floor'], N=s['N'], contrast=f'{a}-{b}', metric=metric, split=split, n=len(delta), delta=delta.mean(),
                                 mc_lo=delta.mean() - 1.96 * se, mc_hi=delta.mean() + 1.96 * se,
                                 mean_a=dfs[a][metric][mask].mean(), mean_b=dfs[b][metric][mask].mean()))
        print(k, s, flush=True)
    pd.DataFrame(rows).to_csv(ru.OUT / 'H4_grid_raw.csv', index=False)


def run_labels(k, n=500):
    s = CORNERS[k]; out = {'setting': json.dumps(s)}
    with setting(s):
        w = hd.make_worlds(n, SEEDS['H4'], N=s['N']); C = hd.channel_mean(0.25)
        for g in ('honest', 'attenuated', 'costly', 'access'):
            cond = ru.cond(g, 'honest'); d = hd.dialogue(w, cond); z, zn, _ = hd.emit_labels(d, SEEDS['labels'])
            ra = ob.recover(d, cond, 'acts', dict(T['honest'])); rl = ob.recover(d, cond, 'labels', dict(T['honest']), labels=z, C=C)
            out[f'{g}/loss_acts'] = ru.logloss_signal(ra, d); out[f'{g}/loss_labels'] = ru.logloss_signal(rl, d)
            out[f'{g}/fam_acts'] = ra['family_post']; out[f'{g}/fam_labels'] = rl['family_post']
            print(k, g, flush=True)
    np.savez_compressed(ru.OUT / f'H4_labels_{k}.npz', **out)


if __name__ == '__main__':
    if sys.argv[1] == 'grid': run_grid()
    else: run_labels(int(sys.argv[2]))
