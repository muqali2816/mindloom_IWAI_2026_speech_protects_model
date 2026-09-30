"""E5 — sensitivity of the key contrasts to auxiliary parameters. PYTHONPATH=. python E5_sensitivity.py

Grid (protocol.json['experiments']['E5_sensitivity']): tau in {0.1,0.2,0.4} x k in {1,2} x lambda in {0.6,0.7,0.85}
x rounds in {12,24} = 36 settings, plus 2 extra settings varying the private-record reliability rho in {0.65, 0.85}
at the default (tau=0.2, k=2, lambda=0.7, rounds=12). Constants are monkeypatched on the core module
(rd.TAU, rd.K, rd.LAM, rd.CFG['rounds'], and for rho: rd.RHO, rd.EV_S, rd.PS_GIVEN_X) before make_worlds/dialogue
and restored afterwards. For every setting the pairs costly x honest, attenuated x honest, honest x honest (beta = 1.5)
run on the same worlds (seed protocol['seeds']['E5'], n = protocol['n_worlds']). Tail metrics use protocol['tail_rounds'] = 4
last rounds in both 12- and 24-round settings.
Writes results/E5_sensitivity.csv (means per setting x pair x split), results/E5_contrasts.csv (paired costly - attenuated,
costly - honest, attenuated - honest with 95% MC intervals), results/E5_manifest.json.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # package root on path
import time, itertools
from math import comb
import numpy as np, pandas as pd
import core as rd
import run_utils as ru

OUT = ru.OUT
SEED = rd.CFG['seeds']['E5']; NW = rd.CFG['n_worlds']; BETA = rd.P['beta_active']
G = rd.CFG['experiments']['E5_sensitivity']
PAIRS = [('costly', 'honest'), ('attenuated', 'honest'), ('honest', 'honest')]
METRICS = ['nonc_A', 'A_dbelief', 'A_hidden_shift', 'A_p_truth', 'B_pC_true', 'B_pW_true', 'nonc_B', 'U_final', 'A_transp_err', 'B_p_truth', 'A_own0']
CONTRASTS = [('costlyxhonest', 'attenuatedxhonest'), ('costlyxhonest', 'honestxhonest'), ('attenuatedxhonest', 'honestxhonest')]
DEFAULT = dict(TAU=rd.TAU, K=rd.K, LAM=rd.LAM, rounds=rd.CFG['rounds'], RHO=rd.RHO, EV_S=rd.EV_S.copy(), PS_GIVEN_X=rd.PS_GIVEN_X.copy())


def set_params(tau, k, lam, rounds, rho):
    rd.TAU = float(tau); rd.K = float(k); rd.LAM = float(lam); rd.CFG['rounds'] = int(rounds)
    rd.RHO = float(rho)
    rd.EV_S = (2 * rd.S_GRID - rd.N) * rd.llr(rd.RHO)
    rd.PS_GIVEN_X = np.stack([np.array([comb(rd.N, s) * (1 - rd.RHO) ** s * rd.RHO ** (rd.N - s) for s in rd.S_GRID]),
                              np.array([comb(rd.N, s) * rd.RHO ** s * (1 - rd.RHO) ** (rd.N - s) for s in rd.S_GRID])])


def restore():
    rd.TAU = DEFAULT['TAU']; rd.K = DEFAULT['K']; rd.LAM = DEFAULT['LAM']; rd.CFG['rounds'] = DEFAULT['rounds']
    rd.RHO = DEFAULT['RHO']; rd.EV_S = DEFAULT['EV_S'].copy(); rd.PS_GIVEN_X = DEFAULT['PS_GIVEN_X'].copy()


def main():
    t0 = time.time()
    settings = [dict(setting=f'tau{t}_k{int(k)}_lam{l}_R{R}', tau=t, k=k, lam=l, rounds=R, rho=DEFAULT['RHO'], grid='main')
                for t, k, l, R in itertools.product(G['tau'], G['k'], G['lambda'], G['rounds'])]
    settings += [dict(setting=f'rho{r}_default', tau=DEFAULT['TAU'], k=DEFAULT['K'], lam=DEFAULT['LAM'], rounds=DEFAULT['rounds'], rho=r, grid='extra_rho')
                 for r in (0.65, 0.85)]
    rows = []; crow = []; timing = {}
    try:
        for st in settings:
            set_params(st['tau'], st['k'], st['lam'], st['rounds'], st['rho'])
            worlds = rd.make_worlds(NW, SEED)
            t = time.time(); dfs = {}
            for a_t, b_t in PAIRS:
                d = rd.dialogue(worlds, ru.cond(a_t, b_t, BETA)); df, _ = rd.metrics(d); dfs[f'{a_t}x{b_t}'] = df
                for split, m in (('all', np.ones(NW, bool)), ('A_own0', df['A_own0'].to_numpy() == 1)):
                    r = {**st, 'pair': f'{a_t}x{b_t}', 'split': split, 'n': int(m.sum())}
                    for c in METRICS:
                        r[c] = float(df.loc[m, c].mean()); r[c + '_se'] = float(df.loc[m, c].std(ddof=1) / np.sqrt(m.sum()))
                    rows.append(r)
            for x, y in CONTRASTS:
                for split, m in (('all', np.ones(NW, bool)), ('A_own0', dfs[x]['A_own0'].to_numpy() == 1)):
                    for r in rd.paired(dfs[x][m][METRICS], dfs[y][m][METRICS], f'{x}-{y}'):
                        r.update(st); r.update(split=split, n=int(m.sum())); crow.append(r)
            timing[st['setting']] = round(time.time() - t, 2)
            print(st['setting'], timing[st['setting']], 's', flush=True)
    finally:
        restore()
    assert rd.TAU == 0.2 and rd.CFG['rounds'] == 12
    pd.DataFrame(rows).to_csv(OUT / 'E5_sensitivity.csv', index=False)
    con = pd.DataFrame(crow); con['sign'] = np.where(con['mc_lo'] > 0, '+', np.where(con['mc_hi'] < 0, '-', '0'))
    con.to_csv(OUT / 'E5_contrasts.csv', index=False)
    ru.write_manifest('E5', {'seed': SEED, 'n_worlds': NW, 'beta': BETA, 'n_settings': len(settings), 'timing_s': timing, 'total_s': round(time.time() - t0, 1)})
    print('done', round(time.time() - t0, 1), 's')


if __name__ == '__main__':
    main()
