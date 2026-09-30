"""H1 — three mechanisms under one public outcome (track A). Run from hybrid_dyad/ with PYTHONPATH=.
(a) calibration grid (partner honest), (b) four conditions at protocol-typical and at calibrated parameters,
(c) 4x4 phase map of types, (d) round-by-round trajectories, (e) beta=0 ablation. All outputs -> results/H1_*.csv."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # package root on path
import json, time
import numpy as np, pandas as pd
import core as hd, run_utils as ru

CFG = hd.CFG; T = CFG['agent_types']; OUT = ru.OUT; SEED = CFG['seeds']['H1']; N_WORLDS = CFG['n_worlds']
w = hd.make_worlds(N_WORLDS, SEED)
TYPES = ['honest', 'attenuated', 'costly', 'access']
KEY = ['evidence_count', 'mean_rate', 'restrict_A', 'restrict_B', 'restrict_joint', 'A_p_truth', 'A_dbelief', 'A_hidden_shift',
       'nonc_A', 'nonc_B', 'nonc_mutual', 'A_ASK', 'A_CONCEDE', 'A_REASSURE', 'A_PRESSURE', 'B_p_truth', 'B_dbelief', 'belief_gap_final',
       'U_final', 'A_transp_err', 'B_transp_err', 'A_pC_true', 'A_pW_true', 'B_pC_true', 'B_pW_true']
SLICES = {'all': lambda df: np.ones(len(df), bool), 'A_own0': lambda df: df.A_own0.to_numpy() == 1,
          'both_own': lambda df: (df.A_own0.to_numpy() == 1) & (df.B_own0.to_numpy() == 1)}


def spec(omega=1.0, c=0.0, gamma=0.0): return dict(omega=float(omega), c=float(c), gamma=float(gamma))


def summarise(df, run, slices=('all', 'A_own0')):
    rows = []
    for s in slices:
        sub = df[SLICES[s](df)]; row = {'run': run, 'split': s, 'n': len(sub)}
        for col in KEY:
            v = sub[col].dropna(); row[col] = v.mean(); row[col + '_se'] = v.std(ddof=1) / np.sqrt(len(v))
        rows.append(row)
    return rows


def contrasts(dfs, pairs, slices=('all', 'A_own0'), tag=''):
    rows = []
    for a, b in pairs:
        for s in slices:
            m = SLICES[s](dfs[a])
            for r in hd.paired(dfs[a][m], dfs[b][m], f'{a}-{b}'):
                r['split'] = s; r['n'] = int(m.sum()); r['set'] = tag; rows.append(r)
    return pd.DataFrame(rows)


t0 = time.time()
# ---------------------------------------------------------------- (a) calibration, partner honest, slice A_own0==1
grid = {'attenuated': ('omega', [0.05, 0.1, 0.2, 0.3]), 'costly': ('c', [0.8, 1.2, 1.6, 2.0]), 'access': ('gamma', [1, 2, 4, 8])}
cal = []
for mech, (par, vals) in grid.items():
    for v in vals:
        df = hd.metrics(hd.dialogue(w, dict(A=spec(**{par: v}), B=dict(T['honest']), beta=1.5)))
        s = df[df.A_own0 == 1]
        cal.append(dict(mechanism=mech, param=par, value=v, typical=float(v == T[mech][par]), n_own0=len(s),
                        nonc_A_own0=s.nonc_A.mean(), nonc_A_own0_se=s.nonc_A.std(ddof=1) / np.sqrt(len(s)), nonc_A_all=df.nonc_A.mean(),
                        evidence_count_own0=s.evidence_count.mean(), restrict_A_own0=s.restrict_A.mean(), A_dbelief_own0=s.A_dbelief.mean(),
                        A_hidden_shift_own0=s.A_hidden_shift.mean()))
cal = pd.DataFrame(cal)
typ = cal[cal.typical == 1].set_index('mechanism').nonc_A_own0
target = float(np.median(typ.to_numpy()))
chosen = {}
for mech in grid:
    sub = cal[cal.mechanism == mech]; k = (sub.nonc_A_own0 - target).abs().idxmin()
    chosen[mech] = dict(param=cal.loc[k, 'param'], value=float(cal.loc[k, 'value']), nonc_A_own0=float(cal.loc[k, 'nonc_A_own0']), residual=float(cal.loc[k, 'nonc_A_own0'] - target))
cal['target_median_typical'] = target
cal.to_csv(OUT / 'H1_calibration.csv', index=False)
json.dump({'target': target, 'typical_nonc_A_own0': typ.to_dict(), 'chosen': chosen,
           'attainable': bool(max(abs(c['residual']) for c in chosen.values()) < 0.02)}, open(OUT / 'H1_calibration_choice.json', 'w'), indent=1)
print('calibration', round(time.time() - t0, 1), 's; target', round(target, 3), chosen)

# ---------------------------------------------------------------- (b) four conditions, partner honest: typical + calibrated
sets = {'typical': {t: dict(T[t]) for t in TYPES},
        'calibrated': {'honest': dict(T['honest']), **{m: spec(**{chosen[m]['param']: chosen[m]['value']}) for m in grid}}}
summ, allc, worlds = [], [], {}
for tag, specs in sets.items():
    dfs = {}
    for t in TYPES:
        d = hd.dialogue(w, dict(A=specs[t], B=dict(T['honest']), beta=1.5)); df = hd.metrics(d); dfs[t] = df
        df.assign(run=t, set=tag).to_csv(OUT / f'H1_{tag}_{t}_worlds.csv', index=False)
        for r in summarise(df, t): r['set'] = tag; summ.append(r)
        if tag == 'typical': worlds[t] = d
    pairs = [(t, 'honest') for t in TYPES[1:]] + [('attenuated', 'costly'), ('attenuated', 'access'), ('costly', 'access')]
    allc.append(contrasts(dfs, pairs, tag=tag))
pd.DataFrame(summ).to_csv(OUT / 'H1_mechanisms_summary.csv', index=False)
pd.concat(allc).to_csv(OUT / 'H1_mechanisms_contrasts.csv', index=False)
print('mechanisms', round(time.time() - t0, 1), 's')

# ---------------------------------------------------------------- (c) phase map 4x4 (typical parameters), beta=1.5 and beta=0
phase = []
for beta in (1.5, 0.0):
    for a in TYPES:
        for b in TYPES:
            df = hd.metrics(hd.dialogue(w, ru.cond(a, b, beta=beta)))
            for s in ('all', 'both_own', 'A_own0'):
                sub = df[SLICES[s](df)]; row = dict(A=a, B=b, beta=beta, split=s, n=len(sub))
                for col in ('nonc_mutual', 'nonc_A', 'nonc_B', 'evidence_count', 'mean_rate', 'belief_gap_final', 'restrict_joint', 'A_p_truth', 'B_p_truth', 'A_ASK', 'B_ASK', 'U_final'):
                    row[col] = sub[col].mean(); row[col + '_se'] = sub[col].std(ddof=1) / np.sqrt(len(sub))
                phase.append(row)
pd.DataFrame(phase).to_csv(OUT / 'H1_phase_map.csv', index=False)
print('phase', round(time.time() - t0, 1), 's')

# ---------------------------------------------------------------- (d) trajectories (typical, partner honest)
traj = []
for t, d in worlds.items():
    q1 = hd.sigmoid(d['L']); truth = d['truth']; pt = np.where(truth[:, None] == 1, q1[:, :, 0], 1 - q1[:, :, 0])
    ptB = np.where(truth[:, None] == 1, q1[:, :, 1], 1 - q1[:, :, 1])
    cum = np.cumsum(d['ev'] != 2, axis=1); own0 = q1[:, 0, 0] > 0.5
    conc = np.cumsum(d['acts'][:, :, 0] == hd.CONCEDE, axis=1) > 0
    for s, m in (('all', np.ones(len(truth), bool)), ('A_own0', own0)):
        for r in range(CFG['rounds'] + 1):
            row = dict(run=t, split=s, round=r, n=int(m.sum()), A_p_truth=pt[m, r].mean(), A_p_truth_se=pt[m, r].std(ddof=1) / np.sqrt(m.sum()),
                       B_p_truth=ptB[m, r].mean(), gap=np.abs(q1[m, r, 0] - q1[m, r, 1]).mean())
            if r >= 1:
                row.update(cum_signals=cum[m, r - 1].mean(), cum_signals_se=cum[m, r - 1].std(ddof=1) / np.sqrt(m.sum()), rate=d['rates'][m, r - 1].mean(),
                           restrict_A=(d['forms'][m, r - 1, 0] == 0).mean(), A_conceded_by=conc[m, r - 1].mean(), A_ASK=(d['acts'][m, r - 1, 0] == hd.ASK).mean(),
                           A_CONCEDE=(d['acts'][m, r - 1, 0] == hd.CONCEDE).mean())
            traj.append(row)
pd.DataFrame(traj).to_csv(OUT / 'H1_trajectories.csv', index=False)

# ---------------------------------------------------------------- (e) beta = 0 ablation, partner honest, typical parameters
summ0, dfs0, dfs15 = [], {}, {}
for t in TYPES:
    dfs15[t] = pd.read_csv(OUT / f'H1_typical_{t}_worlds.csv')
    df0 = hd.metrics(hd.dialogue(w, ru.cond(t, 'honest', beta=0.0))); dfs0[t] = df0
    df0.assign(run=t, set='beta0').to_csv(OUT / f'H1_beta0_{t}_worlds.csv', index=False)
    for r in summarise(df0, t): r['set'] = 'beta0'; summ0.append(r)
pd.DataFrame(summ0).to_csv(OUT / 'H1_beta0_summary.csv', index=False)
c1 = contrasts(dfs0, [(t, 'honest') for t in TYPES[1:]] + [('attenuated', 'costly'), ('attenuated', 'access'), ('costly', 'access')], tag='beta0')
rows = []
for t in TYPES:
    for s in ('all', 'A_own0'):
        m = SLICES[s](dfs15[t])
        for r in hd.paired(dfs0[t][m], dfs15[t][m], f'{t}: beta0-beta1.5'): r['split'] = s; r['n'] = int(m.sum()); r['set'] = 'beta0_vs_beta1.5'; rows.append(r)
pd.concat([c1, pd.DataFrame(rows)]).to_csv(OUT / 'H1_beta0_contrasts.csv', index=False)
print('done', round(time.time() - t0, 1), 's')
