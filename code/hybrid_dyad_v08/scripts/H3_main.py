"""H3 — interventions and measurement channels (track A). Run from hybrid_dyad/ with PYTHONPATH=.
Generators: A in {honest, attenuated, costly, access} (protocol-typical), B honest, seed H3, 2000 shared worlds.
Conditions: base and seven interventions starting at round index 7 (the 8th round; outcomes are read on rounds 8-12 = indices 7..11).
Outputs: results/H3_interventions_summary.csv, H3_interventions_contrasts.csv, H3_P4_check.json, H3_power.csv, H3_power_n80.csv,
per-world tables results/H3_<generator>_<condition>_worlds.csv."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # package root on path
import json, time
import numpy as np, pandas as pd
from scipy import stats
import core as hd, run_utils as ru

CFG = hd.CFG; T = CFG['agent_types']; OUT = ru.OUT; SEED = CFG['seeds']['H3']; N_WORLDS = CFG['n_worlds']; R = CFG['rounds']; T0 = 7
w = hd.make_worlds(N_WORLDS, SEED)
GENS = ['honest', 'attenuated', 'costly', 'access']
CONDS = {'base': {}, 'reassure': {'force_reassure': {'round': T0, 'agent': 'B'}}, 'placebo': {'force_reassure': {'round': T0, 'agent': 'B'}, 'placebo': True},
         'remove_c': {'remove_c_from': T0, 'remove_for': ['A']}, 'remove_gamma': {'remove_gamma_from': T0, 'remove_for': ['A']},
         'force_open_B': {'force_open': {'agent': 'B', 'from': T0}}, 'bypass': {'bypass_from': T0}, 'external_obs': {'external_obs_round': T0, 'external_rho': 0.95}}
OUTC = ['P_concede_A_late', 'nonc_A', 'evidence_count_late', 'evidence_count', 'A_p_truth', 'A_dbelief', 'restrict_A_late', 'restrict_A', 'A_hidden_shift', 'B_p_truth',
        'B_pC_true', 'B_pW_true', 'B_pC_high', 'B_pW_low', 'A_ASK_late']
SLICES = {'all': lambda df: np.ones(len(df), bool), 'A_own0': lambda df: df.A_own0.to_numpy() == 1}


def outcomes(d):
    df = hd.metrics(d); a = d['acts']; cg = np.asarray(hd.P['cost_grid']); wg = np.asarray(hd.P['omega_grid'])
    df['P_concede_A_late'] = (a[:, T0:, 0] == hd.CONCEDE).any(1).astype(float)
    df['evidence_count_late'] = (d['ev'][:, T0:] != 2).sum(1).astype(float)
    df['restrict_A_late'] = (d['forms'][:, T0:, 0] == 0).mean(1)
    df['A_ASK_late'] = (a[:, T0:, 0] == hd.ASK).mean(1)
    df['B_pC_high'] = d['B_pC_partner'][:, int(np.argmin(np.abs(cg - 1.6)))]      # listener's posterior that A's c = 1.6
    df['B_pW_low'] = d['B_pW_partner'][:, int(np.argmin(np.abs(wg - 0.2)))]       # listener's posterior that A's omega = 0.2
    return df


t0 = time.time(); D = {}
for g in GENS:
    for cn, iv in CONDS.items():
        d = hd.dialogue(w, dict(A=dict(T[g]), B=dict(T['honest']), beta=1.5, intervention=iv)); df = outcomes(d); D[(g, cn)] = df
        df.assign(generator=g, condition=cn).to_csv(OUT / f'H3_{g}_{cn}_worlds.csv', index=False)
print('dialogues', round(time.time() - t0, 1), 's')

# ------------------------------------------------------------------ summary and paired contrasts (intervention - base, reassure - placebo)
summ, con = [], []
for (g, cn), df in D.items():
    for s, f in SLICES.items():
        sub = df[f(df)]; row = dict(generator=g, condition=cn, split=s, n=len(sub))
        for c in OUTC: row[c] = sub[c].mean(); row[c + '_se'] = sub[c].std(ddof=1) / np.sqrt(len(sub))
        summ.append(row)
for g in GENS:
    for s, f in SLICES.items():
        m = f(D[(g, 'base')])
        for cn in CONDS:
            if cn == 'base': continue
            for r in hd.paired(D[(g, cn)][m][OUTC], D[(g, 'base')][m][OUTC], f'{cn}-base'): r.update(generator=g, split=s, n=int(m.sum())); con.append(r)
        for r in hd.paired(D[(g, 'reassure')][m][OUTC], D[(g, 'placebo')][m][OUTC], 'reassure-placebo'): r.update(generator=g, split=s, n=int(m.sum())); con.append(r)
summ = pd.DataFrame(summ); con = pd.DataFrame(con)
summ.to_csv(OUT / 'H3_interventions_summary.csv', index=False); con.to_csv(OUT / 'H3_interventions_contrasts.csv', index=False)

# ------------------------------------------------------------------ P4 check (slice A_own0, 95% MC intervals excluding 0 = "moves")
def eff(g, cn, metric, split='A_own0'):
    r = con[(con.generator == g) & (con.contrast == cn) & (con.metric == metric) & (con.split == split)].iloc[0]
    return dict(delta=float(r.delta), lo=float(r.mc_lo), hi=float(r.mc_hi), moves=bool(np.sign(r.mc_lo) == np.sign(r.mc_hi)))
P4 = {'reassure_minus_placebo__P_concede_A_late': {g: eff(g, 'reassure-placebo', 'P_concede_A_late') for g in GENS},
      'remove_c__P_concede_A_late': {g: eff(g, 'remove_c-base', 'P_concede_A_late') for g in GENS},
      'remove_gamma__evidence_count_late': {g: eff(g, 'remove_gamma-base', 'evidence_count_late') for g in GENS},
      'remove_gamma__A_p_truth': {g: eff(g, 'remove_gamma-base', 'A_p_truth') for g in GENS},
      'force_open_B__evidence_count_late': {g: eff(g, 'force_open_B-base', 'evidence_count_late') for g in GENS},
      'force_open_B__A_p_truth': {g: eff(g, 'force_open_B-base', 'A_p_truth') for g in GENS},
      'external_obs__A_dbelief': {g: eff(g, 'external_obs-base', 'A_dbelief') for g in GENS},
      'external_obs__A_p_truth': {g: eff(g, 'external_obs-base', 'A_p_truth') for g in GENS},
      'bypass__evidence_count_late': {g: eff(g, 'bypass-base', 'evidence_count_late') for g in GENS}}
json.dump(P4, open(OUT / 'H3_P4_check.json', 'w'), indent=1)

# ------------------------------------------------------------------ dialogues-per-group table (Welch, alpha .05, 400 resamples)
CHANNELS = {'nonc_A': ('base', 'nonc_A'), 'restrict_A': ('base', 'restrict_A'), 'evidence_count': ('base', 'evidence_count'), 'A_dbelief': ('base', 'A_dbelief'),
            'A_p_truth': ('base', 'A_p_truth'), 'A_hidden_shift': ('base', 'A_hidden_shift'),
            'concede_after_remove_c': ('remove_c', 'P_concede_A_late'), 'concede_after_reassure': ('reassure', 'P_concede_A_late'),
            'evidence_after_force_open_B': ('force_open_B', 'evidence_count_late'), 'p_truth_after_external_obs': ('external_obs', 'A_p_truth'),
            'B_pC_true': ('base', 'B_pC_true'), 'B_pW_true': ('base', 'B_pW_true'), 'B_pC_high': ('base', 'B_pC_high'), 'B_pW_low': ('base', 'B_pW_low')}
PAIRS = [('attenuated', 'costly'), ('attenuated', 'access'), ('costly', 'access')]
NGRID = [8, 12, 17, 25, 35, 50, 70, 100, 140, 200, 400]; NRES = 400; ALPHA = 0.05
rng = np.random.default_rng(SEED + 1)


def welch_power(x, y, n, nres, rng):
    ix = np.stack([rng.choice(len(x), n, replace=False) for _ in range(nres)]); iy = np.stack([rng.choice(len(y), n, replace=False) for _ in range(nres)])
    X = x[ix]; Y = y[iy]; mx, my = X.mean(1), Y.mean(1); vx, vy = X.var(1, ddof=1), Y.var(1, ddof=1)
    se2 = vx / n + vy / n
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (mx - my) / np.sqrt(se2); df = se2 ** 2 / ((vx / n) ** 2 / (n - 1) + (vy / n) ** 2 / (n - 1))
        p = 2 * stats.t.sf(np.abs(t), df)
    zero = se2 == 0; p = np.where(zero, np.where(mx == my, 1.0, 0.0), p)
    return float((p < ALPHA).mean()), float(np.abs(mx - my).mean() / np.sqrt((vx.mean() + vy.mean()) / 2 + 1e-12))


rows = []
for s, f in SLICES.items():
    for g1, g2 in PAIRS:
        for ch, (cn, metric) in CHANNELS.items():
            x = D[(g1, cn)][metric].to_numpy()[f(D[(g1, cn)])]; y = D[(g2, cn)][metric].to_numpy()[f(D[(g2, cn)])]
            dfull = (x.mean() - y.mean()) / np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2 + 1e-12)
            for n in NGRID:
                pw, _ = welch_power(x, y, n, NRES, rng)
                rows.append(dict(split=s, pair=f'{g1} vs {g2}', channel=ch, condition=cn, metric=metric, n_per_group=n, power=pw, cohen_d_full=dfull,
                                 mean_1=x.mean(), mean_2=y.mean()))
pw = pd.DataFrame(rows); pw.to_csv(OUT / 'H3_power.csv', index=False)
n80 = []
for (s, pr, ch), sub in pw.groupby(['split', 'pair', 'channel']):
    sub = sub.sort_values('n_per_group'); ok = sub[sub.power >= 0.8]
    n80.append(dict(split=s, pair=pr, channel=ch, n_80=int(ok.n_per_group.iloc[0]) if len(ok) else np.nan, power_at_400=float(sub.power.iloc[-1]), cohen_d_full=float(sub.cohen_d_full.iloc[0])))
pd.DataFrame(n80).to_csv(OUT / 'H3_power_n80.csv', index=False)
print('done', round(time.time() - t0, 1), 's')
