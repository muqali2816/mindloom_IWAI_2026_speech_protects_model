"""E3 — interventions inside the dyad (track 3). PYTHONPATH=. python E3_interventions.py

Shared worlds (seed protocol['seeds']['E3'], n = protocol['n_worlds']) for four type pairs (beta = 1.5).
Conditions (control 'force_silence' added: B forced to SILENCE at index 7 — displaces B's act without granting relief;
round indices are 0-based as in core.dialogue; 0-based index 7 == 1-based round 8):
  base       : no intervention
  reassure   : B forced to REASSURE at index 7 -> A's concession cost is 0 on A's move at index 8
  placebo    : same forced REASSURE, but no relief is ever granted (B still models A as relieved)
  remove_c_A : A's cost set to 0 from index 7 onward
  remove_c_AB: both agents' cost set to 0 from index 7 onward
  external   : shared unattenuated observation of x with reliability 0.95 delivered to both at index 7
Outcomes added to rd.metrics: conc_A_8_12 = any CONCEDE by A at indices 7..11 (1-based rounds 8-12),
conc_A_9_12 = indices 8..11 (first A move after the forced reassure), the same for B, and the listener
posterior masses P_B(c_A = 1.6), P_B(omega_A = 0.2) (same quantity for every generator).
Writes results/E3_summary.csv, results/E3_contrasts.csv, results/E3_worlds.csv.gz, results/E3_power.csv,
results/E3_power_n80.csv, results/E3_acts_identity.csv, results/E3_manifest.json.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # package root on path
import json, time
import numpy as np, pandas as pd
from scipy import stats
import core as rd
import run_utils as ru

OUT = ru.OUT
SEED = rd.CFG['seeds']['E3']; NW = rd.CFG['n_worlds']; BETA = rd.P['beta_active']
PAIRS = [('costly', 'honest'), ('attenuated', 'honest'), ('costly', 'costly'), ('attenuated', 'attenuated')]
IV = {
    'base': None,
    'reassure': {'force_reassure': {'round': 7, 'agent': 'B'}},
    'placebo': {'force_reassure': {'round': 7, 'agent': 'B'}, 'placebo': True},
    'remove_c_A': {'remove_c_from': 7, 'remove_c_for': ['A']},
    'remove_c_AB': {'remove_c_from': 7},
    'external': {'external_obs_round': 7, 'external_rho': 0.95},
    'force_silence': {'force_act': {'round': 7, 'agent': 'B', 'act': rd.SILENCE}},   # control: B's act at index 7 displaced, no relief
}
CONTRASTS = [('reassure', 'base'), ('placebo', 'reassure'), ('placebo', 'base'), ('remove_c_A', 'base'),
             ('remove_c_AB', 'base'), ('external', 'base'),
             ('force_silence', 'base'), ('reassure', 'force_silence'), ('placebo', 'force_silence')]
POWER_GRID = [8, 12, 17, 25, 35, 50, 70, 100, 140, 200, 400]; N_RESAMPLE = 400; ALPHA = 0.05
CG = np.asarray(rd.P['cost_grid']); WG = np.asarray(rd.P['omega_grid'])


def dialogue_ext(w, cond):
    """Verbatim copy of core.dialogue (checked bitwise below) with one extension: iv['force_act'] = {'round', 'agent', 'act'}
    forces an arbitrary act code (used for the SILENCE control that displaces B's act without granting relief). cond: {'A': spec, 'B': spec, 'B_level': 1|0, 'cost_grid', 'omega_grid', 'prior_c', 'prior_w',
             'n_acts_partner_model', 'intervention': {...}}. spec = {'omega', 'c', 'beta', 'n_acts'}."""
    n = len(w['truth']); R = rd.CFG['rounds']
    cg = cond.get('cost_grid', rd.P['cost_grid']); wg = cond.get('omega_grid', rd.P['omega_grid'])
    nap = int(cond.get('n_acts_partner_model', 5))
    iv = cond.get('intervention', {}) or {}
    A = rd.Level1(0, w, cond['A'], cg, wg, nap, cond.get('prior_c'), cond.get('prior_w'))
    B = rd.Level1(1, w, cond['B'], cg, wg, nap, cond.get('prior_c'), cond.get('prior_w')) if cond.get('B_level', 1) == 1 else rd.Level0(1, w, cond['B'])
    agents = [A, B]
    acts = np.empty((n, R, 2), np.int8); Lh = np.empty((n, R + 1, 2)); posh = np.empty((n, R + 1, 2), np.int8)
    est = np.full((n, R + 1, 2), np.nan); igh = np.full((n, R, 2, 5), np.nan); relief_h = np.zeros((n, R, 2), bool); probh = np.full((n, R, 2, 5), np.nan)
    truth_e = None
    if 'external_obs_round' in iv:
        rng = np.random.default_rng(rd.CFG['seeds'].get('external', 4242))
        truth_e = np.where(rng.random(n) < iv.get('external_rho', 0.95), w['truth'], 1 - w['truth'])
    for i, ag in enumerate(agents):
        Lh[:, 0, i] = ag.logodds(); posh[:, 0, i] = ag.pos; est[:, 0, i] = ag.partner_belief_estimate()
    for r in range(R):
        order = (('A', A), ('B', B)) if cond.get('order', 'AB') == 'AB' else (('B', B), ('A', A))
        for name, ag in order:
            i = ag.i; other = agents[1 - i]
            if 'remove_c_from' in iv and r >= iv['remove_c_from'] and name in iv.get('remove_c_for', ['A', 'B']):
                ag.c_active[:] = 0.
            if 'restore_c_from' in iv and r >= iv['restore_c_from'] and name in iv.get('remove_c_for', ['A', 'B']):
                ag.c_active[:] = ag.c
            if truth_e is not None and r == iv['external_obs_round'] and i == 0:
                A.observe_external(truth_e, iv.get('external_rho', 0.95)); B.observe_external(truth_e, iv.get('external_rho', 0.95))
            prob, ig = ag.act_probs()
            a = rd.draw(prob, w['u'][:, r, i])
            if 'force_reassure' in iv and r == iv['force_reassure']['round'] and name == iv['force_reassure']['agent']:
                a = np.full(n, rd.REASSURE, np.int8)
            if 'force_act' in iv and r == iv['force_act']['round'] and name == iv['force_act']['agent']:
                a = np.full(n, int(iv['force_act']['act']), np.int8)
            acts[:, r, i] = a; probh[:, r, i, :prob.shape[1]] = prob
            if ig is not None: igh[:, r, i, :ig.shape[1]] = ig
            factual, v = ag.after_own_act(a)
            relief_h[:, r, i] = ag.relieved
            # partner observes
            placebo = bool(iv.get('placebo', False))
            if isinstance(other, rd.Level1):
                other.observe_partner_act(a, relief_j=ag.relieved_believed)   # partner's model conditions on the relief it believes it granted
            else:
                other.observe_partner_factual(factual, v)
            other.relieved_believed = ag.relief_partner.copy()
            other.relieved = ag.relief_partner & (not placebo)              # actual relief (placebo: believed but not granted)
            ag.relieved = np.zeros(n, bool); ag.relieved_believed = np.zeros(n, bool)   # consumed by this move
        for i, ag in enumerate(agents):
            Lh[:, r + 1, i] = ag.logodds(); posh[:, r + 1, i] = ag.pos; est[:, r + 1, i] = ag.partner_belief_estimate()
    out = {'truth': w['truth'], 'L': Lh, 'pos': posh, 'acts': acts, 'oracle': rd.oracle_logodds(w), 'est': est, 'ig': igh,
           'relief': relief_h, 'prob': probh, 'private': w['private'], 'A_omega': A.omega, 'A_c': A.c, 'B_omega': B.omega, 'B_c': B.c}
    for name, ag in (('A', A), ('B', B)):
        pc, pw = ag.partner_marginals()
        if pc is not None:
            out[f'{name}_pC_partner'] = pc; out[f'{name}_pW_partner'] = pw
    return out



def extra_metrics(d, df):
    a = d['acts']
    df['conc_A_8_12'] = (a[:, 7:, 0] == rd.CONCEDE).any(axis=1).astype(float)
    df['conc_A_9_12'] = (a[:, 8:, 0] == rd.CONCEDE).any(axis=1).astype(float)
    df['conc_B_8_12'] = (a[:, 7:, 1] == rd.CONCEDE).any(axis=1).astype(float)
    df['conc_A_1_7'] = (a[:, :7, 0] == rd.CONCEDE).any(axis=1).astype(float)
    df['A_concede_8_12'] = (a[:, 7:, 0] == rd.CONCEDE).mean(axis=1)
    df['A_pressure_8_12'] = (a[:, 7:, 0] == rd.PRESSURE).mean(axis=1)
    df['A_silence_8_12'] = (a[:, 7:, 0] == rd.SILENCE).mean(axis=1)
    q1 = rd.sigmoid(d['L']); truth = d['truth']
    pt = np.where(truth[:, None, None] == 1, q1, 1 - q1)
    df['A_p_truth_r7'] = pt[:, 7, 0]                       # belief after round index 6, before the intervention round
    df['A_dbelief_8_12'] = np.abs(q1[:, -1, 0] - q1[:, 7, 0])
    df['B_pC16'] = d['B_pC_partner'][:, int(np.argmin(np.abs(CG - 1.6)))]
    df['B_pW02'] = d['B_pW_partner'][:, int(np.argmin(np.abs(WG - 0.2)))]
    df['B_cmean'] = d['B_pC_partner'] @ CG
    df['B_wmean'] = d['B_pW_partner'] @ WG
    return df


def main():
    t0 = time.time()
    worlds = rd.make_worlds(NW, SEED)
    runs = {}; acts = {}; summary = []; timing = {}
    for a_t, b_t in PAIRS:
        for cname, iv in IV.items():
            key = (f'{a_t}x{b_t}', cname)
            t = time.time()
            d = dialogue_ext(worlds, ru.cond(a_t, b_t, BETA, intervention=iv))
            if cname in ('base', 'reassure'):   # dialogue_ext must reproduce core.dialogue bitwise where no force_act is used
                d0 = rd.dialogue(worlds, ru.cond(a_t, b_t, BETA, intervention=iv))
                assert np.array_equal(d0['acts'], d['acts']) and np.array_equal(d0['L'], d['L']), 'dialogue_ext diverges from core.dialogue'
            df, _ = rd.metrics(d); df = extra_metrics(d, df)
            runs[key] = df; acts[key] = d['acts']; timing['|'.join(key)] = round(time.time() - t, 2)
            for split, m in (('all', np.ones(NW, bool)), ('A_own0', df['A_own0'].to_numpy() == 1)):
                row = ru.summary_row(cname, df[m], split); row['pair'] = key[0]; summary.append(row)
            print(key, timing['|'.join(key)], 's', flush=True)
    summ = pd.DataFrame(summary); summ.to_csv(OUT / 'E3_summary.csv', index=False)
    # long per-world table
    long = pd.concat([df.assign(pair=k[0], condition=k[1]) for k, df in runs.items()], ignore_index=True)
    long.to_csv(OUT / 'E3_worlds.csv.gz', index=False)
    # paired contrasts (shared worlds), all worlds and A_own0 == 1
    rows = []
    for pair in [f'{a}x{b}' for a, b in PAIRS]:
        for x, y in CONTRASTS:
            dx, dy = runs[(pair, x)], runs[(pair, y)]
            for split, m in (('all', np.ones(NW, bool)), ('A_own0', dx['A_own0'].to_numpy() == 1)):
                for r in rd.paired(dx[m], dy[m], f'{x}-{y}'):
                    r.update(pair=pair, split=split, n=int(m.sum())); rows.append(r)
    con = pd.DataFrame(rows); con.to_csv(OUT / 'E3_contrasts.csv', index=False)
    # act identity vs base: share of worlds whose whole act sequence (A and B) is unchanged, and A's acts at indices 8..11
    ident = []
    for pair in [f'{a}x{b}' for a, b in PAIRS]:
        for x, y in CONTRASTS:
            ax_, ay_ = acts[(pair, x)], acts[(pair, y)]
            ident.append({'pair': pair, 'contrast': f'{x}-{y}',
                          'frac_worlds_A_acts_identical_9_12': float((ax_[:, 8:, 0] == ay_[:, 8:, 0]).all(axis=1).mean()),
                          'frac_worlds_all_acts_identical': float((ax_ == ay_).all(axis=(1, 2)).mean()),
                          'frac_A_moves_changed_9_12': float((ax_[:, 8:, 0] != ay_[:, 8:, 0]).mean())})
    pd.DataFrame(ident).to_csv(OUT / 'E3_acts_identity.csv', index=False)
    # ---- simulated power: attenuated x honest vs costly x honest, observation channels
    channels = {
        'tail non-concession A (base)': ('base', 'nonc_A'),
        'A pressure rate (base)': ('base', 'A_pressure'),
        'A silence rate (base)': ('base', 'A_silence'),
        'A assert rate (base)': ('base', 'A_assert'),
        'A private belief change |dq| (base)': ('base', 'A_dbelief'),
        'A truth-weighted belief (base)': ('base', 'A_p_truth'),
        'A concession in 8-12 after cost removal (remove_c_A)': ('remove_c_A', 'conc_A_8_12'),
        'A concession in 9-12 after partner reassure (reassure)': ('reassure', 'conc_A_9_12'),
        'A truth-weighted belief after external evidence (external)': ('external', 'A_p_truth'),
        'listener: P_B(c_A = 1.6) (base)': ('base', 'B_pC16'),
        'listener: P_B(omega_A = 0.2) (base)': ('base', 'B_pW02'),
        'listener: posterior mean c_A (base)': ('base', 'B_cmean'),
    }
    rng = np.random.default_rng(SEED + 1)
    prow = []; n80 = []
    for split in ('all', 'A_own0'):
        for ch, (cname, col) in channels.items():
            g1 = runs[('attenuatedxhonest', cname)]; g2 = runs[('costlyxhonest', cname)]
            if split == 'A_own0':
                g1 = g1[g1['A_own0'] == 1]; g2 = g2[g2['A_own0'] == 1]
            x1 = g1[col].to_numpy(float); x2 = g2[col].to_numpy(float)
            first = None
            for n in POWER_GRID:
                s1 = x1[rng.integers(0, len(x1), (N_RESAMPLE, n))]; s2 = x2[rng.integers(0, len(x2), (N_RESAMPLE, n))]
                v1 = s1.var(axis=1, ddof=1); v2 = s2.var(axis=1, ddof=1)
                ok = (v1 + v2) > 0
                p = np.ones(N_RESAMPLE)
                if ok.any():
                    p[ok] = stats.ttest_ind(s1[ok].T, s2[ok].T, equal_var=False).pvalue
                pw = float((p < ALPHA).mean())
                prow.append({'split': split, 'channel': ch, 'condition': cname, 'metric': col, 'n_per_group': n, 'power': pw,
                             'mean_attenuated': x1.mean(), 'mean_costly': x2.mean(),
                             'cohen_d': (x2.mean() - x1.mean()) / np.sqrt((x1.var(ddof=1) + x2.var(ddof=1)) / 2) if (x1.var() + x2.var()) > 0 else np.nan})
                if first is None and pw >= 0.8: first = n
            n80.append({'split': split, 'channel': ch, 'condition': cname, 'metric': col, 'n_per_group_80pct': first if first else f'not reached; power {pw:.2f} at {POWER_GRID[-1]}',
                        'mean_attenuated': x1.mean(), 'mean_costly': x2.mean(), 'cohen_d': prow[-1]['cohen_d']})
    pd.DataFrame(prow).to_csv(OUT / 'E3_power.csv', index=False)
    pd.DataFrame(n80).to_csv(OUT / 'E3_power_n80.csv', index=False)
    ru.write_manifest('E3', {'seed': SEED, 'n_worlds': NW, 'beta': BETA, 'pairs': PAIRS, 'interventions': IV, 'timing_s': timing,
                             'power': {'grid': POWER_GRID, 'resamples': N_RESAMPLE, 'alpha': ALPHA, 'test': 'Welch t two-sided', 'seed': SEED + 1},
                             'total_s': round(time.time() - t0, 1)})
    print('done', round(time.time() - t0, 1), 's')


if __name__ == '__main__':
    main()
