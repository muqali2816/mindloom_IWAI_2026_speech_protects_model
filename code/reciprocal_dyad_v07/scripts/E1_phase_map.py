"""E1 — phase map of mechanism pairs. PYTHONPATH=. python E1_phase_map.py

All 16 (A_type x B_type) pairs from protocol.json['agent_types'] x beta in {0, 1.5} = 32 conditions,
2000 common worlds (seed protocol.json['seeds']['E1']). Writes
  results/E1_<Atype>x<Btype>_beta<b>_worlds.csv   per-world metrics (core.metrics)
  results/E1_summary.csv                         summary_row for split 'all' and 'both_own' (A_own0==1 & B_own0==1)
  results/E1_trajectories.csv                    per-round means (A_p_truth_t, B_p_truth_t, nonc_mutual_t, U_t) per split
  results/E1_P1_reciprocal.csv                   reciprocal-amplification test (independence product + additive prediction)
  results/E1_P5_transparency.csv                 4x4 tables of transparency error and partner-mechanism recovery
  results/E1_pairs_matrix.csv                    4x4 matrices of key metrics (long format) for the heat-maps
No change to core.py / observer.py.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # package root on path
import json, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
import core as rd
from run_utils import cond, summary_row, OUT, T

SEED = rd.CFG['seeds']['E1']; NW = rd.CFG['n_worlds']; BETAS = rd.CFG['experiments']['E1_phase_map']['beta']
TYPES = list(T.keys())            # honest, attenuated, costly, mixed
worlds = rd.make_worlds(NW, SEED)
R = rd.CFG['rounds']


def splits(df):
    return {'all': np.ones(len(df), bool), 'both_own': ((df['A_own0'] == 1) & (df['B_own0'] == 1)).to_numpy()}


def trajectories(d, mask):
    truth = d['truth'].astype(int); q1 = rd.sigmoid(d['L']); pt = np.where(truth[:, None, None] == 1, q1, 1 - q1)
    a = d['acts']; pos = d['pos'].astype(int)
    out = {'A_p_truth_t': pt[mask, :, 0].mean(0), 'B_p_truth_t': pt[mask, :, 1].mean(0),
           'nonc_mutual_t': ((a[mask, :, 0] != rd.CONCEDE) & (a[mask, :, 1] != rd.CONCEDE)).mean(0),
           'A_concede_t': (a[mask, :, 0] == rd.CONCEDE).mean(0), 'B_concede_t': (a[mask, :, 1] == rd.CONCEDE).mean(0),
           'U_t': (pos[mask, :, 0] != pos[mask, :, 1]).mean(0)}
    return out


t0 = time.time(); rows = []; traj = []; dfs = {}; timing = {}
for beta in BETAS:
    for a_t, b_t in itertools.product(TYPES, TYPES):
        name = f'E1_{a_t}x{b_t}_beta{beta}'; t = time.time()
        d = rd.dialogue(worlds, cond(a_t, b_t, beta)); df, _ = rd.metrics(d)
        df.to_csv(OUT / f'{name}_worlds.csv', index=False); dfs[(a_t, b_t, beta)] = df
        for sp, m in splits(df).items():
            row = summary_row(name, df[m], sp); row.update(A_type=a_t, B_type=b_t, beta=beta); rows.append(row)
            for k, v in trajectories(d, m).items():
                for r, val in enumerate(v):
                    traj.append({'run': name, 'A_type': a_t, 'B_type': b_t, 'beta': beta, 'split': sp, 'metric': k,
                                 'round': r if k in ('A_p_truth_t', 'B_p_truth_t', 'U_t') else r + 1, 'value': val})
        timing[name] = round(time.time() - t, 2)
summary = pd.DataFrame(rows); summary.to_csv(OUT / 'E1_summary.csv', index=False)
pd.DataFrame(traj).to_csv(OUT / 'E1_trajectories.csv', index=False)

# ---- 4x4 matrices (long) for heat-maps
mat = []
for beta in BETAS:
    for sp in ('all', 'both_own'):
        s = summary[(summary.beta == beta) & (summary.split == sp)]
        for _, r in s.iterrows():
            mat.append({'beta': beta, 'split': sp, 'A_type': r.A_type, 'B_type': r.B_type, 'n': r.n,
                        'nonc_mutual': r.nonc_mutual, 'nonc_A': r.nonc_A, 'nonc_B': r.nonc_B,
                        'hidden_shift_mean': (r.A_hidden_shift + r.B_hidden_shift) / 2,
                        'belief_gap_final': r.belief_gap_final, 'U_final': r.U_final, 'no_concession_either': r.no_concession_either,
                        'A_p_truth': r.A_p_truth, 'B_p_truth': r.B_p_truth, 'dist_oracle': r.dist_oracle,
                        'A_transp_err': r.A_transp_err, 'B_transp_err': r.B_transp_err,
                        'A_pC_true': r.A_pC_true, 'A_pW_true': r.A_pW_true, 'B_pC_true': r.B_pC_true, 'B_pW_true': r.B_pW_true})
pd.DataFrame(mat).to_csv(OUT / 'E1_pairs_matrix.csv', index=False)


# ---- P1: reciprocal amplification (common worlds -> paired intervals; bootstrap for products of means)
def boot_ci(fn, n, n_boot=4000, seed=1):
    rng = np.random.default_rng(seed); idx = np.arange(n); vals = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True); vals.append(fn(b))
    return np.percentile(vals, [2.5, 97.5])


p1 = []
for beta in BETAS:
    for typ in ('costly', 'attenuated', 'mixed'):
        cc = dfs[(typ, typ, beta)]; ch = dfs[(typ, 'honest', beta)]; hc = dfs[('honest', typ, beta)]; hh = dfs[('honest', 'honest', beta)]
        for sp in ('all', 'both_own'):
            m = splits(cc)[sp] & splits(ch)[sp] & splits(hc)[sp] & splits(hh)[sp]   # common worlds where all four runs satisfy the split
            obs = cc.nonc_mutual.to_numpy()[m]; pA = ch.nonc_A.to_numpy()[m]; pB = hc.nonc_B.to_numpy()[m]
            mm_hh = hh.nonc_mutual.to_numpy()[m]; mm_ch = ch.nonc_mutual.to_numpy()[m]; mm_hc = hc.nonc_mutual.to_numpy()[m]
            # (1) independence product of the single-agent tail non-concession rates: p_A(typ x honest) * p_B(honest x typ)
            prod_point = pA.mean() * pB.mean(); delta_prod = obs.mean() - prod_point
            lo, hi = boot_ci(lambda b: obs[b].mean() - pA[b].mean() * pB[b].mean(), len(obs))
            # (1b) per-world product (world-paired): nonc_mutual_cc[w] - nonc_A_ch[w] * nonc_B_hc[w]
            dw = obs - pA * pB; se_w = dw.std(ddof=1) / np.sqrt(len(dw))
            # (2) additive prediction: nonc_mutual(hh) + [nonc_mutual(typ x h) - hh] + [nonc_mutual(h x typ) - hh]
            add_w = mm_ch + mm_hc - mm_hh; da = obs - add_w; se_a = da.std(ddof=1) / np.sqrt(len(da))
            p1.append({'beta': beta, 'pair': f'{typ}x{typ}', 'split': sp, 'n_common': int(m.sum()),
                       'nonc_mutual_obs': obs.mean(), 'nonc_A_single': pA.mean(), 'nonc_B_single': pB.mean(),
                       'pred_independence_product': prod_point, 'delta_vs_product': delta_prod, 'delta_vs_product_lo': lo, 'delta_vs_product_hi': hi,
                       'delta_vs_product_perworld': dw.mean(), 'delta_vs_product_perworld_lo': dw.mean() - 1.96 * se_w, 'delta_vs_product_perworld_hi': dw.mean() + 1.96 * se_w,
                       'nonc_mutual_hh': mm_hh.mean(), 'nonc_mutual_typxh': mm_ch.mean(), 'nonc_mutual_hxtyp': mm_hc.mean(),
                       'pred_additive': add_w.mean(), 'delta_vs_additive': da.mean(), 'delta_vs_additive_lo': da.mean() - 1.96 * se_a, 'delta_vs_additive_hi': da.mean() + 1.96 * se_a})
pd.DataFrame(p1).to_csv(OUT / 'E1_P1_reciprocal.csv', index=False)

# ---- P5: transparency error and partner-mechanism recovery by partner type
p5 = []
prior = 1 / len(rd.P['omega_grid'])
for beta in BETAS:
    for sp in ('all', 'both_own'):
        for a_t, b_t in itertools.product(TYPES, TYPES):
            df = dfs[(a_t, b_t, beta)]; m = splits(df)[sp]; x = df[m]
            rec = {'beta': beta, 'split': sp, 'A_type': a_t, 'B_type': b_t, 'n': int(m.sum())}
            for col in ('A_transp_err', 'B_transp_err', 'A_pC_true', 'A_pW_true', 'B_pC_true', 'B_pW_true', 'A_dbelief', 'B_dbelief'):
                rec[col] = x[col].mean(); rec[col + '_se'] = x[col].std(ddof=1) / np.sqrt(len(x))
            rec['A_pW_true_minus_prior'] = rec['A_pW_true'] - prior; rec['A_pC_true_minus_prior'] = rec['A_pC_true'] - prior
            rec['B_pW_true_minus_prior'] = rec['B_pW_true'] - prior; rec['B_pC_true_minus_prior'] = rec['B_pC_true'] - prior
            p5.append(rec)
pd.DataFrame(p5).to_csv(OUT / 'E1_P5_transparency.csv', index=False)

# ---- P5 paired contrasts: same A type, partner attenuated vs honest / costly / mixed (common worlds)
pc = []
for beta in BETAS:
    for sp in ('all', 'both_own'):
        for a_t in TYPES:
            ref = dfs[(a_t, 'honest', beta)]
            for b_t in ('attenuated', 'costly', 'mixed'):
                oth = dfs[(a_t, b_t, beta)]; m = splits(ref)[sp] & splits(oth)[sp]
                for col in ('A_transp_err', 'A_pW_true', 'A_pC_true'):
                    dlt = (oth[col].to_numpy()[m] - ref[col].to_numpy()[m]); se = dlt.std(ddof=1) / np.sqrt(len(dlt))
                    pc.append({'beta': beta, 'split': sp, 'A_type': a_t, 'contrast': f'B={b_t} minus B=honest', 'metric': col, 'n': int(m.sum()),
                               'delta': dlt.mean(), 'mc_lo': dlt.mean() - 1.96 * se, 'mc_hi': dlt.mean() + 1.96 * se})
pd.DataFrame(pc).to_csv(OUT / 'E1_P5_paired.csv', index=False)

# ---- P1 rank check: is costly x costly the maximum of nonc_mutual and of mean hidden shift over the 16 pairs?
rk = []
M = pd.DataFrame(mat)
for beta in BETAS:
    for sp in ('all', 'both_own'):
        s = M[(M.beta == beta) & (M.split == sp)].copy(); s['pair'] = s.A_type + 'x' + s.B_type
        for col in ('nonc_mutual', 'hidden_shift_mean', 'no_concession_either'):
            top = s.sort_values(col, ascending=False).iloc[0]
            rk.append({'beta': beta, 'split': sp, 'metric': col, 'max_pair': top.pair, 'max_value': top[col],
                       'costlyxcostly': float(s[s.pair == 'costlyxcostly'][col].iloc[0]), 'attenuatedxattenuated': float(s[s.pair == 'attenuatedxattenuated'][col].iloc[0]),
                       'honestxhonest': float(s[s.pair == 'honestxhonest'][col].iloc[0]), 'rank_costlyxcostly': int((s[col] > float(s[s.pair == 'costlyxcostly'][col].iloc[0])).sum() + 1)})
pd.DataFrame(rk).to_csv(OUT / 'E1_P1_rank.csv', index=False)

meta = {'seed': SEED, 'n_worlds': NW, 'rounds': R, 'tail_rounds': rd.CFG['tail_rounds'], 'betas': BETAS, 'types': {k: T[k] for k in TYPES},
        'attenuation_rule': rd.ATTENUATION, 'timing_s': timing, 'total_s': round(time.time() - t0, 1)}
(OUT / 'E1_meta.json').write_text(json.dumps(meta, indent=1))
print(json.dumps({'total_s': meta['total_s'], 'n_summary_rows': len(summary), 'n_worlds_both_own_hh_b1.5': int(summary[(summary.run == 'E1_honestxhonest_beta1.5') & (summary.split == 'both_own')].n.iloc[0])}))
