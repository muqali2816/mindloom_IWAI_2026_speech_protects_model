"""E4 — temporal-regime criteria. PYTHONPATH=. python E4_regimes.py

Seed protocol.json['seeds']['E4'], 2000 common worlds, R = 24 rounds (rd.CFG['rounds'] is set to 24 in this script
before make_worlds; tail_rounds stays 4). Conditions: all 16 type pairs x beta in {0, 1.5} (baseline), plus the
hysteresis intervention {'remove_c_from': 7, 'restore_c_from': 10, 'remove_c_for': ['A','B']} for all 16 pairs at
beta = 1.5 and for costly x costly at beta = 0. Round indices in the intervention dict are 0-based code rounds
(c removed on code rounds 7,8,9 = rounds 8-10 in 1-based numbering; restored from code round 10 = round 11).

State per round: MN (both acts != CONCEDE), ONE (exactly one CONCEDE), BOTH (both CONCEDE).
Outputs
  results/E4_persistence.csv        per condition x split: P(MN_{r+1}|MN_r) overall / tail, share of worlds with max MN run >= 6, run-length distribution summary
  results/E4_runlengths.csv         pooled MN run-length histogram per condition x split
  results/E4_transitions.csv        3x3 transition counts/probabilities per round and pooled, per condition x split
  results/E4_occupancy.csv          state occupancy per round
  results/E4_hysteresis.csv         per-round concede rate / U_t / MN_t, intervention vs control, paired deltas
  results/E4_hysteresis_test.csv    return-to-baseline test (2- and 4-round windows after restoration)
  results/E4_forecast.csv           held-out next-act forecast of A: models (a) last A act, (b) last A+B acts, (c) regime + last A act, (d) regime + last A+B acts
  results/E4_<cond>_worlds.csv      per-world metrics for the baseline conditions (core.metrics, tail = last 4 of 24 rounds)
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # package root on path
import json, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
import core as rd
from run_utils import cond, summary_row, OUT, T

R24 = 24
rd.CFG['rounds'] = R24                      # temporary, this script only
SEED = rd.CFG['seeds']['E4']; NW = rd.CFG['n_worlds']; R = R24
TYPES = list(T.keys()); BETAS = [0.0, 1.5]
IV = {'remove_c_from': 7, 'restore_c_from': 10, 'remove_c_for': ['A', 'B']}
worlds = rd.make_worlds(NW, SEED)
MIN_RUN = 6; TAIL8 = 8


def splits(d):
    q1 = rd.sigmoid(d['L'][:, 0, :])
    both = (q1[:, 0] > 0.5) & (q1[:, 1] < 0.5)
    return {'all': np.ones(len(both), bool), 'both_own': both}


def states(d):
    c = d['acts'] == rd.CONCEDE                     # (n,R,2)
    return np.where(~c[:, :, 0] & ~c[:, :, 1], 0, np.where(c[:, :, 0] & c[:, :, 1], 2, 1)).astype(int)   # 0=MN,1=ONE,2=BOTH


def run_lengths(mn):
    """mn (n,R) bool -> list of arrays of maximal run lengths per world, and max run per world."""
    n, Rr = mn.shape; maxrun = np.zeros(n, int); runs = []
    for w in range(n):
        x = mn[w]; cur = 0; lst = []
        for v in x:
            if v: cur += 1
            else:
                if cur: lst.append(cur)
                cur = 0
        if cur: lst.append(cur)
        runs.append(lst); maxrun[w] = max(lst) if lst else 0
    return runs, maxrun


t0 = time.time(); D = {}; timing = {}
for beta in BETAS:
    for a_t, b_t in itertools.product(TYPES, TYPES):
        t = time.time(); name = f'E4_{a_t}x{b_t}_beta{beta}'
        D[(a_t, b_t, beta, 'base')] = rd.dialogue(worlds, cond(a_t, b_t, beta)); timing[name] = round(time.time() - t, 2)
        df, _ = rd.metrics(D[(a_t, b_t, beta, 'base')]); df.to_csv(OUT / f'{name}_worlds.csv', index=False)
for a_t, b_t in itertools.product(TYPES, TYPES):
    D[(a_t, b_t, 1.5, 'iv')] = rd.dialogue(worlds, cond(a_t, b_t, 1.5, intervention=IV))
D[('costly', 'costly', 0.0, 'iv')] = rd.dialogue(worlds, cond('costly', 'costly', 0.0, intervention=IV))
print(json.dumps({'sim_s': round(time.time() - t0, 1)}))

# ---------------------------------------------------------------- (i) persistence, (ii) transitions
pers, rl_rows, tr_rows, occ_rows = [], [], [], []
for (a_t, b_t, beta, kind), d in D.items():
    if kind != 'base': continue
    S = states(d); mn = S == 0
    for sp, m in splits(d).items():
        Sm = S[m]; mnm = mn[m]; n = int(m.sum())
        runs, maxrun = run_lengths(mnm)
        allruns = np.concatenate([np.array(r, int) for r in runs]) if any(runs) else np.array([], int)
        # persistence P(MN_{r+1} | MN_r)
        stay = (mnm[:, 1:] & mnm[:, :-1]).sum(); base = mnm[:, :-1].sum()
        stay_t = (mnm[:, -TAIL8 + 1:] & mnm[:, -TAIL8:-1]).sum(); base_t = mnm[:, -TAIL8:-1].sum()
        # exits from MN per round -> hazard
        pers.append({'A_type': a_t, 'B_type': b_t, 'beta': beta, 'split': sp, 'n': n,
                     'P_stay_MN_all_rounds': stay / base if base else np.nan, 'P_stay_MN_tail8': stay_t / base_t if base_t else np.nan,
                     'share_maxrun_ge6': (maxrun >= MIN_RUN).mean(), 'share_maxrun_ge12': (maxrun >= 12).mean(), 'share_MN_all_24': (maxrun == R).mean(),
                     'mean_maxrun': maxrun.mean(), 'median_maxrun': float(np.median(maxrun)), 'mean_runlength': allruns.mean() if len(allruns) else np.nan,
                     'n_runs': len(allruns), 'share_MN_tail4': mnm[:, -4:].mean(), 'occ_MN_final': (Sm[:, -1] == 0).mean(), 'occ_ONE_final': (Sm[:, -1] == 1).mean(), 'occ_BOTH_final': (Sm[:, -1] == 2).mean()})
        h = np.bincount(allruns, minlength=R + 1) if len(allruns) else np.zeros(R + 1, int)
        hm = np.bincount(maxrun, minlength=R + 1)
        for L in range(R + 1):
            rl_rows.append({'A_type': a_t, 'B_type': b_t, 'beta': beta, 'split': sp, 'run_length': L, 'n_runs': int(h[L]), 'n_worlds_maxrun': int(hm[L])})
        # transitions per round and pooled
        for r in range(R - 1):
            for i in range(3):
                sel = Sm[:, r] == i; tot = sel.sum()
                for j in range(3):
                    cnt = int((Sm[sel, r + 1] == j).sum())
                    tr_rows.append({'A_type': a_t, 'B_type': b_t, 'beta': beta, 'split': sp, 'round_from': r + 1, 'from': i, 'to': j, 'count': cnt, 'prob': cnt / tot if tot else np.nan})
        for i in range(3):
            sel = Sm[:, :-1] == i; tot = sel.sum()
            for j in range(3):
                cnt = int((Sm[:, 1:][sel] == j).sum())
                tr_rows.append({'A_type': a_t, 'B_type': b_t, 'beta': beta, 'split': sp, 'round_from': 0, 'from': i, 'to': j, 'count': cnt, 'prob': cnt / tot if tot else np.nan})
        for r in range(R):
            for i in range(3):
                occ_rows.append({'A_type': a_t, 'B_type': b_t, 'beta': beta, 'split': sp, 'round': r + 1, 'state': i, 'share': (Sm[:, r] == i).mean()})
pd.DataFrame(pers).to_csv(OUT / 'E4_persistence.csv', index=False)
pd.DataFrame(rl_rows).to_csv(OUT / 'E4_runlengths.csv', index=False)
pd.DataFrame(tr_rows).to_csv(OUT / 'E4_transitions.csv', index=False)
pd.DataFrame(occ_rows).to_csv(OUT / 'E4_occupancy.csv', index=False)

# ---------------------------------------------------------------- (iii) hysteresis
hy, hyt = [], []
PRE = list(range(3, 7))                    # code rounds 3..6 (1-based 4..7): pre-removal baseline window
for (a_t, b_t, beta, kind), div in D.items():
    if kind != 'iv': continue
    dbase = D[(a_t, b_t, beta, 'base')]
    for sp, m in splits(dbase).items():
        n = int(m.sum())
        series = {}
        for lab, d in (('control', dbase), ('intervention', div)):
            a = d['acts'][m]; pos = d['pos'][m].astype(int); q1 = rd.sigmoid(d['L'][m])
            series[lab] = {'A_concede': (a[:, :, 0] == rd.CONCEDE).astype(float), 'B_concede': (a[:, :, 1] == rd.CONCEDE).astype(float),
                           'any_concede': ((a[:, :, 0] == rd.CONCEDE) | (a[:, :, 1] == rd.CONCEDE)).astype(float),
                           'MN': ((a[:, :, 0] != rd.CONCEDE) & (a[:, :, 1] != rd.CONCEDE)).astype(float),
                           'U': (pos[:, 1:, 0] != pos[:, 1:, 1]).astype(float), 'belief_gap': np.abs(q1[:, 1:, 0] - q1[:, 1:, 1])}
        for met in series['control']:
            for r in range(R):
                c_ = series['control'][met][:, r]; i_ = series['intervention'][met][:, r]; dl = i_ - c_; se = dl.std(ddof=1) / np.sqrt(n)
                phase = 'pre' if r < IV['remove_c_from'] else 'removed' if r < IV['restore_c_from'] else 'restored'
                hy.append({'A_type': a_t, 'B_type': b_t, 'beta': beta, 'split': sp, 'n': n, 'metric': met, 'round': r + 1, 'code_round': r, 'phase': phase,
                           'control': c_.mean(), 'intervention': i_.mean(), 'delta': dl.mean(), 'mc_lo': dl.mean() - 1.96 * se, 'mc_hi': dl.mean() + 1.96 * se})
            # return-to-baseline tests (paired per world)
            X = series['intervention'][met]; C = series['control'][met]
            pre = X[:, PRE].mean(1); rem = X[:, IV['remove_c_from']:IV['restore_c_from']].mean(1)
            for win in (2, 4):
                post = X[:, IV['restore_c_from']:IV['restore_c_from'] + win].mean(1); ctrl_post = C[:, IV['restore_c_from']:IV['restore_c_from'] + win].mean(1)
                d1 = post - pre; s1 = d1.std(ddof=1) / np.sqrt(n); d2 = post - ctrl_post; s2 = d2.std(ddof=1) / np.sqrt(n)
                hyt.append({'A_type': a_t, 'B_type': b_t, 'beta': beta, 'split': sp, 'n': n, 'metric': met, 'window_rounds_after_restore': win,
                            'pre_removal_mean(code_r3-6)': pre.mean(), 'during_removal_mean': rem.mean(), 'post_restore_mean': post.mean(), 'control_same_rounds': ctrl_post.mean(),
                            'post_minus_pre': d1.mean(), 'post_minus_pre_lo': d1.mean() - 1.96 * s1, 'post_minus_pre_hi': d1.mean() + 1.96 * s1,
                            'post_minus_control': d2.mean(), 'post_minus_control_lo': d2.mean() - 1.96 * s2, 'post_minus_control_hi': d2.mean() + 1.96 * s2,
                            'returned_to_pre_CI': bool((d1.mean() - 1.96 * s1) <= 0 <= (d1.mean() + 1.96 * s1)), 'returned_to_control_CI': bool((d2.mean() - 1.96 * s2) <= 0 <= (d2.mean() + 1.96 * s2))})
pd.DataFrame(hy).to_csv(OUT / 'E4_hysteresis.csv', index=False)
pd.DataFrame(hyt).to_csv(OUT / 'E4_hysteresis_test.csv', index=False)


# ---------------------------------------------------------------- (iv) next-act forecast on held-out worlds
def features(d, t):
    """Context features for predicting A's act at code round t (t >= 3)."""
    a = d['acts']; pos = d['pos'].astype(int)
    lastA = a[:, t - 1, 0].astype(int); lastB = a[:, t - 1, 1].astype(int)
    mn = (a[:, :, 0] != rd.CONCEDE) & (a[:, :, 1] != rd.CONCEDE)
    run3 = mn[:, t - 3:t].all(1).astype(int)              # mutual non-concession on the last 3 rounds
    div = (pos[:, t, 0] != pos[:, t, 1]).astype(int)      # public positions diverge after round t-1
    regime = run3 * 2 + div
    return {'a': lastA, 'b': lastA * 5 + lastB, 'c': regime * 5 + lastA, 'd': regime * 25 + lastA * 5 + lastB}, a[:, t, 0].astype(int)


def fit_predict(dlist, alpha=0.5):
    """dlist: list of dialogue dicts (pooled). Train on world index < n/2, test on the rest. Returns per-model held-out LL and acc arrays (per prediction) and per-world means."""
    ctx_tr = {k: [] for k in 'abcd'}; y_tr = []; ctx_te = {k: [] for k in 'abcd'}; y_te = []; w_te = []
    for d in dlist:
        n = len(d['truth']); half = n // 2
        for t in range(3, R):
            f, y = features(d, t)
            for k in 'abcd':
                ctx_tr[k].append(f[k][:half]); ctx_te[k].append(f[k][half:])
            y_tr.append(y[:half]); y_te.append(y[half:]); w_te.append(np.arange(half, n))
    y_tr = np.concatenate(y_tr); y_te = np.concatenate(y_te); w_te = np.concatenate(w_te)
    out = {}
    for k in 'abcd':
        ctr = np.concatenate(ctx_tr[k]); cte = np.concatenate(ctx_te[k]); K = int(max(ctr.max(), cte.max())) + 1
        cnt = np.zeros((K, 5)); np.add.at(cnt, (ctr, y_tr), 1); prob = (cnt + alpha) / (cnt + alpha).sum(1, keepdims=True)
        p = prob[cte]; ll = np.log(p[np.arange(len(y_te)), y_te]); acc = (p.argmax(1) == y_te).astype(float)
        out[k] = {'ll': ll, 'acc': acc, 'n_contexts_used': int((cnt.sum(1) > 0).sum()), 'K': K}
    return out, y_te, w_te


def boot_diff(x, y, w, n_boot=2000, seed=3):
    """paired bootstrap over held-out worlds of mean(x - y)."""
    uw = np.unique(w); idx = {u: i for i, u in enumerate(uw)}; wi = np.array([idx[v] for v in w])
    per_world = np.bincount(wi, weights=x - y) / np.bincount(wi)
    rng = np.random.default_rng(seed); bs = [per_world[rng.integers(0, len(per_world), len(per_world))].mean() for _ in range(n_boot)]
    return per_world.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


fc = []
groups = {f'{a_t}x{b_t}_beta{beta}': [D[(a_t, b_t, beta, 'base')]] for beta in BETAS for a_t, b_t in itertools.product(TYPES, TYPES)}
groups['pooled16_beta1.5'] = [D[(a_t, b_t, 1.5, 'base')] for a_t, b_t in itertools.product(TYPES, TYPES)]
groups['pooled16_beta0.0'] = [D[(a_t, b_t, 0.0, 'base')] for a_t, b_t in itertools.product(TYPES, TYPES)]
for gname, dl in groups.items():
    res, y_te, w_te = fit_predict(dl)
    base_ll = np.log(np.bincount(y_te, minlength=5) / len(y_te) + 1e-12)[y_te].mean()
    for k, lab in zip('abcd', ('last A act', 'last A + last B act', 'regime(run>=3, positions diverge) + last A act', 'regime + last A + last B act')):
        row = {'group': gname, 'model': k, 'model_desc': lab, 'n_pred_test': len(y_te), 'n_test_worlds': len(np.unique(w_te)), 'rounds_predicted': f'code {3}..{R - 1}',
               'mean_loglik': res[k]['ll'].mean(), 'accuracy': res[k]['acc'].mean(), 'n_contexts': res[k]['n_contexts_used'], 'marginal_loglik': base_ll}
        for ref in 'ab':
            if k in 'cd' or (k == 'b' and ref == 'a'):
                for met in ('ll', 'acc'):
                    m_, lo, hi = boot_diff(res[k][met], res[ref][met], w_te)
                    row[f'd{met}_vs_{ref}'] = m_; row[f'd{met}_vs_{ref}_lo'] = lo; row[f'd{met}_vs_{ref}_hi'] = hi
        fc.append(row)
pd.DataFrame(fc).to_csv(OUT / 'E4_forecast.csv', index=False)

meta = {'seed': SEED, 'n_worlds': NW, 'rounds': R, 'tail_rounds_metrics': rd.CFG['tail_rounds'], 'tail8_for_persistence': TAIL8, 'min_run': MIN_RUN,
        'intervention': IV, 'pre_window_code_rounds': PRE, 'betas': BETAS, 'forecast_train_worlds': 'index < 1000', 'forecast_test_worlds': 'index >= 1000',
        'forecast_smoothing_alpha': 0.5, 'timing_s': timing, 'total_s': round(time.time() - t0, 1)}
(OUT / 'E4_meta.json').write_text(json.dumps(meta, indent=1))
print(json.dumps({'total_s': meta['total_s']}))
