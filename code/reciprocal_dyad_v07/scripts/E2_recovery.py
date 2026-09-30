"""E2: mechanism recovery for agent A. E2a — from public acts (channels 'acts+private', 'acts');
E2b — from speech labels (g x noise). PYTHONPATH=. python E2_recovery.py
Writes results/E2a_recovery.csv, results/E2a_channel_contrasts.csv, results/E2a_sensitivity.csv,
results/E2a_nextact.csv, results/E2b_labels.csv, results/E2b_contrasts.csv and per-world caches results/E2_cache/*.npz.
Does not modify core.py / observer.py."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # package root on path
import json, time
import numpy as np, pandas as pd
import core as rd, observer as ob
from run_utils import cond, OUT, ROOT, T

SEED = rd.CFG['seeds']['E2']; NW = rd.CFG['n_worlds']; BETA = rd.P['beta_active']; R = rd.CFG['rounds']
CACHE = OUT / 'E2_cache'; CACHE.mkdir(exist_ok=True)
HYPS, FAM = ob.hypotheses()                                    # [(0.2,0),(0.6,0),(1,0),(1,0.8),(1,1.6)]
REF = HYPS.index((1.0, 0.0))
PRIOR_OMEGA, PRIOR_C = float(np.mean(rd.P['omega_grid'])), float(np.mean(rd.P['cost_grid']))   # 0.6, 0.8 (task definition)
OBS_PRIOR_OMEGA, OBS_PRIOR_C = float(np.mean([h[0] for h in HYPS])), float(np.mean([h[1] for h in HYPS]))  # observer's uniform prior over 5 hyps
FAMILY_OF = {'honest': 'both', 'attenuated': 'attenuation', 'costly': 'cost', 'mixed': 'neither'}
GENS = [('honest', 'honest'), ('attenuated', 'honest'), ('costly', 'honest'), ('mixed', 'honest'),
        ('costly', 'costly'), ('attenuated', 'attenuated'), ('mixed', 'mixed')]
N_BOOT = 1000; BOOT_SEED = SEED + 777
LOG = []


def log(*a):
    s = ' '.join(str(x) for x in a); LOG.append(s); print(s, flush=True)


def gen_name(a, b):
    return f'{a}x{b}'


def cached(key, fn):
    f = CACHE / f'{key}.npz'
    if f.exists():
        z = np.load(f); return {k: z[k] for k in z.files}
    t = time.time(); res = fn(); out = {k: np.asarray(v) for k, v in res.items() if k != 'hyps'}
    np.savez_compressed(f, **out); log(f'  {key}: {time.time() - t:.1f}s'); return out


def true_params(a_type):
    return float(T[a_type]['omega']), float(T[a_type]['c'])


def per_world_correct(res, a_type):
    """Indicator (n,) that the family posterior favours the true family; NaN vector if undefined."""
    fam = FAMILY_OF[a_type]; pa = res['p_attenuation']
    if fam == 'attenuation': return (pa > 0.5).astype(float)
    if fam == 'cost': return (pa < 0.5).astype(float)
    return np.full(len(pa), np.nan)


def summarise(res, a_type):
    om, c = true_params(a_type); fam = FAMILY_OF[a_type]; pa = res['p_attenuation']
    LL = res['loglik']; mapi = LL.argmax(axis=1)
    row = {'true_family': fam, 'true_omega': om, 'true_c': c, 'n_worlds': len(pa),
           'p_attenuation_mean': float(pa.mean()), 'frac_attenuation_favoured': float((pa > 0.5).mean())}
    if fam in ('attenuation', 'cost'):
        p_true = pa if fam == 'attenuation' else 1 - pa
        row['correct_family'] = float((p_true > 0.5).mean()); row['posterior_on_family'] = float(p_true.mean())
    else:
        row['correct_family'] = np.nan; row['posterior_on_family'] = np.nan
    row['omega_MAE'] = float(np.abs(res['omega_mean'] - om).mean()); row['c_MAE'] = float(np.abs(res['c_mean'] - c).mean())
    row['omega_MAE_prior'] = abs(PRIOR_OMEGA - om); row['c_MAE_prior'] = abs(PRIOR_C - c)
    row['omega_MAE_obsprior'] = abs(OBS_PRIOR_OMEGA - om); row['c_MAE_obsprior'] = abs(OBS_PRIOR_C - c)
    row['omega_mean_est'] = float(res['omega_mean'].mean()); row['c_mean_est'] = float(res['c_mean'].mean())
    if (om, c) in HYPS:
        row['map_true_point'] = float((mapi == HYPS.index((om, c))).mean())
    else:
        row['map_true_point'] = np.nan
    for h, (ho, hc) in enumerate(HYPS):
        row[f'map_share_w{ho}_c{hc}'] = float((mapi == h).mean())
    return row


def boot_paired(x, y, seed):
    """Bootstrap (by worlds) 95% interval of mean(x - y) for per-world indicators."""
    d = (x - y); d = d[~np.isnan(d)]
    if len(d) == 0: return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(d), (N_BOOT, len(d)))
    bs = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(d)


def nextact_rows(res, a_type, gen, channel, ll_true_extra=None):
    """Mean log-probability per act of A's acts under best hypothesis of each family minus under the true point (nats/act)."""
    LL = res['loglik']; om, c = true_params(a_type); fam = FAMILY_OF[a_type]
    att_idx = FAM['attenuation']; cost_idx = FAM['cost']
    att_ex = [i for i in att_idx if i != REF]; cost_ex = [i for i in cost_idx if i != REF]
    if ll_true_extra is not None: ll_true = ll_true_extra
    else: ll_true = LL[:, HYPS.index((om, c))]
    rows = []
    def add(label, ll_alt):
        rows.append({'generator': gen, 'channel': channel, 'true_family': fam, 'comparison': label,
                     'dll_per_act': float(((ll_alt - ll_true) / R).mean()),
                     'dll_per_act_se': float(((ll_alt - ll_true) / R).std(ddof=1) / np.sqrt(len(ll_true))),
                     'frac_alt_better': float((ll_alt > ll_true).mean()),
                     'logp_true_per_act': float((ll_true / R).mean()), 'logp_alt_per_act': float((ll_alt / R).mean())})
    if fam == 'attenuation':
        add('best_wrong_family(cost incl. ref)', LL[:, cost_idx].max(axis=1)); add('best_wrong_family(cost excl. ref)', LL[:, cost_ex].max(axis=1))
        add('best_true_family(attenuation)', LL[:, att_idx].max(axis=1)); add('reference(1,0)', LL[:, REF])
    elif fam == 'cost':
        add('best_wrong_family(attenuation incl. ref)', LL[:, att_idx].max(axis=1)); add('best_wrong_family(attenuation excl. ref)', LL[:, att_ex].max(axis=1))
        add('best_true_family(cost)', LL[:, cost_idx].max(axis=1)); add('reference(1,0)', LL[:, REF])
    elif fam == 'both':
        add('best_attenuation_excl_ref', LL[:, att_ex].max(axis=1)); add('best_cost_excl_ref', LL[:, cost_ex].max(axis=1))
    else:  # mixed: true point off-grid; ll_true_extra is the LL under the true (omega, c)
        add('best_attenuation_family', LL[:, att_idx].max(axis=1)); add('best_cost_family', LL[:, cost_idx].max(axis=1))
        add('best_any_grid_hypothesis', LL.max(axis=1)); add('reference(1,0)', LL[:, REF])
    return rows


def loglik_true_point(d, channel, om, c, beta):
    """LL of A's acts under an off-grid true point (used for mixed generators), same marginalisation as observer.recover."""
    acts = d['acts']; n = acts.shape[0]
    if channel == 'acts+private':
        s = d['private'][:, 0].sum(axis=1); ll = np.zeros(n)
        for sc in range(rd.N + 1):
            m = s == sc
            if m.any(): ll[m] = ob.replay_loglik(acts[m], 0, (om, c), beta, sc)
        return ll
    ps = rd.PS_GIVEN_X.mean(axis=0)
    lls = np.stack([ob.replay_loglik(acts, 0, (om, c), beta, sc) for sc in range(rd.N + 1)], axis=1)
    mx = lls.max(axis=1, keepdims=True)
    return mx[:, 0] + np.log((np.exp(lls - mx) * ps[None, :]).sum(axis=1))


# ----------------------------------------------------------------------------- E2a
t0 = time.time(); log(f'E2 seed={SEED} n_worlds={NW} beta={BETA}')
worlds = rd.make_worlds(NW, SEED)
dialogs = {}; rows_a = []; contrasts = []; nextact = []; per_world = {}
for a_type, b_type in GENS:
    gen = gen_name(a_type, b_type); t = time.time()
    d = rd.dialogue(worlds, cond(a_type, b_type, BETA)); dialogs[gen] = d
    log(f'{gen}: dialogue {time.time() - t:.1f}s; A_own0={float((rd.sigmoid(d["L"][:, 0, 0]) > 0.5).mean()):.3f}')
    for channel in ('acts+private', 'acts'):
        res = cached(f'E2a_{gen}_{channel}', lambda: ob.recover(d, 0, channel, BETA))
        row = dict(generator=gen, A_type=a_type, B_type=b_type, channel=channel, beta_obs=BETA, **summarise(res, a_type))
        rows_a.append(row); per_world[(gen, channel)] = res
        extra = None
        if a_type == 'mixed':
            om, c = true_params(a_type)
            extra = cached(f'E2a_{gen}_{channel}_truepoint', lambda: {'ll': loglik_true_point(d, channel, om, c, BETA)})['ll']
        nextact += nextact_rows(res, a_type, gen, channel, extra)
    # paired channel contrast (bootstrap by worlds)
    cp, ca = per_world[(gen, 'acts+private')], per_world[(gen, 'acts')]
    for metric, fx in (('correct_family', lambda r: per_world_correct(r, a_type)),
                       ('posterior_on_family', lambda r: (r['p_attenuation'] if FAMILY_OF[a_type] == 'attenuation' else 1 - r['p_attenuation']) if FAMILY_OF[a_type] in ('attenuation', 'cost') else np.full(NW, np.nan)),
                       ('omega_abs_err', lambda r: np.abs(r['omega_mean'] - true_params(a_type)[0])),
                       ('c_abs_err', lambda r: np.abs(r['c_mean'] - true_params(a_type)[1]))):
        m, lo, hi, nn = boot_paired(fx(cp), fx(ca), BOOT_SEED)
        contrasts.append({'generator': gen, 'contrast': 'acts+private - acts', 'metric': metric, 'delta': m, 'mc_lo': lo, 'mc_hi': hi, 'n': nn, 'n_boot': N_BOOT})

dfa = pd.DataFrame(rows_a); dfa.to_csv(OUT / 'E2a_recovery.csv', index=False)
pd.DataFrame(contrasts).to_csv(OUT / 'E2a_channel_contrasts.csv', index=False)
log(f'E2a done {time.time() - t0:.0f}s')

# sensitivity: observer with wrong beta (0 instead of 1.5), channel 'acts', generators attenuated / costly vs honest
rows_s = []
for a_type in ('attenuated', 'costly'):
    gen = gen_name(a_type, 'honest'); d = dialogs[gen]
    sens = {}
    for beta_obs in (0.0, BETA):
        if beta_obs == BETA: res = per_world[(gen, 'acts')]
        else: res = cached(f'E2a_sens_{gen}_acts_beta{beta_obs}', lambda: ob.recover(d, 0, 'acts', beta_obs))
        sens[beta_obs] = res
        rows_s.append(dict(generator=gen, A_type=a_type, channel='acts', beta_true=BETA, beta_obs=beta_obs, **summarise(res, a_type)))
        nextact += [dict(r, beta_obs=beta_obs) for r in nextact_rows(res, a_type, gen, f'acts(beta_obs={beta_obs})')]
    # paired contrast beta_obs=0 minus beta_obs=1.5
    r0, r1 = sens[0.0], sens[BETA]
    m, lo, hi, nn = boot_paired(per_world_correct(r0, a_type), per_world_correct(r1, a_type), BOOT_SEED + 1)
    rows_s[-1].update({'delta_correct_beta0_minus_true': m, 'delta_lo': lo, 'delta_hi': hi})
pd.DataFrame(rows_s).to_csv(OUT / 'E2a_sensitivity.csv', index=False)
pd.DataFrame(nextact).to_csv(OUT / 'E2a_nextact.csv', index=False)
log(f'sensitivity done {time.time() - t0:.0f}s')

# ----------------------------------------------------------------------------- E2b
NOISES = {'none': None, 'engine': rd.load_noise('engine'), 'baseline': rd.load_noise('baseline')}
GS = [0.0, 0.25, 0.5, 1.0]
rows_b = []; contrasts_b = []
for gi, a_type in enumerate(('attenuated', 'costly')):
    gen = gen_name(a_type, 'honest'); d = dialogs[gen]
    ref_acts = per_world_correct(per_world[(gen, 'acts')], a_type); ref_priv = per_world_correct(per_world[(gen, 'acts+private')], a_type)
    acts_cf = float(ref_acts.mean()); priv_cf = float(ref_priv.mean())
    for ni, (nname, noise) in enumerate(NOISES.items()):
        for gj, g in enumerate(GS):
            label_seed = SEED + 1000 * (gi + 1) + 100 * ni + gj
            z = rd.emit_labels(d, g, noise, seed=label_seed)
            res = cached(f'E2b_{gen}_g{g}_{nname}', lambda: ob.recover(d, 0, 'labels', BETA, labels=z, g=g, noise=noise))
            row = dict(generator=gen, A_type=a_type, channel='labels', g=g, noise=nname, label_seed=label_seed, **summarise(res, a_type))
            row['acts_correct_family'] = acts_cf; row['acts_private_correct_family'] = priv_cf
            row['gain_over_acts'] = row['correct_family'] - acts_cf
            dec_A = ob.decode_labels(z[:, :, 0], rd.E0 if noise is None else rd.E0 @ noise)
            row['decode_acc_A'] = float((dec_A == d['acts'][:, :, 0]).mean())
            rows_b.append(row)
            cw = per_world_correct(res, a_type)
            for lab, ref in (('labels - acts', ref_acts), ('labels - acts+private', ref_priv)):
                m, lo, hi, nn = boot_paired(cw, ref, BOOT_SEED + 10 * ni + gj)
                contrasts_b.append({'generator': gen, 'g': g, 'noise': nname, 'contrast': lab, 'metric': 'correct_family', 'delta': m, 'mc_lo': lo, 'mc_hi': hi, 'n': nn, 'n_boot': N_BOOT})
dfb = pd.DataFrame(rows_b); dfb.to_csv(OUT / 'E2b_labels.csv', index=False)
pd.DataFrame(contrasts_b).to_csv(OUT / 'E2b_contrasts.csv', index=False)
log(f'E2b done {time.time() - t0:.0f}s')
(OUT / 'E2_run.log').write_text('\n'.join(LOG))
json.dump({'seed': SEED, 'n_worlds': NW, 'beta': BETA, 'hyps': HYPS, 'n_boot': N_BOOT, 'boot_seed': BOOT_SEED, 'prior_means': [PRIOR_OMEGA, PRIOR_C],
           'observer_prior_means': [OBS_PRIOR_OMEGA, OBS_PRIOR_C], 'elapsed_s': round(time.time() - t0, 1), 'numpy': np.__version__},
          open(OUT / 'E2_manifest.json', 'w'), indent=1)
