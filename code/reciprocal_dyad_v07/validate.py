"""Validation checks listed in protocol.json['validation_checks']. Writes results/validation.json.
Run from the package directory: PYTHONPATH=. python validate.py  [--v06 path/to/defensive_dialogue_v06]"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # package root on path
import json, sys, argparse
from pathlib import Path
from math import comb, exp, log
import numpy as np
import core as rd

ROOT = Path(__file__).resolve().parent; OUT = ROOT / 'results'; OUT.mkdir(exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument('--v06', default=str(ROOT.parent / 'dd' / 'defensive_dialogue_v06')); args = ap.parse_args()
V06 = Path(args.v06); checks = {}
T = rd.CFG['agent_types']

# 1-2. regression to v0.6 (naive B, omega=1, cost grid {0,1.6}, omega grid {1}, 4-act partner model)
w6 = rd.make_worlds(2000, 2026092721)
base = dict(cost_grid=[0.0, 1.6], omega_grid=[1.0], n_acts_partner_model=4, B_level=0)
def reg(name, A, **extra):
    d = rd.dialogue(w6, dict(base, A=A, B={'c': 1.6, 'n_acts': 4}, **extra)); b = np.load(V06 / 'results' / f'{name}.npz')
    return bool(np.array_equal(d['acts'], b['acts']) and np.abs(d['L'] - b['L']).max() < 1e-10)
try:
    checks['regression_to_v06_beta0_4acts'] = reg('REG_informed_4acts', {'omega': 1, 'c': 0, 'beta': 0, 'n_acts': 4}, prior_c=[0, 1]) and \
                                              reg('REG_honest_4acts', {'omega': 1, 'c': 0, 'beta': 0, 'n_acts': 4}, prior_c=[1, 0])
    checks['regression_to_v06_latent_beta1.5'] = reg('C1.6__latent__beta1.5', {'omega': 1, 'c': 0, 'beta': 1.5, 'n_acts': 5}, prior_c=[.5, .5]) and \
                                                 reg('C1.6__latent__beta1.5__placebo', {'omega': 1, 'c': 0, 'beta': 1.5, 'n_acts': 5}, prior_c=[.5, .5], intervention={'placebo': True})
except FileNotFoundError as e:
    checks['regression_to_v06_beta0_4acts'] = checks['regression_to_v06_latent_beta1.5'] = f'skipped: {e}'

# 3. omega = 0 for both: no belief change
w = rd.make_worlds(500, 11)
d0 = rd.dialogue(w, dict(A={'omega': 0., 'c': 0, 'beta': 1.5}, B={'omega': 0., 'c': 0, 'beta': 1.5}))
checks['omega_zero_no_belief_change'] = bool(np.abs(d0['L'][:, -1] - d0['L'][:, 0]).max() < 1e-9)

# 4. very large c: no concession
dc = rd.dialogue(w, dict(A={'omega': 1, 'c': 50, 'beta': 0}, B={'omega': 1, 'c': 50, 'beta': 0}))
checks['c_large_no_concession'] = bool((~((dc['acts'] == rd.CONCEDE) & ~dc['relief'])).all())   # concession only when the partner relieved the cost

# 5. role-swap symmetry: mirror worlds (truth flipped, private records flipped and swapped) -> mirrored statistics
def mirror(w):
    return {'truth': 1 - w['truth'], 'private': (1 - w['private'])[:, ::-1].copy(), 'u': w['u'][:, :, ::-1].copy()}
d1 = rd.dialogue(w, dict(A=dict(T['costly'], beta=1.5), B=dict(T['attenuated'], beta=1.5)))
d2 = rd.dialogue(mirror(w), dict(A=dict(T['attenuated'], beta=1.5), B=dict(T['costly'], beta=1.5), order='BA'))
checks['role_swap_symmetry'] = bool(np.array_equal(d1['acts'][:, :, 0], d2['acts'][:, :, 1]) and np.abs(d1['L'][:, :, 0] + d2['L'][:, :, 1]).max() < 1e-9)

# 6. labels at g=0 without noise are a deterministic-noise function of acts only: emission matrix independent of mechanism
E_h = rd.emission_matrix(1.0, 0.0, 0.0); E_c = rd.emission_matrix(1.0, 1.6, 0.0); E_a = rd.emission_matrix(0.2, 0.0, 0.0)
checks['labels_g0_noiseless_equal_acts_recovery'] = bool(np.allclose(E_h, E_c) and np.allclose(E_h, E_a) and np.allclose(rd.E0.sum(1), 1))

# 7-8. policy normalisation, IG non-negative
d = rd.dialogue(w, dict(A=dict(T['costly'], beta=1.5), B=dict(T['honest'], beta=1.5)))
A = rd.Level1(0, w, dict(T['costly'], beta=1.5), rd.P['cost_grid'], rd.P['omega_grid'], 5)
prob, ig = A.act_probs()
checks['policy_normalisation'] = bool(np.abs(prob.sum(1) - 1).max() < 1e-9 and (prob >= 0).all())
checks['IG_nonnegative'] = bool(np.nanmin(d['ig']) > -1e-9)

# 9. scalar re-implementation of one level-1 agent's first-round update and IG on a single world (beta=1.5, honest x honest)
def scalar_check(w, i=0):
    N, rho, lam, k, tau = rd.N, rd.RHO, rd.LAM, rd.K, rd.TAU; lr = log(lam / (1 - lam)); lp = log(rho / (1 - rho)); pr = log(0.75 / 0.25)
    sig = lambda x: 1 / (1 + exp(-x)); cg = rd.P['cost_grid']; wg = rd.P['omega_grid']
    priv = w['private'][i]
    LA = pr + (2 * priv[0].sum() - N) * lp
    pA = sig(LA)
    J = {(x, s, c, o): (pA if x else 1 - pA) * comb(N, s) * ((rho if x else 1 - rho) ** s) * ((1 - rho if x else rho) ** (N - s)) / (len(cg) * len(wg))
         for x in (0, 1) for s in range(N + 1) for c in range(len(cg)) for o in range(len(wg))}
    def ent(D):
        Z = sum(D.values()); return -sum((v / Z) * log(v / Z) for v in D.values() if v > 0)
    def probsB(s, c, o, M, relief):
        q1 = sig(-pr + (2 * s - N) * lp + wg[o] * M); q_own = 1 - q1
        L = [k * (1 - q_own), k * q_own + (0. if relief else cg[c]), 1.1, 1.1, 1.1]
        z = [exp(-(l - min(L)) / tau) for l in L]; return [v / sum(z) for v in z]
    qA = sig(LA); losses = [k * (1 - qA), k * qA, 1.1, 1.1, 1.1]; H0 = ent(J); igs = []
    for a in range(5):
        M = lr if a == 0 else -lr if a == 1 else 0.; expH = 0.
        for b in range(5):
            Jb = {key: v * probsB(key[1], key[2], key[3], M, a == 4)[b] for key, v in J.items()}
            pb = sum(Jb.values())
            if pb > 0: expH += pb * ent(Jb)
        igs.append(H0 - expH)
    score = [(-l + 1.5 * g) / tau for l, g in zip(losses, igs)]; mx = max(score); z = [exp(s - mx) for s in score]
    return [v / sum(z) for v in z], igs
ws = rd.make_worlds(3, 5)
Av = rd.Level1(0, ws, dict(T['honest'], beta=1.5), rd.P['cost_grid'], rd.P['omega_grid'], 5)
pv, igv = Av.act_probs()
ok = True
for i in range(3):
    ps, igs = scalar_check(ws, i)
    ok &= np.abs(np.array(ps) - pv[i]).max() < 1e-9 and np.abs(np.array(igs) - igv[i]).max() < 1e-9
checks['scalar_reimplementation_single_world'] = bool(ok)

# 10. seed repeatability
da = rd.dialogue(rd.make_worlds(300, 77), dict(A=dict(T['mixed'], beta=1.5), B=dict(T['costly'], beta=1.5)))
db = rd.dialogue(rd.make_worlds(300, 77), dict(A=dict(T['mixed'], beta=1.5), B=dict(T['costly'], beta=1.5)))
checks['seed_repeatability'] = bool(np.array_equal(da['acts'], db['acts']) and np.array_equal(da['L'], db['L']))

checks['all_passed'] = all(v is True for v in checks.values())
(OUT / 'validation.json').write_text(json.dumps(checks, indent=1))
print(json.dumps(checks, indent=1))
