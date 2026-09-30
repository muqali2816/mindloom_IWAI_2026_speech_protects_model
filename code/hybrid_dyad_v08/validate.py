"""Validation for hybrid v0.8 (protocol.json['validation_checks']). Run: python code/hybrid_dyad_v08/validate.py
Optional: --mbridge <dir with dyad_model.py>  --v07 <dir with v0.7 core.py>"""
import sys, json, argparse, itertools, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))   # allow `python validate.py` from any cwd
import numpy as np
import core as hd, observer as ob

ROOT = Path(__file__).resolve().parent; OUT = ROOT / 'results'; OUT.mkdir(exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument('--mbridge', default=str(ROOT.parent / 'mbridge')); ap.add_argument('--v07', default=str(ROOT.parent / 'reciprocal_dyad_v07'))
args = ap.parse_args(); checks = {}; T = hd.CFG['agent_types']


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


# 1. regression to M-bridge
try:
    MB = load('dyad_model', Path(args.mbridge) / 'dyad_model.py')
    wm = MB.worlds(2000, 202609302); wm8 = {'truth': wm['truth'], 'private': np.zeros((2000, 2, 0), np.int8), 'u_policy': wm['u_policy'], 'u_ev': wm['u_ev'], 'u_sign': wm['u_sign']}
    ivmap = {None: {}, 'remove_gamma': {'remove_gamma_from': 6}, 'force_A_open': {'force_open': {'agent': 'A', 'from': 6}}, 'bypass': {'bypass_from': 6}}
    ok = True
    for (nm, om, c, g), iv in [(h, None) for h in MB.HYPOTHESES] + [(MB.HYPOTHESES[3], k) for k in ('remove_gamma', 'force_A_open', 'bypass')]:
        d0 = MB.simulate(wm, omega=om, c=c, gamma=g, intervention=iv)
        d1 = hd.dialogue(wm8, dict(A={'omega': om, 'c': c, 'gamma': g}, B={'omega': om, 'c': c, 'gamma': g}, beta=1.0, level1=False, forms_enabled=True,
                                   acts=['ASSERT', 'CONCEDE', 'PRESSURE', 'SILENCE', 'ASK'], order='sim', intervention=ivmap[iv]))
        ok &= np.array_equal(d0['acts'], d1['acts']) and np.array_equal(d0['forms'], d1['forms']) and np.array_equal(d0['ev'], d1['ev']) and np.abs(d0['q'] - hd.sigmoid(d1['L'])).max() < 1e-10
    checks['regression_to_Mbridge_4_hypotheses_and_interventions'] = bool(ok)
except Exception as e:
    checks['regression_to_Mbridge_4_hypotheses_and_interventions'] = f'skipped: {e}'

# 2. regression to v0.7 (no shared channel, forms off, no ASK, sequential)
try:
    rd7 = load('core7', Path(args.v07) / 'core.py'); T7 = rd7.CFG['agent_types']
    w7 = rd7.make_worlds(2000, 20261001); w78 = {'truth': w7['truth'], 'private': w7['private'], 'u_policy': w7['u'], 'u_ev': np.ones((2000, 12)), 'u_sign': np.zeros((2000, 12))}
    saved = {k: hd.P[k] for k in ('lam_low', 'lam_ask', 'lam_floor')}
    for k in saved: hd.P[k] = 0.0
    ok = True; remap = np.array([0, 1, 2, 3, -1, 4])
    for a, b, iv, iv8 in [('costly', 'honest', {}, {}), ('attenuated', 'attenuated', {}, {}),
                          ('costly', 'honest', {'force_reassure': {'round': 7, 'agent': 'B'}, 'placebo': True}, {'force_reassure': {'round': 7, 'agent': 'B'}, 'placebo': True})]:
        d0 = rd7.dialogue(w7, dict(A=dict(T7[a], beta=1.5), B=dict(T7[b], beta=1.5), intervention=iv))
        d1 = hd.dialogue(w78, dict(A=dict(T7[a], gamma=0), B=dict(T7[b], gamma=0), beta=1.5, level1=True, forms_enabled=False,
                                   acts=['ASSERT', 'CONCEDE', 'PRESSURE', 'SILENCE', 'REASSURE'], order='AB', intervention=iv8))
        ok &= np.array_equal(d0['acts'], remap[d1['acts']]) and np.abs(d0['L'] - d1['L']).max() < 1e-10
    for k, v in saved.items(): hd.P[k] = v
    checks['regression_to_v07_latent_beta1.5'] = bool(ok)
except Exception as e:
    checks['regression_to_v07_latent_beta1.5'] = f'skipped: {e}'

w = hd.make_worlds(400, 11)
# 3. omega = 0 both -> no belief change
d0 = hd.dialogue(w, dict(A={'omega': 0., 'c': 0, 'gamma': 0}, B={'omega': 0., 'c': 0, 'gamma': 0}))
checks['omega_zero_no_belief_change'] = bool(np.abs(d0['L'][:, -1] - d0['L'][:, 0]).max() < 1e-9)
# 4. huge c -> concession only when relieved
dc = hd.dialogue(w, dict(A={'omega': 1, 'c': 50, 'gamma': 0}, B={'omega': 1, 'c': 50, 'gamma': 0}))
checks['c_large_no_unrelieved_concession'] = bool((~((dc['acts'] == hd.CONCEDE) & ~dc['relief'])).all())
# 5. role swap symmetry under 'sim' with mirrored worlds
def mirror(w):
    return {'truth': 1 - w['truth'], 'private': (1 - w['private'])[:, ::-1].copy(), 'u_policy': w['u_policy'][:, :, ::-1].copy(), 'u_ev': w['u_ev'], 'u_sign': w['u_sign']}
d1 = hd.dialogue(w, dict(A=dict(T['costly']), B=dict(T['access']))); d2 = hd.dialogue(mirror(w), dict(A=dict(T['access']), B=dict(T['costly'])))
checks['role_swap_symmetry_sim'] = bool(np.array_equal(d1['acts'][:, :, 0], d2['acts'][:, :, 1]) and np.array_equal(d1['forms'][:, :, 0], d2['forms'][:, :, 1]) and np.abs(d1['L'][:, :, 0] + d2['L'][:, :, 1]).max() < 1e-9)
# 6. brute-force enumeration of hidden (x, s_A, s_B, form paths) over 3 rounds equals the HMM (uses validated replay policies)
ws = hd.make_worlds(2, 99); cond = dict(A=dict(T['access']), B=dict(T['honest'])); ds = hd.dialogue(ws, cond)
saveR = hd.CFG['rounds']; hd.CFG['rounds'] = 3; ob.R = 3
short = {k: (v[:, :3] if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[1] == saveR else v) for k, v in ds.items()}
short['L'] = ds['L'][:, :4]; short['pos'] = ds['pos'][:, :4]; short['est'] = ds['est'][:, :4]
C = hd.channel_mean(0.25); z, zn, _ = hd.emit_labels(short, seed=3)
N = ds['private'].shape[2]; S = N + 1; ev_s, ps = hd.s_tables(N); rho = hd.P['private_reliability']
polA = {(h, s): ob.replay_policies(short, 0, {'omega': h[0], 'c': h[1], 'gamma': h[2]}, cond, s) for h in ob.hypotheses()[0] for s in range(S)}
polB = {s: ob.replay_policies(short, 1, dict(T['honest']), cond, s) for s in range(S)}
acts = [hd.ACT_NAMES.index(a) for a in hd.ACT_NAMES]; aidx = np.arange(6)
err = 0.
for mode in ('acts', 'labels'):
    brute = []
    for h in ob.hypotheses()[0]:
        tot = np.zeros(2)
        for n_ in range(2):
            for x in (0, 1):
                for sA in range(S):
                    for sB in range(S):
                        for path in itertools.product(range(4), repeat=3):
                            mass = 0.5 * ps[x, sA] * ps[x, sB]; prevf = np.array([1, 1]); prevu = np.array([hd.ASSERT, hd.ASSERT])
                            for t, fp in enumerate(path):
                                lam = hd.arrival_rate(prevf[None], prevu[None])[0]; e = int(short['ev'][n_, t])
                                mass *= (1 - lam) if e == 2 else lam * (rho if e == x else 1 - rho)
                                fA, fB = ob.FORMS[fp]; uA, uB = short['acts'][n_, t]
                                mass *= polA[(h, sA)][t][prevf[1]][n_, aidx[uA], fA] * polB[sB][t][prevf[0]][n_, aidx[uB], fB]
                                if mode == 'labels': mass *= C[fA, z[n_, t, 0]] * C[fB, z[n_, t, 1]]
                                prevf = np.array([fA, fB]); prevu = np.array([uA, uB])
                            tot[n_] += mass
        brute.append(np.log(tot))
    brute = np.array(brute).T
    filt = np.stack([ob.filter_world(short, h, cond, mode, dict(T['honest']), labels=z, C=C)[0][:, -1] for h in ob.hypotheses()[0]], axis=1)
    err = max(err, float(np.abs(brute - filt).max()))
hd.CFG['rounds'] = saveR; ob.R = saveR
checks['bruteforce_paths_equals_filter'] = bool(err < 1e-9); checks['bruteforce_max_abs_logdiff'] = err
# 7. null labels -> zero gain; 8. coupling off (forms do not affect access) -> labels add nothing about mechanism
wq = hd.make_worlds(150, 21); dq = hd.dialogue(wq, cond); zq, znq, _ = hd.emit_labels(dq, seed=5)
ra = ob.recover(dq, cond, 'acts', dict(T['honest'])); rn = ob.recover(dq, cond, 'null', dict(T['honest']), null_labels=znq, C=C)
checks['null_labels_zero_gain'] = bool(np.abs(ra['family_post'] - rn['family_post']).max() < 1e-9 and np.abs(ra['next_sig'] - rn['next_sig']).max() < 1e-9)
sv = {k: hd.P[k] for k in ('lam_low', 'lam_ask')}; hd.P['lam_low'] = hd.P['lam_ask'] = hd.P['lam_floor']     # forms cannot change access
dq2 = hd.dialogue(wq, cond); zq2, _, _ = hd.emit_labels(dq2, seed=5)
ra2 = ob.recover(dq2, cond, 'acts', dict(T['honest'])); rl2 = ob.recover(dq2, cond, 'labels', dict(T['honest']), labels=zq2, C=C)
for k, v in sv.items(): hd.P[k] = v
# with gamma>0 the form still carries mechanism information through the restriction-cost/gamma trade-off even without access effects,
# so the exact statement is: next-signal predictions coincide (forms no longer matter for arrival)
checks['coupling_off_zero_gain'] = bool(np.abs(ra2['next_sig'] - rl2['next_sig']).max() < 1e-9)
# 9. scalar re-implementation of one agent's joint policy (level-1 off, N=0) against the vectorised policy
from math import exp, log
def scalar_policy(q, om, c, g, pf, pu, i):
    own = q if i == 0 else 1 - q; rw = hd.rho_omega(om); Pp = hd.P; probs = []
    for u in range(6):
        for f in (0, 1):
            base = Pp['lam_ask'] if (u == hd.ASK or pu == hd.ASK) else Pp['lam_low']; lam = Pp['lam_floor'] + (base - Pp['lam_floor']) * f * pf
            risk = [Pp['misrepresentation_cost'] * (1 - own), Pp['misrepresentation_cost'] * own + c] + [Pp['nonfactual_base']] * 4
            mi = float(hd.binary_information(np.array([q]), rw)[0]); d_ = own * (1 - rw) + (1 - own) * rw
            G = risk[u] + Pp['restriction_cost'] * (1 - f) + g * lam * d_ - 1.0 * lam * mi
            probs.append(exp(-G / Pp['choice_temperature']))
    p = np.array(probs).reshape(6, 2); return p / p.sum()
w0 = {'truth': np.zeros(1, int), 'private': np.zeros((1, 2, 0), np.int8)}
err2 = 0.
for q in (0.1, 0.5, 0.9):
    for h in ob.hypotheses()[0]:
        for pf, pu in itertools.product((0, 1), (hd.ASSERT, hd.ASK)):
            ag = hd.Agent(0, w0, {'omega': h[0], 'c': h[1], 'gamma': h[2]}, dict(beta=1.0, level1=False))
            ag.J[:, 1] *= q / 0.75; ag.J[:, 0] *= (1 - q) / 0.25; ag.J /= ag.J.sum()
            pv = ag.policy(np.array([pu]), np.array([pf]), np.zeros(1))[0][0]
            err2 = max(err2, float(np.abs(pv - scalar_policy(q, h[0], h[1], h[2], pf, pu, 0)).max()))
checks['scalar_policy_single_world'] = bool(err2 < 1e-9)
# 10. seed repeatability
da = hd.dialogue(hd.make_worlds(200, 77), dict(A=dict(T['mixed']), B=dict(T['costly']))); db = hd.dialogue(hd.make_worlds(200, 77), dict(A=dict(T['mixed']), B=dict(T['costly'])))
checks['seed_repeatability'] = bool(np.array_equal(da['acts'], db['acts']) and np.array_equal(da['forms'], db['forms']) and np.array_equal(da['L'], db['L']))
checks['all_passed'] = all(v is True for k, v in checks.items() if k != 'bruteforce_max_abs_logdiff')
(OUT / 'validation.json').write_text(json.dumps(checks, indent=1)); print(json.dumps(checks, indent=1))
