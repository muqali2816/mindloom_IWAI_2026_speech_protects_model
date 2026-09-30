"""Hybrid dyad v0.8 — act x form policies, joint verification channel (M-bridge) and level-1 inference
about the partner from the partner's acts (v0.7). See MODEL_SPEC.md. Simulation of stipulated mechanisms only."""
from pathlib import Path
import json
from math import comb
import numpy as np

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'protocol.json').read_text())
P = dict(CFG['common_parameters'])
ACT_NAMES = ['ASSERT', 'CONCEDE', 'PRESSURE', 'SILENCE', 'ASK', 'REASSURE']
ASSERT, CONCEDE, PRESSURE, SILENCE, ASK, REASSURE = range(6)
PREF = np.array([1, 0])
LABELS = ['event.SHIFT', 'function.SEAL', 'insufficient_context', 'regime.BUILD', 'regime.DRAIN', 'regime.EDGE',
          'regime.FLOOD', 'regime.LOCK', 'regime.SEEK', 'regime.UNSEAL', 'regime.VOID']   # M-bridge output order


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12); return np.log(p) - np.log1p(-p)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(np.asarray(x, float), -60, 60)))


def llr(lam):
    lam = np.asarray(lam, float); return np.log(lam) - np.log1p(-lam)


def rho_omega(omega, rho=None):
    rho = P['private_reliability'] if rho is None else rho
    return rho ** omega / (rho ** omega + (1 - rho) ** omega)


def binary_information(q, rw):
    """Mutual information (nats) between x and one binary signal of reliability rw, belief q = P(x=1)."""
    q = np.asarray(q, float); p1 = q * rw + (1 - q) * (1 - rw)
    ent = lambda x: -x * np.log(np.clip(x, 1e-300, None)) - (1 - x) * np.log(np.clip(1 - x, 1e-300, None))
    return ent(p1) - ent(np.asarray(rw, float))


def make_worlds(n, seed, N=None):
    """truth (balanced), private records (n,2,N), and independent uniform streams for policies, arrival, sign, labels."""
    N = P['n_private'] if N is None else N; R = CFG['rounds']
    ss = np.random.SeedSequence(seed).spawn(6)
    truth = np.arange(n) % 2; np.random.default_rng(ss[0]).shuffle(truth)
    eu = np.random.default_rng(ss[1]).random((n, 2, max(N, 1)))[:, :, :N]
    private = np.where(eu < P['private_reliability'], truth[:, None, None], 1 - truth[:, None, None]).astype(np.int8)
    return {'truth': truth, 'private': private, 'u_policy': np.random.default_rng(ss[2]).random((n, R, 2)),
            'u_ev': np.random.default_rng(ss[3]).random((n, R)), 'u_sign': np.random.default_rng(ss[4]).random((n, R)),
            'u_label': np.random.default_rng(ss[5]).random((n, R, 2))}


PRIOR_L = np.array([logit(P['self_prior']), -logit(P['self_prior'])])


def initial_logodds(w):
    ev = (2 * w['private'].astype(float) - 1).sum(axis=2) * llr(P['private_reliability']) if w['private'].shape[2] else np.zeros((len(w['truth']), 2))
    return np.broadcast_to(PRIOR_L, (len(w['truth']), 2)) + ev


def oracle_logodds(w):
    return (2 * w['private'].astype(float) - 1).sum(axis=(1, 2)) * llr(P['private_reliability']) if w['private'].shape[2] else np.zeros(len(w['truth']))


def s_tables(N):
    rho = P['private_reliability']; S = np.arange(N + 1)
    ev_s = (2 * S - N) * llr(rho)
    ps = np.stack([np.array([comb(N, s) * (1 - rho) ** s * rho ** (N - s) for s in S]),
                   np.array([comb(N, s) * rho ** s * (1 - rho) ** (N - s) for s in S])])
    return ev_s, ps


def act_losses(q_own, c_eff, acts):
    """(..., len(acts)) losses for the enabled act menu. q_own = P(own preferred version)."""
    q_own, c_eff = np.broadcast_arrays(np.asarray(q_own, float), np.asarray(c_eff, float))
    b = P['nonfactual_base']
    table = {ASSERT: P['misrepresentation_cost'] * (1 - q_own), CONCEDE: P['misrepresentation_cost'] * q_own + c_eff}
    return np.stack([table.get(a, np.full_like(q_own, b)) for a in acts], axis=-1)


def softmax_last(score, axes):
    score = score - score.max(axis=axes, keepdims=True); p = np.exp(score)
    return p / p.sum(axis=axes, keepdims=True)


def draw(prob_flat, u):
    return np.minimum((prob_flat.cumsum(axis=-1) < np.asarray(u)[..., None]).sum(axis=-1), prob_flat.shape[-1] - 1)


def entropy(J):
    Jf = J.reshape(J.shape[0], -1); return -(Jf * np.log(np.clip(Jf, 1e-300, None))).sum(axis=1)


def shrink_x(J, J_exact, omega):
    """Replace the x-marginal of the exact posterior by the omega-shrunk one (v0.7 rule)."""
    if omega >= 1.0: return J_exact
    p_old = J[:, 1].sum(axis=(1, 2, 3)); p_upd = J_exact[:, 1].sum(axis=(1, 2, 3))
    p_new = sigmoid(logit(p_old) + omega * (logit(p_upd) - logit(p_old)))
    scale = np.stack([(1 - p_new) / np.clip(1 - p_upd, 1e-300, None), p_new / np.clip(p_upd, 1e-300, None)], axis=1)
    Jn = J_exact * scale[:, :, None, None, None]
    return Jn / Jn.sum(axis=(1, 2, 3, 4), keepdims=True)


def arrival_rate(prev_forms, prev_acts, external=None):
    """lambda_t from previous forms (n,2) and previous acts (n,2)."""
    if external is not None: return np.full(len(prev_forms), float(external))
    base = np.where((prev_acts == ASK).any(axis=1), P['lam_ask'], P['lam_low'])
    return P['lam_floor'] + (base - P['lam_floor']) * prev_forms.prod(axis=1)


class Agent:
    def __init__(self, i, w, spec, cond):
        n = len(w['truth']); self.i = i; self.j = 1 - i; self.n = n
        self.omega = float(spec['omega']); self.c = float(spec['c']); self.gamma = float(spec.get('gamma', 0.))
        self.beta = float(spec.get('beta', cond.get('beta', P['beta_active'])))
        self.acts = [ACT_NAMES.index(a) if isinstance(a, str) else int(a) for a in cond.get('acts', ACT_NAMES)]
        self.forms = [0, 1] if cond.get('forms_enabled', True) else [1]
        self.act_index = np.full(6, -1); self.act_index[self.acts] = np.arange(len(self.acts))
        self.level1 = bool(cond.get('level1', True))
        self.N = w['private'].shape[2]; self.ev_s, self.ps = s_tables(self.N)
        self.cg = np.asarray(cond.get('cost_grid', P['cost_grid']), float); self.wg = np.asarray(cond.get('omega_grid', P['omega_grid']), float)
        pc = np.asarray(cond.get('prior_c', np.full(len(self.cg), 1 / len(self.cg))), float)
        pw = np.asarray(cond.get('prior_w', np.full(len(self.wg), 1 / len(self.wg))), float)
        px1 = sigmoid(initial_logodds(w)[:, i])
        base = np.stack([(1 - px1)[:, None] * self.ps[0][None, :], px1[:, None] * self.ps[1][None, :]], axis=1)
        self.J = base[:, :, :, None, None] * pc[None, None, None, :, None] * pw[None, None, None, None, :]
        self.M = np.zeros(n); self.pos = np.full(n, PREF[i], np.int8)
        self.relieved = np.zeros(n, bool); self.relieved_believed = np.zeros(n, bool); self.relief_partner = np.zeros(n, bool)
        self.c_active = np.full(n, self.c); self.gamma_active = np.full(n, self.gamma)

    # beliefs
    def logodds(self): return logit(self.J[:, 1].sum(axis=(1, 2, 3)))
    def q1(self): return sigmoid(self.logodds())
    def q_own(self): q = self.q1(); return q if PREF[self.i] == 1 else 1 - q
    def partner_marginals(self):
        m = self.J.sum(axis=(1, 2)); return m.sum(axis=2), m.sum(axis=1)
    def partner_belief_estimate(self, m_shared):
        q1 = sigmoid(PRIOR_L[self.j] + self.ev_s[None, :, None] + self.wg[None, None, :] * (self.M + m_shared * llr(P['private_reliability']))[:, None, None])
        return (self.J.sum(axis=(1, 3)) * q1).sum(axis=(1, 2))

    # shared signal
    def observe_signal(self, e, got, rho=None):
        rho = P['private_reliability'] if rho is None else rho
        like = np.where(got[:, None], np.where(e[:, None] == np.array([0, 1])[None, :], rho, 1 - rho), 1.)
        Je = self.J * like[:, :, None, None, None]; Je /= Je.sum(axis=(1, 2, 3, 4), keepdims=True)
        self.J = shrink_x(self.J, Je, self.omega)

    # partner model (level-0 over acts, form-free)
    def partner_act_probs(self, M, relief, m_shared):
        q1 = sigmoid(PRIOR_L[self.j] + self.ev_s[None, :, None] + self.wg[None, None, :] * (M + m_shared * llr(P['private_reliability']))[:, None, None])
        q_own_j = q1 if PREF[self.j] == 1 else 1 - q1
        c_eff = np.where(relief[:, None], 0., self.cg[None, :])
        L = act_losses(q_own_j[:, :, None, :], c_eff[:, None, :, None], self.acts)      # (n,S,nC,nW,a)
        return softmax_last(-L / P['choice_temperature'], (-1,))

    def observe_partner_act(self, a_j, relief_j, m_shared):
        if not self.level1: return
        probs = self.partner_act_probs(self.M, relief_j, m_shared)
        idx = self.act_index[np.asarray(a_j, int)]
        like = np.take_along_axis(probs, idx[:, None, None, None, None].repeat(self.N + 1, 1).repeat(len(self.cg), 2).repeat(len(self.wg), 3), axis=4)[..., 0]
        Je = self.J * like[:, None]; Je /= Je.sum(axis=(1, 2, 3, 4), keepdims=True)
        self.J = shrink_x(self.J, Je, self.omega)

    def info_gain_partner(self, m_shared):
        n = self.n; ig = np.zeros((n, len(self.acts)))
        if not self.level1 or self.beta <= 0: return ig
        H0 = entropy(self.J); sgn = 1 if PREF[self.i] == 1 else -1
        for ai, a in enumerate(self.acts):
            M_a = self.M + (llr(P['trusted_precision']) if a == ASSERT else -llr(P['trusted_precision']) if a == CONCEDE else 0.) * sgn
            probs = self.partner_act_probs(M_a, np.full(n, a == REASSURE), m_shared)
            expH = np.zeros(n)
            for b in range(len(self.acts)):
                Jb = self.J * probs[:, None, :, :, :, b]; Zb = Jb.sum(axis=(1, 2, 3, 4))
                Jb = Jb / np.clip(Zb, 1e-300, None)[:, None, None, None, None]
                Jb = shrink_x(self.J, Jb, self.omega)
                expH += Zb * entropy(Jb)
            ig[:, ai] = H0 - expH
        return ig

    # joint act x form policy
    def policy(self, prev_partner_act, prev_partner_form, m_shared, external=None):
        n = self.n; nA = len(self.acts); nF = len(self.forms)
        q = self.q1(); own = q if PREF[self.i] == 1 else 1 - q
        c_eff = np.where(self.relieved, 0., self.c_active)
        L = act_losses(own, c_eff, self.acts)                                         # (n,nA)
        acts_arr = np.asarray(self.acts)
        if external is not None:
            lam = np.full((n, nA, nF), float(external))
        else:
            base = np.where((acts_arr[None, :] == ASK) | (np.asarray(prev_partner_act)[:, None] == ASK), P['lam_ask'], P['lam_low'])   # (n,nA)
            fcol = np.asarray(self.forms, float)[None, None, :]
            lam = P['lam_floor'] + (base[:, :, None] - P['lam_floor']) * fcol * np.asarray(prev_partner_form, float)[:, None, None]
        rw = rho_omega(self.omega)
        disconfirm = own * (1 - rw) + (1 - own) * rw
        ig_sig = lam * binary_information(q, rw)[:, None, None]
        ig_par = self.info_gain_partner(m_shared)[:, :, None]
        G = L[:, :, None] + P['restriction_cost'] * (1 - np.asarray(self.forms, float))[None, None, :] \
            + self.gamma_active[:, None, None] * lam * disconfirm[:, None, None] - self.beta * (ig_sig + ig_par)
        return softmax_last(-G / P['choice_temperature'], (1, 2)), ig_par[:, :, 0], lam

    def after_own_act(self, a):
        factual = (a == ASSERT) | (a == CONCEDE)
        v = np.where(a == ASSERT, PREF[self.i], 1 - PREF[self.i])
        self.M += np.where(factual, np.where(v == 1, 1., -1.) * llr(P['trusted_precision']), 0.)
        self.pos = np.where(factual, v, self.pos).astype(np.int8)
        self.relief_partner = a == REASSURE


def dialogue(w, cond):
    """cond: A, B specs {omega,c,gamma,beta}; acts (names); forms_enabled; level1; order 'sim'|'AB'|'BA';
    intervention: {'bypass_from': t, 'remove_gamma_from': t, 'remove_c_from': t, 'remove_for': ['A','B'],
                   'force_open': {'agent': 'A', 'from': t}, 'force_reassure': {'round': t, 'agent': 'B'}, 'placebo': bool,
                   'external_obs_round': t, 'external_rho': 0.95}."""
    n = len(w['truth']); R = CFG['rounds']; iv = cond.get('intervention', {}) or {}
    A = Agent(0, w, cond['A'], cond); B = Agent(1, w, cond['B'], cond); ag = [A, B]
    nA = len(A.acts); nF = len(A.forms); acts_arr = np.asarray(A.acts)
    acts = np.zeros((n, R, 2), np.int8); forms = np.ones((n, R, 2), np.int8); ev = np.full((n, R), 2, np.int8)
    Lh = np.empty((n, R + 1, 2)); posh = np.empty((n, R + 1, 2), np.int8); est = np.full((n, R + 1, 2), np.nan)
    probh = np.full((n, R, 2, nA * nF), np.nan); rates = np.zeros((n, R)); relief_h = np.zeros((n, R, 2), bool)
    prevf = np.ones((n, 2), np.int8); prevu = np.full((n, 2), ASSERT, np.int8); m_shared = np.zeros(n)
    order = cond.get('order', CFG.get('order', 'sim')); placebo = bool(iv.get('placebo', False))
    for i, a in enumerate(ag):
        Lh[:, 0, i] = a.logodds(); posh[:, 0, i] = a.pos; est[:, 0, i] = a.partner_belief_estimate(m_shared)
    for t in range(R):
        external = iv.get('external_rate', 0.9) if ('bypass_from' in iv and t >= iv['bypass_from']) else None
        rate = arrival_rate(prevf, prevu, external); rates[:, t] = rate
        got = w['u_ev'][:, t] < rate
        sign = np.where(w['u_sign'][:, t] < P['private_reliability'], w['truth'], 1 - w['truth'])
        ev[got, t] = sign[got]; m_shared += np.where(got, 2 * sign - 1, 0)
        for a in ag: a.observe_signal(sign, got)
        if 'external_obs_round' in iv and t == iv['external_obs_round']:
            rng = np.random.default_rng(4242); rho_e = iv.get('external_rho', 0.95)
            e_ext = np.where(rng.random(n) < rho_e, w['truth'], 1 - w['truth'])
            for a in ag: a.observe_signal(e_ext, np.ones(n, bool), rho_e)
        for name, a in (('A', A), ('B', B)):
            if 'remove_gamma_from' in iv and t >= iv['remove_gamma_from'] and name in iv.get('remove_for', ['A', 'B']): a.gamma_active[:] = 0.
            if 'remove_c_from' in iv and t >= iv['remove_c_from'] and name in iv.get('remove_for', ['A', 'B']): a.c_active[:] = 0.
        seq = [0, 1] if order in ('sim', 'AB') else [1, 0]
        chosen = {}
        for i in seq:
            a = ag[i]; j = 1 - i
            if order == 'sim' or i == seq[0]:
                pa, pf = prevu[:, j], prevf[:, j]
            else:
                pa, pf = chosen[j][0], chosen[j][1]      # second mover sees the first mover's current move
            prob, _, _ = a.policy(pa, pf, m_shared, external)
            flat = prob.reshape(n, nA * nF); probh[:, t, i] = flat
            k = draw(flat, w['u_policy'][:, t, i])
            u = acts_arr[k // nF]; f = np.asarray(A.forms)[k % nF]
            if 'force_reassure' in iv and t == iv['force_reassure']['round'] and ('AB'[i] == iv['force_reassure']['agent']):
                u = np.full(n, REASSURE, np.int8)
            if 'force_open' in iv and t >= iv['force_open']['from'] and 'AB'[i] == iv['force_open']['agent']: f = np.ones(n, np.int8)
            chosen[i] = (u.astype(np.int8), f.astype(np.int8))
            if order != 'sim':
                relief_h[:, t, i] = a.relieved
                a.after_own_act(u)
                other = ag[j]; other.observe_partner_act(u, a.relieved_believed, m_shared)
                other.relieved_believed = a.relief_partner.copy(); other.relieved = a.relief_partner & (not placebo)
                a.relieved = np.zeros(n, bool); a.relieved_believed = np.zeros(n, bool)
        if order == 'sim':
            for i in (0, 1):
                relief_h[:, t, i] = ag[i].relieved; ag[i].after_own_act(chosen[i][0])
            for i in (0, 1):
                a, other = ag[i], ag[1 - i]
                other.observe_partner_act(chosen[i][0], a.relieved_believed, m_shared)
            for i in (0, 1):
                a, other = ag[i], ag[1 - i]
                other.relieved_believed = a.relief_partner.copy(); other.relieved = a.relief_partner & (not placebo)
        for i in (0, 1):
            acts[:, t, i] = chosen[i][0]; forms[:, t, i] = chosen[i][1]
        prevf = forms[:, t].copy(); prevu = acts[:, t].copy()
        for i, a in enumerate(ag):
            Lh[:, t + 1, i] = a.logodds(); posh[:, t + 1, i] = a.pos; est[:, t + 1, i] = a.partner_belief_estimate(m_shared)
    out = {'truth': w['truth'], 'private': w['private'], 'L': Lh, 'pos': posh, 'acts': acts, 'forms': forms, 'ev': ev, 'rates': rates,
           'oracle': oracle_logodds(w), 'est': est, 'prob': probh, 'relief': relief_h,
           'A_omega': A.omega, 'A_c': A.c, 'A_gamma': A.gamma, 'B_omega': B.omega, 'B_c': B.c, 'B_gamma': B.gamma}
    for name, a in (('A', A), ('B', B)):
        pc, pw = a.partner_marginals(); out[f'{name}_pC_partner'] = pc; out[f'{name}_pW_partner'] = pw
    return out


# ----------------------------------------------------------------------------- label channel (forms -> LOCK/SEEK rows)

def channel_counts():
    meta = json.loads((ROOT / 'data' / 'annotation_channel.json').read_text())
    counts = np.asarray(meta['counts'], float); gold = meta['gold']
    return np.stack([counts[gold.index('regime.LOCK')], counts[gold.index('regime.SEEK')]]), meta['labels']   # row 0 restrict, row 1 invite


def channel_mean(eta=0.0, pseudo=0.5):
    counts, _ = channel_counts(); sm = (counts + pseudo) / (counts.sum(1, keepdims=True) + pseudo * counts.shape[1])
    bg = sm.mean(0, keepdims=True); return (1 - eta) * sm + eta * bg


def emit_labels(d, seed, pseudo=0.5, eta_gen=None):
    """Form labels for both agents. Rows sampled per world from Dirichlet(counts+pseudo) (eta_gen None) or fixed mean rows mixed with eta_gen."""
    n, R, _ = d['forms'].shape; counts, _ = channel_counts(); rng = np.random.default_rng(seed)
    if eta_gen is None:
        rows = np.stack([rng.dirichlet(counts[f] + pseudo, size=n) for f in (0, 1)], axis=1)   # (n,2,11)
    else:
        rows = np.broadcast_to(channel_mean(eta_gen, pseudo), (n, 2, counts.shape[1])).copy()
    z = np.zeros((n, R, 2), np.int8); znull = np.zeros((n, R, 2), np.int8)
    u = rng.random((n, R, 2)); u2 = rng.random((n, R, 2))
    for i in (0, 1):
        f = d['forms'][:, :, i].astype(int); rf = rows[np.arange(n)[:, None], f]           # (n,R,11)
        z[:, :, i] = draw(rf, u[:, :, i])
        fnull = (d['acts'][:, :, i] == ASK).astype(int); rn = rows[np.arange(n)[:, None], fnull]
        znull[:, :, i] = draw(rn, u2[:, :, i])
    return z, znull, rows


# ----------------------------------------------------------------------------- metrics

def metrics(d):
    import pandas as pd
    truth = d['truth'].astype(int); n = len(truth); tail = CFG['tail_rounds']; a = d['acts']; pos = d['pos'].astype(int)
    q1 = sigmoid(d['L']); pt = np.where(truth[:, None, None] == 1, q1, 1 - q1)
    df = pd.DataFrame({'world': np.arange(n), 'truth': truth,
                       'A_own0': (q1[:, 0, 0] > 0.5).astype(float), 'B_own0': (q1[:, 0, 1] < 0.5).astype(float),
                       'nonc_A': (a[:, -tail:, 0] != CONCEDE).mean(1), 'nonc_B': (a[:, -tail:, 1] != CONCEDE).mean(1),
                       'nonc_mutual': ((a[:, -tail:, 0] != CONCEDE) & (a[:, -tail:, 1] != CONCEDE)).mean(1),
                       'U_final': (pos[:, -tail:, 0] != pos[:, -tail:, 1]).mean(1),
                       'evidence_count': (d['ev'] != 2).sum(1).astype(float), 'mean_rate': d['rates'].mean(1),
                       'restrict_A': (d['forms'][:, :, 0] == 0).mean(1), 'restrict_B': (d['forms'][:, :, 1] == 0).mean(1),
                       'restrict_joint': (d['forms'].prod(2) == 0).mean(1),
                       'A_p_truth0': pt[:, 0, 0], 'A_p_truth': pt[:, -1, 0], 'B_p_truth0': pt[:, 0, 1], 'B_p_truth': pt[:, -1, 1],
                       'A_dbelief': np.abs(q1[:, -1, 0] - q1[:, 0, 0]), 'B_dbelief': np.abs(q1[:, -1, 1] - q1[:, 0, 1]),
                       'belief_gap_final': np.abs(q1[:, -1, 0] - q1[:, -1, 1]),
                       'A_hidden_shift': np.abs(q1[:, -1, 0] - q1[:, 0, 0]) * (pos[:, -1, 0] == pos[:, 0, 0]),
                       'B_hidden_shift': np.abs(q1[:, -1, 1] - q1[:, 0, 1]) * (pos[:, -1, 1] == pos[:, 0, 1]),
                       'A_transp_err': np.abs(d['est'][:, -1, 0] - q1[:, -1, 1]), 'B_transp_err': np.abs(d['est'][:, -1, 1] - q1[:, -1, 0])})
    for i, nm in enumerate('AB'):
        for code, act in enumerate(ACT_NAMES): df[f'{nm}_{act}'] = (a[:, :, i] == code).mean(1)
    cg = np.asarray(P['cost_grid']); wg = np.asarray(P['omega_grid'])
    for nm, pc_, pw_ in (('A', d['B_c'], d['B_omega']), ('B', d['A_c'], d['A_omega'])):
        df[f'{nm}_pC_true'] = d[f'{nm}_pC_partner'][:, int(np.argmin(np.abs(cg - pc_)))]
        df[f'{nm}_pW_true'] = d[f'{nm}_pW_partner'][:, int(np.argmin(np.abs(wg - pw_)))]
    return df


def paired(a, b, label, exclude=('world', 'truth')):
    rows = []
    for col in a.columns:
        if col in exclude: continue
        delta = (a[col] - b[col]).dropna().to_numpy()
        if len(delta) == 0: continue
        se = delta.std(ddof=1) / np.sqrt(len(delta))
        rows.append({'contrast': label, 'metric': col, 'delta': delta.mean(), 'mc_lo': delta.mean() - 1.96 * se, 'mc_hi': delta.mean() + 1.96 * se})
    return rows
