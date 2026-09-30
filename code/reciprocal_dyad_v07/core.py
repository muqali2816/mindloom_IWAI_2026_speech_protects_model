"""Reciprocal dyad v0.7 — two level-1 agents with (omega, c) mechanisms and a speech-label layer.

Lineage: v0.4 kernel (worlds, logit helpers), v0.6 latent-cost listener. See MODEL_SPEC.md.
No NLP, no clinical labels, no fitted human parameters.

Reciprocal dyad v0.7: two Part II agents facing each other. Each treats the partner's act as evidence about the
proposition and about the partner's hidden (c, omega) through a level-0 model of the partner; shared signals arrive at
a fixed rate (no access channel; that is added in hybrid_dyad_v08). Experiments E1-E5 are in scripts/. Results in
results/, report in REPORT_RU.md. Regression: with a passive partner the model reproduces the v0.6 single-agent
results bitwise (validate.py). Simulation of stipulated mechanisms only; no human data.
"""
from pathlib import Path
import json
from math import comb
import numpy as np

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'protocol.json').read_text())
P = CFG['common_parameters']
ACTS = ['assert', 'concede', 'pressure', 'silence', 'reassure']
ASSERT, CONCEDE, PRESSURE, SILENCE, REASSURE = range(5)
PREF = np.array([1, 0])                       # A prefers version 1, B prefers version 0
N = P['n_private']; RHO = P['private_reliability']; LAM = P['trusted_precision']
K = P['misrepresentation_cost']; TAU = P['choice_temperature']; B_NF = P['pressure_base']
LABELS = ['BUILD', 'SEEK', 'UNSEAL', 'LOCK', 'DRAIN', 'FLOOD', 'EDGE', 'VOID', 'SEAL', 'SHIFT', 'INSUFF']
LIDX = {l: i for i, l in enumerate(LABELS)}


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(np.asarray(x, float), -60, 60)))


def llr(lam):
    lam = np.asarray(lam, float)
    return np.log(lam) - np.log1p(-lam)


def make_worlds(n, seed):
    """Balanced truth, private records for both agents, uniform draws for both agents' acts."""
    ss = np.random.SeedSequence(seed).spawn(3)
    truth = np.arange(n) % 2
    np.random.default_rng(ss[0]).shuffle(truth)
    eu = np.random.default_rng(ss[1]).random((n, 2, N))
    private = np.where(eu < RHO, truth[:, None, None], 1 - truth[:, None, None])
    u = np.random.default_rng(ss[2]).random((n, CFG['rounds'], 2))
    return {'truth': truth, 'private': private.astype(np.int8), 'u': u}


PRIOR_L = np.array([logit(P['self_prior']), -logit(P['self_prior'])])   # log-odds for version 1


def initial_logodds(w):
    ev = (2 * w['private'].astype(float) - 1).sum(axis=2) * llr(RHO)
    return np.broadcast_to(PRIOR_L, (len(w['truth']), 2)) + ev


def oracle_logodds(w):
    return (2 * w['private'].astype(float) - 1).sum(axis=(1, 2)) * llr(RHO)


S_GRID = np.arange(N + 1)
EV_S = (2 * S_GRID - N) * llr(RHO)
PS_GIVEN_X = np.stack([np.array([comb(N, s) * (1 - RHO) ** s * RHO ** (N - s) for s in S_GRID]),
                       np.array([comb(N, s) * RHO ** s * (1 - RHO) ** (N - s) for s in S_GRID])])   # (2, S)


def losses(q_own, c_eff, n_acts):
    """(..., n_acts) losses. q_own = P(own preferred version); c_eff = effective concession cost."""
    q_own, c_eff = np.broadcast_arrays(np.asarray(q_own, float), np.asarray(c_eff, float))
    L = np.stack([K * (1 - q_own), K * q_own + c_eff, np.full_like(q_own, B_NF),
                  np.full_like(q_own, P['silence_base']), np.full_like(q_own, P['reassure_base'])], axis=-1)
    return L[..., :n_acts]


def softmax_rows(score):
    score = score - score.max(axis=-1, keepdims=True)
    p = np.exp(score)
    return p / p.sum(axis=-1, keepdims=True)


def draw(prob, u):
    return np.minimum((np.asarray(u)[..., None] > prob.cumsum(axis=-1)).sum(axis=-1), prob.shape[-1] - 1)


def entropy(J):
    Jf = J.reshape(J.shape[0], -1)
    return -(Jf * np.log(np.clip(Jf, 1e-300, None))).sum(axis=1)


ATTENUATION = 'shrink'   # 'shrink': exact joint update, then the x-marginal log-odds move only omega * exact increment
                         # 'mixture': omega*P + (1-omega)/n_acts ; 'power': P^omega renormalised (both act-likelihood tempering)


def temper(prob, omega):
    """Subjective likelihood of the partner's act under act-likelihood tempering (last axis = acts)."""
    if omega >= 1.0 or ATTENUATION == 'shrink':
        return prob
    if ATTENUATION == 'power':
        t = np.clip(prob, 1e-300, None) ** omega
        return t / t.sum(axis=-1, keepdims=True)
    return omega * prob + (1 - omega) / prob.shape[-1]


def posterior_after(J, like_x, omega):
    """J (n,2,S,nC,nW) times like (n,S,nC,nW) broadcast over x; returns normalised posterior and its mass Z.
    Under 'shrink', the x-marginal moves only omega times the exact log-odds increment."""
    Jb = J * like_x[:, None]
    Z = Jb.sum(axis=(1, 2, 3, 4))
    Jb = Jb / np.clip(Z, 1e-300, None)[:, None, None, None, None]
    if ATTENUATION == 'shrink' and omega < 1.0:
        p_old = J[:, 1].sum(axis=(1, 2, 3)); p_upd = Jb[:, 1].sum(axis=(1, 2, 3))
        p_new = sigmoid(logit(p_old) + omega * (logit(p_upd) - logit(p_old)))
        scale = np.stack([(1 - p_new) / np.clip(1 - p_upd, 1e-300, None), p_new / np.clip(p_upd, 1e-300, None)], axis=1)  # (n,2)
        Jb = Jb * scale[:, :, None, None, None]
        Jb = Jb / Jb.sum(axis=(1, 2, 3, 4), keepdims=True)
    return Jb, Z


class Level1:
    """Agent i's joint posterior over (x, s_j, c_j, omega_j) and its policy."""

    def __init__(self, i, w, spec, cost_grid, omega_grid, n_acts_partner, prior_c=None, prior_w=None):
        n = len(w['truth']); self.i = i; self.j = 1 - i
        self.omega = float(spec['omega']); self.c = float(spec['c']); self.beta = float(spec.get('beta', 0.))
        self.n_acts = int(spec.get('n_acts', 5)); self.n_acts_partner = n_acts_partner
        self.cg = np.asarray(cost_grid, float); self.wg = np.asarray(omega_grid, float)
        pc = np.full(len(self.cg), 1 / len(self.cg)) if prior_c is None else np.asarray(prior_c, float)
        pw = np.full(len(self.wg), 1 / len(self.wg)) if prior_w is None else np.asarray(prior_w, float)
        L0 = initial_logodds(w)[:, i]
        px1 = sigmoid(L0)
        base = np.stack([(1 - px1)[:, None] * PS_GIVEN_X[0][None, :], px1[:, None] * PS_GIVEN_X[1][None, :]], axis=1)  # (n,2,S)
        self.J = base[:, :, :, None, None] * pc[None, None, None, :, None] * pw[None, None, None, None, :]        # (n,2,S,nC,nW)
        self.M = np.zeros(n)                     # log-odds shift i believes it induced in j
        self.pos = np.full(n, PREF[i], np.int8)  # own public position (starts at own preferred version)
        self.relief_partner = np.zeros(n, bool)  # partner's concession cost relieved on its next move
        self.relieved = np.zeros(n, bool)        # own concession cost relieved (partner reassured last move)
        self.relieved_believed = np.zeros(n, bool)  # what the partner believes about my relief (placebo differs)
        self.c_active = np.full(n, self.c)       # for interventions (removal)

    # ---- beliefs
    def logodds(self):
        return logit(self.J[:, 1].sum(axis=(1, 2, 3)))

    def q_own(self):
        p1 = sigmoid(self.logodds())
        return p1 if PREF[self.i] == 1 else 1 - p1

    def partner_marginals(self):
        m = self.J.sum(axis=(1, 2))                 # (n, nC, nW)
        return m.sum(axis=2), m.sum(axis=1)         # P(c_j), P(omega_j)

    def partner_belief_estimate(self):
        """i's expected value of partner's P(x=1)."""
        q1 = sigmoid(PRIOR_L[self.j] + EV_S[None, :, None] + self.wg[None, None, :] * self.M[:, None, None])  # (n,S,nW)
        wgt = self.J.sum(axis=(1, 3))               # (n,S,nW)
        return (wgt * q1).sum(axis=(1, 2))

    # ---- model of the partner
    def partner_act_probs(self, M, pos_i, relief):
        """P(a_j | s_j, c_j, omega_j) under i's level-0 model of j. Returns (n,S,nC,nW,n_acts_partner)."""
        q1 = sigmoid(PRIOR_L[self.j] + EV_S[None, :, None] + self.wg[None, None, :] * M[:, None, None])       # (n,S,nW)
        q_own_j = q1 if PREF[self.j] == 1 else 1 - q1
        c_eff = np.where(relief[:, None], 0., self.cg[None, :])                                              # (n,nC)
        L = losses(q_own_j[:, :, None, :], c_eff[:, None, :, None], self.n_acts_partner)                     # (n,S,nC,nW,a)
        return softmax_rows(-L / TAU)

    def info_gain(self):
        n = self.J.shape[0]; ig = np.zeros((n, self.n_acts)); H0 = entropy(self.J)
        for a in range(self.n_acts):
            M_a = self.M + (llr(LAM) if a == ASSERT else -llr(LAM) if a == CONCEDE else 0.) * (1 if PREF[self.i] == 1 else -1)
            pos_a = np.where(a == ASSERT, PREF[self.i], np.where(a == CONCEDE, 1 - PREF[self.i], self.pos))
            probs = temper(self.partner_act_probs(M_a, pos_a, np.full(n, a == REASSURE)), self.omega)    # (n,S,nC,nW,b)
            expH = np.zeros(n)
            for b in range(self.n_acts_partner):
                Jb, Zb = posterior_after(self.J, probs[..., b], self.omega)
                expH += Zb * entropy(Jb)
            ig[:, a] = H0 - expH
        return ig

    def act_probs(self):
        c_eff = np.where(self.relieved, 0., self.c_active)
        L = losses(self.q_own(), c_eff, self.n_acts)
        ig = self.info_gain() if self.beta > 0 else np.zeros_like(L)
        return softmax_rows((-L + self.beta * ig) / TAU), ig

    def after_own_act(self, a):
        factual = (a == ASSERT) | (a == CONCEDE)
        v = np.where(a == ASSERT, PREF[self.i], 1 - PREF[self.i])
        self.M += np.where(factual, np.where(v == 1, 1., -1.) * llr(LAM), 0.)
        self.pos = np.where(factual, v, self.pos).astype(np.int8)
        self.relief_partner = a == REASSURE
        return factual, v

    def observe_partner_act(self, a_j, relief_j):
        probs = temper(self.partner_act_probs(self.M, self.pos, relief_j), self.omega)
        like = np.take_along_axis(probs, a_j[:, None, None, None, None].astype(int).repeat(N + 1, 1).repeat(len(self.cg), 2).repeat(len(self.wg), 3), axis=4)[..., 0]
        self.J, _ = posterior_after(self.J, like, self.omega)

    def observe_external(self, e, rho_e):
        """Unattenuated shared observation e in {0,1} about x with reliability rho_e."""
        like = np.where(e[:, None] == np.array([0, 1])[None, :], rho_e, 1 - rho_e)   # (n,2)
        self.J = self.J * like[:, :, None, None, None]
        self.J /= self.J.sum(axis=(1, 2, 3, 4), keepdims=True)


class Level0:
    """Naive agent (v0.4/v0.6 B): fixed-precision updates on partner assertions, cost-only policy."""

    def __init__(self, i, w, spec, n_acts=4):
        n = len(w['truth']); self.i = i; self.j = 1 - i
        self.omega = float(spec.get('omega', 1.0)); self.c = float(spec['c']); self.beta = 0.
        self.n_acts = int(spec.get('n_acts', n_acts))
        self.L = initial_logodds(w)[:, i].copy()
        self.pos = np.full(n, PREF[i], np.int8); self.relieved = np.zeros(n, bool); self.relieved_believed = np.zeros(n, bool); self.relief_partner = np.zeros(n, bool)
        self.c_active = np.full(n, self.c)

    def logodds(self):
        return self.L

    def q_own(self):
        p1 = sigmoid(self.L); return p1 if PREF[self.i] == 1 else 1 - p1

    def act_probs(self):
        c_eff = np.where(self.relieved, 0., self.c_active)
        L = losses(self.q_own(), c_eff, self.n_acts)
        return softmax_rows(-L / TAU), np.zeros_like(L)

    def after_own_act(self, a):
        factual = (a == ASSERT) | (a == CONCEDE)
        v = np.where(a == ASSERT, PREF[self.i], 1 - PREF[self.i])
        self.pos = np.where(factual, v, self.pos).astype(np.int8)
        self.relief_partner = a == REASSURE
        return factual, v

    def observe_partner_factual(self, factual, v):
        self.L[factual] += self.omega * np.where(v[factual] == 1, 1., -1.) * llr(LAM)

    def observe_external(self, e, rho_e):
        self.L += np.where(e == 1, 1., -1.) * llr(rho_e)

    def partner_marginals(self):
        return None, None

    def partner_belief_estimate(self):
        return np.full(len(self.L), np.nan)


def dialogue(w, cond):
    """cond: {'A': spec, 'B': spec, 'B_level': 1|0, 'cost_grid', 'omega_grid', 'prior_c', 'prior_w',
             'n_acts_partner_model', 'intervention': {...}}. spec = {'omega', 'c', 'beta', 'n_acts'}."""
    n = len(w['truth']); R = CFG['rounds']
    cg = cond.get('cost_grid', P['cost_grid']); wg = cond.get('omega_grid', P['omega_grid'])
    nap = int(cond.get('n_acts_partner_model', 5))
    iv = cond.get('intervention', {}) or {}
    A = Level1(0, w, cond['A'], cg, wg, nap, cond.get('prior_c'), cond.get('prior_w'))
    B = Level1(1, w, cond['B'], cg, wg, nap, cond.get('prior_c'), cond.get('prior_w')) if cond.get('B_level', 1) == 1 else Level0(1, w, cond['B'])
    agents = [A, B]
    acts = np.empty((n, R, 2), np.int8); Lh = np.empty((n, R + 1, 2)); posh = np.empty((n, R + 1, 2), np.int8)
    est = np.full((n, R + 1, 2), np.nan); igh = np.full((n, R, 2, 5), np.nan); relief_h = np.zeros((n, R, 2), bool); probh = np.full((n, R, 2, 5), np.nan)
    truth_e = None
    if 'external_obs_round' in iv:
        rng = np.random.default_rng(CFG['seeds'].get('external', 4242))
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
            a = draw(prob, w['u'][:, r, i])
            if 'force_reassure' in iv and r == iv['force_reassure']['round'] and name == iv['force_reassure']['agent']:
                a = np.full(n, REASSURE, np.int8)
            acts[:, r, i] = a; probh[:, r, i, :prob.shape[1]] = prob
            if ig is not None: igh[:, r, i, :ig.shape[1]] = ig
            factual, v = ag.after_own_act(a)
            relief_h[:, r, i] = ag.relieved
            # partner observes
            placebo = bool(iv.get('placebo', False))
            if isinstance(other, Level1):
                other.observe_partner_act(a, relief_j=ag.relieved_believed)   # partner's model conditions on the relief it believes it granted
            else:
                other.observe_partner_factual(factual, v)
            other.relieved_believed = ag.relief_partner.copy()
            other.relieved = ag.relief_partner & (not placebo)              # actual relief (placebo: believed but not granted)
            ag.relieved = np.zeros(n, bool); ag.relieved_believed = np.zeros(n, bool)   # consumed by this move
        for i, ag in enumerate(agents):
            Lh[:, r + 1, i] = ag.logodds(); posh[:, r + 1, i] = ag.pos; est[:, r + 1, i] = ag.partner_belief_estimate()
    out = {'truth': w['truth'], 'L': Lh, 'pos': posh, 'acts': acts, 'oracle': oracle_logodds(w), 'est': est, 'ig': igh,
           'relief': relief_h, 'prob': probh, 'private': w['private'], 'A_omega': A.omega, 'A_c': A.c, 'B_omega': B.omega, 'B_c': B.c}
    for name, ag in (('A', A), ('B', B)):
        pc, pw = ag.partner_marginals()
        if pc is not None:
            out[f'{name}_pC_partner'] = pc; out[f'{name}_pW_partner'] = pw
    return out


# ----------------------------------------------------------------------------- speech-label layer

E0 = np.array([
    # BUILD SEEK UNSEAL LOCK DRAIN FLOOD EDGE VOID SEAL SHIFT INSUFF
    [0.00, 0.00, 0.00, 0.30, 0.00, 0.50, 0.10, 0.00, 0.00, 0.00, 0.10],   # ASSERT
    [0.60, 0.00, 0.30, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10],   # CONCEDE
    [0.00, 0.00, 0.00, 0.50, 0.20, 0.20, 0.00, 0.00, 0.00, 0.00, 0.10],   # PRESSURE
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.40, 0.00, 0.10],   # SILENCE
    [0.30, 0.50, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10],   # REASSURE
])
C_MAX = max(P['cost_grid'])


def emission_matrix(omega, c, g):
    """Per-act emission distribution for an agent with mechanism (omega, c) and modulation strength g."""
    E = E0.copy()
    if g > 0 and c > 0:
        f = g * c / C_MAX
        for a in (ASSERT, PRESSURE):
            for src in (LIDX['FLOOD'], LIDX['DRAIN']):
                mv = f * E[a, src]; E[a, src] -= mv; E[a, LIDX['LOCK']] += mv
    if g > 0 and omega < 1:
        f = g * (1 - omega)
        mv = f * E[ASSERT, LIDX['LOCK']]; E[ASSERT, LIDX['LOCK']] -= mv; E[ASSERT, LIDX['FLOOD']] += mv
        mv = f * E[ASSERT, LIDX['EDGE']]; E[ASSERT, LIDX['EDGE']] -= mv; E[ASSERT, LIDX['VOID']] += mv
    return E


def emit_labels(d, g, noise=None, seed=0):
    """Labels for both agents' acts. noise: None or (11,11) row-normalised confusion matrix (true -> observed)."""
    rng = np.random.default_rng(seed); n, R, _ = d['acts'].shape
    z = np.empty((n, R, 2), np.int8)
    for i, (om, c) in enumerate(((d['A_omega'], d['A_c']), (d['B_omega'], d['B_c']))):
        E = emission_matrix(om, c, g)
        a = d['acts'][:, :, i]
        u = rng.random((n, R))
        zi = draw(E[a], u)
        pos_change = d['pos'][:, 1:, i] != d['pos'][:, :-1, i]
        shift = pos_change & (rng.random((n, R)) < g * 0.5)
        zi = np.where(shift, LIDX['SHIFT'], zi)
        if noise is not None:
            zi = draw(np.asarray(noise)[zi], rng.random((n, R)))
        z[:, :, i] = zi
    return z


def load_noise(name):
    import pandas as pd
    M = pd.read_csv(ROOT / 'data' / f'{name}_confusion_rownorm_tierA.csv', index_col=0).reindex(index=LABELS, columns=LABELS).fillna(0).to_numpy()
    zero = M.sum(axis=1) == 0
    M[zero] = np.eye(len(LABELS))[zero]           # rows with no support (INSUFF) pass through unchanged
    return M


# ----------------------------------------------------------------------------- metrics

def metrics(d):
    import pandas as pd
    truth = d['truth'].astype(int); n = len(truth); tail = CFG['tail_rounds']; a = d['acts']; pos = d['pos'].astype(int)
    q1 = sigmoid(d['L']); pt = np.where(truth[:, None, None] == 1, q1, 1 - q1)
    nonc = (a[:, -tail:, :] != CONCEDE).mean(axis=1)                       # (n,2)
    df = pd.DataFrame({'world': np.arange(n), 'truth': truth, 'A_right': (truth == 1).astype(float),
                       'A_own0': (q1[:, 0, 0] > 0.5).astype(float), 'B_own0': (q1[:, 0, 1] < 0.5).astype(float),
                       'nonc_A': nonc[:, 0], 'nonc_B': nonc[:, 1], 'nonc_mutual': ((a[:, -tail:, 0] != CONCEDE) & (a[:, -tail:, 1] != CONCEDE)).mean(axis=1),
                       'U_final': (pos[:, -tail:, 0] != pos[:, -tail:, 1]).mean(axis=1),
                       'no_concession_either': (a != CONCEDE).all(axis=(1, 2)).astype(float),
                       'A_p_truth0': pt[:, 0, 0], 'A_p_truth': pt[:, -1, 0], 'B_p_truth0': pt[:, 0, 1], 'B_p_truth': pt[:, -1, 1],
                       'A_dbelief': np.abs(q1[:, -1, 0] - q1[:, 0, 0]), 'B_dbelief': np.abs(q1[:, -1, 1] - q1[:, 0, 1]),
                       'belief_gap_final': np.abs(q1[:, -1, 0] - q1[:, -1, 1]),
                       'A_hidden_shift': np.abs(q1[:, -1, 0] - q1[:, 0, 0]) * (pos[:, -1, 0] == pos[:, 0, 0]),
                       'B_hidden_shift': np.abs(q1[:, -1, 1] - q1[:, 0, 1]) * (pos[:, -1, 1] == pos[:, 0, 1]),
                       'dist_oracle': np.abs(q1[:, -1, :] - sigmoid(d['oracle'])[:, None]).mean(axis=1),
                       'A_transp_err': np.abs(d['est'][:, -1, 0] - q1[:, -1, 1]), 'B_transp_err': np.abs(d['est'][:, -1, 1] - q1[:, -1, 0])})
    for i, nm in enumerate('AB'):
        for code, act in enumerate(ACTS):
            df[f'{nm}_{act}'] = (a[:, :, i] == code).mean(axis=1)
    for nm in 'AB':
        if f'{nm}_pC_partner' in d:
            partner_c = d['B_c'] if nm == 'A' else d['A_c']; partner_w = d['B_omega'] if nm == 'A' else d['A_omega']
            cg = np.asarray(P['cost_grid']); wg = np.asarray(P['omega_grid'])
            ci = int(np.argmin(np.abs(cg - partner_c))); wi = int(np.argmin(np.abs(wg - partner_w)))
            df[f'{nm}_pC_true'] = d[f'{nm}_pC_partner'][:, ci]; df[f'{nm}_pW_true'] = d[f'{nm}_pW_partner'][:, wi]
    tr = {'A_p_truth_t': pt[:, :, 0].mean(0), 'B_p_truth_t': pt[:, :, 1].mean(0),
          'nonc_mutual_t': ((a[:, :, 0] != CONCEDE) & (a[:, :, 1] != CONCEDE)).mean(0),
          'U_t': (pos[:, :, 0] != pos[:, :, 1]).mean(0)}
    return df, tr


def paired(a, b, label, exclude=('world', 'truth', 'A_right')):
    rows = []
    for col in a.columns:
        if col in exclude: continue
        delta = (a[col] - b[col]).dropna().to_numpy()
        if len(delta) == 0: continue
        se = delta.std(ddof=1) / np.sqrt(len(delta))
        rows.append({'contrast': label, 'metric': col, 'delta': delta.mean(), 'mc_lo': delta.mean() - 1.96 * se, 'mc_hi': delta.mean() + 1.96 * se})
    return rows
