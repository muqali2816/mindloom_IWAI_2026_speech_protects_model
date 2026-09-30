"""Exact external observer for the hybrid dyad (MODEL_SPEC §7).

Hidden state per world: (x, s_A, s_B, f_A_prev, f_B_prev). Agents' internal posteriors depend only on public acts,
shared signals and relief, so they are replayed deterministically for each (hypothesis, private count).
Channels: 'acts' (acts + signal record), 'labels' (+ form labels), 'oracle' (+ true forms), 'null' (+ act-only labels).
Order 'sim' only (both agents act from the start-of-round state).
External observer (the "annotator" of Part I placed inside the simulation). It sees the public record of acts and the
shared-signal record, but the forms (invite/restrict) only through noisy labels drawn from the archived LOCK/SEEK
confusion rows (data/annotation_channel.json). It computes, for every world, the exact posterior over the speaker's
mechanism family (attenuation / concession cost / access protection) by filtering over the same level-0 partner model
used by the agents (core.Agent.partner_act_probs); bruteforce_paths_equals_filter in validate.py checks the filter
against explicit path enumeration. Label information is measured as the gain in P(true family) and in next-signal
log-loss relative to acts + signals alone; null labels (drawn from the ASK act instead of the form) must give zero gain.
"""
import numpy as np
import core as hd

R = hd.CFG['rounds']; FORMS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])   # form pair (f_A, f_B), index 2*fA+fB


def hypotheses(cost_grid=None, omega_grid=None, gamma_grid=None):
    cg = list(cost_grid or hd.P['cost_grid']); wg = list(omega_grid or hd.P['omega_grid']); gg = list(gamma_grid or hd.P['gamma_grid'])
    fam = {'attenuation': [(o, 0., 0.) for o in wg], 'cost': [(1., c, 0.) for c in cg], 'access': [(1., 0., 0.)] + [(1., 0., g) for g in gg]}
    hyps = sorted(set(h for v in fam.values() for h in v))
    return hyps, {k: [hyps.index(h) for h in v] for k, v in fam.items()}


def fake_world(n, i, s_count, N):
    priv = np.zeros((n, 2, N), np.int8); priv[:, i, :s_count] = 1
    return {'truth': np.zeros(n, int), 'private': priv}


def replay_policies(d, i, spec, cond, s_count):
    """For agent i with mechanism spec and private count s_count: list over t of policies (2, n, nA, nF) indexed by partner's previous form."""
    n, _, _ = d['acts'].shape; j = 1 - i; N = d['private'].shape[2]
    ag = hd.Agent(i, fake_world(n, i, s_count, N), spec, cond)
    m_shared = np.zeros(n); pols = []
    for t in range(R):
        e = d['ev'][:, t].astype(int); got = e != 2
        m_shared += np.where(got, 2 * e - 1, 0)
        ag.observe_signal(np.where(got, e, 0), got)
        pa = d['acts'][:, t - 1, j] if t > 0 else np.full(n, hd.ASSERT)
        pt = np.stack([ag.policy(pa, np.full(n, fp), m_shared)[0] for fp in (0, 1)], axis=0)
        pols.append(pt)
        ag.after_own_act(d['acts'][:, t, i])
        partner_relieved = (d['acts'][:, t - 1, i] == hd.REASSURE) if t > 0 else np.zeros(n, bool)   # i reassured last round -> j relieved now
        ag.observe_partner_act(d['acts'][:, t, j], partner_relieved, m_shared)
        ag.relieved = d['acts'][:, t, j] == hd.REASSURE
    return pols


def filter_world(d, hyp, cond, channel, spec_B, labels=None, C=None, null_labels=None):
    """Exact HMM for one hypothesis about A. Returns log-evidence (n,), next-signal prob (n,R-1), next-act prob for A (n,R-1,nA)."""
    n = d['acts'].shape[0]; N = d['private'].shape[2]; S = N + 1
    acts = [hd.ACT_NAMES.index(a) if isinstance(a, str) else a for a in cond.get('acts', hd.ACT_NAMES)]
    nA = len(acts); nF = 2; act_index = np.full(6, -1); act_index[acts] = np.arange(nA)
    specA = {'omega': hyp[0], 'c': hyp[1], 'gamma': hyp[2]}
    polA = [replay_policies(d, 0, specA, cond, s) for s in range(S)]      # [s][t] -> (2,n,nA,nF)
    polB = [replay_policies(d, 1, spec_B, cond, s) for s in range(S)]
    ev_s, ps = hd.s_tables(N); rho = hd.P['private_reliability']
    # state tensor (n, 2x, S_A, S_B, 4 formpairs)
    bel = np.zeros((n, 2, S, S, 4)); bel[:, :, :, :, 3] = 0.5 * ps[:, :, None] * ps[:, None, :]      # start: both invite
    logZ = np.zeros(n); logZ_hist = np.zeros((n, R)); next_sig = np.zeros((n, R - 1)); next_act = np.zeros((n, R - 1, nA)); idx = np.arange(n)
    for t in range(R):
        e = d['ev'][:, t].astype(int); prev_acts = d['acts'][:, t - 1] if t > 0 else np.full((n, 2), hd.ASSERT)
        rates = np.stack([hd.arrival_rate(np.broadcast_to(f, (n, 2)), prev_acts) for f in FORMS], axis=1)      # (n,4)
        like_e = np.where(e[:, None, None] == 2, 1 - rates[:, None, :], rates[:, None, :] * np.where(e[:, None, None] == np.arange(2)[None, :, None], rho, 1 - rho))  # (n,2,4)
        bel = bel * like_e[:, :, None, None, :]
        # moves: P(u_A,f_A | s_A, f_B_prev) and P(u_B,f_B | s_B, f_A_prev)
        uA = act_index[d['acts'][:, t, 0]]; uB = act_index[d['acts'][:, t, 1]]
        PA = np.stack([np.stack([polA[s][t][fp][idx, uA] for fp in (0, 1)], axis=1) for s in range(S)], axis=1)   # (n,S_A,2 fBprev,nF)
        PB = np.stack([np.stack([polB[s][t][fp][idx, uB] for fp in (0, 1)], axis=1) for s in range(S)], axis=1)   # (n,S_B,2 fAprev,nF)
        new = np.zeros_like(bel)
        for old in range(4):
            fA_prev, fB_prev = FORMS[old]
            for newp in range(4):
                fA, fB = FORMS[newp]
                trans = PA[:, :, fB_prev, fA][:, :, None] * PB[:, :, fA_prev, fB][:, None, :]          # (n,S_A,S_B)
                emit = 1.
                if channel == 'labels':
                    emit = C[fA, labels[:, t, 0]] * C[fB, labels[:, t, 1]]
                elif channel == 'oracle':
                    emit = ((d['forms'][:, t, 0] == fA) & (d['forms'][:, t, 1] == fB)).astype(float)
                new[:, :, :, :, newp] += bel[:, :, :, :, old] * trans[:, None, :, :] * (emit if np.isscalar(emit) else emit[:, None, None, None])
        if channel == 'null':
            fn = (d['acts'][:, t] == hd.ASK).astype(int)
            new *= (C[fn[:, 0], null_labels[:, t, 0]] * C[fn[:, 1], null_labels[:, t, 1]])[:, None, None, None, None]
        Z = new.sum(axis=(1, 2, 3, 4)); logZ += np.log(np.clip(Z, 1e-300, None)); logZ_hist[:, t] = logZ; bel = new / np.clip(Z, 1e-300, None)[:, None, None, None, None]
        if t < R - 1:
            cur_rates = np.stack([hd.arrival_rate(np.broadcast_to(f, (n, 2)), d['acts'][:, t]) for f in FORMS], axis=1)   # (n,4)
            next_sig[:, t] = (bel.sum(axis=(1, 2, 3)) * cur_rates).sum(1)
            # next act of A: sum over s_A and f_B (current) of P(u | s_A, f_B) marginalised over forms
            wA = bel.sum(axis=(1, 3))                                                                 # (n,S_A,4)
            for s in range(S):
                for fp in range(4):
                    fB = FORMS[fp][1]
                    next_act[:, t] += wA[:, s, fp][:, None] * polA[s][t + 1][fB].sum(axis=2)
    return logZ_hist, next_sig, next_act


def recover(d, cond, channel, spec_B, labels=None, C=None, null_labels=None, hyps=None):
    """Family posterior for agent A and Bayesian-model-averaged next-signal / next-act predictions (weights use data up to t)."""
    hyps_, fam = hypotheses(); hyps_ = hyps or hyps_
    n = d['acts'].shape[0]; LLh = np.zeros((len(hyps_), n, R)); NS = []; NA = []
    for h, hyp in enumerate(hyps_):
        ll, ns, na = filter_world(d, hyp, cond, channel, spec_B, labels, C, null_labels); LLh[h] = ll; NS.append(ns); NA.append(na)
    LL = LLh[:, :, -1].T
    mx = LL.max(axis=1, keepdims=True); w = np.exp(LL - mx); w /= w.sum(axis=1, keepdims=True)
    fam_ll = {}
    for f, ids in fam.items():
        sub = LL[:, ids]; m = sub.max(axis=1, keepdims=True); fam_ll[f] = m[:, 0] + np.log(np.exp(sub - m).mean(axis=1))
    F = np.stack([fam_ll[f] for f in fam], axis=1); F = np.exp(F - F.max(axis=1, keepdims=True)); F /= F.sum(axis=1, keepdims=True)
    H = np.array(hyps_)
    wt = sequential_weights(LLh[:, :, :-1])                       # (H, n, R-1): weights after round t, used to predict t+1
    NS = np.stack(NS, 0); NA = np.stack(NA, 0)
    next_sig = (wt * NS).sum(0); next_act = (wt[..., None] * NA).sum(0)
    return {'hyps': hyps_, 'families': list(fam), 'loglik': LL, 'family_post': F, 'omega_mean': w @ H[:, 0], 'c_mean': w @ H[:, 1], 'gamma_mean': w @ H[:, 2],
            'next_sig': next_sig, 'next_act': next_act, 'w_final': w}


def sequential_weights(LL_t):
    """LL_t: (H, n, R) cumulative log-evidence after each round -> posterior weights per round."""
    m = LL_t.max(axis=0, keepdims=True); w = np.exp(LL_t - m); return w / w.sum(axis=0, keepdims=True)


def summarise(res, true_family, true_hyp):
    fi = res['families'].index(true_family) if true_family in res['families'] else None
    out = {'omega_MAE': float(np.abs(res['omega_mean'] - true_hyp[0]).mean()), 'c_MAE': float(np.abs(res['c_mean'] - true_hyp[1]).mean()),
           'gamma_MAE': float(np.abs(res['gamma_mean'] - true_hyp[2]).mean())}
    if fi is not None:
        out['correct_family'] = float((res['family_post'].argmax(1) == fi).mean()); out['posterior_on_family'] = float(res['family_post'][:, fi].mean())
    return out
