"""Idealised observers for mechanism recovery (E2). See MODEL_SPEC.md §8.

Channels
  'acts+private' : both agents' acts and agent i's private record count are observed
  'acts'         : both agents' acts observed; i's private records marginalised (exact, 5 replays)
  'labels'       : both agents' acts seen only through speech labels. Plug-in decoder: the partner's acts and
                   i's own state-propagating acts are replaced by MAP decodes of the labels; i's own emissions
                   are scored with the exact label likelihood sum_a E(z|a) P(a|history). This is an
                   approximation and a lower bound on a labels-only observer.
Hypotheses: attenuation family {omega in grid, c = 0}, concession-cost family {omega = 1, c in grid};
the reference point (1, 0) belongs to both. Equal family priors, uniform within family.
"""
import numpy as np
import core as rd

R = rd.CFG['rounds']


def hypotheses(cost_grid=None, omega_grid=None):
    cg = list(cost_grid or rd.P['cost_grid']); wg = list(omega_grid or rd.P['omega_grid'])
    att = [(o, 0.0) for o in wg]; cost = [(1.0, c) for c in cg]
    hyps = sorted(set(att) | set(cost))
    fam = {'attenuation': [hyps.index(h) for h in att], 'cost': [hyps.index(h) for h in cost]}
    return hyps, fam


def fake_world(n, i, s_count):
    """World whose agent-i private record sum is s_count (only the sum enters the agent)."""
    priv = np.zeros((n, 2, rd.N), np.int8); priv[:, i, :s_count] = 1
    return {'truth': np.zeros(n, int), 'private': priv, 'u': np.zeros((n, R, 2))}


def replay_loglik(acts, i, hyp, beta, s_count, labels=None, E_eff=None, order='AB'):
    """Log-likelihood (n,) of agent i's observed acts (or labels) given hypothesis (omega, c), beta and private count."""
    n = acts.shape[0]; j = 1 - i
    ag = rd.Level1(i, fake_world(n, i, s_count), {'omega': hyp[0], 'c': hyp[1], 'beta': beta}, rd.P['cost_grid'], rd.P['omega_grid'], 5)
    ll = np.zeros(n); rows = np.arange(n)
    j_relieved_believed = np.zeros(n, bool)
    first = 0 if order == 'AB' else 1
    for r in range(R):
        for who in ((0, 1) if first == 0 else (1, 0)):
            if who == i:
                prob, _ = ag.act_probs()
                if labels is None:
                    ll += np.log(np.clip(prob[rows, acts[:, r, i]], 1e-300, None))
                else:
                    pz = (prob[:, :, None] * E_eff[None, :, :]).sum(axis=1)          # (n, 11)
                    ll += np.log(np.clip(pz[rows, labels[:, r, i]], 1e-300, None))
                ag.after_own_act(acts[:, r, i])
                j_relieved_believed = ag.relief_partner.copy()
                ag.relieved = np.zeros(n, bool); ag.relieved_believed = np.zeros(n, bool)
            else:
                a_j = acts[:, r, j]
                ag.observe_partner_act(a_j, relief_j=j_relieved_believed)
                ag.relieved = a_j == rd.REASSURE; ag.relieved_believed = a_j == rd.REASSURE
    return ll


def decode_labels(z, E_dec):
    """MAP act for each label under a decoding emission matrix (5, 11)."""
    return np.argmax(E_dec, axis=0)[z].astype(np.int8)


def recover(d, i, channel, beta, labels=None, g=0.0, noise=None, decode_with_g=False):
    """Family posterior for agent i. Returns dict with loglik (n,H), family posterior (n,), MAP flags, parameter means."""
    hyps, fam = hypotheses(); n = d['acts'].shape[0]
    acts = d['acts']
    if channel == 'labels':
        assert labels is not None
        E_dec = rd.E0 if noise is None else rd.E0 @ noise
        acts = np.stack([decode_labels(labels[:, :, 0], E_dec), decode_labels(labels[:, :, 1], E_dec)], axis=2)
    LL = np.zeros((n, len(hyps)))
    for h, (om, c) in enumerate(hyps):
        E_eff = None
        if channel == 'labels':
            E_eff = rd.emission_matrix(om, c, g)
            if noise is not None: E_eff = E_eff @ noise
        if channel == 'acts+private':
            s = d['private'][:, i].sum(axis=1)
            ll = np.zeros(n)
            for sc in range(rd.N + 1):
                m = s == sc
                if m.any():
                    sub = {'acts': acts[m]}; lab = labels[m] if labels is not None else None
                    ll[m] = replay_loglik(acts[m], i, (om, c), beta, sc, lab, E_eff)
            LL[:, h] = ll
        else:
            # marginalise private count: P(s|x) with P(x)=1/2 -> P(s) = mean of the two binomials
            ps = rd.PS_GIVEN_X.mean(axis=0)
            lls = np.stack([replay_loglik(acts, i, (om, c), beta, sc, labels, E_eff) for sc in range(rd.N + 1)], axis=1)
            mx = lls.max(axis=1, keepdims=True)
            LL[:, h] = mx[:, 0] + np.log((np.exp(lls - mx) * ps[None, :]).sum(axis=1))
    out = {'hyps': hyps, 'loglik': LL}
    fam_ll = {}
    for f, idx in fam.items():
        sub = LL[:, idx]; mx = sub.max(axis=1, keepdims=True)
        fam_ll[f] = mx[:, 0] + np.log(np.exp(sub - mx).mean(axis=1))
    la, lc = fam_ll['attenuation'], fam_ll['cost']
    out['p_attenuation'] = 1 / (1 + np.exp(np.clip(lc - la, -700, 700)))
    # posterior mean parameters on the joint grid (uniform over unique hypotheses)
    mx = LL.max(axis=1, keepdims=True); w = np.exp(LL - mx); w /= w.sum(axis=1, keepdims=True)
    H = np.array(hyps)
    out['omega_mean'] = w @ H[:, 0]; out['c_mean'] = w @ H[:, 1]
    return out


def summarise(res, true_family, true_omega, true_c):
    pa = res['p_attenuation']; p_true_fam = pa if true_family == 'attenuation' else 1 - pa
    return {'correct_family': float((p_true_fam > 0.5).mean()), 'posterior_on_family': float(p_true_fam.mean()),
            'omega_MAE': float(np.abs(res['omega_mean'] - true_omega).mean()), 'c_MAE': float(np.abs(res['c_mean'] - true_c).mean())}
