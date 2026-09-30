"""H2 — recovery of A's mechanism by the exact observer over channels. Run from hybrid_dyad/ with PYTHONPATH=.
usage: python H2_recovery.py <generator>   generator in GENERATORS below.
Saves results/H2_raw_<generator>.npz with per-world quantities for every channel."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # package root on path
import sys, time, json
import numpy as np
import core as hd, observer as ob, run_utils as ru

T = hd.CFG['agent_types']; SEEDS = hd.CFG['seeds']; N_FULL = hd.CFG['n_worlds']; N_HALF = 1000
# generator name -> (A type, B type)
GENERATORS = {'honest': ('honest', 'honest'), 'attenuated': ('attenuated', 'honest'), 'costly': ('costly', 'honest'),
              'access': ('access', 'honest'), 'mixed': ('mixed', 'honest'),
              'access_x_access': ('access', 'access'), 'costly_x_costly': ('costly', 'costly')}
# channel name -> (observer channel, eta, n worlds)
CHANNELS = {'acts': ('acts', None, N_FULL), 'labels_eta0.25': ('labels', 0.25, N_FULL), 'oracle': ('oracle', None, N_FULL),
            'null': ('null', 0.25, N_FULL),
            'labels_eta0': ('labels', 0.0, N_FULL), 'labels_eta0.5': ('labels', 0.5, N_FULL), 'labels_eta1': ('labels', 1.0, N_HALF)}


def subset(d, m):
    n = d['acts'].shape[0]
    return {k: (v[:m] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n else v) for k, v in d.items()}


def main(gen):
    a_type, b_type = GENERATORS[gen]; cond = ru.cond(a_type, b_type); spec_B = dict(T[b_type])
    w = hd.make_worlds(N_FULL, SEEDS['H2']); d = hd.dialogue(w, cond)
    z, zn, rows = hd.emit_labels(d, SEEDS['labels'])
    df = hd.metrics(d); out = {'A_own0': df['A_own0'].to_numpy(), 'truth': d['truth'], 'ev': d['ev'], 'acts': d['acts'], 'forms': d['forms']}
    timing = {}
    for name, (channel, eta, m) in CHANNELS.items():
        t0 = time.time(); dm = subset(d, m); C = hd.channel_mean(eta) if eta is not None else None
        res = ob.recover(dm, cond, channel, spec_B, labels=z[:m], C=C, null_labels=zn[:m])
        out[f'{name}/family_post'] = res['family_post']; out[f'{name}/omega_mean'] = res['omega_mean']
        out[f'{name}/c_mean'] = res['c_mean']; out[f'{name}/gamma_mean'] = res['gamma_mean']
        out[f'{name}/next_sig'] = res['next_sig']; out[f'{name}/w_final'] = res['w_final']
        out[f'{name}/loss_sig'] = ru.logloss_signal(res, dm)                 # (m, R-1)
        out[f'{name}/loss_act'] = ru.logloss_act(res, dm, hd.ACT_NAMES)      # (m, R-1)
        timing[name] = round(time.time() - t0, 1); print(gen, name, m, timing[name], flush=True)
    out['hyps'] = np.array(res['hyps']); out['families'] = np.array(res['families']); out['timing'] = json.dumps(timing)
    out['true_hyp'] = np.array([T[a_type]['omega'], T[a_type]['c'], T[a_type]['gamma']])
    np.savez_compressed(ru.OUT / f'H2_raw_{gen}.npz', **out)


if __name__ == '__main__':
    main(sys.argv[1])
