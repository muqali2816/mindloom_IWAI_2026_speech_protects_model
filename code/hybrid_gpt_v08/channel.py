"""Finite-calibration uncertainty, not a fitted domain-transfer model."""
import numpy as np
import pandas as pd
import model as M

def calibration(pseudocount=None):
    a=M.P['annotation_dirichlet_pseudocount'] if pseudocount is None else pseudocount
    table=pd.read_csv(M.ROOT/'source/engine_confusion_counts_tierA.csv',index_col=0)
    counts=table.reindex(index=['LOCK','SEEK'],columns=M.LABELS).to_numpy(float)
    return counts+a

def emit(d,seed,alpha=None,uninformative=False):
    """One uncertain channel draw per posterior-predictive world, fixed over its turns.
    Both participants share the world's channel; it does not change per utterance.
    """
    alpha=calibration() if alpha is None else alpha;r=np.random.default_rng(seed)
    n,T=d['y'].shape
    if uninformative:
        C=np.repeat(r.dirichlet(alpha.mean(0),n)[:,None,:],2,axis=1)
    else:C=np.stack([r.dirichlet(alpha[f],n) for f in [0,1]],1)
    pp=C[np.arange(n)[:,None],d['y']%2]
    z=(pp.cumsum(-1)<r.random((n,T,1))).sum(-1).clip(0,len(M.LABELS)-1)
    return z,C

def sample_information(alpha,n=4000,seed=129):
    r=np.random.default_rng(seed);C=np.stack([r.dirichlet(alpha[f],n) for f in [0,1]],1)
    avg=C.mean(1);mi=M.entropy(avg)-M.entropy(C).mean(1)
    overlap=np.minimum(C[:,0],C[:,1]).sum(1)
    return dict(equal_form_prior_MI_nats=np.quantile(mi,[.025,.5,.975]).tolist(),
                row_overlap=np.quantile(overlap,[.025,.5,.975]).tolist(),
                posterior_mean=(alpha/alpha.sum(1,keepdims=True)).tolist(),alpha=alpha.tolist())
