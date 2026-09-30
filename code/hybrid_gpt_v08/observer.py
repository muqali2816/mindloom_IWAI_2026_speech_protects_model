"""External observer of the actual level-1 dyad.

Enumerates 16 parameter pairs and 9 correlated private-count pairs. With forms
observed, likelihood and mechanism posterior are exact on that finite grid.
With forms hidden, a stratified particle filter integrates their histories:
static parameter/private strata are never discarded; form-path particles are
resampled only within a stratum. Dirichlet emission rows are integrated with
path-specific sufficient counts. No future labels or evidence enter a forecast.

Always observes coarse acts AND already delivered verification evidence.
Forecast of E_t additionally conditions on the current utterance's coarse act
and (in label mode) its annotation, both available before E_t is delivered.
"""
import itertools
import numpy as np
import model as M
from channel import calibration

H=np.array(list(M.TYPES.values()))

def systematic(w,u):
    k=len(w);cdf=np.cumsum(w);cdf[-1]=1
    return np.searchsorted(cdf,(np.arange(k)+u)/k,side='right')

def recover_one(d,index,mode='actions_evidence',paths=8,seed=0,labels=None,alpha=None,
                exact_paths=False,information='joint',label_null=False,hypotheses=None):
    rng=np.random.default_rng(seed);alpha=calibration() if alpha is None else np.asarray(alpha)
    hs=H if hypotheses is None else np.asarray(hypotheses,float);nh=len(hs)
    nprivate=M.P['n_private'];strata=np.array(list(itertools.product(range(nh),range(nh),range(nprivate+1),range(nprivate+1))),int)
    ns=len(strata);K=1 if mode=='forms_evidence' or exact_paths else paths
    ids=np.repeat(np.arange(ns),K);st=strata[ids];n=len(ids)
    ps=M.private_likelihood();joint=ps.T@ps/2
    weights=joint[st[:,2],st[:,3]]/(nh*nh)/K
    ag=[M.Agent(0,st[:,2],hs[st[:,0]],d.get('beta',M.P['beta']),d.get('coupling',True)),
        M.Agent(1,st[:,3],hs[st[:,1]],d.get('beta',M.P['beta']),d.get('coupling',True))]
    counts=np.zeros((n,2,len(M.LABELS)),np.int16)
    T=d['y'].shape[1];pred_arrival=[];pred_act=[];history=[];miness=[]
    shared=0
    for t,i in enumerate(d['speaker']):
        pr,_=ag[i].policy(information);u=int(d['y'][index,t]//2);ev=int(d['e'][index,t])
        pred_act.append((weights[:,None]*pr.reshape(n,5,2).sum(2)).sum(0))
        yy=2*u+np.arange(2);pf=pr[:,yy].copy()
        if mode=='forms_evidence':pf[:,1-int(d['y'][index,t]%2)]=0
        use_label=mode=='labels_evidence' and not label_null
        if use_label:
            z=int(labels[index,t]);cp=(alpha[None,:,z]+counts[:,:,z])/(alpha.sum(1)[None,:]+counts.sum(2))
            pf*=cp
        lam=M.arrival(yy[None,:],ag[i].last_partner[:,None],d.get('coupling',True))
        prior_form=weights[:,None]*pf
        pred_arrival.append(float((prior_form*lam).sum()/prior_form.sum()))
        # Exact common-x posterior given this stratum's two private counts and prior shared evidence.
        px=M.sigmoid((2*(st[:,2]+st[:,3])-2*nprivate)*M.logit(M.P['rho_private'])+shared*M.logit(M.P['rho_verification']))
        prob1=px*M.P['rho_verification']+(1-px)*(1-M.P['rho_verification'])
        le=(1-lam) if ev==M.NONE else lam*(prob1 if ev==1 else 1-prob1)[:,None]
        branches=pf*le
        if exact_paths and mode!='forms_evidence':
            parent=np.repeat(np.arange(n),2);f=np.tile([0,1],n)
            weights=(weights[:,None]*branches).reshape(-1)
            ids=ids[parent];st=strata[ids]
            ag=[a.take(parent) for a in ag];counts=counts[parent].copy();n=len(parent)
        else:
            total=branches.sum(1);cond=branches/np.clip(total[:,None],1e-300,None)
            # Correlated random numbers across parameter strata improve comparisons,
            # without changing any stratum's proposal distribution.
            uniforms=np.tile(rng.random(K),ns)
            f=(uniforms>cond[:,0]).astype(int)
            weights*=total
        weights/=weights.sum()
        y=2*u+f
        ag[1-i].observe_move(y);ag[i].own_move(y)
        for a in ag:a.observe_evidence(np.full(n,ev))
        if use_label:counts[np.arange(n),f,z]+=1
        shared+=1 if ev==1 else -1 if ev==0 else 0
        if not exact_paths and K>1:
            W=weights.reshape(ns,K);mass=W.sum(1);wp=W/np.clip(mass[:,None],1e-300,None)
            ess=1/np.clip((wp**2).sum(1),1e-300,None);miness.append(float(ess.min()))
            picks=np.arange(n).reshape(ns,K);ru=rng.random()
            for s in range(ns):
                if mass[s]>0 and ess[s]<K/2:
                    picks[s]=s*K+systematic(wp[s],ru);W[s]=mass[s]/K
            parent=picks.reshape(-1)
            ag=[a.take(parent) for a in ag];counts=counts[parent].copy()
            # Each resampling is within a fixed stratum: ids/st remain unchanged.
            weights=W.reshape(-1)
        post=np.zeros((nh,nh));np.add.at(post,(st[:,0],st[:,1]),weights);history.append(post)
    return dict(posterior=post,history=np.array(history),arrival_probability=np.array(pred_arrival),
                act_probability=np.array(pred_act),min_path_ESS=min(miness) if miness else float(K),
                n_final_paths=n)
