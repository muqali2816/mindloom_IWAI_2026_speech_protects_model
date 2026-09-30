"""Exploratory reciprocal evidence-access model, 2026-09-30.

Both agents infer the same static binary fact, with opposite initial priors.
Their speech actions and relational forms control a shared verification channel.
No natural-language generation, neural measurement, or physiological load.
The one-step partner forecast holds the partner's previous public move fixed;
it is NOT recursive theory of mind or a Nash-equilibrium solution.
"""
import numpy as np

P=dict(rounds=12,rho=.75,lam_low=.35,lam_ask=.90,lam_floor=.05,
       tau=.20,k=2.,b=1.10,restriction_cost=.12)
STATES=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=int) # restriction=0; invite=1
ACTS=['ASSERT_OWN','CONCEDE','PRESSURE','SILENCE','ASK']
HYPOTHESES=[('reference',1.,0.,0.),('attenuation',.1,0.,0.),
            ('concession_cost',1.,1.4,0.),('access_protection',1.,0.,2.)]


def sigmoid(z):return 1/(1+np.exp(-z))


def beliefs(m,omega):
    m=np.asarray(m)
    return sigmoid(np.array([np.log(3.),-np.log(3.)])+m[...,None]*omega*np.log(3.))


def binary_information(q,rho):
    """Mutual information of a binary symmetric signal channel, in nats."""
    q=np.asarray(q);p1=q*rho+(1-q)*(1-rho)
    ent=lambda x: -x*np.log(np.clip(x,1e-300,None))-(1-x)*np.log(np.clip(1-x,1e-300,None))
    return ent(p1)-ent(rho)


def probability(q,omega,c,gamma,partner_form,partner_act,agent,
                epistemic=True,coupling=True,external_rate=None):
    """Joint policy P(coarse act, relational form), shape (n,5,2).

    R = ordinary act loss + cost of restricting + expected public contradiction
    G = R - I(x;next verification). Frequencies are not tempered.
    gamma is public-contradiction aversion, distinct from concession cost c.
    """
    q=np.asarray(q);n=len(q)
    own=q if agent==0 else 1-q
    risk=np.stack([P['k']*(1-own),P['k']*own+c,np.full(n,P['b']),np.full(n,P['b']),np.full(n,P['b'])],axis=1)
    request=(np.arange(5)[None,:,None]==4)|(np.asarray(partner_act)[:,None,None]==4)
    base=np.where(request,P['lam_ask'],P['lam_low'])
    if external_rate is not None:
        lam=np.full((n,5,2),external_rate)
    elif not coupling:
        lam=np.broadcast_to(base,(n,5,2)).copy()
    else:
        lam=P['lam_floor']+(base-P['lam_floor'])*np.asarray(partner_form)[:,None,None]*np.array([0,1])[None,None,:]
    rw=P['rho']**omega/(P['rho']**omega+(1-P['rho'])**omega)
    disconfirm=own*(1-rw)+(1-own)*rw
    ig=lam*binary_information(q,rw)[:,None,None] if epistemic else np.zeros_like(lam)
    risk=risk[:,:,None]+P['restriction_cost']*np.array([1,0])[None,None,:]+np.asarray(gamma)*lam*disconfirm[:,None,None]
    logits=-(risk-ig)/P['tau'];logits-=logits.max(axis=(1,2),keepdims=True)
    pr=np.exp(logits);pr/=pr.sum(axis=(1,2),keepdims=True)
    return pr


def arrival(forms,acts,coupling=True,external_rate=None):
    if external_rate is not None:return np.full(len(forms),external_rate)
    base=np.where((acts==4).any(axis=1),P['lam_ask'],P['lam_low'])
    if not coupling:return base
    return P['lam_floor']+(base-P['lam_floor'])*forms.prod(axis=1)


def worlds(n,seed):
    r=np.random.default_rng(seed);T=P['rounds']
    return dict(truth=r.integers(0,2,n),u_ev=r.random((n,T)),u_sign=r.random((n,T)),
                u_policy=r.random((n,T,2)),u_label=r.random((n,T,2)),u_null=r.random((n,T,2)))


def simulate(w,omega=1.,c=0.,gamma=0.,C=None,epistemic=True,coupling=True,
             intervention=None):
    n=len(w['truth']);T=P['rounds'];m=np.zeros(n,int)
    acts=np.zeros((n,T,2),int);forms=np.ones((n,T,2),int);ev=np.full((n,T),2,int)
    qhist=np.zeros((n,T+1,2));qhist[:,0]=beliefs(m,omega)
    prevf=np.ones((n,2),int);prevu=np.zeros((n,2),int)
    gammas=np.broadcast_to(np.asarray(gamma,float),(2,))
    rates=np.zeros((n,T))
    for t in range(T):
        external=.9 if intervention=='bypass' and t>=6 else None
        rate=arrival(prevf,prevu,coupling,external);rates[:,t]=rate
        got=w['u_ev'][:,t]<rate
        sign=np.where(w['u_sign'][:,t]<P['rho'],w['truth'],1-w['truth'])
        ev[got,t]=sign[got];m+=np.where(got,2*sign-1,0)
        q=beliefs(m,omega);qhist[:,t+1]=q
        for i in range(2):
            g=0. if intervention=='remove_gamma' and t>=6 else gammas[i]
            pp=probability(q[:,i],omega,c,g,prevf[:,1-i],prevu[:,1-i],i,epistemic,coupling,external)
            k=(pp.reshape(n,10).cumsum(1)<w['u_policy'][:,t,i,None]).sum(1).clip(0,9)
            acts[:,t,i]=k//2;forms[:,t,i]=k%2
        if intervention=='force_A_open' and t>=6:forms[:,t,0]=1
        prevf=forms[:,t].copy();prevu=acts[:,t].copy()
    out=dict(truth=w['truth'],q=qhist,ev=ev,acts=acts,forms=forms,rates=rates,omega=omega,c=c,gamma=gammas)
    if C is not None:
        cp=C[forms].cumsum(-1)
        out['labels']=(cp<w['u_label'][:,:,:,None]).sum(-1).clip(0,C.shape[1]-1)
        # Mechanism-independent null channel based only on the observed coarse act.
        fnull=(acts==4).astype(int)
        out['null_labels']=(C[fnull].cumsum(-1)<w['u_null'][:,:,:,None]).sum(-1).clip(0,C.shape[1]-1)
    return out


def metrics(d):
    q=d['q'][:,-1];truth=d['truth'][:,None]
    return dict(p_truth=np.where(truth==1,q,1-q).mean(1),disagreement=np.abs(q[:,0]-q[:,1]),
        evidence_count=(d['ev']!=2).sum(1),restriction_rate=(d['forms']==0).mean((1,2)),
        late_nonconcession=(d['acts'][:,-4:]!=1).mean((1,2)),
        late_ask=(d['acts'][:,-4:]==4).mean((1,2)),
        persistent_disagreement=(np.abs(q[:,0]-q[:,1])>.4).astype(float))


def transition(q,omega,c,gamma,prevacts,currentacts,coupling=True):
    """T[n,old form-pair,new form-pair]=P(new acts/forms | old forms, history)."""
    n=len(q);out=np.empty((n,4,4));idx=np.arange(n)
    for old,state in enumerate(STATES):
        pa=probability(q[:,0],omega,c,gamma,np.full(n,state[1]),prevacts[:,1],0,coupling=coupling)
        pb=probability(q[:,1],omega,c,gamma,np.full(n,state[0]),prevacts[:,0],1,coupling=coupling)
        aa=pa[idx,currentacts[:,0]];bb=pb[idx,currentacts[:,1]]
        for new,f in enumerate(STATES):out[:,old,new]=aa[:,f[0]]*bb[:,f[1]]
    return out


def observe(d,C,mode='actions',hypotheses=HYPOTHESES,coupling=True):
    """Exact external filter on hidden (fact,form pair), within fixed hypotheses.

    Observes fresh evidence and both agents' coarse actions. Depending on mode,
    additionally observes archived-noise form labels or the true simulated form.
    Coarse actions are observed without error; utterance strings are not generated.
    """
    n,T=d['ev'].shape;K=len(hypotheses);idx=np.arange(n)
    bel=np.zeros((K,n,2,4));bel[:,:,:,3]=.5
    logw=np.full((K,n),-np.log(K));m=np.zeros(n,int)
    prevacts=np.zeros((n,2),int)
    nextpred=np.zeros((n,T-1));posterior_history=[];step_logpred=[]
    for t in range(T):
        e=d['ev'][:,t];m+=np.where(e==1,1,np.where(e==0,-1,0))
        rate=np.stack([arrival(np.tile(f,(n,1)),prevacts,coupling=coupling) for f in STATES],axis=1)
        evidence_like=np.empty((n,2,4))
        for x in range(2):
            evidence_like[:,x]=np.where(e[:,None]==2,1-rate,rate*np.where(e[:,None]==x,P['rho'],1-P['rho']))
        for k,(_,om,c,g) in enumerate(hypotheses):
            b=bel[k]*evidence_like
            tr=transition(beliefs(m,om),om,c,g,prevacts,d['acts'][:,t],coupling=coupling)
            nxt=np.einsum('nxs,nsj->nxj',b,tr)
            if mode=='labels':
                z=d['labels'][:,t]
                emit=np.stack([C[f[0],z[:,0]]*C[f[1],z[:,1]] for f in STATES],axis=1)
                nxt*=emit[:,None,:]
            elif mode=='oracle':
                good=(STATES[None,:,:]==d['forms'][:,t,None,:]).all(2)
                nxt*=good[:,None,:]
            elif mode=='null_labels':
                z=d['null_labels'][:,t];f=(d['acts'][:,t]==4).astype(int)
                emit=C[f[:,0],z[:,0]]*C[f[:,1],z[:,1]]
                nxt*=emit[:,None,None]
            normal=nxt.sum((1,2));assert np.all(normal>0)
            bel[k]=nxt/normal[:,None,None];logw[k]+=np.log(normal)
        weights=np.exp(logw-logw.max(0));weights/=weights.sum(0)
        posterior_history.append(weights.T)
        if t<T-1:
            currentrate=np.stack([arrival(np.tile(f,(n,1)),d['acts'][:,t],coupling=coupling) for f in STATES],axis=1)
            perhyp=(bel.sum(2)*currentrate[None,:,:]).sum(2)
            nextpred[:,t]=(weights*perhyp).sum(0)
        prevacts=d['acts'][:,t]
    return dict(posterior=weights.T,next_evidence_probability=nextpred,
                posterior_history=np.stack(posterior_history,1))
