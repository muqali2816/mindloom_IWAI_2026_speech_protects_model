"""Hybrid dyad: joint act/form choices, reciprocal Bayesian inference, gated verification.

Each event: one speaker acts; partner observes (act, form); a new verification
signal may arrive and is observed once by both; roles alternate. Level-1 agents
model a partner's next response as level-0. Planning spans their immediate
verification outcome and the partner's subsequent response, before the next
verification outcome. This is bounded planning, not a recursive equilibrium.
"""
from pathlib import Path
import json,itertools
from math import comb
import numpy as np

ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())
TYPES=P['types']; TYPE_NAMES=list(TYPES)
GRID=np.array(list(itertools.product(P['partner_grid']['omega'],P['partner_grid']['c'],P['partner_grid']['gamma'])),float)
ACTS=['STATE_0','STATE_1','ASK','SILENCE','REASSURE']
STATE_0,STATE_1,ASK,SILENCE,REASSURE=range(5)
Y_ACT=np.repeat(np.arange(5),2);Y_FORM=np.tile([0,1],5)
LABELS=['BUILD','SEEK','UNSEAL','LOCK','DRAIN','FLOOD','EDGE','VOID','SEAL','SHIFT','INSUFF']
NONE=2

def sigmoid(x):return 1/(1+np.exp(-np.clip(np.asarray(x),-700,700)))
def logit(p):
    p=np.clip(p,1e-14,1-1e-14);return np.log(p)-np.log1p(-p)
def softmax(x):
    e=np.exp(x-x.max(-1,keepdims=True));return e/e.sum(-1,keepdims=True)
def entropy(x,axis=-1):return -(x*np.log(np.clip(x,1e-300,None))).sum(axis)
def tempered_rho(omega,rho=None):
    return sigmoid(np.asarray(omega)*logit(P['rho_verification'] if rho is None else rho))
def private_likelihood():
    n=P['n_private'];r=P['rho_private'];s=np.arange(n+1)
    return np.array([[comb(n,int(k))*(r if x else 1-r)**k*(1-r if x else r)**(n-k) for k in s] for x in range(2)])

def arrival(y,partner_y,coupling=True):
    y=np.asarray(y);partner_y=np.asarray(partner_y)
    base=np.where((y//2==ASK)|(partner_y//2==ASK),P['lambda_ask'],P['lambda_base'])
    if not coupling:return base
    return P['lambda_floor']+(base-P['lambda_floor'])*(y%2)*(partner_y%2)

def risk_policy(q,theta,pos,partner_y,relief,coupling=True):
    """Broadcasted cost-only joint-act/form policy. q is probability x=1.
    pos is current public position; a factual action can switch its anchor.
    """
    q=np.asarray(q);theta=np.asarray(theta);pos=np.asarray(pos)
    shape=np.broadcast_shapes(q.shape,theta.shape[:-1],pos.shape,np.shape(partner_y),np.shape(relief))
    q=np.broadcast_to(q,shape);th=np.broadcast_to(theta,shape+(3,));pos=np.broadcast_to(pos,shape)
    old=np.broadcast_to(partner_y,shape);safe=np.broadcast_to(relief,shape)
    acts=Y_ACT;forms=Y_FORM
    factual=acts<2
    declared=np.where(factual,acts,pos[...,None])
    p_declared=np.where(declared==1,q[...,None],1-q[...,None])
    c_eff=th[...,1]*(1-safe)
    loss=np.where(factual,P['misrepresentation_cost']*(1-p_declared)+c_eff[...,None]*(declared!=pos[...,None]),P['nonfactual_cost'])
    lam=arrival(np.arange(10),old[...,None],coupling)
    rr=tempered_rho(th[...,0])
    p_contradiction=p_declared*(1-rr[...,None])+(1-p_declared)*rr[...,None]
    risk=loss+P['restriction_cost']*(1-forms)+th[...,2,None]*lam*p_contradiction
    return softmax(-risk/P['tau']),risk,lam

class Agent:
    def __init__(self,i,s_own,theta,beta=None,coupling=True):
        self.i=i;self.n=len(s_own);self.theta=np.broadcast_to(np.asarray(theta,float),(self.n,3)).copy()
        self.beta=P['beta'] if beta is None else beta;self.coupling=coupling
        n=self.n;ns=P['n_private']+1;nt=len(GRID)
        prior=logit(P['self_prior'])*(1 if i==0 else -1)
        q0=sigmoid(prior+(2*np.asarray(s_own)-P['n_private'])*logit(P['rho_private']))
        px=np.stack([1-q0,q0],1)
        self.J=px[:,:,None,None]*private_likelihood()[None,:,:,None]*np.ones((1,1,1,nt))/nt
        self.sent=np.zeros(n)             # signed factual messages sent; never treated as new private records
        self.shared=np.zeros(n)           # signed, already observed common verification signals
        self.pos=np.full(n,1-i,int);self.partner_pos=np.full(n,i,int)
        self.last_own=np.full(n,2*SILENCE+1,int);self.last_partner=self.last_own.copy()
        self.relief=np.zeros(n);self.granted_relief=np.zeros(n)
    def take(self,idx):
        out=object.__new__(Agent)
        for k,v in self.__dict__.items():setattr(out,k,v[idx].copy() if isinstance(v,np.ndarray) and len(v)==self.n else v)
        out.n=len(idx);return out
    def q(self):return self.J[:,1].sum((1,2))
    def partner_q(self,sent=None,shared=None):
        sent=self.sent if sent is None else sent;shared=self.shared if shared is None else shared
        prior=logit(P['self_prior'])*(-1 if self.i==0 else 1)
        private=(2*np.arange(P['n_private']+1)-P['n_private'])*logit(P['rho_private'])
        return sigmoid(prior+private[None,:,None]+np.asarray(shared)[:,None,None]*logit(tempered_rho(GRID[:,0]))[None,None,:]
            +np.asarray(sent)[:,None,None]*GRID[None,None,:,0]*logit(P['trusted_message_precision']))
    def partner_probs(self,sent=None,shared=None,partner_last_y=None,relief=None):
        q=self.partner_q(sent,shared)
        old=self.last_own if partner_last_y is None else partner_last_y
        safe=self.granted_relief if relief is None else relief
        return risk_policy(q,GRID[None,None,:,:],self.partner_pos[:,None,None],
                           np.asarray(old)[:,None,None],np.asarray(safe)[:,None,None],self.coupling)[0]
    def policy(self,information='joint'):
        _,risk,lam=risk_policy(self.q(),self.theta,self.pos,self.last_partner,self.relief,self.coupling)
        ig=np.zeros_like(risk)
        if self.beta>0:
            rho=tempered_rho(self.theta[:,0]);q=self.q()
            if information=='verification':
                mix=q*rho+(1-q)*(1-rho)
                binary_mi=entropy(np.stack([mix,1-mix],1))-entropy(np.stack([rho,1-rho],1))
                ig=lam*binary_mi[:,None]
            else:
                for y in range(10):
                    a=y//2;rate=lam[:,y]
                    sent=self.sent+(1 if a==1 else -1 if a==0 else 0)
                    hjoint=np.zeros(self.n);expected_response_entropy=np.zeros(self.n)
                    for e in [0,1,NONE]:
                        likelihood=np.stack([rate*np.where(e==x,rho,1-rho) for x in [0,1]],1) if e!=NONE else np.repeat((1-rate)[:,None],2,1)
                        weights=(self.J*likelihood[:,:,None,None]).sum(1)
                        pp=self.partner_probs(sent,self.shared+(1 if e==1 else -1 if e==0 else 0),np.full(self.n,y),np.full(self.n,a==REASSURE))
                        pred=(weights[:,:,:,None]*pp).sum((1,2))
                        hjoint+=entropy(pred)
                        expected_response_entropy+=(weights*entropy(pp)).sum((1,2))
                    conditional_e=entropy(np.stack([1-rate,rate*rho,rate*(1-rho)],1))
                    ig[:,y]=hjoint-conditional_e-expected_response_entropy
        return softmax((-risk+self.beta*ig)/P['tau']),ig
    def observe_move(self,y):
        pp=self.partner_probs();y=np.asarray(y,int)
        like=np.take_along_axis(pp,y[:,None,None,None],axis=3)[:,:,:,0]
        self.J*=like[:,None,:,:];self.J/=self.J.sum((1,2,3),keepdims=True)
        act=y//2;self.partner_pos=np.where(act<2,act,self.partner_pos)
        self.last_partner=y.copy()
        self.relief=(act==REASSURE).astype(float)
    def own_move(self,y,effective_reassurance=True):
        y=np.asarray(y,int);a=y//2
        self.sent+=(a==1).astype(float)-(a==0).astype(float)
        self.pos=np.where(a<2,a,self.pos);self.last_own=y.copy()
        self.granted_relief=((a==REASSURE)&effective_reassurance).astype(float)
        self.relief[:]=0
    def observe_evidence(self,e):
        e=np.asarray(e);rho=tempered_rho(self.theta[:,0])
        like=np.stack([np.where(e==NONE,1.,np.where(e==x,rho,1-rho)) for x in [0,1]],1)
        self.J*=like[:,:,None,None];self.J/=self.J.sum((1,2,3),keepdims=True)
        self.shared+=(e==1).astype(float)-(e==0).astype(float)
    def partner_posterior(self):return self.J.sum((1,2))

def make_worlds(n,seed,rounds=None):
    r=np.random.default_rng(seed);rounds=P['n_rounds'] if rounds is None else rounds
    x=r.integers(0,2,n);u=r.random((n,2,P['n_private']))
    private=np.where(u<P['rho_private'],x[:,None,None],1-x[:,None,None])
    return dict(truth=x,private=private.sum(2),act_u=r.random((n,2*rounds)),arrival_u=r.random((n,2*rounds)),
        signal_u=r.random((n,2*rounds)),label_u=r.random((n,2*rounds)),channel_seed=seed+77)

def simulate(w,theta_A,theta_B,beta=None,coupling=True,information='joint',intervention=None,order='AB'):
    n,T=w['act_u'].shape;agents=[Agent(0,w['private'][:,0],theta_A,beta,coupling),Agent(1,w['private'][:,1],theta_B,beta,coupling)]
    y=np.empty((n,T),int);e=np.full((n,T),NONE,int);q=np.empty((n,T+1,2));pos=np.empty((n,T+1,2),int)
    probabilities=np.empty((n,T,10));ig=np.empty((n,T,10));rate=np.empty((n,T));speaker=np.array([(t%2 if order=='AB' else 1-t%2) for t in range(T)])
    q[:,0]=np.stack([a.q() for a in agents],1);pos[:,0]=np.stack([a.pos for a in agents],1)
    intervention=intervention or {};first=intervention.get('event',T+1)
    for t,i in enumerate(speaker):
        ag=agents[i];other=agents[1-i]
        if t>=first and intervention.get('kind')=='remove_cost':
            for who in intervention.get('agents',[0,1]):agents[who].theta[:,1]=0
        if t>=first and intervention.get('kind')=='remove_gamma':
            for who in intervention.get('agents',[0,1]):agents[who].theta[:,2]=0
        pr,g=ag.policy(information);chosen=(pr.cumsum(1)<w['act_u'][:,t,None]).sum(1).clip(0,9)
        if t==first and intervention.get('kind') in ('reassure','sham','silence'):
            chosen=np.full(n,2*(SILENCE if intervention['kind']=='silence' else REASSURE)+1)
        if t>=first and intervention.get('kind')=='force_open' and i in intervention.get('agents',[0,1]):chosen=(chosen//2)*2+1
        is_sham=t==first and intervention.get('kind')=='sham'
        lam=arrival(chosen,ag.last_partner,coupling)
        if t>=first and intervention.get('kind')=='bypass':lam=np.full(n,.9)
        other.observe_move(chosen);ag.own_move(chosen,effective_reassurance=not is_sham)
        if is_sham:other.relief[:]=0  # local sham only; both agents know no relief was granted
        got=w['arrival_u'][:,t]<lam
        sign=np.where(w['signal_u'][:,t]<P['rho_verification'],w['truth'],1-w['truth'])
        ev=np.where(got,sign,NONE)
        for a in agents:a.observe_evidence(ev)
        y[:,t]=chosen;e[:,t]=ev;rate[:,t]=lam;probabilities[:,t]=pr;ig[:,t]=g
        q[:,t+1]=np.stack([a.q() for a in agents],1);pos[:,t+1]=np.stack([a.pos for a in agents],1)
    return dict(y=y,e=e,q=q,pos=pos,prob=probabilities,ig=ig,rate=rate,truth=w['truth'],private=w['private'],
                theta_A=np.array(theta_A),theta_B=np.array(theta_B),speaker=speaker,
                partner_posterior=np.stack([a.partner_posterior() for a in agents],1),coupling=coupling,beta=P['beta'] if beta is None else beta)

def metrics(d):
    truth=d['truth'][:,None];q=d['q'];n,T=d['y'].shape
    truth_prob=np.where(truth==1,q[:,-1],1-q[:,-1]).mean(1)
    truth_initial=np.where(truth==1,q[:,0],1-q[:,0]).mean(1)
    oldpos=np.stack([d['pos'][:,t,int(i)] for t,i in enumerate(d['speaker'])],1)
    a=d['y']//2;sw=(a<2)&(a!=oldpos)
    return dict(p_truth=truth_prob,learning_gain=truth_prob-truth_initial,evidence_count=(d['e']!=NONE).sum(1),
        final_gap=np.abs(q[:,-1,0]-q[:,-1,1]),mean_abs_belief_change=np.abs(q[:,-1]-q[:,0]).mean(1),
        restriction_rate=(d['y']%2==0).mean(1),late_non_switch=(~sw[:,-8:]).mean(1),
        public_disagreement=(d['pos'][:,-1,0]!=d['pos'][:,-1,1]).astype(float),
        reassure_rate=(a==REASSURE).mean(1))
