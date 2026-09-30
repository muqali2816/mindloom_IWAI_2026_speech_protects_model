"""Independent scalar policy + brute-force short-path observer validation."""
from pathlib import Path
import itertools,json
import numpy as np
import dyad_model as M

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
C=np.array(json.loads((OUT/'annotation_channel.json').read_text())['primary'])
P=M.P

def scalar_policy(q,omega,c,g,partner_form,partner_act,i):
    own=q if i==0 else 1-q
    rw=P['rho']**omega/(P['rho']**omega+(1-P['rho'])**omega)
    probs=[]
    for u in range(5):
        for f in range(2):
            base=P['lam_ask'] if u==4 or partner_act==4 else P['lam_low']
            lam=P['lam_floor']+(base-P['lam_floor'])*f*partner_form
            A=np.array([[lam*rw,lam*(1-rw),1-lam],[lam*(1-rw),lam*rw,1-lam]])
            Q=np.array([1-q,q]);mi=0.
            for x in range(2):
                for e in range(3):
                    if A[x,e]>0:
                        mi+=Q[x]*A[x,e]*np.log(A[x,e]/(Q@A[:,e]))
            risk=[P['k']*(1-own),P['k']*own+c,P['b'],P['b'],P['b']][u]
            public_disconfirm=lam*(own*(1-rw)+(1-own)*rw)
            G=risk+P['restriction_cost']*(1-f)+g*public_disconfirm-mi
            probs.append(np.exp(-G/P['tau']))
    probs=np.array(probs).reshape(5,2);return probs/probs.sum()

checks={};err=0.
for q in [.03,.25,.5,.75,.97]:
    for _,om,c,g in M.HYPOTHESES:
        for pf,pu,i in itertools.product([0,1],[0,4],[0,1]):
            a=scalar_policy(q,om,c,g,pf,pu,i)
            b=M.probability(np.array([q]),om,c,g,np.array([pf]),np.array([pu]),i)[0]
            err=max(err,float(np.abs(a-b).max()))
assert err<1e-12;checks['scalar_policy_max_error']=err

# Explicit enumeration of (hidden truth, 4^3 form paths), no HMM recursion.
w=M.worlds(2,202609304);d=M.simulate(w,gamma=2,C=C)
short={k:v[:,:3] if isinstance(v,np.ndarray) and v.ndim>=2 and v.shape[1]==P['rounds'] else v for k,v in d.items()}
for mode in ['actions','labels']:
    weights=[]
    for _,om,c,g in M.HYPOTHESES:
        per=[]
        for n in range(2):
            total=0.
            for x in [0,1]:
                for path in itertools.product(range(4),repeat=3):
                    mass=.5;prevf=np.array([1,1]);prevu=np.array([0,0]);m=0
                    for t,sg in enumerate(path):
                        base=P['lam_ask'] if 4 in prevu else P['lam_low']
                        lam=P['lam_floor']+(base-P['lam_floor'])*prevf.prod()
                        e=int(short['ev'][n,t]);mass*=1-lam if e==2 else lam*(P['rho'] if e==x else 1-P['rho'])
                        m+=0 if e==2 else 2*e-1
                        qq=1/(1+np.exp(-(np.array([np.log(3),-np.log(3)])+m*om*np.log(3))))
                        newf=M.STATES[sg];newu=short['acts'][n,t]
                        for i in [0,1]:
                            prob=scalar_policy(qq[i],om,c,g,prevf[1-i],prevu[1-i],i)
                            mass*=prob[newu[i],newf[i]]
                            if mode=='labels':mass*=C[newf[i],short['labels'][n,t,i]]
                        prevf=newf;prevu=newu
                    total+=mass
            per.append(total)
        weights.append(per)
    weights=np.array(weights).T;weights/=weights.sum(1,keepdims=True)
    out=M.observe(short,C,mode)['posterior'];er=float(np.abs(weights-out).max())
    assert er<1e-12;checks['brute_paths_'+mode+'_error']=er

# If forms no longer affect access, labels carry no extra mechanism information.
for name,om,c,g in M.HYPOTHESES:
    data=M.simulate(M.worlds(300,202609305),omega=om,c=c,gamma=g,C=C,coupling=False)
    a=M.observe(data,C,'actions',coupling=False)
    b=M.observe(data,C,'labels',coupling=False)
    er=float(np.abs(a['posterior']-b['posterior']).max())
    pr=float(np.abs(a['next_evidence_probability']-b['next_evidence_probability']).max())
    assert er<1e-12 and pr<1e-12
    checks['no_coupling_'+name+'_posterior_error']=er
    checks['no_coupling_'+name+'_prediction_error']=pr
(OUT/'independent_validation.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
