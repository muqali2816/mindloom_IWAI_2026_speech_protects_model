"""Independent scalar arithmetic, causal controls, and short hidden-path enumeration."""
import json,time,math
import numpy as np
from scipy.special import gammaln
import model as M, observer as O, channel as C

def scalar_policy(ag):
    """Independent scalar construction of P(E,response|x,s,theta,own move). n=1."""
    p=M.P;N=p['n_private'];theta=ag.theta[0];q=float(ag.J[0,1].sum())
    oldpos=int(ag.pos[0]);otherpos=int(ag.partner_pos[0]);old=int(ag.last_partner[0]);grid=M.GRID
    sg=lambda v:1/(1+math.exp(-v));lg=lambda v:math.log(v/(1-v))
    rho=lambda om:sg(om*lg(p['rho_verification']))
    def rate(y,other):
        base=p['lambda_ask'] if (y//2==2 or other//2==2) else p['lambda_base']
        return p['lambda_floor']+(base-p['lambda_floor'])*(y%2)*(other%2) if ag.coupling else base
    def costs(q,th,pos,partner,relief):
        out=[]
        for y in range(10):
            a,f=divmod(y,2);decl=a if a<2 else pos;pv=q if decl==1 else 1-q
            ordinary=p['misrepresentation_cost']*(1-pv)+th[1]*(1-relief)*(decl!=pos) if a<2 else p['nonfactual_cost']
            dc=pv*(1-rho(th[0]))+(1-pv)*rho(th[0])
            out.append(ordinary+p['restriction_cost']*(1-f)+th[2]*rate(y,partner)*dc)
        return out
    def soft(z):
        mx=max(z);e=[math.exp(v-mx) for v in z];s=sum(e);return [v/s for v in e]
    J=ag.J[0].reshape(-1);latent=list(np.ndindex(2,N+1,len(grid)));infos=[]
    for y in range(10):
        a=y//2;lam=rate(y,old);like=np.zeros((len(latent),30))
        sent=float(ag.sent[0])+(1 if a==1 else -1 if a==0 else 0)
        for li,(x,s,k) in enumerate(latent):
            th=grid[k]
            for e in range(3):
                pe=(1-lam) if e==2 else lam*(rho(theta[0]) if e==x else 1-rho(theta[0]))
                shared=float(ag.shared[0])+(1 if e==1 else -1 if e==0 else 0)
                prior=lg(p['self_prior'])*(-1 if ag.i==0 else 1)
                qb=sg(prior+(2*s-N)*lg(p['rho_private'])+shared*lg(rho(th[0]))+sent*th[0]*lg(p['trusted_message_precision']))
                pp=soft([-v/p['tau'] for v in costs(qb,th,otherpos,y,a==4)])
                for b in range(10):like[li,e*10+b]=pe*pp[b]
        pred=J@like
        mi=0.
        for li in range(len(J)):
            for o in range(30):
                if like[li,o]>0:mi+=J[li]*like[li,o]*math.log(like[li,o]/pred[o])
        infos.append(mi)
    rr=costs(q,theta,oldpos,old,float(ag.relief[0]))
    return np.array(soft([(-v+ag.beta*g)/p['tau'] for v,g in zip(rr,infos)])),np.array(infos)

def main():
    t0=time.time();checks={};details={};maxp=maxg=0.
    for i,sc,typ in [(0,0,'reference'),(0,1,'attenuated'),(1,2,'concession_cost'),(0,2,'access_protection')]:
        ag=M.Agent(i,np.array([sc]),M.TYPES[typ])
        for step in range(3):
            p,g=ag.policy();sp,sg=scalar_policy(ag)
            maxp=max(maxp,float(np.abs(p[0]-sp).max()));maxg=max(maxg,float(np.abs(g[0]-sg).max()))
            ag.observe_move(np.array([9 if step==0 else 2]));ag.own_move(np.array([2 if i==0 else 1]));ag.observe_evidence(np.array([step%2]))
    checks['scalar_joint_policy']=maxp<1e-11 and maxg<1e-11
    details['scalar_policy_max_error']=maxp;details['scalar_MI_max_error']=maxg
    # Bayesian posterior averaging: all possible partner acts are exhaustive and normalized.
    ag=M.Agent(0,np.array([1]),M.TYPES['attenuated']);J0=ag.J.copy();pr=ag.partner_probs()[0]
    probs=(J0[0].sum(0)[...,None]*pr).sum((0,1));expected=np.zeros_like(J0)
    for y in range(10):
        cp=ag.take(np.array([0]));cp.observe_move(np.array([y]));expected+=probs[y]*cp.J
    details['Bayes_martingale_error']=float(np.abs(expected-J0).max());checks['Bayes_martingale']=details['Bayes_martingale_error']<1e-12
    # No hard-coded log(3): scalar consistency when physical reliability changes.
    oldrho=M.P['rho_verification'];M.P['rho_verification']=.65
    ag=M.Agent(0,np.array([1]),M.TYPES['attenuated']);pv,gv=ag.policy();ps,gs=scalar_policy(ag)
    checks['rho_parameterization']=bool(np.max(np.abs(pv[0]-ps))<1e-11 and np.max(np.abs(gv[0]-gs))<1e-11)
    M.P['rho_verification']=oldrho
    # Mirror a forced public history, changing both truth signs and speaker order.
    n=16;rng=np.random.default_rng(182);s=rng.integers(0,3,(n,2));ta=M.TYPES['access_protection'];tb=M.TYPES['concession_cost']
    A=[M.Agent(0,s[:,0],ta),M.Agent(1,s[:,1],tb)]
    B=[M.Agent(0,2-s[:,1],tb),M.Agent(1,2-s[:,0],ta)]
    mapping=np.array([2,3,0,1,4,5,6,7,8,9]);mirrorerr=0.
    for t in range(8):
        i=t%2;pa,_=A[i].policy();pb,_=B[1-i].policy();mirrorerr=max(mirrorerr,float(np.abs(pa-pb[:,mapping]).max()))
        y=rng.integers(0,10,n);ym=mapping[y]
        A[1-i].observe_move(y);A[i].own_move(y);B[i].observe_move(ym);B[1-i].own_move(ym)
        e=rng.integers(0,3,n);em=np.where(e==2,2,1-e)
        for aa in A:aa.observe_evidence(e)
        for bb in B:bb.observe_evidence(em)
        mirrorerr=max(mirrorerr,float(np.abs(A[0].q()+B[1].q()-1).max()))
    details['mirror_max_error']=mirrorerr;checks['role_symmetry']=mirrorerr<1e-10
    # Factual evidence coming only through partner acts cannot exceed the partner's finite private-record likelihood ratio.
    w=M.make_worlds(100,M.P['seeds']['validation']);d=M.simulate(w,ta,tb)
    q=d['q'];cum=np.cumsum(np.where(d['e']==2,0,2*d['e']-1),axis=1)
    theta=np.array([ta,tb]);inferred=M.logit(q[:,1:])-M.logit(q[:,0,None,:])-cum[:,:,None]*M.logit(M.tempered_rho(theta[:,0]))
    bound=M.P['n_private']*M.logit(M.P['rho_private'])
    details['speech_logBF_max']=float(np.abs(inferred).max());details['speech_logBF_bound']=bound
    checks['no_private_record_double_count']=bool(np.abs(inferred).max()<bound+1e-8)
    checks['normalization_and_MI']=bool(np.abs(d['prob'].sum(2)-1).max()<1e-12 and d['ig'].min()>-1e-10)
    # Position now participates causally in a switching cost.
    ag=M.Agent(0,np.array([1]),tb);p0,_=ag.policy();ag.pos[:]=0;p1,_=ag.policy()
    details['position_policy_effect']=float(np.abs(p0-p1).max());checks['position_is_causal']=details['position_policy_effect']>.01
    # Local sham has no effect on pre-intervention history.
    w=M.make_worlds(100,143);re=M.simulate(w,tb,M.TYPES['reference'],intervention={'kind':'reassure','event':7})
    sh=M.simulate(w,tb,M.TYPES['reference'],intervention={'kind':'sham','event':7})
    checks['local_placebo_pre_history']=bool(np.array_equal(re['y'][:,:7],sh['y'][:,:7]) and np.array_equal(re['q'][:,:8],sh['q'][:,:8]))
    # Disable the gate: conditional form choice and partner-form evidence lose their causal roles.
    ag=M.Agent(0,np.array([1]),ta,coupling=False);pr,_=ag.policy()
    conditional=pr.reshape(1,5,2);conditional/=conditional.sum(2,keepdims=True)
    details['gate_off_conditional_form_difference']=float(np.abs(conditional-conditional[:,:1]).max())
    a0=ag.take(np.array([0]));a1=ag.take(np.array([0]));a0.observe_move(np.array([0]));a1.observe_move(np.array([1]))
    details['gate_off_form_posterior_difference']=float(np.abs(a0.J-a1.J).max())
    checks['gate_off_form_null']=details['gate_off_conditional_form_difference']<1e-11 and details['gate_off_form_posterior_difference']<1e-11
    # Dirichlet collapse: sequential predictive product equals integrated likelihood.
    alpha=C.calibration();r=np.random.default_rng(29);counts=np.zeros_like(alpha);seq=0.
    for _ in range(25):
        f=int(r.integers(2));z=int(r.integers(11));seq+=np.log((alpha[f,z]+counts[f,z])/(alpha[f].sum()+counts[f].sum()));counts[f,z]+=1
    integrated=(gammaln(alpha.sum(1))-gammaln((alpha+counts).sum(1))+ (gammaln(alpha+counts)-gammaln(alpha)).sum(1)).sum()
    details['Dirichlet_identity_error']=float(abs(seq-integrated));checks['Dirichlet_identity']=abs(seq-integrated)<1e-11
    # Exact enumeration of 2-event hidden-form paths, versus the stratified sampler.
    w=M.make_worlds(1,712,rounds=1);d=M.simulate(w,M.TYPES['attenuated'],M.TYPES['reference']);z,_=C.emit(d,992)
    ex=O.recover_one(d,0,'labels_evidence',exact_paths=True,labels=z)
    ap=O.recover_one(d,0,'labels_evidence',paths=128,seed=188,labels=z)
    details['short_exact_paths']=ex['n_final_paths'];details['short_particle_posterior_max_error']=float(np.abs(ex['posterior']-ap['posterior']).max())
    details['short_particle_forecast_max_error']=float(np.abs(ex['arrival_probability']-ap['arrival_probability']).max())
    checks['short_particle_agreement']=details['short_particle_posterior_max_error']<.02 and details['short_particle_forecast_max_error']<.02
    # An explicitly uninformative label channel contributes no likelihood ratio.
    aa=O.recover_one(d,0,'actions_evidence',paths=8,seed=3)
    zz=O.recover_one(d,0,'labels_evidence',paths=8,seed=3,labels=z,label_null=True)
    checks['uninformative_label_null']=bool(np.array_equal(aa['posterior'],zz['posterior']) and np.array_equal(aa['arrival_probability'],zz['arrival_probability']))
    # Oracle-form arrival forecast is exactly the known, gate-defined rate.
    ff=O.recover_one(d,0,'forms_evidence');details['oracle_rate_error']=float(np.abs(ff['arrival_probability']-d['rate'][0]).max())
    checks['oracle_arrival']=details['oracle_rate_error']<1e-12
    out=dict(checks={k:bool(v) for k,v in checks.items()},details=details,all_passed=bool(all(checks.values())),elapsed=time.time()-t0)
    (M.ROOT/'results/independent_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True)
    assert out['all_passed']
if __name__=='__main__':main()
