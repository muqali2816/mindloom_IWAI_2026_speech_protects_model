from pathlib import Path
import time,json,itertools
import numpy as np,pandas as pd
import model as M

OUT=M.ROOT/'results';OUT.mkdir(exist_ok=True)
def ci(x):
    x=np.asarray(x);m=float(x.mean());s=float(x.std(ddof=1)/np.sqrt(len(x)))
    return dict(mean=m,lo=m-1.96*s,hi=m+1.96*s)
def main():
    t0=time.time();w=M.make_worlds(M.P['n_dynamics'],M.P['seeds']['dynamics']);runs={};rows=[];trace=[]
    # Initial private data are interpreted identically in every condition.
    qA=M.sigmoid(M.logit(M.P['self_prior'])+(2*w['private'][:,0]-M.P['n_private'])*M.logit(M.P['rho_private']))
    qB=M.sigmoid(-M.logit(M.P['self_prior'])+(2*w['private'][:,1]-M.P['n_private'])*M.logit(M.P['rho_private']))
    masks={'all':np.ones(len(qA),bool),'both_own':(qA>.5)&(qB<.5)}
    conditions=[]
    for beta in [0.,M.P['beta']]:
        for a,b in itertools.product(M.TYPE_NAMES,M.TYPE_NAMES):
            conditions.append((f'{a}__{b}__beta{beta}',a,b,dict(beta=beta)))
    for a,b in [('reference','reference'),('access_protection','access_protection'),('concession_cost','reference')]:
        conditions.append((f'{a}__{b}__verificationIG',a,b,dict(information='verification')))
    for a,b in [('concession_cost','reference'),('access_protection','access_protection')]:
        for kind in ['reassure','sham','silence','remove_cost','remove_gamma','force_open']:
            conditions.append((f'{a}__{b}__{kind}',a,b,dict(intervention={'kind':kind,'event':7,'agents':[0,1]})))
    for a,b in [('reference','reference'),('access_protection','access_protection')]:
        conditions.append((f'{a}__{b}__no_gate',a,b,dict(coupling=False)))
    for name,a,b,kw in conditions:
        start=time.time();d=M.simulate(w,M.TYPES[a],M.TYPES[b],**kw);mm=M.metrics(d);runs[name]=mm
        for split,mask in masks.items():
            row=dict(condition=name,A=a,B=b,split=split,n=int(mask.sum()))
            row.update({key:float(v[mask].mean()) for key,v in mm.items()});rows.append(row)
        for event in range(d['q'].shape[1]):
            qt=d['q'][:,event];ptrue=np.where(w['truth'][:,None]==1,qt,1-qt).mean(1)
            trace.append(dict(condition=name,event=event,p_truth=float(ptrue.mean()),gap=float(np.abs(qt[:,0]-qt[:,1]).mean())))
        print(name,'seconds',round(time.time()-start,2),'truth',round(mm['p_truth'].mean(),4),'evidence',round(mm['evidence_count'].mean(),3),flush=True)
        pd.DataFrame(rows).to_csv(OUT/'dynamics.csv',index=False)
    contrasts=[]
    comparisons=[('access_protection__access_protection__beta1.0','reference__reference__beta1.0'),
        ('access_protection__access_protection__no_gate','access_protection__access_protection__beta1.0'),
        ('access_protection__access_protection__beta1.0','access_protection__access_protection__beta0.0'),
        ('reference__reference__beta1.0','reference__reference__beta0.0')]
    for a,b in [('concession_cost','reference'),('access_protection','access_protection')]:
        prefix=f'{a}__{b}__'
        comparisons.extend([(prefix+'reassure',prefix+'silence'),(prefix+'reassure',prefix+'sham'),
                            (prefix+'remove_cost',prefix+'beta1.0'),(prefix+'remove_gamma',prefix+'beta1.0'),
                            (prefix+'force_open',prefix+'beta1.0')])
    for a,b in comparisons:
        for split,mask in masks.items():
            for met in runs[a]:contrasts.append(dict(contrast=a+' minus '+b,split=split,metric=met,n=int(mask.sum()),**ci((runs[a][met]-runs[b][met])[mask])))
    pd.DataFrame(contrasts).to_csv(OUT/'dynamics_contrasts.csv',index=False)
    pd.DataFrame(trace).to_csv(OUT/'trajectories.csv',index=False)
    sens=[]
    base_rho=M.P['rho_verification']
    for rho in [.65,.85]:
        M.P['rho_verification']=rho
        for name in ['reference','access_protection']:
            mm=M.metrics(M.simulate(w,M.TYPES[name],M.TYPES[name]))
            sens.append(dict(rho=rho,condition=name,**{k:float(v.mean()) for k,v in mm.items()}))
    M.P['rho_verification']=base_rho
    pd.DataFrame(sens).to_csv(OUT/'rho_sensitivity.csv',index=False)
    (OUT/'dynamics_manifest.json').write_text(json.dumps(dict(n=len(qA),both_own=int(masks['both_own'].sum()),
        conditions=len(conditions),rho_sensitivity_conditions=len(sens),elapsed=time.time()-t0),indent=2))
if __name__=='__main__':main()
