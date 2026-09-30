"""Reporting contrasts; explicit post-run checks, never parameter tuning."""
import json
import numpy as np,pandas as pd
import model as M
import channel as C
from run_dynamics import ci
R=M.ROOT/'results'

def additional_dynamics():
    (R/'calibration_prior_sensitivity.json').write_text(json.dumps({str(a):C.sample_information(C.calibration(a)) for a in [.1,.5,1.]},indent=2))
    w=M.make_worlds(M.P['n_dynamics'],M.P['seeds']['dynamics']);raw={}
    for name,beta in [('reference',0),('reference',1),('access_protection',0),('access_protection',1),('concession_cost',1)]:
        met=M.metrics(M.simulate(w,M.TYPES[name],M.TYPES[name],beta=beta))
        for k,v in met.items():raw[f'{name}__beta{beta}__{k}']=v
    np.savez_compressed(R/'selected_dynamics_worlds.npz',**raw)
    rows=[]
    for beta in [0,1]:
        for metric in ['p_truth','evidence_count','final_gap','public_disagreement']:
            x=raw[f'access_protection__beta{beta}__{metric}']-raw[f'reference__beta{beta}__{metric}']
            rows.append(dict(contrast=f'gamma minus reference beta={beta}',metric=metric,n=len(x),**ci(x)))
    for metric in ['p_truth','evidence_count','final_gap','public_disagreement']:
        x=raw[f'concession_cost__beta1__{metric}']-raw[f'reference__beta1__{metric}']
        rows.append(dict(contrast='cost minus reference beta=1',metric=metric,n=len(x),**ci(x)))
    pd.DataFrame(rows).to_csv(R/'additional_dynamics_contrasts.csv',index=False)

def observer_summary(sensitivity=True):
    d=pd.read_csv(R/'observer_worlds.csv');n=d.world.nunique()
    assert len(d)==3*M.P['n_observer'], 'Wait for complete main observer run'
    rows=[]
    for typ in ['all']+M.TYPE_NAMES:
        s=d if typ=='all' else d[d.generator==typ]
        for met in ['arrival_logloss','act_logloss','posterior_true_A','correct_A','correct_B']:
            piv=s.pivot(index='world',columns='mode',values=met)
            for channel in ['labels_evidence','forms_evidence']:
                # Positive is improvement, with sign depending on metric.
                x=(piv['actions_evidence']-piv[channel]) if 'logloss' in met else (piv[channel]-piv['actions_evidence'])
                rows.append(dict(generator=typ,metric=met,comparison=channel+' over actions_evidence',n=len(x),**ci(x)))
    pd.DataFrame(rows).to_csv(R/'observer_contrasts.csv',index=False)
    d.groupby('mode').mean(numeric_only=True).reset_index().to_csv(R/'observer_overall.csv',index=False)
    raw=np.load(R/'observer_outputs.npz');types=list(M.TYPES)
    conf=[]
    for mode in ['actions_evidence','labels_evidence','forms_evidence']:
        mat=np.zeros((4,4),int)
        for i in range(n):mat[i%4,raw[f'{i}__{mode}__posterior'].sum(1).argmax()]+=1
        for i,a in enumerate(types):
            for j,b in enumerate(types):conf.append(dict(mode=mode,true_A=a,inferred_A=b,n=int(mat[i,j])))
    pd.DataFrame(conf).to_csv(R/'observer_confusion.csv',index=False)
    if not sensitivity:return
    s=pd.read_csv(R/'particle_sensitivity.csv')
    assert len(s)==M.P['observer_sensitivity_worlds']*4, 'Wait for sensitivity completion'
    sc=[]
    for mode in ['actions_evidence','labels_evidence']:
        base=d[(d['mode']==mode)&(d.world<M.P['observer_sensitivity_worlds'])].set_index('world')
        for k in [4,32]:
            sub=s[(s['mode']==mode)&(s.paths==k)].set_index('world').sort_index()
            err=sub.arrival_logloss-base.arrival_logloss
            sc.append(dict(mode=mode,paths=k,n=len(sub),mean_arrival_logloss=sub.arrival_logloss.mean(),
                mean_delta_from_K8=err.mean(),mean_abs_delta_from_K8=abs(err).mean(),
                max_posterior_component_difference=sub.max_posterior_difference_from_main.max(),
                median_max_posterior_component_difference=sub.max_posterior_difference_from_main.median(),
                correct_A=sub.correct_A.mean(),main_correct_A=base.correct_A.mean()))
    pd.DataFrame(sc).to_csv(R/'particle_sensitivity_summary.csv',index=False)
    print(pd.read_csv(R/'observer_overall.csv').to_string(index=False))
    print(pd.read_csv(R/'particle_sensitivity_summary.csv').to_string(index=False))

if __name__=='__main__':
    import sys
    if '--additional-dynamics' in sys.argv:additional_dynamics()
    if '--observer' in sys.argv:observer_summary()
    if '--main-observer' in sys.argv:observer_summary(sensitivity=False)
