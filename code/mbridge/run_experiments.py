"""Run exploratory model checks; all outcomes retained, no result-based tuning."""
from pathlib import Path
import hashlib,importlib.util,json,platform,time
import numpy as np
import pandas as pd
import dyad_model as M

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)

def save(name,rows):
    x=pd.DataFrame(rows);x.to_csv(OUT/name,index=False);return x

def ci(a):
    a=np.asarray(a,float);m=float(a.mean());se=float(a.std(ddof=1)/np.sqrt(len(a)))
    return dict(mean=m,lo=m-1.96*se,hi=m+1.96*se)

def channel():
    spec=importlib.util.spec_from_file_location('archived_scoring',ROOT/'source/baseline_compare.py')
    mod=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    metric=mod.compute(str(ROOT/'source/engine_md3_combined.json'),'en',True)
    assert metric.strict_tot==169 and metric.strict_ok==129
    labels=sorted(set(x for pair in metric.confusion for x in pair))
    gold=sorted(set(a for a,b in metric.confusion))
    counts=np.array([[metric.confusion[a,b] for b in labels] for a in gold])
    save('archive_confusion_counts.csv',[dict(gold=a,predicted=b,count=int(counts[i,j])) for i,a in enumerate(gold) for j,b in enumerate(labels)])
    two=counts[[gold.index('regime.LOCK'),gold.index('regime.SEEK')]].astype(float)
    empirical=two/two.sum(1,keepdims=True)
    smooth=(two+.5)/(two.sum(1,keepdims=True)+.5*len(labels))
    background=smooth.mean(0,keepdims=True)
    primary=.75*smooth+.25*background
    meta=dict(labels=labels,gold=gold,counts=counts.tolist(),restriction_row='regime.LOCK',invite_row='regime.SEEK',
        empirical=empirical.tolist(),smoothed=smooth.tolist(),primary=primary.tolist(),
        smoothing_pseudocount=.5,extra_state_independent_noise=.25,
        warnings=['Only 15 LOCK and 20 SEEK reference cases support these rows.',
                  '169 cases include SHIFT; they are not all Tier-A feature annotations.',
                  'The two empirical rows have disjoint output supports in this archive; raw plug-in noise would identify these two forms perfectly.',
                  'Transport to synthetic dyads is an assumption, not validation. Kappa is not used as classification probability.',
                  'Other output types appear only as noisy outputs of the historical primary-output classifier; no SHIFT or SEAL latent regime is constructed.'])
    (OUT/'annotation_channel.json').write_text(json.dumps(meta,indent=2))
    return primary,smooth,background,empirical

def main():
    C,smooth,bg,raw=channel()
    metadata=dict(status='Exploratory prototype after review; not preregistered; no human validation',
        n_dyads_per_condition=2000,simulation_seed=202609302,observer_seed=202609303,
        params=M.P,hypotheses=M.HYPOTHESES,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        channel='Dirichlet pseudocount 0.5 per output followed by 25% state-independent mixture; specified sensitivity assumptions',
        inference='Exact external filter on (fact,form pair) for four specified parameter points; parameters are not estimated',
        forecast='Agents hold previous partner form and coarse act fixed for their one-step action evaluation',
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'source').glob('*.py')})
    (OUT/'design.json').write_text(json.dumps(metadata,indent=2))
    w=M.worlds(2000,202609302);runs={};summary=[];contrasts=[];traces=[]
    conditions=[(name,dict(omega=om,c=c,gamma=g)) for name,om,c,g in M.HYPOTHESES]
    conditions += [('one_protects',dict(gamma=[0,2])),('both_protect_no_coupling',dict(gamma=2,coupling=False)),
        ('both_protect_no_IG',dict(gamma=2,epistemic=False)),('reference_no_IG',dict(epistemic=False)),
        ('remove_exposure_cost',dict(gamma=2,intervention='remove_gamma')),
        ('force_A_open',dict(gamma=2,intervention='force_A_open')),
        ('bypass_access',dict(gamma=2,intervention='bypass'))]
    for name,kw in conditions:
        d=M.simulate(w,**kw,C=C);mm=M.metrics(d);runs[name]=mm
        row=dict(condition=name,**{key:float(v.mean()) for key,v in mm.items()});summary.append(row)
        for t in range(M.P['rounds']+1):
            q=d['q'][:,t];pt=np.where(d['truth'][:,None]==1,q,1-q).mean(1)
            traces.append(dict(condition=name,round=t,p_truth=float(pt.mean()),disagreement=float(np.abs(q[:,0]-q[:,1]).mean())))
    for a,b in [('access_protection','reference'),('one_protects','reference'),('access_protection','one_protects'),
        ('both_protect_no_coupling','access_protection'),('remove_exposure_cost','access_protection'),
        ('force_A_open','access_protection'),('bypass_access','access_protection'),('access_protection','both_protect_no_IG')]:
        for metric in runs[a]:contrasts.append(dict(contrast=a+' minus '+b,metric=metric,**ci(runs[a][metric]-runs[b][metric])))
    s=save('dyad_summary.csv',summary);save('dyad_contrasts.csv',contrasts);save('dyad_traces.csv',traces)
    print(s.round(4).to_string(index=False),flush=True)
    # Predefined small sensitivity grid, including reduced and increased exposure cost.
    ss=[]
    for g in [0.,.5,1.,2.,4.]:
        for omega in [.3,1.]:
            mm=M.metrics(M.simulate(w,omega=omega,gamma=g))
            ss.append(dict(gamma=g,omega=omega,**{k:float(v.mean()) for k,v in mm.items()}))
    save('parameter_sensitivity.csv',ss)
    w2=M.worlds(2000,202609303);rec=[];pred=[];diff=[];validation={};stored={}
    for k,(name,om,c,g) in enumerate(M.HYPOTHESES):
        d=M.simulate(w2,omega=om,c=c,gamma=g,C=C)
        mode_results={}
        for mode in ['actions','labels','oracle','null_labels']:
            t=time.time();rr=M.observe(d,C,mode);mode_results[mode]=rr
            pp=rr['posterior'][:,k];acc=(rr['posterior'].argmax(1)==k).astype(float)
            rec.append(dict(generator=name,observation=mode,accuracy=float(acc.mean()),p_true_family=float(pp.mean())))
            y=(d['ev'][:,1:]!=2).astype(float);pr=rr['next_evidence_probability'].clip(1e-12,1-1e-12)
            loss=-(y*np.log(pr)+(1-y)*np.log(1-pr))[:,3:].mean(1)
            pred.append(dict(generator=name,observation=mode,**ci(loss)))
            rr['loss_per_world']=loss
            stored[f'{name}__{mode}__posterior']=rr['posterior']
            stored[f'{name}__{mode}__prediction']=rr['next_evidence_probability']
            print('Observer',name,mode,'acc',round(acc.mean(),4),'loss',round(loss.mean(),4),'seconds',round(time.time()-t,1),flush=True)
        for mode in ['labels','oracle','null_labels']:
            diff.append(dict(generator=name,comparison=mode+' versus actions',metric='next_evidence_logloss_gain',
                **ci(mode_results['actions']['loss_per_world']-mode_results[mode]['loss_per_world'])))
        error=float(np.abs(mode_results['actions']['posterior']-mode_results['null_labels']['posterior']).max())
        pred_error=float(np.abs(mode_results['actions']['next_evidence_probability']-mode_results['null_labels']['next_evidence_probability']).max())
        validation[name+'_null_posterior_max_error']=error
        validation[name+'_null_prediction_max_error']=pred_error
        assert error<1e-12 and pred_error<1e-12
    save('mechanism_recovery.csv',rec);save('next_evidence_prediction.csv',pred);save('prediction_gains.csv',diff)
    np.savez_compressed(OUT/'observer_results.npz',**stored)
    # Channel sensitivity on same 4 generators and same shared observer worlds.
    nr=[]
    for eta in [0.,.5,1.]:
        CC=(1-eta)*smooth+eta*bg
        for name,om,c,g in M.HYPOTHESES:
            d=M.simulate(w2,omega=om,c=c,gamma=g,C=CC)
            aa=M.observe(d,CC,'actions');zz=M.observe(d,CC,'labels')
            y=(d['ev'][:,1:]!=2).astype(float)
            def loss(rr):
                p=rr['next_evidence_probability'].clip(1e-12,1-1e-12)
                return -(y*np.log(p)+(1-y)*np.log(1-p))[:,3:].mean(1)
            nr.append(dict(extra_noise=eta,generator=name,**ci(loss(aa)-loss(zz))))
            if eta==1:
                er=float(np.abs(aa['posterior']-zz['posterior']).max())
                validation[name+'_state_independent_channel_error']=er;assert er<1e-12
    save('noise_sensitivity.csv',nr)
    # Validate policy, deterministic Bayes recursion and influence in BOTH directions.
    d=M.simulate(w2,C=C)
    assert np.all((d['q']>=0)&(d['q']<=1))
    for i in [0,1]:
        q=np.linspace(.01,.99,31)
        pr=M.probability(q,1.,0.,2.,np.ones(31),np.zeros(31),i)
        assert np.allclose(pr.sum((1,2)),1.)
        p0=M.probability(q,1.,0.,2.,np.zeros(31),np.zeros(31),i)
        validation[f'agent{i}_partner_form_policy_effect']=float(np.abs(pr-p0).max())
        assert validation[f'agent{i}_partner_form_policy_effect']>0
    validation['empirical_channel_rows_disjoint']=bool(np.sum(np.minimum(raw[0],raw[1]))==0)
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2))
    print('All experiment and control checks finished.',flush=True)

if __name__=='__main__':main()
