import json,time
import numpy as np,pandas as pd
import model as M,observer as O,channel as C
OUT=M.ROOT/'results'
def loss(p,y):
    p=np.clip(p,1e-12,1-1e-12);return -(y*np.log(p)+(1-y)*np.log1p(-p))
def main():
    t0=time.time();n=M.P['n_observer'];k=M.P['observer_paths_per_parameter_private_stratum']
    w=M.make_worlds(n,M.P['seeds']['observer']);true_a=np.arange(n)%4
    theta=np.array(list(M.TYPES.values()))[true_a]
    d=M.simulate(w,theta,M.TYPES['reference'])
    z,Cdraw=C.emit(d,M.P['seeds']['channel']);alpha=C.calibration()
    np.savez_compressed(OUT/'observer_data.npz',**{key:value for key,value in d.items() if isinstance(value,np.ndarray)},labels=z,true_a=true_a,channel_draws=Cdraw)
    (OUT/'annotation_posterior.json').write_text(json.dumps(C.sample_information(alpha),indent=2))
    rows=[];raw={}
    for i in range(n):
        for mode in ['forms_evidence','actions_evidence','labels_evidence']:
            r=O.recover_one(d,i,mode,paths=k,seed=M.P['seeds']['observer']+i*100,labels=z)
            pa=r['posterior'].sum(1);pb=r['posterior'].sum(0)
            y=(d['e'][i]!=M.NONE).astype(float)
            predact=r['act_probability'][np.arange(len(y)),d['y'][i]//2]
            rows.append(dict(world=i,generator=M.TYPE_NAMES[true_a[i]],mode=mode,paths=(1 if mode=='forms_evidence' else k),
                correct_A=int(pa.argmax()==true_a[i]),posterior_true_A=pa[true_a[i]],correct_B=int(pb.argmax()==0),posterior_true_B=pb[0],
                arrival_logloss=float(loss(r['arrival_probability'],y)[4:].mean()),
                act_logloss=float(-np.log(np.clip(predact[4:],1e-12,None)).mean()),min_path_ESS=r['min_path_ESS']))
            raw[f'{i}__{mode}__posterior']=r['posterior'];raw[f'{i}__{mode}__arrival']=r['arrival_probability'];raw[f'{i}__{mode}__act']=r['act_probability']
        pd.DataFrame(rows).to_csv(OUT/'observer_worlds.csv',index=False)
        if i%4==3:print('observer worlds',i+1,'/',n,'seconds',round(time.time()-t0,1),flush=True)
    np.savez_compressed(OUT/'observer_outputs.npz',**raw)
    tab=pd.DataFrame(rows);summary=tab.groupby(['generator','mode']).mean(numeric_only=True).reset_index();summary.to_csv(OUT/'observer_summary.csv',index=False)
    from run_sensitivity import run
    run(main_elapsed=time.time()-t0)

if __name__=='__main__':main()
