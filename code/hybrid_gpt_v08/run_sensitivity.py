"""Resume independent observer path-count checks using four local worker processes."""
from concurrent.futures import ProcessPoolExecutor
import time,json,re,os
import numpy as np,pandas as pd
import model as M,observer as O
R=M.ROOT/'results'
D=Z=RAW=None

def initialize():
    global D,Z,RAW
    D=dict(np.load(R/'observer_data.npz'));Z=D.pop('labels');RAW=np.load(R/'observer_outputs.npz')

def task(job):
    i,mode,k=job
    r=O.recover_one(D,i,mode,paths=k,seed=M.P['seeds']['observer']+i*100,labels=Z)
    pa=r['posterior'].sum(1);y=(D['e'][i]!=M.NONE).astype(float)
    p=np.clip(r['arrival_probability'],1e-12,1-1e-12)
    loss=-(y*np.log(p)+(1-y)*np.log1p(-p))
    return dict(world=i,mode=mode,paths=k,posterior_true_A=float(pa[D['true_a'][i]]),
        correct_A=int(pa.argmax()==D['true_a'][i]),arrival_logloss=float(loss[4:].mean()),
        max_posterior_difference_from_main=float(np.abs(r['posterior']-RAW[f'{i}__{mode}__posterior']).max()))

def run(main_elapsed=None):
    t=time.time();path=R/'particle_sensitivity.csv'
    rows=pd.read_csv(path).to_dict('records') if main_elapsed is None and path.exists() else []
    done={(int(r['world']),r['mode'],int(r['paths'])) for r in rows}
    jobs=[(i,m,k) for i in range(M.P['observer_sensitivity_worlds']) for m in ['actions_evidence','labels_evidence'] for k in [4,32] if (i,m,k) not in done]
    # The seed is a function of the world only; scheduling never changes the samples.
    with ProcessPoolExecutor(max_workers=min(4,os.cpu_count() or 1),initializer=initialize) as pool:
        for result in pool.map(task,jobs,chunksize=1):
            rows.append(result)
            pd.DataFrame(rows).sort_values(['world','mode','paths']).to_csv(path,index=False)
            print('sensitivity tasks',len(rows),'/',M.P['observer_sensitivity_worlds']*4,flush=True)
    if main_elapsed is None:
        log=(R/'observer.log').read_text();match=re.findall(r'observer worlds 96 / 96 seconds ([0-9.]+)',log)
        main_elapsed=float(match[-1]) if match else None
    n=M.P['n_observer'];elapsed=time.time()-t
    (R/'observer_manifest.json').write_text(json.dumps(dict(n_worlds=n,n_per_A_type=n//4,B_true='reference',B_unknown_to_observer=True,
        parameter_pairs=16,private_strata=9,paths=M.P['observer_paths_per_parameter_private_stratum'],
        main_elapsed=main_elapsed,parallel_sensitivity_elapsed=elapsed,forecast_excludes_first_events=4,
        local_sensitivity_workers=4,sensitivity_worlds=M.P['observer_sensitivity_worlds']),indent=2))

if __name__=='__main__':run()
