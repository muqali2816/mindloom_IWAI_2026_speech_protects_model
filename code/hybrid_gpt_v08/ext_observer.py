"""Independent extension of the GPT hybrid v0.8 observer pilot: 480 fresh worlds (120 per A profile, B = reference,
new seed), same code path (observer.recover_one, K=8), three channels. Nothing in the package is modified."""
import sys, time, json; sys.path.insert(0, 'hybrid_dyad_v08')
import numpy as np, pandas as pd
import model as M, observer as O, channel as C
N=480; SEED=2026093099
w=M.make_worlds(N,SEED); true_a=np.arange(N)%4
theta=np.array(list(M.TYPES.values()))[true_a]
d=M.simulate(w,theta,M.TYPES['reference']); z,_=C.emit(d,SEED+77)
def loss(p,y):
    p=np.clip(p,1e-12,1-1e-12); return -(y*np.log(p)+(1-y)*np.log1p(-p))
def task(i):
    rows=[]
    for mode in ['forms_evidence','actions_evidence','labels_evidence']:
        r=O.recover_one(d,i,mode,paths=8,seed=SEED+i*100,labels=z)
        pa=r['posterior'].sum(1); pb=r['posterior'].sum(0); y=(d['e'][i]!=M.NONE).astype(float)
        predact=r['act_probability'][np.arange(len(y)),d['y'][i]//2]
        rows.append(dict(world=i,generator=M.TYPE_NAMES[true_a[i]],mode=mode,correct_A=int(pa.argmax()==true_a[i]),posterior_true_A=float(pa[true_a[i]]),
            correct_B=int(pb.argmax()==0),posterior_true_B=float(pb[0]),arrival_logloss=float(loss(r['arrival_probability'],y)[4:].mean()),
            act_logloss=float(-np.log(np.clip(predact[4:],1e-12,None)).mean()),min_path_ESS=r['min_path_ESS']))
    return rows
if __name__=='__main__':
    shard,nshard=int(sys.argv[1]),int(sys.argv[2]); t0=time.time(); out=[]
    for i in range(shard,N,nshard):
        out+=task(i)
    pd.DataFrame(out).to_csv(f'ext_shards/shard_{shard:02d}.csv',index=False); print('done',shard,round(time.time()-t0))
