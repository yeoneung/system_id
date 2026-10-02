"""Prospective coupled multi-parameter feedback experiments.

Exact OU endpoints; continuous-observation least-squares statistics are
approximated on the recorded mesh. This is NOT an exact sampled-data MLE.
The state selector does not receive the true parameter. The explicitly
labelled oracle_fixed comparator does, to make the comparison stronger.
"""
from pathlib import Path
import argparse, itertools, json, time, csv
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'multiparameter'
POLICIES=['oracle_fixed','fixed','long_blocks','periodic','state_feedback']


def features(d):
    j=np.array([[0.,-1.],[1.,0.]])
    if d==6:
        skew=[np.kron(np.eye(3),j)]
    elif d==8:
        j1=np.array([[0,-1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,1,0]],float)
        j2=np.array([[0,0,-1,0],[0,0,0,1],[1,0,0,0],[0,-1,0,0]],float)
        skew=[np.kron(np.eye(2),s) for s in [j1,j2,j1@j2]]
    else:
        raise ValueError('Declared families have (d,p)=(6,2) or (8,4).')
    rng=np.random.default_rng(2026110100+d)
    o=np.linalg.qr(rng.normal(size=(d,d)))[0]
    bases=np.array([np.eye(d)]+[o@s@o.T for s in skew])
    raw=rng.normal(size=(d,d));known=raw-raw.T
    known*=.15/np.linalg.norm(known,2)
    return bases,known


def dictionary(d,kind):
    vs=list(np.eye(d))
    if kind=='pairs':
        for i,j in itertools.combinations(range(d),2):
            for sign in [-1,1]:
                v=np.zeros(d);v[i]=1/np.sqrt(2);v[j]=sign/np.sqrt(2);vs.append(v)
    return np.array(vs)


def model(d,kind,theta,dt):
    bases,known=features(d);vs=dictionary(d,kind)
    f0=-4*np.eye(d)[None]+3*np.einsum('li,lj->lij',vs,vs)+known
    fs=f0+np.einsum('p,pij->ij',theta,bases)
    aa=[];ll=[];energies=[]
    for f in fs:
        p=solve_continuous_lyapunov(f,-np.eye(d));p=(p+p.T)/2
        a=expm(f*dt);q=p-a@p@a.T;q=(q+q.T)/2
        aa.append(a);ll.append(np.linalg.cholesky(q));energies.append(np.trace(p))
    return bases,vs,f0,np.array(aa),np.array(ll),np.array(energies)


def simulate(d,kind,theta,T,n,dt,seed,noise_mode=None):
    theta=np.asarray(theta);bases,vs,f0,aa,ll,energies=model(d,kind,theta,dt)
    p=len(bases);L=len(vs);kpol=len(POLICIES);steps=round(T/dt)
    assert abs(steps*dt-T)<1e-9
    rng=np.random.default_rng(seed);cache=[]
    def noise():
        if noise_mode=='coarse_pair':
            z=rng.normal(size=(2,n,d));return (z[0]+z[1])/np.sqrt(2)
        if noise_mode=='fine_pair':
            if cache:return cache.pop()
            z=rng.normal(size=(2,n,d));cache.append(z[1]);return z[0]
        return rng.normal(size=(n,d))
    x=np.zeros((kpol,n,d));gram=np.zeros((kpol,n));score=np.zeros((kpol,n,p))
    counts=np.zeros((kpol,n,L),np.int32);act=np.zeros((kpol,n),int)
    oracle=int(np.argmax(energies));path_index=np.arange(n)
    for k in range(steps):
        act[0]=oracle;act[1]=0;act[2]=min(k*L//steps,L-1);act[3]=k%L
        act[4]=np.argmax((x[4]@vs.T)**2,axis=1)
        z=np.broadcast_to(noise(),x.shape)
        y=np.einsum('knij,knj->kni',aa[act],x)+np.einsum('knij,knj->kni',ll[act],z)
        phix=np.einsum('pij,knj->knpi',bases,x,optimize=True)
        phiy=np.einsum('pij,knj->knpi',bases,y,optimize=True)
        fx=np.einsum('knij,knj->kni',f0[act],x)
        fy=np.einsum('knij,knj->kni',f0[act],y)
        # Trapezoidal line integral is Ito for skew features; for I subtract
        # the exact Ito correction d*dt/2. Unresolved Levy areas remain a
        # mesh approximation for the skew scores and are checked by refinement.
        increment=np.einsum('knpi,kni->knp',.5*(phix+phiy),y-x)
        increment[:,:,0]-=.5*d*dt
        increment-=.5*dt*(np.einsum('knpi,kni->knp',phix,fx)+np.einsum('knpi,kni->knp',phiy,fy))
        score+=increment
        gram+=.5*dt*(np.sum(x*x,axis=-1)+np.sum(y*y,axis=-1))
        for r in range(kpol):np.add.at(counts[r],(path_index,act[r]),1)
        x=y
    valid=gram>1e-8*T
    estimates=np.divide(score,gram[:,:,None],out=np.zeros_like(score),where=valid[:,:,None])
    errors=estimates-theta;loss=np.sum(errors**2,axis=-1)
    raw=dict(d=d,p=p,dictionary=kind,T=T,n=n,dt=dt,seed=seed,theta=theta,
             bases=bases,vectors=vs,known_skew=features(d)[1],policies=np.array(POLICIES),
             estimates=estimates,squared_error=loss,coordinate_squared_error=errors**2,
             gram=gram,score=score,terminal_state=x,action_counts=counts,
             oracle_action=oracle,stationary_energy=energies,guard_valid=valid)
    rows=[]
    for r,name in enumerate(POLICIES):
        rows.append(dict(policy=name,d=d,p=p,dictionary=kind,actions=L,T=T,n=n,dt=dt,
            mse=float(loss[r].mean()),mse_se=float(loss[r].std(ddof=1)/np.sqrt(n)),
            bias_squared=float(np.sum(errors[r].mean(axis=0)**2)),
            mean_energy=float(gram[r].mean()/T),guard_failures=int((~valid[r]).sum()),
            oracle_mixture_constant=float(p/max(energies)),
            universal_openloop_lower_constant=float(p/(1/(2*(1-theta[0]))+(d-1)/(2*(4-theta[0]))))))
    return rows,raw


def truths(d,phase):
    if phase=='development':return [np.array([0.,.2]) if d==6 else np.array([0.,.2,-.1,.1])]
    return ([np.array([-.2,.35]),np.array([.2,-.3])] if d==6 else
            [np.array([-.2,.25,-.2,.15]),np.array([.2,-.3,.15,-.25])])


def write_rows(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','confirmation','refinement'],default='development')
    ap.add_argument('--n',type=int,default=256);ap.add_argument('--dt',type=float,default=.02)
    args=ap.parse_args();out=OUT/args.phase;out.mkdir(parents=True,exist_ok=True)
    if args.phase=='development':cases=[(d,kind,0,64.) for d in [6,8] for kind in ['axes','pairs']]
    else:cases=list(itertools.product([6,8],['axes','pairs'],[0,1],[64.,256.]))
    rows_all=[];start=time.time()
    seed_base={'development':2026111100,'confirmation':2026112100,'refinement':2026113100}[args.phase]
    for case,(d,kind,i,T) in enumerate(cases):
        if args.phase=='refinement' and not (kind=='pairs' and T==256):continue
        theta=truths(d,args.phase)[i]
        if args.phase=='refinement':
            coarse,raw0=simulate(d,kind,theta,T,args.n,.02,seed_base+case,'coarse_pair')
            rows,raw=simulate(d,kind,theta,T,args.n,.01,seed_base+case,'fine_pair')
            np.savez_compressed(out/f'case{case}_coarse.npz',**raw0)
            for r,row in enumerate(rows):
                delta=raw['squared_error'][r]-raw0['squared_error'][r]
                row.update(coarse_mse=coarse[r]['mse'],mesh_difference=float(delta.mean()),mesh_se=float(delta.std(ddof=1)/np.sqrt(args.n)))
        else:
            rows,raw=simulate(d,kind,theta,T,args.n,args.dt,seed_base+case)
        for row in rows:row.update(case=case,theta_index=i)
        rows_all+=rows;np.savez_compressed(out/f'case{case}.npz',**raw);write_rows(out/'risk.csv',rows_all)
        print(json.dumps(dict(case=case,d=d,p=len(theta),dictionary=kind,T=T,
            reduction_vs_oracle=100*(1-rows[-1]['mse']/rows[0]['mse']),
            reduction_vs_periodic=100*(1-rows[-1]['mse']/rows[3]['mse']),elapsed=time.time()-start)),flush=True)
    (out/'metadata.json').write_text(json.dumps(dict(phase=args.phase,n=args.n,dt=args.dt,seed_base=seed_base,
        elapsed_seconds=time.time()-start,estimator='Continuous-time LS with trapezoidal mesh approximation; not exact discrete MLE.',
        selector='Parameter-free largest squared projection; equal to one-step expected-energy greedy.',
        primary='Feedback vs oracle stationary optimum and periodic, 32 simultaneous paired contrasts.',
        model='Coupled unknown damping and unknown skew rotations; separate from original 6-state 4-coefficient model.'),indent=2)+'\n')


if __name__=='__main__':main()
