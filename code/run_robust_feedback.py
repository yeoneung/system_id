"""Perturbed six-state identification under a certified finite direction set.

Known feature and drift perturbations are part of the fitted model, not ignored
misspecification. Held-mode OU endpoints are exact; continuous statistics are
trapezoidal approximations with Ito correction. No policy receives theta except
the explicitly named stationary oracles.
"""
from pathlib import Path
import argparse,csv,hashlib,itertools,json,time
import numpy as np
from scipy.linalg import expm,solve_continuous_lyapunov
from matrix_design import a_optimal_weights
from run_multiparameter_feedback import features,write_rows
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/robustness'
POLICIES=['oracle_mixture','oracle_fixed','projection']


def directions(d=6):
    raw=[]
    for v in itertools.product([-1,0,1],repeat=d):
        nz=[i for i,z in enumerate(v) if z]
        if nz and v[nz[0]]==1:raw.append(v)
    raw=np.array(raw,dtype=int)
    vs=raw/np.linalg.norm(raw,axis=1)[:,None]
    powers=3**np.arange(d);lookup=np.full(3**d,-1,int)
    lookup[(raw+1)@powers]=np.arange(len(raw))
    s=np.sum(np.diff(np.sqrt(np.arange(d+1)))**2)
    return vs,lookup,float(1-1/s)


def select(x,vs,lookup):
    """Exactly maximize over signed subsets via sorting; brute-force on ties."""
    n,d=x.shape;order=np.argsort(-np.abs(x),axis=1,kind='stable')
    vals=np.take_along_axis(x,order,axis=1);mag=np.abs(vals)
    quality=np.cumsum(mag,axis=1)**2/np.arange(1,d+1)
    size=np.argmax(quality,axis=1)+1
    coeff=np.zeros_like(x,dtype=int)
    coeff[np.arange(n)[:,None],order]=np.where(np.arange(d)[None]<size[:,None],np.where(vals>=0,1,-1),0)
    first=np.argmax(coeff!=0,axis=1);coeff*=coeff[np.arange(n),first,None]
    idx=lookup[(coeff+1)@(3**np.arange(d))]
    top=np.sort(quality,axis=1)
    ties=np.any(np.isclose(mag[:,:-1],mag[:,1:],rtol=1e-13,atol=1e-14),axis=1)|np.isclose(top[:,-1],top[:,-2],rtol=1e-13,atol=1e-14)
    if np.any(ties):idx[ties]=np.argmax((x[ties]@vs.T)**2,axis=1)
    assert np.all(idx>=0)
    return idx


def build(geometry,level,theta,dt):
    d=6;base,_=features(d);p=len(base);a=1.;b=4.;eta=delta=float(level)
    rng=np.random.default_rng(2026091950+geometry)
    ds=[]
    for _ in range(p):
        z=rng.normal(size=(d,d));z/=np.linalg.norm(z,2);ds.append(z*eta/np.sqrt(p))
    bs=base+np.array(ds)
    e=rng.normal(size=(d,d));e*=delta/np.linalg.norm(e,2)
    vs,lookup,q=directions(d)
    f0=-b*np.eye(d)[None]+(b-a)*np.einsum('li,lj->lij',vs,vs)+e
    fs=f0+np.einsum('p,pij->ij',theta,bs)
    aa=[];ll=[];infos=[]
    for f in fs:
        cov=solve_continuous_lyapunov(f,-np.eye(d));cov=(cov+cov.T)/2
        step=expm(f*dt);noise=cov-step@cov@step.T;noise=(noise+noise.T)/2
        aa.append(step);ll.append(np.linalg.cholesky(noise))
        info=np.einsum('iab,jac,bc->ij',bs,bs,cov,optimize=True);infos.append((info+info.T)/2)
    infos=np.array(infos)
    fixed=int(np.argmin(np.trace(np.linalg.inv(infos),axis1=1,axis2=2)))
    start=np.zeros(len(vs));start[fixed]=1.
    w,c,ok=a_optimal_weights(infos,start=start)
    if not ok:raise RuntimeError('Stationary mixture solver did not converge')
    inv=np.linalg.inv(np.einsum('l,lij->ij',w,infos))
    gap=float(max(0,np.max(np.einsum('ij,lji->l',inv@inv,infos))-c))
    radius=np.sqrt(p)*.15;gamma=a-delta-.15-eta*radius
    kappa=(1-eta)**2*d/(2*(b+delta+.15+eta*radius))
    jol=1/(2*(a-delta))+(d-1)/(2*(b-delta))
    lower=p/((1+eta)**2*jol)
    upper=2*p*(a+(b-a)*q+delta)/((1-eta)**2*d)
    return dict(bs=bs,vs=vs,lookup=lookup,q=q,f0=f0,fs=fs,aa=np.array(aa),ll=np.array(ll),
                infos=infos,weights=w,oracle_constant=c,oracle_gap=gap,oracle_fixed=fixed,
                eta=eta,delta=delta,known_deviation=e,gamma=gamma,kappa=kappa,
                openloop_lower=lower,feedback_upper=upper,certified_factor=lower/upper)


def verify_manifest():
    f=ROOT/'metadata/robustness_manifest.json'
    if not f.exists():raise RuntimeError('Experiment manifest missing; restore metadata from the repository')
    record=json.loads(f.read_text())
    for n,h in record['sha256'].items():
        assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n


def simulate(geometry,level,T,n,dt,seed,theta=None,noise_mode=None):
    theta=np.zeros(2) if theta is None else np.asarray(theta,float)
    m=build(geometry,level,theta,dt);d=6;p=2;kpol=len(POLICIES);L=len(m['vs'])
    steps=round(T/dt);assert abs(steps*dt-T)<1e-10
    rng=np.random.default_rng(seed);cache=[]
    def noise():
        if noise_mode=='coarse_pair':
            z=rng.normal(size=(2,n,d));return (z[0]+z[1])/np.sqrt(2)
        if noise_mode=='fine_pair':
            if cache:return cache.pop()
            z=rng.normal(size=(2,n,d));cache.append(z[1]);return z[0]
        return rng.normal(size=(n,d))
    x=np.zeros((kpol,n,d));score=np.zeros((kpol,n,p));gram=np.zeros((kpol,n,p,p))
    energy=np.zeros((kpol,n));power=np.zeros_like(energy);leak=np.zeros_like(energy)
    counts=np.zeros((kpol,n,L),np.int32);act=np.zeros((kpol,n),int)
    cum=np.cumsum(m['weights']);cum[-1]=1.;trace=np.trace(m['bs'],axis1=1,axis2=2)
    for k in range(steps):
        act[0]=min(np.searchsorted(cum,(k+.5)/steps),L-1)
        act[1]=m['oracle_fixed'];act[2]=select(x[2],m['vs'],m['lookup'])
        z=np.broadcast_to(noise(),x.shape)
        y=np.einsum('knij,knj->kni',m['aa'][act],x)+np.einsum('knij,knj->kni',m['ll'][act],z)
        phix=np.einsum('pij,knj->knpi',m['bs'],x,optimize=True)
        phiy=np.einsum('pij,knj->knpi',m['bs'],y,optimize=True)
        fx=np.einsum('knij,knj->kni',m['f0'][act],x);fy=np.einsum('knij,knj->kni',m['f0'][act],y)
        score+=np.einsum('knpi,kni->knp',.5*(phix+phiy),y-x)-.5*dt*trace
        score-=.5*dt*(np.einsum('knpi,kni->knp',phix,fx)+np.einsum('knpi,kni->knp',phiy,fy))
        gram+=.5*dt*(np.einsum('knpi,knqi->knpq',phix,phix)+np.einsum('knpi,knqi->knpq',phiy,phiy))
        energy+=.5*dt*(np.sum(x*x,axis=-1)+np.sum(y*y,axis=-1))
        v=m['vs'][act]
        power+=.5*dt*9*(np.einsum('kni,kni->kn',v,x)**2+np.einsum('kni,kni->kn',v,y)**2)
        leak+=.5*dt*(np.einsum('kni,ij,knj->kn',x,m['known_deviation'],x)+np.einsum('kni,ij,knj->kn',y,m['known_deviation'],y))
        for j in range(kpol):np.add.at(counts[j],(np.arange(n),act[j]),1)
        x=y
    valid=np.linalg.eigvalsh(gram)[...,0]>=m['kappa']*T/2
    estimates=np.zeros_like(score);estimates[valid]=np.linalg.solve(gram[valid],score[valid])
    errors=estimates-theta;loss=np.sum(errors**2,axis=-1)
    raw=dict(geometry=geometry,level=level,theta=theta,T=T,n=n,dt=dt,seed=seed,policies=np.array(POLICIES),
        estimates=estimates,squared_error=loss,coordinate_squared_error=errors**2,score=score,gram=gram,
        energy=energy,input_energy=power,deviation_energy=leak,terminal_state=x,action_counts=counts,guard_valid=valid)
    raw.update({k:m[k] for k in ['bs','vs','q','known_deviation','weights','infos','oracle_constant','oracle_gap',
                               'oracle_fixed','eta','delta','gamma','kappa','openloop_lower','feedback_upper','certified_factor']})
    rows=[]
    for j,name in enumerate(POLICIES):
        rows.append(dict(geometry=geometry,level=level,T=T,n=n,dt=dt,policy=name,
            mse=float(loss[j].mean()),mse_se=float(loss[j].std(ddof=1)/np.sqrt(n)),
            mean_energy=float(energy[j].mean()/T),mean_input_power=float(power[j].mean()/T),
            guard_failures=int((~valid[j]).sum()),certified_factor=m['certified_factor'],oracle_gap=m['oracle_gap']))
    return rows,raw


def cases():return [(0,0.)]+[(g,e) for g in range(4) for e in [.04,.10]]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','confirmation','refinement'],default='development');args=ap.parse_args()
    phase=args.phase;out=OUT/phase;out.mkdir(parents=True,exist_ok=True)
    if phase!='development':verify_manifest()
    prescribed=[(0,.08)] if phase=='development' else cases()
    allrows=[];start=time.time()
    for case,(g,e) in enumerate(prescribed):
        if phase=='refinement' and not (g in [0,1] and e>0):continue
        T,n=(8.,32) if phase=='development' else (128.,512)
        theta=[.08,-.12] if phase=='development' else [0.,0.]
        seed={'development':2026091960,'confirmation':2026092000,'refinement':2026092100}[phase]+case
        if phase=='refinement':
            coarse,raw0=simulate(g,e,T,n,.02,seed,theta,'coarse_pair')
            rows,raw=simulate(g,e,T,n,.01,seed,theta,'fine_pair')
            np.savez_compressed(out/f'case{case}_coarse.npz',**raw0)
            for j,row in enumerate(rows):
                diff=raw['squared_error'][j]-raw0['squared_error'][j]
                row.update(coarse_mse=coarse[j]['mse'],mesh_difference=float(diff.mean()),mesh_se=float(diff.std(ddof=1)/np.sqrt(n)))
        else:rows,raw=simulate(g,e,T,n,.02,seed,theta)
        for row in rows:row['case']=case
        allrows+=rows;np.savez_compressed(out/f'case{case}.npz',**raw);write_rows(out/'risk.csv',allrows)
        print(json.dumps(dict(phase=phase,case=case,geometry=g,level=e,
            gain_vs_oracle=100*(1-rows[2]['mse']/rows[0]['mse']),elapsed=time.time()-start)),flush=True)
    (out/'metadata.json').write_text(json.dumps(dict(phase=phase,cases=len(allrows)//3,policy_rollouts=sum(r['n'] for r in allrows)*(2 if phase=='refinement' else 1),elapsed_seconds=time.time()-start),indent=2)+'\n')


if __name__=='__main__':main()
