"""A second, non-Clifford six-state/four-unknown confirmation family.

The two extra symmetric features destroy scalar information geometry.
All baselines use the same continuous-statistic mesh approximation.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.linalg import expm,solve_continuous_lyapunov
from matrix_design import a_optimal_weights
from run_multiparameter_feedback import features,dictionary,write_rows

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'coupled_fourparam'
POLICIES=['oracle_mixture','oracle_fixed','periodic','state_feedback']


def model(theta,dt):
    base,known=features(6);rng=np.random.default_rng(2026110601)
    bs=list(base)
    for _ in range(2):
        z=rng.normal(size=(6,6));z=(z+z.T)/2;z-=np.trace(z)*np.eye(6)/6
        z/=np.linalg.norm(z,2);bs.append(z)
    bs=np.array(bs);vs=dictionary(6,'pairs')
    f0=-4*np.eye(6)[None]+3*np.einsum('li,lj->lij',vs,vs)+known
    fs=f0+np.einsum('p,pij->ij',theta,bs);aa=[];ll=[];js=[]
    for f in fs:
        cov=solve_continuous_lyapunov(f,-np.eye(6));cov=(cov+cov.T)/2
        a=expm(f*dt);q=cov-a@cov@a.T;q=(q+q.T)/2
        aa.append(a);ll.append(np.linalg.cholesky(q))
        js.append(np.einsum('iab,jac,bc->ij',bs,bs,cov,optimize=True))
    js=np.array(js);w,c,ok=a_optimal_weights(js)
    assert ok
    # First-order equivalence gap is a numerical diagnostic, not a certificate.
    inv=np.linalg.inv(np.einsum('l,lij->ij',w,js))
    gap=float(np.max(np.einsum('ij,lji->l',inv@inv,js))-c)
    return bs,vs,f0,np.array(aa),np.array(ll),js,w,c,gap


def simulate(theta,T,n,dt,seed,noise_mode=None):
    theta=np.asarray(theta);bs,vs,f0,aa,ll,js,w,constant,gap=model(theta,dt)
    d=6;p=4;L=len(vs);kp=4;steps=round(T/dt);rng=np.random.default_rng(seed);cache=[]
    def noise():
        if noise_mode=='coarse_pair':
            z=rng.normal(size=(2,n,d));return (z[0]+z[1])/np.sqrt(2)
        if noise_mode=='fine_pair':
            if cache:return cache.pop()
            z=rng.normal(size=(2,n,d));cache.append(z[1]);return z[0]
        return rng.normal(size=(n,d))
    x=np.zeros((kp,n,d));g=np.zeros((kp,n,p,p));s=np.zeros((kp,n,p))
    act=np.zeros((kp,n),int);counts=np.zeros((kp,n,L),np.int32)
    oracle=int(np.argmin(np.trace(np.linalg.inv(js),axis1=-2,axis2=-1)))
    bounds=np.rint(np.cumsum(w)*steps).astype(int)
    for k in range(steps):
        act[0]=min(int(np.sum(k>=bounds)),L-1);act[1]=oracle;act[2]=k%L
        act[3]=np.argmax((x[3]@vs.T)**2,axis=1)
        z=np.broadcast_to(noise(),x.shape)
        y=np.einsum('knij,knj->kni',aa[act],x)+np.einsum('knij,knj->kni',ll[act],z)
        px=np.einsum('pij,knj->knpi',bs,x,optimize=True);py=np.einsum('pij,knj->knpi',bs,y,optimize=True)
        fx=np.einsum('knij,knj->kni',f0[act],x);fy=np.einsum('knij,knj->kni',f0[act],y)
        s+=np.einsum('knpi,kni->knp',.5*(px+py),y-x)-.5*dt*np.trace(bs,axis1=-2,axis2=-1)
        s-=.5*dt*(np.einsum('knpi,kni->knp',px,fx)+np.einsum('knpi,kni->knp',py,fy))
        g+=.5*dt*(np.einsum('knpi,knqi->knpq',px,px)+np.einsum('knpi,knqi->knpq',py,py))
        for r in range(kp):np.add.at(counts[r],(np.arange(n),act[r]),1)
        x=y
    valid=np.linalg.eigvalsh(g)[:,:,0]>1e-8*T
    est=np.zeros((kp,n,p));est[valid]=np.linalg.solve(g[valid],s[valid])
    error=est-theta;loss=(error**2).sum(axis=-1)
    raw=dict(theta=theta,T=T,n=n,dt=dt,seed=seed,bases=bs,vectors=vs,policies=np.array(POLICIES),
        estimates=est,squared_error=loss,coordinate_squared_error=error**2,gram=g,score=s,
        terminal_state=x,action_counts=counts,guard_valid=valid,oracle_weights=w,
        oracle_information=js,oracle_constant=constant,oracle_equivalence_gap=gap)
    rows=[]
    for r,name in enumerate(POLICIES):
        rows.append(dict(policy=name,T=T,n=n,dt=dt,mse=float(loss[r].mean()),mse_se=float(loss[r].std(ddof=1)/np.sqrt(n)),
            bias_squared=float(np.sum(error[r].mean(axis=0)**2)),oracle_constant=constant,oracle_equivalence_gap=gap,
            guard_failures=int((~valid[r]).sum())))
    return rows,raw


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','confirmation','refinement'],default='development')
    ap.add_argument('--n',type=int,default=128);args=ap.parse_args()
    out=OUT/args.phase;out.mkdir(parents=True,exist_ok=True)
    truths=[np.array([0.,.2,.1,-.1])] if args.phase=='development' else [np.array([-.2,.35,.15,-.12]),np.array([.2,-.3,-.12,.15])]
    cases=[(0,64.)] if args.phase=='development' else [(i,T) for i in [0,1] for T in [64.,256.]]
    allrows=[];start=time.time();base={'development':2026111200,'confirmation':2026112200,'refinement':2026113200}[args.phase]
    for case,(i,T) in enumerate(cases):
        if args.phase=='refinement' and T!=256:continue
        if args.phase=='refinement':
            coarse,r0=simulate(truths[i],T,args.n,.02,base+case,'coarse_pair')
            rows,raw=simulate(truths[i],T,args.n,.01,base+case,'fine_pair')
            np.savez_compressed(out/f'case{case}_coarse.npz',**r0)
            for r,row in enumerate(rows):
                delta=raw['squared_error'][r]-r0['squared_error'][r]
                row.update(coarse_mse=coarse[r]['mse'],mesh_difference=float(delta.mean()),mesh_se=float(delta.std(ddof=1)/np.sqrt(args.n)))
        else:rows,raw=simulate(truths[i],T,args.n,.02,base+case)
        for row in rows:row.update(case=case,theta_index=i)
        allrows+=rows;write_rows(out/'risk.csv',allrows);np.savez_compressed(out/f'case{case}.npz',**raw)
        print(json.dumps(dict(case=case,T=T,reduction_vs_oracle_mixture=100*(1-rows[-1]['mse']/rows[0]['mse']),
            reduction_vs_periodic=100*(1-rows[-1]['mse']/rows[2]['mse']),equivalence_gap=raw['oracle_equivalence_gap'],elapsed=time.time()-start)),flush=True)


if __name__=='__main__':main()
