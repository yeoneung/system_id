"""Separate mode-update and observation clocks on the identical OU paths.

Two observation meshes are nested on one exact-transition physical path.
Policy updates are unchanged between the two statistic reconstructions.
Continuous-time LS and weighted stopped statistics are numerical quadratures,
not exact discrete-observation likelihood estimators.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from run_robust_feedback import build,select

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/hold'
POLICIES=['oracle_fixed','projection_002','projection_010','projection_050']
HOLDS=[np.inf,.02,.10,.50]


def instrument(x,bs):
    """W=R(Psi^T Psi)^-1 Psi^T and its rowwise divergence."""
    shape=x.shape[:-1];d=x.shape[-1];p=len(bs)
    z=x.reshape(-1,d);r=np.sum(z*z,axis=-1)
    phi=np.einsum('pij,nj->npi',bs,z)
    gram=np.einsum('npi,nqi->npq',phi,phi)
    good=r>0
    inv=np.zeros_like(gram);inv[good]=np.linalg.inv(gram[good])
    V=np.einsum('npq,nqj->npj',inv,phi)
    W=r[:,None,None]*V
    # dG[n,k,l,j] = (B_k e_j)^T B_l x + (B_k x)^T B_l e_j.
    dG=np.einsum('kij,nli->nklj',bs,phi)+np.einsum('nki,lij->nklj',phi,bs)
    div=2*np.einsum('npj,nj->np',V,z)
    div-=r[:,None]*np.einsum('npk,nklj,nlj->np',inv,dG,V)
    div+=r[:,None]*np.einsum('npk,k->np',inv,np.trace(bs,axis1=1,axis2=2))
    # Its value at the single zero state is irrelevant to the continuous
    # integral; this convention also recovers the exact ideal linear field.
    div[~good]=np.trace(bs,axis1=1,axis2=2)
    return W.reshape(*shape,p,d),div.reshape(*shape,p)


def new_stats(k,n,p):
    return dict(score=np.zeros((k,n,p)),gram=np.zeros((k,n,p,p)),
        energy=np.zeros((k,n)),power=np.zeros((k,n)),z=np.zeros((k,n,p)),
        hit=np.zeros((k,n),bool),hit_time=np.full((k,n),np.nan),
        threshold_score=np.zeros((k,n,p)))


def accumulate(s,x,y,act,m,dt,t,Wx,Wy,divx,divy,g):
    bs=m['bs'];px=np.einsum('pij,knj->knpi',bs,x)
    py=np.einsum('pij,knj->knpi',bs,y)
    fx=np.einsum('knij,knj->kni',m['f0'][act],x)
    fy=np.einsum('knij,knj->kni',m['f0'][act],y)
    trace=np.trace(bs,axis1=1,axis2=2)
    s['score']+=np.einsum('knpi,kni->knp',.5*(px+py),y-x)-.5*dt*trace
    s['score']-=.5*dt*(np.einsum('knpi,kni->knp',px,fx)+np.einsum('knpi,kni->knp',py,fy))
    s['gram']+=.5*dt*(np.einsum('knpi,knqi->knpq',px,px)+np.einsum('knpi,knqi->knpq',py,py))
    de=.5*dt*(np.sum(x*x,axis=-1)+np.sum(y*y,axis=-1))
    dz=np.einsum('knpi,kni->knp',.5*(Wx+Wy),y-x)-.25*dt*(divx+divy)
    dz-=.5*dt*(np.einsum('knpi,kni->knp',Wx,fx)+np.einsum('knpi,kni->knp',Wy,fy))
    crossing=(~s['hit'])&(s['energy']+de>=g)
    alpha=np.divide(g-s['energy'],de,out=np.zeros_like(de),where=de>0)
    s['threshold_score'][crossing]=(s['z']+alpha[...,None]*dz)[crossing]
    s['hit_time'][crossing]=(t-dt+alpha*dt)[crossing]
    s['hit']|=crossing;s['energy']+=de;s['z']+=dz
    v=m['vs'][act]
    s['power']+=.5*dt*9*(np.einsum('kni,kni->kn',v,x)**2+np.einsum('kni,kni->kn',v,y)**2)


def simulate(level,theta,T,n,seed,geometry=0):
    dt=.005;theta=np.asarray(theta,float);m=build(geometry,level,theta,dt)
    p=2;d=6;kp=len(POLICIES);steps=round(T/dt)
    assert abs(T-steps*dt)<1e-10 and steps%2==0
    h=.15;epsilon=.1;c=1+3*m['q']+level+(1+level*np.sqrt(p))*h
    g=(1-epsilon)*d*T/(2*c)
    rng=np.random.default_rng(seed)
    x=np.zeros((kp,n,d));Wx,divx=instrument(x,m['bs'])
    stats=[new_stats(kp,n,p) for _ in [.005,.01]]
    act=np.zeros((kp,n),int);act[0]=m['oracle_fixed']
    counts=np.zeros((kp,n,len(m['vs'])),np.int32)
    switches=np.zeros((kp,n),int);last=act.copy()
    for k in range(steps):
        for j,D in enumerate(HOLDS[1:],1):
            if k%round(D/dt)==0:act[j]=select(x[j],m['vs'],m['lookup'])
        if k>0:switches+=act!=last
        last=act.copy()
        if k%2==0:xc=x.copy();Wc=Wx.copy();divc=divx.copy()
        z=np.broadcast_to(rng.normal(size=(n,d)),x.shape)
        y=np.einsum('knij,knj->kni',m['aa'][act],x)+np.einsum('knij,knj->kni',m['ll'][act],z)
        Wy,divy=instrument(y,m['bs'])
        t=(k+1)*dt
        accumulate(stats[0],x,y,act,m,dt,t,Wx,Wy,divx,divy,g)
        if k%2==1:accumulate(stats[1],xc,y,act,m,2*dt,t,Wc,Wy,divc,divy,g)
        for j in range(kp):np.add.at(counts[j],(np.arange(n),act[j]),1)
        x=y;Wx=Wy;divx=divy
    raw=dict(T=T,n=n,seed=seed,level=level,geometry=geometry,theta=theta,
        h=h,epsilon=epsilon,target=g,policies=np.array(POLICIES),holds=np.array(HOLDS),
        observation_meshes=np.array([.005,.01]),bs=m['bs'],vs=m['vs'],
        known_deviation=m['known_deviation'],q=m['q'],kappa=m['kappa'],
        oracle_fixed=m['oracle_fixed'],oracle_gap=m['oracle_gap'],
        oracle_constant=m['oracle_constant'],terminal_state=x,action_counts=counts,
        switches=switches)
    for key in stats[0]:raw[key]=np.stack([s[key] for s in stats])
    valid=np.linalg.eigvalsh(raw['gram'])[...,0]>=m['kappa']*T/2
    terminal=np.zeros_like(raw['score'])
    terminal[valid]=np.linalg.solve(raw['gram'][valid],raw['score'][valid])
    threshold=raw['threshold_score']/g
    estimates=np.stack([terminal,threshold],axis=1)
    raw.update(estimates=estimates,squared_error=np.sum((estimates-theta)**2,axis=-1),
               coordinate_squared_error=(estimates-theta)**2,guard_valid=valid,
               estimators=np.array(['terminal','threshold']))
    return raw


def verify_manifest():
    rec=json.loads((ROOT/'metadata/hold_manifest.json').read_text())
    for name,h in rec['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','confirmation'],default='development')
    args=ap.parse_args();phase=args.phase;out=OUT/phase;out.mkdir(parents=True,exist_ok=True)
    if phase=='confirmation':verify_manifest()
    cases=[(.08,[.07,-.11])] if phase=='development' else [(0.,[0.,0.]),(.04,[0.,0.]),(.04,[.10,-.10]),(.04,[-.10,.10])]
    T,n=(4.,24) if phase=='development' else (128.,512)
    start=time.time()
    for i,(level,theta) in enumerate(cases):
        seed=(202609191700 if phase=='development' else 202609191800)+i
        raw=simulate(level,theta,T,n,seed)
        np.savez_compressed(out/f'case{i}.npz',**raw)
        print(json.dumps(dict(phase=phase,case=i,elapsed=time.time()-start,
            terminal_mse=raw['squared_error'][1,0].mean(axis=-1).tolist(),
            threshold_mse=raw['squared_error'][1,1].mean(axis=-1).tolist())),flush=True)
    (out/'metadata.json').write_text(json.dumps(dict(phase=phase,cases=len(cases),
        physical_policy_rollouts=len(cases)*len(POLICIES)*n,observation_meshes=2,
        note='Two meshes and two estimators reuse the same paths, not additional rollouts',
        elapsed_seconds=time.time()-start),indent=2)+'\n')


if __name__=='__main__':main()
