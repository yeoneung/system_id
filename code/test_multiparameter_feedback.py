"""Independent structural checks, not a formal proof verification."""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm
from run_multiparameter_feedback import features, model, simulate, truths

ROOT=Path(__file__).resolve().parents[1]


def main():
    rng=np.random.default_rng(2026110501);checks={}
    for d in [6,8]:
        bases,k=features(d);p=len(bases);x=rng.normal(size=(37,d))
        phi=np.einsum('pij,nj->npi',bases,x)
        gram=np.einsum('npi,nqi->npq',phi,phi)
        assert np.allclose(gram,np.sum(x*x,axis=1)[:,None,None]*np.eye(p),atol=1e-12)
        assert np.max(np.abs(k+k.T))<1e-13
        assert all(np.linalg.norm(b@k-k@b)>1e-3 for b in bases[1:])
    checks['pointwise_full_parameter_information_and_noncommutation']=True
    for d in [6,8]:
        theta=truths(d,'confirmation')[0];bases,vs,f0,aa,ll,en=model(d,'pairs',theta,.02)
        fs=f0+np.einsum('p,pij->ij',theta,bases)
        for r in [0,len(vs)-1]:
            f=fs[r];block=np.block([[f,np.eye(d)],[np.zeros((d,d)),-f.T]])
            e=expm(.02*block);q=e[:d,d:]@e[:d,:d].T
            assert np.allclose(q,ll[r]@ll[r].T,atol=1e-12)
        # This energy identity has no dependence on the unknown rotations.
        x=rng.normal(size=(20,d));v=vs[3]
        lhs=np.einsum('ni,ij,nj->n',x,fs[3],x)
        rhs=-(4-theta[0])*np.sum(x*x,axis=1)+3*(x@v)**2
        assert np.allclose(lhs,rhs,atol=1e-11)
    checks['exact_transition_covariance_against_van_loan']=True
    checks['radial_generator_identity_with_unknown_rotations']=True
    # Independent noncommuting propagator checks with unknown skew parts.
    for d in [6,8]:
        for _ in range(25):
            theta=truths(d,'confirmation')[1]
            bs,vs,f0,_,_,_=model(d,'axes',theta,.02)
            fs=f0+np.einsum('p,pij->ij',theta,bs)
            durations=rng.uniform(.01,.3,9);actions=rng.integers(d,size=9)
            prop=np.eye(d)
            for h,a in zip(durations,actions):prop=expm(h*fs[a])@prop
            tau=durations.sum();bound=np.exp(-2*(1-theta[0])*tau)+(d-1)*np.exp(-2*(4-theta[0])*tau)
            assert np.linalg.norm(prop,'fro')**2<=bound+1e-12
    checks['skew_perturbed_schatten_propagator_bound_50_cases']=True
    rows,raw=simulate(8,'pairs',truths(8,'confirmation')[0],1.,16,.02,2026110502)
    assert np.all(raw['action_counts'].sum(axis=-1)==50)
    assert np.allclose(raw['squared_error'],((raw['estimates']-raw['theta'])**2).sum(axis=-1))
    assert np.allclose(raw['estimates']*raw['gram'][:,:,None],raw['score'])
    assert all(row['guard_failures']==0 for row in rows)
    checks['raw_score_loss_budget_and_parameter_accounting']=True
    # Conditional scores of the exact Gaussian one-step experiment, estimated
    # from central differences, verify the simulator depends on every parameter.
    for d in [6,8]:
        theta=truths(d,'confirmation')[0];p=len(theta);derivatives=[]
        for i in range(p):
            h=np.zeros(p);h[i]=1e-5
            ap=model(d,'axes',theta+h,.02)[3][0]
            am=model(d,'axes',theta-h,.02)[3][0]
            derivatives.append(((ap-am)/2e-5).reshape(-1))
        assert np.linalg.matrix_rank(np.array(derivatives),tol=1e-8)==p
    checks['sampled_transition_sensitive_to_all_unknown_coefficients']=True
    out=ROOT/'results'/'multiparameter';out.mkdir(exist_ok=True)
    (out/'structural_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
