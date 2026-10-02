"""Checks of the weighted integral and two-clock experiment implementation."""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm
from run_hold_feedback import instrument
from run_robust_feedback import build
from run_multiparameter_feedback import features
ROOT=Path(__file__).resolve().parents[1]


def main():
    rng=np.random.default_rng(202609191710);checks={}
    bs,_=features(6);x=rng.normal(size=(30,6));x=np.vstack([x,np.zeros(6)])
    W,div=instrument(x,bs)
    assert np.allclose(W,np.einsum('pij,nj->npi',bs,x),atol=1e-12)
    assert np.allclose(div,np.trace(bs,axis1=1,axis2=2),atol=1e-12)
    checks['ideal_instrument_and_ito_correction']=True
    m=build(3,.08,[.07,-.11],.005);x=rng.normal(size=(40,6));eps=1e-5
    W,div=instrument(x,m['bs']);fd=np.zeros_like(div)
    for j in range(6):
        xp=x.copy();xm=x.copy();xp[:,j]+=eps;xm[:,j]-=eps
        fd+=(instrument(xp,m['bs'])[0][...,j]-instrument(xm,m['bs'])[0][...,j])/(2*eps)
    error=float(np.max(np.abs(fd-div)));assert error<1e-7
    checks['nonlinear_instrument_divergence_vs_finite_differences']=dict(max_error=error)
    # Two exact half-step OU transitions must have the coarse transition law.
    for j in [0,17,363]:
        A=m['aa'][j];Q=m['ll'][j]@m['ll'][j].T;F=m['fs'][j]
        assert np.max(np.abs(A@A-expm(.01*F)))<1e-13
        # Van Loan independent covariance computation over .01.
        block=np.block([[F,np.eye(6)],[np.zeros((6,6)),-F.T]])
        M=expm(.01*block);Qc=M[:6,6:]@M[:6,:6].T
        assert np.max(np.abs(A@Q@A.T+Q-Qc))<1e-12
    checks['nested_observation_exact_transition_marginals']=True
    z=np.load(ROOT/'results/hold/development/case0.npz')
    rec=np.zeros_like(z['score']);good=z['guard_valid']
    rec[good]=np.linalg.solve(z['gram'][good],z['score'][good])
    assert np.allclose(rec,z['estimates'][:,0])
    assert np.allclose(z['threshold_score']/z['target'],z['estimates'][:,1])
    assert np.all(np.isnan(z['hit_time'])==~z['hit'])
    assert np.all(z['hit_time'][z['hit']]<=float(z['T'])+1e-10)
    assert np.all(z['action_counts'].sum(axis=-1)==round(float(z['T'])/.005))
    for j,D in enumerate(z['holds'][1:],1):
        assert np.all(z['switches'][j]<=np.ceil(z['T']/D)-1)
    checks['development_statistics_stopping_and_switch_budget']=True
    (ROOT/'results/hold/structural_checks.json').write_text(json.dumps(dict(check_groups=4,checks=checks),indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
