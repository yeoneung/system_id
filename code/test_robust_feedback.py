"""Numerical checks of the new perturbation and resource arguments."""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm
from run_robust_feedback import directions,select,build,simulate
ROOT=Path(__file__).resolve().parents[1]


def main():
    rng=np.random.default_rng(2026091970);checks={}
    # Full signed-subset covering: random vectors and the analytic worst case.
    for d in [2,3,6,8]:
        vs,lookup,q=directions(d)
        x=rng.normal(size=(300,d));x/=np.linalg.norm(x,axis=1)[:,None]
        y=np.diff(np.sqrt(np.arange(d+1)));y/=np.linalg.norm(y)
        assert np.min(np.max((x@vs.T)**2,axis=1))>=1-q-1e-12
        assert abs(np.max((vs@y)**2)-(1-q))<1e-12
        x=np.vstack([x,y,np.zeros(d),np.ones(d),np.eye(d)])
        assert np.array_equal(select(x,vs,lookup),np.argmax((x@vs.T)**2,axis=1))
    checks['signed_subset_coverage_and_fast_selector']=True
    # Loewner frame sandwich and information bound, including nonscalar geometry.
    maxgap=0.;deviation=[]
    for geometry in range(4):
        for level in [.04,.10]:
            m=build(geometry,level,np.zeros(2),.02)
            for x in rng.normal(size=(30,6)):
                phi=m['bs']@x;g=phi@phi.T;norm=x@x
                assert np.linalg.eigvalsh(g)[0]>=(1-level)**2*norm-1e-10
                assert np.linalg.eigvalsh(g)[-1]<=(1+level)**2*norm+1e-10
                deviation.append(np.linalg.norm(g/np.trace(g)*2-np.eye(2)))
            jol=1/(2*(1-level))+5/(2*(4-level))
            assert np.max(np.linalg.eigvalsh(m['infos']))<=(1+level)**2*jol+1e-10
            assert m['gamma']>0 and m['oracle_constant']<=np.min(np.trace(np.linalg.inv(m['infos']),axis1=1,axis2=2))+1e-8
            maxgap=max(maxgap,m['oracle_gap'])
    assert min(deviation)>1e-6
    checks['perturbed_frame_stationary_information_and_oracles']={'max_solver_gap':maxgap,'min_sampled_nonscalar_deviation':min(deviation)}
    # Noncommuting transition products for the enlarged PSD trace-budget class.
    slack=[]
    for d in [2,6,8]:
        for trial in range(20):
            a=1.;b=4.;delta=.08;tau=0.;phi=np.eye(d)
            for _ in range(6):
                q,_=np.linalg.qr(rng.normal(size=(d,d)))
                eig=rng.dirichlet(np.ones(d))*(b-a)*rng.uniform(0,1)
                k=(q*eig)@q.T
                e=rng.normal(size=(d,d));e*=delta/np.linalg.norm(e,2)
                rot=rng.normal(size=(d,d));rot=.3*(rot-rot.T)
                dt=rng.uniform(.01,.2);tau+=dt
                phi=expm((-b*np.eye(d)+k+e+rot)*dt)@phi
            bound=np.exp(-2*(a-delta)*tau)+(d-1)*np.exp(-2*(b-delta)*tau)
            slack.append(bound-np.linalg.norm(phi,'fro')**2)
    assert min(slack)>-1e-11
    checks['sixty_convex_gain_noncommuting_propagators']={'min_slack':min(slack)}
    # The extreme rank-one allocation saturates the spectral envelope.
    for d in [2,6,8]:
        for tau in [.01,.5,2.]:
            f=-4*np.eye(d);f[0,0]= -1
            assert abs(np.linalg.norm(expm(f*tau),'fro')**2-(np.exp(-2*tau)+(d-1)*np.exp(-8*tau)))<1e-12
    checks['rank_one_extreme_equality']=True
    # Local-risk bound reduces to the ideal bound and certifies the declared 6D cell.
    vs,_,q=directions(6)
    ideal=6/(2*(1+3*q)*(1/2+5/8))
    m0=build(0,0.,np.zeros(2),.02);m4=build(0,.04,np.zeros(2),.02)
    assert abs(m0['certified_factor']-ideal)<1e-12
    assert m4['certified_factor']>1.12
    checks['robust_bound_and_finite_mode_certificate']={'q':q,'ideal':ideal,'eta_delta_004':m4['certified_factor']}
    # An independent algebraic reconstruction of scores/Grams and reported loss.
    rows,raw=simulate(7,.06,2.,16,.02,2026091971,[.08,-.12])
    good=raw['guard_valid'];rec=np.zeros_like(raw['estimates'])
    rec[good]=np.linalg.solve(raw['gram'][good],raw['score'][good])
    assert np.allclose(rec,raw['estimates'])
    assert np.allclose(np.sum((rec-raw['theta'])**2,axis=-1),raw['squared_error'])
    assert np.all(raw['action_counts'].sum(axis=-1)==100)
    assert np.all(np.linalg.eigvalsh(raw['gram'])>=-1e-12)
    checks['matrix_estimator_loss_and_schedule_accounting']=True
    out=ROOT/'results/robustness';out.mkdir(parents=True,exist_ok=True)
    (out/'structural_checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
