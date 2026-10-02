"""Information and power checks; no stochastic trajectories or fitted parameters."""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/capacity_information'


def switched_case(d, blocks, theta=0., omega=None):
    a,b=1.-theta,4.-theta;k=3.
    if omega is None:omega=np.zeros((d,d))
    # Independent integration of transition, covariance, information, and power.
    state=np.r_[np.eye(d).ravel(),np.zeros(d*d+2)]
    transition=np.eye(d);clock=0.;errors=[];violations=[]
    for gain,duration in blocks:
        f=-b*np.eye(d)+gain+omega
        def rhs(t,y):
            phi=y[:d*d].reshape(d,d);cov=y[d*d:2*d*d].reshape(d,d)
            return np.r_[(f@phi).ravel(),(f@cov+cov@f.T+np.eye(d)).ravel(),np.trace(cov),np.trace(gain@gain@cov)]
        sol=solve_ivp(rhs,(clock,clock+duration),state,rtol=2e-11,atol=3e-12,
                      t_eval=np.linspace(clock,clock+duration,13))
        assert sol.success
        for t,y in zip(sol.t,sol.y.T):
            phi=y[:d*d].reshape(d,d);cov=y[d*d:2*d*d].reshape(d,d)
            c=-np.expm1(-2*a*t)/(2*a)
            spectral=np.exp(-2*a*t)+(d-1)*np.exp(-2*b*t)
            energy=c+(d-1)*(-np.expm1(-2*b*t))/(2*b)
            violations.append([np.sum(phi*phi)-spectral,np.linalg.eigvalsh(cov)[-1]-c,
                               np.trace(cov)-energy,np.trace(gain@gain@cov)-k*k*c])
        transition=expm(f*duration)@transition
        state=sol.y[:,-1];clock+=duration
        errors.append(float(np.linalg.norm(transition-state[:d*d].reshape(d,d))))
    return np.max(violations,axis=0),max(errors),state,clock


def two_blocks(T,theta=0.):
    a,b=1.-theta,4.-theta;k=3.;cov=np.zeros(2);g=power=0.
    for index in [0,1]:
        rates=np.array([b,b]);rates[index]=a
        delta=T/2;stationary=1/(2*rates)
        integrals=stationary*delta+(cov-stationary)*(-np.expm1(-2*rates*delta))/(2*rates)
        g+=sum(integrals);power+=k*k*integrals[index]
        cov=stationary+(cov-stationary)*np.exp(-2*rates*delta)
    return g/T,power/T,cov


def main():
    OUT.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(2026092301)
    cases=[]
    for d in [2,6,8]:
        for j in range(12):
            omega=rng.normal(size=(d,d));omega=.4*(omega-omega.T)
            blocks=[]
            for i in range(5):
                q,_=np.linalg.qr(rng.normal(size=(d,d)))
                lam=np.zeros(d)
                if i%2:lam=3*rng.dirichlet(np.ones(d))
                else:lam[0]=3
                if j%4==0:lam*=.7
                blocks.append(((q*lam)@q.T,float(rng.uniform(.01,.5))))
            violation,error,_,_=switched_case(d,blocks,theta=float(rng.uniform(-.2,.2)),omega=omega)
            cases.append(dict(d=d,max_violation=violation.tolist(),ode_vs_expm=error))
    worst=np.max([x['max_violation'] for x in cases],axis=0)
    assert np.max(worst)<2e-9
    assert max(x['ode_vs_expm'] for x in cases)<2e-9
    equality=[]
    for d in [2,6,8]:
        gain=np.diag([3.]+[0.]*(d-1));T=1.3
        _,_,s,_=switched_case(d,[(gain,T)])
        exact_g=T*(.5+(d-1)/8)-(1-np.exp(-2*T))/4-(d-1)*(1-np.exp(-8*T))/64
        exact_p=4.5*(T-(1-np.exp(-2*T))/2)
        equality.append(max(abs(s[-2]-exact_g),abs(s[-1]-exact_p)))
    assert max(equality)<2e-9
    # Full-circle quadrature, rather than the closed-form cell derivation.
    c=lambda t:4.-3.*max(np.cos(t)**2,np.sin(t)**2)
    cuts=np.arange(0,9)*np.pi/4
    integrals=[sum(quad(lambda t:c(t)**(-m),lo,hi,epsabs=1e-12,epsrel=1e-12)[0]
                   for lo,hi in zip(cuts[:-1],cuts[1:])) for m in [1,2]]
    j_quad=integrals[1]/integrals[0]
    j_exact=.625+3/(20*np.arctan(2.))
    prob=2*quad(lambda t:1/c(t),-np.pi/4,np.pi/4,epsabs=1e-12)[0]/integrals[0]
    assert abs(j_exact-j_quad)<1e-12 and abs(prob-.5)<1e-12
    convergence=[]
    for theta in [-.2,0.,.2]:
        for T in [8.,32.,128.,512.,2048.]:
            j,power,cov=two_blocks(T,theta);target=.5/(1-theta)+.5/(4-theta)
            assert abs(j-target)*T<1.
            # Exact finite-horizon rank-one power balance for drift -b+theta.
            assert abs(power-3*((4-theta)*j-1+sum(cov)/(2*T)))<1e-12
            convergence.append(dict(theta=theta,T=T,J=j,P=power,T_information_deficit=T*(target-j)))
    output=dict(scientific_check_groups=5,new_policy_rollouts=0,
        check_groups=['noncommuting_transition_and_covariance_bounds','nonadaptive_power_bound',
                      'constant_loop_finite_horizon_equalities','full_circle_quadrature_and_equal_frequencies',
                      'two_block_information_and_power_transients'],
        bounds=dict(cases=len(cases),max_violations=worst.tolist(),
                    max_ode_vs_expm_error=max(x['ode_vs_expm'] for x in cases)),
        equality_max_error=max(equality),quadrature_error=abs(j_exact-j_quad),
        example=dict(d=2,p=1,a=1,b=4,limiting_loop_frequencies=[.5,.5],
                     blocks=dict(J=.625,C=1.6,P=4.5),
                     projection=dict(J=j_exact,C=1/j_exact,P=3*(4*j_exact-1)),
                     risk_reduction_percent=100*(1-1/j_exact/1.6),occupation_quadrature=prob),
        two_block_convergence=convergence)
    (OUT/'checks.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='two_block_convergence'},indent=2))

if __name__=='__main__':main()
