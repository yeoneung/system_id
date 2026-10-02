"""Analytical finite-time bounds and consistency checks; no trajectory simulation."""
from pathlib import Path
from fractions import Fraction as F
from math import isqrt, factorial
import csv, json
import numpy as np
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/finite_time'


def log_laplace(d,c,T,lam):
    s=np.sqrt(c*c+2*lam)
    logden=s*T+np.log((1+c/s)/2+(1-c/s)/2*np.exp(-2*s*T))
    return d/2*(c*T-logden)


def bounds(T,epsilon,eta=.04,delta=.04):
    d,p,a,b=6,2,1.,4.
    q=1-1/np.sum(np.diff(np.sqrt(np.arange(d+1)))**2)
    h=T**(-1/3); e=delta+(1+eta*np.sqrt(p))*h
    assert h<=.15 and e<a and 0<epsilon<1
    c=a+(b-a)*q+e
    logtail=d/2*np.log(2/(2-epsilon))-d*c*T*epsilon**2/(4*(1-epsilon))
    tail=np.exp(min(0.,logtail))
    lower=p/((1+eta)**2*(1/(2*(a-e))+(d-1)/(2*(b-e)))+np.pi**2/(T*h*h))
    upper=2*p*c/((1-eta)**2*d*(1-epsilon))+p*T*h*h*tail
    return dict(T=T,epsilon=epsilon,eta=eta,delta=delta,h=h,
                openloop_lower=lower,feedback_upper=upper,tail_upper=tail,
                certified=bool(upper<lower))


def rational_witness():
    # Exact rational enclosures for every sqrt in q. Integer square roots
    # guarantee containment without assuming floating-point rounding direction.
    scale=10**60
    roots=[(F(isqrt(i*scale*scale),scale),F(isqrt(i*scale*scale)+1,scale)) for i in range(7)]
    roots[0]=(F(0),F(0))
    sup=sum((roots[i][1]-roots[i-1][0])**2 for i in range(1,7))
    q_upper=1-1/sup
    assert q_upper<F(308278,10**6)
    assert F(1415,1000)**2>2
    # Use the classical pi < 355/113 enclosure; the remaining arithmetic is exact.
    assert F(355,113)**2<F(987,100)
    eta=delta=F(1,25);h=F(1,100);T=10**6;eps=F(1,250)
    e=delta+(1+eta*F(1415,1000))*h
    c=1+3*F(308278,10**6)+e
    lower=2/((1+eta)**2*(1/(2*(1-e))+5/(2*(4-e)))+F(987,100)/(T*h*h))
    assert 6*T*eps*eps/(4*(1-eps))>24
    assert sum(F(24**n,factorial(n)) for n in range(61))>20_000_000_000
    tail=(2/(2-eps))**3/F(20_000_000_000)
    upper=4*c/((1-eta)**2*6*(1-eps))+2*T*h*h*tail
    assert upper<F(1435,1000)<F(1478,1000)<lower
    return dict(T=T,h=float(h),epsilon=float(eps),
                exact_rational_lower=str(lower),exact_rational_upper=str(upper),
                displayed_lower=1.478,displayed_upper=1.435,
                covering_enclosure_verified=True,exponential_taylor_enclosure_verified=True,
                pi_enclosure_used='pi < 355/113',strict_separation=True)


def checks():
    errors=[]
    for d in [1,2,6]:
        for c,T,lam in [(.7,.4,.15),(1.,2.,.5),(3.,1.2,2.),(.2,5.,.8)]:
            def ode(_,y):
                B,A=y;return [lam-2*c*B-2*B*B,-d*B]
            sol=solve_ivp(ode,[0,T],[0.,0.],rtol=1e-11,atol=1e-13)
            errors.append(abs(sol.y[1,-1]-log_laplace(d,c,T,lam)))
    assert max(errors)<1e-9
    # The chosen Chernoff parameter yields the stated simplified prefactor/exponent.
    for d,c,T,eps in [(6,1.9,128.,.2),(2,.3,9.,.4),(8,4.,32.,.07)]:
        g=(1-eps)*d*T/(2*c)
        lam=c*c*eps*(2-eps)/(2*(1-eps)**2)
        exact=lam*g+log_laplace(d,c,T,lam)
        envelope=d/2*np.log(2/(2-eps))-d*c*T*eps**2/(4*(1-eps))
        assert exact<=envelope+1e-11
    rng=np.random.default_rng(20260919016);maxerr=0.;minslack=np.inf
    for case in [1,2,3,4]:
        z=np.load(ROOT/f'results/robustness/confirmation/case{case}.npz')
        bs,vs,E=z['bs'],z['vs'],z['known_deviation'];eta=float(z['eta']);delta=float(z['delta'])
        for _ in range(64):
            x=rng.normal(size=6);rad=x@x
            psi=np.einsum('pij,j->ip',bs,x)
            Q=psi.T@psi;W=rad*np.linalg.solve(Q,psi.T)
            maxerr=max(maxerr,float(np.linalg.norm(W@psi-rad*np.eye(2))))
            assert np.linalg.eigvalsh(rad/(1-eta)**2*np.eye(2)-W@W.T)[0]>-1e-10
            theta=rng.uniform(-.1,.1,2);v=vs[np.argmax((vs@x)**2)]
            drift=(-4*np.eye(6)+3*np.outer(v,v)+E+np.einsum('p,pij->ij',theta,bs))@x
            c=1+3*float(z['q'])+delta+(1+eta*np.sqrt(2))*.1
            minslack=min(minslack,float(x@drift+c*rad))
    assert maxerr<1e-10 and minslack>=-1e-10
    witness=rational_witness()
    return dict(riccati_transform_12_cases=dict(max_log_error=max(errors)),
                chernoff_envelope_three_cases=True,
                matrix_instrument_and_radial_bound_256_cases=dict(max_error=maxerr,min_radial_slack=minslack),
                rational_finite_horizon_certificate=witness)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    result=checks()
    rows=[bounds(T,eps,level,level) for level in [0.,.04]
          for T,eps in [(1e3,.05),(1e4,.02),(1e5,.008),(1e6,.004)]]
    with (OUT/'bounds.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result.update(new_trajectory_rollouts=0,check_groups=4,rows=rows,
                  statement='Continuous observation, instrument-weighted energy-threshold estimator; not terminal or sampled LS')
    (OUT/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
