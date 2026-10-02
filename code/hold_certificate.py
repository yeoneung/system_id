"""Closed-form information bounds for projection decisions held for Delta.

The controller is sampled; the risk bound uses continuous observations.
No trajectory simulation is used in this file.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parents[1]


def ingredients(D,d=6,a=1.,b=4.,q=None,delta=0.):
    if q is None:q=1-1/np.sum(np.diff(np.sqrt(np.arange(d+1)))**2)
    beta=2*a-delta
    assert D>0 and 0<=q<1 and 0<=delta<a
    assert 2*delta*D*np.exp(delta*D)<=1-q
    def u(rate):return -np.expm1(-rate*D)/rate
    # Small-argument series avoid cancellation in the perturbation integrals.
    def moment(order):
        z=beta*D
        if z<.5:
            term=1.;total=0.
            for n in range(60):
                total+=term/(order+n+1)
                term*=-z/(n+1)
            return D**(order+1)*total
        if order==1:return (1-(1+z)*np.exp(-z))/beta**2
        return (2-(2+2*z+z*z)*np.exp(-z))/beta**3
    H=moment(1);K=D*H-moment(2)
    A=(1-q)*np.exp(-2*a*D)+q*np.exp(-2*b*D)-2*delta*D*np.exp(-beta*D)
    B=u(2*a)+(d-1)*u(2*b)-2*d*delta*H
    IA=(1-q)*u(2*a)+q*u(2*b)-2*delta*H
    IB=d*D/(2*b)+(1/(2*a)-1/(2*b))*D-u(2*a)/(2*a)-(d-1)*u(2*b)/(2*b)-2*d*delta*K
    assert 0<=A<1 and B>0 and IA>0 and IB>0
    J=(IA*B/(1-A)+IB)/D
    return dict(D=D,d=d,a=a,b=b,q=q,delta=delta,A=A,B=B,IA=IA,IB=IB,J_lower=J)


def bound(D,eta=0.,delta=0.,**kwargs):
    x=ingredients(D,delta=delta,**kwargs)
    jol=1/(2*(x['a']-delta))+(x['d']-1)/(2*(x['b']-delta))
    x.update(eta=eta,ratio_lower=(1-eta)**2*x['J_lower']/((1+eta)**2*jol))
    return x


def checks():
    checks={};errors=[]
    for delta in [0.,.04,.1]:
        for D in [.002,.02,.1,.5,1.]:
            v=ingredients(D,delta=delta);q=v['q'];beta=2-delta
            A=lambda t:(1-q)*np.exp(-2*t)+q*np.exp(-8*t)-2*delta*t*np.exp(-beta*t)
            density=lambda t:np.exp(-2*t)+5*np.exp(-8*t)-12*delta*t*np.exp(-beta*t)
            errors.extend([abs(v['IA']-quad(A,0,D)[0]),
                abs(v['B']-quad(density,0,D)[0]),
                abs(v['IB']-quad(lambda t:(D-t)*density(t),0,D)[0])])
    assert max(errors)<1e-12
    checks['closed_forms_vs_quadrature_15_cases']=dict(max_error=max(errors))
    # Algebraic sign of the ideal held-policy advantage: any fixed Delta.
    for d in [2,6,8]:
        for b in [2.,4.,10.]:
            critical=(d-1)/(b+d-1)
            for q in [.5*critical,critical,min(.99,1.1*critical)]:
                for D in [.01,.1,1.,10.]:
                    v=ingredients(D,d=d,b=b,q=q);jol=.5+(d-1)/(2*b)
                    U=-np.expm1(-2*D)/2;V=-np.expm1(-2*b*D)/(2*b)
                    gap=U*V*(b-1)*((d-1)*(1-q)-b*q)/(b*D*(1-v['A']))
                    assert abs(v['J_lower']-jol-gap)<1e-11
    checks['ideal_every_hold_interval_sign_identity']=True
    rng=np.random.default_rng(2026091901701);slacks=[]
    for _ in range(120):
        d=6;delta=.04;D=rng.uniform(.001,.5)
        v=rng.normal(size=d);v/=np.linalg.norm(v)
        H=-4*np.eye(d)+3*np.outer(v,v)
        E=rng.normal(size=(d,d));E*=delta/np.linalg.norm(E,2)
        F=H+E
        x=rng.normal(size=d);r=x@x;align=(v@x)**2/r
        t=D
        A=align*np.exp(-2*t)+(1-align)*np.exp(-8*t)-2*delta*t*np.exp(-(2-delta)*t)
        B=np.exp(-2*t)+5*np.exp(-8*t)-12*delta*t*np.exp(-(2-delta)*t)
        slacks.extend([np.linalg.norm(expm(F*t)@x)**2-A*r,
                       np.linalg.norm(expm(F*t),'fro')**2-B])
    assert min(slacks)>-1e-10
    checks['duhamel_signal_and_noise_lower_bounds_120_cases']=dict(min_slack=min(slacks))
    # Recursion summed exactly, checked against the advertised finite-N bound.
    for D in [.02,.1,.5]:
        v=ingredients(D,delta=.04);A,B=v['A'],v['B']
        for N in [1,7,100]:
            m=0.;energy=0.
            for _ in range(N):energy+=v['IA']*m+v['IB'];m=A*m+B
            closed=N*D*v['J_lower']-v['IA']*B*(1-A**N)/(1-A)**2
            assert abs(energy-closed)<1e-10
    checks['finite_horizon_information_recursion']=True
    return checks


def main():
    out=ROOT/'results/hold';out.mkdir(parents=True,exist_ok=True)
    rows=[bound(D,eta=level,delta=level) for level in [0.,.04] for D in [.02,.1,.5,1.]]
    result=dict(check_groups=4,checks=checks(),bounds=rows,
        statement='Held scheduling, continuous-observation asymptotic risk; finite-N expected information only')
    (out/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
