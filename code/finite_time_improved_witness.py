"""A less conservative local-box choice for the finite-time risk bound."""
from pathlib import Path
from fractions import Fraction as F
from math import factorial,isqrt
import json
ROOT=Path(__file__).resolve().parents[1]


def main():
    scale=10**60
    roots=[(F(isqrt(i*scale*scale),scale),F(isqrt(i*scale*scale)+1,scale)) for i in range(7)]
    roots[0]=(F(0),F(0))
    slo=sum((roots[i][0]-roots[i-1][1])**2 for i in range(1,7))
    shi=sum((roots[i][1]-roots[i-1][0])**2 for i in range(1,7))
    assert F(3,10)<1-1/slo and 1-1/shi<F(308278,10**6)
    assert F(1415,1000)**2>2 and F(355,113)**2<F(987,100)
    T=65536;h=F(3,50);eps=F(9,1000);eta=delta=F(1,25)
    e=delta+(1+eta*F(1415,1000))*h
    c=1+3*F(308278,10**6)+e
    lower=2/((1+eta)**2*(1/(2*(1-e))+5/(2*(4-e)))+F(987,100)/(T*h*h))
    # c_h>1+3(.3)+.04+.06=2; tail exponent is strictly below -16.
    assert 6*2*T*eps*eps/(4*(1-eps))>16
    assert sum(F(16**n,factorial(n)) for n in range(51))>8_000_000
    tail=(2/(2-eps))**3/F(8_000_000)
    upper=4*c/((1-eta)**2*6*(1-eps))+2*T*h*h*tail
    assert upper<F(1481,1000)<F(1493,1000)<lower
    out=dict(check_groups=1,T=T,h=float(h),epsilon=float(eps),
        displayed_upper=1.481,displayed_lower=1.493,exact_upper=str(upper),exact_lower=str(lower),
        covering_enclosure_verified=True,exponential_taylor_enclosure_verified=True,
        strict_separation=True,new_trajectory_rollouts=0,
        scope='Continuous projection and observation; different local box from the retained T=1e6 witness')
    (ROOT/'results/finite_time/improved_witness.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if not k.startswith('exact_')},indent=2))


if __name__=='__main__':main()
