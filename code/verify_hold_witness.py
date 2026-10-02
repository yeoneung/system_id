"""Rational enclosures for the fixed mode-update-rate risk witness."""
from pathlib import Path
from fractions import Fraction as F
from math import factorial
import json
ROOT=Path(__file__).resolve().parents[1]


class I:
    def __init__(self,lo,hi=None):
        self.lo=F(lo);self.hi=F(lo if hi is None else hi);assert self.lo<=self.hi
    def __add__(self,o):
        o=o if isinstance(o,I) else I(o);return I(self.lo+o.lo,self.hi+o.hi)
    __radd__=__add__
    def __neg__(self):return I(-self.hi,-self.lo)
    def __sub__(self,o):return self+-as_i(o)
    def __rsub__(self,o):return as_i(o)+-self
    def __mul__(self,o):
        o=as_i(o);p=[x*y for x in [self.lo,self.hi] for y in [o.lo,o.hi]]
        return I(min(p),max(p))
    __rmul__=__mul__
    def __truediv__(self,o):
        o=as_i(o);assert o.lo*o.hi>0
        return self*I(1/o.hi,1/o.lo)
    def __rtruediv__(self,o):return as_i(o)/self


def as_i(x):return x if isinstance(x,I) else I(x)


def exp_neg(x):
    x=F(x);assert 0<=x<=1
    # Decreasing alternating terms bound exp(-x) between odd/even sums.
    lo=sum(((-x)**n/F(factorial(n)) for n in range(32)),F(0))
    hi=lo+x**32/F(factorial(32))
    return I(lo,hi)


def main():
    D=F(1,10);delta=eta=F(1,25);q=F(308278,10**6);beta=2-delta
    ea=exp_neg(2*D);eb=exp_neg(8*D);ep=exp_neg(beta*D)
    U=(1-ea)/2;V=(1-eb)/8
    H=(1-(1+beta*D)*ep)/(beta*beta)
    M2=(2-(2+2*beta*D+beta*beta*D*D)*ep)/(beta**3)
    K=D*H-M2
    A=(1-q)*ea+q*eb-2*delta*D*ep
    B=U+5*V-12*delta*H
    IA=(1-q)*U+q*V-2*delta*H
    IB=D*F(9,8)-U/2-5*V/8-12*delta*K
    J=(IA*B/(1-A)+IB)/D
    jol=1/(2*(1-delta))+5/(2*(4-delta))
    ratio=(1-eta)**2*J/((1+eta)**2*jol)
    # q's enclosure is certified by the retained integer-sqrt witness.
    old=json.loads((ROOT/'results/finite_time/checks.json').read_text())
    assert old['rational_finite_horizon_certificate']['covering_enclosure_verified']
    assert 2*delta*D/exp_neg(delta*D).lo < 1-q
    assert B.lo>0 and A.lo>=0 and A.hi<1
    assert J.lo>F(14392,10000) and ratio.lo>F(1064,1000)
    out=dict(check_groups=1,Delta=float(D),eta=float(eta),delta=float(delta),
        q_upper=float(q),energy_lower_certified=1.4392,ratio_lower_certified=1.064,
        exact_ratio_lower=str(ratio.lo),exact_energy_lower=str(J.lo),
        exponential_method='exact Fraction alternating Taylor enclosure, degrees 31/32',
        covering_source='retained exact integer-square-root enclosure',strict_separation=True)
    (ROOT/'results/hold/rational_witness.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if not k.startswith('exact_')},indent=2))


if __name__=='__main__':main()
