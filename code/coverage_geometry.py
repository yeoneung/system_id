"""Exact two-dimensional invariant risk; no simulated trajectory or fitted curve."""
from pathlib import Path
import csv,json
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm,solve_continuous_lyapunov
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def exact_gain(a,b,L):
    if L<2 or a<=0 or np.any(np.asarray(b)<a):raise ValueError('Need L>=2 and 0<a<=b.')
    z=np.sqrt(b/a)*np.tan(np.pi/(2*L))
    return 1+(b-a)/(b+a)*z/((1+z*z)*np.arctan(z))


def angular_gain(a,b,L):
    h=np.pi/(2*L)
    f=lambda x:a*np.cos(x)**2+b*np.sin(x)**2
    z=quad(lambda x:1/f(x),0,h,epsabs=1e-13,epsrel=1e-12)[0]
    j=quad(lambda x:1/f(x)**2,0,h,epsabs=1e-13,epsrel=1e-12)[0]/z
    return j/(.5/a+.5/b)


def main():
    out=ROOT/'results/coverage';out.mkdir(exist_ok=True)
    checks={};errors=[]
    for L in [2,3,4,8,32,128]:
        for a in [.3,1.,2.]:
            for r in [1,2,4,10,100,1000]:
                errors.append(abs(exact_gain(a,a*r,L)-angular_gain(a,a*r,L)))
                q=np.sin(np.pi/(2*L))**2
                assert exact_gain(a,a*r,L)>=2*r/((r+1)*(1+(r-1)*q))-1e-12
                assert exact_gain(a,a*r,L)<=2*r/(r+1)+1e-12
    assert max(errors)<1e-9
    checks['closed_form_vs_angular_quadrature_108_cases']=max(errors)
    # Recover the independently derived two-action result from independent quadrature.
    assert abs(1/exact_gain(1,4,2)*1.6-1.3149535200324528)<1e-9
    checks['two_action_risk_constant']=float(1.6/exact_gain(1,4,2))
    for L in [2,4,8,32]:
        assert abs(exact_gain(1,1e14,L)-1)<2e-6
    checks['fixed_dictionary_large_anisotropy_limit']=True
    for r in [2,4,100,10000]:
        assert abs(exact_gain(1,r,10000000)-2*r/(r+1))<1e-9
    checks['dense_dictionary_matches_continuum_ceiling']=True
    # Integrate the covariance on noncommuting action blocks independently
    # of the Schatten proof, checking its finite-horizon information bound.
    rng=np.random.default_rng(2026130102);slacks=[]
    for d in [2,6,8]:
        for _ in range(20):
            a=1.;b=float(rng.uniform(2,20));vs=rng.normal(size=(7,d));vs/=np.linalg.norm(vs,axis=1)[:,None]
            durations=rng.uniform(.01,.4,7);cov=np.zeros((d,d));info=0.
            for v,dt in zip(vs,durations):
                f=-b*np.eye(d)+(b-a)*np.outer(v,v);ps=solve_continuous_lyapunov(f,-np.eye(d));e=expm(f*dt)
                residual=cov-ps;end=e@residual@e.T
                integ=ps*dt+solve_continuous_lyapunov(f,end-residual)
                info+=np.trace(integ);cov=ps+end
            T=durations.sum();bound=T*(.5/a+(d-1)*.5/b)-(1-np.exp(-2*a*T))/(4*a*a)-(d-1)*(1-np.exp(-2*b*T))/(4*b*b)
            assert info<=bound+1e-10;slacks.append(bound-info)
            # One fixed action attains equality, including the transient.
            f=-b*np.eye(d)+(b-a)*np.outer(vs[0],vs[0]);ps=solve_continuous_lyapunov(f,-np.eye(d));e=expm(f*T)
            exact=ps*T+solve_continuous_lyapunov(f,ps-e@ps@e.T)
            assert abs(np.trace(exact)-bound)<1e-10
    checks['finite_horizon_information_60_noncommuting_schedules']=dict(min_slack=float(min(slacks)),constant_action_equality=True)
    # Exact independent stationary samples by Gaussian cone rejection in d=2.
    rng=np.random.default_rng(2026130101);n=200000;a=1.;b=4.;L=4
    angles=np.arange(L)*np.pi/L;v=np.column_stack([np.cos(angles),np.sin(angles)])
    component=rng.integers(L,size=n);x=rng.normal(size=(n,2))/np.sqrt(2*b)
    x+=(np.sqrt(b/a)-1)*(x*v[component]).sum(-1)[:,None]*v[component]
    accepted=np.argmax((x@v.T)**2,axis=1)==component
    energy=(x[accepted]**2).sum(-1);target=exact_gain(a,b,L)*(.5/a+.5/b)
    se=energy.std(ddof=1)/np.sqrt(len(energy));assert abs(energy.mean()-target)<5*se
    checks['gaussian_cone_sampling_crosscheck']=dict(seed=2026130101,proposals=n,accepted=int(accepted.sum()),
        estimated_energy=float(energy.mean()),exact_energy=float(target),mc_se=float(se))
    np.savez_compressed(out/'cone_check.npz',energy=energy,seed=2026130101,proposals=n,a=a,b=b,L=L)
    (out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    rows=[];ratios=np.geomspace(1,1e6,301)
    for L in [2,4,8,32]:
        for r in ratios:rows.append(dict(L=L,anisotropy=r,exact_gain=exact_gain(1,r,L),ceiling=2*r/(r+1)))
    with (out/'exact_curves.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(6.65,2.45))
    for L,color in zip([2,4,8,32],['#b34724','#a57b11','#28785a','#245ea0']):
        axes[0].semilogx(ratios,exact_gain(1,ratios,L),label=f'L={L}',color=color)
    axes[0].semilogx(ratios,2*ratios/(ratios+1),'k--',lw=1,label='All-policy ceiling')
    axes[0].set_ylim(.98,2.05);axes[0].set_xlabel('Damping anisotropy b/a');axes[0].set_ylabel('Open-loop / projection risk constant')
    axes[0].legend(fontsize=6,ncol=2);axes[0].set_title('Exact finite-dictionary risk (d=2)',fontsize=9)
    rr=np.geomspace(1.001,1e5,301);d=6
    for tau,color in zip([.01,.1,1.],['#245ea0','#28785a','#b34724']):
        q=np.minimum(.99,tau/(rr-1));gain=d*rr/((rr+d-1)*(1+(rr-1)*q))
        axes[1].semilogx(rr,gain,color=color,label=r'$(b/a-1)q\leq'+str(tau)+'$')
    axes[1].semilogx(rr,d*rr/(rr+d-1),'k--',lw=1,label='All-policy ceiling')
    axes[1].set_ylim(0,6.2);axes[1].set_xlabel('Damping anisotropy b/a');axes[1].set_ylabel('Guaranteed separation factor')
    axes[1].set_title('Theoretical coverage bounds (d=6)',fontsize=9);axes[1].legend(fontsize=6,loc='lower right')
    for ax in axes:ax.grid(alpha=.18)
    fig.tight_layout();fig.savefig(ROOT/'figures/coverage_geometry.pdf');plt.close(fig)
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
