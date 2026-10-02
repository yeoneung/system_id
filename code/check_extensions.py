"""Two-loop quadrature, bounded-grid HJB, and discrete-time simulations.

The finite-difference HJB solves a bounded reflecting grid problem.
Grid residuals are not whole-space error certificates.
"""
from pathlib import Path
import csv
import json
import math
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.sparse import bmat, coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

ROOT = Path(__file__).resolve().parents[1]


def continuous_value(a, b, angle):
    c = (b-a)*np.sin(angle)/2
    t = c/np.sqrt(a*b)
    gap = 1+2*t/((1+t*t)*(np.pi+2*np.arctan(t)))
    return (1/(2*a)+1/(2*b))*gap


def quadrature_value(a, b, angle):
    A = b-(b-a)*np.cos(angle/2)**2
    B = b-(b-a)*np.sin(angle/2)**2
    c = (b-a)*np.sin(angle)/2
    def radial(phi):
        return A*np.cos(phi)**2+B*np.sin(phi)**2-2*c*np.cos(phi)*np.sin(phi)
    i1 = quad(lambda phi: 1/radial(phi), 0, np.pi/2, epsabs=1e-10, epsrel=1e-11)[0]
    i2 = quad(lambda phi: 1/radial(phi)**2, 0, np.pi/2, epsabs=1e-10, epsrel=1e-11)[0]
    return i2/i1


def generator(matrix, radius, count):
    grid = np.linspace(-radius, radius, count)
    step = grid[1]-grid[0]
    x, y = np.meshgrid(grid, grid, indexing='ij')
    points = np.column_stack([x.ravel(), y.ravel()])
    drift = points @ matrix.T
    n = count*count
    indices = np.arange(n)
    rows, columns, values = [], [], []
    diag = np.zeros(n)
    for axis, jump in [(0, count), (1, 1)]:
        coordinate = np.repeat(np.arange(count), count) if axis == 0 else np.tile(np.arange(count), count)
        for direction in [-1, 1]:
            valid = (coordinate > 0) if direction == -1 else (coordinate < count-1)
            rate = .5/step**2+direction*drift[:, axis]/(2*step)
            # Reflecting ghost point: zero normal derivative cancels drift,
            # doubles the inward diffusion rate at a boundary.
            boundary = (coordinate == count-1) if direction == -1 else (coordinate == 0)
            rate[boundary] = 1/step**2
            assert np.min(rate[valid]) >= -1e-12, (radius, count, float(np.min(rate[valid])))
            rows.append(indices[valid]); columns.append(indices[valid]+direction*jump)
            values.append(rate[valid]); diag[valid] -= rate[valid]
    rows.append(indices); columns.append(indices); values.append(diag)
    Q = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))), shape=(n,n)).tocsr()
    assert np.max(np.abs(np.asarray(Q.sum(axis=1)).ravel())) < 1e-10
    return Q, points


def evaluate_policy(generators, policy, reward):
    n = len(reward)
    Q = sum(Q.multiply((policy == i)[:, None]) for i, Q in enumerate(generators)).tocsr()
    anchor = csr_matrix(([1.], ([0], [n//2])), shape=(1, n))
    augmented = bmat([[Q, -np.ones((n, 1))], [anchor, csr_matrix((1, 1))]], format='csc')
    answer = spsolve(augmented, np.r_[-reward, 0.])
    h, value = answer[:-1], float(answer[-1])
    residual = float(np.max(np.abs(reward+Q@h-value)))
    return h, value, residual


def hjb_case(angles, radius, count, a=1., b=4.):
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    generators = []
    for v in directions:
        Q, points = generator(-b*np.eye(2)+(b-a)*np.outer(v,v), radius, count)
        generators.append(Q)
    reward = np.sum(points**2, axis=1)
    projection = np.argmax((points @ directions.T)**2, axis=1)
    policy = projection.copy()
    projection_value = None
    history = []
    for iteration in range(30):
        h, value, residual = evaluate_policy(generators, policy, reward)
        if projection_value is None:
            projection_value = value
        scores = np.array([Q@h for Q in generators])
        best = np.argmax(scores, axis=0)
        advantage = scores[best, np.arange(len(policy))]-scores[policy, np.arange(len(policy))]
        improve = advantage > 1e-9
        history.append(dict(value=value, switched=int(improve.sum()), evaluation_residual=residual))
        if not improve.any():
            break
        policy[improve] = best[improve]
    else:
        raise AssertionError('Policy iteration did not converge')
    bellman = reward+scores.max(axis=0)-value
    result = dict(angles_radians=list(map(float, angles)), radius=radius, count=count,
                  spacing=2*radius/(count-1), projection_grid_value=projection_value,
                  optimal_grid_value=value, improvement_percent=100*(value/projection_value-1),
                  grid_bellman_residual=float(np.max(np.abs(bellman))),
                  policy_changed_fraction=float(np.mean(policy != projection)), history=history)
    if len(angles) == 2:
        angle=abs(angles[1]-angles[0])
        angle=min(angle,np.pi-angle)
        exact=continuous_value(a,b,angle)
        result.update(exact_whole_space_value=exact, absolute_grid_error=abs(value-exact))
        assert abs(value-projection_value) < 1e-7
    return result, points, policy, projection, h


def discrete_checks():
    rng=np.random.default_rng(72915)
    slacks=[]; exact_errors=[]
    alpha=.85; beta=.4; k=alpha-beta
    for d in [2,3,6]:
        for repeat in range(30):
            Phi=np.eye(d)
            for m in range(1,16):
                U,_=np.linalg.qr(rng.normal(size=(d,d)))
                weights=rng.dirichlet(np.ones(d))*k
                K=(U*weights)@U.T
                Phi=(beta*np.eye(d)+K)@Phi
                slacks.append(alpha**(2*m)+(d-1)*beta**(2*m)-np.linalg.norm(Phi,'fro')**2)
        A=np.diag([alpha]+[beta]*(d-1))
        for m in range(1,20):
            exact_errors.append(abs(np.linalg.norm(np.linalg.matrix_power(A,m),'fro')**2-(alpha**(2*m)+(d-1)*beta**(2*m))))
    assert min(slacks) > -1e-12
    assert max(exact_errors) < 1e-12
    thresholds=np.linspace(0,5,501)
    for small,large in [(0,.1),(.2,1.),(1.,2.)]:
        cdf_small=ndtr(thresholds-small)-ndtr(-thresholds-small)
        cdf_large=ndtr(thresholds-large)-ndtr(-thresholds-large)
        assert np.max(cdf_large-cdf_small) < 1e-12
    return dict(nonadaptive_product_checks=len(slacks),minimum_slack=min(slacks),
                constant_product_equality_max_error=max(exact_errors), folded_normal_cdf_order=True)


def simulate_discrete(alpha=.85,beta=.4,paths=4096,horizons=(256,1024)):
    rng=np.random.default_rng(370102)
    x=np.zeros((3,paths,2)); G=np.zeros((3,paths)); score=np.zeros_like(G); power=np.zeros_like(G)
    rows=[]; selected=np.zeros((3,paths),dtype=int)
    for n in range(max(horizons)):
        selected[0]=0; selected[1]=n%2; selected[2]=np.argmax(x[2]**2,axis=1)
        w=rng.normal(size=(paths,2))
        G+=np.sum(x*x,axis=2)
        score+=np.sum(x*w[None,:,:],axis=2)
        pick=x[np.arange(3)[:,None],np.arange(paths)[None,:],selected]
        power+=(alpha-beta)**2*pick**2
        updated=beta*x+w[None,:,:]
        updated[np.arange(3)[:,None],np.arange(paths)[None,:],selected]+=(alpha-beta)*pick
        x=updated
        N=n+1
        if N in horizons:
            estimates=np.where(G>=N,score/np.maximum(G,1e-300),0)
            errors=estimates**2
            for i,name in enumerate(['constant','alternating','projection']):
                rows.append(dict(horizon=N,policy=name,paths=paths,alpha=alpha,beta=beta,
                    mean_information_per_step=float(G[i].mean()/N),
                    mean_input_power=float(power[i].mean()/N),
                    scaled_mse=float(N*errors[i].mean()),
                    scaled_mse_standard_error=float(N*errors[i].std(ddof=1)/np.sqrt(paths)),
                    paired_mse_difference_vs_constant=float(np.mean(errors[i]-errors[0])),
                    paired_difference_standard_error=float(np.std(errors[i]-errors[0],ddof=1)/np.sqrt(paths)),
                    threshold_failures=int(np.sum(G[i]<N))))
    return rows


def main():
    start=time.time()
    (ROOT/'results').mkdir(exist_ok=True);(ROOT/'figures').mkdir(exist_ok=True)
    root=brentq(lambda t:(1-t*t)*(np.pi+2*np.arctan(t))-2*t,0,1)
    errors=[]
    for a in [.3,1.,2.]:
        for ratio in [1.2,2,4,10,100]:
            for angle in [.1,.4,1.,np.pi/2]:
                exact=continuous_value(a,a*ratio,angle)
                errors.append(abs(exact-quadrature_value(a,a*ratio,angle))/exact)
    assert max(errors)<1e-9
    check=dict(t_star=root,sharp_two_loop_gap=2/(1+root*root),
               maximal_mse_reduction=1-(1+root*root)/2,
               anisotropy_threshold=(root+np.sqrt(1+root*root))**2,
               formula_quadrature_cases=len(errors),formula_max_relative_error=max(errors),
               discrete=discrete_checks())
    results=[]
    cases=[('orthogonal',[-np.pi/4,np.pi/4]),
           ('oblique',[-np.pi/12,np.pi/12]),
           ('irregular_three',[0.,.35,1.35])]
    last={}
    for name,angles in cases:
        resolutions=[(3.,121),(4.,161),(4.,241)]
        if name == 'irregular_three':
            resolutions += [(4.,321),(5.,301)]
        for radius,count in resolutions:
            item,points,policy,projection,h=hjb_case(np.array(angles),radius,count)
            item['case']=name;results.append(item)
            print(name,radius,count,item['projection_grid_value'],item['optimal_grid_value'],item['improvement_percent'],flush=True)
            last[name]=(points,policy,projection,h,count,radius)
    check['hjb_scope']='Bounded reflecting finite-difference grids; no whole-space numerical certificate.'
    check['hjb']=results
    rows=simulate_discrete()
    with (ROOT/'results/discrete_identification.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    check['discrete_identification']=rows
    check['elapsed_seconds']=time.time()-start
    (ROOT/'results/extension_checks.json').write_text(json.dumps(check,indent=2)+'\n')
    angles=np.linspace(.001,np.pi/2,500)
    fig,ax=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for ratio in [2,4,10,100]:
        gaps=[continuous_value(1,ratio,t)/(.5+.5/ratio) for t in angles]
        ax[0].plot(angles*180/np.pi,gaps,label=f'b/a = {ratio}')
    ax[0].axhline(check['sharp_two_loop_gap'],ls='--',c='black',lw=1,label='Sharp two-loop upper bound')
    ax[0].set(xlabel='Angle between two unoriented loops (degrees)',ylabel='Exact optimal risk ratio')
    ax[0].legend(fontsize=8)
    x,policy,projection,h,count,radius=last['irregular_three']
    difference=(policy!=projection).reshape(count,count)
    ax[1].imshow(difference.T,origin='lower',extent=(-radius,radius,-radius,radius),cmap='Blues',interpolation='nearest')
    ax[1].set(xlim=(-2.5,2.5),ylim=(-2.5,2.5),xlabel='$x_1$',ylabel='$x_2$',title='Three-loop grid: improved policy differs')
    fig.savefig(ROOT/'figures/exact_two_loop_and_hjb.pdf');plt.close(fig)
    print(json.dumps({k:v for k,v in check.items() if k not in ['hjb','discrete_identification']},indent=2),flush=True)


if __name__=='__main__':
    main()
