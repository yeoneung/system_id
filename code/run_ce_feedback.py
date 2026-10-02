"""Prospective stationary certainty-equivalent (CE) comparison.

CE interpolates a precomputed Lyapunov-energy table on the known box.
All policies use the same decision clock; two observation meshes reuse
the same exact-OU physical paths. No confirmation loss tunes the design.
"""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from run_multiparameter_feedback import model, features, dictionary, truths

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'ce'
POLICIES = ['oracle_fixed', 'stationary_ce', 'projection']
WIDTH = np.array([.25, .4])
KAPPA = 6 / (2 * (4 + WIDTH[0]))
FINE = .005
OBS = .01
HOLD = .10


def exact_energy(theta):
    bases, known = features(6)
    vs = dictionary(6, 'pairs')
    fs = (-4 * np.eye(6)[None] + 3 * np.einsum('li,lj->lij', vs, vs)
          + known + np.einsum('p,pij->ij', theta, bases))
    return np.array([np.trace(solve_continuous_lyapunov(f, -np.eye(6))) for f in fs])


def build_table():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    axes = [np.linspace(-w, w, round(2*w/.005)+1) for w in WIDTH]
    energies = np.empty((len(axes[0]), len(axes[1]), 36))
    for i, a in enumerate(axes[0]):
        for j, b in enumerate(axes[1]):
            energies[i, j] = exact_energy([a, b])
    np.savez_compressed(OUT/'energy_table.npz', damping=axes[0], rotation=axes[1], energy=energies)
    print(json.dumps(dict(table_shape=list(energies.shape), elapsed=time.time()-t0)), flush=True)


def interpolate(theta, table):
    axes, values = table
    pos = [(np.clip(theta[:, i], a[0], a[-1])-a[0])/(a[1]-a[0]) for i, a in enumerate(axes)]
    idx = [np.clip(np.floor(s).astype(int), 0, len(a)-2) for s, a in zip(pos, axes)]
    w = [s-i for s, i in zip(pos, idx)]
    i, j = idx; x, y = [z[:, None] for z in w]
    return ((1-x)*(1-y)*values[i,j]+x*(1-y)*values[i+1,j]
            +(1-x)*y*values[i,j+1]+x*y*values[i+1,j+1])


def load_table():
    z = np.load(OUT/'energy_table.npz')
    return ([z['damping'], z['rotation']], z['energy'])


def table_check(points, table):
    approx = interpolate(points, table)
    exact = np.array([exact_energy(t) for t in points])
    choice = approx.argmax(axis=1)
    regret = (exact.max(axis=1)-exact[np.arange(len(points)), choice])/exact.max(axis=1)
    return dict(points=len(points), max_relative_score_error=float(np.max(np.abs(approx-exact)/exact)),
                mismatched_choices=int(np.sum(choice != exact.argmax(axis=1))),
                max_relative_action_regret=float(regret.max()))


def increment(x, y, act, dt, bases, f0):
    ax = np.einsum('ij,knj->kni', bases[1], x)
    ay = np.einsum('ij,knj->kni', bases[1], y)
    fx = np.einsum('knij,knj->kni', f0[act], x)
    fy = np.einsum('knij,knj->kni', f0[act], y)
    nx = np.sum(x*x, axis=-1); ny = np.sum(y*y, axis=-1)
    s0 = .5*(ny-nx)-3*dt-.5*dt*(np.sum(x*fx, axis=-1)+np.sum(y*fy, axis=-1))
    s1 = np.sum(.5*(ax+ay)*(y-x), axis=-1)-.5*dt*(np.sum(ax*fx, axis=-1)+np.sum(ay*fy, axis=-1))
    return .5*dt*(nx+ny), np.stack([s0, s1], axis=-1)


def simulate(theta, T, n, seed):
    theta = np.asarray(theta)
    bases, vs, f0, aa, ll, energies = model(6, 'pairs', theta, FINE)
    table = load_table(); rng = np.random.default_rng(seed)
    x = np.zeros((3,n,6)); coarse_x = x.copy()
    gram = np.zeros((2,3,n)); score = np.zeros((2,3,n,2))
    power = np.zeros((3,n)); counts = np.zeros((3,n,36), int)
    act = np.zeros((3,n), int); idx = np.arange(n)
    oracle = int(energies.argmax()); held_steps = round(HOLD/FINE)
    guard_count = 0; clip_count = 0; snapshots = []
    for j in range(round(T/FINE)):
        if j % held_steps == 0:
            t = j*FINE
            valid = (gram[0,1] >= KAPPA*t/2) & (gram[0,1] > 0)
            est = np.divide(score[0,1], gram[0,1,:,None], out=np.zeros((n,2)), where=valid[:,None])
            bounded = np.clip(est, -WIDTH, WIDTH)
            guard_count += int((~valid).sum()); clip_count += int(np.any(est != bounded, axis=1).sum())
            act[0] = oracle
            act[1] = interpolate(bounded, table).argmax(axis=1)
            act[2] = ((x[2]@vs.T)**2).argmax(axis=1)
            for r in range(3): np.add.at(counts[r], (idx, act[r]), 1)
            # Fixed schedule of diagnostics, independent of observed outcomes.
            if (j//held_steps) % max(1, round(T/HOLD)//64) == 0:
                snapshots.append(bounded[:min(8,n)].copy())
        z = np.broadcast_to(rng.normal(size=(n,6)), x.shape)
        y = np.einsum('knij,knj->kni', aa[act], x)+np.einsum('knij,knj->kni', ll[act], z)
        dg, ds = increment(x, y, act, FINE, bases, f0)
        gram[1] += dg; score[1] += ds
        vx = np.sum(vs[act]*x, axis=-1); vy = np.sum(vs[act]*y, axis=-1)
        power += 4.5*FINE*(vx*vx+vy*vy)
        if j % 2 == 1:
            dg, ds = increment(coarse_x, y, act, OBS, bases, f0)
            gram[0] += dg; score[0] += ds; coarse_x = y.copy()
        x = y
    valid = gram >= KAPPA*T/2
    estimates = np.divide(score, gram[:,:,:,None], out=np.zeros_like(score), where=valid[:,:,:,None])
    loss = np.sum((estimates-theta)**2, axis=-1)
    return dict(theta=theta, T=T, n=n, seed=seed, policies=np.array(POLICIES),
                observation_meshes=np.array([OBS,FINE]), hold=HOLD,
                gram=gram, score=score, estimates=estimates, squared_error=loss,
                guard_valid=valid, terminal_state=x, action_counts=counts,
                input_power=power/T, oracle_action=oracle, stationary_energy=energies,
                ce_guard_decisions=guard_count, ce_clipped_decisions=clip_count,
                ce_decisions=n*round(T/HOLD), ce_snapshots=np.concatenate(snapshots))


def check():
    from scipy.linalg import expm
    rng = np.random.default_rng(2026092209)
    table = load_table()
    points = rng.uniform(-WIDTH, WIDTH, (1024,2))
    validation = table_check(points, table)
    assert validation['max_relative_score_error'] < 1e-4
    assert validation['max_relative_action_regret'] < 1e-4
    bases, vs, f0, aa, ll, energy = model(6, 'pairs', [0,.2], FINE)
    # Independent Van Loan transition covariance and semigroup composition.
    errs = []
    for ell in [0,7,35]:
        f = f0[ell]+.2*bases[1]
        e = expm(np.block([[f,np.eye(6)],[np.zeros((6,6)),-f.T]])*FINE)
        q = e[:6,6:]@e[:6,:6].T
        errs.append(np.max(np.abs(q-ll[ell]@ll[ell].T)))
    assert max(errs) < 1e-12
    x = rng.normal(size=(3,11,6)); y = rng.normal(size=x.shape)
    act = rng.integers(0,36,size=(3,11))
    dg, ds = increment(x,y,act,OBS,bases,f0)
    phix=np.einsum('pij,knj->knpi',bases,x); phiy=np.einsum('pij,knj->knpi',bases,y)
    fx=np.einsum('knij,knj->kni',f0[act],x); fy=np.einsum('knij,knj->kni',f0[act],y)
    direct=np.einsum('knpi,kni->knp',.5*(phix+phiy),y-x)
    direct[:,:,0]-=3*OBS
    direct-=.5*OBS*(np.einsum('knpi,kni->knp',phix,fx)+np.einsum('knpi,kni->knp',phiy,fy))
    assert np.max(np.abs(ds-direct)) < 1e-12
    result = dict(check_groups=3, table_validation=validation,
                  van_loan_max_error=float(max(errs)), score_formula_max_error=float(np.max(np.abs(ds-direct))))
    (OUT/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--task',choices=['table','check','development','confirmation'],required=True)
    ap.add_argument('--case',type=int,default=0); args=ap.parse_args()
    if args.task=='table': return build_table()
    if args.task=='check': return check()
    if args.task=='confirmation':
        frozen=json.loads((ROOT/'metadata/ce_manifest.json').read_text())
        assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in frozen['sha256'].items())
        theta=truths(6,'confirmation')[args.case//2]; T=[64.,256.][args.case%2]; n=1024; seed=2026092200+args.case
    else: theta=[.05,-.15]; T=4.; n=32; seed=2026092199
    out=OUT/args.task; out.mkdir(parents=True,exist_ok=True); start=time.time()
    raw=simulate(theta,T,n,seed)
    np.savez_compressed(out/f'case{args.case}.npz',**raw)
    print(json.dumps(dict(task=args.task,case=args.case,elapsed=time.time()-start,
                         mse=raw['squared_error'][0].mean(axis=1).tolist())),flush=True)


if __name__=='__main__': main()
