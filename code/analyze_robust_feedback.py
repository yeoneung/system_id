"""Analyze all 13 prospectively declared contrasts; never regenerate paths."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_robust_feedback import verify_manifest, cases

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/robustness'


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    verify_manifest()
    contrasts, secondary, mesh = [], [], []
    nroll, guard, maxgap, accounting = 0, 0, 0., 0.
    raw_files = sorted(OUT.glob('*/*.npz'))
    for path in raw_files:
        z = np.load(path)
        nroll += int(z['n']) * len(z['policies'])
        truth = z['theta']
        loss = np.sum((z['estimates'] - truth)**2, axis=-1)
        assert np.allclose(loss, z['squared_error'], rtol=1e-13, atol=1e-15)
        valid = np.linalg.eigvalsh(z['gram'])[..., 0] >= z['kappa'] * z['T'] / 2
        assert np.array_equal(valid, z['guard_valid'])
        recovered = np.zeros_like(z['score'])
        recovered[valid] = np.linalg.solve(z['gram'][valid], z['score'][valid])
        assert np.allclose(recovered, z['estimates'], rtol=1e-12, atol=1e-14)
        assert np.all(z['action_counts'].sum(axis=-1) == round(float(z['T']/z['dt'])))
        guard += int((~valid).sum())
        maxgap = max(maxgap, float(z['oracle_gap']))
        # At theta=0, expected energy balance includes the known common E.
        # This residual combines Monte Carlo and mesh error; it is not forced to zero.
        if np.all(truth == 0):
            residual = (z['input_energy']/3 - 4*z['energy'] + 3*z['T']
                        + z['deviation_energy'] - .5*np.sum(z['terminal_state']**2, axis=-1))
            accounting = max(accounting, float(np.max(np.abs(residual.mean(axis=1)/z['T']))))
    for phase, selected in [('confirmation', range(9)), ('refinement', range(1, 5))]:
        for case in selected:
            z = np.load(OUT/phase/f'case{case}.npz')
            g, level = cases()[case]
            assert (int(z['geometry']), float(z['level'])) == (g, level)
            loss = z['squared_error']; n = int(z['n']); T = float(z['T'])
            diff = loss[2] - loss[0]
            se = float(diff.std(ddof=1)/np.sqrt(n))
            crit = float(t.ppf(1-.05/(2*13), n-1))
            mean = float(diff.mean()); base = float(loss[0].mean())
            contrasts.append(dict(phase=phase,case=case,geometry=g,level=level,
                n=n,T=T,dt=float(z['dt']),oracle_mse=base,
                projection_mse=float(loss[2].mean()),difference=mean,paired_se=se,
                adjusted_low=mean-crit*se,adjusted_high=mean+crit*se,
                reduction_percent=-100*mean/base,certified_factor=float(z['certified_factor'])))
            secondary.append(dict(phase=phase,case=case,
                oracle_fixed_mse=float(loss[1].mean()),
                fixed_mixture_estimates_identical=bool(np.array_equal(z['estimates'][0],z['estimates'][1])),
                oracle_input_power=float(z['input_energy'][0].mean()/T),
                projection_input_power=float(z['input_energy'][2].mean()/T),
                input_power_ratio=float(z['input_energy'][2].mean()/z['input_energy'][0].mean()),
                mixture_active_modes=int(np.count_nonzero(z['weights']>1e-10)),
                oracle_gap=float(z['oracle_gap'])))
            if phase == 'refinement':
                coarse = np.load(OUT/phase/f'case{case}_coarse.npz')
                for j,name in enumerate(z['policies']):
                    d = loss[j]-coarse['squared_error'][j]
                    mesh.append(dict(case=case,policy=str(name),
                        coarse_mse=float(coarse['squared_error'][j].mean()),
                        fine_mse=float(loss[j].mean()),difference=float(d.mean()),
                        paired_se=float(d.std(ddof=1)/np.sqrt(n)),
                        relative_change_percent=float(100*d.mean()/coarse['squared_error'][j].mean())))
    assert len(contrasts)==13 and nroll==26208
    write_csv(OUT/'contrasts.csv',contrasts)
    write_csv(OUT/'secondary.csv',secondary)
    write_csv(OUT/'mesh.csv',mesh)
    gains = lambda phase: [r['reduction_percent'] for r in contrasts if r['phase']==phase]
    summary = dict(primary_contrasts=13,family_alpha=.05,interval='paired Student-t, Bonferroni 13',
        negative_adjusted_upper=sum(r['adjusted_high']<0 for r in contrasts),
        confirmation_gain_percent=[min(gains('confirmation')),max(gains('confirmation'))],
        refinement_gain_percent=[min(gains('refinement')),max(gains('refinement'))],
        stored_policy_rollouts=nroll,additional_short_test_rollouts=48,
        raw_files=len(raw_files),guard_failures=guard,maximum_stationary_oracle_gap=maxgap,
        all_stationary_oracles_degenerate_to_best_fixed=all(r['mixture_active_modes']==1 for r in secondary),
        input_power_ratio=[min(r['input_power_ratio'] for r in secondary),max(r['input_power_ratio'] for r in secondary)],
        max_absolute_energy_balance_residual_per_time=accounting,
        mesh_relative_mse_change_percent=[min(r['relative_change_percent'] for r in mesh),max(r['relative_change_percent'] for r in mesh)],
        exact_q=0.30827781826282163,
        protocol_manifest_verified=True,all_stored_losses_and_guards_reconstructed=True,
        unknown_model_misspecification_tested=False,continuous_time_limit_certified=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
    fig, ax = plt.subplots(figsize=(3.4,3.25),layout='constrained')
    for j,r in enumerate(contrasts):
        scale=100/r['oracle_mse']; val=r['reduction_percent']
        lo=-scale*r['adjusted_high']; hi=-scale*r['adjusted_low']
        col={0.:'#777777',.04:'#0072B2',.1:'#D55E00'}[r['level']]
        ax.errorbar(val,j,xerr=[[val-lo],[hi-val]],fmt='o' if j<9 else 's',
                    color=col,ms=3,capsize=2,lw=1)
    labels=[]
    for r in contrasts:
        prefix='C' if r['phase']=='confirmation' else 'F'
        labels.append('C: unperturbed' if r['level']==0 else f"{prefix}: G{r['geometry']+1}, {r['level']:.2f}")
    ax.set_yticks(range(13),labels);ax.invert_yaxis()
    ax.axhline(8.5,color='.7',lw=.7);ax.grid(axis='x',alpha=.25)
    ax.set_xlabel('MSE reduction / oracle mean (%)')
    ax.set_xlim(30,65)
    ax.set_xticks(np.arange(30,65,5))
    fig.savefig(ROOT/'figures/robust_feedback.pdf');plt.close(fig)
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in raw_files}
    provenance=ROOT/'metadata/robustness_data_provenance.json'
    if provenance.exists():
        assert json.loads(provenance.read_text())['sha256']==hashes
    else:
        provenance.write_text(json.dumps(dict(files=len(hashes),sha256=hashes),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
