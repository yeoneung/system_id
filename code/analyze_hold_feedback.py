"""Report all frozen update-clock contrasts and nested observation checks."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_hold_feedback import verify_manifest

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/hold'


def interval(x,family):
    n=len(x);m=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(n))
    radius=float(t.ppf(1-.05/(2*family),n-1)*se)
    return m,se,m-radius,m+radius


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    verify_manifest();rows=[];mesh=[];fail=[];account=[]
    for case in range(4):
        z=np.load(OUT/f'confirmation/case{case}.npz')
        # Check stored losses, guarded terminal inverses, and stopped statistics.
        rec=np.zeros_like(z['score']);good=z['guard_valid']
        rec[good]=np.linalg.solve(z['gram'][good],z['score'][good])
        assert np.allclose(rec,z['estimates'][:,0],rtol=0,atol=1e-12)
        assert np.allclose(z['threshold_score']/z['target'],z['estimates'][:,1])
        assert np.allclose(np.sum((z['estimates']-z['theta'])**2,axis=-1),z['squared_error'])
        assert np.all(z['action_counts'].sum(axis=-1)==round(float(z['T'])/.005))
        assert np.all(np.isnan(z['hit_time'])==~z['hit'])
        for j,D in enumerate(z['holds'][1:],1):
            assert np.all(z['switches'][j]<=np.ceil(z['T']/D)-1)
        loss=z['squared_error'];oracle=loss[1,0,0]
        for e,name in enumerate(z['estimators']):
            for j,D in enumerate(z['holds'][1:],1):
                m,se,lo,hi=interval(loss[1,e,j]-oracle,24)
                rows.append(dict(case=case,level=float(z['level']),theta0=float(z['theta'][0]),
                    theta1=float(z['theta'][1]),hold=float(D),estimator=str(name),
                    mse=float(loss[1,e,j].mean()),oracle_mse=float(oracle.mean()),
                    gain_percent=float(100*(1-loss[1,e,j].mean()/oracle.mean())),
                    paired_difference=m,paired_se=se,adjusted_low=lo,adjusted_high=hi,
                    threshold_failures=int((~z['hit'][1,j]).sum()),
                    guard_failures=int((~z['guard_valid'][1,j]).sum()),
                    mean_switches=float(z['switches'][j].mean()),
                    power_ratio=float(z['power'][1,j].mean()/z['power'][1,0].mean())))
            for j,policy in enumerate(z['policies']):
                m,se,lo,hi=interval(loss[0,e,j]-loss[1,e,j],32)
                mesh.append(dict(case=case,policy=str(policy),estimator=str(name),
                    fine_mse=float(loss[0,e,j].mean()),coarse_mse=float(loss[1,e,j].mean()),
                    relative_change_percent=float(100*m/loss[1,e,j].mean()),
                    difference=m,se=se,adjusted_low=lo,adjusted_high=hi))
        account.append(dict(case=case,guard_failures=int((~z['guard_valid']).sum()),
            projection_threshold_failures=z['hit'][:,1:].shape[-1]*2*3-int(z['hit'][:,1:].sum()),
            physical_policy_rollouts=int(z['n'])*len(z['policies'])))
    write_csv(OUT/'contrasts.csv',rows);write_csv(OUT/'observation_mesh.csv',mesh)
    summary=dict(primary_contrasts=len(rows),family_alpha=.05,negative_adjusted_upper=sum(r['adjusted_high']<0 for r in rows),
        terminal_gain_range=[min(r['gain_percent'] for r in rows if r['estimator']=='terminal'),
                            max(r['gain_percent'] for r in rows if r['estimator']=='terminal')],
        threshold_gain_range=[min(r['gain_percent'] for r in rows if r['estimator']=='threshold'),
                             max(r['gain_percent'] for r in rows if r['estimator']=='threshold')],
        observation_relative_change_range=[min(r['relative_change_percent'] for r in mesh),max(r['relative_change_percent'] for r in mesh)],
        guard_failures=sum(r['guard_failures'] for r in account),cases=account,
        confirmation_physical_policy_rollouts=sum(r['physical_policy_rollouts'] for r in account),
        development_physical_policy_rollouts=96,nested_meshes_share_exact_same_physical_paths=True,
        estimator_reconstruction=True,source_manifest_verified=True,
        finite_horizon_sampled_risk_certified=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(6.7,2.35),sharey=True)
    for e,(ax,name) in enumerate(zip(axes,['terminal','threshold'])):
        for case in range(4):
            rr=[r for r in rows if r['case']==case and r['estimator']==name]
            xs=np.array([r['hold'] for r in rr]);ys=np.array([r['gain_percent'] for r in rr])
            lower=np.array([-100*r['adjusted_high']/r['oracle_mse'] for r in rr])
            upper=np.array([-100*r['adjusted_low']/r['oracle_mse'] for r in rr])
            label=['Ideal, center','Perturbed, center','Perturbed, +/−','Perturbed, −/+'][case]
            ax.errorbar(xs*(1+(case-1.5)*.065),ys,yerr=[ys-lower,upper-ys],marker='o',markersize=3,lw=.85,capsize=2,label=label)
        ax.axhline(0,color='black',lw=.7);ax.set_xscale('log');ax.set_xticks([.02,.1,.5],['0.02','0.10','0.50'])
        ax.set_xlabel('Mode holding time');ax.set_title(['Terminal LS','Information-threshold estimate'][e])
        ax.grid(alpha=.2)
    axes[0].set_ylabel('MSE decrease vs fixed oracle (%)')
    axes[1].legend(fontsize=6,loc='best')
    fig.tight_layout(pad=.6);fig.savefig(ROOT/'figures/hold_feedback.pdf');plt.close(fig)
    rawfiles=sorted(OUT.glob('*/*.npz'))
    provenance=dict(files=len(rawfiles),sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in rawfiles})
    (ROOT/'metadata/hold_data_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
