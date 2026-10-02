from pathlib import Path
import hashlib,json,csv
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'coupled_fourparam'


def main():
    freeze=json.loads((ROOT/'metadata/coupled_fourparam_manifest.json').read_text())
    for name,digest in freeze['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    rows=[];summary={};mesh=[];coordinates=[]
    for phase,m in [('confirmation',40),('refinement',4)]:
        paths=sorted(p for p in (OUT/phase).glob('case*.npz') if '_coarse' not in p.name)
        assert len(paths)==(4 if phase=='confirmation' else 2)
        for path in paths:
            raw=np.load(path);loss=((raw['estimates']-raw['theta'])**2).sum(axis=-1)
            assert np.allclose(loss,raw['squared_error'])
            n=loss.shape[1];crit=t.ppf(1-.05/(2*m),n-1);case=int(path.stem[4:])
            for name,idx in [('oracle_mixture',0),('periodic',2)]:
                delta=loss[-1]-loss[idx];se=delta.std(ddof=1)/np.sqrt(n)
                rows.append(dict(phase=phase,case=case,T=float(raw['T']),dt=float(raw['dt']),n=n,baseline=name,
                    feedback_mse=float(loss[-1].mean()),baseline_mse=float(loss[idx].mean()),difference=float(delta.mean()),
                    paired_se=float(se),adjusted_low=float(delta.mean()-crit*se),adjusted_high=float(delta.mean()+crit*se),
                    reduction_percent=float(100*(1-loss[-1].mean()/loss[idx].mean()))))
            per=((raw['estimates']-raw['theta'])**2).mean(axis=1)
            coordinates.append(dict(phase=phase,case=case,feedback_coordinate_mse=per[-1].tolist(),
                oracle_coordinate_mse=per[0].tolist(),coordinate_reductions_percent=(100*(1-per[-1]/per[0])).tolist(),
                guard_failures=int((~raw['guard_valid']).sum()),oracle_equivalence_gap=float(raw['oracle_equivalence_gap'])))
            if phase=='refinement':
                coarse=np.load(path.with_name(path.stem+'_coarse.npz'))
                for name,idx in [('oracle_mixture',0),('periodic',2),('state_feedback',3)]:
                    delta=loss[idx]-coarse['squared_error'][idx]
                    mesh.append(dict(case=case,policy=name,change=float(delta.mean()),se=float(delta.std(ddof=1)/np.sqrt(n))))
        rr=[r for r in rows if r['phase']==phase]
        summary[phase]=dict(cells=len(paths),paths_per_policy=n,rollouts=len(paths)*n*4*(2 if phase=='refinement' else 1),
            adjusted_improvements=sum(r['adjusted_high']<0 for r in rr),comparisons=len(rr),
            oracle_reduction_range=[min(r['reduction_percent'] for r in rr if r['baseline']=='oracle_mixture'),max(r['reduction_percent'] for r in rr if r['baseline']=='oracle_mixture')],
            periodic_reduction_range=[min(r['reduction_percent'] for r in rr if r['baseline']=='periodic'),max(r['reduction_percent'] for r in rr if r['baseline']=='periodic')])
    # Conservative simultaneous check pooling the 32 structured comparisons
    # with this family's eight comparisons, without selecting any condition.
    with (ROOT/'results/multiparameter/contrasts.csv').open() as f:old=list(csv.DictReader(f))
    old=[r for r in old if r['phase']=='confirmation']
    pooled=sum(float(r['difference'])+t.ppf(1-.05/(2*40),int(r['n'])-1)*float(r['paired_se'])<0 for r in old)
    pooled+=sum(r['adjusted_high']<0 for r in rows if r['phase']=='confirmation')
    summary.update(source_manifest_verified=True,pooled_40_comparison_improvements=int(pooled),development_rollouts=512,
        coordinate_results=coordinates,mesh_results=mesh)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (OUT/'contrasts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(3.2,2.25))
    for name,offset,color,label in [('oracle_mixture',-.13,'#2563a8','Oracle mixture'),('periodic',.13,'#bd4b28','Periodic')]:
        rr=[r for r in rows if r['phase']=='confirmation' and r['baseline']==name]
        for j,r in enumerate(rr):
            # Difference divided by the observed baseline mean is a descriptive
            # scale. Endpoints retain simultaneous difference-test signs;
            # they are not a confidence interval for a population MSE ratio.
            value=r['difference']/r['baseline_mse']
            low=(r['difference']-r['adjusted_low'])/r['baseline_mse']
            high=(r['adjusted_high']-r['difference'])/r['baseline_mse']
            ax.errorbar(j+offset,value,yerr=np.array([[low],[high]]),fmt='o',color=color,capsize=3,label=label if j==0 else None)
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(4),['A/64','A/256','B/64','B/256'])
    ax.set_xlabel('Truth / horizon');ax.set_ylabel('Relative MSE difference')
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(ROOT/'figures/coupled_fourparam.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(3.2,2.25))
    diagnostics=[]
    for case,path in enumerate(sorted((OUT/'confirmation').glob('case*.npz'))):
        raw=np.load(path);coordinate=raw['coordinate_squared_error'].mean(axis=1)
        ratio=coordinate[-1]/coordinate[0]
        ax.plot(np.arange(4),ratio,marker=['o','s','^','D'][case],lw=1,
                label=f"{'A' if case<2 else 'B'}, T={int(raw['T'])}")
        g=raw['gram'].mean(axis=1)/float(raw['T'])
        eig=np.linalg.eigvalsh(g)[:,0]
        diagnostics.append(dict(case=case,oracle_min_mean_information=float(eig[0]),
            feedback_min_mean_information=float(eig[-1]),
            relative_increase_percent=float(100*(eig[-1]/eig[0]-1))))
    ax.axhline(1,color='black',lw=.8);ax.set_ylim(.5,1.04)
    ax.set_xticks(range(4),['Damping','Rotation','Symmetric 1','Symmetric 2'],fontsize=7)
    ax.set_ylabel('Coordinate MSE / oracle MSE');ax.legend(ncol=2,fontsize=7,loc='lower left')
    fig.tight_layout();fig.savefig(ROOT/'figures/coupled_coordinates.pdf');plt.close(fig)
    summary['information_diagnostics']=diagnostics
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['coordinate_results','mesh_results']},indent=2))


if __name__=='__main__':main()
