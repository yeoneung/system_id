"""Rebuild paired comparisons directly from raw multi-parameter estimates."""
from pathlib import Path
import hashlib,json,csv
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'multiparameter'


def main():
    freeze=json.loads((ROOT/'metadata/multiparameter_manifest.json').read_text())
    for name,digest in freeze['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    allrows=[];summary={};plot=[]
    for phase,ncomparisons in [('confirmation',32),('refinement',8)]:
        paths=sorted((p for p in (OUT/phase).glob('case*.npz') if '_coarse' not in p.name),
                     key=lambda p:int(p.stem[4:]))
        assert len(paths)==(16 if phase=='confirmation' else 4)
        for path in paths:
            raw=np.load(path);theta=raw['theta'];loss=((raw['estimates']-theta)**2).sum(axis=-1)
            assert np.allclose(loss,raw['squared_error'])
            assert np.all(raw['action_counts'].sum(axis=-1)==round(float(raw['T']/raw['dt'])))
            n=loss.shape[1];crit=t.ppf(1-.05/(2*ncomparisons),n-1)
            case=int(path.stem.replace('case',''))
            for baseline,index in [('oracle_fixed',0),('periodic',3)]:
                delta=loss[-1]-loss[index];se=delta.std(ddof=1)/np.sqrt(n)
                row=dict(phase=phase,case=case,d=int(raw['d']),p=int(raw['p']),dictionary=str(raw['dictionary']),
                    actions=len(raw['vectors']),T=float(raw['T']),dt=float(raw['dt']),n=n,baseline=baseline,
                    baseline_mse=float(loss[index].mean()),feedback_mse=float(loss[-1].mean()),
                    difference=float(delta.mean()),paired_se=float(se),
                    adjusted_low=float(delta.mean()-crit*se),adjusted_high=float(delta.mean()+crit*se),
                    reduction_percent=float(100*(1-loss[-1].mean()/loss[index].mean())))
                allrows.append(row)
                if phase=='confirmation' and baseline=='oracle_fixed':plot.append(row)
        rows=[r for r in allrows if r['phase']==phase]
        summary[phase]=dict(cells=len(paths),paths_per_policy=n,rollouts=len(paths)*n*5*(2 if phase=='refinement' else 1),
            comparisons=len(rows),adjusted_improvements=sum(r['adjusted_high']<0 for r in rows),
            oracle_reduction_range=[min(r['reduction_percent'] for r in rows if r['baseline']=='oracle_fixed'),max(r['reduction_percent'] for r in rows if r['baseline']=='oracle_fixed')],
            periodic_reduction_range=[min(r['reduction_percent'] for r in rows if r['baseline']=='periodic'),max(r['reduction_percent'] for r in rows if r['baseline']=='periodic')])
    with (OUT/'contrasts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    summary['source_manifest_verified']=True
    summary['development_rollouts']=4*128*5
    summary['scope']='Structured coupled two/four-parameter models; continuous-statistic mesh approximations, not exact sampled-data MLE.'
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(3.2,2.3),sharey=True)
    for ax,(d,color,marker) in zip(axes,[(6,'#2563a8','o'),(8,'#bd4b28','s')]):
        rr=[r for r in plot if r['d']==d]
        for kind,offset in [('axes',-.12),('pairs',.12)]:
            ss=[r for r in rr if r['dictionary']==kind]
            xx=np.arange(4)+offset
            yy=[r['feedback_mse']/r['baseline_mse'] for r in ss]
            ax.plot(xx,yy,marker=marker,markersize=3,linestyle='-' if kind=='axes' else '--',color=color,label=kind)
        ax.axhline(1,color='black',lw=.8);ax.set_ylim(.4,1.05)
        ax.set_title(f'd={d}, p={2 if d==6 else 4}',fontsize=8)
        ax.set_xticks(range(4),['A/64','A/256','B/64','B/256'],rotation=45,fontsize=6)
        ax.tick_params(axis='y',labelsize=7);ax.legend(fontsize=6,loc='lower left')
    axes[0].set_ylabel('MSE / oracle-fixed MSE',fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/'figures/multiparameter_feedback.pdf');plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
