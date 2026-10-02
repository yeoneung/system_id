"""Reconstruct all prospectively declared CE comparisons and comparison plots."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_ce_feedback import KAPPA, POLICIES, load_table, table_check

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/ce'


def interval(delta, family):
    se=delta.std(ddof=1)/np.sqrt(len(delta)); crit=t.ppf(1-.05/(2*family),len(delta)-1)
    return dict(difference=float(delta.mean()), paired_se=float(se),
                adjusted_low=float(delta.mean()-crit*se), adjusted_high=float(delta.mean()+crit*se))


def write_csv(name, rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    freeze=json.loads((ROOT/'metadata/ce_manifest.json').read_text())
    assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in freeze['sha256'].items())
    rows=[]; mesh=[]; descriptive=[]; ce_oracle=[]; snapshots=[]
    for case in range(4):
        raw=np.load(OUT/f'confirmation/case{case}.npz'); n=int(raw['n']); T=float(raw['T'])
        assert n==1024
        reconstructed=np.divide(raw['score'],raw['gram'][:,:,:,None],out=np.zeros_like(raw['score']),
                                where=(raw['gram']>=KAPPA*T/2)[:,:,:,None])
        assert np.allclose(reconstructed,raw['estimates'])
        loss=np.sum((reconstructed-raw['theta'])**2,axis=-1)
        assert np.allclose(loss,raw['squared_error'])
        assert np.all(raw['action_counts'].sum(axis=-1)==round(T/float(raw['hold'])))
        for baseline in [0,1]:
            row=dict(case=case,truth='AB'[case//2],T=T,n=n,baseline=POLICIES[baseline],
                     baseline_mse=float(loss[0,baseline].mean()),projection_mse=float(loss[0,2].mean()),
                     reduction_percent=float(100*(1-loss[0,2].mean()/loss[0,baseline].mean())),
                     **interval(loss[0,2]-loss[0,baseline],8))
            rows.append(row)
        ce_oracle.append(dict(case=case,**interval(loss[0,1]-loss[0,0],4)))
        for i,policy in enumerate(POLICIES):
            descriptive.append(dict(case=case,truth='AB'[case//2],T=T,policy=policy,mse=float(loss[0,i].mean()),
                fine_mse=float(loss[1,i].mean()),energy=float(raw['gram'][0,i].mean()/T),
                mean_power=float(raw['input_power'][i].mean()),terminal_guard_failures=int((~raw['guard_valid'][0,i]).sum()),
                ce_guard_decisions=int(raw['ce_guard_decisions']) if i==1 else 0,
                ce_clipped_decisions=int(raw['ce_clipped_decisions']) if i==1 else 0,
                decisions=int(raw['ce_decisions'])))
            mesh.append(dict(case=case,policy=policy,
                relative_change_percent=float(100*(loss[1,i].mean()/loss[0,i].mean()-1)),
                **interval(loss[1,i]-loss[0,i],12)))
        snapshots.append(raw['ce_snapshots'])
    validation=table_check(np.concatenate(snapshots),load_table())
    write_csv('contrasts.csv',rows); write_csv('descriptive.csv',descriptive)
    write_csv('observation_mesh.csv',mesh); write_csv('ce_vs_oracle.csv',ce_oracle)
    groups={}
    for name in POLICIES[:2]:
        sub=[r for r in rows if r['baseline']==name]
        groups[name]=dict(reduction_range=[min(r['reduction_percent'] for r in sub),max(r['reduction_percent'] for r in sub)],
                          adjusted_improvements=sum(r['adjusted_high']<0 for r in sub))
    summary=dict(cells=4,paths_per_policy=1024,confirmation_policy_rollouts=4*1024*3,
                 development_policy_rollouts=32*3,total_policy_rollouts=4*1024*3+32*3,
                 primary_contrasts=8,comparisons=groups,visited_table_validation=validation,
                 terminal_guard_failures=sum(r['terminal_guard_failures'] for r in descriptive),
                 observation_mesh_percent_range=[min(r['relative_change_percent'] for r in mesh),max(r['relative_change_percent'] for r in mesh)],
                 observation_mesh_adjusted_resolved=sum(r['adjusted_low']*r['adjusted_high']>0 for r in mesh),
                 ce_vs_oracle_adjusted_resolved=sum(r['adjusted_low']*r['adjusted_high']>0 for r in ce_oracle),
                 analysis_check_groups=2,all_cells_reported=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Replot the retained results without changing their frozen files.
    with (ROOT/'results/multiparameter/contrasts.csv').open() as f: old=list(csv.DictReader(f))
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(7,2.25),sharey=True)
    for ax,d in zip(axes[:2],[6,8]):
        for kind,style in [('axes','-o'),('pairs','--s')]:
            sub=[r for r in old if r['phase']=='confirmation' and r['baseline']=='oracle_fixed' and int(r['d'])==d and r['dictionary']==kind]
            ax.plot(range(4),[float(r['feedback_mse'])/float(r['baseline_mse']) for r in sub],style,markersize=4,label=f'Projection: {kind}')
        ax.set_title(f'$d={d},\ p={2 if d==6 else 4}$; $\Delta=.02$',fontsize=9)
    for index,label,style in [(1,'Stationary CE','-^'),(2,'Projection: pairs','--s')]:
        yy=[]
        for case in range(4):
            sub=[r for r in descriptive if r['case']==case]
            yy.append(sub[index]['mse']/sub[0]['mse'])
        axes[2].plot(range(4),yy,style,markersize=4,label=label)
    axes[2].set_title('$d=6,\ p=2$; $\Delta=.10$',fontsize=9)
    for ax in axes:
        ax.axhline(1,color='black',lw=.7);ax.set_ylim(.4,1.1)
        ax.set_xticks(range(4),['A/64','A/256','B/64','B/256'],rotation=30,fontsize=7)
        ax.legend(fontsize=7,loc='lower left');ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('MSE / fixed-oracle MSE',fontsize=9)
    fig.tight_layout();fig.savefig(ROOT/'figures/adaptivity_comparison.pdf');plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
