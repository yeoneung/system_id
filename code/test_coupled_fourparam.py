from pathlib import Path
import itertools,json
import numpy as np
from scipy.linalg import expm
from run_coupled_fourparam import model,simulate

ROOT=Path(__file__).resolve().parents[1]


def main():
    theta=np.array([-.2,.35,.15,-.12]);bs,vs,f0,aa,ll,js,w,c,gap=model(theta,.02)
    assert np.linalg.eigvalsh(np.einsum('iab,jab->ij',bs,bs))[0]>.1
    assert any(np.linalg.norm(a@b-b@a)>1 for a,b in itertools.combinations(bs[1:],2))
    x=np.arange(1,7.);phi=np.einsum('pij,j->pi',bs,x);gram=phi@phi.T
    assert np.linalg.norm(gram-np.trace(gram)*np.eye(4)/4)>1
    for signs in itertools.product([-1,1],repeat=4):
        t=np.array(signs)*np.array([.25,.4,.2,.2])
        fs=f0+np.einsum('p,pij->ij',t,bs)
        assert np.linalg.eigvalsh((fs+fs.transpose(0,2,1))/2).max()<=-.35+1e-12
    f=f0[7]+np.einsum('p,pij->ij',theta,bs)
    e=expm(.02*np.block([[f,np.eye(6)],[np.zeros((6,6)),-f.T]]))
    assert np.allclose(e[:6,6:]@e[:6,:6].T,ll[7]@ll[7].T,atol=1e-12)
    assert gap<1e-5 and np.all(w>=0) and np.isclose(w.sum(),1)
    _,raw=simulate(theta,2,16,.02,2026110602)
    assert np.all(raw['action_counts'].sum(axis=-1)==100)
    assert np.allclose(np.einsum('knij,knj->kni',raw['gram'],raw['estimates']),raw['score'])
    assert np.allclose(((raw['estimates']-theta)**2).sum(axis=-1),raw['squared_error'])
    checks=dict(full_rank_nonisotropic_information=True,noncommuting_unknown_features=True,
        all_16_parameter_corners_common_stability=True,van_loan_covariance=True,
        numerical_a_optimality_residual=True,raw_score_and_budget_accounting=True)
    (ROOT/'results'/'coupled_fourparam').mkdir(exist_ok=True)
    (ROOT/'results'/'coupled_fourparam'/'structural_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
