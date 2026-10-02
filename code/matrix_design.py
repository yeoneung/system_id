"""A-optimal stationary mixture weights for finite action banks."""
import numpy as np
from scipy.optimize import minimize


def a_optimal_weights(information, start=None):
    """Convex A-optimal simplex problem, with a feasible design returned always."""
    m,p,_=information.shape
    if start is None:start=np.full(m,1/m)
    def both(w):
        j=np.einsum('l,lij->ij',w,information)
        try: inv=np.linalg.inv(j)
        except np.linalg.LinAlgError:return 1e20,np.zeros(m)
        if np.linalg.eigvalsh(j)[0]<=1e-13:return 1e20,np.zeros(m)
        return float(np.trace(inv)),-np.einsum('ij,lji->l',inv@inv,information)
    opt=minimize(both,start,jac=True,method='SLSQP',bounds=[(0,1)]*m,
        constraints={'type':'eq','fun':lambda w:w.sum()-1,
                     'jac':lambda w:np.ones(m)},
        options={'ftol':1e-10,'maxiter':150})
    candidates=[start]
    if np.all(np.isfinite(opt.x)):
        w=np.maximum(opt.x,0);w/=w.sum();candidates.append(w)
    w=min(candidates,key=lambda w:both(w)[0])
    return w.copy(),both(w)[0],bool(opt.success)
