# Experimental models and outputs

## Continuous-time model

The simulations use unit diffusion and initial state zero:

$$dX_t = (F_{\ell_t}^{0}+\textstyle\sum_{j=1}^{p}\theta_j B_j)X_t\,dt+dW_t.$$

For the structured families,
$F_\ell^0=-4I+3v_\ell v_\ell^\top+\Omega$, where $\Omega$ is a known
skew-symmetric drift. The first unknown feature is $B_1=I$; the remaining
features are skew-symmetric rotations. The coupled four-parameter family
adds two dense symmetric trace-zero features. The perturbation experiments
change both the known drift and the known feature matrices; they do not
introduce unknown model misspecification.

The projection policy chooses the first maximizer of $(v_\ell^\top X_t)^2$.
Oracles use stationary covariance/information at the true parameter. The CE
policy uses the running guarded estimate, clipped to the known parameter box
for action selection only, and a stationary-energy interpolation table.

Exact Ornstein–Uhlenbeck endpoints are generated for each held loop. The score
uses a trapezoidal integral with its Itô correction; the Gram matrix uses
trapezoidal quadrature. Terminal estimates are unprojected least squares,
with a zero fallback when the information guard fails. Structured and coupled
runs use the numerical guard `1e-8 * T`; perturbation, hold, and CE runs use
their stated model-dependent guards. Held-loop experiments also reconstruct
an information-threshold estimate.

## Protocols

| Family | Model, policies, and uncertainty specification |
| --- | --- |
| Structured | [MULTIPARAMETER_PROTOCOL.md](../protocols/MULTIPARAMETER_PROTOCOL.md) |
| Coupled | [COUPLED_FOURPARAM_PROTOCOL.md](../protocols/COUPLED_FOURPARAM_PROTOCOL.md) |
| Perturbed | [ROBUSTNESS_PROTOCOL.md](../protocols/ROBUSTNESS_PROTOCOL.md) |
| Held decisions | [HOLD_PROTOCOL.md](../protocols/HOLD_PROTOCOL.md) |
| Stationary CE | [CE_PROTOCOL.md](../protocols/CE_PROTOCOL.md) |

Development, confirmation, and refinement outputs are stored separately.
Refinement of the structured, coupled, and perturbation families changes both
clocks and uses fresh paired paths. Held-loop and CE observation refinements
reuse the same physical paths and decisions. Integer seeds are identifiers,
not execution dates.

## Stored arrays

Load NPZ files without enabling pickle:

```python
import numpy as np

with np.load("results/multiparameter/confirmation/case0.npz",
             allow_pickle=False) as data:
    print(data.files)
    print(data["policies"])
    errors = data["estimates"] - data["theta"]
    mse = np.sum(errors**2, axis=-1).mean(axis=-1)
    print(dict(zip(data["policies"], mse)))
```

Files retain the truth, seed, horizon, policy names, per-path estimates,
squared errors, score/Gram statistics, terminal states, and action counts.
They contain per-path sufficient statistics rather than every trajectory
sample. Extra fields depend on the experiment: stationary information and
mixture weights, energy and input power, guard outcomes, switching counts,
stopping times, and CE diagnostic snapshots.

For structured, coupled, and perturbation cases, estimates are indexed by
`(policy, path, parameter)`. CE estimates use
`(observation_mesh, policy, path, parameter)`. Held-loop estimates use
`(observation_mesh, estimator, policy, path, parameter)`; their stored
`observation_meshes`, `estimators`, and `policies` arrays identify each axis.
Use the file's field names and the corresponding analyzer for other arrays.

The 67 NPZ files, including the CE interpolation table and angular sampling
check, are byte-identical to the retained experimental outputs. Numerical CSV
values are retained; export checksums and non-numerical bookkeeping are
maintained separately in `metadata/`.

## Two-loop, HJB, and discrete-time calculations

`check_extensions.py` compares a closed two-loop energy expression with
angular quadrature in 60 configurations. Policy iteration is then run on
reflecting finite-difference grids for orthogonal, oblique, and irregular
three-loop banks. Grid sizes and Bellman residuals are in
`results/extension_checks.json`; these residuals do not certify an error on
the unbounded state space.

The discrete-time simulation uses
$X_{k+1}=(\beta I+(\alpha-\beta)v_{\ell_k}v_{\ell_k}^\top+\theta I)X_k+w_k$
with $\alpha=0.85$, $\beta=0.4$, $\theta=0$, unit Gaussian noise, and
two coordinate loops. Constant, alternating, and projection policies share
innovations. Seed `370102` generates 4,096 path indices at each of the two
horizons. The scalar least-squares estimate is set to zero if its accumulated
information is below the horizon. The output reports every policy, scaled
MSE, paired differences, power, and threshold failures. This is a numerical
comparison without a separate development/confirmation split.

`coverage_geometry.py` also performs a stationary Gaussian cone-sampling
cross-check with 200,000 proposals. The finite-time scripts evaluate bounds
and rational enclosures; their large horizons do not represent new simulated
trajectories. No hardware measurements are included.
