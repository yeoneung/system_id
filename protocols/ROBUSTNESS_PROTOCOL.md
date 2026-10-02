# Perturbed six-state feedback identification

Export note: this description retains the recorded experimental settings.
The checksums in `metadata/` verify this repository snapshot; they are not
an external preregistration or a new prospective freeze.


This protocol is frozen before the new confirmation or refinement trajectories.
It is an internal prospective record, not external preregistration.

Question: does the same projection policy retain finite-horizon vector-risk
gains when the parameter information is nonscalar and the known drift is
perturbed? The experiment varies known model structure.

## Fixed model and comparisons

- Six states, two jointly unknown coefficients, a=1, b=4, theta=(0,0).
- Parameter box [-0.15,0.15]^2. This is the center of the parameter box;
  the estimator receives no true coefficient. The stationary oracles do.
- All unoriented nonzero vectors in {-1,0,1}^6, normalized: 364 modes.
  Its exact worst squared angular error is 0.30827781826282163.
- B_i=A_i+D_i, sum_i ||D_i||_op^2=eta^2, with dense general perturbations.
  Known common drift E has ||E||_op=delta. Both enter the estimator correctly.
  This is sensitivity to model structure, not unmodelled nuisance estimation.
- Geometry seeds: 2026091950 through 2026091953. Each geometry is scaled at
  (eta,delta)=(0.04,0.04) and (0.10,0.10). One unperturbed reference is included.
  All nine declared conditions are reported, whether favorable or unfavorable.
- Fixed horizon T=128, mesh 0.02, 512 independent path indices per policy.
- Three policies with paired Gaussian innovations: A-optimal stationary oracle
  mixture, best stationary fixed-mode oracle, parameter-free projection.
- The mixture uses deterministic long blocks according to its optimal weights.
  The simplex solver starts at the best fixed mode, so at the symmetric
  unperturbed case it can retain a fixed mode instead of spreading equally
  over hundreds of equally optimal modes. Report numerical optimality gaps.
  A stationary oracle is not claimed finite-time optimal over all schedules.
- Seed 2026092000 + case index for confirmation. Case 0 is unperturbed;
  cases 1--8 iterate geometry 0--3, then levels 0.04,0.10.

## Estimation and uncertainty

Use full matrix least squares and the theoretical guard lambda_min(Q_T)
>=kappa*T/2 with kappa=(1-eta)^2*d/[2*(b+delta+0.15+eta*sqrt(2)*0.15)].
Fallback is zero. Report every guard failure. Primary loss is total squared
coefficient error without projection. Coordinate errors are secondary.
OU endpoints under held modes are exact. Score and Gram use trapezoidal
approximations and the trace Ito correction. These are not exact sampled-data
likelihood statistics.

The primary comparison is projection minus oracle-mixture MSE in each of
nine confirmation cells and four preselected fine-mesh cells below. Use
two-sided paired Student-t intervals with Bonferroni correction over all
13 contrasts at family alpha=0.05. Gains are ratios of sample means; adjusted
intervals are for mean differences, not population ratios. Fixed-mode
comparisons and measured input energies are descriptive secondary quantities.

## Mesh refinement

Before confirmation, select geometries 0 and 1 at both nonzero levels:
cases 1--4. Generate 512 fresh paired paths for each policy at meshes 0.02
and 0.01, T=128, seed 2026092100 + case index. Two fine standard normals
are combined into the coarse innovation divided by sqrt(2). This couples
correct marginal transitions but is not an exact OU bridge. Control and
observation spacing change together. Report both absolute MSE changes and
paired SEs; do not declare mesh independence from preserved signs alone.

## Development and accounting

Development uses geometry 0, eta=delta=0.08, theta=(0.08,-0.12), T=8,
32 paired paths per policy, seed 2026091960. A prototype development run
preceded the final stationary-oracle initialization choice; the retained
development output is regenerated with the frozen final source. It is not
part of the primary comparisons. Structural tests have separate seeds and
include a 16-path short accounting check, not an estimation benchmark.

Prescribed stored policy rollouts: confirmation 13,824, refinement 12,288,
development 96. Paired rollouts are not independent sample counts. The
previous studies remain byte-identical and are not pooled into this new
family of inferential comparisons. No result-based cell or seed replacement
is permitted. Export source and protocol checksums are checked before regeneration.
