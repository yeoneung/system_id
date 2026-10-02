# Mode-update clocks and observation clocks

Export note: this description retains the recorded experimental settings.
The checksums in `metadata/` verify this repository snapshot; they are not
an external preregistration or a new prospective freeze.


Prospective internal protocol; not external preregistration.
The question is whether the unchanged projection selector remains useful at
a bounded switching rate, and how terminal and information-threshold estimates
behave when the observation mesh alone is refined.

## Fixed design

- Six states, two unknown coefficients; a=1, b=4, all 364 signed-subset modes.
- Four conditions, all reported: ideal theta=(0,0); geometry 0 at
  eta=delta=.04 and theta=(0,0), (.10,-.10), (-.10,.10).
- Known parameter box [-.15,.15]^2; all perturbations are fitted known parts.
- T=128, 512 independent path indices, paired innovations across four policies:
  the true-parameter stationary best fixed mode, and projection selection at
  holding intervals .02, .10, .50. A mode is held unchanged between decisions.
- The oracle is a stationary comparator, not a finite-horizon minimax oracle.
- Exact held-mode Gaussian transitions at .005. Accumulate observations at
  .005 and at the nested .01 mesh on the SAME physical paths. Both meshes
  share the same mode decisions and terminal states. The control clock does
  not change when the observation clock changes.
- Terminal full-matrix LS uses the original theoretical guard. The weighted
  energy-threshold estimate uses h=.15, epsilon=.10 and the target g defined in hold_certificate.py.
  Its stochastic integral uses trapezoidal Stratonovich integration minus
  the analytic divergence correction. Threshold crossing is interpolated
  within an observation step. Neither estimate is exact sampled-data MLE.
- Continuous-observation bounds do not certify finite-mesh threshold risk;
  these runs measure implementation sensitivity.

## Inference and diagnostics

Primary observation mesh .01. Compare each of 3 projection holding intervals
and each of 2 estimators with the oracle's TERMINAL LS, in all 4 conditions:
24 paired mean squared-error differences. Use two-sided paired Student-t
intervals with Bonferroni family alpha=.05. Report every comparison, including
failures or losses. Report total coefficient loss, stopping failures, guard
failures, switching counts, and input power. Percentage gains are descriptive
ratios of sample means; confidence intervals are for mean differences.

Observation sensitivity uses .005 minus .01 squared errors on identical
physical paths, for 4 policies x 2 estimators x 4 conditions: 32 Bonferroni
paired intervals, a separate secondary family. No additional paths are created
by mesh reconstruction. The oracle's threshold estimate is a diagnostic only:
its failure-to-zero behavior at theta=0 can artificially look favorable.
Mesh agreement alone is not a continuous-observation convergence certificate.

## Development, seeds, and freeze

Development: geometry 0, level .08, theta=(.07,-.11), T=4, n=24,
seed 202609191700. It precedes confirmation and is excluded from inference.
An initial development run was used to check runtime. The zero-state
divergence convention was then corrected to recover the exact ideal linear
instrument. Retained development output is regenerated from the final source.

Confirmation seeds 202609191800 through 202609191803 in listed case order.
The original protocol, simulator, analyzer, tests, analytical checks, and
imported sources were recorded before confirmation. No case or seed was
replaced after viewing confirmation.

Stored new physical policy rollouts: 96 development and 8,192 confirmation.
Two estimators and two observation meshes reuse them. No additional short-test
trajectories are planned; structural checks read the development arrays.
Prior studies remain byte-identical and their tests are not counted as rerun.
