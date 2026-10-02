# Multi-parameter confirmation protocol

Export note: this description retains the recorded experimental settings.
The checksums in `metadata/` verify this repository snapshot; they are not
an external preregistration or a new prospective freeze.


This protocol and the simulator/test sources are hashed before confirmation.
This is an internal prospective freeze, not external preregistration.

## Scientific target

Test feedback versus noise-independent action schedules while simultaneously
estimating unknown damping and rotations in coupled physical state spaces.
Families: (d,p)=(6,2),(8,4). Known dense skew drift and densely conjugated
unknown skew features are fixed by construction seeds.

## Choices fixed before confirmation

- Dictionary: coordinate axes or axes plus signed pair orientations; L=d or d².
- Dynamics: a=1,b=4, known skew spectral norm .15.
- Unknown class: theta0 in [-.25,.25], every rotation in [-.4,.4].
- Truths: d=6 uses (-.2,.35),(.2,-.3); d=8 uses
  (-.2,.25,-.2,.15),(.2,-.3,.15,-.25).
- Horizons 64 and 256. All 16 combinations are reported.
- 1,024 paths per policy/cell; common normal innovations for pairing.
- Confirmation mesh .02; policies choose at the same observation times.
- Comparators: oracle best fixed orientation, first fixed orientation,
  equal-time long blocks, periodic, parameter-free state feedback.
- Oracle fixed uses the true parameter ONLY in that explicitly named baseline.
  It exactly optimizes the continuous-time stationary-mixture criterion for
  this family, because every stationary information is a scalar times I_p.
  This does not make it finite-time-optimal among all schedules.
- Feedback maximizes squared projection, independent of unknown coefficients.
  It equals expected-energy greedy; no superiority over that rule is claimed.
- Primary loss: unprojected squared Euclidean error on ALL unknown coefficients.
- Primary contrasts: feedback minus oracle-fixed and feedback minus periodic,
  16 cells x 2 = 32 Bonferroni-adjusted paired MC t intervals (family 95%).
- Per-coordinate loss and bias are secondary, reported without selecting winners.
- Exact OU endpoints; trapezoidal continuous-time LS statistics with Ito
  correction. This is not an exact sampled-data likelihood or a mesh certificate.
- Refinement: all four pairs-dictionary T=256 cells, 1,024 fresh paired paths,
  meshes .02 and .01; sums of two fine normals give coarse marginal normals,
  not an exact OU bridge. Both control and observation spacing change.
- Refinement primary contrasts: 4 cells x 2 = 8 adjusted intervals at fine mesh.
- Seeds: development 2026111100+case, confirmation 2026112100+case,
  refinement 2026113100+case. Fixed case order is in the simulator.

Development used theta0=0 and different rotation vectors at T=64,n=128.
No policy, dictionary, truth, horizon, or estimator was chosen using
confirmation outcomes. The four development cells are retained.

## Interpretation limits

This is a structured multi-parameter family with isotropic pathwise information,
not an arbitrary unknown matrix. The small dictionaries are not asserted to
satisfy the sufficient worst-case angular covering condition. The selector is parameter-free. Continuous-time asymptotic bounds do not
certify finite-mesh risk.
