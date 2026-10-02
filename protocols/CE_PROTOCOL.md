# Prospective stationary-CE comparison

Export note: this description retains the recorded experimental settings.
The checksums in `metadata/` verify this repository snapshot; they are not
an external preregistration or a new prospective freeze.


The design and analysis sources are frozen before any confirmation trajectory.
This is an internal record, not external preregistration. All four cells and
all declared comparisons are reported, regardless of signs or significance.

## Model and policies

Reuse the retained structured d=6, p=2 model, a=1, b=4, known skew norm .15,
and 36 signed-pair directions. Truths A=(-.2,.35), B=(.2,-.3); T=64,256;
1,024 paired path indices per cell. The known box is [-.25,.25] x [-.4,.4].
Three policies: best stationary fixed-loop oracle at the true parameter,
stationary certainty-equivalent (CE) selection, and projection selection.
Only the oracle is given the true parameter. CE and projection both decide
every .10 time units, from the state/history available at that time.

CE uses the running continuous-score LS approximation with the theoretical
guard G_t >= kappa*t/2, kappa=6/[2(4+.25)], and returns zero if the guard
fails (including t=0). For loop selection only, clip its estimate to the
known box. Choose the first maximizing loop for stationary energy at that
estimate. Evaluate energies by bilinear interpolation of exact Lyapunov
solutions on a .005 x .005 parameter grid (101 x 161 nodes, 36 loops).
This numerical stationary-CE baseline is not an implementation of TOPLE.
It has no separate exploratory phase or horizon-dependent optimization.

Grid approximation is checked before confirmation at 1,024 independently
chosen box points (seed 2026092209). Require maximum relative energy error
and selection regret below 1e-4 at these points; this is an empirical numerical
check, not a uniform interpolation certificate. After confirmation, also
rescore every saved diagnostic estimate using all 36 exact Lyapunov solutions.
Save the first 8 paths at approximately 64 equally spaced decision indices
per cell, with the schedule fixed in the simulator. Report disagreements
and energy regret even if a check exceeds the development threshold.

## Observations, scores, and fairness

Exact Gaussian held-loop transitions generate physical paths on mesh .005.
Independent path indices share Gaussian innovations across the three policies.
Both .01 and nested .005 score reconstructions use these same physical paths,
terminal states, and decisions. CE uses the primary .01 reconstruction for
decisions throughout; the fine estimator is only an observation diagnostic.
The primary terminal estimator for every policy is unprojected guarded LS,
with the same guard as CE. All power, guard, clipping and action counts are saved.
There is no true-parameter use in projection or CE loop selection.

## Outcomes and uncertainty

Primary outcome: total two-coordinate MSE on observation mesh .01.
Eight primary paired contrasts: projection minus CE and projection minus
fixed oracle in each of the four cells. One Bonferroni family, two-sided .05,
paired Student-t intervals (Monte Carlo approximations, not exact Gaussian-loss
confidence coverage). Report reductions as ratios of sample means.
Four CE-minus-oracle comparisons form a secondary Bonferroni family.
Twelve fine-minus-coarse comparisons form a separate secondary family.
No significance criterion controls inclusion, rerunning, or model changes.

Seeds 2026092200, 2026092201, 2026092202, 2026092203 correspond to A/64,
A/256, B/64, B/256. Development alone uses (.05,-.15), T=4, n=32,
seed 2026092199. Three policies produce 12,288 confirmation and 96 development
physical rollouts; two meshes do not multiply these counts.

## Reproduction

Run run_ce_feedback.py --task table, then --task check and --task development.
Run --task confirmation --case 0 through 3, then analyze_ce_feedback.py.
The retained energy table is deterministic. Export checksums are recorded in
metadata/ce_manifest.json.
