# Additional non-Clifford six-state/four-parameter protocol

Export note: this description retains the recorded experimental settings.
The checksums in `metadata/` verify this repository snapshot; they are not
an external preregistration or a new prospective freeze.


This separate protocol is frozen before its confirmation runs. The earlier
structured-family confirmation has already begun; no results from this new
confirmation family have been observed. Development is one different truth
(0,.2,.1,-.1), T=64, n=128, seed 2026111200. Its result is retained.

The model uses d=6, p=4, L=36 signed-pair directions, a=1,b=4. Its features
are I, a dense skew complex structure, and two independent dense symmetric
trace-zero matrices of operator norm one, seeded 2026110601. The known dense
skew drift is the same as in the structured six-dimensional experiment.
The information matrix is generally non-diagonal and is not a scalar times I.
The class widths are (.25,.4,.2,.2), giving common dissipativity gamma=.35.

Confirmation has truths (-.2,.35,.15,-.12),(.2,-.3,-.12,.15), horizons 64,256,
and 1,024 paths per policy in each of four cells. All outcomes are retained.
Policies: oracle A-optimal stationary time mixture, oracle best fixed action,
periodic, and observed-state largest-projection feedback. Both oracle policies
receive the truth, explicitly as strong comparators; feedback does not.
The numerical mixture first-order optimality residual is reported, not
described as a rigorous optimizer certificate. No claim of finite-time oracle
optimality is made.

Primary contrasts: feedback minus oracle mixture and feedback minus periodic,
four cells x two = eight Bonferroni-adjusted paired MC t intervals. Additionally
apply the more conservative 40-comparison adjustment jointly with the 32
structured-family confirmation contrasts. Loss is unprojected squared error
over all four coefficients. Coordinate losses and biases are secondary.

Exact OU endpoints use mesh .02, with trapezoidal continuous-time LS statistics
and the Ito correction for every feature. This is not exact sampled-data MLE.
Refinement repeats BOTH T=256 truth cells with 1,024 fresh paired paths at
.02 and .01, testing four fine-mesh primary comparisons. Both action and
observation spacing change. Coupling pairs marginal Gaussian innovations;
it is not an exact OU bridge.

Seeds are 2026112200+case (confirmation), 2026113200+case (refinement).
No model feature or policy is selected using the new confirmation outcomes.
