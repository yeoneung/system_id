# System identification experiments

Simulation code and retained results for estimating linear stochastic dynamics
while selecting feedback loops from a finite bank. The experiments compare
state-dependent projection selection with fixed loops, periodic schedules,
stationary mixtures, and a certainty-equivalent (CE) selector.

The repository includes simulation settings, seeds, per-path estimates and
sufficient statistics, paired comparisons, numerical checks, and plotting
scripts. All datasets are generated synthetically; no external data or GPU is
required.

## Setup

Use Python 3.11 and the pinned numerical packages:

```bash
git clone https://github.com/yeoneung/system_id.git
cd system_id
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the stored comparisons

```bash
python code/verify_release.py
python code/reproduce.py --suite analysis
```

The analysis suite reconstructs losses and paired comparisons from the stored
arrays and redraws the comparison figures. It runs in a separate copy under
`artifacts/reproduction/workspace/`, leaving the retained results unchanged.
CSV/JSON outputs are in that copy's `results/`; PDFs are in `figures/`. Logs and
an execution report are in `artifacts/reproduction/`.

For the structural tests, analytical numerical checks, HJB grids, and
discrete-time simulation as well:

```bash
python code/reproduce.py --suite all --output artifacts/full-check
```

The HJB suite solves sparse systems with up to 103,041 grid points and can take
several minutes. These commands do not rerun the large continuous-time
confirmation simulations. See [reproduction instructions](docs/REPRODUCING.md)
for full trajectory regeneration and separate suites. Each run requires a new
output directory.

## Experiments

Here, `d` is the state dimension and `p` is the number of unknown coefficients.
Path counts are per policy and condition; policies use paired innovations.

| Experiment | Configuration | Retained confirmation design | Results |
| --- | --- | --- | --- |
| Structured multi-parameter identification | `(d,p)=(6,2),(8,4)`; axes or signed-pair banks | 16 conditions, 1,024 paths, horizons 64 and 256 | [multiparameter](results/multiparameter/) |
| Coupled non-scalar information | `(d,p)=(6,4)`; 36 loops | 4 conditions, 1,024 paths, horizons 64 and 256 | [coupled_fourparam](results/coupled_fourparam/) |
| Known structural perturbations | `(d,p)=(6,2)`; 364 loops | 9 conditions, 512 paths, horizon 128 | [robustness](results/robustness/) |
| Separate control and observation clocks | `(d,p)=(6,2)`; holding times 0.02, 0.10, 0.50 | 4 conditions, 512 paths, horizon 128; two estimators | [hold](results/hold/) |
| Stationary CE comparison | `(d,p)=(6,2)`; 36 loops; holding time 0.10 | 4 conditions, 1,024 paths, horizons 64 and 256 | [ce](results/ce/) |
| Two-loop and HJB calculations | `d=2`; two-loop and irregular three-loop banks | Quadrature and 11 bounded reflecting grids | [extension_checks.json](results/extension_checks.json) |
| Discrete-time identification | `(d,p)=(2,1)`; two loops | 4,096 paired paths at horizons 256 and 1,024 | [discrete_identification.csv](results/discrete_identification.csv) |

Additional scripts check angular coverage, information/power identities, and
finite-time reference bounds. Their outputs are in `results/coverage/`,
`results/capacity_information/`, and `results/finite_time/`.

## Reading the results

`contrasts.csv` files report paired mean squared-error differences and
Bonferroni-adjusted Student-t intervals. Percentage reductions are ratios of
sample means; the intervals concern differences, not population risk ratios.
Protocols list every comparison and the adjustment family.

Continuous-time simulations use exact Gaussian transitions while a loop is
held fixed. Score and information integrals are numerical approximations on
the observation mesh. The stored mesh checks distinguish changes to both
decision and observation spacing from observation-only refinement on the
same physical paths.

The loop gain and available actions are matched across policies. Realized
input power can differ and is reported where measured. Stationary oracles
receive the true parameter as reference comparators; projection and CE do
not. The HJB values refer to bounded reflecting grids.

## Layout

- [`code/`](code/): simulators, analyses, plots, and numerical tests.
- [`results/`](results/): retained arrays and numerical summaries (about 45 MB).
- [`protocols/`](protocols/): model settings, policies, seeds, and comparisons.
- [`metadata/`](metadata/): SHA-256 integrity manifests.
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md): models, estimators, and data fields.
- [`docs/REPRODUCING.md`](docs/REPRODUCING.md): analysis and regeneration commands.

The export manifests identify this repository snapshot. They are not an
external preregistration or evidence of when an experimental design was fixed.
