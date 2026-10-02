# Reproduction

## Stored-data analysis and checks

From the repository root, after installing `requirements.txt`:

```bash
python code/verify_release.py
python code/reproduce.py --suite analysis --output artifacts/analysis
python code/reproduce.py --suite checks --output artifacts/checks
python code/reproduce.py --suite extensions --output artifacts/extensions
```

Or run all three suites with `--suite all`. Each command first verifies the
repository manifest, copies the code and data into its output directory, and
runs with one BLAS/OpenMP thread. Existing output directories are rejected to
avoid mixing runs. Individual script output is saved to `logs/`; `report.json`
records commands, dependency versions, return codes, and elapsed times.

| Suite | Work performed |
| --- | --- |
| `analysis` | Five stored-data analyzers, paired comparisons, diagnostic reconstructions, and two-loop plots |
| `checks` | Four structural test scripts, CE table/transition checks, coverage quadrature and sampling, finite-time and held-loop reference checks, information/power identities |
| `extensions` | Two-loop quadrature, all 11 HJB grids, and both discrete-time horizons |

`all` runs 18 commands. It does not regenerate the large continuous-time
confirmation or refinement datasets. Direct scripts write to `results/` and
`figures/` relative to their own parent directory; use the copied workspace
for further runs. Numerical libraries and sparse solvers may produce small
platform-dependent floating-point differences. The pinned environment uses
Python 3.11, NumPy 1.26.4, SciPy 1.13.1, and Matplotlib 3.9.2.

## Regenerate continuous-time simulations

Full regeneration is substantially more expensive than analyzing stored
arrays. First create a separate workspace, then run the simulators there:

```bash
python code/reproduce.py --suite analysis --output artifacts/regenerated
cd artifacts/regenerated/workspace
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python code/run_multiparameter_feedback.py --phase development --n 128
python code/run_multiparameter_feedback.py --phase confirmation --n 1024
python code/run_multiparameter_feedback.py --phase refinement --n 1024

python code/run_coupled_fourparam.py --phase development --n 128
python code/run_coupled_fourparam.py --phase confirmation --n 1024
python code/run_coupled_fourparam.py --phase refinement --n 1024

python code/run_robust_feedback.py --phase development
python code/run_robust_feedback.py --phase confirmation
python code/run_robust_feedback.py --phase refinement

python code/run_hold_feedback.py --phase development
python code/run_hold_feedback.py --phase confirmation

python code/run_ce_feedback.py --task development
python code/run_ce_feedback.py --task confirmation --case 0
python code/run_ce_feedback.py --task confirmation --case 1
python code/run_ce_feedback.py --task confirmation --case 2
python code/run_ce_feedback.py --task confirmation --case 3
```

The CE energy table is included and checked before confirmation. It can be
recomputed with `python code/run_ce_feedback.py --task table`, but numerical
libraries can change its exact bytes. Keep the retained table for reproducing
the recorded configuration; do not relabel a changed table as the same snapshot.

The robustness analyzer checks the hashes of the retained NPZ files before
accepting them. To analyze newly generated paths, first move the copied
`metadata/robustness_data_provenance.json` to
`metadata/robustness_data_provenance.retained.json`. This preserves the
reference hashes and lets the analyzer record the regenerated dataset's
hashes separately. Do this only in the copied workspace. Then run:

```bash
python code/analyze_multiparameter.py
python code/analyze_coupled_fourparam.py
python code/analyze_robust_feedback.py
python code/analyze_hold_feedback.py
python code/analyze_ce_feedback.py
```

The source/protocol manifests remain the same. Regenerated data are a new
local run and may not be byte-identical across platforms. The root release
manifest always describes the distributed files, not subsequent local runs.

## Integrity and export scope

`metadata/release_manifest.json` covers source, documentation, dependency
specifications, and retained outputs. Per-experiment manifests cover the
simulator dependencies and protocol descriptions used by the runners.

The export preserves model construction, seeds, policy rules, estimators,
comparison families, and raw NPZ files. Packaging changes consist of neutral
documentation, repository-relative metadata paths, removal of LaTeX table
formatting, and extraction of the A-optimal mixture solver from a larger
utility module. No confirmation cases or comparison outcomes are filtered.

These are distribution integrity checks. They do not establish prospective
registration, independent review, or numerical accuracy beyond the checks
that are explicitly run.
