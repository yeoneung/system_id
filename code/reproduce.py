"""Run stored-data analyses and numerical checks in a separate workspace."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time

from verify_release import verify

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "analysis": [
        ["analyze_multiparameter.py"],
        ["analyze_coupled_fourparam.py"],
        ["analyze_robust_feedback.py"],
        ["analyze_hold_feedback.py"],
        ["analyze_ce_feedback.py"],
        ["plot_two_loop_design.py"],
    ],
    "checks": [
        ["test_multiparameter_feedback.py"],
        ["test_coupled_fourparam.py"],
        ["test_robust_feedback.py"],
        ["test_hold_feedback.py"],
        ["run_ce_feedback.py", "--task", "check"],
        ["coverage_geometry.py"],
        ["finite_time_certificate.py"],
        ["hold_certificate.py"],
        ["verify_hold_witness.py"],
        ["finite_time_improved_witness.py"],
        ["check_capacity_information.py"],
    ],
    "extensions": [["check_extensions.py"]],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=[*SUITES, "all"], default="analysis")
    parser.add_argument("--output", type=Path, default=Path("artifacts/reproduction"))
    args = parser.parse_args()
    verified = verify()
    output = args.output.resolve()
    if output.exists():
        parser.error("output already exists; choose a new directory")
    if output == ROOT or ROOT in output.parents and output.parts[len(ROOT.parts)] != "artifacts":
        parser.error("use an output directory under artifacts/ or outside the repository")
    output.mkdir(parents=True)
    workspace = output / "workspace"
    workspace.mkdir()
    for name in ["code", "results", "metadata", "protocols"]:
        shutil.copytree(ROOT / name, workspace / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (workspace / "figures").mkdir()
    logs = output / "logs"
    logs.mkdir()
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", MPLBACKEND="Agg",
               MPLCONFIGDIR=str(output / "matplotlib-cache"))
    commands = (sum(SUITES.values(), []) if args.suite == "all" else SUITES[args.suite])
    report = {
        "suite": args.suite,
        "verified_snapshot_files": verified,
        "environment": {"python": platform.python_version(),
                        "system": platform.system(), "machine": platform.machine(),
                        **{name: importlib.metadata.version(name)
                           for name in ["numpy", "scipy", "matplotlib"]}},
        "blas_threads": 1,
        "runs": [],
    }
    report_path = output / "report.json"
    for i, command in enumerate(commands, 1):
        print(f"[{i}/{len(commands)}] {' '.join(command)}", flush=True)
        log_path = logs / f"{i:02d}_{Path(command[0]).stem}.log"
        start = time.monotonic()
        with log_path.open("w") as log:
            result = subprocess.run([sys.executable, str(workspace / "code" / command[0]),
                                     *command[1:]], cwd=workspace, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        report["runs"].append({"command": command, "returncode": result.returncode,
                               "elapsed_seconds": round(time.monotonic() - start, 3),
                               "log": str(log_path.relative_to(output))})
        report["passed"] = all(item["returncode"] == 0 for item in report["runs"])
        report["complete"] = i == len(commands)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        if result.returncode:
            print(log_path.read_text()[-6000:], file=sys.stderr)
            raise SystemExit(result.returncode)
    print(f"Completed {len(commands)} commands. Outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
