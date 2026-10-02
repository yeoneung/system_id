"""Verify the SHA-256 manifest for the distributed experiment snapshot."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    record = json.loads((root / "metadata/release_manifest.json").read_text())
    failures = []
    for name, expected in record["sha256"].items():
        path = root / name
        if not path.is_file():
            failures.append(f"missing: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(f"changed: {name}")
    if failures:
        raise RuntimeError("Snapshot integrity check failed:\n" + "\n".join(failures))
    print(f"Verified {len(record['sha256'])} snapshot files.", flush=True)
    return len(record["sha256"])


if __name__ == "__main__":
    verify()
