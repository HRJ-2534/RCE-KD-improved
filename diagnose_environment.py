"""Create a comparable fingerprint of two RCE-KD Python environments.

This script deliberately uses only the Python standard library.  It can still
run when importing NumPy/SciPy/scikit-learn is broken.  Run it once in the
known-good instance and once in the failing instance, then compare the JSON
files with ``--compare``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import site
import subprocess
import sys
from typing import Any


PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
}

RELEVANT_ENV = (
    "CONDA_DEFAULT_ENV",
    "CONDA_PREFIX",
    "LD_LIBRARY_PATH",
    "OMP_NUM_THREADS",
    "PATH",
    "PYTHONHOME",
    "PYTHONPATH",
    "VIRTUAL_ENV",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_fingerprint(root: Path) -> dict[str, Any]:
    """Hash actual source/binary package files, including untracked leftovers."""
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    errors: list[str] = []

    if not root.exists():
        return {"root": str(root), "exists": False}

    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    for path in paths:
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        try:
            relative = path.name if root.is_file() else path.relative_to(root).as_posix()
            file_hash = sha256_file(path)
            size = path.stat().st_size
            digest.update(relative.encode("utf-8", errors="surrogateescape"))
            digest.update(b"\0")
            digest.update(file_hash.encode("ascii"))
            digest.update(b"\0")
            count += 1
            total_bytes += size
        except OSError as exc:
            errors.append(f"{path}: {exc}")

    return {
        "root": str(root),
        "exists": True,
        "file_count": count,
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
        "errors": errors,
    }


def distribution_info(distribution_name: str, module_name: str) -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        distribution = importlib.metadata.distribution(distribution_name)
        info["version"] = distribution.version
        info["metadata_path"] = str(distribution._path)  # type: ignore[attr-defined]
    except Exception as exc:  # metadata must not prevent the remaining diagnosis
        info["metadata_error"] = f"{type(exc).__name__}: {exc}"

    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            info["module_spec"] = None
            return info
        info["module_origin"] = spec.origin
        roots = list(spec.submodule_search_locations or [])
        root = Path(roots[0]) if roots else Path(spec.origin or "")
        info["package_tree"] = tree_fingerprint(root)
    except Exception as exc:
        info["spec_error"] = f"{type(exc).__name__}: {exc}"
    return info


def run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def collect() -> dict[str, Any]:
    probe = (
        "import numpy; import scipy; import scipy.stats; import sklearn; "
        "print(numpy.__version__, numpy.__file__); "
        "print(scipy.__version__, scipy.__file__); "
        "print(sklearn.__version__, sklearn.__file__)"
    )
    executable = Path(sys.executable)
    report: dict[str, Any] = {
        "python": {
            "executable": str(executable),
            "executable_sha256": sha256_file(executable),
            "version": sys.version,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "path": sys.path,
            "site_packages": site.getsitepackages(),
            "user_site": site.getusersitepackages(),
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "uname": list(platform.uname()),
            "libc": list(platform.libc_ver()),
        },
        "environment": {name: os.environ.get(name) for name in RELEVANT_ENV},
        "packages": {
            distribution: distribution_info(distribution, module)
            for distribution, module in PACKAGES.items()
        },
        "normal_import": run_command([sys.executable, "-c", probe]),
        "isolated_import": run_command([sys.executable, "-I", "-c", probe]),
        "pip_check": run_command([sys.executable, "-m", "pip", "check"]),
    }
    return report


def differences(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        result: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else key
            if key not in left:
                result.append({"path": child, "old": "<missing>", "new": right[key]})
            elif key not in right:
                result.append({"path": child, "old": left[key], "new": "<missing>"})
            else:
                result.extend(differences(left[key], right[key], child))
        return result
    if left != right:
        return [{"path": path, "old": left, "new": right}]
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="write the current fingerprint to this JSON file")
    parser.add_argument("--compare", nargs=2, metavar=("OLD_JSON", "NEW_JSON"))
    args = parser.parse_args()

    if args.compare:
        old = json.loads(Path(args.compare[0]).read_text(encoding="utf-8"))
        new = json.loads(Path(args.compare[1]).read_text(encoding="utf-8"))
        print(json.dumps(differences(old, new), indent=2, ensure_ascii=False))
        return

    report = collect()
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        print(args.output)
    else:
        print(rendered)


if __name__ == "__main__":
    main()
