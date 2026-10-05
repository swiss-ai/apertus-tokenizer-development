"""Freeze/check the standard HPLT dump inventory across whole-node retries."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(root, pins):
    receipt = root / "HPLT_DUMPS.json"
    old = json.loads(receipt.read_text())
    if old["pins"] != pins:
        raise ValueError("dump input/config/runtime pins changed")
    expected = set(old["dumps"])
    observed = set()
    for state in ("dumps", "completed-dumps"):
        for path in (root / state).rglob("paths_file_*.txt"):
            relative = path.relative_to(root / state).as_posix()
            if relative not in expected or relative in observed:
                raise ValueError("unexpected/duplicate dump completion")
            if sha(path) != old["dumps"][relative]:
                raise ValueError("dump inventory changed")
            observed.add(relative)
    if observed != expected:
        raise ValueError("a pending/completed dump inventory is missing")
    return old


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "check"])
    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help="standard preparation metadata root",
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--marker", required=True, type=Path, help="prepared release completion marker"
    )
    parser.add_argument(
        "--manifest", required=True, type=Path, help="prepared exact file manifest"
    )
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument(
        "--runtime-commit", required=True, help="pinned source-map runtime Git commit"
    )
    parser.add_argument(
        "--implementation-commit", required=True, help="tokenizer producer Git commit"
    )
    args = parser.parse_args()
    marker = json.loads(args.marker.read_text())
    if marker.get("complete") is not True or marker["pins"][
        "examples_manifest_sha256"
    ] != sha(args.manifest):
        raise ValueError("prepared population is not sealed")
    pins = {
        "config": sha(args.config),
        "marker": sha(args.marker),
        "manifest": sha(args.manifest),
        "tokenizer": sha(args.tokenizer),
        "runtime_commit": args.runtime_commit,
        "implementation_commit": args.implementation_commit,
    }
    receipt = args.metadata / "HPLT_DUMPS.json"
    if args.stage == "freeze":
        if receipt.exists():
            raise ValueError("dump inventory is already frozen")
        dumps = {
            path.relative_to(args.metadata / "dumps").as_posix(): sha(path)
            for path in sorted((args.metadata / "dumps").rglob("paths_file_*.txt"))
        }
        if not dumps:
            raise ValueError("no prepared dumps")
        partial = receipt.with_suffix(".partial")
        partial.write_text(
            json.dumps({"pins": pins, "dumps": dumps}, sort_keys=True, indent=2) + "\n"
        )
        os.replace(partial, receipt)
    check(args.metadata, pins)


if __name__ == "__main__":
    main()
