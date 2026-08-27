#!/usr/bin/env python3
"""Verify that numbers in the Apertus docs actually match their source artifacts.

Ported verbatim from tokenizer-lm/scripts/check_number_provenance.py (2026-07-16); only the
default file list differs. The design rationale below is unchanged and load-bearing.

WHY THIS DESIGN (read before "improving" it).

The obvious checker is: pool every value from the result artifacts, then flag any number in
the markdown that does not appear in the pool. **That does not work, and it was measured.**
Against the repo's artifacts, 100% of randomly generated 4-decimal numbers "match" something,
and two of the three known-wrong numbers this project has produced (rho=-0.681, mean=0.0141)
matched a pool and would have passed. Value-presence is not verification.

So a claim must be bound to a SPECIFIC artifact AND key, and compared to THAT value:

    - Apertus v1 AST full-alignment 0.488. <!--prov: results/report_flores60/analysis_results_full.json#ast_boundary_alignment.per_tokenizer.Apertus.overall.full_alignment_rate-->

The checker resolves the key, rounds the artifact value to the precision written in the doc,
and requires some number on that line to equal it. A binding whose artifact or key is missing
is an ERROR, not a skip: a dangling provenance claim is worse than none.

Artifacts live under results/ (gitignored, but present on disk); paths in bindings are relative
to the repo root. Path syntax after '#': dot-separated keys, with [i] for list indices. Keys may
contain spaces and dots-in-names are matched greedily left to right against dict keys.

Modes:
  (default)   check bindings; report unbound decimals as a per-file drift warning
  --strict    also fail when a file with numbers has no ## Provenance section
  --list-unbound FILE   print the unbound decimals in FILE (to help add bindings)

Exit code is nonzero if any binding mismatches or dangles.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PROV_RE = re.compile(r"<!--\s*prov:\s*([^#\s]+)#([^>]+?)\s*-->")
# A decimal, optionally signed, optionally in scientific notation. Not part of a word/version.
NUM_RE = re.compile(r"(?<![\w.])([-+]?\d+\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+[eE][-+]?\d+)(?![\w])")
SEG_RE = re.compile(r"\[([^\]]+)\]|([^.\[\]]+)")

IGNORE_LINE_RE = re.compile(r"^\s*(\||#|>)?\s*$")


def resolve(doc, path: str):
    """Resolve a dot/bracket path. Raises KeyError/IndexError if absent."""
    cur = doc
    for m in SEG_RE.finditer(path):
        seg = m.group(1) if m.group(1) is not None else m.group(2)
        seg = seg.strip()
        if isinstance(cur, list):
            cur = cur[int(seg)]
        elif isinstance(cur, dict):
            if seg not in cur:
                raise KeyError(seg)
            cur = cur[seg]
        else:
            raise KeyError(f"cannot descend into {type(cur).__name__} at {seg!r}")
    return cur


def load(path: Path):
    if path.suffix == ".json":
        return json.loads(path.read_text())
    raise ValueError(f"unsupported artifact type: {path}")


def decimals_of(tok: str) -> int:
    body = tok.split("e")[0].split("E")[0]
    return len(body.split(".")[1]) if "." in body else 0


GENERATED_RE = re.compile(r"DO NOT EDIT BY HAND|^Generated (by|\d{4})", re.M | re.I)
GENERATOR_PATH_RE = re.compile(r"[`\s(]((?:scripts|scratchpad)/[\w./-]+\.py)")


def is_generated(text: str) -> bool:
    """A script-written file IS an artifact. It needs a banner and a regeneration check, not
    per-number bindings, so its numbers are not counted as unbound drift."""
    return bool(GENERATED_RE.search(text[:1200]))


def check_file(md: Path, verbose: bool) -> tuple[list[str], int, int]:
    errors: list[str] = []
    text = md.read_text()
    lines = text.splitlines()
    n_bound = 0
    cache: dict[Path, object] = {}

    for i, line in enumerate(lines, 1):
        for m in PROV_RE.finditer(line):
            n_bound += 1
            art_rel, key = m.group(1), m.group(2)
            art = (REPO / art_rel)
            if not art.exists():
                errors.append(f"{md}:{i}: DANGLING artifact does not exist: {art_rel}")
                continue
            try:
                doc = cache.get(art) or cache.setdefault(art, load(art))
                val = resolve(doc, key)
            except (KeyError, IndexError, ValueError) as e:
                errors.append(f"{md}:{i}: DANGLING key {key!r} not in {art_rel} ({e})")
                continue
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                errors.append(f"{md}:{i}: key {key!r} is {type(val).__name__}, not a number")
                continue

            nums = [t for t in NUM_RE.findall(line)]
            if not nums:
                errors.append(f"{md}:{i}: binding present but no number on the line")
                continue
            ok = any(round(float(t), decimals_of(t)) == round(float(val), decimals_of(t))
                     for t in nums)
            if not ok:
                errors.append(
                    f"{md}:{i}: MISMATCH {art_rel}#{key} = {val!r}; "
                    f"line has {nums}")
            elif verbose:
                print(f"  ok {md.name}:{i}  {key} = {val}")

    if is_generated(text):
        for m in GENERATOR_PATH_RE.finditer(text[:1200]):
            gen = m.group(1)
            if gen.startswith("scratchpad/") or not (REPO / gen).exists():
                errors.append(
                    f"{md}: claims to be generated by {gen!r}, which does not exist in the repo. "
                    f"Its numbers are NOT reproducible; mark them UNVERIFIABLE or restore the script.")
        return errors, n_bound, -1  # -1 signals GENERATED

    bound_lines = {i for i, line in enumerate(lines, 1) if PROV_RE.search(line)}
    unbound = 0
    for i, line in enumerate(lines, 1):
        if i in bound_lines or IGNORE_LINE_RE.match(line):
            continue
        unbound += len(NUM_RE.findall(line))
    return errors, n_bound, unbound


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path,
                    help="markdown files (default: the live Apertus docs)")
    ap.add_argument("--strict", action="store_true",
                    help="also fail on a numbers-bearing file with no ## Provenance section")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    files = args.files or [REPO / "REPORT_focus_candidates.md",
                           REPO / "DEVELOPMENT_RECORD.md",
                           REPO / "BLOG_DRAFT.md"]

    all_errors, total_bound = [], 0
    print(f"{'file':52s} {'bound':>6s} {'unbound':>8s}  status")
    print("-" * 82)
    for f in files:
        f = f if f.is_absolute() else (REPO / f)
        if not f.exists():
            continue
        errs, bound, unbound = check_file(f, args.verbose)
        total_bound += bound
        gen = unbound == -1
        if args.strict and not gen and unbound > 0 and "## Provenance" not in f.read_text():
            errs.append(f"{f}: has {unbound} numbers but no ## Provenance section")
        all_errors += errs
        status = "FAIL" if errs else ("GENERATED" if gen else ("ok" if bound else "NO BINDINGS"))
        try:
            shown = str(f.relative_to(REPO))
        except ValueError:
            shown = str(f)
        ub = "  -" if unbound == -1 else str(unbound)
        print(f"{shown:52s} {bound:6d} {ub:>8s}  {status}")

    if all_errors:
        print(f"\n{len(all_errors)} problem(s):")
        for e in all_errors:
            print(f"  {e}")
        sys.exit(1)
    print(f"\nAll {total_bound} bound claim(s) match their artifacts.")


if __name__ == "__main__":
    main()
