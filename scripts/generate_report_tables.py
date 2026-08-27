#!/usr/bin/env python3
"""Generate the Apertus report tables from artifacts and inject them into the hand-maintained docs.

WHY THIS EXISTS. Whole-report generation was retired 2026-07-16: the docs in this repo are
hand-written and are the only place Apertus results are documented. But the *tables* in them were
hand-transcribed from artifacts, and that is where the errors were. The 2026-07-17 audit found the
vocabulary-usage table stale against a changed corpus, the throughput table matching no artifact at
all, and two vocabulary-utilisation cells transcribed from a superseded panel. Prose stays
hand-written; tables are generated.

HOW IT WORKS. Each managed table sits between markers in the document:

    <!-- BEGIN TABLE: throughput -->
    | Tokenizer | ... |
    ...
    <!-- END TABLE: throughput -->

Only the text between the markers is replaced. Prose, headings and everything else are untouched.
A missing marker pair is an ERROR, never an append: silently appending a second copy of a table is
worse than not writing it, because a reader cannot tell which one is current.

Usage:
    python scripts/generate_report_tables.py --check     # verify docs match artifacts, write nothing
    python scripts/generate_report_tables.py             # inject into the docs
    python scripts/generate_report_tables.py --table throughput --stdout
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

BEGIN = "<!-- BEGIN TABLE: {name} -->"
END = "<!-- END TABLE: {name} -->"


def load(rel: str):
    p = RESULTS / rel
    if not p.exists():
        raise SystemExit(f"ERROR: artifact not found: {p}\n"
                         f"       Run scripts/run_report.sh (or the individual step) first.")
    return json.loads(p.read_text())


def fmt(x, dec):
    """Fixed-decimal, matching how the docs display each column."""
    return f"{x:.{dec}f}"


def trim(x, dec):
    """Fixed-decimal with trailing zeros removed (the vocabulary-usage convention)."""
    s = f"{x:.{dec}f}".rstrip("0").rstrip(".")
    return s or "0"


# --- the six §3.1 / §5 rows, in document order, mapped to artifact keys -------------------------
FOCUS_ROWS = [
    ("Apertus v1", "Apertus"),
    ("preliminary_mul", "preliminary_mul"),
    ("preliminary_enh", "preliminary_enh"),
    ("preliminary_euh", "preliminary_euh"),
    ("preliminary_mul_200k", "preliminary_mul_200k"),
    ("*o200k (200k ref)*", "GPT-4o (o200k)"),
]


def table_throughput() -> str:
    pt = load("report_finewebedu_throughput.json")["finewebedu_eng_throughput"]["per_tokenizer"]
    out = ["| Tokenizer | Vocab | Ships `ignore_merges` | Encode MB/s |", "|---|---|---|---|"]
    for label, key in FOCUS_ROWS:
        r = pt[key]
        im = str(r["shipped_ignore_merges"]).lower()
        if im == "false":
            im = f"**{im}**"          # flag the odd one out, as the prose discusses it
        out.append(f"| {label} | {r['vocab_size']} | {im} | {fmt(r['encode_mb_per_s'], 2)} |")
    return "\n".join(out)


def table_vocab_usage() -> str:
    pt = load("report_nonemitting_tokens.json")["nonemitting_tokens"]["per_tokenizer"]
    out = ["| Tokenizer | Merge tokens | Active % | Rare % | Uncommon % | Unseen % | Scaffold % | Scaffold count |",
           "|---|---|---|---|---|---|---|---|"]
    for label, key in FOCUS_ROWS:
        r = pt[key]
        n, sc = r["n_merge_tokens"], r["pct_scaffold"]
        out.append(f"| {label} | {n:,} | {fmt(r['pct_active'],2)} | {fmt(r['pct_rare'],2)} | "
                   f"{fmt(r['pct_uncommon'],2)} | {fmt(r['pct_unseen'],2)} | {fmt(sc,2)} | "
                   f"{round(n*sc/100):,} |")
    return "\n".join(out)


def table_vocab_usage_ablations() -> str:
    """The full design-ablation table in DEVELOPMENT_RECORD. Row labels and their order are taken
    from the document itself, so the editorial naming stays hand-controlled; only numbers are
    generated. A label that cannot be resolved to an artifact key is an error, not a skipped row."""
    pt = load("report_nonemitting_tokens.json")["nonemitting_tokens"]["per_tokenizer"]
    src = (REPO / "archive/tooling/scripts/build_tokenizer_report.py").read_text()
    body = re.search(r"^DISPLAY\s*=\s*\{(.*?)^\}", src, re.S | re.M).group(1)
    inv = {v: k for k, v in re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', body)}

    def resolve(name):
        if name in inv and inv[name] in pt:
            return inv[name]
        if name in pt:
            return name
        base = name.replace(" (tuned data)", "")
        return inv.get(base) if inv.get(base) in pt else None

    doc = (REPO / "DEVELOPMENT_RECORD.md").read_text().splitlines()
    labels, inside = [], False
    for line in doc:
        if line.strip() == BEGIN.format(name="vocab-usage-ablations"):
            inside = True
            continue
        if line.strip() == END.format(name="vocab-usage-ablations"):
            break
        if inside and line.startswith("| ") and not line.startswith("|---"):
            lab = line.strip().strip("|").split("|")[0].strip()
            if lab != "Tokenizer":
                labels.append(lab)
    if not labels:
        raise SystemExit("ERROR: no existing rows found inside the vocab-usage-ablations markers; "
                         "the row labels are read from the document and cannot be inferred.")
    out = ["| Tokenizer | Vocab util ↑ | Active % | Rare % | Uncommon % | Unseen % | Scaffold % |",
           "|---|---|---|---|---|---|---|"]
    for lab in labels:
        k = resolve(lab)
        if k is None:
            raise SystemExit(f"ERROR: cannot map row label {lab!r} to an artifact key.")
        r = pt[k]
        out.append(f"| {lab} | {fmt(r['util_on_corpus_full_vocab'],3)} | {trim(r['pct_active'],2)} | "
                   f"{trim(r['pct_rare'],2)} | {trim(r['pct_uncommon'],2)} | {trim(r['pct_unseen'],2)} | "
                   f"{trim(r['pct_scaffold'],2)} |")
    return "\n".join(out)


TABLES = {
    "throughput":            ("REPORT_focus_candidates.md", table_throughput),
    "vocab-usage":           ("REPORT_focus_candidates.md", table_vocab_usage),
    "vocab-usage-ablations": ("DEVELOPMENT_RECORD.md",      table_vocab_usage_ablations),
}


def inject(doc_path: Path, name: str, body: str, check: bool) -> bool:
    """Replace the marker block. Returns True if the document already matched."""
    text = doc_path.read_text()
    b, e = BEGIN.format(name=name), END.format(name=name)
    if b not in text or e not in text:
        raise SystemExit(
            f"ERROR: markers for table {name!r} not found in {doc_path.name}.\n"
            f"       Add these around the table, then re-run:\n         {b}\n         ...\n         {e}\n"
            f"       Refusing to append: a second copy of a table is worse than none.")
    pre, rest = text.split(b, 1)
    _, post = rest.split(e, 1)
    new = f"{pre}{b}\n{body}\n{e}{post}"
    if new == text:
        return True
    if not check:
        doc_path.write_text(new)
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", choices=sorted(TABLES) + ["all"], default="all")
    ap.add_argument("--check", action="store_true",
                    help="verify the docs already match the artifacts; write nothing, exit 1 on drift")
    ap.add_argument("--stdout", action="store_true", help="print the table instead of injecting")
    args = ap.parse_args()

    names = sorted(TABLES) if args.table == "all" else [args.table]
    drift = []
    for name in names:
        doc_name, builder = TABLES[name]
        body = builder()
        if args.stdout:
            print(f"--- {name} ({doc_name}) ---\n{body}\n")
            continue
        same = inject(REPO / doc_name, name, body, args.check)
        state = "up to date" if same else ("DRIFT" if args.check else "updated")
        print(f"  {name:24s} -> {doc_name:28s} {state}")
        if args.check and not same:
            drift.append(name)

    if drift:
        print(f"\n{len(drift)} table(s) differ from the artifacts: {drift}\n"
              f"Run without --check to regenerate.")
        sys.exit(1)


if __name__ == "__main__":
    main()
