#!/usr/bin/env python3
"""Emit the AST-alignment vs MBPP rank correlation used in DEVELOPMENT_RECORD.md.

The document claims "Spearman rho(AST align, MBPP) = +0.657 (p=0.011, n=14)". That value was
computed ad hoc and typed in; the panel underneath has since been rebuilt (report_flores60 now
carries REAL StarCoder AST) and one tokenizer gained an MBPP, so it drifted. This script
recomputes it from the current artifacts and writes results/ast_mbpp_correlation.json, so the
document can bind to a key rather than carry a hand-typed number.

Join: every tokenizer that has BOTH a real AST full-alignment (report_flores60) AND a real
(non-string) MBPP (report_extrinsic_index). Keyed by friendly name.

Run: ~/tokenizer-intrinsic-evals/.venv/bin/python scripts/ast_mbpp_correlation.py
"""
import json
import os

from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AST_PANEL = os.path.join(REPO, "results", "report_flores60", "analysis_results_full.json")
EXTRINSIC = os.path.join(REPO, "results", "report_extrinsic_index.json")
OUT = os.path.join(REPO, "results", "ast_mbpp_correlation.json")


def main():
    ast_pt = json.load(open(AST_PANEL))["ast_boundary_alignment"]["per_tokenizer"]
    ex = json.load(open(EXTRINSIC))["extrinsic_index"]["per_tokenizer"]
    ast = {t: v["overall"]["full_alignment_rate"] for t, v in ast_pt.items()}
    mbpp = {t: r["mbpp"] for t, r in ex.items() if not isinstance(r.get("mbpp"), str)}
    both = sorted(t for t in ast if t in mbpp)
    A = [ast[t] for t in both]
    M = [mbpp[t] for t in both]
    rho, p = stats.spearmanr(A, M)

    out = {
        "_meta": {
            "description": "Spearman rank correlation of AST full-alignment (real StarCoder, "
                           "report_flores60) against MBPP pass@1 (report_extrinsic_index), over "
                           "every tokenizer that has both a real value.",
            "ast_source": "results/report_flores60/analysis_results_full.json"
                          "#ast_boundary_alignment.per_tokenizer.<t>.overall.full_alignment_rate",
            "mbpp_source": "results/report_extrinsic_index.json"
                           "#extrinsic_index.per_tokenizer.<t>.mbpp",
        },
        "spearman_rho": round(float(rho), 4),
        "p_value": round(float(p), 4),
        "n": len(both),
        "tokenizers": both,
        "ast": {t: round(ast[t], 4) for t in both},
        "mbpp": {t: round(mbpp[t], 4) for t in both},
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"n={out['n']}  Spearman rho={out['spearman_rho']:+.4f}  p={out['p_value']:.4f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
