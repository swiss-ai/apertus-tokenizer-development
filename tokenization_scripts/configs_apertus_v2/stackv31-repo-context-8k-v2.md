# Stack v3.1 repository context at 8k

Use [the Clariden config](stackv31-repo-context-8k-v2.cfg) or the
[RCP config](../configs_apertus_v2_rcp/stackv31-repo-context-8k-v2.cfg)
with the standard [tokenization workflow](../README.md).

The prepared release is `stackv31-repo-context-8k-v2`, converted directly from
`HuggingFaceCode/stack-v3-train@1f61b735bc0a5698345ce2196730f24bfa467f33`.
It requires the prepared `_SUCCESS` and `examples_manifest.jsonl`, preserves
all four category groups, and writes to distinct 8k output roots. The 8192-token
limit includes BOS/EOS; maps resolve to the sealed prepared source root.
Strict validation checks token/index/map consistency and source coverage.
