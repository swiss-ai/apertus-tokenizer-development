# HPLT finite-score recovery tokenization

The four `configs_apertus_v2/hplt-ia-finite-score-salvage-*.cfg` configurations
consume only the newly recovered eight-language populations from the HPLT
pipeline's finite-stored-score recovery. Other languages and English retain their
existing token releases. This is provisional: finite embedding corruption remains
eligible. The changed selection has a new prepared and token identity.

The producer seals `examples_manifest.jsonl` and `_SUCCESS.json` in each variant.
The standard preparation script reads that exact manifest, groups by language,
and sizes dumps at 4 GiB compressed Parquet. Workers use up to 32 processes and four
tokenizer threads each (capped by the number of files in a dump), with batches bounded by 128 documents and 32 MiB UTF-8.
They preserve whole documents and direct prepared-file row coordinates and IDs.

Use pipeline runtime `fc1e5a86526e38c5f31aa6a099473f5a95b511e7` (provenance and
LC reserve APIs), Datatrove 0.6.0/Arrow 22.0.0, and the repository's
`preliminary_mul_200k/tokenizer.json` SHA-256
`cd403d3f219e2433e3f78b32644b8e6a6134668e15138e6546360330635a96b9`.
The CPU wrapper accepts the installed Clariden EDF explicitly. Run preparation
inside an allocated container using the standard direct-process backend:

```bash
export PYTHONPATH="$PIPELINE_RUNTIME/src"
export TOKENIZATION_LAUNCH_BACKEND=rcp
bash tokenization_scripts/tokenize_script.sh "$CONFIG" --prepare-only
```

The primary whole-node launcher handles preparation, all remaining standard dump
workers, strict validation and the LC reserve in one allocation per variant:

```bash
sbatch --account=infra01 --partition=preemptable --nodes=1 --ntasks=1 \
  --cpus-per-task=128 --mem=512G --gpus=0 --time=06:00:00 --requeue \
  tokenization_scripts/hplt_finite_score_dataset.sbatch \
  "$CONFIG" "$PIPELINE_RUNTIME" "$EDF" "$CONTROL"
```

Use a fresh control root and one allocation per configuration. The launcher
freezes `HPLT_DUMPS.json` before workers start, binding configuration, code, runtime,
prepared marker/manifest and tokenizer hashes. Retries check pending/completed
dump hashes and never recreate a completed dump. Missing, duplicate or changed
completion evidence fails. Do not run two workers on the same dump. Retry only manifests left in `dumps/`;
completed manifests move to `completed-dumps/`. The launcher validates after all dumps complete; the manual equivalent is
`bash tokenization_scripts/validate_tokenization.sh "$CONFIG"` in the same
compatible environment. Strict validation checks all indexes/maps/source-row
coverage, hashes payloads, and seals the variant. Missing or malformed output
prevents release.

The launcher also applies the runtime's existing LC reserve workflow with fraction 0.10, seed
20260907, and `long-context-length-buckets-v3`, then verify and finalize its fresh
sibling `.lc-reserve-v4` root. Combine its kept pairs with the unchanged original
kept pairs under an explicitly inventoried new union identity. Record original,
recovered, reserved and kept counts separately. Preserve source maps and original
paths; final mixture supply comes from measured indexes, not an estimate.
