# Root-aligned tokenization

`tokenize.py` composes DPP's root reader, the tokenizer's `EncodeTokens`
step and DPP's `RootWriter` in native Datatrove Slurm tasks.
A **Data** is a named collection of content. An **alignment root** is a Data that
defines a population of rows and their order. A tokenization job reads
completed text Data and writes one aligned Parquet row per input row. Each row
contains the complete token sequence as packed little-endian int32 bytes in
`tokens`, and its length in `stored_tokens`.

The encoding step adds the configured document boundaries exactly once. The
writer checks root coordinates, token cells and lengths before completion. DPP
uses these rows for mixture sampling and writes training BIN/IDX/MAP files to the
configured mixture output storage.

## Set up compatible code

Use Python 3.10 or newer and a local DPP checkout at the exact commit in
[`dpp-revision.txt`](dpp-revision.txt). DPP is private: access must be granted by its
owner. Keep private source and credentials outside this public repository.
Use this pin for the paired reader/writer interface.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r tests/requirements/runtime.txt
# Set DPP_CHECKOUT to your local DPP source checkout.
git -C "$DPP_CHECKOUT" checkout "$(cat tokenization_scripts/dpp-revision.txt)"
export PYTHONPATH="$DPP_CHECKOUT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m tokenization_scripts.tokenize --help
```

Keep this checkout, the pinned DPP checkout and the Python environment on paths
shared by all worker nodes. `WORKER_ENV` must activate that environment and make
both checkouts importable, for example:

```sh
export TOKENIZER_CHECKOUT="$PWD"
export WORKER_ENV=". /shared/venvs/tokenizer/bin/activate; export PYTHONPATH=$TOKENIZER_CHECKOUT:$DPP_CHECKOUT/src"
```

Submit from the shared checkout. The command schedules native Slurm tasks and a
one-task dependent publication job; it prints their IDs and returns. Wait for
publication to succeed before consuming the token Data.

## Run a job

The **data store** (`data_store`) is the physical directory holding durable Data
and Annotations, grouped by alignment-root identity. The input must already have
complete metadata and data files there. The root Data occupies
`<data_store>/<root-slug>@<revision>/data/root/`. Use the exact alignment-root
reference identifying the completed producer output. The input defaults to that
root Data; pass `--input-ref` to encode another aligned text Data. Choose a
distinct output token Data reference with its own revision.
Use identifiers such as `prepared-text@v1` in Python and CLI arguments.
Construct `TokenizationJob` with keyword arguments. Omit `input_ref` to read the
alignment root, or select another aligned text Data.

The tokenizer JSON must be a local tracked file in a Git checkout with an origin
URL. Supply the full commit containing those exact bytes. The job compares local
bytes with that committed asset before output mutation, then workers load the
verified bytes offline. A mutable directory label is insufficient. The recorded
encoding context includes repository, commit and relative asset path, along with
disabled special-token insertion, truncation and padding. Repository URLs use
the configured origin without machine Git rewrites or embedded credentials.

For Git LFS assets, check out the configured asset revision and materialize the
JSON first, for example with `git lfs pull --include=preliminary_mul_200k/tokenizer.json`
in its checkout. The job verifies its byte size and SHA-256 against the committed
LFS pointer at that exact revision.

From this repository root, with shell variables set to your reviewed identities:

```sh
python -m tokenization_scripts.tokenize \
  --data-store "$DATA_STORE" \
  --root-ref "$ROOT_REF" --output-ref "$TOKEN_REF" \
  --tokenizer-file "$TOKENIZER_JSON" --tokenizer-revision "$ASSET_COMMIT" \
  --prefix-tokens 1 --suffix-tokens 2 \
  --logging-dir "$TOKEN_LOGS" \
  --tasks 8 --workers 2 --threads 4 \
  --partition "$PARTITION" --account "$ACCOUNT" \
  --env-command "$WORKER_ENV"
```

To encode another aligned text collection, add `--input-ref "$TEXT_REF"` with its
exact Data reference.

Prefix and suffix IDs are boundary tokens placed before and after each encoded
text. The shown IDs are examples: check the actual asset and encoding settings.
Both boundary arguments are required; pass an empty string for no prefix/suffix.
The body uses `add_special_tokens=False`; `EncodeTokens` calls DPP's framing
helper before writing. `--text-column` selects the input string field and defaults
to `text`. Empty text retains its row and receives the configured boundaries.
The stored sequence must contain at least one token. Missing input, invalid IDs
and changed row counts or order fail the job.

## Choose resources

`--tasks` counts disjoint ranges of root rows. `--workers` caps concurrent Slurm
tasks. Each task runs one tokenizer process with `--threads` CPUs and the same
Rayon thread count. Defaults are one each; `--mem-per-cpu-gb` defaults to 2 GiB.
Slurm applies this to every allocated CPU; exclusive-node partitions can allocate
more CPUs than `--threads`. Choose a value within the partition's memory per CPU.
Publication uses one CPU and the same memory-per-CPU setting.
`--time` sets each task's wall-time limit and defaults to two hours.
`--batch-size` defaults to 1000
rows per encoding call and write batch; `--batch-bytes` bounds encoding input at
33554432 UTF-8 bytes. A single oversize row runs alone without truncation.
The shared writer also flushes at its byte threshold. Row and batch limits do
not establish a hard process-memory limit; size the allocation for the largest
individual document and its token sequence.

Workers lazily load verified tokenizer bytes and disable asset truncation and
padding. Encoding batches preserve documents and their root coordinates. The
Arrow `tokens` field records the tokenizer's repository, exact asset commit,
relative path, vocabulary and boundary IDs. Both columns are ordinary Data
fields, and `stored_tokens` includes the boundaries.

## Read the output

The completed token Data contains Parquet files and collection metadata.
Finalization checks task completion, root row coverage and decoded storage
integrity. The dependent publication job finalizes the collection after all tasks succeed.

Use the length column directly for counts and sampling preparation:

```python
from data_pipeline_pretrain.pipeline import root_data as rd

collection = rd.ParquetCollection(data_store, root_ref, token_ref)
for batch in collection.batches(["stored_tokens"]):
    print(batch.column("stored_tokens"))
```

Read lengths alongside root text by selecting both collections:

```python
reader = rd.RootParquetReader(
    data_store, root_ref, {None: ["text"], token_ref: ["stored_tokens"]}
)
for document in reader.run():
    print(document.text, document.metadata["stored_tokens"])
```

A zero-based token row identifies the same root row in the text, metadata and
Annotations.
`collection.sequences_at(ordinals, spec=expected_spec)` reads selected complete
sequences and requires the expected `rd.TokenSpec` from the job configuration.

## Retry and verify

Occupied outputs fail by default. Add `--resume` for unfinished outputs only when
inputs, verified asset, boundaries, partitioning and Datatrove logs are compatible,
and no readers or competing writers are active. Completed Datatrove tasks are
reused. Finalization verifies storage before publication, and completed outputs
are immutable. The operator owns the compatibility of an explicitly resumed run.

For a small single-machine smoke test, add `--local` to the command. The local
mode uses spawned worker processes and returns after validation. Its Python
convenience function is `run_job(job)`, returning `CollectionMetadata`.

Python callers can pass an executor factory to `prepare_job(job, factory)` to
construct a native executor from `(pipeline, tasks, logging_dir)`. The function
reserves unfinished output before returning `(writer, executor)`. Local executors
finish synchronously. A Slurm executor needs dependent publication, as shown in
[`slurm.py`](slurm.py); calling `finalize` immediately after submission is premature.
Use fresh Data references and logs for changed inputs, assets or configuration.
