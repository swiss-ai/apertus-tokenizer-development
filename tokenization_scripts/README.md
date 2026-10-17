# Root-aligned tokenization

`preprocess_megatron.py` uses DPP's `TokenWriter` with a batch encoder function.
A **Data** is a named collection of content; a **root Data** fixes its row
population and order. The job reads completed root and primary-text Data,
preserves every row in root order, and writes aligned token Data plus its
automatic `stored_tokens` Annotation, a per-row Int64 token count. Dataset
preparation and physical population changes belong to the preparation producer;
the sampler owns selection and its run maps.

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
python -m tokenization_scripts.preprocess_megatron --help
```

These prerequisites run the job and source checks from paired local checkouts.
Use the producer's supported runtime for deployment. Run one coordinator inside
an allocated machine; scheduler submission remains the owning environment's
responsibility.

## Run a job

The input must already have complete root-aligned metadata and data files beneath
one physical `data_root`. Use the exact root and primary-text Data references
identifying the completed producer outputs. Choose a distinct output token Data
reference with its own revision. The command takes explicit CLI arguments.

The tokenizer JSON must be a local tracked file in a Git checkout with an origin
URL. Supply the full commit containing those exact bytes. The job compares local
bytes with that committed asset before output mutation, then workers load the
verified bytes offline. A mutable directory label is insufficient. The recorded
encoding context includes repository, commit and relative asset path, along with
disabled special-token insertion, truncation and padding. Repository URLs use
the configured origin without machine Git rewrites or embedded credentials.

From this repository root, with shell variables set to your reviewed identities:

```sh
python -m tokenization_scripts.preprocess_megatron \
  --data-root "$DATA_ROOT" \
  --root-ref "$ROOT_REF" --input-ref "$TEXT_REF" --output-ref "$TOKEN_REF" \
  --tokenizer-file "$TOKENIZER_JSON" --tokenizer-revision "$ASSET_COMMIT" \
  --prefix-tokens 1 --suffix-tokens 2 \
  --logging-dir "$TOKEN_LOGS" \
  --tasks 8 --workers 2 --threads 4
```

Prefix and suffix IDs are boundary tokens placed before and after each encoded
text. The shown IDs are examples: check the actual asset and encoding settings.
Both boundary arguments are required; pass an empty string for no prefix/suffix.
The body uses `add_special_tokens=False`. DPP inserts configured boundaries once.
The primary text field comes from the input Data's `primary_content` metadata field;
user IDs are unnecessary for root alignment. Empty text, empty encoded bodies,
wrong output refs, missing input, different row counts or order and invalid IDs
fail the job.

`--tasks` counts disjoint ranges of root rows; `--workers` counts concurrent
processes, and `--threads` sets each process's Rayon pool. Defaults are one each.
Budget workers × threads within allocated CPUs. `--batch-size` defaults to 1000
rows and `--batch-bytes` to 33554432 UTF-8 bytes. A single oversize row runs alone
without truncation. `--max-sequences-per-shard` defaults to 100000 index entries.
The output writer uses uint16/code8 for vocabularies through 65536 IDs, otherwise
nonnegative int32/code4. Output sharding may differ from input Parquet sharding.

`run_producer` guards token Data and automatic lengths. Each Datatrove task writes
BIN/IDX and lengths, including explicit empty pairs for idle tasks. Finalization
checks Datatrove completion, exact pair coverage, token IDs/boundaries, lengths and
corruption before marking both collections complete. Each zero-based sequence number
identifies the corresponding root row in the text, metadata and Annotations.

## Record the code used by a run

Add `--transparency` to the job command. With Python 3.11+ and the ADT SDK
installed, the job records its output identifiers and the tokenizer repository's
actual Git commit in an editable issue. The job checkout must be clean, committed
and pushed; capture works even when the command is launched from another directory.

The default catalogue and issue destination is
[`swiss-ai/apertus-data-transparency-draft`](https://github.com/swiss-ai/apertus-data-transparency-draft).
Each tracked call fetches its latest accepted catalogue automatically. See
DPP's `documentation/transparency.md` at the pinned checkout for SDK installation
and GitHub authentication.

Open the printed issue URL and complete Sources, inputs, notes and any other
required declaration fields there. The skeleton contains output identities and
run code; its declaration checks fail until it is complete. A retry reuses the
pending issue and preserves human edits. Another actual run commit adds a comment.
The issue is created before tokenization, so verify actual output completion and
catalogue acceptance separately.

Use `--transparency owner/repository` for another catalogue. The Python API takes
`TokenizationJob(..., transparency=True)` or the same repository string.
Omitting the flag runs offline with no ADT lookup.

## Retry and verify

Nonempty outputs fail by default. Add `--resume` only when inputs, verified asset,
boundaries, partitioning, resources and Datatrove logs are compatible, and no readers
or competing writers are active. Completed Datatrove tasks are reused; stored
sequences and lengths are validated again. The operator remains responsible for resume compatibility.
Explicit resume can reopen and mutate a completed unmerged output. Tracked runs
protect accepted output references even with resume. Use fresh references and
logging directories for changed data, assets or configuration.

For offline verification and contributor checks, see [development](DEVELOPMENT.md).
