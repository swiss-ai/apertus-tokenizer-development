# Root-aligned tokenization

`preprocess_megatron.py` uses DPP's callback-based `TokenWriter`. It reads completed
exact DPP root and primary-text Data, preserves every row in root order, and writes
new aligned token Data plus its automatic Int64 `stored_tokens` Annotation. Dataset preparation
and physical population changes belong to the preparation producer; the sampler
owns selection and its run maps.

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

The input must already have complete root-aligned metadata and payloads beneath
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

The shown boundary IDs are examples: check the actual asset and encoding recipe.
Both boundary arguments are required; pass an empty string for no prefix/suffix.
The body uses `add_special_tokens=False`. DPP inserts configured boundaries once.
The primary text field comes from the input Data's `primary_content` semantics;
user IDs are unnecessary for root alignment. Empty text, empty encoded bodies,
wrong output refs, missing input, cardinality/alignment problems and invalid IDs
fail the job.

`--tasks` counts deterministic native partitions; `--workers` counts concurrent
processes, and `--threads` sets each process's Rayon pool. Defaults are one each.
Budget workers × threads within allocated CPUs. `--batch-size` defaults to 1000
rows and `--batch-bytes` to 33554432 UTF-8 bytes. A single oversize row runs alone
without truncation. `--max-sequences-per-shard` defaults to 100000 index entries.
The output writer uses uint16/code8 for vocabularies through 65536 IDs, otherwise
nonnegative int32/code4. Output sharding may differ from input Parquet sharding.

`run_producer` guards token Data and automatic lengths. Each native task writes
BIN/IDX and lengths, including explicit empty pairs for idle tasks. Finalization
checks native completion, exact pair coverage, token IDs/boundaries, lengths and
corruption before marking both collections complete. Each sequence ordinal
identifies the corresponding root row in the text, metadata and Annotations.

## Track a run in transparency

Tracking is optional. For a tracked run, use Python 3.11+ and the coordinated ADT
SDK with authenticated access to your chosen catalogue. Follow DPP's
`documentation/transparency.md` in the pinned checkout for installation, destination
setup, producer capture and offline schema/semantic preview. Keep credentials out
of the command and payload.

Create one complete change-set JSON file from your reviewed declarations. It must
declare the new aligned token Data and its automatic Annotation
`annotation/<token-slug>-token-lengths@<token-revision>`, plus any related records.
The token Data names the exact prepared text input; the Annotation describes the
token Data. Use actual catalogue identities and the pushed reproducible producer
commit. Preview this full payload before running; ADT's
`json_dumps(payload, indent=2)` from `adt.lossless` preserves exact decimal values
when writing JSON.

Add these arguments to the job command above:

```sh
--proposal-file "$CHANGE_SET_JSON" --proposal-repo "$CATALOGUE_OWNER_REPO"
```

`CATALOGUE_OWNER_REPO` is an explicit GitHub `owner/repository`. Add
`--proposal-issue N` to reuse an open issue with that exact whole record set;
omitting it lets DPP discover an unambiguous match. The Python API accepts the
same `rd.TrackedProposal` as `TokenizationJob(..., proposal=proposal)`.

DPP checks accepted identities, proposal validity and pending reuse before
mutating outputs. It submits the pending proposal before executing the native
job, so a failed run may leave a pending issue. Omit both proposal arguments for
offline operation: ADT is neither imported nor contacted. Check catalogue
acceptance, payload completion and eligibility separately.

## Retry and verify

Nonempty outputs fail by default. Add `--resume` only when inputs, verified asset,
boundaries, partitioning, resources and native logs are compatible, and no readers
or competing writers are active. Native completed tasks are reused; payloads are
qualified again. The operator remains responsible for resume compatibility.
Explicit resume can reopen and mutate a completed unmerged output. Tracked runs
protect accepted output references even with resume. Use fresh references and
logging directories for changed data, assets or configuration.

For offline verification and contributor checks, see [development](DEVELOPMENT.md).
