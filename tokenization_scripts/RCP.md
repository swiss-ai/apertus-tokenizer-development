# Config-driven tokenizer launcher on RCP

The launch_rcp.py command is the fallback Run:ai scheduler for committed dataset
configs. It does not define dataset columns, grouping, tokenizer identity, input/output
roots, task counts, or context length. Those values are sourced from the pinned config
by the standard tokenize_script.sh and tokenize.sh entrypoints.

The strict launcher JSON pins the clean tokenizer checkout, dataset-config bytes,
processed dataset completion marker, digest-addressed image, staged Python runtime,
private work root, and Run:ai resources. The selected pool may host GPUs, but jobs
always request zero GPUs. Payload jobs set the standard scripts' RCP backend, keep
networked dataset/model clients offline, and mount only /mloscratch.

Run in this order:

    python tokenization_scripts/launch_rcp.py preflight \
      --config /mloscratch/control/TOKENIZER_RCP.json \
      --config-sha256 <sha256>

    python tokenization_scripts/launch_rcp.py render-prepare ...
    python tokenization_scripts/launch_rcp.py submit-prepare ...

After the prepared dump inventory exists, inspect and submit remaining dumps:

    python tokenization_scripts/launch_rcp.py render ...
    python tokenization_scripts/launch_rcp.py submit ...

Completed dump path files move to the configured completed-dumps tree. Re-rendering
therefore schedules only unfinished dumps. Each worker calls tokenize.sh with the
committed config; it never receives launcher-private dataset flags.

When no paths files remain, validate and seal:

    python tokenization_scripts/launch_rcp.py validate ...

Validation requires every bin/idx/map triple, checks index structure and binary size,
parses every token map, verifies map-to-index hashes and sequence counts, scans every
sequence length against MAX_SEQUENCE_TOKENS, hashes the token payload manifest, and
writes _SUCCESS.json last. Transfer to Clariden remains a separate operation.
