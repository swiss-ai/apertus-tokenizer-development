# Historical dataset recipes awaiting root migration

Every `.cfg` in this directory is a retained recipe from the retired dump runtime.
Its original bytes and producing history remain intact. None is a supported config
for the root-aligned job, and none establishes an actual root or token Data ref.

Use the [tokenization runbook](../README.md). Each dataset needs its preparation
producer to establish exact root/input identities and preserve its source-specific
selection, rehydration, lineage and approval scope before its tokenization command
can be ported. Keep frozen payloads/maps and the actual producing commit; the
runtime transition does not rewrite their history.
