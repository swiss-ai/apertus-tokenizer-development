
## DCLM retained output

`configs_apertus_v2/dclm-edu-retained-output-v1.cfg` reads only
`dclm-edu_spp_annotated_robots_v2/data/data/output`. The producer's sibling
`data/data/removed/robots` contains rejected documents and must not enter this
release. The older `dclm-edu_spp_annotated.cfg` scans their common parent; retain it
only to reproduce the historical broad population. The new config uses a fresh
token output root and no expired reservation.

Before launch, pin and inventory every retained prepared file, reconcile the
excluded population with the producer's rejection records, and use a compatible
source-map-enabled tokenizer/pipeline revision. Verify complete source coordinates,
token/index/map consistency and source-aware encoding examples before reserving
and accepting the full replacement. This configuration alone does not establish
a completed replacement or repair historical PII redaction.
