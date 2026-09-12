#!/usr/bin/env python3
"""Generate one ordinary tokenizer config per sealed Apertus SFT source.

The source partition manifest is produced by data-pipeline-pretrain's
`pipelines/apertus-sft-pretrain/partition_sources.py`. Keeping this generator
next to `tokenize_script.sh` avoids introducing a second tokenization path:
each config simply points that existing launcher at one source-pure Parquet
directory. The already-tokenized mixed release is backfilled separately without
running the tokenizer again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
from pathlib import Path

SCHEMA_VERSION = "apertus-sft-source-partition/v1"
SLUG_RE = re.compile(r"[a-z0-9][a-z0-9-]*-[0-9a-f]{12}\Z")
ASSIGNMENT_RE = re.compile(r"^([A-Z][A-Z0-9_]*)=")
REQUIRED_KEYS = {
    "DATASET_NAME",
    "PATH_TO_RAW_DATASET",
    "PATH_TO_PREPROCESSING_METADATA",
    "PATH_TO_OUTPUT_FOLDER",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_config(template: str, replacements: dict[str, str]) -> str:
    seen: set[str] = set()
    lines = []
    for line in template.splitlines():
        match = ASSIGNMENT_RE.match(line)
        key = match.group(1) if match else None
        if key in replacements:
            if key in seen:
                raise ValueError(f"duplicate template assignment: {key}")
            seen.add(key)
            lines.append(f"{key}={shlex.quote(replacements[key])}")
        else:
            lines.append(line)
    if seen != set(replacements):
        raise ValueError(
            f"template lacks assignments: {sorted(set(replacements) - seen)}"
        )
    return "\n".join(lines) + "\n"


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"existing generated config differs: {path}")
        return
    partial = path.with_name(path.name + ".partial")
    partial.unlink(missing_ok=True)
    with partial.open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    os.rename(partial, path)


def generate(
    manifest_path: Path,
    text_root: Path,
    token_root: Path,
    template_path: Path,
    config_dir: Path,
) -> dict:
    if not text_root.is_absolute() or not token_root.is_absolute():
        raise ValueError("text and token roots must be absolute")
    seal_path = text_root / "_SOURCE_PARTITION_SUCCESS.json"
    if manifest_path.read_bytes() != seal_path.read_bytes():
        raise ValueError("partition manifest and source-root seal differ")
    manifest = json.loads(seal_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("source partition schema differs")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("source partition has no source inventory")
    if sum(item["rows"] for item in sources) != manifest["rows"]:
        raise ValueError("source partition rows do not reconcile")
    slugs = [item["slug"] for item in sources]
    if len(slugs) != len(set(slugs)) or not all(
        SLUG_RE.fullmatch(slug) for slug in slugs
    ):
        raise ValueError("source partition slugs are unsafe or duplicated")

    template = template_path.read_text(encoding="utf-8")
    template_keys = {
        match.group(1)
        for line in template.splitlines()
        if (match := ASSIGNMENT_RE.match(line))
    }
    if not REQUIRED_KEYS <= template_keys:
        raise ValueError("tokenizer template lacks a required assignment")
    output = []
    for item in sources:
        slug = item["slug"]
        source_dir = text_root / "data" / slug
        if not source_dir.is_dir() or not list(source_dir.glob("part-*.parquet")):
            raise ValueError(f"source-pure Parquet directory missing: {source_dir}")
        if item["rows"] < 1:
            raise ValueError(f"empty source in partition seal: {slug}")
        # tokenize_script.sh writes to OUTPUT_FOLDER/TOKENIZER_NAME/DATASET_NAME
        # and stores per-run links, dumps, and CSVs under PREPROCESSING_METADATA.
        # Share only the output base; source identity remains a physical folder.
        metadata = token_root / "_preprocessing" / slug
        name = slug
        content = render_config(
            template,
            {
                "DATASET_NAME": name,
                "PATH_TO_RAW_DATASET": str(source_dir),
                "PATH_TO_PREPROCESSING_METADATA": str(metadata),
                "PATH_TO_OUTPUT_FOLDER": str(token_root),
            },
        )
        path = config_dir / f"{slug}.cfg"
        write_new(path, content.encode("utf-8"))
        output.append(
            {
                "source": item["source"],
                "slug": slug,
                "rows": item["rows"],
                "config": path.name,
                "config_sha256": sha256(path),
            }
        )
    result = {
        "schema_version": "apertus-sft-source-tokenizer-configs/v1",
        "partition_seal_sha256": sha256(seal_path),
        "template_sha256": sha256(template_path),
        "text_root": str(text_root),
        "token_root": str(token_root),
        "sources": output,
    }
    write_new(
        config_dir / "_CONFIGS_SUCCESS.json",
        (
            json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8"),
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--text-root", type=Path, required=True)
    parser.add_argument("--token-root", type=Path, required=True)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).resolve().parent
        / "configs_apertus_v2/Apertus-1.5-SFT-mix-pretrain-v1.cfg",
    )
    parser.add_argument("--config-dir", type=Path, required=True)
    args = parser.parse_args()
    result = generate(
        args.partition_manifest,
        args.text_root,
        args.token_root,
        args.template,
        args.config_dir,
    )
    print(
        json.dumps(
            {
                "configs": len(result["sources"]),
                "manifest": str(args.config_dir / "_CONFIGS_SUCCESS.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
