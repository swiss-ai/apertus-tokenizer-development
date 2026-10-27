import argparse
import hashlib
import json
import struct
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from tokenization_scripts import preprocess_megatron, validate_megatron

CATEGORIES = ("programming", "markup", "data", "prose")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_tokenizer(path: Path) -> None:
    tokenizer = Tokenizer(
        WordLevel(
            vocab={
                "<UNK>": 0,
                "<BOS>": 1,
                "<EOS>": 2,
                "alpha": 3,
                "beta": 4,
            },
            unk_token="<UNK>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path))


def _build_fixture(root: Path) -> argparse.Namespace:
    prepared = root / "prepared"
    output = root / "tokens"
    tokenizer = root / "tokenizer.json"
    config = root / "dataset.cfg"
    _build_tokenizer(tokenizer)
    config.write_text("MAX_SEQUENCE_TOKENS=4096\n", encoding="utf-8")
    manifest_rows = []
    for category in CATEGORIES:
        relative = f"examples/{category}/part.parquet"
        source = prepared / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "id": [f"{category}-one"],
                    "text": ["alpha beta"],
                    "sequence_tokens": [4],
                }
            ),
            source,
        )
        manifest_rows.append(
            {
                "relative_path": relative,
                "kind": "examples",
                "category": category,
                "rows": 1,
                "file_bytes": source.stat().st_size,
                "sha256": _sha256(source),
            }
        )
        paths = root / f"{category}.txt"
        paths.write_text(relative + "\n", encoding="utf-8")
        preprocess_megatron.main(
            argparse.Namespace(
                tokenizer_name_or_path=str(tokenizer),
                eos_token=None,
                output_folder=str(output / category / "dump-0"),
                logging_dir=str(root / "logs" / category),
                n_tasks=1,
                n_workers=1,
                dataset=str(prepared),
                paths_file=str(paths),
                column="text",
                id_column="id",
                rehydrate="False",
                extension=".parquet",
                include_boolean_column="",
                tokenizer_batch_size=2,
                max_sequence_tokens=4096,
            )
        )
    manifest = prepared / "examples_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    marker = prepared / "_SUCCESS"
    marker.write_text(
        json.dumps(
            {
                "complete": True,
                "smoke": False,
                "pins": {"examples_manifest_sha256": _sha256(manifest)},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return argparse.Namespace(
        dataset=str(prepared),
        manifest=str(manifest),
        dataset_marker=str(marker),
        output_folder=str(output),
        tokenizer=str(tokenizer),
        config=str(config),
        implementation_commit="a" * 40,
        dataset_name="stack-fixture",
        tokenizer_name="fixture-tokenizer",
        text_column="text",
        id_column="id",
        expected_categories=",".join(CATEGORIES),
        max_sequence_tokens=4096,
        workers=1,
    )


def test_grouped_tokenization_is_validated_and_sealed_marker_last():
    with tempfile.TemporaryDirectory() as temporary:
        args = _build_fixture(Path(temporary))
        args.workers = 2
        result = validate_megatron.validate_and_seal(args)
        output = Path(args.output_folder)
        assert result["complete"] is True
        assert result["totals"]["pairs"] == 4
        assert result["totals"]["sequences"] == 4
        assert result["totals"]["max_sequence_tokens"] == 4
        assert (output / "_SUCCESS.json").read_bytes() == (
            output / "TOKENIZATION_RUN.json"
        ).read_bytes()
        categories = json.loads(
            (output / "CATEGORY_COUNTS.json").read_text(encoding="utf-8")
        )
        assert set(categories["categories"]) == set(CATEGORIES)
        assert all(row["sequences"] == 1 for row in categories["categories"].values())

        # If publication of the final marker is interrupted, byte-identical
        # intermediate manifests are reusable and the marker can be restored.
        success = output / "_SUCCESS.json"
        expected_success = success.read_bytes()
        success.unlink()
        assert validate_megatron.validate_and_seal(args) == result
        assert success.read_bytes() == expected_success


def test_sequence_above_configured_limit_fails_without_sealing():
    with tempfile.TemporaryDirectory() as temporary:
        args = _build_fixture(Path(temporary))
        args.max_sequence_tokens = 3
        with pytest.raises(ValueError, match="outside 1..3"):
            validate_megatron.validate_and_seal(args)
        assert not (Path(args.output_folder) / "_SUCCESS.json").exists()


def test_malformed_index_pointer_fails_without_sealing():
    with tempfile.TemporaryDirectory() as temporary:
        args = _build_fixture(Path(temporary))
        index = next(Path(args.output_folder).rglob("*.idx"))
        sequence_count = 1
        pointers_offset = 34 + 4 * sequence_count
        with index.open("r+b") as stream:
            stream.seek(pointers_offset)
            stream.write(struct.pack("<q", 7))
        with pytest.raises(ValueError, match="pointers are not cumulative"):
            validate_megatron.validate_and_seal(args)
        assert not (Path(args.output_folder) / "_SUCCESS.json").exists()


def test_missing_map_fails_before_validation():
    with tempfile.TemporaryDirectory() as temporary:
        args = _build_fixture(Path(temporary))
        next(Path(args.output_folder).rglob("*.map")).unlink()
        with pytest.raises(ValueError, match="incomplete token pair"):
            validate_megatron.validate_and_seal(args)
        assert not (Path(args.output_folder) / "_SUCCESS.json").exists()


def test_same_size_prepared_payload_change_fails_without_sealing():
    with tempfile.TemporaryDirectory() as temporary:
        args = _build_fixture(Path(temporary))
        source = Path(args.dataset) / "examples" / "programming" / "part.parquet"
        original_size = source.stat().st_size
        with source.open("r+b") as stream:
            stream.seek(4)
            value = stream.read(1)
            assert value
            stream.seek(4)
            stream.write(bytes([value[0] ^ 1]))
        assert source.stat().st_size == original_size
        with pytest.raises(ValueError, match="prepared Parquet digest changed"):
            validate_megatron.validate_and_seal(args)
        assert not (Path(args.output_folder) / "_SUCCESS.json").exists()


def test_lightweight_validation_checks_structure_without_payload_rehashing():
    with tempfile.TemporaryDirectory() as temporary:
        args = _build_fixture(Path(temporary))
        args.validation_mode = validate_megatron.LIGHTWEIGHT_VALIDATION
        args.validator_commit = "b" * 40
        source = Path(args.dataset) / "examples" / "programming" / "part.parquet"
        with source.open("r+b") as stream:
            stream.seek(4)
            value = stream.read(1)
            assert value
            stream.seek(4)
            stream.write(bytes([value[0] ^ 1]))

        result = validate_megatron.validate_and_seal(args)

        assert result["constraints"]["validation_mode"] == (
            validate_megatron.LIGHTWEIGHT_VALIDATION
        )
        assert result["pins"]["validator_commit"] == "b" * 40
        manifest_rows = [
            json.loads(line)
            for line in (Path(args.output_folder) / "TOKENIZATION_MANIFEST.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert all(row["bin_sha256"] is None for row in manifest_rows)
        assert all(row["idx_sha256"] is None for row in manifest_rows)
        assert all(
            validate_megatron.SHA256_RE.fullmatch(row["writer_idx_sha256"])
            for row in manifest_rows
        )


def test_lightweight_validation_detects_truncated_token_payload():
    with tempfile.TemporaryDirectory() as temporary:
        args = _build_fixture(Path(temporary))
        args.validation_mode = validate_megatron.LIGHTWEIGHT_VALIDATION
        token_bin = next(Path(args.output_folder).rglob("*.bin"))
        token_bin.write_bytes(token_bin.read_bytes()[:-1])
        with pytest.raises(ValueError, match="binary size disagrees"):
            validate_megatron.validate_and_seal(args)
        assert not (Path(args.output_folder) / "_SUCCESS.json").exists()
