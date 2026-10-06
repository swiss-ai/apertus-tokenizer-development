import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from data_pipeline_pretrain.pipeline.tokens.megatron_index import read_megatron_index
from data_pipeline_pretrain.pipeline.tokens.megatron_tokenizer import (
    MegatronSequenceEncoder,
    MegatronTokenizedFile,
)
from data_pipeline_pretrain.pipeline.tokens.token_map import read_token_map
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from tokenization_scripts import repair_finetranslations_tokens as repair

COMMIT = "1" * 40


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fixture(root, repeats=False, changed=True):
    root = root.resolve()
    old, new, tokens, output = (
        root / name for name in ("old", "new", "tokens", "output")
    )
    old.mkdir()
    new.mkdir()
    tokenizer_path = root / "tokenizer.json"
    tokenizer = Tokenizer(
        WordLevel(
            {
                "<UNK>": 0,
                "<BOS>": 1,
                "<EOS>": 2,
                "alpha": 3,
                "beta": 4,
                "gamma": 5,
                "delta": 6,
            },
            unk_token="<UNK>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens(["<BOS>", "<EOS>"])
    tokenizer.save(str(tokenizer_path))
    encoder = MegatronSequenceEncoder(str(tokenizer_path))
    reports = []
    for name in ("one.parquet", "two.parquet"):
        original = ["alpha beta", "beta", "delta"]
        corrected = (
            ["alpha gamma delta", "beta", "delta"]
            if changed and name == "one.parquet"
            else original
        )
        pq.write_table(
            pa.table({"id": ["a", "b", "c"], "text": original}),
            old / name,
            row_group_size=1,
        )
        pq.write_table(
            pa.table({"id": ["a", "b", "c"], "text": corrected}),
            new / name,
            row_group_size=2,
        )
        proofs = (
            [
                {
                    "prepared_row": 0,
                    "id": "a",
                    "old_text_sha256": text_sha(original[0]),
                    "new_text_sha256": text_sha(corrected[0]),
                    "changed": True,
                    "full_historical_replay_exact": True,
                }
            ]
            if corrected != original
            else []
        )
        reports.append(
            {
                "path": name,
                "rows": 3,
                "changed_rows": len(proofs),
                "changed": proofs,
                "input_sha256": repair.sha256_file(old / name),
                "output_sha256": repair.sha256_file(new / name),
            }
        )
        writer = MegatronTokenizedFile(
            str(tokens / name.replace(".parquet", "")),
            "00000_tokens",
            token_size=4,
            token_map_manifest={
                "tokenizer": {
                    "sha256": repair.sha256_file(tokenizer_path),
                    "post_processor": {
                        "template": "<BOS> $A <EOS>",
                        "bos_token_id": 1,
                        "eos_token_id": 2,
                    },
                }
            },
        )
        sequence_rows = [0, 0, 1, 2] if repeats else [0, 1, 2]
        for row in sequence_rows:
            writer.write(
                encoder.encode(original[row]),
                {
                    "row": row,
                    "file": {"path": name, **repair.shape(old / name)},
                    "source": {"raw_dataset_root": str(old)},
                },
            )
        writer.close()
    report = {
        "schema": "finetranslations-source-pii-repair/v1",
        "producer_commit": COMMIT,
        "dataset_id": "prepared-v1",
        "source_root": str(old),
        "files": [item["path"] for item in reports],
        "reports": reports,
        "rows": 6,
        "changed_rows": sum(item["changed_rows"] for item in reports),
    }
    repair.write_json(new / "PROCESSING_REPORT.json", report)
    reseal(new)
    return {
        "source_root": tokens,
        "prepared_root": new,
        "tokenizer": tokenizer_path,
        "output_root": output,
        "dataset_id": "tokens-v1",
        "commit": COMMIT,
        "pipeline_commit": "2" * 40,
        "workers": 1,
    }


def reseal(new):
    report = json.loads((new / "PROCESSING_REPORT.json").read_text())
    repair.write_json(
        new / "_SUCCESS.json",
        {
            "complete": True,
            "producer_commit": COMMIT,
            "dataset_id": "prepared-v1",
            "processing_report_sha256": repair.sha256_file(
                new / "PROCESSING_REPORT.json"
            ),
            "rows": report["rows"],
            "changed_rows": report["changed_rows"],
        },
    )


def assert_unsealed(args):
    assert not (args["output_root"] / "_SUCCESS.json").exists()


def test_complete_root_preserves_all_bytes_except_proven_row(tmp_path):
    args = fixture(tmp_path)
    result = repair.repair_root(**args)
    assert result["complete"] and result["sequences"] == 6
    for name in ("one", "two"):
        original = args["source_root"] / name / "00000_tokens"
        corrected = args["output_root"] / name / "00000_tokens"
        before = read_token_map(original.with_suffix(".map").read_bytes())
        after = read_token_map(corrected.with_suffix(".map").read_bytes())
        assert before["records"] == after["records"]
        assert after["manifest"]["raw_dataset_root"] == str(args["prepared_root"])
        assert after["manifest"]["files"][0]["num_row_groups"] == 2
        if name == "two":
            assert (
                original.with_suffix(".bin").read_bytes()
                == corrected.with_suffix(".bin").read_bytes()
            )
            assert (
                original.with_suffix(".bin").stat().st_ino
                == corrected.with_suffix(".bin").stat().st_ino
            )
        else:
            index = read_megatron_index(corrected.with_suffix(".idx"))
            binary = np.frombuffer(
                corrected.with_suffix(".bin").read_bytes(), dtype="<u4"
            )
            expected = MegatronSequenceEncoder(str(args["tokenizer"])).encode(
                "alpha gamma delta"
            )
            assert binary[: index.lengths[0]].tolist() == expected
            assert (
                binary[index.lengths[0] :].tobytes()
                == original.with_suffix(".bin").read_bytes()[16:]
            )


def test_every_repeat_is_reencoded_without_population_change(tmp_path):
    args = fixture(tmp_path, repeats=True)
    repair.repair_root(**args)
    report = json.loads(
        (args["output_root"] / "TOKEN_REPAIR_MANIFEST.json").read_text()
    )
    assert report["sequences"] == 8 and report["changed_sequences"] == 2
    assert report["changed_source_rows"] == 1


def test_no_changes_still_delivers_complete_root(tmp_path):
    args = fixture(tmp_path, changed=False)
    assert repair.repair_root(**args)["sequences"] == 6
    assert len(repair.discover(args["output_root"])) == 2


def test_wrong_original_tokens_fail_exact_reencoding(tmp_path):
    args = fixture(tmp_path)
    path = args["source_root"] / "one/00000_tokens.bin"
    data = bytearray(path.read_bytes())
    data[4] = 9
    path.write_bytes(data)
    with pytest.raises(ValueError, match="reencode original"):
        repair.repair_root(**args)
    assert_unsealed(args)


def test_prepared_same_size_corruption_fails_before_output(tmp_path):
    args = fixture(tmp_path)
    path = args["prepared_root"] / "one.parquet"
    data = bytearray(path.read_bytes())
    data[20] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="sealed row-repair"):
        repair.repair_root(**args)
    assert not args["output_root"].exists()


def test_existing_user_files_are_preserved(tmp_path):
    args = fixture(tmp_path)
    args["output_root"].mkdir()
    sentinel = args["output_root"] / "mine"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        repair.repair_root(**args)
    assert sentinel.read_text() == "preserve"


def test_missing_original_triple_is_not_silently_skipped(tmp_path):
    args = fixture(tmp_path)
    (args["source_root"] / "two/00000_tokens.map").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        repair.repair_root(**args)
    assert not args["output_root"].exists()


def test_tokenizer_identity_mismatch_refuses_seal(tmp_path):
    args = fixture(tmp_path)
    args["tokenizer"].write_text(args["tokenizer"].read_text() + "\n")
    with pytest.raises(ValueError, match="provenance differs"):
        repair.repair_root(**args)
    assert_unsealed(args)


def test_original_mutation_during_copy_refuses_seal(tmp_path):
    args = fixture(tmp_path)
    original_copy = repair._copy_sequences
    touched = False

    def corrupt(source, destination, index, ordinals):
        nonlocal touched
        result = original_copy(source, destination, index, ordinals)
        if not touched:
            path = Path(source.name)
            data = bytearray(path.read_bytes())
            data[-4] ^= 1
            path.write_bytes(data)
            touched = True
        return result

    with (
        patch.object(repair, "_copy_sequences", corrupt),
        pytest.raises(ValueError, match="changed|bytes"),
    ):
        repair.repair_root(**args)
    assert_unsealed(args)


def test_missing_changed_source_coordinate_refuses_seal(tmp_path):
    args = fixture(tmp_path)
    report_path = args["prepared_root"] / "PROCESSING_REPORT.json"
    report = json.loads(report_path.read_text())
    report["reports"][0]["changed"][0]["prepared_row"] = 2
    repair.write_json(report_path, report)
    reseal(args["prepared_root"])
    with pytest.raises(ValueError, match="source row differs"):
        repair.repair_root(**args)
    assert_unsealed(args)


def test_output_alias_inside_old_prepared_is_rejected(tmp_path):
    args = fixture(tmp_path)
    args["output_root"] = tmp_path / "old/tokens"
    with pytest.raises(ValueError, match="historical prepared"):
        repair.repair_root(**args)
    assert not args["output_root"].exists()


def test_changed_row_with_control_token_collision_is_rejected(tmp_path):
    args = fixture(tmp_path)
    path = args["prepared_root"] / "one.parquet"
    pq.write_table(
        pa.table(
            {"id": ["a", "b", "c"], "text": ["alpha <BOS> delta", "beta", "delta"]}
        ),
        path,
    )
    report_path = args["prepared_root"] / "PROCESSING_REPORT.json"
    report = json.loads(report_path.read_text())
    report["reports"][0]["output_sha256"] = repair.sha256_file(path)
    report["reports"][0]["changed"][0]["new_text_sha256"] = text_sha(
        "alpha <BOS> delta"
    )
    repair.write_json(report_path, report)
    reseal(args["prepared_root"])
    with pytest.raises(ValueError, match="collides"):
        repair.repair_root(**args)
    assert_unsealed(args)


def test_changed_row_missing_from_maps_fails_complete_coverage(tmp_path):
    from array import array

    from data_pipeline_pretrain.pipeline.tokens.token_map import pack_token_map

    args = fixture(tmp_path)
    path = args["source_root"] / "one/00000_tokens.map"
    token_map = read_token_map(path.read_bytes())
    files = token_map["manifest"]["files"]
    files[0]["emitted_rows"] = 2
    path.write_bytes(
        pack_token_map(
            files,
            array("Q", [1, 1, 2]),
            token_map["manifest"]["token_count"],
            token_map["manifest"],
        )
    )
    with pytest.raises(ValueError, match="not every prepared changed row"):
        repair.repair_root(**args)
    assert_unsealed(args)


def test_real_cli_forwards_all_paths_and_commit(tmp_path):
    args = fixture(tmp_path)
    command = [
        sys.executable,
        "-m",
        "tokenization_scripts.repair_finetranslations_tokens",
    ]
    for key, value in args.items():
        command.extend(
            [
                "--"
                + ("producer-commit" if key == "commit" else key.replace("_", "-")),
                str(value),
            ]
        )
    subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=os.environ.copy(),
        check=True,
        capture_output=True,
    )
    seal = json.loads((args["output_root"] / "_SUCCESS.json").read_text())
    assert seal["producer_commit"] == COMMIT and seal["complete"]
