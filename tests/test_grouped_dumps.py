import argparse
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from tokenization_scripts import prepare_dumps


class GroupedDumpsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.metadata = self.root / "metadata"
        self.output = self.root / "tokens"
        self.source.mkdir()
        self.tokenizer = self.root / "tokenizer.json"
        self.tokenizer.write_text("tokenizer", encoding="utf-8")

    def build_artifact(self):
        rows = []
        for category, slug, sizes in (
            ("programming", "python", [20, 10]),
            ("data", "json", [15]),
        ):
            for index, size in enumerate(sizes):
                relative = f"languages/{slug}/{index:04d}.parquet"
                path = self.source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x" * size)
                rows.append({"relative_path": relative, "language_slug": slug})
        manifest = self.source / "manifest.jsonl"
        manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (self.source / "languages.json").write_text(
            json.dumps(
                {
                    "languages": [
                        {"slug": "python", "category": "programming"},
                        {"slug": "json", "category": "data"},
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest

    def prepare(self, **overrides):
        values = {
            "dataset_folder": str(self.source),
            "manifest": str(self.source / "manifest.jsonl"),
            "manifest_path_key": "relative_path",
            "filter_in": None,
            "filter_out": None,
            "preprocessing_metadata_folder": str(self.metadata),
            "n_dumps": None,
            "group_fields": "category,language_slug",
            "group_metadata": str(self.source / "languages.json"),
            "group_metadata_root": "languages",
            "group_metadata_lookup_field": "language_slug",
            "group_metadata_id_field": "slug",
            "expected_groups": 2,
            "expected_group_heads": "programming,data",
            "max_dump_bytes": 150_000_000_000,
        }
        values.update(overrides)
        prepare_dumps.main(argparse.Namespace(**values))

    def run_entrypoint(self, config: Path) -> list[str]:
        binary_dir = self.root / "bin"
        binary_dir.mkdir(exist_ok=True)
        calls = self.root / "sbatch.calls"
        srun = binary_dir / "srun"
        srun.write_text(
            "#!/bin/bash\n"
            "while [[ \"$1\" == --* ]]; do shift; done\n"
            "exec \"$@\"\n",
            encoding="utf-8",
        )
        srun.chmod(0o755)
        sbatch = binary_dir / "sbatch"
        sbatch.write_text(
            "#!/bin/bash\n"
            "printf '%s\\n' \"$*\" >>\"$SBATCH_CALLS\"\n"
            "echo 101\n",
            encoding="utf-8",
        )
        sbatch.chmod(0o755)
        environment = dict(os.environ)
        environment["PATH"] = f"{binary_dir}:{environment['PATH']}"
        environment["SBATCH_CALLS"] = str(calls)
        script = (
            Path(__file__).parents[1]
            / "tokenization_scripts"
            / "tokenize_script.sh"
        )
        subprocess.run(
            ["bash", str(script), str(config)],
            capture_output=True,
            text=True,
            check=True,
            env=environment,
        )
        return calls.read_text(encoding="utf-8").splitlines()

    def test_manifest_groups_are_mirrored_into_nested_paths_files(self):
        self.build_artifact()
        self.prepare()
        python_paths = self.metadata / "dumps/programming/python/paths_file_0.txt"
        json_paths = self.metadata / "dumps/data/json/paths_file_0.txt"
        self.assertEqual(
            python_paths.read_text(encoding="utf-8").splitlines(),
            [
                "languages/python/0000.parquet",
                "languages/python/0001.parquet",
            ],
        )
        self.assertEqual(
            json_paths.read_text(encoding="utf-8").splitlines(),
            ["languages/json/0000.parquet"],
        )

    def test_size_limit_splits_one_group_without_flattening_it(self):
        self.build_artifact()
        self.prepare(max_dump_bytes=20)
        python_dumps = sorted(
            (self.metadata / "dumps/programming/python").glob("paths_file_*.txt")
        )
        self.assertEqual(len(python_dumps), 2)
        self.assertEqual(
            sorted(
                line
                for path in python_dumps
                for line in path.read_text(encoding="utf-8").splitlines()
            ),
            [
                "languages/python/0000.parquet",
                "languages/python/0001.parquet",
            ],
        )

    def test_standard_entrypoint_submits_every_group_through_tokenize_sh(self):
        self.build_artifact()
        config = self.root / "grouped.cfg"
        config.write_text(
            "\n".join(
                [
                    f"TOKENIZER={self.tokenizer}",
                    "TOKENIZER_NAME=test-tokenizer",
                    "DATASET_NAME=grouped-fixture",
                    "COLUMN_KEY=content",
                    "ID_COLUMN=content_id",
                    f"PATH_TO_RAW_DATASET={self.source}",
                    f"PATH_TO_OUTPUT_FOLDER={self.output}",
                    f"PATH_TO_PREPROCESSING_METADATA={self.metadata}",
                    "DATASET_OUTPUT_FOLDER_NAME=$PATH_TO_OUTPUT_FOLDER/$TOKENIZER_NAME",
                    f"DATASET_MANIFEST={self.source / 'manifest.jsonl'}",
                    f"REQUIRED_DATASET_MARKER={self.source / '_SUCCESS'}",
                    "MANIFEST_PATH_KEY=relative_path",
                    "DUMP_GROUP_FIELDS=category,language_slug",
                    f"DUMP_GROUP_METADATA={self.source / 'languages.json'}",
                    "DUMP_GROUP_METADATA_ROOT=languages",
                    "DUMP_GROUP_METADATA_LOOKUP_FIELD=language_slug",
                    "DUMP_GROUP_METADATA_ID_FIELD=slug",
                    "EXPECTED_GROUP_COUNT=2",
                    "EXPECTED_GROUP_HEADS=programming,data",
                    "MAX_DUMP_BYTES=150000000000",
                    "INCLUDE_BOOLEAN_COLUMN=apertus_include",
                    "INCLUDE_REASON_COLUMN=exclusion_reason",
                    "INCLUDED_REASON=",
                    "REHYDRATE_FLAG=False",
                    "EXTENSION=.parquet",
                    "NUMBER_OF_DATATROVE_TASKS=4",
                    "TOKENIZER_BATCH_SIZE=1000",
                    "ACCOUNT=infra01",
                    "NODES=1",
                    "PARTITION=normal",
                    "GPUS=0",
                    "CPUS_PER_TASK=4",
                    "NO_REQUEUE=--no-requeue",
                    "TIME=00:10:00",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_entrypoint(config)
        (self.source / "_SUCCESS").write_text("sealed\n", encoding="utf-8")
        submitted = self.run_entrypoint(config)
        self.assertEqual(len(submitted), 2)
        self.assertTrue(all("tokenize.sh" in call for call in submitted))
        self.assertTrue(all("stackv31_tokenize.sh" not in call for call in submitted))
        self.assertTrue(
            any("programming/python/dump-0" in call for call in submitted)
        )
        self.assertTrue(any("data/json/dump-0" in call for call in submitted))
        self.assertTrue(any(f"{config} " in call for call in submitted))

    def test_ungrouped_config_keeps_the_existing_dump_layout(self):
        (self.source / "one.parquet").write_bytes(b"one")
        (self.source / "two.parquet").write_bytes(b"two")
        config = self.root / "ordinary.cfg"
        config.write_text(
            "\n".join(
                [
                    f"TOKENIZER={self.tokenizer}",
                    "TOKENIZER_NAME=test-tokenizer",
                    "DATASET_NAME=ordinary",
                    "COLUMN_KEY=text",
                    f"PATH_TO_RAW_DATASET={self.source}",
                    f"PATH_TO_OUTPUT_FOLDER={self.output}",
                    f"PATH_TO_PREPROCESSING_METADATA={self.metadata}",
                    "DUMPS_NUMBER=1",
                    "REHYDRATE_FLAG=False",
                    "EXTENSION=.parquet",
                    "NUMBER_OF_DATATROVE_TASKS=1",
                    "ACCOUNT=infra01",
                    "NODES=1",
                    "PARTITION=normal",
                    "GPUS=4",
                    "CPUS_PER_TASK=4",
                    "NO_REQUEUE=--no-requeue",
                    "TIME=00:10:00",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        submitted = self.run_entrypoint(config)
        self.assertEqual(len(submitted), 1)
        self.assertIn(
            str(self.output / "test-tokenizer" / "ordinary" / "dump-0"),
            submitted[0],
        )
        self.assertIn("--gres=gpu:4", submitted[0])


if __name__ == "__main__":
    unittest.main()
