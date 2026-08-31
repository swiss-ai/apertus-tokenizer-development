import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZATION_SCRIPTS = REPO_ROOT / "tokenization_scripts"


class TokenizationScriptTest(unittest.TestCase):
    def capture_tokenize_arguments(self, tasks, file_count=1, extra_config=()):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata"
            (metadata / "dumps").mkdir(parents=True)
            (metadata / "dumps" / "paths_file_0.txt").write_text(
                "".join(f"part-{index}.parquet\n" for index in range(file_count)),
                encoding="utf-8",
            )
            output = root / "output"
            output.mkdir()
            config = root / "config.cfg"
            config.write_text(
                "\n".join(
                    [
                        "TOKENIZER=tokenizer.json",
                        "TOKENIZER_NAME=test-tokenizer",
                        "DATASET_NAME=test-data",
                        "COLUMN_KEY=text",
                        "PATH_TO_RAW_DATASET=/unused",
                        f"PATH_TO_PREPROCESSING_METADATA={metadata}",
                        f"PATH_TO_OUTPUT_FOLDER={output}",
                        "DUMPS_NUMBER=1",
                        "REHYDRATE_FLAG=False",
                        "EXTENSION=.parquet",
                        f"NUMBER_OF_DATATROVE_TASKS={tasks}",
                        "ACCOUNT=test",
                        "NODES=1",
                        "GPUS=0",
                        "CPUS_PER_TASK=288",
                        'NO_REQUEUE="--no-requeue"',
                        "TIME=00:10:00",
                        "PARTITION=test",
                    ]
                    + list(extra_config)
                )
                + "\n",
                encoding="utf-8",
            )
            fake_bin = root / "bin"
            fake_bin.mkdir()
            capture = root / "sbatch-args.txt"
            fake_sbatch = fake_bin / "sbatch"
            fake_sbatch.write_text(
                '#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE"\n',
                encoding="utf-8",
            )
            fake_sbatch.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
            environment["CAPTURE"] = str(capture)

            subprocess.run(
                [
                    "bash",
                    str(TOKENIZATION_SCRIPTS / "tokenize_script.sh"),
                    str(config),
                    "--dont_compute_dumps",
                ],
                cwd=TOKENIZATION_SCRIPTS,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            arguments = capture.read_text(encoding="utf-8").splitlines()
            script_index = next(
                index
                for index, argument in enumerate(arguments)
                if argument.endswith("tokenize.sh")
            )
            return arguments[script_index + 1 :]

    def test_measured_whole_node_defaults_reach_worker_script(self):
        arguments = self.capture_tokenize_arguments(tasks=32, file_count=32)
        self.assertEqual(arguments[6], "32")
        self.assertEqual(arguments[-4:], ["10000", "33554432", "32", "9"])

    def test_single_file_dump_keeps_a_wide_thread_pool(self):
        arguments = self.capture_tokenize_arguments(tasks=32, file_count=1)
        self.assertEqual(arguments[6], "1")
        self.assertEqual(arguments[-4:], ["10000", "33554432", "1", "144"])

    def test_worker_default_tracks_small_dump_file_count(self):
        arguments = self.capture_tokenize_arguments(tasks=32, file_count=4)
        self.assertEqual(arguments[6], "4")
        self.assertEqual(arguments[-4:], ["10000", "33554432", "4", "72"])

    def test_explicit_thread_override_is_preserved(self):
        arguments = self.capture_tokenize_arguments(
            tasks=32, file_count=1, extra_config=["TOKENIZER_THREADS=288"]
        )
        self.assertEqual(arguments[-4:], ["10000", "33554432", "1", "288"])

    def test_explicit_runtime_controls_are_forwarded(self):
        arguments = self.capture_tokenize_arguments(
            tasks=32,
            file_count=32,
            extra_config=[
                "TOKENIZER_BATCH_SIZE=500",
                "TOKENIZER_BATCH_BYTES=16777216",
                "TOKENIZER_WORKERS=2",
                "TOKENIZER_THREADS=64",
            ],
        )
        self.assertEqual(arguments[-4:], ["500", "16777216", "2", "64"])

    def test_explicit_worker_override_is_not_capped(self):
        arguments = self.capture_tokenize_arguments(
            tasks=32, file_count=32, extra_config=["TOKENIZER_WORKERS=64"]
        )
        self.assertEqual(arguments[-4:], ["10000", "33554432", "64", "9"])

    def test_negative_one_worker_override_keeps_auto_semantics(self):
        arguments = self.capture_tokenize_arguments(
            tasks=32, file_count=32, extra_config=["TOKENIZER_WORKERS=-1"]
        )
        self.assertEqual(arguments[-4:], ["10000", "33554432", "-1", "9"])

    def test_checked_in_configs_enable_measured_parallelism(self):
        configs = TOKENIZATION_SCRIPTS / "configs_apertus_v2"
        for config in configs.glob("*.cfg"):
            with self.subTest(config=config.name):
                self.assertIn(
                    "NUMBER_OF_DATATROVE_TASKS=32",
                    config.read_text(encoding="utf-8").splitlines(),
                )


if __name__ == "__main__":
    unittest.main()
