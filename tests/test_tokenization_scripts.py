import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZATION_SCRIPTS = REPO_ROOT / "tokenization_scripts"


class TokenizationScriptTest(unittest.TestCase):
    def test_worker_reads_the_fourteenth_argument(self):
        worker = (TOKENIZATION_SCRIPTS / "tokenize.sh").read_text(encoding="utf-8")
        self.assertIn("TOKENIZER_THREADS=${14}", worker)

    def capture_sbatch_arguments(self, tasks, extra_config=()):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata"
            (metadata / "dumps").mkdir(parents=True)
            (metadata / "dumps" / "paths_file_0.txt").write_text(
                "part.parquet\n", encoding="utf-8"
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
            return arguments

    def test_safe_runtime_defaults_reach_worker_script(self):
        arguments = self.capture_sbatch_arguments(tasks=4)
        self.assertEqual(arguments[-4:], ["10000", "33554432", "4", "72"])

    def test_single_worker_thread_pool_is_capped(self):
        arguments = self.capture_sbatch_arguments(tasks=1)
        self.assertEqual(arguments[-4:], ["10000", "33554432", "1", "144"])

    def test_explicit_thread_override_is_preserved(self):
        arguments = self.capture_sbatch_arguments(
            tasks=1, extra_config=["TOKENIZER_THREADS=288"]
        )
        self.assertEqual(arguments[-4:], ["10000", "33554432", "1", "288"])

    def test_explicit_runtime_controls_are_forwarded(self):
        arguments = self.capture_sbatch_arguments(
            tasks=4,
            extra_config=[
                "TOKENIZER_BATCH_SIZE=500",
                "TOKENIZER_BATCH_BYTES=16777216",
                "TOKENIZER_WORKERS=2",
                "TOKENIZER_THREADS=64",
            ],
        )
        self.assertEqual(arguments[-4:], ["500", "16777216", "2", "64"])


if __name__ == "__main__":
    unittest.main()
