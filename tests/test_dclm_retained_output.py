import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "tokenization_scripts"


def input_root(config):
    return Path(
        subprocess.check_output(
            [
                "bash",
                "-c",
                'source "$1"; printf "%s" "$PATH_TO_RAW_DATASET"',
                "bash",
                str(config),
            ],
            text=True,
        )
    )


class DclmRetainedOutputTest(unittest.TestCase):
    def test_dump_preparation_excludes_sibling_rejects(self):
        configs = SCRIPTS / "configs_apertus_v2"
        historical = input_root(configs / "dclm-edu_spp_annotated.cfg")
        retained = input_root(configs / "dclm-edu-retained-output-v1.cfg")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            producer = root / "producer"
            accepted = producer / "data/data/output/part.parquet"
            rejected = producer / "data/data/removed/robots/part.parquet"
            for path in (accepted, rejected):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            selected = producer / retained.relative_to(historical)
            metadata = root / "metadata"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "prepare_dumps.py"),
                    "--dataset-folder",
                    str(selected),
                    "--preprocessing-metadata-folder",
                    str(metadata),
                    "--n-dumps",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            paths = (metadata / "dumps/paths_file_0.txt").read_text().splitlines()
            self.assertEqual(
                {(selected / path).resolve() for path in paths}, {accepted}
            )
            self.assertNotIn(rejected, {(selected / path).resolve() for path in paths})
            self.assertEqual((metadata / "raw-dataset-link").resolve(), selected)


if __name__ == "__main__":
    unittest.main()
