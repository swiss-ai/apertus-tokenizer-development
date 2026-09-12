import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tokenization_scripts/generate_apertus_sft_source_configs.py"
)
SPEC = importlib.util.spec_from_file_location("apertus_sft_source_configs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
configs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(configs)


class SourceConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.text = self.root / "text"
        self.token = self.root / "tokens"
        self.cfg = self.root / "cfg"
        self.text.mkdir()
        self.template = self.root / "template.cfg"
        self.template.write_text(
            "TOKENIZER=../preliminary_mul_200k/tokenizer.json\n"
            "DATASET_NAME=old\n"
            "COLUMN_KEY=text\n"
            "PATH_TO_RAW_DATASET=/old/data\n"
            "PATH_TO_PREPROCESSING_METADATA=/old/meta\n"
            "PATH_TO_OUTPUT_FOLDER=/old/output\n"
            "DUMPS_NUMBER=8\n"
            "REHYDRATE_FLAG=False\n"
            "EXTENSION=.parquet\n"
        )

    def seal(self, sources):
        for item in sources:
            directory = self.text / "data" / item["slug"]
            directory.mkdir(parents=True)
            (directory / "part-00000.parquet").write_bytes(b"fixture")
        path = self.text / "_SOURCE_PARTITION_SUCCESS.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": configs.SCHEMA_VERSION,
                    "rows": sum(item["rows"] for item in sources),
                    "sources": sources,
                }
            )
        )
        return path

    def test_generate_one_config_per_pure_source(self):
        source = {
            "source": "nvidia/OpenMathReasoning",
            "slug": "nvidia-openmathreasoning-123456789abc",
            "rows": 7,
        }
        second = {"source": "smoltalk2", "slug": "smoltalk2-abcdef123456", "rows": 3}
        seal = self.seal([source, second])
        result = configs.generate(seal, self.text, self.token, self.template, self.cfg)
        self.assertEqual(len(result["sources"]), 2)
        self.assertEqual(len(list(self.cfg.glob("*.cfg"))), 2)
        first_cfg = (self.cfg / (source["slug"] + ".cfg")).read_text()
        self.assertIn(f"DATASET_NAME={source['slug']}", first_cfg)
        self.assertIn(
            f"PATH_TO_RAW_DATASET={self.text}/data/{source['slug']}", first_cfg
        )
        self.assertIn(
            f"PATH_TO_PREPROCESSING_METADATA={self.token}/_preprocessing/{source['slug']}",
            first_cfg,
        )
        self.assertIn(f"PATH_TO_OUTPUT_FOLDER={self.token}", first_cfg)
        self.assertEqual(
            configs.generate(seal, self.text, self.token, self.template, self.cfg),
            result,
        )

    def test_reject_unsafe_or_duplicate_source_paths(self):
        bad = {"source": "bad", "slug": "../escape", "rows": 1}
        seal = self.seal([bad])
        with self.assertRaises(ValueError):
            configs.generate(seal, self.text, self.token, self.template, self.cfg)

    def test_reject_manifest_not_matching_seal(self):
        item = {"source": "a", "slug": "a-123456789abc", "rows": 1}
        self.seal([item])
        other = self.root / "other.json"
        other.write_text("{}")
        with self.assertRaises(ValueError):
            configs.generate(other, self.text, self.token, self.template, self.cfg)


if __name__ == "__main__":
    unittest.main()
