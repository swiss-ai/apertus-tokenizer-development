"""Invariants for the CodeAlchemy tokenization configs.

These configs read the *filtered* Parquet produced by
swiss-ai/data-pipeline-pretrain's ``pipelines/code-datasets/code-alchemy``, not the
rehydrated tree. Reading the wrong root would silently tokenize unfiltered,
undecontaminated text, so the roots are asserted rather than left to review.
"""

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "tokenization_scripts" / "configs_apertus_v2"

RELEASED = ("code-dev", "code-dialogue", "code-enhance", "code-qa")
# code-trace is not released: 56.1% of its tokens exceed repeated-8-gram coverage 0.55
# and its documents of at least 16k tokens have zlib p10 of 0.043.
EXCLUDED = ("code-trace",)

FILTERED_INPUT_ROOT = (
    "/capstor/store/cscs/swissai/infra01/datasets/code_alchemy_filtered"
)
FILTERED_OUTPUT_ROOT = (
    "/capstor/store/cscs/swissai/infra01/datasets_tokenized/"
    "code_alchemy_filtered_apertus_v2"
)


def read_config(name):
    path = CONFIG_DIR / f"codealchemy-{name}.cfg"
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


class CodeAlchemyConfigTest(unittest.TestCase):
    def test_exactly_the_released_configurations_exist(self):
        present = sorted(
            path.stem.removeprefix("codealchemy-")
            for path in CONFIG_DIR.glob("codealchemy-*.cfg")
        )
        self.assertEqual(present, sorted(RELEASED))

    def test_excluded_configurations_have_no_config(self):
        for name in EXCLUDED:
            with self.subTest(name):
                self.assertFalse((CONFIG_DIR / f"codealchemy-{name}.cfg").exists())

    def test_each_config_reads_the_filtered_tree(self):
        for name in RELEASED:
            with self.subTest(name):
                values = read_config(name)
                self.assertEqual(
                    values["PATH_TO_RAW_DATASET"],
                    f"{FILTERED_INPUT_ROOT}/{name}/train",
                )

    def test_each_config_writes_the_filtered_tokenized_root(self):
        for name in RELEASED:
            with self.subTest(name):
                values = read_config(name)
                self.assertEqual(values["PATH_TO_OUTPUT_FOLDER"], FILTERED_OUTPUT_ROOT)
                self.assertEqual(
                    values["PATH_TO_PREPROCESSING_METADATA"],
                    f"{FILTERED_OUTPUT_ROOT}/{name}",
                )

    def test_no_config_names_a_superseded_root(self):
        superseded = ("code_alchemy_tokenization_ready", "code_alchemy_apertus_v2")
        for path in CONFIG_DIR.glob("codealchemy-*.cfg"):
            text = path.read_text(encoding="utf-8")
            for root in superseded:
                with self.subTest(config=path.name, root=root):
                    # code_alchemy_filtered_apertus_v2 legitimately contains the second
                    # string as a substring, so compare against the whole path segment.
                    self.assertNotIn(f"/{root}/", text)

    def test_shared_settings_are_consistent(self):
        for name in RELEASED:
            with self.subTest(name):
                values = read_config(name)
                self.assertEqual(values["COLUMN_KEY"], "text")
                self.assertEqual(values["REHYDRATE_FLAG"], "False")
                self.assertEqual(values["TOKENIZER_NAME"], "preliminary_mul_200k")
                self.assertEqual(values["DATASET_NAME"], f"codealchemy-{name}")
                self.assertEqual(values["EXTENSION"], ".parquet")


if __name__ == "__main__":
    unittest.main()
