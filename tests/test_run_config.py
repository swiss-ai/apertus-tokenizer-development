import tempfile
import unittest
from pathlib import Path

from tokenization_scripts.run_all import launcher_command, select_configs
from tokenization_scripts.runner.config import discover_configs, load_config
from tokenization_scripts.runner.transforms import MINHASH_UPSAMPLING


def write_config(path: Path, *, source: str = "/data/input", extra: str = "") -> None:
    path.write_text(
        f"""TOKENIZER=../preliminary_mul_200k/tokenizer.json
TOKENIZER_NAME=preliminary_mul_200k
DATASET_NAME={path.stem}-dataset
COLUMN_KEY=text
PATH_TO_RAW_DATASET={source}
PATH_TO_PREPROCESSING_METADATA=/tokens/{path.stem}
PATH_TO_OUTPUT_FOLDER=/tokens/{path.stem}
DUMPS_NUMBER=4
NUMBER_OF_DATATROVE_TASKS=2
{extra}
""",
        encoding="utf-8",
    )


class RunConfigTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_loads_explicit_transform_without_executing_shell(self):
        path = self.root / "fineweb.cfg"
        write_config(path, extra=f"PREPROCESSING_TRANSFORMS={MINHASH_UPSAMPLING}")

        config = load_config(path)

        self.assertEqual(config.run, "fineweb")
        self.assertEqual(config.tasks, 2)
        self.assertEqual(config.transforms[0].type, MINHASH_UPSAMPLING)
        self.assertEqual(config.plan()["preprocessing"][0]["version"], 1)

    def test_legacy_rehydration_flag_maps_to_named_upsampler(self):
        path = self.root / "legacy.cfg"
        write_config(path, extra="REHYDRATE_FLAG=True")

        config = load_config(path)

        self.assertEqual([item.type for item in config.transforms], [MINHASH_UPSAMPLING])

    def test_empty_input_path_disables_a_run(self):
        path = self.root / "disabled.cfg"
        write_config(path, source="")

        self.assertFalse(load_config(path).enabled)

    def test_discovery_is_stable_and_selection_uses_config_name(self):
        write_config(self.root / "b.cfg")
        write_config(self.root / "a.cfg")
        configs = discover_configs(self.root)

        selected, disabled = select_configs(configs, ["b"], False)

        self.assertEqual([config.run for config in configs], ["a", "b"])
        self.assertEqual([config.run for config in selected], ["b"])
        self.assertEqual(disabled, [])

    def test_launcher_runs_existing_submission_script(self):
        path = self.root / "one.cfg"
        write_config(path)

        command = launcher_command(load_config(path), dont_compute_dumps=True)

        self.assertTrue(command[0].endswith("tokenize_script.sh"))
        self.assertEqual(command[-1], "--dont_compute_dumps")


if __name__ == "__main__":
    unittest.main()
