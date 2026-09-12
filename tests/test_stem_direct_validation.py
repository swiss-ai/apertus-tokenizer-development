import importlib.util
import tempfile
import unittest
from array import array
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tokenization_scripts/validate_stem_direct.py"


class StemDirectValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys

        sys.path.insert(0, str(SCRIPT.parent))
        spec = importlib.util.spec_from_file_location("validate_stem_direct", SCRIPT)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_completed_dumps_require_exact_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "dumps").mkdir()
            completed = root / "completed-dumps"
            completed.mkdir()
            (completed / "paths_file_0.txt").write_text("a.parquet\nb.parquet\n")
            (completed / "paths_file_1.txt").write_text("c.parquet\n")
            self.assertEqual(
                self.module._completed_dumps(root, 2),
                {0: ["a.parquet", "b.parquet"], 1: ["c.parquet"]},
            )
            with self.assertRaisesRegex(ValueError, "completed dump numbers differ"):
                self.module._completed_dumps(root, 3)
            (root / "dumps" / "paths_file_2.txt").write_text("d.parquet\n")
            with self.assertRaisesRegex(ValueError, "unfinished dump"):
                self.module._completed_dumps(root, 2)

    def test_source_row_coordinates_reject_gaps_and_duplicates(self):
        records = array("Q", [0, 1, 2, 0, 1])
        self.module._check_records(records, 0, 3, Path("tokens.map"))
        self.module._check_records(records, 3, 2, Path("tokens.map"))
        records[4] = 0
        with self.assertRaisesRegex(ValueError, "row coordinates differ"):
            self.module._check_records(records, 3, 2, Path("tokens.map"))


if __name__ == "__main__":
    unittest.main()
