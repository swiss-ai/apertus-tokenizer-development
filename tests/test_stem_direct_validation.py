import importlib.util
import tempfile
import unittest
from array import array
from pathlib import Path

import numpy as np

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

    def test_full_binary_validation_rejects_range_boundary_and_literal_collisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary, index = root / "tokens.bin", root / "tokens.idx"
            index.write_bytes(
                b"\0" * 34
                + np.array([3, 4], dtype="<i4").tobytes()
                + np.array([0, 12], dtype="<i8").tobytes()
            )
            info = {"sequence_count": 2, "token_count": 7, "token_bytes": 4}
            cases = [
                ([1, 17, 2, 1, 9, 10, 2], None),
                ([1, 17, 2, 1, 999, 10, 2], "outside tokenizer"),
                ([1, 17, 2, 1, 9, 10, 3], "boundary differs"),
                ([1, 17, 2, 1, 1, 10, 2], "collision inside"),
            ]
            for values, error in cases:
                binary.write_bytes(np.array(values, dtype="<u4").tobytes())
                with self.subTest(values=values):
                    if error:
                        with self.assertRaisesRegex(ValueError, error):
                            self.module._validate_token_values(binary, index, info, 200)
                    else:
                        self.module._validate_token_values(binary, index, info, 200)


if __name__ == "__main__":
    unittest.main()
