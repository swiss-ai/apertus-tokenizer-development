import argparse
import json
import tempfile
import unittest
from pathlib import Path

from tokenization_scripts import stackv31


class Stackv31PlannerTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input_root = self.root / "input"
        self.output_root = self.root / "output"
        self.work_root = self.root / "work"
        self.input_root.mkdir()
        self.tokenizer = self.root / "tokenizer.json"
        self.tokenizer.write_text("tokenizer", encoding="utf-8")

    def artifact(self, languages):
        manifest = []
        category_rows = {}
        total_rows = 0
        included_rows = 0
        for slug, language in languages.items():
            category_rows[slug] = {
                "name": language["name"],
                "category": language["category"],
            }
            for file_index, counts in enumerate(language["files"]):
                relative = f"languages/{slug}/{file_index:04d}.parquet"
                path = self.input_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("parquet placeholder", encoding="utf-8")
                rows, kept, content_bytes, included_bytes = counts
                total_rows += rows
                included_rows += kept
                manifest.append(
                    {
                        "language_slug": slug,
                        "language": language["name"],
                        "relative_path": relative,
                        "rows": rows,
                        "included_rows": kept,
                        "content_bytes": content_bytes,
                        "included_content_bytes": included_bytes,
                    }
                )
        (self.input_root / "manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8"
        )
        (self.input_root / "languages.json").write_text(
            json.dumps({"languages": category_rows}), encoding="utf-8"
        )
        summary = {
            "complete": True,
            "artifact": {"rows": total_rows, "included_rows": included_rows},
            "source": {"revision": "stack-source-revision"},
            "decision": {
                "policy_tag": "policy-v1",
                "policy_sha256": "f" * 64,
                "signals_revision": "signals-revision",
            },
        }
        (self.input_root / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        (self.input_root / "_SUCCESS").write_text("sealed\n", encoding="utf-8")

    def prepare(self, expected_languages, expected_categories, target_jobs=2):
        args = stackv31.parser().parse_args(
            [
                "prepare",
                "--input-root",
                str(self.input_root),
                "--group-manifest",
                str(self.input_root / "manifest.jsonl"),
                "--category-map",
                str(self.input_root / "languages.json"),
                "--output-root",
                str(self.output_root),
                "--work-root",
                str(self.work_root),
                "--tokenizer-path",
                str(self.tokenizer),
                "--expected-languages",
                str(expected_languages),
                "--expected-categories",
                expected_categories,
                "--expected-policy-tag",
                "policy-v1",
                "--expected-signals-revision",
                "signals-revision",
                "--expected-source-revision",
                "stack-source-revision",
                "--implementation-commit",
                "0" * 40,
                "--target-jobs",
                str(target_jobs),
            ]
        )
        return stackv31.prepare(args)

    def test_prepare_groups_languages_by_category_and_uses_manifest_counts(self):
        self.artifact(
            {
                "python": {
                    "name": "Python",
                    "category": "programming",
                    "files": [(3, 2, 300, 200)],
                },
                "json": {
                    "name": "JSON",
                    "category": "data",
                    "files": [(4, 4, 400, 400)],
                },
            }
        )
        run = self.prepare(2, "programming,data")
        self.assertEqual(run["totals"]["expected_sequences"], 6)
        self.assertEqual(run["totals"]["categories"], ["data", "programming"])
        by_slug = {entry["language_slug"]: entry for entry in run["languages"]}
        self.assertEqual(by_slug["python"]["output_relative"], "programming/python")
        self.assertEqual(by_slug["json"]["output_relative"], "data/json")
        self.assertEqual(
            (self.work_root / "paths" / "python.txt").read_text(encoding="utf-8"),
            "languages/python/0000.parquet\n",
        )

    def test_large_language_slices_cover_every_global_rank_once(self):
        plan = [
            {
                "language_slug": "python",
                "tasks": 4,
                "included_content_bytes": 4_000_000,
                "expected_sequences": 0,
            }
        ]
        work = stackv31.assignments(plan, target_jobs=2)
        pieces = [piece for assignment in work for piece in assignment["pieces"]]
        covered = []
        for piece in pieces:
            covered.extend(
                range(
                    piece["rank_offset"],
                    piece["rank_offset"] + piece["local_tasks"],
                )
            )
        self.assertEqual(covered, [0, 1, 2, 3])
        self.assertEqual(len(work), 2)

    def test_run_assignment_plugs_selection_and_group_facts_into_tokenization(self):
        self.artifact(
            {
                "python": {
                    "name": "Python",
                    "category": "programming",
                    "files": [(2, 1, 200, 100)],
                }
            }
        )
        run = self.prepare(1, "programming", target_jobs=1)
        captured = []
        args = argparse.Namespace(work_root=self.work_root, assignment=0, workers=8)
        stackv31.run_assignment(args, preprocess_main=captured.append)
        self.assertEqual(len(captured), 1)
        tokenization = captured[0]
        self.assertEqual(tokenization.include_boolean_column, "apertus_include")
        self.assertEqual(tokenization.local_tasks, 1)
        self.assertEqual(tokenization.rank_offset, 0)
        self.assertEqual(
            tokenization.output_folder,
            str(self.output_root / "programming" / "python"),
        )
        facts = tokenization.provenance_pipeline_json
        self.assertEqual(facts["language_slug"], "python")
        self.assertEqual(facts["language_category"], "programming")
        self.assertEqual(facts["selection"]["policy_tag"], "policy-v1")
        self.assertEqual(run["totals"]["assignments"], 1)

    def test_prepare_rejects_category_disagreement(self):
        self.artifact(
            {
                "python": {
                    "name": "Python",
                    "category": "programming",
                    "files": [(1, 1, 100, 100)],
                }
            }
        )
        manifest_path = self.input_root / "manifest.jsonl"
        row = json.loads(manifest_path.read_text(encoding="utf-8"))
        row["category"] = "prose"
        manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "conflicting categories"):
            self.prepare(1, "programming")


if __name__ == "__main__":
    unittest.main()
