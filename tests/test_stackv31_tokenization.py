import argparse
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from data_pipeline_pretrain.pipeline.tokens import read_token_map
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from tokenization_scripts import preprocess_megatron, stackv31


class Stackv31TokenizationTest(unittest.TestCase):
    def test_selection_rank_slice_and_pipeline_facts_reach_the_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            pq.write_table(
                pa.table(
                    {
                        "content": ["alpha beta", "excluded", "gamma"],
                        "content_id": ["one", "two", "three"],
                        "apertus_include": [True, False, True],
                        "exclusion_reason": ["", "license", ""],
                    }
                ),
                source / "part.parquet",
            )
            paths = root / "paths.txt"
            paths.write_text("part.parquet\n", encoding="utf-8")
            tokenizer_path = root / "tokenizer.json"
            tokenizer = Tokenizer(
                WordLevel(
                    vocab={
                        "<UNK>": 0,
                        "<BOS>": 1,
                        "<EOS>": 2,
                        "alpha": 3,
                        "beta": 4,
                        "gamma": 5,
                    },
                    unk_token="<UNK>",
                )
            )
            tokenizer.pre_tokenizer = Whitespace()
            tokenizer.save(str(tokenizer_path))
            output = root / "tokens"
            facts = {
                "dataset": "stackv31-languages-v1",
                "language_slug": "python",
                "language_category": "programming",
            }
            preprocess_megatron.main(
                argparse.Namespace(
                    tokenizer_name_or_path=str(tokenizer_path),
                    eos_token=None,
                    output_folder=str(output),
                    logging_dir=str(root / "logs"),
                    n_tasks=1,
                    n_workers=1,
                    local_tasks=1,
                    rank_offset=0,
                    dataset=str(source),
                    paths_file=str(paths),
                    column="content",
                    rehydrate="False",
                    extension=".parquet",
                    include_boolean_column="apertus_include",
                    exclusion_reason_column="exclusion_reason",
                    provenance_pipeline_json=facts,
                    tokenizer_batch_size=2,
                )
            )
            token_map = read_token_map((output / "00000_tokens.map").read_bytes())
            self.assertEqual(list(token_map["records"]), [0, 2])
            self.assertEqual(token_map["manifest"]["sequence_count"], 2)
            self.assertEqual(token_map["manifest"]["pipeline"], facts)
            self.assertEqual(token_map["manifest"]["text_column"], "content")

    def test_prepared_group_runs_and_resolves_to_its_parquet_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            parquet = source / "languages" / "python" / "part.parquet"
            parquet.parent.mkdir(parents=True)
            pq.write_table(
                pa.table(
                    {
                        "content": ["alpha", "excluded", "gamma"],
                        "content_id": ["one", "two", "three"],
                        "apertus_include": [True, False, True],
                        "exclusion_reason": ["", "license", ""],
                    }
                ),
                parquet,
            )
            manifest = {
                "language_slug": "python",
                "language": "Python",
                "relative_path": "languages/python/part.parquet",
                "rows": 3,
                "included_rows": 2,
                "content_bytes": 18,
                "included_content_bytes": 10,
            }
            (source / "manifest.jsonl").write_text(
                json.dumps(manifest) + "\n", encoding="utf-8"
            )
            (source / "languages.json").write_text(
                json.dumps(
                    {
                        "languages": {
                            "python": {"name": "Python", "category": "programming"}
                        }
                    }
                ),
                encoding="utf-8",
            )
            summary = {
                "complete": True,
                "source": {"revision": "source-revision"},
                "decision": {
                    "policy_tag": "policy-v1",
                    "policy_sha256": "f" * 64,
                    "signals_revision": "signals-revision",
                },
            }
            (source / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            (source / "_SUCCESS").write_text("sealed\n", encoding="utf-8")
            tokenizer_path = root / "tokenizer.json"
            tokenizer = Tokenizer(
                WordLevel(
                    vocab={
                        "<UNK>": 0,
                        "<BOS>": 1,
                        "<EOS>": 2,
                        "alpha": 3,
                        "gamma": 4,
                    },
                    unk_token="<UNK>",
                )
            )
            tokenizer.pre_tokenizer = Whitespace()
            tokenizer.save(str(tokenizer_path))
            output = root / "tokens"
            work = root / "work"
            prepare_args = stackv31.parser().parse_args(
                [
                    "prepare",
                    "--input-root",
                    str(source),
                    "--group-manifest",
                    str(source / "manifest.jsonl"),
                    "--category-map",
                    str(source / "languages.json"),
                    "--output-root",
                    str(output),
                    "--work-root",
                    str(work),
                    "--tokenizer-path",
                    str(tokenizer_path),
                    "--expected-languages",
                    "1",
                    "--expected-categories",
                    "programming",
                    "--expected-policy-tag",
                    "policy-v1",
                    "--expected-signals-revision",
                    "signals-revision",
                    "--expected-source-revision",
                    "source-revision",
                    "--implementation-commit",
                    "0" * 40,
                    "--target-jobs",
                    "1",
                ]
            )
            stackv31.prepare(prepare_args)
            stackv31.run_assignment(
                argparse.Namespace(work_root=work, assignment=0, workers=1)
            )
            result = stackv31.validate(
                argparse.Namespace(
                    work_root=work,
                    map_root=None,
                    max_map_overhead=100.0,
                    require_assignment_markers=True,
                )
            )
            self.assertTrue(result["complete"])
            self.assertEqual(result["sequences"], 2)
            self.assertEqual(result["resolved_languages"], 1)


if __name__ == "__main__":
    unittest.main()
