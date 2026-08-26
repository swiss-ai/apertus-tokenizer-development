import argparse
import hashlib
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from data_pipeline_pretrain.pipeline.tokens import read_token_map
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from tokenization_scripts import preprocess_megatron


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
            summary = root / "summary.json"
            summary.write_text('{"complete":true}\n', encoding="utf-8")
            manifest = root / "manifest.jsonl"
            manifest.write_text('{"relative_path":"part.parquet"}\n', encoding="utf-8")
            facts = {
                "dataset": "stackv31-languages-v1",
                "selection": {"policy_tag": "policy-v1"},
            }
            preprocess_megatron.main(
                argparse.Namespace(
                    tokenizer_name_or_path=str(tokenizer_path),
                    eos_token=None,
                    output_folder=str(output),
                    logging_dir=str(root / "logs"),
                    n_tasks=1,
                    n_workers=1,
                    dataset=str(source),
                    paths_file=str(paths),
                    column="content",
                    id_column="content_id",
                    rehydrate="False",
                    extension=".parquet",
                    include_boolean_column="apertus_include",
                    exclusion_reason_column="exclusion_reason",
                    provenance_pipeline_json=facts,
                    provenance_group_keys="language_category,language_slug",
                    provenance_group_path="programming/python",
                    provenance_digest_files=(
                        f"artifact_summary={summary},artifact_manifest={manifest}"
                    ),
                    tokenizer_batch_size=2,
                )
            )
            token_map = read_token_map((output / "00000_tokens.map").read_bytes())
            self.assertEqual(list(token_map["records"]), [0, 2])
            self.assertEqual(token_map["manifest"]["sequence_count"], 2)
            self.assertEqual(
                token_map["manifest"]["pipeline"],
                {
                    **facts,
                    "language_category": "programming",
                    "language_slug": "python",
                    "digests": {
                        "artifact_summary": hashlib.sha256(
                            summary.read_bytes()
                        ).hexdigest(),
                        "artifact_manifest": hashlib.sha256(
                            manifest.read_bytes()
                        ).hexdigest(),
                    },
                },
            )
            self.assertEqual(token_map["manifest"]["text_column"], "content")


if __name__ == "__main__":
    unittest.main()
