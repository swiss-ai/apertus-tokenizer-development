#!/usr/bin/env python3
"""
python3 preprocess_megatron.py --tokenizer-name-or-path meta-llama/Meta-Llama-3-8B --output-folder tokenized_datasets/fineweb-edu --n-tasks 16 --dataset datasets/fineweb-edu/raw-dataset-link --paths-file datasets/fineweb-edu/dumps/paths_file_0.txt
"""

import argparse

from data_pipeline_pretrain.pipeline.tokens import (
    MegatronDocumentTokenizer,
    ProvenanceParquetReader,
    Rehydrater,
)
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader


def get_args():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="Tokenizer")
    group.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        required=True,
        help="A path to a directory containing vocabulary files required by the tokenizer or the model id of a predefined tokenizer hosted inside a model repo on the Hugging Face Hub.",
    )
    group.add_argument(
        "--eos-token",
        type=str,
        default=None,
        help="EOS token to add after each document. Default: None",
    )

    group = parser.add_argument_group(title="Output data")
    group.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Path to the output folder to store the tokenized documents",
    )
    group = parser.add_argument_group(title="Miscellaneous configs")
    group.add_argument(
        "--logging-dir",
        type=str,
        default=None,
        help="Path to a folder for storing the logs of the preprocessing step. Default: None",
    )
    group.add_argument(
        "--n-tasks",
        type=int,
        default=8,
        help="Total number of tasks to run the preprocessing step. Default: 8",
    )
    group.add_argument(
        "--n-workers",
        type=int,
        default=-1,
        help="Number of workers executing concurrently --n-tasks tasks. Default: -1, which means --n-workers==--n-tasks",
    )
    group = parser.add_argument_group(title="Dataset configs")
    group.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to a folder recursively containing multiple .parquet files",
    )
    group.add_argument(
        "--paths-file",
        type=str,
        required=True,
        help="A file with one path per line (without the `dataset` prefix) to read",
    )
    group.add_argument(
        "--column",
        type=str,
        default="text",
        help="Column to preprocess from the Dataset. Default: text",
    )
    group.add_argument(
        "--id-column",
        default="id",
        help="Column used as the document id. Default: id",
    )
    group.add_argument(
        "--rehydrate",
        type=str,
        default="False",
        help="Whether to rehydrate the dataset. Default: False",
    )
    group.add_argument(
        "--extension",
        type=str,
        default=".parquet",
        help="File extension to use. e.g. .parquet or .jsonl.zst. Default: .parquet",
    )
    group.add_argument(
        "--include-boolean-column",
        default="",
        help="Optional boolean metadata column deciding which rows are tokenized",
    )
    group.add_argument(
        "--exclusion-reason-column",
        default="exclusion_reason",
        help="Metadata column explaining excluded rows",
    )
    group.add_argument(
        "--tokenizer-batch-size",
        type=int,
        default=10000,
        help="Documents encoded in one tokenizer batch. Default: 10000",
    )

    args = parser.parse_args()

    return args


def main(args):
    n_tasks = args.n_tasks
    # Check number of files > n tasks
    with open(args.paths_file, "rb") as f:
        number_of_files = sum(1 for _ in f)
    n_tasks = min(n_tasks, number_of_files)
    if n_tasks < 1:
        raise ValueError("paths file contains no inputs")

    if "jsonl" in args.extension:
        reader = JsonlReader(
            data_folder=args.dataset,
            paths_file=args.paths_file,
            text_key=args.column,
            id_key=getattr(args, "id_column", "id"),
        )
        write_source_map = False
    else:
        reader = ProvenanceParquetReader(
            data_folder=args.dataset,
            paths_file=args.paths_file,
            text_key=args.column,
            id_key=getattr(args, "id_column", "id"),
        )
        write_source_map = True

    include_column = getattr(args, "include_boolean_column", "")
    if include_column and not write_source_map:
        raise ValueError("row selection is only supported for Parquet inputs")
    selection_steps = []
    if include_column:
        from data_pipeline_pretrain.pipeline.filters import ApertusCodeLicenseFilter

        selection_steps.append(
            ApertusCodeLicenseFilter(
                include_column=include_column,
                reason_column=getattr(
                    args, "exclusion_reason_column", "exclusion_reason"
                ),
            )
        )
    do_rehydrate = args.rehydrate is not None and args.rehydrate.lower() in (
        "true",
        "1",
        "yes",
    )
    preprocess_executor = LocalPipelineExecutor(
        pipeline=[
            reader,
            *selection_steps,
            *([Rehydrater()] if do_rehydrate else []),
            MegatronDocumentTokenizer(
                output_folder=args.output_folder,
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                eos_token=args.eos_token,
                provenance=write_source_map,
                batch_size=getattr(args, "tokenizer_batch_size", 10000),
            ),
        ],
        tasks=n_tasks,
        workers=args.n_workers,
        start_method="spawn",
        logging_dir=args.logging_dir,
    )
    preprocess_executor.run()


if __name__ == "__main__":
    _args = get_args()
    main(_args)
