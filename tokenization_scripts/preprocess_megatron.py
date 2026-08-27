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


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def get_args(argv=None):
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
    group.add_argument(
        "--batch-size",
        type=positive_int,
        default=10_000,
        help="Maximum documents per tokenizer call. Default: 10000",
    )
    group.add_argument(
        "--batch-bytes",
        type=positive_int,
        default=32 * 1024**2,
        help="Maximum UTF-8 input bytes per tokenizer call. Default: 33554432",
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
        "--include-reason-column",
        default="",
        help="Reason metadata paired with --include-boolean-column",
    )
    group.add_argument(
        "--included-reason",
        default="included",
        help="Exact reason value denoting an included row; may be empty",
    )
    group.add_argument(
        "--max-sequence-tokens",
        type=int,
        default=0,
        help="Fail before tokenization when an exact sequence exceeds this length; 0 disables the guard",
    )

    args = parser.parse_args(argv)

    return args


def main(args):
    n_tasks = args.n_tasks
    # Check number of files > n tasks
    with open(args.paths_file, "rb") as f:
        number_of_files = sum(1 for _ in f)
    n_tasks = min(n_tasks, number_of_files)
    if n_tasks < 1:
        raise ValueError("paths file contains no inputs")
    n_workers = min(args.n_workers, n_tasks) if args.n_workers > 0 else args.n_workers

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
        from data_pipeline_pretrain.pipeline.filters import MetadataInclusionFilter

        reason_column = getattr(args, "include_reason_column", "")
        if not reason_column:
            raise ValueError(
                "include boolean column requires an include reason column"
            )
        selection_steps.append(
            MetadataInclusionFilter(
                include_column=include_column,
                reason_column=reason_column,
                included_reason=getattr(args, "included_reason", "included"),
            )
        )
    max_sequence_tokens = int(getattr(args, "max_sequence_tokens", 0) or 0)
    if max_sequence_tokens < 0:
        raise ValueError("max sequence tokens must be non-negative")
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
                batch_size=getattr(
                    args,
                    "batch_size",
                    getattr(args, "tokenizer_batch_size", 10000),
                ),
                batch_bytes=getattr(args, "batch_bytes", 32 * 1024**2),
                max_sequence_tokens=max_sequence_tokens,
            ),
        ],
        tasks=n_tasks,
        workers=n_workers,
        start_method="spawn",
        logging_dir=args.logging_dir,
    )
    preprocess_executor.run()


if __name__ == "__main__":
    _args = get_args()
    main(_args)
