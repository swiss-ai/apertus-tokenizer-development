"""
python3 scripts/tokenization/prepare_dumps.py --dataset-folder /capstor/store/cscs/swissai/a06/datasets_raw/fineweb-2 --filter-in snapshot/data train --filter-out _removed und_ --preprocessing-metadata-folder datasets/fineweb-2 --n-dumps 10
"""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any


def get_parquet_files(path_to_folder: str) -> list[str]:
    files = [
        os.path.join(dp, f)
        for dp, _, fn in os.walk(os.path.expanduser(path_to_folder), followlinks=True)
        for f in fn
    ]

    if len(files) == 0:
        raise ValueError(f"No .parquet files found in {path_to_folder}")

    filtered_files = [
        raw_file
        for raw_file in files
        if Path(raw_file).suffix.lower().endswith(".parquet")
    ]

    return filtered_files


def get_manifest_entries(
    dataset_folder: str, manifest_path: str, path_key: str
) -> list[tuple[str, dict[str, Any]]]:
    """Read an exact, ordered Parquet inventory from a JSONL manifest."""
    root = Path(dataset_folder).resolve()
    entries = []
    seen = set()
    with open(manifest_path, encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Manifest line {line_number} is not a JSON object")
            relative = PurePosixPath(str(row.get(path_key, "")))
            if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"Invalid {path_key!r} on manifest line {line_number}: {relative}"
                )
            path = root.joinpath(*relative.parts)
            if path in seen:
                raise ValueError(f"Duplicate manifest path: {relative}")
            if path.suffix.lower() != ".parquet" or not path.is_file():
                raise ValueError(f"Manifest Parquet file does not exist: {path}")
            seen.add(path)
            entries.append((str(path), row))
    if not entries:
        raise ValueError(f"No Parquet files found in manifest {manifest_path}")
    return entries


def filter_in(list_of_files: list[str], list_of_folders: list[str]) -> list[str]:
    return_list_of_files = []
    for folder in list_of_folders:
        return_list_of_files.extend([file for file in list_of_files if (folder in file and file not in return_list_of_files)])
    return return_list_of_files


def filter_out(list_of_files: list[str], list_of_folders: list[str]) -> list[str]:
    for folder in list_of_folders:
        list_of_files = [file for file in list_of_files if folder not in file]
    return list_of_files


def load_group_metadata(
    path: str, root_key: str, id_field: str
) -> dict[str, dict[str, Any]]:
    """Load lookup metadata used to derive configured output groups."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if root_key:
        if not isinstance(payload, dict) or root_key not in payload:
            raise ValueError(f"Group metadata has no {root_key!r} root")
        payload = payload[root_key]
    if isinstance(payload, dict):
        rows = payload.items()
    elif isinstance(payload, list):
        rows = []
        for index, row in enumerate(payload):
            if not isinstance(row, dict) or not row.get(id_field):
                raise ValueError(
                    f"Group metadata row {index} has no {id_field!r} identifier"
                )
            rows.append((str(row[id_field]), row))
    else:
        raise TypeError("Group metadata root must be an object or an array")

    metadata = {}
    for identifier, row in rows:
        if not isinstance(row, dict):
            raise TypeError(f"Group metadata entry {identifier!r} is not an object")
        identifier = str(identifier)
        if identifier in metadata:
            raise ValueError(f"Duplicate group metadata identifier: {identifier!r}")
        metadata[identifier] = row
    return metadata


def group_entries(
    entries: list[tuple[str, dict[str, Any]]],
    group_fields: list[str],
    metadata: dict[str, dict[str, Any]],
    metadata_lookup_field: str,
) -> dict[tuple[str, ...], list[str]]:
    """Group manifest files by configured row or lookup-metadata fields."""
    if not group_fields:
        return {(): [path for path, _ in entries]}
    grouped: dict[tuple[str, ...], list[str]] = {}
    for path, row in entries:
        lookup = {}
        if metadata:
            identifier = row.get(metadata_lookup_field)
            if identifier is None or str(identifier) not in metadata:
                raise ValueError(
                    f"Manifest entry {path} has no group metadata for "
                    f"{metadata_lookup_field}={identifier!r}"
                )
            lookup = metadata[str(identifier)]
        group = []
        for field in group_fields:
            value = row.get(field) or lookup.get(field)
            component = PurePosixPath(str(value or ""))
            if (
                not value
                or len(component.parts) != 1
                or component.parts[0] in (".", "..")
            ):
                raise ValueError(
                    f"Manifest entry {path} has invalid group field {field}={value!r}"
                )
            group.append(component.parts[0])
        grouped.setdefault(tuple(group), []).append(path)
    return grouped


def split_files(
    files: list[str], n_dumps: int | None, max_dump_bytes: int
) -> list[list[str]]:
    if not files:
        raise ValueError("cannot split an empty file list")
    if n_dumps is not None and n_dumps < 1:
        raise ValueError("number of dumps must be positive")
    sizes = [os.path.getsize(path) for path in files]
    if max_dump_bytes < 1:
        raise ValueError("max dump bytes must be positive")
    dump_count = n_dumps or max(1, (sum(sizes) + max_dump_bytes - 1) // max_dump_bytes)
    dump_count = min(dump_count, len(files))
    dumps = [[] for _ in range(dump_count)]
    dump_sizes = [0] * dump_count
    for path, size in sorted(zip(files, sizes), key=lambda item: -item[1]):
        index = dump_sizes.index(min(dump_sizes))
        dumps[index].append(path)
        dump_sizes[index] += size
    return dumps


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-folder",
        type=str,
        required=True,
        help="Path to a folder containing recursively .parquet files",
    )
    parser.add_argument(
        "--filter-in",
        nargs="+",
        help="Name of the paths to filter in. e.g. --filter-in esp en fr",
        default=None,
    )
    parser.add_argument(
        "--filter-out",
        nargs="+",
        help="Name of the paths to filter out. e.g. --filter out test valid",
        default=None,
    )
    parser.add_argument(
        "--preprocessing-metadata-folder",
        type=str,
        required=True,
        help="Path to a folder to store the generated metadata files",
    )
    parser.add_argument(
        "--n-dumps",
        type=int,
        default=None,
        help="Total number of dumps to split the files into. If None it will automatically compute this value based on the amount and size of parquet files",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional JSONL file containing the exact source inventory",
    )
    parser.add_argument(
        "--manifest-path-key",
        default="relative_path",
        help="Manifest field containing a path relative to the dataset root",
    )
    parser.add_argument(
        "--group-fields",
        default="",
        help="Comma-separated manifest or metadata fields mirrored into token output",
    )
    parser.add_argument(
        "--group-metadata",
        default="",
        help="Optional JSON object or array supplying missing group fields",
    )
    parser.add_argument(
        "--group-metadata-root",
        default="",
        help="Optional top-level key containing the group metadata entries",
    )
    parser.add_argument(
        "--group-metadata-lookup-field",
        default="",
        help="Manifest field whose value selects a group metadata entry",
    )
    parser.add_argument(
        "--group-metadata-id-field",
        default="id",
        help="Identifier field when group metadata entries are an array",
    )
    parser.add_argument(
        "--expected-groups",
        type=int,
        default=0,
        help="Fail unless grouping produces exactly this many groups",
    )
    parser.add_argument(
        "--expected-group-heads",
        default="",
        help="Comma-separated expected values of the first grouping component",
    )
    parser.add_argument(
        "--max-dump-bytes",
        type=int,
        default=150_000_000_000,
        help="Target upper size used when --n-dumps is omitted",
    )
    args = parser.parse_args()

    return args


def main(args):
    print(f"Scanning parquet files in {args.dataset_folder}...")
    entries = (
        get_manifest_entries(
            args.dataset_folder, args.manifest, args.manifest_path_key
        )
        if args.manifest
        else [(path, {}) for path in get_parquet_files(args.dataset_folder)]
    )
    parquet_files = [path for path, _ in entries]
    print(f"Found a total of {len(parquet_files)} in {args.dataset_folder}")
    if args.filter_in:
        selected = set(filter_in(parquet_files, args.filter_in))
        entries = [entry for entry in entries if entry[0] in selected]
    if args.filter_out:
        selected = set(filter_out([entry[0] for entry in entries], args.filter_out))
        entries = [entry for entry in entries if entry[0] in selected]
    parquet_files = [path for path, _ in entries]
    if not parquet_files:
        raise ValueError("No Parquet files remain after filtering")
    size_of_parquet_files = [os.path.getsize(path) for path in parquet_files]
    print(
        f"Total number of files filtered to tokenize: {len(parquet_files)} ({sum(size_of_parquet_files) / 1e9:.2f} GB)"
    )

    group_fields = [
        field.strip() for field in args.group_fields.split(",") if field.strip()
    ]
    if len(group_fields) != len(set(group_fields)):
        raise ValueError("group fields must be unique")
    if group_fields and not args.manifest:
        raise ValueError("group fields require a manifest")
    if args.group_metadata and not group_fields:
        raise ValueError("group metadata requires group fields")
    if args.group_metadata and not args.group_metadata_lookup_field:
        raise ValueError("group metadata requires a manifest lookup field")
    metadata = (
        load_group_metadata(
            args.group_metadata,
            args.group_metadata_root,
            args.group_metadata_id_field,
        )
        if args.group_metadata
        else {}
    )
    grouped = group_entries(
        entries,
        group_fields,
        metadata,
        args.group_metadata_lookup_field,
    )
    if args.expected_groups and len(grouped) != args.expected_groups:
        raise ValueError(f"Expected {args.expected_groups} groups, found {len(grouped)}")
    if args.expected_group_heads:
        expected = {
            value.strip()
            for value in args.expected_group_heads.split(",")
            if value.strip()
        }
        observed = {group[0] for group in grouped if group}
        if observed != expected:
            raise ValueError(
                f"Expected group heads {sorted(expected)}, found {sorted(observed)}"
            )

    PATH_TO_DATASET_SYMLINK = os.path.join(
        args.preprocessing_metadata_folder, "raw-dataset-link"
    )
    PATH_TO_DUMP_FOLDER = os.path.join(args.preprocessing_metadata_folder, "dumps")

    # Create folder to store paths files
    Path(PATH_TO_DUMP_FOLDER).mkdir(parents=True, exist_ok=True)

    # Create symlink to original dataset
    if not os.path.islink(PATH_TO_DATASET_SYMLINK):
        os.symlink(args.dataset_folder, PATH_TO_DATASET_SYMLINK)

    dump_total = 0
    for group, files in sorted(grouped.items()):
        dump_folder = Path(PATH_TO_DUMP_FOLDER).joinpath(*group)
        dump_folder.mkdir(parents=True, exist_ok=True)
        for index, dump_files in enumerate(
            split_files(files, args.n_dumps, args.max_dump_bytes)
        ):
            dump_size = sum(os.path.getsize(path) for path in dump_files)
            print(
                f"[ {'/'.join(group) or 'root'} | Dump {index} | "
                f"{dump_size / 1e9:.2f} GB | {len(dump_files)} Files ]"
            )
            relative_paths = [
                os.path.relpath(path, Path(args.dataset_folder).resolve())
                for path in dump_files
            ]
            (dump_folder / f"paths_file_{index}.txt").write_text(
                "".join(path + "\n" for path in relative_paths), encoding="utf-8"
            )
            dump_total += 1

    print(f"Finished preparing {dump_total} dumps in {len(grouped)} groups!")


if __name__ == "__main__":
    _args = get_args()
    main(_args)
