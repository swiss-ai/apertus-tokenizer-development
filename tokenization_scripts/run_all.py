"""Plan or submit every configured tokenization run from one entry point."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

if __package__:
    from .runner.config import RunConfig, discover_configs
else:
    from runner.config import RunConfig, discover_configs

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = SCRIPT_DIR / "configs_apertus_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "submit"))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--run", action="append", help="config filename without .cfg")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--dont-compute-dumps", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print launcher commands instead of executing them",
    )
    return parser.parse_args()


def select_configs(
    configs: list[RunConfig], runs: list[str] | None, include_disabled: bool
) -> tuple[list[RunConfig], list[RunConfig]]:
    by_name = {config.run: config for config in configs}
    if runs:
        unknown = sorted(set(runs) - by_name.keys())
        if unknown:
            raise ValueError(f"unknown runs: {', '.join(unknown)}")
        selected = [by_name[name] for name in dict.fromkeys(runs)]
    else:
        selected = configs
    disabled = [config for config in selected if not config.enabled]
    if not include_disabled:
        selected = [config for config in selected if config.enabled]
    return selected, disabled


def launcher_command(config: RunConfig, dont_compute_dumps: bool) -> list[str]:
    command = [str(SCRIPT_DIR / "tokenize_script.sh"), str(config.path)]
    if dont_compute_dumps:
        command.append("--dont_compute_dumps")
    return command


def main() -> None:
    args = parse_args()
    configs = discover_configs(args.config_dir)
    selected, disabled = select_configs(
        configs, args.run, args.include_disabled
    )
    if args.command == "plan":
        print(
            json.dumps(
                {
                    "runs": [config.plan() for config in selected],
                    "disabled": [config.run for config in disabled],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if not selected:
        raise ValueError("selection contains no runnable datasets")
    for config in selected:
        if not config.enabled:
            raise ValueError(
                f"{config.run}: PATH_TO_RAW_DATASET is empty; cannot submit"
            )
        command = launcher_command(config, args.dont_compute_dumps)
        print(f"[{config.run}] {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=SCRIPT_DIR, check=True)


if __name__ == "__main__":
    main()
