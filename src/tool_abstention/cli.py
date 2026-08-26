"""Command-line interface for project utilities."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from tool_abstention import __version__
from tool_abstention.config import ProjectConfig, load_yaml_config


def build_parser() -> argparse.ArgumentParser:
    """Create the project argument parser."""
    parser = argparse.ArgumentParser(
        prog="tool-abstention",
        description="Train and evaluate tool-use abstention.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")

    validate = subparsers.add_parser(
        "validate-config", help="validate a project YAML configuration"
    )
    validate.add_argument("path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate-config":
        config = load_yaml_config(args.path, ProjectConfig)
        print(config.model_dump_json(indent=2))
        return
    if args.command is None:
        parser.print_help()
