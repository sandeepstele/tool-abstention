"""Command-line interface for project utilities."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from tool_abstention import __version__
from tool_abstention.config import ProjectConfig, load_yaml_config
from tool_abstention.schemas import SchemaKind, export_schemas, validate_record


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

    export = subparsers.add_parser(
        "export-schemas", help="export canonical public JSON Schemas"
    )
    export.add_argument("directory", type=Path)

    record = subparsers.add_parser(
        "validate-record", help="validate one JSON record against a public schema"
    )
    record.add_argument("kind", choices=("task", "pair", "prediction", "evaluation"))
    record.add_argument("path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-config":
            config = load_yaml_config(args.path, ProjectConfig)
            print(config.model_dump_json(indent=2))
            return
        if args.command == "export-schemas":
            exported = export_schemas(args.directory)
            for name, path in exported.items():
                print(f"{name}: {path}")
            return
        if args.command == "validate-record":
            record = validate_record(args.path, cast("SchemaKind", args.kind))
            print(record.model_dump_json(indent=2))
            return
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    if args.command is None:
        parser.print_help()
