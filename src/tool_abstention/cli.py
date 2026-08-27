"""Command-line interface for project utilities."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from tool_abstention import __version__
from tool_abstention.config import ProjectConfig, load_yaml_config
from tool_abstention.dataset import build_full_dataset
from tool_abstention.harness import evaluate_files
from tool_abstention.inference import (
    MlxBackend,
    load_inference_config,
    load_tasks,
    run_inference,
    select_stratified_smoke,
    write_run_manifest,
)
from tool_abstention.productivity import (
    audit_pairs,
    build_productivity_dataset,
    load_pairs,
)
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

    generate = subparsers.add_parser(
        "generate-productivity", help="build the deterministic productivity slice"
    )
    generate.add_argument("--config", required=True, type=Path)
    generate.add_argument("--output", required=True, type=Path)

    full = subparsers.add_parser(
        "generate-dataset", help="build the complete multi-domain dataset"
    )
    full.add_argument("--config", required=True, type=Path)
    full.add_argument("--output", required=True, type=Path)

    audit = subparsers.add_parser(
        "audit-pairs", help="print every pair for human inspection"
    )
    audit.add_argument("path", type=Path)

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate stored predictions against canonical tasks"
    )
    evaluate.add_argument("--tasks", required=True, type=Path)
    evaluate.add_argument("--predictions", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)

    infer = subparsers.add_parser("infer", help="run resumable local MLX inference")
    infer.add_argument("--config", required=True, type=Path)
    infer.add_argument("--tasks", required=True, type=Path)
    infer.add_argument("--output", required=True, type=Path)
    infer.add_argument("--limit", type=int, default=None)
    infer.add_argument("--stratified-smoke", action="store_true")
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
        if args.command == "generate-productivity":
            manifest = build_productivity_dataset(args.config, args.output)
            print(
                f"generated {manifest['pair_count']} pairs / "
                f"{manifest['task_count']} tasks in {args.output}"
            )
            return
        if args.command == "generate-dataset":
            manifest = build_full_dataset(args.config, args.output)
            print(
                f"generated {manifest['pair_count']} pairs / "
                f"{manifest['task_count']} tasks in {args.output}"
            )
            return
        if args.command == "audit-pairs":
            print(audit_pairs(load_pairs(args.path)))
            return
        if args.command == "evaluate":
            metrics = evaluate_files(args.tasks, args.predictions, args.output)
            print(metrics.model_dump_json(indent=2))
            return
        if args.command == "infer":
            inference_config = load_inference_config(args.config)
            inference_tasks = load_tasks(args.tasks)
            if args.stratified_smoke:
                inference_tasks = select_stratified_smoke(inference_tasks)
            predictions = run_inference(
                inference_tasks,
                MlxBackend(inference_config),
                args.output,
                limit=args.limit,
            )
            write_run_manifest(inference_config, inference_tasks, args.output)
            print(f"stored {len(predictions)} predictions in {args.output}")
            return
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    if args.command is None:
        parser.print_help()
