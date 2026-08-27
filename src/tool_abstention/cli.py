"""Command-line interface for project utilities."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from tool_abstention import __version__
from tool_abstention.calibration import (
    agreement_summary,
    annotation_summary,
    evaluator_agreement_summary,
    export_calibration_packet,
    load_annotations,
)
from tool_abstention.config import ProjectConfig, load_yaml_config
from tool_abstention.dataset import build_full_dataset
from tool_abstention.external import (
    ExternalDecisionRecord,
    evaluate_external_records,
    fetch_external,
    prepare_external,
)
from tool_abstention.harness import evaluate_files
from tool_abstention.inference import (
    SYSTEM_PROMPT,
    MlxBackend,
    PromptExample,
    PromptVariant,
    load_inference_config,
    load_tasks,
    run_inference,
    run_prompt_inference,
    select_stratified_smoke,
    write_run_manifest,
)
from tool_abstention.malformed import analyze_malformed_calls
from tool_abstention.productivity import (
    audit_pairs,
    build_productivity_dataset,
    load_pairs,
)
from tool_abstention.records import EvaluationRecord, PredictionRecord
from tool_abstention.schemas import SchemaKind, export_schemas, validate_record
from tool_abstention.sft import build_sft_dataset, run_sft_training
from tool_abstention.util.hashing import (
    canonical_json_bytes,
    sha256_file,
    sha256_object,
)
from tool_abstention.util.jsonl import read_jsonl, write_jsonl


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

    sft_data = subparsers.add_parser(
        "build-sft", help="export internal train/validation records for MLX SFT"
    )
    sft_data.add_argument("--internal", required=True, type=Path)
    sft_data.add_argument("--output", required=True, type=Path)

    sft_train = subparsers.add_parser(
        "train-sft", help="train a pinned MLX LoRA adapter"
    )
    sft_train.add_argument("--config", required=True, type=Path)
    sft_train.add_argument("--data", required=True, type=Path)
    sft_train.add_argument("--output", required=True, type=Path)

    fetch = subparsers.add_parser(
        "fetch-external", help="fetch pinned external benchmark snapshots"
    )
    fetch.add_argument("--config", required=True, type=Path)
    fetch.add_argument("--output", required=True, type=Path)

    prepare = subparsers.add_parser(
        "prepare-external", help="prepare and leakage-check external benchmarks"
    )
    prepare.add_argument("--config", required=True, type=Path)
    prepare.add_argument("--raw", required=True, type=Path)
    prepare.add_argument("--internal", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)

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

    calibration = subparsers.add_parser(
        "export-calibration", help="export a blinded human calibration packet"
    )
    calibration.add_argument("--tasks", required=True, type=Path)
    calibration.add_argument("--predictions", required=True, type=Path)
    calibration.add_argument("--output", required=True, type=Path)
    calibration.add_argument("--per-cell", type=int, default=4)

    labels = subparsers.add_parser(
        "validate-calibration", help="validate completed human calibration labels"
    )
    labels.add_argument("--annotations", required=True, type=Path)
    labels.add_argument("--mapping", required=True, type=Path)

    agreement = subparsers.add_parser(
        "calibration-agreement", help="compare two independent annotation files"
    )
    agreement.add_argument("--first", required=True, type=Path)
    agreement.add_argument("--second", required=True, type=Path)
    agreement.add_argument("--mapping", required=True, type=Path)

    compare = subparsers.add_parser(
        "compare-calibration", help="compare verified labels with evaluator output"
    )
    compare.add_argument("--annotations", required=True, type=Path)
    compare.add_argument("--mapping", required=True, type=Path)
    compare.add_argument("--evaluations", required=True, type=Path)
    compare.add_argument("--output", type=Path, default=None)

    external_infer = subparsers.add_parser(
        "infer-external", help="run resumable MLX inference on external records"
    )
    external_infer.add_argument("--config", required=True, type=Path)
    external_infer.add_argument("--records", required=True, type=Path)
    external_infer.add_argument("--output", required=True, type=Path)
    external_infer.add_argument("--limit", type=int, default=None)
    external_infer.add_argument("--adapter-path", type=Path, default=None)

    external_eval = subparsers.add_parser(
        "evaluate-external", help="score stored external decision predictions"
    )
    external_eval.add_argument("--records", required=True, type=Path)
    external_eval.add_argument("--predictions", required=True, type=Path)
    external_eval.add_argument("--output", required=True, type=Path)

    malformed = subparsers.add_parser(
        "analyze-malformed", help="compare stored malformed external call attempts"
    )
    malformed.add_argument("--base", required=True, type=Path)
    malformed.add_argument("--sft", required=True, type=Path)
    malformed.add_argument("--output", required=True, type=Path)

    infer = subparsers.add_parser("infer", help="run resumable local MLX inference")
    infer.add_argument("--config", required=True, type=Path)
    infer.add_argument("--tasks", required=True, type=Path)
    infer.add_argument("--output", required=True, type=Path)
    infer.add_argument("--limit", type=int, default=None)
    infer.add_argument("--stratified-smoke", action="store_true")
    infer.add_argument("--prompt-variant", choices=tuple(PromptVariant), default=None)
    infer.add_argument("--adapter-path", type=Path, default=None)
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
        if args.command == "build-sft":
            manifest = build_sft_dataset(args.internal, args.output)
            print(json.dumps(manifest, indent=2))
            return
        if args.command == "train-sft":
            manifest = run_sft_training(args.config, args.data, args.output)
            print(json.dumps(manifest, indent=2))
            return
        if args.command == "fetch-external":
            fetched = fetch_external(args.config, args.output)
            print(json.dumps(fetched, indent=2))
            return
        if args.command == "prepare-external":
            manifest = prepare_external(
                args.config, args.raw, args.internal, args.output
            )
            print(json.dumps(manifest, indent=2))
            return
        if args.command == "audit-pairs":
            print(audit_pairs(load_pairs(args.path)))
            return
        if args.command == "evaluate":
            metrics = evaluate_files(args.tasks, args.predictions, args.output)
            print(metrics.model_dump_json(indent=2))
            return
        if args.command == "export-calibration":
            calibration_tasks = load_tasks(args.tasks)
            calibration_predictions = [
                PredictionRecord.model_validate(value)
                for value in read_jsonl(args.predictions)
            ]
            manifest = export_calibration_packet(
                calibration_tasks,
                calibration_predictions,
                args.output,
                per_cell=args.per_cell,
            )
            print(f"exported {manifest['item_count']} blinded items to {args.output}")
            return
        if args.command == "validate-calibration":
            expected_ids = {
                str(value["audit_id"]) for value in read_jsonl(args.mapping)
            }
            annotations = load_annotations(args.annotations, expected_ids)
            print(json.dumps(annotation_summary(annotations), indent=2))
            return
        if args.command == "calibration-agreement":
            expected_ids = {
                str(value["audit_id"]) for value in read_jsonl(args.mapping)
            }
            first = load_annotations(args.first, expected_ids)
            second = load_annotations(args.second, expected_ids)
            print(json.dumps(agreement_summary(first, second), indent=2))
            return
        if args.command == "compare-calibration":
            mapping = {
                str(value["audit_id"]): str(value["task_id"])
                for value in read_jsonl(args.mapping)
            }
            annotations = load_annotations(args.annotations, set(mapping))
            evaluations = [
                EvaluationRecord.model_validate(value)
                for value in read_jsonl(args.evaluations)
            ]
            comparison = evaluator_agreement_summary(annotations, mapping, evaluations)
            rendered = json.dumps(comparison, indent=2)
            if args.output is not None:
                args.output.write_text(rendered + "\n", encoding="utf-8")
            print(rendered)
            return
        if args.command == "infer-external":
            external_config = load_inference_config(args.config)
            if args.adapter_path is not None:
                external_config = external_config.model_copy(
                    update={"adapter_path": str(args.adapter_path)}
                )
            records = [
                ExternalDecisionRecord.model_validate(value)
                for value in read_jsonl(args.records)
            ]
            examples = [
                PromptExample(
                    id=record.id,
                    messages=(
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *(message.model_dump() for message in record.messages),
                    ),
                    tools=tuple(
                        {
                            "type": "function",
                            "function": function.model_dump(),
                        }
                        for function in record.functions
                    ),
                )
                for record in records
            ]
            predictions = run_prompt_inference(
                examples,
                MlxBackend(external_config),
                args.output,
                limit=args.limit,
            )
            selected = examples[: args.limit] if args.limit is not None else examples
            manifest = {
                "schema_version": 1,
                "model": external_config.model,
                "revision": external_config.revision,
                "adapter_hash": (
                    sha256_file(
                        Path(external_config.adapter_path) / "adapters.safetensors"
                    )
                    if external_config.adapter_path is not None
                    else None
                ),
                "prompt_variant": external_config.prompt_variant,
                "record_ids_hash": sha256_object([example.id for example in selected]),
                "records_hash": sha256_file(args.records),
                "predictions_hash": sha256_file(args.output),
            }
            args.output.with_name("run_manifest.json").write_bytes(
                canonical_json_bytes(manifest) + b"\n"
            )
            print(f"stored {len(predictions)} external predictions in {args.output}")
            return
        if args.command == "evaluate-external":
            records = [
                ExternalDecisionRecord.model_validate(value)
                for value in read_jsonl(args.records)
            ]
            predictions = [
                PredictionRecord.model_validate(value)
                for value in read_jsonl(args.predictions)
            ]
            external_evaluations, external_metrics = evaluate_external_records(
                records, predictions
            )
            write_jsonl(
                args.output / "evaluations.jsonl",
                [
                    evaluation.model_dump(mode="json")
                    for evaluation in external_evaluations
                ],
            )
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "metrics.json").write_bytes(
                canonical_json_bytes(external_metrics.model_dump(mode="json")) + b"\n"
            )
            print(external_metrics.model_dump_json(indent=2))
            return
        if args.command == "analyze-malformed":
            report = analyze_malformed_calls(args.base, args.sft, args.output)
            print(json.dumps(report, indent=2))
            return
        if args.command == "infer":
            inference_config = load_inference_config(args.config)
            if args.prompt_variant is not None:
                inference_config = inference_config.model_copy(
                    update={"prompt_variant": PromptVariant(args.prompt_variant)}
                )
            if args.adapter_path is not None:
                inference_config = inference_config.model_copy(
                    update={"adapter_path": str(args.adapter_path)}
                )
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
