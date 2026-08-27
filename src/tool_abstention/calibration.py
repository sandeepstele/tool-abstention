# ruff: noqa: E501
"""Blinded human-annotation packets and calibration-label validation."""

import csv
import json
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from tool_abstention.records import EvaluationRecord, PredictionRecord, TaskRecord
from tool_abstention.taxonomy import DecisionClass
from tool_abstention.util.hashing import canonical_json_bytes, sha256_object
from tool_abstention.util.jsonl import write_jsonl

ANNOTATION_FIELDS = (
    "audit_id",
    "predicted_behavior",
    "semantic_correctness",
    "format_acceptable",
    "notes",
)

INSTRUCTIONS = """# Human calibration round 1

Open `annotate.html` in a browser. For each item, judge the model response using
only the request, environment, and visible tools shown on the page.

1. Choose the behavior the response actually performed.
2. Decide whether that behavior was semantically correct.
3. Decide whether its output format was acceptable.
4. Use `UNCLEAR` or `UNCERTAIN` instead of guessing.

Meaning matters more than exact wording. A correct fact in a complete sentence can
be semantically correct even if a protocol wanted a shorter answer. A refusal is
semantically correct when the requested capability is unavailable. A tool call on
an already-completed or underspecified task is not correct.

Progress is stored in the browser. When all 60 items are complete, click **Download
completed CSV** and return `annotations.completed.csv`. Do not edit `mapping.jsonl`.
"""


class AnnotatedBehavior(StrEnum):
    CALL = "CALL"
    ANSWER = "ANSWER"
    CLARIFY = "CLARIFY"
    REFUSE = "REFUSE"
    NOOP = "NOOP"
    UNCLEAR = "UNCLEAR"


class TernaryJudgment(StrEnum):
    YES = "YES"
    NO = "NO"
    UNCERTAIN = "UNCERTAIN"


class BinaryJudgment(StrEnum):
    YES = "YES"
    NO = "NO"


class HumanAnnotation(BaseModel):
    """One complete, strict human judgment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    audit_id: str
    predicted_behavior: AnnotatedBehavior
    semantic_correctness: TernaryJudgment
    format_acceptable: BinaryJudgment
    notes: str = ""


def _select_balanced(tasks: list[TaskRecord], per_cell: int) -> list[TaskRecord]:
    cells: dict[tuple[str, DecisionClass], list[TaskRecord]] = {}
    for task in tasks:
        cells.setdefault((task.domain, task.label), []).append(task)
    selected: list[TaskRecord] = []
    for cell in sorted(cells, key=lambda item: (item[0], item[1].value)):
        ranked = sorted(cells[cell], key=lambda task: sha256_object(task.id))
        if len(ranked) < per_cell:
            raise ValueError(f"not enough tasks for calibration cell {cell}")
        selected.extend(ranked[:per_cell])
    return sorted(selected, key=lambda task: sha256_object(task.id))


def _tool_summary(task: TaskRecord) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in task.tools
    ]


def _write_csv(path: Path, audit_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        for audit_id in audit_ids:
            writer.writerow({"audit_id": audit_id})


def _annotation_html(items: list[dict[str, Any]]) -> str:
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tool Abstention Calibration</title><style>
body{{font:16px system-ui;max-width:980px;margin:2rem auto;padding:0 1rem;color:#17202a;background:#f7f9fb}}
.card{{background:white;border:1px solid #ccd6dd;border-radius:12px;padding:1.2rem;margin:1rem 0}}
pre{{white-space:pre-wrap;background:#f1f4f6;padding:.8rem;border-radius:8px}} label{{margin-right:1rem}}
fieldset{{margin:.8rem 0;border:0;padding:0}} button{{padding:.7rem 1rem;margin-right:.5rem}}
.progress{{position:sticky;top:0;background:#17202a;color:white;padding:.8rem;border-radius:8px}}
</style></head><body><h1>Blinded evaluator calibration</h1>
<p>Judge meaning, not exact wording. Do not infer the hidden gold label. Your work is saved in this browser.</p>
<div class="progress" id="progress"></div><div id="root"></div>
<button id="prev">Previous</button><button id="next">Next</button><button id="download">Download completed CSV</button>
<script id="items" type="application/json">{data}</script><script>
const items=JSON.parse(document.getElementById('items').textContent), key='tool-abstention-calibration-v1';
let answers=JSON.parse(localStorage.getItem(key)||'{{}}'), index=0;
const choices=(name,values)=>values.map(v=>`<label><input type="radio" name="${{name}}" value="${{v}}"> ${{v}}</label>`).join('');
function render(){{const x=items[index],a=answers[x.audit_id]||{{}};document.getElementById('progress').textContent=`Item ${{index+1}} / ${{items.length}} — completed ${{Object.keys(answers).filter(k=>answers[k].predicted_behavior&&answers[k].semantic_correctness&&answers[k].format_acceptable).length}}`;
document.getElementById('root').innerHTML=`<div class="card"><h2>${{x.audit_id}}</h2><h3>Request</h3><pre>${{esc(x.query)}}</pre><h3>Environment</h3><pre>${{esc(JSON.stringify(x.environment,null,2))}}</pre><h3>Visible tools</h3><pre>${{esc(JSON.stringify(x.tools,null,2))}}</pre><h3>Model response</h3><pre>${{esc(x.response)}}</pre>
<fieldset><b>What behavior did the response perform?</b><br>${{choices('predicted_behavior',['CALL','ANSWER','CLARIFY','REFUSE','NOOP','UNCLEAR'])}}</fieldset>
<fieldset><b>Was that behavior semantically correct for the request, state, and tools?</b><br>${{choices('semantic_correctness',['YES','NO','UNCERTAIN'])}}</fieldset>
<fieldset><b>Was the output format acceptable?</b><br>${{choices('format_acceptable',['YES','NO'])}}</fieldset>
<label>Optional notes<br><textarea id="notes" rows="3" style="width:100%">${{esc(a.notes||'')}}</textarea></label></div>`;
for(const [k,v] of Object.entries(a)){{const el=document.querySelector(`input[name="${{k}}"] [value="${{v}}"]`)||document.querySelector(`input[name="${{k}}"][value="${{v}}"]`);if(el)el.checked=true;}}
document.querySelectorAll('input').forEach(el=>el.onchange=save);document.getElementById('notes').oninput=save;}}
function esc(s){{return String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function save(){{const id=items[index].audit_id,a=answers[id]||{{}};document.querySelectorAll('input:checked').forEach(el=>a[el.name]=el.value);a.notes=document.getElementById('notes').value;answers[id]=a;localStorage.setItem(key,JSON.stringify(answers));document.getElementById('progress').textContent=`Item ${{index+1}} / ${{items.length}} — completed ${{Object.values(answers).filter(a=>a.predicted_behavior&&a.semantic_correctness&&a.format_acceptable).length}}`;}}
document.getElementById('prev').onclick=()=>{{save();index=Math.max(0,index-1);render();}};document.getElementById('next').onclick=()=>{{save();index=Math.min(items.length-1,index+1);render();}};
document.getElementById('download').onclick=()=>{{save();const fields={json.dumps(ANNOTATION_FIELDS)},q=s=>`"${{String(s??'').replaceAll('"','""')}}"`;let csv=fields.join(',')+'\\n';for(const x of items){{const a=answers[x.audit_id]||{{}};csv+=fields.map(f=>q(f==='audit_id'?x.audit_id:a[f])).join(',')+'\\n';}}const blob=new Blob([csv],{{type:'text/csv'}}),u=URL.createObjectURL(blob),link=document.createElement('a');link.href=u;link.download='annotations.completed.csv';link.click();URL.revokeObjectURL(u);}};render();
</script></body></html>"""


def export_calibration_packet(
    tasks: list[TaskRecord],
    predictions: list[PredictionRecord],
    output: Path,
    *,
    per_cell: int = 4,
) -> dict[str, Any]:
    """Export a deterministic, validation-only, evaluator-blinded packet."""
    if per_cell < 1:
        raise ValueError("per_cell must be positive")
    if any(task.split.value != "validation" for task in tasks):
        raise ValueError("calibration export accepts validation tasks only")
    prediction_by_id = {prediction.task_id: prediction for prediction in predictions}
    if len(prediction_by_id) != len(predictions):
        raise ValueError("prediction task ids must be unique")
    selected = _select_balanced(tasks, per_cell)
    if any(task.id not in prediction_by_id for task in selected):
        raise ValueError("predictions are missing selected calibration tasks")
    output.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    for number, task in enumerate(selected, start=1):
        audit_id = f"audit-{number:03d}"
        prediction = prediction_by_id[task.id]
        items.append(
            {
                "audit_id": audit_id,
                "query": task.query,
                "environment": task.environment,
                "tools": _tool_summary(task),
                "response": prediction.raw_text,
            }
        )
        mappings.append({"audit_id": audit_id, "task_id": task.id})
    _write_csv(output / "annotations.blank.csv", [item["audit_id"] for item in items])
    write_jsonl(output / "mapping.jsonl", mappings)
    (output / "annotate.html").write_text(_annotation_html(items), encoding="utf-8")
    (output / "README.md").write_text(INSTRUCTIONS, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "item_count": len(items),
        "per_domain_class_cell": per_cell,
        "selection_hash": sha256_object([mapping["task_id"] for mapping in mappings]),
        "predictions_hash": sha256_object(
            [
                prediction_by_id[mapping["task_id"]].model_dump(mode="json")
                for mapping in mappings
            ]
        ),
    }
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def load_annotations(path: Path, expected_ids: set[str]) -> list[HumanAnnotation]:
    """Load and strictly validate a completed annotation CSV."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != ANNOTATION_FIELDS:
            raise ValueError(f"annotation columns must be {ANNOTATION_FIELDS}")
        rows = [HumanAnnotation.model_validate(row) for row in reader]
    ids = [row.audit_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("annotation audit ids must be unique")
    if set(ids) != expected_ids:
        raise ValueError("annotation audit ids do not match the calibration packet")
    return rows


def annotation_summary(rows: list[HumanAnnotation]) -> dict[str, Any]:
    """Summarize one complete annotation set without exposing gold labels."""
    return {
        "item_count": len(rows),
        "predicted_behavior": dict(Counter(row.predicted_behavior for row in rows)),
        "semantic_correctness": dict(Counter(row.semantic_correctness for row in rows)),
        "format_acceptable": dict(Counter(row.format_acceptable for row in rows)),
    }


def _cohen_kappa(first: list[str], second: list[str]) -> float | None:
    observed = sum(
        left == right for left, right in zip(first, second, strict=True)
    ) / len(first)
    first_counts = Counter(first)
    second_counts = Counter(second)
    expected = (
        sum(
            first_counts[value] * second_counts[value]
            for value in set(first_counts) | set(second_counts)
        )
        / len(first) ** 2
    )
    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def agreement_summary(
    first: list[HumanAnnotation], second: list[HumanAnnotation]
) -> dict[str, Any]:
    """Compute exact agreement and Cohen's kappa for two independent annotators."""
    first_by_id = {row.audit_id: row for row in first}
    second_by_id = {row.audit_id: row for row in second}
    if set(first_by_id) != set(second_by_id):
        raise ValueError("annotators must label the same audit ids")
    ids = sorted(first_by_id)
    result: dict[str, Any] = {"item_count": len(ids), "fields": {}}
    for field in (
        "predicted_behavior",
        "semantic_correctness",
        "format_acceptable",
    ):
        left = [str(getattr(first_by_id[audit_id], field)) for audit_id in ids]
        right = [str(getattr(second_by_id[audit_id], field)) for audit_id in ids]
        exact = sum(a == b for a, b in zip(left, right, strict=True)) / len(ids)
        result["fields"][field] = {
            "exact_agreement": exact,
            "cohen_kappa": _cohen_kappa(left, right),
        }
    return result


def evaluator_agreement_summary(
    annotations: list[HumanAnnotation],
    mapping: dict[str, str],
    evaluations: list[EvaluationRecord],
) -> dict[str, Any]:
    """Compare calibrated deterministic judgments with verified annotations."""
    annotation_by_id = {row.audit_id: row for row in annotations}
    evaluation_by_id = {row.task_id: row for row in evaluations}
    if set(annotation_by_id) != set(mapping):
        raise ValueError("annotations and mapping must contain the same audit ids")
    if not set(mapping.values()).issubset(evaluation_by_id):
        raise ValueError("evaluations are missing mapped calibration tasks")
    disagreements: list[dict[str, str]] = []
    behavior_matches = 0
    semantic_matches = 0
    protocol_matches = 0
    for audit_id in sorted(mapping):
        annotation = annotation_by_id[audit_id]
        evaluation = evaluation_by_id[mapping[audit_id]]
        predicted = (
            evaluation.predicted_class.value
            if evaluation.predicted_class is not None
            else "UNCLEAR"
        )
        behavior_match = predicted == annotation.predicted_behavior.value
        semantic_match = evaluation.correct == (
            annotation.semantic_correctness is TernaryJudgment.YES
        )
        protocol_match = evaluation.protocol_correct == (
            annotation.format_acceptable is BinaryJudgment.YES
        )
        behavior_matches += behavior_match
        semantic_matches += semantic_match
        protocol_matches += protocol_match
        if not (behavior_match and semantic_match and protocol_match):
            disagreements.append(
                {
                    "audit_id": audit_id,
                    "task_id": mapping[audit_id],
                    "axes": ",".join(
                        axis
                        for axis, matched in (
                            ("behavior", behavior_match),
                            ("semantic", semantic_match),
                            ("protocol", protocol_match),
                        )
                        if not matched
                    ),
                }
            )
    count = len(annotations)
    return {
        "item_count": count,
        "behavior_agreement": behavior_matches / count,
        "semantic_agreement": semantic_matches / count,
        "protocol_agreement": protocol_matches / count,
        "disagreements": disagreements,
    }
