"""Judge-free aggregate metrics for five-class paired evaluation."""

from collections import defaultdict

from pydantic import Field

from tool_abstention.records import ContractModel, EvaluationRecord, TaskRecord
from tool_abstention.taxonomy import DecisionClass, TaskVariant

EVALUATOR_VERSION = "2.0.0"


class ClassMetrics(ContractModel):
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    support: int = Field(ge=0)


class MetricsSummary(ContractModel):
    evaluator_version: str
    task_count: int = Field(gt=0)
    pair_count: int = Field(gt=0)
    accuracy: float = Field(ge=0, le=1)
    behavior_accuracy: float = Field(ge=0, le=1)
    semantic_accuracy: float = Field(ge=0, le=1)
    protocol_compliance_rate: float = Field(ge=0, le=1)
    paired_accuracy: float = Field(ge=0, le=1)
    macro_f1: float = Field(ge=0, le=1)
    act_accuracy: float = Field(ge=0, le=1)
    abstention_accuracy: float = Field(ge=0, le=1)
    tool_hallucination_rate: float = Field(ge=0, le=1)
    per_class: dict[DecisionClass, ClassMetrics]


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(
    tasks: list[TaskRecord], evaluations: list[EvaluationRecord]
) -> MetricsSummary:
    """Compute all core metrics from aligned task and evaluation records."""
    if not tasks:
        raise ValueError("cannot compute metrics for an empty task set")
    task_by_id = {task.id: task for task in tasks}
    evaluation_by_id = {evaluation.task_id: evaluation for evaluation in evaluations}
    if len(task_by_id) != len(tasks) or len(evaluation_by_id) != len(evaluations):
        raise ValueError("task and evaluation ids must be unique")
    if set(task_by_id) != set(evaluation_by_id):
        raise ValueError("task and evaluation ids must match exactly")

    pairs: dict[str, list[bool]] = defaultdict(list)
    true_positive = dict.fromkeys(DecisionClass, 0)
    false_positive = dict.fromkeys(DecisionClass, 0)
    false_negative = dict.fromkeys(DecisionClass, 0)
    support = dict.fromkeys(DecisionClass, 0)
    correct_count = 0
    behavior_correct_count = 0
    semantic_correct_count = 0
    protocol_correct_count = 0
    act_correct = 0
    act_count = 0
    abstain_correct = 0
    abstain_count = 0
    hallucinations = 0

    for task_id, task in task_by_id.items():
        evaluation = evaluation_by_id[task_id]
        correct_count += int(evaluation.correct)
        behavior_correct_count += int(evaluation.behavior_correct)
        semantic_correct_count += int(evaluation.semantic_correct)
        protocol_correct_count += int(evaluation.protocol_correct)
        pairs[task.pair_id].append(evaluation.correct)
        support[task.label] += 1
        if task.variant is TaskVariant.ACT:
            act_count += 1
            act_correct += int(evaluation.correct)
        else:
            abstain_count += 1
            abstain_correct += int(evaluation.correct)
            hallucinations += int(evaluation.predicted_class is DecisionClass.CALL)
        predicted = evaluation.predicted_class
        if predicted is task.label:
            true_positive[task.label] += 1
        else:
            false_negative[task.label] += 1
            if predicted is not None:
                false_positive[predicted] += 1

    if not act_count or not abstain_count:
        raise ValueError("metrics require both act and abstain tasks")
    if any(len(results) != 2 for results in pairs.values()):
        raise ValueError("each pair must contain exactly two evaluated tasks")

    per_class: dict[DecisionClass, ClassMetrics] = {}
    for label in DecisionClass:
        precision = _safe_ratio(
            true_positive[label], true_positive[label] + false_positive[label]
        )
        recall = _safe_ratio(
            true_positive[label], true_positive[label] + false_negative[label]
        )
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        per_class[label] = ClassMetrics(
            precision=precision, recall=recall, f1=f1, support=support[label]
        )
    return MetricsSummary(
        evaluator_version=EVALUATOR_VERSION,
        task_count=len(tasks),
        pair_count=len(pairs),
        accuracy=correct_count / len(tasks),
        behavior_accuracy=behavior_correct_count / len(tasks),
        semantic_accuracy=semantic_correct_count / len(tasks),
        protocol_compliance_rate=protocol_correct_count / len(tasks),
        paired_accuracy=sum(all(results) for results in pairs.values()) / len(pairs),
        macro_f1=sum(metric.f1 for metric in per_class.values()) / len(per_class),
        act_accuracy=act_correct / act_count,
        abstention_accuracy=abstain_correct / abstain_count,
        tool_hallucination_rate=hallucinations / abstain_count,
        per_class=per_class,
    )
