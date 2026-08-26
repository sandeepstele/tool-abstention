"""Canonical taxonomy shared by data, training, and evaluation."""

from enum import StrEnum


class DecisionClass(StrEnum):
    """The single action class and four abstention classes."""

    CALL = "CALL"
    ANSWER = "ANSWER"
    CLARIFY = "CLARIFY"
    REFUSE = "REFUSE"
    NOOP = "NOOP"


class TaskVariant(StrEnum):
    """A member's role in a controlled task pair."""

    ACT = "act"
    ABSTAIN = "abstain"


class DatasetSplit(StrEnum):
    """Leakage-safe dataset partitions."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class PerturbationType(StrEnum):
    """The single controlled change from an act task to its abstain twin."""

    ANSWER_PROVIDED = "answer_provided"
    REQUIRED_ARGUMENT_REMOVED = "required_argument_removed"
    TOOL_REMOVED = "tool_removed"
    ALREADY_SATISFIED = "already_satisfied"


PERTURBATION_LABEL: dict[PerturbationType, DecisionClass] = {
    PerturbationType.ANSWER_PROVIDED: DecisionClass.ANSWER,
    PerturbationType.REQUIRED_ARGUMENT_REMOVED: DecisionClass.CLARIFY,
    PerturbationType.TOOL_REMOVED: DecisionClass.REFUSE,
    PerturbationType.ALREADY_SATISFIED: DecisionClass.NOOP,
}
