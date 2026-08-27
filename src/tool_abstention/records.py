"""Strict canonical records for tasks, predictions, and evaluation."""

import math
from typing import Annotated, Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from tool_abstention.taxonomy import (
    PERTURBATION_LABEL,
    DatasetSplit,
    DecisionClass,
    PerturbationType,
    TaskVariant,
)

type JsonValue = (
    bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
)

Slug = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")]
NonEmptyText = Annotated[str, Field(min_length=1)]


def _ensure_finite(value: JsonValue, *, location: str = "value") -> JsonValue:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location} contains a non-finite number")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _ensure_finite(item, location=f"{location}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _ensure_finite(item, location=f"{location}.{key}")
    return value


class ContractModel(BaseModel):
    """Common strict, immutable behavior for public artifact models."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolDefinition(ContractModel):
    """An OpenAI-style function definition with Draft 2020-12 parameters."""

    name: Slug
    description: NonEmptyText
    parameters: dict[str, JsonValue]

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        _ensure_finite(value, location="parameters")
        try:
            Draft202012Validator.check_schema(value)
        except SchemaError as error:
            raise ValueError(
                f"invalid Draft 2020-12 parameter schema: {error.message}"
            ) from error
        if value.get("type") != "object":
            raise ValueError("tool parameter schema root type must be 'object'")
        return value


class ParsedToolCall(ContractModel):
    """A model-independent canonical function call."""

    name: Slug
    arguments: dict[str, JsonValue]

    @field_validator("arguments")
    @classmethod
    def finite_arguments(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        _ensure_finite(value, location="arguments")
        return value


class ExactAnswerValidator(ContractModel):
    kind: Literal["exact"] = "exact"
    value: NonEmptyText
    case_sensitive: bool = True


class NormalizedTextAnswerValidator(ContractModel):
    kind: Literal["normalized_text"] = "normalized_text"
    value: NonEmptyText


class NumericAnswerValidator(ContractModel):
    kind: Literal["numeric"] = "numeric"
    value: float
    absolute_tolerance: float = Field(ge=0)
    unit: str | None = None

    @field_validator("value", "absolute_tolerance")
    @classmethod
    def finite_number(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("numeric validator values must be finite")
        return value


class SetAnswerValidator(ContractModel):
    kind: Literal["set"] = "set"
    values: tuple[NonEmptyText, ...] = Field(min_length=1)
    case_sensitive: bool = False

    @field_validator("values")
    @classmethod
    def unique_values(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.casefold() for item in value)
        if len(normalized) != len(set(normalized)):
            raise ValueError("set answer values must be unique ignoring case")
        return value


class DomainAnswerValidator(ContractModel):
    kind: Literal["domain"] = "domain"
    validator_id: Slug
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("parameters")
    @classmethod
    def finite_parameters(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        _ensure_finite(value, location="validator parameters")
        return value


AnswerValidator = Annotated[
    ExactAnswerValidator
    | NormalizedTextAnswerValidator
    | NumericAnswerValidator
    | SetAnswerValidator
    | DomainAnswerValidator,
    Field(discriminator="kind"),
]


class CallExpected(ContractModel):
    kind: Literal[DecisionClass.CALL] = DecisionClass.CALL
    tool_name: Slug
    arguments: dict[str, JsonValue]
    expected_result: JsonValue

    @field_validator("arguments")
    @classmethod
    def finite_arguments(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        _ensure_finite(value, location="arguments")
        return value

    @field_validator("expected_result")
    @classmethod
    def finite_result(cls, value: JsonValue) -> JsonValue:
        return _ensure_finite(value, location="expected_result")


class AnswerExpected(ContractModel):
    kind: Literal[DecisionClass.ANSWER] = DecisionClass.ANSWER
    validator: AnswerValidator


class ClarifyExpected(ContractModel):
    kind: Literal[DecisionClass.CLARIFY] = DecisionClass.CLARIFY
    missing_slots: tuple[Slug, ...] = Field(min_length=1)

    @field_validator("missing_slots")
    @classmethod
    def unique_slots(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("missing slots must be unique")
        return value


class RefuseExpected(ContractModel):
    kind: Literal[DecisionClass.REFUSE] = DecisionClass.REFUSE
    unavailable_capability: Slug
    reason: Literal["missing_tool", "unsupported_capability"]


class NoopExpected(ContractModel):
    kind: Literal[DecisionClass.NOOP] = DecisionClass.NOOP
    state_assertion: NonEmptyText
    allowed_markers: tuple[NonEmptyText, ...] = Field(min_length=1)

    @field_validator("allowed_markers")
    @classmethod
    def unique_markers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(marker.casefold() for marker in value)
        if len(normalized) != len(set(normalized)):
            raise ValueError("allowed markers must be unique ignoring case")
        return value


ExpectedBehavior = Annotated[
    CallExpected | AnswerExpected | ClarifyExpected | RefuseExpected | NoopExpected,
    Field(discriminator="kind"),
]


class TaskRecord(ContractModel):
    """One side of a controlled act/abstain pair."""

    id: Slug
    pair_id: Slug
    domain: Slug
    split: DatasetSplit
    variant: TaskVariant
    generator_version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]
    query: NonEmptyText
    tools: tuple[ToolDefinition, ...]
    environment: dict[str, JsonValue] = Field(default_factory=dict)
    label: DecisionClass
    perturbation: PerturbationType | None
    expected: ExpectedBehavior

    @field_validator("tools")
    @classmethod
    def unique_tools(
        cls, value: tuple[ToolDefinition, ...]
    ) -> tuple[ToolDefinition, ...]:
        names = [tool.name for tool in value]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique")
        return value

    @field_validator("environment")
    @classmethod
    def finite_environment(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        _ensure_finite(value, location="environment")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> "TaskRecord":
        expected_id = f"{self.pair_id}-{self.variant.value}"
        if self.id != expected_id:
            raise ValueError(f"task id must be '{expected_id}'")
        if self.expected.kind != self.label:
            raise ValueError("expected behavior kind must match task label")
        if self.variant is TaskVariant.ACT:
            if self.label is not DecisionClass.CALL:
                raise ValueError("act tasks must use the CALL label")
            if self.perturbation is not None:
                raise ValueError("act tasks cannot declare a perturbation")
        else:
            if self.label is DecisionClass.CALL:
                raise ValueError("abstain tasks cannot use the CALL label")
            if self.perturbation is None:
                raise ValueError("abstain tasks must declare a perturbation")
            if PERTURBATION_LABEL[self.perturbation] is not self.label:
                raise ValueError("perturbation does not produce the task label")
        if isinstance(self.expected, CallExpected):
            tools = {tool.name: tool for tool in self.tools}
            tool = tools.get(self.expected.tool_name)
            if tool is None:
                raise ValueError("expected tool must exist in the visible inventory")
            errors = sorted(
                Draft202012Validator(tool.parameters).iter_errors(
                    self.expected.arguments
                ),
                key=lambda error: list(error.absolute_path),
            )
            if errors:
                raise ValueError(
                    f"expected tool arguments are invalid: {errors[0].message}"
                )
        return self


class TaskPair(ContractModel):
    """A validated CALL task and its controlled abstention twin."""

    pair_id: Slug
    act: TaskRecord
    abstain: TaskRecord

    @model_validator(mode="after")
    def validate_pair(self) -> "TaskPair":
        if self.act.pair_id != self.pair_id or self.abstain.pair_id != self.pair_id:
            raise ValueError("pair member pair_id values must match the pair")
        if self.act.variant is not TaskVariant.ACT:
            raise ValueError("pair act member must use the act variant")
        if self.abstain.variant is not TaskVariant.ABSTAIN:
            raise ValueError("pair abstain member must use the abstain variant")
        for field in ("domain", "split", "generator_version"):
            if getattr(self.act, field) != getattr(self.abstain, field):
                raise ValueError(f"pair members must share {field}")
        return self


class PredictionRecord(ContractModel):
    """Raw inference output retained independently of evaluation."""

    task_id: Slug
    raw_text: str
    tool_call: ParsedToolCall | None = None
    latency_ms: float = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    peak_memory_gb: float | None = Field(default=None, ge=0)
    inference_error: NonEmptyText | None = None

    @field_validator("latency_ms", "peak_memory_gb")
    @classmethod
    def finite_latency(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if not math.isfinite(value):
            raise ValueError("timing and memory values must be finite")
        return value

    @model_validator(mode="after")
    def validate_error_state(self) -> "PredictionRecord":
        if self.inference_error is not None and (
            self.raw_text or self.tool_call is not None
        ):
            raise ValueError("failed predictions cannot contain output or a tool call")
        return self


class EvaluationRecord(ContractModel):
    """Deterministic per-example classification result."""

    task_id: Slug
    predicted_class: DecisionClass | None
    correct: bool
    reason_code: Slug
    parsed_tool_call: ParsedToolCall | None = None

    @model_validator(mode="after")
    def validate_unclassified_result(self) -> "EvaluationRecord":
        if self.predicted_class is None and self.correct:
            raise ValueError("an unclassified prediction cannot be correct")
        return self
