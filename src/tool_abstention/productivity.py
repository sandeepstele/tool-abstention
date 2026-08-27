"""Deterministic productivity tools and paired-task generation."""

import copy
import random
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field

from tool_abstention.config import load_yaml_config
from tool_abstention.records import (
    AnswerExpected,
    CallExpected,
    ClarifyExpected,
    ExactAnswerValidator,
    JsonValue,
    NoopExpected,
    RefuseExpected,
    TaskPair,
    TaskRecord,
    ToolDefinition,
)
from tool_abstention.taxonomy import (
    DatasetSplit,
    DecisionClass,
    PerturbationType,
    TaskVariant,
)
from tool_abstention.util.hashing import (
    canonical_json_bytes,
    sha256_file,
    sha256_object,
)
from tool_abstention.util.jsonl import read_jsonl, write_jsonl

type ToolExecutor = Callable[[dict[str, JsonValue], dict[str, JsonValue]], JsonValue]


class ProductivityConfig(BaseModel):
    """Seeded development-slice generation configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = Field(ge=0, le=2**32 - 1)
    pairs_per_class: int = Field(default=25, ge=1, le=25)
    generator_version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]
    split: DatasetSplit = DatasetSplit.TRAIN


CONTACTS: tuple[tuple[str, str], ...] = (
    ("Amina Khan", "amina.khan@example.com"),
    ("Ben Ortiz", "ben.ortiz@example.com"),
    ("Chloe Martin", "chloe.martin@example.com"),
    ("Dev Patel", "dev.patel@example.com"),
    ("Elena Rossi", "elena.rossi@example.com"),
    ("Farah Ali", "farah.ali@example.com"),
    ("Grace Lee", "grace.lee@example.com"),
    ("Hugo Silva", "hugo.silva@example.com"),
    ("Iris Chen", "iris.chen@example.com"),
    ("Jon Bell", "jon.bell@example.com"),
    ("Kira Young", "kira.young@example.com"),
    ("Leo Adams", "leo.adams@example.com"),
    ("Maya Singh", "maya.singh@example.com"),
    ("Noah Kim", "noah.kim@example.com"),
    ("Olivia Cruz", "olivia.cruz@example.com"),
    ("Pavel Novak", "pavel.novak@example.com"),
    ("Quinn Baker", "quinn.baker@example.com"),
    ("Rina Sato", "rina.sato@example.com"),
    ("Sam Wilson", "sam.wilson@example.com"),
    ("Tara Gupta", "tara.gupta@example.com"),
    ("Uma Shah", "uma.shah@example.com"),
    ("Victor Ng", "victor.ng@example.com"),
    ("Willa Jones", "willa.jones@example.com"),
    ("Xavier Reed", "xavier.reed@example.com"),
    ("Yara Haddad", "yara.haddad@example.com"),
)
MESSAGES: tuple[str, ...] = (
    "The review is ready.",
    "Please confirm the meeting.",
    "The draft was updated.",
    "Your report is approved.",
    "The deadline moved to Friday.",
    "Please check the latest notes.",
    "The customer replied.",
    "The deployment completed.",
    "Your access was restored.",
    "The invoice is ready.",
    "The agenda is attached.",
    "Please review the proposal.",
    "The ticket was resolved.",
    "Your request was received.",
    "The migration starts tonight.",
    "Please approve the budget.",
    "The interview is confirmed.",
    "The dashboard is refreshed.",
    "Your package has shipped.",
    "The contract is signed.",
    "Please update the checklist.",
    "The backup completed.",
    "Your reservation is confirmed.",
    "The workshop is postponed.",
    "Please read the incident report.",
)


def _tool_definitions() -> tuple[ToolDefinition, ...]:
    return (
        ToolDefinition(
            name="search_contacts",
            description="Find one contact by full name.",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string", "minLength": 1}},
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="create_event",
            description="Create a calendar event.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "minLength": 1},
                    "date": {"type": "string", "pattern": r"^2027-\d{2}-\d{2}$"},
                },
                "required": ["title", "date"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="close_ticket",
            description="Close an open support ticket.",
            parameters={
                "type": "object",
                "properties": {"ticket_id": {"type": "integer", "minimum": 1}},
                "required": ["ticket_id"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="send_email",
            description="Send an email to an explicit address.",
            parameters={
                "type": "object",
                "properties": {
                    "recipient": {"type": "string", "format": "email"},
                    "message": {"type": "string", "minLength": 1},
                },
                "required": ["recipient", "message"],
                "additionalProperties": False,
            },
        ),
    )


PRODUCTIVITY_TOOLS = _tool_definitions()


def _search_contacts(
    arguments: dict[str, JsonValue], state: dict[str, JsonValue]
) -> JsonValue:
    contacts = state.get("contacts", [])
    assert isinstance(contacts, list)
    for contact in contacts:
        if isinstance(contact, dict) and contact.get("name") == arguments["name"]:
            return copy.deepcopy(contact)
    return None


def _create_event(
    arguments: dict[str, JsonValue], state: dict[str, JsonValue]
) -> JsonValue:
    del state
    title = str(arguments["title"])
    slug = "-".join(title.casefold().split())
    return {
        "event_id": f"evt-{arguments['date']}-{slug}",
        "title": title,
        "date": arguments["date"],
        "status": "created",
    }


def _close_ticket(
    arguments: dict[str, JsonValue], state: dict[str, JsonValue]
) -> JsonValue:
    ticket_id = str(arguments["ticket_id"])
    tickets = state.get("tickets", {})
    assert isinstance(tickets, dict)
    status = tickets.get(ticket_id)
    if status != "open":
        return {"ticket_id": arguments["ticket_id"], "status": status, "changed": False}
    return {"ticket_id": arguments["ticket_id"], "status": "closed", "changed": True}


def _send_email(
    arguments: dict[str, JsonValue], state: dict[str, JsonValue]
) -> JsonValue:
    del state
    return {
        "recipient": arguments["recipient"],
        "message": arguments["message"],
        "status": "sent",
    }


EXECUTORS: dict[str, ToolExecutor] = {
    "search_contacts": _search_contacts,
    "create_event": _create_event,
    "close_ticket": _close_ticket,
    "send_email": _send_email,
}


def execute_tool(
    name: str, arguments: dict[str, JsonValue], state: dict[str, JsonValue]
) -> JsonValue:
    """Validate and execute one deterministic productivity tool."""
    definitions = {tool.name: tool for tool in PRODUCTIVITY_TOOLS}
    if name not in definitions:
        raise ValueError(f"unknown productivity tool: {name}")
    errors = list(
        Draft202012Validator(definitions[name].parameters).iter_errors(arguments)
    )
    if errors:
        raise ValueError(f"invalid arguments for {name}: {errors[0].message}")
    return EXECUTORS[name](copy.deepcopy(arguments), copy.deepcopy(state))


def _base_fields(
    pair_id: str, config: ProductivityConfig, variant: TaskVariant
) -> dict[str, Any]:
    return {
        "id": f"{pair_id}-{variant.value}",
        "pair_id": pair_id,
        "domain": "productivity",
        "split": config.split,
        "variant": variant,
        "generator_version": config.generator_version,
    }


def _contacts_state() -> dict[str, JsonValue]:
    return {"contacts": [{"name": name, "email": email} for name, email in CONTACTS]}


def _answer_pair(
    index: int, config: ProductivityConfig, contact_index: int
) -> TaskPair:
    name, email = CONTACTS[contact_index]
    pair_id = f"productivity-answer-{index:03d}"
    state = _contacts_state()
    result: dict[str, JsonValue] = {"name": name, "email": email}
    answer_query = f"The email address for {name} is {email}."
    answer_question = f"What is {name}'s email address?"
    act = TaskRecord(
        **_base_fields(pair_id, config, TaskVariant.ACT),
        query=f"Find the email address for {name}.",
        tools=PRODUCTIVITY_TOOLS,
        environment=state,
        label=DecisionClass.CALL,
        perturbation=None,
        expected=CallExpected(
            tool_name="search_contacts",
            arguments={"name": name},
            expected_result=result,
        ),
    )
    abstain = TaskRecord(
        **_base_fields(pair_id, config, TaskVariant.ABSTAIN),
        query=f"{answer_query} {answer_question}",
        tools=PRODUCTIVITY_TOOLS,
        environment=state,
        label=DecisionClass.ANSWER,
        perturbation=PerturbationType.ANSWER_PROVIDED,
        expected=AnswerExpected(validator=ExactAnswerValidator(value=email)),
    )
    return TaskPair(pair_id=pair_id, act=act, abstain=abstain)


def _clarify_pair(
    index: int, config: ProductivityConfig, contact_index: int
) -> TaskPair:
    _, email = CONTACTS[contact_index]
    message = MESSAGES[contact_index]
    pair_id = f"productivity-clarify-{index:03d}"
    state = _contacts_state()
    result: dict[str, JsonValue] = {
        "recipient": email,
        "message": message,
        "status": "sent",
    }
    act = TaskRecord(
        **_base_fields(pair_id, config, TaskVariant.ACT),
        query=f'Send "{message}" to {email}.',
        tools=PRODUCTIVITY_TOOLS,
        environment=state,
        label=DecisionClass.CALL,
        perturbation=None,
        expected=CallExpected(
            tool_name="send_email",
            arguments={"recipient": email, "message": message},
            expected_result=result,
        ),
    )
    abstain = TaskRecord(
        **_base_fields(pair_id, config, TaskVariant.ABSTAIN),
        query=f'Send "{message}"',
        tools=PRODUCTIVITY_TOOLS,
        environment=state,
        label=DecisionClass.CLARIFY,
        perturbation=PerturbationType.REQUIRED_ARGUMENT_REMOVED,
        expected=ClarifyExpected(missing_slots=("recipient",)),
    )
    return TaskPair(pair_id=pair_id, act=act, abstain=abstain)


def _refuse_pair(
    index: int, config: ProductivityConfig, contact_index: int
) -> TaskPair:
    del contact_index
    day = index + 1
    title = f"Planning session {index + 1}"
    date = f"2027-06-{day:02d}"
    pair_id = f"productivity-refuse-{index:03d}"
    result: dict[str, JsonValue] = {
        "event_id": f"evt-{date}-planning-session-{index + 1}",
        "title": title,
        "date": date,
        "status": "created",
    }
    query = f'Create a calendar event titled "{title}" on {date}.'
    act = TaskRecord(
        **_base_fields(pair_id, config, TaskVariant.ACT),
        query=query,
        tools=PRODUCTIVITY_TOOLS,
        environment={},
        label=DecisionClass.CALL,
        perturbation=None,
        expected=CallExpected(
            tool_name="create_event",
            arguments={"title": title, "date": date},
            expected_result=result,
        ),
    )
    abstain = TaskRecord(
        **_base_fields(pair_id, config, TaskVariant.ABSTAIN),
        query=query,
        tools=tuple(tool for tool in PRODUCTIVITY_TOOLS if tool.name != "create_event"),
        environment={},
        label=DecisionClass.REFUSE,
        perturbation=PerturbationType.TOOL_REMOVED,
        expected=RefuseExpected(
            unavailable_capability="create_event", reason="missing_tool"
        ),
    )
    return TaskPair(pair_id=pair_id, act=act, abstain=abstain)


def _noop_pair(index: int, config: ProductivityConfig, contact_index: int) -> TaskPair:
    del contact_index
    ticket_id = 1000 + index
    pair_id = f"productivity-noop-{index:03d}"
    query = f"Close support ticket {ticket_id}."
    act = TaskRecord(
        **_base_fields(pair_id, config, TaskVariant.ACT),
        query=query,
        tools=PRODUCTIVITY_TOOLS,
        environment={"tickets": {str(ticket_id): "open"}},
        label=DecisionClass.CALL,
        perturbation=None,
        expected=CallExpected(
            tool_name="close_ticket",
            arguments={"ticket_id": ticket_id},
            expected_result={
                "ticket_id": ticket_id,
                "status": "closed",
                "changed": True,
            },
        ),
    )
    abstain = TaskRecord(
        **_base_fields(pair_id, config, TaskVariant.ABSTAIN),
        query=query,
        tools=PRODUCTIVITY_TOOLS,
        environment={"tickets": {str(ticket_id): "closed"}},
        label=DecisionClass.NOOP,
        perturbation=PerturbationType.ALREADY_SATISFIED,
        expected=NoopExpected(
            state_assertion=f"Support ticket {ticket_id} is already closed.",
            allowed_markers=("already closed", "no action needed"),
        ),
    )
    return TaskPair(pair_id=pair_id, act=act, abstain=abstain)


PAIR_BUILDERS = (_answer_pair, _clarify_pair, _refuse_pair, _noop_pair)


def semantic_pair_changes(pair: TaskPair) -> frozenset[str]:
    """Return changed user-visible semantic dimensions for a pair."""
    changed: set[str] = set()
    for field in ("query", "tools", "environment"):
        if getattr(pair.act, field) != getattr(pair.abstain, field):
            changed.add(field)
    return frozenset(changed)


def validate_semantic_pair(pair: TaskPair) -> None:
    """Require exactly the semantic dimension declared by the perturbation."""
    expected_changes = {
        PerturbationType.ANSWER_PROVIDED: frozenset({"query"}),
        PerturbationType.REQUIRED_ARGUMENT_REMOVED: frozenset({"query"}),
        PerturbationType.TOOL_REMOVED: frozenset({"tools"}),
        PerturbationType.ALREADY_SATISFIED: frozenset({"environment"}),
    }
    assert pair.abstain.perturbation is not None
    actual = semantic_pair_changes(pair)
    expected = expected_changes[pair.abstain.perturbation]
    if actual != expected:
        raise ValueError(
            f"{pair.pair_id}: semantic changes {sorted(actual)} != {sorted(expected)}"
        )


def generate_productivity_pairs(config: ProductivityConfig) -> list[TaskPair]:
    """Generate deterministic, validated pairs for all abstention classes."""
    indices = list(range(len(CONTACTS)))
    random.Random(config.seed).shuffle(indices)
    selected = indices[: config.pairs_per_class]
    pairs = [
        builder(index, config, contact_index)
        for builder in PAIR_BUILDERS
        for index, contact_index in enumerate(selected)
    ]
    for pair in pairs:
        validate_semantic_pair(pair)
        expected = pair.act.expected
        assert isinstance(expected, CallExpected)
        actual_result = execute_tool(
            expected.tool_name, expected.arguments, pair.act.environment
        )
        if actual_result != expected.expected_result:
            raise ValueError(f"{pair.pair_id}: tool result does not match expectation")
    return pairs


def build_productivity_dataset(
    config_path: Path, output_directory: Path
) -> dict[str, Any]:
    """Build deterministic task JSONL and its provenance manifest."""
    config = load_yaml_config(config_path, ProductivityConfig)
    pairs = generate_productivity_pairs(config)
    task_path = output_directory / "tasks.jsonl"
    records = [
        task.model_dump(mode="json")
        for pair in pairs
        for task in (pair.act, pair.abstain)
    ]
    write_jsonl(task_path, records)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generator": "productivity",
        "generator_version": config.generator_version,
        "seed": config.seed,
        "pair_count": len(pairs),
        "task_count": len(records),
        "config_hash": sha256_object(config.model_dump(mode="json")),
        "artifacts": {"tasks.jsonl": {"content_hash": sha256_file(task_path)}},
    }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def load_pairs(path: Path) -> list[TaskPair]:
    """Load task JSONL and reconstruct pairs in stable first-seen order."""
    grouped: dict[str, dict[TaskVariant, TaskRecord]] = {}
    order: list[str] = []
    for value in read_jsonl(path):
        task = TaskRecord.model_validate(value)
        if task.pair_id not in grouped:
            grouped[task.pair_id] = {}
            order.append(task.pair_id)
        if task.variant in grouped[task.pair_id]:
            raise ValueError(f"duplicate {task.variant.value} task for {task.pair_id}")
        grouped[task.pair_id][task.variant] = task
    pairs: list[TaskPair] = []
    for pair_id in order:
        members = grouped[pair_id]
        if set(members) != {TaskVariant.ACT, TaskVariant.ABSTAIN}:
            raise ValueError(f"incomplete pair: {pair_id}")
        pair = TaskPair(
            pair_id=pair_id,
            act=members[TaskVariant.ACT],
            abstain=members[TaskVariant.ABSTAIN],
        )
        validate_semantic_pair(pair)
        pairs.append(pair)
    return pairs


def audit_pairs(pairs: list[TaskPair]) -> str:
    """Render every pair for human inspection."""
    sections: list[str] = []
    for pair in pairs:
        changes = ",".join(sorted(semantic_pair_changes(pair)))
        sections.extend(
            (
                f"PAIR {pair.pair_id} [{pair.abstain.label}] change={changes}",
                f"  ACT: {pair.act.query}",
                f"  ABSTAIN: {pair.abstain.query}",
            )
        )
    return "\n".join(sections)
