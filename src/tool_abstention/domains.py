"""Deterministic finance and weather/geo task domains."""

from collections.abc import Callable
from typing import Any

from tool_abstention.productivity import ProductivityConfig
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
from tool_abstention.taxonomy import DecisionClass, PerturbationType, TaskVariant

type PairBuilder = Callable[[int, ProductivityConfig], TaskPair]


def _fields(
    domain: str, pair_id: str, config: ProductivityConfig, variant: TaskVariant
) -> dict[str, Any]:
    return {
        "id": f"{pair_id}-{variant.value}",
        "pair_id": pair_id,
        "domain": domain,
        "split": config.split,
        "variant": variant,
        "generator_version": config.generator_version,
    }


FINANCE_TOOLS = (
    ToolDefinition(
        name="get_balance",
        description="Get an account balance.",
        parameters={
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        name="transfer",
        description="Transfer money between accounts.",
        parameters={
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "destination": {"type": "string"},
                "amount": {"type": "number", "exclusiveMinimum": 0},
            },
            "required": ["source", "destination", "amount"],
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        name="list_transactions",
        description="List recent account transactions.",
        parameters={
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        name="fx_rate",
        description="Get a currency conversion rate.",
        parameters={
            "type": "object",
            "properties": {
                "base": {"type": "string"},
                "quote": {"type": "string"},
            },
            "required": ["base", "quote"],
            "additionalProperties": False,
        },
    ),
)

WEATHER_TOOLS = (
    ToolDefinition(
        name="get_weather",
        description="Get current weather for a city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        name="get_time",
        description="Get local time for a city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        name="geocode",
        description="Get coordinates for an address.",
        parameters={
            "type": "object",
            "properties": {"address": {"type": "string"}},
            "required": ["address"],
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        name="distance",
        description="Compute distance between two cities.",
        parameters={
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
            },
            "required": ["origin", "destination"],
            "additionalProperties": False,
        },
    ),
)


def execute_domain_tool(
    domain: str, name: str, arguments: dict[str, JsonValue], state: dict[str, JsonValue]
) -> JsonValue:
    """Execute a deterministic finance or weather tool."""
    if domain == "finance":
        if name == "get_balance":
            account = str(arguments["account_id"])
            balances = state.get("balances", {})
            assert isinstance(balances, dict)
            return {"account_id": account, "balance": balances[account]}
        if name == "transfer":
            return {"status": "transferred", "amount": arguments["amount"]}
        if name == "list_transactions":
            return {"account_id": arguments["account_id"], "transactions": []}
        if name == "fx_rate":
            rate = 0.9 + (int(str(state["index"])) / 1000)
            return {
                "base": arguments["base"],
                "quote": arguments["quote"],
                "rate": rate,
            }
    if domain == "weather":
        if name == "get_weather":
            city = str(arguments["city"])
            weather = state.get("weather", {})
            assert isinstance(weather, dict)
            return {"city": city, "temperature_f": weather[city]}
        if name == "get_time":
            return {"city": arguments["city"], "time": state["time"]}
        if name == "geocode":
            coordinates = state["coordinates"]
            assert isinstance(coordinates, dict)
            return {"address": arguments["address"], **coordinates}
        if name == "distance":
            return {
                "origin": arguments["origin"],
                "destination": arguments["destination"],
                "miles": state["miles"],
            }
    raise ValueError(f"unknown {domain} tool: {name}")


CITIES = (
    "Austin",
    "Boston",
    "Chicago",
    "Denver",
    "Eugene",
    "Fresno",
    "Galveston",
    "Houston",
    "Ithaca",
    "Juneau",
    "Knoxville",
    "Lincoln",
    "Madison",
    "Nashville",
    "Oakland",
    "Phoenix",
    "Queens",
    "Raleigh",
    "Seattle",
    "Tampa",
    "Utica",
    "Ventura",
    "Wichita",
    "Yonkers",
    "Zion",
)
CURRENCIES = ("USD", "EUR", "GBP", "JPY", "CAD")


def _finance_pair(
    label: DecisionClass, index: int, config: ProductivityConfig
) -> TaskPair:
    account = f"acct-{1000 + index}"
    other = f"acct-{2000 + index}"
    amount = float(10 + index)
    suffix = label.value.casefold()
    pair_id = f"finance-{suffix}-{index:03d}"
    act_tools: tuple[ToolDefinition, ...]
    abstain_tools: tuple[ToolDefinition, ...]
    act_state: dict[str, JsonValue]
    abstain_state: dict[str, JsonValue]
    if label is DecisionClass.ANSWER:
        query = f"What is the balance of {account}?"
        expected = CallExpected(
            tool_name="get_balance",
            arguments={"account_id": account},
            expected_result={"account_id": account, "balance": 1000 + index},
        )
        abstain_query = (
            f"The balance of {account} is ${1000 + index}. What is its balance?"
        )
        abstain_expected: Any = AnswerExpected(
            validator=ExactAnswerValidator(value=f"${1000 + index}")
        )
        perturbation = PerturbationType.ANSWER_PROVIDED
        act_tools, abstain_tools = FINANCE_TOOLS, FINANCE_TOOLS
        act_state = abstain_state = {"balances": {account: 1000 + index}}
    elif label is DecisionClass.CLARIFY:
        query = f"Transfer ${amount:g} from {account} to {other}."
        expected = CallExpected(
            tool_name="transfer",
            arguments={"source": account, "destination": other, "amount": amount},
            expected_result={"status": "transferred", "amount": amount},
        )
        abstain_query = f"Transfer ${amount:g} to {other}."
        abstain_expected = ClarifyExpected(missing_slots=("source",))
        perturbation = PerturbationType.REQUIRED_ARGUMENT_REMOVED
        act_tools, abstain_tools = FINANCE_TOOLS, FINANCE_TOOLS
        act_state = abstain_state = {"balances": {account: 1000 + index}}
    elif label is DecisionClass.REFUSE:
        base = CURRENCIES[index // 5]
        quote = CURRENCIES[(index % 5 + index // 5 + 1) % len(CURRENCIES)]
        query = f"Get the current {base} to {quote} exchange rate."
        expected = CallExpected(
            tool_name="fx_rate",
            arguments={"base": base, "quote": quote},
            expected_result={"base": base, "quote": quote, "rate": 0.9 + index / 1000},
        )
        abstain_query = query
        abstain_expected = RefuseExpected(
            unavailable_capability="fx_rate", reason="missing_tool"
        )
        perturbation = PerturbationType.TOOL_REMOVED
        act_tools = FINANCE_TOOLS
        abstain_tools = tuple(tool for tool in FINANCE_TOOLS if tool.name != "fx_rate")
        act_state = abstain_state = {"index": index}
    else:
        query = f"Transfer ${amount:g} from {account} to {other}."
        expected = CallExpected(
            tool_name="transfer",
            arguments={"source": account, "destination": other, "amount": amount},
            expected_result={"status": "transferred", "amount": amount},
        )
        abstain_query = query
        abstain_expected = NoopExpected(
            state_assertion="The transfer is already completed.",
            allowed_markers=("already transferred", "no action needed"),
        )
        perturbation = PerturbationType.ALREADY_SATISFIED
        act_tools = abstain_tools = FINANCE_TOOLS
        act_state = {"transfer_status": "pending"}
        abstain_state = {"transfer_status": "completed"}
    act = TaskRecord(
        **_fields("finance", pair_id, config, TaskVariant.ACT),
        query=query,
        tools=act_tools,
        environment=act_state,
        label=DecisionClass.CALL,
        perturbation=None,
        expected=expected,
    )
    abstain = TaskRecord(
        **_fields("finance", pair_id, config, TaskVariant.ABSTAIN),
        query=abstain_query,
        tools=abstain_tools,
        environment=abstain_state,
        label=label,
        perturbation=perturbation,
        expected=abstain_expected,
    )
    return TaskPair(pair_id=pair_id, act=act, abstain=abstain)


def _weather_pair(
    label: DecisionClass, index: int, config: ProductivityConfig
) -> TaskPair:
    city = CITIES[index]
    other = CITIES[(index + 7) % len(CITIES)]
    suffix = label.value.casefold()
    pair_id = f"weather-{suffix}-{index:03d}"
    act_tools: tuple[ToolDefinition, ...]
    abstain_tools: tuple[ToolDefinition, ...]
    act_state: dict[str, JsonValue]
    abstain_state: dict[str, JsonValue]
    if label is DecisionClass.ANSWER:
        temp = 60 + index
        query = f"What is the temperature in {city}?"
        expected = CallExpected(
            tool_name="get_weather",
            arguments={"city": city},
            expected_result={"city": city, "temperature_f": temp},
        )
        abstain_query = f"It is {temp}°F in {city}. What is the temperature there?"
        abstain_expected: Any = AnswerExpected(
            validator=ExactAnswerValidator(value=f"{temp}°F")
        )
        perturbation = PerturbationType.ANSWER_PROVIDED
        act_tools = abstain_tools = WEATHER_TOOLS
        act_state = abstain_state = {"weather": {city: temp}}
    elif label is DecisionClass.CLARIFY:
        query = f"Calculate the distance from {city} to {other}."
        expected = CallExpected(
            tool_name="distance",
            arguments={"origin": city, "destination": other},
            expected_result={
                "origin": city,
                "destination": other,
                "miles": 100 + index,
            },
        )
        abstain_query = f"Calculate the distance from {city}."
        abstain_expected = ClarifyExpected(missing_slots=("destination",))
        perturbation = PerturbationType.REQUIRED_ARGUMENT_REMOVED
        act_tools = abstain_tools = WEATHER_TOOLS
        act_state = abstain_state = {"miles": 100 + index}
    elif label is DecisionClass.REFUSE:
        query = f"What time is it in {city}?"
        expected = CallExpected(
            tool_name="get_time",
            arguments={"city": city},
            expected_result={"city": city, "time": f"{index % 12 + 1:02d}:00"},
        )
        abstain_query = query
        abstain_expected = RefuseExpected(
            unavailable_capability="get_time", reason="missing_tool"
        )
        perturbation = PerturbationType.TOOL_REMOVED
        act_tools = WEATHER_TOOLS
        abstain_tools = tuple(tool for tool in WEATHER_TOOLS if tool.name != "get_time")
        act_state = abstain_state = {"time": f"{index % 12 + 1:02d}:00"}
    else:
        address = f"{100 + index} Main St, {city}"
        coordinates: dict[str, JsonValue] = {
            "lat": 30 + index / 10,
            "lon": -90 - index / 10,
        }
        query = f"Geocode {address}."
        expected = CallExpected(
            tool_name="geocode",
            arguments={"address": address},
            expected_result={"address": address, **coordinates},
        )
        abstain_query = query
        abstain_expected = NoopExpected(
            state_assertion="The address is already geocoded.",
            allowed_markers=("already geocoded", "no action needed"),
        )
        perturbation = PerturbationType.ALREADY_SATISFIED
        act_tools = abstain_tools = WEATHER_TOOLS
        act_state = {"geocoded": {}, "coordinates": coordinates}
        abstain_state = {"geocoded": {address: coordinates}}
    act = TaskRecord(
        **_fields("weather", pair_id, config, TaskVariant.ACT),
        query=query,
        tools=act_tools,
        environment=act_state,
        label=DecisionClass.CALL,
        perturbation=None,
        expected=expected,
    )
    abstain = TaskRecord(
        **_fields("weather", pair_id, config, TaskVariant.ABSTAIN),
        query=abstain_query,
        tools=abstain_tools,
        environment=abstain_state,
        label=label,
        perturbation=perturbation,
        expected=abstain_expected,
    )
    return TaskPair(pair_id=pair_id, act=act, abstain=abstain)


def generate_domain_pairs(domain: str, config: ProductivityConfig) -> list[TaskPair]:
    """Generate 25 pairs per abstention class for finance or weather."""
    builder = _finance_pair if domain == "finance" else _weather_pair
    if domain not in {"finance", "weather"}:
        raise ValueError(f"unsupported domain: {domain}")
    return [
        builder(label, index, config)
        for label in (
            DecisionClass.ANSWER,
            DecisionClass.CLARIFY,
            DecisionClass.REFUSE,
            DecisionClass.NOOP,
        )
        for index in range(config.pairs_per_class)
    ]
