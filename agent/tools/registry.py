from __future__ import annotations
import inspect
import types
from typing import Any, Awaitable, Callable, Union, get_args, get_origin
from pydantic import Field
from core.models import CoreModel
from utils.logger import get_logger


class ToolDefinition(CoreModel):
    name: str
    description: str = ""
    category: str = "general"
    token_cost_est: int = Field(default=500, ge=0)
    timeout_seconds: float = Field(default=60.0, gt=0)
    requires_llm: bool = False
    parameters_schema: dict[str, Any] = Field(default_factory=dict)
    required_parameters: list[str] = Field(default_factory=list)
    parameter_types: dict[str, str] = Field(default_factory=dict)
    parameter_defaults: dict[str, Any] = Field(default_factory=dict)
    parameter_hints: dict[str, str] = Field(default_factory=dict)
    accepts_kwargs: bool = False


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._executors: dict[str, Callable[..., Awaitable[Any]]] = {}
        self._logger = get_logger("agent.tools.registry")

    def register(
        self,
        definition: ToolDefinition,
        execute_fn: Callable[..., Awaitable[Any]],
    ) -> None:
        if definition.name in self._tools:
            self._logger.warning(f"Tool '{definition.name}' already registered, overwriting")

        inferred = self._infer_signature(execute_fn)

        schema = dict(inferred["parameters_schema"])
        schema.update(definition.parameters_schema or {})

        required: list[str] = []
        seen: set[str] = set()
        for item in list(definition.required_parameters) + list(inferred["required_parameters"]):
            key = str(item)
            if key and key not in seen:
                seen.add(key)
                required.append(key)

        accepts_kwargs = bool(definition.accepts_kwargs or inferred["accepts_kwargs"])
        required = [
            item
            for item in required
            if item in schema or accepts_kwargs
        ]

        parameter_types = dict(inferred["parameter_types"])
        for key, value in schema.items():
            if isinstance(value, str):
                parameter_types[key] = value
            elif isinstance(value, dict) and isinstance(value.get("type"), str):
                parameter_types[key] = value["type"]
            else:
                parameter_types.setdefault(key, "any")

        definition.parameters_schema = schema
        definition.required_parameters = required
        definition.parameter_types = parameter_types
        definition.parameter_defaults = dict(inferred["parameter_defaults"])
        definition.parameter_hints = dict(inferred["parameter_hints"])
        definition.accepts_kwargs = accepts_kwargs

        self._tools[definition.name] = definition
        self._executors[definition.name] = execute_fn

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def get_executor(self, name: str) -> Callable[..., Awaitable[Any]] | None:
        return self._executors.get(name)

    def list_all(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def list_by_category(self, category: str) -> list[ToolDefinition]:
        return [tool for tool in self._tools.values() if tool.category == category]

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def find_tools_for_action(self, action_type: str) -> list[ToolDefinition]:
        category_map = {
            "search": "search",
            "read": "read",
            "analyze": "analysis",
            "generate": "generation",
            "memory_store": "memory",
            "memory_recall": "memory",
            "reflect": "analysis",
            "plan": "generation",
            "synthesize": "generation",
        }
        category = category_map.get(action_type, "general")
        return self.list_by_category(category)

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    def get_parameter_requirements(self, name: str) -> dict[str, Any]:
        definition = self._tools.get(name)
        if definition is None:
            return {}

        return {
            "name": definition.name,
            "parameters_schema": dict(definition.parameters_schema),
            "required_parameters": list(definition.required_parameters),
            "parameter_types": dict(definition.parameter_types),
            "parameter_defaults": dict(definition.parameter_defaults),
            "parameter_hints": dict(definition.parameter_hints),
            "accepts_kwargs": definition.accepts_kwargs,
        }

    def validate_parameters(
        self,
        name: str,
        parameters: dict[str, Any] | None,
    ) -> tuple[bool, list[str]]:
        definition = self._tools.get(name)
        if definition is None:
            return False, [f"Tool '{name}' not found"]

        params = parameters if isinstance(parameters, dict) else {}
        missing: list[str] = []

        for key in definition.required_parameters:
            if key not in params:
                missing.append(key)
                continue

            value = params[key]
            if value is None:
                missing.append(key)
            elif isinstance(value, str) and not value.strip():
                missing.append(key)
            elif isinstance(value, (list, tuple, set, dict)) and not value:
                missing.append(key)

        return not missing, missing

    def to_summary(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "category": tool.category,
                "description": tool.description,
                "token_cost_est": tool.token_cost_est,
                "timeout_seconds": tool.timeout_seconds,
                "requires_llm": tool.requires_llm,
                "parameters_schema": tool.parameters_schema,
                "required_parameters": tool.required_parameters,
                "parameter_types": tool.parameter_types,
                "parameter_defaults": tool.parameter_defaults,
                "parameter_hints": tool.parameter_hints,
                "accepts_kwargs": tool.accepts_kwargs,
            }
            for tool in self._tools.values()
        ]

    def _infer_signature(self, execute_fn: Callable[..., Awaitable[Any]]) -> dict[str, Any]:
        schema: dict[str, Any] = {}
        required: list[str] = []
        parameter_types: dict[str, str] = {}
        defaults: dict[str, Any] = {}
        hints: dict[str, str] = {}
        accepts_kwargs = False

        try:
            signature = inspect.signature(execute_fn)
        except Exception:
            return {
                "parameters_schema": schema,
                "required_parameters": required,
                "parameter_types": parameter_types,
                "parameter_defaults": defaults,
                "parameter_hints": hints,
                "accepts_kwargs": accepts_kwargs,
            }

        for name, param in signature.parameters.items():
            if name in ("self", "cls"):
                continue

            if param.kind == inspect.Parameter.VAR_KEYWORD:
                accepts_kwargs = True
                continue

            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                continue

            type_name = self._type_name(param.annotation)
            has_default = param.default is not inspect.Parameter.empty

            schema[name] = type_name
            parameter_types[name] = type_name

            if not has_default:
                required.append(name)
            else:
                defaults[name] = param.default

            hint = f"{name}: {type_name}"
            if has_default:
                hint += f" (default: {self._safe_repr(param.default)})"
            hints[name] = hint

        return {
            "parameters_schema": schema,
            "required_parameters": required,
            "parameter_types": parameter_types,
            "parameter_defaults": defaults,
            "parameter_hints": hints,
            "accepts_kwargs": accepts_kwargs,
        }

    def _type_name(self, annotation: Any) -> str:
        if annotation is inspect.Parameter.empty or annotation is Any:
            return "any"

        if isinstance(annotation, str):
            return annotation

        origin = get_origin(annotation)
        if origin is None:
            return self._python_type_name(annotation)

        if origin in (list, tuple, set, frozenset):
            return "array"

        if origin is dict:
            return "object"

        union_types = {Union, getattr(types, "UnionType", None)} - {None}
        if origin in union_types:
            args = [arg for arg in get_args(annotation) if arg is not type(None)]
            if len(args) == 1:
                return self._type_name(args[0])
            return "any"

        return self._python_type_name(origin)

    def _python_type_name(self, value: Any) -> str:
        mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            set: "array",
            tuple: "array",
            frozenset: "array",
            type(None): "null",
        }

        if value in mapping:
            return mapping[value]

        name = getattr(value, "__name__", str(value))
        lowered = name.lower()

        if lowered == "any":
            return "any"
        if lowered == "object":
            return "object"

        return lowered

    @staticmethod
    def _safe_repr(value: Any) -> str:
        try:
            return repr(value)[:80]
        except Exception:
            return str(value)[:80]