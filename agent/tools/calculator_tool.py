from __future__ import annotations

import ast
import math
import re
from typing import Any, Dict, Optional

from ..agent_types import AgentTool, AgentToolResult, TextContent


_SAFE_GLOBALS: Dict[str, Any] = {
    "acos": math.acos,
    "abs": abs,
    "asin": math.asin,
    "atan": math.atan,
    "atan2": math.atan2,
    "all": all,
    "any": any,
    "bool": bool,
    "ceil": math.ceil,
    "cos": math.cos,
    "degrees": math.degrees,
    "dict": dict,
    "e": math.e,
    "enumerate": enumerate,
    "exp": math.exp,
    "factorial": math.factorial,
    "float": float,
    "floor": math.floor,
    "fsum": math.fsum,
    "gcd": math.gcd,
    "hypot": math.hypot,
    "int": int,
    "len": len,
    "list": list,
    "log": math.log,
    "log10": math.log10,
    "max": max,
    "min": min,
    "pi": math.pi,
    "pow": pow,
    "prod": math.prod,
    "radians": math.radians,
    "range": range,
    "round": round,
    "set": set,
    "sin": math.sin,
    "sorted": sorted,
    "sqrt": math.sqrt,
    "str": str,
    "sum": sum,
    "tan": math.tan,
    "tau": math.tau,
    "tuple": tuple,
    "zip": zip,
}


_ALLOWED_NODE_TYPES = (
    ast.Add,
    ast.And,
    ast.BinOp,
    ast.BoolOp,
    ast.Call,
    ast.Compare,
    ast.comprehension,
    ast.Constant,
    ast.Dict,
    ast.DictComp,
    ast.Div,
    ast.Eq,
    ast.Expression,
    ast.FloorDiv,
    ast.GeneratorExp,
    ast.Gt,
    ast.GtE,
    ast.IfExp,
    ast.In,
    ast.keyword,
    ast.List,
    ast.ListComp,
    ast.Load,
    ast.Lt,
    ast.LtE,
    ast.Mod,
    ast.Mult,
    ast.Name,
    ast.Not,
    ast.NotEq,
    ast.NotIn,
    ast.Or,
    ast.Pow,
    ast.Set,
    ast.SetComp,
    ast.Slice,
    ast.Store,
    ast.Sub,
    ast.Subscript,
    ast.Tuple,
    ast.UAdd,
    ast.UnaryOp,
    ast.USub,
)


def _truncate_text(value: str, limit: int = 4000) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    return value[: limit - 16] + "... [truncated]", True


def _build_error_hint(message: str, *, expression: str, binding_names: list[str]) -> str:
    text = str(message or "").strip()
    expr = str(expression or "").strip()
    if not text:
        return ""

    if text == "only direct function calls are allowed":
        match = re.search(r"\bmath\.([A-Za-z_][A-Za-z0-9_]*)", expr)
        if match:
            return f"use direct calls such as {match.group(1)}(...) instead of math.{match.group(1)}(...)"
        return "use direct calls such as sqrt(...), round(...), or sum(...) without module prefixes"

    if text == "name not allowed: bindings":
        match = re.search(r"""bindings\[['"]([^'"]+)['"]\]""", expr)
        if match:
            key = match.group(1)
            return (
                f"pass bindings={{\"{key}\": ...}} and reference {key} directly in the expression "
                f"instead of bindings[\"{key}\"]"
            )
        return "pass data through the bindings parameter and reference each bound name directly in the expression"

    if text.startswith("function not allowed: "):
        func_name = text.split(":", 1)[1].strip()
        if func_name == "math":
            return "module prefixes are blocked; call functions directly such as sqrt(...) or acos(...)"
        return "use direct allowed calls such as sqrt(...), round(...), sum(...), len(...), min(...), or max(...)"

    if text.startswith("name not allowed: "):
        bad_name = text.split(":", 1)[1].strip()
        if binding_names:
            available = ", ".join(binding_names[:8])
            return f"reference an allowed binding or builtin name instead; available bindings: {available}"
        if bad_name:
            return f"bind {bad_name!r} through the bindings parameter if it is input data"
    return ""


def _sanitize_binding_key(key: Any) -> str:
    text = str(key or "").strip()
    if not text:
        raise ValueError("binding names must be non-empty strings")
    if text.startswith("__"):
        raise ValueError(f"binding name not allowed: {text}")
    if not text.replace("_", "a").isalnum() or text[0].isdigit():
        raise ValueError(f"binding name must be a valid identifier: {text}")
    return text


def _sanitize_binding_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_sanitize_binding_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_binding_value(item) for item in value)
    if isinstance(value, dict):
        return {
            _sanitize_binding_key(key): _sanitize_binding_value(item)
            for key, item in value.items()
        }
    raise ValueError(f"unsupported binding value type: {type(value).__name__}")


class _SafeExpressionValidator(ast.NodeVisitor):
    def __init__(self, allowed_names: set[str]) -> None:
        self.allowed_names = set(allowed_names)
        self.local_names_stack: list[set[str]] = []

    def visit(self, node: ast.AST) -> Any:
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise ValueError(f"unsupported syntax: {type(node).__name__}")
        return super().visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ValueError("only direct function calls are allowed")
        if node.func.id not in self.allowed_names:
            raise ValueError(f"function not allowed: {node.func.id}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        if isinstance(node.ctx, ast.Load):
            if node.id in self.allowed_names:
                return
            if any(node.id in scope for scope in reversed(self.local_names_stack)):
                return
            raise ValueError(f"name not allowed: {node.id}")
        if isinstance(node.ctx, ast.Store):
            if node.id.startswith("__"):
                raise ValueError(f"name not allowed: {node.id}")
            return
        raise ValueError(f"unsupported name context: {type(node.ctx).__name__}")

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        self.visit(node.value)
        self.visit(node.slice)

    def visit_ListComp(self, node: ast.ListComp) -> Any:
        self._visit_comprehension(node.generators, lambda: self.visit(node.elt))

    def visit_SetComp(self, node: ast.SetComp) -> Any:
        self._visit_comprehension(node.generators, lambda: self.visit(node.elt))

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> Any:
        self._visit_comprehension(node.generators, lambda: self.visit(node.elt))

    def visit_DictComp(self, node: ast.DictComp) -> Any:
        self._visit_comprehension(
            node.generators,
            lambda: (self.visit(node.key), self.visit(node.value)),
        )

    def _visit_comprehension(self, generators: list[ast.comprehension], visit_body: Any) -> None:
        self.local_names_stack.append(set())
        try:
            for generator in generators:
                self.visit(generator.iter)
                self._register_target_names(generator.target)
                self.visit(generator.target)
                for if_clause in generator.ifs:
                    self.visit(if_clause)
            visit_body()
        finally:
            self.local_names_stack.pop()

    def _register_target_names(self, target: ast.AST) -> None:
        scope = self.local_names_stack[-1]
        if isinstance(target, ast.Name):
            if target.id.startswith("__"):
                raise ValueError(f"name not allowed: {target.id}")
            scope.add(target.id)
            return
        if isinstance(target, (ast.List, ast.Tuple)):
            for item in target.elts:
                self._register_target_names(item)
            return
        raise ValueError(f"unsupported comprehension target: {type(target).__name__}")


class CalculatorTool(AgentTool):
    name = "calculator"
    description = (
        "Evaluate a restricted single Python-style expression for arithmetic, unit conversions, "
        "small brute-force searches, and checksum logic. Use bindings for long lists/dicts; "
        "statements, assignments, and imports are not supported."
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Restricted single Python expression to evaluate.",
            },
            "bindings": {
                "type": "object",
                "description": (
                    "Optional literal bindings referenced by the expression. Prefer this for longer "
                    "lists/dicts instead of embedding multiline Python."
                ),
            },
        },
        "required": ["expression"],
    }
    label = "Calculator"

    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        signal: Optional[Any] = None,
        on_update: Optional[Any] = None,
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update
        expression = str(params.get("expression", "") or "").strip()
        if not expression:
            return self._error("missing expression", expression="", binding_names=[])

        raw_bindings = params.get("bindings", {})
        if raw_bindings is None:
            raw_bindings = {}
        if not isinstance(raw_bindings, dict):
            return self._error("bindings must be an object", expression=expression, binding_names=[])

        try:
            bindings = {
                _sanitize_binding_key(key): _sanitize_binding_value(value)
                for key, value in raw_bindings.items()
            }
            parsed = ast.parse(expression, mode="eval")
            validator = _SafeExpressionValidator(set(_SAFE_GLOBALS) | set(bindings))
            validator.visit(parsed)
            compiled = compile(parsed, "<calculator>", "eval")
            globals_dict = {"__builtins__": {}}
            globals_dict.update(_SAFE_GLOBALS)
            globals_dict.update(bindings)
            result = eval(compiled, globals_dict, {})
        except Exception as exc:
            return self._error(str(exc), expression=expression, binding_names=sorted(bindings) if "bindings" in locals() else [])

        rendered, truncated = _truncate_text(repr(result))
        return AgentToolResult(
            content=[TextContent(type="text", text=rendered)],
            details={
                "ok": True,
                "expression": expression,
                "binding_names": sorted(bindings),
                "result_text": rendered,
                "result_type": type(result).__name__,
                "truncated": truncated,
            },
        )

    def _error(self, message: str, *, expression: str, binding_names: list[str]) -> AgentToolResult:
        hint = _build_error_hint(message, expression=expression, binding_names=binding_names)
        text = f"calculator error: {message}"
        if hint:
            text += f". Hint: {hint}"
        return AgentToolResult(
            content=[TextContent(type="text", text=text)],
            details={
                "ok": False,
                "error": message,
                "expression": expression,
                "binding_names": binding_names,
                "hint": hint,
            },
        )
