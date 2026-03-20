import asyncio

from agent.tools import CalculatorTool


def _run(params):
    return asyncio.run(CalculatorTool().execute("tc1", params))


def test_calculator_evaluates_arithmetic_expression():
    result = _run({"expression": "(0.312 / 0.12091) * 1000 / 1071.7245 * 0.082057 * 277.039"})

    assert result.details["ok"] is True
    assert result.details["result_type"] == "float"
    assert "54." in result.content[0].text


def test_calculator_supports_bindings_and_small_bruteforce():
    result = _run(
        {
            "expression": "[(w, y) for w in range(2, 10) for y in range(3, 11) if all(((sum((d if i % 2 == 0 else d * w) for i, d in enumerate(num[:-1])) + num[-1]) % 10) == 0 for num in nums)]",
            "bindings": {
                "nums": [
                    [9, 7, 8, 3, 5, 4, 1, 8, 1, 3, 9, 1, 9],
                    [9, 7, 8, 9, 4, 6, 6, 6, 9, 7, 4, 6, 1],
                ]
            },
        }
    )

    assert result.details["ok"] is True
    assert result.details["result_type"] == "list"
    assert result.content[0].text.startswith("[")


def test_calculator_rejects_attribute_access():
    result = _run({"expression": "().__class__"})

    assert result.details["ok"] is False
    assert result.details["error"] == "unsupported syntax: Attribute"
