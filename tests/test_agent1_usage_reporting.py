from agent_build.agent1.usage_reporting import (
    format_cumulative_usage_fragment,
    format_turn_usage_line,
    usage_delta,
)


def test_usage_delta_calculates_expected_fields():
    before = {
        "calls": 2,
        "input_tokens": 100,
        "output_tokens": 30,
        "total_tokens": 130,
        "provider_calls": 1,
        "estimated_calls": 1,
        "hybrid_calls": 0,
        "anomaly_calls": 0,
    }
    after = {
        "calls": 3,
        "input_tokens": 120,
        "output_tokens": 40,
        "total_tokens": 160,
        "provider_calls": 2,
        "estimated_calls": 1,
        "hybrid_calls": 0,
        "anomaly_calls": 1,
    }

    delta = usage_delta(before, after)

    assert delta == {
        "calls": 1,
        "input_tokens": 20,
        "output_tokens": 10,
        "total_tokens": 30,
        "provider_calls": 1,
        "estimated_calls": 0,
        "hybrid_calls": 0,
        "anomaly_calls": 1,
    }


def test_format_turn_usage_line_is_compact_and_token_focused():
    delta = {
        "calls": 1,
        "input_tokens": 12,
        "output_tokens": 8,
        "total_tokens": 20,
        "provider_calls": 1,
        "estimated_calls": 0,
        "hybrid_calls": 0,
        "anomaly_calls": 0,
    }
    line = format_turn_usage_line(delta, {"source": "provider", "provider_cost": 9.99})

    assert line.startswith("[usage] turn")
    assert "in=12" in line
    assert "out=8" in line
    assert "total=20" in line
    assert "source=provider" in line
    assert "mix(p/e/h)=1/0/0" in line
    assert "anomalies=0" in line
    assert "cost" not in line.lower()


def test_format_turn_usage_line_empty_when_no_calls():
    assert format_turn_usage_line({"calls": 0}, {"source": "estimate"}) == ""


def test_format_cumulative_usage_fragment_handles_partial_summary():
    fragment = format_cumulative_usage_fragment({"total_tokens": 15, "calls": 2})
    assert fragment == "usage(total=15,in=0,out=0,calls=2,p/e/h=0/0/0,anom=0)"
