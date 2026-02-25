from eval.core.redaction import redact_data


def test_redaction_masks_secret_values_and_patterns():
    payload = {
        "OPENAI_API_KEY": "sk-abcdefghijklmnopqrstuvwxyz123456",
        "nested": {
            "token": "hf_abcdefghijklmnopqrstuvwxyz",
            "text": "Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345",
        },
    }

    redacted = redact_data(payload)

    assert redacted["OPENAI_API_KEY"] == "[REDACTED]"
    assert redacted["nested"]["token"] == "[REDACTED]"
    assert "[REDACTED]" in redacted["nested"]["text"]


def test_redaction_keeps_usage_token_stats():
    payload = {
        "usage": {
            "input_tokens": 101,
            "output_tokens": 12,
            "total_tokens": 113,
            "cached_tokens": 0,
            "reasoning_tokens": 4,
            "estimate_input_tokens": 99,
            "estimate_output_tokens": 11,
            "estimate_total_tokens": 110,
            "token": "hf_abcdefghijklmnopqrstuvwxyz",
            "refresh_token": "abcd1234",
        }
    }

    redacted = redact_data(payload)
    usage = redacted["usage"]

    assert usage["input_tokens"] == 101
    assert usage["output_tokens"] == 12
    assert usage["total_tokens"] == 113
    assert usage["cached_tokens"] == 0
    assert usage["reasoning_tokens"] == 4
    assert usage["estimate_input_tokens"] == 99
    assert usage["estimate_output_tokens"] == 11
    assert usage["estimate_total_tokens"] == 110
    assert usage["token"] == "[REDACTED]"
    assert usage["refresh_token"] == "[REDACTED]"
