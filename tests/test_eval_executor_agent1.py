import asyncio
from types import SimpleNamespace

from agent_build.agent1 import runtime as agent_runtime
from eval.core.contracts import BenchmarkSample
from eval.executors.agent1_runtime_executor import Agent1RuntimeExecutor


class _FakeAgent:
    def __init__(self):
        self._summary = {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "provider_calls": 0,
            "estimated_calls": 0,
            "hybrid_calls": 0,
            "anomaly_calls": 0,
        }
        self._subs = []

    def subscribe(self, fn):
        self._subs.append(fn)

        def _unsubscribe():
            if fn in self._subs:
                self._subs.remove(fn)

        return _unsubscribe

    def get_usage_summary(self):
        return dict(self._summary)

    def get_last_usage(self):
        return {"source": "provider", "total_tokens": 3}


class _FakeSession(SimpleNamespace):
    pass


def test_per_sample_isolation_uses_unique_memory_paths(monkeypatch, tmp_path):
    seen_paths = []

    def _fake_create_runtime_session(*, log_level=None, enable_event_logger=True, overrides=None, **kwargs):
        del log_level, enable_event_logger, kwargs
        assert overrides is not None
        seen_paths.append(overrides.long_memory_base_path)
        return _FakeSession(agent=_FakeAgent())

    async def _fake_run_user_turn(session, prompt, on_warning=None):
        del on_warning, prompt
        session.agent._summary["calls"] += 1
        session.agent._summary["total_tokens"] += 3
        for sub in list(session.agent._subs):
            sub({"type": "message_end", "message": {"role": "assistant"}})
        return "pred"

    async def _fake_shutdown_runtime_session(session):
        del session
        return None

    monkeypatch.setattr(agent_runtime, "create_runtime_session", _fake_create_runtime_session)
    monkeypatch.setattr(agent_runtime, "run_user_turn", _fake_run_user_turn)
    monkeypatch.setattr(agent_runtime, "shutdown_runtime_session", _fake_shutdown_runtime_session)

    executor = Agent1RuntimeExecutor()

    async def _run():
        s1 = BenchmarkSample(sample_id="a", prompt="q1", ground_truth="g1")
        s2 = BenchmarkSample(sample_id="b", prompt="q2", ground_truth="g2")

        await executor.run_sample(
            s1,
            timeout_s=5,
            sample_index=0,
            run_id="run",
            run_dir=str(tmp_path),
            executor_options={},
        )
        await executor.run_sample(
            s2,
            timeout_s=5,
            sample_index=1,
            run_id="run",
            run_dir=str(tmp_path),
            executor_options={},
        )

    asyncio.run(_run())

    assert len(seen_paths) == 2
    assert seen_paths[0] != seen_paths[1]
    assert "sample_000000" in seen_paths[0]
    assert "sample_000001" in seen_paths[1]


def test_executor_forwards_enable_event_logger_option(monkeypatch, tmp_path):
    seen_values = []

    def _fake_create_runtime_session(*, log_level=None, enable_event_logger=True, overrides=None, **kwargs):
        del log_level, overrides, kwargs
        seen_values.append(bool(enable_event_logger))
        return _FakeSession(agent=_FakeAgent())

    async def _fake_run_user_turn(session, prompt, on_warning=None):
        del session, prompt, on_warning
        return "pred"

    async def _fake_shutdown_runtime_session(session):
        del session
        return None

    monkeypatch.setattr(agent_runtime, "create_runtime_session", _fake_create_runtime_session)
    monkeypatch.setattr(agent_runtime, "run_user_turn", _fake_run_user_turn)
    monkeypatch.setattr(agent_runtime, "shutdown_runtime_session", _fake_shutdown_runtime_session)

    executor = Agent1RuntimeExecutor()

    async def _run() -> None:
        sample = BenchmarkSample(sample_id="a", prompt="q", ground_truth="g")
        await executor.run_sample(
            sample,
            timeout_s=5,
            sample_index=0,
            run_id="run",
            run_dir=str(tmp_path),
            executor_options={"enable_event_logger": False},
        )
        await executor.run_sample(
            sample,
            timeout_s=5,
            sample_index=1,
            run_id="run",
            run_dir=str(tmp_path),
            executor_options={},
        )

    asyncio.run(_run())

    assert seen_values == [False, True]
