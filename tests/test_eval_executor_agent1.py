import asyncio
from pathlib import Path
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


def test_executor_stages_required_files_and_augments_prompt(monkeypatch, tmp_path: Path):
    prompts = []

    def _fake_create_runtime_session(*, log_level=None, enable_event_logger=True, overrides=None, **kwargs):
        del log_level, enable_event_logger, overrides, kwargs
        return _FakeSession(agent=_FakeAgent())

    async def _fake_run_user_turn(session, prompt, on_warning=None):
        del session, on_warning
        prompts.append(str(prompt))
        return "pred"

    async def _fake_shutdown_runtime_session(session):
        del session
        return None

    monkeypatch.setattr(agent_runtime, "create_runtime_session", _fake_create_runtime_session)
    monkeypatch.setattr(agent_runtime, "run_user_turn", _fake_run_user_turn)
    monkeypatch.setattr(agent_runtime, "shutdown_runtime_session", _fake_shutdown_runtime_session)

    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_file = source_dir / "report.txt"
    source_file.write_text("hello", encoding="utf-8")

    sample = BenchmarkSample(
        sample_id="with_file",
        prompt="Read attachment then answer.",
        ground_truth="pred",
        assets={
            "required_files": [
                {
                    "name": "report.txt",
                    "dataset_path": "2023/validation/report.txt",
                    "source_uri": str(source_file),
                }
            ]
        },
    )
    run_dir = tmp_path / "run"
    executor = Agent1RuntimeExecutor()

    async def _run():
        return await executor.run_sample(
            sample,
            timeout_s=5,
            sample_index=0,
            run_id="run",
            run_dir=str(run_dir),
            executor_options={},
        )

    result = asyncio.run(_run())
    assert result.status == "ok"
    assert prompts
    assert "Required attachment files are preloaded in the workspace at:" in prompts[0]
    assert "report.txt" in prompts[0]

    attachments = result.usage.get("attachments")
    assert isinstance(attachments, list)
    assert len(attachments) == 1
    local_path = attachments[0].get("local_path", "")
    assert isinstance(local_path, str) and local_path
    assert Path(local_path).exists()


def test_executor_timeout_error_is_human_readable(monkeypatch, tmp_path):
    def _fake_create_runtime_session(*, log_level=None, enable_event_logger=True, overrides=None, **kwargs):
        del log_level, enable_event_logger, overrides, kwargs
        return _FakeSession(agent=_FakeAgent())

    async def _fake_run_user_turn(session, prompt, on_warning=None):
        del session, prompt, on_warning
        await asyncio.sleep(0.05)
        return "pred"

    async def _fake_shutdown_runtime_session(session):
        del session
        return None

    monkeypatch.setattr(agent_runtime, "create_runtime_session", _fake_create_runtime_session)
    monkeypatch.setattr(agent_runtime, "run_user_turn", _fake_run_user_turn)
    monkeypatch.setattr(agent_runtime, "shutdown_runtime_session", _fake_shutdown_runtime_session)

    executor = Agent1RuntimeExecutor()

    async def _run():
        sample = BenchmarkSample(sample_id="a", prompt="q", ground_truth="g")
        return await executor.run_sample(
            sample,
            timeout_s=0.01,
            sample_index=0,
            run_id="run",
            run_dir=str(tmp_path),
            executor_options={},
        )

    result = asyncio.run(_run())
    assert result.status == "error"
    assert result.error == "timeout after 0.01s"
