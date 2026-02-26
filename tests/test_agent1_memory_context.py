from types import SimpleNamespace

import numpy as np

from agent.agent_types import TextContent
from agent_build.agent1.memory import (
    ContextCompressor,
    DiskMemory,
    EmbeddingIngestionWorker,
    MemoryRetriever,
    SessionConversationMemory,
)


class _FakeEmbedder:
    def get_embedding(self, texts):
        vectors = []
        for text in texts:
            seed = float(sum(ord(ch) for ch in text) % 97)
            vectors.append(np.array([seed, seed / 2.0, 1.0], dtype="float32"))
        return vectors


def test_session_conversation_memory_isolation():
    memory = SessionConversationMemory(embedder=_FakeEmbedder(), max_entries_per_session=10)
    memory.add("alpha", "user: project a details")
    memory.add("beta", "user: project b details")

    alpha_hits = memory.retrieve("project a", top_k=1, session_id="alpha")
    beta_hits = memory.retrieve("project b", top_k=1, session_id="beta")

    assert alpha_hits == ["user: project a details"]
    assert beta_hits == ["user: project b details"]


def test_retriever_filters_disk_memory_by_session(tmp_path):
    embedder = _FakeEmbedder()
    short_memory = SessionConversationMemory(embedder=embedder)
    disk_memory = DiskMemory(filepath=str(tmp_path / "chat"), embedder=embedder, max_entries=20)

    short_memory.add("s1", "user: session one short memory")
    short_memory.add("s2", "user: session two short memory")
    disk_memory.add("assistant: session one long memory", session_id="s1")
    disk_memory.add("assistant: session two long memory", session_id="s2")

    retriever = MemoryRetriever(short_term=short_memory, long_term=disk_memory)
    s1_hits = retriever.retrieve("session", top_k=5, scope="both", session_id="s1")

    assert any("session one" in hit for hit in s1_hits)
    assert all("session two" not in hit for hit in s1_hits)


def test_context_compressor_replaces_old_messages_with_summary(tmp_path):
    embedder = _FakeEmbedder()
    disk_memory = DiskMemory(filepath=str(tmp_path / "chat"), embedder=embedder, max_entries=20)
    compressor = ContextCompressor(trigger_chars=80, target_keep_chars=50)

    agent = SimpleNamespace(
        state=SimpleNamespace(
            messages=[
                SimpleNamespace(role="user", content=[TextContent(type="text", text="A" * 40)]),
                SimpleNamespace(role="assistant", content=[TextContent(type="text", text="B" * 40)]),
                SimpleNamespace(role="user", content=[TextContent(type="text", text="C" * 40)]),
            ]
        )
    )

    changed = compressor.maybe_compress(agent, "compress-session", long_term=disk_memory)

    assert changed is True
    assert "Compressed context summary" in agent.state.messages[0].content[0].text
    long_hits = disk_memory.retrieve("Compressed context", top_k=3, session_id="compress-session")
    assert any("Compressed context summary" in hit for hit in long_hits)


def test_ingestion_worker_writes_to_active_session(tmp_path):
    embedder = _FakeEmbedder()
    short_memory = SessionConversationMemory(embedder=embedder)
    disk_memory = DiskMemory(filepath=str(tmp_path / "chat"), embedder=embedder, max_entries=20)
    worker = EmbeddingIngestionWorker(memory=short_memory, long_term=disk_memory)
    worker.set_active_session("focus")

    import asyncio

    async def _run():
        worker.start()
        worker.enqueue("user", "remember this detail")
        await worker.flush()
        await worker.shutdown()

    asyncio.run(_run())

    hits = short_memory.retrieve("remember", top_k=2, session_id="focus")
    assert hits and "remember this detail" in hits[0]


def test_session_memory_rename_moves_entries():
    memory = SessionConversationMemory(embedder=_FakeEmbedder(), max_entries_per_session=10)
    memory.add("old-session", "user: remember this")
    assert memory.has_session("old-session") is True
    assert memory.rename_session("old-session", "new-session") is True
    assert memory.has_session("old-session") is False
    assert memory.has_session("new-session") is True
    hits = memory.retrieve("remember", top_k=1, session_id="new-session")
    assert hits == ["user: remember this"]


def test_disk_memory_rename_updates_session_id(tmp_path):
    embedder = _FakeEmbedder()
    disk_memory = DiskMemory(filepath=str(tmp_path / "chat"), embedder=embedder, max_entries=20)
    disk_memory.add("assistant: hello world", session_id="old-session")
    assert disk_memory.has_session("old-session") is True
    assert disk_memory.rename_session("old-session", "new-session") is True
    assert disk_memory.has_session("old-session") is False
    assert disk_memory.has_session("new-session") is True


def test_retriever_uses_default_session_when_missing():
    embedder = _FakeEmbedder()
    short_memory = SessionConversationMemory(embedder=embedder)
    short_memory.add("alpha", "user: alpha memory")
    retriever = MemoryRetriever(short_term=short_memory, long_term=None, default_session_id="alpha")

    hits = retriever.retrieve("alpha", top_k=1, scope="short", session_id=None)

    assert hits == ["user: alpha memory"]
