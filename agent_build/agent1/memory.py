import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from agent.agent_types import AgentMessage, TextContent

from POP.embedder import Embedder

from .constants import USER_PROMPT_MARKER
from .message_utils import extract_original_user_message, extract_texts


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass
class MemoryEntry:
    text: str
    embedding: np.ndarray


class ConversationMemory:
    """Short-term in-memory vector store."""

    def __init__(self, embedder: Embedder, max_entries: int = 100) -> None:
        self.embedder = embedder
        self.max_entries = max_entries
        self._entries: List[MemoryEntry] = []

    def add(self, text: str) -> None:
        embedding = self.embedder.get_embedding([text])[0]
        self._entries.append(MemoryEntry(text=text, embedding=embedding))
        if len(self._entries) > self.max_entries:
            self._entries.pop(0)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if not self._entries:
            return []
        query_emb = self.embedder.get_embedding([query])[0]
        scores = [cosine_similarity(query_emb, entry.embedding) for entry in self._entries]
        k = max(1, min(int(top_k or 1), len(self._entries)))
        top_indices = np.argsort(scores)[-k:][::-1]
        return [self._entries[i].text for i in top_indices]


class SessionConversationMemory:
    """Per-session short-term memory with vector retrieval."""

    def __init__(self, embedder: Embedder, max_entries_per_session: int = 100, max_sessions: int = 50) -> None:
        self.embedder = embedder
        self.max_entries_per_session = max_entries_per_session
        self.max_sessions = max_sessions
        self._sessions: Dict[str, ConversationMemory] = {}
        self._session_order: List[str] = []

    def _ensure_session(self, session_id: str) -> ConversationMemory:
        sid = session_id.strip() or "default"
        if sid not in self._sessions:
            self._sessions[sid] = ConversationMemory(self.embedder, max_entries=self.max_entries_per_session)
            self._session_order.append(sid)
            if len(self._session_order) > self.max_sessions:
                oldest = self._session_order.pop(0)
                self._sessions.pop(oldest, None)
        return self._sessions[sid]

    def add(self, session_id: str, text: str) -> None:
        self._ensure_session(session_id).add(text)

    def retrieve(self, query: str, top_k: int = 3, session_id: Optional[str] = None) -> List[str]:
        sid = (session_id or "default").strip() or "default"
        session = self._sessions.get(sid)
        if session is None:
            return []
        return session.retrieve(query, top_k=top_k)


class DiskMemory:
    """
    Persistent text+vector memory split across two files:
      - <base>.text.jsonl
      - <base>.embeddings.npy
    """

    def __init__(self, filepath: str, embedder: Embedder, max_entries: int = 1000) -> None:
        self.base = os.path.splitext(filepath)[0] if filepath.endswith(".jsonl") else filepath
        self.text_path = f"{self.base}.text.jsonl"
        self.emb_path = f"{self.base}.embeddings.npy"
        self.embedder = embedder
        self.max_entries = max_entries
        os.makedirs(os.path.dirname(self.text_path) or ".", exist_ok=True)
        self._n_text = self._count_lines(self.text_path)

    def add(self, text: str, session_id: Optional[str] = None, memory_kind: str = "message") -> None:
        record = {
            "text": text,
            "session_id": (session_id or "").strip() or "default",
            "memory_kind": (memory_kind or "message").strip() or "message",
            "timestamp": time.time(),
        }
        with open(self.text_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._n_text += 1

        vec = self.embedder.get_embedding([text])[0].astype("float32")
        if os.path.exists(self.emb_path):
            matrix = np.load(self.emb_path, mmap_mode=None, allow_pickle=False)
            matrix = np.vstack([matrix, vec[None, :]])
        else:
            matrix = vec[None, :]
        np.save(self.emb_path, matrix)

        self._prune()

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        session_id: Optional[str] = None,
        include_default: bool = True,
    ) -> List[str]:
        if not os.path.exists(self.emb_path) or self._n_text == 0:
            return []

        query_vec = self.embedder.get_embedding([query])[0].astype("float32")
        matrix = np.load(self.emb_path, mmap_mode="r")

        def _norm(arr: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(arr, axis=-1, keepdims=True)
            norms[norms == 0] = 1.0
            return arr / norms

        query_n = _norm(query_vec[None, :])[0]
        matrix_n = _norm(matrix)
        sims = (matrix_n @ query_n).astype("float32")
        k = max(1, min(int(top_k or 1), sims.shape[0]))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        filtered = self._read_records_by_index(idx.tolist(), session_id=session_id, include_default=include_default)
        return [item["text"] for item in filtered]

    def _count_lines(self, path: str) -> int:
        if not os.path.exists(path):
            return 0
        with open(path, "rb") as f:
            return sum(1 for _ in f)

    def _read_lines_by_index(self, indices: Sequence[int]) -> List[str]:
        raw = self._read_raw_lines_by_index(indices)
        parsed: List[str] = []
        for line in raw:
            try:
                parsed.append(str(json.loads(line)["text"]))
            except Exception:
                parsed.append(line.strip())
        return [t for t in parsed if t]

    def _read_raw_lines_by_index(self, indices: Sequence[int]) -> List[str]:
        if not indices:
            return []
        wanted = set(indices)
        found: Dict[int, str] = {}
        with open(self.text_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i in wanted:
                    found[i] = line.strip()
                if len(found) == len(wanted):
                    break
        ordered = [found.get(i, "") for i in indices]
        return [t for t in ordered if t]

    def _read_records_by_index(
        self,
        indices: Sequence[int],
        session_id: Optional[str],
        include_default: bool,
    ) -> List[Dict[str, Any]]:
        target_session = (session_id or "").strip() or None
        results: List[Dict[str, Any]] = []
        ordered_records = self._read_raw_lines_by_index(indices)
        for raw in ordered_records:
            try:
                payload = json.loads(raw) if raw.startswith("{") else {"text": raw}
            except Exception:
                payload = {"text": raw}
            text = str(payload.get("text", "")).strip()
            if not text:
                continue
            record_session = str(payload.get("session_id", "default")).strip() or "default"
            if target_session is not None:
                if record_session == target_session:
                    results.append({"text": text})
                    continue
                if include_default and record_session == "default":
                    results.append({"text": text})
                    continue
                continue
            results.append({"text": text})
        return results

    def _prune(self) -> None:
        if self._n_text <= self.max_entries:
            return
        keep = self.max_entries

        with open(self.text_path, "rb") as f:
            lines = f.readlines()[-keep:]
        with open(self.text_path, "wb") as f:
            f.writelines(lines)
        self._n_text = keep

        if os.path.exists(self.emb_path):
            matrix = np.load(self.emb_path, mmap_mode=None, allow_pickle=False)
            if len(matrix) > keep:
                np.save(self.emb_path, matrix[-keep:])


class MemoryRetriever:
    """Shared retrieval service for prompt injection and memory tool calls."""

    def __init__(self, short_term: SessionConversationMemory, long_term: Optional[DiskMemory] = None) -> None:
        self.short_term = short_term
        self.long_term = long_term

    def retrieve_sections(
        self,
        query: str,
        top_k: int = 3,
        scope: str = "both",
        session_id: str = "default",
    ) -> Tuple[List[str], List[str]]:
        scope = (scope or "both").strip().lower()
        if scope not in {"short", "long", "both"}:
            scope = "both"
        k = max(1, int(top_k or 1))
        short_hits: List[str] = []
        long_hits: List[str] = []
        if scope in {"short", "both"}:
            short_hits = self.short_term.retrieve(query, top_k=k, session_id=session_id)
        if scope in {"long", "both"} and self.long_term is not None:
            long_hits = self.long_term.retrieve(query, top_k=k, session_id=session_id)
        return short_hits, long_hits

    def retrieve(self, query: str, top_k: int = 3, scope: str = "both", session_id: str = "default") -> List[str]:
        short_hits, long_hits = self.retrieve_sections(query, top_k=top_k, scope=scope, session_id=session_id)
        seen = set()
        merged: List[str] = []
        for item in short_hits + long_hits:
            if item not in seen:
                merged.append(item)
                seen.add(item)
        return merged


def format_memory_sections(short_hits: List[str], long_hits: List[str]) -> str:
    sections: List[str] = []
    if short_hits:
        sections.append("|Short-term memory|:\n" + "\n".join(f"- {x}" for x in short_hits))
    if long_hits:
        sections.append("|Long-term memory|:\n" + "\n".join(f"- {x}" for x in long_hits))
    if not sections:
        return "(no relevant memories)"
    return "\n\n".join(sections)


def build_augmented_prompt(user_message: str, memory_text: str) -> str:
    return (
        "Use the memory context as soft background. It may be incomplete or outdated.\n"
        "Prioritize the current user message and ask follow-up questions when needed.\n\n"
        f"Memory context:\n{memory_text}\n\n"
        f"{USER_PROMPT_MARKER}{user_message.strip()}"
    )


class EmbeddingIngestionWorker:
    """Background ingestion worker for embedding writes."""

    def __init__(self, memory: SessionConversationMemory, long_term: Optional[DiskMemory] = None) -> None:
        self.memory = memory
        self.long_term = long_term
        self.active_session_id = "default"
        self._queue: asyncio.Queue[Optional[Tuple[str, str, str]]] = asyncio.Queue()
        self._task: Optional[asyncio.Task[None]] = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    def set_active_session(self, session_id: str) -> None:
        self.active_session_id = session_id.strip() or "default"

    def enqueue(self, role: str, text: str) -> None:
        value = text.strip()
        if not value:
            return
        self._queue.put_nowait((self.active_session_id, role, f"{role}: {value}"))

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is None:
                    return
                session_id, _, text = item
                await asyncio.to_thread(self.memory.add, session_id, text)
                if self.long_term is not None:
                    await asyncio.to_thread(self.long_term.add, text, session_id, "message")
            except Exception as exc:
                print(f"[memory] ingest warning: {exc}")
            finally:
                self._queue.task_done()

    async def flush(self) -> None:
        await self._queue.join()

    async def shutdown(self) -> None:
        if self._task is None:
            return
        await self._queue.put(None)
        await self._queue.join()
        await self._task
        self._task = None


class MemorySubscriber:
    """Consumes agent events and sends user/assistant text to embedding worker."""

    def __init__(self, ingestion_worker: EmbeddingIngestionWorker) -> None:
        self.ingestion_worker = ingestion_worker

    def on_event(self, event: Dict[str, Any]) -> None:
        try:
            if event.get("type") != "message_end":
                return
            message = event.get("message")
            if message is None:
                return
            role = str(getattr(message, "role", "")).strip().lower()
            if role not in {"user", "assistant"}:
                return
            text = "\n".join([x for x in extract_texts(message) if x.strip()]).strip()
            if not text:
                return
            if role == "user":
                text = extract_original_user_message(text)
            if text:
                self.ingestion_worker.enqueue(role, text)
        except Exception as exc:
            print(f"[memory] subscriber warning: {exc}")


class ContextCompressor:
    """Compresses older message context into a lightweight summary."""

    def __init__(self, trigger_chars: int = 20_000, target_keep_chars: int = 12_000) -> None:
        self.trigger_chars = max(2_000, trigger_chars)
        self.target_keep_chars = max(1_000, min(target_keep_chars, self.trigger_chars - 500))

    def maybe_compress(self, agent: Any, session_id: str, long_term: Optional[DiskMemory] = None) -> bool:
        messages = getattr(getattr(agent, "state", None), "messages", None)
        if not isinstance(messages, list) or not messages:
            return False
        total_chars = sum(len("\n".join(extract_texts(msg))) for msg in messages)
        if total_chars < self.trigger_chars:
            return False

        removed: List[Any] = []
        while messages and total_chars > self.target_keep_chars:
            msg = messages.pop(0)
            removed.append(msg)
            total_chars -= len("\n".join(extract_texts(msg)))
        if not removed:
            return False

        summary = self._summarize_messages(removed)
        messages.insert(
            0,
            AgentMessage(role="assistant", content=[TextContent(type="text", text=summary)], timestamp=time.time()),
        )
        if long_term is not None:
            long_term.add(summary, session_id=session_id, memory_kind="compression_summary")
        return True

    def _summarize_messages(self, messages: Sequence[Any]) -> str:
        lines: List[str] = ["Compressed context summary:"]
        for msg in messages[-12:]:
            role = str(getattr(msg, "role", "unknown"))
            text = " ".join(extract_texts(msg)).strip().replace("\n", " ")
            if text:
                lines.append(f"- {role}: {text[:240]}")
        if len(lines) == 1:
            lines.append("- (no text content)")
        return "\n".join(lines)
