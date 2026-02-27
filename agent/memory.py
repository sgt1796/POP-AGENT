import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from .agent_types import AgentMessage, TextContent

from POP.embedder import Embedder

USER_PROMPT_MARKER = "|Current user message|:\n"


def extract_texts(message: Any) -> List[str]:
    texts: List[str] = []
    if not message:
        return texts
    content = getattr(message, "content", None)
    if not content:
        return texts
    for item in content:
        if isinstance(item, TextContent):
            texts.append(item.text or "")
        elif isinstance(item, dict) and item.get("type") == "text":
            texts.append(str(item.get("text", "")))
    return texts


def extract_original_user_message(text: str) -> str:
    if USER_PROMPT_MARKER in text:
        return text.split(USER_PROMPT_MARKER, 1)[1].strip()
    return text.strip()


def extract_bash_exec_command(event: Dict[str, Any]) -> str:
    args = event.get("args")
    if isinstance(args, dict):
        cmd = args.get("cmd")
        if isinstance(cmd, str) and cmd.strip():
            return cmd.strip()

    result = event.get("result")
    details = getattr(result, "details", None)
    if isinstance(details, dict):
        cmd = details.get("command")
        if isinstance(cmd, str) and cmd.strip():
            return cmd.strip()
    if isinstance(result, dict):
        details = result.get("details")
        if isinstance(details, dict):
            cmd = details.get("command")
            if isinstance(cmd, str) and cmd.strip():
                return cmd.strip()
    return ""


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

    def has_session(self, session_id: str) -> bool:
        sid = (session_id or "").strip()
        if not sid:
            return False
        return sid in self._sessions

    def rename_session(self, old_id: str, new_id: str) -> bool:
        old_sid = (old_id or "").strip()
        new_sid = (new_id or "").strip()
        if not old_sid or not new_sid or old_sid == new_sid:
            return False
        if old_sid not in self._sessions:
            return False
        if new_sid in self._sessions:
            target = self._sessions[new_sid]
            source = self._sessions[old_sid]
            try:
                target._entries.extend(source._entries)
                if len(target._entries) > target.max_entries:
                    target._entries = target._entries[-target.max_entries :]
            except Exception:
                pass
            self._sessions.pop(old_sid, None)
            self._session_order = [sid for sid in self._session_order if sid != old_sid]
            return True
        self._sessions[new_sid] = self._sessions.pop(old_sid)
        self._session_order = [new_sid if sid == old_sid else sid for sid in self._session_order]
        if new_sid not in self._session_order:
            self._session_order.append(new_sid)
        return True

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
      - <base>.jsonl
      - <base>.embeddings.npy
    """

    def __init__(self, filepath: str, embedder: Embedder, max_entries: int = 1000) -> None:
        raw_path = str(filepath or "").strip()
        if raw_path.endswith(".jsonl"):
            self.text_path = raw_path
            self.base = os.path.splitext(raw_path)[0]
        else:
            self.base = os.path.splitext(raw_path)[0] if raw_path else raw_path
            self.text_path = f"{self.base}.jsonl"
        self.emb_path = f"{self.base}.embeddings.npy"
        self.embedder = embedder
        self.max_entries = max_entries
        os.makedirs(os.path.dirname(self.text_path) or ".", exist_ok=True)
        legacy_text_path = f"{self.base}.text.jsonl"
        if not os.path.exists(self.text_path) and os.path.exists(legacy_text_path):
            try:
                os.replace(legacy_text_path, self.text_path)
            except Exception:
                # If rename fails, keep using the legacy path to avoid data loss.
                self.text_path = legacy_text_path
        self._n_text = self._count_lines(self.text_path)

    def add(
        self,
        text: str,
        session_id: Optional[str] = None,
        memory_type: str = "message",
        memory_kind: Optional[str] = None,
    ) -> None:
        resolved_type = str(memory_type or "").strip() or str(memory_kind or "").strip() or "message"
        record = {
            "text": text,
            "session_id": (session_id or "").strip() or "default",
            "memory_type": resolved_type,
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

    def has_session(self, session_id: str) -> bool:
        sid = (session_id or "").strip()
        if not sid:
            return False
        if not os.path.exists(self.text_path):
            return False
        with open(self.text_path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw) if raw.startswith("{") else None
                except Exception:
                    payload = None
                if isinstance(payload, dict):
                    record_session = str(payload.get("session_id", "default")).strip() or "default"
                    if record_session == sid:
                        return True
        return False

    def rename_session(self, old_id: str, new_id: str) -> bool:
        old_sid = (old_id or "").strip()
        new_sid = (new_id or "").strip()
        if not old_sid or not new_sid or old_sid == new_sid:
            return False
        if not os.path.exists(self.text_path):
            return False
        updated = False
        tmp_path = f"{self.text_path}.tmp"
        with open(self.text_path, "r", encoding="utf-8") as src, open(tmp_path, "w", encoding="utf-8") as dst:
            for line in src:
                raw = line.rstrip("\n")
                if not raw:
                    continue
                try:
                    payload = json.loads(raw) if raw.startswith("{") else None
                except Exception:
                    payload = None
                if isinstance(payload, dict):
                    record_session = str(payload.get("session_id", "default")).strip() or "default"
                    if record_session == old_sid:
                        payload["session_id"] = new_sid
                        updated = True
                    dst.write(json.dumps(payload, ensure_ascii=False) + "\n")
                else:
                    dst.write(raw + "\n")
        if not updated:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return False
        os.replace(tmp_path, self.text_path)
        return True

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
        retrievable_kinds = {"message", "compression_summary"}
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
            memory_type = (
                str(payload.get("memory_type", payload.get("memory_kind", "message"))).strip().lower() or "message"
            )
            if memory_type not in retrievable_kinds:
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

    def __init__(
        self,
        short_term: SessionConversationMemory,
        long_term: Optional[DiskMemory] = None,
        *,
        default_session_id: str = "default",
    ) -> None:
        self.short_term = short_term
        self.long_term = long_term
        self.default_session_id = str(default_session_id or "default").strip() or "default"

    def set_default_session(self, session_id: str) -> None:
        self.default_session_id = str(session_id or "default").strip() or "default"

    def _resolve_session_id(self, session_id: Optional[str]) -> str:
        sid = (session_id or "").strip()
        if not sid:
            sid = str(self.default_session_id or "").strip()
        return sid or "default"

    def retrieve_sections(
        self,
        query: str,
        top_k: int = 3,
        scope: str = "both",
        session_id: Optional[str] = None,
    ) -> Tuple[List[str], List[str]]:
        scope = (scope or "both").strip().lower()
        if scope not in {"short", "long", "both"}:
            scope = "both"
        k = max(1, int(top_k or 1))
        sid = self._resolve_session_id(session_id)
        short_hits: List[str] = []
        long_hits: List[str] = []
        if scope in {"short", "both"}:
            short_hits = self.short_term.retrieve(query, top_k=k, session_id=sid)
        if scope in {"long", "both"} and self.long_term is not None:
            long_hits = self.long_term.retrieve(query, top_k=k, session_id=sid)
        return short_hits, long_hits

    def retrieve(self, query: str, top_k: int = 3, scope: str = "both", session_id: Optional[str] = None) -> List[str]:
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

    def enqueue(
        self,
        role: str,
        text: str,
        *,
        session_id: Optional[str] = None,
        memory_type: str = "message",
        memory_kind: Optional[str] = None,
    ) -> None:
        value = text.strip()
        if not value:
            return
        kind = str(memory_type or "").strip().lower() or str(memory_kind or "").strip().lower() or "message"
        sid = self.active_session_id
        if session_id is not None:
            sid = session_id
        sid = str(sid or "").strip() or "default"
        if kind == "message":
            role_label = str(role or "unknown").strip().lower() or "unknown"
            payload = f"{role_label}: {value}"
        else:
            payload = value
        self._queue.put_nowait((sid, payload, kind))

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is None:
                    return
                session_id, text, memory_type = item
                if memory_type == "message":
                    await asyncio.to_thread(self.memory.add, session_id, text)
                if self.long_term is not None:
                    await asyncio.to_thread(self.long_term.add, text, session_id, memory_type)
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

    def _read_toolcall_preview(self, assistant_event: Dict[str, Any]) -> Tuple[str, str, Any]:
        partial = assistant_event.get("partial")
        if not isinstance(partial, dict):
            return "", "unknown", None
        content = partial.get("content")
        if not isinstance(content, list):
            return "", "unknown", None
        for item in reversed(content):
            if not isinstance(item, dict):
                continue
            if str(item.get("type", "")).strip() != "toolCall":
                continue
            call_id = str(item.get("id", "")).strip()
            tool_name = str(item.get("name", "")).strip() or "unknown"
            args = item.get("arguments")
            return call_id, tool_name, args
        return "", "unknown", None

    def _enqueue_tool_call(self, text: str) -> None:
        self.ingestion_worker.enqueue("system", text, memory_type="tool_call")

    def _enqueue_error(self, text: str) -> None:
        self.ingestion_worker.enqueue("system", text, memory_type="error")

    def _extract_message_role(self, message: Any) -> str:
        if isinstance(message, dict):
            role = message.get("role", "")
        else:
            role = getattr(message, "role", "")
        return str(role).strip().lower()

    def _extract_message_error(self, message: Any) -> str:
        if isinstance(message, dict):
            value = message.get("error_message")
        else:
            value = getattr(message, "error_message", None)
        return str(value).strip() if value else ""

    def _extract_error_text(self, event: Dict[str, Any]) -> str:
        for key in ("error", "message", "reason"):
            value = event.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        result = event.get("result")
        details = getattr(result, "details", None)
        if isinstance(details, dict):
            value = details.get("error")
            if isinstance(value, str) and value.strip():
                return value.strip()
        if isinstance(result, dict):
            details = result.get("details")
            if isinstance(details, dict):
                value = details.get("error")
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ""

    def _log_message_end(self, message: Any) -> None:
        if message is None:
            return
        role = self._extract_message_role(message)
        if role not in {"user", "assistant"}:
            return
        text = "\n".join([x for x in extract_texts(message) if x.strip()]).strip()
        if not text and isinstance(message, dict):
            raw_text = message.get("text")
            if isinstance(raw_text, str):
                text = raw_text.strip()
        if not text:
            return
        if role == "user":
            text = extract_original_user_message(text)
        if text:
            self.ingestion_worker.enqueue(role, text, memory_type="message")

    def _log_message_error(self, message: Any) -> None:
        error_message = self._extract_message_error(message)
        if not error_message:
            return
        self._enqueue_error(f"assistant_error: {error_message}")

    def _log_message_update_toolcall(self, event: Dict[str, Any]) -> None:
        assistant_event = event.get("assistantMessageEvent") or {}
        if not isinstance(assistant_event, dict):
            return
        assistant_event_type = str(assistant_event.get("type", "")).strip()
        if assistant_event_type not in {"toolcall_start", "toolcall_end"}:
            return
        call_id, tool_name, args = self._read_toolcall_preview(assistant_event)
        parts: List[str] = [assistant_event_type, tool_name]
        if call_id:
            parts.append(f"id={call_id}")
        if args not in (None, "", {}):
            parts.append(f"args={args}")
        self._enqueue_tool_call(" ".join(parts))

    def _log_tool_execution(self, event: Dict[str, Any]) -> None:
        etype = str(event.get("type", "")).strip()
        if etype not in {"tool_execution_start", "tool_execution_end", "tool_execution_error"}:
            return
        tool_name = str(event.get("toolName", "")).strip() or "unknown"
        command = extract_bash_exec_command(event)
        if etype == "tool_execution_start":
            line = f"tool_execution_start {tool_name}"
            if command:
                line = f"{line} cmd={command}"
            self._enqueue_tool_call(line)
            return
        if etype == "tool_execution_end":
            is_error = bool(event.get("isError"))
            line = f"tool_execution_end {tool_name} error={is_error}"
            if command:
                line = f"{line} cmd={command}"
            self._enqueue_tool_call(line)
            if is_error:
                error_text = self._extract_error_text(event)
                if error_text:
                    self._enqueue_error(f"tool_execution_error {tool_name}: {error_text}")
                else:
                    self._enqueue_error(f"tool_execution_error {tool_name}")
            return
        error_text = self._extract_error_text(event)
        if error_text:
            self._enqueue_error(f"tool_execution_error {tool_name}: {error_text}")
        else:
            self._enqueue_error(f"tool_execution_error {tool_name}")

    def on_event(self, event: Dict[str, Any]) -> None:
        try:
            etype = str(event.get("type", "")).strip()
            if etype == "message_end":
                message = event.get("message")
                self._log_message_end(message)
                self._log_message_error(message)
                return
            if etype == "message_update":
                self._log_message_update_toolcall(event)
                return
            self._log_tool_execution(event)
        except Exception as exc:
            print(f"[memory] subscriber warning: {exc}")


class ContextCompressor:
    """Compresses older message context into a lightweight summary."""

    def __init__(self, trigger_chars: int = 20_000, target_keep_chars: int = 12_000) -> None:
        self.trigger_chars = max(1, int(trigger_chars))
        self.target_keep_chars = max(1, min(int(target_keep_chars), self.trigger_chars - 1))

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
            long_term.add(summary, session_id=session_id, memory_type="compression_summary")
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
