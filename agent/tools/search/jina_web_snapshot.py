from __future__ import annotations

from html import unescape
from os import getenv
import re
from typing import Any, Dict, Optional

import requests

from ...agent_types import AgentTool, AgentToolResult, TextContent


_DEFAULT_HTTP_TIMEOUT_S = 20.0
_RETRYABLE_STATUS_CODES = {500, 502, 503, 504, 520, 521, 522, 523, 524}
_DIRECT_HTML_TAG_RE = re.compile(r"(?is)<[^>]+>")
_DIRECT_HTML_BREAK_RE = re.compile(r"(?i)<br\s*/?>")
_DIRECT_HTML_BLOCK_CLOSE_RE = re.compile(r"(?i)</(?:p|div|section|article|aside|header|footer|main|li|ul|ol|h[1-6]|tr|table|blockquote)>")
_DIRECT_HTML_DROP_RE = re.compile(r"(?is)<(?:script|style|noscript)[^>]*>.*?</(?:script|style|noscript)>|<!--.*?-->")


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    key = str(value).strip().lower()
    if key in {"1", "true", "yes", "y", "on"}:
        return True
    if key in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _to_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _resolve_http_timeout_s(value: int) -> float:
    if value and value > 0:
        return max(1.0, float(value))
    return _DEFAULT_HTTP_TIMEOUT_S


def _normalize_content_type(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    return raw.split(";", 1)[0].strip()


def _build_snapshot_headers(
    *,
    use_api_key: bool,
    return_format: str,
    timeout: int,
    target_selector: Any,
    wait_for_selector: Any,
    exclude_selector: Any,
    remove_image: bool,
    links_at_end: bool,
    images_at_end: bool,
    json_response: bool,
    image_caption: bool,
    cookie: Any,
) -> Dict[str, str]:
    target_selector = target_selector or []
    wait_for_selector = wait_for_selector or []
    exclude_selector = exclude_selector or []
    api_key = None
    if use_api_key and getenv("JINAAI_API_KEY"):
        api_key = "Bearer " + getenv("JINAAI_API_KEY", "")
    headers: Dict[str, Optional[str]] = {
        "Authorization": api_key,
        "X-Return-Format": None if return_format == "default" else return_format,
        # requests requires header values to be strings, not ints.
        "X-Timeout": str(timeout) if timeout > 0 else None,
        "X-Target-Selector": ",".join(target_selector) if target_selector else None,
        "X-Wait-For-Selector": ",".join(wait_for_selector) if wait_for_selector else None,
        "X-Remove-Selector": ",".join(exclude_selector) if exclude_selector else None,
        "X-Retain-Images": "none" if remove_image else None,
        "X-With-Links-Summary": "true" if links_at_end else None,
        "X-With-Images-Summary": "true" if images_at_end else None,
        "Accept": "application/json" if json_response else None,
        "X-With-Generated-Alt": "true" if image_caption else None,
        "X-Set-Cookie": str(cookie) if cookie else None,
    }
    return {key: value for key, value in headers.items() if value is not None}


def _fetch_snapshot_text(web_url: str, *, headers: Dict[str, str], http_timeout_s: float) -> str:
    response = requests.get(
        f"https://r.jina.ai/{web_url}",
        headers=headers,
        timeout=http_timeout_s,
    )
    response.raise_for_status()
    return response.text


def _html_to_text(html_text: str) -> str:
    text = _DIRECT_HTML_DROP_RE.sub(" ", str(html_text or ""))
    text = _DIRECT_HTML_BREAK_RE.sub("\n", text)
    text = _DIRECT_HTML_BLOCK_CLOSE_RE.sub("\n", text)
    text = _DIRECT_HTML_TAG_RE.sub(" ", text)
    text = unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _fetch_direct_page_text(web_url: str, *, http_timeout_s: float) -> tuple[str, Dict[str, Any]]:
    response = requests.get(
        web_url,
        timeout=http_timeout_s,
        headers={"User-Agent": "POP-Agent/1.0"},
    )
    response.raise_for_status()
    content_type = _normalize_content_type(response.headers.get("Content-Type", ""))
    raw_text = str(getattr(response, "text", "") or "")
    snapshot_text = _html_to_text(raw_text) if content_type in {"text/html", "application/xhtml+xml"} else raw_text.strip()
    if not snapshot_text:
        raise ValueError("direct fetch returned empty text")
    return snapshot_text, {
        "fallback_source": "direct_http",
        "direct_url": str(getattr(response, "url", "") or web_url),
        "direct_content_type": content_type or None,
    }


def _extract_status_code(exc: Exception) -> Optional[int]:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    try:
        if status_code is not None:
            return int(status_code)
    except Exception:
        pass
    match = re.search(r"\b([45]\d{2})\b", str(exc))
    if match:
        return int(match.group(1))
    return None


def _should_retry_without_selectors(
    status_code: Optional[int],
    *,
    target_selector: Any,
    wait_for_selector: Any,
    exclude_selector: Any,
) -> bool:
    if status_code not in _RETRYABLE_STATUS_CODES:
        return False
    return bool(target_selector or wait_for_selector or exclude_selector)


class JinaWebSnapshotTool(AgentTool):
    name = "jina_web_snapshot"
    description = "Fetch a bounded text snapshot for a known webpage URL using POP.utils.web_snapshot."
    parameters = {
        "type": "object",
        "properties": {
            "web_url": {"type": "string", "description": "URL to snapshot"},
            "url": {"type": "string", "description": "Alias for web_url"},
            "max_chars": {"type": "integer", "description": "Optional max returned characters for the snapshot text."},
            "use_api_key": {"type": "boolean"},
            "return_format": {"type": "string"},
            "timeout": {"type": "number"},
            "target_selector": {"type": "array", "items": {"type": "string"}},
            "wait_for_selector": {"type": "array", "items": {"type": "string"}},
            "exclude_selector": {"type": "array", "items": {"type": "string"}},
            "remove_image": {"type": "boolean"},
            "links_at_end": {"type": "boolean"},
            "images_at_end": {"type": "boolean"},
            "json_response": {"type": "boolean"},
            "image_caption": {
                "type": "boolean",
                "description": (
                    "Caption images in the snapshot using AI. Note: this may consume additional tokens and time."
                ),
            },
            "cookie": {"type": "string"},
        },
        "required": ["web_url"],
    }
    label = "Jina Web Snapshot"

    @staticmethod
    def _error(text: str, details: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(type="text", text=text)], details={"ok": False, **details})

    @staticmethod
    def _ok(text: str, details: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(type="text", text=text)], details={"ok": True, **details})

    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        signal: Optional[Any] = None,
        on_update: Optional[Any] = None,
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update
        web_url = str(params.get("web_url") or params.get("url") or "").strip()
        if not web_url:
            return self._error(
                "jina_web_snapshot error: missing web_url",
                {"error": "missing web_url"},
            )

        kwargs = {
            "use_api_key": _to_bool(params.get("use_api_key"), True),
            "return_format": str(params.get("return_format") or "default"),
            "timeout": _to_int(params.get("timeout"), 0),
            "target_selector": params.get("target_selector") or None,
            "wait_for_selector": params.get("wait_for_selector") or None,
            "exclude_selector": params.get("exclude_selector") or None,
            "remove_image": _to_bool(params.get("remove_image"), False),
            "links_at_end": _to_bool(params.get("links_at_end"), False),
            "images_at_end": _to_bool(params.get("images_at_end"), False),
            "json_response": _to_bool(params.get("json_response"), False),
            "image_caption": _to_bool(params.get("image_caption"), False),
            "cookie": params.get("cookie"),
        }
        max_chars = max(1, _to_int(params.get("max_chars"), 12_000))
        http_timeout_s = _resolve_http_timeout_s(int(kwargs["timeout"]))
        retried_without_selectors = False
        retry_reason: Optional[str] = None
        fallback_details: Dict[str, Any] = {}
        try:
            headers = _build_snapshot_headers(
                use_api_key=bool(kwargs["use_api_key"]),
                return_format=str(kwargs["return_format"]),
                timeout=int(kwargs["timeout"]),
                target_selector=kwargs["target_selector"],
                wait_for_selector=kwargs["wait_for_selector"],
                exclude_selector=kwargs["exclude_selector"],
                remove_image=bool(kwargs["remove_image"]),
                links_at_end=bool(kwargs["links_at_end"]),
                images_at_end=bool(kwargs["images_at_end"]),
                json_response=bool(kwargs["json_response"]),
                image_caption=bool(kwargs["image_caption"]),
                cookie=kwargs["cookie"],
            )
            snapshot = _fetch_snapshot_text(web_url, headers=headers, http_timeout_s=http_timeout_s)
        except requests.HTTPError as exc:
            status_code = _extract_status_code(exc)
            if not _should_retry_without_selectors(
                status_code,
                target_selector=kwargs["target_selector"],
                wait_for_selector=kwargs["wait_for_selector"],
                exclude_selector=kwargs["exclude_selector"],
            ):
                try:
                    snapshot, fallback_details = _fetch_direct_page_text(web_url, http_timeout_s=http_timeout_s)
                    fallback_details["jina_error"] = str(exc)
                    fallback_details["jina_status_code"] = status_code
                except Exception as fallback_exc:
                    return self._error(
                        f"jina_web_snapshot error: {exc}",
                        {
                            "error": str(exc),
                            "url": web_url,
                            "http_timeout_s": http_timeout_s,
                            "jina_status_code": status_code,
                            "direct_fallback_error": str(fallback_exc),
                        },
                    )
            else:
                retried_without_selectors = True
                retry_reason = f"http_{status_code}" if status_code is not None else "http_error"
                try:
                    fallback_headers = _build_snapshot_headers(
                        use_api_key=bool(kwargs["use_api_key"]),
                        return_format=str(kwargs["return_format"]),
                        timeout=int(kwargs["timeout"]),
                        target_selector=None,
                        wait_for_selector=None,
                        exclude_selector=None,
                        remove_image=bool(kwargs["remove_image"]),
                        links_at_end=bool(kwargs["links_at_end"]),
                        images_at_end=bool(kwargs["images_at_end"]),
                        json_response=bool(kwargs["json_response"]),
                        image_caption=bool(kwargs["image_caption"]),
                        cookie=kwargs["cookie"],
                    )
                    snapshot = _fetch_snapshot_text(web_url, headers=fallback_headers, http_timeout_s=http_timeout_s)
                except Exception as retry_exc:
                    return self._error(
                        f"jina_web_snapshot error: {retry_exc}",
                        {
                            "error": str(retry_exc),
                            "url": web_url,
                            "http_timeout_s": http_timeout_s,
                            "retried_without_selectors": True,
                            "retry_reason": retry_reason,
                        },
                    )
        except Exception as exc:
            return self._error(
                f"jina_web_snapshot error: {exc}",
                {"error": str(exc), "url": web_url, "http_timeout_s": http_timeout_s},
            )
        snapshot_text = str(snapshot)
        char_count = len(snapshot_text)
        truncated = char_count > max_chars
        if truncated:
            snapshot_text = snapshot_text[:max_chars]
        return self._ok(
            snapshot_text,
            {
                "url": web_url,
                "char_count": char_count,
                "truncated": truncated,
                "max_chars": max_chars,
                "http_timeout_s": http_timeout_s,
                "retried_without_selectors": retried_without_selectors,
                "retry_reason": retry_reason,
                **fallback_details,
            },
        )


# Backward-compatible class name.
WebSnapshotTool = JinaWebSnapshotTool


__all__ = ["JinaWebSnapshotTool", "WebSnapshotTool"]
