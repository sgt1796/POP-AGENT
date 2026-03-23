from __future__ import annotations

import hashlib
import os
import re
import uuid
from typing import Any, Dict, Optional, Sequence
from urllib import parse as urllib_parse

import requests

from ..agent_types import AgentTool, AgentToolResult, TextContent
from .path_roots import normalize_allowed_roots, path_in_roots


_TRUE_WORDS = {"1", "true", "yes", "y", "on"}
_FALSE_WORDS = {"0", "false", "no", "n", "off"}
_HTML_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_HTML_HREF_RE = re.compile(r"""href\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_INTERSTITIAL_TERMS = ("verify", "verification", "captcha", "robot", "sign in", "login", "access denied")


class _DownloadToolError(RuntimeError):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(str(message or "download failed"))
        self.code = str(code or "download_error")
        self.message = str(message or "download failed")
        self.details = dict(details or {})


def _to_text(value: Any) -> str:
    return str(value or "").strip()


def _resolve_output_path(output_path: str, workspace_root: str, allowed_roots: Sequence[str]) -> str:
    raw = _to_text(output_path)
    if not raw:
        raise _DownloadToolError("missing_output_path", "output_path is required")
    resolved = os.path.realpath(raw if os.path.isabs(raw) else os.path.join(workspace_root, raw))
    if not path_in_roots(resolved, allowed_roots):
        raise _DownloadToolError(
            "path_outside_workspace",
            "output_path must be inside workspace or configured allowed roots",
            {"output_path": raw, "workspace_root": workspace_root, "allowed_roots": list(allowed_roots)},
        )
    return resolved


def _to_positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except Exception as exc:
        raise _DownloadToolError(f"invalid_{name}", f"{name} must be an integer") from exc
    if parsed <= 0:
        raise _DownloadToolError(f"invalid_{name}", f"{name} must be > 0")
    return parsed


def _to_positive_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except Exception as exc:
        raise _DownloadToolError(f"invalid_{name}", f"{name} must be a number") from exc
    if parsed <= 0:
        raise _DownloadToolError(f"invalid_{name}", f"{name} must be > 0")
    return parsed


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = _to_text(value).lower()
    if text in _TRUE_WORDS:
        return True
    if text in _FALSE_WORDS:
        return False
    return default


def _normalize_content_type(value: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "application/octet-stream"
    return raw.split(";", 1)[0].strip() or "application/octet-stream"


def _matches_expected_content_type(actual: str, expected: str) -> bool:
    tokens = [item.strip() for item in str(expected or "").split(",")]
    wanted = [_normalize_content_type(item) for item in tokens if item.strip()]
    if not wanted:
        return True
    for item in wanted:
        if item.endswith("/*"):
            prefix = item.split("/", 1)[0]
            if actual.startswith(f"{prefix}/"):
                return True
            continue
        if actual == item:
            return True
    return False


def _extract_html_hints(html_text: str, base_url: str) -> Dict[str, Any]:
    hints: Dict[str, Any] = {}
    match = _HTML_TITLE_RE.search(html_text)
    if match:
        title = " ".join(match.group(1).split()).strip()
        if title:
            hints["html_title"] = title[:240]
    candidates = []
    seen = set()
    for raw_href in _HTML_HREF_RE.findall(html_text):
        href = urllib_parse.urljoin(base_url, raw_href.strip())
        href_lower = href.lower()
        if not href.startswith(("http://", "https://")):
            continue
        if not (
            href_lower.endswith(".pdf")
            or "/pdf" in href_lower
            or ("download" in href_lower and "pdf" in href_lower)
        ):
            continue
        if href in seen:
            continue
        seen.add(href)
        candidates.append(href)
        if len(candidates) >= 5:
            break
    if candidates:
        hints["pdf_link_candidates"] = candidates
    preview = " ".join(html_text.split()).strip()
    if preview:
        hints["content_preview"] = preview[:300]
    return hints


def _content_type_mismatch_details(
    *,
    response: requests.Response,
    expected_content_type: str,
    content_type: str,
    final_url: str,
) -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "expected_content_type": expected_content_type,
        "content_type": content_type,
    }
    if content_type == "text/html":
        try:
            preview_bytes = b"".join(response.iter_content(chunk_size=16 * 1024))
            html_text = preview_bytes.decode(getattr(response, "encoding", None) or "utf-8", errors="replace")
            details.update(_extract_html_hints(html_text, final_url))
        except Exception:
            pass
    return details


def _content_type_mismatch_recovery_hint(
    *,
    content_type: str,
    mismatch_details: Dict[str, Any],
) -> str:
    if content_type != "text/html":
        return ""

    html_title = _to_text(mismatch_details.get("html_title")).lower()
    content_preview = _to_text(mismatch_details.get("content_preview")).lower()
    candidate_links = mismatch_details.get("pdf_link_candidates") or []
    looks_interstitial = any(term in html_title or term in content_preview for term in _INTERSTITIAL_TERMS)

    if candidate_links:
        if looks_interstitial:
            return (
                "looks like a verification/interstitial page; inspect the final URL or source landing page, "
                "then retry one of the exact PDF links before broad search"
            )
        return (
            "treat this as a landing page; inspect the final URL and retry one of the exact PDF links before broad search"
        )

    if looks_interstitial:
        return (
            "looks like a verification/interstitial page; inspect the final URL or source landing page before broad search"
        )
    return "treat this as a landing page; inspect the final URL or source landing page before broad search"


def _infer_expected_content_type(output_path: str) -> str:
    suffix = os.path.splitext(str(output_path or "").strip())[1].lower()
    if suffix == ".pdf":
        return "application/pdf"
    return ""


class DownloadUrlToFileTool(AgentTool):
    name = "download_url_to_file"
    description = (
        "Download an http/https URL to a workspace or allowed-root file path (streaming, redirects supported). "
        "Useful after openalex_works returns best_oa_pdf_url."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "HTTP or HTTPS URL to download."},
            "output_path": {
                "type": "string",
                "description": "Workspace-relative or absolute output path inside the workspace or allowed roots.",
            },
            "timeout_s": {"type": "number", "description": "Optional request timeout in seconds (default 30)."},
            "max_bytes": {"type": "integer", "description": "Optional maximum bytes to write (default 50MB)."},
            "overwrite": {"type": "boolean", "description": "Overwrite output file if it exists (default false)."},
            "expected_content_type": {
                "type": "string",
                "description": (
                    "Optional expected MIME type, e.g. application/pdf. "
                    "Supports wildcard subtype, e.g. application/*."
                ),
            },
        },
        "required": ["url", "output_path"],
    }
    label = "Download URL To File"

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        *,
        default_timeout_s: float = 30.0,
        default_max_bytes: int = 50_000_000,
        allowed_roots: Optional[Sequence[str]] = None,
    ) -> None:
        self.workspace_root, self.allowed_roots = normalize_allowed_roots(workspace_root, allowed_roots)
        self.default_timeout_s = max(0.1, float(default_timeout_s))
        self.default_max_bytes = max(1, int(default_max_bytes))

    @staticmethod
    def _error(text: str, details: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(type="text", text=text)], details={"ok": False, **details})

    @staticmethod
    def _ok(text: str, details: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(type="text", text=text)], details={"ok": True, **details})

    def _download_impl(self, params: Dict[str, Any]) -> AgentToolResult:
        url = _to_text(params.get("url"))
        if not url:
            return self._error("download_url_to_file error: missing url.", {"error": "missing_url"})

        parsed = urllib_parse.urlparse(url)
        scheme = str(parsed.scheme or "").lower()
        if scheme not in {"http", "https"} or not parsed.netloc:
            return self._error(
                "download_url_to_file error: url must be http or https.",
                {"error": "invalid_url", "url": url},
            )

        try:
            output_path = _resolve_output_path(params.get("output_path", ""), self.workspace_root, self.allowed_roots)
            timeout_s = _to_positive_float(params.get("timeout_s", self.default_timeout_s), "timeout_s")
            max_bytes = _to_positive_int(params.get("max_bytes", self.default_max_bytes), "max_bytes")
        except _DownloadToolError as exc:
            return self._error(
                f"download_url_to_file error: {exc.message}",
                {"error": exc.code, "url": url, **exc.details},
            )

        overwrite = _to_bool(params.get("overwrite"), False)
        expected_content_type = _normalize_content_type(_to_text(params.get("expected_content_type", "")))
        if not _to_text(params.get("expected_content_type", "")):
            expected_content_type = _infer_expected_content_type(output_path)

        if os.path.exists(output_path) and not overwrite:
            return self._error(
                "download_url_to_file error: output file already exists and overwrite=false.",
                {"error": "output_exists", "url": url, "output_path": output_path},
            )

        os.makedirs(os.path.dirname(output_path) or self.workspace_root, exist_ok=True)
        temp_path = f"{output_path}.tmp-{uuid.uuid4().hex}"
        bytes_written = 0
        hasher = hashlib.sha256()
        content_type = "application/octet-stream"
        final_url = url

        try:
            with requests.get(url, stream=True, timeout=timeout_s, allow_redirects=True) as response:
                response.raise_for_status()
                final_url = _to_text(getattr(response, "url", "")) or url
                content_type = _normalize_content_type(str(response.headers.get("Content-Type", "")))
                if expected_content_type and not _matches_expected_content_type(content_type, expected_content_type):
                    mismatch_details = _content_type_mismatch_details(
                        response=response,
                        expected_content_type=expected_content_type,
                        content_type=content_type,
                        final_url=final_url,
                    )
                    recovery_hint = _content_type_mismatch_recovery_hint(
                        content_type=content_type,
                        mismatch_details=mismatch_details,
                    )
                    if recovery_hint:
                        mismatch_details["recovery_hint"] = recovery_hint
                    message = (
                        "expected content type "
                        f"'{expected_content_type}' but received '{content_type}'"
                    )
                    html_title = str(mismatch_details.get("html_title") or "").strip()
                    content_preview = str(mismatch_details.get("content_preview") or "").strip()
                    candidate_links = mismatch_details.get("pdf_link_candidates") or []
                    if html_title:
                        message += f"; landing page title: {html_title}"
                    if final_url:
                        message += f"; final_url: {final_url}"
                    if candidate_links:
                        message += f"; pdf_link_candidates: {', '.join(str(item) for item in candidate_links[:3])}"
                    if content_preview:
                        message += f"; content_preview: {content_preview[:160]}"
                    if recovery_hint:
                        message += f"; recovery_hint: {recovery_hint}"
                    raise _DownloadToolError(
                        "unexpected_content_type",
                        message,
                        mismatch_details,
                    )
                with open(temp_path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        next_size = bytes_written + len(chunk)
                        if next_size > max_bytes:
                            raise _DownloadToolError(
                                "file_too_large",
                                f"download exceeded max_bytes={max_bytes}",
                                {"max_bytes": max_bytes, "bytes_written": bytes_written},
                            )
                        handle.write(chunk)
                        hasher.update(chunk)
                        bytes_written = next_size
            os.replace(temp_path, output_path)
        except _DownloadToolError as exc:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
            return self._error(
                f"download_url_to_file error: {exc.message}",
                {
                    "error": exc.code,
                    "url": url,
                    "output_path": output_path,
                    "final_url": final_url,
                    "bytes_written": bytes_written,
                    **exc.details,
                },
            )
        except requests.RequestException as exc:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            error_code = "http_error" if isinstance(status_code, int) else "network_error"
            return self._error(
                f"download_url_to_file error: {exc}",
                {
                    "error": error_code,
                    "url": url,
                    "output_path": output_path,
                    "final_url": final_url,
                    "bytes_written": bytes_written,
                    "status_code": status_code,
                },
            )
        except OSError as exc:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
            return self._error(
                f"download_url_to_file error: {exc}",
                {
                    "error": "write_error",
                    "url": url,
                    "output_path": output_path,
                    "final_url": final_url,
                    "bytes_written": bytes_written,
                },
            )

        return self._ok(
            (
                f"download_url_to_file: saved {bytes_written} bytes to {output_path} "
                f"(content_type={content_type})"
            ),
            {
                "saved_path": output_path,
                "bytes_written": bytes_written,
                "content_type": content_type,
                "final_url": final_url,
                "sha256": hasher.hexdigest(),
                "url": url,
                "timeout_s": timeout_s,
                "max_bytes": max_bytes,
                "overwrite": overwrite,
            },
        )

    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        signal: Optional[Any] = None,
        on_update: Optional[Any] = None,
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update
        return self._download_impl(params)
