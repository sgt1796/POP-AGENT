from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from ...agent_types import AgentTool, AgentToolResult, TextContent

_VALID_RECENCY = {"day", "week", "month", "year"}
_DATE_FIELDS = (
    "from_date",
    "to_date",
    "from_updated_date",
    "to_updated_date",
)


def _to_text(value: Any) -> str:
    return str(value or "").strip()


def _to_optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = int(text)
    except Exception as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if number <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return number


def _validate_iso_date(name: str, value: str) -> str:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except Exception as exc:
        raise ValueError(f"{name} must be in YYYY-MM-DD format") from exc
    return value


def _value(source: Any, key: str, default: Any = "") -> Any:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _first_non_empty(source: Any, keys: Iterable[str]) -> str:
    for key in keys:
        text = _to_text(_value(source, key, ""))
        if text:
            return text
    return ""


class PerplexitySearchTool(AgentTool):
    name = "perplexity_search"
    description = "Search the web with Perplexity and return ranked source results."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query text."},
            "max_results": {"type": "integer", "description": "Maximum number of search results."},
            "max_tokens_per_page": {
                "type": "integer",
                "description": "Maximum tokens to read per page when extracting snippets.",
            },
            "country": {"type": "string", "description": "Optional 2-letter country code (for localization)."},
            "search_domain_filter": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional allowlist of domains to search within.",
            },
            "search_recency_filter": {
                "type": "string",
                "enum": ["day", "week", "month", "year"],
                "description": "Restrict results by recency window.",
            },
            "from_date": {"type": "string", "description": "Earliest publish date, format YYYY-MM-DD."},
            "to_date": {"type": "string", "description": "Latest publish date, format YYYY-MM-DD."},
            "from_updated_date": {"type": "string", "description": "Earliest updated date, format YYYY-MM-DD."},
            "to_updated_date": {"type": "string", "description": "Latest updated date, format YYYY-MM-DD."},
        },
        "required": ["query"],
    }
    label = "Perplexity Search"

    @staticmethod
    def _error(text: str, details: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(type="text", text=text)], details={"ok": False, **details})

    @staticmethod
    def _ok(text: str, details: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(type="text", text=text)], details={"ok": True, **details})

    def _build_payload(self, params: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        query = _to_text(params.get("query"))
        if not query:
            raise ValueError("missing query")

        payload: Dict[str, Any] = {"query": query}
        filters: Dict[str, Any] = {}

        max_results = _to_optional_positive_int(params.get("max_results"), "max_results")
        if max_results is not None:
            payload["max_results"] = max_results
            filters["max_results"] = max_results

        max_tokens_per_page = _to_optional_positive_int(params.get("max_tokens_per_page"), "max_tokens_per_page")
        if max_tokens_per_page is not None:
            payload["max_tokens_per_page"] = max_tokens_per_page
            filters["max_tokens_per_page"] = max_tokens_per_page

        country = _to_text(params.get("country"))
        if country:
            payload["country"] = country
            filters["country"] = country

        raw_domains = params.get("search_domain_filter")
        if raw_domains is not None:
            if not isinstance(raw_domains, list):
                raise ValueError("search_domain_filter must be an array of strings")
            domains = [_to_text(item) for item in raw_domains if _to_text(item)]
            if domains:
                payload["search_domain_filter"] = domains
                filters["search_domain_filter"] = domains

        recency = _to_text(params.get("search_recency_filter")).lower()
        if recency:
            if recency not in _VALID_RECENCY:
                raise ValueError("search_recency_filter must be one of: day, week, month, year")
            payload["search_recency_filter"] = recency
            filters["search_recency_filter"] = recency

        for field in _DATE_FIELDS:
            raw = _to_text(params.get(field))
            if not raw:
                continue
            payload[field] = _validate_iso_date(field, raw)
            filters[field] = payload[field]

        return payload, filters

    def _normalize_results(self, raw_results: Any) -> List[Dict[str, Any]]:
        items: List[Any] = []
        if raw_results is None:
            return []
        if isinstance(raw_results, list):
            items = raw_results
        else:
            try:
                items = list(raw_results)
            except Exception:
                return []

        normalized: List[Dict[str, Any]] = []
        for item in items:
            title = _first_non_empty(item, ("title", "name", "headline"))
            url = _first_non_empty(item, ("url", "link", "source_url"))
            snippet = _first_non_empty(item, ("snippet", "text", "summary", "content", "excerpt"))
            published_date = _first_non_empty(item, ("date", "published_date", "published_at"))
            updated_date = _first_non_empty(item, ("last_updated", "updated_date", "updated_at"))
            normalized.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "date": published_date,
                    "updated_date": updated_date,
                }
            )
        return normalized

    def _format_text(self, query: str, results: List[Dict[str, Any]]) -> str:
        if not results:
            return f"No results returned for query: {query}"

        lines: List[str] = [f"Perplexity search results for: {query}", ""]
        for idx, item in enumerate(results, start=1):
            title = item.get("title") or "(untitled)"
            lines.append(f"{idx}. {title}")
            if item.get("url"):
                lines.append(f"   URL: {item['url']}")
            if item.get("date"):
                lines.append(f"   Date: {item['date']}")
            if item.get("updated_date"):
                lines.append(f"   Updated: {item['updated_date']}")
            if item.get("snippet"):
                lines.append(f"   Snippet: {item['snippet']}")
        return "\n".join(lines).strip()

    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        signal: Optional[Any] = None,
        on_update: Optional[Any] = None,
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update

        try:
            payload, filters = self._build_payload(params)
        except ValueError as exc:
            return self._error(
                f"perplexity_search error: {exc}",
                {"error": str(exc)},
            )

        try:
            from perplexity import Perplexity  # type: ignore
        except Exception:
            return self._error(
                "perplexity_search error: missing SDK. Install `perplexity` and set PERPLEXITY_API_KEY.",
                {"error": "missing_perplexity_sdk"},
            )

        try:
            client = Perplexity()
            search = client.search.create(**payload)
        except Exception as exc:
            return self._error(
                f"perplexity_search error: {exc}",
                {"error": str(exc), "query": payload.get("query"), "filters": filters},
            )

        raw_results = _value(search, "results", [])
        results = self._normalize_results(raw_results)
        text = self._format_text(str(payload.get("query", "")), results)
        return self._ok(
            text,
            {
                "query": payload.get("query"),
                "count": len(results),
                "filters": filters,
                "results": results,
            },
        )


__all__ = ["PerplexitySearchTool"]
