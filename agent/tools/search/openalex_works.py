from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from ...agent_types import AgentTool, AgentToolResult, TextContent

_BASE_WORKS_URL = "https://api.openalex.org/works"
_ACTIONS = {"search", "fetch_openalex_record"}
_TRUE_WORDS = {"1", "true", "yes", "y", "on"}
_FALSE_WORDS = {"0", "false", "no", "n", "off"}
_SELECT_FIELD_ALIASES = {
    "abstract": ("abstract_inverted_index",),
    "authors": ("authorships",),
    "best_oa_landing_url": ("best_oa_location",),
    "best_oa_pdf_url": ("best_oa_location",),
    "default_relevance_score": ("relevance_score",),
    "is_oa": ("open_access",),
    "oa_url": ("open_access",),
    "source": ("primary_location",),
}


def _to_text(value: Any) -> str:
    return str(value or "").strip()


def _to_optional_positive_int(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    text = _to_text(value)
    if not text:
        return None
    try:
        parsed = int(text)
    except Exception as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0")
    return parsed


def _to_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = _to_text(value).lower()
    if text in _TRUE_WORDS:
        return True
    if text in _FALSE_WORDS:
        return False
    raise ValueError(f"{name} must be a boolean")


def _optional_bool(value: Any, name: str) -> Optional[bool]:
    if value is None:
        return None
    text = _to_text(value)
    if not text:
        return None
    if isinstance(value, bool):
        return value
    return _to_bool(value, name)


def _validate_iso_date(name: str, value: str) -> str:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except Exception as exc:
        raise ValueError(f"{name} must be in YYYY-MM-DD format") from exc
    return value


def _to_select(value: Any) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("select must be an array of strings")
    out: List[str] = []
    for item in value:
        field = _to_text(item)
        if field:
            out.append(field)
    return out


def _expand_select_fields(select: List[str]) -> List[str]:
    expanded: List[str] = []
    seen: set[str] = set()
    for field in select:
        mapped_fields = _SELECT_FIELD_ALIASES.get(field, (field,))
        for mapped in mapped_fields:
            normalized = _to_text(mapped)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            expanded.append(normalized)
    return expanded


def _extract_openalex_id(work_id: str) -> str:
    text = _to_text(work_id)
    if not text:
        return ""
    text = text.rstrip("/")
    upper = text.upper()
    if upper.startswith("HTTPS://API.OPENALEX.ORG/WORKS/"):
        return text.rsplit("/", 1)[-1].strip()
    if upper.startswith("HTTP://API.OPENALEX.ORG/WORKS/"):
        return text.rsplit("/", 1)[-1].strip()
    if upper.startswith("HTTPS://OPENALEX.ORG/"):
        return text.rsplit("/", 1)[-1].strip()
    if upper.startswith("HTTP://OPENALEX.ORG/"):
        return text.rsplit("/", 1)[-1].strip()
    if upper.startswith("OPENALEX:"):
        return text.split(":", 1)[1].strip()
    if upper.startswith("W") and upper[1:].isdigit():
        return text
    return ""


def _extract_doi(work_id: str) -> str:
    text = _to_text(work_id)
    if not text:
        return ""
    lower = text.lower()
    if lower.startswith("https://doi.org/"):
        return text[len("https://doi.org/") :].strip()
    if lower.startswith("http://doi.org/"):
        return text[len("http://doi.org/") :].strip()
    if lower.startswith("doi:"):
        return text.split(":", 1)[1].strip()
    if lower.startswith("10."):
        return text
    return ""


def _reconstruct_abstract(abstract_inverted_index: Any) -> str:
    if not isinstance(abstract_inverted_index, dict):
        return ""
    tokens_by_position: Dict[int, str] = {}
    for token, positions in abstract_inverted_index.items():
        if not isinstance(token, str) or not isinstance(positions, list):
            continue
        for pos in positions:
            try:
                idx = int(pos)
            except Exception:
                continue
            if idx < 0:
                continue
            tokens_by_position[idx] = token
    if not tokens_by_position:
        return ""
    ordered = [tokens_by_position[i] for i in sorted(tokens_by_position.keys())]
    return " ".join(ordered).strip()


def _get_nested(source: Any, *keys: str) -> Any:
    current = source
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _normalize_record(item: Dict[str, Any], include_abstract: bool) -> Dict[str, Any]:
    authorships = item.get("authorships")
    authors: List[str] = []
    if isinstance(authorships, list):
        for authorship in authorships:
            if not isinstance(authorship, dict):
                continue
            author_name = _to_text(_get_nested(authorship, "author", "display_name"))
            if author_name:
                authors.append(author_name)

    doi = _to_text(item.get("doi"))
    open_access_oa_url = _to_text(_get_nested(item, "open_access", "oa_url"))
    best_oa_landing_url = _to_text(_get_nested(item, "best_oa_location", "landing_page_url"))
    best_oa_pdf_url = _to_text(_get_nested(item, "best_oa_location", "pdf_url"))
    source_name = _to_text(_get_nested(item, "primary_location", "source", "display_name"))
    if not source_name:
        source_name = _to_text(_get_nested(item, "host_venue", "display_name"))

    relevance_score = item.get("default_relevance_score")
    if relevance_score is None:
        relevance_score = item.get("relevance_score")

    normalized = {
        "id": _to_text(item.get("id")),
        "title": _to_text(item.get("title")),
        "publication_year": item.get("publication_year"),
        "publication_date": _to_text(item.get("publication_date")),
        "cited_by_count": item.get("cited_by_count"),
        "doi": doi,
        "authors": authors,
        "source": source_name,
        "is_oa": _get_nested(item, "open_access", "is_oa"),
        "oa_url": open_access_oa_url,
        "best_oa_landing_url": best_oa_landing_url,
        "best_oa_pdf_url": best_oa_pdf_url,
        "default_relevance_score": relevance_score,
    }
    if include_abstract:
        normalized["abstract"] = _reconstruct_abstract(item.get("abstract_inverted_index"))
    return normalized


def _format_record_line(index: int, record: Dict[str, Any]) -> str:
    title = _to_text(record.get("title")) or "(untitled)"
    year = _to_text(record.get("publication_year"))
    cited = _to_text(record.get("cited_by_count"))
    source = _to_text(record.get("source"))
    line = f"{index}. {title}"
    extras: List[str] = []
    if year:
        extras.append(f"year={year}")
    if cited:
        extras.append(f"cited_by={cited}")
    if source:
        extras.append(f"source={source}")
    if extras:
        line += " [" + ", ".join(extras) + "]"
    return line


class OpenAlexWorksTool(AgentTool):
    name = "openalex_works"
    description = "Search OpenAlex works and fetch individual work records for research papers."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "fetch_openalex_record"],
                "description": "Action to execute.",
            },
            "query": {"type": "string", "description": "Search query text for action=search."},
            "per_page": {"type": "integer", "description": "Results per page for action=search (1-200)."},
            "cursor": {"type": "string", "description": "OpenAlex cursor for pagination (action=search)."},
            "sort": {"type": "string", "description": "OpenAlex sort expression for action=search."},
            "select": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional OpenAlex fields to select.",
            },
            "publication_year_from": {"type": "integer", "description": "Lower bound for publication year."},
            "publication_year_to": {"type": "integer", "description": "Upper bound for publication year."},
            "from_publication_date": {"type": "string", "description": "Earliest publication date YYYY-MM-DD."},
            "to_publication_date": {"type": "string", "description": "Latest publication date YYYY-MM-DD."},
            "type": {"type": "string", "description": "OpenAlex work type filter (e.g. article)."},
            "open_access_only": {"type": "boolean", "description": "Filter to open-access works only."},
            "has_doi": {"type": "boolean", "description": "Filter works by DOI presence."},
            "filter": {"type": "string", "description": "Raw OpenAlex filter string (advanced)."},
            "work_id": {
                "type": "string",
                "description": "OpenAlex work identifier/URL or DOI for action=fetch_openalex_record.",
            },
            "mailto": {"type": "string", "description": "Optional polite-pool email."},
            "timeout_s": {"type": "number", "description": "HTTP timeout seconds (default 20)."},
            "include_abstract": {
                "type": "boolean",
                "description": "Include reconstructed abstract text from abstract_inverted_index.",
            },
        },
        "required": ["action"],
    }
    label = "OpenAlex Works"

    @staticmethod
    def _error(text: str, details: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(type="text", text=text)], details={"ok": False, **details})

    @staticmethod
    def _ok(text: str, details: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(type="text", text=text)], details={"ok": True, **details})

    def _resolve_mailto(self, params: Dict[str, Any]) -> str:
        explicit = _to_text(params.get("mailto"))
        if explicit:
            return explicit
        return _to_text(os.getenv("OPENALEX_EMAIL"))

    def _resolve_timeout(self, params: Dict[str, Any]) -> float:
        timeout = params.get("timeout_s")
        if timeout is None:
            return 20.0
        text = _to_text(timeout)
        if not text:
            return 20.0
        try:
            parsed = float(text)
        except Exception as exc:
            raise ValueError("timeout_s must be a number") from exc
        if parsed <= 0:
            raise ValueError("timeout_s must be > 0")
        return parsed

    def _request_json(self, url: str, query: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
        api_key = _to_text(os.getenv("OPENALEX_API_KEY"))
        outgoing = {k: v for k, v in query.items() if v is not None and _to_text(v) != ""}
        if api_key:
            outgoing["api_key"] = api_key
        full_url = url
        if outgoing:
            full_url = f"{url}?{urllib_parse.urlencode(outgoing, doseq=True)}"
        req = urllib_request.Request(full_url, headers={"Accept": "application/json"})
        try:
            with urllib_request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
        except urllib_error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8")
            except Exception:
                body = ""
            raise RuntimeError(f"OpenAlex HTTP {exc.code}: {body or exc.reason}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"OpenAlex network error: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"OpenAlex response was not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError("OpenAlex response was not an object")
        return data

    def _build_search_filters(self, params: Dict[str, Any]) -> List[str]:
        filters: List[str] = []

        year_from = _to_optional_positive_int(params.get("publication_year_from"), "publication_year_from")
        year_to = _to_optional_positive_int(params.get("publication_year_to"), "publication_year_to")
        if year_from is not None and year_to is not None and year_from > year_to:
            raise ValueError("publication_year_from must be <= publication_year_to")
        if year_from is not None and year_to is not None:
            filters.append(f"publication_year:{year_from}-{year_to}")
        elif year_from is not None:
            filters.append(f"from_publication_date:{year_from}-01-01")
        elif year_to is not None:
            filters.append(f"to_publication_date:{year_to}-12-31")

        from_date = _to_text(params.get("from_publication_date"))
        to_date = _to_text(params.get("to_publication_date"))
        if from_date:
            filters.append(f"from_publication_date:{_validate_iso_date('from_publication_date', from_date)}")
        if to_date:
            filters.append(f"to_publication_date:{_validate_iso_date('to_publication_date', to_date)}")

        work_type = _to_text(params.get("type"))
        if work_type:
            filters.append(f"type:{work_type}")

        open_access_only = _optional_bool(params.get("open_access_only"), "open_access_only")
        if open_access_only is not None:
            filters.append(f"is_oa:{str(open_access_only).lower()}")

        has_doi = _optional_bool(params.get("has_doi"), "has_doi")
        if has_doi is not None:
            filters.append(f"has_doi:{str(has_doi).lower()}")

        raw_filter = _to_text(params.get("filter"))
        if raw_filter:
            filters.append(raw_filter)
        return filters

    def _build_search_request(self, params: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        query = _to_text(params.get("query"))
        filters = self._build_search_filters(params)
        if not query and not filters:
            raise ValueError("search requires query or at least one filter")

        per_page = _to_optional_positive_int(params.get("per_page"), "per_page")
        if per_page is None:
            per_page = 10
        if per_page > 200:
            raise ValueError("per_page must be <= 200")

        requested_select = _to_select(params.get("select"))
        select = _expand_select_fields(requested_select)
        sort = _to_text(params.get("sort"))
        cursor = _to_text(params.get("cursor"))
        mailto = self._resolve_mailto(params)
        timeout_s = self._resolve_timeout(params)
        include_abstract = _optional_bool(params.get("include_abstract"), "include_abstract") is True

        request_params: Dict[str, Any] = {"per-page": per_page}
        summary_filters: Dict[str, Any] = {
            "per_page": per_page,
            "include_abstract": include_abstract,
        }
        if query:
            request_params["search"] = query
        if filters:
            request_params["filter"] = ",".join(filters)
            summary_filters["filter"] = request_params["filter"]
        if sort:
            request_params["sort"] = sort
            summary_filters["sort"] = sort
        if cursor:
            request_params["cursor"] = cursor
            summary_filters["cursor"] = cursor
        if select:
            request_params["select"] = ",".join(select)
            summary_filters["select"] = list(requested_select)
            summary_filters["api_select"] = list(select)
        if mailto:
            request_params["mailto"] = mailto
            summary_filters["mailto"] = mailto
        return request_params, {
            "query": query,
            "filters": summary_filters,
            "timeout_s": timeout_s,
            "include_abstract": include_abstract,
        }

    def _build_fetch_request(self, params: Dict[str, Any]) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
        work_id = _to_text(params.get("work_id"))
        if not work_id:
            raise ValueError("fetch_openalex_record requires work_id")

        requested_select = _to_select(params.get("select"))
        select = _expand_select_fields(requested_select)
        mailto = self._resolve_mailto(params)
        timeout_s = self._resolve_timeout(params)
        include_abstract = _optional_bool(params.get("include_abstract"), "include_abstract") is True

        request_params: Dict[str, Any] = {}
        if select:
            request_params["select"] = ",".join(select)
        if mailto:
            request_params["mailto"] = mailto

        openalex_id = _extract_openalex_id(work_id)
        doi = _extract_doi(work_id)
        if openalex_id:
            target_url = f"{_BASE_WORKS_URL}/{urllib_parse.quote(openalex_id, safe='')}"
            mode = "openalex_id"
            resolved = openalex_id
        elif doi:
            target_url = _BASE_WORKS_URL
            request_params["filter"] = f"doi:https://doi.org/{doi.lower()}"
            request_params["per-page"] = 1
            mode = "doi_lookup"
            resolved = doi
        else:
            raise ValueError("work_id must be an OpenAlex ID/URL or DOI")

        return target_url, request_params, {
            "mode": mode,
            "work_id": work_id,
            "resolved_work_id": resolved,
            "timeout_s": timeout_s,
            "include_abstract": include_abstract,
            "select": list(requested_select),
            "api_select": list(select),
            "mailto": mailto,
            "doi_lookup_filter": request_params.get("filter"),
        }

    def _format_search_text(self, query: str, records: List[Dict[str, Any]]) -> str:
        if not records:
            if query:
                return f"OpenAlex search returned no results for: {query}"
            return "OpenAlex search returned no results."
        lines = [f"OpenAlex search results ({len(records)}):", ""]
        for idx, record in enumerate(records, start=1):
            lines.append(_format_record_line(idx, record))
            doi = _to_text(record.get("doi"))
            if doi:
                lines.append(f"   DOI: {doi}")
            open_url = _to_text(record.get("best_oa_pdf_url") or record.get("best_oa_landing_url") or record.get("oa_url"))
            if open_url:
                lines.append(f"   OA: {open_url}")
        return "\n".join(lines).strip()

    def _format_fetch_text(self, record: Dict[str, Any]) -> str:
        title = _to_text(record.get("title")) or "(untitled)"
        lines = [f"OpenAlex record: {title}"]
        for key, label in (
            ("id", "ID"),
            ("publication_year", "Year"),
            ("publication_date", "Publication Date"),
            ("cited_by_count", "Cited By"),
            ("doi", "DOI"),
            ("source", "Source"),
        ):
            value = record.get(key)
            text = _to_text(value)
            if text:
                lines.append(f"{label}: {text}")
        if isinstance(record.get("authors"), list) and record["authors"]:
            lines.append(f"Authors: {', '.join(record['authors'])}")
        for key, label in (
            ("oa_url", "OA URL"),
            ("best_oa_landing_url", "Best OA Landing"),
            ("best_oa_pdf_url", "Best OA PDF"),
        ):
            text = _to_text(record.get(key))
            if text:
                lines.append(f"{label}: {text}")
        abstract = _to_text(record.get("abstract"))
        if abstract:
            lines.append(f"Abstract: {abstract}")
        return "\n".join(lines).strip()

    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        signal: Optional[Any] = None,
        on_update: Optional[Any] = None,
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update
        action = _to_text(params.get("action")).lower()
        if action not in _ACTIONS:
            return self._error(
                "openalex_works error: action must be one of search, fetch_openalex_record",
                {"error": "invalid_action", "action": action or None},
            )

        if action == "search":
            request_params: Dict[str, Any] = {}
            try:
                request_params, summary = self._build_search_request(params)
                timeout_s = float(summary["timeout_s"])
                include_abstract = bool(summary["include_abstract"])
                payload = self._request_json(_BASE_WORKS_URL, request_params, timeout_s=timeout_s)
            except ValueError as exc:
                return self._error(f"openalex_works error: {exc}", {"action": action, "error": str(exc)})
            except Exception as exc:
                return self._error(
                    f"openalex_works error: {exc}",
                    {"action": action, "error": str(exc), "request": request_params},
                )

            results = payload.get("results")
            if not isinstance(results, list):
                results = []
            normalized = [_normalize_record(item, include_abstract=include_abstract) for item in results if isinstance(item, dict)]
            meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
            next_cursor = _to_text(meta.get("next_cursor"))
            text = self._format_search_text(_to_text(summary.get("query")), normalized)
            return self._ok(
                text,
                {
                    "action": action,
                    "query": summary.get("query"),
                    "count": len(normalized),
                    "next_cursor": next_cursor or None,
                    "meta": meta,
                    "filters": summary.get("filters", {}),
                    "results": normalized,
                },
            )

        try:
            target_url, request_params, summary = self._build_fetch_request(params)
            timeout_s = float(summary["timeout_s"])
            include_abstract = bool(summary["include_abstract"])
            payload = self._request_json(target_url, request_params, timeout_s=timeout_s)
        except ValueError as exc:
            return self._error(f"openalex_works error: {exc}", {"action": action, "error": str(exc)})
        except Exception as exc:
            return self._error(
                f"openalex_works error: {exc}",
                {"action": action, "error": str(exc)},
            )

        record_payload: Dict[str, Any]
        if summary.get("mode") == "doi_lookup":
            results = payload.get("results")
            if not isinstance(results, list) or not results:
                return self._error(
                    "openalex_works error: no OpenAlex work found for DOI",
                    {"action": action, "error": "doi_not_found", **summary},
                )
            first = results[0]
            if not isinstance(first, dict):
                return self._error(
                    "openalex_works error: DOI lookup returned invalid record",
                    {"action": action, "error": "invalid_record", **summary},
                )
            record_payload = first
        else:
            record_payload = payload

        normalized = _normalize_record(record_payload, include_abstract=include_abstract)
        return self._ok(
            self._format_fetch_text(normalized),
            {
                "action": action,
                **summary,
                "record": normalized,
            },
        )


__all__ = ["OpenAlexWorksTool"]
