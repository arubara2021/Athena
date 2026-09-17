from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from core.models import Source, SourcePlatform, SourceType
from utils.logger import get_logger
from utils.text import normalize_title
from utils.url import get_url_fingerprint

_ARXIV_URL_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v[0-9]+)?",
    re.IGNORECASE,
)
_ARXIV_PREFIX_RE = re.compile(
    r"arxiv[./:]([0-9]{4}\.[0-9]{4,5})(?:v[0-9]+)?",
    re.IGNORECASE,
)
_ARXIV_BARE_RE = re.compile(
    r"^([0-9]{4}\.[0-9]{4,5})(?:v[0-9]+)?$",
)
_ARXIV_OLD_RE = re.compile(
    r"^(?:[a-z\-]+(?:\.[a-z\-]+)?/[0-9]{7})(?:v[0-9]+)?$",
)


class SourceDeduplicator:
    def __init__(
        self,
        max_alternate_urls: int = 5,
        fuzzy_title_threshold: float = 0.85,
    ) -> None:
        self._max_alternate_urls = max(1, max_alternate_urls)
        self._fuzzy_title_threshold = max(0.5, min(1.0, fuzzy_title_threshold))
        self._logger = get_logger("search.dedupe")

    def deduplicate(self, sources: list[Source]) -> list[Source]:
        groups: dict[str, list[Source]] = {}
        order: list[str] = []
        for source in sources:
            if not isinstance(source, Source):
                continue
            key = self._dedupe_key(source)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(source)

        deduplicated: list[Source] = []
        for key in order:
            merged = self._merge_group(groups[key])
            if merged is not None:
                deduplicated.append(merged)

        deduplicated = self._title_dedup_pass(deduplicated)
        deduplicated = self._fuzzy_title_dedup_pass(deduplicated)
        return deduplicated

    def _title_dedup_pass(self, sources: list[Source]) -> list[Source]:
        title_groups: dict[str, list[Source]] = {}
        title_order: list[str] = []
        for source in sources:
            source_type = self._enum_value(source.source_type)
            if source_type != SourceType.PAPER.value:
                continue
            title_key = normalize_title(source.title)
            if not title_key or len(title_key) < 8:
                continue
            if title_key not in title_groups:
                title_groups[title_key] = []
                title_order.append(title_key)
            title_groups[title_key].append(source)

        merged_ids: set[str] = set()
        for title_key in title_order:
            group = title_groups[title_key]
            if len(group) <= 1:
                continue
            best = max(group, key=self._quality_score)
            for source in group:
                if source.source_id != best.source_id:
                    merged_ids.add(source.source_id)

        result: list[Source] = []
        seen_ids: set[str] = set()
        for source in sources:
            if source.source_id in merged_ids:
                continue
            if source.source_id in seen_ids:
                continue
            seen_ids.add(source.source_id)
            result.append(source)
        return result

    def _fuzzy_title_dedup_pass(self, sources: list[Source]) -> list[Source]:
        papers = [
            s for s in sources
            if self._enum_value(s.source_type) == SourceType.PAPER.value
        ]
        non_papers = [
            s for s in sources
            if self._enum_value(s.source_type) != SourceType.PAPER.value
        ]

        merged_ids: set[str] = set()
        for i in range(len(papers)):
            if papers[i].source_id in merged_ids:
                continue
            for j in range(i + 1, len(papers)):
                if papers[j].source_id in merged_ids:
                    continue
                similarity = self._title_similarity(
                    papers[i].title, papers[j].title
                )
                if similarity >= self._fuzzy_title_threshold:
                    loser = papers[j]
                    if self._quality_score(papers[j]) > self._quality_score(papers[i]):
                        loser = papers[i]
                    merged_ids.add(loser.source_id)
                    self._logger.info(
                        f"Fuzzy title dedup merged '{loser.title[:60]}' "
                        f"(similarity={similarity:.2f})"
                    )

        result: list[Source] = []
        for source in sources:
            if source.source_id not in merged_ids:
                result.append(source)
        return result

    def _title_similarity(self, title_a: str, title_b: str) -> float:
        norm_a = normalize_title(title_a)
        norm_b = normalize_title(title_b)
        if not norm_a or not norm_b:
            return 0.0
        if norm_a == norm_b:
            return 1.0
        return SequenceMatcher(None, norm_a, norm_b).ratio()

    def _dedupe_key(self, source: Source) -> str:
        metadata = source.metadata if isinstance(source.metadata, dict) else {}

        arxiv_id = self._extract_arxiv_id(source)
        if arxiv_id:
            return f"arxiv:{arxiv_id}"

        doi = self._extract_doi(metadata, source.url)
        if doi:
            return f"doi:{doi.lower()}"

        title = normalize_title(source.title)
        if not title:
            try:
                return f"url:{get_url_fingerprint(source.url)}"
            except Exception:
                return f"raw:{source.source_id}"

        source_type = self._enum_value(source.source_type)
        if source_type == SourceType.REPOSITORY.value:
            return f"repo:{title}"
        return f"{source_type}:{title}"

    def _extract_arxiv_id(self, source: Source) -> str | None:
        metadata = source.metadata if isinstance(source.metadata, dict) else {}
        values: list[str] = []

        for key in ("arxiv_id", "openalex_id", "id"):
            value = metadata.get(key)
            if value:
                values.append(str(value))

        external_ids = metadata.get("external_ids")
        if isinstance(external_ids, dict):
            for key in ("ArXiv", "arxiv"):
                value = external_ids.get(key)
                if value:
                    values.append(str(value))

        if source.url:
            values.append(str(source.url))

        for value in values:
            cleaned = str(value).lower().strip()

            match = _ARXIV_URL_RE.search(cleaned)
            if match:
                return match.group(1)

            match = _ARXIV_PREFIX_RE.search(cleaned)
            if match:
                return match.group(1)

            if not cleaned.startswith("10."):
                match = _ARXIV_BARE_RE.match(cleaned)
                if match:
                    return match.group(1)

            match = _ARXIV_OLD_RE.match(cleaned)
            if match:
                return match.group(0)

        return None

    def _extract_doi(
        self,
        metadata: dict[str, Any],
        url: str = "",
    ) -> str | None:
        doi = metadata.get("doi")
        if not doi:
            external_ids = metadata.get("external_ids")
            if isinstance(external_ids, dict):
                doi = external_ids.get("DOI") or external_ids.get("doi")

        if not doi and url:
            match = re.search(
                r"doi\.org/(10\.[^\s]+)",
                url.lower(),
            )
            if match:
                doi = match.group(1)

        if not doi:
            return None

        cleaned = str(doi).strip().lower()
        cleaned = cleaned.replace("https://doi.org/", "")
        cleaned = cleaned.replace("http://doi.org/", "")
        cleaned = cleaned.replace("doi:", "")
        cleaned = cleaned.replace("doi.org/", "")

        if not cleaned:
            return None
        if "arxiv" in cleaned:
            return None
        if not cleaned.startswith("10."):
            return None
        return cleaned

    def _merge_group(self, group: list[Source]) -> Source | None:
        if not group:
            return None
        if len(group) == 1:
            return group[0]

        ranked = sorted(group, key=self._quality_score, reverse=True)
        primary = ranked[0]
        others = ranked[1:]

        try:
            data = primary.model_dump()
            authors = list(primary.authors or [])
            seen_authors = {author.lower() for author in authors}
            alternate_urls = [primary.url]
            seen_urls = {primary.url}
            abstract = primary.abstract or ""
            citation_count = primary.citation_count
            has_code = primary.has_code
            year = primary.year
            published_at = primary.published_at
            metadata = dict(primary.metadata or {})

            for other in others:
                if other.url and other.url not in seen_urls:
                    seen_urls.add(other.url)
                    alternate_urls.append(other.url)

                for author in other.authors or []:
                    author_key = author.lower()
                    if author_key not in seen_authors:
                        seen_authors.add(author_key)
                        authors.append(author)

                if other.abstract and len(other.abstract) > len(abstract):
                    abstract = other.abstract

                if other.citation_count is not None:
                    if citation_count is None or other.citation_count > citation_count:
                        citation_count = other.citation_count

                if other.has_code is True:
                    has_code = True

                if other.year is not None and year is None:
                    year = other.year

                if other.published_at is not None and published_at is None:
                    published_at = other.published_at

                metadata = self._merge_metadata(metadata, other.metadata)

            metadata["duplicate_count"] = len(group)
            metadata["alternate_urls"] = alternate_urls[: self._max_alternate_urls]

            data["authors"] = authors
            data["abstract"] = abstract or None
            data["citation_count"] = citation_count
            data["has_code"] = has_code
            data["year"] = year
            data["published_at"] = published_at
            data["metadata"] = metadata

            return Source.model_validate(data)
        except Exception as exc:
            self._logger.warning(f"Deduplication merge failed, using primary source: {exc}")
            return primary

    def _merge_metadata(
        self,
        primary: dict[str, Any] | None,
        other: dict[str, Any] | None,
    ) -> dict[str, Any]:
        merged = dict(primary or {})
        if not isinstance(other, dict):
            return merged
        for key, value in other.items():
            if key not in merged or merged[key] in (None, "", [], {}):
                merged[key] = value
        return merged

    def _quality_score(self, source: Source) -> float:
        score = 0.0
        if source.abstract:
            score += min(len(source.abstract) / 1000, 2.0)
        if source.citation_count is not None:
            score += min(source.citation_count / 100, 3.0)
        if source.authors:
            score += min(len(source.authors), 5) * 0.1
        if source.published_at is not None or source.year is not None:
            score += 0.3
        if source.has_code is True:
            score += 0.5

        metadata = source.metadata if isinstance(source.metadata, dict) else {}
        stars = metadata.get("stars")
        if isinstance(stars, int):
            score += min(stars / 1000, 2.0)
        downloads = metadata.get("downloads")
        if isinstance(downloads, int):
            score += min(downloads / 10000, 1.0)
        if metadata.get("has_pdf") is True:
            score += 0.3

        platform_priority = {
            SourcePlatform.SEMANTIC_SCHOLAR.value: 0.5,
            SourcePlatform.ARXIV.value: 0.45,
            SourcePlatform.OPENALEX.value: 0.4,
            SourcePlatform.GITHUB.value: 0.35,
            SourcePlatform.HUGGINGFACE.value: 0.3,
            SourcePlatform.WIKIPEDIA.value: 0.25,
            SourcePlatform.WEB.value: 0.1,
        }
        score += platform_priority.get(self._enum_value(source.platform), 0.0)
        return score

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value))