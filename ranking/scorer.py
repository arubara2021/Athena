from __future__ import annotations
import difflib
import math
import re
from datetime import datetime, timezone
from typing import Any
from core.models import Difficulty, RankedSource, Source
from core.ranking_config import get_ranking_config
from core.schemas import SearchQuerySchema
from utils.logger import get_logger
from utils.text import clean_text



try:
    from search.query_tokens import concept_tokens, tokenize
except Exception:
    def tokenize(value: Any) -> list[str]:
        return re.findall(r"[a-z0-9]+", str(value or "").lower())
    def concept_tokens(value: Any) -> set[str]:
        return set(tokenize(value))

_STOPWORDS = {
    "about", "and", "are", "can", "could", "for", "from", "how", "into",
    "is", "it", "its", "of", "on", "or", "should", "that", "the", "their",
    "then", "there", "these", "they", "this", "those", "to", "under",
    "use", "used", "using", "was", "were", "what", "when", "where",
    "which", "while", "who", "why", "will", "with", "would", "your",
    "learn", "learning", "source", "sources", "best", "top", "find",
    "please", "want", "need", "help", "give", "show", "make",
}

_PLATFORM_AUTHORITY = {
    "arxiv": 0.95,
    "semantic_scholar": 0.92,
    "openalex": 0.88,
    "core": 0.86,
    "crossref": 0.90,
    "europe_pmc": 0.90,
    "pubmed": 0.92,
    "doaj": 0.84,
    "zenodo": 0.78,
    "datacite": 0.76,
    "github": 0.86,
    "huggingface": 0.84,
    "wikipedia": 0.82,
    "wikibooks": 0.78,
    "wikiversity": 0.76,
    "openstax": 0.88,
    "mit_ocw": 0.90,
    "libretexts": 0.84,
    "open_library": 0.78,
    "internet_archive": 0.76,
    "stack_exchange": 0.74,
    "hacker_news": 0.66,
    "web": 0.58,
    "tavily": 0.60,
    "exa": 0.62,
    "serper": 0.60,
    "serpapi": 0.60,
    "jina": 0.58,
}

_TYPE_QUALITY = {
    "research_paper": 0.92,
    "course": 0.90,
    "book": 0.88,
    "documentation": 0.84,
    "repository": 0.82,
    "model": 0.80,
    "dataset": 0.72,
    "video": 0.68,
    "blog": 0.55,
    "other": 0.45,
}

_EDUCATION_PLATFORMS = {
    "openstax",
    "libretexts",
    "mit_ocw",
    "wikibooks",
    "wikipedia",
    "open_library",
    "wikiversity",
    "khan_academy",
}

_RANGE_PHRASES: tuple[str, ...] = (
    "to advanced",
    "to advance",
    "to expert",
    "to expert level",
    "from basics",
    "from the basics",
    "from scratch",
    "zero to hero",
    "start to finish",
    "basics to advanced",
    "beginner to advanced",
    "beginner to expert",
    "foundations to advanced",
    "fundamentals to advanced",
    "introduction to advanced",
    "intro to advanced",
)

_BEGINNER_MARKERS: tuple[str, ...] = (
    "beginner",
    "beginners",
    "basic",
    "basics",
    "fundamental",
    "fundamentals",
    "intro",
    "introduction",
    "introductory",
    "from zero",
    "no prior",
)

_ADVANCED_MARKERS: tuple[str, ...] = (
    "advanced",
    "advance",
    "advances",
    "expert",
    "experts",
    "in depth",
    "in-depth",
    "deep dive",
    "deep-dive",
    "state of the art",
    "state-of-the-art",
    "cutting edge",
    "cutting-edge",
)

class HeuristicScorer:
    def __init__(self, config: Any | None = None) -> None:
        self._config = config or self._load_config()
        self._logger = get_logger("ranking.scorer")
        self._corpus_stats: dict[str, Any] | None = None

    def score_sources(
        self,
        sources: list[Any],
        query: SearchQuerySchema | str | dict[str, Any] | None = None,
        goal: str = "",
        level: str = "",
    ) -> list[RankedSource]:
        query_schema = self._coerce_query(query)
        query_info = self._query_info(query_schema)
        
        self._compute_corpus_stats(sources)
        
        ranked: list[RankedSource] = []
        for source in sources:
            if not isinstance(source, Source):
                continue
            try:
                score = self.score_source(
                    source,
                    query=query_schema,
                    goal=goal,
                    level=level,
                )
                confidence = self._confidence(source, query_info, score)
                ranked.append(
                    RankedSource(
                        source=source,
                        rank=1,
                        score=score,
                        confidence=confidence,
                        reason="advanced_heuristic",
                    )
                )
            except Exception as exc:
                self._logger.warning(f"Scoring failed for source: {exc}")
                continue
        ranked.sort(key=lambda item: item.score, reverse=True)
        ordered: list[RankedSource] = []
        for index, item in enumerate(ranked, start=1):
            ordered.append(
                RankedSource(
                    source=item.source,
                    rank=index,
                    score=item.score,
                    confidence=item.confidence,
                    reason=item.reason,
                )
            )
        return ordered

    def _compute_corpus_stats(self, sources: list[Any]) -> None:
        total_len = 0
        doc_count = 0
        df: dict[str, int] = {}
        
        for source in sources:
            if not isinstance(source, Source):
                continue
            text = f"{source.title} {source.abstract or ''}".lower()
            tokens = self._tokenize(text)
            if not tokens:
                continue
            
            doc_count += 1
            total_len += len(tokens)
            
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] = df.get(token, 0) + 1
                
        self._corpus_stats = {
            "avgdl": total_len / doc_count if doc_count > 0 else 1.0,
            "df": df,
            "N": doc_count
        }

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _detect_request_intent(
        self,
        level_text: str,
        goal_text: str,
    ) -> tuple[bool, bool, bool]:
        combined = f"{level_text} {goal_text}".strip()

        is_range = any(
            phrase in combined for phrase in _RANGE_PHRASES
        )

        has_beginner = any(
            marker in level_text or marker in goal_text
            for marker in _BEGINNER_MARKERS
        )

        has_advanced = any(
            marker in level_text or marker in goal_text
            for marker in _ADVANCED_MARKERS
        )

        return has_beginner, has_advanced, is_range

    def score_source(
        self,
        source: Source,
        query: SearchQuerySchema | str | dict[str, Any] | None = None,
        goal: str = "",
        level: str = "",
    ) -> float:
        query_schema = self._coerce_query(query)
        query_info = self._query_info(query_schema)
        weights = self._scorer_weights()
        
        relevance = self._relevance(source, query_info)
        authority = self._authority(source)
        type_quality = self._type_quality(source)
        impact = self._impact(source)
        recency = self._recency(source)
        content = self._content(source)
        support = self._support(source)
        authors = self._authors(source)
        
        base_score = (
            weights["relevance"] * relevance
            + weights["authority"] * authority
            + weights["type_quality"] * type_quality
            + weights["impact"] * impact
            + weights["recency"] * recency
            + weights["content"] * content
            + weights["support"] * support
            + weights["authors"] * authors
        )
        
        resolved_goal = clean_text(goal) or clean_text(
            str(getattr(query_schema, "goal", "") or "")
        )
        resolved_level = clean_text(level) or clean_text(
            str(getattr(query_schema, "level", "") or "")
        )

        alignment = self.compute_alignment(
            source,
            resolved_goal,
            resolved_level,
        )
        
        final_score = base_score + alignment

        level_text = resolved_level.lower()
        goal_text = resolved_goal.lower()

        has_beginner, has_advanced, is_range = self._detect_request_intent(
            level_text,
            goal_text,
        )

        is_mixed = is_range or (has_beginner and has_advanced)

        if not is_mixed and (has_beginner or has_advanced):
            grade = self._grade_level(source)
            difficulty = self._difficulty_value(source)

            if has_beginner and (
                difficulty == Difficulty.ADVANCED or grade >= 14.0
            ):
                final_score *= 0.2
            elif has_advanced and (
                difficulty == Difficulty.BEGINNER or grade < 8.0
            ):
                final_score *= 0.5

        return max(0.0, min(1.0, final_score))

    def _grade_level(self, source: Source) -> float:
        text = f"{source.title}. {source.abstract or ''}".strip()
        if not text:
            return 12.0

        words = re.findall(r"\b\w+\b", text)
        if not words:
            return 12.0

        sentences = re.split(r"[.!?]+", text)
        sentences = [s for s in sentences if s.strip()]
        if not sentences:
            return 12.0

        complex_words = 0
        for word in words:
            syllables = len(re.findall(r"[aeiouy]+", word.lower()))
            if syllables >= 3:
                complex_words += 1

        avg_sentence_len = len(words) / len(sentences)
        complex_percent = (complex_words / len(words)) * 100
        fog = 0.4 * (avg_sentence_len + complex_percent)
        return max(0.0, min(20.0, fog))

    def compute_alignment(
        self,
        source: Source,
        goal: str,
        level: str,
    ) -> float:
        level_text = clean_text(level).lower()
        goal_text = clean_text(goal).lower()
        
        if not level_text and not goal_text:
            return 0.0
            
        beginner = (
            "beginner" in level_text
            or "beginner" in goal_text
            or "basics" in goal_text
            or "basic" in goal_text
        )
        advanced = "advanced" in level_text or "advanced" in goal_text
        
        if not beginner and not advanced:
            return 0.0
            
        difficulty = self._difficulty_value(source)
        source_type = self._type_value(source)
        platform = self._enum_value(getattr(source, "platform", "")).lower()
        grade = self._grade_level(source)
        
        settings = self._alignment_settings()
        cap = float(getattr(settings, "cap", 0.30))
        
        adjustment = 0.0
        
        if beginner and advanced:
            difficulty_map = {
                Difficulty.BEGINNER: 0.12,
                Difficulty.INTERMEDIATE: 0.10,
                Difficulty.ADVANCED: 0.12,
                Difficulty.UNKNOWN: 0.02,
            }
            type_map = {
                "documentation": 0.10,
                "course": 0.10,
                "book": 0.10,
                "repository": 0.05,
                "model": 0.05,
                "dataset": 0.04,
                "research_paper": 0.08,
            }
            cap = min(cap, 0.22)
        elif beginner:
            difficulty_map = self._map_or_default(
                getattr(settings, "beginner_difficulty", None),
                {
                    Difficulty.BEGINNER: 0.20,
                    Difficulty.INTERMEDIATE: 0.06,
                    Difficulty.ADVANCED: -0.24,
                    Difficulty.UNKNOWN: 0.00,
                },
            )
            type_map = self._map_or_default(
                getattr(settings, "beginner_source_type", None),
                {
                    "documentation": 0.10,
                    "course": 0.10,
                    "book": 0.10,
                    "video": 0.08,
                    "blog": 0.05,
                    "repository": 0.04,
                    "research_paper": -0.06,
                },
            )
            
            if platform in _EDUCATION_PLATFORMS:
                adjustment += 0.15
                
            if 8.0 <= grade <= 11.0:
                adjustment += 0.10
            elif grade >= 14.0:
                adjustment -= 0.15
                
        else:
            difficulty_map = self._map_or_default(
                getattr(settings, "advanced_difficulty", None),
                {
                    Difficulty.ADVANCED: 0.12,
                    Difficulty.INTERMEDIATE: 0.03,
                    Difficulty.BEGINNER: -0.10,
                    Difficulty.UNKNOWN: 0.00,
                },
            )
            type_map = self._map_or_default(
                getattr(settings, "advanced_source_type", None),
                {
                    "research_paper": 0.05,
                    "repository": 0.04,
                    "model": 0.04,
                    "dataset": 0.03,
                    "book": 0.02,
                },
            )
            
            if grade >= 14.0:
                adjustment += 0.05
            elif grade < 10.0:
                adjustment -= 0.05
                
        adjustment += float(self._map_value(difficulty_map, difficulty, 0.0))
        adjustment += float(self._map_value(type_map, source_type, 0.0))
        
        return max(-cap, min(cap, adjustment))

    def _relevance(self, source: Source, query_info: dict[str, Any]) -> float:
        if self._corpus_stats:
            return self._bm25_relevance(source, query_info)
        return self._token_overlap_relevance(source, query_info)

    def _bm25_relevance(self, source: Source, query_info: dict[str, Any]) -> float:
        k1 = 1.5
        b = 0.75
        query_tokens = query_info.get("tokens", set())
        
        if not query_tokens:
            return 0.50
            
        doc_text = f"{source.title} {source.abstract or ''}".lower()
        doc_tokens = self._tokenize(doc_text)
        
        if not doc_tokens:
            return 0.0
            
        dl = len(doc_tokens)
        avgdl = self._corpus_stats["avgdl"]
        N = self._corpus_stats["N"]
        df = self._corpus_stats["df"]
        
        tf: dict[str, int] = {}
        for t in doc_tokens:
            tf[t] = tf.get(t, 0) + 1
            
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
                
            term_freq = tf[term]
            doc_freq = df.get(term, 0)
            
            idf = math.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
            
            numerator = term_freq * (k1 + 1)
            denominator = term_freq + k1 * (1 - b + b * (dl / avgdl))
            score += idf * (numerator / denominator)
            
        corpus_size = max(1, int(self._corpus_stats.get("N", 0) or 0))
        if corpus_size <= 50:
            denom = 1.0
        elif corpus_size <= 500:
            denom = 2.5
        else:
            denom = 5.0
        normalized = score / (score + denom)
        return max(0.0, min(1.0, normalized))

    def _token_overlap_relevance(self, source: Source, query_info: dict[str, Any]) -> float:
        query_tokens = query_info.get("tokens", set())
        phrases = query_info.get("phrases", [])
        
        if not query_tokens:
            return 0.50
            
        title_text = clean_text(getattr(source, "title", "")).lower()
        abstract_text = clean_text(getattr(source, "abstract", "")).lower()
        metadata_text = self._metadata_text(source).lower()
        full_text = f"{title_text}\n{abstract_text}\n{metadata_text}"
        
        title_tokens = set(self._tokenize(title_text))
        abstract_tokens = set(self._tokenize(abstract_text))
        metadata_tokens = set(self._tokenize(metadata_text))
        
        title_score = self._token_overlap(query_tokens, title_tokens)
        abstract_score = self._token_overlap(query_tokens, abstract_tokens)
        metadata_score = self._token_overlap(query_tokens, metadata_tokens)
        phrase_bonus = self._phrase_bonus(phrases, title_text, full_text)
        
        score = (
            0.55 * title_score
            + 0.30 * abstract_score
            + 0.15 * metadata_score
            + phrase_bonus
        )
        
        if title_score >= 0.99:
            score += 0.10
            
        return max(0.0, min(1.0, score))

    def relevance_factor(
        self,
        source: Source,
        query: SearchQuerySchema | str | dict[str, Any] | None,
    ) -> float:
        query_schema = self._coerce_query(query)
        query_info = self._query_info(query_schema)
        return self._relevance(source, query_info)

    def _load_config(self) -> Any:
        try:
            return get_ranking_config()
        except Exception:
            return None

    def _coerce_query(
        self,
        query: SearchQuerySchema | str | dict[str, Any] | None,
    ) -> Any:
        if query is None:
            return None
        if isinstance(query, SearchQuerySchema):
            return query
        if isinstance(query, str):
            try:
                return SearchQuerySchema(topic=query)
            except Exception:
                return {"topic": query}
        if isinstance(query, dict):
            try:
                return SearchQuerySchema.model_validate(query)
            except Exception:
                return query
        return None

    def _query_info(self, query: Any) -> dict[str, Any]:
        phrases: list[str] = []
        tokens: set[str] = set()
        
        def add_text(value: Any) -> None:
            text = clean_text(value)
            if not text:
                return
            lowered = text.lower()
            if lowered not in phrases and len(lowered) >= 3:
                phrases.append(lowered)
            try:
                raw_tokens = concept_tokens(text)
            except Exception:
                raw_tokens = tokenize(text)
            for token in raw_tokens:
                token_text = str(token).lower()
                if token_text and token_text not in _STOPWORDS:
                    tokens.add(token_text)
                    
        if isinstance(query, SearchQuerySchema):
            add_text(getattr(query, "topic", ""))
            add_text(getattr(query, "goal", ""))
            add_text(getattr(query, "primary_concept", ""))
            add_text(getattr(query, "corrected_topic", ""))
            for keyword in getattr(query, "keywords", []) or []:
                add_text(keyword)
            for short_query in getattr(query, "short_search_queries", []) or []:
                add_text(short_query)
        elif isinstance(query, dict):
            add_text(query.get("topic", ""))
            add_text(query.get("goal", ""))
            add_text(query.get("primary_concept", ""))
            add_text(query.get("corrected_topic", ""))
            for keyword in query.get("keywords", []) or []:
                add_text(keyword)
            for short_query in query.get("short_search_queries", []) or []:
                add_text(short_query)
        elif query is not None:
            add_text(query)
            
        cleaned_phrases: list[str] = []
        for phrase in phrases:
            phrase = clean_text(phrase).lower()
            if not phrase or len(phrase) < 3:
                continue
            if phrase in cleaned_phrases:
                continue
            cleaned_phrases.append(phrase)
            
        return {
            "tokens": tokens,
            "phrases": cleaned_phrases[:12],
        }

    def _source_text(self, source: Source) -> str:
        parts = [
            clean_text(getattr(source, "title", "")),
            clean_text(getattr(source, "abstract", "")),
            clean_text(getattr(source, "summary", "")),
        ]
        metadata = getattr(source, "metadata", {}) or {}
        if isinstance(metadata, dict):
            for key in (
                "tags",
                "topics",
                "subjects",
                "categories",
                "description",
                "pipeline_tag",
                "matched_query",
            ):
                value = metadata.get(key)
                if isinstance(value, str):
                    parts.append(clean_text(value))
                elif isinstance(value, (list, tuple, set)):
                    for item in value:
                        parts.append(clean_text(item))
        return "".join(part for part in parts if part)

    def _metadata_text(self, source: Source) -> str:
        metadata = getattr(source, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            return ""
        parts: list[str] = []
        for key, value in metadata.items():
            if isinstance(value, str):
                parts.append(clean_text(value))
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    parts.append(clean_text(item))
            if len(parts) > 80:
                break
        return " ".join(part for part in parts if part)[:1200]

    def _token_overlap(self, query_tokens: set[str], source_tokens: set[str]) -> float:
        if not query_tokens or not source_tokens:
            return 0.0
        score = 0.0
        candidates = list(source_tokens)
        for term in query_tokens:
            if term in source_tokens:
                score += 1.0
                continue
            best_ratio = 0.0
            term_length = len(term)
            for candidate in candidates:
                if abs(len(candidate) - term_length) > 2:
                    continue
                ratio = difflib.SequenceMatcher(None, term, candidate).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                if best_ratio >= 0.86:
                    break
            if best_ratio >= 0.86:
                score += 0.85
            elif best_ratio >= 0.78:
                score += 0.45
        return min(1.0, score / max(1, len(query_tokens)))

    def _phrase_bonus(self, phrases: list[str], title_text: str, full_text: str) -> float:
        bonus = 0.0
        for phrase in phrases[:10]:
            if phrase and phrase in title_text:
                return 0.18
            if phrase and phrase in full_text:
                bonus = max(bonus, 0.12)
        return bonus

    def _authority(self, source: Source) -> float:
        platform = self._enum_value(getattr(source, "platform", "")).lower()
        authority = _PLATFORM_AUTHORITY.get(platform)
        if authority is not None:
            return float(authority)
        source_type = self._type_value(source)
        fallback_authority = {
            "research_paper": 0.75,
            "course": 0.70,
            "book": 0.70,
            "documentation": 0.65,
            "repository": 0.70,
            "model": 0.65,
            "dataset": 0.60,
            "video": 0.55,
            "blog": 0.45,
            "other": 0.45,
        }
        return float(fallback_authority.get(source_type, 0.45))

    def _type_quality(self, source: Source) -> float:
        source_type = self._type_value(source)
        return float(_TYPE_QUALITY.get(source_type, 0.45))

    def _impact(self, source: Source) -> float:
        citation_count = getattr(source, "citation_count", None)
        if citation_count is not None:
            try:
                citations = float(citation_count)
                if citations <= 0:
                    return 0.20
                return max(0.0, min(1.0, math.log1p(citations) / math.log1p(1000.0)))
            except Exception:
                pass
        metadata = getattr(source, "metadata", {}) or {}
        if isinstance(metadata, dict):
            stars = metadata.get("stars") or metadata.get("stargazers_count")
            if stars is not None:
                try:
                    return max(0.0, min(1.0, math.log1p(float(stars)) / math.log1p(5000.0)))
                except Exception:
                    pass
            downloads = metadata.get("downloads")
            if downloads is not None:
                try:
                    return max(0.0, min(1.0, math.log1p(float(downloads)) / math.log1p(100000.0)))
                except Exception:
                    pass
        return 0.35

    def _recency(self, source: Source) -> float:
        year = getattr(source, "year", None)
        if year is None:
            published_at = getattr(source, "published_at", None)
            if published_at is not None:
                year = getattr(published_at, "year", None)
        if not year:
            return 0.50
        try:
            current_year = datetime.now(timezone.utc).year
            age = max(0, current_year - int(year))
            return max(0.20, math.exp(-age / 8.0))
        except Exception:
            return 0.50

    def _content(self, source: Source) -> float:
        abstract = clean_text(getattr(source, "abstract", ""))
        summary = clean_text(getattr(source, "summary", ""))
        metadata = getattr(source, "metadata", {}) or {}
        score = min(1.0, len(abstract) / 600.0)
        if summary:
            score += 0.25
        if isinstance(metadata, dict):
            score += min(0.20, len(metadata) / 8.0)
        if getattr(source, "has_code", None) is True:
            score += 0.05
        return max(0.0, min(1.0, score))

    def _support(self, source: Source) -> float:
        score = 0.0
        metadata = getattr(source, "metadata", {}) or {}
        if getattr(source, "has_code", None) is True:
            score += 0.45
        if getattr(source, "url", ""):
            score += 0.15
        if isinstance(metadata, dict):
            if metadata.get("pdf_url") or metadata.get("doi"):
                score += 0.25
            if metadata.get("open_access") or metadata.get("is_open_access"):
                score += 0.15
            if metadata.get("repository_url") or metadata.get("html_url"):
                score += 0.15
        return max(0.0, min(1.0, score))

    def _authors(self, source: Source) -> float:
        authors = getattr(source, "authors", []) or []
        if authors:
            return max(0.0, min(1.0, len(authors) / 4.0))
        source_type = self._type_value(source)
        if source_type in ("repository", "model", "dataset"):
            return 0.40
        return 0.20

    def _confidence(self, source: Source, query_info: dict[str, Any], score: float) -> float:
        authority = self._authority(source)
        content = self._content(source)
        metadata = getattr(source, "metadata", {}) or {}
        metadata_score = 0.0
        if isinstance(metadata, dict):
            metadata_score = min(1.0, len(metadata) / 6.0)
        confidence = (
            0.18
            + 0.38 * score
            + 0.18 * authority
            + 0.14 * content
            + 0.10 * metadata_score
        )
        return max(0.05, min(0.98, confidence))

    def _scorer_weights(self) -> dict[str, float]:
        weights = None
        try:
            weights = self._config.scorer_weights()
        except Exception:
            weights = None
        raw = {
            "relevance": getattr(weights, "relevance", 0.42),
            "authority": getattr(weights, "authority", 0.12),
            "type_quality": getattr(weights, "type_quality", 0.10),
            "impact": getattr(weights, "impact", 0.10),
            "recency": getattr(weights, "recency", 0.08),
            "content": getattr(weights, "content", 0.08),
            "support": getattr(weights, "support", 0.05),
            "authors": getattr(weights, "authors", 0.05),
        }
        total = sum(float(value) for value in raw.values())
        if total <= 0:
            total = 1.0
        return {key: float(value) / total for key, value in raw.items()}

    def _alignment_settings(self) -> Any:
        try:
            return self._config.alignment_settings()
        except Exception:
            return None

    def _difficulty_value(self, source: Source) -> Difficulty:
        value = getattr(source, "difficulty", None)
        if isinstance(value, Difficulty):
            return value
        try:
            return Difficulty(str(value).strip().lower())
        except Exception:
            return Difficulty.UNKNOWN

    def _type_value(self, source: Source) -> str:
        return self._enum_value(getattr(source, "source_type", "")).lower()

    def _enum_value(self, value: Any) -> str:
        return str(getattr(value, "value", value) or "")

    def _map_or_default(self, value: Any, default: dict[Any, float]) -> dict[Any, float]:
        if isinstance(value, dict) and value:
            return value
        return default

    def _map_value(self, mapping: dict[Any, Any], key: Any, default: float) -> float:
        if not isinstance(mapping, dict):
            return float(default)
        if key in mapping:
            return float(mapping[key])
        key_value = getattr(key, "value", key)
        if key_value in mapping:
            return float(mapping[key_value])
        if str(key_value) in mapping:
            return float(mapping[str(key_value)])
        return float(default)