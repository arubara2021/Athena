from __future__ import annotations

from typing import Any


def get_settings() -> Any:
    from core.config import get_settings as _get_settings
    return _get_settings()


def get_data_directory() -> Any:
    from core.config import get_data_directory as _get_data_directory
    return _get_data_directory()


def get_output_directory() -> Any:
    from core.config import get_output_directory as _get_output_directory
    return _get_output_directory()


def get_constants() -> Any:
    from core import constants as _constants
    return _constants


def get_search_query_schema() -> Any:
    from core.schemas import SearchQuerySchema
    return SearchQuerySchema


def get_source_platform_enum() -> Any:
    from core.models import SourcePlatform
    return SourcePlatform


def get_source_type_enum() -> Any:
    from core.models import SourceType
    return SourceType


def get_sqlite_store() -> Any:
    from storage.sqlite_store import SQLiteStore
    return SQLiteStore(get_data_directory() / "research_agent.db")


def get_vector_store() -> Any:
    from rag.vector_store import VectorStore
    return VectorStore()


def get_embedder() -> Any:
    from rag.embedder import Embedder
    return Embedder()


def get_conversation(session_id: str | None = None) -> Any:
    from rag.conversation import Conversation
    return Conversation(session_id=session_id)


def get_chat_engine(
    vector_store: Any = None,
    top_k: int = 5,
    session_id: str | None = None,
    source_diversity: int = 0,
) -> Any:
    from rag.chat_engine import RAGChatEngine
    kwargs: dict[str, Any] = {
        "top_k": top_k,
        "session_id": session_id,
        "source_diversity": source_diversity,
    }
    if vector_store is not None:
        kwargs["vector_store"] = vector_store
    return RAGChatEngine(**kwargs)


def get_search_orchestrator(use_llm_expansion: bool = False) -> Any:
    from search.orchestrator import SearchOrchestrator
    return SearchOrchestrator(use_llm_expansion=use_llm_expansion)


def get_consensus_ranker(use_llm: bool = True) -> Any:
    from ranking.consensus_ranker import ConsensusRanker
    return ConsensusRanker(use_llm=use_llm)


def get_learning_path_builder(use_llm: bool = False) -> Any:
    from ranking.learning_path_builder import LearningPathBuilder
    return LearningPathBuilder(use_llm=use_llm)


def get_pipeline(
    use_llm_expansion: bool = False,
    use_llm_ranking: bool = True,
    use_llm_learning_path: bool = False,
    enable_rag: bool = False,
) -> Any:
    from core.pipeline import ResearchPipeline
    return ResearchPipeline(
        search_orchestrator=get_search_orchestrator(
            use_llm_expansion=use_llm_expansion
        ),
        consensus_ranker=get_consensus_ranker(use_llm=use_llm_ranking),
        learning_path_builder=get_learning_path_builder(
            use_llm=use_llm_learning_path
        ),
        enable_rag=enable_rag,
    )


def get_agent_config_class() -> Any:
    from agent.controller import AgentConfig
    return AgentConfig


def get_agent(agent_config: Any = None) -> Any:
    from agent.controller import AutonomousAgent
    if agent_config is None:
        return AutonomousAgent()
    return AutonomousAgent(agent_config=agent_config)


def get_markdown_writer() -> Any:
    from storage.file_manager import FileManager
    from storage.markdown_writer import MarkdownWriter
    return MarkdownWriter(FileManager(get_output_directory()))


def get_tool_registry() -> Any:
    from agent.tools.registry import ToolRegistry
    return ToolRegistry()


def get_tool_registrations() -> tuple[tuple[str, str], ...]:
    return (
        ("agent.tools.search_tools", "register_search_tools"),
        ("agent.tools.read_tools", "register_read_tools"),
        ("agent.tools.analysis_tools", "register_analysis_tools"),
        ("agent.tools.gen_tools", "register_gen_tools"),
        ("agent.tools.memory_tools", "register_memory_tools"),
    )