import pytest
from knowledge.rag_engine import RAGEngine

@pytest.fixture
def engine(tmp_path):
    docs_path = tmp_path / "documents"
    docs_path.mkdir()
    (docs_path / "bio.txt").write_text(
        "Pandu is a software engineer specializing in AI and machine learning. "
        "He works at a startup in Hyderabad building recommendation systems."
    )
    chroma_path = str(tmp_path / ".chroma")
    return RAGEngine(
        documents_path=str(docs_path),
        chroma_path=chroma_path,
        top_k=2,
        min_similarity=0.0,   # 0 so test content is always returned
    )

@pytest.mark.asyncio
async def test_index_and_search(engine):
    engine.index_documents()
    results = await engine.search("What does Pandu do for work?")
    assert len(results) > 0
    combined = " ".join(results).lower()
    assert "software engineer" in combined or "ai" in combined or "startup" in combined

@pytest.mark.asyncio
async def test_search_returns_empty_above_threshold(tmp_path):
    docs_path = tmp_path / "documents"
    docs_path.mkdir()
    (docs_path / "bio.txt").write_text("Pandu likes hiking and photography.")
    engine = RAGEngine(
        documents_path=str(docs_path),
        chroma_path=str(tmp_path / ".chroma"),
        top_k=3,
        min_similarity=0.9999,   # impossibly high
    )
    engine.index_documents()
    results = await engine.search("quantum physics nuclear reactor")
    assert results == []

@pytest.mark.asyncio
async def test_search_as_string_returns_string(engine):
    engine.index_documents()
    result = await engine.search_as_string("AI work")
    assert isinstance(result, str)

@pytest.mark.asyncio
async def test_search_empty_collection_returns_empty(tmp_path):
    empty_docs = tmp_path / "empty_docs"
    empty_docs.mkdir()
    engine = RAGEngine(
        documents_path=str(empty_docs),
        chroma_path=str(tmp_path / ".chroma"),
        top_k=3,
        min_similarity=0.0,
    )
    # Do NOT call index_documents — collection stays empty
    results = await engine.search("anything")
    assert results == []
