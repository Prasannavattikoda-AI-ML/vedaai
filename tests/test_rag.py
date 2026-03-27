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
    await engine.index_documents()
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
    await engine.index_documents()
    results = await engine.search("quantum physics nuclear reactor")
    assert results == []

@pytest.mark.asyncio
async def test_search_as_string_returns_string(engine):
    await engine.index_documents()
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

def test_chunk_text_basic():
    from knowledge.rag_engine import _chunk_text
    text = " ".join([f"word{i}" for i in range(600)])
    chunks = _chunk_text(text)
    assert len(chunks) >= 2
    # First chunk has CHUNK_SIZE words
    assert len(chunks[0].split()) == 512

def test_chunk_text_short_text():
    from knowledge.rag_engine import _chunk_text
    chunks = _chunk_text("short text")
    assert len(chunks) == 1
    assert chunks[0] == "short text"

@pytest.mark.asyncio
async def test_index_documents_idempotent(tmp_path):
    """index_documents called twice should not duplicate chunks."""
    docs_path = tmp_path / "documents"
    docs_path.mkdir()
    (docs_path / "bio.txt").write_text("Pandu is an AI engineer in Hyderabad.")
    engine = RAGEngine(
        documents_path=str(docs_path),
        chroma_path=str(tmp_path / ".chroma"),
        top_k=3,
        min_similarity=0.0,
    )
    await engine.index_documents()
    count_after_first = engine._collection.count()
    await engine.index_documents()
    count_after_second = engine._collection.count()
    assert count_after_first == count_after_second

@pytest.mark.asyncio
async def test_md_files_indexed(tmp_path):
    """Markdown files must be indexed alongside .txt files."""
    docs_path = tmp_path / "documents"
    docs_path.mkdir()
    (docs_path / "info.md").write_text("Pandu specializes in machine learning pipelines.")
    engine = RAGEngine(
        documents_path=str(docs_path),
        chroma_path=str(tmp_path / ".chroma"),
        top_k=3,
        min_similarity=0.0,
    )
    await engine.index_documents()
    results = await engine.search("machine learning")
    assert len(results) > 0
