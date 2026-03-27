import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from adapters.base import Message
from core.agent import ClaudeAgent

def make_msg(is_owner=False, text="hello"):
    return Message(
        channel="telegram", chat_id="c1", message_id="m1",
        sender_id="u1", sender_name="Alice",
        text=text, timestamp=datetime(2026,1,1),
        is_group=False, is_owner=is_owner,
    )

@pytest.fixture
def agent():
    conv = AsyncMock()
    conv.get_history.return_value = []
    rag = AsyncMock()
    rag.search_as_string.return_value = "Pandu is an AI engineer."
    persona = MagicMock()
    persona.name = "Pandu"
    persona.build_prompt.return_value = "You are Pandu."
    return ClaudeAgent(
        api_key="test_key",
        model="claude-sonnet-4-6",
        conversation=conv,
        rag=rag,
        persona=persona,
    )

@pytest.mark.asyncio
async def test_auto_reply_calls_rag(agent):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hi there!", type="text")]
    mock_response.stop_reason = "end_turn"
    with patch.object(agent._client.messages, "create", new=AsyncMock(return_value=mock_response)):
        result = await agent.auto_reply_mode(make_msg(is_owner=False))
    agent._rag.search_as_string.assert_called_once()
    assert result == "Hi there!"

@pytest.mark.asyncio
async def test_auto_reply_handles_empty_rag(agent):
    agent._rag.search_as_string.return_value = ""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Reply", type="text")]
    mock_response.stop_reason = "end_turn"
    with patch.object(agent._client.messages, "create", new=AsyncMock(return_value=mock_response)):
        result = await agent.auto_reply_mode(make_msg())
    assert result == "Reply"

@pytest.mark.asyncio
async def test_assistant_mode_does_not_auto_inject_rag(agent):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Answer", type="text")]
    mock_response.stop_reason = "end_turn"
    with patch.object(agent._client.messages, "create", new=AsyncMock(return_value=mock_response)):
        result = await agent.assistant_mode(make_msg(is_owner=True))
    # RAG must NOT be auto-injected in assistant mode
    agent._rag.search_as_string.assert_not_called()
    assert result == "Answer"

@pytest.mark.asyncio
async def test_assistant_mode_includes_tools(agent):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Answer", type="text")]
    mock_response.stop_reason = "end_turn"
    create_mock = AsyncMock(return_value=mock_response)
    with patch.object(agent._client.messages, "create", new=create_mock):
        await agent.assistant_mode(make_msg(is_owner=True))
    call_kwargs = create_mock.call_args.kwargs
    assert "tools" in call_kwargs

@pytest.mark.asyncio
async def test_auto_reply_retries_on_failure(agent):
    fail_response = AsyncMock(side_effect=Exception("API error"))
    success_response = MagicMock()
    success_response.content = [MagicMock(text="OK", type="text")]
    success_response.stop_reason = "end_turn"
    # First call fails, second succeeds
    create_mock = AsyncMock(side_effect=[Exception("error"), success_response])
    with patch.object(agent._client.messages, "create", new=create_mock):
        with patch("asyncio.sleep", new=AsyncMock()):  # skip the sleep
            result = await agent.auto_reply_mode(make_msg())
    assert result == "OK"
    assert create_mock.call_count == 2

@pytest.mark.asyncio
async def test_auto_reply_fallback_after_two_failures(agent):
    create_mock = AsyncMock(side_effect=Exception("always fails"))
    with patch.object(agent._client.messages, "create", new=create_mock):
        with patch("asyncio.sleep", new=AsyncMock()):
            result = await agent.auto_reply_mode(make_msg())
    assert "unavailable" in result.lower()
    assert create_mock.call_count == 2

@pytest.mark.asyncio
async def test_tool_loop_executes_search_knowledge(agent):
    """_run_tool_loop should call RAG when Claude returns tool_use stop_reason."""
    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.id = "tu_001"
    tool_use_block.name = "search_knowledge"
    tool_use_block.input = {"query": "AI skills"}

    tool_use_response = MagicMock()
    tool_use_response.stop_reason = "tool_use"
    tool_use_response.content = [tool_use_block]

    final_block = MagicMock()
    final_block.type = "text"
    final_block.text = "Based on the knowledge base, Pandu is an AI engineer."
    final_response = MagicMock()
    final_response.stop_reason = "end_turn"
    final_response.content = [final_block]

    create_mock = AsyncMock(side_effect=[tool_use_response, final_response])
    with patch.object(agent._client.messages, "create", new=create_mock):
        result = await agent.assistant_mode(make_msg(is_owner=True, text="What are Pandu's skills?"))

    agent._rag.search_as_string.assert_called_once_with("AI skills")
    assert "AI engineer" in result
    assert create_mock.call_count == 2

@pytest.mark.asyncio
async def test_tool_loop_exception_returns_fallback(agent):
    """An exception inside the tool loop should return FALLBACK_MESSAGE."""
    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.id = "tu_002"
    tool_use_block.name = "search_knowledge"
    tool_use_block.input = {"query": "test"}

    tool_use_response = MagicMock()
    tool_use_response.stop_reason = "tool_use"
    tool_use_response.content = [tool_use_block]

    # Second call (tool result feed-back) raises
    create_mock = AsyncMock(side_effect=[tool_use_response, Exception("network error")])
    with patch.object(agent._client.messages, "create", new=create_mock):
        result = await agent.assistant_mode(make_msg(is_owner=True))

    assert "unavailable" in result.lower()

@pytest.mark.asyncio
async def test_execute_tool_search_knowledge(agent):
    agent._rag.search_as_string.return_value = "Pandu works in Hyderabad"
    msg = make_msg()
    result = await agent._execute_tool("search_knowledge", {"query": "location"}, msg)
    assert "Hyderabad" in result

@pytest.mark.asyncio
async def test_execute_tool_get_history(agent):
    agent._conv.get_history.return_value = [{"role": "user", "content": "hi"}]
    msg = make_msg()
    result = await agent._execute_tool(
        "get_conversation_history", {"chat_id": "c1", "channel": "telegram", "n": 5}, msg
    )
    assert "hi" in result

@pytest.mark.asyncio
async def test_execute_tool_unknown_returns_error(agent):
    msg = make_msg()
    result = await agent._execute_tool("unknown_tool", {}, msg)
    assert "Unknown tool" in result
