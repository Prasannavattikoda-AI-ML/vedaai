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
