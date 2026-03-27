import asyncio
import anthropic
from adapters.base import Message
from core.conversation import ConversationManager
from knowledge.rag_engine import RAGEngine
from knowledge.persona_engine import PersonaEngine


SEARCH_KNOWLEDGE_TOOL = {
    "name": "search_knowledge",
    "description": "Search Pandu's personal knowledge base for relevant information.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"}
        },
        "required": ["query"],
    },
}

GET_HISTORY_TOOL = {
    "name": "get_conversation_history",
    "description": "Retrieve recent conversation history for a specific chat.",
    "input_schema": {
        "type": "object",
        "properties": {
            "chat_id": {"type": "string"},
            "channel": {"type": "string"},
            "n": {"type": "integer", "description": "Number of messages to retrieve", "default": 10},
        },
        "required": ["chat_id", "channel"],
    },
}

FALLBACK_MESSAGE = "I'm unavailable right now, please try again shortly."


class ClaudeAgent:
    def __init__(
        self,
        api_key: str,
        model: str,
        conversation: ConversationManager,
        rag: RAGEngine,
        persona: PersonaEngine,
    ):
        # AsyncAnthropic required — sync client blocks the event loop in asyncio apps
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._conv = conversation
        self._rag = rag
        self._persona = persona

    async def auto_reply_mode(self, message: Message) -> str:
        """Responds to a non-owner message using persona rules + RAG context."""
        history = await self._conv.get_history(message.channel, message.chat_id)
        rag_context = await self._rag.search_as_string(message.text)
        persona_prompt = self._persona.build_prompt(message.text)

        rag_section = f"\nRelevant knowledge:\n{rag_context}" if rag_context else ""
        system = f"{persona_prompt}{rag_section}"

        try:
            response = await self._client.messages.create(
                model=self._model,
                system=system,
                messages=history + [{"role": "user", "content": message.text}],
                max_tokens=1024,
            )
            return response.content[0].text
        except Exception:
            await asyncio.sleep(2)
            try:
                response = await self._client.messages.create(
                    model=self._model,
                    system=system,
                    messages=history + [{"role": "user", "content": message.text}],
                    max_tokens=1024,
                )
                return response.content[0].text
            except Exception:
                return FALLBACK_MESSAGE

    async def assistant_mode(self, message: Message) -> str:
        """Acts as personal assistant for the owner. RAG is tool-accessible, not auto-injected."""
        history = await self._conv.get_history(message.channel, message.chat_id)
        system = f"You are a personal AI assistant for {self._persona.name}. Be concise and direct."
        user_messages = history + [{"role": "user", "content": message.text}]

        try:
            response = await self._client.messages.create(
                model=self._model,
                system=system,
                messages=user_messages,
                tools=[SEARCH_KNOWLEDGE_TOOL, GET_HISTORY_TOOL],
                max_tokens=2048,
            )
            return await self._run_tool_loop(response, message, user_messages)
        except Exception:
            await asyncio.sleep(2)
            try:
                response = await self._client.messages.create(
                    model=self._model,
                    system=system,
                    messages=user_messages,
                    tools=[SEARCH_KNOWLEDGE_TOOL, GET_HISTORY_TOOL],
                    max_tokens=2048,
                )
                return await self._run_tool_loop(response, message, user_messages)
            except Exception:
                return FALLBACK_MESSAGE

    async def _run_tool_loop(self, response, message: Message, messages: list[dict]) -> str:
        """Handles tool_use stop_reason. `messages` is the full history + user turn."""
        try:
            while response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    result = await self._execute_tool(block.name, block.input, message)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
                messages = messages + [
                    {"role": "assistant", "content": response.content},
                    {"role": "user", "content": tool_results},
                ]
                response = await self._client.messages.create(
                    model=self._model,
                    system=f"You are a personal AI assistant for {self._persona.name}.",
                    messages=messages,
                    tools=[SEARCH_KNOWLEDGE_TOOL, GET_HISTORY_TOOL],
                    max_tokens=2048,
                )
        except Exception:
            return FALLBACK_MESSAGE   # also fixes: exceptions in tool loop now handled

        for block in response.content:
            if block.type == "text":
                return block.text
        return FALLBACK_MESSAGE   # fixes: use constant instead of inline string

    async def _execute_tool(self, name: str, inputs: dict, message: Message) -> str:
        """Dispatches a tool call and returns the result as a string."""
        if name == "search_knowledge":
            return await self._rag.search_as_string(inputs["query"])
        if name == "get_conversation_history":
            history = await self._conv.get_history(
                inputs["channel"], inputs["chat_id"], inputs.get("n", 10)
            )
            return str(history)
        return f"Unknown tool: {name}"
