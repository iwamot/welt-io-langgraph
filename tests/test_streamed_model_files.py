"""A file the model generates, whether its reply streams or arrives whole.

LangGraph's `messages` stream hands over a streamed reply as chunks and a
whole one as a single message, and does not repeat the chunks' message when
the node finishes. This drives real LangGraph on a model that replays a
script, so a file has to arrive once on either path.
"""

import asyncio
import base64
from collections.abc import Iterator

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from welt_io_langgraph import renderable_events

IMAGE = base64.b64encode(b"an image").decode()
CONTENT: list = [
    {"type": "text", "text": "here it is"},
    {"type": "image", "base64": IMAGE, "mime_type": "image/png"},
]


class ScriptedModel(BaseChatModel):
    """A chat model that replies with text and an image, streaming or whole."""

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages: list[BaseMessage], *args, **kwargs) -> ChatResult:
        message = AIMessage(content=CONTENT)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self, messages: list[BaseMessage], *args, **kwargs
    ) -> Iterator[ChatGenerationChunk]:
        for index, block in enumerate(CONTENT):
            chunk = AIMessageChunk(content=[{**block, "index": index}])
            yield ChatGenerationChunk(message=chunk)


def reply(model: BaseChatModel) -> list[dict]:
    """Run one turn and return the events it renders."""
    agent = create_agent(model, [], checkpointer=InMemorySaver())

    async def turn() -> list[dict]:
        stream = agent.astream(
            {"messages": [{"role": "user", "content": "draw"}]},
            RunnableConfig(configurable={"thread_id": "one"}),
            stream_mode=["messages", "updates"],
        )
        return [event async for event in renderable_events(stream)]

    return asyncio.run(turn())


EXPECTED = [
    {"data": "here it is"},
    {"file": {"name": "image.png", "bytes": IMAGE}},
]


def test_a_streamed_reply_uploads_its_file_once() -> None:
    assert reply(ScriptedModel()) == EXPECTED


def test_a_whole_reply_uploads_its_file_once() -> None:
    assert reply(ScriptedModel(disable_streaming=True)) == EXPECTED
