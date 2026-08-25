import asyncio
import base64
from collections.abc import AsyncIterator

import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Interrupt

from welt_io_langgraph import (
    decode_interrupt_responses,
    decode_messages,
    renderable_events,
    start_reply,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n"
PNG_BASE64 = base64.b64encode(PNG_BYTES).decode("ascii")


def token(text: str) -> tuple:
    """Build one `messages` stream item carrying a token delta."""
    return ("messages", (AIMessageChunk(content=text), {}))


def interrupted_update(interrupt_id: str, reason: str) -> tuple:
    """Build one `updates` stream item carrying a pending interrupt."""
    return ("updates", {"__interrupt__": (Interrupt(value=reason, id=interrupt_id),)})


class ReplayGraph:
    """A LangGraph-shaped compiled graph that replays scripted items.

    Constructed input data, not a mock: it holds the item lists to stream
    and the inputs it was streamed on, and verifies nothing itself.
    """

    def __init__(self, *scripts: list) -> None:
        self.scripts = list(scripts)
        self.calls: list[tuple] = []

    def astream(
        self, input: dict | Command, config: RunnableConfig, *, stream_mode: list
    ) -> AsyncIterator:
        """Replay the next script."""
        self.calls.append((input, config, stream_mode))
        return _replayed(self.scripts.pop(0))


async def _replayed(items: list) -> AsyncIterator:
    for item in items:
        yield item


CONFIG = RunnableConfig(configurable={"thread_id": "t-1"})


def replies(
    agent: ReplayGraph,
    payload: dict,
    *,
    config: RunnableConfig | None = None,
    files_from: set[str] | None = None,
) -> list[dict]:
    """Stream one reply and gather the events it renders."""
    stream = start_reply(agent, payload, config or CONFIG)

    async def gather() -> list[dict]:
        return [
            event async for event in renderable_events(stream, files_from=files_from)
        ]

    return asyncio.run(gather())


def test_a_turn_streams_the_renderable_events() -> None:
    agent = ReplayGraph([token("hi")])

    assert replies(agent, {"messages": []}) == [{"data": "hi"}]


def test_a_turn_streams_on_the_decoded_messages() -> None:
    agent = ReplayGraph([])
    messages = [{"role": "user", "content": [{"text": "hello"}]}]

    replies(agent, {"messages": messages})

    graph_input, _, _ = agent.calls[0]
    assert graph_input == {"messages": decode_messages(messages)}


def test_a_turn_streams_on_the_config_it_was_given() -> None:
    agent = ReplayGraph([])

    replies(agent, {"messages": []})

    assert agent.calls[0][1] is CONFIG


def test_a_turn_streams_in_the_messages_and_updates_modes() -> None:
    agent = ReplayGraph([])

    replies(agent, {"messages": []})

    assert agent.calls[0][2] == ["messages", "updates"]


def test_a_resume_streams_the_answers_as_a_command() -> None:
    agent = ReplayGraph([token("resumed")])
    responses = {"i-1": {"value": True, "source": "option"}}

    resumed = replies(agent, {"interrupt_responses": responses})

    assert resumed == [{"data": "resumed"}]
    graph_input, _, _ = agent.calls[0]
    assert isinstance(graph_input, Command)
    assert graph_input.resume == decode_interrupt_responses(responses)


def test_a_resume_streams_on_the_config_it_was_given() -> None:
    agent = ReplayGraph([])
    responses = {"i-1": {"value": True, "source": "option"}}

    replies(agent, {"interrupt_responses": responses})

    assert agent.calls[0][1] is CONFIG


def test_a_resume_streams_in_the_messages_and_updates_modes() -> None:
    agent = ReplayGraph([])
    responses = {"i-1": {"value": True, "source": "option"}}

    replies(agent, {"interrupt_responses": responses})

    assert agent.calls[0][2] == ["messages", "updates"]


def test_a_payload_carrying_neither_envelope_raises() -> None:
    agent = ReplayGraph()

    with pytest.raises(KeyError):
        start_reply(agent, {}, CONFIG)


def test_a_stop_ends_the_reply_with_its_interrupts() -> None:
    agent = ReplayGraph([interrupted_update("i-1", "Go?")])

    assert replies(agent, {"messages": []}) == [
        {"interrupt": {"id": "i-1", "name": "", "reason": "Go?"}}
    ]
