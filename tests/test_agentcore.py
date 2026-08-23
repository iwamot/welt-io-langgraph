import asyncio
import base64
from collections.abc import AsyncIterator, Callable, Iterator

import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Interrupt

from welt_io_langgraph import decode_interrupt_responses, decode_messages
from welt_io_langgraph.agentcore import (
    _checked_data,
    _checked_name,
    _drained,
    send_file,
    welt_agent,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n"
PNG_BASE64 = base64.b64encode(PNG_BYTES).decode("ascii")


@pytest.fixture(autouse=True)
def empty_queue() -> Iterator[None]:
    """Start and leave every test with no files queued."""
    _drained()
    yield
    _drained()


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
        self.checkpointer: object | None = object()
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


def replies(
    entrypoint: Callable[[dict], AsyncIterator[dict]], payload: dict
) -> list[dict]:
    """Run the entrypoint on one payload and gather what it streams."""

    async def gather() -> list[dict]:
        return [event async for event in entrypoint(payload)]

    return asyncio.run(gather())


def test_a_turn_streams_the_renderable_events() -> None:
    agent = ReplayGraph([token("hi")])

    entrypoint = welt_agent(agent)

    assert replies(entrypoint, {"messages": []}) == [{"data": "hi"}]


def test_a_turn_runs_on_the_decoded_messages() -> None:
    agent = ReplayGraph([])
    messages = [{"role": "user", "content": [{"text": "hello"}]}]

    replies(welt_agent(agent), {"messages": messages})

    graph_input, _config, stream_mode = agent.calls[0]
    assert graph_input == {"messages": decode_messages(messages)}
    assert stream_mode == ["messages", "updates"]


def test_each_turn_streams_on_a_fresh_thread() -> None:
    agent = ReplayGraph([token("one")], [token("two")])

    entrypoint = welt_agent(agent)
    replies(entrypoint, {"messages": []})
    replies(entrypoint, {"messages": []})

    thread_ids = [config["configurable"]["thread_id"] for _, config, _ in agent.calls]
    assert len(set(thread_ids)) == 2


def test_a_graph_without_a_checkpointer_is_refused() -> None:
    agent = ReplayGraph()
    agent.checkpointer = None

    with pytest.raises(ValueError, match="needs a checkpointer"):
        welt_agent(agent)


class SendingGraph(ReplayGraph):
    """A graph whose stream queues a file the way a tool would."""

    def __init__(self, *, after_last_event: bool = False) -> None:
        super().__init__()
        self.after_last_event = after_last_event

    def astream(
        self, input: dict | Command, config: RunnableConfig, *, stream_mode: list
    ) -> AsyncIterator:
        """Stream two tokens, queueing a file between or after them."""
        return self._items()

    async def _items(self) -> AsyncIterator:
        yield token("before")
        if not self.after_last_event:
            send_file("chart.png", PNG_BYTES)
            yield token("after")
        else:
            send_file("chart.png", PNG_BYTES)


def test_a_file_a_tool_queued_rides_beside_the_reply() -> None:
    entrypoint = welt_agent(SendingGraph(), files_from={"some_tool"})

    assert replies(entrypoint, {"messages": []}) == [
        {"data": "before"},
        {"data": "after"},
        {"file": {"name": "chart.png", "bytes": PNG_BASE64}},
    ]


def test_a_file_queued_after_the_last_event_still_rides_the_reply() -> None:
    entrypoint = welt_agent(SendingGraph(after_last_event=True))

    assert replies(entrypoint, {"messages": []}) == [
        {"data": "before"},
        {"file": {"name": "chart.png", "bytes": PNG_BASE64}},
    ]


def test_a_failed_turns_leftover_files_stay_off_the_next_reply() -> None:
    send_file("stale.txt", b"left behind")

    entrypoint = welt_agent(ReplayGraph([token("fresh")]))

    assert replies(entrypoint, {"messages": []}) == [{"data": "fresh"}]


def test_resume_without_an_interrupted_run_is_refused() -> None:
    entrypoint = welt_agent(ReplayGraph())

    with pytest.raises(RuntimeError, match="No interrupted run"):
        replies(entrypoint, {"interrupt_responses": {}})


def test_an_interrupted_run_resumes_on_its_own_thread() -> None:
    agent = ReplayGraph([interrupted_update("i-1", "Go?")], [token("resumed")])
    responses = {"i-1": {"value": True, "source": "option"}}

    entrypoint = welt_agent(agent)
    first = replies(entrypoint, {"messages": []})
    second = replies(entrypoint, {"interrupt_responses": responses})

    assert first == [{"interrupt": {"id": "i-1", "name": "", "reason": "Go?"}}]
    assert second == [{"data": "resumed"}]
    resume_input, resume_config, _ = agent.calls[1]
    assert isinstance(resume_input, Command)
    assert resume_input.resume == decode_interrupt_responses(responses)
    # The resume rode the interrupted turn's config — same thread.
    assert resume_config is agent.calls[0][1]


def test_the_slot_empties_once_resumed() -> None:
    agent = ReplayGraph([interrupted_update("i-1", "Go?")], [token("resumed")])
    responses = {"i-1": {"value": True, "source": "option"}}

    entrypoint = welt_agent(agent)
    replies(entrypoint, {"messages": []})
    replies(entrypoint, {"interrupt_responses": responses})

    with pytest.raises(RuntimeError, match="No interrupted run"):
        replies(entrypoint, {"interrupt_responses": responses})


def test_a_resume_that_interrupts_again_can_resume_again() -> None:
    agent = ReplayGraph(
        [interrupted_update("i-1", "First?")],
        [interrupted_update("i-2", "Second?")],
        [token("done")],
    )

    entrypoint = welt_agent(agent)
    replies(entrypoint, {"messages": []})
    replies(entrypoint, {"interrupt_responses": {"i-1": {"value": True}}})
    third = replies(entrypoint, {"interrupt_responses": {"i-2": {"value": True}}})

    assert third == [{"data": "done"}]


def test_sent_file_becomes_a_file_wire_event() -> None:
    send_file("chart.png", PNG_BYTES)
    assert _drained() == [{"file": {"name": "chart.png", "bytes": PNG_BASE64}}]


# The checks below go through the private helpers, which take `object`: a
# deliberately wrong value handed to the typed public function would be a
# type error in this file, and the helpers are where the checks live.


def test_a_name_that_is_not_a_string_is_refused() -> None:
    with pytest.raises(TypeError, match="name must be a str, not int"):
        _checked_name(1)


def test_an_empty_name_is_refused() -> None:
    with pytest.raises(ValueError, match="name must not be empty"):
        _checked_name("")


def test_data_that_is_not_bytes_is_refused() -> None:
    with pytest.raises(TypeError, match="data must be bytes, not str"):
        _checked_data("not bytes")


def test_empty_data_is_refused() -> None:
    with pytest.raises(ValueError, match="data must not be empty"):
        _checked_data(b"")


def test_a_refused_file_is_not_queued() -> None:
    with pytest.raises(ValueError):
        send_file("chart.png", b"")
    assert _drained() == []
