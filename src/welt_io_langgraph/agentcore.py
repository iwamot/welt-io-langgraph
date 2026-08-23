"""The AgentCore Runtime entrypoint for a LangGraph agent Welt drives.

`welt_agent` builds the entrypoint that `BedrockAgentCoreApp` serves, so
an agent connects to Welt without rewriting the wiring every deployable
needs: telling a conversation turn from the answers that resume an
interrupted run, decoding each envelope, keeping the interrupted run's
config until its answers arrive, and reducing the stream to the events
Welt renders. The example agent of this repository once wrote this wiring
out by hand; this module is the same wiring as a function.

The interrupted run's config waits inside the returned entrypoint, under
the runtime's own lifecycle: AgentCore Runtime serves each session from
its own microVM, so one slot is enough, and the slot lives and dies with
that microVM — resuming after it was recycled (idle timeout, 8 hours at
most) raises an error the AgentCore Runtime SDK reports as an `error`
event, which Welt renders as its resume-failure notice. The slot is
resume-only: a normal turn always streams on a fresh thread built from
the messages Welt sends, because the Slack thread is the source of truth
for conversation history — letting the checkpointer stack turns into its
own history would double the conversation.

`send_file` hands the Slack thread a file without handing it to the
model: a tool queues the file, and the entrypoint puts it on the wire
beside the events of the reply being streamed. The model never sees what
was sent, so a tool whose file matters to the conversation says what it
holds in its result — or hands it to the model as a file content block
and is named in `files_from` instead.
"""

import base64
from collections.abc import AsyncIterator, Callable, Collection
from typing import Protocol
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from welt_io_langgraph import (
    decode_interrupt_responses,
    decode_messages,
    renderable_events,
)

__all__ = ["send_file", "welt_agent"]


class _StreamingGraph(Protocol):
    """What the entrypoint drives: the compiled graph's streaming face.

    Importing the SDK to name the graph would say what two members
    already say. This names them instead, and a compiled graph satisfies
    it.
    """

    @property
    def checkpointer(self) -> object | None:
        """The checkpointer the graph was compiled with."""
        ...

    def astream(
        self,
        input: dict | Command,
        config: RunnableConfig,
        *,
        stream_mode: list,
    ) -> AsyncIterator:
        """Stream the graph's reply to one input."""
        ...


# The files queued by `send_file`, on their way to the Slack thread. One
# queue for the process, like the interrupt slot is one per entrypoint:
# AgentCore Runtime serves each session from its own microVM, so no other
# reply's files can interleave with the running one's.
_pending_files: list[dict] = []


def send_file(name: str, data: bytes) -> None:
    """
    Queue one file for the Slack thread, beside the reply being streamed.

    The file rides the wire between the events of the running reply, and
    never reaches the model. A tool that wants the model to know what the
    file holds says so in its result — or returns the file as a file
    content block and is named in `files_from`, which puts it in front of
    the model and on the thread both.

    A file queued by a turn that failed before draining does not ride a
    later reply: the entrypoint starts every turn with the queue empty.

    Args:
        name (str): The upload filename, extension included.
        data (bytes): The raw file bytes.

    Raises:
        TypeError: If the name or the data is of the wrong type.
        ValueError: If either is empty. Slack refuses a zero-byte upload,
            and the whole reply fails with it, so an empty file is refused
            here, where the tool that queued it is still on the stack.
    """
    name = _checked_name(name)
    data = _checked_data(data)
    _pending_files.append(
        {"file": {"name": name, "bytes": base64.b64encode(data).decode("ascii")}}
    )


def _checked_name(name: object) -> str:
    """
    Check an upload filename.

    Args:
        name (object): The name the caller passed.

    Returns:
        str: The name.

    Raises:
        TypeError: If it is not a string.
        ValueError: If it is empty.
    """
    if not isinstance(name, str):
        raise TypeError(f"name must be a str, not {type(name).__name__}")
    if not name:
        raise ValueError("name must not be empty")
    return name


def _checked_data(data: object) -> bytes:
    """
    Check a file's bytes.

    Args:
        data (object): The data the caller passed.

    Returns:
        bytes: The data.

    Raises:
        TypeError: If it is not bytes.
        ValueError: If it is empty.
    """
    if not isinstance(data, bytes):
        raise TypeError(f"data must be bytes, not {type(data).__name__}")
    if not data:
        raise ValueError("data must not be empty; Slack refuses an empty upload")
    return data


def _drained() -> list[dict]:
    """
    Take every queued file event off the queue, in order.

    Returns:
        list[dict]: The `file` events queued since the last drain.
    """
    events = _pending_files[:]
    _pending_files.clear()
    return events


def welt_agent(
    agent: _StreamingGraph,
    *,
    files_from: Collection[str] | None = None,
) -> Callable[[dict], AsyncIterator[dict]]:
    """
    Build the AgentCore Runtime entrypoint for an agent Welt drives.

    The returned function is what `BedrockAgentCoreApp` takes::

        app = BedrockAgentCoreApp()
        app.entrypoint(welt_agent(agent, files_from={"create_chart"}))

    It reads which envelope Welt sent — Converse-shaped `messages` for a
    conversation turn, `interrupt_responses` for the answers that resume
    an interrupted run — streams the graph on it, and yields the events
    Welt renders, the files tools queued with `send_file` among them. A
    conversation turn always streams on a fresh thread, because the Slack
    thread is the source of truth for conversation history and the
    messages Welt sends already carry it whole.

    Args:
        agent (CompiledStateGraph): The compiled graph to drive, built
            with a checkpointer — interrupts pause and resume through
            checkpoints, even though the conversation history lives in
            Slack.
        files_from (Collection[str] | None): The names of the tools whose
            file content blocks become `file` events, as
            `renderable_events` takes it. None takes files from none of
            them.

    Returns:
        Callable[[dict], AsyncIterator[dict]]: The entrypoint. It raises
            `RuntimeError` when asked to resume a run its microVM no
            longer holds — the session was recycled while the buttons
            waited — which the AgentCore Runtime SDK reports as an
            `error` event and Welt renders as its resume-failure notice.

    Raises:
        ValueError: If the graph was compiled without a checkpointer,
            which would break the first interrupt at runtime instead of
            here.
    """
    if agent.checkpointer is None:
        raise ValueError(
            "the graph needs a checkpointer; interrupts pause and resume"
            " through checkpoints, even though the conversation history"
            " lives in Slack"
        )
    interrupted_config: RunnableConfig | None = None

    async def entrypoint(payload: dict) -> AsyncIterator[dict]:
        """
        Stream a reply to the conversation or approval answers Welt sent.

        Args:
            payload (dict): The invocation payload, carrying one of the
                two envelopes. What Welt sends is taken as correct, so a
                payload carrying neither is Welt's bug, and the KeyError
                it raises is reported as an `error` event by the SDK.

        Yields:
            dict: The renderable subset of the graph's stream, and the
                `file` events tools queued with `send_file`.

        Raises:
            RuntimeError: If there is no interrupted run to resume.
        """
        nonlocal interrupted_config
        # A failed turn's leftovers stay off this reply.
        _pending_files.clear()

        graph_input: dict | Command
        if "interrupt_responses" in payload:
            config = interrupted_config
            interrupted_config = None
            if config is None:  # The microVM was recycled while the buttons waited.
                raise RuntimeError("No interrupted run to resume in this session.")
            graph_input = Command(
                resume=decode_interrupt_responses(payload["interrupt_responses"])
            )
        else:
            # A fresh thread per turn: Welt sends the whole Slack thread
            # every time, so letting the checkpointer stack turns into its
            # own history would double the conversation.
            config = RunnableConfig(configurable={"thread_id": uuid4().hex})
            graph_input = {"messages": decode_messages(payload["messages"])}

        interrupted = False
        stream = agent.astream(graph_input, config, stream_mode=["messages", "updates"])
        async for event in renderable_events(stream, files_from=files_from):
            if "interrupt" in event:
                interrupted = True
            yield event
            for file_event in _drained():
                yield file_event
        # Files a tool queued after its result's events had already
        # drained — the stream's tail — still belong to this reply.
        for file_event in _drained():
            yield file_event

        if interrupted:
            # Re-stashed on every interrupted stop, so a resume that
            # interrupts again keeps working.
            interrupted_config = config

    return entrypoint
