"""The adapters against the middleware they translate, over real LangGraph.

The unit tests either side of the wire work on payloads written by hand;
this drives `HumanInTheLoopMiddleware` itself — from an entrypoint of its
own, so no model is needed — to pin what the two ends have to agree on:
the request the middleware raises splits into the questions Welt renders,
and the answers rejoin into decisions it resumes from.
"""

import asyncio
from collections.abc import AsyncIterator

from langchain.agents.middleware import (
    AgentState,
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
)
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint
from langgraph.runtime import get_runtime
from langgraph.types import Command

from welt_io_langgraph import decode_interrupt_responses, renderable_events

# Two gated calls in one turn, so the request carries the batch the
# middleware bundles: one action allowing an answer on its behalf, one
# allowing only the two buttons.
TOOL_CALLS = [
    {
        "name": "send_email",
        "args": {"to": "ops@example.com"},
        "id": "call-1",
        "type": "tool_call",
    },
    {
        "name": "delete_file",
        "args": {"path": "/etc/hosts"},
        "id": "call-2",
        "type": "tool_call",
    },
]

middleware = HumanInTheLoopMiddleware(
    interrupt_on={
        "send_email": InterruptOnConfig(
            allowed_decisions=["approve", "reject", "respond"]
        ),
        "delete_file": InterruptOnConfig(allowed_decisions=["approve", "reject"]),
    }
)


@entrypoint(checkpointer=InMemorySaver())
def review(tool_calls: list) -> dict | None:
    """Gate the tool calls of one turn, as the agent's model node would."""
    state = AgentState(messages=[AIMessage(content="", tool_calls=tool_calls)])
    return middleware.after_model(state, get_runtime())


def asked(thread_id: str) -> list[dict]:
    """Run one gated turn and return the questions it stops with."""

    async def turn() -> list[dict]:
        stream = review.astream(TOOL_CALLS, _config(thread_id), stream_mode=["updates"])
        return [event["interrupt"] async for event in _interrupts(stream)]

    return asyncio.run(turn())


def answered(thread_id: str, answers: dict[str, str]) -> list:
    """Answer a stopped turn and return the messages it settles on.

    The answers are the strings Welt sends, by question id: the value of a
    pressed button, or whatever a human typed.
    """

    async def resume() -> list:
        resumed = await review.ainvoke(
            Command(resume=decode_interrupt_responses(answers)), _config(thread_id)
        )
        return resumed["messages"]

    return asyncio.run(resume())


def _config(thread_id: str) -> RunnableConfig:
    return RunnableConfig(configurable={"thread_id": thread_id})


async def _interrupts(stream: AsyncIterator) -> AsyncIterator[dict]:
    async for event in renderable_events(stream):
        if "interrupt" in event:
            yield event


def test_one_request_becomes_one_question_per_gated_call() -> None:
    questions = asked("per-call")

    assert [question["name"] for question in questions] == ["send_email", "delete_file"]
    # One request bundles both calls, so both questions carry its id, each
    # with the index of the action it asks about.
    request_ids = {question["id"].rsplit("#", 1)[0] for question in questions}
    assert len(request_ids) == 1
    assert [question["id"].rsplit("#", 1)[1] for question in questions] == ["0", "1"]


def test_each_question_offers_only_what_its_action_allows() -> None:
    send_email, delete_file = asked("widgets")

    assert [option["label"] for option in delete_file["reason"]["options"]] == [
        "Approve",
        "Reject",
    ]
    assert "input" not in delete_file["reason"]
    assert send_email["reason"]["input"] == {}


def test_a_pressed_reject_stops_the_call_it_was_asked_about() -> None:
    send_email, delete_file = asked("reject")
    answers = {
        send_email["id"]: send_email["reason"]["options"][0]["value"],
        delete_file["id"]: delete_file["reason"]["options"][1]["value"],
    }

    messages = answered("reject", answers)

    # The approved call is left for the tool node to run, so the only
    # message the middleware adds is the rejection of the other one.
    rejection = messages[1]
    assert len(messages) == 2
    assert (rejection.tool_call_id, rejection.status) == ("call-2", "error")


def test_a_typed_answer_reaches_the_model_as_the_tool_s_answer() -> None:
    send_email, delete_file = asked("typed")
    answers = {
        send_email["id"]: "approve",
        delete_file["id"]: delete_file["reason"]["options"][1]["value"],
    }

    messages = answered("typed", answers)

    # Carrying no button's value, the typed word travels on as text — the
    # word matching a button's label decides nothing here, and the model
    # reads the answer as the tool's own.
    answer = messages[1]
    assert (answer.tool_call_id, answer.status, answer.content) == (
        "call-1",
        "success",
        "approve",
    )
