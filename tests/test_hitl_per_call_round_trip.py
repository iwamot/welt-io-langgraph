"""The adapters against `HumanInTheLoopMiddleware`'s `per_call` mode.

The mode raises its interrupt from inside the tool call, so this drives a
whole agent over real LangGraph, on a model that replays a script rather
than calling anything.
"""

import asyncio
from collections.abc import AsyncIterator

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from welt_io_langgraph import decode_interrupt_responses, renderable_events


class ScriptedModel(BaseChatModel):
    """A chat model that asks for the two gated calls, then finishes."""

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages: list[BaseMessage], *args, **kwargs) -> ChatResult:
        # The tool results in the conversation tell the turns apart, so
        # every thread can share one model.
        done = any(message.type == "tool" for message in messages)
        message = AIMessage(content="done") if done else _calling_both_tools()
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: object, **kwargs: object) -> "ScriptedModel":
        return self


def _calling_both_tools() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
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
        ],
    )


ran: list[str] = []


@tool
def send_email(to: str) -> str:
    """Send an email."""
    ran.append(f"send_email:{to}")
    return "sent"


@tool
def delete_file(path: str) -> str:
    """Delete a file."""
    ran.append(f"delete_file:{path}")
    return "deleted"


agent = create_agent(
    ScriptedModel(),
    [send_email, delete_file],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": InterruptOnConfig(
                    allowed_decisions=["approve", "reject", "respond"]
                ),
                "delete_file": InterruptOnConfig(
                    allowed_decisions=["approve", "reject"]
                ),
            },
            interrupt_mode="per_call",
        )
    ],
    checkpointer=InMemorySaver(),
)


def _config(thread_id: str) -> RunnableConfig:
    return RunnableConfig(configurable={"thread_id": thread_id})


def asked(thread_id: str) -> dict[str, dict]:
    """Run the turn up to its stop and return the questions, by tool name."""

    async def turn() -> dict[str, dict]:
        stream = agent.astream(
            {"messages": [{"role": "user", "content": "go"}]},
            _config(thread_id),
            stream_mode=["updates"],
        )
        questions = _questions(stream)
        return {question["name"]: question async for question in questions}

    return asyncio.run(turn())


async def _questions(stream: AsyncIterator) -> AsyncIterator[dict]:
    async for event in renderable_events(stream):
        if "interrupt" in event:
            yield event["interrupt"]


def answered(thread_id: str, answers: dict[str, dict]) -> dict[str, ToolMessage]:
    """Answer a stopped turn and return its tool messages, by tool name."""

    async def resume() -> dict[str, ToolMessage]:
        result = await agent.ainvoke(
            Command(resume=decode_interrupt_responses(answers)), _config(thread_id)
        )
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        return {str(m.name): m for m in tool_messages}

    return asyncio.run(resume())


def _pressed(value: object) -> dict:
    return {"value": value, "source": "option"}


def test_each_gated_call_becomes_a_question_offering_what_it_allows() -> None:
    questions = asked("widgets")

    assert set(questions) == {"send_email", "delete_file"}
    send_email_reason = questions["send_email"]["reason"]
    delete_file_reason = questions["delete_file"]["reason"]
    assert "ops@example.com" in send_email_reason["message"]
    assert send_email_reason["approve"] == {}
    assert send_email_reason["reject"] == {}
    assert send_email_reason["input"] == {
        "label": "Answer instead of running send_email"
    }
    assert delete_file_reason["approve"] == {}
    assert delete_file_reason["reject"] == {}
    assert "input" not in delete_file_reason


def test_the_answers_resume_each_call_with_its_own_decision() -> None:
    ran.clear()
    questions = asked("decisions")

    messages = answered(
        "decisions",
        {
            questions["send_email"]["id"]: _pressed(True),
            questions["delete_file"]["id"]: _pressed(False),
        },
    )

    assert ran == ["send_email:ops@example.com"]
    assert messages["send_email"].content == "sent"
    assert messages["delete_file"].status == "error"


def test_a_typed_answer_stands_in_for_the_tool() -> None:
    ran.clear()
    questions = asked("typed")

    messages = answered(
        "typed",
        {
            questions["send_email"]["id"]: {"value": "not now", "source": "input"},
            questions["delete_file"]["id"]: _pressed(False),
        },
    )

    assert ran == []
    assert messages["send_email"].content == "not now"
