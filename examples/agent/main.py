"""A small AgentCore agent that Welt can drive.

Receives Welt's payload, feeds it to a LangGraph agent, and yields the
renderable subset of its `astream` items — the AgentCore Runtime SDK
emits each one as SSE, which Welt (https://github.com/iwamot/welt)
renders into Slack. The payload carries one of two envelopes:
Converse-shaped `messages` for a conversation turn, or
`interrupt_responses` when a human answered the approval buttons of an
interrupted run.

This example is a standalone deployable; Welt drives it only through the
JSON wire contract, which welt-io-langgraph adapts in both directions.
"""

import os
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from uuid import uuid4

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_aws import ChatBedrockConverse
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.func import task
from langgraph.types import Command, interrupt

from welt_io_langgraph import (
    decode_interrupt_responses,
    decode_messages,
    file_event,
    interrupt_reason,
    renderable_events,
)

app = BedrockAgentCoreApp()

# Where an interrupted run waits for its answers. One slot is enough:
# AgentCore Runtime runs each session in its own microVM, so this process
# never serves two sessions. Resume only: a normal turn always streams on
# a fresh thread built from the messages Welt sends (the Slack thread is
# the source of truth for conversation history, so the checkpointer must
# not stand in for it). No persistence either — the checkpointer below is
# in-memory, and both live and die with the session's microVM (recycled
# on idle timeout, 8 hours at most).
_interrupted_config: RunnableConfig | None = None


@tool
def current_time() -> str:
    """
    Get the current date and time.

    Returns:
        str: The current UTC time in ISO 8601 format.
    """
    return datetime.now(timezone.utc).isoformat()


@tool
def attach_sample_file() -> str:
    """
    Attach a small sample CSV file to the Slack thread.

    Returns:
        str: The outcome of the attachment.
    """
    # A `file_event`-shaped value passed to the custom stream surfaces as
    # a `file` wire event, which Welt uploads to the thread.
    writer = get_stream_writer()
    writer(file_event("sample.csv", b"fruit,count\napple,3\nbanana,5\n"))
    return "Attached sample.csv to the thread."


@tool
def sample_dangerous_action(action: str) -> str:
    """
    Pretend to run a dangerous or irreversible action the user asked for.

    A sample of the approval round trip: the interrupt below pauses the
    run until someone answers in the Slack thread — with the buttons, or
    by typing an instruction into the text field. Nothing is actually
    executed.

    Args:
        action (str): The action to pretend to run.

    Returns:
        str: The outcome of the action.
    """
    answer = interrupt(
        interrupt_reason(
            f"May I run this dangerous action? — {action}",
            [
                {"value": "y", "label": "Approve", "style": "primary"},
                {"value": "n", "label": "Cancel"},
            ],
            input={"label": "Or tell me what to do instead"},
        )
    )
    if answer == "y":
        return f"Ran: {action}. (This example doesn't actually run anything.)"
    if answer == "n":
        return "The action was cancelled by the user."
    return f"The action was not run. The user said instead: {answer}"


@task
def _drafted_report(topic: str) -> str:
    """
    Draft the report body once per run.

    A task rather than a plain function: LangGraph re-executes an
    interrupted tool from its start on resume, but a completed task is
    not re-executed — its saved result is reused. Drafting is the kind
    of work that must not run twice: a redraft (timestamped here to make
    that visible) would silently publish something other than what the
    human approved.

    Args:
        topic (str): The report topic.

    Returns:
        str: The draft report body.
    """
    drafted_at = datetime.now(timezone.utc).isoformat()
    return (
        f"# {topic}\n\nEverything about {topic} is going well.\n\n"
        f"_Drafted at {drafted_at}._\n"
    )


@tool
def sample_draft_report(topic: str) -> str:
    """
    Draft a small report on a topic and ask whether to publish it.

    A sample of work before an interrupt: the draft is written first,
    then the run pauses to show it for the publish decision. Approval
    uploads the approved draft to the thread as report.md.

    Args:
        topic (str): The report topic.

    Returns:
        str: The outcome of the draft.
    """
    draft = _drafted_report(topic).result()
    answer = interrupt(
        interrupt_reason(
            f"May I publish this draft?\n\n```\n{draft}```",
            [
                {"value": "y", "label": "Publish", "style": "primary"},
                {"value": "n", "label": "Discard"},
            ],
            input={"label": "Or tell me what to fix"},
        )
    )
    if answer == "y":
        writer = get_stream_writer()
        writer(file_event("report.md", draft.encode()))
        return (
            "The draft was approved and is already published to the thread"
            " as report.md. The publish flow is complete; no further review"
            " or approval is needed."
        )
    if answer == "n":
        return "The user discarded the draft; nothing was published."
    return f"The draft was not published. The user said instead: {answer}"


agent = create_agent(
    # Any Converse model with access enabled; an empty MODEL_ID means
    # unset, like Welt's own variables.
    model=ChatBedrockConverse(
        model_id=os.environ.get("MODEL_ID") or "global.anthropic.claude-sonnet-4-6"
    ),
    tools=[
        current_time,
        attach_sample_file,
        sample_dangerous_action,
        sample_draft_report,
    ],
    # Interrupts pause and resume through checkpoints, so a checkpointer
    # is required even though the conversation history lives in Slack.
    checkpointer=InMemorySaver(),
)


@app.entrypoint
async def invoke(payload: dict) -> AsyncIterator[dict]:
    """
    Stream a reply to the conversation or approval answers Welt sent.

    Args:
        payload (dict): The invocation payload: Converse-shaped `messages`
            built by Welt from the Slack thread (file blocks
            base64-encoded), or `interrupt_responses` carrying the button
            answers that resume an interrupted run.

    Yields:
        dict: The renderable subset of the agent's `astream` items.
    """
    global _interrupted_config

    if "interrupt_responses" in payload:
        config = _interrupted_config
        _interrupted_config = None
        if config is None:  # The microVM was recycled while the buttons waited.
            # The SDK reports the raise as an `error` event, and Welt renders
            # its resume-failure notice.
            raise RuntimeError("No interrupted run to resume in this session.")
        graph_input = Command(
            resume=decode_interrupt_responses(payload["interrupt_responses"])
        )
    else:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            yield {
                "data": "I received an empty conversation, "
                "so there is nothing to reply to."
            }
            return
        # A fresh thread per turn: Welt sends the whole Slack thread every
        # time, so letting the checkpointer stack turns into its own
        # history would double the conversation.
        config = RunnableConfig(configurable={"thread_id": uuid4().hex})
        graph_input = {"messages": decode_messages(messages)}

    interrupted = False
    # Reduce the stream to the JSON-serializable events Welt renders
    stream = agent.astream(
        graph_input, config, stream_mode=["messages", "updates", "custom"]
    )
    async for event in renderable_events(stream):
        if "interrupt" in event:
            interrupted = True
        yield event

    if interrupted:
        # Re-stashed on every interrupted stop, so a resume that interrupts
        # again keeps working.
        _interrupted_config = config


if __name__ == "__main__":
    app.run()
