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

import json
import os
from base64 import b64encode
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import uuid4

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langchain.agents.middleware.types import AgentState
from langchain.tools import tool
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import ToolCall
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import task
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from welt_io_langgraph import (
    decode_interrupt_responses,
    decode_messages,
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
    return datetime.now(UTC).isoformat()


def _document_name(stem: str) -> str:
    """
    Name a document apart from every other document of the run.

    Converse rejects a request whose messages carry two documents under one
    name, and the tool that returns a document is the only one placed to
    keep it apart — it cannot know what the rest of the run named theirs,
    so it pays the going price of a random tail. The name is the model's
    handle on the document, and the filename Welt uploads it under.

    Args:
        stem (str): The readable part of the name.

    Returns:
        str: The stem, tailed apart from the run's other documents.
    """
    return f"{stem}-{uuid4().hex[:8]}"


@tool
def create_sample_file() -> list[dict]:
    """
    Create a small sample CSV file.

    Returns:
        list[dict]: The outcome, the file carried as a content block —
            which reaches the model, and the Slack thread because the
            entrypoint takes files from this tool.
    """
    name = _document_name("sample")
    csv = b"fruit,count\napple,3\nbanana,5\n"
    return [
        {"type": "text", "text": f"Created {name}.csv."},
        {
            "type": "file",
            # `name` is what Converse and the Slack upload use (dots are
            # invalid there); `filename` is what OpenAI-compatible
            # endpoints type the file by, so it carries the extension.
            "name": name,
            "filename": f"{name}.csv",
            "mime_type": "text/csv",
            "base64": b64encode(csv).decode("ascii"),
        },
    ]


@tool
def sample_dangerous_action(action: str) -> str:
    """
    Pretend to run a dangerous or irreversible action the user asked for.

    A sample of approval from outside the tool: the middleware below names
    this tool in `interrupt_on`, so the run pauses for an answer in the
    Slack thread before this body starts. Nothing here knows about the
    approval — which is what lets a tool the agent did not write, from a
    library or an MCP server, be gated the same way. Nothing is actually
    executed.

    Args:
        action (str): The action to pretend to run.

    Returns:
        str: The outcome of the action.
    """
    return f"Ran: {action}. (This example doesn't actually run anything.)"


def _approval_description(
    tool_call: ToolCall, state: AgentState, runtime: Runtime
) -> str:
    """
    Write the body of the question the middleware asks about one call.

    Without this the middleware writes its own — the tool's name and its
    arguments as Python renders a dict — since it knows nothing about
    Slack. The question is the human's whole view of what they are
    approving, so it is worth writing: the arguments as JSON in a code
    block read the way the rest of the thread does.

    Args:
        tool_call (ToolCall): The call awaiting approval.
        state (AgentState): The agent state, unused here.
        runtime (Runtime): The runtime, unused here.

    Returns:
        str: The markdown Welt shows above the widgets.
    """
    arguments = json.dumps(tool_call["args"], indent=2, ensure_ascii=False)
    return f"May I run `{tool_call['name']}`?\n```\n{arguments}\n```"


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
    drafted_at = datetime.now(UTC).isoformat()
    return (
        f"# {topic}\n\nEverything about {topic} is going well.\n\n"
        f"_Drafted at {drafted_at}._\n"
    )


@tool
def sample_draft_report(topic: str) -> str | list[dict]:
    """
    Draft a small report on a topic and ask whether to publish it.

    A sample of work before an interrupt: the draft is written first,
    then the run pauses to show it for the publish decision. Approval
    returns the approved draft as a markdown file.

    Args:
        topic (str): The report topic.

    Returns:
        str | list[dict]: The outcome, the approved draft carried as a
            content block.
    """
    draft = _drafted_report(topic).result()
    answer = interrupt(
        interrupt_reason(
            f"May I publish this draft?\n\n```\n{draft}```",
            [
                {"value": "Publish", "style": "primary"},
                {"value": "Discard"},
            ],
            input={"label": "Or type your answer"},
        )
    )
    if answer == "Publish":
        name = _document_name("report")
        return [
            {
                "type": "text",
                "text": (
                    "The user answered the publish question in the thread by"
                    f" pressing Publish, so this draft is already published"
                    f" there as {name}.md. The publish flow is complete;"
                    " nothing is left to approve."
                ),
            },
            {
                "type": "file",
                # `name` for Converse and the Slack upload, `filename` with
                # the extension for OpenAI-compatible endpoints — as in
                # create_sample_file.
                "name": name,
                "filename": f"{name}.md",
                "mime_type": "text/markdown",
                "base64": b64encode(draft.encode()).decode("ascii"),
            },
        ]
    if answer == "Discard":
        return "The user discarded the draft; nothing was published."
    return f"The draft was not published. The user answered: {answer}"


# The tools whose files belong in the Slack thread. A tool left out keeps
# its files to the model — this agent has none, but an agent that reads
# documents for the model would.
_FILES_FROM = {"create_sample_file", "sample_draft_report"}

# The model is the one place that decides which Bedrock endpoint and API the
# agent talks to; nothing else in this file depends on that choice.
# ChatBedrockConverse speaks Converse to bedrock-runtime, so MODEL_ID takes
# any Converse model there. An empty MODEL_ID means unset, like Welt's own
# variables.
_model_id = os.environ.get("MODEL_ID") or "global.anthropic.claude-sonnet-4-6"
model = ChatBedrockConverse(model_id=_model_id)
# For bedrock-mantle, Bedrock's OpenAI-compatible endpoint, swap in the
# Mantle model from `langchain-aws[openai]` instead — short-term keys are
# derived from your AWS credentials (or set AWS_BEARER_TOKEN_BEDROCK).
# `base_url` pins the `/openai/v1` base path that `xai.grok-4.*` sits on;
# models served on `/v1` (`openai.gpt-oss-*`, ...) drop it for the default:
# from langchain_aws import ChatOpenAIMantle
# _region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
# model = ChatOpenAIMantle(
#     model=_model_id,
#     use_responses_api=True,
#     base_url=f"https://bedrock-mantle.{_region}.api.aws/openai/v1",
# )

agent = create_agent(
    model=model,
    tools=[
        current_time,
        create_sample_file,
        sample_dangerous_action,
        sample_draft_report,
    ],
    # Approval by declaration: a tool named here pauses before it runs,
    # and the decisions it allows become the widgets Welt renders —
    # `approve` and `reject` as buttons, `respond` as a free-text field
    # whose text is returned to the model in place of the tool's own
    # result. `edit` is left out: rewriting an action's arguments asks for
    # a form the wire has no shape for.
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "sample_dangerous_action": InterruptOnConfig(
                    allowed_decisions=["approve", "reject", "respond"],
                    description=_approval_description,
                )
            }
        )
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
        # The envelope key is the discriminator, so a payload without
        # either one is Welt's bug, and the KeyError it raises is reported
        # as an `error` event by the SDK.
        messages = decode_messages(payload["messages"])
        # A fresh thread per turn: Welt sends the whole Slack thread every
        # time, so letting the checkpointer stack turns into its own
        # history would double the conversation.
        config = RunnableConfig(configurable={"thread_id": uuid4().hex})
        graph_input = {"messages": messages}

    interrupted = False
    # Reduce the stream to the JSON-serializable events Welt renders
    stream = agent.astream(graph_input, config, stream_mode=["messages", "updates"])
    async for event in renderable_events(stream, files_from=_FILES_FROM):
        if "interrupt" in event:
            interrupted = True
        yield event

    if interrupted:
        # Re-stashed on every interrupted stop, so a resume that interrupts
        # again keeps working.
        _interrupted_config = config


if __name__ == "__main__":
    app.run()
