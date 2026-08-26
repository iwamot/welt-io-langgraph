# welt-io-langgraph

[![pypi](https://img.shields.io/pypi/v/welt-io-langgraph.svg)](https://pypi.org/project/welt-io-langgraph/)
[![python](https://img.shields.io/pypi/pyversions/welt-io-langgraph.svg)](https://pypi.org/project/welt-io-langgraph/)
[![langgraph](https://img.shields.io/badge/dynamic/regex?url=https%3A%2F%2Fpypi.org%2Fpypi%2Fwelt-io-langgraph%2Fjson&search=langgraph%28%3E%3D%5B%5Cd.%5D%2B%29&replace=%241&label=langgraph)](https://pypi.org/project/langgraph/)

The [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) (Python) adapter for [Welt](https://github.com/iwamot/welt)'s wire contract.

## Install

```bash
uv add welt-io-langgraph
```

## Usage

`start_reply` and `renderable_events` are the wiring between Welt's payload and a LangGraph agent, so a deployable is your graph plus a short entrypoint:

```python
from collections.abc import AsyncIterator
from uuid import uuid4

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from welt_io_langgraph import renderable_events, start_reply

app = BedrockAgentCoreApp()
agent = create_agent("bedrock_converse:global.anthropic.claude-sonnet-4-6", checkpointer=InMemorySaver())


@app.entrypoint
async def invoke(payload: dict) -> AsyncIterator[dict]:
    config = RunnableConfig(configurable={"thread_id": uuid4().hex})
    async for event in renderable_events(start_reply(agent, payload, config)):
        yield event


if __name__ == "__main__":
    app.run()
```

A fresh thread per turn, because the Slack thread is the source of truth for conversation history. An agent with approval tools keeps the threads it needs to resume; [`examples/agent`](examples/agent) shows that as a map in `main.py`, holding each interrupted run's thread under the ids of the interrupts it raised and dropping the whole stop when the answers arrive.

See [`examples/agent`](examples/agent) for the full version — the smallest complete agent built on this package (text streaming, tool use, file output, file input, and human-approval tools). The sections below cover the entrypoint and the adapters it wires in.

## Supported Versions

### Welt

While both are 0.x, a welt-io-langgraph 0.Y release supports Welt v0.Y. From 1.0 on, a release supports any Welt release that shares its major version, and the minor versions move independently. Support is best effort either way, and other combinations come with no guarantee.

### LangGraph

The badge at the top states the range this release installs against. Every push and pull request runs the suite at both ends of it: the declared floor, and the newest release CI has picked up. That is best effort rather than a guarantee — the floor is where the suite was last seen to pass, so a later release may raise it, and no ceiling is declared at all. `langchain-core` comes along as a dependency and carries no floor of its own, because LangGraph asks for a newer one than anything here needs.

The badge follows the current release. For the range an older release declared, read that release's own metadata on PyPI.

Something misbehaving inside that range is worth an [issue](https://github.com/iwamot/welt-io-langgraph/issues).

## API

The wire between Welt and the agent is JSON, specified by [Welt's wire contract](https://github.com/iwamot/welt/blob/main/docs/wire.md) — plain LangGraph values do not fit it in either direction. Two functions adapt the inbound payload, two the outbound stream. `start_reply` wires the inbound pair into a stream (`interrupt_reason` serves the tools themselves); reach for the pieces directly when your entrypoint needs a shape of its own — messages to edit before the run, a graph to stream some other way. The messages they read are LangChain's, which carry [standard content blocks](https://docs.langchain.com/oss/python/langchain/messages).

### Reply

#### `start_reply(agent, payload, config)`

Starts the stream that replies to Welt's payload. It reads which envelope Welt sent — Converse-shaped `messages` for a conversation turn, `interrupt_responses` for the answers that resume an interrupted run — decodes it, and streams the graph on the result. What comes back is the graph's raw stream, for `renderable_events` to reduce.

Interrupts need the graph compiled with a checkpointer — LangGraph's own requirement, since pausing and resuming run through checkpoints; nothing here checks it. Which thread `config` names stays with the caller: a conversation turn belongs on a fresh one, because the Slack thread is the source of truth for conversation history and letting the checkpointer stack turns into its own history would double the conversation; a resume belongs on the thread the interrupted run stopped in, which the caller kept — under the interrupt ids Welt sends back, or however else suits the agent. Nothing is held here, so nothing here decides how long an unanswered approval stays answerable.

### Inbound

#### `decode_messages(messages)`

Turns Welt's Converse-shaped messages — built from the Slack thread, file bytes base64-encoded — into role/content message dicts that feed the graph input (`{"messages": decoded}`) as-is:

| Converse block | Standard content block |
|---|---|
| Text | Text |
| Image | Image |
| Document | File (the document's name carried as `name`, and with its format as the extension as `filename`) |
| Video | Video |

Each file-carrying block gets the media type LangChain models expect in place of the Converse format token, and the base64 data stays base64 — standard content blocks need no decoding.

#### `decode_interrupt_responses(responses)`

Turns Welt's resume payload — a mapping of interrupt id to the answer a human chose and the widget it came from — into the mapping `Command(resume=...)` takes, answering every pending interrupt at once. A plain interrupt resumes with the answer as the value it was given:

```python
agent.astream(
    Command(resume=decode_interrupt_responses(payload["interrupt_responses"])),
    config,
    stream_mode=["messages", "updates"],
)
```

The interrupt ids are LangGraph's own, as emitted by `renderable_events`; the config must point at the interrupted thread, which the [example agent](examples/agent)'s entrypoint stashes when an interrupt event goes by. Answers to a [`HumanInTheLoopMiddleware`](#gating-tools-with-humanintheloopmiddleware) request are rejoined into the decisions it resumes from, so the host app calls this the same way either way.

#### What arrives is taken as correct

Welt builds the payload and checks its own output against the wire contract before releasing it, so these two functions do no field validation of their own. A payload that departs from the contract is a bug on the sending side rather than an input to guard against, and it surfaces as an ordinary error from whatever touches it first — a `KeyError` or a `TypeError` here, or a refusal from LangChain or Bedrock further on.

The one thing `decode_messages` refuses outright is a content block of a kind Welt never sends. A `messages` turn carries only `text`, `image`, `document`, and `video` blocks; a `toolUse` or `toolResult` block is not a malformed one of those but a forged conversation turn, and rebuilt into history it would let a caller that is not Welt put words the model treats as its own past tool calls and their results into the run. It raises `ValueError`. This is a trust-boundary check, not the field validation the contract otherwise saves you from.

### Outbound

#### `renderable_events(stream, files_from=...)`

Reduces the `(mode, payload)` items of `astream(..., stream_mode=["messages", "updates"])` — whose values Welt does not render — to the events Welt renders:

| LangGraph emits | On the wire | In the Slack thread |
|---|---|---|
| Token deltas | `data` | The streamed reply |
| Tool calls and tool messages | `current_tool_use` / `tool_result` | "Using tool" indicators (tool output stays off the wire) |
| Image / file / video content blocks the model returns, or a tool named in `files_from` returns | `file` | An uploaded file ([size limits](https://github.com/iwamot/welt/blob/main/docs/wire.md#limits)) |
| Pending [interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) | `interrupt` | Buttons and/or a text field |

A run that stops for human input ends its stream with one `interrupt` event per pending interrupt — or per reviewed action, for a [`HumanInTheLoopMiddleware`](#gating-tools-with-humanintheloopmiddleware) request; agents that do not use interrupts see no change.

A tool hands files to the model for either of two reasons — to have it read them, or to give them to the human — and only the agent knows which is which, so name the tools whose files belong in the thread:

```python
async for event in renderable_events(stream, files_from={"create_sample_file"}):
```

A tool left out keeps its files to the model: one that reads a PDF for the model does not drop it into the thread as a side effect. A tool named there returns the file as a content block, which the model reads and Welt uploads:

```python
return [
    {"type": "text", "text": f"Created {name}.csv."},
    {
        "type": "file",
        "name": name,
        "mime_type": "text/csv",
        "base64": b64encode(csv).decode("ascii"),
    },
]
```

A tool message carries the name of the tool that produced it, so nothing else has to be passed in. Uploaded names come from the block's own `name` plus its media type, the block's kind for the rest (`image.png`). That name is also the model's handle on the document — Converse rejects a request whose messages carry two documents under one name, so a tool that returns files has to keep their names apart across the run: the example appends a short uuid to each.

Each event carries only what Welt reads, and an event with nothing to render — a text chunk the model left empty, a file with no bytes — is not sent at all.

#### `interrupt_reason(message, options=..., approve=..., reject=..., input=...)`

Builds the structured reason Welt renders as a message with the specified widgets — the approve and reject buttons Welt words and values itself (`approve`, `reject`), choice buttons of your own (`options`), a free-text field (`input`), or any combination. `approve` and `reject` answer with `True` and `False`, so a question whose decision is approval asks for them by name instead of inventing values; `{}` takes Welt's wording, and a `label` or `style` overrides it. An option's `value` is any JSON value, and the pressed button answers with it as it was declared. With no widget at all the message renders as itself and Welt's default buttons answer it. The specs are [the wire's own shapes](https://github.com/iwamot/welt/blob/main/docs/wire.md#interrupt), typed as `DecisionSpec`, `OptionSpec`, and `InputSpec`, and omitted fields keep Welt's defaults:

```python
answer = interrupt(
    interrupt_reason(
        "Deploy to prod?",
        approve={"label": "Deploy"},
        reject={"label": "Cancel"},
        input={"label": "Or type your answer"},
    )
)
```

Building the reason through this helper is what makes a typo an error. `interrupt` takes its value as `Any`, so a dict literal handed to it directly is checked by nothing, and Welt's reaction to a reason it cannot match is its default buttons — no error, no log, just widgets you did not ask for. The typed parameters catch a misspelled key before the run; the checks inside catch it in runs where no type checker was involved. What they check is the shape, not the size: how many buttons one Slack block holds, and how long a button value may be, are Welt's to enforce.

## Working with interrupts

[Welt's Interrupts doc](https://github.com/iwamot/welt/blob/main/docs/interrupts.md) covers the Slack side: how each reason renders, who can answer, multiple questions, and expiry. On the LangGraph side:

- **`interrupt` needs a checkpointer**, even though the conversation history lives in Slack — pausing and resuming run through checkpoints. An in-memory checkpointer works on AgentCore Runtime, where each session keeps its own microVM.
- **Start each conversation turn on a fresh thread.** Welt sends the whole Slack thread every turn by default, so letting the checkpointer stack turns into its own history would double the conversation. Resume alone reuses the interrupted thread's config. (An agent that keeps its own history instead sets `AGENT_MANAGES_HISTORY` on the Welt side.)
- **A plain interrupt value renders too.** Any non-structured value — `interrupt("Deploy to prod?")` — becomes a question with Welt's default buttons, whose answers arrive as `True` / `False`.
- **Code before `interrupt` runs again on resume.** LangGraph re-executes the interrupted node (or tool) from its start, so wrap whatever precedes an interrupt and must not run twice — side effects, or work that must match what the human approved — in a [LangGraph task](https://docs.langchain.com/oss/python/langgraph/functional-api#task): a completed task is not re-executed on resume; its saved result is reused. The [example agent](examples/agent)'s `sample_draft_report` shows the pattern.

## Gating tools with `HumanInTheLoopMiddleware`

LangChain's [`HumanInTheLoopMiddleware`](https://docs.langchain.com/oss/python/langchain/middleware/built-in#human-in-the-loop) pauses a tool before it runs, named in `interrupt_on` rather than written into the tool — which is what lets a tool the agent did not write, from a library or an MCP server, be gated at all. It works over Welt as-is:

```python
create_agent(
    model=...,
    tools=[send_email],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": InterruptOnConfig(allowed_decisions=["approve", "reject"])
            }
        )
    ],
    checkpointer=InMemorySaver(),
)
```

The decisions an action allows become its widgets, and the answer comes back as the decision the widget stands for. Nothing here words a button: the approve and reject buttons are asked of Welt by name, so what approval is called stays Welt's to say.

| Allowed decision | In the Slack thread | On approval of that answer |
|---|---|---|
| `approve` | Welt's approve button | The tool runs as the model called it |
| `reject` | Welt's reject button | The tool does not run; the model is told it was rejected |
| `respond` | A free-text field labelled after the call | The tool does not run; the typed text reaches the model as the tool's result |
| `edit` | Nothing | — |

- **Write the question's body with `description`.** Left out, the middleware writes its own — a prefix over the tool's name and its arguments as Python renders a dict — since it knows nothing about Slack. `InterruptOnConfig`'s `description` takes a string, or a callable over the call, the state, and the runtime; the [example agent](examples/agent) formats the arguments as JSON in a code block. It is the human's whole view of what they are approving.
- **The free-text field is labelled `Answer instead of running <tool>`.** Welt labels a field `Answer`, which reads the same whether the tool runs or not; naming the call says that answering here replaces running it, and keeps the field distinct from one opened with `interrupt` inside a tool, where what is typed goes to the tool rather than around it.
- **A press is identified by the widget it came from.** Welt says which widget produced each answer, so a press maps to the decision its button stands for and submitted text becomes the `respond` decision — a typed "approve" included, since what it means is read where meaning belongs: it reaches the model as the tool's answer.
- **`edit` has no widget**, rewriting an action's arguments being a form the wire has no shape for. An action allowing `edit` alongside others is asked about with the widgets for the rest; one allowing `edit` alone is passed through to Welt's fallback rendering, whose answers the middleware cannot resume from.
- **One request becomes one question per action.** The middleware bundles every gated call of a turn into a single interrupt and resumes from one decision per action; a Welt stop carries as many questions as it likes, so each action is asked about on its own — buttons per action, answered in any order, and Welt resumes the run once all of them are answered.
- **Write the interrupt yourself when the question depends on the tool's own work.** The middleware knows a call's name and arguments, nothing the tool computes, so showing something the tool produced — a draft, a diff, a dry run — needs `interrupt` inside the tool, as `sample_draft_report` does.

## License

MIT
