# welt-io-langgraph

[![pypi](https://img.shields.io/pypi/v/welt-io-langgraph.svg)](https://pypi.org/project/welt-io-langgraph/)
[![python](https://img.shields.io/pypi/pyversions/welt-io-langgraph.svg)](https://pypi.org/project/welt-io-langgraph/)

The [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) (Python) adapter for [Welt](https://github.com/iwamot/welt)'s wire contract.

## Install

```bash
uv add welt-io-langgraph
```

## Usage

See [`examples/agent`](examples/agent) — the smallest complete agent built on this package (text streaming, tool use, file output, file input, and human-approval tools). The sections below explain the adapters it wires in.

## API

The wire between Welt and the agent is JSON, specified by [Welt's wire contract](https://github.com/iwamot/welt/blob/main/docs/wire.md) — plain LangGraph values do not fit it in either direction. Two functions adapt the inbound payload, three the outbound stream. The adapters target LangGraph 1.x and LangChain 1.x, whose messages carry [standard content blocks](https://docs.langchain.com/oss/python/langchain/messages).

### Inbound

#### `decode_messages(messages)`

Turns Welt's Converse-shaped messages — built from the Slack thread, file bytes base64-encoded — into role/content message dicts that feed the graph input (`{"messages": decoded}`) as-is:

| Converse block | Standard content block |
|---|---|
| Text | Text |
| Image | Image |
| Document | File (the document's name carried as `filename`) |
| Video | Video |

Each file-carrying block gets the media type LangChain models expect in place of the Converse format token, and the base64 data stays base64 — standard content blocks need no decoding. Malformed entries are skipped.

#### `decode_interrupt_responses(responses)`

Turns Welt's resume payload — a mapping of interrupt id to the answer a human chose — into the mapping `Command(resume=...)` takes, answering every pending interrupt at once:

```python
agent.astream(
    Command(resume=decode_interrupt_responses(payload["interrupt_responses"])),
    config,
    stream_mode=["messages", "updates"],
)
```

The interrupt ids are LangGraph's own, as emitted by `renderable_events`; the config must point at the interrupted thread, which the host app stashes when an interrupt event goes by (see the [example agent](examples/agent)). Answers to a [`HumanInTheLoopMiddleware`](#gating-tools-with-humanintheloopmiddleware) request are rejoined into the decisions it resumes from, so the host app calls this the same way either way.

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

#### `file_event(name, data)`

Builds the same `file` event from a filename and raw bytes, for the files the host app attaches itself:

```python
yield file_event("report.csv", csv_bytes)
```

Tools have no use for it — they hand files to the agent as content blocks, and `files_from` decides which of those reach the thread.

#### `interrupt_reason(message, options=..., input=...)`

Builds the structured reason Welt renders as a message with the specified widgets — choice buttons (`options`), a free-text field (`input`), or both. The specs are [the wire's own shapes](https://github.com/iwamot/welt/blob/main/docs/wire.md#interrupt); omitted fields keep Welt's defaults, and a typo becomes an immediate `ValueError` instead of a silent fallback to Welt's default rendering:

```python
answer = interrupt(
    interrupt_reason(
        "Deploy to prod?",
        [
            {"value": "y", "label": "Deploy", "style": "primary"},
            {"value": "n", "label": "Cancel"},
        ],
        input={"label": "Or type your answer"},
    )
)
```

## Working with interrupts

[Welt's Interrupts doc](https://github.com/iwamot/welt/blob/main/docs/interrupts.md) covers the Slack side: how each reason renders, who can answer, multiple questions, and expiry. On the LangGraph side:

- **`interrupt` needs a checkpointer**, even though the conversation history lives in Slack — pausing and resuming run through checkpoints. An in-memory checkpointer works on AgentCore Runtime, where each session keeps its own microVM.
- **Start each conversation turn on a fresh thread.** Welt sends the whole Slack thread every turn by default, so letting the checkpointer stack turns into its own history would double the conversation. Resume alone reuses the interrupted thread's config. (An agent that keeps its own history instead sets `AGENT_MANAGES_HISTORY` on the Welt side.)
- **A plain interrupt value renders too.** Any non-structured value — `interrupt("Deploy to prod?")` — becomes a question with Welt's default **Approve** / **Deny** buttons, whose answers arrive as `y` / `n`.
- **Code before `interrupt` runs again on resume.** LangGraph re-executes the interrupted node (or tool) from its start, so wrap whatever precedes an interrupt and must not run twice — side effects, or work that must match what the human approved — in a [LangGraph task](https://docs.langchain.com/oss/python/langgraph/durable-execution): a completed task is not re-executed on resume; its saved result is reused. The [example agent](examples/agent)'s `sample_draft_report` shows the pattern.

## Gating tools with `HumanInTheLoopMiddleware`

LangChain's [`HumanInTheLoopMiddleware`](https://docs.langchain.com/oss/python/langchain/middleware#human-in-the-loop) pauses a tool before it runs, named in `interrupt_on` rather than written into the tool — which is what lets a tool the agent did not write, from a library or an MCP server, be gated at all. It works over Welt as-is:

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

The decisions an action allows become its widgets, and the answer comes back as the decision the widget stands for:

| Allowed decision | In the Slack thread | On approval of that answer |
|---|---|---|
| `approve` | An **Approve** button | The tool runs as the model called it |
| `reject` | A **Reject** button | The tool does not run; the model is told it was rejected |
| `respond` | A free-text field | The tool does not run; the typed text reaches the model as the tool's result |
| `edit` | Nothing | — |

- **A press is identified by the value it carries.** The buttons carry values the adapter mints, so a press maps to the decision its button stands for and every other answer travels on as text — a typed "approve" included, since the wire says which question was answered but not which widget answered it. What such an answer means is read where meaning belongs: it reaches the model as the tool's answer.
- **`edit` has no widget**, rewriting an action's arguments being a form the wire has no shape for. An action allowing `edit` alongside others is asked about with the widgets for the rest; one allowing `edit` alone is passed through to Welt's fallback rendering, whose answers the middleware cannot resume from.
- **One request becomes one question per action.** The middleware bundles every gated call of a turn into a single interrupt and resumes from one decision per action; a Welt stop carries as many questions as it likes, so each action is asked about on its own — buttons per action, answered in any order, and Welt resumes the run once all of them are answered.
- **Write the interrupt yourself when the question depends on the tool's own work.** The middleware knows a call's name and arguments, nothing the tool computes, so showing something the tool produced — a draft, a diff, a dry run — needs `interrupt` inside the tool, as `sample_draft_report` does.

## Supported Versions

Welt releases first; welt-io-langgraph follows, mirroring the minor version. While both are 0.x, a welt-io-langgraph 0.Y release supports Welt v0.Y — other combinations may work, but come with no guarantee.

## License

MIT
