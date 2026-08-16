# Example Agent

The example agent for [Welt](https://github.com/iwamot/welt): the smallest complete agent that exercises the wire in both directions through welt-io-langgraph.

## Stack

| Package | Role |
|---------|------|
| [Bedrock AgentCore SDK](https://github.com/aws/bedrock-agentcore-sdk-python) | Serves the endpoint |
| [LangChain / LangGraph](https://docs.langchain.com/oss/python/langchain/agents) | Runs the model and the tools (`create_agent`) |
| [langchain-aws](https://github.com/langchain-ai/langchain-aws) | Provides the Bedrock Converse chat model |
| welt-io-langgraph | Adapts the wire to Welt |

## Run Locally

The agent runs on your machine as-is — the AgentCore SDK serves the same HTTP surface locally, on port 8080, that AgentCore Runtime serves in the cloud, and [Welt's local mode](https://github.com/iwamot/welt#quick-start) invokes it there.

Fetch the agent and run it with [uv](https://docs.astral.sh/uv/):

```sh
curl -O https://raw.githubusercontent.com/iwamot/welt-io-langgraph/main/examples/agent/main.py
uv run --with bedrock-agentcore --with langchain --with langchain-aws \
  --with langgraph --with welt-io-langgraph --with "botocore[crt]" main.py
```

The process needs AWS credentials the standard SDK way — environment variables, `AWS_PROFILE`, an SSO session, `aws login` (which is why `botocore[crt]` is included) — because the model runs on Amazon Bedrock. `MODEL_ID` takes any Converse model with access enabled in the Amazon Bedrock console, in the region your credentials point at; unset, the agent uses `global.anthropic.claude-sonnet-4-6`.

One difference from the cloud: AgentCore Runtime gives every session its own microVM, while the local server is a single process for all sessions — the agent stashes an interrupted run in one slot, so keep interrupt experiments to one thread at a time.

## Deploy

Deploy with the [AgentCore CLI](https://github.com/aws/agentcore-cli):

```sh
agentcore create --name WeltExample --framework LangChain_LangGraph --model-provider Bedrock --memory none
cd WeltExample

curl -o app/WeltExample/main.py https://raw.githubusercontent.com/iwamot/welt-io-langgraph/main/examples/agent/main.py
uv add --project app/WeltExample welt-io-langgraph langchain-aws

agentcore deploy
```

The agent defaults to `global.anthropic.claude-sonnet-4-6`, so enable access for it in the Amazon Bedrock console, in the region you deployed to, or point the `MODEL_ID` environment variable at another Converse model. Note the agent runtime ARN from the deploy output: Welt's `AGENT_ARN` points at it.

## Tools

- `current_time` — the minimal tool: plain text streaming, nothing else. Ask "what time is it?" to see tool use in the thread.
- `create_sample_file` — writes a small CSV and returns it as a content block, which the model reads and Welt uploads to the thread. Its name carries a random tail (`sample-3f2a1b9c.csv`) because a document's name has to be unique across the run. Ask it for a sample file.
- `sample_dangerous_action` — a pretend dangerous action (no side effects, no extra AWS permissions) gated by `HumanInTheLoopMiddleware`: the tool itself carries no interrupt code, and the run pauses before its body starts. Welt renders **Approve** / **Reject** buttons plus a free-text field labelled **Answer instead of running sample_dangerous_action** in the Slack thread, and whichever answer comes first resumes the run — a press decides whether the tool runs, while typed text answers on the tool's behalf without running it. Ask "deploy to prod", then press a button or type something like "not now". See [Welt's Interrupts doc](https://github.com/iwamot/welt/blob/main/docs/interrupts.md) for the round trip.
- `sample_draft_report` — drafts a small report, pauses to show the draft for approval, and on approval returns it as a markdown file (`report-8f3a2c1d.md`, tailed for the same reason). The draft-then-ask order is the LangGraph interrupt pitfall: an interrupted tool re-executes from its start on resume, so the drafting runs inside a [LangGraph task](https://docs.langchain.com/oss/python/langgraph/functional-api#task) — its saved result is reused, keeping the published file identical to the approved draft (the draft is timestamped, so a silent redraft would show). Ask "draft a report about apples", then answer the buttons.

The two that produce files are named in the entrypoint's `files_from` — that is what puts their files in the thread, and a tool left out of it would hand its files to the model alone.

## Optional: file input

The agent can also read files uploaded to Slack — disabled by default. To try it, set in Welt's `.env`:

```sh
FILE_INPUT_MODALITIES=image,document
```

These two are what the default model (Anthropic Claude) accepts; `video` needs a model that takes video input — see [Welt's Files doc](https://github.com/iwamot/welt/blob/main/docs/files.md).
