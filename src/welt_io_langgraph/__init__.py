"""Adapters for the two directions of Welt's wire contract.

The wire between Welt and the agent is JSON, and plain LangGraph values do
not fit it in either direction:

- Inbound, Welt sends Bedrock Converse-shaped messages with base64-encoded
  file bytes, while a LangGraph agent consumes LangChain messages whose
  file parts are standard content blocks carrying a media type.
  `decode_messages` rebuilds each message accordingly. Welt resumes an
  interrupted run with a plain mapping of interrupt id to the chosen
  answer; `decode_interrupt_responses` turns it into the mapping
  `Command(resume=...)` takes.
- Outbound, raw `astream` items carry values that are not
  JSON-serializable (message objects, Interrupt objects), which the
  AgentCore Runtime SDK would degrade to a plain string on the SSE wire.
  `renderable_events` reduces the stream to the events Welt renders, with
  generated files base64-encoded — the inbound encoding in reverse.
  `file_event` builds the same `file` event from a name and raw bytes, so
  tools can attach files of their own through LangGraph's custom stream.
  `interrupt_reason` builds the reason shape Welt renders as a message
  with buttons, a free-text field, or both when a tool interrupts for
  human input.
"""

import base64
from collections.abc import AsyncIterator, Sequence

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "decode_interrupt_responses",
    "decode_messages",
    "file_event",
    "interrupt_reason",
    "renderable_events",
]


# The media types LangChain models expect, by Converse format token.
_IMAGE_MIME_TYPES = {
    "gif": "image/gif",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}

_DOCUMENT_MIME_TYPES = {
    "csv": "text/csv",
    "doc": "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "html": "text/html",
    "md": "text/markdown",
    "pdf": "application/pdf",
    "txt": "text/plain",
    "xls": "application/vnd.ms-excel",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

_VIDEO_MIME_TYPES = {
    "flv": "video/x-flv",
    "mkv": "video/x-matroska",
    "mov": "video/quicktime",
    "mp4": "video/mp4",
    "mpeg": "video/mpeg",
    "mpg": "video/mpeg",
    "three_gp": "video/3gpp",
    "webm": "video/webm",
    "wmv": "video/x-ms-wmv",
}


def decode_messages(messages: list) -> list:
    """
    Decode Welt's messages payload into the messages LangGraph consumes.

    A LangGraph agent takes LangChain messages, whose file parts are
    standard content blocks carrying a media type instead of a Converse
    format token, and whose base64 data needs no decoding. This walks the
    payload's `messages` value and rebuilds each message — text blocks
    become text blocks, image blocks image blocks, and document and video
    blocks file and video blocks. The result feeds the graph input
    (`{"messages": decoded}`) as-is. Malformed entries are skipped, since
    they come from the wire rather than the developer; messages left with
    no blocks are dropped.

    Args:
        messages (list): The `messages` value of Welt's payload.

    Returns:
        list: Role/content message dicts for the graph input.
    """
    decoded = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        content = _decoded_content(message.get("content"), files=role == "user")
        if content:
            decoded.append({"role": role, "content": content})
    return decoded


def _decoded_content(content: object, *, files: bool) -> list[dict]:
    """
    Decode one message's Converse content blocks into standard blocks.

    Args:
        content (object): The message's `content` value.
        files (bool): Whether to decode file-carrying blocks — Welt embeds
            them in user messages only; assistant messages keep just their
            text.

    Returns:
        list[dict]: The standard content blocks, in content order.
    """
    if not isinstance(content, list):
        return []
    blocks: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str):
            blocks.append({"type": "text", "text": text})
            continue
        if not files:
            continue
        decoded = (
            _image_block(block.get("image"))
            or _document_block(block.get("document"))
            or _video_block(block.get("video"))
        )
        if decoded is not None:
            blocks.append(decoded)
    return blocks


def _image_block(media: object) -> dict | None:
    """
    Decode a Converse image block into a standard image block.

    Args:
        media (object): The `image` value of a Converse content block.

    Returns:
        dict | None: The image block, or None for blocks without base64
            bytes or with a format that maps to no media type.
    """
    data = _source_bytes(media)
    if data is None or not isinstance(media, dict):
        return None
    mime_type = _media_type(media, _IMAGE_MIME_TYPES)
    if mime_type is None:
        return None
    return {"type": "image", "base64": data, "mime_type": mime_type}


def _document_block(media: object) -> dict | None:
    """
    Decode a Converse document block into a standard file block.

    Args:
        media (object): The `document` value of a Converse content block.

    Returns:
        dict | None: The file block — with the document's name as
            `filename`, which the model integrations pass on — or None for
            blocks without base64 bytes or with a format that maps to no
            media type.
    """
    data = _source_bytes(media)
    if data is None or not isinstance(media, dict):
        return None
    mime_type = _media_type(media, _DOCUMENT_MIME_TYPES)
    if mime_type is None:
        return None
    block: dict = {"type": "file", "base64": data, "mime_type": mime_type}
    name = media.get("name")
    if isinstance(name, str) and name:
        block["filename"] = name
    return block


def _video_block(media: object) -> dict | None:
    """
    Decode a Converse video block into a standard video block.

    Args:
        media (object): The `video` value of a Converse content block.

    Returns:
        dict | None: The video block, or None for blocks without base64
            bytes or with a format that maps to no media type.
    """
    data = _source_bytes(media)
    if data is None or not isinstance(media, dict):
        return None
    mime_type = _media_type(media, _VIDEO_MIME_TYPES)
    if mime_type is None:
        return None
    return {"type": "video", "base64": data, "mime_type": mime_type}


def _media_type(media: dict, mime_types: dict[str, str]) -> str | None:
    """
    Look up the media type of a Converse file block's format token.

    Args:
        media (dict): The block's value (image, document, or video).
        mime_types (dict[str, str]): The media types by format token.

    Returns:
        str | None: The media type, or None for a missing or unknown
            format.
    """
    file_format = media.get("format")
    if not isinstance(file_format, str):
        return None
    return mime_types.get(file_format)


def _source_bytes(media: object) -> str | None:
    """
    Extract the base64 bytes of a Converse file block.

    Args:
        media (object): The block's value (image, document, or video).

    Returns:
        str | None: The base64 string, or None if the block carries none.
    """
    if not isinstance(media, dict):
        return None
    source = media.get("source")
    if not isinstance(source, dict):
        return None
    data = source.get("bytes")
    if not isinstance(data, str) or not data:
        return None
    return data


def decode_interrupt_responses(responses: dict) -> dict:
    """
    Decode Welt's interrupt answers into LangGraph's resume input.

    Welt resumes an interrupted run with a payload mapping each interrupt
    id to the answer a human chose in the thread. LangGraph resumes from
    the same mapping — the returned dict feeds `Command(resume=...)`
    directly, answering every pending interrupt at once. Entries whose
    answer is not a string are skipped, since they come from the wire
    rather than the developer.

    Args:
        responses (dict): The `interrupt_responses` value of Welt's
            payload.

    Returns:
        dict: The interrupt id to answer mapping for `Command(resume=...)`.
    """
    return {
        interrupt_id: answer
        for interrupt_id, answer in responses.items()
        if isinstance(answer, str)
    }


def file_event(name: str, data: bytes) -> dict:
    """
    Build a `file` wire event, which Welt uploads to the Slack thread.

    `renderable_events` emits these for the files a tool or the model
    generates; this builds the same event from arbitrary bytes, for agents
    that attach files of their own. Yield it from the host app alongside
    the reduced stream, or pass it to `get_stream_writer()` inside a tool
    to attach a file from there — `renderable_events` passes it through by
    itself.

    Args:
        name (str): The upload filename, extension included.
        data (bytes): The raw file bytes.

    Returns:
        dict: The `file` event (name plus base64 bytes).

    Raises:
        ValueError: If the name is empty (Welt drops a nameless file).
    """
    if not name:
        raise ValueError("name must not be empty")
    return {"file": {"name": name, "bytes": base64.b64encode(data).decode("ascii")}}


_BUTTON_STYLES = frozenset({"primary", "danger"})


def interrupt_reason(
    message: str,
    options: Sequence[dict] | None = None,
    *,
    input: dict | None = None,
) -> dict:
    """
    Build an interrupt reason that Welt renders as the specified widgets.

    Welt renders this shape as `message` followed by one button per option
    (`options`), a free-text field whose submitted text becomes the
    interrupt's response (`input`), or both — whichever answer comes
    first, a pressed button or the submitted text, settles the question.
    Both widget specs are the wire's own shapes; building them through
    this helper turns a typo into an immediate ValueError instead of a
    silent fallback to Welt's default rendering.

    Args:
        message (str): The text Welt shows above the widgets.
        options (Sequence[dict] | None): One dict per button: a required
            `value` (what the interrupting tool receives as the response
            when the button is pressed), an optional `label` (the button
            text; omitted, Welt shows the value), and an optional `style`
            ("primary" or "danger").
        input (dict | None): The free-text field: an optional `label` (the
            field's label) and an optional `multiline` (whether the field
            accepts multiple lines) — `{}` takes Welt's defaults for both.
            None omits the field.

    Returns:
        dict: The reason to pass to `interrupt`.

    Raises:
        ValueError: If the message is empty, neither options nor input is
            given, or a widget spec is off — an unknown key, a missing
            value, an empty or non-string value/label, a style that is not
            "primary" or "danger", or a non-boolean multiline.
    """
    if not message:
        raise ValueError("message must not be empty")
    if options is None and input is None:
        raise ValueError("options or input must be given")
    reason: dict = {"message": message}
    if options is not None:
        reason["options"] = _built_options(options)
    if input is not None:
        reason["input"] = _built_input(input)
    return reason


_OPTION_KEYS = frozenset({"value", "label", "style"})


def _built_options(options: Sequence[dict]) -> list[dict]:
    """
    Validate and build the `options` entries of a structured reason.

    Only the keys the wire knows are passed through; an omitted label
    stays omitted, leaving its default (the value) to Welt.

    Args:
        options (Sequence[dict]): One dict per button: a required `value`,
            an optional `label`, and an optional `style`.

    Returns:
        list[dict]: The option dicts of the reason shape.

    Raises:
        ValueError: If no options are given, an option carries an unknown
            key, a value is missing, empty, or not a string, a label is
            empty or not a string, or a style is not "primary" or
            "danger".
    """
    if not options:
        raise ValueError("options must not be empty")
    built: list[dict] = []
    for option in options:
        unknown = set(option) - _OPTION_KEYS
        if unknown:
            raise ValueError(f"unknown option keys: {sorted(unknown)}")
        value = option.get("value")
        if not isinstance(value, str) or not value:
            raise ValueError("option value must be a non-empty string")
        entry: dict = {"value": value}
        if "label" in option:
            label = option["label"]
            if not isinstance(label, str) or not label:
                raise ValueError("option label must be a non-empty string")
            entry["label"] = label
        if "style" in option:
            style = option["style"]
            if style not in _BUTTON_STYLES:
                raise ValueError(f"style must be 'primary' or 'danger': {style!r}")
            entry["style"] = style
        built.append(entry)
    return built


_INPUT_KEYS = frozenset({"label", "multiline"})


def _built_input(input_spec: dict) -> dict:
    """
    Validate and build the `input` entry of a structured reason.

    Only the keys the wire knows are passed through; omitted ones stay
    omitted, leaving their defaults to Welt.

    Args:
        input_spec (dict): The field spec: an optional `label` and an
            optional `multiline`.

    Returns:
        dict: The `input` entry of the reason shape.

    Raises:
        ValueError: If the spec carries an unknown key, an empty or
            non-string label, or a non-boolean multiline.
    """
    unknown = set(input_spec) - _INPUT_KEYS
    if unknown:
        raise ValueError(f"unknown input keys: {sorted(unknown)}")
    built: dict = {}
    if "label" in input_spec:
        label = input_spec["label"]
        if not isinstance(label, str) or not label:
            raise ValueError("input label must be a non-empty string")
        built["label"] = label
    if "multiline" in input_spec:
        multiline = input_spec["multiline"]
        if not isinstance(multiline, bool):
            raise ValueError("input multiline must be a bool")
        built["multiline"] = multiline
    return built


async def renderable_events(stream: AsyncIterator) -> AsyncIterator[dict]:
    """
    Reduce a LangGraph stream to the events Welt renders.

    Iterates the `(mode, payload)` items of
    `astream(..., stream_mode=["messages", "updates", "custom"])` and
    yields the wire's renderable subset: text chunks (`data`), tool-use
    indicators (`current_tool_use` / `tool_result`, slimmed so tool output
    stays off the wire), generated files (`file` — the image, file, and
    video content blocks a tool or the model returns, plus every
    `file_event`-shaped value a tool passes to `get_stream_writer()`), and
    interrupts (`interrupt` — each pending interrupt's id and value, the
    value passed through unmodified as the reason since interpreting it is
    the renderer's job). Everything else is dropped.

    Args:
        stream (AsyncIterator): The `(mode, payload)` items of a LangGraph
            stream.

    Yields:
        dict: The renderable wire events, in stream order.
    """
    async for item in stream:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        mode, payload = item
        if mode == "messages":
            for event in _message_events(payload):
                yield event
        elif mode == "updates":
            for event in _interrupt_events(payload):
                yield event
        elif mode == "custom":
            event = _custom_file_event(payload)
            if event is not None:
                yield event


def _message_events(payload: object) -> list[dict]:
    """
    Extract renderable events from a `messages` stream item.

    The payload is a `(message, metadata)` pair. Token deltas
    (AIMessageChunk) carry text — a `data` event — and the opening
    tool-call chunks — a `current_tool_use` event each. A complete
    assistant message (emitted for nodes that return one without
    streaming) carries the same as text and tool calls, plus a `file`
    event per file-carrying content block for models that generate files.
    A tool message becomes a `tool_result` event slimmed to the tool call
    id and status, followed by a `file` event per file-carrying content
    block the tool returned.

    Args:
        payload (object): The `messages` payload of a stream item.

    Returns:
        list[dict]: The renderable events, in message order.
    """
    if not isinstance(payload, tuple) or not payload:
        return []
    message = payload[0]
    kind = getattr(message, "type", None)
    if kind == "AIMessageChunk":
        return _chunk_events(message)
    if kind == "ai":
        return _assistant_events(message)
    if kind == "tool":
        return _tool_events(message)
    return []


def _chunk_events(message: object) -> list[dict]:
    """
    Extract renderable events from an assistant token delta.

    Args:
        message (object): The AIMessageChunk of a `messages` payload.

    Returns:
        list[dict]: A `data` event for the chunk's text and a
            `current_tool_use` event per opening tool-call chunk — the one
            carrying the call's id and name; the argument fragments that
            follow carry neither and stay off the wire.
    """
    events = _data_events(message)
    chunks = getattr(message, "tool_call_chunks", None)
    if not isinstance(chunks, list):
        return events
    events.extend(
        event for chunk in chunks if (event := _tool_use_event(chunk)) is not None
    )
    return events


def _assistant_events(message: object) -> list[dict]:
    """
    Extract renderable events from a complete assistant message.

    Args:
        message (object): The assistant message of a `messages` payload.

    Returns:
        list[dict]: A `data` event for the message's text, a
            `current_tool_use` event per tool call, and a `file` event per
            file-carrying content block, for models that generate files.
    """
    events = _data_events(message)
    calls = getattr(message, "tool_calls", None)
    if isinstance(calls, list):
        events.extend(
            event for call in calls if (event := _tool_use_event(call)) is not None
        )
    events.extend(_file_events(message))
    return events


def _tool_events(message: object) -> list[dict]:
    """
    Extract renderable events from a tool message.

    Args:
        message (object): The tool message of a `messages` payload.

    Returns:
        list[dict]: A `tool_result` event slimmed to the tool call id and
            status — so text tool output stays off the wire — followed by
            a `file` event per file-carrying content block the tool
            returned.
    """
    tool_call_id = getattr(message, "tool_call_id", None)
    status = getattr(message, "status", None)
    events = [
        {
            "tool_result": {
                "toolUseId": tool_call_id if isinstance(tool_call_id, str) else None,
                "status": "error" if status == "error" else "success",
            }
        }
    ]
    events.extend(_file_events(message))
    return events


def _data_events(message: object) -> list[dict]:
    """
    Extract the `data` event of a message's text.

    Args:
        message (object): An assistant message or token delta.

    Returns:
        list[dict]: The `data` event, or nothing for an empty text.
    """
    text = getattr(message, "text", None)
    if not isinstance(text, str) or not text:
        return []
    return [{"data": str(text)}]


def _tool_use_event(call: object) -> dict | None:
    """
    Build a `current_tool_use` event from a tool call or tool-call chunk.

    Args:
        call (object): One entry of a message's `tool_calls` or
            `tool_call_chunks`.

    Returns:
        dict | None: The `current_tool_use` event (name plus toolUseId),
            or None for entries without both — the argument fragments of a
            streamed call.
    """
    if not isinstance(call, dict):
        return None
    name = call.get("name")
    call_id = call.get("id")
    if not isinstance(name, str) or not name:
        return None
    if not isinstance(call_id, str) or not call_id:
        return None
    return {"current_tool_use": {"name": name, "toolUseId": call_id}}


def _file_events(message: object) -> list[dict]:
    """
    Build `file` events from a message's file-carrying content blocks.

    Args:
        message (object): A message whose `content_blocks` may carry
            image, file, video, or audio blocks with base64 data.

    Returns:
        list[dict]: One `file` event per file-carrying block, named after
            the block's kind and media type.
    """
    blocks = getattr(message, "content_blocks", None)
    if not isinstance(blocks, list):
        return []
    events: list[dict] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if not isinstance(kind, str) or kind not in ("image", "file", "video", "audio"):
            continue
        data = block.get("base64")
        if not isinstance(data, str) or not data:
            continue
        events.append(
            {"file": {"name": _file_name(kind, block.get("mime_type")), "bytes": data}}
        )
    return events


# Media subtypes double as filename extensions, except these.
_EXTENSION_BY_SUBTYPE = {
    "3gpp": "3gp",
    "markdown": "md",
    "plain": "txt",
    "quicktime": "mov",
    "x-matroska": "mkv",
}


def _file_name(kind: str, mime_type: object) -> str:
    """
    Synthesize an upload filename for a file-carrying content block.

    Args:
        kind (str): The block's type (image, file, video, or audio).
        mime_type (object): The block's media type, whose subtype provides
            the extension.

    Returns:
        str: The block's kind plus the media subtype as extension.
    """
    subtype = mime_type.split("/", 1)[1] if isinstance(mime_type, str) else ""
    extension = _EXTENSION_BY_SUBTYPE.get(subtype)
    if extension is None:
        extension = subtype if subtype.isalnum() and subtype.islower() else "bin"
    return f"{kind}.{extension}"


def _interrupt_events(payload: object) -> list[dict]:
    """
    Serialize the pending interrupts of an `updates` stream item.

    A run that pauses for human input emits an update whose
    `__interrupt__` value carries one Interrupt per pending question. Each
    becomes an `interrupt` event — the Interrupt's id, with its value
    passed through unmodified as the reason (it is any JSON-serializable
    value by LangGraph's contract, and interpreting it is the renderer's
    job). LangGraph interrupts carry no name, and the wire's `name` slot
    goes to Welt's log only, so it stays empty. The usual node updates
    yield nothing.

    Args:
        payload (object): The `updates` payload of a stream item.

    Returns:
        list[dict]: One `interrupt` event per pending interrupt.
    """
    if not isinstance(payload, dict):
        return []
    interrupts = payload.get("__interrupt__")
    if not isinstance(interrupts, (tuple, list)):
        return []
    events = []
    for interrupt in interrupts:
        interrupt_id = getattr(interrupt, "id", None)
        if not isinstance(interrupt_id, str) or not interrupt_id:
            continue
        events.append(
            {
                "interrupt": {
                    "id": interrupt_id,
                    "name": "",
                    "reason": getattr(interrupt, "value", None),
                }
            }
        )
    return events


def _custom_file_event(payload: object) -> dict | None:
    """
    Pass through a `file` event a tool put on the custom stream.

    A tool attaches a file by passing a `file_event`-shaped value to
    `get_stream_writer()`; other custom values stay off the wire.

    Args:
        payload (object): The `custom` payload of a stream item.

    Returns:
        dict | None: The `file` event, or None for other custom values.
    """
    if not isinstance(payload, dict):
        return None
    file = payload.get("file")
    if not isinstance(file, dict):
        return None
    name = file.get("name")
    data = file.get("bytes")
    if not isinstance(name, str) or not name or not isinstance(data, str):
        return None
    return {"file": {"name": name, "bytes": data}}
