import asyncio
import base64
from collections.abc import AsyncIterator
from types import SimpleNamespace

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.types import Interrupt

from welt_io_langgraph import renderable_events

PNG_BASE64 = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")


def rendered(items: list) -> list[dict]:
    async def source() -> AsyncIterator:
        for item in items:
            yield item

    async def gather() -> list[dict]:
        return [event async for event in renderable_events(source())]

    return asyncio.run(gather())


def test_token_delta_becomes_a_data_event() -> None:
    items = [("messages", (AIMessageChunk(content="hel"), {"langgraph_node": "model"}))]

    assert rendered(items) == [{"data": "hel"}]


def test_empty_token_delta_yields_nothing() -> None:
    assert rendered([("messages", (AIMessageChunk(content=""), {}))]) == []


def test_opening_tool_call_chunk_becomes_a_current_tool_use_event() -> None:
    chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": "current_time",
                "args": "",
                "id": "call-1",
                "index": 0,
                "type": "tool_call_chunk",
            }
        ],
    )

    assert rendered([("messages", (chunk, {}))]) == [
        {"current_tool_use": {"name": "current_time", "toolUseId": "call-1"}}
    ]


def test_argument_fragment_chunks_stay_off_the_wire() -> None:
    chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": None,
                "args": '{"x": 1',
                "id": None,
                "index": 0,
                "type": "tool_call_chunk",
            }
        ],
    )

    assert rendered([("messages", (chunk, {}))]) == []


def test_complete_assistant_message_text_becomes_a_data_event() -> None:
    items = [("messages", (AIMessage("hello"), {}))]

    assert rendered(items) == [{"data": "hello"}]


def test_complete_assistant_message_tool_calls_become_current_tool_use() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "current_time", "args": {}, "id": "call-1", "type": "tool_call"}
        ],
    )

    assert rendered([("messages", (message, {}))]) == [
        {"current_tool_use": {"name": "current_time", "toolUseId": "call-1"}}
    ]


def test_tool_call_without_id_stays_off_the_wire() -> None:
    message = AIMessage(
        content="",
        tool_calls=[{"name": "current_time", "args": {}, "id": None}],
    )

    assert rendered([("messages", (message, {}))]) == []


def test_assistant_message_image_block_becomes_a_file_event() -> None:
    message = AIMessage(
        content=[
            {"type": "text", "text": "here you go"},
            {"type": "image", "base64": PNG_BASE64, "mime_type": "image/png"},
        ]
    )

    assert rendered([("messages", (message, {}))]) == [
        {"data": "here you go"},
        {"file": {"name": "image.png", "bytes": PNG_BASE64}},
    ]


def test_tool_message_is_slimmed_to_id_and_status() -> None:
    message = ToolMessage("arbitrarily large output", tool_call_id="call-1")

    assert rendered([("messages", (message, {}))]) == [
        {"tool_result": {"toolUseId": "call-1", "status": "success"}}
    ]


def test_tool_error_becomes_an_error_status() -> None:
    message = ToolMessage("boom", tool_call_id="call-1", status="error")

    assert rendered([("messages", (message, {}))]) == [
        {"tool_result": {"toolUseId": "call-1", "status": "error"}}
    ]


def test_tool_message_file_blocks_become_file_events_after_the_result() -> None:
    message = ToolMessage(
        content=[
            {"type": "text", "text": "chart rendered"},
            {"type": "image", "base64": PNG_BASE64, "mime_type": "image/png"},
        ],
        tool_call_id="call-1",
    )

    assert rendered([("messages", (message, {}))]) == [
        {"tool_result": {"toolUseId": "call-1", "status": "success"}},
        {"file": {"name": "image.png", "bytes": PNG_BASE64}},
    ]


def test_file_block_kinds_and_media_subtypes_name_the_file() -> None:
    message = AIMessage(
        content=[
            {"type": "file", "base64": "aGk=", "mime_type": "application/pdf"},
            {"type": "video", "base64": "aGk=", "mime_type": "video/3gpp"},
            {"type": "audio", "base64": "aGk=", "mime_type": "audio/mpeg"},
        ]
    )

    assert rendered([("messages", (message, {}))]) == [
        {"file": {"name": "file.pdf", "bytes": "aGk="}},
        {"file": {"name": "video.3gp", "bytes": "aGk="}},
        {"file": {"name": "audio.mpeg", "bytes": "aGk="}},
    ]


def test_file_block_with_odd_media_subtype_gets_a_bin_extension() -> None:
    message = AIMessage(
        content=[
            {
                "type": "file",
                "base64": "aGk=",
                "mime_type": "application/vnd.ms-excel",
            }
        ]
    )

    assert rendered([("messages", (message, {}))]) == [
        {"file": {"name": "file.bin", "bytes": "aGk="}}
    ]


def test_file_block_without_base64_yields_nothing() -> None:
    message = AIMessage(content=[{"type": "image", "url": "https://example.com/a.png"}])

    assert rendered([("messages", (message, {}))]) == []


def test_human_message_yields_nothing() -> None:
    assert rendered([("messages", (HumanMessage("hi"), {}))]) == []


def test_chunk_shaped_value_without_tool_call_chunks_keeps_its_text() -> None:
    # The attribute shape of a token delta, minus the parts a foreign
    # object may lack.
    chunk = SimpleNamespace(type="AIMessageChunk", text="hel", tool_call_chunks=None)

    assert rendered([("messages", (chunk, {}))]) == [{"data": "hel"}]


def test_assistant_shaped_value_without_calls_or_blocks_keeps_its_text() -> None:
    message = SimpleNamespace(
        type="ai", text="hi", tool_calls=None, content_blocks=None
    )

    assert rendered([("messages", (message, {}))]) == [{"data": "hi"}]


def test_non_dict_tool_calls_and_content_blocks_are_skipped() -> None:
    message = SimpleNamespace(
        type="ai",
        text="",
        tool_calls=["not a dict"],
        content_blocks=["not a dict"],
    )

    assert rendered([("messages", (message, {}))]) == []


def test_pending_interrupt_becomes_an_interrupt_event() -> None:
    interrupt = Interrupt(value="Deploy?", id="i-1")
    items = [("updates", {"__interrupt__": (interrupt,)})]

    assert rendered(items) == [
        {"interrupt": {"id": "i-1", "name": "", "reason": "Deploy?"}}
    ]


def test_yields_one_interrupt_event_per_interrupt() -> None:
    items = [
        (
            "updates",
            {
                "__interrupt__": (
                    Interrupt(value="A?", id="i-1"),
                    Interrupt(value="B?", id="i-2"),
                )
            },
        )
    ]

    assert rendered(items) == [
        {"interrupt": {"id": "i-1", "name": "", "reason": "A?"}},
        {"interrupt": {"id": "i-2", "name": "", "reason": "B?"}},
    ]


def test_interrupt_reason_is_passed_through_unmodified() -> None:
    reason = {
        "message": "Deploy to prod?",
        "options": [{"value": "approve", "label": "Deploy", "style": "primary"}],
        "extra": {"nested": [1, {"deep": True}]},
    }
    items = [("updates", {"__interrupt__": (Interrupt(value=reason, id="i-1"),)})]

    rendered_reason = rendered(items)[0]["interrupt"]["reason"]

    assert rendered_reason is reason


def test_interrupt_without_an_id_is_skipped() -> None:
    items = [("updates", {"__interrupt__": (SimpleNamespace(value="Deploy?"),)})]

    assert rendered(items) == []


def test_node_updates_yield_nothing() -> None:
    items = [("updates", {"model": {"messages": [AIMessage("hello")]}})]

    assert rendered(items) == []


def test_non_dict_updates_payload_yields_nothing() -> None:
    assert rendered([("updates", "not a dict")]) == []


def test_custom_file_event_is_passed_through() -> None:
    items = [("custom", {"file": {"name": "report.csv", "bytes": "aGk="}})]

    assert rendered(items) == [{"file": {"name": "report.csv", "bytes": "aGk="}}]


def test_custom_file_event_extra_keys_are_slimmed_away() -> None:
    items = [("custom", {"file": {"name": "report.csv", "bytes": "aGk=", "extra": 1}})]

    assert rendered(items) == [{"file": {"name": "report.csv", "bytes": "aGk="}}]


def test_other_custom_values_stay_off_the_wire() -> None:
    items = [
        ("custom", "progress: 50%"),
        ("custom", {"progress": 0.5}),
        ("custom", {"file": "not a dict"}),
        ("custom", {"file": {"name": "", "bytes": "aGk="}}),
        ("custom", {"file": {"name": "report.csv"}}),
    ]

    assert rendered(items) == []


def test_unrenderable_items_are_dropped() -> None:
    items = [
        "not a tuple",
        ("values", {"messages": []}),
        ("messages", "not a tuple"),
        ("messages", ()),
        ("messages", (object(), {})),
        ("too", "many", "parts"),
    ]

    assert rendered(items) == []


def test_stream_order_is_preserved() -> None:
    chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": "current_time",
                "args": "",
                "id": "call-1",
                "index": 0,
                "type": "tool_call_chunk",
            }
        ],
    )
    items = [
        ("messages", (AIMessageChunk(content="a"), {})),
        ("messages", (chunk, {})),
        ("messages", (ToolMessage("12:00", tool_call_id="call-1"), {})),
        ("messages", (AIMessageChunk(content="b"), {})),
        ("updates", {"__interrupt__": (Interrupt(value="Sure?", id="i-1"),)}),
    ]

    assert rendered(items) == [
        {"data": "a"},
        {"current_tool_use": {"name": "current_time", "toolUseId": "call-1"}},
        {"tool_result": {"toolUseId": "call-1", "status": "success"}},
        {"data": "b"},
        {"interrupt": {"id": "i-1", "name": "", "reason": "Sure?"}},
    ]
