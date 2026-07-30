import asyncio
import base64
import logging
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.types import Interrupt

from welt_io_langgraph import renderable_events

PNG_BASE64 = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")


def rendered(items: list, files_from: set[str] | None = None) -> list[dict]:
    async def source() -> AsyncIterator:
        for item in items:
            yield item

    async def gather() -> list[dict]:
        return [
            event async for event in renderable_events(source(), files_from=files_from)
        ]

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


def _charting_tool_message() -> ToolMessage:
    return ToolMessage(
        content=[
            {"type": "text", "text": "chart rendered"},
            {"type": "image", "base64": PNG_BASE64, "mime_type": "image/png"},
        ],
        tool_call_id="call-1",
        name="render_chart",
    )


def test_files_of_a_tool_named_in_files_from_follow_the_result() -> None:
    items = [("messages", (_charting_tool_message(), {}))]

    assert rendered(items, {"render_chart"}) == [
        {"tool_result": {"toolUseId": "call-1", "status": "success"}},
        {"file": {"name": "image.png", "bytes": PNG_BASE64}},
    ]


def test_files_of_a_tool_left_out_of_files_from_stay_off_the_wire() -> None:
    items = [("messages", (_charting_tool_message(), {}))]
    only_the_result = [{"tool_result": {"toolUseId": "call-1", "status": "success"}}]

    assert rendered(items, {"file_read"}) == only_the_result
    assert rendered(items, set()) == only_the_result
    assert rendered(items) == only_the_result


def test_block_name_names_the_upload() -> None:
    message = ToolMessage(
        content=[
            {
                "type": "file",
                "name": "sample-3f2a1b9c",
                "mime_type": "text/csv",
                "base64": "aGk=",
            }
        ],
        tool_call_id="call-1",
        name="create_sample_file",
    )

    assert rendered([("messages", (message, {}))], {"create_sample_file"}) == [
        {"tool_result": {"toolUseId": "call-1", "status": "success"}},
        {"file": {"name": "sample-3f2a1b9c.csv", "bytes": "aGk="}},
    ]


def test_a_nameless_block_falls_back_to_its_kind() -> None:
    message = AIMessage(
        content=[
            {"type": "image", "name": "", "base64": "aGk=", "mime_type": "image/png"},
            {"type": "file", "base64": "aGk=", "mime_type": "application/pdf"},
        ]
    )

    assert rendered([("messages", (message, {}))]) == [
        {"file": {"name": "image.png", "bytes": "aGk="}},
        {"file": {"name": "file.pdf", "bytes": "aGk="}},
    ]


def test_files_of_an_unnamed_tool_message_stay_off_the_wire() -> None:
    message = ToolMessage(
        content=[{"type": "image", "base64": PNG_BASE64, "mime_type": "image/png"}],
        tool_call_id="call-1",
    )

    assert rendered([("messages", (message, {}))], {"render_chart"}) == [
        {"tool_result": {"toolUseId": "call-1", "status": "success"}}
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


def test_a_file_with_no_bytes_stays_off_the_wire() -> None:
    message = ToolMessage(
        content=[
            {"type": "file", "name": "sample", "mime_type": "text/csv", "base64": ""}
        ],
        tool_call_id="call-1",
        name="create_sample_file",
    )

    assert rendered([("messages", (message, {}))], {"create_sample_file"}) == [
        {"tool_result": {"toolUseId": "call-1", "status": "success"}}
    ]


def test_an_empty_file_is_logged_against_the_tool_that_returned_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    message = ToolMessage(
        content=[{"type": "file", "mime_type": "text/csv", "base64": ""}],
        tool_call_id="call-1",
        name="create_sample_file",
    )

    with caplog.at_level(logging.WARNING):
        rendered([("messages", (message, {}))], {"create_sample_file"})

    assert "create_sample_file" in caplog.text
    assert "file.csv" in caplog.text


def test_an_empty_file_from_the_model_is_logged_against_the_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    message = AIMessage(
        content=[{"type": "image", "base64": "", "mime_type": "image/png"}]
    )

    with caplog.at_level(logging.WARNING):
        assert rendered([("messages", (message, {}))]) == []

    assert "the model" in caplog.text


def test_human_message_yields_nothing() -> None:
    assert rendered([("messages", (HumanMessage("hi"), {}))]) == []


def test_a_value_that_only_looks_like_a_message_yields_nothing() -> None:
    # The stream is read through LangChain's own types rather than by
    # guessing at attributes, so a foreign object carrying the same names
    # is not mistaken for a message.
    lookalike = SimpleNamespace(type="AIMessageChunk", text="hel", tool_call_chunks=[])

    assert rendered([("messages", (lookalike, {}))]) == []


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


def hitl_request(*actions: dict, configs: list | None = None) -> dict:
    """Build a HumanInTheLoopMiddleware request out of reviewed actions.

    Each action carries its allowed decisions under `allowed`, which the
    request keeps in its review configs rather than its actions.
    """
    if configs is None:
        configs = [
            {
                "action_name": action["name"],
                "allowed_decisions": action.get("allowed", ["approve", "reject"]),
            }
            for action in actions
        ]
    return {
        "action_requests": [
            {key: value for key, value in action.items() if key != "allowed"}
            for action in actions
        ],
        "review_configs": configs,
    }


def test_hitl_request_becomes_a_question_for_its_action() -> None:
    request = hitl_request(
        {
            "name": "send_email",
            "args": {"to": "ops@example.com"},
            "description": "Tool execution requires approval\n\nTool: send_email",
            "allowed": ["approve", "reject", "respond"],
        }
    )
    items = [("updates", {"__interrupt__": (Interrupt(value=request, id="i-1"),)})]

    assert rendered(items) == [
        {
            "interrupt": {
                "id": "i-1#0",
                "name": "send_email",
                "reason": {
                    "message": "Tool execution requires approval\n\nTool: send_email",
                    "options": [
                        {
                            "value": "welt-io:hitl:approve",
                            "label": "Approve",
                            "style": "primary",
                        },
                        {
                            "value": "welt-io:hitl:reject",
                            "label": "Reject",
                            "style": "danger",
                        },
                    ],
                    "input": {},
                },
            }
        }
    ]


def test_hitl_request_splits_into_one_question_per_action() -> None:
    request = hitl_request(
        {"name": "send_email", "args": {}, "description": "Send it?"},
        {
            "name": "ask_expert",
            "args": {},
            "description": "Answer for it?",
            "allowed": ["respond"],
        },
    )
    items = [("updates", {"__interrupt__": (Interrupt(value=request, id="i-1"),)})]

    assert rendered(items) == [
        {
            "interrupt": {
                "id": "i-1#0",
                "name": "send_email",
                "reason": {
                    "message": "Send it?",
                    "options": [
                        {
                            "value": "welt-io:hitl:approve",
                            "label": "Approve",
                            "style": "primary",
                        },
                        {
                            "value": "welt-io:hitl:reject",
                            "label": "Reject",
                            "style": "danger",
                        },
                    ],
                },
            }
        },
        {
            "interrupt": {
                "id": "i-1#1",
                "name": "ask_expert",
                "reason": {"message": "Answer for it?", "input": {}},
            }
        },
    ]


def test_hitl_action_without_a_description_is_asked_about_by_name() -> None:
    request = hitl_request({"name": "send_email", "args": {}})
    items = [("updates", {"__interrupt__": (Interrupt(value=request, id="i-1"),)})]

    assert rendered(items)[0]["interrupt"]["reason"]["message"] == "send_email"


def test_hitl_request_allowing_only_edit_is_passed_through() -> None:
    request = hitl_request({"name": "send_email", "args": {}, "allowed": ["edit"]})
    items = [("updates", {"__interrupt__": (Interrupt(value=request, id="i-1"),)})]

    assert rendered(items) == [
        {"interrupt": {"id": "i-1", "name": "", "reason": request}}
    ]


def test_hitl_request_without_review_configs_is_passed_through() -> None:
    request = {"action_requests": [{"name": "send_email", "args": {}}]}
    items = [("updates", {"__interrupt__": (Interrupt(value=request, id="i-1"),)})]

    assert rendered(items) == [
        {"interrupt": {"id": "i-1", "name": "", "reason": request}}
    ]


def test_hitl_request_with_no_actions_is_passed_through() -> None:
    request = {"action_requests": [], "review_configs": []}
    items = [("updates", {"__interrupt__": (Interrupt(value=request, id="i-1"),)})]

    assert rendered(items) == [
        {"interrupt": {"id": "i-1", "name": "", "reason": request}}
    ]


def test_hitl_action_without_its_review_config_is_passed_through() -> None:
    request = hitl_request(
        {"name": "send_email", "args": {}},
        configs=[{"action_name": "another_tool", "allowed_decisions": ["approve"]}],
    )
    items = [("updates", {"__interrupt__": (Interrupt(value=request, id="i-1"),)})]

    assert rendered(items) == [
        {"interrupt": {"id": "i-1", "name": "", "reason": request}}
    ]


def test_hitl_request_with_a_malformed_review_config_is_passed_through() -> None:
    request = hitl_request({"name": "send_email", "args": {}}, configs=["not a dict"])
    items = [("updates", {"__interrupt__": (Interrupt(value=request, id="i-1"),)})]

    assert rendered(items) == [
        {"interrupt": {"id": "i-1", "name": "", "reason": request}}
    ]


def test_hitl_request_with_a_malformed_action_is_passed_through() -> None:
    request = {
        "action_requests": ["not a dict"],
        "review_configs": [{"action_name": "x", "allowed_decisions": ["approve"]}],
    }
    items = [("updates", {"__interrupt__": (Interrupt(value=request, id="i-1"),)})]

    assert rendered(items) == [
        {"interrupt": {"id": "i-1", "name": "", "reason": request}}
    ]


def test_hitl_action_without_a_name_is_passed_through() -> None:
    request = {
        "action_requests": [{"name": "", "args": {}}],
        "review_configs": [{"action_name": "", "allowed_decisions": ["approve"]}],
    }
    items = [("updates", {"__interrupt__": (Interrupt(value=request, id="i-1"),)})]

    assert rendered(items) == [
        {"interrupt": {"id": "i-1", "name": "", "reason": request}}
    ]


def test_hitl_review_config_without_decisions_is_passed_through() -> None:
    request = hitl_request(
        {"name": "send_email", "args": {}},
        configs=[{"action_name": "send_email", "allowed_decisions": "approve"}],
    )
    items = [("updates", {"__interrupt__": (Interrupt(value=request, id="i-1"),)})]

    assert rendered(items) == [
        {"interrupt": {"id": "i-1", "name": "", "reason": request}}
    ]


def test_hitl_action_allowing_edit_is_asked_with_the_rest_of_its_widgets() -> None:
    request = hitl_request(
        {"name": "send_email", "args": {}, "allowed": ["approve", "edit"]}
    )
    items = [("updates", {"__interrupt__": (Interrupt(value=request, id="i-1"),)})]

    reason = rendered(items)[0]["interrupt"]["reason"]

    assert [option["label"] for option in reason["options"]] == ["Approve"]
    assert "input" not in reason
