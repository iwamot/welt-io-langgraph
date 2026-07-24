import base64

from welt_io_langgraph import decode_messages


def encoded(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def test_decodes_image_document_and_video_blocks() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"image": {"format": "png", "source": {"bytes": encoded(b"img")}}},
                {"document": {"format": "pdf", "source": {"bytes": encoded(b"doc")}}},
                {"video": {"format": "mp4", "source": {"bytes": encoded(b"vid")}}},
            ],
        }
    ]

    assert decode_messages(messages) == [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "base64": encoded(b"img"),
                    "mime_type": "image/png",
                },
                {
                    "type": "file",
                    "base64": encoded(b"doc"),
                    "mime_type": "application/pdf",
                },
                {
                    "type": "video",
                    "base64": encoded(b"vid"),
                    "mime_type": "video/mp4",
                },
            ],
        }
    ]


def test_document_name_becomes_the_filename() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "document": {
                        "format": "pdf",
                        "name": "report",
                        "source": {"bytes": encoded(b"doc")},
                    }
                }
            ],
        }
    ]

    assert decode_messages(messages)[0]["content"] == [
        {
            "type": "file",
            "base64": encoded(b"doc"),
            "mime_type": "application/pdf",
            "filename": "report",
        }
    ]


def test_empty_document_name_stays_omitted() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"document": {"format": "csv", "name": "", "source": {"bytes": "aGk="}}}
            ],
        }
    ]

    assert "filename" not in decode_messages(messages)[0]["content"][0]


def test_text_blocks_become_text_blocks() -> None:
    messages = [{"role": "user", "content": [{"text": "hello"}]}]

    assert decode_messages(messages) == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    ]


def test_block_order_is_preserved() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"document": {"format": "pdf", "source": {"bytes": "aGk="}}},
                {"text": "<@U0123456>: what does the report say?"},
                {"image": {"format": "png", "source": {"bytes": "aGk="}}},
            ],
        }
    ]

    kinds = [block["type"] for block in decode_messages(messages)[0]["content"]]

    assert kinds == ["file", "text", "image"]


def test_assistant_messages_keep_their_text_only() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [
                {"text": "here you go"},
                {"image": {"format": "png", "source": {"bytes": "aGk="}}},
            ],
        }
    ]

    assert decode_messages(messages) == [
        {"role": "assistant", "content": [{"type": "text", "text": "here you go"}]}
    ]


def test_conversation_roles_and_order_are_preserved() -> None:
    messages = [
        {"role": "user", "content": [{"text": "hi"}]},
        {"role": "assistant", "content": [{"text": "hello"}]},
        {"role": "user", "content": [{"text": "again"}]},
    ]

    assert [message["role"] for message in decode_messages(messages)] == [
        "user",
        "assistant",
        "user",
    ]


def test_unknown_format_block_is_skipped() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"text": "look"},
                {"image": {"format": "tiff", "source": {"bytes": "aGk="}}},
            ],
        }
    ]

    assert decode_messages(messages)[0]["content"] == [{"type": "text", "text": "look"}]


def test_unknown_document_format_block_is_skipped() -> None:
    messages = [
        {
            "role": "user",
            "content": [{"document": {"format": "epub", "source": {"bytes": "aGk="}}}],
        }
    ]

    assert decode_messages(messages) == []


def test_unknown_video_format_block_is_skipped() -> None:
    messages = [
        {
            "role": "user",
            "content": [{"video": {"format": "avi", "source": {"bytes": "aGk="}}}],
        }
    ]

    assert decode_messages(messages) == []


def test_missing_format_block_is_skipped() -> None:
    messages = [{"role": "user", "content": [{"image": {"source": {"bytes": "aGk="}}}]}]

    assert decode_messages(messages) == []


def test_message_left_with_no_blocks_is_dropped() -> None:
    messages = [
        {"role": "user", "content": []},
        {"role": "user", "content": [{"text": "kept"}]},
    ]

    assert decode_messages(messages) == [
        {"role": "user", "content": [{"type": "text", "text": "kept"}]}
    ]


def test_no_op_on_empty_messages() -> None:
    assert decode_messages([]) == []


def test_skips_non_dict_message() -> None:
    assert decode_messages(["not a dict"]) == []


def test_skips_unknown_role() -> None:
    assert decode_messages([{"role": "system", "content": [{"text": "x"}]}]) == []


def test_skips_non_list_content() -> None:
    assert decode_messages([{"role": "user", "content": "not a list"}]) == []


def test_skips_non_dict_block() -> None:
    assert decode_messages([{"role": "user", "content": ["not a dict"]}]) == []


def test_skips_non_dict_media() -> None:
    messages = [{"role": "user", "content": [{"image": "not a dict"}]}]

    assert decode_messages(messages) == []


def test_skips_non_dict_source() -> None:
    messages = [{"role": "user", "content": [{"image": {"source": "not a dict"}}]}]

    assert decode_messages(messages) == []


def test_skips_bytes_that_are_not_str() -> None:
    messages = [{"role": "user", "content": [{"image": {"source": {"bytes": b"raw"}}}]}]

    assert decode_messages(messages) == []


def test_skips_empty_bytes() -> None:
    messages = [{"role": "user", "content": [{"image": {"source": {"bytes": ""}}}]}]

    assert decode_messages(messages) == []
