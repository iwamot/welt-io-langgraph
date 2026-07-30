import base64

from welt_io_langgraph import decode_messages


def encoded(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def user_message(*content: object) -> dict:
    return {"role": "user", "content": list(content)}


def image_block(raw: bytes = b"img") -> dict:
    return {"image": {"format": "png", "source": {"bytes": encoded(raw)}}}


def document_block(name: str = "report") -> dict:
    return {
        "document": {
            "format": "pdf",
            "name": name,
            "source": {"bytes": encoded(b"doc")},
        }
    }


def test_decodes_image_document_and_video_blocks() -> None:
    messages = [
        user_message(
            image_block(),
            document_block(),
            {"video": {"format": "mp4", "source": {"bytes": encoded(b"vid")}}},
        )
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
                    "filename": "report",
                },
                {
                    "type": "video",
                    "base64": encoded(b"vid"),
                    "mime_type": "video/mp4",
                },
            ],
        }
    ]


def test_the_three_gp_token_becomes_the_media_type_langchain_takes() -> None:
    block = {"video": {"format": "three_gp", "source": {"bytes": encoded(b"vid")}}}

    decoded = decode_messages([user_message(block)])

    assert decoded[0]["content"][0]["mime_type"] == "video/3gpp"


def test_text_blocks_become_standard_text_blocks() -> None:
    messages = [
        user_message({"text": "<@U1>: hello"}),
        {"role": "assistant", "content": [{"text": "hi"}]},
    ]

    assert decode_messages(messages) == [
        {"role": "user", "content": [{"type": "text", "text": "<@U1>: hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    ]


def test_decodes_across_multiple_messages() -> None:
    messages = [
        user_message({"text": "<@U1>: one"}),
        {"role": "assistant", "content": [{"text": "two"}]},
        user_message({"text": "<@U1>: three"}),
    ]

    assert [message["role"] for message in decode_messages(messages)] == [
        "user",
        "assistant",
        "user",
    ]


def test_leaves_the_input_untouched() -> None:
    messages = [user_message(image_block())]

    decode_messages(messages)

    assert messages == [user_message(image_block())]


def test_an_empty_conversation_decodes_to_an_empty_one() -> None:
    assert decode_messages([]) == []
