import base64

import pytest

from welt_io_langgraph import file_event


def test_builds_a_file_event_with_base64_bytes() -> None:
    data = b"col_a,col_b\n1,2\n"

    assert file_event("report.csv", data) == {
        "file": {
            "name": "report.csv",
            "bytes": base64.b64encode(data).decode("ascii"),
        }
    }


def test_empty_bytes_encode_to_an_empty_string() -> None:
    assert file_event("empty.bin", b"") == {"file": {"name": "empty.bin", "bytes": ""}}


def test_empty_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="name"):
        file_event("", b"data")
