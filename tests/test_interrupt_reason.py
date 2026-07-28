import pytest
from jsonschema.exceptions import ValidationError

from welt_io_langgraph import interrupt_reason


def test_builds_message_and_options() -> None:
    reason = interrupt_reason(
        "Deploy to prod?",
        [
            {"value": "approve", "label": "Deploy"},
            {"value": "reject", "label": "Cancel"},
        ],
    )

    assert reason == {
        "message": "Deploy to prod?",
        "options": [
            {"value": "approve", "label": "Deploy"},
            {"value": "reject", "label": "Cancel"},
        ],
    }


@pytest.mark.parametrize("style", ["primary", "danger"])
def test_option_style_is_carried(style: str) -> None:
    reason = interrupt_reason("Sure?", [{"value": "y", "label": "Yes", "style": style}])

    assert reason["options"] == [{"value": "y", "label": "Yes", "style": style}]


def test_option_without_label_leaves_the_default_to_welt() -> None:
    reason = interrupt_reason("Sure?", [{"value": "y"}])

    assert reason["options"] == [{"value": "y"}]


def test_styled_and_unstyled_options_mix() -> None:
    reason = interrupt_reason(
        "Sure?",
        [{"value": "y", "label": "Yes", "style": "primary"}, {"value": "n"}],
    )

    assert reason["options"] == [
        {"value": "y", "label": "Yes", "style": "primary"},
        {"value": "n"},
    ]


def test_input_builds_message_and_input() -> None:
    reason = interrupt_reason("Which city should I check?", input={"label": "City"})

    assert reason == {
        "message": "Which city should I check?",
        "input": {"label": "City"},
    }


def test_empty_input_takes_welt_defaults() -> None:
    reason = interrupt_reason("Notes?", input={})

    assert reason == {"message": "Notes?", "input": {}}


def test_multiline_input() -> None:
    reason = interrupt_reason(
        "Describe the change.", input={"label": "Draft", "multiline": True}
    )

    assert reason["input"] == {"label": "Draft", "multiline": True}


def test_options_and_input_carry_both() -> None:
    reason = interrupt_reason(
        "Which city should I check?",
        [{"value": "tokyo", "label": "Tokyo"}, {"value": "osaka", "label": "Osaka"}],
        input={"label": "City"},
    )

    assert reason == {
        "message": "Which city should I check?",
        "options": [
            {"value": "tokyo", "label": "Tokyo"},
            {"value": "osaka", "label": "Osaka"},
        ],
        "input": {"label": "City"},
    }


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        (("", [{"value": "y"}]), {}),
        (("Sure?", []), {}),
        (("Sure?", [{"value": "y", "styl": "primary"}]), {}),
        (("Sure?", [{"label": "Yes"}]), {}),
        (("Sure?", [{"value": ""}]), {}),
        (("Sure?", [{"value": "y", "label": ""}]), {}),
        (("Sure?", [{"value": "y", "style": "warning"}]), {}),
        (("Sure?", [{"value": "y"} for _ in range(26)]), {}),
        (("Sure?", [{"value": "v" * 1801}]), {}),
        (("Sure?",), {}),
        (("",), {"input": {}}),
        (("Sure?",), {"input": {"label": ""}}),
        (("Sure?",), {"input": {"label": 42}}),
        (("Sure?",), {"input": {"multiline": "yes"}}),
        (("Sure?",), {"input": {"placeholder": "Type here"}}),
        (("Sure?", [{"value": ""}]), {"input": {}}),
        (("Sure?", []), {"input": {}}),
    ],
)
def test_rejects_a_reason_welt_would_not_render(args: tuple, kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        interrupt_reason(*args, **kwargs)


def test_the_error_names_the_offending_field() -> None:
    with pytest.raises(ValidationError) as caught:
        interrupt_reason("Sure?", [{"value": "y", "style": "warning"}])

    assert caught.value.json_path == "$.options[0].style"
