import pytest

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


def test_empty_message_is_rejected() -> None:
    with pytest.raises(ValueError, match="message"):
        interrupt_reason("", [{"value": "y"}])


def test_empty_options_are_rejected() -> None:
    with pytest.raises(ValueError, match="options"):
        interrupt_reason("Sure?", [])


def test_unknown_option_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown option keys"):
        interrupt_reason("Sure?", [{"value": "y", "text": "Yes"}])


def test_missing_option_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="value"):
        interrupt_reason("Sure?", [{"label": "Yes"}])


def test_empty_option_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="value"):
        interrupt_reason("Sure?", [{"value": "", "label": "Yes"}])


def test_empty_option_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="label"):
        interrupt_reason("Sure?", [{"value": "y", "label": ""}])


def test_unknown_style_is_rejected() -> None:
    with pytest.raises(ValueError, match="style"):
        interrupt_reason("Sure?", [{"value": "y", "style": "default"}])


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


def test_empty_message_with_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="message"):
        interrupt_reason("", input={})


def test_empty_input_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="label"):
        interrupt_reason("Sure?", input={"label": ""})


def test_non_string_input_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="label"):
        interrupt_reason("Sure?", input={"label": 42})


def test_non_bool_input_multiline_is_rejected() -> None:
    with pytest.raises(ValueError, match="multiline"):
        interrupt_reason("Sure?", input={"multiline": "yes"})


def test_unknown_input_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown input keys"):
        interrupt_reason("Sure?", input={"placeholder": "Type here"})


def test_no_widgets_is_rejected() -> None:
    with pytest.raises(ValueError, match="options or input"):
        interrupt_reason("Sure?")


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


def test_malformed_options_with_input_are_rejected() -> None:
    with pytest.raises(ValueError, match="value"):
        interrupt_reason("Sure?", [{"value": "", "label": "X"}], input={})
    with pytest.raises(ValueError, match="options"):
        interrupt_reason("Sure?", [], input={})
