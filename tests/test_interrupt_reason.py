from typing import Literal

import pytest

from welt_io_langgraph import (
    _checked_input,
    _checked_message,
    _checked_option,
    _checked_options,
    interrupt_reason,
)


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
def test_option_style_is_carried(style: Literal["primary", "danger"]) -> None:
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
        (("Sure?", [{"value": ""}]), {}),
        (("Sure?", [{"value": "y", "label": ""}]), {}),
        (("Sure?",), {}),
        (("",), {"input": {}}),
        (("Sure?",), {"input": {"label": ""}}),
        (("Sure?", []), {"input": {}}),
    ],
)
def test_rejects_a_reason_welt_would_not_render(args: tuple, kwargs: dict) -> None:
    with pytest.raises(ValueError):
        interrupt_reason(*args, **kwargs)


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        (("Sure?", [{"value": "y"} for _ in range(26)]), {}),
        (("Sure?", [{"value": "v" * 1801}]), {}),
        (("v" * 12_001, [{"value": "y"}]), {}),
    ],
)
def test_welts_own_rendering_caps_are_left_to_welt(args: tuple, kwargs: dict) -> None:
    assert interrupt_reason(*args, **kwargs)


# --- what the type checker cannot reach --------------------------------------
#
# A caller who builds the options in a variable, or runs no type checker at
# all, gets an error from these instead of Welt's default buttons in the
# thread. They take `object` for the same reason: a deliberately wrong value
# written against the typed signature would not survive `ty`.


@pytest.mark.parametrize("message", [42, None, b"Sure?"])
def test_a_message_that_is_not_a_string_is_refused(message: object) -> None:
    with pytest.raises(TypeError):
        _checked_message(message)


@pytest.mark.parametrize("options", [{"value": "y"}, "y", 42, None])
def test_options_that_are_not_a_sequence_are_refused(options: object) -> None:
    with pytest.raises(TypeError):
        _checked_options(options)


@pytest.mark.parametrize("option", ["y", None, [("value", "y")]])
def test_an_option_that_is_not_a_dict_is_refused(option: object) -> None:
    with pytest.raises(TypeError):
        _checked_option(option)


@pytest.mark.parametrize(
    "option",
    [{"value": 42}, {"value": "y", "label": 42}, {"value": "y", "label": None}],
)
def test_an_option_value_of_the_wrong_type_is_refused(option: object) -> None:
    with pytest.raises(TypeError):
        _checked_option(option)


@pytest.mark.parametrize(
    "option",
    [
        {"label": "Yes"},
        {"value": "y", "style": "warning"},
        {"value": "y", "style": None},
        {"value": "y", "labl": "Yes"},
    ],
)
def test_an_option_welt_would_not_render_is_refused(option: object) -> None:
    with pytest.raises(ValueError):
        _checked_option(option)


def test_the_error_names_the_key_that_was_misspelled() -> None:
    with pytest.raises(ValueError) as caught:
        _checked_option({"value": "y", "labl": "Yes"})

    assert "labl" in str(caught.value)


@pytest.mark.parametrize(
    "input_spec",
    [
        "City",
        None,
        {"label": 42},
        {"label": None},
        {"multiline": "yes"},
        {"multiline": None},
    ],
)
def test_an_input_of_the_wrong_type_is_refused(input_spec: object) -> None:
    with pytest.raises(TypeError):
        _checked_input(input_spec)


def test_an_input_key_welt_does_not_know_is_refused() -> None:
    with pytest.raises(ValueError):
        _checked_input({"placeholder": "Type here"})
