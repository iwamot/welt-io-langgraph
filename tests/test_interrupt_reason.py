import json
from copy import deepcopy
from itertools import product

import pytest

from welt_io_langgraph import (
    DecisionSpec,
    InputSpec,
    OptionSpec,
    _checked_decision,
    _checked_input,
    _checked_message,
    _checked_option,
    _checked_options,
    interrupt_reason,
)


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        (("", [{"value": "y"}]), {}),
        (("Sure?", []), {}),
        (("Sure?", [{"value": "y", "label": ""}]), {}),
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


def test_widget_combinations_preserve_values_and_omissions() -> None:
    decisions: tuple[DecisionSpec | None, ...] = (
        None,
        {},
        {"label": "Deploy"},
        {"label": "Cancel", "style": "primary"},
        {"style": "danger"},
    )
    inputs: tuple[InputSpec | None, ...] = (
        None,
        {},
        {"label": "City"},
        {"label": "Draft", "multiline": True},
        {"multiline": False},
    )
    plain: list[OptionSpec] = [
        {"value": value}
        for value in (True, False, None, 42, "", ["a"], {"decision": "hold"})
    ]
    options: tuple[list[OptionSpec] | None, ...] = (
        None,
        plain,
        [
            {"value": "approve", "label": "Deploy"},
            {"value": "reject", "label": "Cancel"},
        ],
        [
            {"value": "approve", "label": "Deploy", "style": "primary"},
            {"value": "reject", "label": "Cancel", "style": "danger"},
            {"value": "n"},
        ],
    )
    for message, approve, reject, input_spec, choices in product(
        ("Sure?", "日本語\n質問"), decisions, decisions, inputs, options
    ):
        before = deepcopy((approve, reject, input_spec, choices))
        expected: dict = {"message": message}
        for key, value in (
            ("approve", approve),
            ("reject", reject),
            ("input", input_spec),
            ("options", choices),
        ):
            if value is not None:
                expected[key] = value
        actual = interrupt_reason(
            message, choices, approve=approve, reject=reject, input=input_spec
        )
        assert json.dumps(actual, sort_keys=True) == json.dumps(
            expected, sort_keys=True
        )
        assert (approve, reject, input_spec, choices) == before


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
    [{"value": "y", "label": 42}, {"value": "y", "label": None}],
)
def test_an_option_label_of_the_wrong_type_is_refused(option: object) -> None:
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


@pytest.mark.parametrize("spec", ["yes", None, {"label": 42}, {"label": None}])
def test_a_decision_of_the_wrong_type_is_refused(spec: object) -> None:
    with pytest.raises(TypeError):
        _checked_decision(spec, "approve")


@pytest.mark.parametrize(
    "spec", [{"label": ""}, {"style": "default"}, {"style": None}, {"value": True}]
)
def test_a_decision_welt_would_not_render_is_refused(spec: object) -> None:
    with pytest.raises(ValueError):
        _checked_decision(spec, "approve")


def test_the_decision_error_names_the_argument() -> None:
    with pytest.raises(ValueError) as caught:
        _checked_decision({"label": ""}, "reject")

    assert "reject" in str(caught.value)
