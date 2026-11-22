import json
from copy import deepcopy
from itertools import permutations, product

from welt_io_langgraph import decode_interrupt_responses


def _pressed(value: object) -> dict:
    return {"value": value, "source": "option"}


def _typed(text: str) -> dict:
    return {"value": text, "source": "input"}


def test_plain_answers_preserve_values_ids_and_insertion_order() -> None:
    values = ("approve", "later", True, False, None, [1], {"nested": [False]}, "", 0)
    for size in range(4):
        for chosen in product(values, repeat=size):
            answers = {
                f"i-{size - i}": _typed(value)
                if isinstance(value, str) and i % 2
                else _pressed(value)
                for i, value in enumerate(chosen)
            }
            before = deepcopy(answers)
            expected = {f"i-{size - i}": value for i, value in enumerate(chosen)}
            actual = decode_interrupt_responses(answers)
            # JSON distinguishes booleans from numbers, unlike Python equality.
            assert json.dumps(actual, sort_keys=True) == json.dumps(
                expected, sort_keys=True
            )
            assert list(actual) == list(expected)
            if actual:
                actual[next(iter(actual))] = "changed"
            assert answers == before


def test_hitl_answers_rejoin_by_action_index_independently_of_arrival_order() -> None:
    cases = (
        (_pressed(True), {"type": "approve"}),
        (_pressed(False), {"type": "reject"}),
        (_typed("approve"), {"type": "respond", "message": "approve"}),
        (
            _typed("ask ops first\n日本語"),
            {"type": "respond", "message": "ask ops first\n日本語"},
        ),
    )
    for selected in product(cases, repeat=3):
        for order in permutations(range(3)):
            answers = {f"welt-io:hitl:{i}:i-1": selected[i][0] for i in order}
            before = deepcopy(answers)
            assert decode_interrupt_responses(answers) == {
                "i-1": {"decisions": [decision for _, decision in selected]}
            }
            assert answers == before
        per_call = {
            f"welt-io:hitl-call:i-{i}": answer for i, (answer, _) in enumerate(selected)
        }
        assert decode_interrupt_responses(per_call) == {
            f"i-{i}": decision for i, (_, decision) in enumerate(selected)
        }


def test_a_pressed_value_no_question_offered_rejects() -> None:
    answers = {"welt-io:hitl:0:i-1": _pressed("edit")}

    assert decode_interrupt_responses(answers) == {
        "i-1": {"decisions": [{"type": "reject"}]}
    }


def test_hitl_answers_leaving_a_gap_travel_on_for_the_middleware_to_refuse() -> None:
    answers = {
        "welt-io:hitl:0:i-1": _pressed(True),
        "welt-io:hitl:2:i-1": _pressed(True),
    }

    assert decode_interrupt_responses(answers) == {
        "i-1": {"decisions": [{"type": "approve"}, {"type": "approve"}]}
    }


def test_hitl_answers_keep_the_place_of_their_first_answer() -> None:
    answers = {
        "i-1": _pressed("y"),
        "welt-io:hitl:0:i-2": _pressed("approve"),
        "i-3": _pressed("n"),
        "welt-io:hitl:1:i-2": _pressed("reject"),
    }

    assert list(decode_interrupt_responses(answers)) == ["i-1", "i-2", "i-3"]


def test_hitl_requests_are_rejoined_one_by_one() -> None:
    answers = {
        "welt-io:hitl:0:i-1": _pressed(True),
        "welt-io:hitl:0:i-2": _pressed("reject"),
    }

    assert decode_interrupt_responses(answers) == {
        "i-1": {"decisions": [{"type": "approve"}]},
        "i-2": {"decisions": [{"type": "reject"}]},
    }


def test_an_id_outside_the_adapters_namespace_answers_a_plain_interrupt() -> None:
    answers = {
        "i-1#0": _pressed("y"),
        "welt-io:hitl:x:i-2": _pressed("y"),
        "welt-io:hitl:0:": _pressed("y"),
        "welt-io:hitl:0": _pressed("y"),
    }

    assert decode_interrupt_responses(answers) == {
        "i-1#0": "y",
        "welt-io:hitl:x:i-2": "y",
        "welt-io:hitl:0:": "y",
        "welt-io:hitl:0": "y",
    }
