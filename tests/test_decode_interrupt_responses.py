from welt_io_langgraph import decode_interrupt_responses


def _pressed(value: object) -> dict:
    return {"value": value, "source": "option"}


def _typed(text: str) -> dict:
    return {"value": text, "source": "input"}


def test_answers_become_the_resume_mapping() -> None:
    answers = {"i-1": _pressed("approve"), "i-2": _typed("later")}

    assert decode_interrupt_responses(answers) == {"i-1": "approve", "i-2": "later"}


def test_an_answer_travels_on_as_the_value_it_was_given() -> None:
    answers = {"i-1": _pressed(True), "i-2": _pressed(None), "i-3": _pressed([1])}

    assert decode_interrupt_responses(answers) == {"i-1": True, "i-2": None, "i-3": [1]}


def test_answer_order_is_preserved() -> None:
    answers = {"i-2": _pressed(False), "i-1": _pressed(True)}

    assert list(decode_interrupt_responses(answers)) == ["i-2", "i-1"]


def test_the_input_is_left_untouched() -> None:
    answers = {"i-1": _pressed("y")}

    decoded = decode_interrupt_responses(answers)
    decoded["i-1"] = "changed"

    assert answers == {"i-1": {"value": "y", "source": "option"}}


def test_no_answers_decode_to_no_resume_input() -> None:
    assert decode_interrupt_responses({}) == {}


def test_hitl_answers_are_rejoined_into_decisions() -> None:
    answers = {
        "welt-io:hitl:0:i-1": _pressed(True),
        "welt-io:hitl:1:i-1": _pressed(False),
        "welt-io:hitl:2:i-1": _typed("ask ops first"),
    }

    assert decode_interrupt_responses(answers) == {
        "i-1": {
            "decisions": [
                {"type": "approve"},
                {"type": "reject"},
                {"type": "respond", "message": "ask ops first"},
            ]
        }
    }


def test_hitl_decisions_follow_action_order() -> None:
    answers = {
        "welt-io:hitl:1:i-1": _pressed(False),
        "welt-io:hitl:0:i-1": _pressed(True),
    }

    assert decode_interrupt_responses(answers) == {
        "i-1": {"decisions": [{"type": "approve"}, {"type": "reject"}]}
    }


def test_a_typed_answer_is_a_respond_whatever_it_reads_like() -> None:
    answers = {"welt-io:hitl:0:i-1": _typed("approve")}

    assert decode_interrupt_responses(answers) == {
        "i-1": {"decisions": [{"type": "respond", "message": "approve"}]}
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
